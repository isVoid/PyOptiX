# -------------------------------------------------------------------------------
#
# Numba extensions for general CUDA / OptiX support
#
# -------------------------------------------------------------------------------

from enum import IntEnum
from operator import add, mul, sub
from typing import List, Tuple

from llvmlite import ir
from numba import cuda, float32, int32, types, uchar, uint8, uint32
from numba.core import cgutils
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    signature,
)
from numba.cuda.cudadecl import register, register_attr, register_global
from numba.cuda.cudadrv import nvvm
from numba.cuda.cudaimpl import lower
from numba.cuda.types import dim3

import optix


class VectorType(types.Type):
    def __init__(self, name, base_type, attr_names):
        self._base_type = base_type
        self._attr_names = attr_names
        super().__init__(name=name)

    @property
    def base_type(self):
        return self._base_type

    @property
    def attr_names(self):
        return self._attr_names

    @property
    def num_elements(self):
        return len(self._attr_names)


def make_vector_type(
    name: str, base_type: types.Type, attr_names: List[str]
) -> types.Type:
    """Create a vector type.

    Parameters
    ----------
    name: str
        The name of the type.
    base_type: numba.types.Type
        The primitive type for each element in the vector.
    attr_names: list of str
        Name for each attribute.
    """

    class _VectorType(VectorType):
        """Internal instantiation of VectorType."""

        pass

    class VectorTypeModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(attr_name, base_type) for attr_name in attr_names]
            super().__init__(dmm, fe_type, members)

    vector_type = _VectorType(name, base_type, attr_names)
    register_model(_VectorType)(VectorTypeModel)
    for attr_name in attr_names:
        make_attribute_wrapper(_VectorType, attr_name, attr_name)

    return vector_type


def make_vector_type_factory(
    vector_type: types.Type, overloads: List[Tuple[types.Type]]
):
    """Make a factory function for ``vector_type``

    Parameters
    ----------
    vector_type: VectorType
        The type to create factory function for.
    overloads: List of argument types tuples
        A list containing different overloads of the factory function. Each
        base type in the tuple should either be primitive type or VectorType.
    """

    def func():
        pass

    class FactoryTemplate(ConcreteTemplate):
        key = func
        cases = [signature(vector_type, *arglist) for arglist in overloads]

    def make_lower_factory(fml_arg_list):
        """Meta function to create a lowering for the factory function. Flattens
        the arguments by converting vector_type into load instructions for each
        of its attributes. Such as float2 -> float2.x, float2.y.
        """

        def lower_factory(context, builder, sig, actual_args):
            # A list of elements to assign from
            source_list = []
            # Convert the list of argument types to a list of load IRs.
            for argidx, fml_arg in enumerate(fml_arg_list):
                if isinstance(fml_arg, VectorType):
                    pxy = cgutils.create_struct_proxy(fml_arg)(
                        context, builder, actual_args[argidx]
                    )
                    source_list += [getattr(pxy, attr) for attr in fml_arg.attr_names]
                else:
                    # assumed primitive type
                    source_list.append(actual_args[argidx])

            if len(source_list) != vector_type.num_elements:
                raise numba.core.TypingError(
                    f"Unmatched number of source elements ({len(source_list)}) "
                    "and target elements ({vector_type.num_elements})."
                )

            out = cgutils.create_struct_proxy(vector_type)(context, builder)

            for attr_name, source in zip(vector_type.attr_names, source_list):
                setattr(out, attr_name, source)
            return out._getvalue()

        return lower_factory

    func.__name__ = f"make_{vector_type.name.lower()}"
    register(FactoryTemplate)
    register_global(func, types.Function(FactoryTemplate))
    for arglist in overloads:
        lower_factory = make_lower_factory(arglist)
        lower(func, *arglist)(lower_factory)
    return func


def lower_vector_type_binops(
    binop, vector_type: VectorType, overloads: List[Tuple[types.Type]]
):
    """Lower binops for ``vector_type``

    Parameters
    ----------
    binop: operation
        The binop to lower
    vector_type: VectorType
        The type to lower op for.
    overloads: List of argument types tuples
        A list containing different overloads of the binop. Expected to be either
            - vector_type x vector_type
            - primitive_type x vector_type
            - vector_type x primitive_type.
        In case one of the oprand is primitive_type, the operation is broadcasted.
    """
    # Should we assume the above are the only possible cases?
    class Vector_op_template(ConcreteTemplate):
        key = binop
        cases = [signature(vector_type, *arglist) for arglist in overloads]

    def make_lower_op(fml_arg_list):
        def op_impl(context, builder, sig, actual_args):
            def _make_load_IR(typ, actual_arg):
                if isinstance(typ, VectorType):
                    pxy = cgutils.create_struct_proxy(typ)(context, builder, actual_arg)
                    oprands = [getattr(pxy, attr) for attr in typ.attr_names]
                else:
                    # Assumed primitive type, broadcast
                    oprands = [actual_arg for _ in range(vector_type.num_elements)]
                return oprands

            def element_wise_op(lhs, rhs, res, attr):
                setattr(
                    res,
                    attr,
                    context.compile_internal(
                        builder,
                        lambda x, y: binop(x, y),
                        signature(types.float32, types.float32, types.float32),
                        (lhs, rhs),
                    ),
                )

            lhs_typ, rhs_typ = fml_arg_list
            # Construct a list of load IRs
            lhs = _make_load_IR(lhs_typ, actual_args[0])
            rhs = _make_load_IR(rhs_typ, actual_args[1])

            if not len(lhs) == len(rhs) == vector_type.num_elements:
                raise numba.core.TypingError(
                    f"Unmatched number of lhs elements ({len(lhs)}), rhs elements ({len(rhs)}) "
                    "and target elements ({vector_type.num_elements})."
                )

            out = cgutils.create_struct_proxy(vector_type)(context, builder)
            for attr, l, r in zip(vector_type.attr_names, lhs, rhs):
                element_wise_op(l, r, out, attr)

            return out._getvalue()

        return op_impl

    register_global(binop, types.Function(Vector_op_template))
    for arglist in overloads:
        impl = make_lower_op(arglist)
        lower(binop, *arglist)(impl)


# Register basic types
uchar4 = make_vector_type("UChar4", uchar, ["x", "y", "z", "w"])
float3 = make_vector_type("Float3", float32, ["x", "y", "z"])
float2 = make_vector_type("Float2", float32, ["x", "y"])
uint3 = make_vector_type("UInt3", uint32, ["x", "y", "z"])

# Register factory functions
make_uchar4 = make_vector_type_factory(uchar4, [(uchar,) * 4])
make_float3 = make_vector_type_factory(float3, [(float32,) * 3, (float2, float32)])
make_float2 = make_vector_type_factory(float2, [(float32,) * 2])
make_uint3 = make_vector_type_factory(uint3, [(uint32,) * 3])

# Lower Vector Type Ops
## float3
lower_vector_type_binops(
    add, float3, [(float3, float3), (float32, float3), (float3, float32)]
)
lower_vector_type_binops(
    mul, float3, [(float3, float3), (float32, float3), (float3, float32)]
)
## float2
lower_vector_type_binops(
    mul, float2, [(float2, float2), (float32, float2), (float2, float32)]
)
lower_vector_type_binops(
    sub, float2, [(float2, float2), (float32, float2), (float2, float32)]
)

# Temporary Payload Parameter Pack
class PayloadPack(types.Type):
    def __init__(self):
        super().__init__(name="PayloadPack")


payload_pack = PayloadPack()


@register_model(PayloadPack)
class PayloadPackModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("p0", types.uint32),
            ("p1", types.uint32),
            ("p2", types.uint32),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(PayloadPack, "p0", "p0")
make_attribute_wrapper(PayloadPack, "p1", "p1")
make_attribute_wrapper(PayloadPack, "p2", "p2")

# OptiX typedefs and enums
# -----------

OptixVisibilityMask = types.Integer("OptixVisibilityMask", bitwidth=32, signed=False)
OptixTraversableHandle = types.Integer(
    "OptixTraversableHandle", bitwidth=64, signed=False
)


class OptixRayFlags(IntEnum):
    OPTIX_RAY_FLAG_NONE = 0
    OPTIX_RAY_FLAG_DISABLE_ANYHIT = 1 << 0
    OPTIX_RAY_FLAG_ENFORCE_ANYHIT = 1 << 1
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1 << 2
    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT = (1 << 3,)
    OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1 << 4
    OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1 << 5
    OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT = 1 << 6
    OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT = 1 << 7


# OptiX types
# -----------

# Typing for OptiX types


class SbtDataPointer(types.RawPointer):
    def __init__(self):
        super().__init__(name="SbtDataPointer")


sbt_data_pointer = SbtDataPointer()


# Models for OptiX types


@register_model(SbtDataPointer)
class SbtDataPointerModel(models.OpaqueModel):
    pass


# Params
# ------------

# Structures as declared in triangle.h


class ParamsStruct:
    fields = (
        ("image", "uchar4*"),
        ("image_width", "unsigned int"),
        ("image_height", "unsigned int"),
        ("cam_eye", "float3"),
        ("cam_u", "float3"),
        ("cam_v", "float3"),
        ("cam_w", "float3"),
        ("handle", "OptixTraversableHandle"),
    )


# "Declare" a global called params
params = ParamsStruct()


class Params(types.Type):
    def __init__(self):
        super().__init__(name="ParamsType")


params_type = Params()


# ParamsStruct data model


@register_model(Params)
class ParamsModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("image", types.CPointer(uchar4)),
            ("image_width", types.uint32),
            ("image_height", types.uint32),
            ("cam_eye", float3),
            ("cam_u", float3),
            ("cam_v", float3),
            ("cam_w", float3),
            ("handle", OptixTraversableHandle),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(Params, "image", "image")
make_attribute_wrapper(Params, "image_width", "image_width")
make_attribute_wrapper(Params, "image_height", "image_height")
make_attribute_wrapper(Params, "cam_eye", "cam_eye")
make_attribute_wrapper(Params, "cam_u", "cam_u")
make_attribute_wrapper(Params, "cam_v", "cam_v")
make_attribute_wrapper(Params, "cam_w", "cam_w")
make_attribute_wrapper(Params, "handle", "handle")


@typeof_impl.register(ParamsStruct)
def typeof_params(val, c):
    return params_type


# ParamsStruct lowering
# The below makes 'param' a global variable, accessible from any user defined
# kernels.


@lower_constant(Params)
def constant_params(context, builder, ty, pyval):
    try:
        gvar = builder.module.get_global("params")
    except KeyError:
        llty = context.get_value_type(ty)
        gvar = cgutils.add_global_variable(
            builder.module, llty, "params", addrspace=nvvm.ADDRSPACE_CONSTANT
        )
        gvar.linkage = "external"
        gvar.global_constant = True

    return builder.load(gvar)


# MissData
# ------------

# Structures as declared in triangle.h
class MissDataStruct:
    fields = ("bg_color", "float3")


MissData = MissDataStruct()


class MissData(types.Type):
    def __init__(self):
        super().__init__(name="MissDataType")


miss_data_type = MissData()


@register_model(MissData)
class MissDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("bg_color", float3),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(MissData, "bg_color", "bg_color")


@typeof_impl.register(MissDataStruct)
def typeof_miss_data(val, c):
    return miss_data_type


# MissData Constructor
@type_callable(MissDataStruct)
def type_miss_data_struct(context):
    def typer(sbt_data_pointer):
        if isinstance(sbt_data_pointer, SbtDataPointer):
            return miss_data_type

    return typer


@lower(MissDataStruct, sbt_data_pointer)
def lower_miss_data_ctor(context, builder, sig, args):
    # Anyway to err if this ctor is not called inside __miss__* program?
    # TODO: Optimize
    ptr = args[0]
    ptr = builder.bitcast(ptr, context.get_value_type(miss_data_type).as_pointer())

    bg_color_ptr = cgutils.gep_inbounds(builder, ptr, 0, 0)

    xptr = cgutils.gep_inbounds(builder, bg_color_ptr, 0, 0)
    yptr = cgutils.gep_inbounds(builder, bg_color_ptr, 0, 1)
    zptr = cgutils.gep_inbounds(builder, bg_color_ptr, 0, 2)

    output_miss_data = cgutils.create_struct_proxy(miss_data_type)(context, builder)
    output_bg_color_ptr = cgutils.gep_inbounds(
        builder, output_miss_data._getpointer(), 0, 0
    )
    output_bg_color_x_ptr = cgutils.gep_inbounds(builder, output_bg_color_ptr, 0, 0)
    output_bg_color_y_ptr = cgutils.gep_inbounds(builder, output_bg_color_ptr, 0, 1)
    output_bg_color_z_ptr = cgutils.gep_inbounds(builder, output_bg_color_ptr, 0, 2)

    x = builder.load(xptr)
    y = builder.load(yptr)
    z = builder.load(zptr)

    builder.store(x, output_bg_color_x_ptr)
    builder.store(y, output_bg_color_y_ptr)
    builder.store(z, output_bg_color_z_ptr)

    # Doesn't seem to do what's expected?
    # miss_data.bg_color.x = builder.load(xptr)
    # miss_data.bg_color.y = builder.load(yptr)
    # miss_data.bg_color.z = builder.load(zptr)
    return output_miss_data._getvalue()


# OptiX functions
# ---------------

# Here we "prototype" the OptiX functions that the user will call in their
# kernels, so that Numba has something to refer to when compiling the kernel.


def _optix_GetLaunchIndex():
    pass


def _optix_GetLaunchDimensions():
    pass


def _optix_GetSbtDataPointer():
    pass


def _optix_SetPayload_0():
    pass


def _optix_SetPayload_1():
    pass


def _optix_SetPayload_2():
    pass


def _optix_GetTriangleBarycentrics():
    pass


def _optix_Trace():
    pass


# Monkey-patch the functions into the optix module, so the user can write
# optix.GetLaunchIndex etc., for symmetry with the rest of the API implemented
# in PyOptiX.

optix.GetLaunchIndex = _optix_GetLaunchIndex
optix.GetLaunchDimensions = _optix_GetLaunchDimensions
optix.GetSbtDataPointer = _optix_GetSbtDataPointer
optix.GetTriangleBarycentrics = _optix_GetTriangleBarycentrics
optix.SetPayload_0 = _optix_SetPayload_0
optix.SetPayload_1 = _optix_SetPayload_1
optix.SetPayload_2 = _optix_SetPayload_2

optix.Trace = _optix_Trace


# OptiX function typing


@register
class OptixGetLaunchIndex(ConcreteTemplate):
    key = optix.GetLaunchIndex
    cases = [signature(dim3)]


@register
class OptixGetLaunchDimensions(ConcreteTemplate):
    key = optix.GetLaunchDimensions
    cases = [signature(dim3)]


@register
class OptixGetSbtDataPointer(ConcreteTemplate):
    key = optix.GetSbtDataPointer
    cases = [signature(sbt_data_pointer)]


def registerSetPayload(reg):
    class OptixSetPayloadReg(ConcreteTemplate):
        key = getattr(optix, "SetPayload_" + str(reg))
        cases = [signature(types.void, uint32)]

    register(OptixSetPayloadReg)
    return OptixSetPayloadReg


OptixSetPayload_0 = registerSetPayload(0)
OptixSetPayload_1 = registerSetPayload(1)
OptixSetPayload_2 = registerSetPayload(2)


@register
class OptixGetTriangleBarycentrics(ConcreteTemplate):
    key = optix.GetTriangleBarycentrics
    cases = [signature(float2)]


@register
class OptixTrace(ConcreteTemplate):
    key = optix.Trace
    cases = [
        signature(
            payload_pack,
            OptixTraversableHandle,
            float3,
            float3,
            float32,
            float32,
            float32,
            OptixVisibilityMask,
            uint32,
            uint32,
            uint32,
            uint32,
        )
    ]


@register_attr
class OptixModuleTemplate(AttributeTemplate):
    key = types.Module(optix)

    def resolve_GetLaunchIndex(self, mod):
        return types.Function(OptixGetLaunchIndex)

    def resolve_GetLaunchDimensions(self, mod):
        return types.Function(OptixGetLaunchDimensions)

    def resolve_GetSbtDataPointer(self, mod):
        return types.Function(OptixGetSbtDataPointer)

    def resolve_SetPayload_0(self, mod):
        return types.Function(OptixSetPayload_0)

    def resolve_SetPayload_1(self, mod):
        return types.Function(OptixSetPayload_1)

    def resolve_SetPayload_2(self, mod):
        return types.Function(OptixSetPayload_2)

    def resolve_GetTriangleBarycentrics(self, mod):
        return types.Function(OptixGetTriangleBarycentrics)

    def resolve_Trace(self, mod):
        return types.Function(OptixTrace)


# OptiX function lowering


@lower(optix.GetLaunchIndex)
def lower_optix_getLaunchIndex(context, builder, sig, args):
    def get_launch_index(axis):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.IntType(32), []),
            f"call ($0), _optix_get_launch_index_{axis}, ();",
            "=r",
        )
        return builder.call(asm, [])

    index = cgutils.create_struct_proxy(dim3)(context, builder)
    index.x = get_launch_index("x")
    index.y = get_launch_index("y")
    index.z = get_launch_index("z")
    return index._getvalue()


@lower(optix.GetLaunchDimensions)
def lower_optix_getLaunchDimensions(context, builder, sig, args):
    def get_launch_dimensions(axis):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.IntType(32), []),
            f"call ($0), _optix_get_launch_dimension_{axis}, ();",
            "=r",
        )
        return builder.call(asm, [])

    index = cgutils.create_struct_proxy(dim3)(context, builder)
    index.x = get_launch_dimensions("x")
    index.y = get_launch_dimensions("y")
    index.z = get_launch_dimensions("z")
    return index._getvalue()


@lower(optix.GetSbtDataPointer)
def lower_optix_getSbtDataPointer(context, builder, sig, args):
    asm = ir.InlineAsm(
        ir.FunctionType(ir.IntType(64), []),
        "call ($0), _optix_get_sbt_data_ptr_64, ();",
        "=l",
    )
    ptr = builder.call(asm, [])
    ptr = builder.inttoptr(ptr, ir.IntType(8).as_pointer())
    return ptr


def lower_optix_SetPayloadReg(reg):
    def lower_optix_SetPayload_impl(context, builder, sig, args):
        asm = ir.InlineAsm(
            ir.FunctionType(ir.VoidType(), [ir.IntType(32), ir.IntType(32)]),
            f"call _optix_set_payload, ($0, $1);",
            "r,r",
        )
        builder.call(asm, [context.get_constant(types.int32, reg), args[0]])

    lower(getattr(optix, f"SetPayload_{reg}"), uint32)(lower_optix_SetPayload_impl)


lower_optix_SetPayloadReg(0)
lower_optix_SetPayloadReg(1)
lower_optix_SetPayloadReg(2)


@lower(optix.GetTriangleBarycentrics)
def lower_optix_getTriangleBarycentrics(context, builder, sig, args):
    f2 = cgutils.create_struct_proxy(float2)(context, builder)
    retty = ir.LiteralStructType([ir.FloatType(), ir.FloatType()])
    asm = ir.InlineAsm(
        ir.FunctionType(retty, []),
        "call ($0, $1), _optix_get_triangle_barycentrics, ();",
        "=f,=f",
    )
    ret = builder.call(asm, [])
    f2.x = builder.extract_value(ret, 0)
    f2.y = builder.extract_value(ret, 1)
    return f2._getvalue()


@lower(
    optix.Trace,
    OptixTraversableHandle,
    float3,
    float3,
    float32,
    float32,
    float32,
    OptixVisibilityMask,
    uint32,
    uint32,
    uint32,
    uint32,
)
def lower_optix_Trace(context, builder, sig, args):
    # Only implements the version that accepts 3 payload registers
    # TODO: Optimize returns, adapt to 0-32 payload registers.

    (
        handle,
        rayOrigin,
        rayDirection,
        tmin,
        tmax,
        rayTime,
        visibilityMask,
        rayFlags,
        SBToffset,
        SBTstride,
        missSBTIndex,
    ) = args

    rayOrigin = cgutils.create_struct_proxy(float3)(context, builder, rayOrigin)
    rayDirection = cgutils.create_struct_proxy(float3)(context, builder, rayDirection)
    output = cgutils.create_struct_proxy(payload_pack)(context, builder)

    ox, oy, oz = rayOrigin.x, rayOrigin.y, rayOrigin.z
    dx, dy, dz = rayDirection.x, rayDirection.y, rayDirection.z

    n_payload_registers = 3
    n_stub_output_operands = 32 - n_payload_registers
    outputs = [output.p0, output.p1, output.p2] + [
        builder.load(builder.alloca(ir.IntType(32)))
        for _ in range(n_stub_output_operands)
    ]

    retty = ir.LiteralStructType([ir.IntType(32)] * 32)
    asm = ir.InlineAsm(
        ir.FunctionType(retty, []),
        "call "
        "($0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,"
        "$30,$31),"
        "_optix_trace_typed_32,"
        "($32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48,$49,$50,$51,$52,$53,$54,$55,$56,$57,$58,$59,"
        "$60,$61,$62,$63,$64,$65,$66,$67,$68,$69,$70,$71,$72,$73,$74,$75,$76,$77,$78,$79,$80);",
        "=r," * 32 + "r,l,f,f,f,f,f,f,f,f,f,r,r,r,r,r,r," + "r," * 31 + "r",
        side_effect=True,
    )

    zero = context.get_constant(types.int32, 0)
    c_payload_registers = context.get_constant(types.int32, n_payload_registers)
    args = [
        zero,
        handle,
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        tmin,
        tmax,
        rayTime,
        visibilityMask,
        rayFlags,
        SBToffset,
        SBTstride,
        missSBTIndex,
        c_payload_registers,
    ] + outputs
    ret = builder.call(asm, args)
    output.p0 = builder.extract_value(ret, 0)
    output.p1 = builder.extract_value(ret, 1)
    output.p2 = builder.extract_value(ret, 2)
    return output._getvalue()
