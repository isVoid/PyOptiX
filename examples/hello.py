#!/usr/bin/env python3


import optix
import cupy  as cp    # CUDA bindings
import numpy as np    # Packing of structures in C-compatible format

import array
import ctypes         # C interop helpers
from PIL import Image # Image IO

from llvmlite import ir

from numba import types
from numba.core import cgutils
from numba.core.extending import models, register_model, typeof_impl
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                         signature)
from numba.cuda import compile_ptx_for_current_device
from numba.cuda.cudadecl import register, register_attr, register_global
from numba.cuda.cudaimpl import lower, lower_attr
from numba.cuda.types import dim3


#-------------------------------------------------------------------------------
#
# Util 
#
#-------------------------------------------------------------------------------
pix_width  = 512
pix_height = 512

class Logger:
    def __init__( self ):
        self.num_mssgs = 0

    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
        self.num_mssgs += 1


def log_callback( level, tag, mssg ):
    print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
    

def round_up( val, mult_of ):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of 


def  get_aligned_itemsize( formats, alignment ):
    names = []
    for i in range( len(formats ) ):
        names.append( 'x'+str(i) )

    temp_dtype = np.dtype( { 
        'names'   : names,
        'formats' : formats, 
        'align'   : True
        } )
    return round_up( temp_dtype.itemsize, alignment )


def array_to_device_memory( numpy_array, stream=cp.cuda.Stream() ):

    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )
    return d_mem


def compile_cuda( cuda_file ):
    with open( cuda_file, 'rb' ) as f:
        src = f.read()
    from pynvrtc.compiler import Program
    prog = Program( src.decode(), cuda_file )
    ptx  = prog.compile( [
        '-use_fast_math', 
        '-lineinfo',
        '-default-device',
        '-std=c++11',
        '-rdc',
        'true',
        #'-IC:\\ProgramData\\NVIDIA Corporation\OptiX SDK 7.2.0\include',
        #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
        '-I/usr/local/cuda/include',
        '-I/home/gmarkall/numbadev/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64/include/'
        ] )
    return ptx


#-------------------------------------------------------------------------------
#
# Hello kernel and Numba extensions
#
#-------------------------------------------------------------------------------

class RayGenDataStruct:
    pass


class ParamsStruct:
    def __init__(self, image, image_width):
        self.image = image
        self.image_width = image_width


class RayGenData(types.Type):
    def __init__(self):
        super().__init__(name='RayGenDataType')

ray_gen_data = RayGenData()


@register_attr
class RayGenData_attrs(AttributeTemplate):
    key = ray_gen_data

    def resolve_r(self, mod):
        return types.float32

    def resolve_g(self, mod):
        return types.float32

    def resolve_b(self, mod):
        return types.float32


class UChar4(types.Type):
    def __init__(self):
        super().__init__(name="UChar4")


uchar4 = UChar4()


class Params(types.Type):
    def __init__(self):
        super().__init__(name='ParamsType')


params_type = Params()


@register_attr
class Params_attrs(AttributeTemplate):
    key = params_type

    def resolve_image(self, mod):
        return types.CPointer(uchar4)

    def resolve_image_width(self, mod):
        return types.uint32


@register_model(RayGenData)
class RayGenDataModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('r', types.float32),
            ('g', types.float32),
            ('b', types.float32),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(UChar4)
class Uchar4Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.uchar),
            ('y', types.uchar),
            ('z', types.uchar),
            ('w', types.uchar),
        ]
        super().__init__(dmm, fe_type, members)


@register_model(Params)
class ParamsModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('image', types.CPointer(uchar4)),
            ('image_width', types.uint32)
        ]
        super().__init__(dmm, fe_type, members)


@typeof_impl.register(ParamsStruct)
def typeof_params(val, c):
    return params_type


params = ParamsStruct(None, None)


@lower_constant(Params)
def constant_params(context, builder, ty, pyval):
    try:
        gvar = builder.module.get_global('params')
    except KeyError:
        llty = context.get_value_type(ty)
        gvar = cgutils.add_global_variable(builder.module, llty, 'params')
        gvar.linkage = 'external'
        gvar.global_constant = True

    return builder.load(gvar)


@lower_attr(Params, 'image')
def params_image_width(context, builder, sig, arg):
    return builder.extract_value(arg, 0)


@lower_attr(Params, 'image_width')
def params_image_width(context, builder, sig, arg):
    return builder.extract_value(arg, 1)


def __raygen__hello():
    launch_index = optix.GetLaunchIndex();
    rtData = optix.GetSbtDataPointer();

    f0 = types.float32(0.0)
    f255 = types.float32(255.0)

    idx = launch_index.y * params.image_width + launch_index.x

    params.image[idx] = make_uchar4(
            max(f0, min(f255, rtData.r * f255)),
            max(f0, min(f255, rtData.g * f255)),
            max(f0, min(f255, rtData.b * f255)),
            255
    )


def _optix_GetLaunchIndex():
    pass


def _optix_GetSbtDataPointer():
    pass


def make_uchar4(x, y, z, w):
    pass


optix.GetLaunchIndex = _optix_GetLaunchIndex
optix.GetSbtDataPointer = _optix_GetSbtDataPointer


@register
class OptixGetLaunchIndex(ConcreteTemplate):
    key = optix.GetLaunchIndex
    cases = [signature(dim3)]


@register
class OptixGetSbtDataPointer(ConcreteTemplate):
    key = optix.GetSbtDataPointer
    cases = [signature(ray_gen_data)]


@register
class MakeUChar4(ConcreteTemplate):
    key = make_uchar4
    cases = [signature(uchar4, types.uchar, types.uchar, types.uchar,
                       types.uchar)]


register_global(make_uchar4, types.Function(MakeUChar4))


@register_attr
class OptixModuleTemplate(AttributeTemplate):
    key = types.Module(optix)

    def resolve_GetLaunchIndex(self, mod):
        return types.Function(OptixGetLaunchIndex)

    def resolve_GetSbtDataPointer(self, mod):
        return types.Function(OptixGetSbtDataPointer)


@lower(optix.GetLaunchIndex)
def lower_optix_getLaunchIndex(context, builder, sig, args):
    def get_launch_index(axis):
        asm = ir.InlineAsm(ir.FunctionType(ir.IntType(32), []),
                           f"call ($0), _optix_get_launch_index_{axis}, ();",
                           "=r")
        return builder.call(asm, [])

    index = cgutils.create_struct_proxy(dim3)(context, builder)
    index.x = get_launch_index('x')
    index.y = get_launch_index('y')
    index.z = get_launch_index('z')
    return index._getvalue()


@lower(optix.GetSbtDataPointer)
def lower_optix_getSbtDataPointer(context, builder, sig, args):
    asm = ir.InlineAsm(ir.FunctionType(ir.IntType(64), []),
                       f"call ($0), _optix_get_sbt_data_ptr_64, ();",
                       "=l")
    ptr = builder.call(asm, [])
    # Note - here we're dereferencing the pointer and loading the values,
    # for convenience in a PoC. A proper implementation would return the
    # pointer and the lowering would deal appropriately with it.
    #breakpoint()
    ptr = builder.inttoptr(ptr,
                          context.get_value_type(ray_gen_data).as_pointer())
                          #ir.PointerType(context.get_value_type(ray_gen_data)))
    rgd = cgutils.create_struct_proxy(ray_gen_data)(context, builder)
    rptr = cgutils.gep_inbounds(builder, ptr, 0, 0)
    gptr = cgutils.gep_inbounds(builder, ptr, 0, 1)
    bptr = cgutils.gep_inbounds(builder, ptr, 0, 2)
    rgd.r = builder.load(rptr)
    rgd.g = builder.load(gptr)
    rgd.b = builder.load(bptr)
    return rgd._getvalue()


@lower(make_uchar4, types.uchar, types.uchar, types.uchar, types.uchar)
def lower_make_uchar4(context, builder, sig, args):
    uc4 = cgutils.create_struct_proxy(uchar4)(context, builder)
    uc4.x = args[0]
    uc4.y = args[1]
    uc4.z = args[2]
    uc4.w = args[3]
    return uc4._getvalue()


@lower_attr(ray_gen_data, 'r')
def ray_gen_data_r(context, builder, sig, args):
    return builder.extract_value(args, 0)


@lower_attr(ray_gen_data, 'g')
def ray_gen_data_r(context, builder, sig, args):
    return builder.extract_value(args, 1)


@lower_attr(ray_gen_data, 'b')
def ray_gen_data_r(context, builder, sig, args):
    return builder.extract_value(args, 2)


def compile_numba(f, sig=()):
    return compile_ptx_for_current_device(f, sig)


#-------------------------------------------------------------------------------
#
# Optix setup
#
#-------------------------------------------------------------------------------

def init_optix():
    print( "Initializing cuda ..." )
    cp.cuda.runtime.free( 0 )

    print( "Initializing optix ..." )
    optix.init()


def create_ctx():
    print( "Creating optix device context ..." )

    # Note that log callback data is no longer needed.  We can
    # instead send a callable class instance as the log-function
    # which stores any data needed
    global logger
    logger = Logger()
    
    # OptiX param struct fields can be set with optional
    # keyword constructor arguments.
    ctx_options = optix.DeviceContextOptions( 
            logCallbackFunction = logger,
            logCallbackLevel    = 4
            )

    # They can also be set and queried as properties on the struct
    ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL 

    cu_ctx = 0 
    return optix.deviceContextCreate( cu_ctx, ctx_options )


def set_pipeline_options():
    return optix.PipelineCompileOptions(
        usesMotionBlur        = False,
        traversableGraphFlags = 
            optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        numPayloadValues      = 2,
        numAttributeValues    = 2,
        exceptionFlags        = optix.EXCEPTION_FLAG_NONE,
        pipelineLaunchParamsVariableName = "params"
        )


def create_module( ctx, pipeline_options, hello_ptx ):
    print( "Creating optix module ..." )
    
    formats = ['u8', 'u4']
    itemsize = get_aligned_itemsize( formats, 16 )
    params_dtype = np.dtype( { 
        'names'   : ['image', 'image_width' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )

    bound_value = array.array( 'i', [pix_width] )
    bound_value_entry = optix.ModuleCompileBoundValueEntry(
        pipelineParamOffsetInBytes = params_dtype.fields['image_width'][1],
        boundValue  = bound_value,
        annotation  = "my_bound_value"
        )


    module_options = optix.ModuleCompileOptions(
        maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
        boundValues      = [ bound_value_entry ],
        debugLevel       = optix.COMPILE_DEBUG_LEVEL_LINEINFO
    )

    module, log = ctx.moduleCreateFromPTX(
            module_options,
            pipeline_options,
            hello_ptx
            )
    print( "\tModule create log: <<<{}>>>".format( log ) )
    return module


def create_program_groups( ctx, module ):
    print( "Creating program groups ... " )

    # TODO: optix.ProgramGroup.Options() ?
    program_group_options = optix.ProgramGroupOptions()

    # TODO: optix.ProgramGroup.Kind.RAYGEN ?
    raygen_prog_group_desc                          = optix.ProgramGroupDesc()
    raygen_prog_group_desc.kind                     = \
        optix.PROGRAM_GROUP_KIND_RAYGEN 
    raygen_prog_group_desc.raygenModule             = module
    raygen_prog_group_desc.raygenEntryFunctionName  = "__raygen__hello"

    log = ""
    raygen_prog_group = ctx.programGroupCreate(
            [ raygen_prog_group_desc ], 
            program_group_options,
            log
            )
    print( "\tProgramGroup raygen create log: <<<{}>>>".format( log ) )

    # Leave miss group's module and entryfunc name null
    miss_prog_group_desc  = optix.ProgramGroupDesc()
    miss_prog_group_desc.kind = optix.PROGRAM_GROUP_KIND_MISS

    miss_prog_group = ctx.programGroupCreate(
            [ miss_prog_group_desc ],
            program_group_options,
            log
            )
    print( "\tProgramGroup miss create log: <<<{}>>>".format( log ) )

    return ( raygen_prog_group[0], miss_prog_group[0] )


def create_pipeline( ctx, raygen_prog_group, pipeline_compile_options ):
    print( "Creating pipeline ... " )
    max_trace_depth  = 0
    program_groups = [ raygen_prog_group ]

    pipeline_link_options               = optix.PipelineLinkOptions() 
    pipeline_link_options.maxTraceDepth = max_trace_depth
    pipeline_link_options.debugLevel    = optix.COMPILE_DEBUG_LEVEL_FULL

    log = ""
    pipeline = ctx.pipelineCreate(
            pipeline_compile_options,
            pipeline_link_options,
            program_groups,
            log
            )

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes( prog_group, stack_sizes )

    (dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size) = \
        optix.util.computeStackSizes( 
            stack_sizes, 
            0,  # maxTraceDepth
            0,  # maxCCDepth
            0   # maxDCDepth
            )
    
    pipeline.setStackSize( 
            dc_stack_size_from_trav,
            dc_stack_size_from_state, 
            cc_stack_size,
            2  # maxTraversableDepth
            )

    return pipeline


def create_sbt( raygen_prog_group, miss_prog_group ):
    print( "Creating sbt ... " )

    global d_raygen_sbt
    global d_miss_sbt

    header_format = '{}B'.format( optix.SBT_RECORD_HEADER_SIZE )

    #
    # raygen record
    #
    formats  = [ header_format, 'f4', 'f4', 'f4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( { 
        'names'   : ['header', 'r', 'g', 'b' ],
        'formats' : formats, 
        'itemsize': itemsize,
        'align'   : True
        } )
    h_raygen_sbt = np.array( [ (0, 0.462, 0.725, 0.0 ) ], dtype=dtype )
    optix.sbtRecordPackHeader( raygen_prog_group, h_raygen_sbt )
    d_raygen_sbt = array_to_device_memory( h_raygen_sbt )
    
    #
    # miss record
    #
    formats  = [ header_format, 'i4']
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( { 
        'names'   : ['header', 'x' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_miss_sbt = np.array( [ (0, 127 ) ], dtype=dtype )
    optix.sbtRecordPackHeader( miss_prog_group, h_miss_sbt )
    d_miss_sbt = array_to_device_memory( h_miss_sbt )
    
    sbt = optix.ShaderBindingTable()
    sbt.raygenRecord                = d_raygen_sbt.ptr
    sbt.missRecordBase              = d_miss_sbt.ptr
    sbt.missRecordStrideInBytes     = d_miss_sbt.mem.size
    sbt.missRecordCount             = 1
    return sbt


def launch( pipeline, sbt ):
    print( "Launching ... " )

    pix_bytes  = pix_width*pix_height*4
    
    h_pix = np.zeros( (pix_width,pix_height,4), 'B' )
    h_pix[0:256, 0:256] = [255, 128, 0, 255]
    d_pix = cp.array( h_pix )

    formats = ['u8', 'u4']
    itemsize = get_aligned_itemsize( formats, 8 )
    params_dtype = np.dtype( { 
        'names'   : ['image', 'image_width' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_params = np.array( [ ( d_pix.data.ptr, pix_width ) ], dtype=params_dtype )
    d_params = array_to_device_memory( h_params )

    stream = cp.cuda.Stream()
    optix.launch( 
        pipeline, 
        stream.ptr, 
        d_params.ptr, 
        h_params.dtype.itemsize, 
        sbt,
        pix_width,
        pix_height,
        1 # depth
        )

    stream.synchronize()

    h_pix = cp.asnumpy( d_pix )
    return h_pix


#-------------------------------------------------------------------------------
#
# main
#
#-------------------------------------------------------------------------------


def main():
    cuda_hello_ptx = compile_cuda( "examples/hello.cu" )
    print(cuda_hello_ptx)
    hello_ptx, resty = compile_numba(__raygen__hello)
    # Demangle name
    hello_ptx = hello_ptx.replace('_ZN6cudapy8__main__19__raygen__hello$241E',
                                  '__raygen__hello')

    # Hack for global var things
    hello_ptx = hello_ptx.replace('.extern .global .align 8 .b8 params[16];',
                                  '.visible .const .align 8 .b8 params[16];')
    hello_ptx = hello_ptx.replace('ld.global', 'ld.const')
    print(hello_ptx)

    init_optix()

    ctx              = create_ctx()
    pipeline_options = set_pipeline_options()
    module           = create_module( ctx, pipeline_options, hello_ptx )
    raygen_prog_group, miss_prog_group = create_program_groups( ctx, module )
    pipeline         = create_pipeline( ctx, raygen_prog_group, pipeline_options )
    sbt              = create_sbt( raygen_prog_group, miss_prog_group ) 
    pix              = launch( pipeline, sbt ) 

    print( "Total number of log messages: {}".format( logger.num_mssgs ) )

    img = Image.fromarray( pix, 'RGBA' )
    img.save( 'my.png' )
    img.show()


if __name__ == "__main__":
    main()
