#!/usr/bin/env python3


import ctypes  # C interop helpers
import math
from enum import Enum

import cupy as cp  # CUDA bindings
import numpy as np  # Packing of structures in C-compatible format
from numba import cuda, float32, int32, types, uint8, uint32
from numba.core.extending import overload
from numba.cuda import get_current_device
from numba.cuda.compiler import compile_cuda as numba_compile_cuda
from numba.cuda.libdevice import fast_powf, float_as_int, int_as_float
from numba_support import (
    MissDataStruct,
    OptixRayFlags,
    OptixVisibilityMask,
    float2,
    float3,
    make_float2,
    make_float3,
    make_uchar4,
    make_uint3,
    params,
    uchar4,
    uint3,
)
from PIL import Image, ImageOps  # Image IO
from vec_math import clamp, dot, normalize

import optix

# -------------------------------------------------------------------------------
#
# Util
#
# -------------------------------------------------------------------------------
pix_width = 1024
pix_height = 768


class Logger:
    def __init__(self):
        self.num_mssgs = 0

    def __call__(self, level, tag, mssg):
        print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))
        self.num_mssgs += 1


def log_callback(level, tag, mssg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))


def round_up(val, mult_of):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def get_aligned_itemsize(formats, alignment):
    names = []
    for i in range(len(formats)):
        names.append("x" + str(i))

    temp_dtype = np.dtype({"names": names, "formats": formats, "align": True})
    return round_up(temp_dtype.itemsize, alignment)


def array_to_device_memory(numpy_array, stream=cp.cuda.Stream()):

    byte_size = numpy_array.size * numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p(numpy_array.ctypes.data)
    d_mem = cp.cuda.memory.alloc(byte_size)
    d_mem.copy_from_async(h_ptr, byte_size, stream)
    return d_mem


def compile_cuda(cuda_file):
    with open(cuda_file, "rb") as f:
        src = f.read()
    from pynvrtc.compiler import Program

    prog = Program(src.decode(), cuda_file)
    ptx = prog.compile(
        [
            "-use_fast_math",
            "-lineinfo",
            "-default-device",
            "-std=c++11",
            "-rdc",
            "true",
            #'-IC:\\ProgramData\\NVIDIA Corporation\OptiX SDK 7.2.0\include',
            #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
            "-I/usr/local/cuda/include",
            f"-I{optix.include_path}",
        ]
    )
    return ptx


# -------------------------------------------------------------------------------
#
# Optix setup
#
# -------------------------------------------------------------------------------


def init_optix():
    print("Initializing cuda ...")
    cp.cuda.runtime.free(0)

    print("Initializing optix ...")
    optix.init()


def create_ctx():
    print("Creating optix device context ...")

    # Note that log callback data is no longer needed.  We can
    # instead send a callable class instance as the log-function
    # which stores any data needed
    global logger
    logger = Logger()

    # OptiX param struct fields can be set with optional
    # keyword constructor arguments.
    ctx_options = optix.DeviceContextOptions(
        logCallbackFunction=logger, logCallbackLevel=4
    )

    # They can also be set and queried as properties on the struct
    ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL

    cu_ctx = 0
    return optix.deviceContextCreate(cu_ctx, ctx_options)


def create_accel(ctx):

    accel_options = optix.AccelBuildOptions(
        buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
        operation=optix.BUILD_OPERATION_BUILD,
    )

    global vertices
    vertices = cp.array([-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0], dtype="f4")

    triangle_input_flags = [optix.GEOMETRY_FLAG_NONE]
    triangle_input = optix.BuildInputTriangleArray()
    triangle_input.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    triangle_input.numVertices = len(vertices)
    triangle_input.vertexBuffers = [vertices.data.ptr]
    triangle_input.flags = triangle_input_flags
    triangle_input.numSbtRecords = 1

    gas_buffer_sizes = ctx.accelComputeMemoryUsage([accel_options], [triangle_input])

    d_temp_buffer_gas = cp.cuda.alloc(gas_buffer_sizes.tempSizeInBytes)
    d_gas_output_buffer = cp.cuda.alloc(gas_buffer_sizes.outputSizeInBytes)

    gas_handle = ctx.accelBuild(
        0,  # CUDA stream
        [accel_options],
        [triangle_input],
        d_temp_buffer_gas.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [],  # emitted properties
    )

    return (gas_handle, d_gas_output_buffer)


def set_pipeline_options():
    return optix.PipelineCompileOptions(
        usesMotionBlur=False,
        traversableGraphFlags=int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
        numPayloadValues=3,
        numAttributeValues=3,
        exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
        pipelineLaunchParamsVariableName="params",
        usesPrimitiveTypeFlags=optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE,
    )


def create_module(ctx, pipeline_options, ptx):
    print("Creating optix module ...")

    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_LINEINFO,
    )

    module, log = ctx.moduleCreateFromPTX(module_options, pipeline_options, ptx)
    print("\tModule create log: <<<{}>>>".format(log))
    return module


def create_program_groups(ctx, raygen_module, miss_prog_module, hitgroup_module):
    print("Creating program groups ... ")

    program_group_options = optix.ProgramGroupOptions()

    raygen_prog_group_desc = optix.ProgramGroupDesc()
    raygen_prog_group_desc.raygenModule = raygen_module
    raygen_prog_group_desc.raygenEntryFunctionName = "__raygen__rg"

    miss_prog_group_desc = optix.ProgramGroupDesc()
    miss_prog_group_desc.missModule = miss_prog_module
    miss_prog_group_desc.missEntryFunctionName = "__miss__ms"

    hitgroup_prog_group_desc = optix.ProgramGroupDesc()
    hitgroup_prog_group_desc.hitgroupModuleCH = hitgroup_module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"

    prog_group, log = ctx.programGroupCreate(
        [raygen_prog_group_desc, miss_prog_group_desc, hitgroup_prog_group_desc],
        program_group_options,
    )
    print("\tProgramGroup create log: <<<{}>>>".format(log))

    return prog_group


def create_pipeline(ctx, program_groups, pipeline_compile_options):
    print("Creating pipeline ... ")

    max_trace_depth = 1
    pipeline_link_options = optix.PipelineLinkOptions()
    pipeline_link_options.maxTraceDepth = max_trace_depth
    pipeline_link_options.debugLevel = optix.COMPILE_DEBUG_LEVEL_FULL

    log = ""
    pipeline = ctx.pipelineCreate(
        pipeline_compile_options, pipeline_link_options, program_groups, log
    )

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes(prog_group, stack_sizes)

    (
        dc_stack_size_from_trav,
        dc_stack_size_from_state,
        cc_stack_size,
    ) = optix.util.computeStackSizes(
        stack_sizes, max_trace_depth, 0, 0  # maxCCDepth  # maxDCDepth
    )

    pipeline.setStackSize(
        dc_stack_size_from_trav,
        dc_stack_size_from_state,
        cc_stack_size,
        1,  # maxTraversableDepth
    )

    return pipeline


def create_sbt(prog_groups):
    print("Creating sbt ... ")

    (raygen_prog_group, miss_prog_group, hitgroup_prog_group) = prog_groups

    global d_raygen_sbt
    global d_miss_sbt

    header_format = "{}B".format(optix.SBT_RECORD_HEADER_SIZE)

    #
    # raygen record
    #
    formats = [header_format]
    itemsize = get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype(
        {"names": ["header"], "formats": formats, "itemsize": itemsize, "align": True}
    )
    h_raygen_sbt = np.array([0], dtype=dtype)
    optix.sbtRecordPackHeader(raygen_prog_group, h_raygen_sbt)
    global d_raygen_sbt
    d_raygen_sbt = array_to_device_memory(h_raygen_sbt)

    #
    # miss record
    #
    formats = [header_format, "f4", "f4", "f4"]
    itemsize = get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype(
        {
            "names": ["header", "r", "g", "b"],
            "formats": formats,
            "itemsize": itemsize,
            "align": True,
        }
    )
    h_miss_sbt = np.array([(0, 0.3, 0.1, 0.2)], dtype=dtype)
    optix.sbtRecordPackHeader(miss_prog_group, h_miss_sbt)
    global d_miss_sbt
    d_miss_sbt = array_to_device_memory(h_miss_sbt)

    #
    # hitgroup record
    #
    formats = [header_format]
    itemsize = get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype(
        {"names": ["header"], "formats": formats, "itemsize": itemsize, "align": True}
    )
    h_hitgroup_sbt = np.array([(0)], dtype=dtype)
    optix.sbtRecordPackHeader(hitgroup_prog_group, h_hitgroup_sbt)
    global d_hitgroup_sbt
    d_hitgroup_sbt = array_to_device_memory(h_hitgroup_sbt)

    sbt = optix.ShaderBindingTable()
    sbt.raygenRecord = d_raygen_sbt.ptr
    sbt.missRecordBase = d_miss_sbt.ptr
    sbt.missRecordStrideInBytes = d_miss_sbt.mem.size
    sbt.missRecordCount = 1
    sbt.hitgroupRecordBase = d_hitgroup_sbt.ptr
    sbt.hitgroupRecordStrideInBytes = d_hitgroup_sbt.mem.size
    sbt.hitgroupRecordCount = 1
    return sbt


def launch(pipeline, sbt, trav_handle):
    print("Launching ... ")

    pix_bytes = pix_width * pix_height * 4

    h_pix = np.zeros((pix_width, pix_height, 4), "B")
    d_pix = cp.array(h_pix)

    params = [
        ("u8", "image", d_pix.data.ptr),
        ("u4", "image_width", pix_width),
        ("u4", "image_height", pix_height),
        ("f4", "cam_eye_x", 0),
        ("f4", "cam_eye_y", 0),
        ("f4", "cam_eye_z", 2.0),
        ("f4", "cam_U_x", 1.10457),
        ("f4", "cam_U_y", 0),
        ("f4", "cam_U_z", 0),
        ("f4", "cam_V_x", 0),
        ("f4", "cam_V_y", 0.828427),
        ("f4", "cam_V_z", 0),
        ("f4", "cam_W_x", 0),
        ("f4", "cam_W_y", 0),
        ("f4", "cam_W_z", -2.0),
        ("u8", "trav_handle", trav_handle),
    ]

    formats = [x[0] for x in params]
    names = [x[1] for x in params]
    values = [x[2] for x in params]
    itemsize = get_aligned_itemsize(formats, 8)
    params_dtype = np.dtype(
        {"names": names, "formats": formats, "itemsize": itemsize, "align": True}
    )
    h_params = np.array([tuple(values)], dtype=params_dtype)
    d_params = array_to_device_memory(h_params)

    stream = cp.cuda.Stream()
    optix.launch(
        pipeline,
        stream.ptr,
        d_params.ptr,
        h_params.dtype.itemsize,
        sbt,
        pix_width,
        pix_height,
        1,  # depth
    )

    stream.synchronize()

    h_pix = cp.asnumpy(d_pix)
    return h_pix


# Numba compilation
# -----------------

# An equivalent to the compile_cuda function for Python kernels. The types of
# the arguments to the kernel must be provided, if there are any.


def compile_numba(f, sig=(), fastmath=True, debug=False, lineinfo=False):
    # Based on numba.cuda.compile_ptx. We don't just use
    # compile_ptx_for_current_device because it generates a kernel with a
    # mangled name. For proceeding beyond this prototype, an option should be
    # added to compile_ptx in Numba to not mangle the function name.

    nvvm_options = {
        "debug": debug,
        "lineinfo": lineinfo,
        "fastmath": fastmath,
        "opt": 0 if debug else 3,
    }

    cres = numba_compile_cuda(
        f,
        None,
        sig,
        fastmath=fastmath,
        debug=debug,
        lineinfo=lineinfo,
        nvvm_options=nvvm_options,
    )
    fname = cres.fndesc.llvm_func_name
    tgt = cres.target_context
    filename = cres.type_annotation.filename
    linenum = int(cres.type_annotation.linenum)
    lib, kernel = tgt.prepare_cuda_kernel(
        cres.library, cres.fndesc, debug, nvvm_options, filename, linenum
    )
    cc = get_current_device().compute_capability
    ptx = lib.get_asm_str(cc=cc)

    # Demangle name
    mangled_name = kernel.name
    original_name = cres.library.name
    return ptx.replace(mangled_name, original_name)


# -------------------------------------------------------------------------------
#
# User code / kernel - the following section is what we'd expect a user of
# PyOptiX to write.
#
# -------------------------------------------------------------------------------

# Helpers
@cuda.jit(device=True, fastmath=True)
def toSRGB(c):
    # Use float32 for constants
    invGamma = float32(1.0) / float32(2.4)
    powed = make_float3(
        math.pow(c.x, invGamma), math.pow(c.y, invGamma), math.pow(c.z, invGamma)
    )

    return make_float3(
        float32(12.92) * c.x
        if c.x < float32(0.0031308)
        else float32(1.055) * powed.x - float32(0.055),
        float32(12.92) * c.y
        if c.y < float32(0.0031308)
        else float32(1.055) * powed.y - float32(0.055),
        float32(12.92) * c.z
        if c.z < float32(0.0031308)
        else float32(1.055) * powed.z - float32(0.055),
    )


@cuda.jit(device=True, fastmath=True)
def quantizeUnsigned8Bits(x):
    x = clamp(x, float32(0.0), float32(1.0))
    N, Np1 = (1 << 8) - 1, 1 << 8
    return uint8(min(uint32(x * float32(Np1)), uint32(N)))


@cuda.jit(device=True, fastmath=True)
def make_color(c):
    srgb = toSRGB(clamp(c, float32(0.0), float32(1.0)))
    return make_uchar4(
        quantizeUnsigned8Bits(srgb.x),
        quantizeUnsigned8Bits(srgb.y),
        quantizeUnsigned8Bits(srgb.z),
        uint8(255),
    )


# ray functions
@cuda.jit(device=True, fastmath=True)
def setPayload(p):
    optix.SetPayload_0(float_as_int(p.x))
    optix.SetPayload_1(float_as_int(p.y))
    optix.SetPayload_2(float_as_int(p.z))


@cuda.jit(device=True, fastmath=True)
def computeRay(idx, dim):
    U = params.cam_u
    V = params.cam_v
    W = params.cam_w
    # Normalizing coordinates to [-1.0, 1.0]
    d = float32(2.0) * make_float2(
        float32(idx.x) / float32(dim.x), float32(idx.y) / float32(dim.y)
    ) - float32(1.0)

    origin = params.cam_eye
    direction = normalize(d.x * U + d.y * V + W)
    return origin, direction


def __raygen__rg():
    # Lookup our location within the launch grid
    idx = optix.GetLaunchIndex()
    dim = optix.GetLaunchDimensions()

    # Map our launch idx to a screen location and create a ray from the camera
    # location through the screen
    ray_origin, ray_direction = computeRay(make_uint3(idx.x, idx.y, 0), dim)

    # Trace the ray against our scene hierarchy
    payload_pack = optix.Trace(
        params.handle,
        ray_origin,
        ray_direction,
        float32(0.0),  # Min intersection distance
        float32(1e16),  # Max intersection distance
        float32(0.0),  # rayTime -- used for motion blur
        OptixVisibilityMask(255),  # Specify always visible
        OptixRayFlags.OPTIX_RAY_FLAG_NONE,
        uint32(0),  # SBT offset   -- See SBT discussion
        uint32(1),  # SBT stride   -- See SBT discussion
        uint32(0),  # missSBTIndex -- See SBT discussion
    )
    result = make_float3(
        int_as_float(payload_pack.p0),
        int_as_float(payload_pack.p1),
        int_as_float(payload_pack.p2),
    )

    # Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color(result)


def __miss__ms():
    miss_data = MissDataStruct(optix.GetSbtDataPointer())
    setPayload(miss_data.bg_color)


def __closesthit__ch():
    # When built-in triangle intersection is used, a number of fundamental
    # attributes are provided by the OptiX API, indlucing barycentric coordinates.
    barycentrics = optix.GetTriangleBarycentrics()

    setPayload(make_float3(barycentrics, float32(1.0)))


# -------------------------------------------------------------------------------
#
# main
#
# -------------------------------------------------------------------------------


def main():
    raygen_ptx = compile_numba(__raygen__rg, fastmath=True)
    miss_ptx = compile_numba(__miss__ms, fastmath=True)
    hitgroup_ptx = compile_numba(__closesthit__ch, fastmath=True)

    init_optix()

    ctx = create_ctx()
    gas_handle, d_gas_output_buffer = create_accel(ctx)
    pipeline_options = set_pipeline_options()

    raygen_module = create_module(ctx, pipeline_options, raygen_ptx)
    miss_module = create_module(ctx, pipeline_options, miss_ptx)
    hitgroup_module = create_module(ctx, pipeline_options, hitgroup_ptx)

    prog_groups = create_program_groups(
        ctx, raygen_module, miss_module, hitgroup_module
    )
    pipeline = create_pipeline(ctx, prog_groups, pipeline_options)
    sbt = create_sbt(prog_groups)
    pix = launch(pipeline, sbt, gas_handle)

    print("Total number of log messages: {}".format(logger.num_mssgs))

    pix = pix.reshape((pix_height, pix_width, 4))  # PIL expects [ y, x ] resolution
    img = ImageOps.flip(Image.fromarray(pix, "RGBA"))  # PIL expects y = 0 at bottom
    img.save("triangle.png")
    img.show()


if __name__ == "__main__":
    main()
