#!/usr/bin/env python3

#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import optix
import os
import cupy  as cp    # CUDA bindings
import numpy as np    # Packing of structures in C-compatible format

import array
import ctypes         # C interop helpers
from PIL import Image, ImageOps # Image IO
from pynvrtc.compiler import Program

import path_util


#-------------------------------------------------------------------------------
#
# Util 
#
#-------------------------------------------------------------------------------


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


def optix_version_gte( version ):
    if optix.version()[0] >  version[0]:
        return True
    if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
        return True
    return False


def compile_cuda( cuda_file ):
    with open( cuda_file, 'rb' ) as f:
        src = f.read()
    nvrtc_dll = os.environ.get('NVRTC_DLL')
    if nvrtc_dll is None:
        nvrtc_dll = ''
    print("NVRTC_DLL = {}".format(nvrtc_dll))
    prog = Program( src.decode(), cuda_file,
                    lib_name= nvrtc_dll )
    compile_options = [
        '-use_fast_math', 
        '-lineinfo',
        '-default-device',
        '-std=c++11',
        '-rdc',
        'true',
        #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
        f'-I{path_util.cuda_tk_path}',
        f'-I{path_util.include_path}'
    ]
    # Optix 7.0 compiles need path to system stddef.h
    # the value of optix.stddef_path is compiled in constant. When building
    # the module, the value can be specified via an environment variable, e.g.
    #   export PYOPTIX_STDDEF_DIR="/usr/include/linux"
    if (optix.version()[1] == 0):
        compile_options.append( f'-I{path_util.stddef_path}' )

    ptx  = prog.compile( compile_options )
    return ptx


#-------------------------------------------------------------------------------
#
# Optix setup
#
#-------------------------------------------------------------------------------

pix_width = 768
pix_height = 768


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
    if optix.version()[1] >= 2:
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL 

    cu_ctx = 0 
    return optix.deviceContextCreate( cu_ctx, ctx_options )


def build_triangle_gas( ctx ):

    NUM_KEYS = 3

    motion_options           = optix.MotionOptions()
    motion_options.numKeys   = NUM_KEYS
    motion_options.timeBegin = 0.0
    motion_options.timeEnd   = 1.0
    motion_options.flags     = optix.MOTION_FLAG_NONE

    accel_options = optix.AccelBuildOptions(
        buildFlags = int( optix.BUILD_FLAG_ALLOW_COMPACTION ),
        operation  = optix.BUILD_OPERATION_BUILD,
        motionOptions = motion_options
        )    

    NUM_VERTS = 3
    global vertices
    vertices_0 = cp.array( [ 
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.5, 1.0, 0.0, 0.0,
        ], dtype = 'f4' )

    vertices_1 = cp.array( [
        0.5, 0.0, 0.0, 0.0,
        1.5, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0,
        ], dtype = 'f4' )

    vertices_2 = cp.array( [
        0.5, -0.5, 0.0, 0.0,
        1.5, -0.5, 0.0, 0.0,
        1.0, 0.5, 0.0, 0.0
        ], dtype = 'f4' )

    triangle_input_flags                = [ optix.GEOMETRY_FLAG_DISABLE_ANYHIT ]
    triangle_input                      = optix.BuildInputTriangleArray()
    triangle_input.vertexFormat         = optix.VERTEX_FORMAT_FLOAT3
    triangle_input.numVertices          = NUM_VERTS
    triangle_input.vertexBuffers        = [ vertices_0.data.ptr, vertices_1.data.ptr, vertices_2.data.ptr ]
    triangle_input.flags                = triangle_input_flags
    triangle_input.numSbtRecords        = 1
    triangle_input.sbtIndexOffsetBuffer = 0

    gas_buffer_sizes = ctx.accelComputeMemoryUsage( [accel_options], [triangle_input ] )
    d_temp_buffer    = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes )
    d_output_buffer  = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes ) 

    d_result = cp.array( [ 0 ], dtype = 'u8' )
    emit_property = optix.AccelEmitDesc(
        type = optix.PROPERTY_TYPE_COMPACTED_SIZE,
        result = d_result.data.ptr
        )

    gas_handle = ctx.accelBuild(
        0,  # CUDA stream
        [ accel_options ],
        [ triangle_input ],
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        [ emit_property ]
    )

    compacted_gas_size = cp.asnumpy( d_result )

    if compacted_gas_size < gas_buffer_sizes.outputSizeInBytes:

        d_compacted_output_buffer = cp.cuda.alloc( compacted_gas_size )
        compacted_gas_handle = ctx.accelCompact( 
            0,  #CUDA stream
            gas_handle,
            d_compacted_output_buffer,
            compacted_gas_size
        )
        return compacted_gas_handle, d_compacted_output_buffer
    else:
        return gas_handle, d_output_buffer


def build_sphere_gas(ctx):

    accel_options = optix.AccelBuildOptions(
        buildFlags = int( optix.BUILD_FLAG_ALLOW_COMPACTION ),
        operation  = optix.BUILD_OPERATION_BUILD,
    )  

    aabb = cp.array( [ 
        -1.5, -1.0, -0.5,
        -0.5, 0.0, 0.5
        ], dtype = 'f4')

    sphere_input_flags = [ optix.GEOMETRY_FLAG_DISABLE_ANYHIT ]
    sphere_input = optix.BuildInputCustomPrimitiveArray(
        aabbBuffers = [ aabb.data.ptr ],
        numPrimitives = 1,
        flags = sphere_input_flags,
        numSbtRecords = 1
    )

    gas_buffer_sizes = ctx.accelComputeMemoryUsage( [accel_options], [sphere_input] )
    d_temp_buffer    = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes )
    d_output_buffer  = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes )

    d_result = cp.array( [ 0 ], dtype = 'u8' )
    emit_property = optix.AccelEmitDesc(
        type = optix.PROPERTY_TYPE_COMPACTED_SIZE,
        result = d_result.data.ptr
        )

    sphere_gas_handle = ctx.accelBuild(
        0,  # CUDA stream
        [ accel_options ],
        [ sphere_input ],
        d_temp_buffer.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [ emit_property ]
        )

    compacted_gas_size = cp.asnumpy( d_result )

    if compacted_gas_size < gas_buffer_sizes.outputSizeInBytes:
        d_final_output_buffer = cp.cuda.alloc( compacted_gas_size )
        final_gas_handle = ctx.accelCompact( 
            0,  #CUDA stream
            sphere_gas_handle,
            d_final_output_buffer,
            compacted_gas_size
        )
    else:
        d_final_output_buffer = d_output_buffer
        final_gas_handle = sphere_gas_handle

    motion_keys = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.5,
        0.0, 0.0, 1.0, 0.0
    ]

    motion_options           = optix.MotionOptions()
    motion_options.numKeys   = 2
    motion_options.timeBegin = 0.0
    motion_options.timeEnd   = 1.0
    motion_options.flags     = optix.MOTION_FLAG_NONE

    motion_transform         = optix.MatrixMotionTransform( 
                                final_gas_handle,
                                motion_options,
                                motion_keys
                                )

    xform_bytes = optix.getDeviceRepresentation( motion_transform )
    d_sphere_motion_transform = cp.array( np.frombuffer( xform_bytes, dtype='B' ) )

    sphere_motion_transform_handle = optix.convertPointerToTraversableHandle(
                                    ctx,
                                    d_sphere_motion_transform.data.ptr,
                                    optix.TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM
                                    )

    return sphere_motion_transform_handle, d_final_output_buffer


def build_ias(ctx, sphere_motion_transform_handle, triangle_gas_handle):

    instance_data = [ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ]

    sphere_instance                   = optix.Instance(instance_data)
    sphere_instance.flags             = optix.INSTANCE_FLAG_NONE
    sphere_instance.instanceId        = 1
    sphere_instance.sbtOffset         = 0
    sphere_instance.visibilityMask    = 1
    sphere_instance.traversableHandle = sphere_motion_transform_handle 

    triangle_instance                   = optix.Instance(instance_data)
    triangle_instance.flags             = optix.INSTANCE_FLAG_NONE
    triangle_instance.instanceId        = 0
    triangle_instance.sbtOffset         = 1
    triangle_instance.visibilityMask    = 1
    triangle_instance.traversableHandle = triangle_gas_handle

    instances       = [ sphere_instance, triangle_instance ]
    instances_bytes = optix.getDeviceRepresentation( instances ) 

    d_instances = cp.array( np.frombuffer( instances_bytes, dtype='B' ) )

    instance_input = optix.BuildInputInstanceArray()
    instance_input.instances    = d_instances.data.ptr
    instance_input.numInstances = len(instances) 

    accel_options = optix.AccelBuildOptions() 
    accel_options.buildFlags              = optix.BUILD_FLAG_NONE
    accel_options.operation               = optix.BUILD_OPERATION_BUILD

    accel_options.motionOptions.numKeys   = 2
    accel_options.motionOptions.timeBegin = 0.0
    accel_options.motionOptions.timeEnd   = 1.0
    accel_options.motionOptions.flags     = optix.MOTION_FLAG_NONE

    ias_buffer_sizes = ctx.accelComputeMemoryUsage( [accel_options], [instance_input] )
    d_temp_buffer  = cp.cuda.alloc( ias_buffer_sizes.tempSizeInBytes ) 
    d_ias_output_buffer = cp.cuda.alloc( ias_buffer_sizes.outputSizeInBytes )

    ias_handle = ctx.accelBuild(
        0,    # CUDA stream
        [ accel_options ], 
        [ instance_input ],   
        d_temp_buffer.ptr,
        ias_buffer_sizes.tempSizeInBytes,
        d_ias_output_buffer.ptr,
        ias_buffer_sizes.outputSizeInBytes,
        [] # emitted properties
        )

    return ias_handle, d_ias_output_buffer


def create_module(ctx):

    module_compile_options = optix.ModuleCompileOptions()
    module_compile_options.maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT
    module_compile_options.optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT
    module_compile_options.debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT

    pipeline_compile_options = optix.PipelineCompileOptions()
    pipeline_compile_options.numPayloadValues = 3
    pipeline_compile_options.numAttributeValues = 3
    pipeline_compile_options.usesMotionBlur = True
    pipeline_compile_options.exceptionFlags = optix.EXCEPTION_FLAG_NONE
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params"

    simple_motion_blur_cu = os.path.join(os.path.dirname(__file__), 'simpleMotionBlur.cu')
    simple_motion_blur_ptx = compile_cuda( simple_motion_blur_cu )

    module, log = ctx.moduleCreateFromPTX(
        module_compile_options,
        pipeline_compile_options,
        simple_motion_blur_ptx
    )

    return module, pipeline_compile_options


def create_program_groups(ctx, module):

    program_group_options = optix.ProgramGroupOptions()
    
    raygen_program_group_desc = optix.ProgramGroupDesc()
    raygen_program_group_desc.raygenModule = module
    raygen_program_group_desc.raygenEntryFunctionName = "__raygen__rg"

    log = None
    raygen_program_group = None
    if optix_version_gte( (7,4) ):
        program_group_options = optix.ProgramGroupOptions()
        raygen_prog_group, log = ctx.programGroupCreate(
            [ raygen_program_group_desc ],
            program_group_options
            )
    else:
        raygen_prog_group, log = ctx.programGroupCreate(
            [ raygen_program_group_desc ]
            )
    print( "\tProgramGroup raygen create log: <<<{}>>>".format( log ) )

    miss_prog_group_desc                        = optix.ProgramGroupDesc()
    miss_prog_group_desc.missModule             = module
    miss_prog_group_desc.missEntryFunctionName  = "__miss__camera"
    miss_prog_group = None
    if optix_version_gte( (7,4) ):
        program_group_options = optix.ProgramGroupOptions()
        miss_prog_group, log = ctx.programGroupCreate(
            [ miss_prog_group_desc ],
            program_group_options
            )
    else:
        miss_prog_group, log = ctx.programGroupCreate(
            [ miss_prog_group_desc ]
            )
    print( "\tProgramGroup mis create log: <<<{}>>>".format( log ) )

    hitgroup_prog_group_desc                             = optix.ProgramGroupDesc()
    hitgroup_prog_group_desc.hitgroupModuleCH            = module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__camera"
    hitgroup_prog_group_desc.hitgroupModuleIS            = module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameIS = "__intersection__sphere"

    triangle_hitgroup_prog_group = None
    if optix_version_gte( (7,4) ):
        program_group_options = optix.ProgramGroupOptions()
        triangle_hitgroup_prog_group, log = ctx.programGroupCreate(
            [ hitgroup_prog_group_desc ],
            program_group_options
            )
    else:
        triangle_hitgroup_prog_group, log = ctx.programGroupCreate(
            [ hitgroup_prog_group_desc ]
            )
    
    sphere_hitgroup_prog_group = None
    if optix_version_gte( (7,4) ):
        program_group_options = optix.ProgramGroupOptions()
        sphere_hitgroup_prog_group, log = ctx.programGroupCreate(
            [ hitgroup_prog_group_desc ],
            program_group_options
            )
    else:
        sphere_hitgroup_prog_group, log = ctx.programGroupCreate(
            [ hitgroup_prog_group_desc ]
            )
    print( "\tProgramGroup hitgroup create log: <<<{}>>>".format( log ) )

    return [ raygen_prog_group[0], miss_prog_group[0], \
        triangle_hitgroup_prog_group[0], sphere_hitgroup_prog_group[0] ]


def create_pipeline(ctx, program_groups, pipeline_compile_options):

    pipeline_link_options = optix.PipelineLinkOptions()
    pipeline_link_options.maxTraceDepth = 2
    pipeline_link_options.debugLevel = optix.COMPILE_DEBUG_LEVEL_FULL

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

    ( dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size ) = \
        optix.util.computeStackSizes(
            stack_sizes,
            1,  # maxTraceDepth
            0,  # maxCCDepth
            0   # maxDCDepth
            )

    pipeline.setStackSize(
        dc_stack_size_from_trav,
        dc_stack_size_from_state,
        cc_stack_size,
        3   # maxTraversableDepth ( 3 since largest depth is IAS->MT->GAS )
    )

    return pipeline


def create_sbt( program_groups ):
    print( "Creating sbt ... " )

    (raygen_prog_group, miss_prog_group, triangle_hit_prog_group, \
        sphere_hit_prog_group) = program_groups

    header_format = '{}B'.format( optix.SBT_RECORD_HEADER_SIZE )

    #
    # raygen record
    #
    formats = [ header_format ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( {
        'names'     : ['header'],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True 
        } )
    h_raygen_sbt = np.array( [ 0 ], dtype = dtype )
    optix.sbtRecordPackHeader( raygen_prog_group, h_raygen_sbt )
    global d_raygen_sbt 
    d_rayen_sbt = array_to_device_memory( h_raygen_sbt )

    #
    # miss records
    #
    formats = [ header_format, 'f4','f4','f4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( {
        'names'     : [ 'header', 'r', 'g', 'b' ],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True 
        } )
    h_miss_sbt = np.array( [ (0, 0.1, 0.1, 0.1 ) ], dtype=dtype )
    optix.sbtRecordPackHeader( miss_prog_group, h_miss_sbt )
    global d_miss_sbt
    d_miss_sbt = array_to_device_memory( h_miss_sbt )

    #
    # hit group records
    #
    formats = [ header_format, 'f4','f4','f4',
                               'f4','f4','f4',
                               'f4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    hit_sbt_dtype = np.dtype( {
        'names'     : [ 'header','r','g','b',
                                 'x','y','z',
                                 'rad' ],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'aign'      : True
        } )
    h_sphere_hitgroup_sbt = np.array( [ (0, 0.9,  0.1, 0.1,
                                           -1.0, -0.5, 0.1,
                                            0.5 
                                            ) ], dtype=hit_sbt_dtype )

    h_triangle_hitgroup_sbt = np.array( [ (0, 0.1, 0.1, 0.9,
                                              0.0, 0.0, 0.0,    # unused
                                              0.0               # unused 
                                              ) ], dtype=hit_sbt_dtype )

    optix.sbtRecordPackHeader( sphere_hit_prog_group, h_sphere_hitgroup_sbt )
    optix.sbtRecordPackHeader( triangle_hit_prog_group, h_triangle_hitgroup_sbt )

    h_hitgroup_sbt = np.array( [ h_sphere_hitgroup_sbt, h_triangle_hitgroup_sbt ], 
        dtype=hit_sbt_dtype )

    d_hitgroup_sbt = array_to_device_memory( h_hitgroup_sbt )

    return optix.ShaderBindingTable(
        raygenRecord                = d_rayen_sbt.ptr,
        missRecordBase              = d_miss_sbt.ptr,
        missRecordStrideInBytes     = d_miss_sbt.mem.size,
        missRecordCount             = 1,
        hitgroupRecordBase          = d_hitgroup_sbt.ptr,
        hitgroupRecordStrideInBytes = d_hitgroup_sbt.mem.size,
        hitgroupRecordCount         = 2     # sphere, triangle
    )


def launch( pipeline, sbt, trav_handle ):
    print( "Launching ... " )

    pix_bytes = pix_width * pix_height * 4

    h_pix = np.zeros( (pix_width, pix_height, 4 ), 'B' )
    h_pix[0:pix_width, 0:pix_height] = [255, 128, 0, 255]
    d_pix = cp.array( h_pix )

    params = [
        ( 'u4', 'image_width',       pix_width ),
        ( 'u4', 'image_height',     pix_height ),
        ( 'u8', 'accum',        d_pix.data.ptr ),
        ( 'u8', 'frame',        d_pix.data.ptr ),
        ( 'u4', 'subframe index',            0 ),
        ( 'f4', 'cam_eye_x',                 0 ),
        ( 'f4', 'cam_eye_y',                 0 ),
        ( 'f4', 'cam_eye_z',               5.0 ),
        ( 'f4', 'cam_U_x',      1.10457        ),
        ( 'f4', 'cam_U_y',      0              ),
        ( 'f4', 'cam_U_z',      0              ),
        ( 'f4', 'cam_V_x',      0              ),
        ( 'f4', 'cam_V_y',      0.828427       ),
        ( 'f4', 'cam_V_z',      0              ),
        ( 'f4', 'cam_W_x',      0              ),
        ( 'f4', 'cam_W_y',      0              ),
        ( 'f4', 'cam_W_z',      -2.0           ),
        ( 'u8', 'trav_handle',  trav_handle    )
    ]

    formats = [ x[0] for x in params ] 
    names   = [ x[1] for x in params ] 
    values  = [ x[2] for x in params ] 
    itemsize = get_aligned_itemsize( formats, 8 )
    params_dtype = np.dtype( { 
        'names'   : names, 
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_params = np.array( [ tuple(values) ], dtype=params_dtype )
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

    ctx                                = create_ctx()
    tri_gas_handle, d_tri_gas_buffer   = build_triangle_gas(ctx)
    sphere_motion_transform_handle, d_sphere_gas_buffer = build_sphere_gas(ctx)
    ias_handle, d_ias_buffer           = build_ias(ctx, sphere_motion_transform_handle, tri_gas_handle)
    module, pipeline_compile_options   = create_module(ctx)
    program_groups                     = create_program_groups(ctx, module)
    pipeline                           = create_pipeline(ctx, program_groups, pipeline_compile_options)
    sbt                                = create_sbt( program_groups ) 
    pix                                = launch( pipeline, sbt, ias_handle ) 

    print( "Total number of log messages: {}".format( logger.num_mssgs ) )

    pix = pix.reshape( ( pix_height, pix_width, 4 ) )     # PIL expects [ y, x ] resolution
    img = ImageOps.flip( Image.fromarray( pix, 'RGBA' ) ) # PIL expects y = 0 at bottom
    img.show()
    img.save( 'my.png' )


if __name__ == "__main__":
    main()
