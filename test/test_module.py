import optix as ox
import cupy as cp

import array
import pytest

import sample_ptx
import tutil 


if tutil.optix_version_gte( (7,2) ): 
    class TestModuleCompileBoundValyeEntry:
        def test_compile_bound_value_entry( self ):
            bound_value = array.array( 'f', [0.1, 0.2, 0.3] )
            bound_value_entry = ox.ModuleCompileBoundValueEntry(
                pipelineParamOffsetInBytes = 4,
                boundValue  = bound_value,
                annotation  = "my_bound_value"
            )

            assert bound_value_entry.pipelineParamOffsetInBytes == 4
            with pytest.raises( AttributeError ):
                print( bound_value_entry.boundValue )
            assert bound_value_entry.annotation == "my_bound_value"

            bound_value_entry.pipelineParamOffsetInBytes = 8
            assert bound_value_entry.pipelineParamOffsetInBytes == 8
            bound_value_entry.annotation = "new_bound_value"
            assert bound_value_entry.annotation == "new_bound_value"


class TestModule:
    if tutil.optix_version_gte( (7,2) ): 
        def test_options( self ):
            mod_opts = ox.ModuleCompileOptions(
                maxRegisterCount = 64,
                optLevel         = ox.COMPILE_OPTIMIZATION_LEVEL_1,
                debugLevel       = tutil.default_debug_level(),
                boundValues      = []
            )
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level() 
            # ox.ModuleCompileOptions.boundValues is write-only
            with pytest.raises( AttributeError ):
                print( mod_opts.boundValues )

            mod_opts = ox.ModuleCompileOptions()
            assert mod_opts.maxRegisterCount == ox.COMPILE_DEFAULT_MAX_REGISTER_COUNT
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_DEFAULT
            assert mod_opts.debugLevel       == tutil.default_debug_level()
            mod_opts.maxRegisterCount = 64
            mod_opts.optLevel         = ox.COMPILE_OPTIMIZATION_LEVEL_1
            mod_opts.debugLevel       = tutil.default_debug_level()
            mod_opts.boundValues = [ ox.ModuleCompileBoundValueEntry() ];
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()
    elif tutil.optix_version_gte( (7,1) ): 
        def test_options( self ):
            mod_opts = ox.ModuleCompileOptions(
                maxRegisterCount = 64,
                optLevel         = ox.COMPILE_OPTIMIZATION_LEVEL_1,
                debugLevel       = tutil.default_debug_level()
            )
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()

            mod_opts = ox.ModuleCompileOptions()
            assert mod_opts.maxRegisterCount == ox.COMPILE_DEFAULT_MAX_REGISTER_COUNT
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_DEFAULT
            assert mod_opts.debugLevel       == ox.COMPILE_DEBUG_LEVEL_DEFAULT
            mod_opts.maxRegisterCount = 64
            mod_opts.optLevel         = ox.COMPILE_OPTIMIZATION_LEVEL_1
            mod_opts.debugLevel       = tutil.default_debug_level()
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()
    else:
        def test_options( self ):
            mod_opts = ox.ModuleCompileOptions(
                maxRegisterCount = 64,
                optLevel         = ox.COMPILE_OPTIMIZATION_LEVEL_1,
                debugLevel       = tutil.default_debug_level()
            )
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()

            mod_opts = ox.ModuleCompileOptions()
            assert mod_opts.maxRegisterCount == ox.COMPILE_DEFAULT_MAX_REGISTER_COUNT
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_DEFAULT
            assert mod_opts.debugLevel       == tutil.default_debug_level()
            mod_opts.maxRegisterCount = 64
            mod_opts.optLevel         = ox.COMPILE_OPTIMIZATION_LEVEL_1
            mod_opts.debugLevel       = ox.COMPILE_DEBUG_LEVEL_FULL
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == ox.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == ox.COMPILE_DEBUG_LEVEL_FULL

    def test_create_destroy( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        module_opts   = ox.ModuleCompileOptions()
        pipeline_opts = ox.PipelineCompileOptions()
        mod, log = ctx.moduleCreateFromPTX(
            module_opts,
            pipeline_opts,
            sample_ptx.sample_ptx,
            )
        assert type(mod) is ox.Module
        assert type(log) is str

        mod.destroy()
        ctx.destroy()

    if tutil.optix_version_gte( (7,1) ): 
        def test_builtin_is_module_get( self ):
            ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
            module_opts     = ox.ModuleCompileOptions()
            pipeline_opts   = ox.PipelineCompileOptions()
            builtin_is_opts = ox.BuiltinISOptions()
            builtin_is_opts.builtinISModuleType = ox.PRIMITIVE_TYPE_TRIANGLE

            is_mod = ctx.builtinISModuleGet(
                module_opts,
                pipeline_opts,
                builtin_is_opts
            )
            assert type( is_mod ) is ox.Module
            is_mod.destroy()
            ctx.destroy()
