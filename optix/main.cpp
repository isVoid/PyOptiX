
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NOMINMAX 
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>


namespace py = pybind11;

#define PYOPTIX_CHECK( call )                                                  \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
            throw std::runtime_error( optixGetErrorString( res )  );           \
    } while( 0 )

#define PYOPTIX_CHECK_LOG( call )                                              \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
            throw std::runtime_error(                                          \
                    std::string( optixGetErrorString( res ) ) +                \
                    ": " + log_buf                                             \
                    );                                                         \
    } while( 0 )



#define COMMA ,

#if OPTIX_VERSION >= 70100
#    define IF_OPTIX71( code ) code
#    define IF_OPTIX71_ELSE( code0, code1 ) code0
#else
#    define IF_OPTIX71( code ) 
#    define IF_OPTIX71_ELSE( code0, code1 ) code1
#endif

#if OPTIX_VERSION >= 70200
#    define IF_OPTIX72( code ) code
#else
#    define IF_OPTIX72( code ) 
#endif

#if OPTIX_VERSION >= 70300
#    define IF_OPTIX73( code ) code
#else
#    define IF_OPTIX73( code ) 
#endif

#if OPTIX_VERSION >= 70400
#    define IF_OPTIX74( code ) code
#else
#    define IF_OPTIX74( code ) 
#endif



namespace pyoptix
{

void context_log_cb(
    unsigned int level,
    const char* tag,
    const char* message,
    void* cbdata
    );

void convertBuildInputs(
    py::list build_inputs_in,
    std::vector<OptixBuildInput>& build_inputs
    );
//------------------------------------------------------------------------------
//
// Opaque type struct wrappers
//
//------------------------------------------------------------------------------

struct DeviceContext
{
    OptixDeviceContext deviceContext = 0;
    py::object         logCallbackFunction;
};
bool operator==( const DeviceContext& a, const DeviceContext& b) { return a.deviceContext == b.deviceContext; }

struct Module
{

    OptixModule module = 0;
};
bool operator==( const Module& a, const Module& b) { return a.module == b.module; }

struct ProgramGroup
{
    OptixProgramGroup programGroup = 0;
};
bool operator==( const ProgramGroup& a, const ProgramGroup& b) { return a.programGroup== b.programGroup; }


struct Pipeline
{
    OptixPipeline pipeline = 0;
};
bool operator==( const Pipeline& a, const Pipeline& b) { return a.pipeline== b.pipeline; }

struct Denoiser
{
    OptixDenoiser denoiser = 0;
};
bool operator==( const Denoiser& a, const Denoiser& b) { return a.denoiser== b.denoiser; }

//------------------------------------------------------------------------------
//
// Proxy objets to modify some functionality in the optix param structs
//
//------------------------------------------------------------------------------

struct DeviceContextOptions
{
    DeviceContextOptions(
       py::object log_callback_function,
       int32_t    log_callback_level
       IF_OPTIX72( COMMA OptixDeviceContextValidationMode validation_mode )
       )
    {
        logCallbackFunction         = log_callback_function;
        if( !logCallbackFunction.is_none() )
        {
            options.logCallbackFunction = pyoptix::context_log_cb;
            options.logCallbackData     = logCallbackFunction.ptr();
        }

        options.logCallbackLevel    = log_callback_level;
        IF_OPTIX72( options.validationMode      = validation_mode; )
    }


    // Log callback needs additional backing
    py::object logCallbackFunction;
    OptixDeviceContextOptions options;
};


struct BuildInputTriangleArray
{
    BuildInputTriangleArray(
        const py::list&        vertexBuffers_, // list of CUdeviceptr
        OptixVertexFormat      vertexFormat,
        unsigned int           vertexStrideInBytes,
        CUdeviceptr            indexBuffer,
        unsigned int           numIndexTriplets,
        OptixIndicesFormat     indexFormat,
        unsigned int           indexStrideInBytes,
        CUdeviceptr            preTransform,
        const py::list&        flags_, // list of uint32_t
        unsigned int           numSbtRecords,
        CUdeviceptr            sbtIndexOffsetBuffer,
        unsigned int           sbtIndexOffsetSizeInBytes,
        unsigned int           sbtIndexOffsetStrideInBytes,
        unsigned int           primitiveIndexOffset
        IF_OPTIX71( COMMA OptixTransformFormat   transformFormat )
        )
    {
        memset(&build_input, 0, sizeof(OptixBuildInputTriangleArray));
        vertexBuffers                           = vertexBuffers_.cast<std::vector<CUdeviceptr> >();
        build_input.vertexFormat                = vertexFormat;
        build_input.vertexStrideInBytes         = vertexStrideInBytes;
        build_input.indexBuffer                 = indexBuffer;
        build_input.numIndexTriplets            = numIndexTriplets;
        build_input.indexFormat                 = indexFormat;
        build_input.indexStrideInBytes          = indexStrideInBytes;
        build_input.preTransform                = preTransform;
        flags                                   = flags_.cast<std::vector<unsigned int> >();
        build_input.numSbtRecords               = numSbtRecords;
        build_input.sbtIndexOffsetBuffer        = sbtIndexOffsetBuffer;
        build_input.sbtIndexOffsetSizeInBytes   = sbtIndexOffsetSizeInBytes;
        build_input.sbtIndexOffsetStrideInBytes = sbtIndexOffsetStrideInBytes;
        build_input.primitiveIndexOffset        = primitiveIndexOffset;
        build_input.numSbtRecords               = numSbtRecords;
        IF_OPTIX71( 
        build_input.transformFormat             = transformFormat;
        )
    }

    void sync()
    {
        build_input.vertexBuffers = vertexBuffers.data();
        build_input.flags         = flags.data();
    }


    std::vector<unsigned int> flags;
    std::vector<CUdeviceptr>  vertexBuffers;
    OptixBuildInputTriangleArray build_input;
};


#if OPTIX_VERSION >= 70200
struct BuildInputCurveArray
{
    BuildInputCurveArray(
        OptixPrimitiveType  curveType,
        unsigned int        numPrimitives, 
        const py::list&     vertexBuffers_,
        unsigned int        numVertices,
        unsigned int        vertexStrideInBytes,
        const py::list&     widthBuffers_,
        unsigned int        widthStrideInBytes,
        const py::list&     normalBuffers_,
        unsigned int        normalStrideInBytes,
        CUdeviceptr         indexBuffer,
        unsigned int        indexStrideInBytes,
        unsigned int        flag,
        unsigned int        primitiveIndexOffset
        )
    {
        memset(&build_input, 0, sizeof(OptixBuildInputCurveArray));
        build_input.curveType              = curveType;
        build_input.numPrimitives          = numPrimitives;
        vertexBuffers                      = vertexBuffers_.cast<std::vector<CUdeviceptr> >();
        build_input.numVertices            = numVertices;
        build_input.vertexStrideInBytes    = vertexStrideInBytes;
        widthBuffers                       = widthBuffers_.cast<std::vector<CUdeviceptr> >();
        build_input.widthStrideInBytes     = widthStrideInBytes;
        normalBuffers                      = normalBuffers_.cast<std::vector<CUdeviceptr> >();
        build_input.normalStrideInBytes    = normalStrideInBytes;
        build_input.indexBuffer            = indexBuffer;
        build_input.indexStrideInBytes     = indexStrideInBytes;
        build_input.flag                   = flag;
        build_input.primitiveIndexOffset   = primitiveIndexOffset;
    }
    
    void sync()
    {
        build_input.vertexBuffers = vertexBuffers.data();
        build_input.widthBuffers  = widthBuffers.data();
        build_input.normalBuffers = normalBuffers.data();
    }


    std::vector<CUdeviceptr>  vertexBuffers;
    std::vector<CUdeviceptr>  widthBuffers;
    std::vector<CUdeviceptr>  normalBuffers;
    OptixBuildInputCurveArray build_input;
};
#endif // OPTIX_VERSION >= 70200


struct BuildInputCustomPrimitiveArray
{
    BuildInputCustomPrimitiveArray(
        const py::list&     aabbBuffers_,    // list of CUdeviceptr 
        unsigned int        numPrimitives,
        unsigned int        strideInBytes,
        const py::list&     flags_,          // list of uint32_t
        unsigned int        numSbtRecords,
        CUdeviceptr         sbtIndexOffsetBuffer,
        unsigned int        sbtIndexOffsetSizeInBytes,
        unsigned int        sbtIndexOffsetStrideInBytes,
        unsigned int        primitiveIndexOffset
        )
    {
        aabbBuffers                             = aabbBuffers_.cast<std::vector<CUdeviceptr> >();
        build_input.numPrimitives               = numPrimitives;
        build_input.strideInBytes               = strideInBytes;
        flags                                   = flags_.cast<std::vector<unsigned int> >();
        build_input.numSbtRecords               = numSbtRecords;
        build_input.sbtIndexOffsetBuffer        = sbtIndexOffsetBuffer;
        build_input.sbtIndexOffsetSizeInBytes   = sbtIndexOffsetSizeInBytes;
        build_input.sbtIndexOffsetStrideInBytes = sbtIndexOffsetStrideInBytes;
        build_input.primitiveIndexOffset        = primitiveIndexOffset;
    }


    void sync()
    {
        build_input.aabbBuffers = aabbBuffers.data();
        build_input.flags       = flags.data();
    }


    std::vector<unsigned int> flags;
    std::vector<CUdeviceptr>  aabbBuffers;
    OptixBuildInputCustomPrimitiveArray build_input;
};


struct BuildInputInstanceArray
{
    BuildInputInstanceArray(
        CUdeviceptr     instances,
        CUdeviceptr     instancePointers,
        unsigned int    numInstances
        )
    {
        if( instances && instancePointers )
            throw std::runtime_error( 
                "BuildInputInstanceArray created with both instances and instance pointers" 
            );
        build_input.instances = instances;
        if( instances )
            setInstances( instances );
        if( instancePointers )
            setInstancePointers( instancePointers );
            
        build_input.numInstances = numInstances;
    }


    void setInstances( CUdeviceptr instances )
    {
        build_type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        build_input.instances = instances;
    }
    
    void setInstancePointers( CUdeviceptr instances )
    {
        build_type = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS;
        build_input.instances = instances;
    }

    OptixBuildInputType          build_type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    OptixBuildInputInstanceArray build_input{};
};


struct PipelineCompileOptions
{
    PipelineCompileOptions(
        bool      usesMotionBlur,
        uint32_t  traversableGraphFlags,
        int32_t   numPayloadValues,
        int32_t   numAttributeValues,
        uint32_t  exceptionFlags,
        const char* pipelineLaunchParamsVariableName_
        IF_OPTIX71( COMMA int32_t  usesPrimitiveTypeFlags )
        )
    {
        options.usesMotionBlur         = usesMotionBlur;
        options.traversableGraphFlags  = traversableGraphFlags;
        options.numPayloadValues       = numPayloadValues;
        options.numAttributeValues     = numAttributeValues;
        options.exceptionFlags         = exceptionFlags;
        IF_OPTIX71(
        options.usesPrimitiveTypeFlags = usesPrimitiveTypeFlags;
        )
        if( pipelineLaunchParamsVariableName_ )
            pipelineLaunchParamsVariableName =
                pipelineLaunchParamsVariableName_;
    }

    void sync()
    {
        options.pipelineLaunchParamsVariableName =
            pipelineLaunchParamsVariableName.c_str();
    }

    // Strings need extra backing
    std::string pipelineLaunchParamsVariableName;
    OptixPipelineCompileOptions options;
};


struct PipelineLinkOptions
{
    PipelineLinkOptions(
        unsigned int maxTraceDepth,
        OptixCompileDebugLevel debugLevel
        )
    {
        options.maxTraceDepth = maxTraceDepth;
        options.debugLevel    = debugLevel;
    }

    OptixPipelineLinkOptions options{};
};


struct ShaderBindingTable
{
    ShaderBindingTable(
        CUdeviceptr  raygenRecord,
        CUdeviceptr  exceptionRecord,
        CUdeviceptr  missRecordBase,
        unsigned int missRecordStrideInBytes,
        unsigned int missRecordCount,
        CUdeviceptr  hitgroupRecordBase,
        unsigned int hitgroupRecordStrideInBytes,
        unsigned int hitgroupRecordCount,
        CUdeviceptr  callablesRecordBase,
        unsigned int callablesRecordStrideInBytes,
        unsigned int callablesRecordCount
    )
    {
        sbt.raygenRecord                 = raygenRecord;
        sbt.exceptionRecord              = exceptionRecord;
        sbt.missRecordBase               = missRecordBase;
        sbt.missRecordStrideInBytes      = missRecordStrideInBytes;
        sbt.missRecordCount              = missRecordCount;
        sbt.hitgroupRecordBase           = hitgroupRecordBase;
        sbt.hitgroupRecordStrideInBytes  = hitgroupRecordStrideInBytes;
        sbt.hitgroupRecordCount          = hitgroupRecordCount;
        sbt.callablesRecordBase          = callablesRecordBase;
        sbt.callablesRecordStrideInBytes = callablesRecordStrideInBytes;
        sbt.callablesRecordCount         = callablesRecordCount;
    }



    OptixShaderBindingTable sbt;
};


#if OPTIX_VERSION >= 70200
struct ModuleCompileBoundValueEntry
{
    ModuleCompileBoundValueEntry(
        size_t pipelineParamOffsetInBytes,
        const py::buffer& boundValue,
        const std::string& annotation
        )
    {
        entry.pipelineParamOffsetInBytes = pipelineParamOffsetInBytes;
        setBoundValue( boundValue );
        setAnnotation( annotation );
    }


    ModuleCompileBoundValueEntry(const ModuleCompileBoundValueEntry& other)
    {
        value      = other.value;
        annotation = other.annotation;
        entry      = other.entry;
    }


    ModuleCompileBoundValueEntry( ModuleCompileBoundValueEntry&& other )
    {
        value      = std::move( other.value );
        annotation = std::move( other.annotation );
        entry      = other.entry;
    }


    void setBoundValue( const py::buffer& val )
    {
        py::buffer_info binfo = val.request();
        if( binfo.ndim != 1 )
            throw std::runtime_error(
                "Multi-dimensional array passed as value for "
                "optix.ModuleCompileBoundValueEntry.boundValue" );

        size_t byte_size = binfo.itemsize * binfo.shape[0];
        const std::byte* bytes = reinterpret_cast<const std::byte*>(binfo.ptr);
        value.clear();
        std::copy( bytes, bytes+byte_size, std::back_inserter( value ) );

    }


    void setAnnotation( const std::string& val )
    {
        annotation = val;
    }


    void sync()
    {
        entry.annotation    = annotation.c_str();
        entry.sizeInBytes   = value.size();
        entry.boundValuePtr = value.data();
    }


    OptixModuleCompileBoundValueEntry entry;
    std::string            annotation;
    std::vector<std::byte> value;
};
#endif // OPTIX_VERSION >= 70200


#if OPTIX_VERSION >= 70400
struct PayloadType
{
    PayloadType() {}
    PayloadType( const py::list&  payload_semantics )
    {
        setPayloadSemantics( payload_semantics );
    }


    void setPayloadSemantics( const py::list& val )
    {
        payload_semantics = val.cast<std::vector<uint32_t> >();
    }


    void sync()
    {
        payload_type.numPayloadValues = payload_semantics.size();
        if( !payload_semantics.empty() )
            payload_type.payloadSemantics = payload_semantics.data();
        else 
            payload_type.payloadSemantics = nullptr; 
    }


    OptixPayloadType  payload_type{};
    std::vector<uint32_t> payload_semantics;
};
#endif // OPTIX_VERSION >= 70400


struct ModuleCompileOptions
{
    ModuleCompileOptions(
        int32_t                       maxRegisterCount,
        OptixCompileOptimizationLevel optLevel,
        OptixCompileDebugLevel        debugLevel
        IF_OPTIX72( COMMA std::vector<pyoptix::ModuleCompileBoundValueEntry>&& bound_values )
        IF_OPTIX74( COMMA std::vector<pyoptix::PayloadType>&& payload_types )
        )
    {
        options.maxRegisterCount = maxRegisterCount;
        options.optLevel         = optLevel;
        options.debugLevel       = debugLevel;

        IF_OPTIX72(
        pyboundValues = std::move( bound_values );
        )

        IF_OPTIX74(
        pypayloadTypes = std::move( payload_types );
        )
    }


    void sync()
    {
#if OPTIX_VERSION >= 70200
        boundValues.clear();
        for( auto& pybve : pyboundValues )
        {
            pybve.sync();
            boundValues.push_back( pybve.entry );
        }

        options.boundValues    = boundValues.empty() ?
                                 nullptr             :
                                 boundValues.data();
        options.numBoundValues = static_cast<uint32_t>( boundValues.size() );
#endif 

#if OPTIX_VERSION >= 70400
        payloadTypes.clear();
        for( auto& pypt : pypayloadTypes )
        {
            pypt.sync();
            payloadTypes.push_back( pypt.payload_type);
        }

        std::cerr << "*****PAYLOAD TYPES SIZE: " << payloadTypes.size() << std::endl;
        options.payloadTypes    = payloadTypes.empty() ?
                                 nullptr             :
                                 payloadTypes.data();
        options.numPayloadTypes= static_cast<uint32_t>( payloadTypes.size() );
        std::cerr << "*****numPAYLOAD TYPES : " << options.numPayloadTypes << std::endl;
#endif
    }

    OptixModuleCompileOptions options{};

#if OPTIX_VERSION >= 70200
    std::vector<pyoptix::ModuleCompileBoundValueEntry> pyboundValues;
    std::vector<OptixModuleCompileBoundValueEntry>     boundValues;
#endif
#if OPTIX_VERSION >= 70400
    std::vector<pyoptix::PayloadType> pypayloadTypes;
    std::vector<OptixPayloadType>     payloadTypes;
#endif
};



#if OPTIX_VERSION >= 70100
struct BuiltinISOptions
{
    OptixBuiltinISOptions options;
};
#endif


struct ProgramGroupDesc
{
    ProgramGroupDesc(
        uint32_t                  flags,

        const char*               raygenEntryFunctionName,
        const pyoptix::Module     raygenModule,

        const char*               missEntryFunctionName,
        const pyoptix::Module     missModule,

        const char*               exceptionEntryFunctionName,
        const pyoptix::Module     exceptionModule,

        const char*               callablesEntryFunctionNameDC,
        const pyoptix::Module     callablesModuleDC,

        const char*               callablesEntryFunctionNameCC,
        const pyoptix::Module     callablesModuleCC,

        const char*               hitgroupEntryFunctionNameCH,
        const pyoptix::Module     hitgroupModuleCH,
        const char*               hitgroupEntryFunctionNameAH,
        const pyoptix::Module     hitgroupModuleAH,
        const char*               hitgroupEntryFunctionNameIS,
        const pyoptix::Module     hitgroupModuleIS
        )
    {
        program_group_desc.flags = flags;

        // TODO: check for bad inputs and throw exception (eg, passing in kind = RAYGEN and a missModule)
        if( raygenEntryFunctionName )
        {
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            entryFunctionName0 = raygenEntryFunctionName;
        }
        else if( missEntryFunctionName )
        {
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            entryFunctionName0 = missEntryFunctionName;
        }
        else if( exceptionEntryFunctionName )
        {
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
            entryFunctionName0 = exceptionEntryFunctionName;
        }
        else if( callablesEntryFunctionNameDC || callablesEntryFunctionNameCC )
        {
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            entryFunctionName0 = callablesEntryFunctionNameDC ? callablesEntryFunctionNameDC : "";
            entryFunctionName1 = callablesEntryFunctionNameCC ? callablesEntryFunctionNameCC : "";
        }
        else if( hitgroupEntryFunctionNameCH || hitgroupEntryFunctionNameAH || hitgroupEntryFunctionNameIS )
        {
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            entryFunctionName0 = hitgroupEntryFunctionNameCH ? hitgroupEntryFunctionNameCH : "";
            entryFunctionName1 = hitgroupEntryFunctionNameAH ? hitgroupEntryFunctionNameAH : "";
            entryFunctionName2 = hitgroupEntryFunctionNameIS ? hitgroupEntryFunctionNameIS : "";
        }

        if( raygenModule.module )
        {
            program_group_desc.raygen.module = raygenModule.module;
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        }
        else if( missModule.module )
        {
            program_group_desc.miss.module = missModule.module;
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        }
        else if( exceptionModule.module )
        {
            program_group_desc.exception.module = exceptionModule.module;
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        }
        else if( callablesModuleDC.module || callablesModuleCC.module )
        {
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            program_group_desc.callables.moduleDC = callablesModuleDC.module ? callablesModuleDC.module : nullptr;
            program_group_desc.callables.moduleCC = callablesModuleDC.module ? callablesModuleCC.module : nullptr;
        }
        else if( hitgroupModuleCH.module || hitgroupModuleAH.module || hitgroupModuleIS.module )
        {
            program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            program_group_desc.hitgroup.moduleCH = hitgroupModuleCH.module ? hitgroupModuleCH.module : nullptr;
            program_group_desc.hitgroup.moduleAH = hitgroupModuleAH.module ? hitgroupModuleAH.module : nullptr;
            program_group_desc.hitgroup.moduleIS = hitgroupModuleIS.module ? hitgroupModuleIS.module : nullptr;
        }
    }

    std::string entryFunctionName0;
    std::string entryFunctionName1;
    std::string entryFunctionName2;
    OptixProgramGroupDesc program_group_desc{};
};


#if OPTIX_VERSION >= 70400
struct ProgramGroupOptions
{
    ProgramGroupOptions( const pyoptix::PayloadType& payload_type )
    {
         setPayloadType( payload_type );
    }

    void setPayloadType( const pyoptix::PayloadType& payload_type_ )
    {
        payload_type = payload_type_.payload_type;
        if( payload_type.numPayloadValues> 0 )
            options.payloadType = &payload_type;
        else
            options.payloadType = nullptr; 
    }

    OptixPayloadType         payload_type{};
    OptixProgramGroupOptions options{};
};
#endif // OPTIX_VERSION >= 70400



//------------------------------------------------------------------------------
//
// Helpers
//
//------------------------------------------------------------------------------

constexpr size_t LOG_BUFFER_MAX_SIZE = 2048u;

void context_log_cb( unsigned int level, const char* tag, const char* message, void* cbdata  )
{
    py::object cb = py::reinterpret_borrow<py::object>(
        reinterpret_cast<PyObject*>( cbdata )
        );
    cb( level, tag, message );
}


void convertBuildInputs(
        py::list build_inputs_in,
        std::vector<OptixBuildInput>& build_inputs
        )
{
    build_inputs.resize( build_inputs_in.size() );
    int32_t idx = 0;
    for( auto list_elem : build_inputs_in )
    {
        if( py::isinstance<pyoptix::BuildInputTriangleArray>( list_elem ) )
        {
            pyoptix::BuildInputTriangleArray& tri_array= list_elem.cast<pyoptix::BuildInputTriangleArray&>();
            tri_array.sync();
            build_inputs[idx].type          = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build_inputs[idx].triangleArray = tri_array.build_input;
        }
#if OPTIX_VERSION >= 70100	
        else if( py::isinstance<pyoptix::BuildInputCurveArray>( list_elem ) )
        {
            pyoptix::BuildInputCurveArray& curve_array= list_elem.cast<pyoptix::BuildInputCurveArray&>();
            curve_array.sync();
            build_inputs[idx].type          = OPTIX_BUILD_INPUT_TYPE_CURVES;
            build_inputs[idx].curveArray    = curve_array.build_input;
        }
#endif
        else if( py::isinstance<pyoptix::BuildInputCustomPrimitiveArray>( list_elem ) )
        {
            pyoptix::BuildInputCustomPrimitiveArray& cp_array = 
                list_elem.cast<pyoptix::BuildInputCustomPrimitiveArray&>();

            cp_array.sync();
            build_inputs[idx].type                 = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
            build_inputs[idx].customPrimitiveArray = cp_array.build_input;
        }
        else if( py::isinstance<pyoptix::BuildInputInstanceArray>( list_elem ) )
        {
            pyoptix::BuildInputInstanceArray& inst_array = 
                list_elem.cast<pyoptix::BuildInputInstanceArray&>();
            build_inputs[idx].type          = inst_array.build_type; 
            build_inputs[idx].instanceArray = inst_array.build_input;
        }
        else
        {
            throw std::runtime_error(
                "Context.accelComputeMemoryUsage called with non-build input types"
                " in buildInputs param"
                );
        }

        ++idx;
    }
}

//------------------------------------------------------------------------------
//
// OptiX API error checked wrappers
//
//------------------------------------------------------------------------------

void init()
{
    PYOPTIX_CHECK( optixInit() );
}

py::tuple version()
{
    unsigned int major = OPTIX_VERSION / 10000;
    unsigned int minor = (OPTIX_VERSION % 10000) / 100;
    unsigned int micro = OPTIX_VERSION % 100;

    return py::make_tuple( major, minor, micro);
}

const char* getErrorName(
       OptixResult result
    )
{
    return optixGetErrorName( result );
}

const char* getErrorString(
       OptixResult result
    )
{
    return optixGetErrorString( result );
}

pyoptix::DeviceContext deviceContextCreate(
       uintptr_t fromContext,
       const pyoptix::DeviceContextOptions& options
    )
{
    pyoptix::DeviceContext ctx{};
    ctx.logCallbackFunction = options.logCallbackFunction;

    PYOPTIX_CHECK(
        optixDeviceContextCreate(
            reinterpret_cast<CUcontext>( fromContext ),
            &options.options,
            &(ctx.deviceContext)
        )
    );
    return ctx;
}

void deviceContextDestroy(
       pyoptix::DeviceContext context
    )
{
    PYOPTIX_CHECK(
        optixDeviceContextDestroy(
            context.deviceContext
        )
    );
}

py::object deviceContextGetProperty(
       pyoptix::DeviceContext context,
       OptixDeviceProperty property
    )
{
    switch( property )
    {
        // uint32_t
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH:
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH:
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS:
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS:
        case OPTIX_DEVICE_PROPERTY_RTCORE_VERSION:
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID:
        case OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK:
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS:
        case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET:
        {
            uint32_t value = 0u;
            PYOPTIX_CHECK(
                optixDeviceContextGetProperty(
                    context.deviceContext,
                    property,
                    &value,
                    sizeof( uint32_t )
                )
            );
            return py::int_( value );
        }
        default:
        {
            throw std::runtime_error(
                "Unrecognized optix.DeviceProperty passed to "
                "DeviceContext.getProperty()"
                );
        }
    }
}

void deviceContextSetLogCallback(
       pyoptix::DeviceContext context,
       py::object             callbackFunction,
       uint32_t               callbackLevel
    )
{
    context.logCallbackFunction = callbackFunction;
    OptixLogCallback cb         = nullptr;
    void*            cb_data    = nullptr;
    if( !context.logCallbackFunction.is_none() )
    {
        cb      = context_log_cb;
        cb_data = context.logCallbackFunction.ptr();
    }

    PYOPTIX_CHECK(
        optixDeviceContextSetLogCallback(
            context.deviceContext,
            cb,
            cb_data,
            callbackLevel
        )
    );
}

void deviceContextSetCacheEnabled(
       pyoptix::DeviceContext context,
       int                enabled
    )
{
    PYOPTIX_CHECK(
        optixDeviceContextSetCacheEnabled(
            context.deviceContext,
            enabled
        )
    );
}

void deviceContextSetCacheLocation(
       pyoptix::DeviceContext context,
       const char* location
    )
{
    PYOPTIX_CHECK(
        optixDeviceContextSetCacheLocation(
            context.deviceContext,
            location
        )
    );
}

void deviceContextSetCacheDatabaseSizes(
       pyoptix::DeviceContext context,
       size_t lowWaterMark,
       size_t highWaterMark
    )
{
    PYOPTIX_CHECK(
        optixDeviceContextSetCacheDatabaseSizes(
            context.deviceContext,
            lowWaterMark,
            highWaterMark
        )
    );
}

py::bool_ deviceContextGetCacheEnabled(
       pyoptix::DeviceContext context
    )
{
    int32_t enabled = 0;
    PYOPTIX_CHECK(
        optixDeviceContextGetCacheEnabled(
            context.deviceContext,
            &enabled
        )
    );

    return py::bool_( enabled );
}

py::str deviceContextGetCacheLocation(
       pyoptix::DeviceContext context
    )
{
   constexpr size_t locationSize = 1024u;
   char location[ locationSize ];
    PYOPTIX_CHECK(
        optixDeviceContextGetCacheLocation(
            context.deviceContext,
            location,
            locationSize
        )
    );
    return py::str( location );
}

py::tuple deviceContextGetCacheDatabaseSizes(
       pyoptix::DeviceContext context
    )
{
    size_t lowWaterMark;
    size_t highWaterMark;
    PYOPTIX_CHECK(
        optixDeviceContextGetCacheDatabaseSizes(
            context.deviceContext,
            &lowWaterMark,
            &highWaterMark
        )
    );
    return py::make_tuple( lowWaterMark, highWaterMark );
}


pyoptix::Pipeline pipelineCreate(
       pyoptix::DeviceContext                 context,
       const pyoptix::PipelineCompileOptions& pipelineCompileOptions,
       const pyoptix::PipelineLinkOptions&    pipelineLinkOptions,
       const py::list&                        programGroups,
       std::string&                           logString
    )
{
    std::vector<OptixProgramGroup> pgs;
    for( const auto list_elem : programGroups )
    {
        pyoptix::ProgramGroup pygroup = list_elem.cast<pyoptix::ProgramGroup>();
        pgs.push_back( pygroup.programGroup );
    }

    size_t log_buf_size = LOG_BUFFER_MAX_SIZE;
    char   log_buf[ LOG_BUFFER_MAX_SIZE ];
    log_buf[0] = '\0';

    pyoptix::Pipeline pipeline{};
    PYOPTIX_CHECK_LOG(
        optixPipelineCreate(
            context.deviceContext,
            &pipelineCompileOptions.options,
            &pipelineLinkOptions.options,
            pgs.data(),
            static_cast<uint32_t>( pgs.size() ),
            log_buf,
            &log_buf_size,
            &pipeline.pipeline
        )
    );

    logString = log_buf;
    return pipeline;
}

void pipelineDestroy(
       pyoptix::Pipeline pipeline
    )
{
    PYOPTIX_CHECK(
        optixPipelineDestroy(
            pipeline.pipeline
        )
    );
}

void pipelineSetStackSize(
       pyoptix::Pipeline pipeline,
       unsigned int  directCallableStackSizeFromTraversal,
       unsigned int  directCallableStackSizeFromState,
       unsigned int  continuationStackSize,
       unsigned int  maxTraversableGraphDepth
    )
{
    PYOPTIX_CHECK(
        optixPipelineSetStackSize(
            pipeline.pipeline,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize,
            maxTraversableGraphDepth
        )
    );
}

py::tuple moduleCreateFromPTX(
       const pyoptix::DeviceContext&          context,
             pyoptix::ModuleCompileOptions&   moduleCompileOptions,
             pyoptix::PipelineCompileOptions& pipelineCompileOptions,
       const std::string&                     PTX
       )
{
    size_t log_buf_size = LOG_BUFFER_MAX_SIZE;
    char   log_buf[ LOG_BUFFER_MAX_SIZE ];
    log_buf[0] = '\0';

    moduleCompileOptions.sync();
    pipelineCompileOptions.sync();

    pyoptix::Module module;
    PYOPTIX_CHECK_LOG(
        optixModuleCreateFromPTX(
            context.deviceContext,
            &moduleCompileOptions.options,
            &pipelineCompileOptions.options,
            PTX.c_str(),
            static_cast<size_t>( PTX.size()+1 ),
            log_buf,
            &log_buf_size,
            &module.module
        )
    );
    return py::make_tuple( module, py::str(log_buf) );
}

void moduleDestroy(
       pyoptix::Module module
    )
{
    PYOPTIX_CHECK(
        optixModuleDestroy(
            module.module
        )
    );
}


#if OPTIX_VERSION >= 70100
pyoptix::Module builtinISModuleGet(
       const pyoptix::DeviceContext&          context,
             pyoptix::ModuleCompileOptions&   moduleCompileOptions,
             pyoptix::PipelineCompileOptions& pipelineCompileOptions,
       const pyoptix::BuiltinISOptions&       builtinISOptions
    )
{
    moduleCompileOptions.sync();
    pipelineCompileOptions.sync();

    pyoptix::Module module;
    PYOPTIX_CHECK(
        optixBuiltinISModuleGet(
            context.deviceContext,
            &moduleCompileOptions.options,
            &pipelineCompileOptions.options,
            &builtinISOptions.options,
            &module.module
        )
    );
    return module;
}
#endif // OPTIX_VERSION >= 70100


OptixStackSizes programGroupGetStackSize(
       pyoptix::ProgramGroup programGroup
    )
{
    OptixStackSizes sizes{};
    PYOPTIX_CHECK(
        optixProgramGroupGetStackSize(
            programGroup.programGroup,
            &sizes
        )
    );
    return sizes;
}

py::tuple programGroupCreate(
       pyoptix::DeviceContext context,
       const py::list&        programDescriptions
       IF_OPTIX74( COMMA const pyoptix::ProgramGroupOptions& options )
    )
{
    size_t log_buf_size = LOG_BUFFER_MAX_SIZE;
    char   log_buf[ LOG_BUFFER_MAX_SIZE ];
    log_buf[0] = '\0';

    std::vector<OptixProgramGroupDesc> program_groups_descs;
    for( auto list_elem : programDescriptions )
    {
        pyoptix::ProgramGroupDesc& pydesc = list_elem.cast<pyoptix::ProgramGroupDesc&>();

        switch( pydesc.program_group_desc.kind )
        {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
                pydesc.program_group_desc.raygen.entryFunctionName =
                    !pydesc.entryFunctionName0.empty() ?
                    pydesc.entryFunctionName0.c_str() :
                    nullptr;
                break;
            case OPTIX_PROGRAM_GROUP_KIND_MISS:
                pydesc.program_group_desc.miss.entryFunctionName =
                    !pydesc.entryFunctionName0.empty() ?
                    pydesc.entryFunctionName0.c_str() :
                    nullptr;
                break;
            case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
                pydesc.program_group_desc.exception.entryFunctionName =
                    !pydesc.entryFunctionName0.empty() ?
                    pydesc.entryFunctionName0.c_str() :
                    nullptr;
                break;
            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
                pydesc.program_group_desc.hitgroup.entryFunctionNameCH =
                    !pydesc.entryFunctionName0.empty() ?
                    pydesc.entryFunctionName0.c_str() :
                    nullptr;
                pydesc.program_group_desc.hitgroup.entryFunctionNameAH =
                    !pydesc.entryFunctionName1.empty() ?
                    pydesc.entryFunctionName1.c_str() :
                    nullptr;
                pydesc.program_group_desc.hitgroup.entryFunctionNameIS =
                    !pydesc.entryFunctionName2.empty() ?
                    pydesc.entryFunctionName2.c_str() :
                    nullptr;
                break;
            case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
                pydesc.program_group_desc.callables.entryFunctionNameDC =
                    !pydesc.entryFunctionName0.empty() ?
                    pydesc.entryFunctionName0.c_str() :
                    nullptr;
                pydesc.program_group_desc.callables.entryFunctionNameCC =
                    !pydesc.entryFunctionName1.empty() ?
                    pydesc.entryFunctionName1.c_str() :
                    nullptr;
                break;

        }

        program_groups_descs.push_back( pydesc.program_group_desc );
    }
    std::vector<OptixProgramGroup> program_groups( programDescriptions.size() );

#if OPTIX_VERSION < 70400
       const OptixProgramGroupOptions opts{};
#else
       const OptixProgramGroupOptions& opts = options.options;
#endif

    PYOPTIX_CHECK_LOG(
        optixProgramGroupCreate(
            context.deviceContext,
            program_groups_descs.data(),
            static_cast<uint32_t>( program_groups_descs.size() ),
            &opts,
            log_buf,
            &log_buf_size,
            program_groups.data()
        )
    );

    py::list pygroups;
    for( auto& group : program_groups )
        pygroups.append( pyoptix::ProgramGroup{ group } );

    return py::make_tuple( pygroups, py::str(log_buf) );
}

void programGroupDestroy(
       pyoptix::ProgramGroup programGroup
    )
{
    PYOPTIX_CHECK(
        optixProgramGroupDestroy(
            programGroup.programGroup
        )
    );
}

void launch(
       pyoptix::Pipeline              pipeline,
       uintptr_t                      stream,
       CUdeviceptr                    pipelineParams,
       size_t                         pipelineParamsSize,
       const pyoptix::ShaderBindingTable& sbt,
       uint32_t                       width,
       uint32_t                       height,
       uint32_t                       depth
    )
{
    PYOPTIX_CHECK(
        optixLaunch(
            pipeline.pipeline,
            reinterpret_cast<CUstream>( stream ),
            pipelineParams,
            pipelineParamsSize,
            &sbt.sbt,
            width,
            height,
            depth
        )
    );
}

void sbtRecordPackHeader(
       pyoptix::ProgramGroup programGroup,
       py::buffer sbtRecord
    )
{
    py::buffer_info binfo = sbtRecord.request();
    // TODO: sanity check buffer

    PYOPTIX_CHECK(
        optixSbtRecordPackHeader(
            programGroup.programGroup,
            binfo.ptr
        )
    );
}

OptixAccelBufferSizes accelComputeMemoryUsage(
       pyoptix::DeviceContext   context,
       const py::list&          accelOptions, 
       const py::list&          buildInputs    
    )
{
    const uint32_t num_inputs = buildInputs.size();
    if( accelOptions.size() != num_inputs )
        throw std::runtime_error(
            "Context.accelComputeMemoryUsage called with mismatched number "
            "of accel options and build inputs"
            );
    auto accel_options = accelOptions.cast<std::vector<OptixAccelBuildOptions> >();

    std::vector<OptixBuildInput> build_inputs;
    convertBuildInputs( buildInputs, build_inputs );

    OptixAccelBufferSizes bufferSizes{};
    PYOPTIX_CHECK(
        optixAccelComputeMemoryUsage(
            context.deviceContext,
            accel_options.data(),
            build_inputs.data(),
            num_inputs,
            &bufferSizes
        )
    );
    return bufferSizes;
}

OptixTraversableHandle accelBuild(
       pyoptix::DeviceContext        context,
       uintptr_t                     stream,
       const py::list&               accelOptions, //AccelBuildOptions
       const py::list&               buildInputs,  // 
       CUdeviceptr                   tempBuffer,
       size_t                        tempBufferSizeInBytes,
       CUdeviceptr                   outputBuffer,
       size_t                        outputBufferSizeInBytes,
       const py::list&               emittedProperties // AccelEmitDesc
    )
{
    const uint32_t num_inputs = buildInputs.size();
    if( accelOptions.size() != num_inputs )
        throw std::runtime_error(
            "Context.accelComputeMemoryUsage called with mismatched number "
            "of accel options and build inputs"
            );
    auto accel_options = accelOptions.cast<std::vector<OptixAccelBuildOptions> >();

    std::vector<OptixBuildInput> build_inputs;
    convertBuildInputs( buildInputs, build_inputs );

    const uint32_t num_properties = emittedProperties.size();
    auto emitted_properties = emittedProperties.cast<std::vector<OptixAccelEmitDesc> >();

    OptixTraversableHandle output_handle;
    PYOPTIX_CHECK(
        optixAccelBuild(
            context.deviceContext,
            reinterpret_cast<CUstream>( stream ),
            accel_options.data(),
            build_inputs.data(),
            num_inputs,
            tempBuffer,
            tempBufferSizeInBytes,
            outputBuffer,
            outputBufferSizeInBytes,
            &output_handle,
            emitted_properties.empty() ? nullptr : emitted_properties.data(),
            num_properties
        )
    );

    return output_handle;
}

void accelGetRelocationInfo(
       pyoptix::DeviceContext context,
       OptixTraversableHandle handle,
       OptixAccelRelocationInfo* info
    )
{
    PYOPTIX_CHECK(
        optixAccelGetRelocationInfo(
            context.deviceContext,
            handle,
            info
        )
    );
}

void accelCheckRelocationCompatibility(
       pyoptix::DeviceContext context,
       const OptixAccelRelocationInfo* info,
       int* compatible
    )
{
    PYOPTIX_CHECK(
        optixAccelCheckRelocationCompatibility(
            context.deviceContext,
            info,
            compatible
        )
    );
}

void accelRelocate(
       pyoptix::DeviceContext          context,
       uintptr_t           stream,
       const OptixAccelRelocationInfo* info,
       CUdeviceptr                     instanceTraversableHandles,
       size_t                          numInstanceTraversableHandles,
       CUdeviceptr                     targetAccel,
       size_t                          targetAccelSizeInBytes,
       OptixTraversableHandle*         targetHandle
    )
{
    PYOPTIX_CHECK(
        optixAccelRelocate(
            context.deviceContext,
            reinterpret_cast<CUstream>( stream ),
            info,
            instanceTraversableHandles,
            numInstanceTraversableHandles,
            targetAccel,
            targetAccelSizeInBytes,
            targetHandle
        )
    );
}

void accelCompact(
       pyoptix::DeviceContext  context,
       uintptr_t   stream,
       OptixTraversableHandle  inputHandle,
       CUdeviceptr             outputBuffer,
       size_t                  outputBufferSizeInBytes,
       OptixTraversableHandle* outputHandle
    )
{
    PYOPTIX_CHECK(
        optixAccelCompact(
            context.deviceContext,
            reinterpret_cast<CUstream>( stream ),
            inputHandle,
            outputBuffer,
            outputBufferSizeInBytes,
            outputHandle
        )
    );
}

void convertPointerToTraversableHandle(
       pyoptix::DeviceContext  onDevice,
       CUdeviceptr             pointer,
       OptixTraversableType    traversableType,
       OptixTraversableHandle* traversableHandle
    )
{
    PYOPTIX_CHECK(
        optixConvertPointerToTraversableHandle(
            onDevice.deviceContext,
            pointer,
            traversableType,
            traversableHandle
        )
    );
}

#if OPTIX_VERSION >= 70300
void denoiserCreate( 
       pyoptix::DeviceContext context,
       OptixDenoiserModelKind modelKind,
       const OptixDenoiserOptions* options,
       pyoptix::Denoiser* denoiser
    )
{
    PYOPTIX_CHECK(
        optixDenoiserCreate(
            context.deviceContext,
            modelKind,
            options,
            &denoiser->denoiser
        )
    );
}
#else
void denoiserCreate(
       pyoptix::DeviceContext context,
       const OptixDenoiserOptions* options,
       pyoptix::Denoiser* denoiser
    )
{
    PYOPTIX_CHECK( 
        optixDenoiserCreate(
            context.deviceContext,
            options,
            &denoiser->denoiser
        )
    );
}
#endif


#if OPTIX_VERSION < 70300
void denoiserSetModel(
       pyoptix::Denoiser denoiser,
       OptixDenoiserModelKind kind,
       void* data,
       size_t sizeInBytes
    )
{
    PYOPTIX_CHECK(
        optixDenoiserSetModel(
            denoiser.denoiser,
            kind,
            data,
            sizeInBytes
        )
    );
}
#endif


void denoiserDestroy(
       pyoptix::Denoiser denoiser
    )
{
    PYOPTIX_CHECK(
        optixDenoiserDestroy(
            denoiser.denoiser
        )
    );
}

void denoiserComputeMemoryResources(
       const pyoptix::Denoiser denoiser,
       unsigned int        outputWidth,
       unsigned int        outputHeight,
       OptixDenoiserSizes* returnSizes
    )
{
    PYOPTIX_CHECK(
        optixDenoiserComputeMemoryResources(
            denoiser.denoiser,
            outputWidth,
            outputHeight,
            returnSizes
        )
    );
}

void denoiserSetup(
       pyoptix::Denoiser denoiser,
       uintptr_t stream,
       unsigned int  inputWidth,
       unsigned int  inputHeight,
       CUdeviceptr   denoiserState,
       size_t        denoiserStateSizeInBytes,
       CUdeviceptr   scratch,
       size_t        scratchSizeInBytes
    )
{
    PYOPTIX_CHECK(
        optixDenoiserSetup(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            inputWidth,
            inputHeight,
            denoiserState,
            denoiserStateSizeInBytes,
            scratch,
            scratchSizeInBytes
        )
    );
}

#if OPTIX_VERSION >= 70300
void denoiserInvoke(
       pyoptix::Denoiser              denoiser,
       uintptr_t                      stream,
       const OptixDenoiserParams*     params,
       CUdeviceptr                    denoiserState,
       size_t                         denoiserStateSizeInBytes,
       const OptixDenoiserGuideLayer* guideLayer,
       const OptixDenoiserLayer*      layers,
       unsigned int                   numLayers,
       unsigned int                   inputOffsetX,
       unsigned int                   inputOffsetY,
       CUdeviceptr                    scratch,
       size_t                         scratchSizeInBytes
    )
{
    PYOPTIX_CHECK(
        optixDenoiserInvoke(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            params,
            denoiserState,
            denoiserStateSizeInBytes,
            guideLayer,
            layers,
            numLayers,
            inputOffsetX,
            inputOffsetY,
            scratch,
            scratchSizeInBytes
        )
    );
}
#else
void denoiserInvoke(
       pyoptix::Denoiser          denoiser,
       uintptr_t                  stream,
       const OptixDenoiserParams* params,
       CUdeviceptr                denoiserState,
       size_t                     denoiserStateSizeInBytes,
       const OptixImage2D*        inputLayers,
       unsigned int               numInputLayers,
       unsigned int               inputOffsetX,
       unsigned int               inputOffsetY,
       const OptixImage2D*        outputLayer,
       CUdeviceptr                scratch,
       size_t                     scratchSizeInBytes
    )
{
    PYOPTIX_CHECK(
        optixDenoiserInvoke(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            params,
            denoiserState,
            denoiserStateSizeInBytes,
            inputLayers,
            numInputLayers,
            inputOffsetX,
            inputOffsetY,
	    outputLayer,
            scratch,
            scratchSizeInBytes
        )
    );
}
#endif

void denoiserComputeIntensity(
       pyoptix::Denoiser   denoiser,
       uintptr_t stream,
       const OptixImage2D* inputImage,
       CUdeviceptr         outputIntensity,
       CUdeviceptr         scratch,
       size_t              scratchSizeInBytes
    )
{
    PYOPTIX_CHECK(
        optixDenoiserComputeIntensity(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            inputImage,
            outputIntensity,
            scratch,
            scratchSizeInBytes
        )
    );
}

#if OPTIX_VERSION >= 70200
void denoiserComputeAverageColor(
       pyoptix::Denoiser   denoiser,
       uintptr_t stream,
       const OptixImage2D* inputImage,
       CUdeviceptr         outputAverageColor,
       CUdeviceptr         scratch,
       size_t              scratchSizeInBytes
    )
{
    PYOPTIX_CHECK(
        optixDenoiserComputeAverageColor(
            denoiser.denoiser,
            reinterpret_cast<CUstream>( stream ),
            inputImage,
            outputAverageColor,
            scratch,
            scratchSizeInBytes
        )
    );
}
#endif

namespace util
{

void accumulateStackSizes(
        pyoptix::ProgramGroup programGroup,
        OptixStackSizes&  stackSizes
        )
{
    PYOPTIX_CHECK(
        optixUtilAccumulateStackSizes( programGroup.programGroup, &stackSizes )
    );
}

py::tuple computeStackSizes(
        const OptixStackSizes& stackSizes,
        unsigned int           maxTraceDepth,
        unsigned int           maxCCDepth,
        unsigned int           maxDCDepth
        )
{
    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;

    PYOPTIX_CHECK(
        optixUtilComputeStackSizes(
            &stackSizes,
            maxTraceDepth,
            maxCCDepth,
            maxDCDepth,
            &directCallableStackSizeFromTraversal,
            &directCallableStackSizeFromState,
            &continuationStackSize
            )
        );

    return py::make_tuple(
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize 
        );
}

py::tuple computeStackSizesDCSplit(
        const OptixStackSizes& stackSizes,
        unsigned int           dssDCFromTraversal,
        unsigned int           dssDCFromState,
        unsigned int           maxTraceDepth,
        unsigned int           maxCCDepth,
        unsigned int           maxDCDepthFromTraversal,
        unsigned int           maxDCDepthFromState
        )
{
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;

    PYOPTIX_CHECK(
        optixUtilComputeStackSizesDCSplit(
            &stackSizes,
            dssDCFromTraversal,
            dssDCFromState,
            maxTraceDepth,
            maxCCDepth,
            maxDCDepthFromTraversal,
            maxDCDepthFromState,
            &directCallableStackSizeFromTraversal,
            &directCallableStackSizeFromState,
            &continuationStackSize
            )
        );

    return py::make_tuple(
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize
        );
}


py::tuple computeStackSizesCssCCTree(
        const OptixStackSizes* stackSizes,
        unsigned int           cssCCTree,
        unsigned int           maxTraceDepth,
        unsigned int           maxDCDepth
        )
{
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;

    PYOPTIX_CHECK(
        optixUtilComputeStackSizesCssCCTree(
            stackSizes,
            cssCCTree,
            maxTraceDepth,
            maxDCDepth,
            &directCallableStackSizeFromTraversal,
            &directCallableStackSizeFromState,
            &continuationStackSize
            )
        );

    return py::make_tuple(
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize
        );
}


py::tuple computeStackSizesSimplePathTracer(
        pyoptix::ProgramGroup        programGroupRG,
        pyoptix::ProgramGroup        programGroupMS1,
        py::list                     programGroupCH1,
        pyoptix::ProgramGroup        programGroupMS2,
        py::list                     programGroupCH2 
        )
{
    unsigned int directCallableStackSizeFromTraversal;
    unsigned int directCallableStackSizeFromState;
    unsigned int continuationStackSize;

    auto ch1_py = programGroupCH1.cast<std::vector<pyoptix::ProgramGroup> >();
    auto ch2_py = programGroupCH2.cast<std::vector<pyoptix::ProgramGroup> >();
    std::vector<OptixProgramGroup> ch1;
    std::vector<OptixProgramGroup> ch2;
    for( auto& pypg : ch1_py )
        ch1.push_back( pypg.programGroup );
    for( auto pypg : ch2_py )
        ch2.push_back( pypg.programGroup );

    PYOPTIX_CHECK(
        optixUtilComputeStackSizesSimplePathTracer(
            programGroupRG.programGroup,
            programGroupMS1.programGroup,
            ch1.data(),
            ch1.size(),
            programGroupMS2.programGroup,
            ch2.data(),
            ch2.size(),
            &directCallableStackSizeFromTraversal,
            &directCallableStackSizeFromState,
            &continuationStackSize
            )
        );

    return py::make_tuple(
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize
        );
}


} // end namespace util
} // end namespace pyoptix


PYBIND11_MODULE( optix, m )
{
    m.doc() = R"pbdoc(
        OptiX API
        -----------------------

        .. currentmodule:: optix

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    cudaFree(0); 
    pyoptix::init();
    
    //---------------------------------------------------------------------------
    //
    // Module Methods
    //
    //---------------------------------------------------------------------------
    m.def( "version", &pyoptix::version);
    m.def( "deviceContextCreate", &pyoptix::deviceContextCreate);
    m.def( "getErrorName", &pyoptix::getErrorName );
    m.def( "getErrorString", &pyoptix::getErrorString );
    m.def( "launch", &pyoptix::launch );
    m.def( "sbtRecordPackHeader", &pyoptix::sbtRecordPackHeader );
    m.def( "convertPointerToTraversableHandle", &pyoptix::convertPointerToTraversableHandle );


    //--------------------------------------------------------------------------
    //
    // Structs for interfacing with CUDA
    //
    //--------------------------------------------------------------------------
    auto m_util = m.def_submodule( "util", nullptr /*TODO: docstring*/ );
    m_util.def( "accumulateStackSizes", &pyoptix::util::accumulateStackSizes );
    m_util.def( "computeStackSizes", &pyoptix::util::computeStackSizes );
    m_util.def( "computeStackSizesDCSplit", &pyoptix::util::computeStackSizesDCSplit );
    m_util.def( "computeStackSizesCssCCTree", &pyoptix::util::computeStackSizesCssCCTree );
    m_util.def( "computeStackSizesSimplePathTracer", &pyoptix::util::computeStackSizesSimplePathTracer );


    //--------------------------------------------------------------------------
    //
    // defines
    //
    //--------------------------------------------------------------------------

    m.attr( "SBT_RECORD_HEADER_SIZE"             ) = OPTIX_SBT_RECORD_HEADER_SIZE;
    m.attr( "SBT_RECORD_ALIGNMENT"               ) = OPTIX_SBT_RECORD_ALIGNMENT;
    m.attr( "ACCEL_BUFFER_BYTE_ALIGNMENT"        ) = OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
    m.attr( "INSTANCE_BYTE_ALIGNMENT"            ) = OPTIX_INSTANCE_BYTE_ALIGNMENT;
    m.attr( "AABB_BUFFER_BYTE_ALIGNMENT"         ) = OPTIX_AABB_BUFFER_BYTE_ALIGNMENT;
    m.attr( "GEOMETRY_TRANSFORM_BYTE_ALIGNMENT"  ) = OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT;
    m.attr( "TRANSFORM_BYTE_ALIGNMENT"           ) = OPTIX_TRANSFORM_BYTE_ALIGNMENT;
    m.attr( "COMPILE_DEFAULT_MAX_REGISTER_COUNT" ) = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;


    //--------------------------------------------------------------------------
    //
    // Enumerations
    //
    //--------------------------------------------------------------------------

    py::enum_<OptixResult>(m, "Result", py::arithmetic())
        .value( "SUCCESS", OPTIX_SUCCESS )
        .value( "ERROR_INVALID_VALUE", OPTIX_ERROR_INVALID_VALUE )
        .value( "ERROR_HOST_OUT_OF_MEMORY", OPTIX_ERROR_HOST_OUT_OF_MEMORY )
        .value( "ERROR_INVALID_OPERATION", OPTIX_ERROR_INVALID_OPERATION )
        .value( "ERROR_FILE_IO_ERROR", OPTIX_ERROR_FILE_IO_ERROR )
        .value( "ERROR_INVALID_FILE_FORMAT", OPTIX_ERROR_INVALID_FILE_FORMAT )
        .value( "ERROR_DISK_CACHE_INVALID_PATH", OPTIX_ERROR_DISK_CACHE_INVALID_PATH )
        .value( "ERROR_DISK_CACHE_PERMISSION_ERROR", OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR )
        .value( "ERROR_DISK_CACHE_DATABASE_ERROR", OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR )
        .value( "ERROR_DISK_CACHE_INVALID_DATA", OPTIX_ERROR_DISK_CACHE_INVALID_DATA )
        .value( "ERROR_LAUNCH_FAILURE", OPTIX_ERROR_LAUNCH_FAILURE )
        .value( "ERROR_INVALID_DEVICE_CONTEXT", OPTIX_ERROR_INVALID_DEVICE_CONTEXT )
        .value( "ERROR_CUDA_NOT_INITIALIZED", OPTIX_ERROR_CUDA_NOT_INITIALIZED )
#if OPTIX_VERSION >= 70200
        .value( "ERROR_VALIDATION_FAILURE", OPTIX_ERROR_VALIDATION_FAILURE )
#endif
        .value( "ERROR_INVALID_PTX", OPTIX_ERROR_INVALID_PTX )
        .value( "ERROR_INVALID_LAUNCH_PARAMETER", OPTIX_ERROR_INVALID_LAUNCH_PARAMETER )
        .value( "ERROR_INVALID_PAYLOAD_ACCESS", OPTIX_ERROR_INVALID_PAYLOAD_ACCESS )
        .value( "ERROR_INVALID_ATTRIBUTE_ACCESS", OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS )
        .value( "ERROR_INVALID_FUNCTION_USE", OPTIX_ERROR_INVALID_FUNCTION_USE )
        .value( "ERROR_INVALID_FUNCTION_ARGUMENTS", OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS )
        .value( "ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY", OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY )
        .value( "ERROR_PIPELINE_LINK_ERROR", OPTIX_ERROR_PIPELINE_LINK_ERROR )
        .value( "ERROR_INTERNAL_COMPILER_ERROR", OPTIX_ERROR_INTERNAL_COMPILER_ERROR )
        .value( "ERROR_DENOISER_MODEL_NOT_SET", OPTIX_ERROR_DENOISER_MODEL_NOT_SET )
        .value( "ERROR_DENOISER_NOT_INITIALIZED", OPTIX_ERROR_DENOISER_NOT_INITIALIZED )
        .value( "ERROR_ACCEL_NOT_COMPATIBLE", OPTIX_ERROR_ACCEL_NOT_COMPATIBLE )
        .value( "ERROR_NOT_SUPPORTED", OPTIX_ERROR_NOT_SUPPORTED )
        .value( "ERROR_UNSUPPORTED_ABI_VERSION", OPTIX_ERROR_UNSUPPORTED_ABI_VERSION )
        .value( "ERROR_FUNCTION_TABLE_SIZE_MISMATCH", OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH )
        .value( "ERROR_INVALID_ENTRY_FUNCTION_OPTIONS", OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS )
        .value( "ERROR_LIBRARY_NOT_FOUND", OPTIX_ERROR_LIBRARY_NOT_FOUND )
        .value( "ERROR_ENTRY_SYMBOL_NOT_FOUND", OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND )
#if OPTIX_VERSION >= 70200
        .value( "ERROR_LIBRARY_UNLOAD_FAILURE", OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE )
#endif
        .value( "ERROR_CUDA_ERROR", OPTIX_ERROR_CUDA_ERROR )
        .value( "ERROR_INTERNAL_ERROR", OPTIX_ERROR_INTERNAL_ERROR )
        .value( "ERROR_UNKNOWN", OPTIX_ERROR_UNKNOWN )
        .export_values();

    py::enum_<OptixDeviceProperty>(m, "DeviceProperty", py::arithmetic())
        .value( "DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS )
        .value( "DEVICE_PROPERTY_RTCORE_VERSION", OPTIX_DEVICE_PROPERTY_RTCORE_VERSION )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID )
        .value( "DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK", OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS )
        .value( "DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET", OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET )
        .export_values();

#if OPTIX_VERSION >= 70200
    py::enum_<OptixDeviceContextValidationMode>(m, "DeviceContextValidationMode", py::arithmetic())
        .value( "DEVICE_CONTEXT_VALIDATION_MODE_OFF", OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF )
        .value( "DEVICE_CONTEXT_VALIDATION_MODE_ALL", OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL )
        .export_values();
#endif

    py::enum_<OptixGeometryFlags>(m, "GeometryFlags", py::arithmetic())
        .value( "GEOMETRY_FLAG_NONE", OPTIX_GEOMETRY_FLAG_NONE )
        .value( "GEOMETRY_FLAG_DISABLE_ANYHIT", OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT )
        .value( "GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL", OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL )
        .export_values();

    py::enum_<OptixHitKind>(m, "HitKind", py::arithmetic())
        .value( "HIT_KIND_TRIANGLE_FRONT_FACE", OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE )
        .value( "HIT_KIND_TRIANGLE_BACK_FACE", OPTIX_HIT_KIND_TRIANGLE_BACK_FACE )
        .export_values();

    py::enum_<OptixIndicesFormat>(m, "IndicesFormat", py::arithmetic())
        IF_OPTIX71(
        .value( "INDICES_FORMAT_NONE", OPTIX_INDICES_FORMAT_NONE )
        )
        .value( "INDICES_FORMAT_UNSIGNED_SHORT3", OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 )
        .value( "INDICES_FORMAT_UNSIGNED_INT3", OPTIX_INDICES_FORMAT_UNSIGNED_INT3 )
        .export_values();

    py::enum_<OptixVertexFormat>(m, "VertexFormat", py::arithmetic())
        IF_OPTIX71(
        .value( "VERTEX_FORMAT_NONE", OPTIX_VERTEX_FORMAT_NONE )
        )
        .value( "VERTEX_FORMAT_FLOAT3", OPTIX_VERTEX_FORMAT_FLOAT3 )
        .value( "VERTEX_FORMAT_FLOAT2", OPTIX_VERTEX_FORMAT_FLOAT2 )
        .value( "VERTEX_FORMAT_HALF3", OPTIX_VERTEX_FORMAT_HALF3 )
        .value( "VERTEX_FORMAT_HALF2", OPTIX_VERTEX_FORMAT_HALF2 )
        .value( "VERTEX_FORMAT_SNORM16_3", OPTIX_VERTEX_FORMAT_SNORM16_3 )
        .value( "VERTEX_FORMAT_SNORM16_2", OPTIX_VERTEX_FORMAT_SNORM16_2 )
        .export_values();

#if OPTIX_VERSION >= 70100
    py::enum_<OptixTransformFormat>(m, "TransformFormat", py::arithmetic())
        .value( "TRANSFORM_FORMAT_NONE", OPTIX_TRANSFORM_FORMAT_NONE )
        .value( "TRANSFORM_FORMAT_MATRIX_FLOAT12", OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 )
        .export_values();

    py::enum_<OptixPrimitiveType>(m, "PrimitiveType", py::arithmetic())
        .value( "PRIMITIVE_TYPE_CUSTOM", OPTIX_PRIMITIVE_TYPE_CUSTOM )
        .value( "PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_ROUND_LINEAR", OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR )
        .value( "PRIMITIVE_TYPE_TRIANGLE", OPTIX_PRIMITIVE_TYPE_TRIANGLE )
        .export_values();

    py::enum_<OptixPrimitiveTypeFlags>(m, "PmitiveTypeFlags", py::arithmetic() )
        .value( "PRIMITIVE_TYPE_FLAGS_CUSTOM", OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM )
        .value( "PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE", OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE )
        .value( "PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR", OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR )
        .value( "PRIMITIVE_TYPE_FLAGS_TRIANGLE", OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE )
        .export_values();
#endif

    py::enum_<OptixBuildInputType>(m, "BuildInputType", py::arithmetic())
        .value( "BUILD_INPUT_TYPE_TRIANGLES", OPTIX_BUILD_INPUT_TYPE_TRIANGLES )
        .value( "BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES", OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES )
        .value( "BUILD_INPUT_TYPE_INSTANCES", OPTIX_BUILD_INPUT_TYPE_INSTANCES )
        .value( "BUILD_INPUT_TYPE_INSTANCE_POINTERS", OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS )
        IF_OPTIX71(
        .value( "BUILD_INPUT_TYPE_CURVES", OPTIX_BUILD_INPUT_TYPE_CURVES )
        )
        .export_values();

    py::enum_<OptixInstanceFlags>(m, "InstanceFlags", py::arithmetic())
        .value( "INSTANCE_FLAG_NONE", OPTIX_INSTANCE_FLAG_NONE )
        .value( "INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING", OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING )
        .value( "INSTANCE_FLAG_FLIP_TRIANGLE_FACING", OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING )
        .value( "INSTANCE_FLAG_DISABLE_ANYHIT", OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT )
        .value( "INSTANCE_FLAG_ENFORCE_ANYHIT", OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT )
#if OPTIX_VERSION < 70400
        .value( "INSTANCE_FLAG_DISABLE_TRANSFORM", OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM )
#endif
        .export_values();

    py::enum_<OptixBuildFlags>(m, "BuildFlags", py::arithmetic())
        .value( "BUILD_FLAG_NONE", OPTIX_BUILD_FLAG_NONE )
        .value( "BUILD_FLAG_ALLOW_UPDATE", OPTIX_BUILD_FLAG_ALLOW_UPDATE )
        .value( "BUILD_FLAG_ALLOW_COMPACTION", OPTIX_BUILD_FLAG_ALLOW_COMPACTION )
        .value( "BUILD_FLAG_PREFER_FAST_TRACE", OPTIX_BUILD_FLAG_PREFER_FAST_TRACE )
        .value( "BUILD_FLAG_PREFER_FAST_BUILD", OPTIX_BUILD_FLAG_PREFER_FAST_BUILD )
        .value( "BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS", OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS )
        .export_values();

    py::enum_<OptixBuildOperation>(m, "BuildOperation", py::arithmetic())
        .value( "BUILD_OPERATION_BUILD", OPTIX_BUILD_OPERATION_BUILD )
        .value( "BUILD_OPERATION_UPDATE", OPTIX_BUILD_OPERATION_UPDATE )
        .export_values();

    py::enum_<OptixMotionFlags>(m, "MotionFlags", py::arithmetic())
        .value( "MOTION_FLAG_NONE", OPTIX_MOTION_FLAG_NONE )
        .value( "MOTION_FLAG_START_VANISH", OPTIX_MOTION_FLAG_START_VANISH )
        .value( "MOTION_FLAG_END_VANISH", OPTIX_MOTION_FLAG_END_VANISH )
        .export_values();

    py::enum_<OptixAccelPropertyType>(m, "AccelPropertyType", py::arithmetic())
        .value( "PROPERTY_TYPE_COMPACTED_SIZE", OPTIX_PROPERTY_TYPE_COMPACTED_SIZE )
        .value( "PROPERTY_TYPE_AABBS", OPTIX_PROPERTY_TYPE_AABBS )
        .export_values();

    py::enum_<OptixTraversableType>(m, "TraversableType", py::arithmetic())
        .value( "TRAVERSABLE_TYPE_STATIC_TRANSFORM", OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM )
        .value( "TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM", OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM )
        .value( "TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM", OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM )
        .export_values();

    py::enum_<OptixPixelFormat>(m, "PixelFormat", py::arithmetic())
        .value( "PIXEL_FORMAT_HALF3", OPTIX_PIXEL_FORMAT_HALF3 )
        .value( "PIXEL_FORMAT_HALF4", OPTIX_PIXEL_FORMAT_HALF4 )
        .value( "PIXEL_FORMAT_FLOAT3", OPTIX_PIXEL_FORMAT_FLOAT3 )
        .value( "PIXEL_FORMAT_FLOAT4", OPTIX_PIXEL_FORMAT_FLOAT4 )
        .value( "PIXEL_FORMAT_UCHAR3", OPTIX_PIXEL_FORMAT_UCHAR3 )
        .value( "PIXEL_FORMAT_UCHAR4", OPTIX_PIXEL_FORMAT_UCHAR4 )
        .export_values();

    py::enum_<OptixDenoiserModelKind>(m, "DenoiserModelKind", py::arithmetic())
#if OPTIX_VERSION < 70300
        .value( "DENOISER_MODEL_KIND_USER", OPTIX_DENOISER_MODEL_KIND_USER )
#endif
        .value( "DENOISER_MODEL_KIND_LDR", OPTIX_DENOISER_MODEL_KIND_LDR )
        .value( "DENOISER_MODEL_KIND_HDR", OPTIX_DENOISER_MODEL_KIND_HDR )
        .value( "DENOISER_MODEL_KIND_AOV", OPTIX_DENOISER_MODEL_KIND_AOV )
        IF_OPTIX73( .value( "DENOISER_MODEL_KIND_TEMPORAL", OPTIX_DENOISER_MODEL_KIND_AOV ) )
        IF_OPTIX74( .value( "DENOISER_MODEL_KIND_TEMPORAL_AOV", OPTIX_DENOISER_MODEL_KIND_AOV ) )
        .export_values();

    py::enum_<OptixRayFlags>(m, "RayFlags", py::arithmetic())
        .value( "RAY_FLAG_NONE", OPTIX_RAY_FLAG_NONE )
        .value( "RAY_FLAG_DISABLE_ANYHIT", OPTIX_RAY_FLAG_DISABLE_ANYHIT )
        .value( "RAY_FLAG_ENFORCE_ANYHIT", OPTIX_RAY_FLAG_ENFORCE_ANYHIT )
        .value( "RAY_FLAG_TERMINATE_ON_FIRST_HIT", OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT )
        .value( "RAY_FLAG_DISABLE_CLOSESTHIT", OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT )
        .value( "RAY_FLAG_CULL_BACK_FACING_TRIANGLES", OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES )
        .value( "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES", OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES )
        .value( "RAY_FLAG_CULL_DISABLED_ANYHIT", OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT )
        .value( "RAY_FLAG_CULL_ENFORCED_ANYHIT", OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT )
        .export_values();

    py::enum_<OptixTransformType>(m, "TransformType", py::arithmetic())
        .value( "TRANSFORM_TYPE_NONE", OPTIX_TRANSFORM_TYPE_NONE )
        .value( "TRANSFORM_TYPE_STATIC_TRANSFORM", OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM )
        .value( "TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM", OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM )
        .value( "TRANSFORM_TYPE_SRT_MOTION_TRANSFORM", OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM )
        .value( "TRANSFORM_TYPE_INSTANCE", OPTIX_TRANSFORM_TYPE_INSTANCE )
        .export_values();

    py::enum_<OptixTraversableGraphFlags>(m, "TraversableGraphFlags", py::arithmetic())
        .value( "TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY", OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY )
        .value( "TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS", OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS )
        .value( "TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING", OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING )
        .export_values();

    py::enum_<OptixCompileOptimizationLevel>(m, "CompileOptimizationLevel", py::arithmetic())
        .value( "COMPILE_OPTIMIZATION_DEFAULT", OPTIX_COMPILE_OPTIMIZATION_DEFAULT )
        .value( "COMPILE_OPTIMIZATION_LEVEL_0", OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 )
        .value( "COMPILE_OPTIMIZATION_LEVEL_1", OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 )
        .value( "COMPILE_OPTIMIZATION_LEVEL_2", OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 )
        .value( "COMPILE_OPTIMIZATION_LEVEL_3", OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 )
        .export_values();

    py::enum_<OptixCompileDebugLevel>(m, "CompileDebugLevel", py::arithmetic())
        IF_OPTIX71(
        .value( "COMPILE_DEBUG_LEVEL_DEFAULT",  OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT  )
        )
        .value( "COMPILE_DEBUG_LEVEL_NONE",     OPTIX_COMPILE_DEBUG_LEVEL_NONE     )
#if OPTIX_VERSION < 70400
        .value( "COMPILE_DEBUG_LEVEL_LINEINFO", OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO )
        .value( "COMPILE_DEBUG_LEVEL_FULL",     OPTIX_COMPILE_DEBUG_LEVEL_FULL     )
#else
        .value( "COMPILE_DEBUG_LEVEL_MINIMAL",  OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL  )
        .value( "COMPILE_DEBUG_LEVEL_MODERATE", OPTIX_COMPILE_DEBUG_LEVEL_MODERATE )
        .value( "COMPILE_DEBUG_LEVEL_FULL",     OPTIX_COMPILE_DEBUG_LEVEL_FULL     )
#endif
        .export_values();


#if OPTIX_VERSION >= 70400
    py::enum_<OptixPayloadTypeID>(m, "PayloadTypeID", py::arithmetic())
        .value( "PAYLOAD_TYPE_DEFAULT", OPTIX_PAYLOAD_TYPE_DEFAULT )
        .value( "PAYLOAD_TYPE_ID_0", OPTIX_PAYLOAD_TYPE_ID_0 )
        .value( "PAYLOAD_TYPE_ID_1", OPTIX_PAYLOAD_TYPE_ID_1 )
        .value( "PAYLOAD_TYPE_ID_2", OPTIX_PAYLOAD_TYPE_ID_2 )
        .value( "PAYLOAD_TYPE_ID_3", OPTIX_PAYLOAD_TYPE_ID_3 )
        .value( "PAYLOAD_TYPE_ID_4", OPTIX_PAYLOAD_TYPE_ID_4 )
        .value( "PAYLOAD_TYPE_ID_5", OPTIX_PAYLOAD_TYPE_ID_5 )
        .value( "PAYLOAD_TYPE_ID_6", OPTIX_PAYLOAD_TYPE_ID_6 )
        .value( "PAYLOAD_TYPE_ID_7", OPTIX_PAYLOAD_TYPE_ID_7 )
        .export_values();

    py::enum_<OptixPayloadSemantics>(m, "PayloadSemantics", py::arithmetic())
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_NONE", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_NONE )
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_READ", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ )
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE )
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE )

        .value( "PAYLOAD_SEMANTICS_CH_NONE", OPTIX_PAYLOAD_SEMANTICS_CH_NONE )
        .value( "PAYLOAD_SEMANTICS_CH_READ", OPTIX_PAYLOAD_SEMANTICS_CH_READ )
        .value( "PAYLOAD_SEMANTICS_CH_WRITE", OPTIX_PAYLOAD_SEMANTICS_CH_WRITE )
        .value( "PAYLOAD_SEMANTICS_CH_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE )

        .value( "PAYLOAD_SEMANTICS_MS_NONE", OPTIX_PAYLOAD_SEMANTICS_MS_NONE )
        .value( "PAYLOAD_SEMANTICS_MS_READ", OPTIX_PAYLOAD_SEMANTICS_MS_READ )
        .value( "PAYLOAD_SEMANTICS_MS_WRITE", OPTIX_PAYLOAD_SEMANTICS_MS_WRITE )
        .value( "PAYLOAD_SEMANTICS_MS_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_MS_WRITE )

        .value( "PAYLOAD_SEMANTICS_AH_NONE", OPTIX_PAYLOAD_SEMANTICS_AH_NONE )
        .value( "PAYLOAD_SEMANTICS_AH_READ", OPTIX_PAYLOAD_SEMANTICS_AH_READ )
        .value( "PAYLOAD_SEMANTICS_AH_WRITE", OPTIX_PAYLOAD_SEMANTICS_AH_WRITE )
        .value( "PAYLOAD_SEMANTICS_AH_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE )

        .value( "PAYLOAD_SEMANTICS_IS_NONE", OPTIX_PAYLOAD_SEMANTICS_IS_NONE )
        .value( "PAYLOAD_SEMANTICS_IS_READ", OPTIX_PAYLOAD_SEMANTICS_IS_READ )
        .value( "PAYLOAD_SEMANTICS_IS_WRITE", OPTIX_PAYLOAD_SEMANTICS_IS_WRITE )
        .value( "PAYLOAD_SEMANTICS_IS_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE )
        .export_values();
#endif //OPTIX_VERSION >= 70400


    py::enum_<OptixProgramGroupKind>(m, "ProgramGroupKind", py::arithmetic())
        .value( "PROGRAM_GROUP_KIND_RAYGEN", OPTIX_PROGRAM_GROUP_KIND_RAYGEN )
        .value( "PROGRAM_GROUP_KIND_MISS", OPTIX_PROGRAM_GROUP_KIND_MISS )
        .value( "PROGRAM_GROUP_KIND_EXCEPTION", OPTIX_PROGRAM_GROUP_KIND_EXCEPTION )
        .value( "PROGRAM_GROUP_KIND_HITGROUP", OPTIX_PROGRAM_GROUP_KIND_HITGROUP )
        .value( "PROGRAM_GROUP_KIND_CALLABLES", OPTIX_PROGRAM_GROUP_KIND_CALLABLES )
        .export_values();

    py::enum_<OptixProgramGroupFlags>(m, "ProgramGroupFlags", py::arithmetic())
        .value( "PROGRAM_GROUP_FLAGS_NONE", OPTIX_PROGRAM_GROUP_FLAGS_NONE )
        .export_values();
py::enum_<OptixExceptionCodes>(m, "ExceptionCodes", py::arithmetic())
        .value( "EXCEPTION_CODE_STACK_OVERFLOW", OPTIX_EXCEPTION_CODE_STACK_OVERFLOW )
        .value( "EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED", OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED )
        .value( "EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED", OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED )
        .value( "EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE", OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE )
        .value( "EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT", OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT )
        .value( "EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT", OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT )
#if OPTIX_VERSION >= 70100
        .value( "EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE", OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE )
        .value( "EXCEPTION_CODE_INVALID_RAY", OPTIX_EXCEPTION_CODE_INVALID_RAY )
        .value( "EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH", OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH )
        .value( "EXCEPTION_CODE_BUILTIN_IS_MISMATCH", OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH )
        .value( "EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS", OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS )
#endif
#if OPTIX_VERSION >= 70200
        .value( "EXCEPTION_CODE_CALLABLE_INVALID_SBT", OPTIX_EXCEPTION_CODE_CALLABLE_INVALID_SBT )
        .value( "EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD", OPTIX_EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD )
        .value( "EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD", OPTIX_EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD )
#endif
        .export_values();

    py::enum_<OptixExceptionFlags>(m, "ExceptionFlags", py::arithmetic())
        .value( "EXCEPTION_FLAG_NONE", OPTIX_EXCEPTION_FLAG_NONE )
        .value( "EXCEPTION_FLAG_STACK_OVERFLOW", OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW )
        .value( "EXCEPTION_FLAG_TRACE_DEPTH", OPTIX_EXCEPTION_FLAG_TRACE_DEPTH )
        .value( "EXCEPTION_FLAG_USER", OPTIX_EXCEPTION_FLAG_USER )
        .value( "EXCEPTION_FLAG_DEBUG", OPTIX_EXCEPTION_FLAG_DEBUG )
        .export_values();

    py::enum_<OptixQueryFunctionTableOptions>(m, "QueryFunctionTableOptions", py::arithmetic())
        .value( "QUERY_FUNCTION_TABLE_OPTION_DUMMY", OPTIX_QUERY_FUNCTION_TABLE_OPTION_DUMMY )
        .export_values();


    /*
#if OPTIX_VERSION >= 70400
    py::enum_<OptixPayloadSemantics>(m, "PayloadSemantics", py::arithmetic())
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_NONE", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_NONE )
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_READ", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ )
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE )
        .value( "PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE )
        .value( "PAYLOAD_SEMANTICS_CH_READ", OPTIX_PAYLOAD_SEMANTICS_CH_READ )
        .value( "PAYLOAD_SEMANTICS_CH_WRITE", OPTIX_PAYLOAD_SEMANTICS_CH_WRITE )
        .value( "PAYLOAD_SEMANTICS_CH_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE )
        .value( "PAYLOAD_SEMANTICS_MS_NONE", OPTIX_PAYLOAD_SEMANTICS_MS_NONE )
        .value( "PAYLOAD_SEMANTICS_MS_READ", OPTIX_PAYLOAD_SEMANTICS_MS_READ )
        .value( "PAYLOAD_SEMANTICS_MS_WRITE", OPTIX_PAYLOAD_SEMANTICS_MS_WRITE )
        .value( "PAYLOAD_SEMANTICS_MS_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE )
        .value( "PAYLOAD_SEMANTICS_AH_NONE", OPTIX_PAYLOAD_SEMANTICS_AH_NONE )
        .value( "PAYLOAD_SEMANTICS_AH_READ", OPTIX_PAYLOAD_SEMANTICS_AH_READ )
        .value( "PAYLOAD_SEMANTICS_AH_WRITE", OPTIX_PAYLOAD_SEMANTICS_AH_WRITE )
        .value( "PAYLOAD_SEMANTICS_AH_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE )
        .value( "PAYLOAD_SEMANTICS_IS_NONE", OPTIX_PAYLOAD_SEMANTICS_IS_NONE )
        .value( "PAYLOAD_SEMANTICS_IS_READ", OPTIX_PAYLOAD_SEMANTICS_IS_READ )
        .value( "PAYLOAD_SEMANTICS_IS_WRITE", OPTIX_PAYLOAD_SEMANTICS_IS_WRITE )
        .value( "PAYLOAD_SEMANTICS_IS_READ_WRITE", OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE )
        .export_values();
#endif
*/


    //---------------------------------------------------------------------------
    //
    // Opaque types
    //
    //---------------------------------------------------------------------------

    py::class_<pyoptix::DeviceContext>( m, "DeviceContext" )
        .def( "destroy", &pyoptix::deviceContextDestroy )
        .def( "getProperty", &pyoptix::deviceContextGetProperty )
        .def( "setLogCallback", &pyoptix::deviceContextSetLogCallback )
        .def( "setCacheEnabled", &pyoptix::deviceContextSetCacheEnabled )
        .def( "setCacheLocation", &pyoptix::deviceContextSetCacheLocation )
        .def( "setCacheDatabaseSizes", &pyoptix::deviceContextSetCacheDatabaseSizes )
        .def( "getCacheEnabled", &pyoptix::deviceContextGetCacheEnabled )
        .def( "getCacheLocation", &pyoptix::deviceContextGetCacheLocation )
        .def( "getCacheDatabaseSizes", &pyoptix::deviceContextGetCacheDatabaseSizes )
        .def( "pipelineCreate", &pyoptix::pipelineCreate )
        .def( "moduleCreateFromPTX", &pyoptix::moduleCreateFromPTX )
        IF_OPTIX71(
        .def( "builtinISModuleGet", &pyoptix::builtinISModuleGet )
        )
        .def( "programGroupCreate", &pyoptix::programGroupCreate )
        .def( "accelComputeMemoryUsage", &pyoptix::accelComputeMemoryUsage )
        .def( "accelBuild", &pyoptix::accelBuild )
        .def( "accelGetRelocationInfo", &pyoptix::accelGetRelocationInfo )
        .def( "accelCheckRelocationCompatibility", &pyoptix::accelCheckRelocationCompatibility )
        .def( "accelRelocate", &pyoptix::accelRelocate )
        .def( "accelCompact", &pyoptix::accelCompact )
        .def( "denoiserCreate", &pyoptix::denoiserCreate )
        .def(py::self == py::self)
        ;

    py::class_<pyoptix::Module>( m, "Module" )
        .def( "destroy", &pyoptix::moduleDestroy )
        .def(py::self == py::self)
        ;

    py::class_<pyoptix::ProgramGroup>( m, "ProgramGroup" )
        .def( "getStackSize", &pyoptix::programGroupGetStackSize )
        .def( "destroy", &pyoptix::programGroupDestroy )
        .def(py::self == py::self)
        ;

    py::class_<pyoptix::Pipeline>( m, "Pipeline" )
        .def( "destroy", &pyoptix::pipelineDestroy )
        .def( "setStackSize", &pyoptix::pipelineSetStackSize )
        .def(py::self == py::self)
        ;

    py::class_<pyoptix::Denoiser>( m, "Denoiser" )
#if OPTIX_VERSION < 70300
        .def( "setModel", &pyoptix::denoiserSetModel )
#endif
        .def( "destroy", &pyoptix::denoiserDestroy )
        .def( "computeMemoryResources", &pyoptix::denoiserComputeMemoryResources )
        .def( "setup", &pyoptix::denoiserSetup )
        .def( "invoke", &pyoptix::denoiserInvoke )
        .def( "computeIntensity", &pyoptix::denoiserComputeIntensity )
        IF_OPTIX73( .def( "computeAverageColor", &pyoptix::denoiserComputeAverageColor ) )
        .def(py::self == py::self)
        ;


    //---------------------------------------------------------------------------
    //
    // Param types
    //
    //---------------------------------------------------------------------------

    py::class_<pyoptix::DeviceContextOptions>(m, "DeviceContextOptions")
        .def(
            py::init< py::object, int32_t, OptixDeviceContextValidationMode>(),
            py::arg( "logCallbackFunction" )=py::none(),
            py::arg( "logCallbackLevel"    )=0,
            IF_OPTIX72( py::arg( "validationMode" )=OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF )
        )
        .def_property( "logCallbackFunction",
            [](const pyoptix::DeviceContextOptions& self)
            { return self.logCallbackFunction; },
            [](pyoptix::DeviceContextOptions& self, py::object val)
            {
                self.logCallbackFunction= val;
                self.options.logCallbackFunction = pyoptix::context_log_cb;
                self.options.logCallbackData = val.ptr();
            }
        )
        .def_property("logCallbackLevel",
            [](const pyoptix::DeviceContextOptions& self)
            { return self.options.logCallbackLevel;},
            [](pyoptix::DeviceContextOptions& self, int32_t val)
            { self.options.logCallbackLevel = val; }
        )
#if OPTIX_VERSION >= 70200
        .def_property("validationMode",
            [](const pyoptix::DeviceContextOptions& self)
            { return self.options.validationMode; },
            [](pyoptix::DeviceContextOptions& self, OptixDeviceContextValidationMode val)
            { self.options.validationMode = val; }
        )
#endif
        ;


    py::class_<pyoptix::BuildInputTriangleArray>(m, "BuildInputTriangleArray")
        .def(
            py::init<
                const py::list&,
                OptixVertexFormat,
                unsigned int,
                CUdeviceptr,
                unsigned int,
                OptixIndicesFormat,
                unsigned int,
                CUdeviceptr,
                const py::list&,
                unsigned int,
                CUdeviceptr,
                unsigned int,
                unsigned int,
                unsigned int
                IF_OPTIX71( COMMA  OptixTransformFormat )
            >(),
            py::arg( "vertexBuffers_"              ) = py::list(), // list of CUdeviceptr
            py::arg( "vertexFormat"                ) = 
                IF_OPTIX71_ELSE( OPTIX_VERTEX_FORMAT_NONE, static_cast<OptixVertexFormat>(0x0000u) 
            ),
            py::arg( "vertexStrideInBytes"         ) = 0u,
            py::arg( "indexBuffer"                 ) = 0u,
            py::arg( "numIndexTriplets"            ) = 0u,
            py::arg( "indexFormat"                 ) = 
                IF_OPTIX71_ELSE( OPTIX_INDICES_FORMAT_NONE, static_cast<OptixIndicesFormat>(0x0000u) 
            ),
            py::arg( "indexStrideInBytes"          ) = 0u,
            py::arg( "preTransform"                ) = 0u,
            py::arg( "flags_"                      ) = py::list(), // list of uint32_t
            py::arg( "numSbtRecords"               ) = 0u,
            py::arg( "sbtIndexOffsetBuffer"        ) = 0u,
            py::arg( "sbtIndexOffsetSizeInBytes"   ) = 0u,
            py::arg( "sbtIndexOffsetStrideInBytes" ) = 0u,
            py::arg( "primitiveIndexOffset"        ) = 0u
            IF_OPTIX71( COMMA  
            py::arg( "transformFormat"             ) = OPTIX_TRANSFORM_FORMAT_NONE
            )
        )
        .def_property( "vertexBuffers",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return py::cast( self.vertexBuffers ); },
            [](pyoptix::BuildInputTriangleArray& self, py::list& val)
            { self.vertexBuffers =  val.cast<std::vector<CUdeviceptr> >(); }
            )
        .def_property( "numVertices",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.numVertices; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.numVertices = val; }
            )
        .def_property( "vertexFormat",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.vertexFormat; },
            [](pyoptix::BuildInputTriangleArray& self, OptixVertexFormat val)
            { self.build_input.vertexFormat = val; }
            )
        .def_property( "vertexStrideInBytes",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.vertexStrideInBytes; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.vertexStrideInBytes = val; }
            )
        .def_property( "indexBuffer",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.indexBuffer; },
            [](pyoptix::BuildInputTriangleArray& self, CUdeviceptr val)
            { self.build_input.indexBuffer = val; }
            )
        .def_property( "numIndexTriplets",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.numIndexTriplets; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.numIndexTriplets = val; }
            )
        .def_property( "indexFormat",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.indexFormat; },
            [](pyoptix::BuildInputTriangleArray& self, OptixIndicesFormat val)
            { self.build_input.indexFormat = val; }
            )
        .def_property( "indexStrideInBytes",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.indexStrideInBytes; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.indexStrideInBytes = val; }
            )
        .def_property( "preTransform",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.preTransform; },
            [](pyoptix::BuildInputTriangleArray& self, CUdeviceptr val)
            { self.build_input.preTransform = val; }
            )
        .def_property( "flags",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return py::cast( self.flags ); },
            [](pyoptix::BuildInputTriangleArray& self, py::list& val)
            { self.flags =  val.cast<std::vector<unsigned int> >(); }
            )
        .def_property( "numSbtRecords",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.numSbtRecords; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.numSbtRecords = val; }
            )
        .def_property( "sbtIndexOffsetBuffer",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.sbtIndexOffsetBuffer; },
            [](pyoptix::BuildInputTriangleArray& self, CUdeviceptr val)
            { self.build_input.sbtIndexOffsetBuffer = val; }
            )
        .def_property( "sbtIndexOffsetSizeInBytes",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.sbtIndexOffsetSizeInBytes; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.sbtIndexOffsetSizeInBytes = val; }
            )
        .def_property( "sbtIndexOffsetStrideInBytes",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.sbtIndexOffsetStrideInBytes; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.sbtIndexOffsetStrideInBytes = val; }
            )
        .def_property( "primitiveIndexOffset",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.primitiveIndexOffset; },
            [](pyoptix::BuildInputTriangleArray& self, unsigned int val)
            { self.build_input.primitiveIndexOffset = val; }
            )
#if OPTIX_VERSION >= 70100
        .def_property( "transformFormat",
            []( const pyoptix::BuildInputTriangleArray& self )
            { return self.build_input.transformFormat; },
            [](pyoptix::BuildInputTriangleArray& self, OptixTransformFormat val)
            { self.build_input.transformFormat = val; }
            )
#endif
        ;


#if OPTIX_VERSION > 70100
    py::class_<pyoptix::BuildInputCurveArray>(m, "BuildInputCurveArray")
        .def( 
            py::init< 
                OptixPrimitiveType,
                unsigned int, 
                const py::list&,
                unsigned int,
                unsigned int,
                const py::list&,
                unsigned int,
                const py::list&,
                unsigned int,
                CUdeviceptr,
                unsigned int,
                unsigned int,
                unsigned int
            >(), 
            py::arg( "curveType"                   ) = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
            py::arg( "numPrimitives"               ) = 0u,
            py::arg( "vertexBuffers"               ) = py::list(), // list of CUdeviceptr
            py::arg( "numVertices"                 ) = 0u,
            py::arg( "vertexStrideInBytes"         ) = 0u,
            py::arg( "widthBuffers"                ) = py::list(), // list of CUdeviceptr
            py::arg( "widthStrideInBytes"          ) = 0u,
            py::arg( "normalBuffers"               ) = py::list(), // list of CUdeviceptr
            py::arg( "normalStrideInBytes"         ) = 0u,
            py::arg( "indexBuffer"                 ) = 0llu,
            py::arg( "indexStrideInBytes"          ) = 0u,
            py::arg( "flag"                        ) = 0u,
            py::arg( "primitiveIndexOffset"        ) = 0u
        )
        .def_property( "curveType", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.curveType; }, 
            [](pyoptix::BuildInputCurveArray& self, OptixPrimitiveType val) 
            { self.build_input.curveType = val; }
            )
        .def_property( "numPrimitives", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.numPrimitives; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.numPrimitives = val; }
            )
        .def_property( "vertexBuffers", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return py::cast( self.vertexBuffers ); }, 
            [](pyoptix::BuildInputCurveArray& self, py::list& val) 
            { self.vertexBuffers =  val.cast<std::vector<CUdeviceptr> >(); }
            )
        .def_property( "numVertices", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.numVertices; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.numVertices = val; }
            )
        .def_property( "vertexStrideInBytes", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.vertexStrideInBytes; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.vertexStrideInBytes = val; }
            )
        .def_property( "widthBuffers", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return py::cast( self.widthBuffers ); }, 
            [](pyoptix::BuildInputCurveArray& self, py::list& val) 
            { self.widthBuffers =  val.cast<std::vector<CUdeviceptr> >(); }
            )
        .def_property( "widthStrideInBytes", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.widthStrideInBytes; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.widthStrideInBytes = val; }
            )
        .def_property( "normalBuffers", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return py::cast( self.normalBuffers ); }, 
            [](pyoptix::BuildInputCurveArray& self, py::list& val) 
            { self.normalBuffers =  val.cast<std::vector<CUdeviceptr> >(); }
            )
        .def_property( "normalStrideInBytes", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.normalStrideInBytes; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.normalStrideInBytes = val; }
            )
        .def_property( "indexBuffer", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.indexBuffer; }, 
            [](pyoptix::BuildInputCurveArray& self, CUdeviceptr val) 
            { self.build_input.indexBuffer = val; }
            )
        .def_property( "indexStrideInBytes", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.indexStrideInBytes; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.indexStrideInBytes = val; }
            )
        .def_property( "flag", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.flag; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.flag = val; }
            )
        .def_property( "primitiveIndexOffset", 
            []( const pyoptix::BuildInputCurveArray& self ) 
            { return self.build_input.primitiveIndexOffset; }, 
            [](pyoptix::BuildInputCurveArray& self, unsigned int val) 
            { self.build_input.primitiveIndexOffset = val; }
            )
         ;
#endif // OPTIX_VERSION > 70100


    /* NOTE: Not very useful in python host-side
    py::class_<OptixAabb>(m, "Aabb")
        .def( py::init([]() { return std::unique_ptr<OptixAabb>(new OptixAabb{} ); } ) )
        .def_readwrite( "minX", &OptixAabb::minX )
        .def_readwrite( "minY", &OptixAabb::minY )
        .def_readwrite( "minZ", &OptixAabb::minZ )
        .def_readwrite( "maxX", &OptixAabb::maxX )
        .def_readwrite( "maxY", &OptixAabb::maxY )
        .def_readwrite( "maxZ", &OptixAabb::maxZ )
        ;
    */


    py::class_<pyoptix::BuildInputCustomPrimitiveArray>(m, "BuildInputCustomPrimitiveArray")
        .def( 
            py::init< 
                const py::list&,
                unsigned int,
                unsigned int,
                const py::list&,
                unsigned int,
                CUdeviceptr,
                unsigned int,
                unsigned int,
                unsigned int
                >(), 
            py::arg( "aabbBuffers"                 ) = 0u,
            py::arg( "numPrimitives"               ) = 0u,
            py::arg( "strideInBytes"               ) = 0u,
            py::arg( "flags"                       ) = 0u,
            py::arg( "numSbtRecords"               ) = 0u,
            py::arg( "sbtIndexOffsetBuffer"        ) = 0u,
            py::arg( "sbtIndexOffsetSizeInBytes"   ) = 0u,
            py::arg( "sbtIndexOffsetStrideInBytes" ) = 0u,
            py::arg( "primitiveIndexOffset"        ) = 0u
        )
        .def_property( "aabbBuffers", 
            []( const pyoptix::BuildInputCustomPrimitiveArray& self ) 
            { return py::cast( self.aabbBuffers ); }, 
            [](pyoptix::BuildInputCustomPrimitiveArray& self, py::list& val) 
            { self.aabbBuffers =  val.cast<std::vector<CUdeviceptr> >(); }
            )
        .def_property( "numPrimitives", 
            []( const pyoptix::BuildInputCustomPrimitiveArray& self ) 
            { return self.build_input.numPrimitives; }, 
            [](pyoptix::BuildInputCustomPrimitiveArray& self, unsigned int val) 
            { self.build_input.numPrimitives = val; }
            )
        .def_property( "strideInBytes", 
            []( const pyoptix::BuildInputCustomPrimitiveArray& self ) 
            { return self.build_input.strideInBytes; }, 
            [](pyoptix::BuildInputCustomPrimitiveArray& self, unsigned int val) 
            { self.build_input.strideInBytes = val; }
            )
        .def_property( "flags",
            []( const pyoptix::BuildInputCustomPrimitiveArray& self )
            { return py::cast( self.flags ); },
            [](pyoptix::BuildInputCustomPrimitiveArray& self, py::list& val)
            { self.flags =  val.cast<std::vector<unsigned int> >(); }
            )
        .def_property( "numSbtRecords",
            []( const pyoptix::BuildInputCustomPrimitiveArray& self )
            { return self.build_input.numSbtRecords; },
            [](pyoptix::BuildInputCustomPrimitiveArray& self, unsigned int val)
            { self.build_input.numSbtRecords = val; }
            )
        .def_property( "sbtIndexOffsetBuffer",
            []( const pyoptix::BuildInputCustomPrimitiveArray& self )
            { return self.build_input.sbtIndexOffsetBuffer; },
            [](pyoptix::BuildInputCustomPrimitiveArray& self, CUdeviceptr val)
            { self.build_input.sbtIndexOffsetBuffer = val; }
            )
        .def_property( "sbtIndexOffsetSizeInBytes",
            []( const pyoptix::BuildInputCustomPrimitiveArray& self )
            { return self.build_input.sbtIndexOffsetSizeInBytes; },
            [](pyoptix::BuildInputCustomPrimitiveArray& self, unsigned int val)
            { self.build_input.sbtIndexOffsetSizeInBytes = val; }
            )
        .def_property( "sbtIndexOffsetStrideInBytes",
            []( const pyoptix::BuildInputCustomPrimitiveArray& self )
            { return self.build_input.sbtIndexOffsetStrideInBytes; },
            [](pyoptix::BuildInputCustomPrimitiveArray& self, unsigned int val)
            { self.build_input.sbtIndexOffsetStrideInBytes = val; }
            )
        .def_property( "primitiveIndexOffset",
            []( const pyoptix::BuildInputCustomPrimitiveArray& self )
            { return self.build_input.primitiveIndexOffset; },
            [](pyoptix::BuildInputCustomPrimitiveArray& self, unsigned int val)
            { self.build_input.primitiveIndexOffset = val; }
            )
        ;


    py::class_<pyoptix::BuildInputInstanceArray>(m, "BuildInputInstanceArray")
        .def( 
            py::init< 
                CUdeviceptr,
                CUdeviceptr,
                unsigned int
                >(), 
            py::arg( "instances"        ) = 0u,
            py::arg( "instancePointers" ) = 0u,
            py::arg( "numInstances"     ) = 0u
        )
        .def_property( "instances",
            []( const pyoptix::BuildInputInstanceArray& self )
            { return self.build_input.instances; },
            [](pyoptix::BuildInputInstanceArray& self, CUdeviceptr val)
            { self.setInstances( val ); }
            )
        .def_property( "instancePointers",
            []( const pyoptix::BuildInputInstanceArray& self )
            { return self.build_input.instances; },
            [](pyoptix::BuildInputInstanceArray& self, CUdeviceptr val)
            { self.setInstancePointers( val ); }
            )
        .def_property( "numInstances", 
            []( const pyoptix::BuildInputInstanceArray& self ) 
            { return self.build_input.numInstances; }, 
            [](pyoptix::BuildInputInstanceArray& self, unsigned int val) 
            { self.build_input.numInstances = val; }
            )
        ;


    /* NOTE: Wrapper type OptixBuildInput not used in python bindings
    py::class_<OptixBuildInput>(m, "BuildInput")
        .def( py::init([]() { return std::unique_ptr<OptixBuildInput>(new OptixBuildInput{} ); } ) )
        .def_readwrite( "type", &OptixBuildInput::type )
        .def_readwrite( "triangleArray", &OptixBuildInput::triangleArray )
#if OPTIX_VERSION >= 70100
        .def_readwrite( "curveArray", &OptixBuildInput::curveArray )
        .def_readwrite( "customPrimitiveArray", &OptixBuildInput::customPrimitiveArray )
#else
        .def_readwrite( "aabbArray", &OptixBuildInput::aabbArray )
#endif
        .def_readwrite( "instanceArray", &OptixBuildInput::instanceArray )
        ;
    */

    py::class_<OptixInstance>(m, "Instance")
        .def( py::init([]() { return std::unique_ptr<OptixInstance>(new OptixInstance{} ); } ) )
        .def_readwrite( "instanceId", &OptixInstance::instanceId )
        .def_readwrite( "sbtOffset", &OptixInstance::sbtOffset )
        .def_readwrite( "visibilityMask", &OptixInstance::visibilityMask )
        .def_readwrite( "flags", &OptixInstance::flags )
        .def_readwrite( "traversableHandle", &OptixInstance::traversableHandle )
        ;

    py::class_<OptixMotionOptions>(m, "MotionOptions")
        .def( py::init([]() { return std::unique_ptr<OptixMotionOptions>(new OptixMotionOptions{} ); } ) )
        .def_readwrite( "numKeys", &OptixMotionOptions::numKeys )
        .def_readwrite( "flags", &OptixMotionOptions::flags )
        .def_readwrite( "timeBegin", &OptixMotionOptions::timeBegin )
        .def_readwrite( "timeEnd", &OptixMotionOptions::timeEnd )
        ;

    py::class_<OptixAccelBuildOptions>(m, "AccelBuildOptions")
        .def( py::init(
                []( unsigned int buildFlags,
                    OptixBuildOperation operation,
                    const OptixMotionOptions& motionOptions
                    )
                {
                    auto opts = std::unique_ptr<OptixAccelBuildOptions>(new OptixAccelBuildOptions{} );
                    opts->buildFlags = buildFlags;
                    opts->operation = operation;
                    opts->motionOptions = motionOptions;
                    return opts;
                }
            ),
            py::arg( "buildFlags"    ) = 0,
            py::arg( "operation"     ) = OPTIX_BUILD_OPERATION_BUILD,
            py::arg( "motionOptions" ) = OptixMotionOptions{}
        )
        .def_readwrite( "buildFlags", &OptixAccelBuildOptions::buildFlags )
        .def_readwrite( "operation", &OptixAccelBuildOptions::operation )
        .def_readwrite( "motionOptions", &OptixAccelBuildOptions::motionOptions )
        ;

    py::class_<OptixAccelBufferSizes>(m, "AccelBufferSizes")
        .def( py::init([]() { return std::unique_ptr<OptixAccelBufferSizes>(new OptixAccelBufferSizes{} ); } ) )
        .def_readwrite( "outputSizeInBytes", &OptixAccelBufferSizes::outputSizeInBytes )
        .def_readwrite( "tempSizeInBytes", &OptixAccelBufferSizes::tempSizeInBytes )
        .def_readwrite( "tempUpdateSizeInBytes", &OptixAccelBufferSizes::tempUpdateSizeInBytes )
        ;

    py::class_<OptixAccelEmitDesc>(m, "AccelEmitDesc")
        .def( py::init([]() { return std::unique_ptr<OptixAccelEmitDesc>(new OptixAccelEmitDesc{} ); } ) )
        .def_readwrite( "result", &OptixAccelEmitDesc::result )
        .def_readwrite( "type", &OptixAccelEmitDesc::type )
        ;

    py::class_<OptixAccelRelocationInfo>(m, "AccelRelocationInfo")
        .def( py::init([]() { return std::unique_ptr<OptixAccelRelocationInfo>(new OptixAccelRelocationInfo{} ); } ) )
        ;

    py::class_<OptixStaticTransform>(m, "StaticTransform")
        .def( py::init([]() { return std::unique_ptr<OptixStaticTransform>(new OptixStaticTransform{} ); } ) )
        .def_readwrite( "child", &OptixStaticTransform::child )
        ;

    py::class_<OptixMatrixMotionTransform>(m, "MatrixMotionTransform")
        .def( py::init([]() { return std::unique_ptr<OptixMatrixMotionTransform>(new OptixMatrixMotionTransform{} ); } ) )
        .def_readwrite( "child", &OptixMatrixMotionTransform::child )
        .def_readwrite( "motionOptions", &OptixMatrixMotionTransform::motionOptions )
        ;

    py::class_<OptixSRTData>(m, "SRTData")
        .def( py::init([]() { return std::unique_ptr<OptixSRTData>(new OptixSRTData{} ); } ) )
        .def_readwrite( "tz", &OptixSRTData::tz )
        ;

    py::class_<OptixSRTMotionTransform>(m, "SRTMotionTransform")
        .def( py::init([]() { return std::unique_ptr<OptixSRTMotionTransform>(new OptixSRTMotionTransform{} ); } ) )
        .def_readwrite( "child", &OptixSRTMotionTransform::child )
        .def_readwrite( "motionOptions", &OptixSRTMotionTransform::motionOptions )
        ;

    py::class_<OptixImage2D>(m, "Image2D")
        .def( py::init([]() { return std::unique_ptr<OptixImage2D>(new OptixImage2D{} ); } ) )
        .def_readwrite( "data", &OptixImage2D::data )
        .def_readwrite( "width", &OptixImage2D::width )
        .def_readwrite( "height", &OptixImage2D::height )
        .def_readwrite( "rowStrideInBytes", &OptixImage2D::rowStrideInBytes )
        .def_readwrite( "pixelStrideInBytes", &OptixImage2D::pixelStrideInBytes )
        .def_readwrite( "format", &OptixImage2D::format )
        ;

#if OPTIX_VERSION == 70300
    py::class_<OptixDenoiserOptions>(m, "DenoiserOptions")
        .def( py::init([]() { return std::unique_ptr<OptixDenoiserOptions>(new OptixDenoiserOptions{} ); } ) )
        .def_readwrite( "guideAlbedo", &OptixDenoiserOptions::guideAlbedo )
        .def_readwrite( "guideNormal", &OptixDenoiserOptions::guideNormal )
        ;
#elif OPTIX_VERSION <= 70200
    py::class_<OptixDenoiserOptions>(m, "DenoiserOptions")
        .def( py::init([]() { return std::unique_ptr<OptixDenoiserOptions>(new OptixDenoiserOptions{} ); } ) )
        .def_readwrite( "inputKind", &OptixDenoiserOptions::inputKind )
        ;
#endif

    py::class_<OptixDenoiserParams>(m, "DenoiserParams")
        .def( py::init([]() { return std::unique_ptr<OptixDenoiserParams>(new OptixDenoiserParams{} ); } ) )
        .def_readwrite( "denoiseAlpha", &OptixDenoiserParams::denoiseAlpha )
        .def_readwrite( "hdrIntensity", &OptixDenoiserParams::hdrIntensity )
        .def_readwrite( "blendFactor", &OptixDenoiserParams::blendFactor )
        IF_OPTIX72(
        .def_readwrite( "hdrAverageColor", &OptixDenoiserParams::hdrAverageColor )
        )
        ;

    py::class_<OptixDenoiserSizes>(m, "DenoiserSizes")
        .def( py::init([]() { return std::unique_ptr<OptixDenoiserSizes>(new OptixDenoiserSizes{} ); } ) )
        .def_readwrite( "stateSizeInBytes", &OptixDenoiserSizes::stateSizeInBytes )
#if OPTIX_VERSION > 70000
        .def_readwrite( "withOverlapScratchSizeInBytes", &OptixDenoiserSizes::withOverlapScratchSizeInBytes )
        .def_readwrite( "withoutOverlapScratchSizeInBytes", &OptixDenoiserSizes::withoutOverlapScratchSizeInBytes )
#else
        .def_readwrite( "minimumScratchSizeInBytes", &OptixDenoiserSizes::minimumScratchSizeInBytes )
        .def_readwrite( "recommendedScratchSizeInBytes", &OptixDenoiserSizes::recommendedScratchSizeInBytes )
#endif
        .def_readwrite( "overlapWindowSizeInPixels", &OptixDenoiserSizes::overlapWindowSizeInPixels )
        ;


#if OPTIX_VERSION >= 70200
    py::class_<pyoptix::ModuleCompileBoundValueEntry>(
            m, "ModuleCompileBoundValueEntry"
            )
        .def(
            py::init< size_t, py::buffer, const std::string& >(),
            py::arg( "pipelineParamOffsetInBytes" ) = 0u,
            py::arg( "boundValue"                 ) = py::bytes(),
            py::arg( "annotation"                 ) = ""
            )
        .def_property( "pipelineParamOffsetInBytes",
            [](const pyoptix::ModuleCompileBoundValueEntry& self)
            { return self.entry.pipelineParamOffsetInBytes; },
            [](pyoptix::ModuleCompileBoundValueEntry& self, size_t val)
            { self.entry.pipelineParamOffsetInBytes = val;  }
            )
        .def_property_readonly( "sizeInBytes",
            [](const pyoptix::ModuleCompileBoundValueEntry& self)
            { return self.entry.sizeInBytes; }
            )
        .def_property( "boundValue",
            //[](const pyoptix::ModuleCompileBoundValueEntry& self)
            //{ return self.boundValue; },
            nullptr,
            [](pyoptix::ModuleCompileBoundValueEntry& self, py::buffer val)
            { self.setBoundValue( val );  }
            )
        .def_property( "annotation",
            [](const pyoptix::ModuleCompileBoundValueEntry& self)
            { return self.annotation; },
            [](pyoptix::ModuleCompileBoundValueEntry& self,
               std::string&& val)
            { self.setAnnotation( std::move( val ) );  }
            )
        ;
#endif // OPTIX_VERSION >= 70200


#if OPTIX_VERSION >= 70400
    py::class_<pyoptix::PayloadType>(
            m, "PayloadType"
            )
        .def(
            py::init<py::list>(),
            py::arg( "payloadSemantics" ) = py::list()
            )
        .def_property( "pipelineParamOffsetInBytes",
            [](const pyoptix::ModuleCompileBoundValueEntry& self)
            { return self.entry.pipelineParamOffsetInBytes; },
            [](pyoptix::ModuleCompileBoundValueEntry& self, size_t val)
            { self.entry.pipelineParamOffsetInBytes = val;  }
            )
        .def_property( "payloadSemantics",
            //[](const pyoptix::PayloadType& self)
            //{ return self.payloadSemantics; },
            nullptr,
            [](pyoptix::PayloadType& self, py::list val)
            { self.setPayloadSemantics( val );  }
            )
        ;
#endif // OPTIX_VERSION >= 70400


    py::class_<pyoptix::ModuleCompileOptions>(m, "ModuleCompileOptions")
        .def(
            py::init<
                int32_t,
                OptixCompileOptimizationLevel,
                OptixCompileDebugLevel
                IF_OPTIX72( COMMA std::vector<pyoptix::ModuleCompileBoundValueEntry>&& )
                IF_OPTIX74( COMMA std::vector<pyoptix::PayloadType>&&  )
                >(),
            py::arg( "maxRegisterCount" ) = 0u,
            py::arg( "optLevel"         ) = OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            py::arg( "debugLevel"       ) = 
            IF_OPTIX71_ELSE( 
                OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT, OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO 
            )
            IF_OPTIX72( COMMA py::arg( "boundValues"  ) = std::vector<pyoptix::ModuleCompileBoundValueEntry>() )
            IF_OPTIX74( COMMA py::arg( "payloadTypes" ) = std::vector<pyoptix::PayloadType>() )
            )
        .def_property( "maxRegisterCount",
            [](const pyoptix::ModuleCompileOptions& self)
            { return self.options.maxRegisterCount; },
            [](pyoptix::ModuleCompileOptions& self, int32_t val)
            { self.options.maxRegisterCount = val;  }
            )
        .def_property( "optLevel",
            [](const pyoptix::ModuleCompileOptions& self)
            { return self.options.optLevel; },
            [](pyoptix::ModuleCompileOptions& self,
               OptixCompileOptimizationLevel val)
            { self.options.optLevel = val; }
            )
        .def_property( "debugLevel",
            [](const pyoptix::ModuleCompileOptions& self)
            { return self.options.debugLevel; },
            [](pyoptix::ModuleCompileOptions& self,
               OptixCompileDebugLevel val)
            { self.options.debugLevel = val; }
            )
#if OPTIX_VERSION >= 70200
        .def_property( "boundValues",
            // This doesnt do what you probably want it to so disable it
            //[](const pyoptix::ModuleCompileOptions& self)
            //{ return self.boundValues; },
            nullptr,
            [](pyoptix::ModuleCompileOptions& self,
               std::vector<pyoptix::ModuleCompileBoundValueEntry>&& val )
            { self.pyboundValues = std::move( val ); }
            )
#endif
#if OPTIX_VERSION >= 70400
        .def_property( "payloadTypes",
            // This doesnt do what you probably want it to so disable it
            //[](const pyoptix::PayloadType& self)
            //{ return self.payloadTypes; },
            nullptr,
            [](pyoptix::ModuleCompileOptions& self,
               std::vector<pyoptix::PayloadType>&& val )
            { self.pypayloadTypes = std::move( val ); }
            )
#endif
        ;


    py::class_<pyoptix::ProgramGroupDesc>(m, "ProgramGroupDesc")
        .def(
            py::init<
                uint32_t,

                const char*,             //  raygenEntryFunctionName
                const pyoptix::Module,   //  raygenModule

                const char*,             //  missEntryFunctionName
                const pyoptix::Module,   //  missModule

                const char*,             //  exceptionEntryFunctionName
                const pyoptix::Module,   //  exceptionModule

                const char*,             //  callablesEntryFunctionNameDC
                const pyoptix::Module,   //  callablesModuleDC

                const char*,             //  callablesEntryFunctionNameCC
                const pyoptix::Module,   //  callablesModuleCC

                const char*,             //  hitgroupEntryFunctionNameCH
                const pyoptix::Module,   //  hitgroupModuleCH
                const char*,             //  hitgroupEntryFunctionNameAH
                const pyoptix::Module,   //  hitgroupModuleAH
                const char*,             //  hitgroupEntryFunctionNameIS
                const pyoptix::Module    //  hitgroupModuleIS
                >(),
            py::arg( "flags"                        ) = 0u,
            py::arg( "raygenEntryFunctionName"      ) = nullptr,
            py::arg( "raygenModule"                 ) = pyoptix::Module{},
            py::arg( "missEntryFunctionName"        ) = nullptr,
            py::arg( "missModule"                   ) = pyoptix::Module{},
            py::arg( "exceptionEntryFunctionName"   ) = nullptr,
            py::arg( "exceptionModule"              ) = pyoptix::Module{},
            py::arg( "callablesEntryFunctionNameDC" ) = nullptr,
            py::arg( "callablesModuleDC"            ) = pyoptix::Module{},
            py::arg( "callablesEntryFunctionNameCC" ) = nullptr,
            py::arg( "callablesModuleCC"            ) = pyoptix::Module{},
            py::arg( "hitgroupEntryFunctionNameCH"  ) = nullptr,
            py::arg( "hitgroupModuleCH"             ) = pyoptix::Module{},
            py::arg( "hitgroupEntryFunctionNameAH"  ) = nullptr,
            py::arg( "hitgroupModuleAH"             ) = pyoptix::Module{},
            py::arg( "hitgroupEntryFunctionNameIS"  ) = nullptr,
            py::arg( "hitgroupModuleIS"             ) = pyoptix::Module{}
            )
        .def_property( "flags",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.program_group_desc.flags; },
            []( pyoptix::ProgramGroupDesc& self, uint32_t flags )
            { self.program_group_desc.flags = flags; }
        )
        .def_property( "raygenModule",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.raygen.module }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind          = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                self.program_group_desc.raygen.module = module.module;
            }
        )
        .def_property( "raygenEntryFunctionName",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName0; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                self.entryFunctionName0      = name;
            }
        )
        .def_property( "missModule",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.miss.module }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind        = OPTIX_PROGRAM_GROUP_KIND_MISS;
                self.program_group_desc.miss.module = module.module;
            }
        )
        .def_property( "missEntryFunctionName",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName0; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
                self.entryFunctionName0      = name;
            }
        )
        .def_property( "exceptionModule",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.exception.module }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind             = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
                self.program_group_desc.exception.module = module.module;
            }
        )
        .def_property( "exceptionEntryFunctionName",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName0; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
                self.entryFunctionName0      = name;
            }
        )
        .def_property( "callablesModuleDC",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.callables.moduleDC }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind               = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                self.program_group_desc.callables.moduleDC = module.module;
            }
        )
        .def_property( "callablesEntryFunctionNameDC",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName0; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                self.entryFunctionName0      = name;
            }
        )
        .def_property( "callablesModuleCC",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.callables.moduleCC }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind               = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                self.program_group_desc.callables.moduleCC = module.module;
            }
        )
        .def_property( "callablesEntryFunctionNameCC",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName1; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
                self.entryFunctionName1      = name;
            }
        )
        .def_property( "hitgroupModuleCH",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.hitgroup.moduleCH }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                self.program_group_desc.hitgroup.moduleCH = module.module;
            }
        )
        .def_property( "hitgroupEntryFunctionNameCH",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName0; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                self.entryFunctionName0      = name;
            }
        )
        .def_property( "hitgroupModuleAH",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.hitgroup.moduleAH }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                self.program_group_desc.hitgroup.moduleAH = module.module;
            }
        )
        .def_property( "hitgroupEntryFunctionNameAH",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName1; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                self.entryFunctionName1      = name;
            }
        )
        .def_property( "hitgroupModuleIS",
            []( pyoptix::ProgramGroupDesc& self )
            { return pyoptix::Module{ self.program_group_desc.hitgroup.moduleIS }; },
            []( pyoptix::ProgramGroupDesc& self, const pyoptix::Module& module )
            {
                self.program_group_desc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                self.program_group_desc.hitgroup.moduleIS = module.module;
            }
        )
        .def_property( "hitgroupEntryFunctionNameIS",
            []( pyoptix::ProgramGroupDesc& self )
            { return self.entryFunctionName2; },
            []( pyoptix::ProgramGroupDesc& self, const std::string& name )
            {
                self.program_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                self.entryFunctionName2      = name;
            }
        )
        ;


#if OPTIX_VERSION >= 70400
    py::class_<pyoptix::ProgramGroupOptions>(m, "ProgramGroupOptions")
        .def(
            py::init< 
                pyoptix::PayloadType
                >(),
            py::arg( "payloadType" ) = pyoptix::PayloadType{}
        )
        .def_property( "payloadType",
            // This doesnt do what you probably want it to so disable it
            //[](const pyoptix::ProgramGroupOptions& self)
            //{ return self.payload_type; },
            nullptr,
            []( pyoptix::ProgramGroupOptions& self, const pyoptix::PayloadType& payload_type )
            {
                self.setPayloadType( payload_type );
            }
        )
        ;
#endif // OPTIX_VERSION >= 70400


    py::class_<pyoptix::PipelineCompileOptions>(m, "PipelineCompileOptions")
        .def(
            py::init<
                bool,
                uint32_t,
                int32_t,
                int32_t,
                uint32_t,
                const char*
                IF_OPTIX71( COMMA int32_t )
	    >(),
            py::arg( "usesMotionBlur" )=0,
            py::arg( "traversableGraphFlags" )=0u,
            py::arg( "numPayloadValues" )=0,
            py::arg( "numAttributeValues" )=0,
            py::arg( "exceptionFlags" )=0u,
            py::arg( "pipelineLaunchParamsVariableName" )=nullptr
            IF_OPTIX71( COMMA py::arg( "usesPrimitiveTypeFlags" )=0 )
            )
        .def_property( "usesMotionBlur",
            [](const pyoptix::PipelineCompileOptions& self)
            { return self.options.usesMotionBlur; },
            [](pyoptix::PipelineCompileOptions& self, bool val)
            { self.options.usesMotionBlur = val; }
        )
        .def_property( "traversableGraphFlags",
            [](const pyoptix::PipelineCompileOptions& self)
            { return self.options.traversableGraphFlags; },
            [](pyoptix::PipelineCompileOptions& self, uint32_t val)
            { self.options.traversableGraphFlags = val; }
        )
        .def_property( "numPayloadValues",
            [](const pyoptix::PipelineCompileOptions& self)
            { return self.options.numPayloadValues; },
            [](pyoptix::PipelineCompileOptions& self, int val)
            { self.options.numPayloadValues = val; }
        )
        .def_property( "numAttributeValues",
            [](const pyoptix::PipelineCompileOptions& self)
            { return self.options.numAttributeValues; },
            [](pyoptix::PipelineCompileOptions& self, int val)
            { self.options.numAttributeValues = val; }
        )
        .def_property( "exceptionFlags",
            [](const pyoptix::PipelineCompileOptions& self)
            { return self.options.exceptionFlags; },
            [](pyoptix::PipelineCompileOptions& self, uint32_t val)
            { self.options.exceptionFlags = val; }
        )
        .def_readwrite(
            "pipelineLaunchParamsVariableName",
            &pyoptix::PipelineCompileOptions::pipelineLaunchParamsVariableName
        )
#if OPTIX_VERSION >= 70100
        .def_property( "usesPrimitiveTypeFlags",
            [](const pyoptix::PipelineCompileOptions& self)
            { return self.options.usesPrimitiveTypeFlags; },
            [](pyoptix::PipelineCompileOptions& self, uint32_t val)
            { self.options.usesPrimitiveTypeFlags = val; }
        )
#endif
        ;

    py::class_<pyoptix::PipelineLinkOptions>(m, "PipelineLinkOptions")
        .def(
            py::init<
                uint32_t,
                OptixCompileDebugLevel
                >(),
            py::arg( "maxTraceDepth" ) = 0u,
            py::arg( "debugLevel"       ) = 
            IF_OPTIX71_ELSE( 
                OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT, OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO 
                )
            )
        .def_property( "maxTraceDepth",
            [](const pyoptix::PipelineLinkOptions& self)
            { return self.options.maxTraceDepth; },
            [](pyoptix::PipelineLinkOptions& self, uint32_t val)
            { self.options.maxTraceDepth = val; }
            )
        .def_property( "debugLevel",
            [](const pyoptix::PipelineLinkOptions& self)
            { return self.options.debugLevel; },
            [](pyoptix::PipelineLinkOptions& self, OptixCompileDebugLevel val)
            { self.options.debugLevel = val; }
            )
        ;

    py::class_<pyoptix::ShaderBindingTable>(m, "ShaderBindingTable")
        .def(
            py::init<
                CUdeviceptr,
                CUdeviceptr,
                CUdeviceptr,
                uint32_t,
                uint32_t,
                CUdeviceptr,
                uint32_t,
                uint32_t,
                CUdeviceptr,
                uint32_t,
                uint32_t
                >(),
            py::arg( "raygenRecord"            ) = 0,
            py::arg( "exceptionRecord"         ) = 0,
            py::arg( "missRecordBase"          ) = 0,
            py::arg( "missRecordStrideInBytes" ) = 0,
            py::arg( "missRecordCount"         ) = 0,
            py::arg( "hitgroupRecordBase"          ) = 0,
            py::arg( "hitgroupRecordStrideInBytes" ) = 0,
            py::arg( "hitgroupRecordCount"         ) = 0,
            py::arg( "callablesRecordBase"          ) = 0,
            py::arg( "callablesRecordStrideInBytes" ) = 0,
            py::arg( "callablesRecordCount"         ) = 0
            )
        .def_property( "raygenRecord",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.raygenRecord; },
            [](pyoptix::ShaderBindingTable& self, CUdeviceptr val)
            { self.sbt.raygenRecord = val; }
            )
        .def_property( "exceptionRecord",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.exceptionRecord; },
            [](pyoptix::ShaderBindingTable& self, CUdeviceptr val)
            { self.sbt.exceptionRecord = val; }
            )
        .def_property( "missRecordBase",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.missRecordBase; },
            [](pyoptix::ShaderBindingTable& self, CUdeviceptr val)
            { self.sbt.missRecordBase = val; }
            )
        .def_property( "missRecordStrideInBytes",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.missRecordStrideInBytes; },
            [](pyoptix::ShaderBindingTable& self, uint32_t val)
            { self.sbt.missRecordStrideInBytes = val; }
            )
        .def_property( "missRecordCount",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.missRecordCount; },
            [](pyoptix::ShaderBindingTable& self, uint32_t val)
            { self.sbt.missRecordCount = val; }
            )
        .def_property( "hitgroupRecordBase",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.hitgroupRecordBase; },
            [](pyoptix::ShaderBindingTable& self, CUdeviceptr val)
            { self.sbt.hitgroupRecordBase = val; }
            )
        .def_property( "hitgroupRecordStrideInBytes",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.hitgroupRecordStrideInBytes; },
            [](pyoptix::ShaderBindingTable& self, uint32_t val)
            { self.sbt.hitgroupRecordStrideInBytes = val; }
            )
        .def_property( "hitgroupRecordCount",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.hitgroupRecordCount; },
            [](pyoptix::ShaderBindingTable& self, uint32_t val)
            { self.sbt.hitgroupRecordCount = val; }
            )
        .def_property( "callablesRecordBase",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.callablesRecordBase; },
            [](pyoptix::ShaderBindingTable& self, CUdeviceptr val)
            { self.sbt.callablesRecordBase = val; }
            )
        .def_property( "callablesRecordStrideInBytes",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.callablesRecordStrideInBytes; },
            [](pyoptix::ShaderBindingTable& self, uint32_t val)
            { self.sbt.callablesRecordStrideInBytes = val; }
            )
        .def_property( "callablesRecordCount",
            [](const pyoptix::ShaderBindingTable& self)
            { return self.sbt.callablesRecordCount; },
            [](pyoptix::ShaderBindingTable& self, uint32_t val)
            { self.sbt.callablesRecordCount = val; }
            )
        ;

    py::class_<OptixStackSizes>(m, "StackSizes")
        .def( py::init( []()
           { return std::unique_ptr<OptixStackSizes>(new OptixStackSizes{} ); }
        ) )
        .def_readwrite( "cssRG", &OptixStackSizes::cssRG )
        .def_readwrite( "cssMS", &OptixStackSizes::cssMS )
        .def_readwrite( "cssCH", &OptixStackSizes::cssCH )
        .def_readwrite( "cssAH", &OptixStackSizes::cssAH )
        .def_readwrite( "cssIS", &OptixStackSizes::cssIS )
        .def_readwrite( "cssCC", &OptixStackSizes::cssCC )
        .def_readwrite( "dssDC", &OptixStackSizes::dssDC )
        ;

#if OPTIX_VERSION >= 70100
    py::class_<pyoptix::BuiltinISOptions>(m, "BuiltinISOptions")
        .def(
            py::init<OptixPrimitiveType, bool>(),
            py::arg( "builtinISModuleType" ) = OPTIX_PRIMITIVE_TYPE_TRIANGLE,
            py::arg( "usesMotionBlur" )=false
            )
        .def_property( "builtinISModuleType",
            [](const pyoptix::BuiltinISOptions& self)
            { return self.options.builtinISModuleType; },
            [](pyoptix::BuiltinISOptions& self, OptixPrimitiveType val )
            { self.options.builtinISModuleType = val; }
        )
        .def_property( "usesMotionBlur",
            [](const pyoptix::BuiltinISOptions& self)
            { return self.options.usesMotionBlur; },
            [](pyoptix::BuiltinISOptions& self, bool val )
            { self.options.usesMotionBlur = val; }
        )
        ;
#endif // OPTIX_VERSION >= 70100

    
    py::class_<OptixTraversableHandle>(m, "TraversableHandle")
        .def( py::init( []()
           { return std::unique_ptr<OptixTraversableHandle>(new OptixTraversableHandle{} ); }
        ) )
        ;
}
