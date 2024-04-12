//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) Microsoft Corporation.
//  Licensed under the MIT license.
//
//  File:       D3D12Shader.h
//  Content:    D3D12 Shader Types and APIs
//
//////////////////////////////////////////////////////////////////////////////

#ifndef __D3D12SHADER_H__
#define __D3D12SHADER_H__

#include "d3dcommon.h"

typedef enum D3D12_SHADER_VERSION_TYPE
{
    D3D12_SHVER_PIXEL_SHADER          = 0,
    D3D12_SHVER_VERTEX_SHADER         = 1,
    D3D12_SHVER_GEOMETRY_SHADER       = 2,
    
    // D3D11 Shaders
    D3D12_SHVER_HULL_SHADER           = 3,
    D3D12_SHVER_DOMAIN_SHADER         = 4,
    D3D12_SHVER_COMPUTE_SHADER        = 5,

    // D3D12 Shaders
    D3D12_SHVER_LIBRARY               = 6,

    D3D12_SHVER_RAY_GENERATION_SHADER = 7,
    D3D12_SHVER_INTERSECTION_SHADER   = 8,
    D3D12_SHVER_ANY_HIT_SHADER        = 9,
    D3D12_SHVER_CLOSEST_HIT_SHADER    = 10,
    D3D12_SHVER_MISS_SHADER           = 11,
    D3D12_SHVER_CALLABLE_SHADER       = 12,

    D3D12_SHVER_MESH_SHADER           = 13,
    D3D12_SHVER_AMPLIFICATION_SHADER  = 14,

    D3D12_SHVER_RESERVED0             = 0xFFF0,
} D3D12_SHADER_VERSION_TYPE;

#define D3D12_SHVER_GET_TYPE(_Version) \
    (((_Version) >> 16) & 0xffff)
#define D3D12_SHVER_GET_MAJOR(_Version) \
    (((_Version) >> 4) & 0xf)
#define D3D12_SHVER_GET_MINOR(_Version) \
    (((_Version) >> 0) & 0xf)

// Slot ID for library function return
#define D3D_RETURN_PARAMETER_INDEX (-1)

typedef D3D_RESOURCE_RETURN_TYPE D3D12_RESOURCE_RETURN_TYPE;

typedef D3D_CBUFFER_TYPE D3D12_CBUFFER_TYPE;


typedef struct _D3D12_SIGNATURE_PARAMETER_DESC
{
    LPCSTR                      SemanticName;   // Name of the semantic
    UINT                        SemanticIndex;  // Index of the semantic
    UINT                        Register;       // Number of member variables
    D3D_NAME                    SystemValueType;// A predefined system value, or D3D_NAME_UNDEFINED if not applicable
    D3D_REGISTER_COMPONENT_TYPE ComponentType;  // Scalar type (e.g. uint, float, etc.)
    BYTE                        Mask;           // Mask to indicate which components of the register
                                                // are used (combination of D3D10_COMPONENT_MASK values)
    BYTE                        ReadWriteMask;  // Mask to indicate whether a given component is 
                                                // never written (if this is an output signature) or
                                                // always read (if this is an input signature).
                                                // (combination of D3D_MASK_* values)
    UINT                        Stream;         // Stream index
    D3D_MIN_PRECISION           MinPrecision;   // Minimum desired interpolation precision
} D3D12_SIGNATURE_PARAMETER_DESC;

typedef struct _D3D12_SHADER_BUFFER_DESC
{
    LPCSTR                  Name;           // Name of the constant buffer
    D3D_CBUFFER_TYPE        Type;           // Indicates type of buffer content
    UINT                    Variables;      // Number of member variables
    UINT                    Size;           // Size of CB (in bytes)
    UINT                    uFlags;         // Buffer description flags
} D3D12_SHADER_BUFFER_DESC;

typedef struct _D3D12_SHADER_VARIABLE_DESC
{
    LPCSTR                  Name;           // Name of the variable
    UINT                    StartOffset;    // Offset in constant buffer's backing store
    UINT                    Size;           // Size of variable (in bytes)
    UINT                    uFlags;         // Variable flags
    LPVOID                  DefaultValue;   // Raw pointer to default value
    UINT                    StartTexture;   // First texture index (or -1 if no textures used)
    UINT                    TextureSize;    // Number of texture slots possibly used.
    UINT                    StartSampler;   // First sampler index (or -1 if no textures used)
    UINT                    SamplerSize;    // Number of sampler slots possibly used.
} D3D12_SHADER_VARIABLE_DESC;

typedef struct _D3D12_SHADER_TYPE_DESC
{
    D3D_SHADER_VARIABLE_CLASS   Class;          // Variable class (e.g. object, matrix, etc.)
    D3D_SHADER_VARIABLE_TYPE    Type;           // Variable type (e.g. float, sampler, etc.)
    UINT                        Rows;           // Number of rows (for matrices, 1 for other numeric, 0 if not applicable)
    UINT                        Columns;        // Number of columns (for vectors & matrices, 1 for other numeric, 0 if not applicable)
    UINT                        Elements;       // Number of elements (0 if not an array)
    UINT                        Members;        // Number of members (0 if not a structure)
    UINT                        Offset;         // Offset from the start of structure (0 if not a structure member)
    LPCSTR                      Name;           // Name of type, can be NULL
} D3D12_SHADER_TYPE_DESC;

typedef D3D_TESSELLATOR_DOMAIN D3D12_TESSELLATOR_DOMAIN;

typedef D3D_TESSELLATOR_PARTITIONING D3D12_TESSELLATOR_PARTITIONING;

typedef D3D_TESSELLATOR_OUTPUT_PRIMITIVE D3D12_TESSELLATOR_OUTPUT_PRIMITIVE;

typedef struct _D3D12_SHADER_DESC
{
    UINT                    Version;                     // Shader version
    LPCSTR                  Creator;                     // Creator string
    UINT                    Flags;                       // Shader compilation/parse flags
    
    UINT                    ConstantBuffers;             // Number of constant buffers
    UINT                    BoundResources;              // Number of bound resources
    UINT                    InputParameters;             // Number of parameters in the input signature
    UINT                    OutputParameters;            // Number of parameters in the output signature

    UINT                    InstructionCount;            // Number of emitted instructions
    UINT                    TempRegisterCount;           // Number of temporary registers used 
    UINT                    TempArrayCount;              // Number of temporary arrays used
    UINT                    DefCount;                    // Number of constant defines 
    UINT                    DclCount;                    // Number of declarations (input + output)
    UINT                    TextureNormalInstructions;   // Number of non-categorized texture instructions
    UINT                    TextureLoadInstructions;     // Number of texture load instructions
    UINT                    TextureCompInstructions;     // Number of texture comparison instructions
    UINT                    TextureBiasInstructions;     // Number of texture bias instructions
    UINT                    TextureGradientInstructions; // Number of texture gradient instructions
    UINT                    FloatInstructionCount;       // Number of floating point arithmetic instructions used
    UINT                    IntInstructionCount;         // Number of signed integer arithmetic instructions used
    UINT                    UintInstructionCount;        // Number of unsigned integer arithmetic instructions used
    UINT                    StaticFlowControlCount;      // Number of static flow control instructions used
    UINT                    DynamicFlowControlCount;     // Number of dynamic flow control instructions used
    UINT                    MacroInstructionCount;       // Number of macro instructions used
    UINT                    ArrayInstructionCount;       // Number of array instructions used
    UINT                    CutInstructionCount;         // Number of cut instructions used
    UINT                    EmitInstructionCount;        // Number of emit instructions used
    D3D_PRIMITIVE_TOPOLOGY  GSOutputTopology;            // Geometry shader output topology
    UINT                    GSMaxOutputVertexCount;      // Geometry shader maximum output vertex count
    D3D_PRIMITIVE           InputPrimitive;              // GS/HS input primitive
    UINT                    PatchConstantParameters;     // Number of parameters in the patch constant signature
    UINT                    cGSInstanceCount;            // Number of Geometry shader instances
    UINT                    cControlPoints;              // Number of control points in the HS->DS stage
    D3D_TESSELLATOR_OUTPUT_PRIMITIVE HSOutputPrimitive;  // Primitive output by the tessellator
    D3D_TESSELLATOR_PARTITIONING HSPartitioning;         // Partitioning mode of the tessellator
    D3D_TESSELLATOR_DOMAIN  TessellatorDomain;           // Domain of the tessellator (quad, tri, isoline)
    // instruction counts
    UINT cBarrierInstructions;                           // Number of barrier instructions in a compute shader
    UINT cInterlockedInstructions;                       // Number of interlocked instructions
    UINT cTextureStoreInstructions;                      // Number of texture writes
} D3D12_SHADER_DESC;

typedef struct _D3D12_SHADER_INPUT_BIND_DESC
{
    LPCSTR                      Name;           // Name of the resource
    D3D_SHADER_INPUT_TYPE       Type;           // Type of resource (e.g. texture, cbuffer, etc.)
    UINT                        BindPoint;      // Starting bind point
    UINT                        BindCount;      // Number of contiguous bind points (for arrays)

    UINT                        uFlags;         // Input binding flags
    D3D_RESOURCE_RETURN_TYPE    ReturnType;     // Return type (if texture)
    D3D_SRV_DIMENSION           Dimension;      // Dimension (if texture)
    UINT                        NumSamples;     // Number of samples (0 if not MS texture)
    UINT                        Space;          // Register space
    UINT uID;                                   // Range ID in the bytecode
} D3D12_SHADER_INPUT_BIND_DESC;

#define D3D_SHADER_REQUIRES_DOUBLES                                                         0x00000001
#define D3D_SHADER_REQUIRES_EARLY_DEPTH_STENCIL                                             0x00000002
#define D3D_SHADER_REQUIRES_UAVS_AT_EVERY_STAGE                                             0x00000004
#define D3D_SHADER_REQUIRES_64_UAVS                                                         0x00000008
#define D3D_SHADER_REQUIRES_MINIMUM_PRECISION                                               0x00000010
#define D3D_SHADER_REQUIRES_11_1_DOUBLE_EXTENSIONS                                          0x00000020
#define D3D_SHADER_REQUIRES_11_1_SHADER_EXTENSIONS                                          0x00000040
#define D3D_SHADER_REQUIRES_LEVEL_9_COMPARISON_FILTERING                                    0x00000080
#define D3D_SHADER_REQUIRES_TILED_RESOURCES                                                 0x00000100
#define D3D_SHADER_REQUIRES_STENCIL_REF                                                     0x00000200
#define D3D_SHADER_REQUIRES_INNER_COVERAGE                                                  0x00000400
#define D3D_SHADER_REQUIRES_TYPED_UAV_LOAD_ADDITIONAL_FORMATS                               0x00000800
#define D3D_SHADER_REQUIRES_ROVS                                                            0x00001000
#define D3D_SHADER_REQUIRES_VIEWPORT_AND_RT_ARRAY_INDEX_FROM_ANY_SHADER_FEEDING_RASTERIZER  0x00002000
#define D3D_SHADER_REQUIRES_WAVE_OPS                                                        0x00004000
#define D3D_SHADER_REQUIRES_INT64_OPS                                                       0x00008000
#define D3D_SHADER_REQUIRES_VIEW_ID                                                         0x00010000
#define D3D_SHADER_REQUIRES_BARYCENTRICS                                                    0x00020000
#define D3D_SHADER_REQUIRES_NATIVE_16BIT_OPS                                                0x00040000
#define D3D_SHADER_REQUIRES_SHADING_RATE                                                    0x00080000
#define D3D_SHADER_REQUIRES_RAYTRACING_TIER_1_1                                             0x00100000
#define D3D_SHADER_REQUIRES_SAMPLER_FEEDBACK                                                0x00200000
#define D3D_SHADER_REQUIRES_ATOMIC_INT64_ON_TYPED_RESOURCE                                  0x00400000
#define D3D_SHADER_REQUIRES_ATOMIC_INT64_ON_GROUP_SHARED                                    0x00800000
#define D3D_SHADER_REQUIRES_DERIVATIVES_IN_MESH_AND_AMPLIFICATION_SHADERS                   0x01000000
#define D3D_SHADER_REQUIRES_RESOURCE_DESCRIPTOR_HEAP_INDEXING                               0x02000000
#define D3D_SHADER_REQUIRES_SAMPLER_DESCRIPTOR_HEAP_INDEXING                                0x04000000
#define D3D_SHADER_REQUIRES_WAVE_MMA                                                        0x08000000
#define D3D_SHADER_REQUIRES_ATOMIC_INT64_ON_DESCRIPTOR_HEAP_RESOURCE                        0x10000000
#define D3D_SHADER_FEATURE_ADVANCED_TEXTURE_OPS                                             0x20000000
#define D3D_SHADER_FEATURE_WRITEABLE_MSAA_TEXTURES                                          0x40000000


typedef struct _D3D12_LIBRARY_DESC
{
    LPCSTR    Creator;           // The name of the originator of the library.
    UINT      Flags;             // Compilation flags.
    UINT      FunctionCount;     // Number of functions exported from the library.
} D3D12_LIBRARY_DESC;

typedef struct _D3D12_FUNCTION_DESC
{
    UINT                    Version;                     // Shader version
    LPCSTR                  Creator;                     // Creator string
    UINT                    Flags;                       // Shader compilation/parse flags
    
    UINT                    ConstantBuffers;             // Number of constant buffers
    UINT                    BoundResources;              // Number of bound resources

    UINT                    InstructionCount;            // Number of emitted instructions
    UINT                    TempRegisterCount;           // Number of temporary registers used 
    UINT                    TempArrayCount;              // Number of temporary arrays used
    UINT                    DefCount;                    // Number of constant defines 
    UINT                    DclCount;                    // Number of declarations (input + output)
    UINT                    TextureNormalInstructions;   // Number of non-categorized texture instructions
    UINT                    TextureLoadInstructions;     // Number of texture load instructions
    UINT                    TextureCompInstructions;     // Number of texture comparison instructions
    UINT                    TextureBiasInstructions;     // Number of texture bias instructions
    UINT                    TextureGradientInstructions; // Number of texture gradient instructions
    UINT                    FloatInstructionCount;       // Number of floating point arithmetic instructions used
    UINT                    IntInstructionCount;         // Number of signed integer arithmetic instructions used
    UINT                    UintInstructionCount;        // Number of unsigned integer arithmetic instructions used
    UINT                    StaticFlowControlCount;      // Number of static flow control instructions used
    UINT                    DynamicFlowControlCount;     // Number of dynamic flow control instructions used
    UINT                    MacroInstructionCount;       // Number of macro instructions used
    UINT                    ArrayInstructionCount;       // Number of array instructions used
    UINT                    MovInstructionCount;         // Number of mov instructions used
    UINT                    MovcInstructionCount;        // Number of movc instructions used
    UINT                    ConversionInstructionCount;  // Number of type conversion instructions used
    UINT                    BitwiseInstructionCount;     // Number of bitwise arithmetic instructions used
    D3D_FEATURE_LEVEL       MinFeatureLevel;             // Min target of the function byte code
    UINT64                  RequiredFeatureFlags;        // Required feature flags

    LPCSTR                  Name;                        // Function name
    INT                     FunctionParameterCount;      // Number of logical parameters in the function signature (not including return)
    BOOL                    HasReturn;                   // TRUE, if function returns a value, false - it is a subroutine
    BOOL                    Has10Level9VertexShader;     // TRUE, if there is a 10L9 VS blob
    BOOL                    Has10Level9PixelShader;      // TRUE, if there is a 10L9 PS blob
} D3D12_FUNCTION_DESC;

typedef struct _D3D12_PARAMETER_DESC
{
    LPCSTR                      Name;               // Parameter name.
    LPCSTR                      SemanticName;       // Parameter semantic name (+index).
    D3D_SHADER_VARIABLE_TYPE    Type;               // Element type.
    D3D_SHADER_VARIABLE_CLASS   Class;              // Scalar/Vector/Matrix.
    UINT                        Rows;               // Rows are for matrix parameters.
    UINT                        Columns;            // Components or Columns in matrix.
    D3D_INTERPOLATION_MODE      InterpolationMode;  // Interpolation mode.
    D3D_PARAMETER_FLAGS         Flags;              // Parameter modifiers.

    UINT                        FirstInRegister;    // The first input register for this parameter.
    UINT                        FirstInComponent;   // The first input register component for this parameter.
    UINT                        FirstOutRegister;   // The first output register for this parameter.
    UINT                        FirstOutComponent;  // The first output register component for this parameter.
} D3D12_PARAMETER_DESC;


//////////////////////////////////////////////////////////////////////////////
// Interfaces ////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3D12ShaderReflectionType ID3D12ShaderReflectionType;
typedef interface ID3D12ShaderReflectionType *LPD3D12SHADERREFLECTIONTYPE;

typedef interface ID3D12ShaderReflectionVariable ID3D12ShaderReflectionVariable;
typedef interface ID3D12ShaderReflectionVariable *LPD3D12SHADERREFLECTIONVARIABLE;

typedef interface ID3D12ShaderReflectionConstantBuffer ID3D12ShaderReflectionConstantBuffer;
typedef interface ID3D12ShaderReflectionConstantBuffer *LPD3D12SHADERREFLECTIONCONSTANTBUFFER;

typedef interface ID3D12ShaderReflection ID3D12ShaderReflection;
typedef interface ID3D12ShaderReflection *LPD3D12SHADERREFLECTION;

typedef interface ID3D12LibraryReflection ID3D12LibraryReflection;
typedef interface ID3D12LibraryReflection *LPD3D12LIBRARYREFLECTION;

typedef interface ID3D12FunctionReflection ID3D12FunctionReflection;
typedef interface ID3D12FunctionReflection *LPD3D12FUNCTIONREFLECTION;

typedef interface ID3D12FunctionParameterReflection ID3D12FunctionParameterReflection;
typedef interface ID3D12FunctionParameterReflection *LPD3D12FUNCTIONPARAMETERREFLECTION;


// {E913C351-783D-48CA-A1D1-4F306284AD56}
interface DECLSPEC_UUID("E913C351-783D-48CA-A1D1-4F306284AD56") ID3D12ShaderReflectionType;
DEFINE_GUID(IID_ID3D12ShaderReflectionType, 
0xe913c351, 0x783d, 0x48ca, 0xa1, 0xd1, 0x4f, 0x30, 0x62, 0x84, 0xad, 0x56);

#undef INTERFACE
#define INTERFACE ID3D12ShaderReflectionType

DECLARE_INTERFACE(ID3D12ShaderReflectionType)
{
    STDMETHOD(GetDesc)(THIS_ _Out_ D3D12_SHADER_TYPE_DESC *pDesc) PURE;
    
    STDMETHOD_(ID3D12ShaderReflectionType*, GetMemberTypeByIndex)(THIS_ _In_ UINT Index) PURE;
    STDMETHOD_(ID3D12ShaderReflectionType*, GetMemberTypeByName)(THIS_ _In_ LPCSTR Name) PURE;
    STDMETHOD_(LPCSTR, GetMemberTypeName)(THIS_ _In_ UINT Index) PURE;

    STDMETHOD(IsEqual)(THIS_ _In_ ID3D12ShaderReflectionType* pType) PURE;
    STDMETHOD_(ID3D12ShaderReflectionType*, GetSubType)(THIS) PURE;
    STDMETHOD_(ID3D12ShaderReflectionType*, GetBaseClass)(THIS) PURE;
    STDMETHOD_(UINT, GetNumInterfaces)(THIS) PURE;
    STDMETHOD_(ID3D12ShaderReflectionType*, GetInterfaceByIndex)(THIS_ _In_ UINT uIndex) PURE;
    STDMETHOD(IsOfType)(THIS_ _In_ ID3D12ShaderReflectionType* pType) PURE;
    STDMETHOD(ImplementsInterface)(THIS_ _In_ ID3D12ShaderReflectionType* pBase) PURE;
};

// {8337A8A6-A216-444A-B2F4-314733A73AEA}
interface DECLSPEC_UUID("8337A8A6-A216-444A-B2F4-314733A73AEA") ID3D12ShaderReflectionVariable;
DEFINE_GUID(IID_ID3D12ShaderReflectionVariable, 
0x8337a8a6, 0xa216, 0x444a, 0xb2, 0xf4, 0x31, 0x47, 0x33, 0xa7, 0x3a, 0xea);

#undef INTERFACE
#define INTERFACE ID3D12ShaderReflectionVariable

DECLARE_INTERFACE(ID3D12ShaderReflectionVariable)
{
    STDMETHOD(GetDesc)(THIS_ _Out_ D3D12_SHADER_VARIABLE_DESC *pDesc) PURE;
    
    STDMETHOD_(ID3D12ShaderReflectionType*, GetType)(THIS) PURE;
    STDMETHOD_(ID3D12ShaderReflectionConstantBuffer*, GetBuffer)(THIS) PURE;

    STDMETHOD_(UINT, GetInterfaceSlot)(THIS_ _In_ UINT uArrayIndex) PURE;
};

// {C59598B4-48B3-4869-B9B1-B1618B14A8B7}
interface DECLSPEC_UUID("C59598B4-48B3-4869-B9B1-B1618B14A8B7") ID3D12ShaderReflectionConstantBuffer;
DEFINE_GUID(IID_ID3D12ShaderReflectionConstantBuffer, 
0xc59598b4, 0x48b3, 0x4869, 0xb9, 0xb1, 0xb1, 0x61, 0x8b, 0x14, 0xa8, 0xb7);

#undef INTERFACE
#define INTERFACE ID3D12ShaderReflectionConstantBuffer

DECLARE_INTERFACE(ID3D12ShaderReflectionConstantBuffer)
{
    STDMETHOD(GetDesc)(THIS_ D3D12_SHADER_BUFFER_DESC *pDesc) PURE;
    
    STDMETHOD_(ID3D12ShaderReflectionVariable*, GetVariableByIndex)(THIS_ _In_ UINT Index) PURE;
    STDMETHOD_(ID3D12ShaderReflectionVariable*, GetVariableByName)(THIS_ _In_ LPCSTR Name) PURE;
};

// The ID3D12ShaderReflection IID may change from SDK version to SDK version
// if the reflection API changes.  This prevents new code with the new API
// from working with an old binary.  Recompiling with the new header
// will pick up the new IID.

// {5A58797D-A72C-478D-8BA2-EFC6B0EFE88E}
interface DECLSPEC_UUID("5A58797D-A72C-478D-8BA2-EFC6B0EFE88E") ID3D12ShaderReflection;
DEFINE_GUID(IID_ID3D12ShaderReflection, 
0x5a58797d, 0xa72c, 0x478d, 0x8b, 0xa2, 0xef, 0xc6, 0xb0, 0xef, 0xe8, 0x8e);

#undef INTERFACE
#define INTERFACE ID3D12ShaderReflection

DECLARE_INTERFACE_(ID3D12ShaderReflection, IUnknown)
{
    STDMETHOD(QueryInterface)(THIS_ _In_ REFIID iid,
                              _Out_ LPVOID *ppv) PURE;
    STDMETHOD_(ULONG, AddRef)(THIS) PURE;
    STDMETHOD_(ULONG, Release)(THIS) PURE;

    STDMETHOD(GetDesc)(THIS_ _Out_ D3D12_SHADER_DESC *pDesc) PURE;
    
    STDMETHOD_(ID3D12ShaderReflectionConstantBuffer*, GetConstantBufferByIndex)(THIS_ _In_ UINT Index) PURE;
    STDMETHOD_(ID3D12ShaderReflectionConstantBuffer*, GetConstantBufferByName)(THIS_ _In_ LPCSTR Name) PURE;
    
    STDMETHOD(GetResourceBindingDesc)(THIS_ _In_ UINT ResourceIndex,
                                      _Out_ D3D12_SHADER_INPUT_BIND_DESC *pDesc) PURE;
    
    STDMETHOD(GetInputParameterDesc)(THIS_ _In_ UINT ParameterIndex,
                                     _Out_ D3D12_SIGNATURE_PARAMETER_DESC *pDesc) PURE;
    STDMETHOD(GetOutputParameterDesc)(THIS_ _In_ UINT ParameterIndex,
                                      _Out_ D3D12_SIGNATURE_PARAMETER_DESC *pDesc) PURE;
    STDMETHOD(GetPatchConstantParameterDesc)(THIS_ _In_ UINT ParameterIndex,
                                             _Out_ D3D12_SIGNATURE_PARAMETER_DESC *pDesc) PURE;

    STDMETHOD_(ID3D12ShaderReflectionVariable*, GetVariableByName)(THIS_ _In_ LPCSTR Name) PURE;

    STDMETHOD(GetResourceBindingDescByName)(THIS_ _In_ LPCSTR Name,
                                            _Out_ D3D12_SHADER_INPUT_BIND_DESC *pDesc) PURE;

    STDMETHOD_(UINT, GetMovInstructionCount)(THIS) PURE;
    STDMETHOD_(UINT, GetMovcInstructionCount)(THIS) PURE;
    STDMETHOD_(UINT, GetConversionInstructionCount)(THIS) PURE;
    STDMETHOD_(UINT, GetBitwiseInstructionCount)(THIS) PURE;
    
    STDMETHOD_(D3D_PRIMITIVE, GetGSInputPrimitive)(THIS) PURE;
    STDMETHOD_(BOOL, IsSampleFrequencyShader)(THIS) PURE;

    STDMETHOD_(UINT, GetNumInterfaceSlots)(THIS) PURE;
    STDMETHOD(GetMinFeatureLevel)(THIS_ _Out_ enum D3D_FEATURE_LEVEL* pLevel) PURE;

    STDMETHOD_(UINT, GetThreadGroupSize)(THIS_
                                         _Out_opt_ UINT* pSizeX,
                                         _Out_opt_ UINT* pSizeY,
                                         _Out_opt_ UINT* pSizeZ) PURE;

    STDMETHOD_(UINT64, GetRequiresFlags)(THIS) PURE;
};

// {8E349D19-54DB-4A56-9DC9-119D87BDB804}
interface DECLSPEC_UUID("8E349D19-54DB-4A56-9DC9-119D87BDB804") ID3D12LibraryReflection;
DEFINE_GUID(IID_ID3D12LibraryReflection, 
0x8e349d19, 0x54db, 0x4a56, 0x9d, 0xc9, 0x11, 0x9d, 0x87, 0xbd, 0xb8, 0x4);

#undef INTERFACE
#define INTERFACE ID3D12LibraryReflection

DECLARE_INTERFACE_(ID3D12LibraryReflection, IUnknown)
{
    STDMETHOD(QueryInterface)(THIS_ _In_ REFIID iid, _Out_ LPVOID * ppv) PURE;
    STDMETHOD_(ULONG, AddRef)(THIS) PURE;
    STDMETHOD_(ULONG, Release)(THIS) PURE;

    STDMETHOD(GetDesc)(THIS_ _Out_ D3D12_LIBRARY_DESC * pDesc) PURE;
    
    STDMETHOD_(ID3D12FunctionReflection *, GetFunctionByIndex)(THIS_ _In_ INT FunctionIndex) PURE;
};

// {1108795C-2772-4BA9-B2A8-D464DC7E2799}
interface DECLSPEC_UUID("1108795C-2772-4BA9-B2A8-D464DC7E2799") ID3D12FunctionReflection;
DEFINE_GUID(IID_ID3D12FunctionReflection, 
0x1108795c, 0x2772, 0x4ba9, 0xb2, 0xa8, 0xd4, 0x64, 0xdc, 0x7e, 0x27, 0x99);

#undef INTERFACE
#define INTERFACE ID3D12FunctionReflection

DECLARE_INTERFACE(ID3D12FunctionReflection)
{
    STDMETHOD(GetDesc)(THIS_ _Out_ D3D12_FUNCTION_DESC * pDesc) PURE;
    
    STDMETHOD_(ID3D12ShaderReflectionConstantBuffer *, GetConstantBufferByIndex)(THIS_ _In_ UINT BufferIndex) PURE;
    STDMETHOD_(ID3D12ShaderReflectionConstantBuffer *, GetConstantBufferByName)(THIS_ _In_ LPCSTR Name) PURE;
    
    STDMETHOD(GetResourceBindingDesc)(THIS_ _In_ UINT ResourceIndex,
                                      _Out_ D3D12_SHADER_INPUT_BIND_DESC * pDesc) PURE;
    
    STDMETHOD_(ID3D12ShaderReflectionVariable *, GetVariableByName)(THIS_ _In_ LPCSTR Name) PURE;

    STDMETHOD(GetResourceBindingDescByName)(THIS_ _In_ LPCSTR Name,
                                            _Out_ D3D12_SHADER_INPUT_BIND_DESC * pDesc) PURE;

    // Use D3D_RETURN_PARAMETER_INDEX to get description of the return value.
    STDMETHOD_(ID3D12FunctionParameterReflection *, GetFunctionParameter)(THIS_ _In_ INT ParameterIndex) PURE;
};

// {EC25F42D-7006-4F2B-B33E-02CC3375733F}
interface DECLSPEC_UUID("EC25F42D-7006-4F2B-B33E-02CC3375733F") ID3D12FunctionParameterReflection;
DEFINE_GUID(IID_ID3D12FunctionParameterReflection, 
0xec25f42d, 0x7006, 0x4f2b, 0xb3, 0x3e, 0x2, 0xcc, 0x33, 0x75, 0x73, 0x3f);

#undef INTERFACE
#define INTERFACE ID3D12FunctionParameterReflection

DECLARE_INTERFACE(ID3D12FunctionParameterReflection)
{
    STDMETHOD(GetDesc)(THIS_ _Out_ D3D12_PARAMETER_DESC * pDesc) PURE;
};


//////////////////////////////////////////////////////////////////////////////
// APIs //////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#ifdef __cplusplus
}
#endif //__cplusplus
    
#endif //__D3D12SHADER_H__

