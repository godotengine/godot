///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// RDAT_LibraryTypes.inl                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines types used in Dxil Library Runtime Data (RDAT).                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifdef DEF_RDAT_ENUMS

RDAT_ENUM_START(DxilResourceFlag, uint32_t)
  RDAT_ENUM_VALUE(None,                     0)
  RDAT_ENUM_VALUE(UAVGloballyCoherent,      1 << 0)
  RDAT_ENUM_VALUE(UAVCounter,               1 << 1)
  RDAT_ENUM_VALUE(UAVRasterizerOrderedView, 1 << 2)
  RDAT_ENUM_VALUE(DynamicIndexing,          1 << 3)
  RDAT_ENUM_VALUE(Atomics64Use,             1 << 4)
RDAT_ENUM_END()

#endif // DEF_RDAT_ENUMS

#ifdef DEF_DXIL_ENUMS

// Enums using RDAT_DXIL_ENUM_START use existing definitions of enums, rather
// than redefining the enum locally.  The definition here is mainly to
// implement the ToString function.
// A static_assert under DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL is used to
// check one enum value that would change if the enum were to be updated,
// making sure this definition is updated as well.

RDAT_DXIL_ENUM_START(hlsl::DXIL::ResourceClass, uint32_t)
  RDAT_ENUM_VALUE_NODEF(SRV)
  RDAT_ENUM_VALUE_NODEF(UAV)
  RDAT_ENUM_VALUE_NODEF(CBuffer)
  RDAT_ENUM_VALUE_NODEF(Sampler)
  RDAT_ENUM_VALUE_NODEF(Invalid)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::ResourceClass::Invalid == 4, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::ResourceKind, uint32_t)
  RDAT_ENUM_VALUE_NODEF(Invalid)
  RDAT_ENUM_VALUE_NODEF(Texture1D)
  RDAT_ENUM_VALUE_NODEF(Texture2D)
  RDAT_ENUM_VALUE_NODEF(Texture2DMS)
  RDAT_ENUM_VALUE_NODEF(Texture3D)
  RDAT_ENUM_VALUE_NODEF(TextureCube)
  RDAT_ENUM_VALUE_NODEF(Texture1DArray)
  RDAT_ENUM_VALUE_NODEF(Texture2DArray)
  RDAT_ENUM_VALUE_NODEF(Texture2DMSArray)
  RDAT_ENUM_VALUE_NODEF(TextureCubeArray)
  RDAT_ENUM_VALUE_NODEF(TypedBuffer)
  RDAT_ENUM_VALUE_NODEF(RawBuffer)
  RDAT_ENUM_VALUE_NODEF(StructuredBuffer)
  RDAT_ENUM_VALUE_NODEF(CBuffer)
  RDAT_ENUM_VALUE_NODEF(Sampler)
  RDAT_ENUM_VALUE_NODEF(TBuffer)
  RDAT_ENUM_VALUE_NODEF(RTAccelerationStructure)
  RDAT_ENUM_VALUE_NODEF(FeedbackTexture2D)
  RDAT_ENUM_VALUE_NODEF(FeedbackTexture2DArray)
  RDAT_ENUM_VALUE_NODEF(NumEntries)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::ResourceKind::NumEntries == 19, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::ShaderKind, uint32_t)
  RDAT_ENUM_VALUE_NODEF(Pixel)
  RDAT_ENUM_VALUE_NODEF(Vertex)
  RDAT_ENUM_VALUE_NODEF(Geometry)
  RDAT_ENUM_VALUE_NODEF(Hull)
  RDAT_ENUM_VALUE_NODEF(Domain)
  RDAT_ENUM_VALUE_NODEF(Compute)
  RDAT_ENUM_VALUE_NODEF(Library)
  RDAT_ENUM_VALUE_NODEF(RayGeneration)
  RDAT_ENUM_VALUE_NODEF(Intersection)
  RDAT_ENUM_VALUE_NODEF(AnyHit)
  RDAT_ENUM_VALUE_NODEF(ClosestHit)
  RDAT_ENUM_VALUE_NODEF(Miss)
  RDAT_ENUM_VALUE_NODEF(Callable)
  RDAT_ENUM_VALUE_NODEF(Mesh)
  RDAT_ENUM_VALUE_NODEF(Amplification)
  RDAT_ENUM_VALUE_NODEF(Invalid)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::ShaderKind::Invalid == 15, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::SemanticKind, uint32_t)
  /* <py::lines('SemanticKind-ENUM')>hctdb_instrhelp.get_rdat_enum_decl("SemanticKind", nodef=True)</py>*/
  // SemanticKind-ENUM:BEGIN
  RDAT_ENUM_VALUE_NODEF(Arbitrary)
  RDAT_ENUM_VALUE_NODEF(VertexID)
  RDAT_ENUM_VALUE_NODEF(InstanceID)
  RDAT_ENUM_VALUE_NODEF(Position)
  RDAT_ENUM_VALUE_NODEF(RenderTargetArrayIndex)
  RDAT_ENUM_VALUE_NODEF(ViewPortArrayIndex)
  RDAT_ENUM_VALUE_NODEF(ClipDistance)
  RDAT_ENUM_VALUE_NODEF(CullDistance)
  RDAT_ENUM_VALUE_NODEF(OutputControlPointID)
  RDAT_ENUM_VALUE_NODEF(DomainLocation)
  RDAT_ENUM_VALUE_NODEF(PrimitiveID)
  RDAT_ENUM_VALUE_NODEF(GSInstanceID)
  RDAT_ENUM_VALUE_NODEF(SampleIndex)
  RDAT_ENUM_VALUE_NODEF(IsFrontFace)
  RDAT_ENUM_VALUE_NODEF(Coverage)
  RDAT_ENUM_VALUE_NODEF(InnerCoverage)
  RDAT_ENUM_VALUE_NODEF(Target)
  RDAT_ENUM_VALUE_NODEF(Depth)
  RDAT_ENUM_VALUE_NODEF(DepthLessEqual)
  RDAT_ENUM_VALUE_NODEF(DepthGreaterEqual)
  RDAT_ENUM_VALUE_NODEF(StencilRef)
  RDAT_ENUM_VALUE_NODEF(DispatchThreadID)
  RDAT_ENUM_VALUE_NODEF(GroupID)
  RDAT_ENUM_VALUE_NODEF(GroupIndex)
  RDAT_ENUM_VALUE_NODEF(GroupThreadID)
  RDAT_ENUM_VALUE_NODEF(TessFactor)
  RDAT_ENUM_VALUE_NODEF(InsideTessFactor)
  RDAT_ENUM_VALUE_NODEF(ViewID)
  RDAT_ENUM_VALUE_NODEF(Barycentrics)
  RDAT_ENUM_VALUE_NODEF(ShadingRate)
  RDAT_ENUM_VALUE_NODEF(CullPrimitive)
  RDAT_ENUM_VALUE_NODEF(Invalid)
  // SemanticKind-ENUM:END
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::ComponentType, uint32_t)
  RDAT_ENUM_VALUE_NODEF(Invalid)
  RDAT_ENUM_VALUE_NODEF(I1)
  RDAT_ENUM_VALUE_NODEF(I16)
  RDAT_ENUM_VALUE_NODEF(U16)
  RDAT_ENUM_VALUE_NODEF(I32)
  RDAT_ENUM_VALUE_NODEF(U32)
  RDAT_ENUM_VALUE_NODEF(I64)
  RDAT_ENUM_VALUE_NODEF(U64)
  RDAT_ENUM_VALUE_NODEF(F16)
  RDAT_ENUM_VALUE_NODEF(F32)
  RDAT_ENUM_VALUE_NODEF(F64)
  RDAT_ENUM_VALUE_NODEF(SNormF16)
  RDAT_ENUM_VALUE_NODEF(UNormF16)
  RDAT_ENUM_VALUE_NODEF(SNormF32)
  RDAT_ENUM_VALUE_NODEF(UNormF32)
  RDAT_ENUM_VALUE_NODEF(SNormF64)
  RDAT_ENUM_VALUE_NODEF(UNormF64)
  RDAT_ENUM_VALUE_NODEF(PackedS8x32)
  RDAT_ENUM_VALUE_NODEF(PackedU8x32)
  RDAT_ENUM_VALUE_NODEF(LastEntry)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::ComponentType::LastEntry == 19, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::InterpolationMode, uint32_t)
  RDAT_ENUM_VALUE_NODEF(Undefined)
  RDAT_ENUM_VALUE_NODEF(Constant)
  RDAT_ENUM_VALUE_NODEF(Linear)
  RDAT_ENUM_VALUE_NODEF(LinearCentroid)
  RDAT_ENUM_VALUE_NODEF(LinearNoperspective)
  RDAT_ENUM_VALUE_NODEF(LinearNoperspectiveCentroid)
  RDAT_ENUM_VALUE_NODEF(LinearSample)
  RDAT_ENUM_VALUE_NODEF(LinearNoperspectiveSample)
  RDAT_ENUM_VALUE_NODEF(Invalid)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::InterpolationMode::Invalid == 8, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

#endif // DEF_DXIL_ENUMS

#ifdef DEF_RDAT_TYPES

#define RECORD_TYPE RuntimeDataResourceInfo
RDAT_STRUCT_TABLE(RuntimeDataResourceInfo, ResourceTable)
  RDAT_ENUM(uint32_t, hlsl::DXIL::ResourceClass, Class)
  RDAT_ENUM(uint32_t, hlsl::DXIL::ResourceKind, Kind)
  RDAT_VALUE(uint32_t, ID)
  RDAT_VALUE(uint32_t, Space)
  RDAT_VALUE(uint32_t, LowerBound)
  RDAT_VALUE(uint32_t, UpperBound)
  RDAT_STRING(Name)
  RDAT_FLAGS(uint32_t, DxilResourceFlag, Flags)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE RuntimeDataFunctionInfo
RDAT_STRUCT_TABLE(RuntimeDataFunctionInfo, FunctionTable)
  // full function name
  RDAT_STRING(Name)
  // unmangled function name
  RDAT_STRING(UnmangledName)
  // list of global resources used by this function
  RDAT_RECORD_ARRAY_REF(RuntimeDataResourceInfo, Resources)
  // list of external function names this function calls
  RDAT_STRING_ARRAY_REF(FunctionDependencies)
  // Shader type, or library function
  RDAT_ENUM(uint32_t, hlsl::DXIL::ShaderKind, ShaderKind)
  // Payload Size:
  // 1) any/closest hit or miss shader: payload size
  // 2) call shader: parameter size 
  RDAT_VALUE(uint32_t, PayloadSizeInBytes)
  // attribute size for closest hit and any hit
  RDAT_VALUE(uint32_t, AttributeSizeInBytes)
  // first 32 bits of feature flag
  RDAT_VALUE(uint32_t, FeatureInfo1)
  // second 32 bits of feature flag
  RDAT_VALUE(uint32_t, FeatureInfo2)
  // valid shader stage flag.
  RDAT_VALUE(uint32_t, ShaderStageFlag)
  // minimum shader target.
  RDAT_VALUE(uint32_t, MinShaderTarget)

#if DEF_RDAT_TYPES == DEF_RDAT_TYPES_USE_HELPERS
  // void SetFeatureFlags(uint64_t flags) convenience method
  void SetFeatureFlags(uint64_t flags) {
    FeatureInfo1 = flags & 0xffffffff;
    FeatureInfo2 = (flags >> 32) & 0xffffffff;
  }
#endif

#if DEF_RDAT_TYPES == DEF_RDAT_READER_DECL
  // uint64_t GetFeatureFlags() convenience method
  uint64_t GetFeatureFlags();
#elif DEF_RDAT_TYPES == DEF_RDAT_READER_IMPL
  // uint64_t GetFeatureFlags() convenience method
  uint64_t RuntimeDataFunctionInfo_Reader::GetFeatureFlags() {
    return asRecord() ? (((uint64_t)asRecord()->FeatureInfo2 << 32) |
                         (uint64_t)asRecord()->FeatureInfo1)
                      : 0;
  }
#endif

RDAT_STRUCT_END()
#undef RECORD_TYPE

#endif // DEF_RDAT_TYPES
