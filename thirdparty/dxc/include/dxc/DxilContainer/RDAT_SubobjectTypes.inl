///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// RDAT_SubobjectTypes.inl                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Defines types used in Dxil Library Runtime Data (RDAT).                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifdef DEF_RDAT_ENUMS

// Nothing yet

#endif // DEF_RDAT_ENUMS

#ifdef DEF_DXIL_ENUMS

RDAT_DXIL_ENUM_START(hlsl::DXIL::SubobjectKind, uint32_t)
  RDAT_ENUM_VALUE_NODEF(StateObjectConfig)
  RDAT_ENUM_VALUE_NODEF(GlobalRootSignature)
  RDAT_ENUM_VALUE_NODEF(LocalRootSignature)
  RDAT_ENUM_VALUE_NODEF(SubobjectToExportsAssociation)
  RDAT_ENUM_VALUE_NODEF(RaytracingShaderConfig)
  RDAT_ENUM_VALUE_NODEF(RaytracingPipelineConfig)
  RDAT_ENUM_VALUE_NODEF(HitGroup)
  RDAT_ENUM_VALUE_NODEF(RaytracingPipelineConfig1)
  // No need to define this here
  //RDAT_ENUM_VALUE_NODEF(NumKinds)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::SubobjectKind::NumKinds == 13, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::StateObjectFlags, uint32_t)
  RDAT_ENUM_VALUE_NODEF(AllowLocalDependenciesOnExternalDefinitions)
  RDAT_ENUM_VALUE_NODEF(AllowExternalDependenciesOnLocalDefinitions)
  RDAT_ENUM_VALUE_NODEF(AllowStateObjectAdditions)
  // No need to define these masks here
  //RDAT_ENUM_VALUE_NODEF(ValidMask_1_4)
  //RDAT_ENUM_VALUE_NODEF(ValidMask)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::StateObjectFlags::ValidMask == 0x7, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::HitGroupType, uint32_t)
  RDAT_ENUM_VALUE_NODEF(Triangle)
  RDAT_ENUM_VALUE_NODEF(ProceduralPrimitive)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::HitGroupType::LastEntry == 2, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

RDAT_DXIL_ENUM_START(hlsl::DXIL::RaytracingPipelineFlags, uint32_t)
  RDAT_ENUM_VALUE_NODEF(None)
  RDAT_ENUM_VALUE_NODEF(SkipTriangles)
  RDAT_ENUM_VALUE_NODEF(SkipProceduralPrimitives)
  // No need to define mask here
  //RDAT_ENUM_VALUE_NODEF(ValidMask)
#if DEF_RDAT_ENUMS == DEF_RDAT_DUMP_IMPL
  static_assert((unsigned)hlsl::DXIL::RaytracingPipelineFlags::ValidMask == 0x300, "otherwise, RDAT_DXIL_ENUM definition needs updating");
#endif
RDAT_ENUM_END()

#endif // DEF_DXIL_ENUMS

#ifdef DEF_RDAT_TYPES

#define RECORD_TYPE StateObjectConfig_t
RDAT_STRUCT(StateObjectConfig_t)
  RDAT_FLAGS(uint32_t, hlsl::DXIL::StateObjectFlags, Flags)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE RootSignature_t
RDAT_STRUCT(RootSignature_t)
  RDAT_BYTES(Data)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE SubobjectToExportsAssociation_t
RDAT_STRUCT(SubobjectToExportsAssociation_t)
  RDAT_STRING(Subobject)
  RDAT_STRING_ARRAY_REF(Exports)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE RaytracingShaderConfig_t
RDAT_STRUCT(RaytracingShaderConfig_t)
  RDAT_VALUE(uint32_t, MaxPayloadSizeInBytes)
  RDAT_VALUE(uint32_t, MaxAttributeSizeInBytes)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE RaytracingPipelineConfig_t
RDAT_STRUCT(RaytracingPipelineConfig_t)
  RDAT_VALUE(uint32_t, MaxTraceRecursionDepth)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE HitGroup_t
RDAT_STRUCT(HitGroup_t)
  RDAT_ENUM(uint32_t, hlsl::DXIL::HitGroupType, Type)
  RDAT_STRING(AnyHit)
  RDAT_STRING(ClosestHit)
  RDAT_STRING(Intersection)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE RaytracingPipelineConfig1_t
RDAT_STRUCT(RaytracingPipelineConfig1_t)
  RDAT_VALUE(uint32_t, MaxTraceRecursionDepth)
  RDAT_FLAGS(uint32_t, hlsl::DXIL::RaytracingPipelineFlags, Flags)
RDAT_STRUCT_END()
#undef RECORD_TYPE

#define RECORD_TYPE RuntimeDataSubobjectInfo
RDAT_STRUCT_TABLE(RuntimeDataSubobjectInfo, SubobjectTable)
  RDAT_ENUM(uint32_t, hlsl::DXIL::SubobjectKind, Kind)
  RDAT_STRING(Name)
  RDAT_UNION()
    RDAT_UNION_IF(StateObjectConfig, ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::StateObjectConfig))
      RDAT_RECORD_VALUE(StateObjectConfig_t, StateObjectConfig)
    RDAT_UNION_ELIF(RootSignature,
                    ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::GlobalRootSignature) ||
                    ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::LocalRootSignature))
      RDAT_RECORD_VALUE(RootSignature_t, RootSignature)
    RDAT_UNION_ELIF(SubobjectToExportsAssociation, ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::SubobjectToExportsAssociation))
      RDAT_RECORD_VALUE(SubobjectToExportsAssociation_t, SubobjectToExportsAssociation)
    RDAT_UNION_ELIF(RaytracingShaderConfig, ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::RaytracingShaderConfig))
      RDAT_RECORD_VALUE(RaytracingShaderConfig_t, RaytracingShaderConfig)
    RDAT_UNION_ELIF(RaytracingPipelineConfig, ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::RaytracingPipelineConfig))
      RDAT_RECORD_VALUE(RaytracingPipelineConfig_t, RaytracingPipelineConfig)
    RDAT_UNION_ELIF(HitGroup, ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::HitGroup))
      RDAT_RECORD_VALUE(HitGroup_t, HitGroup)
    RDAT_UNION_ELIF(RaytracingPipelineConfig1, ((uint32_t)pRecord->Kind == (uint32_t)hlsl::DXIL::SubobjectKind::RaytracingPipelineConfig1))
      RDAT_RECORD_VALUE(RaytracingPipelineConfig1_t, RaytracingPipelineConfig1)
    RDAT_UNION_ENDIF()
  RDAT_UNION_END()

// Note: this is how one could inject custom code into one of the definition modes:
#if DEF_RDAT_TYPES == DEF_RDAT_READER
  // Add custom code here that only gets added to the reader class definition
#endif

RDAT_STRUCT_END()
#undef RECORD_TYPE

#endif // DEF_RDAT_TYPES
