//
// Copyright (C) 2014-2015 LunarG, Inc.
// Copyright (C) 2022-2025 Arm Limited.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// 1) Programmatically fill in instruction/operand information.
//    This can be used for disassembly, printing documentation, etc.
//
// 2) Print documentation from this parameterization.
//

#include "doc.h"
#include "spvUtil.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <mutex>

namespace spv {
    extern "C" {
        // Include C-based headers that don't have a namespace
        #include "GLSL.ext.KHR.h"
        #include "GLSL.ext.EXT.h"
        #include "GLSL.ext.AMD.h"
        #include "GLSL.ext.NV.h"
        #include "GLSL.ext.ARM.h"
        #include "GLSL.ext.QCOM.h"
    }
}

namespace spv {

//
// Whole set of functions that translate enumerants to their text strings for
// the specification (or their sanitized versions for auto-generating the
// spirv headers.
//
// Also, for masks the ceilings are declared next to these, to help keep them in sync.
// Ceilings should be
//  - one more than the maximum value an enumerant takes on, for non-mask enumerants
//    (for non-sparse enums, this is the number of enumerants)
//  - the number of bits consumed by the set of masks
//    (for non-sparse mask enums, this is the number of enumerants)
//

const char* SourceString(int source)
{
    switch (source) {
    case 0:  return "Unknown";
    case 1:  return "ESSL";
    case 2:  return "GLSL";
    case 3:  return "OpenCL_C";
    case 4:  return "OpenCL_CPP";
    case 5:  return "HLSL";

    default: return "Bad";
    }
}

const char* ExecutionModelString(int model)
{
    switch (model) {
    case 0:  return "Vertex";
    case 1:  return "TessellationControl";
    case 2:  return "TessellationEvaluation";
    case 3:  return "Geometry";
    case 4:  return "Fragment";
    case 5:  return "GLCompute";
    case 6:  return "Kernel";
    case (int)ExecutionModel::TaskNV: return "TaskNV";
    case (int)ExecutionModel::MeshNV: return "MeshNV";
    case (int)ExecutionModel::TaskEXT: return "TaskEXT";
    case (int)ExecutionModel::MeshEXT: return "MeshEXT";

    default: return "Bad";

    case (int)ExecutionModel::RayGenerationKHR: return "RayGenerationKHR";
    case (int)ExecutionModel::IntersectionKHR:  return "IntersectionKHR";
    case (int)ExecutionModel::AnyHitKHR:        return "AnyHitKHR";
    case (int)ExecutionModel::ClosestHitKHR:    return "ClosestHitKHR";
    case (int)ExecutionModel::MissKHR:          return "MissKHR";
    case (int)ExecutionModel::CallableKHR:      return "CallableKHR";
    }
}

const char* AddressingString(int addr)
{
    switch (addr) {
    case 0:  return "Logical";
    case 1:  return "Physical32";
    case 2:  return "Physical64";

    case (int)AddressingModel::PhysicalStorageBuffer64EXT: return "PhysicalStorageBuffer64EXT";

    default: return "Bad";
    }
}

const char* MemoryString(int mem)
{
    switch (mem) {
    case (int)MemoryModel::Simple:     return "Simple";
    case (int)MemoryModel::GLSL450:    return "GLSL450";
    case (int)MemoryModel::OpenCL:     return "OpenCL";
    case (int)MemoryModel::VulkanKHR:  return "VulkanKHR";

    default: return "Bad";
    }
}

const int ExecutionModeCeiling = 40;

const char* ExecutionModeString(int mode)
{
    switch (mode) {
    case 0:  return "Invocations";
    case 1:  return "SpacingEqual";
    case 2:  return "SpacingFractionalEven";
    case 3:  return "SpacingFractionalOdd";
    case 4:  return "VertexOrderCw";
    case 5:  return "VertexOrderCcw";
    case 6:  return "PixelCenterInteger";
    case 7:  return "OriginUpperLeft";
    case 8:  return "OriginLowerLeft";
    case 9:  return "EarlyFragmentTests";
    case 10: return "PointMode";
    case 11: return "Xfb";
    case 12: return "DepthReplacing";
    case 13: return "Bad";
    case 14: return "DepthGreater";
    case 15: return "DepthLess";
    case 16: return "DepthUnchanged";
    case 17: return "LocalSize";
    case 18: return "LocalSizeHint";
    case 19: return "InputPoints";
    case 20: return "InputLines";
    case 21: return "InputLinesAdjacency";
    case 22: return "Triangles";
    case 23: return "InputTrianglesAdjacency";
    case 24: return "Quads";
    case 25: return "Isolines";
    case 26: return "OutputVertices";
    case 27: return "OutputPoints";
    case 28: return "OutputLineStrip";
    case 29: return "OutputTriangleStrip";
    case 30: return "VecTypeHint";
    case 31: return "ContractionOff";
    case 32: return "Bad";

    case (int)ExecutionMode::Initializer:                   return "Initializer";
    case (int)ExecutionMode::Finalizer:                     return "Finalizer";
    case (int)ExecutionMode::SubgroupSize:                  return "SubgroupSize";
    case (int)ExecutionMode::SubgroupsPerWorkgroup:         return "SubgroupsPerWorkgroup";
    case (int)ExecutionMode::SubgroupsPerWorkgroupId:       return "SubgroupsPerWorkgroupId";
    case (int)ExecutionMode::LocalSizeId:                   return "LocalSizeId";
    case (int)ExecutionMode::LocalSizeHintId:               return "LocalSizeHintId";

    case (int)ExecutionMode::PostDepthCoverage:             return "PostDepthCoverage";
    case (int)ExecutionMode::DenormPreserve:                return "DenormPreserve";
    case (int)ExecutionMode::DenormFlushToZero:             return "DenormFlushToZero";
    case (int)ExecutionMode::SignedZeroInfNanPreserve:      return "SignedZeroInfNanPreserve";
    case (int)ExecutionMode::RoundingModeRTE:               return "RoundingModeRTE";
    case (int)ExecutionMode::RoundingModeRTZ:               return "RoundingModeRTZ";

    case (int)ExecutionMode::NonCoherentTileAttachmentReadQCOM: return "NonCoherentTileAttachmentReadQCOM";
    case (int)ExecutionMode::TileShadingRateQCOM:               return "TileShadingRateQCOM";

    case (int)ExecutionMode::EarlyAndLateFragmentTestsAMD:  return "EarlyAndLateFragmentTestsAMD";
    case (int)ExecutionMode::StencilRefUnchangedFrontAMD:   return "StencilRefUnchangedFrontAMD";
    case (int)ExecutionMode::StencilRefLessFrontAMD:        return "StencilRefLessFrontAMD";
    case (int)ExecutionMode::StencilRefGreaterBackAMD:      return "StencilRefGreaterBackAMD";
    case (int)ExecutionMode::StencilRefReplacingEXT:        return "StencilRefReplacingEXT";
    case (int)ExecutionMode::SubgroupUniformControlFlowKHR: return "SubgroupUniformControlFlow";
    case (int)ExecutionMode::MaximallyReconvergesKHR:       return "MaximallyReconverges";

    case (int)ExecutionMode::OutputLinesNV:                 return "OutputLinesNV";
    case (int)ExecutionMode::OutputPrimitivesNV:            return "OutputPrimitivesNV";
    case (int)ExecutionMode::OutputTrianglesNV:             return "OutputTrianglesNV";
    case (int)ExecutionMode::DerivativeGroupQuadsNV:        return "DerivativeGroupQuadsNV";
    case (int)ExecutionMode::DerivativeGroupLinearNV:       return "DerivativeGroupLinearNV";

    case (int)ExecutionMode::PixelInterlockOrderedEXT:         return "PixelInterlockOrderedEXT";
    case (int)ExecutionMode::PixelInterlockUnorderedEXT:       return "PixelInterlockUnorderedEXT";
    case (int)ExecutionMode::SampleInterlockOrderedEXT:        return "SampleInterlockOrderedEXT";
    case (int)ExecutionMode::SampleInterlockUnorderedEXT:      return "SampleInterlockUnorderedEXT";
    case (int)ExecutionMode::ShadingRateInterlockOrderedEXT:   return "ShadingRateInterlockOrderedEXT";
    case (int)ExecutionMode::ShadingRateInterlockUnorderedEXT: return "ShadingRateInterlockUnorderedEXT";

    case (int)ExecutionMode::MaxWorkgroupSizeINTEL:    return "MaxWorkgroupSizeINTEL";
    case (int)ExecutionMode::MaxWorkDimINTEL:          return "MaxWorkDimINTEL";
    case (int)ExecutionMode::NoGlobalOffsetINTEL:      return "NoGlobalOffsetINTEL";
    case (int)ExecutionMode::NumSIMDWorkitemsINTEL:    return "NumSIMDWorkitemsINTEL";

    case (int)ExecutionMode::RequireFullQuadsKHR:      return "RequireFullQuadsKHR";
    case (int)ExecutionMode::QuadDerivativesKHR:       return "QuadDerivativesKHR";

    case (int)ExecutionMode::NonCoherentColorAttachmentReadEXT:        return "NonCoherentColorAttachmentReadEXT";
    case (int)ExecutionMode::NonCoherentDepthAttachmentReadEXT:        return "NonCoherentDepthAttachmentReadEXT";
    case (int)ExecutionMode::NonCoherentStencilAttachmentReadEXT:      return "NonCoherentStencilAttachmentReadEXT";

    case (int)ExecutionMode::Shader64BitIndexingEXT:                   return "Shader64BitIndexingEXT";

    case ExecutionModeCeiling:
    default: return "Bad";
    }
}

const char* StorageClassString(int StorageClass)
{
    switch (StorageClass) {
    case 0:  return "UniformConstant";
    case 1:  return "Input";
    case 2:  return "Uniform";
    case 3:  return "Output";
    case 4:  return "Workgroup";
    case 5:  return "CrossWorkgroup";
    case 6:  return "Private";
    case 7:  return "Function";
    case 8:  return "Generic";
    case 9:  return "PushConstant";
    case 10: return "AtomicCounter";
    case 11: return "Image";
    case 12: return "StorageBuffer";

    case (int)StorageClass::TileAttachmentQCOM:       return "TileAttachmentQCOM";
    case (int)StorageClass::RayPayloadKHR:            return "RayPayloadKHR";
    case (int)StorageClass::HitAttributeKHR:          return "HitAttributeKHR";
    case (int)StorageClass::IncomingRayPayloadKHR:    return "IncomingRayPayloadKHR";
    case (int)StorageClass::ShaderRecordBufferKHR:    return "ShaderRecordBufferKHR";
    case (int)StorageClass::CallableDataKHR:          return "CallableDataKHR";
    case (int)StorageClass::IncomingCallableDataKHR:  return "IncomingCallableDataKHR";

    case (int)StorageClass::PhysicalStorageBufferEXT: return "PhysicalStorageBufferEXT";
    case (int)StorageClass::TaskPayloadWorkgroupEXT:  return "TaskPayloadWorkgroupEXT";
    case (int)StorageClass::HitObjectAttributeNV:     return "HitObjectAttributeNV";
    case (int)StorageClass::TileImageEXT:             return "TileImageEXT";
    case (int)StorageClass::HitObjectAttributeEXT:    return "HitObjectAttributeEXT";
    default: return "Bad";
    }
}

const int DecorationCeiling = 45;

const char* DecorationString(int decoration)
{
    switch (decoration) {
    case 0:  return "RelaxedPrecision";
    case 1:  return "SpecId";
    case 2:  return "Block";
    case 3:  return "BufferBlock";
    case 4:  return "RowMajor";
    case 5:  return "ColMajor";
    case 6:  return "ArrayStride";
    case 7:  return "MatrixStride";
    case 8:  return "GLSLShared";
    case 9:  return "GLSLPacked";
    case 10: return "CPacked";
    case 11: return "BuiltIn";
    case 12: return "Bad";
    case 13: return "NoPerspective";
    case 14: return "Flat";
    case 15: return "Patch";
    case 16: return "Centroid";
    case 17: return "Sample";
    case 18: return "Invariant";
    case 19: return "Restrict";
    case 20: return "Aliased";
    case 21: return "Volatile";
    case 22: return "Constant";
    case 23: return "Coherent";
    case 24: return "NonWritable";
    case 25: return "NonReadable";
    case 26: return "Uniform";
    case 27: return "Bad";
    case 28: return "SaturatedConversion";
    case 29: return "Stream";
    case 30: return "Location";
    case 31: return "Component";
    case 32: return "Index";
    case 33: return "Binding";
    case 34: return "DescriptorSet";
    case 35: return "Offset";
    case 36: return "XfbBuffer";
    case 37: return "XfbStride";
    case 38: return "FuncParamAttr";
    case 39: return "FP Rounding Mode";
    case 40: return "FP Fast Math Mode";
    case 41: return "Linkage Attributes";
    case 42: return "NoContraction";
    case 43: return "InputAttachmentIndex";
    case 44: return "Alignment";

    case DecorationCeiling:
    default:  return "Bad";

    case (int)Decoration::WeightTextureQCOM:           return "DecorationWeightTextureQCOM";
    case (int)Decoration::BlockMatchTextureQCOM:       return "DecorationBlockMatchTextureQCOM";
    case (int)Decoration::BlockMatchSamplerQCOM:       return "DecorationBlockMatchSamplerQCOM";
    case (int)Decoration::ExplicitInterpAMD:           return "ExplicitInterpAMD";
    case (int)Decoration::OverrideCoverageNV:          return "OverrideCoverageNV";
    case (int)Decoration::PassthroughNV:               return "PassthroughNV";
    case (int)Decoration::ViewportRelativeNV:          return "ViewportRelativeNV";
    case (int)Decoration::SecondaryViewportRelativeNV: return "SecondaryViewportRelativeNV";
    case (int)Decoration::PerPrimitiveNV:              return "PerPrimitiveNV";
    case (int)Decoration::PerViewNV:                   return "PerViewNV";
    case (int)Decoration::PerTaskNV:                   return "PerTaskNV";

    case (int)Decoration::PerVertexKHR:                return "PerVertexKHR";

    case (int)Decoration::NonUniformEXT:           return "DecorationNonUniformEXT";
    case (int)Decoration::HlslCounterBufferGOOGLE: return "DecorationHlslCounterBufferGOOGLE";
    case (int)Decoration::HlslSemanticGOOGLE:      return "DecorationHlslSemanticGOOGLE";
    case (int)Decoration::RestrictPointerEXT:      return "DecorationRestrictPointerEXT";
    case (int)Decoration::AliasedPointerEXT:       return "DecorationAliasedPointerEXT";

    case (int)Decoration::HitObjectShaderRecordBufferNV:  return "DecorationHitObjectShaderRecordBufferNV";
    case (int)Decoration::HitObjectShaderRecordBufferEXT:  return "DecorationHitObjectShaderRecordBufferEXT";

    case (int)Decoration::SaturatedToLargestFloat8NormalConversionEXT: return "DecorationSaturatedToLargestFloat8NormalConversionEXT";
    }
}

const char* BuiltInString(int builtIn)
{
    switch (builtIn) {
    case 0:  return "Position";
    case 1:  return "PointSize";
    case 2:  return "Bad";
    case 3:  return "ClipDistance";
    case 4:  return "CullDistance";
    case 5:  return "VertexId";
    case 6:  return "InstanceId";
    case 7:  return "PrimitiveId";
    case 8:  return "InvocationId";
    case 9:  return "Layer";
    case 10: return "ViewportIndex";
    case 11: return "TessLevelOuter";
    case 12: return "TessLevelInner";
    case 13: return "TessCoord";
    case 14: return "PatchVertices";
    case 15: return "FragCoord";
    case 16: return "PointCoord";
    case 17: return "FrontFacing";
    case 18: return "SampleId";
    case 19: return "SamplePosition";
    case 20: return "SampleMask";
    case 21: return "Bad";
    case 22: return "FragDepth";
    case 23: return "HelperInvocation";
    case 24: return "NumWorkgroups";
    case 25: return "WorkgroupSize";
    case 26: return "WorkgroupId";
    case 27: return "LocalInvocationId";
    case 28: return "GlobalInvocationId";
    case 29: return "LocalInvocationIndex";
    case 30: return "WorkDim";
    case 31: return "GlobalSize";
    case 32: return "EnqueuedWorkgroupSize";
    case 33: return "GlobalOffset";
    case 34: return "GlobalLinearId";
    case 35: return "Bad";
    case 36: return "SubgroupSize";
    case 37: return "SubgroupMaxSize";
    case 38: return "NumSubgroups";
    case 39: return "NumEnqueuedSubgroups";
    case 40: return "SubgroupId";
    case 41: return "SubgroupLocalInvocationId";
    case 42: return "VertexIndex";                 // TBD: put next to VertexId?
    case 43: return "InstanceIndex";               // TBD: put next to InstanceId?

    case 4416: return "SubgroupEqMaskKHR";
    case 4417: return "SubgroupGeMaskKHR";
    case 4418: return "SubgroupGtMaskKHR";
    case 4419: return "SubgroupLeMaskKHR";
    case 4420: return "SubgroupLtMaskKHR";
    case 4438: return "DeviceIndex";
    case 4440: return "ViewIndex";
    case 4424: return "BaseVertex";
    case 4425: return "BaseInstance";
    case 4426: return "DrawIndex";
    case 4432: return "PrimitiveShadingRateKHR";
    case 4444: return "ShadingRateKHR";
    case 5014: return "FragStencilRefEXT";

    case (int)BuiltIn::TileOffsetQCOM:     return "TileOffsetQCOM";
    case (int)BuiltIn::TileDimensionQCOM:  return "TileDimensionQCOM";
    case (int)BuiltIn::TileApronSizeQCOM:  return "TileApronSizeQCOM";

    case 4992: return "BaryCoordNoPerspAMD";
    case 4993: return "BaryCoordNoPerspCentroidAMD";
    case 4994: return "BaryCoordNoPerspSampleAMD";
    case 4995: return "BaryCoordSmoothAMD";
    case 4996: return "BaryCoordSmoothCentroidAMD";
    case 4997: return "BaryCoordSmoothSampleAMD";
    case 4998: return "BaryCoordPullModelAMD";
    case (int)BuiltIn::LaunchIdKHR:                 return "LaunchIdKHR";
    case (int)BuiltIn::LaunchSizeKHR:               return "LaunchSizeKHR";
    case (int)BuiltIn::WorldRayOriginKHR:           return "WorldRayOriginKHR";
    case (int)BuiltIn::WorldRayDirectionKHR:        return "WorldRayDirectionKHR";
    case (int)BuiltIn::ObjectRayOriginKHR:          return "ObjectRayOriginKHR";
    case (int)BuiltIn::ObjectRayDirectionKHR:       return "ObjectRayDirectionKHR";
    case (int)BuiltIn::RayTminKHR:                  return "RayTminKHR";
    case (int)BuiltIn::RayTmaxKHR:                  return "RayTmaxKHR";
    case (int)BuiltIn::CullMaskKHR:                 return "CullMaskKHR";
    case (int)BuiltIn::HitTriangleVertexPositionsKHR: return "HitTriangleVertexPositionsKHR";
    case (int)BuiltIn::HitMicroTriangleVertexPositionsNV: return "HitMicroTriangleVertexPositionsNV";
    case (int)BuiltIn::HitMicroTriangleVertexBarycentricsNV: return "HitMicroTriangleVertexBarycentricsNV";
    case (int)BuiltIn::HitKindFrontFacingMicroTriangleNV: return "HitKindFrontFacingMicroTriangleNV";
    case (int)BuiltIn::HitKindBackFacingMicroTriangleNV: return "HitKindBackFacingMicroTriangleNV";
    case (int)BuiltIn::HitIsSphereNV:               return "HitIsSphereNV";
    case (int)BuiltIn::HitIsLSSNV:                  return "HitIsLSSNV";
    case (int)BuiltIn::HitSpherePositionNV:         return "HitSpherePositionNV";
    case (int)BuiltIn::HitSphereRadiusNV:           return "HitSphereRadiusNV";
    case (int)BuiltIn::HitLSSPositionsNV:           return "HitLSSPositionsNV";
    case (int)BuiltIn::HitLSSRadiiNV:               return "HitLLSSRadiiNV";
    case (int)BuiltIn::InstanceCustomIndexKHR:      return "InstanceCustomIndexKHR";
    case (int)BuiltIn::RayGeometryIndexKHR:         return "RayGeometryIndexKHR";
    case (int)BuiltIn::ObjectToWorldKHR:            return "ObjectToWorldKHR";
    case (int)BuiltIn::WorldToObjectKHR:            return "WorldToObjectKHR";
    case (int)BuiltIn::HitTNV:                      return "HitTNV";
    case (int)BuiltIn::HitKindKHR:                  return "HitKindKHR";
    case (int)BuiltIn::IncomingRayFlagsKHR:         return "IncomingRayFlagsKHR";
    case (int)BuiltIn::ViewportMaskNV:              return "ViewportMaskNV";
    case (int)BuiltIn::SecondaryPositionNV:         return "SecondaryPositionNV";
    case (int)BuiltIn::SecondaryViewportMaskNV:     return "SecondaryViewportMaskNV";
    case (int)BuiltIn::PositionPerViewNV:           return "PositionPerViewNV";
    case (int)BuiltIn::ViewportMaskPerViewNV:       return "ViewportMaskPerViewNV";
//    case (int)BuiltIn::FragmentSizeNV:             return "FragmentSizeNV";        // superseded by BuiltInFragSizeEXT
//    case (int)BuiltIn::InvocationsPerPixelNV:      return "InvocationsPerPixelNV"; // superseded by BuiltInFragInvocationCountEXT
    case (int)BuiltIn::BaryCoordKHR:                return "BaryCoordKHR";
    case (int)BuiltIn::BaryCoordNoPerspKHR:         return "BaryCoordNoPerspKHR";
    case (int)BuiltIn::ClusterIDNV:                 return "ClusterIDNV";

    case (int)BuiltIn::FragSizeEXT:                 return "FragSizeEXT";
    case (int)BuiltIn::FragInvocationCountEXT:      return "FragInvocationCountEXT";

    case 5264: return "FullyCoveredEXT";

    case (int)BuiltIn::TaskCountNV:           return "TaskCountNV";
    case (int)BuiltIn::PrimitiveCountNV:      return "PrimitiveCountNV";
    case (int)BuiltIn::PrimitiveIndicesNV:    return "PrimitiveIndicesNV";
    case (int)BuiltIn::ClipDistancePerViewNV: return "ClipDistancePerViewNV";
    case (int)BuiltIn::CullDistancePerViewNV: return "CullDistancePerViewNV";
    case (int)BuiltIn::LayerPerViewNV:        return "LayerPerViewNV";
    case (int)BuiltIn::MeshViewCountNV:       return "MeshViewCountNV";
    case (int)BuiltIn::MeshViewIndicesNV:     return "MeshViewIndicesNV";
    case (int)BuiltIn::WarpsPerSMNV:           return "WarpsPerSMNV";
    case (int)BuiltIn::SMCountNV:              return "SMCountNV";
    case (int)BuiltIn::WarpIDNV:               return "WarpIDNV";
    case (int)BuiltIn::SMIDNV:                 return "SMIDNV";
    case (int)BuiltIn::CurrentRayTimeNV:       return "CurrentRayTimeNV";
    case (int)BuiltIn::PrimitivePointIndicesEXT:        return "PrimitivePointIndicesEXT";
    case (int)BuiltIn::PrimitiveLineIndicesEXT:         return "PrimitiveLineIndicesEXT";
    case (int)BuiltIn::PrimitiveTriangleIndicesEXT:     return "PrimitiveTriangleIndicesEXT";
    case (int)BuiltIn::CullPrimitiveEXT:                return "CullPrimitiveEXT";
    case (int)BuiltIn::CoreCountARM:           return "CoreCountARM";
    case (int)BuiltIn::CoreIDARM:              return "CoreIDARM";
    case (int)BuiltIn::CoreMaxIDARM:           return "CoreMaxIDARM";
    case (int)BuiltIn::WarpIDARM:              return "WarpIDARM";
    case (int)BuiltIn::WarpMaxIDARM:           return "BuiltInWarpMaxIDARM";

    default: return "Bad";
    }
}

const char* DimensionString(int dim)
{
    switch (dim) {
    case 0:  return "1D";
    case 1:  return "2D";
    case 2:  return "3D";
    case 3:  return "Cube";
    case 4:  return "Rect";
    case 5:  return "Buffer";
    case 6:  return "SubpassData";
    case (int)Dim::TileImageDataEXT:  return "TileImageDataEXT";

    default: return "Bad";
    }
}

const char* SamplerAddressingModeString(int mode)
{
    switch (mode) {
    case 0:  return "None";
    case 1:  return "ClampToEdge";
    case 2:  return "Clamp";
    case 3:  return "Repeat";
    case 4:  return "RepeatMirrored";

    default: return "Bad";
    }
}

const char* SamplerFilterModeString(int mode)
{
    switch (mode) {
    case 0: return "Nearest";
    case 1: return "Linear";

    default: return "Bad";
    }
}

const char* ImageFormatString(int format)
{
    switch (format) {
    case  0: return "Unknown";

    // ES/Desktop float
    case  1: return "Rgba32f";
    case  2: return "Rgba16f";
    case  3: return "R32f";
    case  4: return "Rgba8";
    case  5: return "Rgba8Snorm";

    // Desktop float
    case  6: return "Rg32f";
    case  7: return "Rg16f";
    case  8: return "R11fG11fB10f";
    case  9: return "R16f";
    case 10: return "Rgba16";
    case 11: return "Rgb10A2";
    case 12: return "Rg16";
    case 13: return "Rg8";
    case 14: return "R16";
    case 15: return "R8";
    case 16: return "Rgba16Snorm";
    case 17: return "Rg16Snorm";
    case 18: return "Rg8Snorm";
    case 19: return "R16Snorm";
    case 20: return "R8Snorm";

    // ES/Desktop int
    case 21: return "Rgba32i";
    case 22: return "Rgba16i";
    case 23: return "Rgba8i";
    case 24: return "R32i";

    // Desktop int
    case 25: return "Rg32i";
    case 26: return "Rg16i";
    case 27: return "Rg8i";
    case 28: return "R16i";
    case 29: return "R8i";

    // ES/Desktop uint
    case 30: return "Rgba32ui";
    case 31: return "Rgba16ui";
    case 32: return "Rgba8ui";
    case 33: return "R32ui";

    // Desktop uint
    case 34: return "Rgb10a2ui";
    case 35: return "Rg32ui";
    case 36: return "Rg16ui";
    case 37: return "Rg8ui";
    case 38: return "R16ui";
    case 39: return "R8ui";
    case 40: return "R64ui";
    case 41: return "R64i";

    default:
        return "Bad";
    }
}

const char* ImageChannelOrderString(int format)
{
    switch (format) {
    case 0:  return "R";
    case 1:  return "A";
    case 2:  return "RG";
    case 3:  return "RA";
    case 4:  return "RGB";
    case 5:  return "RGBA";
    case 6:  return "BGRA";
    case 7:  return "ARGB";
    case 8:  return "Intensity";
    case 9:  return "Luminance";
    case 10: return "Rx";
    case 11: return "RGx";
    case 12: return "RGBx";
    case 13: return "Depth";
    case 14: return "DepthStencil";
    case 15: return "sRGB";
    case 16: return "sRGBx";
    case 17: return "sRGBA";
    case 18: return "sBGRA";

    default:
        return "Bad";
    }
}

const char* ImageChannelDataTypeString(int type)
{
    switch (type)
    {
    case 0: return "SnormInt8";
    case 1: return "SnormInt16";
    case 2: return "UnormInt8";
    case 3: return "UnormInt16";
    case 4: return "UnormShort565";
    case 5: return "UnormShort555";
    case 6: return "UnormInt101010";
    case 7: return "SignedInt8";
    case 8: return "SignedInt16";
    case 9: return "SignedInt32";
    case 10: return "UnsignedInt8";
    case 11: return "UnsignedInt16";
    case 12: return "UnsignedInt32";
    case 13: return "HalfFloat";
    case 14: return "Float";
    case 15: return "UnormInt24";
    case 16: return "UnormInt101010_2";

    default:
        return "Bad";
    }
}

const int ImageOperandsCeiling = 17;

const char* ImageOperandsString(int format)
{
    switch (format) {
    case (int)ImageOperandsShift::Bias:                    return "Bias";
    case (int)ImageOperandsShift::Lod:                     return "Lod";
    case (int)ImageOperandsShift::Grad:                    return "Grad";
    case (int)ImageOperandsShift::ConstOffset:             return "ConstOffset";
    case (int)ImageOperandsShift::Offset:                  return "Offset";
    case (int)ImageOperandsShift::ConstOffsets:            return "ConstOffsets";
    case (int)ImageOperandsShift::Sample:                  return "Sample";
    case (int)ImageOperandsShift::MinLod:                  return "MinLod";
    case (int)ImageOperandsShift::MakeTexelAvailableKHR:   return "MakeTexelAvailableKHR";
    case (int)ImageOperandsShift::MakeTexelVisibleKHR:     return "MakeTexelVisibleKHR";
    case (int)ImageOperandsShift::NonPrivateTexelKHR:      return "NonPrivateTexelKHR";
    case (int)ImageOperandsShift::VolatileTexelKHR:        return "VolatileTexelKHR";
    case (int)ImageOperandsShift::SignExtend:              return "SignExtend";
    case (int)ImageOperandsShift::ZeroExtend:              return "ZeroExtend";
    case (int)ImageOperandsShift::Nontemporal:             return "Nontemporal";
    case (int)ImageOperandsShift::Offsets:                 return "Offsets";

    case ImageOperandsCeiling:
    default:
        return "Bad";
    }
}

const char* FPFastMathString(int mode)
{
    switch (mode) {
    case 0: return "NotNaN";
    case 1: return "NotInf";
    case 2: return "NSZ";
    case 3: return "AllowRecip";
    case 4: return "Fast";

    default:     return "Bad";
    }
}

const char* FPRoundingModeString(int mode)
{
    switch (mode) {
    case 0:  return "RTE";
    case 1:  return "RTZ";
    case 2:  return "RTP";
    case 3:  return "RTN";

    default: return "Bad";
    }
}

const char* LinkageTypeString(int type)
{
    switch (type) {
    case 0:  return "Export";
    case 1:  return "Import";

    default: return "Bad";
    }
}

const char* FuncParamAttrString(int attr)
{
    switch (attr) {
    case 0:  return "Zext";
    case 1:  return "Sext";
    case 2:  return "ByVal";
    case 3:  return "Sret";
    case 4:  return "NoAlias";
    case 5:  return "NoCapture";
    case 6:  return "NoWrite";
    case 7:  return "NoReadWrite";

    default: return "Bad";
    }
}

const char* AccessQualifierString(int attr)
{
    switch (attr) {
    case 0:  return "ReadOnly";
    case 1:  return "WriteOnly";
    case 2:  return "ReadWrite";

    default: return "Bad";
    }
}

const int SelectControlCeiling = 2;

const char* SelectControlString(int cont)
{
    switch (cont) {
    case 0:  return "Flatten";
    case 1:  return "DontFlatten";

    case SelectControlCeiling:
    default: return "Bad";
    }
}

const int LoopControlCeiling = (int)LoopControlShift::PartialCount + 1;

const char* LoopControlString(int cont)
{
    switch (cont) {
    case (int)LoopControlShift::Unroll:             return "Unroll";
    case (int)LoopControlShift::DontUnroll:         return "DontUnroll";
    case (int)LoopControlShift::DependencyInfinite: return "DependencyInfinite";
    case (int)LoopControlShift::DependencyLength:   return "DependencyLength";
    case (int)LoopControlShift::MinIterations:      return "MinIterations";
    case (int)LoopControlShift::MaxIterations:      return "MaxIterations";
    case (int)LoopControlShift::IterationMultiple:  return "IterationMultiple";
    case (int)LoopControlShift::PeelCount:          return "PeelCount";
    case (int)LoopControlShift::PartialCount:       return "PartialCount";

    case LoopControlCeiling:
    default: return "Bad";
    }
}

const int FunctionControlCeiling = 4;

const char* FunctionControlString(int cont)
{
    switch (cont) {
    case 0:  return "Inline";
    case 1:  return "DontInline";
    case 2:  return "Pure";
    case 3:  return "Const";

    case FunctionControlCeiling:
    default: return "Bad";
    }
}

const char* MemorySemanticsString(int mem)
{
    // Note: No bits set (None) means "Relaxed"
    switch (mem) {
    case 0: return "Bad"; // Note: this is a placeholder for 'Consume'
    case 1: return "Acquire";
    case 2: return "Release";
    case 3: return "AcquireRelease";
    case 4: return "SequentiallyConsistent";
    case 5: return "Bad"; // Note: reserved for future expansion
    case 6: return "UniformMemory";
    case 7: return "SubgroupMemory";
    case 8: return "WorkgroupMemory";
    case 9: return "CrossWorkgroupMemory";
    case 10: return "AtomicCounterMemory";
    case 11: return "ImageMemory";

    default:     return "Bad";
    }
}

const int MemoryAccessCeiling = 6;

const char* MemoryAccessString(int mem)
{
    switch (mem) {
    case (int)MemoryAccessShift::Volatile:                 return "Volatile";
    case (int)MemoryAccessShift::Aligned:                  return "Aligned";
    case (int)MemoryAccessShift::Nontemporal:              return "Nontemporal";
    case (int)MemoryAccessShift::MakePointerAvailableKHR:  return "MakePointerAvailableKHR";
    case (int)MemoryAccessShift::MakePointerVisibleKHR:    return "MakePointerVisibleKHR";
    case (int)MemoryAccessShift::NonPrivatePointerKHR:     return "NonPrivatePointerKHR";

    default: return "Bad";
    }
}

const int CooperativeMatrixOperandsCeiling = 6;

const char* CooperativeMatrixOperandsString(int op)
{
    switch (op) {
    case (int)CooperativeMatrixOperandsShift::MatrixASignedComponentsKHR:  return "ASignedComponentsKHR";
    case (int)CooperativeMatrixOperandsShift::MatrixBSignedComponentsKHR:  return "BSignedComponentsKHR";
    case (int)CooperativeMatrixOperandsShift::MatrixCSignedComponentsKHR:  return "CSignedComponentsKHR";
    case (int)CooperativeMatrixOperandsShift::MatrixResultSignedComponentsKHR:  return "ResultSignedComponentsKHR";
    case (int)CooperativeMatrixOperandsShift::SaturatingAccumulationKHR:   return "SaturatingAccumulationKHR";

    default: return "Bad";
    }
}

const int TensorAddressingOperandsCeiling = 3;

const char* TensorAddressingOperandsString(int op)
{
    switch (op) {
    case (int)TensorAddressingOperandsShift::TensorView:  return "TensorView";
    case (int)TensorAddressingOperandsShift::DecodeFunc:  return "DecodeFunc";

    default: return "Bad";
    }
}

const char* ScopeString(int mem)
{
    switch (mem) {
    case 0:  return "CrossDevice";
    case 1:  return "Device";
    case 2:  return "Workgroup";
    case 3:  return "Subgroup";
    case 4:  return "Invocation";

    default: return "Bad";
    }
}

const char* GroupOperationString(int gop)
{

    switch (gop)
    {
    case (int)GroupOperation::Reduce:  return "Reduce";
    case (int)GroupOperation::InclusiveScan:  return "InclusiveScan";
    case (int)GroupOperation::ExclusiveScan:  return "ExclusiveScan";
    case (int)GroupOperation::ClusteredReduce:  return "ClusteredReduce";
    case (int)GroupOperation::PartitionedReduceNV:  return "PartitionedReduceNV";
    case (int)GroupOperation::PartitionedInclusiveScanNV:  return "PartitionedInclusiveScanNV";
    case (int)GroupOperation::PartitionedExclusiveScanNV:  return "PartitionedExclusiveScanNV";

    default: return "Bad";
    }
}

const char* KernelEnqueueFlagsString(int flag)
{
    switch (flag)
    {
    case 0:  return "NoWait";
    case 1:  return "WaitKernel";
    case 2:  return "WaitWorkGroup";

    default: return "Bad";
    }
}

const char* KernelProfilingInfoString(int info)
{
    switch (info)
    {
    case 0:  return "CmdExecTime";

    default: return "Bad";
    }
}

const char* CapabilityString(int info)
{
    switch (info)
    {
    case 0:  return "Matrix";
    case 1:  return "Shader";
    case 2:  return "Geometry";
    case 3:  return "Tessellation";
    case 4:  return "Addresses";
    case 5:  return "Linkage";
    case 6:  return "Kernel";
    case 7:  return "Vector16";
    case 8:  return "Float16Buffer";
    case 9:  return "Float16";
    case 10: return "Float64";
    case 11: return "Int64";
    case 12: return "Int64Atomics";
    case 13: return "ImageBasic";
    case 14: return "ImageReadWrite";
    case 15: return "ImageMipmap";
    case 16: return "Bad";
    case 17: return "Pipes";
    case 18: return "Groups";
    case 19: return "DeviceEnqueue";
    case 20: return "LiteralSampler";
    case 21: return "AtomicStorage";
    case 22: return "Int16";
    case 23: return "TessellationPointSize";
    case 24: return "GeometryPointSize";
    case 25: return "ImageGatherExtended";
    case 26: return "Bad";
    case 27: return "StorageImageMultisample";
    case 28: return "UniformBufferArrayDynamicIndexing";
    case 29: return "SampledImageArrayDynamicIndexing";
    case 30: return "StorageBufferArrayDynamicIndexing";
    case 31: return "StorageImageArrayDynamicIndexing";
    case 32: return "ClipDistance";
    case 33: return "CullDistance";
    case 34: return "ImageCubeArray";
    case 35: return "SampleRateShading";
    case 36: return "ImageRect";
    case 37: return "SampledRect";
    case 38: return "GenericPointer";
    case 39: return "Int8";
    case 40: return "InputAttachment";
    case 41: return "SparseResidency";
    case 42: return "MinLod";
    case 43: return "Sampled1D";
    case 44: return "Image1D";
    case 45: return "SampledCubeArray";
    case 46: return "SampledBuffer";
    case 47: return "ImageBuffer";
    case 48: return "ImageMSArray";
    case 49: return "StorageImageExtendedFormats";
    case 50: return "ImageQuery";
    case 51: return "DerivativeControl";
    case 52: return "InterpolationFunction";
    case 53: return "TransformFeedback";
    case 54: return "GeometryStreams";
    case 55: return "StorageImageReadWithoutFormat";
    case 56: return "StorageImageWriteWithoutFormat";
    case 57: return "MultiViewport";
    case 61: return "GroupNonUniform";
    case 62: return "GroupNonUniformVote";
    case 63: return "GroupNonUniformArithmetic";
    case 64: return "GroupNonUniformBallot";
    case 65: return "GroupNonUniformShuffle";
    case 66: return "GroupNonUniformShuffleRelative";
    case 67: return "GroupNonUniformClustered";
    case 68: return "GroupNonUniformQuad";

    case (int)Capability::SubgroupBallotKHR: return "SubgroupBallotKHR";
    case (int)Capability::DrawParameters:    return "DrawParameters";
    case (int)Capability::SubgroupVoteKHR:   return "SubgroupVoteKHR";
    case (int)Capability::GroupNonUniformRotateKHR: return "GroupNonUniformRotateKHR";

    case (int)Capability::StorageUniformBufferBlock16: return "StorageUniformBufferBlock16";
    case (int)Capability::StorageUniform16:            return "StorageUniform16";
    case (int)Capability::StoragePushConstant16:       return "StoragePushConstant16";
    case (int)Capability::StorageInputOutput16:        return "StorageInputOutput16";

    case (int)Capability::StorageBuffer8BitAccess:             return "StorageBuffer8BitAccess";
    case (int)Capability::UniformAndStorageBuffer8BitAccess:   return "UniformAndStorageBuffer8BitAccess";
    case (int)Capability::StoragePushConstant8:                return "StoragePushConstant8";

    case (int)Capability::DeviceGroup: return "DeviceGroup";
    case (int)Capability::MultiView:   return "MultiView";

    case (int)Capability::DenormPreserve:           return "DenormPreserve";
    case (int)Capability::DenormFlushToZero:        return "DenormFlushToZero";
    case (int)Capability::SignedZeroInfNanPreserve: return "SignedZeroInfNanPreserve";
    case (int)Capability::RoundingModeRTE:          return "RoundingModeRTE";
    case (int)Capability::RoundingModeRTZ:          return "RoundingModeRTZ";

    case (int)Capability::StencilExportEXT: return "StencilExportEXT";

    case (int)Capability::Float16ImageAMD:       return "Float16ImageAMD";
    case (int)Capability::ImageGatherBiasLodAMD: return "ImageGatherBiasLodAMD";
    case (int)Capability::FragmentMaskAMD:       return "FragmentMaskAMD";
    case (int)Capability::ImageReadWriteLodAMD:  return "ImageReadWriteLodAMD";

    case (int)Capability::AtomicStorageOps:             return "AtomicStorageOps";

    case (int)Capability::SampleMaskPostDepthCoverage:  return "SampleMaskPostDepthCoverage";
    case (int)Capability::GeometryShaderPassthroughNV:     return "GeometryShaderPassthroughNV";
    case (int)Capability::ShaderViewportIndexLayerNV:      return "ShaderViewportIndexLayerNV";
    case (int)Capability::ShaderViewportMaskNV:            return "ShaderViewportMaskNV";
    case (int)Capability::ShaderStereoViewNV:              return "ShaderStereoViewNV";
    case (int)Capability::PerViewAttributesNV:             return "PerViewAttributesNV";
    case (int)Capability::GroupNonUniformPartitionedNV:    return "GroupNonUniformPartitionedNV";
    case (int)Capability::RayTracingNV:                    return "RayTracingNV";
    case (int)Capability::RayTracingMotionBlurNV:          return "RayTracingMotionBlurNV";
    case (int)Capability::RayTracingKHR:                   return "RayTracingKHR";
    case (int)Capability::RayCullMaskKHR:                  return "RayCullMaskKHR";
    case (int)Capability::RayQueryKHR:                     return "RayQueryKHR";
    case (int)Capability::RayTracingProvisionalKHR:        return "RayTracingProvisionalKHR";
    case (int)Capability::RayTraversalPrimitiveCullingKHR: return "RayTraversalPrimitiveCullingKHR";
    case (int)Capability::RayTracingPositionFetchKHR:      return "RayTracingPositionFetchKHR";
    case (int)Capability::DisplacementMicromapNV:           return "DisplacementMicromapNV";
    case (int)Capability::RayTracingOpacityMicromapEXT:    return "RayTracingOpacityMicromapEXT";
    case (int)Capability::RayTracingDisplacementMicromapNV: return "RayTracingDisplacementMicromapNV";
    case (int)Capability::RayQueryPositionFetchKHR:        return "RayQueryPositionFetchKHR";
    case (int)Capability::ComputeDerivativeGroupQuadsNV:   return "ComputeDerivativeGroupQuadsNV";
    case (int)Capability::ComputeDerivativeGroupLinearNV:  return "ComputeDerivativeGroupLinearNV";
    case (int)Capability::FragmentBarycentricKHR:          return "FragmentBarycentricKHR";
    case (int)Capability::MeshShadingNV:                   return "MeshShadingNV";
    case (int)Capability::ImageFootprintNV:                return "ImageFootprintNV";
    case (int)Capability::MeshShadingEXT:                  return "MeshShadingEXT";
//    case (int)Capability::ShadingRateNV:                   return "ShadingRateNV";  // superseded by FragmentDensityEXT
    case (int)Capability::SampleMaskOverrideCoverageNV:    return "SampleMaskOverrideCoverageNV";
    case (int)Capability::FragmentDensityEXT:              return "FragmentDensityEXT";

    case (int)Capability::FragmentFullyCoveredEXT: return "FragmentFullyCoveredEXT";

    case (int)Capability::ShaderNonUniformEXT:                          return "ShaderNonUniformEXT";
    case (int)Capability::RuntimeDescriptorArrayEXT:                    return "RuntimeDescriptorArrayEXT";
    case (int)Capability::InputAttachmentArrayDynamicIndexingEXT:       return "InputAttachmentArrayDynamicIndexingEXT";
    case (int)Capability::UniformTexelBufferArrayDynamicIndexingEXT:    return "UniformTexelBufferArrayDynamicIndexingEXT";
    case (int)Capability::StorageTexelBufferArrayDynamicIndexingEXT:    return "StorageTexelBufferArrayDynamicIndexingEXT";
    case (int)Capability::UniformBufferArrayNonUniformIndexingEXT:      return "UniformBufferArrayNonUniformIndexingEXT";
    case (int)Capability::SampledImageArrayNonUniformIndexingEXT:       return "SampledImageArrayNonUniformIndexingEXT";
    case (int)Capability::StorageBufferArrayNonUniformIndexingEXT:      return "StorageBufferArrayNonUniformIndexingEXT";
    case (int)Capability::StorageImageArrayNonUniformIndexingEXT:       return "StorageImageArrayNonUniformIndexingEXT";
    case (int)Capability::InputAttachmentArrayNonUniformIndexingEXT:    return "InputAttachmentArrayNonUniformIndexingEXT";
    case (int)Capability::UniformTexelBufferArrayNonUniformIndexingEXT: return "UniformTexelBufferArrayNonUniformIndexingEXT";
    case (int)Capability::StorageTexelBufferArrayNonUniformIndexingEXT: return "StorageTexelBufferArrayNonUniformIndexingEXT";

    case (int)Capability::VulkanMemoryModelKHR:                return "VulkanMemoryModelKHR";
    case (int)Capability::VulkanMemoryModelDeviceScopeKHR:     return "VulkanMemoryModelDeviceScopeKHR";

    case (int)Capability::PhysicalStorageBufferAddressesEXT:   return "PhysicalStorageBufferAddressesEXT";

    case (int)Capability::VariablePointers:                    return "VariablePointers";

    case (int)Capability::CooperativeMatrixNV:     return "CooperativeMatrixNV";
    case (int)Capability::CooperativeMatrixKHR:    return "CooperativeMatrixKHR";
    case (int)Capability::CooperativeMatrixReductionsNV:           return "CooperativeMatrixReductionsNV";
    case (int)Capability::CooperativeMatrixConversionsNV:          return "CooperativeMatrixConversionsNV";
    case (int)Capability::CooperativeMatrixPerElementOperationsNV: return "CooperativeMatrixPerElementOperationsNV";
    case (int)Capability::CooperativeMatrixTensorAddressingNV:     return "CooperativeMatrixTensorAddressingNV";
    case (int)Capability::CooperativeMatrixBlockLoadsNV:           return "CooperativeMatrixBlockLoadsNV";
    case (int)Capability::TensorAddressingNV:                      return "TensorAddressingNV";

    case (int)Capability::ShaderSMBuiltinsNV:      return "ShaderSMBuiltinsNV";

    case (int)Capability::CooperativeVectorNV:                     return "CooperativeVectorNV";
    case (int)Capability::CooperativeVectorTrainingNV:             return "CooperativeVectorTrainingNV";

    case (int)Capability::FragmentShaderSampleInterlockEXT:        return "FragmentShaderSampleInterlockEXT";
    case (int)Capability::FragmentShaderPixelInterlockEXT:         return "FragmentShaderPixelInterlockEXT";
    case (int)Capability::FragmentShaderShadingRateInterlockEXT:   return "FragmentShaderShadingRateInterlockEXT";

    case (int)Capability::TileImageColorReadAccessEXT:           return "TileImageColorReadAccessEXT";
    case (int)Capability::TileImageDepthReadAccessEXT:           return "TileImageDepthReadAccessEXT";
    case (int)Capability::TileImageStencilReadAccessEXT:         return "TileImageStencilReadAccessEXT";

    case (int)Capability::CooperativeMatrixLayoutsARM:             return "CooperativeMatrixLayoutsARM";
    case (int)Capability::TensorsARM:                              return "TensorsARM";

    case (int)Capability::FragmentShadingRateKHR:                  return "FragmentShadingRateKHR";

    case (int)Capability::DemoteToHelperInvocationEXT:             return "DemoteToHelperInvocationEXT";
    case (int)Capability::AtomicFloat16VectorNV:                   return "AtomicFloat16VectorNV";
    case (int)Capability::ShaderClockKHR:                          return "ShaderClockKHR";
    case (int)Capability::QuadControlKHR:                          return "QuadControlKHR";
    case (int)Capability::Int64ImageEXT:                           return "Int64ImageEXT";

    case (int)Capability::IntegerFunctions2INTEL:              return "IntegerFunctions2INTEL";

    case (int)Capability::ExpectAssumeKHR:                         return "ExpectAssumeKHR";

    case (int)Capability::AtomicFloat16AddEXT:                     return "AtomicFloat16AddEXT";
    case (int)Capability::AtomicFloat32AddEXT:                     return "AtomicFloat32AddEXT";
    case (int)Capability::AtomicFloat64AddEXT:                     return "AtomicFloat64AddEXT";
    case (int)Capability::AtomicFloat16MinMaxEXT:                  return "AtomicFloat16MinMaxEXT";
    case (int)Capability::AtomicFloat32MinMaxEXT:                  return "AtomicFloat32MinMaxEXT";
    case (int)Capability::AtomicFloat64MinMaxEXT:                  return "AtomicFloat64MinMaxEXT";

    case (int)Capability::WorkgroupMemoryExplicitLayoutKHR:            return "WorkgroupMemoryExplicitLayoutKHR";
    case (int)Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR:  return "WorkgroupMemoryExplicitLayout8BitAccessKHR";
    case (int)Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR: return "WorkgroupMemoryExplicitLayout16BitAccessKHR";
    case (int)Capability::CoreBuiltinsARM:                             return "CoreBuiltinsARM";

    case (int)Capability::ShaderInvocationReorderNV:                return "ShaderInvocationReorderNV";
    case (int)Capability::ShaderInvocationReorderEXT:               return "ShaderInvocationReorderEXT";

    case (int)Capability::TextureSampleWeightedQCOM:           return "TextureSampleWeightedQCOM";
    case (int)Capability::TextureBoxFilterQCOM:                return "TextureBoxFilterQCOM";
    case (int)Capability::TextureBlockMatchQCOM:               return "TextureBlockMatchQCOM";
    case (int)Capability::TileShadingQCOM:                     return "TileShadingQCOM";
    case (int)Capability::TextureBlockMatch2QCOM:              return "TextureBlockMatch2QCOM";

    case (int)Capability::CooperativeMatrixConversionQCOM:     return "CooperativeMatrixConversionQCOM";

    case (int)Capability::ReplicatedCompositesEXT:             return "ReplicatedCompositesEXT";

    case (int)Capability::DotProductKHR:                       return "DotProductKHR";
    case (int)Capability::DotProductInputAllKHR:               return "DotProductInputAllKHR";
    case (int)Capability::DotProductInput4x8BitKHR:            return "DotProductInput4x8BitKHR";
    case (int)Capability::DotProductInput4x8BitPackedKHR:      return "DotProductInput4x8BitPackedKHR";

    case (int)Capability::RayTracingClusterAccelerationStructureNV:   return "RayTracingClusterAccelerationStructureNV";

    case (int)Capability::RayTracingSpheresGeometryNV:             return "RayTracingSpheresGeometryNV";
    case (int)Capability::RayTracingLinearSweptSpheresGeometryNV:  return "RayTracingLinearSweptSpheresGeometryNV";

    case (int)Capability::BFloat16TypeKHR:                     return "BFloat16TypeKHR";
    case (int)Capability::BFloat16DotProductKHR:               return "BFloat16DotProductKHR";
    case (int)Capability::BFloat16CooperativeMatrixKHR:        return "BFloat16CooperativeMatrixKHR";

    case (int)Capability::Float8EXT:                           return "Float8EXT";
    case (int)Capability::Float8CooperativeMatrixEXT:          return "Float8CooperativeMatrixEXT";

    case (int)Capability::Shader64BitIndexingEXT:              return "CapabilityShader64BitIndexingEXT";

    default: return "Bad";
    }
}

const char* OpcodeString(int op)
{
    switch (op) {
    case 0:   return "OpNop";
    case 1:   return "OpUndef";
    case 2:   return "OpSourceContinued";
    case 3:   return "OpSource";
    case 4:   return "OpSourceExtension";
    case 5:   return "OpName";
    case 6:   return "OpMemberName";
    case 7:   return "OpString";
    case 8:   return "OpLine";
    case 9:   return "Bad";
    case 10:  return "OpExtension";
    case 11:  return "OpExtInstImport";
    case 12:  return "OpExtInst";
    case 13:  return "Bad";
    case 14:  return "OpMemoryModel";
    case 15:  return "OpEntryPoint";
    case 16:  return "OpExecutionMode";
    case 17:  return "OpCapability";
    case 18:  return "Bad";
    case 19:  return "OpTypeVoid";
    case 20:  return "OpTypeBool";
    case 21:  return "OpTypeInt";
    case 22:  return "OpTypeFloat";
    case 23:  return "OpTypeVector";
    case 24:  return "OpTypeMatrix";
    case 25:  return "OpTypeImage";
    case 26:  return "OpTypeSampler";
    case 27:  return "OpTypeSampledImage";
    case 28:  return "OpTypeArray";
    case 29:  return "OpTypeRuntimeArray";
    case 30:  return "OpTypeStruct";
    case 31:  return "OpTypeOpaque";
    case 32:  return "OpTypePointer";
    case 33:  return "OpTypeFunction";
    case 34:  return "OpTypeEvent";
    case 35:  return "OpTypeDeviceEvent";
    case 36:  return "OpTypeReserveId";
    case 37:  return "OpTypeQueue";
    case 38:  return "OpTypePipe";
    case 39:  return "OpTypeForwardPointer";
    case 40:  return "Bad";
    case 41:  return "OpConstantTrue";
    case 42:  return "OpConstantFalse";
    case 43:  return "OpConstant";
    case 44:  return "OpConstantComposite";
    case 45:  return "OpConstantSampler";
    case 46:  return "OpConstantNull";
    case 47:  return "Bad";
    case 48:  return "OpSpecConstantTrue";
    case 49:  return "OpSpecConstantFalse";
    case 50:  return "OpSpecConstant";
    case 51:  return "OpSpecConstantComposite";
    case 52:  return "OpSpecConstantOp";
    case 53:  return "Bad";
    case 54:  return "OpFunction";
    case 55:  return "OpFunctionParameter";
    case 56:  return "OpFunctionEnd";
    case 57:  return "OpFunctionCall";
    case 58:  return "Bad";
    case 59:  return "OpVariable";
    case 60:  return "OpImageTexelPointer";
    case 61:  return "OpLoad";
    case 62:  return "OpStore";
    case 63:  return "OpCopyMemory";
    case 64:  return "OpCopyMemorySized";
    case 65:  return "OpAccessChain";
    case 66:  return "OpInBoundsAccessChain";
    case 67:  return "OpPtrAccessChain";
    case 68:  return "OpArrayLength";
    case 69:  return "OpGenericPtrMemSemantics";
    case 70:  return "OpInBoundsPtrAccessChain";
    case 71:  return "OpDecorate";
    case 72:  return "OpMemberDecorate";
    case 73:  return "OpDecorationGroup";
    case 74:  return "OpGroupDecorate";
    case 75:  return "OpGroupMemberDecorate";
    case 76:  return "Bad";
    case 77:  return "OpVectorExtractDynamic";
    case 78:  return "OpVectorInsertDynamic";
    case 79:  return "OpVectorShuffle";
    case 80:  return "OpCompositeConstruct";
    case 81:  return "OpCompositeExtract";
    case 82:  return "OpCompositeInsert";
    case 83:  return "OpCopyObject";
    case 84:  return "OpTranspose";
    case (int)Op::OpCopyLogical: return "OpCopyLogical";
    case 85:  return "Bad";
    case 86:  return "OpSampledImage";
    case 87:  return "OpImageSampleImplicitLod";
    case 88:  return "OpImageSampleExplicitLod";
    case 89:  return "OpImageSampleDrefImplicitLod";
    case 90:  return "OpImageSampleDrefExplicitLod";
    case 91:  return "OpImageSampleProjImplicitLod";
    case 92:  return "OpImageSampleProjExplicitLod";
    case 93:  return "OpImageSampleProjDrefImplicitLod";
    case 94:  return "OpImageSampleProjDrefExplicitLod";
    case 95:  return "OpImageFetch";
    case 96:  return "OpImageGather";
    case 97:  return "OpImageDrefGather";
    case 98:  return "OpImageRead";
    case 99:  return "OpImageWrite";
    case 100: return "OpImage";
    case 101: return "OpImageQueryFormat";
    case 102: return "OpImageQueryOrder";
    case 103: return "OpImageQuerySizeLod";
    case 104: return "OpImageQuerySize";
    case 105: return "OpImageQueryLod";
    case 106: return "OpImageQueryLevels";
    case 107: return "OpImageQuerySamples";
    case 108: return "Bad";
    case 109: return "OpConvertFToU";
    case 110: return "OpConvertFToS";
    case 111: return "OpConvertSToF";
    case 112: return "OpConvertUToF";
    case 113: return "OpUConvert";
    case 114: return "OpSConvert";
    case 115: return "OpFConvert";
    case 116: return "OpQuantizeToF16";
    case 117: return "OpConvertPtrToU";
    case 118: return "OpSatConvertSToU";
    case 119: return "OpSatConvertUToS";
    case 120: return "OpConvertUToPtr";
    case 121: return "OpPtrCastToGeneric";
    case 122: return "OpGenericCastToPtr";
    case 123: return "OpGenericCastToPtrExplicit";
    case 124: return "OpBitcast";
    case 125: return "Bad";
    case 126: return "OpSNegate";
    case 127: return "OpFNegate";
    case 128: return "OpIAdd";
    case 129: return "OpFAdd";
    case 130: return "OpISub";
    case 131: return "OpFSub";
    case 132: return "OpIMul";
    case 133: return "OpFMul";
    case 134: return "OpUDiv";
    case 135: return "OpSDiv";
    case 136: return "OpFDiv";
    case 137: return "OpUMod";
    case 138: return "OpSRem";
    case 139: return "OpSMod";
    case 140: return "OpFRem";
    case 141: return "OpFMod";
    case 142: return "OpVectorTimesScalar";
    case 143: return "OpMatrixTimesScalar";
    case 144: return "OpVectorTimesMatrix";
    case 145: return "OpMatrixTimesVector";
    case 146: return "OpMatrixTimesMatrix";
    case 147: return "OpOuterProduct";
    case 148: return "OpDot";
    case 149: return "OpIAddCarry";
    case 150: return "OpISubBorrow";
    case 151: return "OpUMulExtended";
    case 152: return "OpSMulExtended";
    case 153: return "Bad";
    case 154: return "OpAny";
    case 155: return "OpAll";
    case 156: return "OpIsNan";
    case 157: return "OpIsInf";
    case 158: return "OpIsFinite";
    case 159: return "OpIsNormal";
    case 160: return "OpSignBitSet";
    case 161: return "OpLessOrGreater";
    case 162: return "OpOrdered";
    case 163: return "OpUnordered";
    case 164: return "OpLogicalEqual";
    case 165: return "OpLogicalNotEqual";
    case 166: return "OpLogicalOr";
    case 167: return "OpLogicalAnd";
    case 168: return "OpLogicalNot";
    case 169: return "OpSelect";
    case 170: return "OpIEqual";
    case 171: return "OpINotEqual";
    case 172: return "OpUGreaterThan";
    case 173: return "OpSGreaterThan";
    case 174: return "OpUGreaterThanEqual";
    case 175: return "OpSGreaterThanEqual";
    case 176: return "OpULessThan";
    case 177: return "OpSLessThan";
    case 178: return "OpULessThanEqual";
    case 179: return "OpSLessThanEqual";
    case 180: return "OpFOrdEqual";
    case 181: return "OpFUnordEqual";
    case 182: return "OpFOrdNotEqual";
    case 183: return "OpFUnordNotEqual";
    case 184: return "OpFOrdLessThan";
    case 185: return "OpFUnordLessThan";
    case 186: return "OpFOrdGreaterThan";
    case 187: return "OpFUnordGreaterThan";
    case 188: return "OpFOrdLessThanEqual";
    case 189: return "OpFUnordLessThanEqual";
    case 190: return "OpFOrdGreaterThanEqual";
    case 191: return "OpFUnordGreaterThanEqual";
    case 192: return "Bad";
    case 193: return "Bad";
    case 194: return "OpShiftRightLogical";
    case 195: return "OpShiftRightArithmetic";
    case 196: return "OpShiftLeftLogical";
    case 197: return "OpBitwiseOr";
    case 198: return "OpBitwiseXor";
    case 199: return "OpBitwiseAnd";
    case 200: return "OpNot";
    case 201: return "OpBitFieldInsert";
    case 202: return "OpBitFieldSExtract";
    case 203: return "OpBitFieldUExtract";
    case 204: return "OpBitReverse";
    case 205: return "OpBitCount";
    case 206: return "Bad";
    case 207: return "OpDPdx";
    case 208: return "OpDPdy";
    case 209: return "OpFwidth";
    case 210: return "OpDPdxFine";
    case 211: return "OpDPdyFine";
    case 212: return "OpFwidthFine";
    case 213: return "OpDPdxCoarse";
    case 214: return "OpDPdyCoarse";
    case 215: return "OpFwidthCoarse";
    case 216: return "Bad";
    case 217: return "Bad";
    case 218: return "OpEmitVertex";
    case 219: return "OpEndPrimitive";
    case 220: return "OpEmitStreamVertex";
    case 221: return "OpEndStreamPrimitive";
    case 222: return "Bad";
    case 223: return "Bad";
    case 224: return "OpControlBarrier";
    case 225: return "OpMemoryBarrier";
    case 226: return "Bad";
    case 227: return "OpAtomicLoad";
    case 228: return "OpAtomicStore";
    case 229: return "OpAtomicExchange";
    case 230: return "OpAtomicCompareExchange";
    case 231: return "OpAtomicCompareExchangeWeak";
    case 232: return "OpAtomicIIncrement";
    case 233: return "OpAtomicIDecrement";
    case 234: return "OpAtomicIAdd";
    case 235: return "OpAtomicISub";
    case 236: return "OpAtomicSMin";
    case 237: return "OpAtomicUMin";
    case 238: return "OpAtomicSMax";
    case 239: return "OpAtomicUMax";
    case 240: return "OpAtomicAnd";
    case 241: return "OpAtomicOr";
    case 242: return "OpAtomicXor";
    case 243: return "Bad";
    case 244: return "Bad";
    case 245: return "OpPhi";
    case 246: return "OpLoopMerge";
    case 247: return "OpSelectionMerge";
    case 248: return "OpLabel";
    case 249: return "OpBranch";
    case 250: return "OpBranchConditional";
    case 251: return "OpSwitch";
    case 252: return "OpKill";
    case 253: return "OpReturn";
    case 254: return "OpReturnValue";
    case 255: return "OpUnreachable";
    case 256: return "OpLifetimeStart";
    case 257: return "OpLifetimeStop";
    case 258: return "Bad";
    case 259: return "OpGroupAsyncCopy";
    case 260: return "OpGroupWaitEvents";
    case 261: return "OpGroupAll";
    case 262: return "OpGroupAny";
    case 263: return "OpGroupBroadcast";
    case 264: return "OpGroupIAdd";
    case 265: return "OpGroupFAdd";
    case 266: return "OpGroupFMin";
    case 267: return "OpGroupUMin";
    case 268: return "OpGroupSMin";
    case 269: return "OpGroupFMax";
    case 270: return "OpGroupUMax";
    case 271: return "OpGroupSMax";
    case 272: return "Bad";
    case 273: return "Bad";
    case 274: return "OpReadPipe";
    case 275: return "OpWritePipe";
    case 276: return "OpReservedReadPipe";
    case 277: return "OpReservedWritePipe";
    case 278: return "OpReserveReadPipePackets";
    case 279: return "OpReserveWritePipePackets";
    case 280: return "OpCommitReadPipe";
    case 281: return "OpCommitWritePipe";
    case 282: return "OpIsValidReserveId";
    case 283: return "OpGetNumPipePackets";
    case 284: return "OpGetMaxPipePackets";
    case 285: return "OpGroupReserveReadPipePackets";
    case 286: return "OpGroupReserveWritePipePackets";
    case 287: return "OpGroupCommitReadPipe";
    case 288: return "OpGroupCommitWritePipe";
    case 289: return "Bad";
    case 290: return "Bad";
    case 291: return "OpEnqueueMarker";
    case 292: return "OpEnqueueKernel";
    case 293: return "OpGetKernelNDrangeSubGroupCount";
    case 294: return "OpGetKernelNDrangeMaxSubGroupSize";
    case 295: return "OpGetKernelWorkGroupSize";
    case 296: return "OpGetKernelPreferredWorkGroupSizeMultiple";
    case 297: return "OpRetainEvent";
    case 298: return "OpReleaseEvent";
    case 299: return "OpCreateUserEvent";
    case 300: return "OpIsValidEvent";
    case 301: return "OpSetUserEventStatus";
    case 302: return "OpCaptureEventProfilingInfo";
    case 303: return "OpGetDefaultQueue";
    case 304: return "OpBuildNDRange";
    case 305: return "OpImageSparseSampleImplicitLod";
    case 306: return "OpImageSparseSampleExplicitLod";
    case 307: return "OpImageSparseSampleDrefImplicitLod";
    case 308: return "OpImageSparseSampleDrefExplicitLod";
    case 309: return "OpImageSparseSampleProjImplicitLod";
    case 310: return "OpImageSparseSampleProjExplicitLod";
    case 311: return "OpImageSparseSampleProjDrefImplicitLod";
    case 312: return "OpImageSparseSampleProjDrefExplicitLod";
    case 313: return "OpImageSparseFetch";
    case 314: return "OpImageSparseGather";
    case 315: return "OpImageSparseDrefGather";
    case 316: return "OpImageSparseTexelsResident";
    case 317: return "OpNoLine";
    case 318: return "OpAtomicFlagTestAndSet";
    case 319: return "OpAtomicFlagClear";
    case 320: return "OpImageSparseRead";

    case (int)Op::OpModuleProcessed: return "OpModuleProcessed";
    case (int)Op::OpExecutionModeId: return "OpExecutionModeId";
    case (int)Op::OpDecorateId:      return "OpDecorateId";

    case 333: return "OpGroupNonUniformElect";
    case 334: return "OpGroupNonUniformAll";
    case 335: return "OpGroupNonUniformAny";
    case 336: return "OpGroupNonUniformAllEqual";
    case 337: return "OpGroupNonUniformBroadcast";
    case 338: return "OpGroupNonUniformBroadcastFirst";
    case 339: return "OpGroupNonUniformBallot";
    case 340: return "OpGroupNonUniformInverseBallot";
    case 341: return "OpGroupNonUniformBallotBitExtract";
    case 342: return "OpGroupNonUniformBallotBitCount";
    case 343: return "OpGroupNonUniformBallotFindLSB";
    case 344: return "OpGroupNonUniformBallotFindMSB";
    case 345: return "OpGroupNonUniformShuffle";
    case 346: return "OpGroupNonUniformShuffleXor";
    case 347: return "OpGroupNonUniformShuffleUp";
    case 348: return "OpGroupNonUniformShuffleDown";
    case 349: return "OpGroupNonUniformIAdd";
    case 350: return "OpGroupNonUniformFAdd";
    case 351: return "OpGroupNonUniformIMul";
    case 352: return "OpGroupNonUniformFMul";
    case 353: return "OpGroupNonUniformSMin";
    case 354: return "OpGroupNonUniformUMin";
    case 355: return "OpGroupNonUniformFMin";
    case 356: return "OpGroupNonUniformSMax";
    case 357: return "OpGroupNonUniformUMax";
    case 358: return "OpGroupNonUniformFMax";
    case 359: return "OpGroupNonUniformBitwiseAnd";
    case 360: return "OpGroupNonUniformBitwiseOr";
    case 361: return "OpGroupNonUniformBitwiseXor";
    case 362: return "OpGroupNonUniformLogicalAnd";
    case 363: return "OpGroupNonUniformLogicalOr";
    case 364: return "OpGroupNonUniformLogicalXor";
    case 365: return "OpGroupNonUniformQuadBroadcast";
    case 366: return "OpGroupNonUniformQuadSwap";

    case (int)Op::OpTerminateInvocation: return "OpTerminateInvocation";

    case 4421: return "OpSubgroupBallotKHR";
    case 4422: return "OpSubgroupFirstInvocationKHR";
    case 4428: return "OpSubgroupAllKHR";
    case 4429: return "OpSubgroupAnyKHR";
    case 4430: return "OpSubgroupAllEqualKHR";
    case 4432: return "OpSubgroupReadInvocationKHR";
    case 4433: return "OpExtInstWithForwardRefsKHR";

    case (int)Op::OpGroupNonUniformQuadAllKHR: return "OpGroupNonUniformQuadAllKHR";
    case (int)Op::OpGroupNonUniformQuadAnyKHR: return "OpGroupNonUniformQuadAnyKHR";

    case (int)Op::OpAtomicFAddEXT: return "OpAtomicFAddEXT";
    case (int)Op::OpAtomicFMinEXT: return "OpAtomicFMinEXT";
    case (int)Op::OpAtomicFMaxEXT: return "OpAtomicFMaxEXT";

    case (int)Op::OpAssumeTrueKHR: return "OpAssumeTrueKHR";
    case (int)Op::OpExpectKHR: return "OpExpectKHR";

    case 5000: return "OpGroupIAddNonUniformAMD";
    case 5001: return "OpGroupFAddNonUniformAMD";
    case 5002: return "OpGroupFMinNonUniformAMD";
    case 5003: return "OpGroupUMinNonUniformAMD";
    case 5004: return "OpGroupSMinNonUniformAMD";
    case 5005: return "OpGroupFMaxNonUniformAMD";
    case 5006: return "OpGroupUMaxNonUniformAMD";
    case 5007: return "OpGroupSMaxNonUniformAMD";

    case 5011: return "OpFragmentMaskFetchAMD";
    case 5012: return "OpFragmentFetchAMD";

    case (int)Op::OpReadClockKHR:               return "OpReadClockKHR";

    case (int)Op::OpDecorateStringGOOGLE:       return "OpDecorateStringGOOGLE";
    case (int)Op::OpMemberDecorateStringGOOGLE: return "OpMemberDecorateStringGOOGLE";

    case (int)Op::OpReportIntersectionKHR:             return "OpReportIntersectionKHR";
    case (int)Op::OpIgnoreIntersectionNV:              return "OpIgnoreIntersectionNV";
    case (int)Op::OpIgnoreIntersectionKHR:             return "OpIgnoreIntersectionKHR";
    case (int)Op::OpTerminateRayNV:                    return "OpTerminateRayNV";
    case (int)Op::OpTerminateRayKHR:                   return "OpTerminateRayKHR";
    case (int)Op::OpTraceNV:                           return "OpTraceNV";
    case (int)Op::OpTraceRayMotionNV:                  return "OpTraceRayMotionNV";
    case (int)Op::OpTraceRayKHR:                       return "OpTraceRayKHR";
    case (int)Op::OpTypeAccelerationStructureKHR:      return "OpTypeAccelerationStructureKHR";
    case (int)Op::OpExecuteCallableNV:                 return "OpExecuteCallableNV";
    case (int)Op::OpExecuteCallableKHR:                return "OpExecuteCallableKHR";
    case (int)Op::OpConvertUToAccelerationStructureKHR: return "OpConvertUToAccelerationStructureKHR";

    case (int)Op::OpGroupNonUniformPartitionNV:       return "OpGroupNonUniformPartitionNV";
    case (int)Op::OpImageSampleFootprintNV:           return "OpImageSampleFootprintNV";
    case (int)Op::OpWritePackedPrimitiveIndices4x8NV: return "OpWritePackedPrimitiveIndices4x8NV";
    case (int)Op::OpEmitMeshTasksEXT:                 return "OpEmitMeshTasksEXT";
    case (int)Op::OpSetMeshOutputsEXT:                return "OpSetMeshOutputsEXT";

    case (int)Op::OpGroupNonUniformRotateKHR:         return "OpGroupNonUniformRotateKHR";

    case (int)Op::OpTypeRayQueryKHR:                                                   return "OpTypeRayQueryKHR";
    case (int)Op::OpRayQueryInitializeKHR:                                             return "OpRayQueryInitializeKHR";
    case (int)Op::OpRayQueryTerminateKHR:                                              return "OpRayQueryTerminateKHR";
    case (int)Op::OpRayQueryGenerateIntersectionKHR:                                   return "OpRayQueryGenerateIntersectionKHR";
    case (int)Op::OpRayQueryConfirmIntersectionKHR:                                    return "OpRayQueryConfirmIntersectionKHR";
    case (int)Op::OpRayQueryProceedKHR:                                                return "OpRayQueryProceedKHR";
    case (int)Op::OpRayQueryGetIntersectionTypeKHR:                                    return "OpRayQueryGetIntersectionTypeKHR";
    case (int)Op::OpRayQueryGetRayTMinKHR:                                             return "OpRayQueryGetRayTMinKHR";
    case (int)Op::OpRayQueryGetRayFlagsKHR:                                            return "OpRayQueryGetRayFlagsKHR";
    case (int)Op::OpRayQueryGetIntersectionTKHR:                                       return "OpRayQueryGetIntersectionTKHR";
    case (int)Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR:                     return "OpRayQueryGetIntersectionInstanceCustomIndexKHR";
    case (int)Op::OpRayQueryGetIntersectionInstanceIdKHR:                              return "OpRayQueryGetIntersectionInstanceIdKHR";
    case (int)Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR:  return "OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR";
    case (int)Op::OpRayQueryGetIntersectionGeometryIndexKHR:                           return "OpRayQueryGetIntersectionGeometryIndexKHR";
    case (int)Op::OpRayQueryGetIntersectionPrimitiveIndexKHR:                          return "OpRayQueryGetIntersectionPrimitiveIndexKHR";
    case (int)Op::OpRayQueryGetIntersectionBarycentricsKHR:                            return "OpRayQueryGetIntersectionBarycentricsKHR";
    case (int)Op::OpRayQueryGetIntersectionFrontFaceKHR:                               return "OpRayQueryGetIntersectionFrontFaceKHR";
    case (int)Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR:                     return "OpRayQueryGetIntersectionCandidateAABBOpaqueKHR";
    case (int)Op::OpRayQueryGetIntersectionObjectRayDirectionKHR:                      return "OpRayQueryGetIntersectionObjectRayDirectionKHR";
    case (int)Op::OpRayQueryGetIntersectionObjectRayOriginKHR:                         return "OpRayQueryGetIntersectionObjectRayOriginKHR";
    case (int)Op::OpRayQueryGetWorldRayDirectionKHR:                                   return "OpRayQueryGetWorldRayDirectionKHR";
    case (int)Op::OpRayQueryGetWorldRayOriginKHR:                                      return "OpRayQueryGetWorldRayOriginKHR";
    case (int)Op::OpRayQueryGetIntersectionObjectToWorldKHR:                           return "OpRayQueryGetIntersectionObjectToWorldKHR";
    case (int)Op::OpRayQueryGetIntersectionWorldToObjectKHR:                           return "OpRayQueryGetIntersectionWorldToObjectKHR";
    case (int)Op::OpRayQueryGetIntersectionTriangleVertexPositionsKHR:                 return "OpRayQueryGetIntersectionTriangleVertexPositionsKHR";
    case (int)Op::OpRayQueryGetClusterIdNV:                                            return "OpRayQueryGetIntersectionClusterIdNV";

    case (int)Op::OpRayQueryGetIntersectionSpherePositionNV:                           return "OpRayQueryGetIntersectionSpherePositionNV";
    case (int)Op::OpRayQueryGetIntersectionSphereRadiusNV:                             return "OpRayQueryGetIntersectionSphereRadiusNV";
    case (int)Op::OpRayQueryGetIntersectionLSSHitValueNV:                              return "OpRayQueryGetIntersectionLSSHitValueNV";
    case (int)Op::OpRayQueryGetIntersectionLSSPositionsNV:                             return "OpRayQueryGetIntersectionLSSPositionsNV";
    case (int)Op::OpRayQueryGetIntersectionLSSRadiiNV:                                 return "OpRayQueryGetIntersectionLSSRadiiNV";
    case (int)Op::OpRayQueryIsSphereHitNV:                                             return "OpRayQueryIsSphereHitNV";
    case (int)Op::OpRayQueryIsLSSHitNV:                                                return "OpRayQueryIsLSSHitNV";

    case (int)Op::OpTypeCooperativeMatrixNV:         return "OpTypeCooperativeMatrixNV";
    case (int)Op::OpCooperativeMatrixLoadNV:         return "OpCooperativeMatrixLoadNV";
    case (int)Op::OpCooperativeMatrixStoreNV:        return "OpCooperativeMatrixStoreNV";
    case (int)Op::OpCooperativeMatrixMulAddNV:       return "OpCooperativeMatrixMulAddNV";
    case (int)Op::OpCooperativeMatrixLengthNV:       return "OpCooperativeMatrixLengthNV";
    case (int)Op::OpTypeCooperativeMatrixKHR:        return "OpTypeCooperativeMatrixKHR";
    case (int)Op::OpCooperativeMatrixLoadKHR:        return "OpCooperativeMatrixLoadKHR";
    case (int)Op::OpCooperativeMatrixStoreKHR:       return "OpCooperativeMatrixStoreKHR";
    case (int)Op::OpCooperativeMatrixMulAddKHR:      return "OpCooperativeMatrixMulAddKHR";
    case (int)Op::OpCooperativeMatrixLengthKHR:      return "OpCooperativeMatrixLengthKHR";
    case (int)Op::OpDemoteToHelperInvocationEXT:     return "OpDemoteToHelperInvocationEXT";
    case (int)Op::OpIsHelperInvocationEXT:           return "OpIsHelperInvocationEXT";

    case (int)Op::OpCooperativeMatrixConvertNV:      return "OpCooperativeMatrixConvertNV";
    case (int)Op::OpCooperativeMatrixTransposeNV:    return "OpCooperativeMatrixTransposeNV";
    case (int)Op::OpCooperativeMatrixReduceNV:       return "OpCooperativeMatrixReduceNV";
    case (int)Op::OpCooperativeMatrixLoadTensorNV:   return "OpCooperativeMatrixLoadTensorNV";
    case (int)Op::OpCooperativeMatrixStoreTensorNV:  return "OpCooperativeMatrixStoreTensorNV";
    case (int)Op::OpCooperativeMatrixPerElementOpNV: return "OpCooperativeMatrixPerElementOpNV";
    case (int)Op::OpTypeTensorLayoutNV:              return "OpTypeTensorLayoutNV";
    case (int)Op::OpTypeTensorViewNV:                return "OpTypeTensorViewNV";
    case (int)Op::OpCreateTensorLayoutNV:            return "OpCreateTensorLayoutNV";
    case (int)Op::OpTensorLayoutSetBlockSizeNV:      return "OpTensorLayoutSetBlockSizeNV";
    case (int)Op::OpTensorLayoutSetDimensionNV:      return "OpTensorLayoutSetDimensionNV";
    case (int)Op::OpTensorLayoutSetStrideNV:         return "OpTensorLayoutSetStrideNV";
    case (int)Op::OpTensorLayoutSliceNV:             return "OpTensorLayoutSliceNV";
    case (int)Op::OpTensorLayoutSetClampValueNV:     return "OpTensorLayoutSetClampValueNV";
    case (int)Op::OpCreateTensorViewNV:              return "OpCreateTensorViewNV";
    case (int)Op::OpTensorViewSetDimensionNV:        return "OpTensorViewSetDimensionNV";
    case (int)Op::OpTensorViewSetStrideNV:           return "OpTensorViewSetStrideNV";
    case (int)Op::OpTensorViewSetClipNV:             return "OpTensorViewSetClipNV";

    case (int)Op::OpTypeTensorARM:                   return "OpTypeTensorARM";
    case (int)Op::OpTensorReadARM:                   return "OpTensorReadARM";
    case (int)Op::OpTensorWriteARM:                  return "OpTensorWriteARM";
    case (int)Op::OpTensorQuerySizeARM:              return "OpTensorQuerySizeARM";

    case (int)Op::OpTypeCooperativeVectorNV:         return "OpTypeCooperativeVectorNV";
    case (int)Op::OpCooperativeVectorMatrixMulNV:    return "OpCooperativeVectorMatrixMulNV";
    case (int)Op::OpCooperativeVectorMatrixMulAddNV: return "OpCooperativeVectorMatrixMulAddNV";
    case (int)Op::OpCooperativeVectorLoadNV:         return "OpCooperativeVectorLoadNV";
    case (int)Op::OpCooperativeVectorStoreNV:        return "OpCooperativeVectorStoreNV";
    case (int)Op::OpCooperativeVectorOuterProductAccumulateNV:   return "OpCooperativeVectorOuterProductAccumulateNV";
    case (int)Op::OpCooperativeVectorReduceSumAccumulateNV:      return "OpCooperativeVectorReduceSumAccumulateNV";

    case (int)Op::OpBeginInvocationInterlockEXT:     return "OpBeginInvocationInterlockEXT";
    case (int)Op::OpEndInvocationInterlockEXT:       return "OpEndInvocationInterlockEXT";

    case (int)Op::OpTypeHitObjectNV:                     return "OpTypeHitObjectNV";
    case (int)Op::OpHitObjectTraceRayNV:                 return "OpHitObjectTraceRayNV";
    case (int)Op::OpHitObjectTraceRayMotionNV:           return "OpHitObjectTraceRayMotionNV";
    case (int)Op::OpHitObjectRecordHitNV:                return "OpHitObjectRecordHitNV";
    case (int)Op::OpHitObjectRecordHitMotionNV:          return "OpHitObjectRecordHitMotionNV";
    case (int)Op::OpHitObjectRecordHitWithIndexNV:       return "OpHitObjectRecordHitWithIndexNV";
    case (int)Op::OpHitObjectRecordHitWithIndexMotionNV: return "OpHitObjectRecordHitWithIndexMotionNV";
    case (int)Op::OpHitObjectRecordMissNV:               return "OpHitObjectRecordMissNV";
    case (int)Op::OpHitObjectRecordMissMotionNV:         return "OpHitObjectRecordMissMotionNV";
    case (int)Op::OpHitObjectRecordEmptyNV:              return "OpHitObjectRecordEmptyNV";
    case (int)Op::OpHitObjectExecuteShaderNV:            return "OpHitObjectExecuteShaderNV";
    case (int)Op::OpReorderThreadWithHintNV:             return "OpReorderThreadWithHintNV";
    case (int)Op::OpReorderThreadWithHitObjectNV:        return "OpReorderThreadWithHitObjectNV";
    case (int)Op::OpHitObjectGetCurrentTimeNV:           return "OpHitObjectGetCurrentTimeNV";
    case (int)Op::OpHitObjectGetAttributesNV:            return "OpHitObjectGetAttributesNV";
    case (int)Op::OpHitObjectGetHitKindNV:               return "OpHitObjectGetFrontFaceNV";
    case (int)Op::OpHitObjectGetPrimitiveIndexNV:        return "OpHitObjectGetPrimitiveIndexNV";
    case (int)Op::OpHitObjectGetGeometryIndexNV:         return "OpHitObjectGetGeometryIndexNV";
    case (int)Op::OpHitObjectGetInstanceIdNV:            return "OpHitObjectGetInstanceIdNV";
    case (int)Op::OpHitObjectGetInstanceCustomIndexNV:   return "OpHitObjectGetInstanceCustomIndexNV";
    case (int)Op::OpHitObjectGetObjectRayDirectionNV:    return "OpHitObjectGetObjectRayDirectionNV";
    case (int)Op::OpHitObjectGetObjectRayOriginNV:       return "OpHitObjectGetObjectRayOriginNV";
    case (int)Op::OpHitObjectGetWorldRayDirectionNV:     return "OpHitObjectGetWorldRayDirectionNV";
    case (int)Op::OpHitObjectGetWorldRayOriginNV:        return "OpHitObjectGetWorldRayOriginNV";
    case (int)Op::OpHitObjectGetWorldToObjectNV:         return "OpHitObjectGetWorldToObjectNV";
    case (int)Op::OpHitObjectGetObjectToWorldNV:         return "OpHitObjectGetObjectToWorldNV";
    case (int)Op::OpHitObjectGetRayTMaxNV:               return "OpHitObjectGetRayTMaxNV";
    case (int)Op::OpHitObjectGetRayTMinNV:               return "OpHitObjectGetRayTMinNV";
    case (int)Op::OpHitObjectIsEmptyNV:                  return "OpHitObjectIsEmptyNV";
    case (int)Op::OpHitObjectIsHitNV:                    return "OpHitObjectIsHitNV";
    case (int)Op::OpHitObjectIsMissNV:                   return "OpHitObjectIsMissNV";
    case (int)Op::OpHitObjectGetShaderBindingTableRecordIndexNV: return "OpHitObjectGetShaderBindingTableRecordIndexNV";
    case (int)Op::OpHitObjectGetShaderRecordBufferHandleNV:   return "OpHitObjectGetShaderRecordBufferHandleNV";
    case (int)Op::OpHitObjectGetClusterIdNV:             return "OpHitObjectGetClusterIdNV";
    case (int)Op::OpHitObjectGetSpherePositionNV:        return "OpHitObjectGetSpherePositionNV";
    case (int)Op::OpHitObjectGetSphereRadiusNV:          return "OpHitObjectGetSphereRadiusNV";
    case (int)Op::OpHitObjectGetLSSPositionsNV:          return "OpHitObjectGetLSSPositionsNV";
    case (int)Op::OpHitObjectGetLSSRadiiNV:              return "OpHitObjectGetLSSRadiiNV";
    case (int)Op::OpHitObjectIsSphereHitNV:              return "OpHitObjectIsSphereHitNV";
    case (int)Op::OpHitObjectIsLSSHitNV:                 return "OpHitObjectIsLSSHitNV";

    case (int)Op::OpFetchMicroTriangleVertexBarycentricNV:       return "OpFetchMicroTriangleVertexBarycentricNV";
    case (int)Op::OpFetchMicroTriangleVertexPositionNV:    return "OpFetchMicroTriangleVertexPositionNV";

    case (int)Op::OpColorAttachmentReadEXT:          return "OpColorAttachmentReadEXT";
    case (int)Op::OpDepthAttachmentReadEXT:          return "OpDepthAttachmentReadEXT";
    case (int)Op::OpStencilAttachmentReadEXT:        return "OpStencilAttachmentReadEXT";

    case (int)Op::OpImageSampleWeightedQCOM:         return "OpImageSampleWeightedQCOM";
    case (int)Op::OpImageBoxFilterQCOM:              return "OpImageBoxFilterQCOM";
    case (int)Op::OpImageBlockMatchSADQCOM:          return "OpImageBlockMatchSADQCOM";
    case (int)Op::OpImageBlockMatchSSDQCOM:          return "OpImageBlockMatchSSDQCOM";
    case (int)Op::OpImageBlockMatchWindowSSDQCOM:    return "OpImageBlockMatchWindowSSDQCOM";
    case (int)Op::OpImageBlockMatchWindowSADQCOM:    return "OpImageBlockMatchWindowSADQCOM";
    case (int)Op::OpImageBlockMatchGatherSSDQCOM:    return "OpImageBlockMatchGatherSSDQCOM";
    case (int)Op::OpImageBlockMatchGatherSADQCOM:    return "OpImageBlockMatchGatherSADQCOM";

    case (int)Op::OpBitCastArrayQCOM:                return "OpBitCastArrayQCOM";
    case (int)Op::OpCompositeConstructCoopMatQCOM:   return "OpCompositeConstructCoopMatQCOM";
    case (int)Op::OpCompositeExtractCoopMatQCOM:     return "OpCompositeExtractCoopMatQCOM";
    case (int)Op::OpExtractSubArrayQCOM:             return "OpExtractSubArrayQCOM";

    case (int)Op::OpConstantCompositeReplicateEXT: return "OpConstantCompositeReplicateEXT";
    case (int)Op::OpSpecConstantCompositeReplicateEXT: return "OpSpecConstantCompositeReplicateEXT";
    case (int)Op::OpCompositeConstructReplicateEXT: return "OpCompositeConstructReplicateEXT";

    case (int)Op::OpSDotKHR: return "OpSDotKHR";
    case (int)Op::OpUDotKHR: return "OpUDotKHR";
    case (int)Op::OpSUDotKHR: return "OpSUDotKHR";
    case (int)Op::OpSDotAccSatKHR: return "OpSDotAccSatKHR";
    case (int)Op::OpUDotAccSatKHR: return "OpUDotAccSatKHR";
    case (int)Op::OpSUDotAccSatKHR: return "OpSUDotAccSatKHR";

    case (int)Op::OpTypeHitObjectEXT:                     return "OpTypeHitObjectEXT";
    case (int)Op::OpHitObjectTraceRayEXT:                 return "OpHitObjectTraceRayEXT";
    case (int)Op::OpHitObjectTraceRayMotionEXT:           return "OpHitObjectTraceRayMotionEXT";
    case (int)Op::OpHitObjectRecordMissEXT:               return "OpHitObjectRecordMissEXT";
    case (int)Op::OpHitObjectRecordMissMotionEXT:         return "OpHitObjectRecordMissMotionEXT";
    case (int)Op::OpHitObjectRecordEmptyEXT:              return "OpHitObjectRecordEmptyEXT";
    case (int)Op::OpHitObjectExecuteShaderEXT:            return "OpHitObjectExecuteShaderEXT";
    case (int)Op::OpReorderThreadWithHintEXT:             return "OpReorderThreadWithHintEXT";
    case (int)Op::OpReorderThreadWithHitObjectEXT:        return "OpReorderThreadWithHitObjectEXT";
    case (int)Op::OpHitObjectGetCurrentTimeEXT:           return "OpHitObjectGetCurrentTimeEXT";
    case (int)Op::OpHitObjectGetAttributesEXT:            return "OpHitObjectGetAttributesEXT";
    case (int)Op::OpHitObjectGetHitKindEXT:               return "OpHitObjectGetHitKindEXT";
    case (int)Op::OpHitObjectGetPrimitiveIndexEXT:        return "OpHitObjectGetPrimitiveIndexEXT";
    case (int)Op::OpHitObjectGetGeometryIndexEXT:         return "OpHitObjectGetGeometryIndexEXT";
    case (int)Op::OpHitObjectGetInstanceIdEXT:            return "OpHitObjectGetInstanceIdEXT";
    case (int)Op::OpHitObjectGetInstanceCustomIndexEXT:   return "OpHitObjectGetInstanceCustomIndexEXT";
    case (int)Op::OpHitObjectGetObjectRayDirectionEXT:    return "OpHitObjectGetObjectRayDirectionEXT";
    case (int)Op::OpHitObjectGetObjectRayOriginEXT:       return "OpHitObjectGetObjectRayOriginEXT";
    case (int)Op::OpHitObjectGetWorldRayDirectionEXT:     return "OpHitObjectGetWorldRayDirectionEXT";
    case (int)Op::OpHitObjectGetWorldRayOriginEXT:        return "OpHitObjectGetWorldRayOriginEXT";
    case (int)Op::OpHitObjectGetWorldToObjectEXT:         return "OpHitObjectGetWorldToObjectEXT";
    case (int)Op::OpHitObjectGetObjectToWorldEXT:         return "OpHitObjectGetObjectToWorldEXT";
    case (int)Op::OpHitObjectGetRayTMaxEXT:               return "OpHitObjectGetRayTMaxEXT";
    case (int)Op::OpHitObjectGetRayTMinEXT:               return "OpHitObjectGetRayTMinEXT";
    case (int)Op::OpHitObjectGetRayFlagsEXT:              return "OpHitObjectGetRayFlagsEXT";
    case (int)Op::OpHitObjectIsEmptyEXT:                  return "OpHitObjectIsEmptyEXT";
    case (int)Op::OpHitObjectIsHitEXT:                    return "OpHitObjectIsHitEXT";
    case (int)Op::OpHitObjectIsMissEXT:                   return "OpHitObjectIsMissEXT";
    case (int)Op::OpHitObjectGetShaderBindingTableRecordIndexEXT: return "OpHitObjectGetShaderBindingTableRecordIndexEXT";
    case (int)Op::OpHitObjectGetShaderRecordBufferHandleEXT:   return "OpHitObjectGetShaderRecordBufferHandleEXT";
    case (int)Op::OpHitObjectSetShaderBindingTableRecordIndexEXT: return "OpHitObjectSetShaderBindingTableRecordIndexEXT";
    case (int)Op::OpHitObjectReorderExecuteShaderEXT:     return "OpHitObjectReorderExecuteEXT";
    case (int)Op::OpHitObjectTraceReorderExecuteEXT:      return "OpHitObjectTraceReorderExecuteEXT";
    case (int)Op::OpHitObjectTraceMotionReorderExecuteEXT: return "OpHitObjectTraceMotionReorderExecuteEXT";
    case (int)Op::OpHitObjectRecordFromQueryEXT:          return "OpHitObjectRecordFromQueryEXT";
    case (int)Op::OpHitObjectGetIntersectionTriangleVertexPositionsEXT: return "OpHitObjectGetIntersectionTriangleVertexPositionsEXT";

    default:
        return "Bad";
    }
}

// The set of objects that hold all the instruction/operand
// parameterization information.
InstructionParameters InstructionDesc[OpCodeMask + 1];
OperandParameters ExecutionModeOperands[ExecutionModeCeiling];
OperandParameters DecorationOperands[DecorationCeiling];

EnumDefinition OperandClassParams[OperandCount];
EnumParameters ExecutionModeParams[ExecutionModeCeiling];
EnumParameters ImageOperandsParams[ImageOperandsCeiling];
EnumParameters DecorationParams[DecorationCeiling];
EnumParameters LoopControlParams[FunctionControlCeiling];
EnumParameters SelectionControlParams[SelectControlCeiling];
EnumParameters FunctionControlParams[FunctionControlCeiling];
EnumParameters MemoryAccessParams[MemoryAccessCeiling];
EnumParameters CooperativeMatrixOperandsParams[CooperativeMatrixOperandsCeiling];
EnumParameters TensorAddressingOperandsParams[TensorAddressingOperandsCeiling];

// Set up all the parameterizing descriptions of the opcodes, operands, etc.
void Parameterize()
{
    // only do this once.
    static std::once_flag initialized;
    std::call_once(initialized, [](){

        // Exceptions to having a result <id> and a resulting type <id>.
        // (Everything is initialized to have both).

        InstructionDesc[enumCast(Op::OpNop)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpSource)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpSourceContinued)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpSourceExtension)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpExtension)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpExtInstImport)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpCapability)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpMemoryModel)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpEntryPoint)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpExecutionMode)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpExecutionModeId)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpTypeVoid)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeBool)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeInt)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeFloat)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeVector)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeMatrix)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeImage)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeSampler)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeSampledImage)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeArray)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeRuntimeArray)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeStruct)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeOpaque)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypePointer)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeForwardPointer)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpTypeFunction)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeEvent)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeDeviceEvent)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeReserveId)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeQueue)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypePipe)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpFunctionEnd)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpStore)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpImageWrite)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpDecorationGroup)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpDecorate)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpDecorateId)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpDecorateStringGOOGLE)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpMemberDecorate)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpMemberDecorateStringGOOGLE)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpGroupDecorate)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpGroupMemberDecorate)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpName)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpMemberName)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpString)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpLine)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpNoLine)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpCopyMemory)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpCopyMemorySized)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpEmitVertex)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpEndPrimitive)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpEmitStreamVertex)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpEndStreamPrimitive)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpControlBarrier)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpMemoryBarrier)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpAtomicStore)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpLoopMerge)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpSelectionMerge)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpLabel)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpBranch)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpBranchConditional)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpSwitch)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpKill)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpTerminateInvocation)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpReturn)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpReturnValue)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpUnreachable)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpLifetimeStart)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpLifetimeStop)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpCommitReadPipe)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpCommitWritePipe)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpGroupCommitWritePipe)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpGroupCommitReadPipe)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpCaptureEventProfilingInfo)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpSetUserEventStatus)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpRetainEvent)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpReleaseEvent)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpGroupWaitEvents)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpAtomicFlagClear)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpModuleProcessed)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixNV)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixKHR)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpBeginInvocationInterlockEXT)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpEndInvocationInterlockEXT)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpAssumeTrueKHR)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpTypeTensorLayoutNV)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeTensorViewNV)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreTensorNV)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpTypeCooperativeVectorNV)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpCooperativeVectorStoreNV)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].setResultAndType(false, false);
        InstructionDesc[enumCast(Op::OpCooperativeVectorReduceSumAccumulateNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpTypeTensorARM)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTensorReadARM)].setResultAndType(true, true);
        InstructionDesc[enumCast(Op::OpTensorWriteARM)].setResultAndType(false, false);

        // Specific additional context-dependent operands

        ExecutionModeOperands[enumCast(ExecutionMode::Invocations)].push(OperandLiteralNumber, "'Number of <<Invocation,invocations>>'");

        ExecutionModeOperands[enumCast(ExecutionMode::LocalSize)].push(OperandLiteralNumber, "'x size'");
        ExecutionModeOperands[enumCast(ExecutionMode::LocalSize)].push(OperandLiteralNumber, "'y size'");
        ExecutionModeOperands[enumCast(ExecutionMode::LocalSize)].push(OperandLiteralNumber, "'z size'");

        ExecutionModeOperands[enumCast(ExecutionMode::LocalSizeHint)].push(OperandLiteralNumber, "'x size'");
        ExecutionModeOperands[enumCast(ExecutionMode::LocalSizeHint)].push(OperandLiteralNumber, "'y size'");
        ExecutionModeOperands[enumCast(ExecutionMode::LocalSizeHint)].push(OperandLiteralNumber, "'z size'");

        ExecutionModeOperands[enumCast(ExecutionMode::OutputVertices)].push(OperandLiteralNumber, "'Vertex count'");
        ExecutionModeOperands[enumCast(ExecutionMode::VecTypeHint)].push(OperandLiteralNumber, "'Vector type'");

        DecorationOperands[enumCast(Decoration::Stream)].push(OperandLiteralNumber, "'Stream Number'");
        DecorationOperands[enumCast(Decoration::Location)].push(OperandLiteralNumber, "'Location'");
        DecorationOperands[enumCast(Decoration::Component)].push(OperandLiteralNumber, "'Component'");
        DecorationOperands[enumCast(Decoration::Index)].push(OperandLiteralNumber, "'Index'");
        DecorationOperands[enumCast(Decoration::Binding)].push(OperandLiteralNumber, "'Binding Point'");
        DecorationOperands[enumCast(Decoration::DescriptorSet)].push(OperandLiteralNumber, "'Descriptor Set'");
        DecorationOperands[enumCast(Decoration::Offset)].push(OperandLiteralNumber, "'Byte Offset'");
        DecorationOperands[enumCast(Decoration::XfbBuffer)].push(OperandLiteralNumber, "'XFB Buffer Number'");
        DecorationOperands[enumCast(Decoration::XfbStride)].push(OperandLiteralNumber, "'XFB Stride'");
        DecorationOperands[enumCast(Decoration::ArrayStride)].push(OperandLiteralNumber, "'Array Stride'");
        DecorationOperands[enumCast(Decoration::MatrixStride)].push(OperandLiteralNumber, "'Matrix Stride'");
        DecorationOperands[enumCast(Decoration::BuiltIn)].push(OperandLiteralNumber, "See <<BuiltIn,*BuiltIn*>>");
        DecorationOperands[enumCast(Decoration::FPRoundingMode)].push(OperandFPRoundingMode, "'Floating-Point Rounding Mode'");
        DecorationOperands[enumCast(Decoration::FPFastMathMode)].push(OperandFPFastMath, "'Fast-Math Mode'");
        DecorationOperands[enumCast(Decoration::LinkageAttributes)].push(OperandLiteralString, "'Name'");
        DecorationOperands[enumCast(Decoration::LinkageAttributes)].push(OperandLinkageType, "'Linkage Type'");
        DecorationOperands[enumCast(Decoration::FuncParamAttr)].push(OperandFuncParamAttr, "'Function Parameter Attribute'");
        DecorationOperands[enumCast(Decoration::SpecId)].push(OperandLiteralNumber, "'Specialization Constant ID'");
        DecorationOperands[enumCast(Decoration::InputAttachmentIndex)].push(OperandLiteralNumber, "'Attachment Index'");
        DecorationOperands[enumCast(Decoration::Alignment)].push(OperandLiteralNumber, "'Alignment'");

        OperandClassParams[OperandSource].set(0, SourceString, nullptr);
        OperandClassParams[OperandExecutionModel].set(0, ExecutionModelString, nullptr);
        OperandClassParams[OperandAddressing].set(0, AddressingString, nullptr);
        OperandClassParams[OperandMemory].set(0, MemoryString, nullptr);
        OperandClassParams[OperandExecutionMode].set(ExecutionModeCeiling, ExecutionModeString, ExecutionModeParams);
        OperandClassParams[OperandExecutionMode].setOperands(ExecutionModeOperands);
        OperandClassParams[OperandStorage].set(0, StorageClassString, nullptr);
        OperandClassParams[OperandDimensionality].set(0, DimensionString, nullptr);
        OperandClassParams[OperandSamplerAddressingMode].set(0, SamplerAddressingModeString, nullptr);
        OperandClassParams[OperandSamplerFilterMode].set(0, SamplerFilterModeString, nullptr);
        OperandClassParams[OperandSamplerImageFormat].set(0, ImageFormatString, nullptr);
        OperandClassParams[OperandImageChannelOrder].set(0, ImageChannelOrderString, nullptr);
        OperandClassParams[OperandImageChannelDataType].set(0, ImageChannelDataTypeString, nullptr);
        OperandClassParams[OperandImageOperands].set(ImageOperandsCeiling, ImageOperandsString, ImageOperandsParams, true);
        OperandClassParams[OperandFPFastMath].set(0, FPFastMathString, nullptr, true);
        OperandClassParams[OperandFPRoundingMode].set(0, FPRoundingModeString, nullptr);
        OperandClassParams[OperandLinkageType].set(0, LinkageTypeString, nullptr);
        OperandClassParams[OperandFuncParamAttr].set(0, FuncParamAttrString, nullptr);
        OperandClassParams[OperandAccessQualifier].set(0, AccessQualifierString, nullptr);
        OperandClassParams[OperandDecoration].set(DecorationCeiling, DecorationString, DecorationParams);
        OperandClassParams[OperandDecoration].setOperands(DecorationOperands);
        OperandClassParams[OperandBuiltIn].set(0, BuiltInString, nullptr);
        OperandClassParams[OperandSelect].set(SelectControlCeiling, SelectControlString, SelectionControlParams, true);
        OperandClassParams[OperandLoop].set(LoopControlCeiling, LoopControlString, LoopControlParams, true);
        OperandClassParams[OperandFunction].set(FunctionControlCeiling, FunctionControlString, FunctionControlParams, true);
        OperandClassParams[OperandMemorySemantics].set(0, MemorySemanticsString, nullptr, true);
        OperandClassParams[OperandMemoryAccess].set(MemoryAccessCeiling, MemoryAccessString, MemoryAccessParams, true);
        OperandClassParams[OperandScope].set(0, ScopeString, nullptr);
        OperandClassParams[OperandGroupOperation].set(0, GroupOperationString, nullptr);
        OperandClassParams[OperandKernelEnqueueFlags].set(0, KernelEnqueueFlagsString, nullptr);
        OperandClassParams[OperandKernelProfilingInfo].set(0, KernelProfilingInfoString, nullptr, true);
        OperandClassParams[OperandCapability].set(0, CapabilityString, nullptr);
        OperandClassParams[OperandCooperativeMatrixOperands].set(CooperativeMatrixOperandsCeiling, CooperativeMatrixOperandsString, CooperativeMatrixOperandsParams, true);
        OperandClassParams[OperandTensorAddressingOperands].set(TensorAddressingOperandsCeiling, TensorAddressingOperandsString, TensorAddressingOperandsParams, true);
        OperandClassParams[OperandOpcode].set(OpCodeMask + 1, OpcodeString, nullptr);

        // set name of operator, an initial set of <id> style operands, and the description

        InstructionDesc[enumCast(Op::OpSource)].operands.push(OperandSource, "");
        InstructionDesc[enumCast(Op::OpSource)].operands.push(OperandLiteralNumber, "'Version'");
        InstructionDesc[enumCast(Op::OpSource)].operands.push(OperandId, "'File'", true);
        InstructionDesc[enumCast(Op::OpSource)].operands.push(OperandLiteralString, "'Source'", true);

        InstructionDesc[enumCast(Op::OpSourceContinued)].operands.push(OperandLiteralString, "'Continued Source'");

        InstructionDesc[enumCast(Op::OpSourceExtension)].operands.push(OperandLiteralString, "'Extension'");

        InstructionDesc[enumCast(Op::OpName)].operands.push(OperandId, "'Target'");
        InstructionDesc[enumCast(Op::OpName)].operands.push(OperandLiteralString, "'Name'");

        InstructionDesc[enumCast(Op::OpMemberName)].operands.push(OperandId, "'Type'");
        InstructionDesc[enumCast(Op::OpMemberName)].operands.push(OperandLiteralNumber, "'Member'");
        InstructionDesc[enumCast(Op::OpMemberName)].operands.push(OperandLiteralString, "'Name'");

        InstructionDesc[enumCast(Op::OpString)].operands.push(OperandLiteralString, "'String'");

        InstructionDesc[enumCast(Op::OpLine)].operands.push(OperandId, "'File'");
        InstructionDesc[enumCast(Op::OpLine)].operands.push(OperandLiteralNumber, "'Line'");
        InstructionDesc[enumCast(Op::OpLine)].operands.push(OperandLiteralNumber, "'Column'");

        InstructionDesc[enumCast(Op::OpExtension)].operands.push(OperandLiteralString, "'Name'");

        InstructionDesc[enumCast(Op::OpExtInstImport)].operands.push(OperandLiteralString, "'Name'");

        InstructionDesc[enumCast(Op::OpCapability)].operands.push(OperandCapability, "'Capability'");

        InstructionDesc[enumCast(Op::OpMemoryModel)].operands.push(OperandAddressing, "");
        InstructionDesc[enumCast(Op::OpMemoryModel)].operands.push(OperandMemory, "");

        InstructionDesc[enumCast(Op::OpEntryPoint)].operands.push(OperandExecutionModel, "");
        InstructionDesc[enumCast(Op::OpEntryPoint)].operands.push(OperandId, "'Entry Point'");
        InstructionDesc[enumCast(Op::OpEntryPoint)].operands.push(OperandLiteralString, "'Name'");
        InstructionDesc[enumCast(Op::OpEntryPoint)].operands.push(OperandVariableIds, "'Interface'");

        InstructionDesc[enumCast(Op::OpExecutionMode)].operands.push(OperandId, "'Entry Point'");
        InstructionDesc[enumCast(Op::OpExecutionMode)].operands.push(OperandExecutionMode, "'Mode'");
        InstructionDesc[enumCast(Op::OpExecutionMode)].operands.push(OperandOptionalLiteral, "See <<Execution_Mode,Execution Mode>>");

        InstructionDesc[enumCast(Op::OpExecutionModeId)].operands.push(OperandId, "'Entry Point'");
        InstructionDesc[enumCast(Op::OpExecutionModeId)].operands.push(OperandExecutionMode, "'Mode'");
        InstructionDesc[enumCast(Op::OpExecutionModeId)].operands.push(OperandVariableIds, "See <<Execution_Mode,Execution Mode>>");

        InstructionDesc[enumCast(Op::OpTypeInt)].operands.push(OperandLiteralNumber, "'Width'");
        InstructionDesc[enumCast(Op::OpTypeInt)].operands.push(OperandLiteralNumber, "'Signedness'");

        InstructionDesc[enumCast(Op::OpTypeFloat)].operands.push(OperandLiteralNumber, "'Width'");
        InstructionDesc[enumCast(Op::OpTypeFloat)].operands.push(OperandOptionalLiteral, "'FP Encoding'");

        InstructionDesc[enumCast(Op::OpTypeVector)].operands.push(OperandId, "'Component Type'");
        InstructionDesc[enumCast(Op::OpTypeVector)].operands.push(OperandLiteralNumber, "'Component Count'");

        InstructionDesc[enumCast(Op::OpTypeMatrix)].operands.push(OperandId, "'Column Type'");
        InstructionDesc[enumCast(Op::OpTypeMatrix)].operands.push(OperandLiteralNumber, "'Column Count'");

        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandId, "'Sampled Type'");
        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandDimensionality, "");
        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandLiteralNumber, "'Depth'");
        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandLiteralNumber, "'Arrayed'");
        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandLiteralNumber, "'MS'");
        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandLiteralNumber, "'Sampled'");
        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandSamplerImageFormat, "");
        InstructionDesc[enumCast(Op::OpTypeImage)].operands.push(OperandAccessQualifier, "", true);

        InstructionDesc[enumCast(Op::OpTypeSampledImage)].operands.push(OperandId, "'Image Type'");

        InstructionDesc[enumCast(Op::OpTypeArray)].operands.push(OperandId, "'Element Type'");
        InstructionDesc[enumCast(Op::OpTypeArray)].operands.push(OperandId, "'Length'");

        InstructionDesc[enumCast(Op::OpTypeRuntimeArray)].operands.push(OperandId, "'Element Type'");

        InstructionDesc[enumCast(Op::OpTypeStruct)].operands.push(OperandVariableIds, "'Member 0 type', +\n'member 1 type', +\n...");

        InstructionDesc[enumCast(Op::OpTypeOpaque)].operands.push(OperandLiteralString, "The name of the opaque type.");

        InstructionDesc[enumCast(Op::OpTypePointer)].operands.push(OperandStorage, "");
        InstructionDesc[enumCast(Op::OpTypePointer)].operands.push(OperandId, "'Type'");

        InstructionDesc[enumCast(Op::OpTypeForwardPointer)].operands.push(OperandId, "'Pointer Type'");
        InstructionDesc[enumCast(Op::OpTypeForwardPointer)].operands.push(OperandStorage, "");

        InstructionDesc[enumCast(Op::OpTypePipe)].operands.push(OperandAccessQualifier, "'Qualifier'");

        InstructionDesc[enumCast(Op::OpTypeFunction)].operands.push(OperandId, "'Return Type'");
        InstructionDesc[enumCast(Op::OpTypeFunction)].operands.push(OperandVariableIds, "'Parameter 0 Type', +\n'Parameter 1 Type', +\n...");

        InstructionDesc[enumCast(Op::OpConstant)].operands.push(OperandVariableLiterals, "'Value'");

        InstructionDesc[enumCast(Op::OpConstantComposite)].operands.push(OperandVariableIds, "'Constituents'");

        InstructionDesc[enumCast(Op::OpConstantSampler)].operands.push(OperandSamplerAddressingMode, "");
        InstructionDesc[enumCast(Op::OpConstantSampler)].operands.push(OperandLiteralNumber, "'Param'");
        InstructionDesc[enumCast(Op::OpConstantSampler)].operands.push(OperandSamplerFilterMode, "");

        InstructionDesc[enumCast(Op::OpSpecConstant)].operands.push(OperandVariableLiterals, "'Value'");

        InstructionDesc[enumCast(Op::OpSpecConstantComposite)].operands.push(OperandVariableIds, "'Constituents'");

        InstructionDesc[enumCast(Op::OpSpecConstantOp)].operands.push(OperandLiteralNumber, "'Opcode'");
        InstructionDesc[enumCast(Op::OpSpecConstantOp)].operands.push(OperandVariableIds, "'Operands'");

        InstructionDesc[enumCast(Op::OpVariable)].operands.push(OperandStorage, "");
        InstructionDesc[enumCast(Op::OpVariable)].operands.push(OperandId, "'Initializer'", true);

        InstructionDesc[enumCast(Op::OpFunction)].operands.push(OperandFunction, "");
        InstructionDesc[enumCast(Op::OpFunction)].operands.push(OperandId, "'Function Type'");

        InstructionDesc[enumCast(Op::OpFunctionCall)].operands.push(OperandId, "'Function'");
        InstructionDesc[enumCast(Op::OpFunctionCall)].operands.push(OperandVariableIds, "'Argument 0', +\n'Argument 1', +\n...");

        InstructionDesc[enumCast(Op::OpExtInst)].operands.push(OperandId, "'Set'");
        InstructionDesc[enumCast(Op::OpExtInst)].operands.push(OperandLiteralNumber, "'Instruction'");
        InstructionDesc[enumCast(Op::OpExtInst)].operands.push(OperandVariableIds, "'Operand 1', +\n'Operand 2', +\n...");

        InstructionDesc[enumCast(Op::OpExtInstWithForwardRefsKHR)].operands.push(OperandId, "'Set'");
        InstructionDesc[enumCast(Op::OpExtInstWithForwardRefsKHR)].operands.push(OperandLiteralNumber, "'Instruction'");
        InstructionDesc[enumCast(Op::OpExtInstWithForwardRefsKHR)].operands.push(OperandVariableIds, "'Operand 1', +\n'Operand 2', +\n...");

        InstructionDesc[enumCast(Op::OpLoad)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpLoad)].operands.push(OperandMemoryAccess, "", true);
        InstructionDesc[enumCast(Op::OpLoad)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpLoad)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpStore)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpStore)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpStore)].operands.push(OperandMemoryAccess, "", true);
        InstructionDesc[enumCast(Op::OpStore)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpStore)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpPhi)].operands.push(OperandVariableIds, "'Variable, Parent, ...'");

        InstructionDesc[enumCast(Op::OpDecorate)].operands.push(OperandId, "'Target'");
        InstructionDesc[enumCast(Op::OpDecorate)].operands.push(OperandDecoration, "");
        InstructionDesc[enumCast(Op::OpDecorate)].operands.push(OperandVariableLiterals, "See <<Decoration,'Decoration'>>.");

        InstructionDesc[enumCast(Op::OpDecorateId)].operands.push(OperandId, "'Target'");
        InstructionDesc[enumCast(Op::OpDecorateId)].operands.push(OperandDecoration, "");
        InstructionDesc[enumCast(Op::OpDecorateId)].operands.push(OperandVariableIds, "See <<Decoration,'Decoration'>>.");

        InstructionDesc[enumCast(Op::OpDecorateStringGOOGLE)].operands.push(OperandId, "'Target'");
        InstructionDesc[enumCast(Op::OpDecorateStringGOOGLE)].operands.push(OperandDecoration, "");
        InstructionDesc[enumCast(Op::OpDecorateStringGOOGLE)].operands.push(OperandVariableLiteralStrings, "'Literal Strings'");

        InstructionDesc[enumCast(Op::OpMemberDecorate)].operands.push(OperandId, "'Structure Type'");
        InstructionDesc[enumCast(Op::OpMemberDecorate)].operands.push(OperandLiteralNumber, "'Member'");
        InstructionDesc[enumCast(Op::OpMemberDecorate)].operands.push(OperandDecoration, "");
        InstructionDesc[enumCast(Op::OpMemberDecorate)].operands.push(OperandVariableLiterals, "See <<Decoration,'Decoration'>>.");

        InstructionDesc[enumCast(Op::OpMemberDecorateStringGOOGLE)].operands.push(OperandId, "'Structure Type'");
        InstructionDesc[enumCast(Op::OpMemberDecorateStringGOOGLE)].operands.push(OperandLiteralNumber, "'Member'");
        InstructionDesc[enumCast(Op::OpMemberDecorateStringGOOGLE)].operands.push(OperandDecoration, "");
        InstructionDesc[enumCast(Op::OpMemberDecorateStringGOOGLE)].operands.push(OperandVariableLiteralStrings, "'Literal Strings'");

        InstructionDesc[enumCast(Op::OpGroupDecorate)].operands.push(OperandId, "'Decoration Group'");
        InstructionDesc[enumCast(Op::OpGroupDecorate)].operands.push(OperandVariableIds, "'Targets'");

        InstructionDesc[enumCast(Op::OpGroupMemberDecorate)].operands.push(OperandId, "'Decoration Group'");
        InstructionDesc[enumCast(Op::OpGroupMemberDecorate)].operands.push(OperandVariableIdLiteral, "'Targets'");

        InstructionDesc[enumCast(Op::OpVectorExtractDynamic)].operands.push(OperandId, "'Vector'");
        InstructionDesc[enumCast(Op::OpVectorExtractDynamic)].operands.push(OperandId, "'Index'");

        InstructionDesc[enumCast(Op::OpVectorInsertDynamic)].operands.push(OperandId, "'Vector'");
        InstructionDesc[enumCast(Op::OpVectorInsertDynamic)].operands.push(OperandId, "'Component'");
        InstructionDesc[enumCast(Op::OpVectorInsertDynamic)].operands.push(OperandId, "'Index'");

        InstructionDesc[enumCast(Op::OpVectorShuffle)].operands.push(OperandId, "'Vector 1'");
        InstructionDesc[enumCast(Op::OpVectorShuffle)].operands.push(OperandId, "'Vector 2'");
        InstructionDesc[enumCast(Op::OpVectorShuffle)].operands.push(OperandVariableLiterals, "'Components'");

        InstructionDesc[enumCast(Op::OpCompositeConstruct)].operands.push(OperandVariableIds, "'Constituents'");

        InstructionDesc[enumCast(Op::OpCompositeExtract)].operands.push(OperandId, "'Composite'");
        InstructionDesc[enumCast(Op::OpCompositeExtract)].operands.push(OperandVariableLiterals, "'Indexes'");

        InstructionDesc[enumCast(Op::OpCompositeInsert)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpCompositeInsert)].operands.push(OperandId, "'Composite'");
        InstructionDesc[enumCast(Op::OpCompositeInsert)].operands.push(OperandVariableLiterals, "'Indexes'");

        InstructionDesc[enumCast(Op::OpCopyObject)].operands.push(OperandId, "'Operand'");

        InstructionDesc[enumCast(Op::OpCopyMemory)].operands.push(OperandId, "'Target'");
        InstructionDesc[enumCast(Op::OpCopyMemory)].operands.push(OperandId, "'Source'");
        InstructionDesc[enumCast(Op::OpCopyMemory)].operands.push(OperandMemoryAccess, "", true);

        InstructionDesc[enumCast(Op::OpCopyMemorySized)].operands.push(OperandId, "'Target'");
        InstructionDesc[enumCast(Op::OpCopyMemorySized)].operands.push(OperandId, "'Source'");
        InstructionDesc[enumCast(Op::OpCopyMemorySized)].operands.push(OperandId, "'Size'");
        InstructionDesc[enumCast(Op::OpCopyMemorySized)].operands.push(OperandMemoryAccess, "", true);

        InstructionDesc[enumCast(Op::OpSampledImage)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpSampledImage)].operands.push(OperandId, "'Sampler'");

        InstructionDesc[enumCast(Op::OpImage)].operands.push(OperandId, "'Sampled Image'");

        InstructionDesc[enumCast(Op::OpImageRead)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageRead)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageRead)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageRead)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageWrite)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageWrite)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageWrite)].operands.push(OperandId, "'Texel'");
        InstructionDesc[enumCast(Op::OpImageWrite)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageWrite)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleDrefImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleDrefImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleDrefImplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSampleDrefImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleDrefImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleDrefExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleDrefExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleDrefExplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSampleDrefExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleDrefExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleProjImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleProjImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleProjImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleProjImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleProjExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleProjExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleProjExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleProjExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleProjDrefImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefImplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSampleProjDrefExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefExplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleProjDrefExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageFetch)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageFetch)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageFetch)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageFetch)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageGather)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageGather)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageGather)].operands.push(OperandId, "'Component'");
        InstructionDesc[enumCast(Op::OpImageGather)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageGather)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageDrefGather)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageDrefGather)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageDrefGather)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageDrefGather)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageDrefGather)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefImplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefExplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleDrefExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleProjImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleProjExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefImplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefImplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefImplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefImplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefImplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefExplicitLod)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefExplicitLod)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefExplicitLod)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefExplicitLod)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseSampleProjDrefExplicitLod)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseFetch)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageSparseFetch)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseFetch)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseFetch)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseGather)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseGather)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseGather)].operands.push(OperandId, "'Component'");
        InstructionDesc[enumCast(Op::OpImageSparseGather)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseGather)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseDrefGather)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSparseDrefGather)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseDrefGather)].operands.push(OperandId, "'D~ref~'");
        InstructionDesc[enumCast(Op::OpImageSparseDrefGather)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseDrefGather)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseRead)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageSparseRead)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSparseRead)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSparseRead)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpImageSparseTexelsResident)].operands.push(OperandId, "'Resident Code'");

        InstructionDesc[enumCast(Op::OpImageQuerySizeLod)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageQuerySizeLod)].operands.push(OperandId, "'Level of Detail'");

        InstructionDesc[enumCast(Op::OpImageQuerySize)].operands.push(OperandId, "'Image'");

        InstructionDesc[enumCast(Op::OpImageQueryLod)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageQueryLod)].operands.push(OperandId, "'Coordinate'");

        InstructionDesc[enumCast(Op::OpImageQueryLevels)].operands.push(OperandId, "'Image'");

        InstructionDesc[enumCast(Op::OpImageQuerySamples)].operands.push(OperandId, "'Image'");

        InstructionDesc[enumCast(Op::OpImageQueryFormat)].operands.push(OperandId, "'Image'");

        InstructionDesc[enumCast(Op::OpImageQueryOrder)].operands.push(OperandId, "'Image'");

        InstructionDesc[enumCast(Op::OpAccessChain)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpAccessChain)].operands.push(OperandVariableIds, "'Indexes'");

        InstructionDesc[enumCast(Op::OpInBoundsAccessChain)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpInBoundsAccessChain)].operands.push(OperandVariableIds, "'Indexes'");

        InstructionDesc[enumCast(Op::OpPtrAccessChain)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpPtrAccessChain)].operands.push(OperandId, "'Element'");
        InstructionDesc[enumCast(Op::OpPtrAccessChain)].operands.push(OperandVariableIds, "'Indexes'");

        InstructionDesc[enumCast(Op::OpInBoundsPtrAccessChain)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpInBoundsPtrAccessChain)].operands.push(OperandId, "'Element'");
        InstructionDesc[enumCast(Op::OpInBoundsPtrAccessChain)].operands.push(OperandVariableIds, "'Indexes'");

        InstructionDesc[enumCast(Op::OpSNegate)].operands.push(OperandId, "'Operand'");

        InstructionDesc[enumCast(Op::OpFNegate)].operands.push(OperandId, "'Operand'");

        InstructionDesc[enumCast(Op::OpNot)].operands.push(OperandId, "'Operand'");

        InstructionDesc[enumCast(Op::OpAny)].operands.push(OperandId, "'Vector'");

        InstructionDesc[enumCast(Op::OpAll)].operands.push(OperandId, "'Vector'");

        InstructionDesc[enumCast(Op::OpConvertFToU)].operands.push(OperandId, "'Float Value'");

        InstructionDesc[enumCast(Op::OpConvertFToS)].operands.push(OperandId, "'Float Value'");

        InstructionDesc[enumCast(Op::OpConvertSToF)].operands.push(OperandId, "'Signed Value'");

        InstructionDesc[enumCast(Op::OpConvertUToF)].operands.push(OperandId, "'Unsigned Value'");

        InstructionDesc[enumCast(Op::OpUConvert)].operands.push(OperandId, "'Unsigned Value'");

        InstructionDesc[enumCast(Op::OpSConvert)].operands.push(OperandId, "'Signed Value'");

        InstructionDesc[enumCast(Op::OpFConvert)].operands.push(OperandId, "'Float Value'");

        InstructionDesc[enumCast(Op::OpSatConvertSToU)].operands.push(OperandId, "'Signed Value'");

        InstructionDesc[enumCast(Op::OpSatConvertUToS)].operands.push(OperandId, "'Unsigned Value'");

        InstructionDesc[enumCast(Op::OpConvertPtrToU)].operands.push(OperandId, "'Pointer'");

        InstructionDesc[enumCast(Op::OpConvertUToPtr)].operands.push(OperandId, "'Integer Value'");

        InstructionDesc[enumCast(Op::OpPtrCastToGeneric)].operands.push(OperandId, "'Pointer'");

        InstructionDesc[enumCast(Op::OpGenericCastToPtr)].operands.push(OperandId, "'Pointer'");

        InstructionDesc[enumCast(Op::OpGenericCastToPtrExplicit)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpGenericCastToPtrExplicit)].operands.push(OperandStorage, "'Storage'");

        InstructionDesc[enumCast(Op::OpGenericPtrMemSemantics)].operands.push(OperandId, "'Pointer'");

        InstructionDesc[enumCast(Op::OpBitcast)].operands.push(OperandId, "'Operand'");

        InstructionDesc[enumCast(Op::OpQuantizeToF16)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpTranspose)].operands.push(OperandId, "'Matrix'");

        InstructionDesc[enumCast(Op::OpCopyLogical)].operands.push(OperandId, "'Operand'");

        InstructionDesc[enumCast(Op::OpIsNan)].operands.push(OperandId, "'x'");

        InstructionDesc[enumCast(Op::OpIsInf)].operands.push(OperandId, "'x'");

        InstructionDesc[enumCast(Op::OpIsFinite)].operands.push(OperandId, "'x'");

        InstructionDesc[enumCast(Op::OpIsNormal)].operands.push(OperandId, "'x'");

        InstructionDesc[enumCast(Op::OpSignBitSet)].operands.push(OperandId, "'x'");

        InstructionDesc[enumCast(Op::OpLessOrGreater)].operands.push(OperandId, "'x'");
        InstructionDesc[enumCast(Op::OpLessOrGreater)].operands.push(OperandId, "'y'");

        InstructionDesc[enumCast(Op::OpOrdered)].operands.push(OperandId, "'x'");
        InstructionDesc[enumCast(Op::OpOrdered)].operands.push(OperandId, "'y'");

        InstructionDesc[enumCast(Op::OpUnordered)].operands.push(OperandId, "'x'");
        InstructionDesc[enumCast(Op::OpUnordered)].operands.push(OperandId, "'y'");

        InstructionDesc[enumCast(Op::OpArrayLength)].operands.push(OperandId, "'Structure'");
        InstructionDesc[enumCast(Op::OpArrayLength)].operands.push(OperandLiteralNumber, "'Array member'");

        InstructionDesc[enumCast(Op::OpIAdd)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpIAdd)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFAdd)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFAdd)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpISub)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpISub)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFSub)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFSub)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpIMul)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpIMul)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFMul)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFMul)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpUDiv)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpUDiv)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSDiv)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSDiv)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFDiv)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFDiv)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpUMod)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpUMod)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSRem)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSRem)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSMod)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSMod)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFRem)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFRem)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFMod)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFMod)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpVectorTimesScalar)].operands.push(OperandId, "'Vector'");
        InstructionDesc[enumCast(Op::OpVectorTimesScalar)].operands.push(OperandId, "'Scalar'");

        InstructionDesc[enumCast(Op::OpMatrixTimesScalar)].operands.push(OperandId, "'Matrix'");
        InstructionDesc[enumCast(Op::OpMatrixTimesScalar)].operands.push(OperandId, "'Scalar'");

        InstructionDesc[enumCast(Op::OpVectorTimesMatrix)].operands.push(OperandId, "'Vector'");
        InstructionDesc[enumCast(Op::OpVectorTimesMatrix)].operands.push(OperandId, "'Matrix'");

        InstructionDesc[enumCast(Op::OpMatrixTimesVector)].operands.push(OperandId, "'Matrix'");
        InstructionDesc[enumCast(Op::OpMatrixTimesVector)].operands.push(OperandId, "'Vector'");

        InstructionDesc[enumCast(Op::OpMatrixTimesMatrix)].operands.push(OperandId, "'LeftMatrix'");
        InstructionDesc[enumCast(Op::OpMatrixTimesMatrix)].operands.push(OperandId, "'RightMatrix'");

        InstructionDesc[enumCast(Op::OpOuterProduct)].operands.push(OperandId, "'Vector 1'");
        InstructionDesc[enumCast(Op::OpOuterProduct)].operands.push(OperandId, "'Vector 2'");

        InstructionDesc[enumCast(Op::OpDot)].operands.push(OperandId, "'Vector 1'");
        InstructionDesc[enumCast(Op::OpDot)].operands.push(OperandId, "'Vector 2'");

        InstructionDesc[enumCast(Op::OpIAddCarry)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpIAddCarry)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpISubBorrow)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpISubBorrow)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpUMulExtended)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpUMulExtended)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSMulExtended)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSMulExtended)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpShiftRightLogical)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpShiftRightLogical)].operands.push(OperandId, "'Shift'");

        InstructionDesc[enumCast(Op::OpShiftRightArithmetic)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpShiftRightArithmetic)].operands.push(OperandId, "'Shift'");

        InstructionDesc[enumCast(Op::OpShiftLeftLogical)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpShiftLeftLogical)].operands.push(OperandId, "'Shift'");

        InstructionDesc[enumCast(Op::OpLogicalOr)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpLogicalOr)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpLogicalAnd)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpLogicalAnd)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpLogicalEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpLogicalEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpLogicalNotEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpLogicalNotEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpLogicalNot)].operands.push(OperandId, "'Operand'");

        InstructionDesc[enumCast(Op::OpBitwiseOr)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpBitwiseOr)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpBitwiseXor)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpBitwiseXor)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpBitwiseAnd)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpBitwiseAnd)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpBitFieldInsert)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpBitFieldInsert)].operands.push(OperandId, "'Insert'");
        InstructionDesc[enumCast(Op::OpBitFieldInsert)].operands.push(OperandId, "'Offset'");
        InstructionDesc[enumCast(Op::OpBitFieldInsert)].operands.push(OperandId, "'Count'");

        InstructionDesc[enumCast(Op::OpBitFieldSExtract)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpBitFieldSExtract)].operands.push(OperandId, "'Offset'");
        InstructionDesc[enumCast(Op::OpBitFieldSExtract)].operands.push(OperandId, "'Count'");

        InstructionDesc[enumCast(Op::OpBitFieldUExtract)].operands.push(OperandId, "'Base'");
        InstructionDesc[enumCast(Op::OpBitFieldUExtract)].operands.push(OperandId, "'Offset'");
        InstructionDesc[enumCast(Op::OpBitFieldUExtract)].operands.push(OperandId, "'Count'");

        InstructionDesc[enumCast(Op::OpBitReverse)].operands.push(OperandId, "'Base'");

        InstructionDesc[enumCast(Op::OpBitCount)].operands.push(OperandId, "'Base'");

        InstructionDesc[enumCast(Op::OpSelect)].operands.push(OperandId, "'Condition'");
        InstructionDesc[enumCast(Op::OpSelect)].operands.push(OperandId, "'Object 1'");
        InstructionDesc[enumCast(Op::OpSelect)].operands.push(OperandId, "'Object 2'");

        InstructionDesc[enumCast(Op::OpIEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpIEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFOrdEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFOrdEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFUnordEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFUnordEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpINotEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpINotEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFOrdNotEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFOrdNotEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFUnordNotEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFUnordNotEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpULessThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpULessThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSLessThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSLessThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFOrdLessThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFOrdLessThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFUnordLessThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFUnordLessThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpUGreaterThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpUGreaterThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSGreaterThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSGreaterThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFOrdGreaterThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFOrdGreaterThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFUnordGreaterThan)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFUnordGreaterThan)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpULessThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpULessThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSLessThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSLessThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFOrdLessThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFOrdLessThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFUnordLessThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFUnordLessThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpUGreaterThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpUGreaterThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpSGreaterThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpSGreaterThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFOrdGreaterThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFOrdGreaterThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpFUnordGreaterThanEqual)].operands.push(OperandId, "'Operand 1'");
        InstructionDesc[enumCast(Op::OpFUnordGreaterThanEqual)].operands.push(OperandId, "'Operand 2'");

        InstructionDesc[enumCast(Op::OpDPdx)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpDPdy)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpFwidth)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpDPdxFine)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpDPdyFine)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpFwidthFine)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpDPdxCoarse)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpDPdyCoarse)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpFwidthCoarse)].operands.push(OperandId, "'P'");

        InstructionDesc[enumCast(Op::OpEmitStreamVertex)].operands.push(OperandId, "'Stream'");

        InstructionDesc[enumCast(Op::OpEndStreamPrimitive)].operands.push(OperandId, "'Stream'");

        InstructionDesc[enumCast(Op::OpControlBarrier)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpControlBarrier)].operands.push(OperandScope, "'Memory'");
        InstructionDesc[enumCast(Op::OpControlBarrier)].operands.push(OperandMemorySemantics, "'Semantics'");

        InstructionDesc[enumCast(Op::OpMemoryBarrier)].operands.push(OperandScope, "'Memory'");
        InstructionDesc[enumCast(Op::OpMemoryBarrier)].operands.push(OperandMemorySemantics, "'Semantics'");

        InstructionDesc[enumCast(Op::OpImageTexelPointer)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpImageTexelPointer)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageTexelPointer)].operands.push(OperandId, "'Sample'");

        InstructionDesc[enumCast(Op::OpAtomicLoad)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicLoad)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicLoad)].operands.push(OperandMemorySemantics, "'Semantics'");

        InstructionDesc[enumCast(Op::OpAtomicStore)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicStore)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicStore)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicStore)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicExchange)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicExchange)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicExchange)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicExchange)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicCompareExchange)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchange)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchange)].operands.push(OperandMemorySemantics, "'Equal'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchange)].operands.push(OperandMemorySemantics, "'Unequal'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchange)].operands.push(OperandId, "'Value'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchange)].operands.push(OperandId, "'Comparator'");

        InstructionDesc[enumCast(Op::OpAtomicCompareExchangeWeak)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchangeWeak)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchangeWeak)].operands.push(OperandMemorySemantics, "'Equal'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchangeWeak)].operands.push(OperandMemorySemantics, "'Unequal'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchangeWeak)].operands.push(OperandId, "'Value'");
        InstructionDesc[enumCast(Op::OpAtomicCompareExchangeWeak)].operands.push(OperandId, "'Comparator'");

        InstructionDesc[enumCast(Op::OpAtomicIIncrement)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicIIncrement)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicIIncrement)].operands.push(OperandMemorySemantics, "'Semantics'");

        InstructionDesc[enumCast(Op::OpAtomicIDecrement)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicIDecrement)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicIDecrement)].operands.push(OperandMemorySemantics, "'Semantics'");

        InstructionDesc[enumCast(Op::OpAtomicIAdd)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicIAdd)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicIAdd)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicIAdd)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicFAddEXT)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicFAddEXT)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicFAddEXT)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicFAddEXT)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAssumeTrueKHR)].operands.push(OperandId, "'Condition'");

        InstructionDesc[enumCast(Op::OpExpectKHR)].operands.push(OperandId, "'Value'");
        InstructionDesc[enumCast(Op::OpExpectKHR)].operands.push(OperandId, "'ExpectedValue'");

        InstructionDesc[enumCast(Op::OpAtomicISub)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicISub)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicISub)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicISub)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicUMin)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicUMin)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicUMin)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicUMin)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicUMax)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicUMax)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicUMax)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicUMax)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicSMin)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicSMin)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicSMin)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicSMin)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicSMax)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicSMax)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicSMax)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicSMax)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicFMinEXT)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicFMinEXT)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicFMinEXT)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicFMinEXT)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicFMaxEXT)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicFMaxEXT)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicFMaxEXT)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicFMaxEXT)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicAnd)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicAnd)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicAnd)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicAnd)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicOr)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicOr)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicOr)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicOr)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicXor)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicXor)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicXor)].operands.push(OperandMemorySemantics, "'Semantics'");
        InstructionDesc[enumCast(Op::OpAtomicXor)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpAtomicFlagTestAndSet)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicFlagTestAndSet)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicFlagTestAndSet)].operands.push(OperandMemorySemantics, "'Semantics'");

        InstructionDesc[enumCast(Op::OpAtomicFlagClear)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpAtomicFlagClear)].operands.push(OperandScope, "'Scope'");
        InstructionDesc[enumCast(Op::OpAtomicFlagClear)].operands.push(OperandMemorySemantics, "'Semantics'");

        InstructionDesc[enumCast(Op::OpLoopMerge)].operands.push(OperandId, "'Merge Block'");
        InstructionDesc[enumCast(Op::OpLoopMerge)].operands.push(OperandId, "'Continue Target'");
        InstructionDesc[enumCast(Op::OpLoopMerge)].operands.push(OperandLoop, "");
        InstructionDesc[enumCast(Op::OpLoopMerge)].operands.push(OperandOptionalLiteral, "");

        InstructionDesc[enumCast(Op::OpSelectionMerge)].operands.push(OperandId, "'Merge Block'");
        InstructionDesc[enumCast(Op::OpSelectionMerge)].operands.push(OperandSelect, "");

        InstructionDesc[enumCast(Op::OpBranch)].operands.push(OperandId, "'Target Label'");

        InstructionDesc[enumCast(Op::OpBranchConditional)].operands.push(OperandId, "'Condition'");
        InstructionDesc[enumCast(Op::OpBranchConditional)].operands.push(OperandId, "'True Label'");
        InstructionDesc[enumCast(Op::OpBranchConditional)].operands.push(OperandId, "'False Label'");
        InstructionDesc[enumCast(Op::OpBranchConditional)].operands.push(OperandVariableLiterals, "'Branch weights'");

        InstructionDesc[enumCast(Op::OpSwitch)].operands.push(OperandId, "'Selector'");
        InstructionDesc[enumCast(Op::OpSwitch)].operands.push(OperandId, "'Default'");
        InstructionDesc[enumCast(Op::OpSwitch)].operands.push(OperandVariableLiteralId, "'Target'");


        InstructionDesc[enumCast(Op::OpReturnValue)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpLifetimeStart)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpLifetimeStart)].operands.push(OperandLiteralNumber, "'Size'");

        InstructionDesc[enumCast(Op::OpLifetimeStop)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpLifetimeStop)].operands.push(OperandLiteralNumber, "'Size'");

        InstructionDesc[enumCast(Op::OpGroupAsyncCopy)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupAsyncCopy)].operands.push(OperandId, "'Destination'");
        InstructionDesc[enumCast(Op::OpGroupAsyncCopy)].operands.push(OperandId, "'Source'");
        InstructionDesc[enumCast(Op::OpGroupAsyncCopy)].operands.push(OperandId, "'Num Elements'");
        InstructionDesc[enumCast(Op::OpGroupAsyncCopy)].operands.push(OperandId, "'Stride'");
        InstructionDesc[enumCast(Op::OpGroupAsyncCopy)].operands.push(OperandId, "'Event'");

        InstructionDesc[enumCast(Op::OpGroupWaitEvents)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupWaitEvents)].operands.push(OperandId, "'Num Events'");
        InstructionDesc[enumCast(Op::OpGroupWaitEvents)].operands.push(OperandId, "'Events List'");

        InstructionDesc[enumCast(Op::OpGroupAll)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupAll)].operands.push(OperandId, "'Predicate'");

        InstructionDesc[enumCast(Op::OpGroupAny)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupAny)].operands.push(OperandId, "'Predicate'");

        InstructionDesc[enumCast(Op::OpGroupBroadcast)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupBroadcast)].operands.push(OperandId, "'Value'");
        InstructionDesc[enumCast(Op::OpGroupBroadcast)].operands.push(OperandId, "'LocalId'");

        InstructionDesc[enumCast(Op::OpGroupIAdd)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupIAdd)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupIAdd)].operands.push(OperandId, "'X'");

        InstructionDesc[enumCast(Op::OpGroupFAdd)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupFAdd)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupFAdd)].operands.push(OperandId, "'X'");

        InstructionDesc[enumCast(Op::OpGroupUMin)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupUMin)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupUMin)].operands.push(OperandId, "'X'");

        InstructionDesc[enumCast(Op::OpGroupSMin)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupSMin)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupSMin)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupFMin)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupFMin)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupFMin)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupUMax)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupUMax)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupUMax)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupSMax)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupSMax)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupSMax)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupFMax)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupFMax)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupFMax)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpReadPipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpReadPipe)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpReadPipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpReadPipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpWritePipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpWritePipe)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpWritePipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpWritePipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpReservedReadPipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpReservedReadPipe)].operands.push(OperandId, "'Reserve Id'");
        InstructionDesc[enumCast(Op::OpReservedReadPipe)].operands.push(OperandId, "'Index'");
        InstructionDesc[enumCast(Op::OpReservedReadPipe)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpReservedReadPipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpReservedReadPipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpReservedWritePipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpReservedWritePipe)].operands.push(OperandId, "'Reserve Id'");
        InstructionDesc[enumCast(Op::OpReservedWritePipe)].operands.push(OperandId, "'Index'");
        InstructionDesc[enumCast(Op::OpReservedWritePipe)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpReservedWritePipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpReservedWritePipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpReserveReadPipePackets)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpReserveReadPipePackets)].operands.push(OperandId, "'Num Packets'");
        InstructionDesc[enumCast(Op::OpReserveReadPipePackets)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpReserveReadPipePackets)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpReserveWritePipePackets)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpReserveWritePipePackets)].operands.push(OperandId, "'Num Packets'");
        InstructionDesc[enumCast(Op::OpReserveWritePipePackets)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpReserveWritePipePackets)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpCommitReadPipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpCommitReadPipe)].operands.push(OperandId, "'Reserve Id'");
        InstructionDesc[enumCast(Op::OpCommitReadPipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpCommitReadPipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpCommitWritePipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpCommitWritePipe)].operands.push(OperandId, "'Reserve Id'");
        InstructionDesc[enumCast(Op::OpCommitWritePipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpCommitWritePipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpIsValidReserveId)].operands.push(OperandId, "'Reserve Id'");

        InstructionDesc[enumCast(Op::OpGetNumPipePackets)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpGetNumPipePackets)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpGetNumPipePackets)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpGetMaxPipePackets)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpGetMaxPipePackets)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpGetMaxPipePackets)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpGroupReserveReadPipePackets)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupReserveReadPipePackets)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpGroupReserveReadPipePackets)].operands.push(OperandId, "'Num Packets'");
        InstructionDesc[enumCast(Op::OpGroupReserveReadPipePackets)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpGroupReserveReadPipePackets)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpGroupReserveWritePipePackets)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupReserveWritePipePackets)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpGroupReserveWritePipePackets)].operands.push(OperandId, "'Num Packets'");
        InstructionDesc[enumCast(Op::OpGroupReserveWritePipePackets)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpGroupReserveWritePipePackets)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpGroupCommitReadPipe)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupCommitReadPipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpGroupCommitReadPipe)].operands.push(OperandId, "'Reserve Id'");
        InstructionDesc[enumCast(Op::OpGroupCommitReadPipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpGroupCommitReadPipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpGroupCommitWritePipe)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupCommitWritePipe)].operands.push(OperandId, "'Pipe'");
        InstructionDesc[enumCast(Op::OpGroupCommitWritePipe)].operands.push(OperandId, "'Reserve Id'");
        InstructionDesc[enumCast(Op::OpGroupCommitWritePipe)].operands.push(OperandId, "'Packet Size'");
        InstructionDesc[enumCast(Op::OpGroupCommitWritePipe)].operands.push(OperandId, "'Packet Alignment'");

        InstructionDesc[enumCast(Op::OpBuildNDRange)].operands.push(OperandId, "'GlobalWorkSize'");
        InstructionDesc[enumCast(Op::OpBuildNDRange)].operands.push(OperandId, "'LocalWorkSize'");
        InstructionDesc[enumCast(Op::OpBuildNDRange)].operands.push(OperandId, "'GlobalWorkOffset'");

        InstructionDesc[enumCast(Op::OpCaptureEventProfilingInfo)].operands.push(OperandId, "'Event'");
        InstructionDesc[enumCast(Op::OpCaptureEventProfilingInfo)].operands.push(OperandId, "'Profiling Info'");
        InstructionDesc[enumCast(Op::OpCaptureEventProfilingInfo)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpSetUserEventStatus)].operands.push(OperandId, "'Event'");
        InstructionDesc[enumCast(Op::OpSetUserEventStatus)].operands.push(OperandId, "'Status'");

        InstructionDesc[enumCast(Op::OpIsValidEvent)].operands.push(OperandId, "'Event'");

        InstructionDesc[enumCast(Op::OpRetainEvent)].operands.push(OperandId, "'Event'");

        InstructionDesc[enumCast(Op::OpReleaseEvent)].operands.push(OperandId, "'Event'");

        InstructionDesc[enumCast(Op::OpGetKernelWorkGroupSize)].operands.push(OperandId, "'Invoke'");
        InstructionDesc[enumCast(Op::OpGetKernelWorkGroupSize)].operands.push(OperandId, "'Param'");
        InstructionDesc[enumCast(Op::OpGetKernelWorkGroupSize)].operands.push(OperandId, "'Param Size'");
        InstructionDesc[enumCast(Op::OpGetKernelWorkGroupSize)].operands.push(OperandId, "'Param Align'");

        InstructionDesc[enumCast(Op::OpGetKernelPreferredWorkGroupSizeMultiple)].operands.push(OperandId, "'Invoke'");
        InstructionDesc[enumCast(Op::OpGetKernelPreferredWorkGroupSizeMultiple)].operands.push(OperandId, "'Param'");
        InstructionDesc[enumCast(Op::OpGetKernelPreferredWorkGroupSizeMultiple)].operands.push(OperandId, "'Param Size'");
        InstructionDesc[enumCast(Op::OpGetKernelPreferredWorkGroupSizeMultiple)].operands.push(OperandId, "'Param Align'");

        InstructionDesc[enumCast(Op::OpGetKernelNDrangeSubGroupCount)].operands.push(OperandId, "'ND Range'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeSubGroupCount)].operands.push(OperandId, "'Invoke'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeSubGroupCount)].operands.push(OperandId, "'Param'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeSubGroupCount)].operands.push(OperandId, "'Param Size'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeSubGroupCount)].operands.push(OperandId, "'Param Align'");

        InstructionDesc[enumCast(Op::OpGetKernelNDrangeMaxSubGroupSize)].operands.push(OperandId, "'ND Range'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeMaxSubGroupSize)].operands.push(OperandId, "'Invoke'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeMaxSubGroupSize)].operands.push(OperandId, "'Param'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeMaxSubGroupSize)].operands.push(OperandId, "'Param Size'");
        InstructionDesc[enumCast(Op::OpGetKernelNDrangeMaxSubGroupSize)].operands.push(OperandId, "'Param Align'");

        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Queue'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Flags'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'ND Range'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Num Events'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Wait Events'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Ret Event'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Invoke'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Param'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Param Size'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandId, "'Param Align'");
        InstructionDesc[enumCast(Op::OpEnqueueKernel)].operands.push(OperandVariableIds, "'Local Size'");

        InstructionDesc[enumCast(Op::OpEnqueueMarker)].operands.push(OperandId, "'Queue'");
        InstructionDesc[enumCast(Op::OpEnqueueMarker)].operands.push(OperandId, "'Num Events'");
        InstructionDesc[enumCast(Op::OpEnqueueMarker)].operands.push(OperandId, "'Wait Events'");
        InstructionDesc[enumCast(Op::OpEnqueueMarker)].operands.push(OperandId, "'Ret Event'");

        InstructionDesc[enumCast(Op::OpGroupNonUniformElect)].operands.push(OperandScope, "'Execution'");

        InstructionDesc[enumCast(Op::OpGroupNonUniformAll)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformAll)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformAny)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformAny)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformAllEqual)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformAllEqual)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformBroadcast)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBroadcast)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBroadcast)].operands.push(OperandId, "ID");

        InstructionDesc[enumCast(Op::OpGroupNonUniformBroadcastFirst)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBroadcastFirst)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformBallot)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBallot)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformInverseBallot)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformInverseBallot)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotBitExtract)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotBitExtract)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotBitExtract)].operands.push(OperandId, "Bit");

        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotBitCount)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotBitCount)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotBitCount)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotFindLSB)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotFindLSB)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotFindMSB)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBallotFindMSB)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffle)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffle)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffle)].operands.push(OperandId, "'Id'");

        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleXor)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleXor)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleXor)].operands.push(OperandId, "Mask");

        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleUp)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleUp)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleUp)].operands.push(OperandId, "Offset");

        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleDown)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleDown)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformShuffleDown)].operands.push(OperandId, "Offset");

        InstructionDesc[enumCast(Op::OpGroupNonUniformIAdd)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformIAdd)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformIAdd)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformIAdd)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformFAdd)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFAdd)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFAdd)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFAdd)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformIMul)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformIMul)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformIMul)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformIMul)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformFMul)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMul)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMul)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMul)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformSMin)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformSMin)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformSMin)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformSMin)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformUMin)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformUMin)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformUMin)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformUMin)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformFMin)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMin)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMin)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMin)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformSMax)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformSMax)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformSMax)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformSMax)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformUMax)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformUMax)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformUMax)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformUMax)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformFMax)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMax)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMax)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformFMax)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseAnd)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseAnd)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseAnd)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseAnd)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseOr)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseOr)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseOr)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseOr)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseXor)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseXor)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseXor)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformBitwiseXor)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalAnd)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalAnd)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalAnd)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalAnd)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalOr)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalOr)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalOr)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalOr)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalXor)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalXor)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalXor)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformLogicalXor)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadBroadcast)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadBroadcast)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadBroadcast)].operands.push(OperandId, "'Id'");

        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadSwap)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadSwap)].operands.push(OperandId, "X");
        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadSwap)].operands.push(OperandId, "'Direction'");

        InstructionDesc[enumCast(Op::OpSubgroupBallotKHR)].operands.push(OperandId, "'Predicate'");

        InstructionDesc[enumCast(Op::OpSubgroupFirstInvocationKHR)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpSubgroupAnyKHR)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpSubgroupAnyKHR)].operands.push(OperandId, "'Predicate'");

        InstructionDesc[enumCast(Op::OpSubgroupAllKHR)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpSubgroupAllKHR)].operands.push(OperandId, "'Predicate'");

        InstructionDesc[enumCast(Op::OpSubgroupAllEqualKHR)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpSubgroupAllEqualKHR)].operands.push(OperandId, "'Predicate'");

        InstructionDesc[enumCast(Op::OpGroupNonUniformRotateKHR)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformRotateKHR)].operands.push(OperandId, "'X'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformRotateKHR)].operands.push(OperandId, "'Delta'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformRotateKHR)].operands.push(OperandId, "'ClusterSize'", true);

        InstructionDesc[enumCast(Op::OpSubgroupReadInvocationKHR)].operands.push(OperandId, "'Value'");
        InstructionDesc[enumCast(Op::OpSubgroupReadInvocationKHR)].operands.push(OperandId, "'Index'");

        InstructionDesc[enumCast(Op::OpModuleProcessed)].operands.push(OperandLiteralString, "'process'");

        InstructionDesc[enumCast(Op::OpGroupIAddNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupIAddNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupIAddNonUniformAMD)].operands.push(OperandId, "'X'");

        InstructionDesc[enumCast(Op::OpGroupFAddNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupFAddNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupFAddNonUniformAMD)].operands.push(OperandId, "'X'");

        InstructionDesc[enumCast(Op::OpGroupUMinNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupUMinNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupUMinNonUniformAMD)].operands.push(OperandId, "'X'");

        InstructionDesc[enumCast(Op::OpGroupSMinNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupSMinNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupSMinNonUniformAMD)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupFMinNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupFMinNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupFMinNonUniformAMD)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupUMaxNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupUMaxNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupUMaxNonUniformAMD)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupSMaxNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupSMaxNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupSMaxNonUniformAMD)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupFMaxNonUniformAMD)].operands.push(OperandScope, "'Execution'");
        InstructionDesc[enumCast(Op::OpGroupFMaxNonUniformAMD)].operands.push(OperandGroupOperation, "'Operation'");
        InstructionDesc[enumCast(Op::OpGroupFMaxNonUniformAMD)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpFragmentMaskFetchAMD)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpFragmentMaskFetchAMD)].operands.push(OperandId, "'Coordinate'");

        InstructionDesc[enumCast(Op::OpFragmentFetchAMD)].operands.push(OperandId, "'Image'");
        InstructionDesc[enumCast(Op::OpFragmentFetchAMD)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpFragmentFetchAMD)].operands.push(OperandId, "'Fragment Index'");

        InstructionDesc[enumCast(Op::OpGroupNonUniformPartitionNV)].operands.push(OperandId, "X");

        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadAllKHR)].operands.push(OperandId, "'Predicate'");
        InstructionDesc[enumCast(Op::OpGroupNonUniformQuadAnyKHR)].operands.push(OperandId, "'Predicate'");
        InstructionDesc[enumCast(Op::OpTypeAccelerationStructureKHR)].setResultAndType(true, false);

        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'Ray Flags'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'Cull Mask'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'Ray Origin'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'Ray Direction'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpTraceNV)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpTraceNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Ray Flags'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Cull Mask'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Ray Origin'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Ray Direction'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Time'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpTraceRayMotionNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'Ray Flags'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'Cull Mask'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'Ray Origin'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'Ray Direction'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpTraceRayKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpReportIntersectionKHR)].operands.push(OperandId, "'Hit Parameter'");
        InstructionDesc[enumCast(Op::OpReportIntersectionKHR)].operands.push(OperandId, "'Hit Kind'");

        InstructionDesc[enumCast(Op::OpIgnoreIntersectionNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpIgnoreIntersectionKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpTerminateRayNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpTerminateRayKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpExecuteCallableNV)].operands.push(OperandId, "SBT Record Index");
        InstructionDesc[enumCast(Op::OpExecuteCallableNV)].operands.push(OperandId, "CallableData ID");
        InstructionDesc[enumCast(Op::OpExecuteCallableNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpExecuteCallableKHR)].operands.push(OperandId, "SBT Record Index");
        InstructionDesc[enumCast(Op::OpExecuteCallableKHR)].operands.push(OperandId, "CallableData");
        InstructionDesc[enumCast(Op::OpExecuteCallableKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpConvertUToAccelerationStructureKHR)].operands.push(OperandId, "Value");
        InstructionDesc[enumCast(Op::OpConvertUToAccelerationStructureKHR)].setResultAndType(true, true);

        // Ray Query
        InstructionDesc[enumCast(Op::OpTypeAccelerationStructureKHR)].setResultAndType(true, false);
        InstructionDesc[enumCast(Op::OpTypeRayQueryKHR)].setResultAndType(true, false);

        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'AccelerationS'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'CullMask'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'Tmin'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].operands.push(OperandId, "'Tmax'");
        InstructionDesc[enumCast(Op::OpRayQueryInitializeKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpRayQueryTerminateKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryTerminateKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpRayQueryGenerateIntersectionKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGenerateIntersectionKHR)].operands.push(OperandId, "'THit'");
        InstructionDesc[enumCast(Op::OpRayQueryGenerateIntersectionKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpRayQueryConfirmIntersectionKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryConfirmIntersectionKHR)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpRayQueryProceedKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryProceedKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTypeKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTypeKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTypeKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetRayTMinKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetRayTMinKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetRayFlagsKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetRayFlagsKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceCustomIndexKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceIdKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceIdKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceIdKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionGeometryIndexKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionGeometryIndexKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionGeometryIndexKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionPrimitiveIndexKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionPrimitiveIndexKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionPrimitiveIndexKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionBarycentricsKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionBarycentricsKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionBarycentricsKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionFrontFaceKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionFrontFaceKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionFrontFaceKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionCandidateAABBOpaqueKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectRayDirectionKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectRayDirectionKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectRayDirectionKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectRayOriginKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectRayOriginKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectRayOriginKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetWorldRayDirectionKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetWorldRayDirectionKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetWorldRayOriginKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetWorldRayOriginKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectToWorldKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectToWorldKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionObjectToWorldKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionWorldToObjectKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionWorldToObjectKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionWorldToObjectKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTriangleVertexPositionsKHR)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTriangleVertexPositionsKHR)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionTriangleVertexPositionsKHR)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetClusterIdNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetClusterIdNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetClusterIdNV)].setResultAndType(true, true);
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionSpherePositionNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionSpherePositionNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionSpherePositionNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionSphereRadiusNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionSphereRadiusNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionSphereRadiusNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSHitValueNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSHitValueNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSHitValueNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSPositionsNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSPositionsNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSPositionsNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSRadiiNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSRadiiNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryGetIntersectionLSSRadiiNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryIsSphereHitNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryIsSphereHitNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryIsSphereHitNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpRayQueryIsLSSHitNV)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpRayQueryIsLSSHitNV)].operands.push(OperandId, "'Committed'");
        InstructionDesc[enumCast(Op::OpRayQueryIsLSSHitNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageSampleFootprintNV)].operands.push(OperandId, "'Sampled Image'");
        InstructionDesc[enumCast(Op::OpImageSampleFootprintNV)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpImageSampleFootprintNV)].operands.push(OperandId, "'Granularity'");
        InstructionDesc[enumCast(Op::OpImageSampleFootprintNV)].operands.push(OperandId, "'Coarse'");
        InstructionDesc[enumCast(Op::OpImageSampleFootprintNV)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleFootprintNV)].operands.push(OperandVariableIds, "", true);

        InstructionDesc[enumCast(Op::OpWritePackedPrimitiveIndices4x8NV)].operands.push(OperandId, "'Index Offset'");
        InstructionDesc[enumCast(Op::OpWritePackedPrimitiveIndices4x8NV)].operands.push(OperandId, "'Packed Indices'");

        InstructionDesc[enumCast(Op::OpEmitMeshTasksEXT)].operands.push(OperandId, "'groupCountX'");
        InstructionDesc[enumCast(Op::OpEmitMeshTasksEXT)].operands.push(OperandId, "'groupCountY'");
        InstructionDesc[enumCast(Op::OpEmitMeshTasksEXT)].operands.push(OperandId, "'groupCountZ'");
        InstructionDesc[enumCast(Op::OpEmitMeshTasksEXT)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpEmitMeshTasksEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpSetMeshOutputsEXT)].operands.push(OperandId, "'vertexCount'");
        InstructionDesc[enumCast(Op::OpSetMeshOutputsEXT)].operands.push(OperandId, "'primitiveCount'");
        InstructionDesc[enumCast(Op::OpSetMeshOutputsEXT)].setResultAndType(false, false);


        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixNV)].operands.push(OperandId, "'Component Type'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixNV)].operands.push(OperandId, "'Scope'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixNV)].operands.push(OperandId, "'Rows'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixNV)].operands.push(OperandId, "'Columns'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadNV)].operands.push(OperandId, "'Stride'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadNV)].operands.push(OperandId, "'Column Major'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadNV)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadNV)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadNV)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].operands.push(OperandId, "'Stride'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].operands.push(OperandId, "'Column Major'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreNV)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpCooperativeMatrixMulAddNV)].operands.push(OperandId, "'A'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixMulAddNV)].operands.push(OperandId, "'B'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixMulAddNV)].operands.push(OperandId, "'C'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixLengthNV)].operands.push(OperandId, "'Type'");

        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixKHR)].operands.push(OperandId, "'Component Type'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixKHR)].operands.push(OperandId, "'Scope'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixKHR)].operands.push(OperandId, "'Rows'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixKHR)].operands.push(OperandId, "'Columns'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeMatrixKHR)].operands.push(OperandId, "'Use'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadKHR)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadKHR)].operands.push(OperandId, "'Memory Layout'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadKHR)].operands.push(OperandId, "'Stride'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadKHR)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadKHR)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadKHR)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].operands.push(OperandId, "'Memory Layout'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].operands.push(OperandId, "'Stride'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreKHR)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpCooperativeMatrixMulAddKHR)].operands.push(OperandId, "'A'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixMulAddKHR)].operands.push(OperandId, "'B'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixMulAddKHR)].operands.push(OperandId, "'C'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixMulAddKHR)].operands.push(OperandCooperativeMatrixOperands, "'Cooperative Matrix Operands'", true);

        InstructionDesc[enumCast(Op::OpCooperativeMatrixLengthKHR)].operands.push(OperandId, "'Type'");

        InstructionDesc[enumCast(Op::OpTypeCooperativeVectorNV)].operands.push(OperandId, "'Component Type'");
        InstructionDesc[enumCast(Op::OpTypeCooperativeVectorNV)].operands.push(OperandId, "'Components'");

        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'Input'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'InputInterpretation'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'Matrix'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'MatrixOffset'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'MatrixInterpretation'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'M'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'K'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'MemoryLayout'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'Transpose'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandId, "'MatrixStride'", true);
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulNV)].operands.push(OperandCooperativeMatrixOperands, "'Cooperative Matrix Operands'", true);

        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'Input'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'InputInterpretation'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'Matrix'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'MatrixOffset'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'MatrixInterpretation'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'Bias'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'BiasOffset'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'BiasInterpretation'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'M'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'K'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'MemoryLayout'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'Transpose'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandId, "'MatrixStride'", true);
        InstructionDesc[enumCast(Op::OpCooperativeVectorMatrixMulAddNV)].operands.push(OperandCooperativeMatrixOperands, "'Cooperative Matrix Operands'", true);

        InstructionDesc[enumCast(Op::OpCooperativeVectorLoadNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorLoadNV)].operands.push(OperandId, "'Offset'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorLoadNV)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorLoadNV)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpCooperativeVectorLoadNV)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpCooperativeVectorStoreNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorStoreNV)].operands.push(OperandId, "'Offset'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorStoreNV)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorStoreNV)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorStoreNV)].operands.push(OperandLiteralNumber, "", true);
        InstructionDesc[enumCast(Op::OpCooperativeVectorStoreNV)].operands.push(OperandId, "", true);

        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].operands.push(OperandId, "'Offset'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].operands.push(OperandId, "'A'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].operands.push(OperandId, "'B'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].operands.push(OperandId, "'MemoryLayout'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].operands.push(OperandId, "'MatrixInterpretation'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorOuterProductAccumulateNV)].operands.push(OperandId, "'MatrixStride'", true);

        InstructionDesc[enumCast(Op::OpCooperativeVectorReduceSumAccumulateNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorReduceSumAccumulateNV)].operands.push(OperandId, "'Offset'");
        InstructionDesc[enumCast(Op::OpCooperativeVectorReduceSumAccumulateNV)].operands.push(OperandId, "'V'");

        InstructionDesc[enumCast(Op::OpDemoteToHelperInvocationEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpReadClockKHR)].operands.push(OperandScope, "'Scope'");

        InstructionDesc[enumCast(Op::OpTypeHitObjectNV)].setResultAndType(true, false);

        InstructionDesc[enumCast(Op::OpHitObjectGetShaderRecordBufferHandleNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetShaderRecordBufferHandleNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpReorderThreadWithHintNV)].operands.push(OperandId, "'Hint'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHintNV)].operands.push(OperandId, "'Bits'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHintNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectNV)].operands.push(OperandId, "'Hint'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectNV)].operands.push(OperandId, "'Bits'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectGetCurrentTimeNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetCurrentTimeNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetHitKindNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetHitKindNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetPrimitiveIndexNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetPrimitiveIndexNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetGeometryIndexNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetGeometryIndexNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceIdNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceIdNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceCustomIndexNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceCustomIndexNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayDirectionNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayDirectionNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayOriginNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayOriginNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayDirectionNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayDirectionNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayOriginNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayOriginNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetWorldToObjectNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetWorldToObjectNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetObjectToWorldNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetObjectToWorldNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMaxNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMaxNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMinNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMinNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetShaderBindingTableRecordIndexNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetShaderBindingTableRecordIndexNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsEmptyNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsEmptyNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsHitNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsHitNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsMissNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsMissNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetAttributesNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetAttributesNV)].operands.push(OperandId, "'HitObjectAttribute'");
        InstructionDesc[enumCast(Op::OpHitObjectGetAttributesNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectExecuteShaderNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectExecuteShaderNV)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectExecuteShaderNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'InstanceId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'PrimitiveId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'GeometryIndex'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'HitKind'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].operands.push(OperandId, "'HitObject Attribute'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'InstanceId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'PrimitiveId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'GeometryIndex'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'HitKind'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'Current Time'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].operands.push(OperandId, "'HitObject Attribute'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitMotionNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'InstanceId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'PrimitiveId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'GeometryIndex'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'HitKind'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'SBT Record Index'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].operands.push(OperandId, "'HitObject Attribute'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'InstanceId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'PrimitiveId'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'GeometryIndex'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'HitKind'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'SBT Record Index'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'Current Time'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].operands.push(OperandId, "'HitObject Attribute'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordHitWithIndexMotionNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordMissNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissNV)].operands.push(OperandId, "'SBT Index'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].operands.push(OperandId, "'SBT Index'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].operands.push(OperandId, "'Current Time'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordEmptyNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordEmptyNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'Cullmask'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'Cullmask'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'Time'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionNV)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectGetClusterIdNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetClusterIdNV)].setResultAndType(true, true);
        InstructionDesc[enumCast(Op::OpHitObjectGetSpherePositionNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetSpherePositionNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetSphereRadiusNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetSphereRadiusNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetLSSPositionsNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetLSSPositionsNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetLSSRadiiNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetLSSRadiiNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsSphereHitNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsSphereHitNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsLSSHitNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsLSSHitNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexBarycentricNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexBarycentricNV)].operands.push(OperandId, "'Instance ID'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexBarycentricNV)].operands.push(OperandId, "'Geometry Index'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexBarycentricNV)].operands.push(OperandId, "'Primitive Index'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexBarycentricNV)].operands.push(OperandId, "'Barycentrics'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexBarycentricNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexPositionNV)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexPositionNV)].operands.push(OperandId, "'Instance ID'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexPositionNV)].operands.push(OperandId, "'Geometry Index'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexPositionNV)].operands.push(OperandId, "'Primitive Index'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexPositionNV)].operands.push(OperandId, "'Barycentrics'");
        InstructionDesc[enumCast(Op::OpFetchMicroTriangleVertexPositionNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpColorAttachmentReadEXT)].operands.push(OperandId, "'Attachment'");
        InstructionDesc[enumCast(Op::OpColorAttachmentReadEXT)].operands.push(OperandId, "'Sample'", true);
        InstructionDesc[enumCast(Op::OpStencilAttachmentReadEXT)].operands.push(OperandId, "'Sample'", true);
        InstructionDesc[enumCast(Op::OpDepthAttachmentReadEXT)].operands.push(OperandId, "'Sample'", true);

        InstructionDesc[enumCast(Op::OpImageSampleWeightedQCOM)].operands.push(OperandId, "'source texture'");
        InstructionDesc[enumCast(Op::OpImageSampleWeightedQCOM)].operands.push(OperandId, "'texture coordinates'");
        InstructionDesc[enumCast(Op::OpImageSampleWeightedQCOM)].operands.push(OperandId, "'weights texture'");
        InstructionDesc[enumCast(Op::OpImageSampleWeightedQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageSampleWeightedQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageBoxFilterQCOM)].operands.push(OperandId, "'source texture'");
        InstructionDesc[enumCast(Op::OpImageBoxFilterQCOM)].operands.push(OperandId, "'texture coordinates'");
        InstructionDesc[enumCast(Op::OpImageBoxFilterQCOM)].operands.push(OperandId, "'box size'");
        InstructionDesc[enumCast(Op::OpImageBoxFilterQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageBoxFilterQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageBlockMatchSADQCOM)].operands.push(OperandId, "'target texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSADQCOM)].operands.push(OperandId, "'target coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSADQCOM)].operands.push(OperandId, "'reference texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSADQCOM)].operands.push(OperandId, "'reference coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSADQCOM)].operands.push(OperandId, "'block size'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSADQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageBlockMatchSADQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageBlockMatchSSDQCOM)].operands.push(OperandId, "'target texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSSDQCOM)].operands.push(OperandId, "'target coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSSDQCOM)].operands.push(OperandId, "'reference texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSSDQCOM)].operands.push(OperandId, "'reference coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSSDQCOM)].operands.push(OperandId, "'block size'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchSSDQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageBlockMatchSSDQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSSDQCOM)].operands.push(OperandId, "'target texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSSDQCOM)].operands.push(OperandId, "'target coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSSDQCOM)].operands.push(OperandId, "'reference texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSSDQCOM)].operands.push(OperandId, "'reference coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSSDQCOM)].operands.push(OperandId, "'block size'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSSDQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSSDQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSADQCOM)].operands.push(OperandId, "'target texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSADQCOM)].operands.push(OperandId, "'target coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSADQCOM)].operands.push(OperandId, "'reference texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSADQCOM)].operands.push(OperandId, "'reference coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSADQCOM)].operands.push(OperandId, "'block size'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSADQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageBlockMatchWindowSADQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSSDQCOM)].operands.push(OperandId, "'target texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSSDQCOM)].operands.push(OperandId, "'target coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSSDQCOM)].operands.push(OperandId, "'reference texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSSDQCOM)].operands.push(OperandId, "'reference coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSSDQCOM)].operands.push(OperandId, "'block size'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSSDQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSSDQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSADQCOM)].operands.push(OperandId, "'target texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSADQCOM)].operands.push(OperandId, "'target coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSADQCOM)].operands.push(OperandId, "'reference texture'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSADQCOM)].operands.push(OperandId, "'reference coordinates'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSADQCOM)].operands.push(OperandId, "'block size'");
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSADQCOM)].operands.push(OperandImageOperands, "", true);
        InstructionDesc[enumCast(Op::OpImageBlockMatchGatherSADQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpBitCastArrayQCOM)].operands.push(OperandId, "'source array'");
        InstructionDesc[enumCast(Op::OpBitCastArrayQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpCompositeConstructCoopMatQCOM)].operands.push(OperandId, "'source array'");
        InstructionDesc[enumCast(Op::OpCompositeConstructCoopMatQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpCompositeExtractCoopMatQCOM)].operands.push(OperandId, "'source cooperative matrix'");
        InstructionDesc[enumCast(Op::OpCompositeExtractCoopMatQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpExtractSubArrayQCOM)].operands.push(OperandId, "'source array'");
        InstructionDesc[enumCast(Op::OpExtractSubArrayQCOM)].operands.push(OperandId, "'start index'");
        InstructionDesc[enumCast(Op::OpExtractSubArrayQCOM)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpConstantCompositeReplicateEXT)].operands.push(OperandId, "'Value'");
        InstructionDesc[enumCast(Op::OpSpecConstantCompositeReplicateEXT)].operands.push(OperandId, "'Value'");
        InstructionDesc[enumCast(Op::OpCompositeConstructReplicateEXT)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixConvertNV)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixTransposeNV)].operands.push(OperandId, "'Matrix'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixReduceNV)].operands.push(OperandId, "'Matrix'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixReduceNV)].operands.push(OperandLiteralNumber, "'ReduceMask'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixReduceNV)].operands.push(OperandId, "'CombineFunc'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixPerElementOpNV)].operands.push(OperandId, "'Matrix'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixPerElementOpNV)].operands.push(OperandId, "'Operation'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixPerElementOpNV)].operands.push(OperandVariableIds, "'Operands'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadTensorNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadTensorNV)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadTensorNV)].operands.push(OperandId, "'TensorLayout'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadTensorNV)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixLoadTensorNV)].operands.push(OperandTensorAddressingOperands, "'Tensor Addressing Operands'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreTensorNV)].operands.push(OperandId, "'Pointer'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreTensorNV)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreTensorNV)].operands.push(OperandId, "'TensorLayout'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreTensorNV)].operands.push(OperandMemoryAccess, "'Memory Access'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixStoreTensorNV)].operands.push(OperandTensorAddressingOperands, "'Tensor Addressing Operands'");

        InstructionDesc[enumCast(Op::OpCooperativeMatrixReduceNV)].operands.push(OperandId, "'Matrix'");
        InstructionDesc[enumCast(Op::OpCooperativeMatrixReduceNV)].operands.push(OperandLiteralNumber, "'ReduceMask'");

        InstructionDesc[enumCast(Op::OpTypeTensorLayoutNV)].operands.push(OperandId, "'Dim'");
        InstructionDesc[enumCast(Op::OpTypeTensorLayoutNV)].operands.push(OperandId, "'ClampMode'");

        InstructionDesc[enumCast(Op::OpTypeTensorViewNV)].operands.push(OperandId, "'Dim'");
        InstructionDesc[enumCast(Op::OpTypeTensorViewNV)].operands.push(OperandId, "'HasDimensions'");
        InstructionDesc[enumCast(Op::OpTypeTensorViewNV)].operands.push(OperandVariableIds, "'p'");

        InstructionDesc[enumCast(Op::OpTensorLayoutSetBlockSizeNV)].operands.push(OperandId, "'TensorLayout'");
        InstructionDesc[enumCast(Op::OpTensorLayoutSetBlockSizeNV)].operands.push(OperandVariableIds, "'BlockSize'");

        InstructionDesc[enumCast(Op::OpTensorLayoutSetDimensionNV)].operands.push(OperandId, "'TensorLayout'");
        InstructionDesc[enumCast(Op::OpTensorLayoutSetDimensionNV)].operands.push(OperandVariableIds, "'Dim'");

        InstructionDesc[enumCast(Op::OpTensorLayoutSetStrideNV)].operands.push(OperandId, "'TensorLayout'");
        InstructionDesc[enumCast(Op::OpTensorLayoutSetStrideNV)].operands.push(OperandVariableIds, "'Stride'");

        InstructionDesc[enumCast(Op::OpTensorLayoutSliceNV)].operands.push(OperandId, "'TensorLayout'");
        InstructionDesc[enumCast(Op::OpTensorLayoutSliceNV)].operands.push(OperandVariableIds, "'Operands'");

        InstructionDesc[enumCast(Op::OpTensorLayoutSetClampValueNV)].operands.push(OperandId, "'TensorLayout'");
        InstructionDesc[enumCast(Op::OpTensorLayoutSetClampValueNV)].operands.push(OperandId, "'Value'");

        InstructionDesc[enumCast(Op::OpTensorViewSetDimensionNV)].operands.push(OperandId, "'TensorView'");
        InstructionDesc[enumCast(Op::OpTensorViewSetDimensionNV)].operands.push(OperandVariableIds, "'Dim'");

        InstructionDesc[enumCast(Op::OpTensorViewSetStrideNV)].operands.push(OperandId, "'TensorView'");
        InstructionDesc[enumCast(Op::OpTensorViewSetStrideNV)].operands.push(OperandVariableIds, "'Stride'");

        InstructionDesc[enumCast(Op::OpTensorViewSetClipNV)].operands.push(OperandId, "'TensorView'");
        InstructionDesc[enumCast(Op::OpTensorViewSetClipNV)].operands.push(OperandId, "'ClipRowOffset'");
        InstructionDesc[enumCast(Op::OpTensorViewSetClipNV)].operands.push(OperandId, "'ClipRowSpan'");
        InstructionDesc[enumCast(Op::OpTensorViewSetClipNV)].operands.push(OperandId, "'ClipColOffset'");
        InstructionDesc[enumCast(Op::OpTensorViewSetClipNV)].operands.push(OperandId, "'ClipColSpan'");

        InstructionDesc[enumCast(Op::OpSDotKHR)].operands.push(OperandId, "'Vector1'");
        InstructionDesc[enumCast(Op::OpSDotKHR)].operands.push(OperandId, "'Vector2'");
        InstructionDesc[enumCast(Op::OpSDotKHR)].operands.push(OperandLiteralNumber, "'PackedVectorFormat'");

        InstructionDesc[enumCast(Op::OpUDotKHR)].operands.push(OperandId, "'Vector1'");
        InstructionDesc[enumCast(Op::OpUDotKHR)].operands.push(OperandId, "'Vector2'");
        InstructionDesc[enumCast(Op::OpUDotKHR)].operands.push(OperandLiteralNumber, "'PackedVectorFormat'");

        InstructionDesc[enumCast(Op::OpSUDotKHR)].operands.push(OperandId, "'Vector1'");
        InstructionDesc[enumCast(Op::OpSUDotKHR)].operands.push(OperandId, "'Vector2'");
        InstructionDesc[enumCast(Op::OpSUDotKHR)].operands.push(OperandLiteralNumber, "'PackedVectorFormat'");

        InstructionDesc[enumCast(Op::OpSDotAccSatKHR)].operands.push(OperandId, "'Vector1'");
        InstructionDesc[enumCast(Op::OpSDotAccSatKHR)].operands.push(OperandId, "'Vector2'");
        InstructionDesc[enumCast(Op::OpSDotAccSatKHR)].operands.push(OperandId, "'Accumulator'");
        InstructionDesc[enumCast(Op::OpSDotAccSatKHR)].operands.push(OperandLiteralNumber, "'PackedVectorFormat'");

        InstructionDesc[enumCast(Op::OpUDotAccSatKHR)].operands.push(OperandId, "'Vector1'");
        InstructionDesc[enumCast(Op::OpUDotAccSatKHR)].operands.push(OperandId, "'Vector2'");
        InstructionDesc[enumCast(Op::OpUDotAccSatKHR)].operands.push(OperandId, "'Accumulator'");
        InstructionDesc[enumCast(Op::OpUDotAccSatKHR)].operands.push(OperandLiteralNumber, "'PackedVectorFormat'");

        InstructionDesc[enumCast(Op::OpSUDotAccSatKHR)].operands.push(OperandId, "'Vector1'");
        InstructionDesc[enumCast(Op::OpSUDotAccSatKHR)].operands.push(OperandId, "'Vector2'");
        InstructionDesc[enumCast(Op::OpSUDotAccSatKHR)].operands.push(OperandId, "'Accumulator'");
        InstructionDesc[enumCast(Op::OpSUDotAccSatKHR)].operands.push(OperandLiteralNumber, "'PackedVectorFormat'");

        InstructionDesc[enumCast(Op::OpTypeTensorARM)].operands.push(OperandId, "'Element Type'");
        InstructionDesc[enumCast(Op::OpTypeTensorARM)].operands.push(OperandId, "'Rank'");

        InstructionDesc[enumCast(Op::OpTensorReadARM)].operands.push(OperandId, "'Tensor'");
        InstructionDesc[enumCast(Op::OpTensorReadARM)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpTensorReadARM)].operands.push(OperandLiteralNumber, "'Tensor Operand'", true);
        InstructionDesc[enumCast(Op::OpTensorReadARM)].operands.push(OperandVariableIds, "'Tensor Operands'");

        InstructionDesc[enumCast(Op::OpTensorWriteARM)].operands.push(OperandId, "'Tensor'");
        InstructionDesc[enumCast(Op::OpTensorWriteARM)].operands.push(OperandId, "'Coordinate'");
        InstructionDesc[enumCast(Op::OpTensorWriteARM)].operands.push(OperandId, "'Object'");
        InstructionDesc[enumCast(Op::OpTensorWriteARM)].operands.push(OperandLiteralNumber, "'Tensor Operand'", true);
        InstructionDesc[enumCast(Op::OpTensorWriteARM)].operands.push(OperandVariableIds, "'Tensor Operands'");

        InstructionDesc[enumCast(Op::OpTensorQuerySizeARM)].operands.push(OperandId, "'Tensor'");
        InstructionDesc[enumCast(Op::OpTensorQuerySizeARM)].operands.push(OperandId, "'Dimension'", true);

        InstructionDesc[enumCast(Op::OpTypeHitObjectEXT)].setResultAndType(true, false);

        InstructionDesc[enumCast(Op::OpHitObjectGetShaderRecordBufferHandleNV)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetShaderRecordBufferHandleNV)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetShaderRecordBufferHandleEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetShaderRecordBufferHandleEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpReorderThreadWithHintEXT)].operands.push(OperandId, "'Hint'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHintEXT)].operands.push(OperandId, "'Bits'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHintEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectEXT)].operands.push(OperandId, "'Hint'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectEXT)].operands.push(OperandId, "'Bits'");
        InstructionDesc[enumCast(Op::OpReorderThreadWithHitObjectEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectGetCurrentTimeEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetCurrentTimeEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetHitKindEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetHitKindEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetPrimitiveIndexEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetPrimitiveIndexEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetGeometryIndexEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetGeometryIndexEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceIdEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceIdEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceCustomIndexEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetInstanceCustomIndexEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayDirectionEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayDirectionEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayOriginEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetObjectRayOriginEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayDirectionEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayDirectionEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayOriginEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetWorldRayOriginEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetWorldToObjectEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetWorldToObjectEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetObjectToWorldEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetObjectToWorldEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMaxEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMaxEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMinEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetRayTMinEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetRayFlagsEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetRayFlagsEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetShaderBindingTableRecordIndexEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetShaderBindingTableRecordIndexEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsEmptyEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsEmptyEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsHitEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsHitEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectIsMissEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectIsMissEXT)].setResultAndType(true, true);

        InstructionDesc[enumCast(Op::OpHitObjectGetAttributesEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetAttributesEXT)].operands.push(OperandId, "'HitObjectAttribute'");
        InstructionDesc[enumCast(Op::OpHitObjectGetAttributesEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectExecuteShaderEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectExecuteShaderEXT)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectExecuteShaderEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].operands.push(OperandId, "'SBT Index'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'SBT Index'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].operands.push(OperandId, "'Current Time'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordMissMotionEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordEmptyEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordEmptyEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'Cullmask'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'Cullmask'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'Time'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceRayMotionEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectSetShaderBindingTableRecordIndexEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectSetShaderBindingTableRecordIndexEXT)].operands.push(OperandId, "'SBT Record Index'");
        InstructionDesc[enumCast(Op::OpHitObjectSetShaderBindingTableRecordIndexEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectReorderExecuteShaderEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectReorderExecuteShaderEXT)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectReorderExecuteShaderEXT)].operands.push(OperandId, "'Hint'");
        InstructionDesc[enumCast(Op::OpHitObjectReorderExecuteShaderEXT)].operands.push(OperandId, "'Bits'");
        InstructionDesc[enumCast(Op::OpHitObjectReorderExecuteShaderEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Cullmask'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Hint'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].operands.push(OperandId, "'Bits'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceReorderExecuteEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Acceleration Structure'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'RayFlags'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Cullmask'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'SBT Record Offset'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'SBT Record Stride'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Miss Index'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Origin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'TMin'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Direction'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'TMax'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Time'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Payload'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Hint'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].operands.push(OperandId, "'Bits'");
        InstructionDesc[enumCast(Op::OpHitObjectTraceMotionReorderExecuteEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectRecordFromQueryEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordFromQueryEXT)].operands.push(OperandId, "'RayQuery'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordFromQueryEXT)].operands.push(OperandId, "'SBT Record Index'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordFromQueryEXT)].operands.push(OperandId, "'HitObjectAttribute'");
        InstructionDesc[enumCast(Op::OpHitObjectRecordFromQueryEXT)].setResultAndType(false, false);

        InstructionDesc[enumCast(Op::OpHitObjectGetIntersectionTriangleVertexPositionsEXT)].operands.push(OperandId, "'HitObject'");
        InstructionDesc[enumCast(Op::OpHitObjectGetIntersectionTriangleVertexPositionsEXT)].setResultAndType(true, true);

    });
}

} // end spv namespace
