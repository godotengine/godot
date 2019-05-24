// Copyright (c) 2018 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "source/val/validate.h"

#include <algorithm>

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateEntryPoint(ValidationState_t& _, const Instruction* inst) {
  const auto entry_point_id = inst->GetOperandAs<uint32_t>(1);
  auto entry_point = _.FindDef(entry_point_id);
  if (!entry_point || SpvOpFunction != entry_point->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpEntryPoint Entry Point <id> '" << _.getIdName(entry_point_id)
           << "' is not a function.";
  }
  // don't check kernel function signatures
  const SpvExecutionModel execution_model =
      inst->GetOperandAs<SpvExecutionModel>(0);
  if (execution_model != SpvExecutionModelKernel) {
    // TODO: Check the entry point signature is void main(void), may be subject
    // to change
    const auto entry_point_type_id = entry_point->GetOperandAs<uint32_t>(3);
    const auto entry_point_type = _.FindDef(entry_point_type_id);
    if (!entry_point_type || 3 != entry_point_type->words().size()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "OpEntryPoint Entry Point <id> '" << _.getIdName(entry_point_id)
             << "'s function parameter count is not zero.";
    }
  }

  auto return_type = _.FindDef(entry_point->type_id());
  if (!return_type || SpvOpTypeVoid != return_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpEntryPoint Entry Point <id> '" << _.getIdName(entry_point_id)
           << "'s function return type is not void.";
  }

  const auto* execution_modes = _.GetExecutionModes(entry_point_id);
  if (_.HasCapability(SpvCapabilityShader)) {
    switch (execution_model) {
      case SpvExecutionModelFragment:
        if (execution_modes &&
            execution_modes->count(SpvExecutionModeOriginUpperLeft) &&
            execution_modes->count(SpvExecutionModeOriginLowerLeft)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points can only specify "
                    "one of OriginUpperLeft or OriginLowerLeft execution "
                    "modes.";
        }
        if (!execution_modes ||
            (!execution_modes->count(SpvExecutionModeOriginUpperLeft) &&
             !execution_modes->count(SpvExecutionModeOriginLowerLeft))) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points require either an "
                    "OriginUpperLeft or OriginLowerLeft execution mode.";
        }
        if (execution_modes &&
            1 < std::count_if(execution_modes->begin(), execution_modes->end(),
                              [](const SpvExecutionMode& mode) {
                                switch (mode) {
                                  case SpvExecutionModeDepthGreater:
                                  case SpvExecutionModeDepthLess:
                                  case SpvExecutionModeDepthUnchanged:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points can specify at most "
                    "one of DepthGreater, DepthLess or DepthUnchanged "
                    "execution modes.";
        }
        break;
      case SpvExecutionModelTessellationControl:
      case SpvExecutionModelTessellationEvaluation:
        if (execution_modes &&
            1 < std::count_if(execution_modes->begin(), execution_modes->end(),
                              [](const SpvExecutionMode& mode) {
                                switch (mode) {
                                  case SpvExecutionModeSpacingEqual:
                                  case SpvExecutionModeSpacingFractionalEven:
                                  case SpvExecutionModeSpacingFractionalOdd:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Tessellation execution model entry points can specify at "
                    "most one of SpacingEqual, SpacingFractionalOdd or "
                    "SpacingFractionalEven execution modes.";
        }
        if (execution_modes &&
            1 < std::count_if(execution_modes->begin(), execution_modes->end(),
                              [](const SpvExecutionMode& mode) {
                                switch (mode) {
                                  case SpvExecutionModeTriangles:
                                  case SpvExecutionModeQuads:
                                  case SpvExecutionModeIsolines:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Tessellation execution model entry points can specify at "
                    "most one of Triangles, Quads or Isolines execution modes.";
        }
        if (execution_modes &&
            1 < std::count_if(execution_modes->begin(), execution_modes->end(),
                              [](const SpvExecutionMode& mode) {
                                switch (mode) {
                                  case SpvExecutionModeVertexOrderCw:
                                  case SpvExecutionModeVertexOrderCcw:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Tessellation execution model entry points can specify at "
                    "most one of VertexOrderCw or VertexOrderCcw execution "
                    "modes.";
        }
        break;
      case SpvExecutionModelGeometry:
        if (!execution_modes ||
            1 != std::count_if(execution_modes->begin(), execution_modes->end(),
                               [](const SpvExecutionMode& mode) {
                                 switch (mode) {
                                   case SpvExecutionModeInputPoints:
                                   case SpvExecutionModeInputLines:
                                   case SpvExecutionModeInputLinesAdjacency:
                                   case SpvExecutionModeTriangles:
                                   case SpvExecutionModeInputTrianglesAdjacency:
                                     return true;
                                   default:
                                     return false;
                                 }
                               })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Geometry execution model entry points must specify "
                    "exactly one of InputPoints, InputLines, "
                    "InputLinesAdjacency, Triangles or InputTrianglesAdjacency "
                    "execution modes.";
        }
        if (!execution_modes ||
            1 != std::count_if(execution_modes->begin(), execution_modes->end(),
                               [](const SpvExecutionMode& mode) {
                                 switch (mode) {
                                   case SpvExecutionModeOutputPoints:
                                   case SpvExecutionModeOutputLineStrip:
                                   case SpvExecutionModeOutputTriangleStrip:
                                     return true;
                                   default:
                                     return false;
                                 }
                               })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Geometry execution model entry points must specify "
                    "exactly one of OutputPoints, OutputLineStrip or "
                    "OutputTriangleStrip execution modes.";
        }
        break;
      default:
        break;
    }
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    switch (execution_model) {
      case SpvExecutionModelGLCompute:
        if (!execution_modes ||
            !execution_modes->count(SpvExecutionModeLocalSize)) {
          bool ok = false;
          for (auto& i : _.ordered_instructions()) {
            if (i.opcode() == SpvOpDecorate) {
              if (i.operands().size() > 2) {
                if (i.GetOperandAs<SpvDecoration>(1) == SpvDecorationBuiltIn &&
                    i.GetOperandAs<SpvBuiltIn>(2) == SpvBuiltInWorkgroupSize) {
                  ok = true;
                  break;
                }
              }
            }
          }
          if (!ok) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << "In the Vulkan environment, GLCompute execution model "
                      "entry points require either the LocalSize execution "
                      "mode or an object decorated with WorkgroupSize must be "
                      "specified.";
          }
        }
        break;
      default:
        break;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExecutionMode(ValidationState_t& _,
                                   const Instruction* inst) {
  const auto entry_point_id = inst->GetOperandAs<uint32_t>(0);
  const auto found = std::find(_.entry_points().cbegin(),
                               _.entry_points().cend(), entry_point_id);
  if (found == _.entry_points().cend()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpExecutionMode Entry Point <id> '"
           << _.getIdName(entry_point_id)
           << "' is not the Entry Point "
              "operand of an OpEntryPoint.";
  }

  const auto mode = inst->GetOperandAs<SpvExecutionMode>(1);
  const auto* models = _.GetExecutionModels(entry_point_id);
  switch (mode) {
    case SpvExecutionModeInvocations:
    case SpvExecutionModeInputPoints:
    case SpvExecutionModeInputLines:
    case SpvExecutionModeInputLinesAdjacency:
    case SpvExecutionModeInputTrianglesAdjacency:
    case SpvExecutionModeOutputLineStrip:
    case SpvExecutionModeOutputTriangleStrip:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExecutionModel& model) {
                         return model == SpvExecutionModelGeometry;
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with the Geometry execution "
                  "model.";
      }
      break;
    case SpvExecutionModeOutputPoints:
      if (!std::all_of(models->begin(), models->end(),
                       [&_](const SpvExecutionModel& model) {
                         switch (model) {
                           case SpvExecutionModelGeometry:
                             return true;
                           case SpvExecutionModelMeshNV:
                             return _.HasCapability(SpvCapabilityMeshShadingNV);
                           default:
                             return false;
                         }
                       })) {
        if (_.HasCapability(SpvCapabilityMeshShadingNV)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with the Geometry or "
                    "MeshNV execution model.";
        } else {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with the Geometry "
                    "execution "
                    "model.";
        }
      }
      break;
    case SpvExecutionModeSpacingEqual:
    case SpvExecutionModeSpacingFractionalEven:
    case SpvExecutionModeSpacingFractionalOdd:
    case SpvExecutionModeVertexOrderCw:
    case SpvExecutionModeVertexOrderCcw:
    case SpvExecutionModePointMode:
    case SpvExecutionModeQuads:
    case SpvExecutionModeIsolines:
      if (!std::all_of(
              models->begin(), models->end(),
              [](const SpvExecutionModel& model) {
                return (model == SpvExecutionModelTessellationControl) ||
                       (model == SpvExecutionModelTessellationEvaluation);
              })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with a tessellation "
                  "execution model.";
      }
      break;
    case SpvExecutionModeTriangles:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExecutionModel& model) {
                         switch (model) {
                           case SpvExecutionModelGeometry:
                           case SpvExecutionModelTessellationControl:
                           case SpvExecutionModelTessellationEvaluation:
                             return true;
                           default:
                             return false;
                         }
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with a Geometry or "
                  "tessellation execution model.";
      }
      break;
    case SpvExecutionModeOutputVertices:
      if (!std::all_of(models->begin(), models->end(),
                       [&_](const SpvExecutionModel& model) {
                         switch (model) {
                           case SpvExecutionModelGeometry:
                           case SpvExecutionModelTessellationControl:
                           case SpvExecutionModelTessellationEvaluation:
                             return true;
                           case SpvExecutionModelMeshNV:
                             return _.HasCapability(SpvCapabilityMeshShadingNV);
                           default:
                             return false;
                         }
                       })) {
        if (_.HasCapability(SpvCapabilityMeshShadingNV)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with a Geometry, "
                    "tessellation or MeshNV execution model.";
        } else {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with a Geometry or "
                    "tessellation execution model.";
        }
      }
      break;
    case SpvExecutionModePixelCenterInteger:
    case SpvExecutionModeOriginUpperLeft:
    case SpvExecutionModeOriginLowerLeft:
    case SpvExecutionModeEarlyFragmentTests:
    case SpvExecutionModeDepthReplacing:
    case SpvExecutionModeDepthGreater:
    case SpvExecutionModeDepthLess:
    case SpvExecutionModeDepthUnchanged:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExecutionModel& model) {
                         return model == SpvExecutionModelFragment;
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with the Fragment execution "
                  "model.";
      }
      break;
    case SpvExecutionModeLocalSizeHint:
    case SpvExecutionModeVecTypeHint:
    case SpvExecutionModeContractionOff:
    case SpvExecutionModeLocalSizeHintId:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExecutionModel& model) {
                         return model == SpvExecutionModelKernel;
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with the Kernel execution "
                  "model.";
      }
      break;
    case SpvExecutionModeLocalSize:
    case SpvExecutionModeLocalSizeId:
      if (!std::all_of(models->begin(), models->end(),
                       [&_](const SpvExecutionModel& model) {
                         switch (model) {
                           case SpvExecutionModelKernel:
                           case SpvExecutionModelGLCompute:
                             return true;
                           case SpvExecutionModelTaskNV:
                           case SpvExecutionModelMeshNV:
                             return _.HasCapability(SpvCapabilityMeshShadingNV);
                           default:
                             return false;
                         }
                       })) {
        if (_.HasCapability(SpvCapabilityMeshShadingNV)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with a Kernel, GLCompute, "
                    "MeshNV, or TaskNV execution model.";
        } else {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with a Kernel or "
                    "GLCompute "
                    "execution model.";
        }
      }
    default:
      break;
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if (mode == SpvExecutionModeOriginLowerLeft) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "In the Vulkan environment, the OriginLowerLeft execution mode "
                "must not be used.";
    }
    if (mode == SpvExecutionModePixelCenterInteger) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "In the Vulkan environment, the PixelCenterInteger execution "
                "mode must not be used.";
    }
  }

  if (spvIsWebGPUEnv(_.context()->target_env)) {
    if (mode != SpvExecutionModeOriginUpperLeft &&
        mode != SpvExecutionModeDepthReplacing &&
        mode != SpvExecutionModeDepthGreater &&
        mode != SpvExecutionModeDepthLess &&
        mode != SpvExecutionModeDepthUnchanged &&
        mode != SpvExecutionModeLocalSize &&
        mode != SpvExecutionModeLocalSizeHint) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Execution mode must be one of OriginUpperLeft, "
                "DepthReplacing, DepthGreater, DepthLess, DepthUnchanged, "
                "LocalSize, or LocalSizeHint for WebGPU environment.";
    }
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ModeSettingPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpEntryPoint:
      if (auto error = ValidateEntryPoint(_, inst)) return error;
      break;
    case SpvOpExecutionMode:
    case SpvOpExecutionModeId:
      if (auto error = ValidateExecutionMode(_, inst)) return error;
      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
