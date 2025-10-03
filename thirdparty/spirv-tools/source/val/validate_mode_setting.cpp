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
#include <algorithm>

#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateEntryPoint(ValidationState_t& _, const Instruction* inst) {
  const auto entry_point_id = inst->GetOperandAs<uint32_t>(1);
  auto entry_point = _.FindDef(entry_point_id);
  if (!entry_point || spv::Op::OpFunction != entry_point->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpEntryPoint Entry Point <id> " << _.getIdName(entry_point_id)
           << " is not a function.";
  }

  // Only check the shader execution models
  const spv::ExecutionModel execution_model =
      inst->GetOperandAs<spv::ExecutionModel>(0);
  if (execution_model != spv::ExecutionModel::Kernel) {
    const auto entry_point_type_id = entry_point->GetOperandAs<uint32_t>(3);
    const auto entry_point_type = _.FindDef(entry_point_type_id);
    if (!entry_point_type || 3 != entry_point_type->words().size()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << _.VkErrorID(4633) << "OpEntryPoint Entry Point <id> "
             << _.getIdName(entry_point_id)
             << "s function parameter count is not zero.";
    }
  }

  auto return_type = _.FindDef(entry_point->type_id());
  if (!return_type || spv::Op::OpTypeVoid != return_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << _.VkErrorID(4633) << "OpEntryPoint Entry Point <id> "
           << _.getIdName(entry_point_id)
           << "s function return type is not void.";
  }

  const auto* execution_modes = _.GetExecutionModes(entry_point_id);
  if (_.HasCapability(spv::Capability::Shader)) {
    switch (execution_model) {
      case spv::ExecutionModel::Fragment:
        if (execution_modes &&
            execution_modes->count(spv::ExecutionMode::OriginUpperLeft) &&
            execution_modes->count(spv::ExecutionMode::OriginLowerLeft)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points can only specify "
                    "one of OriginUpperLeft or OriginLowerLeft execution "
                    "modes.";
        }
        if (!execution_modes ||
            (!execution_modes->count(spv::ExecutionMode::OriginUpperLeft) &&
             !execution_modes->count(spv::ExecutionMode::OriginLowerLeft))) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points require either an "
                    "OriginUpperLeft or OriginLowerLeft execution mode.";
        }
        if (execution_modes &&
            1 < std::count_if(execution_modes->begin(), execution_modes->end(),
                              [](const spv::ExecutionMode& mode) {
                                switch (mode) {
                                  case spv::ExecutionMode::DepthGreater:
                                  case spv::ExecutionMode::DepthLess:
                                  case spv::ExecutionMode::DepthUnchanged:
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
        if (execution_modes &&
            1 < std::count_if(
                    execution_modes->begin(), execution_modes->end(),
                    [](const spv::ExecutionMode& mode) {
                      switch (mode) {
                        case spv::ExecutionMode::PixelInterlockOrderedEXT:
                        case spv::ExecutionMode::PixelInterlockUnorderedEXT:
                        case spv::ExecutionMode::SampleInterlockOrderedEXT:
                        case spv::ExecutionMode::SampleInterlockUnorderedEXT:
                        case spv::ExecutionMode::ShadingRateInterlockOrderedEXT:
                        case spv::ExecutionMode::
                            ShadingRateInterlockUnorderedEXT:
                          return true;
                        default:
                          return false;
                      }
                    })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points can specify at most "
                    "one fragment shader interlock execution mode.";
        }
        if (execution_modes &&
            1 < std::count_if(
                    execution_modes->begin(), execution_modes->end(),
                    [](const spv::ExecutionMode& mode) {
                      switch (mode) {
                        case spv::ExecutionMode::StencilRefUnchangedFrontAMD:
                        case spv::ExecutionMode::StencilRefLessFrontAMD:
                        case spv::ExecutionMode::StencilRefGreaterFrontAMD:
                          return true;
                        default:
                          return false;
                      }
                    })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points can specify at most "
                    "one of StencilRefUnchangedFrontAMD, "
                    "StencilRefLessFrontAMD or StencilRefGreaterFrontAMD "
                    "execution modes.";
        }
        if (execution_modes &&
            1 < std::count_if(
                    execution_modes->begin(), execution_modes->end(),
                    [](const spv::ExecutionMode& mode) {
                      switch (mode) {
                        case spv::ExecutionMode::StencilRefUnchangedBackAMD:
                        case spv::ExecutionMode::StencilRefLessBackAMD:
                        case spv::ExecutionMode::StencilRefGreaterBackAMD:
                          return true;
                        default:
                          return false;
                      }
                    })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Fragment execution model entry points can specify at most "
                    "one of StencilRefUnchangedBackAMD, "
                    "StencilRefLessBackAMD or StencilRefGreaterBackAMD "
                    "execution modes.";
        }
        break;
      case spv::ExecutionModel::TessellationControl:
      case spv::ExecutionModel::TessellationEvaluation:
        if (execution_modes &&
            1 < std::count_if(
                    execution_modes->begin(), execution_modes->end(),
                    [](const spv::ExecutionMode& mode) {
                      switch (mode) {
                        case spv::ExecutionMode::SpacingEqual:
                        case spv::ExecutionMode::SpacingFractionalEven:
                        case spv::ExecutionMode::SpacingFractionalOdd:
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
                              [](const spv::ExecutionMode& mode) {
                                switch (mode) {
                                  case spv::ExecutionMode::Triangles:
                                  case spv::ExecutionMode::Quads:
                                  case spv::ExecutionMode::Isolines:
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
                              [](const spv::ExecutionMode& mode) {
                                switch (mode) {
                                  case spv::ExecutionMode::VertexOrderCw:
                                  case spv::ExecutionMode::VertexOrderCcw:
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
      case spv::ExecutionModel::Geometry:
        if (!execution_modes ||
            1 != std::count_if(
                     execution_modes->begin(), execution_modes->end(),
                     [](const spv::ExecutionMode& mode) {
                       switch (mode) {
                         case spv::ExecutionMode::InputPoints:
                         case spv::ExecutionMode::InputLines:
                         case spv::ExecutionMode::InputLinesAdjacency:
                         case spv::ExecutionMode::Triangles:
                         case spv::ExecutionMode::InputTrianglesAdjacency:
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
                               [](const spv::ExecutionMode& mode) {
                                 switch (mode) {
                                   case spv::ExecutionMode::OutputPoints:
                                   case spv::ExecutionMode::OutputLineStrip:
                                   case spv::ExecutionMode::OutputTriangleStrip:
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
      case spv::ExecutionModel::MeshEXT:
        if (!execution_modes ||
            1 != std::count_if(execution_modes->begin(), execution_modes->end(),
                               [](const spv::ExecutionMode& mode) {
                                 switch (mode) {
                                   case spv::ExecutionMode::OutputPoints:
                                   case spv::ExecutionMode::OutputLinesEXT:
                                   case spv::ExecutionMode::OutputTrianglesEXT:
                                     return true;
                                   default:
                                     return false;
                                 }
                               })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "MeshEXT execution model entry points must specify exactly "
                    "one of OutputPoints, OutputLinesEXT, or "
                    "OutputTrianglesEXT Execution Modes.";
        } else if (2 != std::count_if(
                            execution_modes->begin(), execution_modes->end(),
                            [](const spv::ExecutionMode& mode) {
                              switch (mode) {
                                case spv::ExecutionMode::OutputPrimitivesEXT:
                                case spv::ExecutionMode::OutputVertices:
                                  return true;
                                default:
                                  return false;
                              }
                            })) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "MeshEXT execution model entry points must specify both "
                    "OutputPrimitivesEXT and OutputVertices Execution Modes.";
        }
        break;
      default:
        break;
    }
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    switch (execution_model) {
      case spv::ExecutionModel::GLCompute:
        if (!execution_modes ||
            !execution_modes->count(spv::ExecutionMode::LocalSize)) {
          bool ok = false;
          for (auto& i : _.ordered_instructions()) {
            if (i.opcode() == spv::Op::OpDecorate) {
              if (i.operands().size() > 2) {
                if (i.GetOperandAs<spv::Decoration>(1) ==
                        spv::Decoration::BuiltIn &&
                    i.GetOperandAs<spv::BuiltIn>(2) ==
                        spv::BuiltIn::WorkgroupSize) {
                  ok = true;
                  break;
                }
              }
            }
            if (i.opcode() == spv::Op::OpExecutionModeId) {
              const auto mode = i.GetOperandAs<spv::ExecutionMode>(1);
              if (mode == spv::ExecutionMode::LocalSizeId) {
                ok = true;
                break;
              }
            }
          }
          if (!ok) {
            return _.diag(SPV_ERROR_INVALID_DATA, inst)
                   << _.VkErrorID(6426)
                   << "In the Vulkan environment, GLCompute execution model "
                      "entry points require either the LocalSize or "
                      "LocalSizeId execution mode or an object decorated with "
                      "WorkgroupSize must be specified.";
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
           << "OpExecutionMode Entry Point <id> " << _.getIdName(entry_point_id)
           << " is not the Entry Point "
              "operand of an OpEntryPoint.";
  }

  const auto mode = inst->GetOperandAs<spv::ExecutionMode>(1);
  if (inst->opcode() == spv::Op::OpExecutionModeId) {
    size_t operand_count = inst->operands().size();
    for (size_t i = 2; i < operand_count; ++i) {
      const auto operand_id = inst->GetOperandAs<uint32_t>(2);
      const auto* operand_inst = _.FindDef(operand_id);
      if (mode == spv::ExecutionMode::SubgroupsPerWorkgroupId ||
          mode == spv::ExecutionMode::LocalSizeHintId ||
          mode == spv::ExecutionMode::LocalSizeId) {
        if (!spvOpcodeIsConstant(operand_inst->opcode())) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "For OpExecutionModeId all Extra Operand ids must be "
                    "constant "
                    "instructions.";
        }
      } else {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpExecutionModeId is only valid when the Mode operand is an "
                  "execution mode that takes Extra Operands that are id "
                  "operands.";
      }
    }
  } else if (mode == spv::ExecutionMode::SubgroupsPerWorkgroupId ||
             mode == spv::ExecutionMode::LocalSizeHintId ||
             mode == spv::ExecutionMode::LocalSizeId) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "OpExecutionMode is only valid when the Mode operand is an "
              "execution mode that takes no Extra Operands, or takes Extra "
              "Operands that are not id operands.";
  }

  const auto* models = _.GetExecutionModels(entry_point_id);
  switch (mode) {
    case spv::ExecutionMode::Invocations:
    case spv::ExecutionMode::InputPoints:
    case spv::ExecutionMode::InputLines:
    case spv::ExecutionMode::InputLinesAdjacency:
    case spv::ExecutionMode::InputTrianglesAdjacency:
    case spv::ExecutionMode::OutputLineStrip:
    case spv::ExecutionMode::OutputTriangleStrip:
      if (!std::all_of(models->begin(), models->end(),
                       [](const spv::ExecutionModel& model) {
                         return model == spv::ExecutionModel::Geometry;
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with the Geometry execution "
                  "model.";
      }
      break;
    case spv::ExecutionMode::OutputPoints:
      if (!std::all_of(
              models->begin(), models->end(),
              [&_](const spv::ExecutionModel& model) {
                switch (model) {
                  case spv::ExecutionModel::Geometry:
                    return true;
                  case spv::ExecutionModel::MeshNV:
                    return _.HasCapability(spv::Capability::MeshShadingNV);
                  case spv::ExecutionModel::MeshEXT:
                    return _.HasCapability(spv::Capability::MeshShadingEXT);
                  default:
                    return false;
                }
              })) {
        if (_.HasCapability(spv::Capability::MeshShadingNV) ||
            _.HasCapability(spv::Capability::MeshShadingEXT)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with the Geometry "
                    "MeshNV or MeshEXT execution model.";
        } else {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with the Geometry "
                    "execution "
                    "model.";
        }
      }
      break;
    case spv::ExecutionMode::SpacingEqual:
    case spv::ExecutionMode::SpacingFractionalEven:
    case spv::ExecutionMode::SpacingFractionalOdd:
    case spv::ExecutionMode::VertexOrderCw:
    case spv::ExecutionMode::VertexOrderCcw:
    case spv::ExecutionMode::PointMode:
    case spv::ExecutionMode::Quads:
    case spv::ExecutionMode::Isolines:
      if (!std::all_of(
              models->begin(), models->end(),
              [](const spv::ExecutionModel& model) {
                return (model == spv::ExecutionModel::TessellationControl) ||
                       (model == spv::ExecutionModel::TessellationEvaluation);
              })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with a tessellation "
                  "execution model.";
      }
      break;
    case spv::ExecutionMode::Triangles:
      if (!std::all_of(models->begin(), models->end(),
                       [](const spv::ExecutionModel& model) {
                         switch (model) {
                           case spv::ExecutionModel::Geometry:
                           case spv::ExecutionModel::TessellationControl:
                           case spv::ExecutionModel::TessellationEvaluation:
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
    case spv::ExecutionMode::OutputVertices:
      if (!std::all_of(
              models->begin(), models->end(),
              [&_](const spv::ExecutionModel& model) {
                switch (model) {
                  case spv::ExecutionModel::Geometry:
                  case spv::ExecutionModel::TessellationControl:
                  case spv::ExecutionModel::TessellationEvaluation:
                    return true;
                  case spv::ExecutionModel::MeshNV:
                    return _.HasCapability(spv::Capability::MeshShadingNV);
                  case spv::ExecutionModel::MeshEXT:
                    return _.HasCapability(spv::Capability::MeshShadingEXT);
                  default:
                    return false;
                }
              })) {
        if (_.HasCapability(spv::Capability::MeshShadingNV) ||
            _.HasCapability(spv::Capability::MeshShadingEXT)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with a Geometry, "
                    "tessellation, MeshNV or MeshEXT execution model.";
        } else {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with a Geometry or "
                    "tessellation execution model.";
        }
      }
      break;
    case spv::ExecutionMode::OutputLinesEXT:
    case spv::ExecutionMode::OutputTrianglesEXT:
    case spv::ExecutionMode::OutputPrimitivesEXT:
      if (!std::all_of(models->begin(), models->end(),
                       [](const spv::ExecutionModel& model) {
                         return (model == spv::ExecutionModel::MeshEXT ||
                                 model == spv::ExecutionModel::MeshNV);
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with the MeshEXT or MeshNV "
                  "execution "
                  "model.";
      }
      break;
    case spv::ExecutionMode::PixelCenterInteger:
    case spv::ExecutionMode::OriginUpperLeft:
    case spv::ExecutionMode::OriginLowerLeft:
    case spv::ExecutionMode::EarlyFragmentTests:
    case spv::ExecutionMode::DepthReplacing:
    case spv::ExecutionMode::DepthGreater:
    case spv::ExecutionMode::DepthLess:
    case spv::ExecutionMode::DepthUnchanged:
    case spv::ExecutionMode::PixelInterlockOrderedEXT:
    case spv::ExecutionMode::PixelInterlockUnorderedEXT:
    case spv::ExecutionMode::SampleInterlockOrderedEXT:
    case spv::ExecutionMode::SampleInterlockUnorderedEXT:
    case spv::ExecutionMode::ShadingRateInterlockOrderedEXT:
    case spv::ExecutionMode::ShadingRateInterlockUnorderedEXT:
    case spv::ExecutionMode::EarlyAndLateFragmentTestsAMD:
    case spv::ExecutionMode::StencilRefUnchangedFrontAMD:
    case spv::ExecutionMode::StencilRefGreaterFrontAMD:
    case spv::ExecutionMode::StencilRefLessFrontAMD:
    case spv::ExecutionMode::StencilRefUnchangedBackAMD:
    case spv::ExecutionMode::StencilRefGreaterBackAMD:
    case spv::ExecutionMode::StencilRefLessBackAMD:
      if (!std::all_of(models->begin(), models->end(),
                       [](const spv::ExecutionModel& model) {
                         return model == spv::ExecutionModel::Fragment;
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with the Fragment execution "
                  "model.";
      }
      break;
    case spv::ExecutionMode::LocalSizeHint:
    case spv::ExecutionMode::VecTypeHint:
    case spv::ExecutionMode::ContractionOff:
    case spv::ExecutionMode::LocalSizeHintId:
      if (!std::all_of(models->begin(), models->end(),
                       [](const spv::ExecutionModel& model) {
                         return model == spv::ExecutionModel::Kernel;
                       })) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "Execution mode can only be used with the Kernel execution "
                  "model.";
      }
      break;
    case spv::ExecutionMode::LocalSize:
    case spv::ExecutionMode::LocalSizeId:
      if (mode == spv::ExecutionMode::LocalSizeId && !_.IsLocalSizeIdAllowed())
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << "LocalSizeId mode is not allowed by the current environment.";

      if (!std::all_of(
              models->begin(), models->end(),
              [&_](const spv::ExecutionModel& model) {
                switch (model) {
                  case spv::ExecutionModel::Kernel:
                  case spv::ExecutionModel::GLCompute:
                    return true;
                  case spv::ExecutionModel::TaskNV:
                  case spv::ExecutionModel::MeshNV:
                    return _.HasCapability(spv::Capability::MeshShadingNV);
                  case spv::ExecutionModel::TaskEXT:
                  case spv::ExecutionModel::MeshEXT:
                    return _.HasCapability(spv::Capability::MeshShadingEXT);
                  default:
                    return false;
                }
              })) {
        if (_.HasCapability(spv::Capability::MeshShadingNV) ||
            _.HasCapability(spv::Capability::MeshShadingEXT)) {
          return _.diag(SPV_ERROR_INVALID_DATA, inst)
                 << "Execution mode can only be used with a Kernel, GLCompute, "
                    "MeshNV, MeshEXT, TaskNV or TaskEXT execution model.";
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
    if (mode == spv::ExecutionMode::OriginLowerLeft) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4653)
             << "In the Vulkan environment, the OriginLowerLeft execution mode "
                "must not be used.";
    }
    if (mode == spv::ExecutionMode::PixelCenterInteger) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4654)
             << "In the Vulkan environment, the PixelCenterInteger execution "
                "mode must not be used.";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateMemoryModel(ValidationState_t& _,
                                 const Instruction* inst) {
  // Already produced an error if multiple memory model instructions are
  // present.
  if (_.memory_model() != spv::MemoryModel::VulkanKHR &&
      _.HasCapability(spv::Capability::VulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "VulkanMemoryModelKHR capability must only be specified if "
              "the VulkanKHR memory model is used.";
  }

  if (spvIsOpenCLEnv(_.context()->target_env)) {
    if ((_.addressing_model() != spv::AddressingModel::Physical32) &&
        (_.addressing_model() != spv::AddressingModel::Physical64)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Addressing model must be Physical32 or Physical64 "
             << "in the OpenCL environment.";
    }
    if (_.memory_model() != spv::MemoryModel::OpenCL) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Memory model must be OpenCL in the OpenCL environment.";
    }
  }

  if (spvIsVulkanEnv(_.context()->target_env)) {
    if ((_.addressing_model() != spv::AddressingModel::Logical) &&
        (_.addressing_model() !=
         spv::AddressingModel::PhysicalStorageBuffer64)) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << _.VkErrorID(4635)
             << "Addressing model must be Logical or PhysicalStorageBuffer64 "
             << "in the Vulkan environment.";
    }
  }
  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ModeSettingPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpEntryPoint:
      if (auto error = ValidateEntryPoint(_, inst)) return error;
      break;
    case spv::Op::OpExecutionMode:
    case spv::Op::OpExecutionModeId:
      if (auto error = ValidateExecutionMode(_, inst)) return error;
      break;
    case spv::Op::OpMemoryModel:
      if (auto error = ValidateMemoryModel(_, inst)) return error;
      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
