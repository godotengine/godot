// Copyright (c) 2021 Google LLC.
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

#include "source/lint/divergence_analysis.h"

#include "source/opt/basic_block.h"
#include "source/opt/control_dependence.h"
#include "source/opt/dataflow.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"

namespace spvtools {
namespace lint {

void DivergenceAnalysis::EnqueueSuccessors(opt::Instruction* inst) {
  // Enqueue control dependents of block, if applicable.
  // There are two ways for a dependence source to be updated:
  // 1. control -> control: source block is marked divergent.
  // 2. data -> control: branch condition is marked divergent.
  uint32_t block_id;
  if (inst->IsBlockTerminator()) {
    block_id = context().get_instr_block(inst)->id();
  } else if (inst->opcode() == spv::Op::OpLabel) {
    block_id = inst->result_id();
    opt::BasicBlock* bb = context().cfg()->block(block_id);
    // Only enqueue phi instructions, as other uses don't affect divergence.
    bb->ForEachPhiInst([this](opt::Instruction* phi) { Enqueue(phi); });
  } else {
    opt::ForwardDataFlowAnalysis::EnqueueUsers(inst);
    return;
  }
  if (!cd_.HasBlock(block_id)) {
    return;
  }
  for (const spvtools::opt::ControlDependence& dep :
       cd_.GetDependenceTargets(block_id)) {
    opt::Instruction* target_inst =
        context().cfg()->block(dep.target_bb_id())->GetLabelInst();
    Enqueue(target_inst);
  }
}

opt::DataFlowAnalysis::VisitResult DivergenceAnalysis::Visit(
    opt::Instruction* inst) {
  if (inst->opcode() == spv::Op::OpLabel) {
    return VisitBlock(inst->result_id());
  } else {
    return VisitInstruction(inst);
  }
}

opt::DataFlowAnalysis::VisitResult DivergenceAnalysis::VisitBlock(uint32_t id) {
  if (!cd_.HasBlock(id)) {
    return opt::DataFlowAnalysis::VisitResult::kResultFixed;
  }
  DivergenceLevel& cur_level = divergence_[id];
  if (cur_level == DivergenceLevel::kDivergent) {
    return opt::DataFlowAnalysis::VisitResult::kResultFixed;
  }
  DivergenceLevel orig = cur_level;
  for (const spvtools::opt::ControlDependence& dep :
       cd_.GetDependenceSources(id)) {
    if (divergence_[dep.source_bb_id()] > cur_level) {
      cur_level = divergence_[dep.source_bb_id()];
      divergence_source_[id] = dep.source_bb_id();
    } else if (dep.source_bb_id() != 0) {
      uint32_t condition_id = dep.GetConditionID(*context().cfg());
      DivergenceLevel dep_level = divergence_[condition_id];
      // Check if we are along the chain of unconditional branches starting from
      // the branch target.
      if (follow_unconditional_branches_[dep.branch_target_bb_id()] !=
          follow_unconditional_branches_[dep.target_bb_id()]) {
        // We must have reconverged in order to reach this block.
        // Promote partially uniform to divergent.
        if (dep_level == DivergenceLevel::kPartiallyUniform) {
          dep_level = DivergenceLevel::kDivergent;
        }
      }
      if (dep_level > cur_level) {
        cur_level = dep_level;
        divergence_source_[id] = condition_id;
        divergence_dependence_source_[id] = dep.source_bb_id();
      }
    }
  }
  return cur_level > orig ? VisitResult::kResultChanged
                          : VisitResult::kResultFixed;
}

opt::DataFlowAnalysis::VisitResult DivergenceAnalysis::VisitInstruction(
    opt::Instruction* inst) {
  if (inst->IsBlockTerminator()) {
    // This is called only when the condition has changed, so return changed.
    return VisitResult::kResultChanged;
  }
  if (!inst->HasResultId()) {
    return VisitResult::kResultFixed;
  }
  uint32_t id = inst->result_id();
  DivergenceLevel& cur_level = divergence_[id];
  if (cur_level == DivergenceLevel::kDivergent) {
    return opt::DataFlowAnalysis::VisitResult::kResultFixed;
  }
  DivergenceLevel orig = cur_level;
  cur_level = ComputeInstructionDivergence(inst);
  return cur_level > orig ? VisitResult::kResultChanged
                          : VisitResult::kResultFixed;
}

DivergenceAnalysis::DivergenceLevel
DivergenceAnalysis::ComputeInstructionDivergence(opt::Instruction* inst) {
  // TODO(kuhar): Check to see if inst is decorated with Uniform or UniformId
  // and use that to short circuit other checks. Uniform is for subgroups which
  // would satisfy derivative groups too. UniformId takes a scope, so if it is
  // subgroup or greater it could satisfy derivative group and
  // Device/QueueFamily could satisfy fully uniform.
  uint32_t id = inst->result_id();
  // Handle divergence roots.
  if (inst->opcode() == spv::Op::OpFunctionParameter) {
    divergence_source_[id] = 0;
    return divergence_[id] = DivergenceLevel::kDivergent;
  } else if (inst->IsLoad()) {
    spvtools::opt::Instruction* var = inst->GetBaseAddress();
    if (var->opcode() != spv::Op::OpVariable) {
      // Assume divergent.
      divergence_source_[id] = 0;
      return DivergenceLevel::kDivergent;
    }
    DivergenceLevel ret = ComputeVariableDivergence(var);
    if (ret > DivergenceLevel::kUniform) {
      divergence_source_[inst->result_id()] = 0;
    }
    return divergence_[id] = ret;
  }
  // Get the maximum divergence of the operands.
  DivergenceLevel ret = DivergenceLevel::kUniform;
  inst->ForEachInId([this, inst, &ret](const uint32_t* op) {
    if (!op) return;
    if (divergence_[*op] > ret) {
      divergence_source_[inst->result_id()] = *op;
      ret = divergence_[*op];
    }
  });
  divergence_[inst->result_id()] = ret;
  return ret;
}

DivergenceAnalysis::DivergenceLevel
DivergenceAnalysis::ComputeVariableDivergence(opt::Instruction* var) {
  uint32_t type_id = var->type_id();
  spvtools::opt::analysis::Pointer* type =
      context().get_type_mgr()->GetType(type_id)->AsPointer();
  assert(type != nullptr);
  uint32_t def_id = var->result_id();
  DivergenceLevel ret;
  switch (type->storage_class()) {
    case spv::StorageClass::Function:
    case spv::StorageClass::Generic:
    case spv::StorageClass::AtomicCounter:
    case spv::StorageClass::StorageBuffer:
    case spv::StorageClass::PhysicalStorageBuffer:
    case spv::StorageClass::Output:
    case spv::StorageClass::Workgroup:
    case spv::StorageClass::Image:  // Image atomics probably aren't uniform.
    case spv::StorageClass::Private:
      ret = DivergenceLevel::kDivergent;
      break;
    case spv::StorageClass::Input:
      ret = DivergenceLevel::kDivergent;
      // If this variable has a Flat decoration, it is partially uniform.
      // TODO(kuhar): Track access chain indices and also consider Flat members
      // of a structure.
      context().get_decoration_mgr()->WhileEachDecoration(
          def_id, static_cast<uint32_t>(spv::Decoration::Flat),
          [&ret](const opt::Instruction&) {
            ret = DivergenceLevel::kPartiallyUniform;
            return false;
          });
      break;
    case spv::StorageClass::UniformConstant:
      // May be a storage image which is also written to; mark those as
      // divergent.
      if (!var->IsVulkanStorageImage() || var->IsReadOnlyPointer()) {
        ret = DivergenceLevel::kUniform;
      } else {
        ret = DivergenceLevel::kDivergent;
      }
      break;
    case spv::StorageClass::Uniform:
    case spv::StorageClass::PushConstant:
    case spv::StorageClass::CrossWorkgroup:  // Not for shaders; default
                                             // uniform.
    default:
      ret = DivergenceLevel::kUniform;
      break;
  }
  return ret;
}

void DivergenceAnalysis::Setup(opt::Function* function) {
  // TODO(kuhar): Run functions called by |function| so we can detect
  // reconvergence caused by multiple returns.
  cd_.ComputeControlDependenceGraph(
      *context().cfg(), *context().GetPostDominatorAnalysis(function));
  context().cfg()->ForEachBlockInPostOrder(
      function->entry().get(), [this](const opt::BasicBlock* bb) {
        uint32_t id = bb->id();
        if (bb->terminator() == nullptr ||
            bb->terminator()->opcode() != spv::Op::OpBranch) {
          follow_unconditional_branches_[id] = id;
        } else {
          uint32_t target_id = bb->terminator()->GetSingleWordInOperand(0);
          // Target is guaranteed to have been visited before us in postorder.
          follow_unconditional_branches_[id] =
              follow_unconditional_branches_[target_id];
        }
      });
}

std::ostream& operator<<(std::ostream& os,
                         DivergenceAnalysis::DivergenceLevel level) {
  switch (level) {
    case DivergenceAnalysis::DivergenceLevel::kUniform:
      return os << "uniform";
    case DivergenceAnalysis::DivergenceLevel::kPartiallyUniform:
      return os << "partially uniform";
    case DivergenceAnalysis::DivergenceLevel::kDivergent:
      return os << "divergent";
    default:
      return os << "<invalid divergence level>";
  }
}

}  // namespace lint
}  // namespace spvtools
