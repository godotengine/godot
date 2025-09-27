// Copyright (c) 2016 Google Inc.
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

#include "source/opt/ir_loader.h"

#include <utility>

#include "DebugInfo.h"
#include "OpenCLDebugInfo100.h"
#include "source/ext_inst.h"
#include "source/opt/ir_context.h"
#include "source/opt/log.h"
#include "source/opt/reflect.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kExtInstSetIndex = 4;
constexpr uint32_t kLexicalScopeIndex = 5;
constexpr uint32_t kInlinedAtIndex = 6;
}  // namespace

IrLoader::IrLoader(const MessageConsumer& consumer, Module* m)
    : consumer_(consumer),
      module_(m),
      source_("<instruction>"),
      inst_index_(0),
      last_dbg_scope_(kNoDebugScope, kNoInlinedAt) {}

bool IsLineInst(const spv_parsed_instruction_t* inst) {
  const auto opcode = static_cast<spv::Op>(inst->opcode);
  if (IsOpLineInst(opcode)) return true;
  if (opcode != spv::Op::OpExtInst) return false;
  if (inst->ext_inst_type != SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100)
    return false;
  const uint32_t ext_inst_index = inst->words[kExtInstSetIndex];
  const NonSemanticShaderDebugInfo100Instructions ext_inst_key =
      NonSemanticShaderDebugInfo100Instructions(ext_inst_index);
  return ext_inst_key == NonSemanticShaderDebugInfo100DebugLine ||
         ext_inst_key == NonSemanticShaderDebugInfo100DebugNoLine;
}

bool IrLoader::AddInstruction(const spv_parsed_instruction_t* inst) {
  ++inst_index_;
  if (IsLineInst(inst)) {
    module()->SetContainsDebugInfo();
    last_line_inst_.reset();
    dbg_line_info_.emplace_back(module()->context(), *inst, last_dbg_scope_);
    return true;
  }

  // If it is a DebugScope or DebugNoScope of debug extension, we do not
  // create a new instruction, but simply keep the information in
  // struct DebugScope.
  const auto opcode = static_cast<spv::Op>(inst->opcode);
  if (opcode == spv::Op::OpExtInst &&
      spvExtInstIsDebugInfo(inst->ext_inst_type)) {
    const uint32_t ext_inst_index = inst->words[kExtInstSetIndex];
    if (inst->ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100 ||
        inst->ext_inst_type ==
            SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100) {
      const CommonDebugInfoInstructions ext_inst_key =
          CommonDebugInfoInstructions(ext_inst_index);
      if (ext_inst_key == CommonDebugInfoDebugScope) {
        uint32_t inlined_at = 0;
        if (inst->num_words > kInlinedAtIndex)
          inlined_at = inst->words[kInlinedAtIndex];
        last_dbg_scope_ =
            DebugScope(inst->words[kLexicalScopeIndex], inlined_at);
        module()->SetContainsDebugInfo();
        return true;
      }
      if (ext_inst_key == CommonDebugInfoDebugNoScope) {
        last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
        module()->SetContainsDebugInfo();
        return true;
      }
    } else {
      const DebugInfoInstructions ext_inst_key =
          DebugInfoInstructions(ext_inst_index);
      if (ext_inst_key == DebugInfoDebugScope) {
        uint32_t inlined_at = 0;
        if (inst->num_words > kInlinedAtIndex)
          inlined_at = inst->words[kInlinedAtIndex];
        last_dbg_scope_ =
            DebugScope(inst->words[kLexicalScopeIndex], inlined_at);
        module()->SetContainsDebugInfo();
        return true;
      }
      if (ext_inst_key == DebugInfoDebugNoScope) {
        last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
        module()->SetContainsDebugInfo();
        return true;
      }
    }
  }

  std::unique_ptr<Instruction> spv_inst(
      new Instruction(module()->context(), *inst, std::move(dbg_line_info_)));
  if (!spv_inst->dbg_line_insts().empty()) {
    if (extra_line_tracking_ &&
        (!spv_inst->dbg_line_insts().back().IsNoLine())) {
      last_line_inst_ = std::unique_ptr<Instruction>(
          spv_inst->dbg_line_insts().back().Clone(module()->context()));
      if (last_line_inst_->IsDebugLineInst())
        last_line_inst_->SetResultId(module()->context()->TakeNextId());
    }
    dbg_line_info_.clear();
  } else if (last_line_inst_ != nullptr) {
    last_line_inst_->SetDebugScope(last_dbg_scope_);
    spv_inst->dbg_line_insts().push_back(*last_line_inst_);
    last_line_inst_ = std::unique_ptr<Instruction>(
        spv_inst->dbg_line_insts().back().Clone(module()->context()));
    if (last_line_inst_->IsDebugLineInst())
      last_line_inst_->SetResultId(module()->context()->TakeNextId());
  }

  const char* src = source_.c_str();
  spv_position_t loc = {inst_index_, 0, 0};

  // Handle function and basic block boundaries first, then normal
  // instructions.
  if (opcode == spv::Op::OpFunction) {
    if (function_ != nullptr) {
      Error(consumer_, src, loc, "function inside function");
      return false;
    }
    function_ = MakeUnique<Function>(std::move(spv_inst));
  } else if (opcode == spv::Op::OpFunctionEnd) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc,
            "OpFunctionEnd without corresponding OpFunction");
      return false;
    }
    if (block_ != nullptr) {
      Error(consumer_, src, loc, "OpFunctionEnd inside basic block");
      return false;
    }
    function_->SetFunctionEnd(std::move(spv_inst));
    module_->AddFunction(std::move(function_));
    function_ = nullptr;
  } else if (opcode == spv::Op::OpLabel) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc, "OpLabel outside function");
      return false;
    }
    if (block_ != nullptr) {
      Error(consumer_, src, loc, "OpLabel inside basic block");
      return false;
    }
    block_ = MakeUnique<BasicBlock>(std::move(spv_inst));
  } else if (spvOpcodeIsBlockTerminator(opcode)) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc, "terminator instruction outside function");
      return false;
    }
    if (block_ == nullptr) {
      Error(consumer_, src, loc, "terminator instruction outside basic block");
      return false;
    }
    if (last_dbg_scope_.GetLexicalScope() != kNoDebugScope)
      spv_inst->SetDebugScope(last_dbg_scope_);
    block_->AddInstruction(std::move(spv_inst));
    function_->AddBasicBlock(std::move(block_));
    block_ = nullptr;
    last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
    last_line_inst_.reset();
    dbg_line_info_.clear();
  } else {
    if (function_ == nullptr) {  // Outside function definition
      SPIRV_ASSERT(consumer_, block_ == nullptr);
      if (opcode == spv::Op::OpCapability) {
        module_->AddCapability(std::move(spv_inst));
      } else if (opcode == spv::Op::OpExtension) {
        module_->AddExtension(std::move(spv_inst));
      } else if (opcode == spv::Op::OpExtInstImport) {
        module_->AddExtInstImport(std::move(spv_inst));
      } else if (opcode == spv::Op::OpMemoryModel) {
        module_->SetMemoryModel(std::move(spv_inst));
      } else if (opcode == spv::Op::OpSamplerImageAddressingModeNV) {
        module_->SetSampledImageAddressMode(std::move(spv_inst));
      } else if (opcode == spv::Op::OpEntryPoint) {
        module_->AddEntryPoint(std::move(spv_inst));
      } else if (opcode == spv::Op::OpExecutionMode ||
                 opcode == spv::Op::OpExecutionModeId) {
        module_->AddExecutionMode(std::move(spv_inst));
      } else if (IsDebug1Inst(opcode)) {
        module_->AddDebug1Inst(std::move(spv_inst));
      } else if (IsDebug2Inst(opcode)) {
        module_->AddDebug2Inst(std::move(spv_inst));
      } else if (IsDebug3Inst(opcode)) {
        module_->AddDebug3Inst(std::move(spv_inst));
      } else if (IsAnnotationInst(opcode)) {
        module_->AddAnnotationInst(std::move(spv_inst));
      } else if (IsTypeInst(opcode)) {
        module_->AddType(std::move(spv_inst));
      } else if (IsConstantInst(opcode) || opcode == spv::Op::OpVariable ||
                 opcode == spv::Op::OpUndef) {
        module_->AddGlobalValue(std::move(spv_inst));
      } else if (opcode == spv::Op::OpExtInst &&
                 spvExtInstIsDebugInfo(inst->ext_inst_type)) {
        module_->AddExtInstDebugInfo(std::move(spv_inst));
      } else if (opcode == spv::Op::OpExtInst &&
                 spvExtInstIsNonSemantic(inst->ext_inst_type)) {
        // If there are no functions, add the non-semantic instructions to the
        // global values. Otherwise append it to the list of the last function.
        auto func_begin = module_->begin();
        auto func_end = module_->end();
        if (func_begin == func_end) {
          module_->AddGlobalValue(std::move(spv_inst));
        } else {
          (--func_end)->AddNonSemanticInstruction(std::move(spv_inst));
        }
      } else {
        Errorf(consumer_, src, loc,
               "Unhandled inst type (opcode: %d) found outside function "
               "definition.",
               opcode);
        return false;
      }
    } else {
      if (opcode == spv::Op::OpLoopMerge || opcode == spv::Op::OpSelectionMerge)
        last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
      if (last_dbg_scope_.GetLexicalScope() != kNoDebugScope)
        spv_inst->SetDebugScope(last_dbg_scope_);
      if (opcode == spv::Op::OpExtInst &&
          spvExtInstIsDebugInfo(inst->ext_inst_type)) {
        const uint32_t ext_inst_index = inst->words[kExtInstSetIndex];
        if (inst->ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100) {
          const OpenCLDebugInfo100Instructions ext_inst_key =
              OpenCLDebugInfo100Instructions(ext_inst_index);
          switch (ext_inst_key) {
            case OpenCLDebugInfo100DebugDeclare: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            case OpenCLDebugInfo100DebugValue: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            default: {
              Errorf(consumer_, src, loc,
                     "Debug info extension instruction other than DebugScope, "
                     "DebugNoScope, DebugFunctionDefinition, DebugDeclare, and "
                     "DebugValue found inside function",
                     opcode);
              return false;
            }
          }
        } else if (inst->ext_inst_type ==
                   SPV_EXT_INST_TYPE_NONSEMANTIC_SHADER_DEBUGINFO_100) {
          const NonSemanticShaderDebugInfo100Instructions ext_inst_key =
              NonSemanticShaderDebugInfo100Instructions(ext_inst_index);
          switch (ext_inst_key) {
            case NonSemanticShaderDebugInfo100DebugDeclare:
            case NonSemanticShaderDebugInfo100DebugValue:
            case NonSemanticShaderDebugInfo100DebugScope:
            case NonSemanticShaderDebugInfo100DebugNoScope:
            case NonSemanticShaderDebugInfo100DebugFunctionDefinition: {
              if (block_ == nullptr) {  // Inside function but outside blocks
                Errorf(consumer_, src, loc,
                       "Debug info extension instruction found inside function "
                       "but outside block",
                       opcode);
              } else {
                block_->AddInstruction(std::move(spv_inst));
              }
              break;
            }
            default: {
              Errorf(consumer_, src, loc,
                     "Debug info extension instruction other than DebugScope, "
                     "DebugNoScope, DebugDeclare, and DebugValue found inside "
                     "function",
                     opcode);
              return false;
            }
          }
        } else {
          const DebugInfoInstructions ext_inst_key =
              DebugInfoInstructions(ext_inst_index);
          switch (ext_inst_key) {
            case DebugInfoDebugDeclare: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            case DebugInfoDebugValue: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            default: {
              Errorf(consumer_, src, loc,
                     "Debug info extension instruction other than DebugScope, "
                     "DebugNoScope, DebugDeclare, and DebugValue found inside "
                     "function",
                     opcode);
              return false;
            }
          }
        }
      } else {
        if (block_ == nullptr) {  // Inside function but outside blocks
          if (opcode != spv::Op::OpFunctionParameter) {
            Errorf(consumer_, src, loc,
                   "Non-OpFunctionParameter (opcode: %d) found inside "
                   "function but outside basic block",
                   opcode);
            return false;
          }
          function_->AddParameter(std::move(spv_inst));
        } else {
          block_->AddInstruction(std::move(spv_inst));
        }
      }
    }
  }
  return true;
}

// Resolves internal references among the module, functions, basic blocks, etc.
// This function should be called after adding all instructions.
void IrLoader::EndModule() {
  if (block_ && function_) {
    // We're in the middle of a basic block, but the terminator is missing.
    // Register the block anyway.  This lets us write tests with less
    // boilerplate.
    function_->AddBasicBlock(std::move(block_));
    block_ = nullptr;
  }
  if (function_) {
    // We're in the middle of a function, but the OpFunctionEnd is missing.
    // Register the function anyway.  This lets us write tests with less
    // boilerplate.
    module_->AddFunction(std::move(function_));
    function_ = nullptr;
  }
  for (auto& function : *module_) {
    for (auto& bb : function) bb.SetParent(&function);
  }

  // Copy any trailing Op*Line instruction into the module
  module_->SetTrailingDbgLineInfo(std::move(dbg_line_info_));
}

}  // namespace opt
}  // namespace spvtools
