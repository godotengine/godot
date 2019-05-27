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

#include "source/opt/log.h"
#include "source/opt/reflect.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

IrLoader::IrLoader(const MessageConsumer& consumer, Module* m)
    : consumer_(consumer),
      module_(m),
      source_("<instruction>"),
      inst_index_(0) {}

bool IrLoader::AddInstruction(const spv_parsed_instruction_t* inst) {
  ++inst_index_;
  const auto opcode = static_cast<SpvOp>(inst->opcode);
  if (IsDebugLineInst(opcode)) {
    dbg_line_info_.push_back(Instruction(module()->context(), *inst));
    return true;
  }

  std::unique_ptr<Instruction> spv_inst(
      new Instruction(module()->context(), *inst, std::move(dbg_line_info_)));
  dbg_line_info_.clear();

  const char* src = source_.c_str();
  spv_position_t loc = {inst_index_, 0, 0};

  // Handle function and basic block boundaries first, then normal
  // instructions.
  if (opcode == SpvOpFunction) {
    if (function_ != nullptr) {
      Error(consumer_, src, loc, "function inside function");
      return false;
    }
    function_ = MakeUnique<Function>(std::move(spv_inst));
  } else if (opcode == SpvOpFunctionEnd) {
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
  } else if (opcode == SpvOpLabel) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc, "OpLabel outside function");
      return false;
    }
    if (block_ != nullptr) {
      Error(consumer_, src, loc, "OpLabel inside basic block");
      return false;
    }
    block_ = MakeUnique<BasicBlock>(std::move(spv_inst));
  } else if (IsTerminatorInst(opcode)) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc, "terminator instruction outside function");
      return false;
    }
    if (block_ == nullptr) {
      Error(consumer_, src, loc, "terminator instruction outside basic block");
      return false;
    }
    block_->AddInstruction(std::move(spv_inst));
    function_->AddBasicBlock(std::move(block_));
    block_ = nullptr;
  } else {
    if (function_ == nullptr) {  // Outside function definition
      SPIRV_ASSERT(consumer_, block_ == nullptr);
      if (opcode == SpvOpCapability) {
        module_->AddCapability(std::move(spv_inst));
      } else if (opcode == SpvOpExtension) {
        module_->AddExtension(std::move(spv_inst));
      } else if (opcode == SpvOpExtInstImport) {
        module_->AddExtInstImport(std::move(spv_inst));
      } else if (opcode == SpvOpMemoryModel) {
        module_->SetMemoryModel(std::move(spv_inst));
      } else if (opcode == SpvOpEntryPoint) {
        module_->AddEntryPoint(std::move(spv_inst));
      } else if (opcode == SpvOpExecutionMode) {
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
      } else if (IsConstantInst(opcode) || opcode == SpvOpVariable ||
                 opcode == SpvOpUndef) {
        module_->AddGlobalValue(std::move(spv_inst));
      } else {
        Errorf(consumer_, src, loc,
               "Unhandled inst type (opcode: %d) found outside function definition.",
               opcode);
        return false;
      }
    } else {
      if (block_ == nullptr) {  // Inside function but outside blocks
        if (opcode != SpvOpFunctionParameter) {
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
}

}  // namespace opt
}  // namespace spvtools
