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

#ifndef SOURCE_OPT_IR_LOADER_H_
#define SOURCE_OPT_IR_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/instruction.h"
#include "source/opt/module.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {

// Loader class for constructing SPIR-V in-memory IR representation. Methods in
// this class are designed to work with the interface for spvBinaryParse() in
// libspirv.h so that we can leverage the syntax checks implemented behind it.
//
// The user is expected to call SetModuleHeader() to fill in the module's
// header, and then AddInstruction() for each decoded instruction, and finally
// EndModule() to finalize the module. The instructions processed in sequence
// by AddInstruction() should comprise a valid SPIR-V module.
class IrLoader {
 public:
  // Instantiates a builder to construct the given |module| gradually.
  // All internal messages will be communicated to the outside via the given
  // message |consumer|. This instance only keeps a reference to the |consumer|,
  // so the |consumer| should outlive this instance.
  IrLoader(const MessageConsumer& consumer, Module* m);

  // Sets the source name of the module.
  void SetSource(const std::string& src) { source_ = src; }

  Module* module() const { return module_; }

  // Sets the fields in the module's header to the given parameters.
  void SetModuleHeader(uint32_t magic, uint32_t version, uint32_t generator,
                       uint32_t bound, uint32_t reserved) {
    module_->SetHeader({magic, version, generator, bound, reserved});
  }
  // Adds an instruction to the module. Returns true if no error occurs. This
  // method will properly capture and store the data provided in |inst| so that
  // |inst| is no longer needed after returning.
  bool AddInstruction(const spv_parsed_instruction_t* inst);
  // Finalizes the module construction. This must be called after the module
  // header has been set and all instructions have been added.  This is
  // forgiving in the case of a missing terminator instruction on a basic block,
  // or a missing OpFunctionEnd.  Resolves internal bookkeeping.
  void EndModule();

  // Sets whether extra OpLine instructions should be injected to better
  // track line information.
  void SetExtraLineTracking(bool flag) { extra_line_tracking_ = flag; }

 private:
  // Consumer for communicating messages to outside.
  const MessageConsumer& consumer_;
  // The module to be built.
  Module* module_;
  // The source name of the module.
  std::string source_;
  // The last used instruction index.
  uint32_t inst_index_;
  // The current Function under construction.
  std::unique_ptr<Function> function_;
  // The current BasicBlock under construction.
  std::unique_ptr<BasicBlock> block_;
  // Line related debug instructions accumulated thus far.
  std::vector<Instruction> dbg_line_info_;
  // If doing extra line tracking, this is the line instruction that should be
  // applied to the next instruction.  Otherwise it always contains null.
  std::unique_ptr<Instruction> last_line_inst_;

  // The last DebugScope information that IrLoader::AddInstruction() handled.
  DebugScope last_dbg_scope_;

  // When true, do extra line information tracking: Additional OpLine
  // instructions will be injected to help track line info more robustly during
  // transformations.
  bool extra_line_tracking_ = true;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_IR_LOADER_H_
