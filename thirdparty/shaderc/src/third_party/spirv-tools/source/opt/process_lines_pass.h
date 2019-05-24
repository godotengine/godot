// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#ifndef SOURCE_OPT_PROPAGATE_LINES_PASS_H_
#define SOURCE_OPT_PROPAGATE_LINES_PASS_H_

#include "source/opt/function.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

namespace {

// Constructor Parameters
static const int kLinesPropagateLines = 0;
static const int kLinesEliminateDeadLines = 1;

}  // anonymous namespace

// See optimizer.hpp for documentation.
class ProcessLinesPass : public Pass {
  using LineProcessFunction =
      std::function<bool(Instruction*, uint32_t*, uint32_t*, uint32_t*)>;

 public:
  ProcessLinesPass(uint32_t func_id);
  ~ProcessLinesPass() override = default;

  const char* name() const override { return "propagate-lines"; }

  // See optimizer.hpp for this pass' user documentation.
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisNameMap | IRContext::kAnalysisConstants |
           IRContext::kAnalysisTypes;
  }

 private:
  // If |inst| has no debug line instruction, create one with
  // |file_id, line, col|. If |inst| has debug line instructions, set
  // |file_id, line, col| from the last. |file_id| equals 0 indicates no line
  // info is available. Return true if |inst| modified.
  bool PropagateLine(Instruction* inst, uint32_t* file_id, uint32_t* line,
                     uint32_t* col);

  // If last debug line instruction of |inst| matches |file_id, line, col|,
  // delete all debug line instructions of |inst|. If they do not match,
  // replace all debug line instructions of |inst| with new line instruction
  // set from |file_id, line, col|. If |inst| has no debug line instructions,
  // do not modify |inst|. |file_id| equals 0 indicates no line info is
  // available. Return true if |inst| modified.
  bool EliminateDeadLines(Instruction* inst, uint32_t* file_id, uint32_t* line,
                          uint32_t* col);

  // Apply lpfn() to all type, constant, global variable and function
  // instructions in their physical order.
  bool ProcessLines();

  // A function that calls either PropagateLine or EliminateDeadLines.
  // Initialized by the class constructor.
  LineProcessFunction line_process_func_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_PROPAGATE_LINES_PASS_H_
