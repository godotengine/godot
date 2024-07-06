// Copyright (c) 2022 Google LLC.
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

#ifndef SOURCE_DIFF_DIFF_H_
#define SOURCE_DIFF_DIFF_H_

#include "source/opt/ir_context.h"

namespace spvtools {
namespace diff {

struct Options {
  bool ignore_set_binding = false;
  bool ignore_location = false;
  bool indent = false;
  bool no_header = false;
  bool color_output = false;
  bool dump_id_map = false;
};

// Given two SPIR-V modules, this function outputs the textual diff of their
// assembly in `out`.  The diff is *semantic*, so that the ordering of certain
// instructions wouldn't matter.
//
// The output is a disassembly of src, with diff(1)-style + and - lines that
// show how the src is changed into dst.  To make this disassembly
// self-consistent, the ids that are output are all in the space of the src
// module; e.g. any + lines (showing instructions from the dst module) have
// their ids mapped to the matched instruction in the src module (or a new id
// allocated in the src module if unmatched).
spv_result_t Diff(opt::IRContext* src, opt::IRContext* dst, std::ostream& out,
                  Options options);

}  // namespace diff
}  // namespace spvtools

#endif  // SOURCE_DIFF_DIFF_H_
