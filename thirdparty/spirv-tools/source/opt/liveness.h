// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#ifndef SOURCE_OPT_LIVENESS_H_
#define SOURCE_OPT_LIVENESS_H_

#include <cstdint>
#include <unordered_set>

namespace spvtools {
namespace opt {

class IRContext;
class Instruction;

namespace analysis {

class Type;

// This class represents the liveness of the input variables of a module
class LivenessManager {
 public:
  LivenessManager(IRContext* ctx);

  // Copy liveness info into |live_locs| and |builtin_locs|.
  void GetLiveness(std::unordered_set<uint32_t>* live_locs,
                   std::unordered_set<uint32_t>* live_builtins);

  // Return true if builtin |bi| is being analyzed.
  bool IsAnalyzedBuiltin(uint32_t bi);

  // Return the result type of |ac| when applied to |cur_type_id|. Set
  // |no_loc| to true if no loc found. Set |is_patch| indicates if the variable
  // is a patch variable. Set |input| if the variable is an input variable.
  // Otherwise it is assumed that the variable is an output variable.
  uint32_t AnalyzeAccessChainLoc(const Instruction* ac, uint32_t curr_type_id,
                                 uint32_t* offset, bool* no_loc, bool is_patch,
                                 bool input = true);

  // Return size of |type_id| in units of locations
  uint32_t GetLocSize(const analysis::Type* type) const;

 private:
  IRContext* context() const { return ctx_; }

  // Initialize analysis
  void InitializeAnalysis();

  // Analyze |id| for builtin var and struct members. Return true if builtins
  // found.
  bool AnalyzeBuiltIn(uint32_t id);

  // Mark all live locations resulting from |user| of |var| at |loc|.
  void MarkRefLive(const Instruction* user, Instruction* var);

  // Mark |count| locations starting at location |start|.
  void MarkLocsLive(uint32_t start, uint32_t count);

  // Return type of the member |index| in the aggregate type |agg_type_id|.
  uint32_t GetComponentType(uint32_t index, uint32_t agg_type_id) const;

  // Return offset of member |index| in the aggregate type |agg_type_id| in
  // units of input locations.
  uint32_t GetLocOffset(uint32_t index, uint32_t agg_type_id) const;

  // Populate live_locs_ and live_builtins_
  void ComputeLiveness();

  // IR context that owns this liveness manager.
  IRContext* ctx_;

  // True if live_locs_ and live_builtins_ are computed
  bool computed_;

  // Live locations
  std::unordered_set<uint32_t> live_locs_;

  // Live builtins
  std::unordered_set<uint32_t> live_builtins_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LIVENESS_H_
