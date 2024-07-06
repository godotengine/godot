// Copyright (c) 2017 Pierre Moreau
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

#ifndef SOURCE_OPT_DECORATION_MANAGER_H_
#define SOURCE_OPT_DECORATION_MANAGER_H_

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {
namespace analysis {

// A class for analyzing and managing decorations in an Module.
class DecorationManager {
 public:
  // Constructs a decoration manager from the given |module|
  explicit DecorationManager(Module* module) : module_(module) {
    AnalyzeDecorations();
  }
  DecorationManager() = delete;

  // Removes all decorations (direct and through groups) where |pred| is
  // true and that apply to |id| so that they no longer apply to |id|.  Returns
  // true if something changed.
  //
  // If |id| is part of a group, it will be removed from the group if it
  // does not use all of the group's decorations, or, if there are no
  // decorations that apply to the group.
  //
  // If decoration groups become empty, the |OpGroupDecorate| and
  // |OpGroupMemberDecorate| instructions will be killed.
  //
  // Decoration instructions that apply directly to |id| will be killed.
  //
  // If |id| is a decoration group and all of the group's decorations are
  // removed, then the |OpGroupDecorate| and
  // |OpGroupMemberDecorate| for the group will be killed, but not the defining
  // |OpDecorationGroup| instruction.
  bool RemoveDecorationsFrom(
      uint32_t id, std::function<bool(const Instruction&)> pred =
                       [](const Instruction&) { return true; });

  // Removes all decorations from the result id of |inst|.
  //
  // NOTE: This is only meant to be called from ir_context, as only metadata
  // will be removed, and no actual instruction.
  void RemoveDecoration(Instruction* inst);

  // Returns a vector of all decorations affecting |id|. If a group is applied
  // to |id|, the decorations of that group are returned rather than the group
  // decoration instruction. If |include_linkage| is not set, linkage
  // decorations won't be returned.
  std::vector<Instruction*> GetDecorationsFor(uint32_t id,
                                              bool include_linkage);
  std::vector<const Instruction*> GetDecorationsFor(uint32_t id,
                                                    bool include_linkage) const;
  // Returns whether two IDs have the same decorations. Two
  // spv::Op::OpGroupDecorate instructions that apply the same decorations but
  // to different IDs, still count as being the same.
  bool HaveTheSameDecorations(uint32_t id1, uint32_t id2) const;

  // Returns whether two IDs have the same decorations. Two
  // spv::Op::OpGroupDecorate instructions that apply the same decorations but
  // to different IDs, still count as being the same.
  bool HaveSubsetOfDecorations(uint32_t id1, uint32_t id2) const;

  // Returns whether the two decorations instructions are the same and are
  // applying the same decorations; unless |ignore_target| is false, the targets
  // to which they are applied to does not matter, except for the member part.
  //
  // This is only valid for OpDecorate, OpMemberDecorate and OpDecorateId; it
  // will return false for other opcodes.
  bool AreDecorationsTheSame(const Instruction* inst1, const Instruction* inst2,
                             bool ignore_target) const;

  // Returns whether a decoration instruction for |id| with decoration
  // |decoration| exists or not.
  bool HasDecoration(uint32_t id, uint32_t decoration) const;
  bool HasDecoration(uint32_t id, spv::Decoration decoration) const;

  // |f| is run on each decoration instruction for |id| with decoration
  // |decoration|. Processed are all decorations which target |id| either
  // directly or indirectly by Decoration Groups.
  void ForEachDecoration(uint32_t id, uint32_t decoration,
                         std::function<void(const Instruction&)> f) const;

  // |f| is run on each decoration instruction for |id| with decoration
  // |decoration|. Processes all decoration which target |id| either directly or
  // indirectly through decoration groups. If |f| returns false, iteration is
  // terminated and this function returns false.
  bool WhileEachDecoration(uint32_t id, uint32_t decoration,
                           std::function<bool(const Instruction&)> f) const;

  // |f| is run on each decoration instruction for |id| with decoration
  // |decoration|. Processes all decoration which target |id| either directly or
  // indirectly through decoration groups. If |f| returns true, iteration is
  // terminated and this function returns true. Otherwise returns false.
  bool FindDecoration(uint32_t id, uint32_t decoration,
                      std::function<bool(const Instruction&)> f);

  // Clone all decorations from one id |from|.
  // The cloned decorations are assigned to the given id |to| and are
  // added to the module. The purpose is to decorate cloned instructions.
  // This function does not check if the id |to| is already decorated.
  void CloneDecorations(uint32_t from, uint32_t to);

  // Same as above, but only clone the decoration if the decoration operand is
  // in |decorations_to_copy|.  This function has the extra restriction that
  // |from| and |to| must not be an object, not a type.
  void CloneDecorations(
      uint32_t from, uint32_t to,
      const std::vector<spv::Decoration>& decorations_to_copy);

  // Informs the decoration manager of a new decoration that it needs to track.
  void AddDecoration(Instruction* inst);

  // Add decoration with |opcode| and operands |opnds|.
  void AddDecoration(spv::Op opcode, const std::vector<Operand> opnds);

  // Add |decoration| of |inst_id| to module.
  void AddDecoration(uint32_t inst_id, uint32_t decoration);

  // Add |decoration, decoration_value| of |inst_id| to module.
  void AddDecorationVal(uint32_t inst_id, uint32_t decoration,
                        uint32_t decoration_value);

  // Add |decoration, decoration_value| of |inst_id, member| to module.
  void AddMemberDecoration(uint32_t inst_id, uint32_t member,
                           uint32_t decoration, uint32_t decoration_value);

  friend bool operator==(const DecorationManager&, const DecorationManager&);
  friend bool operator!=(const DecorationManager& lhs,
                         const DecorationManager& rhs) {
    return !(lhs == rhs);
  }

 private:
  // Analyzes the defs and uses in the given |module| and populates data
  // structures in this class. Does nothing if |module| is nullptr.
  void AnalyzeDecorations();

  template <typename T>
  std::vector<T> InternalGetDecorationsFor(uint32_t id, bool include_linkage);

  // Tracks decoration information of an ID.
  struct TargetData {
    std::vector<Instruction*> direct_decorations;    // All decorate
                                                     // instructions applied
                                                     // to the tracked ID.
    std::vector<Instruction*> indirect_decorations;  // All instructions
                                                     // applying a group to
                                                     // the tracked ID.
    std::vector<Instruction*> decorate_insts;  // All decorate instructions
                                               // applying the decorations
                                               // of the tracked ID to
                                               // targets.
                                               // It is empty if the
                                               // tracked ID is not a
                                               // group.
  };

  friend bool operator==(const TargetData& lhs, const TargetData& rhs) {
    if (!std::is_permutation(lhs.direct_decorations.begin(),
                             lhs.direct_decorations.end(),
                             rhs.direct_decorations.begin())) {
      return false;
    }
    if (!std::is_permutation(lhs.indirect_decorations.begin(),
                             lhs.indirect_decorations.end(),
                             rhs.indirect_decorations.begin())) {
      return false;
    }
    if (!std::is_permutation(lhs.decorate_insts.begin(),
                             lhs.decorate_insts.end(),
                             rhs.decorate_insts.begin())) {
      return false;
    }
    return true;
  }

  // Mapping from ids to the instructions applying a decoration to those ids.
  // In other words, for each id you get all decoration instructions
  // referencing that id, be it directly (spv::Op::OpDecorate,
  // spv::Op::OpMemberDecorate and spv::Op::OpDecorateId), or indirectly
  // (spv::Op::OpGroupDecorate, spv::Op::OpMemberGroupDecorate).
  std::unordered_map<uint32_t, TargetData> id_to_decoration_insts_;
  // The enclosing module.
  Module* module_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DECORATION_MANAGER_H_
