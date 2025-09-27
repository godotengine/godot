// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_OPT_ELIMINATE_DEAD_MEMBERS_PASS_H_
#define SOURCE_OPT_ELIMINATE_DEAD_MEMBERS_PASS_H_

#include "source/opt/def_use_manager.h"
#include "source/opt/function.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// Remove unused members from structures.  The remaining members will remain at
// the same offset.
class EliminateDeadMembersPass : public MemPass {
 public:
  const char* name() const override { return "eliminate-dead-members"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisCombinators | IRContext::kAnalysisCFG |
           IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisLoopAnalysis |
           IRContext::kAnalysisScalarEvolution |
           IRContext::kAnalysisRegisterPressure |
           IRContext::kAnalysisValueNumberTable |
           IRContext::kAnalysisStructuredCFG |
           IRContext::kAnalysisBuiltinVarId |
           IRContext::kAnalysisIdToFuncMapping;
  }

 private:
  // Populate |used_members_| with the member of structures that are live in the
  // current context.
  void FindLiveMembers();

  // Add to |used_members_| the member of structures that are live in
  // |function|.
  void FindLiveMembers(const Function& function);
  // Add to |used_members_| the member of structures that are live in |inst|.
  void FindLiveMembers(const Instruction* inst);

  // Add to |used_members_| the members that are live in the |OpStore|
  // instruction |inst|.
  void MarkMembersAsLiveForStore(const Instruction* inst);

  // Add to |used_members_| the members that are live in the |OpCopyMemory*|
  // instruction |inst|.
  void MarkMembersAsLiveForCopyMemory(const Instruction* inst);

  // Add to |used_members_| the members that are live in the
  // |OpCompositeExtract| instruction |inst|.
  void MarkMembersAsLiveForExtract(const Instruction* inst);

  // Add to |used_members_| the members that are live in the |Op*AccessChain|
  // instruction |inst|.
  void MarkMembersAsLiveForAccessChain(const Instruction* inst);

  // Add the member referenced by the OpArrayLength instruction |inst| to
  // |uses_members_|.
  void MarkMembersAsLiveForArrayLength(const Instruction* inst);

  // Remove dead members from structs and updates any instructions that need to
  // be updated as a consequence.  Return true if something changed.
  bool RemoveDeadMembers();

  // Update |inst|, which must be an |OpMemberName| or |OpMemberDecorate|
  // instruction, so it references the correct member after the struct is
  // updated.  Return true if something changed.
  bool UpdateOpMemberNameOrDecorate(Instruction* inst);

  // Update |inst|, which must be an |OpGroupMemberDecorate| instruction, so it
  // references the correct member after the struct is updated.  Return true if
  // something changed.
  bool UpdateOpGroupMemberDecorate(Instruction* inst);

  // Update the |OpTypeStruct| instruction |inst| my removing the members that
  // are not live.  Return true if something changed.
  bool UpdateOpTypeStruct(Instruction* inst);

  // Update the |OpConstantComposite| instruction |inst| to match the change
  // made to the type that was being generated.  Return true if something
  // changed.
  bool UpdateConstantComposite(Instruction* inst);

  // Update the |Op*AccessChain| instruction |inst| to reference the correct
  // members. All members referenced in the access chain must be live.  This
  // function must be called after the |OpTypeStruct| instruction for the type
  // has been updated.  Return true if something changed.
  bool UpdateAccessChain(Instruction* inst);

  // Update the |OpCompositeExtract| instruction |inst| to reference the correct
  // members. All members referenced in the instruction must be live.  This
  // function must be called after the |OpTypeStruct| instruction for the type
  // has been updated.  Return true if something changed.
  bool UpdateCompsiteExtract(Instruction* inst);

  // Update the |OpCompositeInsert| instruction |inst| to reference the correct
  // members. If the member being inserted is not live, then |inst| is killed.
  // This function must be called after the |OpTypeStruct| instruction for the
  // type has been updated.  Return true if something changed.
  bool UpdateCompositeInsert(Instruction* inst);

  // Update the |OpArrayLength| instruction |inst| to reference the correct
  // member. The member referenced in the instruction must be live.  Return true
  // if something changed.
  bool UpdateOpArrayLength(Instruction* inst);

  // Add all of the members of type |type_id| and members of any subtypes to
  // |used_members_|.
  void MarkTypeAsFullyUsed(uint32_t type_id);

  // Add all of the members of the type of the operand |in_idx| in |inst| and
  // members of any subtypes to |uses_members_|.
  void MarkOperandTypeAsFullyUsed(const Instruction* inst, uint32_t in_idx);

  // Return the index of the member that use to be the |member_idx|th member of
  // |type_id|.  If the member has been removed, |kRemovedMember| is returned.
  uint32_t GetNewMemberIndex(uint32_t type_id, uint32_t member_idx);

  // A map from a type id to a set of indices representing the members of the
  // type that are used, and must be kept.
  std::unordered_map<uint32_t, std::set<uint32_t>> used_members_;
  void MarkStructOperandsAsFullyUsed(const Instruction* inst);
  void MarkPointeeTypeAsFullUsed(uint32_t ptr_type_id);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_ELIMINATE_DEAD_MEMBERS_PASS_H_
