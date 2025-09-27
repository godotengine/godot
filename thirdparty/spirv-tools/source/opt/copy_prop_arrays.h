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

#ifndef SOURCE_OPT_COPY_PROP_ARRAYS_H_
#define SOURCE_OPT_COPY_PROP_ARRAYS_H_

#include <memory>
#include <vector>

#include "source/opt/mem_pass.h"

namespace spvtools {
namespace opt {

// This pass implements a simple array copy propagation.  It does not do a full
// array data flow.  It looks for simple cases that meet the following
// conditions:
//
// 1) The source must never be stored to.
// 2) The target must be stored to exactly once.
// 3) The store to the target must be a store to the entire array, and be a
// copy of the entire source.
// 4) All loads of the target must be dominated by the store.
//
// The hard part is keeping all of the types correct.  We do not want to
// have to do too large a search to update everything, which may not be
// possible, so we give up if we see any instruction that might be hard to
// update.

class CopyPropagateArrays : public MemPass {
 public:
  const char* name() const override { return "copy-propagate-arrays"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse | IRContext::kAnalysisCFG |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisDecorations |
           IRContext::kAnalysisDominatorAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Represents one index in the OpAccessChain instruction. It can be either
  // an instruction's result_id (OpConstant by ex), or a immediate value.
  // Immediate values are used to prepare the final access chain without
  // creating OpConstant instructions until done.
  struct AccessChainEntry {
    bool is_result_id;
    union {
      uint32_t result_id;
      uint32_t immediate;
    };

    bool operator!=(const AccessChainEntry& other) const {
      return other.is_result_id != is_result_id || other.result_id != result_id;
    }
  };

  // The class used to identify a particular memory object.  This memory object
  // will be owned by a particular variable, meaning that the memory is part of
  // that variable.  It could be the entire variable or a member of the
  // variable.
  class MemoryObject {
   public:
    // Construction a memory object that is owned by |var_inst|.  The iterator
    // |begin| and |end| traverse a container of integers that identify which
    // member of |var_inst| this memory object will represent.  These integers
    // are interpreted the same way they would be in an |OpAccessChain|
    // instruction.
    template <class iterator>
    MemoryObject(Instruction* var_inst, iterator begin, iterator end);

    // Change |this| to now point to the member identified by |access_chain|
    // (starting from the current member).  The elements in |access_chain| are
    // interpreted the same as the indices in the |OpAccessChain|
    // instruction.
    void PushIndirection(const std::vector<AccessChainEntry>& access_chain);

    // Change |this| to now represent the first enclosing object to which it
    // belongs.  (Remove the last element off the access_chain). It is invalid
    // to call this function if |this| does not represent a member of its owner.
    void PopIndirection() {
      assert(IsMember());
      access_chain_.pop_back();
    }

    // Returns true if |this| represents a member of its owner, and not the
    // entire variable.
    bool IsMember() const { return !access_chain_.empty(); }

    // Returns the number of members in the object represented by |this|.  If
    // |this| does not represent a composite type, the return value will be 0.
    uint32_t GetNumberOfMembers();

    // Returns the owning variable that the memory object is contained in.
    Instruction* GetVariable() const { return variable_inst_; }

    // Returns a vector of integers that can be used to access the specific
    // member that |this| represents starting from the owning variable.  These
    // values are to be interpreted the same way the indices are in an
    // |OpAccessChain| instruction.
    const std::vector<AccessChainEntry>& AccessChain() const {
      return access_chain_;
    }

    // Converts all immediate values in the AccessChain their OpConstant
    // equivalent.
    void BuildConstants();

    // Returns the type id of the pointer type that can be used to point to this
    // memory object.
    uint32_t GetPointerTypeId(const CopyPropagateArrays* pass) const {
      analysis::DefUseManager* def_use_mgr =
          GetVariable()->context()->get_def_use_mgr();
      analysis::TypeManager* type_mgr =
          GetVariable()->context()->get_type_mgr();

      Instruction* var_pointer_inst =
          def_use_mgr->GetDef(GetVariable()->type_id());

      uint32_t member_type_id = pass->GetMemberTypeId(
          var_pointer_inst->GetSingleWordInOperand(1), GetAccessIds());

      uint32_t member_pointer_type_id = type_mgr->FindPointerToType(
          member_type_id, static_cast<spv::StorageClass>(
                              var_pointer_inst->GetSingleWordInOperand(0)));
      return member_pointer_type_id;
    }

    // Returns the storage class of the memory object.
    spv::StorageClass GetStorageClass() const {
      analysis::TypeManager* type_mgr =
          GetVariable()->context()->get_type_mgr();
      const analysis::Pointer* pointer_type =
          type_mgr->GetType(GetVariable()->type_id())->AsPointer();
      return pointer_type->storage_class();
    }

    // Returns true if |other| represents memory that is contains inside of the
    // memory represented by |this|.
    bool Contains(MemoryObject* other);

   private:
    // The variable that owns this memory object.
    Instruction* variable_inst_;

    // The access chain to reach the particular member the memory object
    // represents.  It should be interpreted the same way the indices in an
    // |OpAccessChain| are interpreted.
    std::vector<AccessChainEntry> access_chain_;
    std::vector<uint32_t> GetAccessIds() const;
  };

  // Returns the memory object being stored to |var_inst| in the store
  // instruction |store_inst|, if one exists, that can be used in place of
  // |var_inst| in all of the loads of |var_inst|.  This code is conservative
  // and only identifies very simple cases.  If no such memory object can be
  // found, the return value is |nullptr|.
  std::unique_ptr<CopyPropagateArrays::MemoryObject> FindSourceObjectIfPossible(
      Instruction* var_inst, Instruction* store_inst);

  // Replaces all loads of |var_inst| with a load from |source| instead.
  // |insertion_pos| is a position where it is possible to construct the
  // address of |source| and also dominates all of the loads of |var_inst|.
  void PropagateObject(Instruction* var_inst, MemoryObject* source,
                       Instruction* insertion_pos);

  // Returns true if all of the references to |ptr_inst| can be rewritten and
  // are dominated by |store_inst|.
  bool HasValidReferencesOnly(Instruction* ptr_inst, Instruction* store_inst);

  // Returns a memory object that at one time was equivalent to the value in
  // |result|.  If no such memory object exists, the return value is |nullptr|.
  std::unique_ptr<MemoryObject> GetSourceObjectIfAny(uint32_t result);

  // Returns the memory object that is loaded by |load_inst|.  If a memory
  // object cannot be identified, the return value is |nullptr|.  The opcode of
  // |load_inst| must be |OpLoad|.
  std::unique_ptr<MemoryObject> BuildMemoryObjectFromLoad(
      Instruction* load_inst);

  // Returns the memory object that at some point was equivalent to the result
  // of |extract_inst|.  If a memory object cannot be identified, the return
  // value is |nullptr|.  The opcode of |extract_inst| must be
  // |OpCompositeExtract|.
  std::unique_ptr<MemoryObject> BuildMemoryObjectFromExtract(
      Instruction* extract_inst);

  // Returns the memory object that at some point was equivalent to the result
  // of |construct_inst|.  If a memory object cannot be identified, the return
  // value is |nullptr|.  The opcode of |constuct_inst| must be
  // |OpCompositeConstruct|.
  std::unique_ptr<MemoryObject> BuildMemoryObjectFromCompositeConstruct(
      Instruction* conststruct_inst);

  // Returns the memory object that at some point was equivalent to the result
  // of |insert_inst|.  If a memory object cannot be identified, the return
  // value is |nullptr\.  The opcode of |insert_inst| must be
  // |OpCompositeInsert|.  This function looks for a series of
  // |OpCompositeInsert| instructions that insert the elements one at a time in
  // order from beginning to end.
  std::unique_ptr<MemoryObject> BuildMemoryObjectFromInsert(
      Instruction* insert_inst);

  // Return true if the given entry can represent the given value.
  bool IsAccessChainIndexValidAndEqualTo(const AccessChainEntry& entry,
                                         uint32_t value) const;

  // Return true if |type_id| is a pointer type whose pointee type is an array.
  bool IsPointerToArrayType(uint32_t type_id);

  // Returns true if there are not stores using |ptr_inst| or something derived
  // from it.
  bool HasNoStores(Instruction* ptr_inst);

  // Creates an |OpAccessChain| instruction whose result is a pointer the memory
  // represented by |source|.  The new instruction will be placed before
  // |insertion_point|.  |insertion_point| must be part of a function.  Returns
  // the new instruction.
  Instruction* BuildNewAccessChain(Instruction* insertion_point,
                                   MemoryObject* source) const;

  // Rewrites all uses of |original_ptr| to use |new_pointer_inst| updating
  // types of other instructions as needed.  This function should not be called
  // if |CanUpdateUses(original_ptr_inst, new_pointer_inst->type_id())| returns
  // false.
  void UpdateUses(Instruction* original_ptr_inst,
                  Instruction* new_pointer_inst);

  // Return true if |UpdateUses| is able to change all of the uses of
  // |original_ptr_inst| to |type_id| and still have valid code.
  bool CanUpdateUses(Instruction* original_ptr_inst, uint32_t type_id);

  // Returns a store to |var_inst| that writes to the entire variable, and is
  // the only store that does so.  Note it does not look through OpAccessChain
  // instruction, so partial stores are not considered.
  Instruction* FindStoreInstruction(const Instruction* var_inst) const;

  // Return the type id of the member of the type |id| access using
  // |access_chain|. The elements of |access_chain| are to be interpreted the
  // same way the indexes are used in an |OpCompositeExtract| instruction.
  uint32_t GetMemberTypeId(uint32_t id,
                           const std::vector<uint32_t>& access_chain) const;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_COPY_PROP_ARRAYS_H_
