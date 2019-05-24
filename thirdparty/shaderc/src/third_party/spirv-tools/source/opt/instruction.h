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

#ifndef SOURCE_OPT_INSTRUCTION_H_
#define SOURCE_OPT_INSTRUCTION_H_

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "source/opcode.h"
#include "source/operand.h"
#include "source/util/ilist_node.h"
#include "source/util/small_vector.h"

#include "source/latest_version_glsl_std_450_header.h"
#include "source/latest_version_spirv_header.h"
#include "source/opt/reflect.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace opt {

class Function;
class IRContext;
class Module;
class InstructionList;

// Relaxed logical addressing:
//
// In the logical addressing model, pointers cannot be stored or loaded.  This
// is a useful assumption because it simplifies the aliasing significantly.
// However, for the purpose of legalizing code generated from HLSL, we will have
// to allow storing and loading of pointers to opaque objects and runtime
// arrays.  This relaxation of the rule still implies that function and private
// scope variables do not have any aliasing, so we can treat them as before.
// This will be call the relaxed logical addressing model.
//
// This relaxation of the rule will be allowed by |GetBaseAddress|, but it will
// enforce that no other pointers are stored or loaded.

// About operand:
//
// In the SPIR-V specification, the term "operand" is used to mean any single
// SPIR-V word following the leading wordcount-opcode word. Here, the term
// "operand" is used to mean a *logical* operand. A logical operand may consist
// of multiple SPIR-V words, which together make up the same component. For
// example, a logical operand of a 64-bit integer needs two words to express.
//
// Further, we categorize logical operands into *in* and *out* operands.
// In operands are operands actually serve as input to operations, while out
// operands are operands that represent ids generated from operations (result
// type id or result id). For example, for "OpIAdd %rtype %rid %inop1 %inop2",
// "%inop1" and "%inop2" are in operands, while "%rtype" and "%rid" are out
// operands.

// A *logical* operand to a SPIR-V instruction. It can be the type id, result
// id, or other additional operands carried in an instruction.
struct Operand {
  using OperandData = utils::SmallVector<uint32_t, 2>;
  Operand(spv_operand_type_t t, OperandData&& w)
      : type(t), words(std::move(w)) {}

  Operand(spv_operand_type_t t, const OperandData& w) : type(t), words(w) {}

  spv_operand_type_t type;  // Type of this logical operand.
  OperandData words;        // Binary segments of this logical operand.

  friend bool operator==(const Operand& o1, const Operand& o2) {
    return o1.type == o2.type && o1.words == o2.words;
  }

  // TODO(antiagainst): create fields for literal number kind, width, etc.
};

inline bool operator!=(const Operand& o1, const Operand& o2) {
  return !(o1 == o2);
}

// A SPIR-V instruction. It contains the opcode and any additional logical
// operand, including the result id (if any) and result type id (if any). It
// may also contain line-related debug instruction (OpLine, OpNoLine) directly
// appearing before this instruction. Note that the result id of an instruction
// should never change after the instruction being built. If the result id
// needs to change, the user should create a new instruction instead.
class Instruction : public utils::IntrusiveNodeBase<Instruction> {
 public:
  using OperandList = std::vector<Operand>;
  using iterator = OperandList::iterator;
  using const_iterator = OperandList::const_iterator;

  // Creates a default OpNop instruction.
  // This exists solely for containers that can't do without. Should be removed.
  Instruction()
      : utils::IntrusiveNodeBase<Instruction>(),
        context_(nullptr),
        opcode_(SpvOpNop),
        has_type_id_(false),
        has_result_id_(false),
        unique_id_(0) {}

  // Creates a default OpNop instruction.
  Instruction(IRContext*);
  // Creates an instruction with the given opcode |op| and no additional logical
  // operands.
  Instruction(IRContext*, SpvOp);
  // Creates an instruction using the given spv_parsed_instruction_t |inst|. All
  // the data inside |inst| will be copied and owned in this instance. And keep
  // record of line-related debug instructions |dbg_line| ahead of this
  // instruction, if any.
  Instruction(IRContext* c, const spv_parsed_instruction_t& inst,
              std::vector<Instruction>&& dbg_line = {});

  // Creates an instruction with the given opcode |op|, type id: |ty_id|,
  // result id: |res_id| and input operands: |in_operands|.
  Instruction(IRContext* c, SpvOp op, uint32_t ty_id, uint32_t res_id,
              const OperandList& in_operands);

  // TODO: I will want to remove these, but will first have to remove the use of
  // std::vector<Instruction>.
  Instruction(const Instruction&) = default;
  Instruction& operator=(const Instruction&) = default;

  Instruction(Instruction&&);
  Instruction& operator=(Instruction&&);

  virtual ~Instruction() = default;

  // Returns a newly allocated instruction that has the same operands, result,
  // and type as |this|.  The new instruction is not linked into any list.
  // It is the responsibility of the caller to make sure that the storage is
  // removed. It is the caller's responsibility to make sure that there is only
  // one instruction for each result id.
  Instruction* Clone(IRContext* c) const;

  IRContext* context() const { return context_; }

  SpvOp opcode() const { return opcode_; }
  // Sets the opcode of this instruction to a specific opcode. Note this may
  // invalidate the instruction.
  // TODO(qining): Remove this function when instruction building and insertion
  // is well implemented.
  void SetOpcode(SpvOp op) { opcode_ = op; }
  uint32_t type_id() const {
    return has_type_id_ ? GetSingleWordOperand(0) : 0;
  }
  uint32_t result_id() const {
    return has_result_id_ ? GetSingleWordOperand(has_type_id_ ? 1 : 0) : 0;
  }
  uint32_t unique_id() const {
    assert(unique_id_ != 0);
    return unique_id_;
  }
  // Returns the vector of line-related debug instructions attached to this
  // instruction and the caller can directly modify them.
  std::vector<Instruction>& dbg_line_insts() { return dbg_line_insts_; }
  const std::vector<Instruction>& dbg_line_insts() const {
    return dbg_line_insts_;
  }

  // Same semantics as in the base class except the list the InstructionList
  // containing |pos| will now assume ownership of |this|.
  // inline void MoveBefore(Instruction* pos);
  // inline void InsertAfter(Instruction* pos);

  // Begin and end iterators for operands.
  iterator begin() { return operands_.begin(); }
  iterator end() { return operands_.end(); }
  const_iterator begin() const { return operands_.cbegin(); }
  const_iterator end() const { return operands_.cend(); }
  // Const begin and end iterators for operands.
  const_iterator cbegin() const { return operands_.cbegin(); }
  const_iterator cend() const { return operands_.cend(); }

  // Gets the number of logical operands.
  uint32_t NumOperands() const {
    return static_cast<uint32_t>(operands_.size());
  }
  // Gets the number of SPIR-V words occupied by all logical operands.
  uint32_t NumOperandWords() const {
    return NumInOperandWords() + TypeResultIdCount();
  }
  // Gets the |index|-th logical operand.
  inline Operand& GetOperand(uint32_t index);
  inline const Operand& GetOperand(uint32_t index) const;
  // Adds |operand| to the list of operands of this instruction.
  // It is the responsibility of the caller to make sure
  // that the instruction remains valid.
  inline void AddOperand(Operand&& operand);
  // Gets the |index|-th logical operand as a single SPIR-V word. This method is
  // not expected to be used with logical operands consisting of multiple SPIR-V
  // words.
  uint32_t GetSingleWordOperand(uint32_t index) const;
  // Sets the |index|-th in-operand's data to the given |data|.
  inline void SetInOperand(uint32_t index, Operand::OperandData&& data);
  // Sets the |index|-th operand's data to the given |data|.
  // This is for in-operands modification only, but with |index| expressed in
  // terms of operand index rather than in-operand index.
  inline void SetOperand(uint32_t index, Operand::OperandData&& data);
  // Replace all of the in operands with those in |new_operands|.
  inline void SetInOperands(OperandList&& new_operands);
  // Sets the result type id.
  inline void SetResultType(uint32_t ty_id);
  // Sets the result id
  inline void SetResultId(uint32_t res_id);
  inline bool HasResultId() const { return has_result_id_; }
  // Remove the |index|-th operand
  void RemoveOperand(uint32_t index) {
    operands_.erase(operands_.begin() + index);
  }

  // The following methods are similar to the above, but are for in operands.
  uint32_t NumInOperands() const {
    return static_cast<uint32_t>(operands_.size() - TypeResultIdCount());
  }
  uint32_t NumInOperandWords() const;
  Operand& GetInOperand(uint32_t index) {
    return GetOperand(index + TypeResultIdCount());
  }
  const Operand& GetInOperand(uint32_t index) const {
    return GetOperand(index + TypeResultIdCount());
  }
  uint32_t GetSingleWordInOperand(uint32_t index) const {
    return GetSingleWordOperand(index + TypeResultIdCount());
  }
  void RemoveInOperand(uint32_t index) {
    operands_.erase(operands_.begin() + index + TypeResultIdCount());
  }

  // Returns true if this instruction is OpNop.
  inline bool IsNop() const;
  // Turns this instruction to OpNop. This does not clear out all preceding
  // line-related debug instructions.
  inline void ToNop();

  // Runs the given function |f| on this instruction and optionally on the
  // preceding debug line instructions.  The function will always be run
  // if this is itself a debug line instruction.
  inline void ForEachInst(const std::function<void(Instruction*)>& f,
                          bool run_on_debug_line_insts = false);
  inline void ForEachInst(const std::function<void(const Instruction*)>& f,
                          bool run_on_debug_line_insts = false) const;

  // Runs the given function |f| on this instruction and optionally on the
  // preceding debug line instructions.  The function will always be run
  // if this is itself a debug line instruction. If |f| returns false,
  // iteration is terminated and this function returns false.
  inline bool WhileEachInst(const std::function<bool(Instruction*)>& f,
                            bool run_on_debug_line_insts = false);
  inline bool WhileEachInst(const std::function<bool(const Instruction*)>& f,
                            bool run_on_debug_line_insts = false) const;

  // Runs the given function |f| on all operand ids.
  //
  // |f| should not transform an ID into 0, as 0 is an invalid ID.
  inline void ForEachId(const std::function<void(uint32_t*)>& f);
  inline void ForEachId(const std::function<void(const uint32_t*)>& f) const;

  // Runs the given function |f| on all "in" operand ids.
  inline void ForEachInId(const std::function<void(uint32_t*)>& f);
  inline void ForEachInId(const std::function<void(const uint32_t*)>& f) const;

  // Runs the given function |f| on all "in" operand ids. If |f| returns false,
  // iteration is terminated and this function returns false.
  inline bool WhileEachInId(const std::function<bool(uint32_t*)>& f);
  inline bool WhileEachInId(
      const std::function<bool(const uint32_t*)>& f) const;

  // Runs the given function |f| on all "in" operands.
  inline void ForEachInOperand(const std::function<void(uint32_t*)>& f);
  inline void ForEachInOperand(
      const std::function<void(const uint32_t*)>& f) const;

  // Runs the given function |f| on all "in" operands. If |f| returns false,
  // iteration is terminated and this function return false.
  inline bool WhileEachInOperand(const std::function<bool(uint32_t*)>& f);
  inline bool WhileEachInOperand(
      const std::function<bool(const uint32_t*)>& f) const;

  // Returns true if any operands can be labels
  inline bool HasLabels() const;

  // Pushes the binary segments for this instruction into the back of *|binary|.
  void ToBinaryWithoutAttachedDebugInsts(std::vector<uint32_t>* binary) const;

  // Replaces the operands to the instruction with |new_operands|. The caller
  // is responsible for building a complete and valid list of operands for
  // this instruction.
  void ReplaceOperands(const OperandList& new_operands);

  // Returns true if the instruction annotates an id with a decoration.
  inline bool IsDecoration() const;

  // Returns true if the instruction is known to be a load from read-only
  // memory.
  bool IsReadOnlyLoad() const;

  // Returns the instruction that gives the base address of an address
  // calculation.  The instruction must be a load, as defined by |IsLoad|,
  // store, copy, or access chain instruction.  In logical addressing mode, will
  // return an OpVariable or OpFunctionParameter instruction. For relaxed
  // logical addressing, it would also return a load of a pointer to an opaque
  // object.  For physical addressing mode, could return other types of
  // instructions.
  Instruction* GetBaseAddress() const;

  // Returns true if the instruction loads from memory or samples an image, and
  // stores the result into an id. It considers only core instructions.
  // Memory-to-memory instructions are not considered loads.
  inline bool IsLoad() const;

  // Returns true if the instruction declares a variable that is read-only.
  bool IsReadOnlyVariable() const;

  // The following functions check for the various descriptor types defined in
  // the Vulkan specification section 13.1.

  // Returns true if the instruction defines a pointer type that points to a
  // storage image.
  bool IsVulkanStorageImage() const;

  // Returns true if the instruction defines a pointer type that points to a
  // sampled image.
  bool IsVulkanSampledImage() const;

  // Returns true if the instruction defines a pointer type that points to a
  // storage texel buffer.
  bool IsVulkanStorageTexelBuffer() const;

  // Returns true if the instruction defines a pointer type that points to a
  // storage buffer.
  bool IsVulkanStorageBuffer() const;

  // Returns true if the instruction defines a pointer type that points to a
  // uniform buffer.
  bool IsVulkanUniformBuffer() const;

  // Returns true if the instruction is an atom operation that uses original
  // value.
  inline bool IsAtomicWithLoad() const;

  // Returns true if the instruction is an atom operation.
  inline bool IsAtomicOp() const;

  // Returns true if this instruction is a branch or switch instruction (either
  // conditional or not).
  bool IsBranch() const { return spvOpcodeIsBranch(opcode()); }

  // Returns true if this instruction causes the function to finish execution
  // and return to its caller
  bool IsReturn() const { return spvOpcodeIsReturn(opcode()); }

  // Returns true if this instruction exits this function or aborts execution.
  bool IsReturnOrAbort() const { return spvOpcodeIsReturnOrAbort(opcode()); }

  // Returns the id for the |element|'th subtype. If the |this| is not a
  // composite type, this function returns 0.
  uint32_t GetTypeComponent(uint32_t element) const;

  // Returns true if this instruction is a basic block terminator.
  bool IsBlockTerminator() const {
    return spvOpcodeIsBlockTerminator(opcode());
  }

  // Returns true if |this| is an instruction that define an opaque type.  Since
  // runtime array have similar characteristics they are included as opaque
  // types.
  bool IsOpaqueType() const;

  // Returns true if |this| is an instruction which could be folded into a
  // constant value.
  bool IsFoldable() const;

  // Returns true if |this| is an instruction which could be folded into a
  // constant value by |FoldScalar|.
  bool IsFoldableByFoldScalar() const;

  // Returns true if we are allowed to fold or otherwise manipulate the
  // instruction that defines |id| in the given context. This includes not
  // handling NaN values.
  bool IsFloatingPointFoldingAllowed() const;

  inline bool operator==(const Instruction&) const;
  inline bool operator!=(const Instruction&) const;
  inline bool operator<(const Instruction&) const;

  Instruction* InsertBefore(std::vector<std::unique_ptr<Instruction>>&& list);
  Instruction* InsertBefore(std::unique_ptr<Instruction>&& i);
  using utils::IntrusiveNodeBase<Instruction>::InsertBefore;

  // Returns true if |this| is an instruction defining a constant, but not a
  // Spec constant.
  inline bool IsConstant() const;

  // Returns true if |this| is an instruction with an opcode safe to move
  bool IsOpcodeCodeMotionSafe() const;

  // Pretty-prints |inst|.
  //
  // Provides the disassembly of a specific instruction. Utilizes |inst|'s
  // context to provide the correct interpretation of types, constants, etc.
  //
  // |options| are the disassembly options. SPV_BINARY_TO_TEXT_OPTION_NO_HEADER
  // is always added to |options|.
  std::string PrettyPrint(uint32_t options = 0u) const;

  // Returns true if the result can be a vector and the result of each component
  // depends on the corresponding component of any vector inputs.
  bool IsScalarizable() const;

  // Return true if the only effect of this instructions is the result.
  bool IsOpcodeSafeToDelete() const;

  // Returns true if it is valid to use the result of |inst| as the base
  // pointer for a load or store.  In this case, valid is defined by the relaxed
  // logical addressing rules when using logical addressing.  Normal validation
  // rules for physical addressing.
  bool IsValidBasePointer() const;

  // Dump this instruction on stderr.  Useful when running interactive
  // debuggers.
  void Dump() const;

 private:
  // Returns the total count of result type id and result id.
  uint32_t TypeResultIdCount() const {
    if (has_type_id_ && has_result_id_) return 2;
    if (has_type_id_ || has_result_id_) return 1;
    return 0;
  }

  // Returns true if the instruction declares a variable that is read-only.  The
  // first version assumes the module is a shader module.  The second assumes a
  // kernel.
  bool IsReadOnlyVariableShaders() const;
  bool IsReadOnlyVariableKernel() const;

  // Returns true if the result of |inst| can be used as the base image for an
  // instruction that samples a image, reads an image, or writes to an image.
  bool IsValidBaseImage() const;

  IRContext* context_;  // IR Context
  SpvOp opcode_;        // Opcode
  bool has_type_id_;    // True if the instruction has a type id
  bool has_result_id_;  // True if the instruction has a result id
  uint32_t unique_id_;  // Unique instruction id
  // All logical operands, including result type id and result id.
  OperandList operands_;
  // Opline and OpNoLine instructions preceding this instruction. Note that for
  // Instructions representing OpLine or OpNonLine itself, this field should be
  // empty.
  std::vector<Instruction> dbg_line_insts_;

  friend InstructionList;
};

// Pretty-prints |inst| to |str| and returns |str|.
//
// Provides the disassembly of a specific instruction. Utilizes |inst|'s context
// to provide the correct interpretation of types, constants, etc.
//
// Disassembly uses raw ids (not pretty printed names).
std::ostream& operator<<(std::ostream& str, const Instruction& inst);

inline bool Instruction::operator==(const Instruction& other) const {
  return unique_id() == other.unique_id();
}

inline bool Instruction::operator!=(const Instruction& other) const {
  return !(*this == other);
}

inline bool Instruction::operator<(const Instruction& other) const {
  return unique_id() < other.unique_id();
}

inline Operand& Instruction::GetOperand(uint32_t index) {
  assert(index < operands_.size() && "operand index out of bound");
  return operands_[index];
}

inline const Operand& Instruction::GetOperand(uint32_t index) const {
  assert(index < operands_.size() && "operand index out of bound");
  return operands_[index];
}

inline void Instruction::AddOperand(Operand&& operand) {
  operands_.push_back(std::move(operand));
}

inline void Instruction::SetInOperand(uint32_t index,
                                      Operand::OperandData&& data) {
  SetOperand(index + TypeResultIdCount(), std::move(data));
}

inline void Instruction::SetOperand(uint32_t index,
                                    Operand::OperandData&& data) {
  assert(index < operands_.size() && "operand index out of bound");
  assert(index >= TypeResultIdCount() && "operand is not a in-operand");
  operands_[index].words = std::move(data);
}

inline void Instruction::SetInOperands(OperandList&& new_operands) {
  // Remove the old in operands.
  operands_.erase(operands_.begin() + TypeResultIdCount(), operands_.end());
  // Add the new in operands.
  operands_.insert(operands_.end(), new_operands.begin(), new_operands.end());
}

inline void Instruction::SetResultId(uint32_t res_id) {
  // TODO(dsinclair): Allow setting a result id if there wasn't one
  // previously. Need to make room in the operands_ array to place the result,
  // and update the has_result_id_ flag.
  assert(has_result_id_);

  // TODO(dsinclair): Allow removing the result id. This needs to make sure,
  // if there was a result id previously to remove it from the operands_ array
  // and reset the has_result_id_ flag.
  assert(res_id != 0);

  auto ridx = has_type_id_ ? 1 : 0;
  operands_[ridx].words = {res_id};
}

inline void Instruction::SetResultType(uint32_t ty_id) {
  // TODO(dsinclair): Allow setting a type id if there wasn't one
  // previously. Need to make room in the operands_ array to place the result,
  // and update the has_type_id_ flag.
  assert(has_type_id_);

  // TODO(dsinclair): Allow removing the type id. This needs to make sure,
  // if there was a type id previously to remove it from the operands_ array
  // and reset the has_type_id_ flag.
  assert(ty_id != 0);

  operands_.front().words = {ty_id};
}

inline bool Instruction::IsNop() const {
  return opcode_ == SpvOpNop && !has_type_id_ && !has_result_id_ &&
         operands_.empty();
}

inline void Instruction::ToNop() {
  opcode_ = SpvOpNop;
  has_type_id_ = false;
  has_result_id_ = false;
  operands_.clear();
}

inline bool Instruction::WhileEachInst(
    const std::function<bool(Instruction*)>& f, bool run_on_debug_line_insts) {
  if (run_on_debug_line_insts) {
    for (auto& dbg_line : dbg_line_insts_) {
      if (!f(&dbg_line)) return false;
    }
  }
  return f(this);
}

inline bool Instruction::WhileEachInst(
    const std::function<bool(const Instruction*)>& f,
    bool run_on_debug_line_insts) const {
  if (run_on_debug_line_insts) {
    for (auto& dbg_line : dbg_line_insts_) {
      if (!f(&dbg_line)) return false;
    }
  }
  return f(this);
}

inline void Instruction::ForEachInst(const std::function<void(Instruction*)>& f,
                                     bool run_on_debug_line_insts) {
  WhileEachInst(
      [&f](Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts);
}

inline void Instruction::ForEachInst(
    const std::function<void(const Instruction*)>& f,
    bool run_on_debug_line_insts) const {
  WhileEachInst(
      [&f](const Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts);
}

inline void Instruction::ForEachId(const std::function<void(uint32_t*)>& f) {
  for (auto& opnd : operands_)
    if (spvIsIdType(opnd.type)) f(&opnd.words[0]);
}

inline void Instruction::ForEachId(
    const std::function<void(const uint32_t*)>& f) const {
  for (const auto& opnd : operands_)
    if (spvIsIdType(opnd.type)) f(&opnd.words[0]);
}

inline bool Instruction::WhileEachInId(
    const std::function<bool(uint32_t*)>& f) {
  for (auto& opnd : operands_) {
    if (spvIsInIdType(opnd.type)) {
      if (!f(&opnd.words[0])) return false;
    }
  }
  return true;
}

inline bool Instruction::WhileEachInId(
    const std::function<bool(const uint32_t*)>& f) const {
  for (const auto& opnd : operands_) {
    if (spvIsInIdType(opnd.type)) {
      if (!f(&opnd.words[0])) return false;
    }
  }
  return true;
}

inline void Instruction::ForEachInId(const std::function<void(uint32_t*)>& f) {
  WhileEachInId([&f](uint32_t* id) {
    f(id);
    return true;
  });
}

inline void Instruction::ForEachInId(
    const std::function<void(const uint32_t*)>& f) const {
  WhileEachInId([&f](const uint32_t* id) {
    f(id);
    return true;
  });
}

inline bool Instruction::WhileEachInOperand(
    const std::function<bool(uint32_t*)>& f) {
  for (auto& opnd : operands_) {
    switch (opnd.type) {
      case SPV_OPERAND_TYPE_RESULT_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
        break;
      default:
        if (!f(&opnd.words[0])) return false;
        break;
    }
  }
  return true;
}

inline bool Instruction::WhileEachInOperand(
    const std::function<bool(const uint32_t*)>& f) const {
  for (const auto& opnd : operands_) {
    switch (opnd.type) {
      case SPV_OPERAND_TYPE_RESULT_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
        break;
      default:
        if (!f(&opnd.words[0])) return false;
        break;
    }
  }
  return true;
}

inline void Instruction::ForEachInOperand(
    const std::function<void(uint32_t*)>& f) {
  WhileEachInOperand([&f](uint32_t* op) {
    f(op);
    return true;
  });
}

inline void Instruction::ForEachInOperand(
    const std::function<void(const uint32_t*)>& f) const {
  WhileEachInOperand([&f](const uint32_t* op) {
    f(op);
    return true;
  });
}

inline bool Instruction::HasLabels() const {
  switch (opcode_) {
    case SpvOpSelectionMerge:
    case SpvOpBranch:
    case SpvOpLoopMerge:
    case SpvOpBranchConditional:
    case SpvOpSwitch:
    case SpvOpPhi:
      return true;
      break;
    default:
      break;
  }
  return false;
}

bool Instruction::IsDecoration() const {
  return spvOpcodeIsDecoration(opcode());
}

bool Instruction::IsLoad() const { return spvOpcodeIsLoad(opcode()); }

bool Instruction::IsAtomicWithLoad() const {
  return spvOpcodeIsAtomicWithLoad(opcode());
}

bool Instruction::IsAtomicOp() const { return spvOpcodeIsAtomicOp(opcode()); }

bool Instruction::IsConstant() const {
  return IsCompileTimeConstantInst(opcode());
}
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INSTRUCTION_H_
