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

#include "NonSemanticShaderDebugInfo100.h"
#include "OpenCLDebugInfo100.h"
#include "source/binary.h"
#include "source/common_debug_info.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/latest_version_spirv_header.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/opt/reflect.h"
#include "source/util/ilist_node.h"
#include "source/util/small_vector.h"
#include "source/util/string_utils.h"
#include "spirv-tools/libspirv.h"

constexpr uint32_t kNoDebugScope = 0;
constexpr uint32_t kNoInlinedAt = 0;

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

  template <class InputIt>
  Operand(spv_operand_type_t t, InputIt firstOperandData,
          InputIt lastOperandData)
      : type(t), words(firstOperandData, lastOperandData) {}

  spv_operand_type_t type;  // Type of this logical operand.
  OperandData words;        // Binary segments of this logical operand.

  uint32_t AsId() const {
    assert(spvIsIdType(type));
    assert(words.size() == 1);
    return words[0];
  }

  // Returns a string operand as a std::string.
  std::string AsString() const {
    assert(type == SPV_OPERAND_TYPE_LITERAL_STRING);
    return spvtools::utils::MakeString(words);
  }

  // Returns a literal integer operand as a uint64_t
  uint64_t AsLiteralUint64() const {
    assert(type == SPV_OPERAND_TYPE_LITERAL_INTEGER ||
           type == SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER ||
           type == SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER ||
           type == SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER);
    assert(1 <= words.size());
    assert(words.size() <= 2);
    uint64_t result = 0;
    if (words.size() > 0) {  // Needed to avoid maybe-uninitialized GCC warning
      uint32_t low = words[0];
      result = uint64_t(low);
    }
    if (words.size() > 1) {
      uint32_t high = words[1];
      result = result | (uint64_t(high) << 32);
    }
    return result;
  }

  friend bool operator==(const Operand& o1, const Operand& o2) {
    return o1.type == o2.type && o1.words == o2.words;
  }

  // TODO(antiagainst): create fields for literal number kind, width, etc.
};

inline bool operator!=(const Operand& o1, const Operand& o2) {
  return !(o1 == o2);
}

// This structure is used to represent a DebugScope instruction from
// the OpenCL.100.DebugInfo extended instruction set. Note that we can
// ignore the result id of DebugScope instruction because it is not
// used for anything. We do not keep it to reduce the size of
// structure.
// TODO: Let validator check that the result id is not used anywhere.
class DebugScope {
 public:
  DebugScope(uint32_t lexical_scope, uint32_t inlined_at)
      : lexical_scope_(lexical_scope), inlined_at_(inlined_at) {}

  inline bool operator!=(const DebugScope& d) const {
    return lexical_scope_ != d.lexical_scope_ || inlined_at_ != d.inlined_at_;
  }

  // Accessor functions for |lexical_scope_|.
  uint32_t GetLexicalScope() const { return lexical_scope_; }
  void SetLexicalScope(uint32_t scope) { lexical_scope_ = scope; }

  // Accessor functions for |inlined_at_|.
  uint32_t GetInlinedAt() const { return inlined_at_; }
  void SetInlinedAt(uint32_t at) { inlined_at_ = at; }

  // Pushes the binary segments for this DebugScope instruction into
  // the back of *|binary|.
  void ToBinary(uint32_t type_id, uint32_t result_id, uint32_t ext_set,
                std::vector<uint32_t>* binary) const;

 private:
  // The result id of the lexical scope in which this debug scope is
  // contained. The value is kNoDebugScope if there is no scope.
  uint32_t lexical_scope_;

  // The result id of DebugInlinedAt if instruction in this debug scope
  // is inlined. The value is kNoInlinedAt if it is not inlined.
  uint32_t inlined_at_;
};

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
        opcode_(spv::Op::OpNop),
        has_type_id_(false),
        has_result_id_(false),
        unique_id_(0),
        dbg_scope_(kNoDebugScope, kNoInlinedAt) {}

  // Creates a default OpNop instruction.
  Instruction(IRContext*);
  // Creates an instruction with the given opcode |op| and no additional logical
  // operands.
  Instruction(IRContext*, spv::Op);
  // Creates an instruction using the given spv_parsed_instruction_t |inst|. All
  // the data inside |inst| will be copied and owned in this instance. And keep
  // record of line-related debug instructions |dbg_line| ahead of this
  // instruction, if any.
  Instruction(IRContext* c, const spv_parsed_instruction_t& inst,
              std::vector<Instruction>&& dbg_line = {});

  Instruction(IRContext* c, const spv_parsed_instruction_t& inst,
              const DebugScope& dbg_scope);

  // Creates an instruction with the given opcode |op|, type id: |ty_id|,
  // result id: |res_id| and input operands: |in_operands|.
  Instruction(IRContext* c, spv::Op op, uint32_t ty_id, uint32_t res_id,
              const OperandList& in_operands);

  // TODO: I will want to remove these, but will first have to remove the use of
  // std::vector<Instruction>.
  Instruction(const Instruction&) = default;
  Instruction& operator=(const Instruction&) = default;

  Instruction(Instruction&&);
  Instruction& operator=(Instruction&&);

  ~Instruction() override = default;

  // Returns a newly allocated instruction that has the same operands, result,
  // and type as |this|.  The new instruction is not linked into any list.
  // It is the responsibility of the caller to make sure that the storage is
  // removed. It is the caller's responsibility to make sure that there is only
  // one instruction for each result id.
  Instruction* Clone(IRContext* c) const;

  IRContext* context() const { return context_; }

  spv::Op opcode() const { return opcode_; }
  // Sets the opcode of this instruction to a specific opcode. Note this may
  // invalidate the instruction.
  // TODO(qining): Remove this function when instruction building and insertion
  // is well implemented.
  void SetOpcode(spv::Op op) { opcode_ = op; }
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

  const Instruction* dbg_line_inst() const {
    return dbg_line_insts_.empty() ? nullptr : &dbg_line_insts_[0];
  }

  // Clear line-related debug instructions attached to this instruction.
  void clear_dbg_line_insts() { dbg_line_insts_.clear(); }

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
  inline bool HasResultType() const { return has_type_id_; }
  // Sets the result id
  inline void SetResultId(uint32_t res_id);
  inline bool HasResultId() const { return has_result_id_; }
  // Sets DebugScope.
  inline void SetDebugScope(const DebugScope& scope);
  inline const DebugScope& GetDebugScope() const { return dbg_scope_; }
  // Add debug line inst. Renew result id if Debug[No]Line
  void AddDebugLine(const Instruction* inst);
  // Updates DebugInlinedAt of DebugScope and OpLine.
  void UpdateDebugInlinedAt(uint32_t new_inlined_at);
  // Clear line-related debug instructions attached to this instruction
  // along with def-use entries.
  void ClearDbgLineInsts();
  // Return true if Shader100:Debug[No]Line
  bool IsDebugLineInst() const;
  // Return true if Op[No]Line or Shader100:Debug[No]Line
  bool IsLineInst() const;
  // Return true if OpLine or Shader100:DebugLine
  bool IsLine() const;
  // Return true if OpNoLine or Shader100:DebugNoLine
  bool IsNoLine() const;
  inline uint32_t GetDebugInlinedAt() const {
    return dbg_scope_.GetInlinedAt();
  }
  // Updates lexical scope of DebugScope and OpLine.
  void UpdateLexicalScope(uint32_t scope);
  // Updates OpLine and DebugScope based on the information of |from|.
  void UpdateDebugInfoFrom(const Instruction* from);
  // Remove the |index|-th operand
  void RemoveOperand(uint32_t index) {
    operands_.erase(operands_.begin() + index);
  }
  // Insert an operand before the |index|-th operand
  void InsertOperand(uint32_t index, Operand&& operand) {
    operands_.insert(operands_.begin() + index, operand);
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

  // Returns true if it's an OpBranchConditional instruction
  // with branch weights.
  bool HasBranchWeights() const;

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

  // Returns true if the instruction generates a pointer that is definitely
  // read-only.  This is determined by analysing the pointer type's storage
  // class and decorations that target the pointer's id.  It does not analyse
  // other instructions that the pointer may be derived from.  Thus if 'true' is
  // returned, the pointer is definitely read-only, while if 'false' is returned
  // it is possible that the pointer may actually be read-only if it is derived
  // from another pointer that is decorated as read-only.
  bool IsReadOnlyPointer() const;

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

  // Returns true if the instruction defines a variable in StorageBuffer or
  // Uniform storage class with a pointer type that points to a storage buffer.
  bool IsVulkanStorageBufferVariable() const;

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

  // Takes ownership of the instruction owned by |i| and inserts it immediately
  // before |this|. Returns the inserted instruction.
  Instruction* InsertBefore(std::unique_ptr<Instruction>&& i);
  // Takes ownership of the instructions in |list| and inserts them in order
  // immediately before |this|.  Returns the first inserted instruction.
  // Assumes the list is non-empty.
  Instruction* InsertBefore(std::vector<std::unique_ptr<Instruction>>&& list);
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

  // Returns debug opcode of an OpenCL.100.DebugInfo instruction. If
  // it is not an OpenCL.100.DebugInfo instruction, just returns
  // OpenCLDebugInfo100InstructionsMax.
  OpenCLDebugInfo100Instructions GetOpenCL100DebugOpcode() const;

  // Returns debug opcode of an NonSemantic.Shader.DebugInfo.100 instruction. If
  // it is not an NonSemantic.Shader.DebugInfo.100 instruction, just return
  // NonSemanticShaderDebugInfo100InstructionsMax.
  NonSemanticShaderDebugInfo100Instructions GetShader100DebugOpcode() const;

  // Returns debug opcode of an OpenCL.100.DebugInfo or
  // NonSemantic.Shader.DebugInfo.100 instruction. Since these overlap, we
  // return the OpenCLDebugInfo code
  CommonDebugInfoInstructions GetCommonDebugOpcode() const;

  // Returns true if it is an OpenCL.DebugInfo.100 instruction.
  bool IsOpenCL100DebugInstr() const {
    return GetOpenCL100DebugOpcode() != OpenCLDebugInfo100InstructionsMax;
  }

  // Returns true if it is an NonSemantic.Shader.DebugInfo.100 instruction.
  bool IsShader100DebugInstr() const {
    return GetShader100DebugOpcode() !=
           NonSemanticShaderDebugInfo100InstructionsMax;
  }
  bool IsCommonDebugInstr() const {
    return GetCommonDebugOpcode() != CommonDebugInfoInstructionsMax;
  }

  // Returns true if this instructions a non-semantic instruction.
  bool IsNonSemanticInstruction() const;

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

  // Returns true if the instruction generates a read-only pointer, with the
  // same caveats documented in the comment for IsReadOnlyPointer.  The first
  // version assumes the module is a shader module.  The second assumes a
  // kernel.
  bool IsReadOnlyPointerShaders() const;
  bool IsReadOnlyPointerKernel() const;

  // Returns true if the result of |inst| can be used as the base image for an
  // instruction that samples a image, reads an image, or writes to an image.
  bool IsValidBaseImage() const;

  IRContext* context_;  // IR Context
  spv::Op opcode_;      // Opcode
  bool has_type_id_;    // True if the instruction has a type id
  bool has_result_id_;  // True if the instruction has a result id
  uint32_t unique_id_;  // Unique instruction id
  // All logical operands, including result type id and result id.
  OperandList operands_;
  // Op[No]Line or Debug[No]Line instructions preceding this instruction. Note
  // that for Instructions representing Op[No]Line or Debug[No]Line themselves,
  // this field should be empty.
  std::vector<Instruction> dbg_line_insts_;

  // DebugScope that wraps this instruction.
  DebugScope dbg_scope_;

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

inline void Instruction::SetDebugScope(const DebugScope& scope) {
  dbg_scope_ = scope;
  for (auto& i : dbg_line_insts_) {
    i.dbg_scope_ = scope;
  }
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
  return opcode_ == spv::Op::OpNop && !has_type_id_ && !has_result_id_ &&
         operands_.empty();
}

inline void Instruction::ToNop() {
  opcode_ = spv::Op::OpNop;
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
  for (auto& operand : operands_)
    if (spvIsIdType(operand.type)) f(&operand.words[0]);
}

inline void Instruction::ForEachId(
    const std::function<void(const uint32_t*)>& f) const {
  for (const auto& operand : operands_)
    if (spvIsIdType(operand.type)) f(&operand.words[0]);
}

inline bool Instruction::WhileEachInId(
    const std::function<bool(uint32_t*)>& f) {
  for (auto& operand : operands_) {
    if (spvIsInIdType(operand.type) && !f(&operand.words[0])) {
      return false;
    }
  }
  return true;
}

inline bool Instruction::WhileEachInId(
    const std::function<bool(const uint32_t*)>& f) const {
  for (const auto& operand : operands_) {
    if (spvIsInIdType(operand.type) && !f(&operand.words[0])) {
      return false;
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
  for (auto& operand : operands_) {
    switch (operand.type) {
      case SPV_OPERAND_TYPE_RESULT_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
        break;
      default:
        if (!f(&operand.words[0])) return false;
        break;
    }
  }
  return true;
}

inline bool Instruction::WhileEachInOperand(
    const std::function<bool(const uint32_t*)>& f) const {
  for (const auto& operand : operands_) {
    switch (operand.type) {
      case SPV_OPERAND_TYPE_RESULT_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
        break;
      default:
        if (!f(&operand.words[0])) return false;
        break;
    }
  }
  return true;
}

inline void Instruction::ForEachInOperand(
    const std::function<void(uint32_t*)>& f) {
  WhileEachInOperand([&f](uint32_t* operand) {
    f(operand);
    return true;
  });
}

inline void Instruction::ForEachInOperand(
    const std::function<void(const uint32_t*)>& f) const {
  WhileEachInOperand([&f](const uint32_t* operand) {
    f(operand);
    return true;
  });
}

inline bool Instruction::HasLabels() const {
  switch (opcode_) {
    case spv::Op::OpSelectionMerge:
    case spv::Op::OpBranch:
    case spv::Op::OpLoopMerge:
    case spv::Op::OpBranchConditional:
    case spv::Op::OpSwitch:
    case spv::Op::OpPhi:
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
  return IsConstantInst(opcode()) && !IsSpecConstantInst(opcode());
}
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INSTRUCTION_H_
