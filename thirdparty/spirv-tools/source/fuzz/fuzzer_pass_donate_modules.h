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

#ifndef SOURCE_FUZZ_FUZZER_PASS_DONATE_MODULES_H_
#define SOURCE_FUZZ_FUZZER_PASS_DONATE_MODULES_H_

#include <vector>

#include "source/fuzz/fuzzer_pass.h"
#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass that randomly adds code from other SPIR-V modules to the module
// being transformed.
class FuzzerPassDonateModules : public FuzzerPass {
 public:
  FuzzerPassDonateModules(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations,
      std::vector<fuzzerutil::ModuleSupplier> donor_suppliers);

  void Apply() override;

  // Donates the global declarations and functions of |donor_ir_context| into
  // the fuzzer pass's IR context.  |make_livesafe| dictates whether the
  // functions of the donated module will be made livesafe (see
  // FactFunctionIsLivesafe).
  void DonateSingleModule(opt::IRContext* donor_ir_context, bool make_livesafe);

 private:
  // Adapts a storage class coming from a donor module so that it will work
  // in a recipient module, e.g. by changing Uniform to Private.
  static spv::StorageClass AdaptStorageClass(
      spv::StorageClass donor_storage_class);

  // Identifies all external instruction set imports in |donor_ir_context| and
  // populates |original_id_to_donated_id| with a mapping from the donor's id
  // for such an import to a corresponding import in the recipient.  Aborts if
  // no such corresponding import is available.
  void HandleExternalInstructionImports(
      opt::IRContext* donor_ir_context,
      std::map<uint32_t, uint32_t>* original_id_to_donated_id);

  // Considers all types, globals, constants and undefs in |donor_ir_context|.
  // For each instruction, uses |original_to_donated_id| to map its result id to
  // either (1) the id of an existing identical instruction in the recipient, or
  // (2) to a fresh id, in which case the instruction is also added to the
  // recipient (with any operand ids that it uses being remapped via
  // |original_id_to_donated_id|).
  void HandleTypesAndValues(
      opt::IRContext* donor_ir_context,
      std::map<uint32_t, uint32_t>* original_id_to_donated_id);

  // Helper method for HandleTypesAndValues, to handle a single type/value.
  void HandleTypeOrValue(
      const opt::Instruction& type_or_value,
      std::map<uint32_t, uint32_t>* original_id_to_donated_id);

  // Assumes that |donor_ir_context| does not exhibit recursion.  Considers the
  // functions in |donor_ir_context|'s call graph in a reverse-topologically-
  // sorted order (leaves-to-root), adding each function to the recipient
  // module, rewritten to use fresh ids and using |original_id_to_donated_id| to
  // remap ids.  The |make_livesafe| argument captures whether the functions in
  // the module are required to be made livesafe before being added to the
  // recipient.
  void HandleFunctions(opt::IRContext* donor_ir_context,
                       std::map<uint32_t, uint32_t>* original_id_to_donated_id,
                       bool make_livesafe);

  // During donation we will have to ignore some instructions, e.g. because they
  // use opcodes that we cannot support or because they reference the ids of
  // instructions that have not been donated.  This function encapsulates the
  // logic for deciding which whether instruction |instruction| from
  // |donor_ir_context| can be donated.
  bool CanDonateInstruction(
      opt::IRContext* donor_ir_context, const opt::Instruction& instruction,
      const std::map<uint32_t, uint32_t>& original_id_to_donated_id,
      const std::set<uint32_t>& skipped_instructions) const;

  // We treat the OpArrayLength instruction specially.  In the donor shader this
  // instruction yields the length of a runtime array that is the final member
  // of a struct.  During donation, we will have converted the runtime array
  // type, and the associated struct field, into a fixed-size array.
  //
  // Instead of donating this instruction, we turn it into an OpCopyObject
  // instruction that copies the size of the fixed-size array.
  void HandleOpArrayLength(
      const opt::Instruction& instruction,
      std::map<uint32_t, uint32_t>* original_id_to_donated_id,
      std::vector<protobufs::Instruction>* donated_instructions) const;

  // The instruction |instruction| is required to be an instruction that cannot
  // be easily donated, either because it uses an unsupported opcode, has an
  // unsupported result type, or uses id operands that could not be donated.
  //
  // If |instruction| generates a result id, the function attempts to add a
  // substitute for |instruction| to |donated_instructions| that has the correct
  // result type.  If this cannot be done, the instruction's result id is added
  // to |skipped_instructions|.  The mapping from donor ids to recipient ids is
  // managed by |original_id_to_donated_id|.
  void HandleDifficultInstruction(
      const opt::Instruction& instruction,
      std::map<uint32_t, uint32_t>* original_id_to_donated_id,
      std::vector<protobufs::Instruction>* donated_instructions,
      std::set<uint32_t>* skipped_instructions);

  // Adds an instruction based in |instruction| to |donated_instructions| in a
  // form ready for donation.  The original instruction comes from
  // |donor_ir_context|, and |original_id_to_donated_id| maps ids from
  // |donor_ir_context| to corresponding ids in the recipient module.
  void PrepareInstructionForDonation(
      const opt::Instruction& instruction, opt::IRContext* donor_ir_context,
      std::map<uint32_t, uint32_t>* original_id_to_donated_id,
      std::vector<protobufs::Instruction>* donated_instructions);

  // Tries to create a protobufs::LoopLimiterInfo given a loop header basic
  // block. Returns true if successful and outputs loop limiter into the |out|
  // variable. Otherwise, returns false. |out| contains an undefined value when
  // this function returns false.
  bool CreateLoopLimiterInfo(
      opt::IRContext* donor_ir_context, const opt::BasicBlock& loop_header,
      const std::map<uint32_t, uint32_t>& original_id_to_donated_id,
      protobufs::LoopLimiterInfo* out);

  // Requires that |donated_instructions| represents a prepared version of the
  // instructions of |function_to_donate| (which comes from |donor_ir_context|)
  // ready for donation, and |original_id_to_donated_id| maps ids from
  // |donor_ir_context| to their corresponding ids in the recipient module.
  //
  // Attempts to add a livesafe version of the function, based on
  // |donated_instructions|, to the recipient module. Returns true if the
  // donation was successful, false otherwise.
  bool MaybeAddLivesafeFunction(
      const opt::Function& function_to_donate, opt::IRContext* donor_ir_context,
      const std::map<uint32_t, uint32_t>& original_id_to_donated_id,
      const std::vector<protobufs::Instruction>& donated_instructions);

  // Returns true if and only if |instruction| is a scalar, vector, matrix,
  // array or struct; i.e. it is not an opaque type.
  bool IsBasicType(const opt::Instruction& instruction) const;

  // Functions that supply SPIR-V modules
  std::vector<fuzzerutil::ModuleSupplier> donor_suppliers_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_DONATE_MODULES_H_
