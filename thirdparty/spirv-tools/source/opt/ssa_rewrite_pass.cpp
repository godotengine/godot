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

// This file implements the SSA rewriting algorithm proposed in
//
//      Simple and Efficient Construction of Static Single Assignment Form.
//      Braun M., Buchwald S., Hack S., Leißa R., Mallon C., Zwinkau A. (2013)
//      In: Jhala R., De Bosschere K. (eds)
//      Compiler Construction. CC 2013.
//      Lecture Notes in Computer Science, vol 7791.
//      Springer, Berlin, Heidelberg
//
//      https://link.springer.com/chapter/10.1007/978-3-642-37051-9_6
//
// In contrast to common eager algorithms based on dominance and dominance
// frontier information, this algorithm works backwards from load operations.
//
// When a target variable is loaded, it queries the variable's reaching
// definition.  If the reaching definition is unknown at the current location,
// it searches backwards in the CFG, inserting Phi instructions at join points
// in the CFG along the way until it finds the desired store instruction.
//
// The algorithm avoids repeated lookups using memoization.
//
// For reducible CFGs, which are a superset of the structured CFGs in SPIRV,
// this algorithm is proven to produce minimal SSA.  That is, it inserts the
// minimal number of Phi instructions required to ensure the SSA property, but
// some Phi instructions may be dead
// (https://en.wikipedia.org/wiki/Static_single_assignment_form).

#include "source/opt/ssa_rewrite_pass.h"

#include <memory>
#include <sstream>

#include "source/opcode.h"
#include "source/opt/cfg.h"
#include "source/opt/mem_pass.h"
#include "source/opt/types.h"

// Debug logging (0: Off, 1-N: Verbosity level).  Replace this with the
// implementation done for
// https://github.com/KhronosGroup/SPIRV-Tools/issues/1351
// #define SSA_REWRITE_DEBUGGING_LEVEL 3

#ifdef SSA_REWRITE_DEBUGGING_LEVEL
#include <ostream>
#else
#define SSA_REWRITE_DEBUGGING_LEVEL 0
#endif

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kStoreValIdInIdx = 1;
constexpr uint32_t kVariableInitIdInIdx = 1;
}  // namespace

std::string SSARewriter::PhiCandidate::PrettyPrint(const CFG* cfg) const {
  std::ostringstream str;
  str << "%" << result_id_ << " = Phi[%" << var_id_ << ", BB %" << bb_->id()
      << "](";
  if (phi_args_.size() > 0) {
    uint32_t arg_ix = 0;
    for (uint32_t pred_label : cfg->preds(bb_->id())) {
      uint32_t arg_id = phi_args_[arg_ix++];
      str << "[%" << arg_id << ", bb(%" << pred_label << ")] ";
    }
  }
  str << ")";
  if (copy_of_ != 0) {
    str << "  [COPY OF " << copy_of_ << "]";
  }
  str << ((is_complete_) ? "  [COMPLETE]" : "  [INCOMPLETE]");

  return str.str();
}

SSARewriter::PhiCandidate& SSARewriter::CreatePhiCandidate(uint32_t var_id,
                                                           BasicBlock* bb) {
  // TODO(1841): Handle id overflow.
  uint32_t phi_result_id = pass_->context()->TakeNextId();
  auto result = phi_candidates_.emplace(
      phi_result_id, PhiCandidate(var_id, phi_result_id, bb));
  PhiCandidate& phi_candidate = result.first->second;
  return phi_candidate;
}

void SSARewriter::ReplacePhiUsersWith(const PhiCandidate& phi_to_remove,
                                      uint32_t repl_id) {
  for (uint32_t user_id : phi_to_remove.users()) {
    PhiCandidate* user_phi = GetPhiCandidate(user_id);
    BasicBlock* bb = pass_->context()->get_instr_block(user_id);
    if (user_phi) {
      // If the user is a Phi candidate, replace all arguments that refer to
      // |phi_to_remove.result_id()| with |repl_id|.
      for (uint32_t& arg : user_phi->phi_args()) {
        if (arg == phi_to_remove.result_id()) {
          arg = repl_id;
        }
      }
    } else if (bb->id() == user_id) {
      // The phi candidate is the definition of the variable at basic block
      // |bb|.  We must change this to the replacement.
      WriteVariable(phi_to_remove.var_id(), bb, repl_id);
    } else {
      // For regular loads, traverse the |load_replacement_| table looking for
      // instances of |phi_to_remove|.
      for (auto& it : load_replacement_) {
        if (it.second == phi_to_remove.result_id()) {
          it.second = repl_id;
        }
      }
    }
  }
}

uint32_t SSARewriter::TryRemoveTrivialPhi(PhiCandidate* phi_candidate) {
  uint32_t same_id = 0;
  for (uint32_t arg_id : phi_candidate->phi_args()) {
    if (arg_id == same_id || arg_id == phi_candidate->result_id()) {
      // This is a self-reference operand or a reference to the same value ID.
      continue;
    }
    if (same_id != 0) {
      // This Phi candidate merges at least two values.  Therefore, it is not
      // trivial.
      assert(phi_candidate->copy_of() == 0 &&
             "Phi candidate transitioning from copy to non-copy.");
      return phi_candidate->result_id();
    }
    same_id = arg_id;
  }

  // The previous logic has determined that this Phi candidate |phi_candidate|
  // is trivial.  It is essentially the copy operation phi_candidate->phi_result
  // = Phi(same, same, same, ...).  Since it is not necessary, we can re-route
  // all the users of |phi_candidate->phi_result| to all its users, and remove
  // |phi_candidate|.

  // Mark the Phi candidate as a trivial copy of |same_id|, so it won't be
  // generated.
  phi_candidate->MarkCopyOf(same_id);

  assert(same_id != 0 && "Completed Phis cannot have %0 in their arguments");

  // Since |phi_candidate| always produces |same_id|, replace all the users of
  // |phi_candidate| with |same_id|.
  ReplacePhiUsersWith(*phi_candidate, same_id);

  return same_id;
}

uint32_t SSARewriter::AddPhiOperands(PhiCandidate* phi_candidate) {
  assert(phi_candidate->phi_args().size() == 0 &&
         "Phi candidate already has arguments");

  bool found_0_arg = false;
  for (uint32_t pred : pass_->cfg()->preds(phi_candidate->bb()->id())) {
    BasicBlock* pred_bb = pass_->cfg()->block(pred);

    // If |pred_bb| is not sealed, use %0 to indicate that
    // |phi_candidate| needs to be completed after the whole CFG has
    // been processed.
    //
    // Note that we cannot call GetReachingDef() in these cases
    // because this would generate an empty Phi candidate in
    // |pred_bb|.  When |pred_bb| is later processed, a new definition
    // for |phi_candidate->var_id_| will be lost because
    // |phi_candidate| will still be reached by the empty Phi.
    //
    // Consider:
    //
    //       BB %23:
    //           %38 = Phi[%i](%int_0[%1], %39[%25])
    //
    //           ...
    //
    //       BB %25: [Starts unsealed]
    //       %39 = Phi[%i]()
    //       %34 = ...
    //       OpStore %i %34    -> Currdef(%i) at %25 is %34
    //       OpBranch %23
    //
    // When we first create the Phi in %38, we add an operandless Phi in
    // %39 to hold the unknown reaching def for %i.
    //
    // But then, when we go to complete %39 at the end.  The reaching def
    // for %i in %25's predecessor is %38 itself.  So we miss the fact
    // that %25 has a def for %i that should be used.
    //
    // By making the argument %0, we make |phi_candidate| incomplete,
    // which will cause it to be completed after the whole CFG has
    // been scanned.
    uint32_t arg_id = IsBlockSealed(pred_bb)
                          ? GetReachingDef(phi_candidate->var_id(), pred_bb)
                          : 0;
    phi_candidate->phi_args().push_back(arg_id);

    if (arg_id == 0) {
      found_0_arg = true;
    } else {
      // If this argument is another Phi candidate, add |phi_candidate| to the
      // list of users for the defining Phi.
      PhiCandidate* defining_phi = GetPhiCandidate(arg_id);
      if (defining_phi && defining_phi != phi_candidate) {
        defining_phi->AddUser(phi_candidate->result_id());
      }
    }
  }

  // If we could not fill-in all the arguments of this Phi, mark it incomplete
  // so it gets completed after the whole CFG has been processed.
  if (found_0_arg) {
    phi_candidate->MarkIncomplete();
    incomplete_phis_.push(phi_candidate);
    return phi_candidate->result_id();
  }

  // Try to remove |phi_candidate|, if it's trivial.
  uint32_t repl_id = TryRemoveTrivialPhi(phi_candidate);
  if (repl_id == phi_candidate->result_id()) {
    // |phi_candidate| is complete and not trivial.  Add it to the
    // list of Phi candidates to generate.
    phi_candidate->MarkComplete();
    phis_to_generate_.push_back(phi_candidate);
  }

  return repl_id;
}

uint32_t SSARewriter::GetValueAtBlock(uint32_t var_id, BasicBlock* bb) {
  assert(bb != nullptr);
  const auto& bb_it = defs_at_block_.find(bb);
  if (bb_it != defs_at_block_.end()) {
    const auto& current_defs = bb_it->second;
    const auto& var_it = current_defs.find(var_id);
    if (var_it != current_defs.end()) {
      return var_it->second;
    }
  }
  return 0;
}

uint32_t SSARewriter::GetReachingDef(uint32_t var_id, BasicBlock* bb) {
  // If |var_id| has a definition in |bb|, return it.
  uint32_t val_id = GetValueAtBlock(var_id, bb);
  if (val_id != 0) return val_id;

  // Otherwise, look up the value for |var_id| in |bb|'s predecessors.
  auto& predecessors = pass_->cfg()->preds(bb->id());
  if (predecessors.size() == 1) {
    // If |bb| has exactly one predecessor, we look for |var_id|'s definition
    // there.
    val_id = GetReachingDef(var_id, pass_->cfg()->block(predecessors[0]));
  } else if (predecessors.size() > 1) {
    // If there is more than one predecessor, this is a join block which may
    // require a Phi instruction.  This will act as |var_id|'s current
    // definition to break potential cycles.
    PhiCandidate& phi_candidate = CreatePhiCandidate(var_id, bb);

    // Set the value for |bb| to avoid an infinite recursion.
    WriteVariable(var_id, bb, phi_candidate.result_id());
    val_id = AddPhiOperands(&phi_candidate);
  }

  // If we could not find a store for this variable in the path from the root
  // of the CFG, the variable is not defined, so we use undef.
  if (val_id == 0) {
    val_id = pass_->GetUndefVal(var_id);
    if (val_id == 0) {
      return 0;
    }
  }

  WriteVariable(var_id, bb, val_id);

  return val_id;
}

void SSARewriter::SealBlock(BasicBlock* bb) {
  auto result = sealed_blocks_.insert(bb);
  (void)result;
  assert(result.second == true &&
         "Tried to seal the same basic block more than once.");
}

void SSARewriter::ProcessStore(Instruction* inst, BasicBlock* bb) {
  auto opcode = inst->opcode();
  assert((opcode == spv::Op::OpStore || opcode == spv::Op::OpVariable) &&
         "Expecting a store or a variable definition instruction.");

  uint32_t var_id = 0;
  uint32_t val_id = 0;
  if (opcode == spv::Op::OpStore) {
    (void)pass_->GetPtr(inst, &var_id);
    val_id = inst->GetSingleWordInOperand(kStoreValIdInIdx);
  } else if (inst->NumInOperands() >= 2) {
    var_id = inst->result_id();
    val_id = inst->GetSingleWordInOperand(kVariableInitIdInIdx);
  }
  if (pass_->IsTargetVar(var_id)) {
    WriteVariable(var_id, bb, val_id);
    pass_->context()->get_debug_info_mgr()->AddDebugValueForVariable(
        inst, var_id, val_id, inst);

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
    std::cerr << "\tFound store '%" << var_id << " = %" << val_id << "': "
              << inst->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
              << "\n";
#endif
  }
}

bool SSARewriter::ProcessLoad(Instruction* inst, BasicBlock* bb) {
  // Get the pointer that we are using to load from.
  uint32_t var_id = 0;
  (void)pass_->GetPtr(inst, &var_id);

  // Get the immediate reaching definition for |var_id|.
  //
  // In the presence of variable pointers, the reaching definition may be
  // another pointer.  For example, the following fragment:
  //
  //  %2 = OpVariable %_ptr_Input_float Input
  // %11 = OpVariable %_ptr_Function__ptr_Input_float Function
  //       OpStore %11 %2
  // %12 = OpLoad %_ptr_Input_float %11
  // %13 = OpLoad %float %12
  //
  // corresponds to the pseudo-code:
  //
  // layout(location = 0) in flat float *%2
  // float %13;
  // float *%12;
  // float **%11;
  // *%11 = %2;
  // %12 = *%11;
  // %13 = *%12;
  //
  // which ultimately, should correspond to:
  //
  // %13 = *%2;
  //
  // During rewriting, the pointer %12 is found to be replaceable by %2 (i.e.,
  // load_replacement_[12] is 2). However, when processing the load
  // %13 = *%12, the type of %12's reaching definition is another float
  // pointer (%2), instead of a float value.
  //
  // When this happens, we need to continue looking up the reaching definition
  // chain until we get to a float value or a non-target var (i.e. a variable
  // that cannot be SSA replaced, like %2 in this case since it is a function
  // argument).
  analysis::DefUseManager* def_use_mgr = pass_->context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = pass_->context()->get_type_mgr();
  analysis::Type* load_type = type_mgr->GetType(inst->type_id());
  uint32_t val_id = 0;
  bool found_reaching_def = false;
  while (!found_reaching_def) {
    if (!pass_->IsTargetVar(var_id)) {
      // If the variable we are loading from is not an SSA target (globals,
      // function parameters), do nothing.
      return true;
    }

    val_id = GetReachingDef(var_id, bb);
    if (val_id == 0) {
      return false;
    }

    // If the reaching definition is a pointer type different than the type of
    // the instruction we are analyzing, then it must be a reference to another
    // pointer (otherwise, this would be invalid SPIRV).  We continue
    // de-referencing it by making |val_id| be |var_id|.
    //
    // NOTE: if there is no reaching definition instruction, it means |val_id|
    // is an undef.
    Instruction* reaching_def_inst = def_use_mgr->GetDef(val_id);
    if (reaching_def_inst &&
        !type_mgr->GetType(reaching_def_inst->type_id())->IsSame(load_type)) {
      var_id = val_id;
    } else {
      found_reaching_def = true;
    }
  }

  // Schedule a replacement for the result of this load instruction with
  // |val_id|. After all the rewriting decisions are made, every use of
  // this load will be replaced with |val_id|.
  uint32_t load_id = inst->result_id();
  assert(load_replacement_.count(load_id) == 0);
  load_replacement_[load_id] = val_id;
  PhiCandidate* defining_phi = GetPhiCandidate(val_id);
  if (defining_phi) {
    defining_phi->AddUser(load_id);
  }

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cerr << "\tFound load: "
            << inst->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
            << " (replacement for %" << load_id << " is %" << val_id << ")\n";
#endif

  return true;
}

void SSARewriter::PrintPhiCandidates() const {
  std::cerr << "\nPhi candidates:\n";
  for (const auto& phi_it : phi_candidates_) {
    std::cerr << "\tBB %" << phi_it.second.bb()->id() << ": "
              << phi_it.second.PrettyPrint(pass_->cfg()) << "\n";
  }
  std::cerr << "\n";
}

void SSARewriter::PrintReplacementTable() const {
  std::cerr << "\nLoad replacement table\n";
  for (const auto& it : load_replacement_) {
    std::cerr << "\t%" << it.first << " -> %" << it.second << "\n";
  }
  std::cerr << "\n";
}

bool SSARewriter::GenerateSSAReplacements(BasicBlock* bb) {
#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cerr << "Generating SSA replacements for block: " << bb->id() << "\n";
  std::cerr << bb->PrettyPrint(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
            << "\n";
#endif

  for (auto& inst : *bb) {
    auto opcode = inst.opcode();
    if (opcode == spv::Op::OpStore || opcode == spv::Op::OpVariable) {
      ProcessStore(&inst, bb);
    } else if (inst.opcode() == spv::Op::OpLoad) {
      if (!ProcessLoad(&inst, bb)) {
        return false;
      }
    }
  }

  // Seal |bb|. This means that all the stores in it have been scanned and
  // it's ready to feed them into its successors.
  SealBlock(bb);

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  PrintPhiCandidates();
  PrintReplacementTable();
  std::cerr << "\n\n";
#endif
  return true;
}

uint32_t SSARewriter::GetReplacement(std::pair<uint32_t, uint32_t> repl) {
  uint32_t val_id = repl.second;
  auto it = load_replacement_.find(val_id);
  while (it != load_replacement_.end()) {
    val_id = it->second;
    it = load_replacement_.find(val_id);
  }
  return val_id;
}

uint32_t SSARewriter::GetPhiArgument(const PhiCandidate* phi_candidate,
                                     uint32_t ix) {
  assert(phi_candidate->IsReady() &&
         "Tried to get the final argument from an incomplete/trivial Phi");

  uint32_t arg_id = phi_candidate->phi_args()[ix];
  while (arg_id != 0) {
    PhiCandidate* phi_user = GetPhiCandidate(arg_id);
    if (phi_user == nullptr || phi_user->IsReady()) {
      // If the argument is not a Phi or it's a Phi candidate ready to be
      // emitted, return it.
      return arg_id;
    }
    arg_id = phi_user->copy_of();
  }

  assert(false &&
         "No Phi candidates in the copy-of chain are ready to be generated");

  return 0;
}

bool SSARewriter::ApplyReplacements() {
  bool modified = false;

#if SSA_REWRITE_DEBUGGING_LEVEL > 2
  std::cerr << "\n\nApplying replacement decisions to IR\n\n";
  PrintPhiCandidates();
  PrintReplacementTable();
  std::cerr << "\n\n";
#endif

  // Add Phi instructions from completed Phi candidates.
  std::vector<Instruction*> generated_phis;
  for (const PhiCandidate* phi_candidate : phis_to_generate_) {
#if SSA_REWRITE_DEBUGGING_LEVEL > 2
    std::cerr << "Phi candidate: " << phi_candidate->PrettyPrint(pass_->cfg())
              << "\n";
#endif

    assert(phi_candidate->is_complete() &&
           "Tried to instantiate a Phi instruction from an incomplete Phi "
           "candidate");

    auto* local_var = pass_->get_def_use_mgr()->GetDef(phi_candidate->var_id());

    // Build the vector of operands for the new OpPhi instruction.
    uint32_t type_id = pass_->GetPointeeTypeId(local_var);
    std::vector<Operand> phi_operands;
    uint32_t arg_ix = 0;
    std::unordered_map<uint32_t, uint32_t> already_seen;
    for (uint32_t pred_label : pass_->cfg()->preds(phi_candidate->bb()->id())) {
      uint32_t op_val_id = GetPhiArgument(phi_candidate, arg_ix++);
      if (already_seen.count(pred_label) == 0) {
        phi_operands.push_back(
            {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {op_val_id}});
        phi_operands.push_back(
            {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {pred_label}});
        already_seen[pred_label] = op_val_id;
      } else {
        // It is possible that there are two edges from the same parent block.
        // Since the OpPhi can have only one entry for each parent, we have to
        // make sure the two edges are consistent with each other.
        assert(already_seen[pred_label] == op_val_id &&
               "Inconsistent value for duplicate edges.");
      }
    }

    // Generate a new OpPhi instruction and insert it in its basic
    // block.
    std::unique_ptr<Instruction> phi_inst(
        new Instruction(pass_->context(), spv::Op::OpPhi, type_id,
                        phi_candidate->result_id(), phi_operands));
    generated_phis.push_back(phi_inst.get());
    pass_->get_def_use_mgr()->AnalyzeInstDef(&*phi_inst);
    pass_->context()->set_instr_block(&*phi_inst, phi_candidate->bb());
    auto insert_it = phi_candidate->bb()->begin();
    insert_it = insert_it.InsertBefore(std::move(phi_inst));
    pass_->context()->get_decoration_mgr()->CloneDecorations(
        phi_candidate->var_id(), phi_candidate->result_id(),
        {spv::Decoration::RelaxedPrecision});

    // Add DebugValue for the new OpPhi instruction.
    insert_it->SetDebugScope(local_var->GetDebugScope());
    pass_->context()->get_debug_info_mgr()->AddDebugValueForVariable(
        &*insert_it, phi_candidate->var_id(), phi_candidate->result_id(),
        &*insert_it);

    modified = true;
  }

  // Scan uses for all inserted Phi instructions. Do this separately from the
  // registration of the Phi instruction itself to avoid trying to analyze
  // uses of Phi instructions that have not been registered yet.
  for (Instruction* phi_inst : generated_phis) {
    pass_->get_def_use_mgr()->AnalyzeInstUse(&*phi_inst);
  }

#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cerr << "\n\nReplacing the result of load instructions with the "
               "corresponding SSA id\n\n";
#endif

  // Apply replacements from the load replacement table.
  for (auto& repl : load_replacement_) {
    uint32_t load_id = repl.first;
    uint32_t val_id = GetReplacement(repl);
    Instruction* load_inst =
        pass_->context()->get_def_use_mgr()->GetDef(load_id);

#if SSA_REWRITE_DEBUGGING_LEVEL > 2
    std::cerr << "\t"
              << load_inst->PrettyPrint(
                     SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)
              << "  (%" << load_id << " -> %" << val_id << ")\n";
#endif

    // Remove the load instruction and replace all the uses of this load's
    // result with |val_id|.  Kill any names or decorates using the load's
    // result before replacing to prevent incorrect replacement in those
    // instructions.
    pass_->context()->KillNamesAndDecorates(load_id);
    pass_->context()->ReplaceAllUsesWith(load_id, val_id);
    pass_->context()->KillInst(load_inst);
    modified = true;
  }

  return modified;
}

void SSARewriter::FinalizePhiCandidate(PhiCandidate* phi_candidate) {
  assert(phi_candidate->phi_args().size() > 0 &&
         "Phi candidate should have arguments");

  uint32_t ix = 0;
  for (uint32_t pred : pass_->cfg()->preds(phi_candidate->bb()->id())) {
    BasicBlock* pred_bb = pass_->cfg()->block(pred);
    uint32_t& arg_id = phi_candidate->phi_args()[ix++];
    if (arg_id == 0) {
      // If |pred_bb| is still not sealed, it means it's unreachable. In this
      // case, we just use Undef as an argument.
      arg_id = IsBlockSealed(pred_bb)
                   ? GetReachingDef(phi_candidate->var_id(), pred_bb)
                   : pass_->GetUndefVal(phi_candidate->var_id());
    }
  }

  // This candidate is now completed.
  phi_candidate->MarkComplete();

  // If |phi_candidate| is not trivial, add it to the list of Phis to
  // generate.
  if (TryRemoveTrivialPhi(phi_candidate) == phi_candidate->result_id()) {
    // If we could not remove |phi_candidate|, it means that it is complete
    // and not trivial. Add it to the list of Phis to generate.
    assert(!phi_candidate->copy_of() && "A completed Phi cannot be trivial.");
    phis_to_generate_.push_back(phi_candidate);
  }
}

void SSARewriter::FinalizePhiCandidates() {
#if SSA_REWRITE_DEBUGGING_LEVEL > 1
  std::cerr << "Finalizing Phi candidates:\n\n";
  PrintPhiCandidates();
  std::cerr << "\n";
#endif

  // Now, complete the collected candidates.
  while (incomplete_phis_.size() > 0) {
    PhiCandidate* phi_candidate = incomplete_phis_.front();
    incomplete_phis_.pop();
    FinalizePhiCandidate(phi_candidate);
  }
}

Pass::Status SSARewriter::RewriteFunctionIntoSSA(Function* fp) {
#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cerr << "Function before SSA rewrite:\n"
            << fp->PrettyPrint(0) << "\n\n\n";
#endif

  // Collect variables that can be converted into SSA IDs.
  pass_->CollectTargetVars(fp);

  // Generate all the SSA replacements and Phi candidates. This will
  // generate incomplete and trivial Phis.
  bool succeeded = pass_->cfg()->WhileEachBlockInReversePostOrder(
      fp->entry().get(), [this](BasicBlock* bb) {
        if (!GenerateSSAReplacements(bb)) {
          return false;
        }
        return true;
      });

  if (!succeeded) {
    return Pass::Status::Failure;
  }

  // Remove trivial Phis and add arguments to incomplete Phis.
  FinalizePhiCandidates();

  // Finally, apply all the replacements in the IR.
  bool modified = ApplyReplacements();

#if SSA_REWRITE_DEBUGGING_LEVEL > 0
  std::cerr << "\n\n\nFunction after SSA rewrite:\n"
            << fp->PrettyPrint(0) << "\n";
#endif

  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

Pass::Status SSARewritePass::Process() {
  Status status = Status::SuccessWithoutChange;
  for (auto& fn : *get_module()) {
    if (fn.IsDeclaration()) {
      continue;
    }
    status =
        CombineStatus(status, SSARewriter(this).RewriteFunctionIntoSSA(&fn));
    // Kill DebugDeclares for target variables.
    for (auto var_id : seen_target_vars_) {
      context()->get_debug_info_mgr()->KillDebugDeclares(var_id);
    }
    if (status == Status::Failure) {
      break;
    }
  }
  return status;
}

}  // namespace opt
}  // namespace spvtools
