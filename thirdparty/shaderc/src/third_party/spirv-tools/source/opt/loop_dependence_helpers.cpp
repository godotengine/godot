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

#include "source/opt/loop_dependence.h"

#include <ostream>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/instruction.h"
#include "source/opt/scalar_analysis.h"
#include "source/opt/scalar_analysis_nodes.h"

namespace spvtools {
namespace opt {

bool LoopDependenceAnalysis::IsZIV(
    const std::pair<SENode*, SENode*>& subscript_pair) {
  return CountInductionVariables(subscript_pair.first, subscript_pair.second) ==
         0;
}

bool LoopDependenceAnalysis::IsSIV(
    const std::pair<SENode*, SENode*>& subscript_pair) {
  return CountInductionVariables(subscript_pair.first, subscript_pair.second) ==
         1;
}

bool LoopDependenceAnalysis::IsMIV(
    const std::pair<SENode*, SENode*>& subscript_pair) {
  return CountInductionVariables(subscript_pair.first, subscript_pair.second) >
         1;
}

SENode* LoopDependenceAnalysis::GetLowerBound(const Loop* loop) {
  Instruction* cond_inst = loop->GetConditionInst();
  if (!cond_inst) {
    return nullptr;
  }
  Instruction* lower_inst = GetOperandDefinition(cond_inst, 0);
  switch (cond_inst->opcode()) {
    case SpvOpULessThan:
    case SpvOpSLessThan:
    case SpvOpULessThanEqual:
    case SpvOpSLessThanEqual:
    case SpvOpUGreaterThan:
    case SpvOpSGreaterThan:
    case SpvOpUGreaterThanEqual:
    case SpvOpSGreaterThanEqual: {
      // If we have a phi we are looking at the induction variable. We look
      // through the phi to the initial value of the phi upon entering the loop.
      if (lower_inst->opcode() == SpvOpPhi) {
        lower_inst = GetOperandDefinition(lower_inst, 0);
        // We don't handle looking through multiple phis.
        if (lower_inst->opcode() == SpvOpPhi) {
          return nullptr;
        }
      }
      return scalar_evolution_.SimplifyExpression(
          scalar_evolution_.AnalyzeInstruction(lower_inst));
    }
    default:
      return nullptr;
  }
}

SENode* LoopDependenceAnalysis::GetUpperBound(const Loop* loop) {
  Instruction* cond_inst = loop->GetConditionInst();
  if (!cond_inst) {
    return nullptr;
  }
  Instruction* upper_inst = GetOperandDefinition(cond_inst, 1);
  switch (cond_inst->opcode()) {
    case SpvOpULessThan:
    case SpvOpSLessThan: {
      // When we have a < condition we must subtract 1 from the analyzed upper
      // instruction.
      SENode* upper_bound = scalar_evolution_.SimplifyExpression(
          scalar_evolution_.CreateSubtraction(
              scalar_evolution_.AnalyzeInstruction(upper_inst),
              scalar_evolution_.CreateConstant(1)));
      return upper_bound;
    }
    case SpvOpUGreaterThan:
    case SpvOpSGreaterThan: {
      // When we have a > condition we must add 1 to the analyzed upper
      // instruction.
      SENode* upper_bound =
          scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
              scalar_evolution_.AnalyzeInstruction(upper_inst),
              scalar_evolution_.CreateConstant(1)));
      return upper_bound;
    }
    case SpvOpULessThanEqual:
    case SpvOpSLessThanEqual:
    case SpvOpUGreaterThanEqual:
    case SpvOpSGreaterThanEqual: {
      // We don't need to modify the results of analyzing when we have <= or >=.
      SENode* upper_bound = scalar_evolution_.SimplifyExpression(
          scalar_evolution_.AnalyzeInstruction(upper_inst));
      return upper_bound;
    }
    default:
      return nullptr;
  }
}

bool LoopDependenceAnalysis::IsWithinBounds(int64_t value, int64_t bound_one,
                                            int64_t bound_two) {
  if (bound_one < bound_two) {
    // If |bound_one| is the lower bound.
    return (value >= bound_one && value <= bound_two);
  } else if (bound_one > bound_two) {
    // If |bound_two| is the lower bound.
    return (value >= bound_two && value <= bound_one);
  } else {
    // Both bounds have the same value.
    return value == bound_one;
  }
}

bool LoopDependenceAnalysis::IsProvablyOutsideOfLoopBounds(
    const Loop* loop, SENode* distance, SENode* coefficient) {
  // We test to see if we can reduce the coefficient to an integral constant.
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (!coefficient_constant) {
    PrintDebug(
        "IsProvablyOutsideOfLoopBounds could not reduce coefficient to a "
        "SEConstantNode so must exit.");
    return false;
  }

  SENode* lower_bound = GetLowerBound(loop);
  SENode* upper_bound = GetUpperBound(loop);
  if (!lower_bound || !upper_bound) {
    PrintDebug(
        "IsProvablyOutsideOfLoopBounds could not get both the lower and upper "
        "bounds so must exit.");
    return false;
  }
  // If the coefficient is positive we calculate bounds as upper - lower
  // If the coefficient is negative we calculate bounds as lower - upper
  SENode* bounds = nullptr;
  if (coefficient_constant->FoldToSingleValue() >= 0) {
    PrintDebug(
        "IsProvablyOutsideOfLoopBounds found coefficient >= 0.\n"
        "Using bounds as upper - lower.");
    bounds = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.CreateSubtraction(upper_bound, lower_bound));
  } else {
    PrintDebug(
        "IsProvablyOutsideOfLoopBounds found coefficient < 0.\n"
        "Using bounds as lower - upper.");
    bounds = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.CreateSubtraction(lower_bound, upper_bound));
  }

  // We can attempt to deal with symbolic cases by subtracting |distance| and
  // the bound nodes. If we can subtract, simplify and produce a SEConstantNode
  // we can produce some information.
  SEConstantNode* distance_minus_bounds =
      scalar_evolution_
          .SimplifyExpression(
              scalar_evolution_.CreateSubtraction(distance, bounds))
          ->AsSEConstantNode();
  if (distance_minus_bounds) {
    PrintDebug(
        "IsProvablyOutsideOfLoopBounds found distance - bounds as a "
        "SEConstantNode with value " +
        ToString(distance_minus_bounds->FoldToSingleValue()));
    // If distance - bounds > 0 we prove the distance is outwith the loop
    // bounds.
    if (distance_minus_bounds->FoldToSingleValue() > 0) {
      PrintDebug(
          "IsProvablyOutsideOfLoopBounds found distance escaped the loop "
          "bounds.");
      return true;
    }
  }

  return false;
}

const Loop* LoopDependenceAnalysis::GetLoopForSubscriptPair(
    const std::pair<SENode*, SENode*>& subscript_pair) {
  // Collect all the SERecurrentNodes.
  std::vector<SERecurrentNode*> source_nodes =
      std::get<0>(subscript_pair)->CollectRecurrentNodes();
  std::vector<SERecurrentNode*> destination_nodes =
      std::get<1>(subscript_pair)->CollectRecurrentNodes();

  // Collect all the loops stored by the SERecurrentNodes.
  std::unordered_set<const Loop*> loops{};
  for (auto source_nodes_it = source_nodes.begin();
       source_nodes_it != source_nodes.end(); ++source_nodes_it) {
    loops.insert((*source_nodes_it)->GetLoop());
  }
  for (auto destination_nodes_it = destination_nodes.begin();
       destination_nodes_it != destination_nodes.end();
       ++destination_nodes_it) {
    loops.insert((*destination_nodes_it)->GetLoop());
  }

  // If we didn't find 1 loop |subscript_pair| is a subscript over multiple or 0
  // loops. We don't handle this so return nullptr.
  if (loops.size() != 1) {
    PrintDebug("GetLoopForSubscriptPair found loops.size() != 1.");
    return nullptr;
  }
  return *loops.begin();
}

DistanceEntry* LoopDependenceAnalysis::GetDistanceEntryForLoop(
    const Loop* loop, DistanceVector* distance_vector) {
  if (!loop) {
    return nullptr;
  }

  DistanceEntry* distance_entry = nullptr;
  for (size_t loop_index = 0; loop_index < loops_.size(); ++loop_index) {
    if (loop == loops_[loop_index]) {
      distance_entry = &(distance_vector->GetEntries()[loop_index]);
      break;
    }
  }

  return distance_entry;
}

DistanceEntry* LoopDependenceAnalysis::GetDistanceEntryForSubscriptPair(
    const std::pair<SENode*, SENode*>& subscript_pair,
    DistanceVector* distance_vector) {
  const Loop* loop = GetLoopForSubscriptPair(subscript_pair);

  return GetDistanceEntryForLoop(loop, distance_vector);
}

SENode* LoopDependenceAnalysis::GetTripCount(const Loop* loop) {
  BasicBlock* condition_block = loop->FindConditionBlock();
  if (!condition_block) {
    return nullptr;
  }
  Instruction* induction_instr = loop->FindConditionVariable(condition_block);
  if (!induction_instr) {
    return nullptr;
  }
  Instruction* cond_instr = loop->GetConditionInst();
  if (!cond_instr) {
    return nullptr;
  }

  size_t iteration_count = 0;

  // We have to check the instruction type here. If the condition instruction
  // isn't a supported type we can't calculate the trip count.
  if (loop->IsSupportedCondition(cond_instr->opcode())) {
    if (loop->FindNumberOfIterations(induction_instr, &*condition_block->tail(),
                                     &iteration_count)) {
      return scalar_evolution_.CreateConstant(
          static_cast<int64_t>(iteration_count));
    }
  }

  return nullptr;
}

SENode* LoopDependenceAnalysis::GetFirstTripInductionNode(const Loop* loop) {
  BasicBlock* condition_block = loop->FindConditionBlock();
  if (!condition_block) {
    return nullptr;
  }
  Instruction* induction_instr = loop->FindConditionVariable(condition_block);
  if (!induction_instr) {
    return nullptr;
  }
  int64_t induction_initial_value = 0;
  if (!loop->GetInductionInitValue(induction_instr, &induction_initial_value)) {
    return nullptr;
  }

  SENode* induction_init_SENode = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateConstant(induction_initial_value));
  return induction_init_SENode;
}

SENode* LoopDependenceAnalysis::GetFinalTripInductionNode(
    const Loop* loop, SENode* induction_coefficient) {
  SENode* first_trip_induction_node = GetFirstTripInductionNode(loop);
  if (!first_trip_induction_node) {
    return nullptr;
  }
  // Get trip_count as GetTripCount - 1
  // This is because the induction variable is not stepped on the first
  // iteration of the loop
  SENode* trip_count =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateSubtraction(
          GetTripCount(loop), scalar_evolution_.CreateConstant(1)));
  // Return first_trip_induction_node + trip_count * induction_coefficient
  return scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
      first_trip_induction_node,
      scalar_evolution_.CreateMultiplyNode(trip_count, induction_coefficient)));
}

std::set<const Loop*> LoopDependenceAnalysis::CollectLoops(
    const std::vector<SERecurrentNode*>& recurrent_nodes) {
  // We don't handle loops with more than one induction variable. Therefore we
  // can identify the number of induction variables by collecting all of the
  // loops the collected recurrent nodes belong to.
  std::set<const Loop*> loops{};
  for (auto recurrent_nodes_it = recurrent_nodes.begin();
       recurrent_nodes_it != recurrent_nodes.end(); ++recurrent_nodes_it) {
    loops.insert((*recurrent_nodes_it)->GetLoop());
  }

  return loops;
}

int64_t LoopDependenceAnalysis::CountInductionVariables(SENode* node) {
  if (!node) {
    return -1;
  }

  std::vector<SERecurrentNode*> recurrent_nodes = node->CollectRecurrentNodes();

  // We don't handle loops with more than one induction variable. Therefore we
  // can identify the number of induction variables by collecting all of the
  // loops the collected recurrent nodes belong to.
  std::set<const Loop*> loops = CollectLoops(recurrent_nodes);

  return static_cast<int64_t>(loops.size());
}

std::set<const Loop*> LoopDependenceAnalysis::CollectLoops(
    SENode* source, SENode* destination) {
  if (!source || !destination) {
    return std::set<const Loop*>{};
  }

  std::vector<SERecurrentNode*> source_nodes = source->CollectRecurrentNodes();
  std::vector<SERecurrentNode*> destination_nodes =
      destination->CollectRecurrentNodes();

  std::set<const Loop*> loops = CollectLoops(source_nodes);
  std::set<const Loop*> destination_loops = CollectLoops(destination_nodes);

  loops.insert(std::begin(destination_loops), std::end(destination_loops));

  return loops;
}

int64_t LoopDependenceAnalysis::CountInductionVariables(SENode* source,
                                                        SENode* destination) {
  if (!source || !destination) {
    return -1;
  }

  std::set<const Loop*> loops = CollectLoops(source, destination);

  return static_cast<int64_t>(loops.size());
}

Instruction* LoopDependenceAnalysis::GetOperandDefinition(
    const Instruction* instruction, int id) {
  return context_->get_def_use_mgr()->GetDef(
      instruction->GetSingleWordInOperand(id));
}

std::vector<Instruction*> LoopDependenceAnalysis::GetSubscripts(
    const Instruction* instruction) {
  Instruction* access_chain = GetOperandDefinition(instruction, 0);

  std::vector<Instruction*> subscripts;

  for (auto i = 1u; i < access_chain->NumInOperandWords(); ++i) {
    subscripts.push_back(GetOperandDefinition(access_chain, i));
  }

  return subscripts;
}

SENode* LoopDependenceAnalysis::GetConstantTerm(const Loop* loop,
                                                SERecurrentNode* induction) {
  SENode* offset = induction->GetOffset();
  SENode* lower_bound = GetLowerBound(loop);
  if (!offset || !lower_bound) {
    return nullptr;
  }
  SENode* constant_term = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateSubtraction(offset, lower_bound));
  return constant_term;
}

bool LoopDependenceAnalysis::CheckSupportedLoops(
    std::vector<const Loop*> loops) {
  for (auto loop : loops) {
    if (!IsSupportedLoop(loop)) {
      return false;
    }
  }
  return true;
}

void LoopDependenceAnalysis::MarkUnsusedDistanceEntriesAsIrrelevant(
    const Instruction* source, const Instruction* destination,
    DistanceVector* distance_vector) {
  std::vector<Instruction*> source_subscripts = GetSubscripts(source);
  std::vector<Instruction*> destination_subscripts = GetSubscripts(destination);

  std::set<const Loop*> used_loops{};

  for (Instruction* source_inst : source_subscripts) {
    SENode* source_node = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.AnalyzeInstruction(source_inst));
    std::vector<SERecurrentNode*> recurrent_nodes =
        source_node->CollectRecurrentNodes();
    for (SERecurrentNode* recurrent_node : recurrent_nodes) {
      used_loops.insert(recurrent_node->GetLoop());
    }
  }

  for (Instruction* destination_inst : destination_subscripts) {
    SENode* destination_node = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.AnalyzeInstruction(destination_inst));
    std::vector<SERecurrentNode*> recurrent_nodes =
        destination_node->CollectRecurrentNodes();
    for (SERecurrentNode* recurrent_node : recurrent_nodes) {
      used_loops.insert(recurrent_node->GetLoop());
    }
  }

  for (size_t i = 0; i < loops_.size(); ++i) {
    if (used_loops.find(loops_[i]) == used_loops.end()) {
      distance_vector->GetEntries()[i].dependence_information =
          DistanceEntry::DependenceInformation::IRRELEVANT;
    }
  }
}

bool LoopDependenceAnalysis::IsSupportedLoop(const Loop* loop) {
  std::vector<Instruction*> inductions{};
  loop->GetInductionVariables(inductions);
  if (inductions.size() != 1) {
    return false;
  }
  Instruction* induction = inductions[0];
  SENode* induction_node = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.AnalyzeInstruction(induction));
  if (!induction_node->AsSERecurrentNode()) {
    return false;
  }
  SENode* induction_step =
      induction_node->AsSERecurrentNode()->GetCoefficient();
  if (!induction_step->AsSEConstantNode()) {
    return false;
  }
  if (!(induction_step->AsSEConstantNode()->FoldToSingleValue() == 1 ||
        induction_step->AsSEConstantNode()->FoldToSingleValue() == -1)) {
    return false;
  }
  return true;
}

void LoopDependenceAnalysis::PrintDebug(std::string debug_msg) {
  if (debug_stream_) {
    (*debug_stream_) << debug_msg << "\n";
  }
}

bool Constraint::operator==(const Constraint& other) const {
  // A distance of |d| is equivalent to a line |x - y = -d|
  if ((GetType() == ConstraintType::Distance &&
       other.GetType() == ConstraintType::Line) ||
      (GetType() == ConstraintType::Line &&
       other.GetType() == ConstraintType::Distance)) {
    auto is_distance = AsDependenceLine() != nullptr;

    auto as_distance =
        is_distance ? AsDependenceDistance() : other.AsDependenceDistance();
    auto distance = as_distance->GetDistance();

    auto line = other.AsDependenceLine();

    auto scalar_evolution = distance->GetParentAnalysis();

    auto neg_distance = scalar_evolution->SimplifyExpression(
        scalar_evolution->CreateNegation(distance));

    return *scalar_evolution->CreateConstant(1) == *line->GetA() &&
           *scalar_evolution->CreateConstant(-1) == *line->GetB() &&
           *neg_distance == *line->GetC();
  }

  if (GetType() != other.GetType()) {
    return false;
  }

  if (AsDependenceDistance()) {
    return *AsDependenceDistance()->GetDistance() ==
           *other.AsDependenceDistance()->GetDistance();
  }

  if (AsDependenceLine()) {
    auto this_line = AsDependenceLine();
    auto other_line = other.AsDependenceLine();
    return *this_line->GetA() == *other_line->GetA() &&
           *this_line->GetB() == *other_line->GetB() &&
           *this_line->GetC() == *other_line->GetC();
  }

  if (AsDependencePoint()) {
    auto this_point = AsDependencePoint();
    auto other_point = other.AsDependencePoint();

    return *this_point->GetSource() == *other_point->GetSource() &&
           *this_point->GetDestination() == *other_point->GetDestination();
  }

  return true;
}

bool Constraint::operator!=(const Constraint& other) const {
  return !(*this == other);
}

}  // namespace opt
}  // namespace spvtools
