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

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/scalar_analysis.h"
#include "source/opt/scalar_analysis_nodes.h"

namespace spvtools {
namespace opt {

using SubscriptPair = std::pair<SENode*, SENode*>;

namespace {

// Calculate the greatest common divisor of a & b using Stein's algorithm.
// https://en.wikipedia.org/wiki/Binary_GCD_algorithm
int64_t GreatestCommonDivisor(int64_t a, int64_t b) {
  // Simple cases
  if (a == b) {
    return a;
  } else if (a == 0) {
    return b;
  } else if (b == 0) {
    return a;
  }

  // Both even
  if (a % 2 == 0 && b % 2 == 0) {
    return 2 * GreatestCommonDivisor(a / 2, b / 2);
  }

  // Even a, odd b
  if (a % 2 == 0 && b % 2 == 1) {
    return GreatestCommonDivisor(a / 2, b);
  }

  // Odd a, even b
  if (a % 2 == 1 && b % 2 == 0) {
    return GreatestCommonDivisor(a, b / 2);
  }

  // Both odd, reduce the larger argument
  if (a > b) {
    return GreatestCommonDivisor((a - b) / 2, b);
  } else {
    return GreatestCommonDivisor((b - a) / 2, a);
  }
}

// Check if node is affine, ie in the form: a0*i0 + a1*i1 + ... an*in + c
// and contains only the following types of nodes: SERecurrentNode, SEAddNode
// and SEConstantNode
bool IsInCorrectFormForGCDTest(SENode* node) {
  bool children_ok = true;

  if (auto add_node = node->AsSEAddNode()) {
    for (auto child : add_node->GetChildren()) {
      children_ok &= IsInCorrectFormForGCDTest(child);
    }
  }

  bool this_ok = node->AsSERecurrentNode() || node->AsSEAddNode() ||
                 node->AsSEConstantNode();

  return children_ok && this_ok;
}

// If |node| is an SERecurrentNode then returns |node| or if |node| is an
// SEAddNode returns a vector of SERecurrentNode that are its children.
std::vector<SERecurrentNode*> GetAllTopLevelRecurrences(SENode* node) {
  auto nodes = std::vector<SERecurrentNode*>{};
  if (auto recurrent_node = node->AsSERecurrentNode()) {
    nodes.push_back(recurrent_node);
  }

  if (auto add_node = node->AsSEAddNode()) {
    for (auto child : add_node->GetChildren()) {
      auto child_nodes = GetAllTopLevelRecurrences(child);
      nodes.insert(nodes.end(), child_nodes.begin(), child_nodes.end());
    }
  }

  return nodes;
}

// If |node| is an SEConstantNode then returns |node| or if |node| is an
// SEAddNode returns a vector of SEConstantNode that are its children.
std::vector<SEConstantNode*> GetAllTopLevelConstants(SENode* node) {
  auto nodes = std::vector<SEConstantNode*>{};
  if (auto recurrent_node = node->AsSEConstantNode()) {
    nodes.push_back(recurrent_node);
  }

  if (auto add_node = node->AsSEAddNode()) {
    for (auto child : add_node->GetChildren()) {
      auto child_nodes = GetAllTopLevelConstants(child);
      nodes.insert(nodes.end(), child_nodes.begin(), child_nodes.end());
    }
  }

  return nodes;
}

bool AreOffsetsAndCoefficientsConstant(
    const std::vector<SERecurrentNode*>& nodes) {
  for (auto node : nodes) {
    if (!node->GetOffset()->AsSEConstantNode() ||
        !node->GetOffset()->AsSEConstantNode()) {
      return false;
    }
  }
  return true;
}

// Fold all SEConstantNode that appear in |recurrences| and |constants| into a
// single integer value.
int64_t CalculateConstantTerm(const std::vector<SERecurrentNode*>& recurrences,
                              const std::vector<SEConstantNode*>& constants) {
  int64_t constant_term = 0;
  for (auto recurrence : recurrences) {
    constant_term +=
        recurrence->GetOffset()->AsSEConstantNode()->FoldToSingleValue();
  }

  for (auto constant : constants) {
    constant_term += constant->FoldToSingleValue();
  }

  return constant_term;
}

int64_t CalculateGCDFromCoefficients(
    const std::vector<SERecurrentNode*>& recurrences, int64_t running_gcd) {
  for (SERecurrentNode* recurrence : recurrences) {
    auto coefficient = recurrence->GetCoefficient()->AsSEConstantNode();

    running_gcd = GreatestCommonDivisor(
        running_gcd, std::abs(coefficient->FoldToSingleValue()));
  }

  return running_gcd;
}

// Compare 2 fractions while first normalizing them, e.g. 2/4 and 4/8 will both
// be simplified to 1/2 and then determined to be equal.
bool NormalizeAndCompareFractions(int64_t numerator_0, int64_t denominator_0,
                                  int64_t numerator_1, int64_t denominator_1) {
  auto gcd_0 =
      GreatestCommonDivisor(std::abs(numerator_0), std::abs(denominator_0));
  auto gcd_1 =
      GreatestCommonDivisor(std::abs(numerator_1), std::abs(denominator_1));

  auto normalized_numerator_0 = numerator_0 / gcd_0;
  auto normalized_denominator_0 = denominator_0 / gcd_0;
  auto normalized_numerator_1 = numerator_1 / gcd_1;
  auto normalized_denominator_1 = denominator_1 / gcd_1;

  return normalized_numerator_0 == normalized_numerator_1 &&
         normalized_denominator_0 == normalized_denominator_1;
}

}  // namespace

bool LoopDependenceAnalysis::GetDependence(const Instruction* source,
                                           const Instruction* destination,
                                           DistanceVector* distance_vector) {
  // Start off by finding and marking all the loops in |loops_| that are
  // irrelevant to the dependence analysis.
  MarkUnsusedDistanceEntriesAsIrrelevant(source, destination, distance_vector);

  Instruction* source_access_chain = GetOperandDefinition(source, 0);
  Instruction* destination_access_chain = GetOperandDefinition(destination, 0);

  auto num_access_chains =
      (source_access_chain->opcode() == SpvOpAccessChain) +
      (destination_access_chain->opcode() == SpvOpAccessChain);

  // If neither is an access chain, then they are load/store to a variable.
  if (num_access_chains == 0) {
    if (source_access_chain != destination_access_chain) {
      // Not the same location, report independence
      return true;
    } else {
      // Accessing the same variable
      for (auto& entry : distance_vector->GetEntries()) {
        entry = DistanceEntry();
      }
      return false;
    }
  }

  // If only one is an access chain, it could be accessing a part of a struct
  if (num_access_chains == 1) {
    auto source_is_chain = source_access_chain->opcode() == SpvOpAccessChain;
    auto access_chain =
        source_is_chain ? source_access_chain : destination_access_chain;
    auto variable =
        source_is_chain ? destination_access_chain : source_access_chain;

    auto location_in_chain = GetOperandDefinition(access_chain, 0);

    if (variable != location_in_chain) {
      // Not the same location, report independence
      return true;
    } else {
      // Accessing the same variable
      for (auto& entry : distance_vector->GetEntries()) {
        entry = DistanceEntry();
      }
      return false;
    }
  }

  // If the access chains aren't collecting from the same structure there is no
  // dependence.
  Instruction* source_array = GetOperandDefinition(source_access_chain, 0);
  Instruction* destination_array =
      GetOperandDefinition(destination_access_chain, 0);

  // Nested access chains are not supported yet, bail out.
  if (source_array->opcode() == SpvOpAccessChain ||
      destination_array->opcode() == SpvOpAccessChain) {
    for (auto& entry : distance_vector->GetEntries()) {
      entry = DistanceEntry();
    }
    return false;
  }

  if (source_array != destination_array) {
    PrintDebug("Proved independence through different arrays.");
    return true;
  }

  // To handle multiple subscripts we must get every operand in the access
  // chains past the first.
  std::vector<Instruction*> source_subscripts = GetSubscripts(source);
  std::vector<Instruction*> destination_subscripts = GetSubscripts(destination);

  auto sets_of_subscripts =
      PartitionSubscripts(source_subscripts, destination_subscripts);

  auto first_coupled = std::partition(
      std::begin(sets_of_subscripts), std::end(sets_of_subscripts),
      [](const std::set<std::pair<Instruction*, Instruction*>>& set) {
        return set.size() == 1;
      });

  // Go through each subscript testing for independence.
  // If any subscript results in independence, we prove independence between the
  // load and store.
  // If we can't prove independence we store what information we can gather in
  // a DistanceVector.
  for (auto it = std::begin(sets_of_subscripts); it < first_coupled; ++it) {
    auto source_subscript = std::get<0>(*(*it).begin());
    auto destination_subscript = std::get<1>(*(*it).begin());

    SENode* source_node = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.AnalyzeInstruction(source_subscript));
    SENode* destination_node = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.AnalyzeInstruction(destination_subscript));

    // Check the loops are in a form we support.
    auto subscript_pair = std::make_pair(source_node, destination_node);

    const Loop* loop = GetLoopForSubscriptPair(subscript_pair);
    if (loop) {
      if (!IsSupportedLoop(loop)) {
        PrintDebug(
            "GetDependence found an unsupported loop form. Assuming <=> for "
            "loop.");
        DistanceEntry* distance_entry =
            GetDistanceEntryForSubscriptPair(subscript_pair, distance_vector);
        if (distance_entry) {
          distance_entry->direction = DistanceEntry::Directions::ALL;
        }
        continue;
      }
    }

    // If either node is simplified to a CanNotCompute we can't perform any
    // analysis so must assume <=> dependence and return.
    if (source_node->GetType() == SENode::CanNotCompute ||
        destination_node->GetType() == SENode::CanNotCompute) {
      // Record the <=> dependence if we can get a DistanceEntry
      PrintDebug(
          "GetDependence found source_node || destination_node as "
          "CanNotCompute. Abandoning evaluation for this subscript.");
      DistanceEntry* distance_entry =
          GetDistanceEntryForSubscriptPair(subscript_pair, distance_vector);
      if (distance_entry) {
        distance_entry->direction = DistanceEntry::Directions::ALL;
      }
      continue;
    }

    // We have no induction variables so can apply a ZIV test.
    if (IsZIV(subscript_pair)) {
      PrintDebug("Found a ZIV subscript pair");
      if (ZIVTest(subscript_pair)) {
        PrintDebug("Proved independence with ZIVTest.");
        return true;
      }
    }

    // We have only one induction variable so should attempt an SIV test.
    if (IsSIV(subscript_pair)) {
      PrintDebug("Found a SIV subscript pair.");
      if (SIVTest(subscript_pair, distance_vector)) {
        PrintDebug("Proved independence with SIVTest.");
        return true;
      }
    }

    // We have multiple induction variables so should attempt an MIV test.
    if (IsMIV(subscript_pair)) {
      PrintDebug("Found a MIV subscript pair.");
      if (GCDMIVTest(subscript_pair)) {
        PrintDebug("Proved independence with the GCD test.");
        auto current_loops = CollectLoops(source_node, destination_node);

        for (auto current_loop : current_loops) {
          auto distance_entry =
              GetDistanceEntryForLoop(current_loop, distance_vector);
          distance_entry->direction = DistanceEntry::Directions::NONE;
        }
        return true;
      }
    }
  }

  for (auto it = first_coupled; it < std::end(sets_of_subscripts); ++it) {
    auto coupled_instructions = *it;
    std::vector<SubscriptPair> coupled_subscripts{};

    for (const auto& elem : coupled_instructions) {
      auto source_subscript = std::get<0>(elem);
      auto destination_subscript = std::get<1>(elem);

      SENode* source_node = scalar_evolution_.SimplifyExpression(
          scalar_evolution_.AnalyzeInstruction(source_subscript));
      SENode* destination_node = scalar_evolution_.SimplifyExpression(
          scalar_evolution_.AnalyzeInstruction(destination_subscript));

      coupled_subscripts.push_back({source_node, destination_node});
    }

    auto supported = true;

    for (const auto& subscript : coupled_subscripts) {
      auto loops = CollectLoops(std::get<0>(subscript), std::get<1>(subscript));

      auto is_subscript_supported =
          std::all_of(std::begin(loops), std::end(loops),
                      [this](const Loop* l) { return IsSupportedLoop(l); });

      supported = supported && is_subscript_supported;
    }

    if (DeltaTest(coupled_subscripts, distance_vector)) {
      return true;
    }
  }

  // We were unable to prove independence so must gather all of the direction
  // information we found.
  PrintDebug(
      "Couldn't prove independence.\n"
      "All possible direction information has been collected in the input "
      "DistanceVector.");

  return false;
}

bool LoopDependenceAnalysis::ZIVTest(
    const std::pair<SENode*, SENode*>& subscript_pair) {
  auto source = std::get<0>(subscript_pair);
  auto destination = std::get<1>(subscript_pair);

  PrintDebug("Performing ZIVTest");
  // If source == destination, dependence with direction = and distance 0.
  if (source == destination) {
    PrintDebug("ZIVTest found EQ dependence.");
    return false;
  } else {
    PrintDebug("ZIVTest found independence.");
    // Otherwise we prove independence.
    return true;
  }
}

bool LoopDependenceAnalysis::SIVTest(
    const std::pair<SENode*, SENode*>& subscript_pair,
    DistanceVector* distance_vector) {
  DistanceEntry* distance_entry =
      GetDistanceEntryForSubscriptPair(subscript_pair, distance_vector);
  if (!distance_entry) {
    PrintDebug(
        "SIVTest could not find a DistanceEntry for subscript_pair. Exiting");
  }

  SENode* source_node = std::get<0>(subscript_pair);
  SENode* destination_node = std::get<1>(subscript_pair);

  int64_t source_induction_count = CountInductionVariables(source_node);
  int64_t destination_induction_count =
      CountInductionVariables(destination_node);

  // If the source node has no induction variables we can apply a
  // WeakZeroSrcTest.
  if (source_induction_count == 0) {
    PrintDebug("Found source has no induction variable.");
    if (WeakZeroSourceSIVTest(
            source_node, destination_node->AsSERecurrentNode(),
            destination_node->AsSERecurrentNode()->GetCoefficient(),
            distance_entry)) {
      PrintDebug("Proved independence with WeakZeroSourceSIVTest.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    }
  }

  // If the destination has no induction variables we can apply a
  // WeakZeroDestTest.
  if (destination_induction_count == 0) {
    PrintDebug("Found destination has no induction variable.");
    if (WeakZeroDestinationSIVTest(
            source_node->AsSERecurrentNode(), destination_node,
            source_node->AsSERecurrentNode()->GetCoefficient(),
            distance_entry)) {
      PrintDebug("Proved independence with WeakZeroDestinationSIVTest.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    }
  }

  // We now need to collect the SERecurrentExpr nodes from source and
  // destination. We do not handle cases where source or destination have
  // multiple SERecurrentExpr nodes.
  std::vector<SERecurrentNode*> source_recurrent_nodes =
      source_node->CollectRecurrentNodes();
  std::vector<SERecurrentNode*> destination_recurrent_nodes =
      destination_node->CollectRecurrentNodes();

  if (source_recurrent_nodes.size() == 1 &&
      destination_recurrent_nodes.size() == 1) {
    PrintDebug("Found source and destination have 1 induction variable.");
    SERecurrentNode* source_recurrent_expr = *source_recurrent_nodes.begin();
    SERecurrentNode* destination_recurrent_expr =
        *destination_recurrent_nodes.begin();

    // If the coefficients are identical we can apply a StrongSIVTest.
    if (source_recurrent_expr->GetCoefficient() ==
        destination_recurrent_expr->GetCoefficient()) {
      PrintDebug("Found source and destination share coefficient.");
      if (StrongSIVTest(source_node, destination_node,
                        source_recurrent_expr->GetCoefficient(),
                        distance_entry)) {
        PrintDebug("Proved independence with StrongSIVTest");
        distance_entry->dependence_information =
            DistanceEntry::DependenceInformation::DIRECTION;
        distance_entry->direction = DistanceEntry::Directions::NONE;
        return true;
      }
    }

    // If the coefficients are of equal magnitude and opposite sign we can
    // apply a WeakCrossingSIVTest.
    if (source_recurrent_expr->GetCoefficient() ==
        scalar_evolution_.CreateNegation(
            destination_recurrent_expr->GetCoefficient())) {
      PrintDebug("Found source coefficient = -destination coefficient.");
      if (WeakCrossingSIVTest(source_node, destination_node,
                              source_recurrent_expr->GetCoefficient(),
                              distance_entry)) {
        PrintDebug("Proved independence with WeakCrossingSIVTest");
        distance_entry->dependence_information =
            DistanceEntry::DependenceInformation::DIRECTION;
        distance_entry->direction = DistanceEntry::Directions::NONE;
        return true;
      }
    }
  }

  return false;
}

bool LoopDependenceAnalysis::StrongSIVTest(SENode* source, SENode* destination,
                                           SENode* coefficient,
                                           DistanceEntry* distance_entry) {
  PrintDebug("Performing StrongSIVTest.");
  // If both source and destination are SERecurrentNodes we can perform tests
  // based on distance.
  // If either source or destination contain value unknown nodes or if one or
  // both are not SERecurrentNodes we must attempt a symbolic test.
  std::vector<SEValueUnknown*> source_value_unknown_nodes =
      source->CollectValueUnknownNodes();
  std::vector<SEValueUnknown*> destination_value_unknown_nodes =
      destination->CollectValueUnknownNodes();
  if (source_value_unknown_nodes.size() > 0 ||
      destination_value_unknown_nodes.size() > 0) {
    PrintDebug(
        "StrongSIVTest found symbolics. Will attempt SymbolicStrongSIVTest.");
    return SymbolicStrongSIVTest(source, destination, coefficient,
                                 distance_entry);
  }

  if (!source->AsSERecurrentNode() || !destination->AsSERecurrentNode()) {
    PrintDebug(
        "StrongSIVTest could not simplify source and destination to "
        "SERecurrentNodes so will exit.");
    distance_entry->direction = DistanceEntry::Directions::ALL;
    return false;
  }

  // Build an SENode for distance.
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const Loop* subscript_loop = GetLoopForSubscriptPair(subscript_pair);
  SENode* source_constant_term =
      GetConstantTerm(subscript_loop, source->AsSERecurrentNode());
  SENode* destination_constant_term =
      GetConstantTerm(subscript_loop, destination->AsSERecurrentNode());
  if (!source_constant_term || !destination_constant_term) {
    PrintDebug(
        "StrongSIVTest could not collect the constant terms of either source "
        "or destination so will exit.");
    return false;
  }
  SENode* constant_term_delta =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateSubtraction(
          destination_constant_term, source_constant_term));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  // We must check the offset delta and coefficient are constants.
  int64_t distance = 0;
  SEConstantNode* delta_constant = constant_term_delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    PrintDebug(
        "StrongSIVTest found delta value and coefficient value as constants "
        "with values:\n"
        "\tdelta value: " +
        ToString(delta_value) +
        "\n\tcoefficient value: " + ToString(coefficient_value) + "\n");
    // Check if the distance is not integral to try to prove independence.
    if (delta_value % coefficient_value != 0) {
      PrintDebug(
          "StrongSIVTest proved independence through distance not being an "
          "integer.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / coefficient_value;
      PrintDebug("StrongSIV test found distance as " + ToString(distance));
    }
  } else {
    // If we can't fold delta and coefficient to single values we can't produce
    // distance.
    // As a result we can't perform the rest of the pass and must assume
    // dependence in all directions.
    PrintDebug("StrongSIVTest could not produce a distance. Must exit.");
    distance_entry->distance = DistanceEntry::Directions::ALL;
    return false;
  }

  // Next we gather the upper and lower bounds as constants if possible. If
  // distance > upper_bound - lower_bound we prove independence.
  SENode* lower_bound = GetLowerBound(subscript_loop);
  SENode* upper_bound = GetUpperBound(subscript_loop);
  if (lower_bound && upper_bound) {
    PrintDebug("StrongSIVTest found bounds.");
    SENode* bounds = scalar_evolution_.SimplifyExpression(
        scalar_evolution_.CreateSubtraction(upper_bound, lower_bound));

    if (bounds->GetType() == SENode::SENodeType::Constant) {
      int64_t bounds_value = bounds->AsSEConstantNode()->FoldToSingleValue();
      PrintDebug(
          "StrongSIVTest found upper_bound - lower_bound as a constant with "
          "value " +
          ToString(bounds_value));

      // If the absolute value of the distance is > upper bound - lower bound
      // then we prove independence.
      if (llabs(distance) > llabs(bounds_value)) {
        PrintDebug(
            "StrongSIVTest proved independence through distance escaping the "
            "loop bounds.");
        distance_entry->dependence_information =
            DistanceEntry::DependenceInformation::DISTANCE;
        distance_entry->direction = DistanceEntry::Directions::NONE;
        distance_entry->distance = distance;
        return true;
      }
    }
  } else {
    PrintDebug("StrongSIVTest was unable to gather lower and upper bounds.");
  }

  // Otherwise we can get a direction as follows
  //             { < if distance > 0
  // direction = { = if distance == 0
  //             { > if distance < 0
  PrintDebug(
      "StrongSIVTest could not prove independence. Gathering direction "
      "information.");
  if (distance > 0) {
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DISTANCE;
    distance_entry->direction = DistanceEntry::Directions::LT;
    distance_entry->distance = distance;
    return false;
  }
  if (distance == 0) {
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DISTANCE;
    distance_entry->direction = DistanceEntry::Directions::EQ;
    distance_entry->distance = 0;
    return false;
  }
  if (distance < 0) {
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DISTANCE;
    distance_entry->direction = DistanceEntry::Directions::GT;
    distance_entry->distance = distance;
    return false;
  }

  // We were unable to prove independence or discern any additional information
  // Must assume <=> direction.
  PrintDebug(
      "StrongSIVTest was unable to determine any dependence information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::SymbolicStrongSIVTest(
    SENode* source, SENode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing SymbolicStrongSIVTest.");
  SENode* source_destination_delta = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateSubtraction(source, destination));
  // By cancelling out the induction variables by subtracting the source and
  // destination we can produce an expression of symbolics and constants. This
  // expression can be compared to the loop bounds to find if the offset is
  // outwith the bounds.
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const Loop* subscript_loop = GetLoopForSubscriptPair(subscript_pair);
  if (IsProvablyOutsideOfLoopBounds(subscript_loop, source_destination_delta,
                                    coefficient)) {
    PrintDebug(
        "SymbolicStrongSIVTest proved independence through loop bounds.");
    distance_entry->dependence_information =
        DistanceEntry::DependenceInformation::DIRECTION;
    distance_entry->direction = DistanceEntry::Directions::NONE;
    return true;
  }
  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "SymbolicStrongSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::WeakZeroSourceSIVTest(
    SENode* source, SERecurrentNode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing WeakZeroSourceSIVTest.");
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const Loop* subscript_loop = GetLoopForSubscriptPair(subscript_pair);
  // Build an SENode for distance.
  SENode* destination_constant_term =
      GetConstantTerm(subscript_loop, destination);
  SENode* delta = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateSubtraction(source, destination_constant_term));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  int64_t distance = 0;
  SEConstantNode* delta_constant = delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    PrintDebug(
        "WeakZeroSourceSIVTest folding delta and coefficient to constants.");
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    // Check if the distance is not integral.
    if (delta_value % coefficient_value != 0) {
      PrintDebug(
          "WeakZeroSourceSIVTest proved independence through distance not "
          "being an integer.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / coefficient_value;
      PrintDebug(
          "WeakZeroSourceSIVTest calculated distance with the following "
          "values\n"
          "\tdelta value: " +
          ToString(delta_value) +
          "\n\tcoefficient value: " + ToString(coefficient_value) +
          "\n\tdistance: " + ToString(distance) + "\n");
    }
  } else {
    PrintDebug(
        "WeakZeroSourceSIVTest was unable to fold delta and coefficient to "
        "constants.");
  }

  // If we can prove the distance is outside the bounds we prove independence.
  SEConstantNode* lower_bound =
      GetLowerBound(subscript_loop)->AsSEConstantNode();
  SEConstantNode* upper_bound =
      GetUpperBound(subscript_loop)->AsSEConstantNode();
  if (lower_bound && upper_bound) {
    PrintDebug("WeakZeroSourceSIVTest found bounds as SEConstantNodes.");
    int64_t lower_bound_value = lower_bound->FoldToSingleValue();
    int64_t upper_bound_value = upper_bound->FoldToSingleValue();
    if (!IsWithinBounds(llabs(distance), lower_bound_value,
                        upper_bound_value)) {
      PrintDebug(
          "WeakZeroSourceSIVTest proved independence through distance escaping "
          "the loop bounds.");
      PrintDebug(
          "Bound values were as follow\n"
          "\tlower bound value: " +
          ToString(lower_bound_value) +
          "\n\tupper bound value: " + ToString(upper_bound_value) +
          "\n\tdistance value: " + ToString(distance) + "\n");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DISTANCE;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      distance_entry->distance = distance;
      return true;
    }
  } else {
    PrintDebug(
        "WeakZeroSourceSIVTest was unable to find lower and upper bound as "
        "SEConstantNodes.");
  }

  // Now we want to see if we can detect to peel the first or last iterations.

  // We get the FirstTripValue as GetFirstTripInductionNode() +
  // GetConstantTerm(destination)
  SENode* first_trip_SENode =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
          GetFirstTripInductionNode(subscript_loop),
          GetConstantTerm(subscript_loop, destination)));

  // If source == FirstTripValue, peel_first.
  if (first_trip_SENode) {
    PrintDebug("WeakZeroSourceSIVTest built first_trip_SENode.");
    if (first_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroSourceSIVTest has found first_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(first_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (source == first_trip_SENode) {
      // We have found that peeling the first iteration will break dependency.
      PrintDebug(
          "WeakZeroSourceSIVTest has found peeling first iteration will break "
          "dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_first = true;
      return false;
    }
  } else {
    PrintDebug("WeakZeroSourceSIVTest was unable to build first_trip_SENode");
  }

  // We get the LastTripValue as GetFinalTripInductionNode(coefficient) +
  // GetConstantTerm(destination)
  SENode* final_trip_SENode =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
          GetFinalTripInductionNode(subscript_loop, coefficient),
          GetConstantTerm(subscript_loop, destination)));

  // If source == LastTripValue, peel_last.
  if (final_trip_SENode) {
    PrintDebug("WeakZeroSourceSIVTest built final_trip_SENode.");
    if (first_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroSourceSIVTest has found final_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(final_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (source == final_trip_SENode) {
      // We have found that peeling the last iteration will break dependency.
      PrintDebug(
          "WeakZeroSourceSIVTest has found peeling final iteration will break "
          "dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_last = true;
      return false;
    }
  } else {
    PrintDebug("WeakZeroSourceSIVTest was unable to build final_trip_SENode");
  }

  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "WeakZeroSourceSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::WeakZeroDestinationSIVTest(
    SERecurrentNode* source, SENode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing WeakZeroDestinationSIVTest.");
  // Build an SENode for distance.
  std::pair<SENode*, SENode*> subscript_pair =
      std::make_pair(source, destination);
  const Loop* subscript_loop = GetLoopForSubscriptPair(subscript_pair);
  SENode* source_constant_term = GetConstantTerm(subscript_loop, source);
  SENode* delta = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateSubtraction(destination, source_constant_term));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  int64_t distance = 0;
  SEConstantNode* delta_constant = delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    PrintDebug(
        "WeakZeroDestinationSIVTest folding delta and coefficient to "
        "constants.");
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    // Check if the distance is not integral.
    if (delta_value % coefficient_value != 0) {
      PrintDebug(
          "WeakZeroDestinationSIVTest proved independence through distance not "
          "being an integer.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / coefficient_value;
      PrintDebug(
          "WeakZeroDestinationSIVTest calculated distance with the following "
          "values\n"
          "\tdelta value: " +
          ToString(delta_value) +
          "\n\tcoefficient value: " + ToString(coefficient_value) +
          "\n\tdistance: " + ToString(distance) + "\n");
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to fold delta and coefficient "
        "to constants.");
  }

  // If we can prove the distance is outside the bounds we prove independence.
  SEConstantNode* lower_bound =
      GetLowerBound(subscript_loop)->AsSEConstantNode();
  SEConstantNode* upper_bound =
      GetUpperBound(subscript_loop)->AsSEConstantNode();
  if (lower_bound && upper_bound) {
    PrintDebug("WeakZeroDestinationSIVTest found bounds as SEConstantNodes.");
    int64_t lower_bound_value = lower_bound->FoldToSingleValue();
    int64_t upper_bound_value = upper_bound->FoldToSingleValue();
    if (!IsWithinBounds(llabs(distance), lower_bound_value,
                        upper_bound_value)) {
      PrintDebug(
          "WeakZeroDestinationSIVTest proved independence through distance "
          "escaping the loop bounds.");
      PrintDebug(
          "Bound values were as follows\n"
          "\tlower bound value: " +
          ToString(lower_bound_value) +
          "\n\tupper bound value: " + ToString(upper_bound_value) +
          "\n\tdistance value: " + ToString(distance));
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DISTANCE;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      distance_entry->distance = distance;
      return true;
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to find lower and upper bound "
        "as SEConstantNodes.");
  }

  // Now we want to see if we can detect to peel the first or last iterations.

  // We get the FirstTripValue as GetFirstTripInductionNode() +
  // GetConstantTerm(source)
  SENode* first_trip_SENode = scalar_evolution_.SimplifyExpression(
      scalar_evolution_.CreateAddNode(GetFirstTripInductionNode(subscript_loop),
                                      GetConstantTerm(subscript_loop, source)));

  // If destination == FirstTripValue, peel_first.
  if (first_trip_SENode) {
    PrintDebug("WeakZeroDestinationSIVTest built first_trip_SENode.");
    if (first_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroDestinationSIVTest has found first_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(first_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (destination == first_trip_SENode) {
      // We have found that peeling the first iteration will break dependency.
      PrintDebug(
          "WeakZeroDestinationSIVTest has found peeling first iteration will "
          "break dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_first = true;
      return false;
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to build first_trip_SENode");
  }

  // We get the LastTripValue as GetFinalTripInductionNode(coefficient) +
  // GetConstantTerm(source)
  SENode* final_trip_SENode =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateAddNode(
          GetFinalTripInductionNode(subscript_loop, coefficient),
          GetConstantTerm(subscript_loop, source)));

  // If destination == LastTripValue, peel_last.
  if (final_trip_SENode) {
    PrintDebug("WeakZeroDestinationSIVTest built final_trip_SENode.");
    if (final_trip_SENode->AsSEConstantNode()) {
      PrintDebug(
          "WeakZeroDestinationSIVTest has found final_trip_SENode as an "
          "SEConstantNode with value: " +
          ToString(final_trip_SENode->AsSEConstantNode()->FoldToSingleValue()) +
          "\n");
    }
    if (destination == final_trip_SENode) {
      // We have found that peeling the last iteration will break dependency.
      PrintDebug(
          "WeakZeroDestinationSIVTest has found peeling final iteration will "
          "break dependency");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::PEEL;
      distance_entry->peel_last = true;
      return false;
    }
  } else {
    PrintDebug(
        "WeakZeroDestinationSIVTest was unable to build final_trip_SENode");
  }

  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "WeakZeroDestinationSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

bool LoopDependenceAnalysis::WeakCrossingSIVTest(
    SENode* source, SENode* destination, SENode* coefficient,
    DistanceEntry* distance_entry) {
  PrintDebug("Performing WeakCrossingSIVTest.");
  // We currently can't handle symbolic WeakCrossingSIVTests. If either source
  // or destination are not SERecurrentNodes we must exit.
  if (!source->AsSERecurrentNode() || !destination->AsSERecurrentNode()) {
    PrintDebug(
        "WeakCrossingSIVTest found source or destination != SERecurrentNode. "
        "Exiting");
    distance_entry->direction = DistanceEntry::Directions::ALL;
    return false;
  }

  // Build an SENode for distance.
  SENode* offset_delta =
      scalar_evolution_.SimplifyExpression(scalar_evolution_.CreateSubtraction(
          destination->AsSERecurrentNode()->GetOffset(),
          source->AsSERecurrentNode()->GetOffset()));

  // Scalar evolution doesn't perform division, so we must fold to constants and
  // do it manually.
  int64_t distance = 0;
  SEConstantNode* delta_constant = offset_delta->AsSEConstantNode();
  SEConstantNode* coefficient_constant = coefficient->AsSEConstantNode();
  if (delta_constant && coefficient_constant) {
    PrintDebug(
        "WeakCrossingSIVTest folding offset_delta and coefficient to "
        "constants.");
    int64_t delta_value = delta_constant->FoldToSingleValue();
    int64_t coefficient_value = coefficient_constant->FoldToSingleValue();
    // Check if the distance is not integral or if it has a non-integral part
    // equal to 1/2.
    if (delta_value % (2 * coefficient_value) != 0 &&
        static_cast<float>(delta_value % (2 * coefficient_value)) /
                static_cast<float>(2 * coefficient_value) !=
            0.5) {
      PrintDebug(
          "WeakCrossingSIVTest proved independence through distance escaping "
          "the loop bounds.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DIRECTION;
      distance_entry->direction = DistanceEntry::Directions::NONE;
      return true;
    } else {
      distance = delta_value / (2 * coefficient_value);
    }

    if (distance == 0) {
      PrintDebug("WeakCrossingSIVTest found EQ dependence.");
      distance_entry->dependence_information =
          DistanceEntry::DependenceInformation::DISTANCE;
      distance_entry->direction = DistanceEntry::Directions::EQ;
      distance_entry->distance = 0;
      return false;
    }
  } else {
    PrintDebug(
        "WeakCrossingSIVTest was unable to fold offset_delta and coefficient "
        "to constants.");
  }

  // We were unable to prove independence or discern any additional information.
  // Must assume <=> direction.
  PrintDebug(
      "WeakCrossingSIVTest was unable to determine any dependence "
      "information.");
  distance_entry->direction = DistanceEntry::Directions::ALL;
  return false;
}

// Perform the GCD test if both, the source and the destination nodes, are in
// the form a0*i0 + a1*i1 + ... an*in + c.
bool LoopDependenceAnalysis::GCDMIVTest(
    const std::pair<SENode*, SENode*>& subscript_pair) {
  auto source = std::get<0>(subscript_pair);
  auto destination = std::get<1>(subscript_pair);

  // Bail out if source/destination is in an unexpected form.
  if (!IsInCorrectFormForGCDTest(source) ||
      !IsInCorrectFormForGCDTest(destination)) {
    return false;
  }

  auto source_recurrences = GetAllTopLevelRecurrences(source);
  auto dest_recurrences = GetAllTopLevelRecurrences(destination);

  // Bail out if all offsets and coefficients aren't constant.
  if (!AreOffsetsAndCoefficientsConstant(source_recurrences) ||
      !AreOffsetsAndCoefficientsConstant(dest_recurrences)) {
    return false;
  }

  // Calculate the GCD of all coefficients.
  auto source_constants = GetAllTopLevelConstants(source);
  int64_t source_constant =
      CalculateConstantTerm(source_recurrences, source_constants);

  auto dest_constants = GetAllTopLevelConstants(destination);
  int64_t destination_constant =
      CalculateConstantTerm(dest_recurrences, dest_constants);

  int64_t delta = std::abs(source_constant - destination_constant);

  int64_t running_gcd = 0;

  running_gcd = CalculateGCDFromCoefficients(source_recurrences, running_gcd);
  running_gcd = CalculateGCDFromCoefficients(dest_recurrences, running_gcd);

  return delta % running_gcd != 0;
}

using PartitionedSubscripts =
    std::vector<std::set<std::pair<Instruction*, Instruction*>>>;
PartitionedSubscripts LoopDependenceAnalysis::PartitionSubscripts(
    const std::vector<Instruction*>& source_subscripts,
    const std::vector<Instruction*>& destination_subscripts) {
  PartitionedSubscripts partitions{};

  auto num_subscripts = source_subscripts.size();

  // Create initial partitions with one subscript pair per partition.
  for (size_t i = 0; i < num_subscripts; ++i) {
    partitions.push_back({{source_subscripts[i], destination_subscripts[i]}});
  }

  // Iterate over the loops to create all partitions
  for (auto loop : loops_) {
    int64_t k = -1;

    for (size_t j = 0; j < partitions.size(); ++j) {
      auto& current_partition = partitions[j];

      // Does |loop| appear in |current_partition|
      auto it = std::find_if(
          current_partition.begin(), current_partition.end(),
          [loop,
           this](const std::pair<Instruction*, Instruction*>& elem) -> bool {
            auto source_recurrences =
                scalar_evolution_.AnalyzeInstruction(std::get<0>(elem))
                    ->CollectRecurrentNodes();
            auto destination_recurrences =
                scalar_evolution_.AnalyzeInstruction(std::get<1>(elem))
                    ->CollectRecurrentNodes();

            source_recurrences.insert(source_recurrences.end(),
                                      destination_recurrences.begin(),
                                      destination_recurrences.end());

            auto loops_in_pair = CollectLoops(source_recurrences);
            auto end_it = loops_in_pair.end();

            return std::find(loops_in_pair.begin(), end_it, loop) != end_it;
          });

      auto has_loop = it != current_partition.end();

      if (has_loop) {
        if (k == -1) {
          k = j;
        } else {
          // Add |partitions[j]| to |partitions[k]| and discard |partitions[j]|
          partitions[static_cast<size_t>(k)].insert(current_partition.begin(),
                                                    current_partition.end());
          current_partition.clear();
        }
      }
    }
  }

  // Remove empty (discarded) partitions
  partitions.erase(
      std::remove_if(
          partitions.begin(), partitions.end(),
          [](const std::set<std::pair<Instruction*, Instruction*>>& partition) {
            return partition.empty();
          }),
      partitions.end());

  return partitions;
}

Constraint* LoopDependenceAnalysis::IntersectConstraints(
    Constraint* constraint_0, Constraint* constraint_1,
    const SENode* lower_bound, const SENode* upper_bound) {
  if (constraint_0->AsDependenceNone()) {
    return constraint_1;
  } else if (constraint_1->AsDependenceNone()) {
    return constraint_0;
  }

  // Both constraints are distances. Either the same distance or independent.
  if (constraint_0->AsDependenceDistance() &&
      constraint_1->AsDependenceDistance()) {
    auto dist_0 = constraint_0->AsDependenceDistance();
    auto dist_1 = constraint_1->AsDependenceDistance();

    if (*dist_0->GetDistance() == *dist_1->GetDistance()) {
      return constraint_0;
    } else {
      return make_constraint<DependenceEmpty>();
    }
  }

  // Both constraints are points. Either the same point or independent.
  if (constraint_0->AsDependencePoint() && constraint_1->AsDependencePoint()) {
    auto point_0 = constraint_0->AsDependencePoint();
    auto point_1 = constraint_1->AsDependencePoint();

    if (*point_0->GetSource() == *point_1->GetSource() &&
        *point_0->GetDestination() == *point_1->GetDestination()) {
      return constraint_0;
    } else {
      return make_constraint<DependenceEmpty>();
    }
  }

  // Both constraints are lines/distances.
  if ((constraint_0->AsDependenceDistance() ||
       constraint_0->AsDependenceLine()) &&
      (constraint_1->AsDependenceDistance() ||
       constraint_1->AsDependenceLine())) {
    auto is_distance_0 = constraint_0->AsDependenceDistance() != nullptr;
    auto is_distance_1 = constraint_1->AsDependenceDistance() != nullptr;

    auto a0 = is_distance_0 ? scalar_evolution_.CreateConstant(1)
                            : constraint_0->AsDependenceLine()->GetA();
    auto b0 = is_distance_0 ? scalar_evolution_.CreateConstant(-1)
                            : constraint_0->AsDependenceLine()->GetB();
    auto c0 =
        is_distance_0
            ? scalar_evolution_.SimplifyExpression(
                  scalar_evolution_.CreateNegation(
                      constraint_0->AsDependenceDistance()->GetDistance()))
            : constraint_0->AsDependenceLine()->GetC();

    auto a1 = is_distance_1 ? scalar_evolution_.CreateConstant(1)
                            : constraint_1->AsDependenceLine()->GetA();
    auto b1 = is_distance_1 ? scalar_evolution_.CreateConstant(-1)
                            : constraint_1->AsDependenceLine()->GetB();
    auto c1 =
        is_distance_1
            ? scalar_evolution_.SimplifyExpression(
                  scalar_evolution_.CreateNegation(
                      constraint_1->AsDependenceDistance()->GetDistance()))
            : constraint_1->AsDependenceLine()->GetC();

    if (a0->AsSEConstantNode() && b0->AsSEConstantNode() &&
        c0->AsSEConstantNode() && a1->AsSEConstantNode() &&
        b1->AsSEConstantNode() && c1->AsSEConstantNode()) {
      auto constant_a0 = a0->AsSEConstantNode()->FoldToSingleValue();
      auto constant_b0 = b0->AsSEConstantNode()->FoldToSingleValue();
      auto constant_c0 = c0->AsSEConstantNode()->FoldToSingleValue();

      auto constant_a1 = a1->AsSEConstantNode()->FoldToSingleValue();
      auto constant_b1 = b1->AsSEConstantNode()->FoldToSingleValue();
      auto constant_c1 = c1->AsSEConstantNode()->FoldToSingleValue();

      // a & b can't both be zero, otherwise it wouldn't be line.
      if (NormalizeAndCompareFractions(constant_a0, constant_b0, constant_a1,
                                       constant_b1)) {
        // Slopes are equal, either parallel lines or the same line.

        if (constant_b0 == 0 && constant_b1 == 0) {
          if (NormalizeAndCompareFractions(constant_c0, constant_a0,
                                           constant_c1, constant_a1)) {
            return constraint_0;
          }

          return make_constraint<DependenceEmpty>();
        } else if (NormalizeAndCompareFractions(constant_c0, constant_b0,
                                                constant_c1, constant_b1)) {
          // Same line.
          return constraint_0;
        } else {
          // Parallel lines can't intersect, report independence.
          return make_constraint<DependenceEmpty>();
        }

      } else {
        // Lines are not parallel, therefore, they must intersect.

        // Calculate intersection.
        if (upper_bound->AsSEConstantNode() &&
            lower_bound->AsSEConstantNode()) {
          auto constant_lower_bound =
              lower_bound->AsSEConstantNode()->FoldToSingleValue();
          auto constant_upper_bound =
              upper_bound->AsSEConstantNode()->FoldToSingleValue();

          auto up = constant_b1 * constant_c0 - constant_b0 * constant_c1;
          // Both b or both a can't be 0, so down is never 0
          // otherwise would have entered the parallel line section.
          auto down = constant_b1 * constant_a0 - constant_b0 * constant_a1;

          auto x_coord = up / down;

          int64_t y_coord = 0;
          int64_t arg1 = 0;
          int64_t const_b_to_use = 0;

          if (constant_b1 != 0) {
            arg1 = constant_c1 - constant_a1 * x_coord;
            y_coord = arg1 / constant_b1;
            const_b_to_use = constant_b1;
          } else if (constant_b0 != 0) {
            arg1 = constant_c0 - constant_a0 * x_coord;
            y_coord = arg1 / constant_b0;
            const_b_to_use = constant_b0;
          }

          if (up % down == 0 &&
              arg1 % const_b_to_use == 0 &&  // Coordinates are integers.
              constant_lower_bound <=
                  x_coord &&  // x_coord is within loop bounds.
              x_coord <= constant_upper_bound &&
              constant_lower_bound <=
                  y_coord &&  // y_coord is within loop bounds.
              y_coord <= constant_upper_bound) {
            // Lines intersect at integer coordinates.
            return make_constraint<DependencePoint>(
                scalar_evolution_.CreateConstant(x_coord),
                scalar_evolution_.CreateConstant(y_coord),
                constraint_0->GetLoop());

          } else {
            return make_constraint<DependenceEmpty>();
          }

        } else {
          // Not constants, bail out.
          return make_constraint<DependenceNone>();
        }
      }

    } else {
      // Not constants, bail out.
      return make_constraint<DependenceNone>();
    }
  }

  // One constraint is a line/distance and the other is a point.
  if ((constraint_0->AsDependencePoint() &&
       (constraint_1->AsDependenceLine() ||
        constraint_1->AsDependenceDistance())) ||
      (constraint_1->AsDependencePoint() &&
       (constraint_0->AsDependenceLine() ||
        constraint_0->AsDependenceDistance()))) {
    auto point_0 = constraint_0->AsDependencePoint() != nullptr;

    auto point = point_0 ? constraint_0->AsDependencePoint()
                         : constraint_1->AsDependencePoint();

    auto line_or_distance = point_0 ? constraint_1 : constraint_0;

    auto is_distance = line_or_distance->AsDependenceDistance() != nullptr;

    auto a = is_distance ? scalar_evolution_.CreateConstant(1)
                         : line_or_distance->AsDependenceLine()->GetA();
    auto b = is_distance ? scalar_evolution_.CreateConstant(-1)
                         : line_or_distance->AsDependenceLine()->GetB();
    auto c =
        is_distance
            ? scalar_evolution_.SimplifyExpression(
                  scalar_evolution_.CreateNegation(
                      line_or_distance->AsDependenceDistance()->GetDistance()))
            : line_or_distance->AsDependenceLine()->GetC();

    auto x = point->GetSource();
    auto y = point->GetDestination();

    if (a->AsSEConstantNode() && b->AsSEConstantNode() &&
        c->AsSEConstantNode() && x->AsSEConstantNode() &&
        y->AsSEConstantNode()) {
      auto constant_a = a->AsSEConstantNode()->FoldToSingleValue();
      auto constant_b = b->AsSEConstantNode()->FoldToSingleValue();
      auto constant_c = c->AsSEConstantNode()->FoldToSingleValue();

      auto constant_x = x->AsSEConstantNode()->FoldToSingleValue();
      auto constant_y = y->AsSEConstantNode()->FoldToSingleValue();

      auto left_hand_side = constant_a * constant_x + constant_b * constant_y;

      if (left_hand_side == constant_c) {
        // Point is on line, return point
        return point_0 ? constraint_0 : constraint_1;
      } else {
        // Point not on line, report independence (empty constraint).
        return make_constraint<DependenceEmpty>();
      }

    } else {
      // Not constants, bail out.
      return make_constraint<DependenceNone>();
    }
  }

  return nullptr;
}

// Propagate constraints function as described in section 5 of Practical
// Dependence Testing, Goff, Kennedy, Tseng, 1991.
SubscriptPair LoopDependenceAnalysis::PropagateConstraints(
    const SubscriptPair& subscript_pair,
    const std::vector<Constraint*>& constraints) {
  SENode* new_first = subscript_pair.first;
  SENode* new_second = subscript_pair.second;

  for (auto& constraint : constraints) {
    // In the paper this is a[k]. We're extracting the coefficient ('a') of a
    // recurrent expression with respect to the loop 'k'.
    SENode* coefficient_of_recurrent =
        scalar_evolution_.GetCoefficientFromRecurrentTerm(
            new_first, constraint->GetLoop());

    // In the paper this is a'[k].
    SENode* coefficient_of_recurrent_prime =
        scalar_evolution_.GetCoefficientFromRecurrentTerm(
            new_second, constraint->GetLoop());

    if (constraint->GetType() == Constraint::Distance) {
      DependenceDistance* as_distance = constraint->AsDependenceDistance();

      // In the paper this is a[k]*d
      SENode* rhs = scalar_evolution_.CreateMultiplyNode(
          coefficient_of_recurrent, as_distance->GetDistance());

      // In the paper this is a[k] <- 0
      SENode* zeroed_coefficient =
          scalar_evolution_.BuildGraphWithoutRecurrentTerm(
              new_first, constraint->GetLoop());

      // In the paper this is e <- e - a[k]*d.
      new_first = scalar_evolution_.CreateSubtraction(zeroed_coefficient, rhs);
      new_first = scalar_evolution_.SimplifyExpression(new_first);

      // In the paper this is a'[k] - a[k].
      SENode* new_child = scalar_evolution_.SimplifyExpression(
          scalar_evolution_.CreateSubtraction(coefficient_of_recurrent_prime,
                                              coefficient_of_recurrent));

      // In the paper this is a'[k]'i[k].
      SERecurrentNode* prime_recurrent =
          scalar_evolution_.GetRecurrentTerm(new_second, constraint->GetLoop());

      if (!prime_recurrent) continue;

      // As we hash the nodes we need to create a new node when we update a
      // child.
      SENode* new_recurrent = scalar_evolution_.CreateRecurrentExpression(
          constraint->GetLoop(), prime_recurrent->GetOffset(), new_child);
      // In the paper this is a'[k] <- a'[k] - a[k].
      new_second = scalar_evolution_.UpdateChildNode(
          new_second, prime_recurrent, new_recurrent);
    }
  }

  new_second = scalar_evolution_.SimplifyExpression(new_second);
  return std::make_pair(new_first, new_second);
}

bool LoopDependenceAnalysis::DeltaTest(
    const std::vector<SubscriptPair>& coupled_subscripts,
    DistanceVector* dv_entry) {
  std::vector<Constraint*> constraints(loops_.size());

  std::vector<bool> loop_appeared(loops_.size());

  std::generate(std::begin(constraints), std::end(constraints),
                [this]() { return make_constraint<DependenceNone>(); });

  // Separate SIV and MIV subscripts
  std::vector<SubscriptPair> siv_subscripts{};
  std::vector<SubscriptPair> miv_subscripts{};

  for (const auto& subscript_pair : coupled_subscripts) {
    if (IsSIV(subscript_pair)) {
      siv_subscripts.push_back(subscript_pair);
    } else {
      miv_subscripts.push_back(subscript_pair);
    }
  }

  // Delta Test
  while (!siv_subscripts.empty()) {
    std::vector<bool> results(siv_subscripts.size());

    std::vector<DistanceVector> current_distances(
        siv_subscripts.size(), DistanceVector(loops_.size()));

    // Apply SIV test to all SIV subscripts, report independence if any of them
    // is independent
    std::transform(
        std::begin(siv_subscripts), std::end(siv_subscripts),
        std::begin(current_distances), std::begin(results),
        [this](SubscriptPair& p, DistanceVector& d) { return SIVTest(p, &d); });

    if (std::accumulate(std::begin(results), std::end(results), false,
                        std::logical_or<bool>{})) {
      return true;
    }

    // Derive new constraint vector.
    std::vector<std::pair<Constraint*, size_t>> all_new_constrants{};

    for (size_t i = 0; i < siv_subscripts.size(); ++i) {
      auto loop = GetLoopForSubscriptPair(siv_subscripts[i]);

      auto loop_id =
          std::distance(std::begin(loops_),
                        std::find(std::begin(loops_), std::end(loops_), loop));

      loop_appeared[loop_id] = true;
      auto distance_entry = current_distances[i].GetEntries()[loop_id];

      if (distance_entry.dependence_information ==
          DistanceEntry::DependenceInformation::DISTANCE) {
        // Construct a DependenceDistance.
        auto node = scalar_evolution_.CreateConstant(distance_entry.distance);

        all_new_constrants.push_back(
            {make_constraint<DependenceDistance>(node, loop), loop_id});
      } else {
        // Construct a DependenceLine.
        const auto& subscript_pair = siv_subscripts[i];
        SENode* source_node = std::get<0>(subscript_pair);
        SENode* destination_node = std::get<1>(subscript_pair);

        int64_t source_induction_count = CountInductionVariables(source_node);
        int64_t destination_induction_count =
            CountInductionVariables(destination_node);

        SENode* a = nullptr;
        SENode* b = nullptr;
        SENode* c = nullptr;

        if (destination_induction_count != 0) {
          a = destination_node->AsSERecurrentNode()->GetCoefficient();
          c = scalar_evolution_.CreateNegation(
              destination_node->AsSERecurrentNode()->GetOffset());
        } else {
          a = scalar_evolution_.CreateConstant(0);
          c = scalar_evolution_.CreateNegation(destination_node);
        }

        if (source_induction_count != 0) {
          b = scalar_evolution_.CreateNegation(
              source_node->AsSERecurrentNode()->GetCoefficient());
          c = scalar_evolution_.CreateAddNode(
              c, source_node->AsSERecurrentNode()->GetOffset());
        } else {
          b = scalar_evolution_.CreateConstant(0);
          c = scalar_evolution_.CreateAddNode(c, source_node);
        }

        a = scalar_evolution_.SimplifyExpression(a);
        b = scalar_evolution_.SimplifyExpression(b);
        c = scalar_evolution_.SimplifyExpression(c);

        all_new_constrants.push_back(
            {make_constraint<DependenceLine>(a, b, c, loop), loop_id});
      }
    }

    // Calculate the intersection between the new and existing constraints.
    std::vector<Constraint*> intersection = constraints;
    for (const auto& constraint_to_intersect : all_new_constrants) {
      auto loop_id = std::get<1>(constraint_to_intersect);
      auto loop = loops_[loop_id];
      intersection[loop_id] = IntersectConstraints(
          intersection[loop_id], std::get<0>(constraint_to_intersect),
          GetLowerBound(loop), GetUpperBound(loop));
    }

    // Report independence if an empty constraint (DependenceEmpty) is found.
    auto first_empty =
        std::find_if(std::begin(intersection), std::end(intersection),
                     [](Constraint* constraint) {
                       return constraint->AsDependenceEmpty() != nullptr;
                     });
    if (first_empty != std::end(intersection)) {
      return true;
    }
    std::vector<SubscriptPair> new_siv_subscripts{};
    std::vector<SubscriptPair> new_miv_subscripts{};

    auto equal =
        std::equal(std::begin(constraints), std::end(constraints),
                   std::begin(intersection),
                   [](Constraint* a, Constraint* b) { return *a == *b; });

    // If any constraints have changed, propagate them into the rest of the
    // subscripts possibly creating new ZIV/SIV subscripts.
    if (!equal) {
      std::vector<SubscriptPair> new_subscripts(miv_subscripts.size());

      // Propagate constraints into MIV subscripts
      std::transform(std::begin(miv_subscripts), std::end(miv_subscripts),
                     std::begin(new_subscripts),
                     [this, &intersection](SubscriptPair& subscript_pair) {
                       return PropagateConstraints(subscript_pair,
                                                   intersection);
                     });

      // If a ZIV subscript is returned, apply test, otherwise, update untested
      // subscripts.
      for (auto& subscript : new_subscripts) {
        if (IsZIV(subscript) && ZIVTest(subscript)) {
          return true;
        } else if (IsSIV(subscript)) {
          new_siv_subscripts.push_back(subscript);
        } else {
          new_miv_subscripts.push_back(subscript);
        }
      }
    }

    // Set new constraints and subscripts to test.
    std::swap(siv_subscripts, new_siv_subscripts);
    std::swap(miv_subscripts, new_miv_subscripts);
    std::swap(constraints, intersection);
  }

  // Create the dependence vector from the constraints.
  for (size_t i = 0; i < loops_.size(); ++i) {
    // Don't touch entries for loops that weren't tested.
    if (loop_appeared[i]) {
      auto current_constraint = constraints[i];
      auto& current_distance_entry = (*dv_entry).GetEntries()[i];

      if (auto dependence_distance =
              current_constraint->AsDependenceDistance()) {
        if (auto constant_node =
                dependence_distance->GetDistance()->AsSEConstantNode()) {
          current_distance_entry.dependence_information =
              DistanceEntry::DependenceInformation::DISTANCE;

          current_distance_entry.distance = constant_node->FoldToSingleValue();
          if (current_distance_entry.distance == 0) {
            current_distance_entry.direction = DistanceEntry::Directions::EQ;
          } else if (current_distance_entry.distance < 0) {
            current_distance_entry.direction = DistanceEntry::Directions::GT;
          } else {
            current_distance_entry.direction = DistanceEntry::Directions::LT;
          }
        }
      } else if (auto dependence_point =
                     current_constraint->AsDependencePoint()) {
        auto source = dependence_point->GetSource();
        auto destination = dependence_point->GetDestination();

        if (source->AsSEConstantNode() && destination->AsSEConstantNode()) {
          current_distance_entry = DistanceEntry(
              source->AsSEConstantNode()->FoldToSingleValue(),
              destination->AsSEConstantNode()->FoldToSingleValue());
        }
      }
    }
  }

  // Test any remaining MIV subscripts and report independence if found.
  std::vector<bool> results(miv_subscripts.size());

  std::transform(std::begin(miv_subscripts), std::end(miv_subscripts),
                 std::begin(results),
                 [this](const SubscriptPair& p) { return GCDMIVTest(p); });

  return std::accumulate(std::begin(results), std::end(results), false,
                         std::logical_or<bool>{});
}

}  // namespace opt
}  // namespace spvtools
