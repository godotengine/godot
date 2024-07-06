// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/call_graph.h"

#include <queue>

namespace spvtools {
namespace fuzz {

CallGraph::CallGraph(opt::IRContext* context) {
  // Initialize function in-degree, call graph edges and corresponding maximum
  // loop nesting depth to 0, empty and 0 respectively.
  for (auto& function : *context->module()) {
    function_in_degree_[function.result_id()] = 0;
    call_graph_edges_[function.result_id()] = std::set<uint32_t>();
    function_max_loop_nesting_depth_[function.result_id()] = 0;
  }

  // Record the maximum loop nesting depth for each edge, by keeping a map from
  // pairs of function ids, where (A, B) represents a function call from A to B,
  // to the corresponding maximum depth.
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> call_to_max_depth;

  // Compute |function_in_degree_|, |call_graph_edges_| and |call_to_max_depth|.
  BuildGraphAndGetDepthOfFunctionCalls(context, &call_to_max_depth);

  // Compute |functions_in_topological_order_|.
  ComputeTopologicalOrderOfFunctions();

  // Compute |function_max_loop_nesting_depth_|.
  ComputeInterproceduralFunctionCallDepths(call_to_max_depth);
}

void CallGraph::BuildGraphAndGetDepthOfFunctionCalls(
    opt::IRContext* context,
    std::map<std::pair<uint32_t, uint32_t>, uint32_t>* call_to_max_depth) {
  // Consider every function.
  for (auto& function : *context->module()) {
    // Avoid considering the same callee of this function multiple times by
    // recording known callees.
    std::set<uint32_t> known_callees;
    // Consider every function call instruction in every block.
    for (auto& block : function) {
      for (auto& instruction : block) {
        if (instruction.opcode() != spv::Op::OpFunctionCall) {
          continue;
        }
        // Get the id of the function being called.
        uint32_t callee = instruction.GetSingleWordInOperand(0);

        // Get the loop nesting depth of this function call.
        uint32_t loop_nesting_depth =
            context->GetStructuredCFGAnalysis()->LoopNestingDepth(block.id());
        // If inside a loop header, consider the function call nested inside the
        // loop headed by the block.
        if (block.IsLoopHeader()) {
          loop_nesting_depth++;
        }

        // Update the map if we have not seen this pair (caller, callee)
        // before or if this function call is from a greater depth.
        if (!known_callees.count(callee) ||
            call_to_max_depth->at({function.result_id(), callee}) <
                loop_nesting_depth) {
          call_to_max_depth->insert(
              {{function.result_id(), callee}, loop_nesting_depth});
        }

        if (known_callees.count(callee)) {
          // We have already considered a call to this function - ignore it.
          continue;
        }
        // Increase the callee's in-degree and add an edge to the call graph.
        function_in_degree_[callee]++;
        call_graph_edges_[function.result_id()].insert(callee);
        // Mark the callee as 'known'.
        known_callees.insert(callee);
      }
    }
  }
}

void CallGraph::ComputeTopologicalOrderOfFunctions() {
  // This is an implementation of Kahn’s algorithm for topological sorting.

  // Initialise |functions_in_topological_order_|.
  functions_in_topological_order_.clear();

  // Get a copy of the initial in-degrees of all functions.  The algorithm
  // involves decrementing these values, hence why we work on a copy.
  std::map<uint32_t, uint32_t> function_in_degree = GetFunctionInDegree();

  // Populate a queue with all those function ids with in-degree zero.
  std::queue<uint32_t> queue;
  for (auto& entry : function_in_degree) {
    if (entry.second == 0) {
      queue.push(entry.first);
    }
  }

  // Pop ids from the queue, adding them to the sorted order and decreasing the
  // in-degrees of their successors.  A successor who's in-degree becomes zero
  // gets added to the queue.
  while (!queue.empty()) {
    auto next = queue.front();
    queue.pop();
    functions_in_topological_order_.push_back(next);
    for (auto successor : GetDirectCallees(next)) {
      assert(function_in_degree.at(successor) > 0 &&
             "The in-degree cannot be zero if the function is a successor.");
      function_in_degree[successor] = function_in_degree.at(successor) - 1;
      if (function_in_degree.at(successor) == 0) {
        queue.push(successor);
      }
    }
  }

  assert(functions_in_topological_order_.size() == function_in_degree.size() &&
         "Every function should appear in the sort.");

  return;
}

void CallGraph::ComputeInterproceduralFunctionCallDepths(
    const std::map<std::pair<uint32_t, uint32_t>, uint32_t>&
        call_to_max_depth) {
  // Find the maximum loop nesting depth that each function can be
  // called from, by considering them in topological order.
  for (uint32_t function_id : functions_in_topological_order_) {
    const auto& callees = call_graph_edges_[function_id];

    // For each callee, update its maximum loop nesting depth, if a call from
    // |function_id| increases it.
    for (uint32_t callee : callees) {
      uint32_t max_depth_from_this_function =
          function_max_loop_nesting_depth_[function_id] +
          call_to_max_depth.at({function_id, callee});
      if (function_max_loop_nesting_depth_[callee] <
          max_depth_from_this_function) {
        function_max_loop_nesting_depth_[callee] = max_depth_from_this_function;
      }
    }
  }
}

void CallGraph::PushDirectCallees(uint32_t function_id,
                                  std::queue<uint32_t>* queue) const {
  for (auto callee : GetDirectCallees(function_id)) {
    queue->push(callee);
  }
}

std::set<uint32_t> CallGraph::GetIndirectCallees(uint32_t function_id) const {
  std::set<uint32_t> result;
  std::queue<uint32_t> queue;
  PushDirectCallees(function_id, &queue);

  while (!queue.empty()) {
    auto next = queue.front();
    queue.pop();
    if (result.count(next)) {
      continue;
    }
    result.insert(next);
    PushDirectCallees(next, &queue);
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
