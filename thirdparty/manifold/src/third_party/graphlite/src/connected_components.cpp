// Copyright 2022 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <deque>

#include "graph.h"

namespace manifold {

int ConnectedComponents(std::vector<int>& components, const Graph& graph) {
  if (!graph.size()) {
    return 0;
  }
  components.resize(graph.size());
  std::fill(components.begin(), components.end(), -1);

  std::deque<int> queue;
  int numComponent = 0;
  for (auto it = graph.begin(); it != graph.end(); ++it) {
    const int& root = *it;
    if (components[root] >= 0) continue;  // skip visited nodes

    // new component
    components[root] = numComponent;
    queue.emplace_back(root);
    // traverse all connected nodes
    while (!queue.empty()) {
      const auto [n_begin, n_end] = graph.neighbors(queue.front());
      queue.pop_front();
      for (auto n_it = n_begin; n_it != n_end; ++n_it) {
        const int& neighbor = *n_it;
        if (components[neighbor] < 0) {  // unvisited
          components[neighbor] = numComponent;
          queue.emplace_back(neighbor);
        }
      }
    }
    ++numComponent;
  }
  return numComponent;
}
}  // namespace manifold