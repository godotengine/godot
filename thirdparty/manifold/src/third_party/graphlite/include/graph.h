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

#pragma once
#include "graph_lite.h"

namespace manifold {

typedef typename graph_lite::Graph<
    int, void, void, graph_lite::EdgeDirection::UNDIRECTED,
    graph_lite::MultiEdge::DISALLOWED, graph_lite::SelfLoop::DISALLOWED,
    graph_lite::Map::UNORDERED_MAP, graph_lite::Container::VEC>
    Graph;

int ConnectedComponents(std::vector<int>& components, const Graph& g);
}  // namespace manifold