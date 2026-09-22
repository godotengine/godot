// Copyright 2025 The Manifold Authors.
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

#include "tree2d.h"

#include "parallel.h"

#ifndef ZoneScoped
#if __has_include(<tracy/Tracy.hpp>)
#include <tracy/Tracy.hpp>
#else
#define FrameMarkStart(x)
#define FrameMarkEnd(x)
// putting ZoneScoped in a function will instrument the function execution when
// TRACY_ENABLE is set, which allows the profiler to record more accurate
// timing.
#define ZoneScoped
#define ZoneScopedN(name)
#endif
#endif

namespace manifold {

// Not really a proper KD-tree, but a kd tree with k = 2 and alternating x/y
// partition.
// Recursive sorting is not the most efficient, but simple and guaranteed to
// result in a balanced tree.
void BuildTwoDTreeImpl(VecView<PolyVert> points, bool sortX) {
  auto cmpx = [](const PolyVert& a, const PolyVert& b) {
    return a.pos.x < b.pos.x;
  };
  auto cmpy = [](const PolyVert& a, const PolyVert& b) {
    return a.pos.y < b.pos.y;
  };
  if (sortX)
    manifold::stable_sort(points.begin(), points.end(), cmpx);
  else
    manifold::stable_sort(points.begin(), points.end(), cmpy);
  if (points.size() < 2) return;
  BuildTwoDTreeImpl(points.view(0, points.size() / 2), !sortX);
  BuildTwoDTreeImpl(points.view(points.size() / 2 + 1), !sortX);
}

void BuildTwoDTree(VecView<PolyVert> points) {
  ZoneScoped;
  // don't even bother...
  if (points.size() <= 8) return;
  BuildTwoDTreeImpl(points, true);
}
}  // namespace manifold
