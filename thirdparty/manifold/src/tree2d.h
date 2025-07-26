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

#pragma once

#include "manifold/common.h"
#include "manifold/optional_assert.h"
#include "manifold/polygon.h"
#include "manifold/vec_view.h"

namespace manifold {

void BuildTwoDTreeImpl(VecView<PolyVert> points, bool sortX);

void BuildTwoDTree(VecView<PolyVert> points);

template <typename F>
void QueryTwoDTree(VecView<PolyVert> points, Rect r, F f) {
  if (points.size() <= 8) {
    for (const auto& p : points)
      if (r.Contains(p.pos)) f(p);
    return;
  }
  Rect current;
  current.min = vec2(-std::numeric_limits<double>::infinity());
  current.max = vec2(std::numeric_limits<double>::infinity());

  int level = 0;
  VecView<PolyVert> currentView = points;
  std::array<Rect, 64> rectStack;
  std::array<VecView<PolyVert>, 64> viewStack;
  std::array<int, 64> levelStack;
  int stackPointer = 0;

  while (1) {
    if (currentView.size() <= 8) {
      for (const auto& p : currentView)
        if (r.Contains(p.pos)) f(p);
      if (--stackPointer < 0) break;
      level = levelStack[stackPointer];
      currentView = viewStack[stackPointer];
      current = rectStack[stackPointer];
      continue;
    }

    // these are conceptual left/right trees
    Rect left = current;
    Rect right = current;
    const PolyVert middle = currentView[currentView.size() / 2];
    if (level % 2 == 0)
      left.max.x = right.min.x = middle.pos.x;
    else
      left.max.y = right.min.y = middle.pos.y;

    if (r.Contains(middle.pos)) f(middle);
    if (left.DoesOverlap(r)) {
      if (right.DoesOverlap(r)) {
        DEBUG_ASSERT(stackPointer < 64, logicErr, "Stack overflow");
        rectStack[stackPointer] = right;
        viewStack[stackPointer] = currentView.view(currentView.size() / 2 + 1);
        levelStack[stackPointer] = level + 1;
        stackPointer++;
      }
      current = left;
      currentView = currentView.view(0, currentView.size() / 2);
      level++;
    } else {
      current = right;
      currentView = currentView.view(currentView.size() / 2 + 1);
      level++;
    }
  }
}
}  // namespace manifold
