// Copyright 2026 The Manifold Authors.
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

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "manifold/polygon.h"
#include "shared.h"

namespace manifold {

struct HalfedgeTriangulation {
  std::vector<Halfedge> halfedges;
  size_t contourEnd = 0;
  double epsilon = -1;

  void AddContours(const PolygonsIdx& polys) {
    size_t numContourEdges = 0;
    for (const SimplePolygonIdx& poly : polys) numContourEdges += poly.size();
    halfedges.reserve(halfedges.size() + numContourEdges);
    edge2halfedge.reserve(edge2halfedge.size() + numContourEdges);
    for (const SimplePolygonIdx& poly : polys) {
      for (size_t i = 0; i < poly.size(); ++i) {
        const int start = poly[i].idx;
        const int end = poly[i + 1 < poly.size() ? i + 1 : 0].idx;
        // Store the exterior contour halfedge, opposite the filled contour.
        AddHalfedge(end, start);
      }
    }
    contourEnd = halfedges.size();
  }

  void ReserveTriangles(size_t numTri) {
    halfedges.reserve(contourEnd + 3 * numTri);
    edge2halfedge.reserve(edge2halfedge.size() + numTri);
  }

  void AddTriangle(int first, int second, int third) {
    AddHalfedge(first, second);
    AddHalfedge(second, third);
    AddHalfedge(third, first);
  }

  size_t NumTri() const { return (halfedges.size() - contourEnd) / 3; }

  std::vector<ivec3> Triangles() const {
    std::vector<ivec3> triangles;
    triangles.reserve(NumTri());
    for (size_t edge = contourEnd; edge < halfedges.size(); edge += 3) {
      triangles.push_back({halfedges[edge].startVert,
                           halfedges[edge + 1].startVert,
                           halfedges[edge + 2].startVert});
    }
    return triangles;
  }

  void Finalize() {
#ifdef MANIFOLD_DEBUG
    DEBUG_ASSERT(edge2halfedge.empty(), topologyErr,
                 "triangulation has unpaired halfedges");
    for (size_t i = 0; i < halfedges.size(); ++i) {
      const int pair = halfedges[i].pairedHalfedge;
      DEBUG_ASSERT(pair >= 0 && pair < static_cast<int>(halfedges.size()),
                   topologyErr, "invalid paired halfedge");
      DEBUG_ASSERT(halfedges[pair].pairedHalfedge == static_cast<int>(i),
                   topologyErr, "halfedge pair is not reciprocal");
      DEBUG_ASSERT(halfedges[i].startVert == halfedges[pair].endVert &&
                       halfedges[i].endVert == halfedges[pair].startVert,
                   topologyErr, "halfedge pair endpoints do not match");
    }
#endif
    edge2halfedge.clear();
    edge2halfedge.rehash(0);
  }

 private:
  std::unordered_map<uint64_t, std::vector<int>> edge2halfedge;

  static uint64_t EdgeKey(int start, int end) {
    return (uint64_t{static_cast<uint32_t>(start)} << 32) |
           static_cast<uint32_t>(end);
  }

  void AddHalfedge(int start, int end) {
    const int halfedge = halfedges.size();
    Halfedge data = {start, end, -1, -1};
    auto reverse = edge2halfedge.find(EdgeKey(end, start));
    if (reverse != edge2halfedge.end() && !reverse->second.empty()) {
      data.pairedHalfedge = reverse->second.back();
      halfedges[data.pairedHalfedge].pairedHalfedge = halfedge;
      reverse->second.pop_back();
      if (reverse->second.empty()) edge2halfedge.erase(reverse);
    } else {
      edge2halfedge[EdgeKey(start, end)].push_back(halfedge);
    }
    halfedges.push_back(data);
  }
};

HalfedgeTriangulation TriangulateIdxHalfedges(const PolygonsIdx& polys,
                                              double epsilon = -1,
                                              bool allowConvex = true);

}  // namespace manifold
