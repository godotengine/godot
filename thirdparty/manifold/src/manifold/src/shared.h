// Copyright 2021 The Manifold Authors.
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

#include "par.h"
#include "sparse.h"
#include "utils.h"
#include "vec.h"

namespace manifold {

/** @addtogroup Private
 *  @{
 */
inline glm::vec3 SafeNormalize(glm::vec3 v) {
  v = glm::normalize(v);
  return glm::isfinite(v.x) ? v : glm::vec3(0);
}

inline float MaxPrecision(float minPrecision, const Box& bBox) {
  float precision = glm::max(minPrecision, kTolerance * bBox.Scale());
  return glm::isfinite(precision) ? precision : -1;
}

inline int NextHalfedge(int current) {
  ++current;
  if (current % 3 == 0) current -= 3;
  return current;
}

inline glm::mat3 NormalTransform(const glm::mat4x3& transform) {
  return glm::inverse(glm::transpose(glm::mat3(transform)));
}

/**
 * By using the closest axis-aligned projection to the normal instead of a
 * projection along the normal, we avoid introducing any rounding error.
 */
inline glm::mat3x2 GetAxisAlignedProjection(glm::vec3 normal) {
  glm::vec3 absNormal = glm::abs(normal);
  float xyzMax;
  glm::mat2x3 projection;
  if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
    projection = glm::mat2x3(1.0f, 0.0f, 0.0f,  //
                             0.0f, 1.0f, 0.0f);
    xyzMax = normal.z;
  } else if (absNormal.y > absNormal.x) {
    projection = glm::mat2x3(0.0f, 0.0f, 1.0f,  //
                             1.0f, 0.0f, 0.0f);
    xyzMax = normal.y;
  } else {
    projection = glm::mat2x3(0.0f, 1.0f, 0.0f,  //
                             0.0f, 0.0f, 1.0f);
    xyzMax = normal.x;
  }
  if (xyzMax < 0) projection[0] *= -1.0f;
  return glm::transpose(projection);
}

inline glm::vec3 GetBarycentric(const glm::vec3& v, const glm::mat3& triPos,
                                float precision) {
  const glm::mat3 edges(triPos[2] - triPos[1], triPos[0] - triPos[2],
                        triPos[1] - triPos[0]);
  const glm::vec3 d2(glm::dot(edges[0], edges[0]), glm::dot(edges[1], edges[1]),
                     glm::dot(edges[2], edges[2]));
  const int longSide = d2[0] > d2[1] && d2[0] > d2[2] ? 0
                       : d2[1] > d2[2]                ? 1
                                                      : 2;
  const glm::vec3 crossP = glm::cross(edges[0], edges[1]);
  const float area2 = glm::dot(crossP, crossP);
  const float tol2 = precision * precision;

  glm::vec3 uvw(0);
  for (const int i : {0, 1, 2}) {
    const glm::vec3 dv = v - triPos[i];
    if (glm::dot(dv, dv) < tol2) {
      // Return exactly equal if within tolerance of vert.
      uvw[i] = 1;
      return uvw;
    }
  }

  if (d2[longSide] < tol2) {  // point
    return glm::vec3(1, 0, 0);
  } else if (area2 > d2[longSide] * tol2) {  // triangle
    for (const int i : {0, 1, 2}) {
      const int j = Next3(i);
      const glm::vec3 crossPv = glm::cross(edges[i], v - triPos[j]);
      const float area2v = glm::dot(crossPv, crossPv);
      // Return exactly equal if within tolerance of edge.
      uvw[i] = area2v < d2[i] * tol2 ? 0 : glm::dot(crossPv, crossP);
    }
    uvw /= (uvw[0] + uvw[1] + uvw[2]);
    return uvw;
  } else {  // line
    const int nextV = Next3(longSide);
    const float alpha =
        glm::dot(v - triPos[nextV], edges[longSide]) / d2[longSide];
    uvw[longSide] = 0;
    uvw[nextV] = 1 - alpha;
    const int lastV = Next3(nextV);
    uvw[lastV] = alpha;
    return uvw;
  }
}

/**
 * The fundamental component of the halfedge data structure used for storing and
 * operating on the Manifold.
 */
struct Halfedge {
  int startVert, endVert;
  int pairedHalfedge;
  int face;
  bool IsForward() const { return startVert < endVert; }
  bool operator<(const Halfedge& other) const {
    return startVert == other.startVert ? endVert < other.endVert
                                        : startVert < other.startVert;
  }
};

struct Barycentric {
  int tri;
  glm::vec4 uvw;
};

struct TriRef {
  /// The unique ID of the mesh instance of this triangle. If .meshID and .tri
  /// match for two triangles, then they are coplanar and came from the same
  /// face.
  int meshID;
  /// The OriginalID of the mesh this triangle came from. This ID is ideal for
  /// reapplying properties like UV coordinates to the output mesh.
  int originalID;
  /// The triangle index of the original triangle this was part of:
  /// Mesh.triVerts[tri].
  int tri;

  bool SameFace(const TriRef& other) const {
    return meshID == other.meshID && tri == other.tri;
  }
};

/**
 * This is a temporary edge structure which only stores edges forward and
 * references the halfedge it was created from.
 */
struct TmpEdge {
  int first, second, halfedgeIdx;

  TmpEdge() {}
  TmpEdge(int start, int end, int idx) {
    first = glm::min(start, end);
    second = glm::max(start, end);
    halfedgeIdx = idx;
  }

  bool operator<(const TmpEdge& other) const {
    return first == other.first ? second < other.second : first < other.first;
  }
};
/** @} */

struct Halfedge2Tmp {
  void operator()(thrust::tuple<TmpEdge&, const Halfedge&, int> inout) {
    const Halfedge& halfedge = thrust::get<1>(inout);
    int idx = thrust::get<2>(inout);
    if (!halfedge.IsForward()) idx = -1;

    thrust::get<0>(inout) = TmpEdge(halfedge.startVert, halfedge.endVert, idx);
  }
};

struct TmpInvalid {
  bool operator()(const TmpEdge& edge) { return edge.halfedgeIdx < 0; }
};

Vec<TmpEdge> inline CreateTmpEdges(const Vec<Halfedge>& halfedge) {
  Vec<TmpEdge> edges(halfedge.size());
  for_each_n(autoPolicy(edges.size()),
             zip(edges.begin(), halfedge.begin(), countAt(0)), edges.size(),
             Halfedge2Tmp());
  int numEdge =
      remove_if<decltype(edges.begin())>(
          autoPolicy(edges.size()), edges.begin(), edges.end(), TmpInvalid()) -
      edges.begin();
  ASSERT(numEdge == halfedge.size() / 2, topologyErr, "Not oriented!");
  edges.resize(numEdge);
  return edges;
}

template <const bool inverted>
struct ReindexEdge {
  VecView<const TmpEdge> edges;
  SparseIndices& indices;

  void operator()(size_t i) {
    int& edge = indices.Get(i, inverted);
    edge = edges[edge].halfedgeIdx;
  }
};

#ifdef MANIFOLD_DEBUG
inline std::ostream& operator<<(std::ostream& stream, const Halfedge& edge) {
  return stream << "startVert = " << edge.startVert
                << ", endVert = " << edge.endVert
                << ", pairedHalfedge = " << edge.pairedHalfedge
                << ", face = " << edge.face;
}

inline std::ostream& operator<<(std::ostream& stream, const Barycentric& bary) {
  return stream << "tri = " << bary.tri << ", uvw = " << bary.uvw;
}

inline std::ostream& operator<<(std::ostream& stream, const TriRef& ref) {
  return stream << "meshID: " << ref.meshID
                << ", originalID: " << ref.originalID << ", tri: " << ref.tri;
}
#endif
}  // namespace manifold
