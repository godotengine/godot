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

#include "manifold/parallel.h"
#include "manifold/sparse.h"
#include "manifold/utils.h"
#include "manifold/vec.h"

namespace manifold {

/** @addtogroup Private
 *  @{
 */
inline vec3 SafeNormalize(vec3 v) {
  v = glm::normalize(v);
  return std::isfinite(v.x) ? v : vec3(0);
}

inline double MaxPrecision(double minPrecision, const Box& bBox) {
  double precision = std::max(minPrecision, kTolerance * bBox.Scale());
  return std::isfinite(precision) ? precision : -1;
}

inline int NextHalfedge(int current) {
  ++current;
  if (current % 3 == 0) current -= 3;
  return current;
}

inline mat3 NormalTransform(const mat4x3& transform) {
  return glm::inverse(glm::transpose(mat3(transform)));
}

/**
 * By using the closest axis-aligned projection to the normal instead of a
 * projection along the normal, we avoid introducing any rounding error.
 */
inline mat3x2 GetAxisAlignedProjection(vec3 normal) {
  vec3 absNormal = glm::abs(normal);
  double xyzMax;
  mat2x3 projection;
  if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
    projection = mat2x3(1.0, 0.0, 0.0,  //
                        0.0, 1.0, 0.0);
    xyzMax = normal.z;
  } else if (absNormal.y > absNormal.x) {
    projection = mat2x3(0.0, 0.0, 1.0,  //
                        1.0, 0.0, 0.0);
    xyzMax = normal.y;
  } else {
    projection = mat2x3(0.0, 1.0, 0.0,  //
                        0.0, 0.0, 1.0);
    xyzMax = normal.x;
  }
  if (xyzMax < 0) projection[0] *= -1.0;
  return glm::transpose(projection);
}

inline vec3 GetBarycentric(const vec3& v, const mat3& triPos,
                           double precision) {
  const mat3 edges(triPos[2] - triPos[1], triPos[0] - triPos[2],
                   triPos[1] - triPos[0]);
  const vec3 d2(glm::dot(edges[0], edges[0]), glm::dot(edges[1], edges[1]),
                glm::dot(edges[2], edges[2]));
  const int longSide = d2[0] > d2[1] && d2[0] > d2[2] ? 0
                       : d2[1] > d2[2]                ? 1
                                                      : 2;
  const vec3 crossP = glm::cross(edges[0], edges[1]);
  const double area2 = glm::dot(crossP, crossP);
  const double tol2 = precision * precision;

  vec3 uvw(0);
  for (const int i : {0, 1, 2}) {
    const vec3 dv = v - triPos[i];
    if (glm::dot(dv, dv) < tol2) {
      // Return exactly equal if within tolerance of vert.
      uvw[i] = 1;
      return uvw;
    }
  }

  if (d2[longSide] < tol2) {  // point
    return vec3(1, 0, 0);
  } else if (area2 > d2[longSide] * tol2) {  // triangle
    for (const int i : {0, 1, 2}) {
      const int j = Next3(i);
      const vec3 crossPv = glm::cross(edges[i], v - triPos[j]);
      const double area2v = glm::dot(crossPv, crossPv);
      // Return exactly equal if within tolerance of edge.
      uvw[i] = area2v < d2[i] * tol2 ? 0 : glm::dot(crossPv, crossP);
    }
    uvw /= (uvw[0] + uvw[1] + uvw[2]);
    return uvw;
  } else {  // line
    const int nextV = Next3(longSide);
    const double alpha =
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
  vec4 uvw;
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
    first = std::min(start, end);
    second = std::max(start, end);
    halfedgeIdx = idx;
  }

  bool operator<(const TmpEdge& other) const {
    return first == other.first ? second < other.second : first < other.first;
  }
};
/** @} */

Vec<TmpEdge> inline CreateTmpEdges(const Vec<Halfedge>& halfedge) {
  Vec<TmpEdge> edges(halfedge.size());
  for_each_n(autoPolicy(edges.size()), countAt(0), edges.size(),
             [&edges, &halfedge](const int idx) {
               const Halfedge& half = halfedge[idx];
               edges[idx] = TmpEdge(half.startVert, half.endVert,
                                    half.IsForward() ? idx : -1);
             });

  size_t numEdge =
      remove_if(edges.begin(), edges.end(),
                [](const TmpEdge& edge) { return edge.halfedgeIdx < 0; }) -
      edges.begin();
  DEBUG_ASSERT(numEdge == halfedge.size() / 2, topologyErr, "Not oriented!");
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
                << ", pairedHalfedge = " << edge.pairedHalfedge;
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
