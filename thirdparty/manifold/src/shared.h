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

#include "parallel.h"
#include "utils.h"
#include "vec.h"

namespace manifold {

inline vec3 SafeNormalize(vec3 v) {
  v = la::normalize(v);
  return std::isfinite(v.x) ? v : vec3(0.0);
}

inline double MaxEpsilon(double minEpsilon, const Box& bBox) {
  double epsilon = std::max(minEpsilon, kPrecision * bBox.Scale());
  return std::isfinite(epsilon) ? epsilon : -1;
}

inline int NextHalfedge(int current) {
  current += current % 3 == 2 ? -2 : 1;
  return current;
}

inline int PrevHalfedge(int current) {
  current += current % 3 == 0 ? 2 : -1;
  return current;
}

/**
When a transform is applied to the verts of an object, and different transform
is needed for the normals. Note that applying this transform will stretch their
length, so they should be renormalized after.
*/
inline mat3 NormalTransform(const mat3x4& transform) {
  return la::inverse(la::transpose(mat3(transform)));
}

/**
This is the corresponding normal transform for the inverse of the input
transform.
*/
inline mat3 InverseNormalTransform(const mat3x4& transform) {
  return la::inverse(la::transpose(la::inverse(mat3(transform))));
}

/**
 * Symbolic perturbation primitives shared by Boolean3 and Boolean2.
 * Carefully designed to minimize FP rounding error and eliminate it at edge
 * cases.
 */

inline double withSign(bool pos, double v) { return pos ? v : -v; }

/**
 * Interpolate the (y, z) of segment aL-aR at the given x. The choice of
 * (x - aL) vs (x - aR) is the smaller in magnitude, which keeps FP error
 * low near either endpoint. Domain check via DEBUG_ASSERT.
 */
inline vec2 Interpolate(vec3 aL, vec3 aR, double x) {
  const double dxL = x - aL.x;
  const double dxR = x - aR.x;
  DEBUG_ASSERT(dxL * dxR <= 0, logicErr,
               "Boolean manifold error: not in domain");
  const bool useL = fabs(dxL) < fabs(dxR);
  const vec3 dLR = aR - aL;
  const double lambda = (useL ? dxL : dxR) / dLR.x;
  if (!std::isfinite(lambda) || !std::isfinite(dLR.y) || !std::isfinite(dLR.z))
    return vec2(aL.y, aL.z);
  vec2 yz;
  yz[0] = lambda * dLR.y + (useL ? aL.y : aR.y);
  yz[1] = lambda * dLR.z + (useL ? aL.z : aR.z);
  return yz;
}

/**
 * `p < q` with symbolic perturbation: when `p == q` exactly, `dir < 0`
 * acts as the tiebreaker. Used to give consistent strict-ordering answers
 * regardless of which side of an FP equality we land on.
 */
inline bool Shadows(double p, double q, double dir) {
  return p == q ? dir < 0 : p < q;
}

/**
 * By using the closest axis-aligned projection to the normal instead of a
 * projection along the normal, we avoid introducing any rounding error.
 */
inline mat2x3 GetAxisAlignedProjection(vec3 normal) {
  vec3 absNormal = la::abs(normal);
  double xyzMax;
  mat3x2 projection;
  if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
    projection = mat3x2({1.0, 0.0, 0.0},  //
                        {0.0, 1.0, 0.0});
    xyzMax = normal.z;
  } else if (absNormal.y > absNormal.x) {
    projection = mat3x2({0.0, 0.0, 1.0},  //
                        {1.0, 0.0, 0.0});
    xyzMax = normal.y;
  } else {
    projection = mat3x2({0.0, 1.0, 0.0},  //
                        {0.0, 0.0, 1.0});
    xyzMax = normal.x;
  }
  if (xyzMax < 0) projection[0] *= -1.0;
  return la::transpose(projection);
}

inline vec3 GetBarycentric(const vec3& v, const mat3& triPos,
                           double tolerance) {
  const mat3 edges(triPos[2] - triPos[1], triPos[0] - triPos[2],
                   triPos[1] - triPos[0]);
  const vec3 d2(la::dot(edges[0], edges[0]), la::dot(edges[1], edges[1]),
                la::dot(edges[2], edges[2]));
  const int longSide = d2[0] > d2[1] && d2[0] > d2[2] ? 0
                       : d2[1] > d2[2]                ? 1
                                                      : 2;
  const vec3 crossP = la::cross(edges[0], edges[1]);
  const double area2 = la::dot(crossP, crossP);
  const double tol2 = tolerance * tolerance;

  vec3 uvw(0.0);
  for (const int i : {0, 1, 2}) {
    const vec3 dv = v - triPos[i];
    if (la::dot(dv, dv) < tol2) {
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
      const vec3 crossPv = la::cross(edges[i], v - triPos[j]);
      const double area2v = la::dot(crossPv, crossPv);
      // Return exactly equal if within tolerance of edge.
      uvw[i] = area2v < d2[i] * tol2 ? 0 : la::dot(crossPv, crossP);
    }
    uvw /= (uvw[0] + uvw[1] + uvw[2]);
    return uvw;
  } else {  // line
    const int nextV = Next3(longSide);
    const double alpha =
        la::dot(v - triPos[nextV], edges[longSide]) / d2[longSide];
    uvw[longSide] = 0;
    uvw[nextV] = 1 - alpha;
    const int lastV = Next3(nextV);
    uvw[lastV] = alpha;
    return uvw;
  }
}

/**
 * Temporary or value-style halfedge record. Persistent Manifold storage uses
 * Halfedges below, which derives endVert from the next halfedge in each face.
 */
struct Halfedge {
  int startVert, endVert;
  int pairedHalfedge;
  int propVert;
  bool IsForward() const { return startVert < endVert; }
  bool operator<(const Halfedge& other) const {
    return startVert == other.startVert ? endVert < other.endVert
                                        : startVert < other.startVert;
  }
};

class Halfedges {
 public:
  Halfedges() = default;
  explicit Halfedges(size_t size) { resize_nofill(size); }
  explicit Halfedges(const VecView<const Halfedge>& edges) { FromData(edges); }

  size_t size() const { return start_.size(); }
  bool empty() const { return start_.empty(); }

  int Start(int idx) const { return start_[idx]; }
  int End(int idx) const { return start_[NextHalfedge(idx)]; }
  int Pair(int idx) const { return paired_[idx]; }
  int Prop(int idx) const { return propVert_[idx]; }

  void SetStart(int idx, int vert) { start_[idx] = vert; }
  void SetEnd(int idx, int vert) { start_[NextHalfedge(idx)] = vert; }
  void SetPair(int idx, int pair) { paired_[idx] = pair; }
  void SetProp(int idx, int prop) { propVert_[idx] = prop; }

  bool IsForward(int idx) const { return Start(idx) < End(idx); }

  Halfedge Get(int idx) const {
    return {Start(idx), End(idx), Pair(idx), Prop(idx)};
  }

  void Set(int idx, int startVert, int pairedHalfedge, int propVert) {
    SetStart(idx, startVert);
    SetPair(idx, pairedHalfedge);
    SetProp(idx, propVert);
  }

  void push_back(int startVert, int pairedHalfedge, int propVert) {
    start_.push_back(startVert);
    paired_.push_back(pairedHalfedge);
    propVert_.push_back(propVert);
  }

  void resize(size_t newSize) {
    start_.resize(newSize, -1);
    paired_.resize(newSize, -1);
    propVert_.resize(newSize, -1);
  }

  void resize_nofill(size_t newSize) {
    start_.resize_nofill(newSize);
    paired_.resize_nofill(newSize);
    propVert_.resize_nofill(newSize);
  }

  void clear(bool shrink = true) {
    start_.clear(shrink);
    paired_.clear(shrink);
    propVert_.clear(shrink);
  }

  void MakeUnique() {
    start_.MakeUnique();
    paired_.MakeUnique();
    propVert_.MakeUnique();
  }

  Vec<Halfedge> ToData() const {
    Vec<Halfedge> data(size());
    for_each_n(autoPolicy(size()), countAt(0), size(),
               [this, &data](int idx) { data[idx] = Get(idx); });
    return data;
  }

  void FromData(const VecView<const Halfedge>& data) {
    clear(true);
    resize_nofill(data.size());
    for_each_n(autoPolicy(data.size()), countAt(0), data.size(),
               [this, &data](int idx) {
                 start_[idx] = data[idx].startVert;
                 paired_[idx] = data[idx].pairedHalfedge;
                 propVert_[idx] = data[idx].propVert;
               });
#ifdef MANIFOLD_DEBUG
    DEBUG_ASSERT(data.size() % 3 == 0, topologyErr,
                 "Halfedges::FromData requires triangle faces!");
    for (size_t idx = 0; idx < data.size(); ++idx) {
      const int next = NextHalfedge(static_cast<int>(idx));
      if (data[idx].endVert >= 0 && data[next].startVert >= 0) {
        DEBUG_ASSERT(data[idx].endVert == data[next].startVert, topologyErr,
                     "Halfedges::FromData requires triangle-ordered edges!");
      }
    }
#endif
  }

  SharedVec<int> start_;
  SharedVec<int> paired_;
  SharedVec<int> propVert_;
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
  /// If set as an input of MeshGL, it is passed along unchanged. This is how
  /// the user can tell us not to collapse certain edges: those that divide
  /// difference faceIDs. If not set, this is always -1.
  int faceID;
  /// Triangles with the same coplanar ID are coplanar. Starts as a canonical
  /// triangle index, but after boolean operations it may refer to a triangle
  /// that is no longer present in this mesh.
  int coplanarID;

  bool SameFace(const TriRef& other) const {
    return meshID == other.meshID && coplanarID == other.coplanarID &&
           faceID == other.faceID;
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

Vec<TmpEdge> inline CreateTmpEdges(const Halfedges& halfedge) {
  Vec<TmpEdge> edges(halfedge.size());
  for_each_n(autoPolicy(edges.size()), countAt(0), edges.size(),
             [&edges, &halfedge](const int idx) {
               edges[idx] = TmpEdge(halfedge.Start(idx), halfedge.End(idx),
                                    halfedge.IsForward(idx) ? idx : -1);
             });

  size_t numEdge =
      remove_if(edges.begin(), edges.end(),
                [](const TmpEdge& edge) { return edge.halfedgeIdx < 0; }) -
      edges.begin();
  DEBUG_ASSERT(numEdge == halfedge.size() / 2, topologyErr, "Not oriented!");
  edges.resize(numEdge);
  return edges;
}

#ifdef MANIFOLD_DEBUG
inline std::ostream& operator<<(std::ostream& stream, const Halfedge& edge) {
  return stream << "startVert = " << edge.startVert
                << ", endVert = " << edge.endVert
                << ", pairedHalfedge = " << edge.pairedHalfedge
                << ", propVert = " << edge.propVert;
}

inline std::ostream& operator<<(std::ostream& stream, const Barycentric& bary) {
  return stream << "tri = " << bary.tri << ", uvw = " << bary.uvw;
}

inline std::ostream& operator<<(std::ostream& stream, const TriRef& ref) {
  return stream << "meshID: " << ref.meshID
                << ", originalID: " << ref.originalID
                << ", faceID: " << ref.faceID
                << ", coplanarID: " << ref.coplanarID;
}
#endif
}  // namespace manifold
