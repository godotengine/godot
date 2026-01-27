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

#include "impl.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <optional>

#include "csg_tree.h"
#include "disjoint_sets.h"
#include "hashtable.h"
#include "manifold/optional_assert.h"
#include "mesh_fixes.h"
#include "parallel.h"
#include "shared.h"
#include "svd.h"

namespace {
using namespace manifold;

/**
 * Returns arc cosine of 𝑥.
 *
 * @return value in range [0,M_PI]
 * @return NAN if 𝑥 ∈ {NAN,+INFINITY,-INFINITY}
 * @return NAN if 𝑥 ∉ [-1,1]
 */
double sun_acos(double x) {
  /*
   * Origin of acos function: FreeBSD /usr/src/lib/msun/src/e_acos.c
   * Changed the use of union to memcpy to avoid undefined behavior.
   * ====================================================
   * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
   *
   * Developed at SunSoft, a Sun Microsystems, Inc. business.
   * Permission to use, copy, modify, and distribute this
   * software is freely granted, provided that this notice
   * is preserved.
   * ====================================================
   */
  constexpr double pio2_hi =
      1.57079632679489655800e+00; /* 0x3FF921FB, 0x54442D18 */
  constexpr double pio2_lo =
      6.12323399573676603587e-17; /* 0x3C91A626, 0x33145C07 */
  constexpr double pS0 =
      1.66666666666666657415e-01; /* 0x3FC55555, 0x55555555 */
  constexpr double pS1 =
      -3.25565818622400915405e-01; /* 0xBFD4D612, 0x03EB6F7D */
  constexpr double pS2 =
      2.01212532134862925881e-01; /* 0x3FC9C155, 0x0E884455 */
  constexpr double pS3 =
      -4.00555345006794114027e-02; /* 0xBFA48228, 0xB5688F3B */
  constexpr double pS4 =
      7.91534994289814532176e-04; /* 0x3F49EFE0, 0x7501B288 */
  constexpr double pS5 =
      3.47933107596021167570e-05; /* 0x3F023DE1, 0x0DFDF709 */
  constexpr double qS1 =
      -2.40339491173441421878e+00; /* 0xC0033A27, 0x1C8A2D4B */
  constexpr double qS2 =
      2.02094576023350569471e+00; /* 0x40002AE5, 0x9C598AC8 */
  constexpr double qS3 =
      -6.88283971605453293030e-01; /* 0xBFE6066C, 0x1B8D0159 */
  constexpr double qS4 =
      7.70381505559019352791e-02; /* 0x3FB3B8C5, 0xB12E9282 */
  auto R = [=](double z) {
    double p, q;
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    return p / q;
  };
  double z, w, s, c, df;
  uint64_t xx;
  uint32_t hx, lx, ix;
  memcpy(&xx, &x, sizeof(xx));
  hx = xx >> 32;
  ix = hx & 0x7fffffff;
  /* |x| >= 1 or nan */
  if (ix >= 0x3ff00000) {
    lx = xx;
    if (((ix - 0x3ff00000) | lx) == 0) {
      /* acos(1)=0, acos(-1)=pi */
      if (hx >> 31) return 2 * pio2_hi + 0x1p-120f;
      return 0;
    }
    return 0 / (x - x);
  }
  /* |x| < 0.5 */
  if (ix < 0x3fe00000) {
    if (ix <= 0x3c600000) /* |x| < 2**-57 */
      return pio2_hi + 0x1p-120f;
    return pio2_hi - (x - (pio2_lo - x * R(x * x)));
  }
  /* x < -0.5 */
  if (hx >> 31) {
    z = (1.0 + x) * 0.5;
    s = sqrt(z);
    w = R(z) * s - pio2_lo;
    return 2 * (pio2_hi - (s + w));
  }
  /* x > 0.5 */
  z = (1.0 - x) * 0.5;
  s = sqrt(z);
  memcpy(&xx, &s, sizeof(xx));
  xx &= 0xffffffff00000000;
  memcpy(&df, &xx, sizeof(xx));
  c = (z - df * df) / (s + df);
  w = R(z) * s + c;
  return 2 * (df + w);
}

struct Transform4x3 {
  const mat3x4 transform;

  vec3 operator()(vec3 position) { return transform * vec4(position, 1.0); }
};

struct UpdateMeshID {
  const HashTableD<uint32_t> meshIDold2new;

  void operator()(TriRef& ref) { ref.meshID = meshIDold2new[ref.meshID]; }
};

int GetLabels(std::vector<int>& components,
              const Vec<std::pair<int, int>>& edges, int numNodes) {
  DisjointSets uf(numNodes);
  for (auto edge : edges) {
    if (edge.first == -1 || edge.second == -1) continue;
    uf.unite(edge.first, edge.second);
  }

  return uf.connectedComponents(components);
}
}  // namespace

namespace manifold {

#if (MANIFOLD_PAR == 1)
#if (TBB_VERSION_MAJOR < 2021)
tbb::task_arena gc_arena(1, 1);
#else
tbb::task_arena gc_arena(1, 1, tbb::task_arena::priority::low);
#endif
#endif

std::atomic<uint32_t> Manifold::Impl::meshIDCounter_(1);

uint32_t Manifold::Impl::ReserveIDs(uint32_t n) {
  return Manifold::Impl::meshIDCounter_.fetch_add(n, std::memory_order_relaxed);
}

/**
 * Create either a unit tetrahedron, cube or octahedron. The cube is in the
 * first octant, while the others are symmetric about the origin.
 */
Manifold::Impl::Impl(Shape shape, const mat3x4 m) {
  std::vector<vec3> vertPos;
  std::vector<ivec3> triVerts;
  switch (shape) {
    case Shape::Tetrahedron:
      vertPos = {{-1.0, -1.0, 1.0},
                 {-1.0, 1.0, -1.0},
                 {1.0, -1.0, -1.0},
                 {1.0, 1.0, 1.0}};
      triVerts = {{2, 0, 1}, {0, 3, 1}, {2, 3, 0}, {3, 2, 1}};
      break;
    case Shape::Cube:
      vertPos = {{0.0, 0.0, 0.0},  //
                 {0.0, 0.0, 1.0},  //
                 {0.0, 1.0, 0.0},  //
                 {0.0, 1.0, 1.0},  //
                 {1.0, 0.0, 0.0},  //
                 {1.0, 0.0, 1.0},  //
                 {1.0, 1.0, 0.0},  //
                 {1.0, 1.0, 1.0}};
      triVerts = {{1, 0, 4}, {2, 4, 0},  //
                  {1, 3, 0}, {3, 1, 5},  //
                  {3, 2, 0}, {3, 7, 2},  //
                  {5, 4, 6}, {5, 1, 4},  //
                  {6, 4, 2}, {7, 6, 2},  //
                  {7, 3, 5}, {7, 5, 6}};
      break;
    case Shape::Octahedron:
      vertPos = {{1.0, 0.0, 0.0},   //
                 {-1.0, 0.0, 0.0},  //
                 {0.0, 1.0, 0.0},   //
                 {0.0, -1.0, 0.0},  //
                 {0.0, 0.0, 1.0},   //
                 {0.0, 0.0, -1.0}};
      triVerts = {{0, 2, 4}, {1, 5, 3},  //
                  {2, 1, 4}, {3, 5, 0},  //
                  {1, 3, 4}, {0, 5, 2},  //
                  {3, 0, 4}, {2, 5, 1}};
      break;
  }
  vertPos_ = vertPos;
  for (auto& v : vertPos_) v = m * vec4(v, 1.0);
  CreateHalfedges(triVerts);
  Finish();
  InitializeOriginal();
  MarkCoplanar();
}

void Manifold::Impl::RemoveUnreferencedVerts() {
  ZoneScoped;
  const int numVert = NumVert();
  Vec<int> keep(numVert, 0);
  auto policy = autoPolicy(numVert, 1e5);
  for_each(policy, halfedge_.cbegin(), halfedge_.cend(), [&keep](Halfedge h) {
    if (h.startVert >= 0) {
      reinterpret_cast<std::atomic<int>*>(&keep[h.startVert])
          ->store(1, std::memory_order_relaxed);
    }
  });

  for_each_n(policy, countAt(0), numVert, [&keep, this](int v) {
    if (keep[v] == 0) {
      vertPos_[v] = vec3(NAN);
    }
  });
}

void Manifold::Impl::InitializeOriginal(bool keepFaceID) {
  const int meshID = ReserveIDs(1);
  meshRelation_.originalID = meshID;
  auto& triRef = meshRelation_.triRef;
  triRef.resize_nofill(NumTri());
  for_each_n(autoPolicy(NumTri(), 1e5), countAt(0), NumTri(),
             [meshID, keepFaceID, &triRef](const int tri) {
               triRef[tri] = {meshID, meshID, -1,
                              keepFaceID ? triRef[tri].coplanarID : tri};
             });
  meshRelation_.meshIDtransform.clear();
  meshRelation_.meshIDtransform[meshID] = {meshID};
}

void Manifold::Impl::MarkCoplanar() {
  ZoneScoped;
  const int numTri = NumTri();
  struct TriPriority {
    double area2;
    int tri;
  };
  Vec<TriPriority> triPriority(numTri);
  for_each_n(autoPolicy(numTri), countAt(0), numTri,
             [&triPriority, this](int tri) {
               meshRelation_.triRef[tri].coplanarID = -1;
               if (halfedge_[3 * tri].startVert < 0) {
                 triPriority[tri] = {0, tri};
                 return;
               }
               const vec3 v = vertPos_[halfedge_[3 * tri].startVert];
               triPriority[tri] = {
                   length2(cross(vertPos_[halfedge_[3 * tri].endVert] - v,
                                 vertPos_[halfedge_[3 * tri + 1].endVert] - v)),
                   tri};
             });

  stable_sort(triPriority.begin(), triPriority.end(),
              [](auto a, auto b) { return a.area2 > b.area2; });

  Vec<int> interiorHalfedges;
  for (const auto tp : triPriority) {
    if (meshRelation_.triRef[tp.tri].coplanarID >= 0) continue;

    meshRelation_.triRef[tp.tri].coplanarID = tp.tri;
    if (halfedge_[3 * tp.tri].startVert < 0) continue;
    const vec3 base = vertPos_[halfedge_[3 * tp.tri].startVert];
    const vec3 normal = faceNormal_[tp.tri];
    interiorHalfedges.resize(3);
    interiorHalfedges[0] = 3 * tp.tri;
    interiorHalfedges[1] = 3 * tp.tri + 1;
    interiorHalfedges[2] = 3 * tp.tri + 2;
    while (!interiorHalfedges.empty()) {
      const int h =
          NextHalfedge(halfedge_[interiorHalfedges.back()].pairedHalfedge);
      interiorHalfedges.pop_back();
      if (meshRelation_.triRef[h / 3].coplanarID >= 0) continue;

      const vec3 v = vertPos_[halfedge_[h].endVert];
      if (std::abs(dot(v - base, normal)) < tolerance_) {
        meshRelation_.triRef[h / 3].coplanarID = tp.tri;

        if (interiorHalfedges.empty() ||
            h != halfedge_[interiorHalfedges.back()].pairedHalfedge) {
          interiorHalfedges.push_back(h);
        } else {
          interiorHalfedges.pop_back();
        }
        const int hNext = NextHalfedge(h);
        interiorHalfedges.push_back(hNext);
      }
    }
  }
}

/**
 * Dereference duplicate property vertices if they are exactly floating-point
 * equal. These unreferenced properties are then removed by CompactProps.
 */
void Manifold::Impl::DedupePropVerts() {
  ZoneScoped;
  const size_t numProp = NumProp();
  if (numProp == 0) return;

  Vec<std::pair<int, int>> vert2vert(halfedge_.size(), {-1, -1});
  for_each_n(autoPolicy(halfedge_.size(), 1e4), countAt(0), halfedge_.size(),
             [&vert2vert, numProp, this](const int edgeIdx) {
               const Halfedge edge = halfedge_[edgeIdx];
               if (edge.pairedHalfedge < 0) return;
               const int edgeFace = edgeIdx / 3;
               const int pairFace = edge.pairedHalfedge / 3;

               if (meshRelation_.triRef[edgeFace].meshID !=
                   meshRelation_.triRef[pairFace].meshID)
                 return;

               const int prop0 = halfedge_[edgeIdx].propVert;
               const int prop1 =
                   halfedge_[NextHalfedge(edge.pairedHalfedge)].propVert;
               bool propEqual = true;
               for (size_t p = 0; p < numProp; ++p) {
                 if (properties_[numProp * prop0 + p] !=
                     properties_[numProp * prop1 + p]) {
                   propEqual = false;
                   break;
                 }
               }
               if (propEqual) {
                 vert2vert[edgeIdx] = std::make_pair(prop0, prop1);
               }
             });

  std::vector<int> vertLabels;
  const size_t numPropVert = NumPropVert();
  const int numLabels = GetLabels(vertLabels, vert2vert, numPropVert);

  std::vector<int> label2vert(numLabels);
  for (size_t v = 0; v < numPropVert; ++v) label2vert[vertLabels[v]] = v;
  for (Halfedge& edge : halfedge_)
    edge.propVert = label2vert[vertLabels[edge.propVert]];
}

constexpr int kRemovedHalfedge = -2;

struct HalfedgePairData {
  int largeVert;
  int tri;
  int edgeIndex;

  bool operator<(const HalfedgePairData& other) const {
    return largeVert < other.largeVert ||
           (largeVert == other.largeVert && tri < other.tri);
  }
};

template <bool useProp, typename F>
struct PrepHalfedges {
  VecView<Halfedge> halfedges;
  const VecView<ivec3> triProp;
  const VecView<ivec3> triVert;
  F& f;

  void operator()(const int tri) {
    const ivec3& props = triProp[tri];
    for (const int i : {0, 1, 2}) {
      const int j = Next3(i);
      const int k = Next3(j);
      const int e = 3 * tri + i;
      const int v0 = useProp ? props[i] : triVert[tri][i];
      const int v1 = useProp ? props[j] : triVert[tri][j];
      DEBUG_ASSERT(v0 != v1, logicErr, "topological degeneracy");
      halfedges[e] = {v0, v1, -1, props[i]};
      f(e, v0, v1);
    }
  }
};

/**
 * Create the halfedge_ data structure from a list of triangles. If the optional
 * prop2vert array is missing, it's assumed these triangles are are pointing to
 * both vert and propVert indices. If prop2vert is present, the triangles are
 * assumed to be pointing to propVert indices only. The prop2vert array is used
 * to map the propVert indices to vert indices.
 */
void Manifold::Impl::CreateHalfedges(const Vec<ivec3>& triProp,
                                     const Vec<ivec3>& triVert) {
  ZoneScoped;
  const size_t numTri = triProp.size();
  const int numHalfedge = 3 * numTri;
  // drop the old value first to avoid copy
  halfedge_.clear(true);
  halfedge_.resize_nofill(numHalfedge);
  auto policy = autoPolicy(numTri, 1e5);

  int vertCount = static_cast<int>(vertPos_.size());
  Vec<int> ids(numHalfedge);
  {
    ZoneScopedN("PrepHalfedges");
    if (vertCount < (1 << 18)) {
      // For small vertex count, it is faster to just do sorting
      Vec<uint64_t> edge(numHalfedge);
      auto setEdge = [&edge](int e, int v0, int v1) {
        edge[e] = static_cast<uint64_t>(v0 < v1 ? 1 : 0) << 63 |
                  (static_cast<uint64_t>(std::min(v0, v1))) << 32 |
                  static_cast<uint64_t>(std::max(v0, v1));
      };
      if (triVert.empty()) {
        for_each_n(policy, countAt(0), numTri,
                   PrepHalfedges<true, decltype(setEdge)>{halfedge_, triProp,
                                                          triVert, setEdge});
      } else {
        for_each_n(policy, countAt(0), numTri,
                   PrepHalfedges<false, decltype(setEdge)>{halfedge_, triProp,
                                                           triVert, setEdge});
      }
      sequence(ids.begin(), ids.end());
      stable_sort(ids.begin(), ids.end(), [&edge](const int& a, const int& b) {
        return edge[a] < edge[b];
      });
    } else {
      // For larger vertex count, we separate the ids into slices for halfedges
      // with the same smaller vertex.
      // We first copy them there (as HalfedgePairData), and then do sorting
      // locally for each slice.
      // This helps with memory locality, and is faster for larger meshes.
      Vec<HalfedgePairData> entries(numHalfedge);
      Vec<int> offsets(vertCount * 2, 0);
      auto setOffset = [&offsets, vertCount](int _e, int v0, int v1) {
        const int offset = v0 > v1 ? 0 : vertCount;
        AtomicAdd(offsets[std::min(v0, v1) + offset], 1);
      };
      if (triVert.empty()) {
        for_each_n(policy, countAt(0), numTri,
                   PrepHalfedges<true, decltype(setOffset)>{
                       halfedge_, triProp, triVert, setOffset});
      } else {
        for_each_n(policy, countAt(0), numTri,
                   PrepHalfedges<false, decltype(setOffset)>{
                       halfedge_, triProp, triVert, setOffset});
      }
      exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
      for_each_n(policy, countAt(0), numTri,
                 [this, &offsets, &entries, vertCount](const int tri) {
                   for (const int i : {0, 1, 2}) {
                     const int e = 3 * tri + i;
                     const int v0 = halfedge_[e].startVert;
                     const int v1 = halfedge_[e].endVert;
                     const int offset = v0 > v1 ? 0 : vertCount;
                     const int start = std::min(v0, v1);
                     const int index = AtomicAdd(offsets[start + offset], 1);
                     entries[index] = {std::max(v0, v1), tri, e};
                   }
                 });
      for_each_n(policy, countAt(0), offsets.size(), [&](const int v) {
        int start = v == 0 ? 0 : offsets[v - 1];
        int end = offsets[v];
        for (int i = start; i < end; ++i) ids[i] = i;
        std::sort(ids.begin() + start, ids.begin() + end,
                  [&entries](int a, int b) { return entries[a] < entries[b]; });
        for (int i = start; i < end; ++i) ids[i] = entries[ids[i]].edgeIndex;
      });
    }
  }

  // Mark opposed triangles for removal - this may strand unreferenced verts
  // which are removed later by RemoveUnreferencedVerts() and Finish().
  const int numEdge = numHalfedge / 2;

  const auto body = [&](int i, int consecutiveStart, int segmentEnd) {
    const int pair0 = ids[i];
    Halfedge& h0 = halfedge_[pair0];
    int k = consecutiveStart + numEdge;
    while (1) {
      const int pair1 = ids[k];
      Halfedge& h1 = halfedge_[pair1];
      if (h0.startVert != h1.endVert || h0.endVert != h1.startVert) break;
      if (h1.pairedHalfedge != kRemovedHalfedge &&
          halfedge_[NextHalfedge(pair0)].endVert ==
              halfedge_[NextHalfedge(pair1)].endVert) {
        h0.pairedHalfedge = h1.pairedHalfedge = kRemovedHalfedge;
        // Reorder so that remaining edges pair up
        if (k != i + numEdge) std::swap(ids[i + numEdge], ids[k]);
        break;
      }
      ++k;
      if (k >= segmentEnd + numEdge) break;
    }
    if (i + 1 == segmentEnd) return consecutiveStart;
    Halfedge& h1 = halfedge_[ids[i + 1]];
    if (h1.startVert == h0.startVert && h1.endVert == h0.endVert)
      return consecutiveStart;
    return i + 1;
  };

#if MANIFOLD_PAR == 1
  Vec<std::pair<int, int>> ranges;
  const int increment = std::min(
      std::max(numEdge / tbb::this_task_arena::max_concurrency() / 2, 1024),
      numEdge);
  const auto duplicated = [&](int a, int b) {
    const Halfedge& h0 = halfedge_[ids[a]];
    const Halfedge& h1 = halfedge_[ids[b]];
    return h0.startVert == h1.startVert && h0.endVert == h1.endVert;
  };
  int end = 0;
  while (end < numEdge) {
    const int start = end;
    end = std::min(end + increment, numEdge);
    // make sure duplicated halfedges are in the same partition
    while (end < numEdge && duplicated(end - 1, end)) end++;
    ranges.push_back(std::make_pair(start, end));
  }
  for_each(ExecutionPolicy::Par, ranges.begin(), ranges.end(),
           [&](const std::pair<int, int>& range) {
             const auto [start, end] = range;
             int consecutiveStart = start;
             for (int i = start; i < end; ++i)
               consecutiveStart = body(i, consecutiveStart, end);
           });
#else
  int consecutiveStart = 0;
  for (int i = 0; i < numEdge; ++i)
    consecutiveStart = body(i, consecutiveStart, numEdge);
#endif
  for_each_n(policy, countAt(0), numEdge, [this, &ids, numEdge](int i) {
    const int pair0 = ids[i];
    const int pair1 = ids[i + numEdge];
    if (halfedge_[pair0].pairedHalfedge != kRemovedHalfedge) {
      halfedge_[pair0].pairedHalfedge = pair1;
      halfedge_[pair1].pairedHalfedge = pair0;
    } else {
      halfedge_[pair0] = halfedge_[pair1] = {-1, -1, -1};
    }
  });
}

/**
 * Does a full recalculation of the face bounding boxes, including updating
 * the collider, but does not resort the faces.
 */
void Manifold::Impl::Update() {
  CalculateBBox();
  Vec<Box> faceBox;
  Vec<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  collider_.UpdateBoxes(faceBox);
}

void Manifold::Impl::MakeEmpty(Error status) {
  bBox_ = Box();
  vertPos_.clear();
  halfedge_.clear();
  vertNormal_.clear();
  faceNormal_.clear();
  halfedgeTangent_.clear();
  meshRelation_ = MeshRelationD();
  status_ = status;
}

void Manifold::Impl::Warp(std::function<void(vec3&)> warpFunc) {
  WarpBatch([&warpFunc](VecView<vec3> vecs) {
    for_each(ExecutionPolicy::Seq, vecs.begin(), vecs.end(), warpFunc);
  });
}

void Manifold::Impl::WarpBatch(std::function<void(VecView<vec3>)> warpFunc) {
  warpFunc(vertPos_.view());
  CalculateBBox();
  if (!IsFinite()) {
    MakeEmpty(Error::NonFiniteVertex);
    return;
  }
  Update();
  faceNormal_.clear();  // force recalculation of triNormal
  SetEpsilon();
  Finish();
  MarkCoplanar();
  meshRelation_.originalID = -1;
}

Manifold::Impl Manifold::Impl::Transform(const mat3x4& transform_) const {
  ZoneScoped;
  if (transform_ == mat3x4(la::identity)) return *this;
  auto policy = autoPolicy(NumVert());
  Impl result;
  if (status_ != Manifold::Error::NoError) {
    result.status_ = status_;
    return result;
  }
  if (!all(la::isfinite(transform_))) {
    result.MakeEmpty(Error::NonFiniteVertex);
    return result;
  }
  result.collider_ = collider_;
  result.meshRelation_ = meshRelation_;
  result.epsilon_ = epsilon_;
  result.tolerance_ = tolerance_;
  result.numProp_ = numProp_;
  result.properties_ = properties_;
  result.bBox_ = bBox_;
  result.halfedge_ = halfedge_;
  result.halfedgeTangent_.resize(halfedgeTangent_.size());

  result.meshRelation_.originalID = -1;
  for (auto& m : result.meshRelation_.meshIDtransform) {
    m.second.transform = transform_ * Mat4(m.second.transform);
  }

  result.vertPos_.resize(NumVert());
  result.faceNormal_.resize(faceNormal_.size());
  result.vertNormal_.resize(vertNormal_.size());
  transform(vertPos_.begin(), vertPos_.end(), result.vertPos_.begin(),
            Transform4x3({transform_}));

  mat3 normalTransform = NormalTransform(transform_);
  transform(faceNormal_.begin(), faceNormal_.end(), result.faceNormal_.begin(),
            TransformNormals({normalTransform}));
  transform(vertNormal_.begin(), vertNormal_.end(), result.vertNormal_.begin(),
            TransformNormals({normalTransform}));

  const bool invert = la::determinant(mat3(transform_)) < 0;

  if (halfedgeTangent_.size() > 0) {
    for_each_n(policy, countAt(0), halfedgeTangent_.size(),
               TransformTangents({result.halfedgeTangent_, 0, mat3(transform_),
                                  invert, halfedgeTangent_, halfedge_}));
  }

  if (invert) {
    for_each_n(policy, countAt(0), result.NumTri(),
               FlipTris({result.halfedge_}));
  }

  // This optimization does a cheap collider update if the transform is
  // axis-aligned.
  if (!result.collider_.Transform(transform_)) result.Update();

  result.CalculateBBox();
  // Scale epsilon by the norm of the 3x3 portion of the transform.
  result.epsilon_ *= SpectralNorm(mat3(transform_));
  // Maximum of inherited epsilon loss and translational epsilon loss.
  result.SetEpsilon(result.epsilon_);
  return result;
}

/**
 * Sets epsilon based on the bounding box, and limits its minimum value
 * by the optional input.
 */
void Manifold::Impl::SetEpsilon(double minEpsilon, bool useSingle) {
  epsilon_ = MaxEpsilon(minEpsilon, bBox_);
  double minTol = epsilon_;
  if (useSingle)
    minTol =
        std::max(minTol, std::numeric_limits<float>::epsilon() * bBox_.Scale());
  tolerance_ = std::max(tolerance_, minTol);
}

/**
 * If face normals are already present, this function uses them to compute
 * vertex normals (angle-weighted pseudo-normals); otherwise it also computes
 * the face normals. Face normals are only calculated when needed because
 * nearly degenerate faces will accrue rounding error, while the Boolean can
 * retain their original normal, which is more accurate and can help with
 * merging coplanar faces.
 *
 * If the face normals have been invalidated by an operation like Warp(),
 * ensure you do faceNormal_.resize(0) before calling this function to force
 * recalculation.
 */
void Manifold::Impl::CalculateNormals() {
  ZoneScoped;
  vertNormal_.resize(NumVert());
  auto policy = autoPolicy(NumTri());

  std::vector<std::atomic<int>> vertHalfedgeMap(NumVert());
  for_each_n(policy, countAt(0), NumVert(), [&](const size_t vert) {
    vertHalfedgeMap[vert] = std::numeric_limits<int>::max();
  });

  auto atomicMin = [&vertHalfedgeMap](int value, int vert) {
    if (vert < 0) return;
    int old = std::numeric_limits<int>::max();
    while (!vertHalfedgeMap[vert].compare_exchange_strong(old, value))
      if (old < value) break;
  };
  if (faceNormal_.size() != NumTri()) {
    faceNormal_.resize(NumTri());
    for_each_n(policy, countAt(0), NumTri(), [&](const int face) {
      vec3& triNormal = faceNormal_[face];
      if (halfedge_[3 * face].startVert < 0) {
        triNormal = vec3(0, 0, 1);
        return;
      }

      ivec3 triVerts;
      for (int i : {0, 1, 2}) {
        int v = halfedge_[3 * face + i].startVert;
        triVerts[i] = v;
        atomicMin(3 * face + i, v);
      }

      vec3 edge[3];
      for (int i : {0, 1, 2}) {
        const int j = (i + 1) % 3;
        edge[i] = la::normalize(vertPos_[triVerts[j]] - vertPos_[triVerts[i]]);
      }
      triNormal = la::normalize(la::cross(edge[0], edge[1]));
      if (std::isnan(triNormal.x)) triNormal = vec3(0, 0, 1);
    });
  } else {
    for_each_n(policy, countAt(0), halfedge_.size(),
               [&](const int i) { atomicMin(i, halfedge_[i].startVert); });
  }

  for_each_n(policy, countAt(0), NumVert(), [&](const size_t vert) {
    int firstEdge = vertHalfedgeMap[vert].load();
    // not referenced
    if (firstEdge == std::numeric_limits<int>::max()) {
      vertNormal_[vert] = vec3(0.0);
      return;
    }
    vec3 normal = vec3(0.0);
    ForVert(firstEdge, [&](int edge) {
      ivec3 triVerts = {halfedge_[edge].startVert, halfedge_[edge].endVert,
                        halfedge_[NextHalfedge(edge)].endVert};
      vec3 currEdge =
          la::normalize(vertPos_[triVerts[1]] - vertPos_[triVerts[0]]);
      vec3 prevEdge =
          la::normalize(vertPos_[triVerts[0]] - vertPos_[triVerts[2]]);

      // if it is not finite, this means that the triangle is degenerate, and we
      // should just exclude it from the normal calculation...
      if (!la::isfinite(currEdge[0]) || !la::isfinite(prevEdge[0])) return;
      double dot = -la::dot(prevEdge, currEdge);
      double phi = dot >= 1 ? 0 : (dot <= -1 ? kPi : sun_acos(dot));
      normal += phi * faceNormal_[edge / 3];
    });
    vertNormal_[vert] = SafeNormalize(normal);
  });
}

/**
 * Remaps all the contained meshIDs to new unique values to represent new
 * instances of these meshes.
 */
void Manifold::Impl::IncrementMeshIDs() {
  HashTable<uint32_t> meshIDold2new(meshRelation_.meshIDtransform.size() * 2);
  // Update keys of the transform map
  std::map<int, Relation> oldTransforms;
  std::swap(meshRelation_.meshIDtransform, oldTransforms);
  const int numMeshIDs = oldTransforms.size();
  int nextMeshID = ReserveIDs(numMeshIDs);
  for (const auto& pair : oldTransforms) {
    meshIDold2new.D().Insert(pair.first, nextMeshID);
    meshRelation_.meshIDtransform[nextMeshID++] = pair.second;
  }

  const size_t numTri = NumTri();
  for_each_n(autoPolicy(numTri, 1e5), meshRelation_.triRef.begin(), numTri,
             UpdateMeshID({meshIDold2new.D()}));
}

#ifdef MANIFOLD_DEBUG
/**
 * Debugging output using high precision OBJ files with specialized comments
 */
std::ostream& operator<<(std::ostream& stream, const Manifold::Impl& impl) {
  stream << std::setprecision(19);  // for double precision
  stream << std::fixed;             // for uniformity in output numbers
  stream << "# ======= begin mesh ======" << std::endl;
  stream << "# tolerance = " << impl.tolerance_ << std::endl;
  stream << "# epsilon = " << impl.epsilon_ << std::endl;
  // TODO: Mesh relation, vertex normal and face normal
  for (const vec3& v : impl.vertPos_)
    stream << "v " << v.x << " " << v.y << " " << v.z << std::endl;
  std::vector<ivec3> triangles;
  triangles.reserve(impl.halfedge_.size() / 3);
  for (size_t i = 0; i < impl.halfedge_.size(); i += 3)
    triangles.emplace_back(impl.halfedge_[i].startVert + 1,
                           impl.halfedge_[i + 1].startVert + 1,
                           impl.halfedge_[i + 2].startVert + 1);
  sort(triangles.begin(), triangles.end());
  for (const auto& tri : triangles)
    stream << "f " << tri.x << " " << tri.y << " " << tri.z << std::endl;
  stream << "# ======== end mesh =======" << std::endl;
  return stream;
}

/**
 * Import a mesh from a Wavefront OBJ file that was exported with Write.  This
 * function is the counterpart to Write and should be used with it.  This
 * function is not guaranteed to be able to import OBJ files not written by the
 * Write function.
 */
Manifold Manifold::ReadOBJ(std::istream& stream) {
  if (!stream.good()) return Invalid();

  MeshGL64 mesh;
  std::optional<double> epsilon;
  stream >> std::setprecision(19);
  while (true) {
    char c = stream.get();
    if (stream.eof()) break;
    switch (c) {
      case '#': {
        char c = stream.get();
        if (c == ' ') {
          constexpr int SIZE = 10;
          std::array<char, SIZE> tmp;
          stream.get(tmp.data(), SIZE, '\n');
          if (strncmp(tmp.data(), "tolerance", SIZE) == 0) {
            // skip 3 letters
            for (int _ : {0, 1, 2}) stream.get();
            stream >> mesh.tolerance;
          } else if (strncmp(tmp.data(), "epsilon =", SIZE) == 0) {
            double tmp;
            stream >> tmp;
            epsilon = {tmp};
          } else {
            // add it back because it is not what we want
            int end = 0;
            while (end < SIZE && tmp[end] != 0) end++;
            while (--end > -1) stream.putback(tmp[end]);
          }
          c = stream.get();
        }
        // just skip the remaining comment
        while (c != '\n' && !stream.eof()) {
          c = stream.get();
        }
        break;
      }
      case 'v':
        for (int _ : {0, 1, 2}) {
          double x;
          stream >> x;
          mesh.vertProperties.push_back(x);
        }
        break;
      case 'f':
        for (int _ : {0, 1, 2}) {
          uint64_t x;
          stream >> x;
          mesh.triVerts.push_back(x - 1);
        }
        break;
      case '\r':
      case '\n':
        break;
      default:
        DEBUG_ASSERT(false, userErr, "unexpected character in Manifold import");
    }
  }
  auto m = std::make_shared<Manifold::Impl>(mesh);
  if (epsilon) m->SetEpsilon(*epsilon);
  return Manifold(m);
}

/**
 * Export the mesh to a Wavefront OBJ file in a way that preserves the full
 * 64-bit precision of the vertex positions, as well as storing metadata such as
 * the tolerance and epsilon. Useful for debugging and testing.  Files written
 * by WriteOBJ should be read back in with ReadOBJ.
 */
bool Manifold::WriteOBJ(std::ostream& stream) const {
  if (!stream.good()) return false;
  stream << *this->GetCsgLeafNode().GetImpl();
  return true;
}
#endif

}  // namespace manifold
