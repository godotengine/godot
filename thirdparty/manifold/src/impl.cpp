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
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef MANIFOLD_NO_IOSTREAM
#include <iomanip>
#endif
#include <map>
#include <optional>
#include <regex>
#ifndef MANIFOLD_NO_IOSTREAM
#include <sstream>
#endif

#include "csg_tree.h"
#include "disjoint_sets.h"
#include "hashtable.h"
#include "manifold/manifold.h"
#include "manifold/optional_assert.h"
#include "mesh_fixes.h"
#include "parallel.h"
#include "shared.h"
#include "svd.h"

namespace {
using namespace manifold;

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

#ifndef MANIFOLD_NO_IOSTREAM
template <typename T>
double FromChars(T buffer) {
  double tmp;
  std::istringstream iss(buffer);
  iss >> std::setprecision(19);
  iss >> tmp;
  return tmp;
}
#endif
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
  vertPos_ = Vec(vertPos);
  for (auto& v : vertPos_) v = m * vec4(v, 1.0);
  CreateHalfedges(triVerts);
  InitializeOriginal();
  CalculateBBox();
  SetEpsilon();
  SortGeometry();
  SetNormalsAndCoplanar();
}

void Manifold::Impl::RemoveUnreferencedVerts() {
  ZoneScoped;
  const int numVert = NumVert();
  Vec<int> keep(numVert, 0);
  auto policy = autoPolicy(numVert, 1e5);
  for_each_n(policy, countAt(0), halfedge_.size(), [&keep, this](int edge) {
    const int startVert = halfedge_.Start(edge);
    if (startVert >= 0) {
      reinterpret_cast<std::atomic<int>*>(&keep[startVert])
          ->store(1, std::memory_order_relaxed);
    }
  });

  for_each_n(policy, countAt(0), numVert, [&keep, this](int v) {
    if (keep[v] == 0) {
      vertPos_[v] = vec3(NAN);
    }
  });
}

void Manifold::Impl::EagerTransformPropNormals(
    const Halfedges& halfedge, const MeshRelationD& meshRelation,
    const mat3& normalTransform, Vec<double>& properties, int numPropVert,
    int stride, int offset) {
  // Short-circuit when no meshID carries normals. OR semantics (any has
  // it), unlike AllHaveNormals() - mixed inputs still need the per-meshID
  // iteration below to rotate the with-normals subset.
  bool anyHasNormals = false;
  for (const auto& m : meshRelation.meshIDtransform) {
    if (m.second.hasNormals) {
      anyHasNormals = true;
      break;
    }
  }
  if (!anyHasNormals) return;
  Vec<bool> propVisited(numPropVert, false);
  for (size_t e = 0; e < halfedge.size(); ++e) {
    if (!TriHasNormals(meshRelation, e / 3)) continue;
    const int prop = halfedge.Prop(e);
    if (prop < 0 || propVisited[prop]) continue;
    propVisited[prop] = true;
    vec3 n;
    for (const int i : {0, 1, 2})
      n[i] = properties[(offset + prop) * stride + i];
    // Re-normalize as we transform: non-orthogonal transforms (scale) and
    // barycentric interpolation upstream both leave non-unit values that
    // would otherwise compound and break downstream lighting / smoothing.
    n = SafeNormalize(normalTransform * n);
    for (const int i : {0, 1, 2})
      properties[(offset + prop) * stride + i] = n[i];
  }
}

void Manifold::Impl::InitializeOriginal() {
  const int meshID = ReserveIDs(1);
  meshRelation_.originalID = meshID;
  auto& triRef = meshRelation_.triRef;
  triRef.resize_nofill(NumTri());
  for_each_n(autoPolicy(NumTri(), 1e5), countAt(0), NumTri(),
             [meshID, &triRef](const int tri) {
               triRef[tri] = {meshID, meshID, -1, triRef[tri].coplanarID};
             });
  // Preserve the AND-across-old-Relations state so AsOriginal keeps the
  // recording when it builds a fresh Relation. Primitives start with an
  // empty map, which AllHaveNormals() returns false for.
  const bool hadNormals = AllHaveNormals();
  meshRelation_.meshIDtransform.clear();
  meshRelation_.meshIDtransform[meshID] = {meshID, la::identity, false,
                                           hadNormals};
}

void Manifold::Impl::SetNormalsAndCoplanar() {
  ZoneScoped;
  const int numTri = NumTri();
  faceNormal_.resize(numTri);
  struct TriPriority {
    double area2;
    int tri;
  };
  Vec<TriPriority> triPriority(numTri);
  for_each_n(autoPolicy(numTri), countAt(0), numTri,
             [&triPriority, this](int tri) {
               meshRelation_.triRef[tri].coplanarID = -1;
               if (halfedge_.Start(3 * tri) < 0) {
                 triPriority[tri] = {0, tri};
                 return;
               }
               const vec3 v = vertPos_[halfedge_.Start(3 * tri)];
               const vec3 n = cross(vertPos_[halfedge_.End(3 * tri)] - v,
                                    vertPos_[halfedge_.End(3 * tri + 1)] - v);
               faceNormal_[tri] = normalize(n);
               if (std::isnan(faceNormal_[tri].x))
                 faceNormal_[tri] = vec3(0, 0, 1);
               triPriority[tri] = {length2(n), tri};
             });

  stable_sort(triPriority.begin(), triPriority.end(),
              [](auto a, auto b) { return a.area2 > b.area2; });

  Vec<int> interiorHalfedges;
  for (const auto tp : triPriority) {
    if (meshRelation_.triRef[tp.tri].coplanarID >= 0) continue;

    meshRelation_.triRef[tp.tri].coplanarID = tp.tri;
    if (halfedge_.Start(3 * tp.tri) < 0) continue;
    const vec3 base = vertPos_[halfedge_.Start(3 * tp.tri)];
    const vec3 normal = faceNormal_[tp.tri];
    interiorHalfedges.resize(3);
    interiorHalfedges[0] = 3 * tp.tri;
    interiorHalfedges[1] = 3 * tp.tri + 1;
    interiorHalfedges[2] = 3 * tp.tri + 2;
    while (!interiorHalfedges.empty()) {
      const int h = NextHalfedge(halfedge_.Pair(interiorHalfedges.back()));
      interiorHalfedges.pop_back();
      if (meshRelation_.triRef[h / 3].coplanarID >= 0) continue;

      const vec3 v = vertPos_[halfedge_.End(h)];
      if (std::abs(dot(v - base, normal)) < tolerance_) {
        const size_t tri = h / 3;
        meshRelation_.triRef[tri].coplanarID = tp.tri;
        faceNormal_[tri] = normal;

        if (interiorHalfedges.empty() ||
            h != halfedge_.Pair(interiorHalfedges.back())) {
          interiorHalfedges.push_back(h);
        } else {
          interiorHalfedges.pop_back();
        }
        const int hNext = NextHalfedge(h);
        interiorHalfedges.push_back(hNext);
      }
    }
  }
  CalculateVertNormals();
}

/**
 * Dereference duplicate property vertices if they are exactly floating-point
 * equal. These unreferenced properties are then removed by CompactProps.
 */
void Manifold::Impl::DedupePropVerts() {
  ZoneScoped;
  const size_t numProp = NumProp();
  if (numProp == 0) return;

  halfedge_.MakeUnique();
  Vec<std::pair<int, int>> vert2vert(halfedge_.size(), {-1, -1});
  for_each_n(autoPolicy(halfedge_.size(), 1e4), countAt(0), halfedge_.size(),
             [&vert2vert, numProp, this](const int edgeIdx) {
               const int pair = halfedge_.Pair(edgeIdx);
               if (pair < 0) return;
               const int edgeFace = edgeIdx / 3;
               const int pairFace = pair / 3;

               if (meshRelation_.triRef[edgeFace].meshID !=
                   meshRelation_.triRef[pairFace].meshID)
                 return;

               const int prop0 = halfedge_.Prop(edgeIdx);
               const int prop1 = halfedge_.Prop(NextHalfedge(pair));
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
  for (size_t edge = 0; edge < halfedge_.size(); ++edge) {
    halfedge_.SetProp(edge, label2vert[vertLabels[halfedge_.Prop(edge)]]);
  }
}

struct CreateHalfedge {
  int startVert;
  int endVert;
  int propVert;
};

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
  VecView<CreateHalfedge> halfedges;
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
      halfedges[e] = {v0, v1, props[i]};
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
  Vec<CreateHalfedge> halfedge(numHalfedge);
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
                   PrepHalfedges<true, decltype(setEdge)>{halfedge, triProp,
                                                          triVert, setEdge});
      } else {
        for_each_n(policy, countAt(0), numTri,
                   PrepHalfedges<false, decltype(setEdge)>{halfedge, triProp,
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
                       halfedge, triProp, triVert, setOffset});
      } else {
        for_each_n(policy, countAt(0), numTri,
                   PrepHalfedges<false, decltype(setOffset)>{
                       halfedge, triProp, triVert, setOffset});
      }
      exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());
      for_each_n(policy, countAt(0), numTri,
                 [&halfedge, &offsets, &entries, vertCount](const int tri) {
                   for (const int i : {0, 1, 2}) {
                     const int e = 3 * tri + i;
                     const int v0 = halfedge[e].startVert;
                     const int v1 = halfedge[e].endVert;
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
  Vec<unsigned char> removed(numHalfedge, false);

  const auto body = [&](int i, int consecutiveStart, int segmentEnd) {
    const int pair0 = ids[i];
    const CreateHalfedge h0 = halfedge[pair0];
    int k = consecutiveStart + numEdge;
    while (1) {
      const int pair1 = ids[k];
      const CreateHalfedge h1 = halfedge[pair1];
      if (h0.startVert != h1.endVert || h0.endVert != h1.startVert) break;
      if (!removed[pair1] && halfedge[NextHalfedge(pair0)].endVert ==
                                 halfedge[NextHalfedge(pair1)].endVert) {
        removed[pair0] = true;
        removed[pair1] = true;
        if (i + numEdge != k) {
          // Reorder so that remaining edges pair up, while preserving relative
          // order between the edges (triangle id order)
          // cannot directly use move and move_backward because we need to keep
          // removed halfedges in-place
          int dir = i + numEdge < k ? 1 : -1;
          int a = k;
          int b = k + dir;
          auto isRemoved = [&removed, &ids](int x) { return removed[ids[x]]; };
          auto inRange = [&a, dir, i, numEdge]() {
            return (dir > 0 ? a >= i + numEdge : a <= i + numEdge);
          };
          while (1) {
            do {
              a -= dir;
            } while (inRange() && isRemoved(a));
            if (!inRange()) break;
            do {
              b -= dir;
            } while (isRemoved(b) && b != k);
            ids[b] = ids[a];
          }
          ids[i + numEdge] = pair1;
        }
        break;
      }
      ++k;
      if (k >= segmentEnd + numEdge) break;
    }
    if (i + 1 == segmentEnd) return consecutiveStart;
    const CreateHalfedge h1 = halfedge[ids[i + 1]];
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
    const CreateHalfedge h0 = halfedge[ids[a]];
    const CreateHalfedge h1 = halfedge[ids[b]];
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

  halfedge_.clear(true);
  halfedge_.resize_nofill(numHalfedge);
  for_each_n(policy, countAt(0), numEdge,
             [this, &halfedge, &ids, &removed, numEdge](int i) {
               const int pair0 = ids[i];
               const int pair1 = ids[i + numEdge];
               if (!removed[pair0]) {
                 halfedge_.SetStart(pair0, halfedge[pair0].startVert);
                 halfedge_.SetProp(pair0, halfedge[pair0].propVert);
                 halfedge_.SetPair(pair0, pair1);
                 halfedge_.SetStart(pair1, halfedge[pair1].startVert);
                 halfedge_.SetProp(pair1, halfedge[pair1].propVert);
                 halfedge_.SetPair(pair1, pair0);
               } else {
                 halfedge_.SetStart(pair0, -1);
                 halfedge_.SetProp(pair0, 0);
                 halfedge_.SetPair(pair0, -1);
                 halfedge_.SetStart(pair1, -1);
                 halfedge_.SetProp(pair1, 0);
                 halfedge_.SetPair(pair1, -1);
               }
             });
#ifdef MANIFOLD_DEBUG
  for (int edge = 0; edge < numHalfedge; ++edge) {
    const int next = NextHalfedge(edge);
    if (!removed[edge] && !removed[next]) {
      DEBUG_ASSERT(halfedge[edge].endVert == halfedge[next].startVert,
                   topologyErr,
                   "CreateHalfedges requires triangle-ordered edges!");
    }
  }
#endif
}

void Manifold::Impl::MakeEmpty(Error status) {
  bBox_ = Box();
  vertPos_.clear();
  halfedge_.MakeUnique();
  halfedge_.clear();
  vertNormal_.clear();
  faceNormal_.clear();
  halfedgeTangent_.clear();
  meshRelation_ = MeshRelationD();
  collider_ = {};
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
  SetEpsilon();
  SortGeometry();
  SetNormalsAndCoplanar();
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

  if (numProp_ >= 3) {
    EagerTransformPropNormals(halfedge_, meshRelation_, normalTransform,
                              result.properties_, NumPropVert(), numProp_);
  }

  const bool invert = la::determinant(mat3(transform_)) < 0;

  if (halfedgeTangent_.size() > 0) {
    for_each_n(policy, countAt(0), halfedgeTangent_.size(),
               TransformTangents({result.halfedgeTangent_, 0, mat3(transform_),
                                  invert, halfedgeTangent_, halfedge_}));
  }

  if (invert) {
    result.halfedge_.MakeUnique();
    for_each_n(policy, countAt(0), result.NumTri(),
               FlipTris({result.halfedge_}));
  }

  result.CalculateBBox();
  // Scale epsilon by the norm of the 3x3 portion of the transform.
  result.epsilon_ *= SpectralNorm(mat3(transform_));
  // Maximum of inherited epsilon loss and translational epsilon loss.
  result.SetEpsilon(result.epsilon_);

  if (!result.IsEmpty()) {
    if (Collider::IsAxisAligned(transform_)) {
      result.collider_ = collider_;
      result.collider_.Transform(transform_);
    } else if (!result.IsEmpty()) {
      result.collider_ = collider_;
      Vec<Box> faceBox;
      Vec<uint32_t> faceMorton;
      result.GetFaceBoxMorton(faceBox, faceMorton);
      result.collider_.UpdateBoxes(faceBox);
    }
  }
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
 * This function uses the face normals to compute
 * vertex normals (angle-weighted pseudo-normals). Face normals should only be
 * calculated when needed because nearly degenerate faces will accrue rounding
 * error, while the Boolean can retain their original normal, which is more
 * accurate and can help with merging coplanar faces.
 */
void Manifold::Impl::CalculateVertNormals() {
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

  for_each_n(policy, countAt(0), halfedge_.size(),
             [&](const int i) { atomicMin(i, halfedge_.Start(i)); });

  for_each_n(policy, countAt(0), NumVert(), [&](const size_t vert) {
    int firstEdge = vertHalfedgeMap[vert].load();
    // not referenced
    if (firstEdge == std::numeric_limits<int>::max()) {
      vertNormal_[vert] = vec3(0.0);
      return;
    }
    vec3 normal = vec3(0.0);
    ForVert(firstEdge, [&](int edge) {
      ivec3 triVerts = {halfedge_.Start(edge), halfedge_.End(edge),
                        halfedge_.End(NextHalfedge(edge))};
      vec3 currEdge =
          la::normalize(vertPos_[triVerts[1]] - vertPos_[triVerts[0]]);
      vec3 prevEdge =
          la::normalize(vertPos_[triVerts[0]] - vertPos_[triVerts[2]]);

      // if it is not finite, this means that the triangle is degenerate, and we
      // should just exclude it from the normal calculation...
      if (!la::isfinite(currEdge[0]) || !la::isfinite(prevEdge[0])) return;
      double dot = -la::dot(prevEdge, currEdge);
      double phi = dot >= 1 ? 0 : (dot <= -1 ? kPi : math::acos(dot));
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
  ZoneScoped;
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

#ifndef MANIFOLD_NO_IOSTREAM
static std::ostream& WriteOBJWithEpsilon(std::ostream& stream,
                                         const MeshGL64& mesh,
                                         std::optional<double> epsilon) {
  auto useHexFloat = []() {
    const char* v = std::getenv("MANIFOLD_OBJ_HEX_FLOAT");
    if (v == nullptr) return false;
    return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
           std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "on") == 0 ||
           std::strcmp(v, "ON") == 0;
  };
  const bool hexFloat = useHexFloat();
  auto writeValue = [&](double value) {
    if (hexFloat) {
      // Use explicit C-format hex to keep text stable across standard library
      // implementations.
      char buf[128];
      std::snprintf(buf, sizeof(buf), "%.13a", value);
      stream << buf;
    } else {
      stream << value;
    }
  };

  stream << std::setprecision(19);  // for double precision
  if (!hexFloat) {
    stream << std::fixed;  // for uniformity in output numbers
  }
  stream << "# ======= begin mesh ======" << std::endl;
  stream << "# float_format = " << (hexFloat ? "hexfloat" : "fixed")
         << std::endl;
  stream << "# tolerance = ";
  writeValue(mesh.tolerance);
  stream << std::endl;
  if (epsilon.has_value()) {
    stream << "# epsilon = ";
    writeValue(epsilon.value());
    stream << std::endl;
  }
  for (size_t i = 0; i < mesh.NumVert(); i++) {
    stream << "v";
    size_t offset = i * mesh.numProp;
    for (size_t j : {0, 1, 2}) {
      stream << " ";
      writeValue(mesh.vertProperties[offset + j]);
    }
    stream << std::endl;
  }
  std::vector<std::array<uint64_t, 3>> triangles;
  triangles.reserve(mesh.NumTri());
  for (size_t i = 0; i < mesh.NumTri(); i++)
    triangles.push_back({mesh.triVerts[3 * i] + 1, mesh.triVerts[3 * i + 1] + 1,
                         mesh.triVerts[3 * i + 2] + 1});
  sort(triangles.begin(), triangles.end());
  for (const auto& tri : triangles)
    stream << "f " << tri[0] << " " << tri[1] << " " << tri[2] << std::endl;
  stream << "# ======== end mesh =======" << std::endl;
  return stream;
}

static std::pair<MeshGL64, std::optional<double>> ReadOBJWithEpsilon(
    std::istream& stream) {
  static const std::string FLOAT_PATTERN =
      "(-?\\d+(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?)";
  static const std::string FACE_ELEMENT = "(\\d+)(?:\\S+)?";
  static const std::string TRAILING_SPACES = "(?:\\s*)";
  static const std::string SEPARATOR = "\\s+";
  static const std::regex TOLERANCE_COMMENT_PATTERN(
      "^# tolerance = " + FLOAT_PATTERN + TRAILING_SPACES);
  static const std::regex EPSILON_COMMENT_PATTERN(
      "^# epsilon = " + FLOAT_PATTERN + TRAILING_SPACES);
  static const std::regex VERTEX_PATTERN("^v" + SEPARATOR + FLOAT_PATTERN +
                                         SEPARATOR + FLOAT_PATTERN + SEPARATOR +
                                         FLOAT_PATTERN + TRAILING_SPACES);
  static const std::regex FACE_PATTERN("^f" + SEPARATOR + FACE_ELEMENT +
                                       SEPARATOR + FACE_ELEMENT + SEPARATOR +
                                       FACE_ELEMENT + TRAILING_SPACES);

  MeshGL64 mesh;
  std::optional<double> epsilon;
  if (!stream.good()) return std::make_pair(mesh, epsilon);

  constexpr size_t BUFFER_SIZE = 1000;
  std::array<char, BUFFER_SIZE> buffer;
  std::cmatch m;

  while (!stream.eof()) {
    // extract line, and skip the line if it is longer than BUFFER_SIZE
    // because the lines we care about should not be that long
    // not using std::basic_istream<...>::getline because getline throws when
    // the size exceeds the limit, but we don't want exception related code
    size_t i = 0;
    char c;
    while (!stream.eof() && (c = stream.get()) != '\n' && c != '\r')
      if (i < BUFFER_SIZE) buffer[i++] = c;
    if (i == BUFFER_SIZE) continue;
    buffer[i] = '\0';
    // check pattern...
    if (std::regex_match(buffer.data(), m, TOLERANCE_COMMENT_PATTERN)) {
      mesh.tolerance = FromChars(m[1]);
    } else if (std::regex_match(buffer.data(), m, EPSILON_COMMENT_PATTERN)) {
      epsilon = {FromChars(m[1])};
    } else if (std::regex_match(buffer.data(), m, VERTEX_PATTERN)) {
      for (int j : {0, 1, 2})
        mesh.vertProperties.push_back(FromChars(m[j + 1]));
    } else if (std::regex_match(buffer.data(), m, FACE_PATTERN)) {
      for (int j : {0, 1, 2})
        mesh.triVerts.push_back(std::stoi(m[j + 1].str()) - 1);
    }
  }

  return std::make_pair(mesh, epsilon);
}

/**
 * Import a mesh from a Wavefront OBJ file.
 */
MeshGL64 ReadOBJ(std::istream& stream) {
  return ReadOBJWithEpsilon(stream).first;
}

/**
 * Import a mesh from a Wavefront OBJ file.
 * This supports reading tolerance and epsilon values from WriteOBJ.
 */
Manifold Manifold::ReadOBJ(std::istream& stream) {
  if (!stream.good()) return Invalid();
  auto [mesh, epsilon] = ReadOBJWithEpsilon(stream);
  auto impl = std::make_shared<Impl>(mesh);
  if (epsilon) impl->SetEpsilon(epsilon.value());
  return Manifold(impl);
}

/**
 * Export the mesh to a Wavefront OBJ file in a way that preserves the full
 * 64-bit precision of the vertex positions.
 */
bool WriteOBJ(std::ostream& stream, const MeshGL64& mesh) {
  if (!stream.good()) return false;
  WriteOBJWithEpsilon(stream, mesh, {});
  return true;
}

/**
 * Debugging output using high precision OBJ files with specialized comments
 */
std::ostream& operator<<(std::ostream& stream, const Manifold::Impl& impl) {
  MeshGL64 mesh = GetMeshGLImpl<double, uint64_t>(impl, -1);
  return WriteOBJWithEpsilon(stream, mesh, {impl.epsilon_});
}

/**
 * Export the mesh to a Wavefront OBJ file in a way that preserves the full
 * 64-bit precision of the vertex positions, as well as storing metadata such as
 * the tolerance and epsilon.
 */
bool Manifold::WriteOBJ(std::ostream& stream) const {
  if (!stream.good()) return false;
  stream << *this->GetCsgLeafNode().GetImpl();
  return true;
}
#endif
}  // namespace manifold
