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

#include <unordered_map>

#include "impl.h"
#include "parallel.h"
#include "shared.h"

namespace {
using namespace manifold;

ivec3 TriOf(int edge) {
  ivec3 triEdge;
  triEdge[0] = edge;
  triEdge[1] = NextHalfedge(triEdge[0]);
  triEdge[2] = NextHalfedge(triEdge[1]);
  return triEdge;
}

bool Is01Longest(vec2 v0, vec2 v1, vec2 v2) {
  const vec2 e[3] = {v1 - v0, v2 - v1, v0 - v2};
  double l[3];
  for (int i : {0, 1, 2}) l[i] = la::dot(e[i], e[i]);
  return l[0] > l[1] && l[0] > l[2];
}

struct DuplicateEdge {
  const Halfedge* sortedHalfedge;

  bool operator()(int edge) {
    const Halfedge& halfedge = sortedHalfedge[edge];
    const Halfedge& nextHalfedge = sortedHalfedge[edge + 1];
    return halfedge.startVert == nextHalfedge.startVert &&
           halfedge.endVert == nextHalfedge.endVert;
  }
};

struct ShortEdge {
  VecView<const Halfedge> halfedge;
  VecView<const vec3> vertPos;
  const double epsilon;
  const int firstNewVert;

  bool operator()(int edge) const {
    const Halfedge& half = halfedge[edge];
    if (half.pairedHalfedge < 0 ||
        (half.startVert < firstNewVert && half.endVert < firstNewVert))
      return false;
    // Flag short edges
    const vec3 delta = vertPos[half.endVert] - vertPos[half.startVert];
    return la::dot(delta, delta) < epsilon * epsilon;
  }
};

struct FlagEdge {
  VecView<const Halfedge> halfedge;
  VecView<const TriRef> triRef;
  const int firstNewVert;

  bool operator()(int edge) const {
    const Halfedge& half = halfedge[edge];
    if (half.pairedHalfedge < 0 || half.startVert < firstNewVert) return false;
    // Flag redundant edges - those where the startVert is surrounded by only
    // two original triangles.
    const TriRef ref0 = triRef[edge / 3];
    int current = NextHalfedge(half.pairedHalfedge);
    TriRef ref1 = triRef[current / 3];
    bool ref1Updated = !ref0.SameFace(ref1);
    while (current != edge) {
      current = NextHalfedge(halfedge[current].pairedHalfedge);
      int tri = current / 3;
      const TriRef ref = triRef[tri];
      if (!ref.SameFace(ref0) && !ref.SameFace(ref1)) {
        if (!ref1Updated) {
          ref1 = ref;
          ref1Updated = true;
        } else {
          return false;
        }
      }
    }
    return true;
  }
};

struct SwappableEdge {
  VecView<const Halfedge> halfedge;
  VecView<const vec3> vertPos;
  VecView<const vec3> triNormal;
  const double tolerance;
  const int firstNewVert;

  bool operator()(int edge) const {
    const Halfedge& half = halfedge[edge];
    if (half.pairedHalfedge < 0) return false;
    if (half.startVert < firstNewVert && half.endVert < firstNewVert &&
        halfedge[NextHalfedge(edge)].endVert < firstNewVert &&
        halfedge[NextHalfedge(half.pairedHalfedge)].endVert < firstNewVert)
      return false;

    int tri = edge / 3;
    ivec3 triEdge = TriOf(edge);
    mat2x3 projection = GetAxisAlignedProjection(triNormal[tri]);
    vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedge[triEdge[i]].startVert];
    if (CCW(v[0], v[1], v[2], tolerance) > 0 || !Is01Longest(v[0], v[1], v[2]))
      return false;

    // Switch to neighbor's projection.
    edge = half.pairedHalfedge;
    tri = edge / 3;
    triEdge = TriOf(edge);
    projection = GetAxisAlignedProjection(triNormal[tri]);
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedge[triEdge[i]].startVert];
    return CCW(v[0], v[1], v[2], tolerance) > 0 ||
           Is01Longest(v[0], v[1], v[2]);
  }
};

struct FlagStore {
#if MANIFOLD_PAR == 1
  tbb::combinable<std::vector<size_t>> store;
#endif
  std::vector<size_t> s;

  template <typename Pred, typename F>
  void run_seq(size_t n, Pred pred, F f) {
    for (size_t i = 0; i < n; ++i)
      if (pred(i)) s.push_back(i);
    for (size_t i : s) f(i);
    s.clear();
  }

#if MANIFOLD_PAR == 1
  template <typename Pred, typename F>
  void run_par(size_t n, Pred pred, F f) {
    // Test pred in parallel, store i into thread-local vectors when pred(i) is
    // true. After testing pred, iterate and call f over the indices in
    // ascending order by using a heap in a single thread
    auto& store = this->store;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&store, &pred](const auto& r) {
                        auto& local = store.local();
                        for (auto i = r.begin(); i < r.end(); ++i) {
                          if (pred(i)) local.push_back(i);
                        }
                      });

    std::vector<std::vector<size_t>> stores;
    std::vector<size_t> result;
    store.combine_each(
        [&](auto& data) { stores.emplace_back(std::move(data)); });
    std::vector<size_t> sizes;
    size_t total_size = 0;
    for (const auto& tmp : stores) {
      sizes.push_back(total_size);
      total_size += tmp.size();
    }
    result.resize(total_size);
    for_each_n(ExecutionPolicy::Seq, countAt(0), stores.size(), [&](size_t i) {
      std::copy(stores[i].begin(), stores[i].end(), result.begin() + sizes[i]);
    });
    stable_sort(autoPolicy(result.size()), result.begin(), result.end());
    for (size_t x : result) f(x);
  }
#endif

  template <typename Pred, typename F>
  void run(size_t n, Pred pred, F f) {
#if MANIFOLD_PAR == 1
    if (n > 1e5) {
      run_par(n, pred, f);
    } else
#endif
    {
      run_seq(n, pred, f);
    }
  }
};

}  // namespace

namespace manifold {

/**
 * Duplicates just enough verts to covert an even-manifold to a proper
 * 2-manifold, splitting non-manifold verts and edges with too many triangles.
 */
void Manifold::Impl::CleanupTopology() {
  if (!halfedge_.size()) return;
  DEBUG_ASSERT(IsManifold(), logicErr, "polygon mesh is not manifold!");

  // In the case of a very bad triangulation, it is possible to create pinched
  // verts. They must be removed before edge collapse.
  SplitPinchedVerts();
  DedupeEdges();
}

/**
 * Collapses degenerate triangles by removing edges shorter than tolerance_ and
 * any edge that is preceeded by an edge that joins the same two face relations.
 * It also performs edge swaps on the long edges of degenerate triangles, though
 * there are some configurations of degenerates that cannot be removed this way.
 *
 * Before collapsing edges, the mesh is checked for duplicate edges (more than
 * one pair of triangles sharing the same edge), which are removed by
 * duplicating one vert and adding two triangles. These degenerate triangles are
 * likely to be collapsed again in the subsequent simplification.
 *
 * Note when an edge collapse would result in something non-manifold, the
 * vertices are duplicated in such a way as to remove handles or separate
 * meshes, thus decreasing the Genus(). It only increases when meshes that have
 * collapsed to just a pair of triangles are removed entirely.
 *
 * Verts with index less than firstNewVert will be left uncollapsed. This is
 * zero by default so that everything can be collapsed.
 *
 * Rather than actually removing the edges, this step merely marks them for
 * removal, by setting vertPos to NaN and halfedge to {-1, -1, -1, -1}.
 */
void Manifold::Impl::SimplifyTopology(int firstNewVert) {
  if (!halfedge_.size()) return;

  CleanupTopology();
  CollapseShortEdges(firstNewVert);
  CollapseColinearEdges(firstNewVert);
  SwapDegenerates(firstNewVert);
}

void Manifold::Impl::RemoveDegenerates(int firstNewVert) {
  if (!halfedge_.size()) return;

  CleanupTopology();
  CollapseShortEdges(firstNewVert);
  SwapDegenerates(firstNewVert);
}

void Manifold::Impl::CollapseShortEdges(int firstNewVert) {
  ZoneScopedN("CollapseShortEdge");
  FlagStore s;
  size_t numFlagged = 0;
  const size_t nbEdges = halfedge_.size();

  std::vector<int> scratchBuffer;
  scratchBuffer.reserve(10);
  // Short edges get to skip several checks and hence remove more classes of
  // degenerate triangles than flagged edges do, but this could in theory lead
  // to error stacking where a vertex moves too far. For this reason this is
  // restricted to epsilon, rather than tolerance.
  ShortEdge se{halfedge_, vertPos_, epsilon_, firstNewVert};
  s.run(nbEdges, se, [&](size_t i) {
    const bool didCollapse = CollapseEdge(i, scratchBuffer);
    if (didCollapse) numFlagged++;
    scratchBuffer.resize(0);
  });

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose > 0 && numFlagged > 0) {
    std::cout << "collapsed " << numFlagged << " short edges" << std::endl;
  }
#endif
}

void Manifold::Impl::CollapseColinearEdges(int firstNewVert) {
  FlagStore s;
  size_t numFlagged = 0;
  const size_t nbEdges = halfedge_.size();
  std::vector<int> scratchBuffer;
  scratchBuffer.reserve(10);
  while (1) {
    ZoneScopedN("CollapseFlaggedEdge");
    numFlagged = 0;
    // Collapse colinear edges, but only remove new verts, i.e. verts with
    // index
    // >= firstNewVert. This is used to keep the Boolean from changing the
    // non-intersecting parts of the input meshes. Colinear is defined not by a
    // local check, but by the global MarkCoplanar function, which keeps this
    // from being vulnerable to error stacking.
    FlagEdge se{halfedge_, meshRelation_.triRef, firstNewVert};
    s.run(nbEdges, se, [&](size_t i) {
      const bool didCollapse = CollapseEdge(i, scratchBuffer);
      if (didCollapse) numFlagged++;
      scratchBuffer.resize(0);
    });
    if (numFlagged == 0) break;

#ifdef MANIFOLD_DEBUG
    if (ManifoldParams().verbose > 0 && numFlagged > 0) {
      std::cout << "collapsed " << numFlagged << " colinear edges" << std::endl;
    }
#endif
  }
}

void Manifold::Impl::SwapDegenerates(int firstNewVert) {
  ZoneScopedN("RecursiveEdgeSwap");
  FlagStore s;
  size_t numFlagged = 0;
  const size_t nbEdges = halfedge_.size();
  std::vector<int> scratchBuffer;
  scratchBuffer.reserve(10);

  SwappableEdge se{halfedge_, vertPos_, faceNormal_, tolerance_, firstNewVert};
  std::vector<int> edgeSwapStack;
  std::vector<int> visited(halfedge_.size(), -1);
  int tag = 0;
  s.run(nbEdges, se, [&](size_t i) {
    numFlagged++;
    tag++;
    RecursiveEdgeSwap(i, tag, visited, edgeSwapStack, scratchBuffer);
    while (!edgeSwapStack.empty()) {
      int last = edgeSwapStack.back();
      edgeSwapStack.pop_back();
      RecursiveEdgeSwap(last, tag, visited, edgeSwapStack, scratchBuffer);
    }
  });

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose > 0 && numFlagged > 0) {
    std::cout << "swapped " << numFlagged << " edges" << std::endl;
  }
#endif
}

// Deduplicate the given 4-manifold edge by duplicating endVert, thus making the
// edges distinct. Also duplicates startVert if it becomes pinched.
void Manifold::Impl::DedupeEdge(const int edge) {
  // Orbit endVert
  const int startVert = halfedge_[edge].startVert;
  const int endVert = halfedge_[edge].endVert;
  const int endProp = halfedge_[NextHalfedge(edge)].propVert;
  int current = halfedge_[NextHalfedge(edge)].pairedHalfedge;
  while (current != edge) {
    const int vert = halfedge_[current].startVert;
    if (vert == startVert) {
      // Single topological unit needs 2 faces added to be split
      const int newVert = vertPos_.size();
      vertPos_.push_back(vertPos_[endVert]);
      if (vertNormal_.size() > 0) vertNormal_.push_back(vertNormal_[endVert]);
      current = halfedge_[NextHalfedge(current)].pairedHalfedge;
      const int opposite = halfedge_[NextHalfedge(edge)].pairedHalfedge;

      UpdateVert(newVert, current, opposite);

      int newHalfedge = halfedge_.size();
      int oldFace = current / 3;
      int outsideVert = halfedge_[current].startVert;
      halfedge_.push_back({endVert, newVert, -1, endProp});
      halfedge_.push_back({newVert, outsideVert, -1, endProp});
      halfedge_.push_back(
          {outsideVert, endVert, -1, halfedge_[current].propVert});
      PairUp(newHalfedge + 2, halfedge_[current].pairedHalfedge);
      PairUp(newHalfedge + 1, current);
      if (meshRelation_.triRef.size() > 0)
        meshRelation_.triRef.push_back(meshRelation_.triRef[oldFace]);
      if (faceNormal_.size() > 0) faceNormal_.push_back(faceNormal_[oldFace]);

      newHalfedge += 3;
      oldFace = opposite / 3;
      outsideVert = halfedge_[opposite].startVert;
      halfedge_.push_back({newVert, endVert, -1, endProp});  // fix prop
      halfedge_.push_back({endVert, outsideVert, -1, endProp});
      halfedge_.push_back(
          {outsideVert, newVert, -1, halfedge_[opposite].propVert});
      PairUp(newHalfedge + 2, halfedge_[opposite].pairedHalfedge);
      PairUp(newHalfedge + 1, opposite);
      PairUp(newHalfedge, newHalfedge - 3);
      if (meshRelation_.triRef.size() > 0)
        meshRelation_.triRef.push_back(meshRelation_.triRef[oldFace]);
      if (faceNormal_.size() > 0) faceNormal_.push_back(faceNormal_[oldFace]);

      break;
    }

    current = halfedge_[NextHalfedge(current)].pairedHalfedge;
  }

  if (current == edge) {
    // Separate topological unit needs no new faces to be split
    const int newVert = vertPos_.size();
    vertPos_.push_back(vertPos_[endVert]);
    if (vertNormal_.size() > 0) vertNormal_.push_back(vertNormal_[endVert]);

    ForVert(NextHalfedge(current), [this, newVert](int e) {
      halfedge_[e].startVert = newVert;
      halfedge_[halfedge_[e].pairedHalfedge].endVert = newVert;
    });
  }

  // Orbit startVert
  const int pair = halfedge_[edge].pairedHalfedge;
  current = halfedge_[NextHalfedge(pair)].pairedHalfedge;
  while (current != pair) {
    const int vert = halfedge_[current].startVert;
    if (vert == endVert) {
      break;  // Connected: not a pinched vert
    }
    current = halfedge_[NextHalfedge(current)].pairedHalfedge;
  }

  if (current == pair) {
    // Split the pinched vert the previous split created.
    const int newVert = vertPos_.size();
    vertPos_.push_back(vertPos_[endVert]);
    if (vertNormal_.size() > 0) vertNormal_.push_back(vertNormal_[endVert]);

    ForVert(NextHalfedge(current), [this, newVert](int e) {
      halfedge_[e].startVert = newVert;
      halfedge_[halfedge_[e].pairedHalfedge].endVert = newVert;
    });
  }
}

void Manifold::Impl::PairUp(int edge0, int edge1) {
  halfedge_[edge0].pairedHalfedge = edge1;
  halfedge_[edge1].pairedHalfedge = edge0;
}

// Traverses CW around startEdge.endVert from startEdge to endEdge
// (edgeEdge.endVert must == startEdge.endVert), updating each edge to point
// to vert instead.
void Manifold::Impl::UpdateVert(int vert, int startEdge, int endEdge) {
  int current = startEdge;
  while (current != endEdge) {
    halfedge_[current].endVert = vert;
    current = NextHalfedge(current);
    halfedge_[current].startVert = vert;
    current = halfedge_[current].pairedHalfedge;
    DEBUG_ASSERT(current != startEdge, logicErr, "infinite loop in decimator!");
  }
}

// In the event that the edge collapse would create a non-manifold edge,
// instead we duplicate the two verts and attach the manifolds the other way
// across this edge.
void Manifold::Impl::FormLoop(int current, int end) {
  int startVert = vertPos_.size();
  vertPos_.push_back(vertPos_[halfedge_[current].startVert]);
  int endVert = vertPos_.size();
  vertPos_.push_back(vertPos_[halfedge_[current].endVert]);

  int oldMatch = halfedge_[current].pairedHalfedge;
  int newMatch = halfedge_[end].pairedHalfedge;

  UpdateVert(startVert, oldMatch, newMatch);
  UpdateVert(endVert, end, current);

  halfedge_[current].pairedHalfedge = newMatch;
  halfedge_[newMatch].pairedHalfedge = current;
  halfedge_[end].pairedHalfedge = oldMatch;
  halfedge_[oldMatch].pairedHalfedge = end;

  RemoveIfFolded(end);
}

void Manifold::Impl::CollapseTri(const ivec3& triEdge) {
  if (halfedge_[triEdge[1]].pairedHalfedge == -1) return;
  int pair1 = halfedge_[triEdge[1]].pairedHalfedge;
  int pair2 = halfedge_[triEdge[2]].pairedHalfedge;
  halfedge_[pair1].pairedHalfedge = pair2;
  halfedge_[pair2].pairedHalfedge = pair1;
  for (int i : {0, 1, 2}) {
    halfedge_[triEdge[i]] = {-1, -1, -1, halfedge_[triEdge[i]].propVert};
  }
}

void Manifold::Impl::RemoveIfFolded(int edge) {
  const ivec3 tri0edge = TriOf(edge);
  const ivec3 tri1edge = TriOf(halfedge_[edge].pairedHalfedge);
  if (halfedge_[tri0edge[1]].pairedHalfedge == -1) return;
  if (halfedge_[tri0edge[1]].endVert == halfedge_[tri1edge[1]].endVert) {
    if (halfedge_[tri0edge[1]].pairedHalfedge == tri1edge[2]) {
      if (halfedge_[tri0edge[2]].pairedHalfedge == tri1edge[1]) {
        for (int i : {0, 1, 2})
          vertPos_[halfedge_[tri0edge[i]].startVert] = vec3(NAN);
      } else {
        vertPos_[halfedge_[tri0edge[1]].startVert] = vec3(NAN);
      }
    } else {
      if (halfedge_[tri0edge[2]].pairedHalfedge == tri1edge[1]) {
        vertPos_[halfedge_[tri1edge[1]].startVert] = vec3(NAN);
      }
    }
    PairUp(halfedge_[tri0edge[1]].pairedHalfedge,
           halfedge_[tri1edge[2]].pairedHalfedge);
    PairUp(halfedge_[tri0edge[2]].pairedHalfedge,
           halfedge_[tri1edge[1]].pairedHalfedge);
    for (int i : {0, 1, 2}) {
      halfedge_[tri0edge[i]] = {-1, -1, -1};
      halfedge_[tri1edge[i]] = {-1, -1, -1};
    }
  }
}

// Collapses the given edge by removing startVert - returns false if the edge
// cannot be collapsed. May split the mesh topologically if the collapse would
// have resulted in a 4-manifold edge. Do not collapse an edge if startVert is
// pinched - the vert would be marked NaN, but other edges could still be
// pointing to it.
bool Manifold::Impl::CollapseEdge(const int edge, std::vector<int>& edges) {
  Vec<TriRef>& triRef = meshRelation_.triRef;

  const Halfedge toRemove = halfedge_[edge];
  if (toRemove.pairedHalfedge < 0) return false;

  const int endVert = toRemove.endVert;
  const ivec3 tri0edge = TriOf(edge);
  const ivec3 tri1edge = TriOf(toRemove.pairedHalfedge);

  const vec3 pNew = vertPos_[endVert];
  const vec3 pOld = vertPos_[toRemove.startVert];
  const vec3 delta = pNew - pOld;
  const bool shortEdge = la::dot(delta, delta) < epsilon_ * epsilon_;

  // Orbit startVert
  int start = halfedge_[tri1edge[1]].pairedHalfedge;
  int current = tri1edge[2];
  if (!shortEdge) {
    current = start;
    TriRef refCheck = triRef[toRemove.pairedHalfedge / 3];
    vec3 pLast = vertPos_[halfedge_[tri1edge[1]].endVert];
    while (current != tri1edge[0]) {
      current = NextHalfedge(current);
      vec3 pNext = vertPos_[halfedge_[current].endVert];
      const int tri = current / 3;
      const TriRef ref = triRef[tri];
      const mat2x3 projection = GetAxisAlignedProjection(faceNormal_[tri]);
      // Don't collapse if the edge is not redundant (this may have changed due
      // to the collapse of neighbors).
      if (!ref.SameFace(refCheck)) {
        const TriRef oldRef = refCheck;
        refCheck = triRef[edge / 3];
        if (!ref.SameFace(refCheck)) {
          return false;
        }
        if (ref.meshID != oldRef.meshID || ref.faceID != oldRef.faceID ||
            la::dot(faceNormal_[toRemove.pairedHalfedge / 3],
                    faceNormal_[tri]) < -0.5) {
          // Restrict collapse to colinear edges when the edge separates faces
          // or the edge is sharp. This ensures large shifts are not introduced
          // parallel to the tangent plane.
          if (CCW(projection * pLast, projection * pOld, projection * pNew,
                  epsilon_) != 0)
            return false;
        }
      }

      // Don't collapse edge if it would cause a triangle to invert.
      if (CCW(projection * pNext, projection * pLast, projection * pNew,
              epsilon_) < 0)
        return false;

      pLast = pNext;
      current = halfedge_[current].pairedHalfedge;
    }
  }

  // Orbit endVert
  {
    int current = halfedge_[tri0edge[1]].pairedHalfedge;
    while (current != tri1edge[2]) {
      current = NextHalfedge(current);
      edges.push_back(current);
      current = halfedge_[current].pairedHalfedge;
    }
  }

  // Remove toRemove.startVert and replace with endVert.
  vertPos_[toRemove.startVert] = vec3(NAN);
  CollapseTri(tri1edge);

  // Orbit startVert
  const int tri0 = edge / 3;
  const int tri1 = toRemove.pairedHalfedge / 3;
  current = start;
  while (current != tri0edge[2]) {
    current = NextHalfedge(current);

    if (NumProp() > 0) {
      // Update the shifted triangles to the vertBary of endVert
      const int tri = current / 3;
      if (triRef[tri].SameFace(triRef[tri0])) {
        halfedge_[current].propVert = halfedge_[NextHalfedge(edge)].propVert;
      } else if (triRef[tri].SameFace(triRef[tri1])) {
        halfedge_[current].propVert =
            halfedge_[toRemove.pairedHalfedge].propVert;
      }
    }

    const int vert = halfedge_[current].endVert;
    const int next = halfedge_[current].pairedHalfedge;
    for (size_t i = 0; i < edges.size(); ++i) {
      if (vert == halfedge_[edges[i]].endVert) {
        FormLoop(edges[i], current);
        start = next;
        edges.resize(i);
        break;
      }
    }
    current = next;
  }

  UpdateVert(endVert, start, tri0edge[2]);
  CollapseTri(tri0edge);
  RemoveIfFolded(start);
  return true;
}

void Manifold::Impl::RecursiveEdgeSwap(const int edge, int& tag,
                                       std::vector<int>& visited,
                                       std::vector<int>& edgeSwapStack,
                                       std::vector<int>& edges) {
  Vec<TriRef>& triRef = meshRelation_.triRef;

  if (edge < 0) return;
  const int pair = halfedge_[edge].pairedHalfedge;
  if (pair < 0) return;

  // avoid infinite recursion
  if (visited[edge] == tag && visited[pair] == tag) return;

  const ivec3 tri0edge = TriOf(edge);
  const ivec3 tri1edge = TriOf(pair);

  mat2x3 projection = GetAxisAlignedProjection(faceNormal_[edge / 3]);
  vec2 v[4];
  for (int i : {0, 1, 2})
    v[i] = projection * vertPos_[halfedge_[tri0edge[i]].startVert];
  // Only operate on the long edge of a degenerate triangle.
  if (CCW(v[0], v[1], v[2], tolerance_) > 0 || !Is01Longest(v[0], v[1], v[2]))
    return;

  // Switch to neighbor's projection.
  projection = GetAxisAlignedProjection(faceNormal_[pair / 3]);
  for (int i : {0, 1, 2})
    v[i] = projection * vertPos_[halfedge_[tri0edge[i]].startVert];
  v[3] = projection * vertPos_[halfedge_[tri1edge[2]].startVert];

  auto SwapEdge = [&]() {
    // The 0-verts are swapped to the opposite 2-verts.
    const int v0 = halfedge_[tri0edge[2]].startVert;
    const int v1 = halfedge_[tri1edge[2]].startVert;
    halfedge_[tri0edge[0]].startVert = v1;
    halfedge_[tri0edge[2]].endVert = v1;
    halfedge_[tri1edge[0]].startVert = v0;
    halfedge_[tri1edge[2]].endVert = v0;
    PairUp(tri0edge[0], halfedge_[tri1edge[2]].pairedHalfedge);
    PairUp(tri1edge[0], halfedge_[tri0edge[2]].pairedHalfedge);
    PairUp(tri0edge[2], tri1edge[2]);
    // Both triangles are now subsets of the neighboring triangle.
    const int tri0 = tri0edge[0] / 3;
    const int tri1 = tri1edge[0] / 3;
    faceNormal_[tri0] = faceNormal_[tri1];
    triRef[tri0] = triRef[tri1];
    const double l01 = la::length(v[1] - v[0]);
    const double l02 = la::length(v[2] - v[0]);
    const double a = std::max(0.0, std::min(1.0, l02 / l01));
    // Update properties if applicable
    if (properties_.size() > 0) {
      Vec<double>& prop = properties_;
      halfedge_[tri0edge[1]].propVert = halfedge_[tri1edge[0]].propVert;
      halfedge_[tri0edge[0]].propVert = halfedge_[tri1edge[2]].propVert;
      halfedge_[tri0edge[2]].propVert = halfedge_[tri1edge[2]].propVert;
      const int numProp = NumProp();
      const int newProp = prop.size() / numProp;
      const int propIdx0 = halfedge_[tri1edge[0]].propVert;
      const int propIdx1 = halfedge_[tri1edge[1]].propVert;
      for (int p = 0; p < numProp; ++p) {
        prop.push_back(a * prop[numProp * propIdx0 + p] +
                       (1 - a) * prop[numProp * propIdx1 + p]);
      }
      halfedge_[tri1edge[0]].propVert = newProp;
      halfedge_[tri0edge[2]].propVert = newProp;
    }

    // if the new edge already exists, duplicate the verts and split the mesh.
    int current = halfedge_[tri1edge[0]].pairedHalfedge;
    const int endVert = halfedge_[tri1edge[1]].endVert;
    while (current != tri0edge[1]) {
      current = NextHalfedge(current);
      if (halfedge_[current].endVert == endVert) {
        FormLoop(tri0edge[2], current);
        RemoveIfFolded(tri0edge[2]);
        return;
      }
      current = halfedge_[current].pairedHalfedge;
    }
  };

  // Only operate if the other triangles are not degenerate.
  if (CCW(v[1], v[0], v[3], tolerance_) <= 0) {
    if (!Is01Longest(v[1], v[0], v[3])) return;
    // Two facing, long-edge degenerates can swap.
    SwapEdge();
    const vec2 e23 = v[3] - v[2];
    if (la::dot(e23, e23) < tolerance_ * tolerance_) {
      tag++;
      CollapseEdge(tri0edge[2], edges);
      edges.resize(0);
    } else {
      visited[edge] = tag;
      visited[pair] = tag;
      edgeSwapStack.insert(edgeSwapStack.end(), {tri1edge[1], tri1edge[0],
                                                 tri0edge[1], tri0edge[0]});
    }
    return;
  } else if (CCW(v[0], v[3], v[2], tolerance_) <= 0 ||
             CCW(v[1], v[2], v[3], tolerance_) <= 0) {
    return;
  }
  // Normal path
  SwapEdge();
  visited[edge] = tag;
  visited[pair] = tag;
  edgeSwapStack.insert(edgeSwapStack.end(),
                       {halfedge_[tri1edge[0]].pairedHalfedge,
                        halfedge_[tri0edge[1]].pairedHalfedge});
}

void Manifold::Impl::SplitPinchedVerts() {
  ZoneScoped;

  auto nbEdges = halfedge_.size();
#if MANIFOLD_PAR == 1
  if (nbEdges > 1e4) {
    std::mutex mutex;
    std::vector<size_t> pinched;
    // This parallelized version is non-trivial so we can't reuse the code
    //
    // The idea here is to identify cycles of halfedges that can be iterated
    // through using ForVert. Pinched verts are vertices where there are
    // multiple cycles associated with the vertex. Each cycle is identified with
    // the largest halfedge index within the cycle, and when there are multiple
    // cycles associated with the same starting vertex but with different ids,
    // it means we have a pinched vertex. This check is done by using a single
    // atomic cas operation, the expected case is either invalid id (the vertex
    // was not processed) or with the same id.
    //
    // The local store is to store the processed halfedges, so to avoid
    // repetitive processing. Note that it only approximates the processed
    // halfedges because it is thread local. This is why we need a vector to
    // deduplicate the probematic halfedges we found.
    std::vector<std::atomic<size_t>> largestEdge(NumVert());
    for_each(ExecutionPolicy::Par, countAt(0), countAt(NumVert()),
             [&largestEdge](size_t i) {
               largestEdge[i].store(std::numeric_limits<size_t>::max());
             });
    tbb::combinable<std::vector<bool>> store(
        [nbEdges]() { return std::vector<bool>(nbEdges, false); });
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, nbEdges),
        [&store, &mutex, &pinched, &largestEdge, this](const auto& r) {
          auto& local = store.local();
          std::vector<size_t> pinchedLocal;
          for (auto i = r.begin(); i < r.end(); ++i) {
            if (local[i]) continue;
            local[i] = true;
            const int vert = halfedge_[i].startVert;
            if (vert == -1) continue;
            size_t largest = i;
            ForVert(i, [&local, &largest](int current) {
              local[current] = true;
              largest = std::max(largest, static_cast<size_t>(current));
            });
            size_t expected = std::numeric_limits<size_t>::max();
            if (!largestEdge[vert].compare_exchange_strong(expected, largest) &&
                expected != largest) {
              // we know that there is another loop...
              pinchedLocal.push_back(largest);
            }
          }
          if (!pinchedLocal.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            pinched.insert(pinched.end(), pinchedLocal.begin(),
                           pinchedLocal.end());
          }
        });

    manifold::stable_sort(pinched.begin(), pinched.end());
    std::vector<bool> halfedgeProcessed(nbEdges, false);
    for (size_t i : pinched) {
      if (halfedgeProcessed[i]) continue;
      vertPos_.push_back(vertPos_[halfedge_[i].startVert]);
      const int vert = NumVert() - 1;
      ForVert(i, [this, vert, &halfedgeProcessed](int current) {
        halfedgeProcessed[current] = true;
        halfedge_[current].startVert = vert;
        halfedge_[halfedge_[current].pairedHalfedge].endVert = vert;
      });
    }
  } else
#endif
  {
    std::vector<bool> vertProcessed(NumVert(), false);
    std::vector<bool> halfedgeProcessed(nbEdges, false);
    for (size_t i = 0; i < nbEdges; ++i) {
      if (halfedgeProcessed[i]) continue;
      int vert = halfedge_[i].startVert;
      if (vert == -1) continue;
      if (vertProcessed[vert]) {
        vertPos_.push_back(vertPos_[vert]);
        vert = NumVert() - 1;
      } else {
        vertProcessed[vert] = true;
      }
      ForVert(i, [this, &halfedgeProcessed, vert](int current) {
        halfedgeProcessed[current] = true;
        halfedge_[current].startVert = vert;
        halfedge_[halfedge_[current].pairedHalfedge].endVert = vert;
      });
    }
  }
}

void Manifold::Impl::DedupeEdges() {
  while (1) {
    ZoneScopedN("DedupeEdge");

    const size_t nbEdges = halfedge_.size();
    std::vector<size_t> duplicates;
    auto localLoop = [&](size_t start, size_t end, std::vector<bool>& local,
                         std::vector<size_t>& results) {
      // Iterate over all halfedges that start with the same vertex, and check
      // for halfedges with the same ending vertex.
      // Note: we use Vec and linear search when the number of neighbor is
      // small because unordered_set requires allocations and is expensive.
      // We switch to unordered_set when the number of neighbor is
      // larger to avoid making things quadratic.
      // We do it in two pass, the first pass to find the minimal halfedges with
      // the target start and end verts, the second pass flag all the duplicated
      // halfedges that are not having the minimal index as duplicates.
      // This ensures deterministic result.
      //
      // The local store is to store the processed halfedges, so to avoid
      // repetitive processing. Note that it only approximates the processed
      // halfedges because it is thread local.
      Vec<std::pair<int, int>> endVerts;
      std::unordered_map<int, int> endVertSet;
      for (auto i = start; i < end; ++i) {
        if (local[i] || halfedge_[i].startVert == -1 ||
            halfedge_[i].endVert == -1)
          continue;
        // we want to keep the allocation
        endVerts.clear(false);
        endVertSet.clear();

        // first iteration, populate entries
        // this makes sure we always report the same set of entries
        ForVert(i, [&local, &endVerts, &endVertSet, this](int current) {
          local[current] = true;
          if (halfedge_[current].startVert == -1 ||
              halfedge_[current].endVert == -1) {
            return;
          }
          int endV = halfedge_[current].endVert;
          if (endVertSet.empty()) {
            auto iter = std::find_if(endVerts.begin(), endVerts.end(),
                                     [endV](const std::pair<int, int>& pair) {
                                       return pair.first == endV;
                                     });
            if (iter != endVerts.end()) {
              iter->second = std::min(iter->second, current);
            } else {
              endVerts.push_back({endV, current});
              if (endVerts.size() > 32) {
                endVertSet.insert(endVerts.begin(), endVerts.end());
                endVerts.clear(false);
              }
            }
          } else {
            auto pair = endVertSet.insert({endV, current});
            if (!pair.second)
              pair.first->second = std::min(pair.first->second, current);
          }
        });
        // second iteration, actually check for duplicates
        // we always report the same set of duplicates, excluding the smallest
        // halfedge in the set of duplicates
        ForVert(i, [&endVerts, &endVertSet, &results, this](int current) {
          if (halfedge_[current].startVert == -1 ||
              halfedge_[current].endVert == -1) {
            return;
          }
          int endV = halfedge_[current].endVert;
          if (endVertSet.empty()) {
            auto iter = std::find_if(endVerts.begin(), endVerts.end(),
                                     [endV](const std::pair<int, int>& pair) {
                                       return pair.first == endV;
                                     });
            if (iter->second != current) results.push_back(current);
          } else {
            auto iter = endVertSet.find(endV);
            if (iter->second != current) results.push_back(current);
          }
        });
      }
    };
#if MANIFOLD_PAR == 1
    if (nbEdges > 1e4) {
      std::mutex mutex;
      tbb::combinable<std::vector<bool>> store(
          [nbEdges]() { return std::vector<bool>(nbEdges, false); });
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, nbEdges),
          [&store, &mutex, &duplicates, &localLoop](const auto& r) {
            auto& local = store.local();
            std::vector<size_t> duplicatesLocal;
            localLoop(r.begin(), r.end(), local, duplicatesLocal);
            if (!duplicatesLocal.empty()) {
              std::lock_guard<std::mutex> lock(mutex);
              duplicates.insert(duplicates.end(), duplicatesLocal.begin(),
                                duplicatesLocal.end());
            }
          });
      manifold::stable_sort(duplicates.begin(), duplicates.end());
      duplicates.resize(
          std::distance(duplicates.begin(),
                        std::unique(duplicates.begin(), duplicates.end())));
    } else
#endif
    {
      std::vector<bool> local(nbEdges, false);
      localLoop(0, nbEdges, local, duplicates);
    }

    size_t numFlagged = 0;
    for (size_t i : duplicates) {
      DedupeEdge(i);
      numFlagged++;
    }

    if (numFlagged == 0) break;

#ifdef MANIFOLD_DEBUG
    if (ManifoldParams().verbose > 0) {
      std::cout << "found " << numFlagged << " duplicate edges to split"
                << std::endl;
    }
#endif
  }
}
}  // namespace manifold
