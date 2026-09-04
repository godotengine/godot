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
#include <unordered_set>

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

struct FlagStore {
#if MANIFOLD_PAR == 1
  tbb::combinable<Vec<size_t>> store;
#endif
  Vec<size_t> s;

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
                        for (auto i = r.begin(); i < r.end(); i++)
                          if (pred(i)) local.push_back(i);
                      });

    std::vector<Vec<size_t>> stores;
    Vec<size_t> result;
    store.combine_each(
        [&](auto& data) { stores.emplace_back(std::move(data)); });
    Vec<size_t> sizes;
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
  halfedge_.MakeUnique();

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
  halfedge_.MakeUnique();

  CleanupTopology();
  CollapseShortEdges(firstNewVert);
  CollapseColinearEdges(firstNewVert);
  SwapDegenerates(firstNewVert);
  // Merging verts causes their normals to change
  CalculateVertNormals();
}

void Manifold::Impl::RemoveDegenerates(int firstNewVert) {
  if (!halfedge_.size()) return;
  halfedge_.MakeUnique();

  CleanupTopology();
  CollapseShortEdges(firstNewVert);
  SwapDegenerates(firstNewVert);
  // Merging verts causes their normals to change
  CalculateVertNormals();
}

void Manifold::Impl::CollapseShortEdges(int firstNewVert) {
  ZoneScopedN("CollapseShortEdge");
  FlagStore s;
  size_t numFlagged = 0;
  const size_t nbEdges = halfedge_.size();

  Vec<int> scratchBuffer;
  scratchBuffer.reserve(10);
  // Short edges get to skip several checks and hence remove more classes of
  // degenerate triangles than flagged edges do, but this could in theory lead
  // to error stacking where a vertex moves too far. For this reason this is
  // restricted to epsilon, rather than tolerance. However, in the case of a
  // Boolean operation, we set firstNewVert in order to only operate on
  // newly-created verts, which means error stacking is not a concern, so we
  // allow collapsing up to tolerance in that case.
  const double tol = firstNewVert == 0 ? epsilon_ : tolerance_;

  auto shortEdge = [&](int edge) {
    const int pair = halfedge_.Pair(edge);
    if (pair < 0) return false;
    const int start = halfedge_.Start(edge);
    const int end = halfedge_.End(edge);
    if (start < firstNewVert && end < firstNewVert) return false;
    // Flag short edges
    const vec3 delta = vertPos_[end] - vertPos_[start];
    const double lenSq = la::dot(delta, delta);
    // To ensure tolerance_-scale errors don't stack, only collapse these edges
    // if they connect a new vert to an old vert, since old verts are only
    // allowed to move by epsilon_.
    const double maxLen = end < firstNewVert ? tol * tol : epsilon_ * epsilon_;
    return lenSq < maxLen;
  };

  s.run(nbEdges, shortEdge, [&](size_t i) {
    const bool didCollapse = CollapseEdge(i, scratchBuffer, tol, firstNewVert);
    if (didCollapse) numFlagged++;
    scratchBuffer.resize(0);
  });

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().verbose >= 2 && numFlagged > 0) {
    std::cout << "collapsed " << numFlagged << " short edges" << std::endl;
  }
#endif
}

void Manifold::Impl::CollapseColinearEdges(int firstNewVert) {
  FlagStore s;
  size_t numFlagged = 0;
  const size_t nbEdges = halfedge_.size();
  Vec<int> scratchBuffer;
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
    auto colinearEdge = [&](int edge) {
      const int pair = halfedge_.Pair(edge);
      if (pair < 0 || halfedge_.Start(edge) < firstNewVert) return false;
      // Flag redundant edges - those where the startVert is surrounded by only
      // two original triangles.
      const TriRef ref0 = meshRelation_.triRef[edge / 3];
      int current = NextHalfedge(pair);
      TriRef ref1 = meshRelation_.triRef[current / 3];
      bool ref1Updated = !ref0.SameFace(ref1);
      while (current != edge) {
        current = NextHalfedge(halfedge_.Pair(current));
        int tri = current / 3;
        const TriRef ref = meshRelation_.triRef[tri];
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
    };

    s.run(nbEdges, colinearEdge, [&](size_t i) {
      const bool didCollapse = CollapseEdge(i, scratchBuffer);
      if (didCollapse) numFlagged++;
      scratchBuffer.resize(0);
    });
    if (numFlagged == 0) break;

#ifdef MANIFOLD_DEBUG
    if (ManifoldParams().verbose >= 2 && numFlagged > 0) {
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
  Vec<int> scratchBuffer;
  scratchBuffer.reserve(10);

  auto swappableEdge = [&](int edge) {
    const int pair = halfedge_.Pair(edge);
    if (pair < 0) return false;
    const ivec3 triEdge = TriOf(edge);
    const ivec3 pairTriEdge = TriOf(pair);
    if (halfedge_.Start(triEdge[0]) < firstNewVert &&
        halfedge_.Start(triEdge[1]) < firstNewVert &&
        halfedge_.Start(triEdge[2]) < firstNewVert &&
        halfedge_.Start(pairTriEdge[2]) < firstNewVert)
      return false;

    int tri = edge / 3;
    mat2x3 projection = GetAxisAlignedProjection(faceNormal_[tri]);
    vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos_[halfedge_.Start(triEdge[i])];
    if (CCW(v[0], v[1], v[2], tolerance_) > 0 || !Is01Longest(v[0], v[1], v[2]))
      return false;

    // Switch to neighbor's projection.
    edge = pair;
    tri = edge / 3;
    projection = GetAxisAlignedProjection(faceNormal_[tri]);
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos_[halfedge_.Start(pairTriEdge[i])];
    return CCW(v[0], v[1], v[2], tolerance_) > 0 ||
           Is01Longest(v[0], v[1], v[2]);
  };

  Vec<int> edgeSwapStack;
  Vec<int> visited(halfedge_.size(), -1);
  int tag = 0;
  s.run(nbEdges, swappableEdge, [&](size_t i) {
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
  if (ManifoldParams().verbose >= 2 && numFlagged > 0) {
    std::cout << "swapped " << numFlagged << " edges" << std::endl;
  }
#endif
}

// Deduplicate the given 4-manifold edge by duplicating endVert, thus making the
// edges distinct. Also duplicates startVert if it becomes pinched.
void Manifold::Impl::DedupeEdge(const int edge) {
  // Orbit endVert
  const int nextEdge = NextHalfedge(edge);
  const int startVert = halfedge_.Start(edge);
  const int endVert = halfedge_.Start(nextEdge);
  const int endProp = halfedge_.Prop(nextEdge);
  int current = halfedge_.Pair(nextEdge);
  while (current != edge) {
    const int vert = halfedge_.Start(current);
    if (vert == startVert) {
      // Single topological unit needs 2 faces added to be split
      const int newVert = vertPos_.size();
      vertPos_.push_back(vertPos_[endVert]);
      if (vertNormal_.size() > 0) vertNormal_.push_back(vertNormal_[endVert]);
      current = halfedge_.Pair(NextHalfedge(current));
      const int opposite = halfedge_.Pair(nextEdge);

      UpdateVert(newVert, current, opposite);

      int newHalfedge = halfedge_.size();
      int oldFace = current / 3;
      int outsideVert = halfedge_.Start(current);
      halfedge_.push_back(endVert, -1, endProp);
      halfedge_.push_back(newVert, -1, endProp);
      halfedge_.push_back(outsideVert, -1, halfedge_.Prop(current));
      PairUp(newHalfedge + 2, halfedge_.Pair(current));
      PairUp(newHalfedge + 1, current);
      if (meshRelation_.triRef.size() > 0)
        meshRelation_.triRef.push_back(meshRelation_.triRef[oldFace]);
      if (faceNormal_.size() > 0) faceNormal_.push_back(faceNormal_[oldFace]);

      newHalfedge += 3;
      oldFace = opposite / 3;
      outsideVert = halfedge_.Start(opposite);
      halfedge_.push_back(newVert, -1, endProp);  // fix prop
      halfedge_.push_back(endVert, -1, endProp);
      halfedge_.push_back(outsideVert, -1, halfedge_.Prop(opposite));
      PairUp(newHalfedge + 2, halfedge_.Pair(opposite));
      PairUp(newHalfedge + 1, opposite);
      PairUp(newHalfedge, newHalfedge - 3);
      if (meshRelation_.triRef.size() > 0)
        meshRelation_.triRef.push_back(meshRelation_.triRef[oldFace]);
      if (faceNormal_.size() > 0) faceNormal_.push_back(faceNormal_[oldFace]);

      break;
    }

    current = halfedge_.Pair(NextHalfedge(current));
  }

  if (current == edge) {
    // Separate topological unit needs no new faces to be split
    const int newVert = vertPos_.size();
    vertPos_.push_back(vertPos_[endVert]);
    if (vertNormal_.size() > 0) vertNormal_.push_back(vertNormal_[endVert]);

    ForVert(NextHalfedge(current), [this, newVert](int e) {
      halfedge_.SetStart(e, newVert);
      halfedge_.SetEnd(halfedge_.Pair(e), newVert);
    });
  }

  // Orbit startVert
  const int pair = halfedge_.Pair(edge);
  current = halfedge_.Pair(NextHalfedge(pair));
  while (current != pair) {
    const int vert = halfedge_.Start(current);
    if (vert == endVert) {
      break;  // Connected: not a pinched vert
    }
    current = halfedge_.Pair(NextHalfedge(current));
  }

  if (current == pair) {
    // Split the pinched vert the previous split created.
    const int newVert = vertPos_.size();
    vertPos_.push_back(vertPos_[endVert]);
    if (vertNormal_.size() > 0) vertNormal_.push_back(vertNormal_[endVert]);

    ForVert(NextHalfedge(current), [this, newVert](int e) {
      halfedge_.SetStart(e, newVert);
      halfedge_.SetEnd(halfedge_.Pair(e), newVert);
    });
  }
}

void Manifold::Impl::PairUp(int edge0, int edge1) {
  halfedge_.SetPair(edge0, edge1);
  halfedge_.SetPair(edge1, edge0);
}

// Traverses CW around startEdge.endVert from startEdge to endEdge
// (edgeEdge.endVert must == startEdge.endVert), updating each edge to point
// to vert instead.
void Manifold::Impl::UpdateVert(int vert, int startEdge, int endEdge) {
  int current = startEdge;
  while (current != endEdge) {
    halfedge_.SetEnd(current, vert);
    current = NextHalfedge(current);
    halfedge_.SetStart(current, vert);
    current = halfedge_.Pair(current);
    DEBUG_ASSERT(current != startEdge, logicErr, "infinite loop in decimator!");
  }
}

// In the event that the edge collapse would create a non-manifold edge,
// instead we duplicate the two verts and attach the manifolds the other way
// across this edge.
void Manifold::Impl::FormLoop(int current, int end) {
  int startVert = vertPos_.size();
  vertPos_.push_back(vertPos_[halfedge_.Start(current)]);
  int endVert = vertPos_.size();
  vertPos_.push_back(vertPos_[halfedge_.End(current)]);

  int oldMatch = halfedge_.Pair(current);
  int newMatch = halfedge_.Pair(end);

  UpdateVert(startVert, oldMatch, newMatch);
  UpdateVert(endVert, end, current);

  PairUp(current, newMatch);
  PairUp(end, oldMatch);

  RemoveIfFolded(end);
}

void Manifold::Impl::CollapseTri(const ivec3& triEdge) {
  if (halfedge_.Pair(triEdge[1]) == -1) return;
  int pair1 = halfedge_.Pair(triEdge[1]);
  int pair2 = halfedge_.Pair(triEdge[2]);
  PairUp(pair1, pair2);
  for (int i : {0, 1, 2}) {
    halfedge_.Set(triEdge[i], -1, -1, halfedge_.Prop(triEdge[i]));
  }
}

void Manifold::Impl::RemoveIfFolded(int edge) {
  const ivec3 tri0edge = TriOf(edge);
  const ivec3 tri1edge = TriOf(halfedge_.Pair(edge));
  if (halfedge_.Pair(tri0edge[1]) == -1) return;
  if (halfedge_.Start(tri0edge[2]) == halfedge_.Start(tri1edge[2])) {
    if (halfedge_.Pair(tri0edge[1]) == tri1edge[2]) {
      if (halfedge_.Pair(tri0edge[2]) == tri1edge[1]) {
        for (int i : {0, 1, 2})
          vertPos_[halfedge_.Start(tri0edge[i])] = vec3(NAN);
      } else {
        vertPos_[halfedge_.Start(tri0edge[1])] = vec3(NAN);
      }
    } else {
      if (halfedge_.Pair(tri0edge[2]) == tri1edge[1]) {
        vertPos_[halfedge_.Start(tri1edge[1])] = vec3(NAN);
      }
    }
    PairUp(halfedge_.Pair(tri0edge[1]), halfedge_.Pair(tri1edge[2]));
    PairUp(halfedge_.Pair(tri0edge[2]), halfedge_.Pair(tri1edge[1]));
    for (int i : {0, 1, 2}) {
      halfedge_.Set(tri0edge[i], -1, -1, -1);
      halfedge_.Set(tri1edge[i], -1, -1, -1);
    }
  }
}

// Collapses the given edge by removing startVert - returns false if the edge
// cannot be collapsed. May split the mesh topologically if the collapse would
// have resulted in a 4-manifold edge. Do not collapse an edge if startVert is
// pinched - the vert would be marked NaN, but other edges could still be
// pointing to it.
bool Manifold::Impl::CollapseEdge(const int edge, Vec<int>& edges, double tol,
                                  int firstNewVert) {
  Vec<TriRef>& triRef = meshRelation_.triRef;
  if (tol < 0) tol = epsilon_;

  const int pair = halfedge_.Pair(edge);
  if (pair < 0) return false;

  const ivec3 tri0edge = TriOf(edge);
  const ivec3 tri1edge = TriOf(pair);
  const int startVert = halfedge_.Start(tri0edge[0]);
  const int endVert = halfedge_.Start(tri0edge[1]);

  const vec3 pNew = vertPos_[endVert];
  const vec3 pOld = vertPos_[startVert];
  const vec3 delta = pNew - pOld;
  // We don't check that startVert is still new here - it may have been
  // collapsed to a different neighbor. However, it's still fine to collapse it
  // further, as it's still only collapsing its own original neighbors together,
  // which can't stack errors arbitrarily far.
  const double maxLen =
      endVert < firstNewVert ? tol * tol : epsilon_ * epsilon_;
  const bool shortEdge = la::dot(delta, delta) < maxLen;

  // Orbit startVert
  int start = halfedge_.Pair(tri1edge[1]);
  int current = tri1edge[2];
  if (!shortEdge) {
    current = start;
    TriRef refCheck = triRef[pair / 3];
    vec3 pLast = vertPos_[halfedge_.Start(tri1edge[2])];
    while (current != tri1edge[0]) {
      current = NextHalfedge(current);
      vec3 pNext = vertPos_[halfedge_.End(current)];
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
            la::dot(faceNormal_[pair / 3], faceNormal_[tri]) < -0.5) {
          // Restrict collapse to colinear edges when the edge separates faces
          // or the edge is sharp. This ensures large shifts are not introduced
          // parallel to the tangent plane.
          if (CCW(projection * pLast, projection * pOld, projection * pNew,
                  tol) != 0)
            return false;
        }
      }

      // Don't collapse edge if it would cause a triangle to invert.
      if (CCW(projection * pNext, projection * pLast, projection * pNew,
              epsilon_) < 0)
        return false;

      pLast = pNext;
      current = halfedge_.Pair(current);
    }
  }

  // Orbit endVert
  {
    int current = halfedge_.Pair(tri0edge[1]);
    while (current != tri1edge[2]) {
      current = NextHalfedge(current);
      edges.push_back(current);
      current = halfedge_.Pair(current);
    }
  }

  // Remove toRemove.startVert and replace with endVert.
  vertPos_[startVert] = vec3(NAN);
  CollapseTri(tri1edge);

  // Orbit startVert
  const int tri0 = edge / 3;
  const int tri1 = pair / 3;
  current = start;
  while (current != tri0edge[2]) {
    current = NextHalfedge(current);

    if (NumProp() > 0) {
      // Update the shifted triangles to the vertBary of endVert
      const int tri = current / 3;
      if (triRef[tri].SameFace(triRef[tri0])) {
        halfedge_.SetProp(current, halfedge_.Prop(NextHalfedge(edge)));
      } else if (triRef[tri].SameFace(triRef[tri1])) {
        halfedge_.SetProp(current, halfedge_.Prop(pair));
      }
    }

    const int vert = halfedge_.End(current);
    const int next = halfedge_.Pair(current);
    for (size_t i = 0; i < edges.size(); ++i) {
      if (vert == halfedge_.End(edges[i])) {
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
                                       Vec<int>& visited,
                                       Vec<int>& edgeSwapStack,
                                       Vec<int>& edges) {
  Vec<TriRef>& triRef = meshRelation_.triRef;

  if (edge < 0) return;
  const int pair = halfedge_.Pair(edge);
  if (pair < 0) return;

  // avoid infinite recursion
  if (visited[edge] == tag && visited[pair] == tag) return;

  const ivec3 tri0edge = TriOf(edge);
  const ivec3 tri1edge = TriOf(pair);

  mat2x3 projection = GetAxisAlignedProjection(faceNormal_[edge / 3]);
  vec2 v[4];
  for (int i : {0, 1, 2})
    v[i] = projection * vertPos_[halfedge_.Start(tri0edge[i])];
  // Only operate on the long edge of a degenerate triangle.
  if (CCW(v[0], v[1], v[2], tolerance_) > 0 || !Is01Longest(v[0], v[1], v[2]))
    return;

  // Switch to neighbor's projection.
  projection = GetAxisAlignedProjection(faceNormal_[pair / 3]);
  for (int i : {0, 1, 2})
    v[i] = projection * vertPos_[halfedge_.Start(tri0edge[i])];
  v[3] = projection * vertPos_[halfedge_.Start(tri1edge[2])];

  auto SwapEdge = [&]() {
    // The 0-verts are swapped to the opposite 2-verts.
    const int v0 = halfedge_.Start(tri0edge[2]);
    const int v1 = halfedge_.Start(tri1edge[2]);
    halfedge_.SetStart(tri0edge[0], v1);
    halfedge_.SetEnd(tri0edge[2], v1);
    halfedge_.SetStart(tri1edge[0], v0);
    halfedge_.SetEnd(tri1edge[2], v0);
    PairUp(tri0edge[0], halfedge_.Pair(tri1edge[2]));
    PairUp(tri1edge[0], halfedge_.Pair(tri0edge[2]));
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
      halfedge_.SetProp(tri0edge[1], halfedge_.Prop(tri1edge[0]));
      halfedge_.SetProp(tri0edge[0], halfedge_.Prop(tri1edge[2]));
      halfedge_.SetProp(tri0edge[2], halfedge_.Prop(tri1edge[2]));
      const int numProp = NumProp();
      const int newProp = prop.size() / numProp;
      const int propIdx0 = halfedge_.Prop(tri1edge[0]);
      const int propIdx1 = halfedge_.Prop(tri1edge[1]);
      for (int p = 0; p < numProp; ++p) {
        prop.push_back(a * prop[numProp * propIdx0 + p] +
                       (1 - a) * prop[numProp * propIdx1 + p]);
      }
      halfedge_.SetProp(tri1edge[0], newProp);
      halfedge_.SetProp(tri0edge[2], newProp);
    }

    // if the new edge already exists, duplicate the verts and split the mesh.
    int current = halfedge_.Pair(tri1edge[0]);
    const int endVert = halfedge_.End(tri1edge[1]);
    while (current != tri0edge[1]) {
      current = NextHalfedge(current);
      if (halfedge_.End(current) == endVert) {
        FormLoop(tri0edge[2], current);
        RemoveIfFolded(tri0edge[2]);
        return;
      }
      current = halfedge_.Pair(current);
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
      for (auto edge : {tri1edge[1], tri1edge[0], tri0edge[1], tri0edge[0]})
        edgeSwapStack.push_back(edge);
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
  for (auto edge : {halfedge_.Pair(tri1edge[0]), halfedge_.Pair(tri0edge[1])})
    edgeSwapStack.push_back(edge);
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
    // the smallest halfedge index within the cycle, and when there are multiple
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
    for_each(ExecutionPolicy::Par, countAt(0_uz), countAt(NumVert()),
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
            const int vert = halfedge_.Start(i);
            if (vert == -1) continue;
            size_t smallest = i;
            ForVert(i, [&local, &smallest](int current) {
              local[current] = true;
              smallest = std::min(smallest, static_cast<size_t>(current));
            });
            size_t got = std::numeric_limits<size_t>::max();
            if (!largestEdge[vert].compare_exchange_strong(got, smallest) &&
                got != smallest) {
              // we know that there is another loop...
              // put all to pinched
              pinchedLocal.push_back(smallest);
              pinchedLocal.push_back(got);
            }
          }
          if (!pinchedLocal.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            pinched.insert(pinched.end(), pinchedLocal.begin(),
                           pinchedLocal.end());
          }
        });

    manifold::stable_sort(pinched.begin(), pinched.end());
    pinched.resize(std::distance(
        pinched.begin(), manifold::unique(pinched.begin(), pinched.end())));
    std::unordered_set<int> processedVerts;
    for (size_t i : pinched) {
      const int startVert = halfedge_.Start(i);
      if (processedVerts.find(startVert) == processedVerts.end()) {
        processedVerts.insert(startVert);
        continue;
      }
      vertPos_.push_back(vertPos_[startVert]);
      const int vert = NumVert() - 1;
      ForVert(i, [this, vert](int current) {
        halfedge_.SetStart(current, vert);
        halfedge_.SetEnd(halfedge_.Pair(current), vert);
      });
    }
  } else
#endif
  {
    std::vector<bool> vertProcessed(NumVert(), false);
    std::vector<bool> halfedgeProcessed(nbEdges, false);
    for (size_t i = 0; i < nbEdges; ++i) {
      if (halfedgeProcessed[i]) continue;
      int vert = halfedge_.Start(i);
      if (vert == -1) continue;
      if (vertProcessed[vert]) {
        vertPos_.push_back(vertPos_[vert]);
        vert = NumVert() - 1;
        ForVert(i, [this, &halfedgeProcessed, vert](int current) {
          halfedgeProcessed[current] = true;
          halfedge_.SetStart(current, vert);
          halfedge_.SetEnd(halfedge_.Pair(current), vert);
        });
      } else {
        vertProcessed[vert] = true;
        ForVert(i, [this, &halfedgeProcessed, vert](int current) {
          halfedgeProcessed[current] = true;
        });
      }
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
        if (local[i]) continue;
        const int startVert = halfedge_.Start(i);
        const int endVert = halfedge_.End(i);
        if (startVert == -1 || endVert == -1) continue;
        // we want to keep the allocation
        endVerts.clear(false);
        endVertSet.clear();

        // first iteration, populate entries
        // this makes sure we always report the same set of entries
        ForVert(i, [&local, &endVerts, &endVertSet, this](int current) {
          local[current] = true;
          const int startVert = halfedge_.Start(current);
          const int endV = halfedge_.End(current);
          if (startVert == -1 || endV == -1) return;
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
          const int startVert = halfedge_.Start(current);
          const int endV = halfedge_.End(current);
          if (startVert == -1 || endV == -1) return;
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
    if (ManifoldParams().verbose >= 2) {
      std::cout << "found " << numFlagged << " duplicate edges to split"
                << std::endl;
    }
#endif
  }
}
}  // namespace manifold
