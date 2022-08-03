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
#include "par.h"

namespace {
using namespace manifold;

__host__ __device__ glm::ivec3 TriOf(int edge) {
  glm::ivec3 triEdge;
  triEdge[0] = edge;
  triEdge[1] = NextHalfedge(triEdge[0]);
  triEdge[2] = NextHalfedge(triEdge[1]);
  return triEdge;
}

__host__ __device__ bool Is01Longest(glm::vec2 v0, glm::vec2 v1, glm::vec2 v2) {
  const glm::vec2 e[3] = {v1 - v0, v2 - v1, v0 - v2};
  float l[3];
  for (int i : {0, 1, 2}) l[i] = glm::dot(e[i], e[i]);
  return l[0] > l[1] && l[0] > l[2];
}

struct DuplicateEdge {
  const Halfedge* sortedHalfedge;

  __host__ __device__ bool operator()(int edge) {
    const Halfedge& halfedge = sortedHalfedge[edge];
    const Halfedge& nextHalfedge = sortedHalfedge[edge + 1];
    return halfedge.startVert == nextHalfedge.startVert &&
           halfedge.endVert == nextHalfedge.endVert;
  }
};

struct ShortEdge {
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const float precision;

  __host__ __device__ bool operator()(int edge) {
    if (halfedge[edge].pairedHalfedge < 0) return false;
    // Flag short edges
    const glm::vec3 delta =
        vertPos[halfedge[edge].endVert] - vertPos[halfedge[edge].startVert];
    return glm::dot(delta, delta) < precision * precision;
  }
};

struct FlagEdge {
  const Halfedge* halfedge;
  const BaryRef* triBary;

  __host__ __device__ bool operator()(int edge) {
    if (halfedge[edge].pairedHalfedge < 0) return false;
    // Flag redundant edges - those where the startVert is surrounded by only
    // two original triangles.
    const BaryRef ref0 = triBary[edge / 3];
    int current = NextHalfedge(halfedge[edge].pairedHalfedge);
    const BaryRef ref1 = triBary[current / 3];
    while (current != edge) {
      current = NextHalfedge(halfedge[current].pairedHalfedge);
      int tri = current / 3;
      const BaryRef ref = triBary[tri];
      if ((ref.meshID != ref0.meshID || ref.tri != ref0.tri) &&
          (ref.meshID != ref1.meshID || ref.tri != ref1.tri))
        return false;
    }
    return true;
  }
};

struct SwappableEdge {
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const glm::vec3* triNormal;
  const float precision;

  __host__ __device__ bool operator()(int edge) {
    if (halfedge[edge].pairedHalfedge < 0) return false;

    int tri = halfedge[edge].face;
    glm::ivec3 triedge = TriOf(edge);
    glm::mat3x2 projection = GetAxisAlignedProjection(triNormal[tri]);
    glm::vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedge[triedge[i]].startVert];
    // if (CCW(v[0], v[1], v[2], precision) < 0) printf("tri %d is CW!\n", tri);
    if (CCW(v[0], v[1], v[2], precision) > 0 || !Is01Longest(v[0], v[1], v[2]))
      return false;

    // Switch to neighbor's projection.
    edge = halfedge[edge].pairedHalfedge;
    tri = halfedge[edge].face;
    triedge = TriOf(edge);
    projection = GetAxisAlignedProjection(triNormal[tri]);
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedge[triedge[i]].startVert];
    return CCW(v[0], v[1], v[2], precision) > 0 ||
           Is01Longest(v[0], v[1], v[2]);
  }
};
}  // namespace

namespace manifold {

/**
 * Collapses degenerate triangles by removing edges shorter than precision_ and
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
 * Rather than actually removing the edges, this step merely marks them for
 * removal, by setting vertPos to NaN and halfedge to {-1, -1, -1, -1}.
 */
void Manifold::Impl::SimplifyTopology() {
  auto policy = autoPolicy(halfedge_.size());

  VecDH<Halfedge> halfedge(halfedge_);
  VecDH<int> idx(halfedge_.size());
  sequence(policy, idx.begin(), idx.end());
  sort_by_key(policy, halfedge.begin(), halfedge.end(), idx.begin());

  VecDH<int> flaggedEdges(halfedge_.size());

  int numFlagged =
      copy_if<decltype(flaggedEdges.begin())>(
          policy, idx.begin(), idx.end() - 1, countAt(0), flaggedEdges.begin(),
          DuplicateEdge({halfedge.cptrD()})) -
      flaggedEdges.begin();
  flaggedEdges.resize(numFlagged);

  for (const int edge : flaggedEdges) DedupeEdge(edge);

  flaggedEdges.resize(halfedge_.size());
  numFlagged =
      copy_if<decltype(flaggedEdges.begin())>(
          policy, countAt(0), countAt(halfedge_.size()), flaggedEdges.begin(),
          ShortEdge({halfedge_.cptrD(), vertPos_.cptrD(), precision_})) -
      flaggedEdges.begin();
  flaggedEdges.resize(numFlagged);

  for (const int edge : flaggedEdges) CollapseEdge(edge);

  flaggedEdges.resize(halfedge_.size());
  numFlagged =
      copy_if<decltype(flaggedEdges.begin())>(
          policy, countAt(0), countAt(halfedge_.size()), flaggedEdges.begin(),
          FlagEdge({halfedge_.cptrD(), meshRelation_.triBary.cptrD()})) -
      flaggedEdges.begin();
  flaggedEdges.resize(numFlagged);

  for (const int edge : flaggedEdges) CollapseEdge(edge);

  flaggedEdges.resize(halfedge_.size());
  numFlagged =
      copy_if<decltype(flaggedEdges.begin())>(
          policy, countAt(0), countAt(halfedge_.size()), flaggedEdges.begin(),
          SwappableEdge({halfedge_.cptrD(), vertPos_.cptrD(),
                         faceNormal_.cptrD(), precision_})) -
      flaggedEdges.begin();
  flaggedEdges.resize(numFlagged);

  for (const int edge : flaggedEdges) {
    RecursiveEdgeSwap(edge);
  }
}

void Manifold::Impl::DedupeEdge(const int edge) {
  // Orbit endVert
  const int startVert = halfedge_[edge].startVert;
  const int endVert = halfedge_[edge].endVert;
  int current = halfedge_[NextHalfedge(edge)].pairedHalfedge;
  while (current != edge) {
    const int vert = halfedge_[current].startVert;
    if (vert == startVert) {
      const int newVert = vertPos_.size();
      vertPos_.push_back(vertPos_[endVert]);
      if (vertNormal_.size() > 0) vertNormal_.push_back(vertNormal_[endVert]);
      current = halfedge_[NextHalfedge(current)].pairedHalfedge;
      const int opposite = halfedge_[NextHalfedge(edge)].pairedHalfedge;

      UpdateVert(newVert, current, opposite);

      int newHalfedge = halfedge_.size();
      int newFace = newHalfedge / 3;
      int oldFace = halfedge_[current].face;
      int outsideVert = halfedge_[current].startVert;
      halfedge_.push_back({endVert, newVert, -1, newFace});
      halfedge_.push_back({newVert, outsideVert, -1, newFace});
      halfedge_.push_back({outsideVert, endVert, -1, newFace});
      PairUp(newHalfedge + 2, halfedge_[current].pairedHalfedge);
      PairUp(newHalfedge + 1, current);
      if (meshRelation_.triBary.size() > 0)
        meshRelation_.triBary.push_back(meshRelation_.triBary[oldFace]);
      if (faceNormal_.size() > 0) faceNormal_.push_back(faceNormal_[oldFace]);

      newHalfedge += 3;
      ++newFace;
      oldFace = halfedge_[opposite].face;
      outsideVert = halfedge_[opposite].startVert;
      halfedge_.push_back({newVert, endVert, -1, newFace});
      halfedge_.push_back({endVert, outsideVert, -1, newFace});
      halfedge_.push_back({outsideVert, newVert, -1, newFace});
      PairUp(newHalfedge + 2, halfedge_[opposite].pairedHalfedge);
      PairUp(newHalfedge + 1, opposite);
      PairUp(newHalfedge, newHalfedge - 3);
      if (meshRelation_.triBary.size() > 0)
        meshRelation_.triBary.push_back(meshRelation_.triBary[oldFace]);
      if (faceNormal_.size() > 0) faceNormal_.push_back(faceNormal_[oldFace]);

      break;
    }

    current = halfedge_[NextHalfedge(current)].pairedHalfedge;
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
  while (startEdge != endEdge) {
    halfedge_[startEdge].endVert = vert;
    startEdge = NextHalfedge(startEdge);
    halfedge_[startEdge].startVert = vert;
    startEdge = halfedge_[startEdge].pairedHalfedge;
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

void Manifold::Impl::CollapseTri(const glm::ivec3& triEdge) {
  int pair1 = halfedge_[triEdge[1]].pairedHalfedge;
  int pair2 = halfedge_[triEdge[2]].pairedHalfedge;
  halfedge_[pair1].pairedHalfedge = pair2;
  halfedge_[pair2].pairedHalfedge = pair1;
  for (int i : {0, 1, 2}) {
    halfedge_[triEdge[i]] = {-1, -1, -1, -1};
  }
}

void Manifold::Impl::RemoveIfFolded(int edge) {
  const glm::ivec3 tri0edge = TriOf(edge);
  const glm::ivec3 tri1edge = TriOf(halfedge_[edge].pairedHalfedge);
  if (halfedge_[tri0edge[1]].endVert == halfedge_[tri1edge[1]].endVert) {
    for (int i : {0, 1, 2}) {
      vertPos_[halfedge_[tri0edge[i]].startVert] = glm::vec3(NAN);
      halfedge_[tri0edge[i]] = {-1, -1, -1, -1};
      halfedge_[tri1edge[i]] = {-1, -1, -1, -1};
    }
  }
}

void Manifold::Impl::CollapseEdge(const int edge) {
  VecDH<BaryRef>& triBary = meshRelation_.triBary;

  const Halfedge toRemove = halfedge_[edge];
  if (toRemove.pairedHalfedge < 0) return;

  const int endVert = toRemove.endVert;
  const glm::ivec3 tri0edge = TriOf(edge);
  const glm::ivec3 tri1edge = TriOf(toRemove.pairedHalfedge);

  const glm::vec3 pNew = vertPos_[endVert];
  const glm::vec3 pOld = vertPos_[toRemove.startVert];
  const glm::vec3 delta = pNew - pOld;
  const bool shortEdge = glm::dot(delta, delta) < precision_ * precision_;

  std::vector<int> edges;
  // Orbit endVert
  int current = halfedge_[tri0edge[1]].pairedHalfedge;
  while (current != tri1edge[2]) {
    current = NextHalfedge(current);
    edges.push_back(current);
    current = halfedge_[current].pairedHalfedge;
  }

  // Orbit startVert
  int start = halfedge_[tri1edge[1]].pairedHalfedge;
  const BaryRef ref0 = triBary[edge / 3];
  const BaryRef ref1 = triBary[toRemove.pairedHalfedge / 3];
  if (!shortEdge) {
    current = start;
    glm::vec3 pLast = vertPos_[halfedge_[tri1edge[1]].endVert];
    while (current != tri0edge[2]) {
      current = NextHalfedge(current);
      glm::vec3 pNext = vertPos_[halfedge_[current].endVert];
      const int tri = current / 3;
      const BaryRef ref = triBary[tri];
      // Don't collapse if the edge is not redundant (this may have changed due
      // to the collapse of neighbors).
      if ((ref.meshID != ref0.meshID || ref.tri != ref0.tri) &&
          (ref.meshID != ref1.meshID || ref.tri != ref1.tri))
        return;

      // Don't collapse edge if it would cause a triangle to invert.
      const glm::mat3x2 projection = GetAxisAlignedProjection(faceNormal_[tri]);
      if (CCW(projection * pNext, projection * pLast, projection * pNew,
              precision_) < 0)
        return;

      pLast = pNext;
      current = halfedge_[current].pairedHalfedge;
    }
  }

  // Remove toRemove.startVert and replace with endVert.
  vertPos_[toRemove.startVert] = glm::vec3(NAN);
  CollapseTri(tri1edge);

  // Orbit startVert
  current = start;
  while (current != tri0edge[2]) {
    current = NextHalfedge(current);

    if (!shortEdge) {
      // Update the shifted triangles to the vertBary of endVert
      const int tri = current / 3;
      const int vIdx = current - 3 * tri;
      triBary[tri].vertBary[vIdx] =
          (ref0.meshID == triBary[tri].meshID && ref0.tri == triBary[tri].tri)
              ? ref0.vertBary[(edge + 1) % 3]
              : ref1.vertBary[toRemove.pairedHalfedge % 3];
    }

    const int vert = halfedge_[current].endVert;
    const int next = halfedge_[current].pairedHalfedge;
    for (int i = 0; i < edges.size(); ++i) {
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
}

void Manifold::Impl::RecursiveEdgeSwap(const int edge) {
  VecDH<BaryRef>& triBary = meshRelation_.triBary;

  if (edge < 0) return;
  const int pair = halfedge_[edge].pairedHalfedge;
  if (pair < 0) return;

  const glm::ivec3 tri0edge = TriOf(edge);
  const glm::ivec3 tri1edge = TriOf(pair);
  const glm::ivec3 perm0 = TriOf(edge % 3);
  const glm::ivec3 perm1 = TriOf(pair % 3);

  glm::mat3x2 projection = GetAxisAlignedProjection(faceNormal_[edge / 3]);
  glm::vec2 v[4];
  for (int i : {0, 1, 2})
    v[i] = projection * vertPos_[halfedge_[tri0edge[i]].startVert];
  // Only operate on the long edge of a degenerate triangle.
  if (CCW(v[0], v[1], v[2], precision_) > 0 || !Is01Longest(v[0], v[1], v[2]))
    return;

  // Switch to neighbor's projection.
  projection = GetAxisAlignedProjection(faceNormal_[halfedge_[pair].face]);
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
    const int tri0 = halfedge_[tri0edge[0]].face;
    const int tri1 = halfedge_[tri1edge[0]].face;
    faceNormal_[tri0] = faceNormal_[tri1];
    triBary[tri0] = triBary[tri1];
    triBary[tri0].vertBary[perm0[1]] = triBary[tri1].vertBary[perm1[0]];
    triBary[tri0].vertBary[perm0[0]] = triBary[tri1].vertBary[perm1[2]];
    // Calculate a new barycentric coordinate for the split triangle.
    const glm::vec3 uvw0 = UVW(triBary[tri1].vertBary[perm1[0]],
                               meshRelation_.barycentric.cptrH());
    const glm::vec3 uvw1 = UVW(triBary[tri1].vertBary[perm1[1]],
                               meshRelation_.barycentric.cptrH());
    const float l01 = glm::length(v[1] - v[0]);
    const float l02 = glm::length(v[2] - v[0]);
    const float a = glm::max(0.0f, glm::min(1.0f, l02 / l01));
    const glm::vec3 uvw2 = a * uvw0 + (1 - a) * uvw1;
    // And assign it.
    const int newBary = meshRelation_.barycentric.size();
    meshRelation_.barycentric.push_back(uvw2);
    triBary[tri1].vertBary[perm1[0]] = newBary;
    triBary[tri0].vertBary[perm0[2]] = newBary;

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
  if (CCW(v[1], v[0], v[3], precision_) <= 0) {
    if (!Is01Longest(v[1], v[0], v[3])) return;
    // Two facing, long-edge degenerates can swap.
    SwapEdge();
    const glm::vec2 e23 = v[3] - v[2];
    if (glm::dot(e23, e23) < precision_ * precision_) {
      CollapseEdge(tri0edge[2]);
    } else {
      RecursiveEdgeSwap(tri0edge[0]);
      RecursiveEdgeSwap(tri0edge[1]);
      RecursiveEdgeSwap(tri1edge[0]);
      RecursiveEdgeSwap(tri1edge[1]);
    }
    return;
  } else if (CCW(v[0], v[3], v[2], precision_) <= 0 ||
             CCW(v[1], v[2], v[3], precision_) <= 0) {
    return;
  }
  // Normal path
  SwapEdge();
  RecursiveEdgeSwap(halfedge_[tri0edge[1]].pairedHalfedge);
  RecursiveEdgeSwap(halfedge_[tri1edge[0]].pairedHalfedge);
}
}  // namespace manifold
