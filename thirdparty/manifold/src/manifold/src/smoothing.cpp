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

#include <map>

#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;

__host__ __device__ glm::vec3 OrthogonalTo(glm::vec3 in, glm::vec3 ref) {
  in -= glm::dot(in, ref) * ref;
  return in;
}

/**
 * The total number of verts if a triangle is subdivided naturally such that
 * each edge has edgeVerts verts along it (edgeVerts >= -1).
 */
__host__ __device__ int VertsPerTri(int edgeVerts) {
  return (edgeVerts * edgeVerts + edgeVerts) / 2;
}

struct Barycentric {
  int tri;
  glm::vec3 uvw;
};

struct ReindexHalfedge {
  int* half2Edge;

  __host__ __device__ void operator()(thrust::tuple<int, TmpEdge> in) {
    const int edge = thrust::get<0>(in);
    const int halfedge = thrust::get<1>(in).halfedgeIdx;

    half2Edge[halfedge] = edge;
  }
};

struct EdgeVerts {
  glm::vec3* vertPos;
  const int startIdx;
  const int n;

  __host__ __device__ void operator()(thrust::tuple<int, TmpEdge> in) {
    int edge = thrust::get<0>(in);
    TmpEdge edgeVerts = thrust::get<1>(in);

    float invTotal = 1.0f / n;
    for (int i = 1; i < n; ++i)
      vertPos[startIdx + (n - 1) * edge + i - 1] =
          (float(n - i) * vertPos[edgeVerts.first] +
           float(i) * vertPos[edgeVerts.second]) *
          invTotal;
  }
};

struct InteriorVerts {
  glm::vec3* vertPos;
  glm::vec3* uvw;
  BaryRef* triBary;
  glm::vec3* uvwNew;
  BaryRef* triBaryNew;
  const glm::vec3* uvwOld;
  const int startIdx;
  const int n;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<int, BaryRef> in) {
    const int tri = thrust::get<0>(in);
    const BaryRef baryOld = thrust::get<1>(in);

    glm::mat3 uvwOldTri;
    for (int i : {0, 1, 2}) uvwOldTri[i] = UVW(baryOld.vertBary[i], uvwOld);

    const float invTotal = 1.0f / n;
    int posTri = tri * n * n;
    int posBary = tri * VertsPerTri(n + 1);
    int pos = startIdx + tri * VertsPerTri(n - 2);
    for (int i = 0; i <= n; ++i) {
      for (int j = 0; j <= n - i; ++j) {
        const int k = n - i - j;
        const float u = invTotal * j;
        const float v = invTotal * k;
        const float w = invTotal * i;
        const int first = posBary;
        uvw[posBary] = {u, v, w};
        uvwNew[posBary] = uvwOldTri * uvw[posBary];
        ++posBary;
        if (j == n - i) continue;

        // The three retained verts are denoted by their index - 3. uvw entries
        // are added for them out of laziness of indexing only.
        const int a = (k == n) ? -2 : first;
        const int b = (i == n - 1) ? -1 : first + n - i + 1;
        const int c = (j == n - 1) ? -3 : first + 1;
        glm::ivec3 vertBary(c, a, b);
        triBary[posTri] = {-1, -1, tri, vertBary};
        triBaryNew[posTri++] = {baryOld.meshID, baryOld.originalID, baryOld.tri,
                                vertBary};
        if (j < n - 1 - i) {
          int d = b + 1;  // d cannot be a retained vert
          vertBary = {b, d, c};
          triBary[posTri] = {-1, -1, tri, vertBary};
          triBaryNew[posTri++] = {baryOld.meshID, baryOld.originalID,
                                  baryOld.tri, vertBary};
        }

        if (i == 0 || j == 0 || k == 0) continue;

        vertPos[pos++] = u * vertPos[halfedge[3 * tri].startVert] +      //
                         v * vertPos[halfedge[3 * tri + 1].startVert] +  //
                         w * vertPos[halfedge[3 * tri + 2].startVert];
      }
    }
  }
};

struct SplitTris {
  glm::ivec3* triVerts;
  const Halfedge* halfedge;
  const int* half2Edge;
  const int edgeIdx;
  const int triIdx;
  const int n;

  __host__ __device__ int EdgeVert(int i, int inHalfedge) const {
    bool forward = halfedge[inHalfedge].IsForward();
    int edge = forward ? half2Edge[inHalfedge]
                       : half2Edge[halfedge[inHalfedge].pairedHalfedge];
    return edgeIdx + (n - 1) * edge + (forward ? i - 1 : n - 1 - i);
  }

  __host__ __device__ int TriVert(int i, int j, int tri) const {
    --i;
    --j;
    int m = n - 2;
    int vertsPerTri = (m * m + m) / 2;
    int vertOffset = (i * (2 * m - i + 1)) / 2 + j;
    return triIdx + vertsPerTri * tri + vertOffset;
  }

  __host__ __device__ int Vert(int i, int j, int tri) const {
    bool edge0 = i == 0;
    bool edge1 = j == 0;
    bool edge2 = j == n - i;
    if (edge0) {
      if (edge1)
        return halfedge[3 * tri + 1].startVert;
      else if (edge2)
        return halfedge[3 * tri].startVert;
      else
        return EdgeVert(n - j, 3 * tri);
    } else if (edge1) {
      if (edge2)
        return halfedge[3 * tri + 2].startVert;
      else
        return EdgeVert(i, 3 * tri + 1);
    } else if (edge2)
      return EdgeVert(j, 3 * tri + 2);
    else
      return TriVert(i, j, tri);
  }

  __host__ __device__ void operator()(int tri) {
    int pos = n * n * tri;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n - i; ++j) {
        int a = Vert(i, j, tri);
        int b = Vert(i + 1, j, tri);
        int c = Vert(i, j + 1, tri);
        triVerts[pos++] = glm::ivec3(c, a, b);
        if (j < n - 1 - i) {
          int d = Vert(i + 1, j + 1, tri);
          triVerts[pos++] = glm::ivec3(b, d, c);
        }
      }
    }
  }
};

struct SmoothBezier {
  const glm::vec3* vertPos;
  const glm::vec3* triNormal;
  const glm::vec3* vertNormal;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec4&, Halfedge> inOut) {
    glm::vec4& tangent = thrust::get<0>(inOut);
    const Halfedge edge = thrust::get<1>(inOut);

    const glm::vec3 startV = vertPos[edge.startVert];
    const glm::vec3 edgeVec = vertPos[edge.endVert] - startV;
    const glm::vec3 edgeNormal =
        (triNormal[edge.face] + triNormal[halfedge[edge.pairedHalfedge].face]) /
        2.0f;
    glm::vec3 dir = glm::normalize(glm::cross(glm::cross(edgeNormal, edgeVec),
                                              vertNormal[edge.startVert]));

    const float weight = glm::abs(glm::dot(dir, glm::normalize(edgeVec)));
    // Quadratic weighted bezier for circular interpolation
    const glm::vec4 bz2 =
        weight *
        glm::vec4(startV + dir * glm::length(edgeVec) / (2 * weight), 1.0f);
    // Equivalent cubic weighted bezier
    const glm::vec4 bz3 = glm::mix(glm::vec4(startV, 1.0f), bz2, 2 / 3.0f);
    // Convert from homogeneous form to geometric form
    tangent = glm::vec4(glm::vec3(bz3) / bz3.w - startV, bz3.w);
  }
};

struct TriBary2Vert {
  Barycentric* vertBary;
  int* lock;
  const glm::vec3* uvw;
  const Halfedge* halfedge;

  __host__ __device__ void operator()(thrust::tuple<BaryRef, int> in) {
    const BaryRef baryRef = thrust::get<0>(in);
    const int tri = thrust::get<1>(in);

    for (int i : {0, 1, 2}) {
      int vert = halfedge[3 * tri + i].startVert;
      if (AtomicAdd(lock[vert], 1) != 0) continue;
      vertBary[vert] = {baryRef.tri, UVW(baryRef.vertBary[i], uvw)};
    }
  }
};

struct InterpTri {
  const Halfedge* halfedge;
  const glm::vec4* halfedgeTangent;
  const glm::vec3* vertPos;

  __host__ __device__ glm::vec4 Homogeneous(glm::vec4 v) const {
    v.x *= v.w;
    v.y *= v.w;
    v.z *= v.w;
    return v;
  }

  __host__ __device__ glm::vec4 Homogeneous(glm::vec3 v) const {
    return glm::vec4(v, 1.0f);
  }

  __host__ __device__ glm::vec3 HNormalize(glm::vec4 v) const {
    return glm::vec3(v) / v.w;
  }

  __host__ __device__ glm::vec4 Bezier(glm::vec3 point,
                                       glm::vec4 tangent) const {
    return Homogeneous(glm::vec4(point, 0) + tangent);
  }

  __host__ __device__ glm::mat2x4 CubicBezier2Linear(glm::vec4 p0, glm::vec4 p1,
                                                     glm::vec4 p2, glm::vec4 p3,
                                                     float x) const {
    glm::mat2x4 out;
    glm::vec4 p12 = glm::mix(p1, p2, x);
    out[0] = glm::mix(glm::mix(p0, p1, x), p12, x);
    out[1] = glm::mix(p12, glm::mix(p2, p3, x), x);
    return out;
  }

  __host__ __device__ glm::vec3 BezierPoint(glm::mat2x4 points, float x) const {
    return HNormalize(glm::mix(points[0], points[1], x));
  }

  __host__ __device__ glm::vec3 BezierTangent(glm::mat2x4 points) const {
    return glm::normalize(HNormalize(points[1]) - HNormalize(points[0]));
  }

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec3&, Barycentric> inOut) {
    glm::vec3& pos = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut).tri;
    const glm::vec3 uvw = thrust::get<1>(inOut).uvw;

    glm::vec4 posH(0);
    const glm::mat3 corners = {vertPos[halfedge[3 * tri].startVert],
                               vertPos[halfedge[3 * tri + 1].startVert],
                               vertPos[halfedge[3 * tri + 2].startVert]};

    for (const int i : {0, 1, 2}) {
      if (uvw[i] == 1) {
        pos = glm::vec3(corners[i]);
        return;
      }
    }

    const glm::mat3x4 tangentR = {halfedgeTangent[3 * tri],
                                  halfedgeTangent[3 * tri + 1],
                                  halfedgeTangent[3 * tri + 2]};
    const glm::mat3x4 tangentL = {
        halfedgeTangent[halfedge[3 * tri + 2].pairedHalfedge],
        halfedgeTangent[halfedge[3 * tri].pairedHalfedge],
        halfedgeTangent[halfedge[3 * tri + 1].pairedHalfedge]};

    for (const int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      const int k = (i + 2) % 3;
      const float x = uvw[k] / (1 - uvw[i]);

      const glm::mat2x4 bez = CubicBezier2Linear(
          Homogeneous(corners[j]), Bezier(corners[j], tangentR[j]),
          Bezier(corners[k], tangentL[k]), Homogeneous(corners[k]), x);
      const glm::vec3 end = BezierPoint(bez, x);
      const glm::vec3 tangent = BezierTangent(bez);

      const glm::vec3 jBitangent = SafeNormalize(OrthogonalTo(
          glm::vec3(tangentL[j]), SafeNormalize(glm::vec3(tangentR[j]))));
      const glm::vec3 kBitangent = SafeNormalize(OrthogonalTo(
          glm::vec3(tangentR[k]), -SafeNormalize(glm::vec3(tangentL[k]))));
      const glm::vec3 normal = SafeNormalize(
          glm::cross(glm::mix(jBitangent, kBitangent, x), tangent));
      const glm::vec3 delta = OrthogonalTo(
          glm::mix(glm::vec3(tangentL[j]), glm::vec3(tangentR[k]), x), normal);
      const float deltaW = glm::mix(tangentL[j].w, tangentR[k].w, x);

      const glm::mat2x4 bez1 = CubicBezier2Linear(
          Homogeneous(end), Homogeneous(glm::vec4(end + delta, deltaW)),
          Bezier(corners[i], glm::mix(tangentR[i], tangentL[i], x)),
          Homogeneous(corners[i]), uvw[i]);
      const glm::vec3 p = BezierPoint(bez1, uvw[i]);
      float w = uvw[j] * uvw[j] * uvw[k] * uvw[k];
      posH += Homogeneous(glm::vec4(p, w));
    }
    pos = HNormalize(posH);
  }
};
}  // namespace

namespace manifold {

/**
 * Calculates halfedgeTangent_, allowing the manifold to be refined and
 * smoothed. The tangents form weighted cubic Beziers along each edge. This
 * function creates circular arcs where possible (minimizing maximum curvature),
 * constrained to the vertex normals. Where sharpenedEdges are specified, the
 * tangents are shortened that intersect the sharpened edge, concentrating the
 * curvature there, while the tangents of the sharp edges themselves are aligned
 * for continuity.
 */
void Manifold::Impl::CreateTangents(
    const std::vector<Smoothness>& sharpenedEdges) {
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.resize(numHalfedge);

  for_each_n(autoPolicy(numHalfedge),
             zip(halfedgeTangent_.begin(), halfedge_.cbegin()), numHalfedge,
             SmoothBezier({vertPos_.cptrD(), faceNormal_.cptrD(),
                           vertNormal_.cptrD(), halfedge_.cptrD()}));

  if (!sharpenedEdges.empty()) {
    const VecDH<BaryRef>& triBary = meshRelation_.triBary;

    // sharpenedEdges are referenced to the input Mesh, but the triangles have
    // been sorted in creating the Manifold, so the indices are converted using
    // meshRelation_.
    std::vector<int> oldHalfedge2New(halfedge_.size());
    for (int tri = 0; tri < NumTri(); ++tri) {
      int oldTri = triBary[tri].tri;
      for (int i : {0, 1, 2}) oldHalfedge2New[3 * oldTri + i] = 3 * tri + i;
    }

    using Pair = std::pair<Smoothness, Smoothness>;
    // Fill in missing pairs with default smoothness = 1.
    std::map<int, Pair> edges;
    for (Smoothness edge : sharpenedEdges) {
      if (edge.smoothness == 1) continue;
      edge.halfedge = oldHalfedge2New[edge.halfedge];
      int pair = halfedge_[edge.halfedge].pairedHalfedge;
      if (edges.find(pair) == edges.end()) {
        edges[edge.halfedge] = {edge, {pair, 1}};
      } else {
        edges[pair].second = edge;
      }
    }

    std::map<int, std::vector<Pair>> vertTangents;
    for (const auto& value : edges) {
      const Pair edge = value.second;
      vertTangents[halfedge_[edge.first.halfedge].startVert].push_back(edge);
      vertTangents[halfedge_[edge.second.halfedge].startVert].push_back(
          {edge.second, edge.first});
    }

    VecDH<glm::vec4>& tangent = halfedgeTangent_;
    for (const auto& value : vertTangents) {
      const std::vector<Pair>& vert = value.second;
      // Sharp edges that end are smooth at their terminal vert.
      if (vert.size() == 1) continue;
      if (vert.size() == 2) {  // Make continuous edge
        const int first = vert[0].first.halfedge;
        const int second = vert[1].first.halfedge;
        const glm::vec3 newTangent = glm::normalize(glm::vec3(tangent[first]) -
                                                    glm::vec3(tangent[second]));
        tangent[first] =
            glm::vec4(glm::length(glm::vec3(tangent[first])) * newTangent,
                      tangent[first].w);
        tangent[second] =
            glm::vec4(-glm::length(glm::vec3(tangent[second])) * newTangent,
                      tangent[second].w);

        auto SmoothHalf = [&](int first, int last, float smoothness) {
          int current = NextHalfedge(halfedge_[first].pairedHalfedge);
          while (current != last) {
            const float cosBeta = glm::dot(
                newTangent, glm::normalize(glm::vec3(tangent[current])));
            const float factor =
                (1 - smoothness) * cosBeta * cosBeta + smoothness;
            tangent[current] = glm::vec4(factor * glm::vec3(tangent[current]),
                                         tangent[current].w);
            current = NextHalfedge(halfedge_[current].pairedHalfedge);
          }
        };

        SmoothHalf(first, second,
                   (vert[0].second.smoothness + vert[1].first.smoothness) / 2);
        SmoothHalf(second, first,
                   (vert[1].second.smoothness + vert[0].first.smoothness) / 2);

      } else {  // Sharpen vertex uniformly
        float smoothness = 0;
        for (const Pair& pair : vert) {
          smoothness += pair.first.smoothness;
          smoothness += pair.second.smoothness;
        }
        smoothness /= 2 * vert.size();

        const int start = vert[0].first.halfedge;
        int current = start;
        do {
          tangent[current] = glm::vec4(smoothness * glm::vec3(tangent[current]),
                                       tangent[current].w);
          current = NextHalfedge(halfedge_[current].pairedHalfedge);
        } while (current != start);
      }
    }
  }
}

/**
 * Split each edge into n pieces and sub-triangulate each triangle accordingly.
 * This function doesn't run Finish(), as that is expensive and it'll need to be
 * run after the new vertices have moved, which is a likely scenario after
 * refinement (smoothing).
 */
Manifold::Impl::MeshRelationD Manifold::Impl::Subdivide(int n) {
  if (n < 2) return meshRelation_;
  faceNormal_.resize(0);
  vertNormal_.resize(0);
  int numVert = NumVert();
  int numEdge = NumEdge();
  int numTri = NumTri();
  // Append new verts
  int vertsPerEdge = n - 1;
  int triVertStart = numVert + numEdge * vertsPerEdge;
  vertPos_.resize(triVertStart + numTri * VertsPerTri(n - 2));

  MeshRelationD relation;
  relation.barycentric.resize(numTri * VertsPerTri(n + 1));
  relation.triBary.resize(n * n * numTri);
  MeshRelationD oldMeshRelation = std::move(meshRelation_);
  meshRelation_.barycentric.resize(relation.barycentric.size());
  meshRelation_.triBary.resize(relation.triBary.size());
  meshRelation_.originalID = oldMeshRelation.originalID;

  VecDH<TmpEdge> edges = CreateTmpEdges(halfedge_);
  VecDH<int> half2Edge(2 * numEdge);
  auto policy = autoPolicy(numEdge);
  for_each_n(policy, zip(countAt(0), edges.begin()), numEdge,
             ReindexHalfedge({half2Edge.ptrD()}));
  for_each_n(policy, zip(countAt(0), edges.begin()), numEdge,
             EdgeVerts({vertPos_.ptrD(), numVert, n}));
  for_each_n(
      policy, zip(countAt(0), oldMeshRelation.triBary.begin()), numTri,
      InteriorVerts({vertPos_.ptrD(), relation.barycentric.ptrD(),
                     relation.triBary.ptrD(), meshRelation_.barycentric.ptrD(),
                     meshRelation_.triBary.ptrD(),
                     oldMeshRelation.barycentric.cptrD(), triVertStart, n,
                     halfedge_.ptrD()}));
  // Create subtriangles
  VecDH<glm::ivec3> triVerts(n * n * numTri);
  for_each_n(policy, countAt(0), numTri,
             SplitTris({triVerts.ptrD(), halfedge_.cptrD(), half2Edge.cptrD(),
                        numVert, triVertStart, n}));
  CreateHalfedges(triVerts);
  return relation;
}

void Manifold::Impl::Refine(int n) {
  Manifold::Impl old = *this;
  MeshRelationD relation = Subdivide(n);

  if (old.halfedgeTangent_.size() == old.halfedge_.size()) {
    VecDH<Barycentric> vertBary(NumVert());
    VecDH<int> lock(NumVert(), 0);
    auto policy = autoPolicy(NumTri());
    for_each_n(policy, zip(relation.triBary.begin(), countAt(0)), NumTri(),
               TriBary2Vert({vertBary.ptrD(), lock.ptrD(),
                             relation.barycentric.cptrD(), halfedge_.cptrD()}));

    for_each_n(policy, zip(vertPos_.begin(), vertBary.begin()), NumVert(),
               InterpTri({old.halfedge_.cptrD(), old.halfedgeTangent_.cptrD(),
                          old.vertPos_.cptrD()}));
  }

  halfedgeTangent_.resize(0);
  Finish();
}
}  // namespace manifold
