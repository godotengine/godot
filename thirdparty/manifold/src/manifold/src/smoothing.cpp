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

#include <glm/gtx/quaternion.hpp>

#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;

glm::vec3 OrthogonalTo(glm::vec3 in, glm::vec3 ref) {
  in -= glm::dot(in, ref) * ref;
  return in;
}

// Calculate a tangent vector in the form of a weighted cubic Bezier taking as
// input the desired tangent direction (length doesn't matter) and the edge
// vector to the neighboring vertex. In a symmetric situation where the tangents
// at each end are mirror images of each other, this will result in a circular
// arc.
glm::vec4 CircularTangent(const glm::vec3& tangent, const glm::vec3& edgeVec) {
  const glm::vec3 dir = SafeNormalize(tangent);

  float weight = glm::abs(glm::dot(dir, SafeNormalize(edgeVec)));
  if (weight == 0) {
    weight = 1;
  }
  // Quadratic weighted bezier for circular interpolation
  const glm::vec4 bz2 =
      weight * glm::vec4(dir * glm::length(edgeVec) / (2 * weight), 1);
  // Equivalent cubic weighted bezier
  const glm::vec4 bz3 = glm::mix(glm::vec4(0, 0, 0, 1), bz2, 2 / 3.0f);
  // Convert from homogeneous form to geometric form
  return glm::vec4(glm::vec3(bz3) / bz3.w, bz3.w);
}

struct SmoothBezier {
  const Manifold::Impl* impl;
  VecView<const glm::vec3> vertNormal;

  void operator()(thrust::tuple<glm::vec4&, Halfedge, int> inOut) {
    glm::vec4& tangent = thrust::get<0>(inOut);
    const Halfedge edge = thrust::get<1>(inOut);
    const int edgeIdx = thrust::get<2>(inOut);

    if (impl->IsInsideQuad(edgeIdx)) {
      tangent = glm::vec4(0, 0, 0, -1);
      return;
    }

    const glm::vec3 edgeVec =
        impl->vertPos_[edge.endVert] - impl->vertPos_[edge.startVert];
    const glm::vec3 edgeNormal =
        impl->faceNormal_[edge.face] +
        impl->faceNormal_[impl->halfedge_[edge.pairedHalfedge].face];
    glm::vec3 dir =
        glm::cross(glm::cross(edgeNormal, edgeVec), vertNormal[edge.startVert]);
    tangent = CircularTangent(dir, edgeVec);
  }
};

struct InterpTri {
  const Manifold::Impl* impl;

  glm::vec4 Homogeneous(glm::vec4 v) const {
    v.x *= v.w;
    v.y *= v.w;
    v.z *= v.w;
    return v;
  }

  glm::vec4 Homogeneous(glm::vec3 v) const { return glm::vec4(v, 1.0f); }

  glm::vec3 HNormalize(glm::vec4 v) const {
    return v.w == 0 ? v : (glm::vec3(v) / v.w);
  }

  glm::vec4 Scale(glm::vec4 v, float scale) const {
    return glm::vec4(scale * glm::vec3(v), v.w);
  }

  glm::vec4 Bezier(glm::vec3 point, glm::vec4 tangent) const {
    return Homogeneous(glm::vec4(point, 0) + tangent);
  }

  glm::mat2x4 CubicBezier2Linear(glm::vec4 p0, glm::vec4 p1, glm::vec4 p2,
                                 glm::vec4 p3, float x) const {
    glm::mat2x4 out;
    glm::vec4 p12 = glm::mix(p1, p2, x);
    out[0] = glm::mix(glm::mix(p0, p1, x), p12, x);
    out[1] = glm::mix(p12, glm::mix(p2, p3, x), x);
    return out;
  }

  glm::vec3 BezierPoint(glm::mat2x4 points, float x) const {
    return HNormalize(glm::mix(points[0], points[1], x));
  }

  glm::vec3 BezierTangent(glm::mat2x4 points) const {
    return SafeNormalize(HNormalize(points[1]) - HNormalize(points[0]));
  }

  glm::vec3 RotateFromTo(glm::vec3 v, glm::quat start, glm::quat end) const {
    return end * glm::conjugate(start) * v;
  }

  glm::mat2x4 Bezier2Bezier(const glm::mat2x3& corners,
                            const glm::mat2x4& tangentsX,
                            const glm::mat2x4& tangentsY, float x,
                            const glm::bvec2& pointedEnds,
                            const glm::vec3& anchor) const {
    const glm::mat2x4 bez = CubicBezier2Linear(
        Homogeneous(corners[0]), Bezier(corners[0], tangentsX[0]),
        Bezier(corners[1], tangentsX[1]), Homogeneous(corners[1]), x);
    const glm::vec3 end = BezierPoint(bez, x);
    const glm::vec3 tangent = BezierTangent(bez);

    const glm::mat2x3 nTangentsX(SafeNormalize(glm::vec3(tangentsX[0])),
                                 -SafeNormalize(glm::vec3(tangentsX[1])));
    const glm::mat2x3 biTangents = {
        SafeNormalize(OrthogonalTo(
            glm::vec3(tangentsY[0]) + kTolerance * (anchor - corners[0]),
            nTangentsX[0])),
        SafeNormalize(OrthogonalTo(
            glm::vec3(tangentsY[1]) + kTolerance * (anchor - corners[1]),
            nTangentsX[1]))};

    const glm::quat q0 =
        glm::quat_cast(glm::mat3(nTangentsX[0], biTangents[0],
                                 glm::cross(nTangentsX[0], biTangents[0])));
    const glm::quat q1 =
        glm::quat_cast(glm::mat3(nTangentsX[1], biTangents[1],
                                 glm::cross(nTangentsX[1], biTangents[1])));
    const glm::quat qTmp = glm::slerp(q0, q1, x);
    const glm::quat q =
        glm::rotation(qTmp * glm::vec3(1, 0, 0), tangent) * qTmp;

    const glm::vec3 end0 = pointedEnds[0]
                               ? glm::vec3(0)
                               : RotateFromTo(glm::vec3(tangentsY[0]), q0, q);
    const glm::vec3 end1 = pointedEnds[1]
                               ? glm::vec3(0)
                               : RotateFromTo(glm::vec3(tangentsY[1]), q1, q);
    const glm::vec3 delta = glm::mix(end0, end1, x);
    const float deltaW = glm::mix(tangentsY[0].w, tangentsY[1].w, x);

    return {Homogeneous(end), glm::vec4(delta, deltaW)};
  }

  glm::vec3 Bezier2D(const glm::mat4x3& corners, const glm::mat4& tangentsX,
                     const glm::mat4& tangentsY, float x, float y,
                     const glm::vec3& centroid, bool isTriangle) const {
    glm::mat2x4 bez0 = Bezier2Bezier(
        {corners[0], corners[1]}, {tangentsX[0], tangentsX[1]},
        {tangentsY[0], tangentsY[1]}, x, {isTriangle, false}, centroid);
    glm::mat2x4 bez1 = Bezier2Bezier(
        {corners[2], corners[3]}, {tangentsX[2], tangentsX[3]},
        {tangentsY[2], tangentsY[3]}, 1 - x, {false, isTriangle}, centroid);

    const float flatLen =
        isTriangle ? x * glm::length(corners[1] - corners[2])
                   : glm::length(glm::mix(corners[0], corners[1], x) -
                                 glm::mix(corners[3], corners[2], x));
    const float scale = glm::length(glm::vec3(bez0[0] - bez1[0])) / flatLen;

    const glm::mat2x4 bez = CubicBezier2Linear(
        bez0[0], Bezier(glm::vec3(bez0[0]), Scale(bez0[1], scale)),
        Bezier(glm::vec3(bez1[0]), Scale(bez1[1], scale)), bez1[0], y);
    return BezierPoint(bez, y);
  }

  void operator()(thrust::tuple<glm::vec3&, Barycentric> inOut) {
    glm::vec3& pos = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut).tri;
    const glm::vec4 uvw = thrust::get<1>(inOut).uvw;

    const glm::ivec4 halfedges = impl->GetHalfedges(tri);
    const glm::mat4x3 corners = {
        impl->vertPos_[impl->halfedge_[halfedges[0]].startVert],
        impl->vertPos_[impl->halfedge_[halfedges[1]].startVert],
        impl->vertPos_[impl->halfedge_[halfedges[2]].startVert],
        halfedges[3] < 0
            ? glm::vec3(0)
            : impl->vertPos_[impl->halfedge_[halfedges[3]].startVert]};

    for (const int i : {0, 1, 2, 3}) {
      if (uvw[i] == 1) {
        pos = corners[i];
        return;
      }
    }

    glm::vec4 posH(0);

    if (halfedges[3] < 0) {  // tri
      const glm::mat3x4 tangentR = {impl->halfedgeTangent_[halfedges[0]],
                                    impl->halfedgeTangent_[halfedges[1]],
                                    impl->halfedgeTangent_[halfedges[2]]};
      const glm::mat3x4 tangentL = {
          impl->halfedgeTangent_[impl->halfedge_[halfedges[2]].pairedHalfedge],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[0]].pairedHalfedge],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[1]].pairedHalfedge]};
      const glm::vec3 centroid = glm::mat3(corners) * glm::vec3(1.0f / 3);

      for (const int i : {0, 1, 2}) {
        const int j = Next3(i);
        const int k = Prev3(i);
        const float x = uvw[k] / (1 - uvw[i]);
        const glm::vec3 p =
            Bezier2D({corners[i], corners[j], corners[k], corners[i]},
                     {tangentR[i], tangentL[j], tangentR[k], tangentL[i]},
                     {tangentL[i], tangentR[j], tangentL[k], tangentR[i]},
                     1 - uvw[i], x, centroid, true);
        posH += Homogeneous(glm::vec4(p, uvw[i]));
      }
    } else {  // quad
      const glm::mat4 tangentsX = {
          impl->halfedgeTangent_[halfedges[0]],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[0]].pairedHalfedge],
          impl->halfedgeTangent_[halfedges[2]],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[2]].pairedHalfedge]};
      const glm::mat4 tangentsY = {
          impl->halfedgeTangent_[impl->halfedge_[halfedges[3]].pairedHalfedge],
          impl->halfedgeTangent_[halfedges[1]],
          impl->halfedgeTangent_[impl->halfedge_[halfedges[1]].pairedHalfedge],
          impl->halfedgeTangent_[halfedges[3]]};
      const glm::vec3 centroid = corners * glm::vec4(0.25);
      const float x = uvw[1] + uvw[2];
      const float y = uvw[2] + uvw[3];
      const glm::vec3 pX =
          Bezier2D(corners, tangentsX, tangentsY, x, y, centroid, false);
      const glm::vec3 pY =
          Bezier2D({corners[1], corners[2], corners[3], corners[0]},
                   {tangentsY[1], tangentsY[2], tangentsY[3], tangentsY[0]},
                   {tangentsX[1], tangentsX[2], tangentsX[3], tangentsX[0]}, y,
                   1 - x, centroid, false);
      posH += Homogeneous(glm::vec4(pX, x * (1 - x)));
      posH += Homogeneous(glm::vec4(pY, y * (1 - y)));
    }
    pos = HNormalize(posH);
  }
};
}  // namespace

namespace manifold {

/**
 * Get the property normal associated with the startVert of this halfedge, where
 * normalIdx shows the beginning of where normals are stored in the properties.
 */
glm::vec3 Manifold::Impl::GetNormal(int halfedge, int normalIdx) const {
  const int tri = halfedge / 3;
  const int j = halfedge % 3;
  const int prop = meshRelation_.triProperties[tri][j];
  glm::vec3 normal;
  for (const int i : {0, 1, 2}) {
    normal[i] =
        meshRelation_.properties[prop * meshRelation_.numProp + normalIdx + i];
  }
  return normal;
}

/**
 * Returns true if this halfedge should be marked as the interior of a quad, as
 * defined by its two triangles referring to the same face, and those triangles
 * having no further face neighbors beyond.
 */
bool Manifold::Impl::IsInsideQuad(int halfedge) const {
  if (halfedgeTangent_.size() > 0) {
    return halfedgeTangent_[halfedge].w < 0;
  }
  const int tri = halfedge_[halfedge].face;
  const TriRef ref = meshRelation_.triRef[tri];
  const int pair = halfedge_[halfedge].pairedHalfedge;
  const int pairTri = halfedge_[pair].face;
  const TriRef pairRef = meshRelation_.triRef[pairTri];
  if (!ref.SameFace(pairRef)) return false;

  auto SameFace = [this](int halfedge, const TriRef& ref) {
    return ref.SameFace(
        meshRelation_.triRef[halfedge_[halfedge].pairedHalfedge / 3]);
  };

  int neighbor = NextHalfedge(halfedge);
  if (SameFace(neighbor, ref)) return false;
  neighbor = NextHalfedge(neighbor);
  if (SameFace(neighbor, ref)) return false;
  neighbor = NextHalfedge(pair);
  if (SameFace(neighbor, pairRef)) return false;
  neighbor = NextHalfedge(neighbor);
  if (SameFace(neighbor, pairRef)) return false;
  return true;
}

/**
 * Returns true if this halfedge is an interior of a quad, as defined by its
 * halfedge tangent having negative weight.
 */
bool Manifold::Impl::IsMarkedInsideQuad(int halfedge) const {
  return halfedgeTangent_.size() > 0 && halfedgeTangent_[halfedge].w < 0;
}

// sharpenedEdges are referenced to the input Mesh, but the triangles have
// been sorted in creating the Manifold, so the indices are converted using
// meshRelation_.
std::vector<Smoothness> Manifold::Impl::UpdateSharpenedEdges(
    const std::vector<Smoothness>& sharpenedEdges) const {
  std::unordered_map<int, int> oldHalfedge2New;
  for (int tri = 0; tri < NumTri(); ++tri) {
    int oldTri = meshRelation_.triRef[tri].tri;
    for (int i : {0, 1, 2}) oldHalfedge2New[3 * oldTri + i] = 3 * tri + i;
  }
  std::vector<Smoothness> newSharp = sharpenedEdges;
  for (Smoothness& edge : newSharp) {
    edge.halfedge = oldHalfedge2New[edge.halfedge];
  }
  return newSharp;
}

// Find faces containing at least 3 triangles - these will not have
// interpolated normals - all their vert normals must match their face normal.
Vec<bool> Manifold::Impl::FlatFaces() const {
  const int numTri = NumTri();
  Vec<bool> triIsFlatFace(numTri, false);
  for_each_n(autoPolicy(numTri), countAt(0), numTri,
             [this, &triIsFlatFace](const int tri) {
               const TriRef& ref = meshRelation_.triRef[tri];
               int faceNeighbors = 0;
               glm::ivec3 faceTris = {-1, -1, -1};
               for (const int j : {0, 1, 2}) {
                 const int neighborTri =
                     halfedge_[halfedge_[3 * tri + j].pairedHalfedge].face;
                 const TriRef& jRef = meshRelation_.triRef[neighborTri];
                 if (jRef.SameFace(ref)) {
                   ++faceNeighbors;
                   faceTris[j] = neighborTri;
                 }
               }
               if (faceNeighbors > 1) {
                 triIsFlatFace[tri] = true;
                 for (const int j : {0, 1, 2}) {
                   if (faceTris[j] >= 0) {
                     triIsFlatFace[faceTris[j]] = true;
                   }
                 }
               }
             });
  return triIsFlatFace;
}

// Returns a vector of length numVert that has a tri that is part of a
// neighboring flat face if there is only one flat face. If there are none it
// gets -1, and if there are more than one it gets -2.
Vec<int> Manifold::Impl::VertFlatFace(const Vec<bool>& flatFaces) const {
  Vec<int> vertFlatFace(NumVert(), -1);
  Vec<TriRef> vertRef(NumVert(), {-1, -1, -1});
  for (int tri = 0; tri < NumTri(); ++tri) {
    if (flatFaces[tri]) {
      for (const int j : {0, 1, 2}) {
        const int vert = halfedge_[3 * tri + j].startVert;
        if (vertRef[vert].SameFace(meshRelation_.triRef[tri])) continue;
        vertRef[vert] = meshRelation_.triRef[tri];
        vertFlatFace[vert] = vertFlatFace[vert] == -1 ? tri : -2;
      }
    }
  }
  return vertFlatFace;
}

std::vector<Smoothness> Manifold::Impl::SharpenEdges(
    float minSharpAngle, float minSmoothness) const {
  std::vector<Smoothness> sharpenedEdges;
  const float minRadians = glm::radians(minSharpAngle);
  for (int e = 0; e < halfedge_.size(); ++e) {
    if (!halfedge_[e].IsForward()) continue;
    const int pair = halfedge_[e].pairedHalfedge;
    const float dihedral =
        glm::acos(glm::dot(faceNormal_[e / 3], faceNormal_[pair / 3]));
    if (dihedral > minRadians) {
      sharpenedEdges.push_back({e, minSmoothness});
      sharpenedEdges.push_back({pair, minSmoothness});
    }
  }
  return sharpenedEdges;
}

/**
 * Sharpen tangents that intersect an edge to sharpen that edge. The weight is
 * unchanged, as this has a squared effect on radius of curvature, except
 * in the case of zero radius, which is marked with weight = 0.
 */
void Manifold::Impl::SharpenTangent(int halfedge, float smoothness) {
  halfedgeTangent_[halfedge] =
      glm::vec4(smoothness * glm::vec3(halfedgeTangent_[halfedge]),
                smoothness == 0 ? 0 : halfedgeTangent_[halfedge].w);
}

/**
 * Instead of calculating the internal shared normals like CalculateNormals
 * does, this method fills in vertex properties, unshared across edges that
 * are bent more than minSharpAngle.
 */
void Manifold::Impl::SetNormals(int normalIdx, float minSharpAngle) {
  if (IsEmpty()) return;
  if (normalIdx < 0) return;

  const int oldNumProp = NumProp();
  const int numTri = NumTri();

  Vec<bool> triIsFlatFace = FlatFaces();
  Vec<int> vertFlatFace = VertFlatFace(triIsFlatFace);
  Vec<int> vertNumSharp(NumVert(), 0);
  for (int e = 0; e < halfedge_.size(); ++e) {
    if (!halfedge_[e].IsForward()) continue;
    const int pair = halfedge_[e].pairedHalfedge;
    const int tri1 = e / 3;
    const int tri2 = pair / 3;
    const float dihedral =
        glm::degrees(glm::acos(glm::dot(faceNormal_[tri1], faceNormal_[tri2])));
    if (dihedral > minSharpAngle) {
      ++vertNumSharp[halfedge_[e].startVert];
      ++vertNumSharp[halfedge_[e].endVert];
    } else {
      const bool faceSplit =
          triIsFlatFace[tri1] != triIsFlatFace[tri2] ||
          (triIsFlatFace[tri1] && triIsFlatFace[tri2] &&
           !meshRelation_.triRef[tri1].SameFace(meshRelation_.triRef[tri2]));
      if (vertFlatFace[halfedge_[e].startVert] == -2 && faceSplit) {
        ++vertNumSharp[halfedge_[e].startVert];
      }
      if (vertFlatFace[halfedge_[e].endVert] == -2 && faceSplit) {
        ++vertNumSharp[halfedge_[e].endVert];
      }
    }
  }

  const int numProp = glm::max(oldNumProp, normalIdx + 3);
  Vec<float> oldProperties(numProp * NumPropVert(), 0);
  meshRelation_.properties.swap(oldProperties);
  meshRelation_.numProp = numProp;
  if (meshRelation_.triProperties.size() == 0) {
    meshRelation_.triProperties.resize(numTri);
    for_each_n(autoPolicy(numTri), countAt(0), numTri, [this](int tri) {
      for (const int j : {0, 1, 2})
        meshRelation_.triProperties[tri][j] = halfedge_[3 * tri + j].startVert;
    });
  }
  Vec<glm::ivec3> oldTriProp(numTri, {-1, -1, -1});
  meshRelation_.triProperties.swap(oldTriProp);

  for (int tri = 0; tri < numTri; ++tri) {
    for (const int i : {0, 1, 2}) {
      if (meshRelation_.triProperties[tri][i] >= 0) continue;
      int startEdge = 3 * tri + i;
      const int vert = halfedge_[startEdge].startVert;

      if (vertNumSharp[vert] < 2) {
        const glm::vec3 normal = vertFlatFace[vert] >= 0
                                     ? faceNormal_[vertFlatFace[vert]]
                                     : vertNormal_[vert];
        int lastProp = -1;
        ForVert(startEdge, [&](int current) {
          const int thisTri = current / 3;
          const int j = current - 3 * thisTri;
          const int prop = oldTriProp[thisTri][j];
          meshRelation_.triProperties[thisTri][j] = prop;
          if (prop == lastProp) return;
          lastProp = prop;
          auto start = oldProperties.begin() + prop * oldNumProp;
          std::copy(start, start + oldNumProp,
                    meshRelation_.properties.begin() + prop * numProp);
          for (const int i : {0, 1, 2})
            meshRelation_.properties[prop * numProp + normalIdx + i] =
                normal[i];
        });
      } else {
        const glm::vec3 centerPos = vertPos_[vert];
        // Length degree
        std::vector<int> group;
        // Length number of normals
        std::vector<glm::vec3> normals;
        int current = startEdge;
        int prevFace = halfedge_[current].face;

        do {
          int next = NextHalfedge(halfedge_[current].pairedHalfedge);
          const int face = halfedge_[next].face;

          const float dihedral = glm::degrees(
              glm::acos(glm::dot(faceNormal_[face], faceNormal_[prevFace])));
          if (dihedral > minSharpAngle ||
              triIsFlatFace[face] != triIsFlatFace[prevFace] ||
              (triIsFlatFace[face] && triIsFlatFace[prevFace] &&
               !meshRelation_.triRef[face].SameFace(
                   meshRelation_.triRef[prevFace]))) {
            break;
          }
          current = next;
          prevFace = face;
        } while (current != startEdge);

        const int endEdge = current;

        struct FaceEdge {
          int face;
          glm::vec3 edgeVec;
        };

        ForVert<FaceEdge>(
            endEdge,
            [this, centerPos](int current) {
              return FaceEdge(
                  {halfedge_[current].face,
                   glm::normalize(vertPos_[halfedge_[current].endVert] -
                                  centerPos)});
            },
            [this, &triIsFlatFace, &normals, &group, minSharpAngle](
                int current, const FaceEdge& here, const FaceEdge& next) {
              const float dihedral = glm::degrees(glm::acos(
                  glm::dot(faceNormal_[here.face], faceNormal_[next.face])));
              if (dihedral > minSharpAngle ||
                  triIsFlatFace[here.face] != triIsFlatFace[next.face] ||
                  (triIsFlatFace[here.face] && triIsFlatFace[next.face] &&
                   !meshRelation_.triRef[here.face].SameFace(
                       meshRelation_.triRef[next.face]))) {
                normals.push_back(glm::vec3(0));
              }
              group.push_back(normals.size() - 1);
              float dot = glm::dot(here.edgeVec, next.edgeVec);
              const float phi =
                  dot >= 1 ? kTolerance
                           : (dot <= -1 ? glm::pi<float>() : glm::acos(dot));
              normals.back() += faceNormal_[next.face] * phi;
            });

        for (auto& normal : normals) {
          normal = glm::normalize(normal);
        }

        int lastGroup = 0;
        int lastProp = -1;
        int newProp = -1;
        int idx = 0;
        ForVert(endEdge, [&](int current1) {
          const int thisTri = current1 / 3;
          const int j = current1 - 3 * thisTri;
          const int prop = oldTriProp[thisTri][j];
          auto start = oldProperties.begin() + prop * oldNumProp;

          if (group[idx] != lastGroup && group[idx] != 0 && prop == lastProp) {
            lastGroup = group[idx];
            newProp = NumPropVert();
            meshRelation_.properties.resize(meshRelation_.properties.size() +
                                            numProp);
            std::copy(start, start + oldNumProp,
                      meshRelation_.properties.begin() + newProp * numProp);
            for (const int i : {0, 1, 2}) {
              meshRelation_.properties[newProp * numProp + normalIdx + i] =
                  normals[group[idx]][i];
            }
          } else if (prop != lastProp) {
            lastProp = prop;
            newProp = prop;
            std::copy(start, start + oldNumProp,
                      meshRelation_.properties.begin() + prop * numProp);
            for (const int i : {0, 1, 2})
              meshRelation_.properties[prop * numProp + normalIdx + i] =
                  normals[group[idx]][i];
          }

          meshRelation_.triProperties[thisTri][j] = newProp;
          ++idx;
        });
      }
    }
  }
}

/**
 * Tangents get flattened to create sharp edges by setting their weight to zero.
 * This is the natural limit of reducing the weight to increase the sharpness
 * smoothly. This limit gives a decent shape, but it causes the parameterization
 * to be stretched and compresses it near the edges, which is good for resolving
 * tight curvature, but bad for property interpolation. This function fixes the
 * parameter stretch at the limit for sharp edges, since there is no curvature
 * to resolve. Note this also changes the overall shape - making it more evenly
 * curved.
 */
void Manifold::Impl::LinearizeFlatTangents() {
  const int n = halfedgeTangent_.size();
  for_each_n(
      autoPolicy(n), zip(halfedgeTangent_.begin(), countAt(0)), n,
      [this](thrust::tuple<glm::vec4&, int> inOut) {
        glm::vec4& tangent = thrust::get<0>(inOut);
        const int halfedge = thrust::get<1>(inOut);
        glm::vec4& otherTangent =
            halfedgeTangent_[halfedge_[halfedge].pairedHalfedge];

        const glm::bvec2 flat(tangent.w == 0, otherTangent.w == 0);
        if (!halfedge_[halfedge].IsForward() || (!flat[0] && !flat[1])) {
          return;
        }

        const glm::vec3 edgeVec = vertPos_[halfedge_[halfedge].endVert] -
                                  vertPos_[halfedge_[halfedge].startVert];

        if (flat[0] && flat[1]) {
          tangent = glm::vec4(edgeVec / 3.0f, 1);
          otherTangent = glm::vec4(-edgeVec / 3.0f, 1);
        } else if (flat[0]) {
          tangent = glm::vec4((edgeVec + glm::vec3(otherTangent)) / 2.0f, 1);
        } else {
          otherTangent = glm::vec4((-edgeVec + glm::vec3(tangent)) / 2.0f, 1);
        }
      });
}

/**
 * Calculates halfedgeTangent_, allowing the manifold to be refined and
 * smoothed. The tangents form weighted cubic Beziers along each edge. This
 * function creates circular arcs where possible (minimizing maximum curvature),
 * constrained to the indicated property normals. Across edges that form
 * discontinuities in the normals, the tangent vectors are zero-length, allowing
 * the shape to form a sharp corner with minimal oscillation.
 */
void Manifold::Impl::CreateTangents(int normalIdx) {
  ZoneScoped;
  const int numVert = NumVert();
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.resize(0);
  Vec<glm::vec4> tangent(numHalfedge);

  Vec<glm::vec3> vertNormal(numVert);
  Vec<glm::ivec2> vertSharpHalfedge(numVert, glm::ivec2(-1));
  for (int e = 0; e < numHalfedge; ++e) {
    const int vert = halfedge_[e].startVert;
    auto& sharpHalfedge = vertSharpHalfedge[vert];
    if (sharpHalfedge[0] >= 0 && sharpHalfedge[1] >= 0) continue;

    int idx = 0;
    // Only used when there is only one.
    glm::vec3& lastNormal = vertNormal[vert];

    ForVert<glm::vec3>(
        e,
        [normalIdx, this](int halfedge) {
          return GetNormal(halfedge, normalIdx);
        },
        [&sharpHalfedge, &idx, &lastNormal](int halfedge,
                                            const glm::vec3& normal,
                                            const glm::vec3& nextNormal) {
          const glm::vec3 diff = nextNormal - normal;
          if (glm::dot(diff, diff) > kTolerance * kTolerance) {
            if (idx < 2) {
              sharpHalfedge[idx++] = halfedge;
            } else {
              sharpHalfedge[0] = -1;  // marks more than 2 sharp edges
            }
          }
          lastNormal = normal;
        });
  }

  for_each_n(autoPolicy(numHalfedge),
             zip(tangent.begin(), halfedge_.cbegin(), countAt(0)), numHalfedge,
             SmoothBezier({this, vertNormal}));

  halfedgeTangent_.swap(tangent);

  for (int vert = 0; vert < numVert; ++vert) {
    const int first = vertSharpHalfedge[vert][0];
    const int second = vertSharpHalfedge[vert][1];
    if (second == -1) continue;
    if (first != -1) {  // Make continuous edge
      const glm::vec3 newTangent = glm::normalize(glm::cross(
          GetNormal(first, normalIdx), GetNormal(second, normalIdx)));
      if (!isfinite(newTangent[0])) continue;

      halfedgeTangent_[first] = CircularTangent(
          newTangent, vertPos_[halfedge_[first].endVert] - vertPos_[vert]);
      halfedgeTangent_[second] = CircularTangent(
          -newTangent, vertPos_[halfedge_[second].endVert] - vertPos_[vert]);

      ForVert(first, [this, first, second](int current) {
        if (current != first && current != second &&
            !IsMarkedInsideQuad(current)) {
          SharpenTangent(current, 0);
        }
      });
    } else {  // Sharpen vertex uniformly
      ForVert(second, [this](int current) {
        if (!IsMarkedInsideQuad(current)) {
          SharpenTangent(current, 0);
        }
      });
    }
  }
  LinearizeFlatTangents();
}

/**
 * Calculates halfedgeTangent_, allowing the manifold to be refined and
 * smoothed. The tangents form weighted cubic Beziers along each edge. This
 * function creates circular arcs where possible (minimizing maximum curvature),
 * constrained to the vertex normals. Where sharpenedEdges are specified, the
 * tangents are shortened that intersect the sharpened edge, concentrating the
 * curvature there, while the tangents of the sharp edges themselves are aligned
 * for continuity.
 */
void Manifold::Impl::CreateTangents(std::vector<Smoothness> sharpenedEdges) {
  ZoneScoped;
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.resize(0);
  Vec<glm::vec4> tangent(numHalfedge);

  Vec<bool> triIsFlatFace = FlatFaces();
  Vec<int> vertFlatFace = VertFlatFace(triIsFlatFace);
  Vec<glm::vec3> vertNormal = vertNormal_;
  for (int v = 0; v < NumVert(); ++v) {
    if (vertFlatFace[v] >= 0) {
      vertNormal[v] = faceNormal_[vertFlatFace[v]];
    }
  }

  for_each_n(autoPolicy(numHalfedge),
             zip(tangent.begin(), halfedge_.cbegin(), countAt(0)), numHalfedge,
             SmoothBezier({this, vertNormal}));

  halfedgeTangent_.swap(tangent);

  // Add sharpened edges around faces, just on the face side.
  for (int tri = 0; tri < NumTri(); ++tri) {
    if (!triIsFlatFace[tri]) continue;
    for (const int j : {0, 1, 2}) {
      const int tri2 = halfedge_[3 * tri + j].pairedHalfedge / 3;
      if (!triIsFlatFace[tri2] ||
          !meshRelation_.triRef[tri].SameFace(meshRelation_.triRef[tri2])) {
        sharpenedEdges.push_back({3 * tri + j, 0});
      }
    }
  }

  if (sharpenedEdges.empty()) return;

  using Pair = std::pair<Smoothness, Smoothness>;
  // Fill in missing pairs with default smoothness = 1.
  std::map<int, Pair> edges;
  for (Smoothness edge : sharpenedEdges) {
    if (edge.smoothness >= 1) continue;
    const bool forward = halfedge_[edge.halfedge].IsForward();
    const int pair = halfedge_[edge.halfedge].pairedHalfedge;
    const int idx = forward ? edge.halfedge : pair;
    if (edges.find(idx) == edges.end()) {
      edges[idx] = {edge, {pair, 1}};
      if (!forward) std::swap(edges[idx].first, edges[idx].second);
    } else {
      Smoothness& e = forward ? edges[idx].first : edges[idx].second;
      e.smoothness = glm::min(edge.smoothness, e.smoothness);
    }
  }

  std::map<int, std::vector<Pair>> vertTangents;
  for (const auto& value : edges) {
    const Pair edge = value.second;
    vertTangents[halfedge_[edge.first.halfedge].startVert].push_back(edge);
    vertTangents[halfedge_[edge.second.halfedge].startVert].push_back(
        {edge.second, edge.first});
  }

  for (const auto& value : vertTangents) {
    const std::vector<Pair>& vert = value.second;
    // Sharp edges that end are smooth at their terminal vert.
    if (vert.size() == 1) continue;
    if (vert.size() == 2) {  // Make continuous edge
      const int first = vert[0].first.halfedge;
      const int second = vert[1].first.halfedge;
      const glm::vec3 newTangent =
          glm::normalize(glm::vec3(halfedgeTangent_[first]) -
                         glm::vec3(halfedgeTangent_[second]));

      const glm::vec3 pos = vertPos_[halfedge_[first].startVert];
      halfedgeTangent_[first] =
          CircularTangent(newTangent, vertPos_[halfedge_[first].endVert] - pos);
      halfedgeTangent_[second] = CircularTangent(
          -newTangent, vertPos_[halfedge_[second].endVert] - pos);

      float smoothness =
          (vert[0].second.smoothness + vert[1].first.smoothness) / 2;
      ForVert(first, [this, &smoothness, &vert, first, second](int current) {
        if (current == second) {
          smoothness =
              (vert[1].second.smoothness + vert[0].first.smoothness) / 2;
        } else if (current != first && !IsMarkedInsideQuad(current)) {
          SharpenTangent(current, smoothness);
        }
      });
    } else {  // Sharpen vertex uniformly
      float smoothness = 0;
      for (const Pair& pair : vert) {
        smoothness += pair.first.smoothness;
        smoothness += pair.second.smoothness;
      }
      smoothness /= 2 * vert.size();

      ForVert(vert[0].first.halfedge, [this, smoothness](int current) {
        if (!IsMarkedInsideQuad(current)) {
          SharpenTangent(current, smoothness);
        }
      });
    }
  }
  LinearizeFlatTangents();
}

void Manifold::Impl::Refine(std::function<int(glm::vec3)> edgeDivisions) {
  if (IsEmpty()) return;
  Manifold::Impl old = *this;
  Vec<Barycentric> vertBary = Subdivide(edgeDivisions);
  if (vertBary.size() == 0) return;

  if (old.halfedgeTangent_.size() == old.halfedge_.size()) {
    for_each_n(autoPolicy(NumTri()), zip(vertPos_.begin(), vertBary.begin()),
               NumVert(), InterpTri({&old}));
    // Make original since the subdivided faces have been warped into
    // being non-coplanar, and hence not being related to the original faces.
    meshRelation_.originalID = ReserveIDs(1);
    InitializeOriginal();
    CreateFaces();
  }

  halfedgeTangent_.resize(0);
  Finish();
}

}  // namespace manifold
