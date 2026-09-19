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

#include "execution_impl.h"
#include "impl.h"
#include "parallel.h"

namespace {
using namespace manifold;

// Returns a normalized vector orthogonal to ref, in the plane of ref and in,
// unless in and ref are colinear, in which case it falls back to the plane of
// ref and altIn.
vec3 OrthogonalTo(vec3 in, vec3 altIn, vec3 ref) {
  vec3 out = in - la::dot(in, ref) * ref;
  if (la::dot(out, out) < kPrecision * la::dot(in, in)) {
    out = altIn - la::dot(altIn, ref) * ref;
  }
  return SafeNormalize(out);
}

double Wrap(double radians) {
  return radians < -kPi  ? radians + kTwoPi
         : radians > kPi ? radians - kTwoPi
                         : radians;
}

// Minimum sharp angle in degrees, below which edges are considered coplanar.
// Floating point noise in the dihedral angle computation can reach ~1e-6
// degrees for nearly-parallel face normals; this threshold must exceed that.
constexpr double kMinSharpAngle = 1e-4;

// Get the angle between two unit-vectors.
double AngleBetween(vec3 a, vec3 b) {
  const double dot = la::dot(a, b);
  return dot >= 1 ? 0 : (dot <= -1 ? kPi : math::acos(dot));
}

bool EqualNormals(vec3 a, vec3 b) {
  return la::dot(SafeNormalize(a), SafeNormalize(b)) > 0.9999;
}

// Calculate a tangent vector in the form of a weighted cubic Bezier taking as
// input the desired tangent direction (length doesn't matter) and the edge
// vector to the neighboring vertex. In a symmetric situation where the tangents
// at each end are mirror images of each other, this will result in a circular
// arc.
vec4 CircularTangent(const vec3& tangent, const vec3& edgeVec) {
  const vec3 dir = SafeNormalize(tangent);

  double weight = std::max(0.5, la::dot(dir, SafeNormalize(edgeVec)));
  // Quadratic weighted bezier for circular interpolation
  const vec4 bz2 = vec4(dir * 0.5 * la::length(edgeVec), weight);
  // Equivalent cubic weighted bezier
  const vec4 bz3 = la::lerp(vec4(0, 0, 0, 1), bz2, 2 / 3.0);
  // Convert from homogeneous form to geometric form
  return vec4(vec3(bz3) / bz3.w, bz3.w);
}

struct InterpTri {
  VecView<vec3> vertPos;
  VecView<const Barycentric> vertBary;
  const Manifold::Impl* impl;

  static vec4 Homogeneous(vec4 v) {
    v.x *= v.w;
    v.y *= v.w;
    v.z *= v.w;
    return v;
  }

  static vec4 Homogeneous(vec3 v) { return vec4(v, 1.0); }

  static vec3 HNormalize(vec4 v) {
    return v.w == 0 ? vec3(v) : (vec3(v) / v.w);
  }

  static vec4 Scale(vec4 v, double scale) { return vec4(scale * vec3(v), v.w); }

  static vec4 Bezier(vec3 point, vec4 tangent) {
    return Homogeneous(vec4(point, 0) + tangent);
  }

  static mat4x2 CubicBezier2Linear(vec4 p0, vec4 p1, vec4 p2, vec4 p3,
                                   double x) {
    mat4x2 out;
    vec4 p12 = la::lerp(p1, p2, x);
    out[0] = la::lerp(la::lerp(p0, p1, x), p12, x);
    out[1] = la::lerp(p12, la::lerp(p2, p3, x), x);
    return out;
  }

  static vec3 BezierPoint(mat4x2 points, double x) {
    return HNormalize(la::lerp(points[0], points[1], x));
  }

  static vec3 BezierTangent(mat4x2 points) {
    return SafeNormalize(HNormalize(points[1]) - HNormalize(points[0]));
  }

  static vec3 RotateFromTo(vec3 v, quat start, quat end) {
    return la::qrot(end, la::qrot(la::qconj(start), v));
  }

  static quat Slerp(const quat& x, const quat& y, double a, bool longWay) {
    quat z = y;
    double cosTheta = la::dot(x, y);

    // Take the long way around the sphere only when requested
    if ((cosTheta < 0) != longWay) {
      z = -y;
      cosTheta = -cosTheta;
    }

    if (std::abs(cosTheta) > 1.0 - std::numeric_limits<double>::epsilon()) {
      return la::lerp(x, z, a);  // for numerical stability
    } else {
      double angle = math::acos(cosTheta);
      return (math::sin((1.0 - a) * angle) * x + math::sin(a * angle) * z) /
             math::sin(angle);
    }
  }

  static mat4x2 Bezier2Bezier(const mat3x2& corners, const mat4x2& tangentsX,
                              const mat4x2& tangentsY, double x,
                              const vec3& anchor) {
    const mat4x2 bez = CubicBezier2Linear(
        Homogeneous(corners[0]), Bezier(corners[0], tangentsX[0]),
        Bezier(corners[1], tangentsX[1]), Homogeneous(corners[1]), x);
    const vec3 end = BezierPoint(bez, x);
    const vec3 tangent = BezierTangent(bez);

    const mat3x2 nTangentsX(SafeNormalize(vec3(tangentsX[0])),
                            -SafeNormalize(vec3(tangentsX[1])));
    const mat3x2 biTangents = {
        OrthogonalTo(vec3(tangentsY[0]), (anchor - corners[0]), nTangentsX[0]),
        OrthogonalTo(vec3(tangentsY[1]), (anchor - corners[1]), nTangentsX[1])};

    const quat q0 = la::rotation_quat(mat3(
        nTangentsX[0], biTangents[0], la::cross(nTangentsX[0], biTangents[0])));
    const quat q1 = la::rotation_quat(mat3(
        nTangentsX[1], biTangents[1], la::cross(nTangentsX[1], biTangents[1])));
    const vec3 edge = corners[1] - corners[0];
    const bool longWay =
        la::dot(nTangentsX[0], edge) + la::dot(nTangentsX[1], edge) < 0;
    const quat qTmp = Slerp(q0, q1, x, longWay);
    const quat q = la::qmul(la::rotation_quat(la::qxdir(qTmp), tangent), qTmp);

    const vec3 delta = la::lerp(RotateFromTo(vec3(tangentsY[0]), q0, q),
                                RotateFromTo(vec3(tangentsY[1]), q1, q), x);
    const double deltaW = la::lerp(tangentsY[0].w, tangentsY[1].w, x);

    return {Homogeneous(end), vec4(delta, deltaW)};
  }

  static vec3 Bezier2D(const mat3x4& corners, const mat4& tangentsX,
                       const mat4& tangentsY, double x, double y,
                       const vec3& centroid) {
    mat4x2 bez0 =
        Bezier2Bezier({corners[0], corners[1]}, {tangentsX[0], tangentsX[1]},
                      {tangentsY[0], tangentsY[1]}, x, centroid);
    mat4x2 bez1 =
        Bezier2Bezier({corners[2], corners[3]}, {tangentsX[2], tangentsX[3]},
                      {tangentsY[2], tangentsY[3]}, 1 - x, centroid);

    const mat4x2 bez =
        CubicBezier2Linear(bez0[0], Bezier(vec3(bez0[0]), bez0[1]),
                           Bezier(vec3(bez1[0]), bez1[1]), bez1[0], y);
    return BezierPoint(bez, y);
  }

  void operator()(const int vert) {
    vec3& pos = vertPos[vert];
    const int tri = vertBary[vert].tri;
    const vec4 uvw = vertBary[vert].uvw;

    const ivec4 halfedges = impl->GetHalfedges(tri);
    const mat3x4 corners = {
        impl->vertPos_[impl->halfedge_.Start(halfedges[0])],
        impl->vertPos_[impl->halfedge_.Start(halfedges[1])],
        impl->vertPos_[impl->halfedge_.Start(halfedges[2])],
        halfedges[3] < 0 ? vec3(0.0)
                         : impl->vertPos_[impl->halfedge_.Start(halfedges[3])]};

    for (const int i : {0, 1, 2, 3}) {
      if (uvw[i] == 1) {
        pos = corners[i];
        return;
      }
    }

    vec4 posH(0.0);

    if (halfedges[3] < 0) {  // tri
      const mat4x3 tangentR = {impl->halfedgeTangent_[halfedges[0]],
                               impl->halfedgeTangent_[halfedges[1]],
                               impl->halfedgeTangent_[halfedges[2]]};
      const mat4x3 tangentL = {
          impl->halfedgeTangent_[impl->halfedge_.Pair(halfedges[2])],
          impl->halfedgeTangent_[impl->halfedge_.Pair(halfedges[0])],
          impl->halfedgeTangent_[impl->halfedge_.Pair(halfedges[1])]};
      const vec3 centroid = mat3(corners) * vec3(1.0 / 3);

      for (const int i : {0, 1, 2}) {
        const int j = Next3(i);
        const int k = Prev3(i);
        const double x = uvw[k] / (1 - uvw[i]);

        const mat4x2 bez =
            Bezier2Bezier({corners[j], corners[k]}, {tangentR[j], tangentL[k]},
                          {tangentL[j], tangentR[k]}, x, centroid);

        const mat4x2 bez1 = CubicBezier2Linear(
            bez[0], Bezier(vec3(bez[0]), bez[1]),
            Bezier(corners[i], la::lerp(tangentR[i], tangentL[i], x)),
            Homogeneous(corners[i]), uvw[i]);
        const vec3 p = BezierPoint(bez1, uvw[i]);
        posH += Homogeneous(vec4(p, uvw[j] * uvw[k]));
      }
    } else {  // quad
      const mat4 tangentsX = {
          impl->halfedgeTangent_[halfedges[0]],
          impl->halfedgeTangent_[impl->halfedge_.Pair(halfedges[0])],
          impl->halfedgeTangent_[halfedges[2]],
          impl->halfedgeTangent_[impl->halfedge_.Pair(halfedges[2])]};
      const mat4 tangentsY = {
          impl->halfedgeTangent_[impl->halfedge_.Pair(halfedges[3])],
          impl->halfedgeTangent_[halfedges[1]],
          impl->halfedgeTangent_[impl->halfedge_.Pair(halfedges[1])],
          impl->halfedgeTangent_[halfedges[3]]};
      const vec3 centroid = corners * vec4(0.25);
      const double x = uvw[1] + uvw[2];
      const double y = uvw[2] + uvw[3];
      const vec3 pX = Bezier2D(corners, tangentsX, tangentsY, x, y, centroid);
      const vec3 pY =
          Bezier2D({corners[1], corners[2], corners[3], corners[0]},
                   {tangentsY[1], tangentsY[2], tangentsY[3], tangentsY[0]},
                   {tangentsX[1], tangentsX[2], tangentsX[3], tangentsX[0]}, y,
                   1 - x, centroid);
      posH += Homogeneous(vec4(pX, x * (1 - x)));
      posH += Homogeneous(vec4(pY, y * (1 - y)));
    }
    pos = HNormalize(posH);
    if (la::any(!la::isfinite(pos))) {
      pos = corners[0];
    }
  }
};
}  // namespace

namespace manifold {

/**
 * Get the property normal associated with the startVert of this halfedge, where
 * normalIdx shows the beginning of where normals are stored in the properties.
 */
vec3 Manifold::Impl::GetNormal(int halfedge, int normalIdx) const {
  const int prop = halfedge_.Prop(halfedge);
  vec3 normal;
  for (const int i : {0, 1, 2}) {
    normal[i] = properties_[prop * numProp_ + normalIdx + i];
  }
  // hasNormals=true means CalculateNormals (or a flagged round-trip) wrote
  // world-frame values, kept world-frame through Transform/Compose. Without
  // the flag, treat the slot as per-mesh frame and re-rotate to world like
  // pre-#1718 callers expected.
  const int meshID = meshRelation_.triRef[halfedge / 3].meshID;
  auto it = meshRelation_.meshIDtransform.find(meshID);
  if (it != meshRelation_.meshIDtransform.end() && !it->second.hasNormals) {
    normal = it->second.GetNormalTransform() * normal;
  }
  return normal;
}

/**
 * Returns a circular tangent for the requested halfedge, orthogonal to the
 * given normal vector, and avoiding folding when the tangent needs to be more
 * than 90 degrees from the edge vector.
 */
vec4 Manifold::Impl::TangentFromNormal(const vec3& normal, int halfedge) const {
  const vec3 edgeVec =
      vertPos_[halfedge_.End(halfedge)] - vertPos_[halfedge_.Start(halfedge)];
  const vec3 edgeNormal =
      faceNormal_[halfedge / 3] + faceNormal_[halfedge_.Pair(halfedge) / 3];
  const vec3 biTangent = la::dot(normal, edgeNormal) < 0
                             ? la::cross(edgeNormal, edgeVec)
                             : la::cross(normal, edgeVec);
  return CircularTangent(la::cross(biTangent, normal), edgeVec);
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
  const int tri = halfedge / 3;
  const TriRef ref = meshRelation_.triRef[tri];
  const int pair = halfedge_.Pair(halfedge);
  const int pairTri = pair / 3;
  const TriRef pairRef = meshRelation_.triRef[pairTri];
  if (!ref.SameFace(pairRef)) return false;

  auto SameFace = [this](int halfedge, const TriRef& ref) {
    return ref.SameFace(meshRelation_.triRef[halfedge_.Pair(halfedge) / 3]);
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
// meshRelation_.faceID, which temporarily holds the mapping.
std::vector<Smoothness> Manifold::Impl::UpdateSharpenedEdges(
    const std::vector<Smoothness>& sharpenedEdges) const {
  std::unordered_map<int, int> oldHalfedge2New;
  for (size_t tri = 0; tri < NumTri(); ++tri) {
    int oldTri = meshRelation_.triRef[tri].faceID;
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
  for_each_n(autoPolicy(numTri, 1e5), countAt(0), numTri,
             [this, &triIsFlatFace](const int tri) {
               const TriRef& ref = meshRelation_.triRef[tri];
               int faceNeighbors = 0;
               ivec3 faceTris = {-1, -1, -1};
               for (const int j : {0, 1, 2}) {
                 const int neighborTri = halfedge_.Pair(3 * tri + j) / 3;
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
  Vec<TriRef> vertRef(NumVert(), {-1, -1, -1, -1});
  for (size_t tri = 0; tri < NumTri(); ++tri) {
    if (flatFaces[tri]) {
      for (const int j : {0, 1, 2}) {
        const int vert = halfedge_.Start(3 * tri + j);
        if (vertRef[vert].SameFace(meshRelation_.triRef[tri])) continue;
        vertRef[vert] = meshRelation_.triRef[tri];
        vertFlatFace[vert] = vertFlatFace[vert] == -1 ? tri : -2;
      }
    }
  }
  return vertFlatFace;
}

Vec<int> Manifold::Impl::VertHalfedge() const {
  Vec<int> vertHalfedge(NumVert());
  Vec<uint8_t> counters(NumVert(), 0);
  for_each_n(autoPolicy(halfedge_.size(), 1e5), countAt(0), halfedge_.size(),
             [&vertHalfedge, &counters, this](const int idx) {
               const int start = halfedge_.Start(idx);
               auto old = std::atomic_exchange(
                   reinterpret_cast<std::atomic<uint8_t>*>(&counters[start]),
                   static_cast<uint8_t>(1));
               if (old == 1) return;
               // arbitrary, last one wins.
               vertHalfedge[start] = idx;
             });
  return vertHalfedge;
}

std::vector<Smoothness> Manifold::Impl::SharpenEdges(
    double minSharpAngle, double minSmoothness) const {
  std::vector<Smoothness> sharpenedEdges;
  minSharpAngle = std::max(minSharpAngle, kMinSharpAngle);
  const double minRadians = radians(minSharpAngle);
  for (size_t e = 0; e < halfedge_.size(); ++e) {
    if (!halfedge_.IsForward(e)) continue;
    const size_t pair = halfedge_.Pair(e);
    const double dihedral =
        AngleBetween(faceNormal_[e / 3], faceNormal_[pair / 3]);
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
void Manifold::Impl::SharpenTangent(int halfedge, double smoothness) {
  halfedgeTangent_[halfedge] =
      vec4(smoothness * vec3(halfedgeTangent_[halfedge]),
           smoothness == 0 ? 0 : halfedgeTangent_[halfedge].w);
}

/**
 * Instead of calculating the internal shared normals like CalculateNormals
 * does, this method fills in vertex properties, unshared across edges that
 * are bent more than minSharpAngle.
 */
void Manifold::Impl::SetNormals(int normalIdx, double minSharpAngle) {
  if (IsEmpty()) return;
  if (normalIdx < 0) return;
  // Ensure minSharpAngle is large enough to avoid treating nearly-coplanar
  // faces as sharp due to floating point noise in the dihedral computation.
  minSharpAngle = std::max(minSharpAngle, kMinSharpAngle);
  halfedge_.MakeUnique();

  const int oldNumProp = NumProp();

  Vec<int> vertNumSharp(NumVert(), 0);
  for (size_t e = 0; e < halfedge_.size(); ++e) {
    if (!halfedge_.IsForward(e)) continue;
    const int pair = halfedge_.Pair(e);
    const int tri1 = e / 3;
    const int tri2 = pair / 3;
    const double dihedral =
        degrees(AngleBetween(faceNormal_[tri1], faceNormal_[tri2]));
    if (dihedral > minSharpAngle) {
      ++vertNumSharp[halfedge_.Start(e)];
      ++vertNumSharp[halfedge_.End(e)];
    }
  }

  const int numProp = std::max(oldNumProp, normalIdx + 3);
  Vec<double> oldProperties(numProp * NumPropVert(), 0);
  properties_.swap(oldProperties);
  numProp_ = numProp;

  Vec<int> oldHalfedgeProp(halfedge_.size());
  for_each_n(autoPolicy(halfedge_.size(), 1e5), countAt(0), halfedge_.size(),
             [this, &oldHalfedgeProp](int i) {
               oldHalfedgeProp[i] = halfedge_.Prop(i);
               halfedge_.SetProp(i, -1);
             });

  // Cached per-meshID inverse-normal-transform for the legacy non-zero
  // normalIdx path. Lazily populated on first lookup; reused across all
  // verts in the loop below.
  // TODO: drop this and its only caller below when the non-zero normalIdx
  // parameter on CalculateNormals is removed.
  std::map<int, mat3> meshIDtoNormalTransform;
  auto getTransform = [&](int meshID) {
    if (meshIDtoNormalTransform.find(meshID) == meshIDtoNormalTransform.end()) {
      meshIDtoNormalTransform[meshID] =
          meshRelation_.meshIDtransform[meshID].GetInverseNormalTransform();
    }
    return meshIDtoNormalTransform[meshID];
  };

  const int numEdge = halfedge_.size();
  for (int startEdge = 0; startEdge < numEdge; ++startEdge) {
    if (halfedge_.Prop(startEdge) >= 0) continue;
    const int vert = halfedge_.Start(startEdge);

    if (vertNumSharp[vert] < 2) {  // vertex has single normal
      const vec3 worldNormal = vertNormal_[vert];
      // Non-zero normalIdx is the legacy deferred-transform path: store in
      // per-mesh frame so GetMeshGL's runTransform application on export
      // recovers world frame even after later transforms. Standard slot 0
      // uses the eager-transform contract: store world-frame directly.
      // Caveat: for legacy idx!=0, if a single propVert is shared between
      // triangles of different meshIDs, we pick startEdge's meshID for the
      // per-mesh-frame mapping. Other meshIDs reading the same propVert
      // through a different runTransform on export will get a wrong
      // rotation. Same shape as master; out of scope here.
      const vec3 normal =
          normalIdx == 0
              ? worldNormal
              : getTransform(meshRelation_.triRef[startEdge / 3].meshID) *
                    worldNormal;
      int lastProp = -1;
      ForVert(startEdge, [&](int current) {
        const int prop = oldHalfedgeProp[current];
        halfedge_.SetProp(current, prop);
        if (prop == lastProp) return;
        lastProp = prop;
        // update property vertex
        auto start = oldProperties.begin() + prop * oldNumProp;
        std::copy(start, start + oldNumProp,
                  properties_.begin() + prop * numProp);
        for (const int i : {0, 1, 2})
          properties_[prop * numProp + normalIdx + i] = normal[i];
      });
      continue;
    }

    // vertex has multiple normals
    const vec3 centerPos = vertPos_[vert];
    // Length degree
    std::vector<int> groups;
    // Length number of normals
    std::vector<vec3> normals;
    std::vector<int> meshIds;
    int current = startEdge;
    int prevFace = current / 3;

    do {  // find a sharp edge to start on
      int next = NextHalfedge(halfedge_.Pair(current));
      const int face = next / 3;

      const double dihedral =
          degrees(AngleBetween(faceNormal_[face], faceNormal_[prevFace]));
      if (dihedral > minSharpAngle) {
        break;
      }
      current = next;
      prevFace = face;
    } while (current != startEdge);

    const int endEdge = current;

    struct FaceEdge {
      int face;
      vec3 normalizedEdge;
    };

    // calculate pseudo-normals between each sharp edge
    ForVert<FaceEdge>(
        endEdge,
        [&](int current) {
          const int vert = halfedge_.End(current);
          return FaceEdge(
              {current / 3, SafeNormalize(vertPos_[vert] - centerPos)});
        },
        [&](int, const FaceEdge& here, FaceEdge& next) {
          const double dihedral = degrees(
              AngleBetween(faceNormal_[here.face], faceNormal_[next.face]));
          if (dihedral > minSharpAngle) {
            normals.push_back(vec3(0.0));
            meshIds.push_back(meshRelation_.triRef[next.face].meshID);
          }
          groups.push_back(normals.size() - 1);
          if (std::isfinite(next.normalizedEdge.x)) {
            vec3 dir = SafeNormalize(
                la::cross(next.normalizedEdge, here.normalizedEdge));
            normals.back() +=
                dir * AngleBetween(here.normalizedEdge, next.normalizedEdge);
          } else {
            next.normalizedEdge = here.normalizedEdge;
          }
        });

    for (int i = 0; i < normals.size(); ++i) {
      vec3 n = normals[i];
      // Same frame-storage rule as the single-normal path above.
      if (normalIdx != 0) n = getTransform(meshIds[i]) * n;
      normals[i] = SafeNormalize(n);
    }

    int lastGroup = 0;
    int lastProp = -1;
    int newProp = -1;
    int idx = 0;
    ForVert(endEdge, [&](int current1) {
      const int prop = oldHalfedgeProp[current1];
      auto start = oldProperties.begin() + prop * oldNumProp;

      if (groups[idx] != lastGroup && groups[idx] != 0 && prop == lastProp) {
        // split property vertex, duplicating but with an updated normal
        lastGroup = groups[idx];
        newProp = NumPropVert();
        properties_.resize(properties_.size() + numProp);
        std::copy(start, start + oldNumProp,
                  properties_.begin() + newProp * numProp);
        for (const int i : {0, 1, 2}) {
          properties_[newProp * numProp + normalIdx + i] =
              normals[groups[idx]][i];
        }
      } else if (prop != lastProp) {
        // update property vertex
        lastProp = prop;
        newProp = prop;
        std::copy(start, start + oldNumProp,
                  properties_.begin() + prop * numProp);
        for (const int i : {0, 1, 2})
          properties_[prop * numProp + normalIdx + i] = normals[groups[idx]][i];
      }

      // point to updated property vertex
      halfedge_.SetProp(current1, newProp);
      ++idx;
    });
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
  for_each_n(autoPolicy(n, 1e4), countAt(0), n, [this](const int halfedge) {
    vec4& tangent = halfedgeTangent_[halfedge];
    const int pair = halfedge_.Pair(halfedge);
    vec4& otherTangent = halfedgeTangent_[pair];

    const bool flat[2] = {tangent.w == 0, otherTangent.w == 0};
    if (!halfedge_.IsForward(halfedge) || (!flat[0] && !flat[1])) {
      return;
    }

    const vec3 edgeVec =
        vertPos_[halfedge_.End(halfedge)] - vertPos_[halfedge_.Start(halfedge)];

    if (flat[0] && flat[1]) {
      tangent = vec4(edgeVec / 3.0, 1);
      otherTangent = vec4(-edgeVec / 3.0, 1);
    } else if (flat[0]) {
      tangent = vec4((edgeVec + vec3(otherTangent)) / 2.0, 1);
    } else {
      otherTangent = vec4((-edgeVec + vec3(tangent)) / 2.0, 1);
    }
  });
}

/**
 * Redistribute the tangents around each vertex so that the angles between them
 * have the same ratios as the angles of the triangles between the corresponding
 * edges. This avoids folding the output shape and gives smoother results. There
 * must be at least one fixed halfedge on a vertex for that vertex to be
 * operated on. If there is only one, then that halfedge is not treated as
 * fixed, but the whole circle is turned to an average orientation.
 */
void Manifold::Impl::DistributeTangents(const Vec<bool>& fixedHalfedges) {
  const int numHalfedge = fixedHalfedges.size();
  for_each_n(
      autoPolicy(numHalfedge, 1e4), countAt(0), numHalfedge,
      [this, &fixedHalfedges](int halfedge) {
        if (!fixedHalfedges[halfedge] || IsMarkedInsideQuad(halfedge)) return;

        vec3 normal(0.0);
        Vec<double> currentAngle;
        Vec<double> desiredAngle;

        const vec3 approxNormal = vertNormal_[halfedge_.Start(halfedge)];
        const vec3 center = vertPos_[halfedge_.Start(halfedge)];
        vec3 lastEdgeVec =
            SafeNormalize(vertPos_[halfedge_.End(halfedge)] - center);
        const vec3 firstTangent =
            SafeNormalize(vec3(halfedgeTangent_[halfedge]));
        vec3 lastTangent = firstTangent;
        int current = halfedge;
        do {
          current = NextHalfedge(halfedge_.Pair(current));
          if (IsMarkedInsideQuad(current)) continue;
          const vec3 thisEdgeVec =
              SafeNormalize(vertPos_[halfedge_.End(current)] - center);
          const vec3 thisTangent =
              SafeNormalize(vec3(halfedgeTangent_[current]));
          normal += la::cross(thisTangent, lastTangent);
          // cumulative sum
          desiredAngle.push_back(
              AngleBetween(thisEdgeVec, lastEdgeVec) +
              (desiredAngle.size() > 0 ? desiredAngle.back() : 0));
          if (current == halfedge) {
            currentAngle.push_back(kTwoPi);
          } else {
            currentAngle.push_back(AngleBetween(thisTangent, firstTangent));
            if (la::dot(approxNormal, la::cross(thisTangent, firstTangent)) <
                0) {
              currentAngle.back() = kTwoPi - currentAngle.back();
            }
          }
          lastEdgeVec = thisEdgeVec;
          lastTangent = thisTangent;
        } while (!fixedHalfedges[current]);

        if (currentAngle.size() == 1 || la::dot(normal, normal) == 0) return;

        const double scale = currentAngle.back() / desiredAngle.back();
        double offset = 0;
        if (current == halfedge) {  // only one - find average offset
          for (size_t i = 0; i < currentAngle.size(); ++i) {
            offset += Wrap(currentAngle[i] - scale * desiredAngle[i]);
          }
          offset /= currentAngle.size();
        }

        current = halfedge;
        size_t i = 0;
        do {
          current = NextHalfedge(halfedge_.Pair(current));
          if (current != halfedge && fixedHalfedges[current]) break;
          if (IsMarkedInsideQuad(current)) continue;
          desiredAngle[i] *= scale;
          const double lastAngle = i > 0 ? desiredAngle[i - 1] : 0;
          // shrink obtuse angles
          if (desiredAngle[i] - lastAngle > kPi) {
            desiredAngle[i] = lastAngle + kPi;
          } else if (i + 1 < desiredAngle.size() &&
                     scale * desiredAngle[i + 1] - desiredAngle[i] > kPi) {
            desiredAngle[i] = scale * desiredAngle[i + 1] - kPi;
          }
          const double angle = currentAngle[i] - desiredAngle[i] - offset;
          vec3 tangent(halfedgeTangent_[current]);
          const quat q = la::rotation_quat(la::normalize(normal), angle);
          const vec3 newTangent = la::qrot(q, tangent);
          for (const int j : {0, 1, 2}) {
            halfedgeTangent_[current][j] = newTangent[j];
          }
          ++i;
        } while (!fixedHalfedges[current]);
      });
}

/**
 * Calculates halfedgeTangent_, allowing the manifold to be refined and
 * smoothed. The tangents form weighted cubic Beziers along each edge. This
 * function creates circular arcs where possible (minimizing maximum curvature),
 * constrained to the indicated property normals. Where a vert has multiple
 * normals, the tangents will form a crease. Where the only normal is that of a
 * flat face, the tangents at its ends will be made colinear. Zero-length
 * normals are considered missing and will defer to their neighboring normals
 * instead. If all normals are missing, the vertex pseudonormal will be used.
 */
void Manifold::Impl::CreateTangents(int normalIdx) {
  ZoneScoped;
  const int numVert = NumVert();
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.clear();
  Vec<vec4> tangent(numHalfedge);
  Vec<bool> fixedHalfedge(numHalfedge, false);

  // special flags for tangent.w
  constexpr double kInsideQuad = -1;
  constexpr double kMissingNormal = -3;

  Vec<int> vertHalfedge = VertHalfedge();
  for_each_n(
      autoPolicy(numVert, 1e4), vertHalfedge.begin(), numVert, [&](int e) {
        struct FlatNormal {
          bool isFlatFace;
          vec3 normal;
        };

        ivec2 faceEdges(-1, -1);
        int startHalfedge = -1;
        vec3 lastNormal(0.0);

        ForVert<FlatNormal>(
            e,
            [normalIdx, this](int halfedge) {
              const vec3 normal = GetNormal(halfedge, normalIdx);
              return FlatNormal(
                  {EqualNormals(normal, faceNormal_[halfedge / 3]), normal});
            },
            [&](int halfedge, const FlatNormal& here, const FlatNormal& next) {
              // Tangents not known at first are used as temporary storage for
              // normals and w is set to a negative flag value. This starts with
              // the flag clear.
              tangent[halfedge].w = 1;

              if (here.isFlatFace != next.isFlatFace) {
                // Record the two halfedges that border a single flat face.
                if (faceEdges[0] == -1) {
                  faceEdges[0] = halfedge;
                } else if (faceEdges[1] == -1) {
                  faceEdges[1] = halfedge;
                } else {
                  faceEdges[0] = -2;
                }
              }

              if (next.normal == vec3(0.) || here.normal == vec3(0.)) {
                if (here.normal != vec3(0.)) {  // next missing
                  lastNormal = here.normal;
                } else if (next.normal != vec3(0.)) {  // here missing
                  if (startHalfedge < 0) startHalfedge = halfedge;
                } else {  // both missing
                  if (startHalfedge < 0) startHalfedge = -2;
                }
                tangent[halfedge] = {lastNormal, kMissingNormal};
              }

              if (IsInsideQuad(halfedge))
                tangent[halfedge] = {lastNormal, kInsideQuad};

              if (tangent[halfedge].w < 0) return;

              // calculate tangents
              if (EqualNormals(next.normal, here.normal)) {
                tangent[halfedge] = TangentFromNormal(here.normal, halfedge);
              } else {
                // tangents at the intersection of two normals are fixed.
                fixedHalfedge[halfedge] = true;
                // Override the flat face logic if more than one normal.
                faceEdges[0] = -2;

                const vec3 edgeVec = vertPos_[halfedge_.End(halfedge)] -
                                     vertPos_[halfedge_.Start(halfedge)];
                const vec3 dir = la::cross(here.normal, next.normal);
                tangent[halfedge] = CircularTangent(
                    (la::dot(dir, edgeVec) < 0 ? -1.0 : 1.0) * dir, edgeVec);
              }
            });

        if (startHalfedge != -1 && lastNormal == vec3(0.)) {
          // Use vert pseudo normal if no normals are present at all.
          const vec3 normal = vertNormal_[halfedge_.Start(e)];
          ForVert(e, [&](int halfedge) {
            if (tangent[halfedge].w != kInsideQuad)
              tangent[halfedge] = TangentFromNormal(normal, halfedge);
          });
          return;
        }

        if (startHalfedge >= 0) {
          // Orbit the vertex backwards, pulling the next normal from the
          // tangent where it is stored temporarily.
          int current = startHalfedge;
          vec3 prevNormal =
              GetNormal(NextHalfedge(halfedge_.Pair(current)), normalIdx);
          do {
            DEBUG_ASSERT(prevNormal != vec3(0.), logicErr,
                         "missing prevNormal");
            if (tangent[current].w == kMissingNormal) {
              vec3 nextNormal = tangent[current].xyz();
              if (nextNormal == vec3(0.)) {
                nextNormal = lastNormal;
              }

              if (EqualNormals(prevNormal, nextNormal)) {
                tangent[current] = TangentFromNormal(prevNormal, current);
              } else {
                const vec3 dir = la::cross(prevNormal, nextNormal);
                const vec3 edgeVec = vertPos_[halfedge_.End(current)] -
                                     vertPos_[halfedge_.Start(current)];
                tangent[current] = CircularTangent(
                    (la::dot(dir, edgeVec) < 0 ? -1.0 : 1.0) * dir, edgeVec);
              }
            }
            vec3 currentNormal = GetNormal(current, normalIdx);
            if (currentNormal != vec3(0.)) {
              prevNormal = currentNormal;
            }
            current = halfedge_.Pair(PrevHalfedge(current));

          } while (current != startHalfedge);
        }

        if (faceEdges[0] >= 0 && faceEdges[1] >= 0) {
          // When only a single flat face is present with a single shared normal
          // for the entire vert, the tangents on either side of it should be
          // aligned to give a continuous curve.
          const vec3 edge0 = vertPos_[halfedge_.End(faceEdges[0])] -
                             vertPos_[halfedge_.Start(faceEdges[0])];
          const vec3 edge1 = vertPos_[halfedge_.End(faceEdges[1])] -
                             vertPos_[halfedge_.Start(faceEdges[1])];
          const vec3 newTangent = la::normalize(edge0) - la::normalize(edge1);
          tangent[faceEdges[0]] = CircularTangent(newTangent, edge0);
          tangent[faceEdges[1]] = CircularTangent(-newTangent, edge1);
          // Fix these tangents to keep them even to the edges.
          fixedHalfedge[faceEdges[0]] = true;
          fixedHalfedge[faceEdges[1]] = true;
        }
      });

  halfedgeTangent_ = std::move(tangent);
  DistributeTangents(fixedHalfedge);
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
void Manifold::Impl::CreateTangents(std::vector<Smoothness> sharpenedEdges,
                                    ExecutionContext::Impl* ctx) {
  ZoneScoped;
  const int numHalfedge = halfedge_.size();
  halfedgeTangent_.clear();
  Vec<vec4> tangent(numHalfedge);
  Vec<bool> fixedHalfedge(numHalfedge, false);

  Vec<int> vertHalfedge = VertHalfedge();
  Vec<bool> triIsFlatFace = FlatFaces();
  Vec<int> vertFlatFace = VertFlatFace(triIsFlatFace);
  Vec<vec3> vertNormal = vertNormal_;
  for (size_t v = 0; v < NumVert(); ++v) {
    if (vertFlatFace[v] >= 0) {
      vertNormal[v] = faceNormal_[vertFlatFace[v]];
    }
  }
  ADVANCE_PHASE_OR_RETURN(ctx);

  for_each_n(autoPolicy(numHalfedge, 1e4), countAt(0), numHalfedge, ctx,
             [&tangent, &vertNormal, this](const int edgeIdx) {
               tangent[edgeIdx] =
                   IsInsideQuad(edgeIdx)
                       ? vec4(0, 0, 0, -1)
                       : TangentFromNormal(vertNormal[halfedge_.Start(edgeIdx)],
                                           edgeIdx);
             });

  halfedgeTangent_ = std::move(tangent);
  ADVANCE_PHASE_OR_RETURN(ctx);

  // Add sharpened edges around faces, just on the face side.
  for (size_t tri = 0; tri < NumTri(); ++tri) {
    if (!triIsFlatFace[tri]) continue;
    for (const int j : {0, 1, 2}) {
      const int tri2 = halfedge_.Pair(3 * tri + j) / 3;
      if (!triIsFlatFace[tri2] ||
          !meshRelation_.triRef[tri].SameFace(meshRelation_.triRef[tri2])) {
        sharpenedEdges.push_back({3 * tri + j, 0});
      }
    }
  }
  ADVANCE_PHASE_OR_RETURN(ctx);

  using Pair = std::pair<Smoothness, Smoothness>;
  // Fill in missing pairs with default smoothness = 1.
  std::map<int, Pair> edges;
  for (Smoothness edge : sharpenedEdges) {
    if (edge.smoothness >= 1) continue;
    const bool forward = halfedge_.IsForward(edge.halfedge);
    const int pair = halfedge_.Pair(edge.halfedge);
    const int idx = forward ? edge.halfedge : pair;
    if (edges.find(idx) == edges.end()) {
      edges[idx] = {edge, {static_cast<size_t>(pair), 1}};
      if (!forward) std::swap(edges[idx].first, edges[idx].second);
    } else {
      Smoothness& e = forward ? edges[idx].first : edges[idx].second;
      e.smoothness = std::min(edge.smoothness, e.smoothness);
    }
  }

  std::map<int, std::vector<Pair>> vertTangents;
  for (const auto& value : edges) {
    const Pair edge = value.second;
    vertTangents[halfedge_.Start(edge.first.halfedge)].push_back(edge);
    vertTangents[halfedge_.Start(edge.second.halfedge)].push_back(
        {edge.second, edge.first});
  }
  ADVANCE_PHASE_OR_RETURN(ctx);

  const int numVert = NumVert();
  for_each_n(
      autoPolicy(numVert, 1e4), countAt(0), numVert, ctx,
      [this, &vertTangents, &fixedHalfedge, &vertHalfedge,
       &triIsFlatFace](int v) {
        auto it = vertTangents.find(v);
        if (it == vertTangents.end()) {
          fixedHalfedge[vertHalfedge[v]] = true;
          return;
        }
        const std::vector<Pair>& vert = it->second;
        // Sharp edges that end are smooth at their terminal vert.
        if (vert.size() == 1) return;
        if (vert.size() == 2) {  // Make continuous edge
          const int first = vert[0].first.halfedge;
          const int second = vert[1].first.halfedge;
          fixedHalfedge[first] = true;
          fixedHalfedge[second] = true;
          const vec3 newTangent = la::normalize(vec3(halfedgeTangent_[first]) -
                                                vec3(halfedgeTangent_[second]));

          const vec3 pos = vertPos_[halfedge_.Start(first)];
          halfedgeTangent_[first] =
              CircularTangent(newTangent, vertPos_[halfedge_.End(first)] - pos);
          halfedgeTangent_[second] = CircularTangent(
              -newTangent, vertPos_[halfedge_.End(second)] - pos);

          double smoothness =
              (vert[0].second.smoothness + vert[1].first.smoothness) / 2;
          ForVert(first, [this, &smoothness, &vert, first,
                          second](int current) {
            if (current == second) {
              smoothness =
                  (vert[1].second.smoothness + vert[0].first.smoothness) / 2;
            } else if (current != first && !IsMarkedInsideQuad(current)) {
              SharpenTangent(current, smoothness);
            }
          });
        } else {  // Sharpen vertex uniformly
          double smoothness = 0;
          double denom = 0;
          for (const Pair& pair : vert) {
            smoothness += pair.first.smoothness;
            smoothness += pair.second.smoothness;
            denom += pair.first.smoothness == 0 ? 0 : 1;
            denom += pair.second.smoothness == 0 ? 0 : 1;
          }
          smoothness /= denom;

          ForVert(vert[0].first.halfedge,
                  [this, &triIsFlatFace, smoothness](int current) {
                    if (!IsMarkedInsideQuad(current)) {
                      const int pair = halfedge_.Pair(current);
                      SharpenTangent(current, triIsFlatFace[current / 3] ||
                                                      triIsFlatFace[pair / 3]
                                                  ? 0
                                                  : smoothness);
                    }
                  });
        }
      });
  ADVANCE_PHASE_OR_RETURN(ctx);

  LinearizeFlatTangents();
  ADVANCE_PHASE_OR_RETURN(ctx);

  DistributeTangents(fixedHalfedge);
  ADVANCE_PHASE_OR_RETURN(ctx);
}

bool Manifold::Impl::ValidTangents() const {
  const int numHalfedge = halfedge_.size();
  return all_of(autoPolicy(numHalfedge, 1e4), countAt(0), countAt(numHalfedge),
                [&](const size_t edgeIdx) {
                  const bool inQuad = IsMarkedInsideQuad(edgeIdx);
                  const size_t pair = halfedge_.Pair(edgeIdx);
                  if (inQuad != IsMarkedInsideQuad(pair)) return false;
                  if (!inQuad) return true;
                  // missing tangents cannot be adjacent
                  if (IsMarkedInsideQuad(NextHalfedge(edgeIdx)) ||
                      IsMarkedInsideQuad(PrevHalfedge(edgeIdx)) ||
                      IsMarkedInsideQuad(NextHalfedge(pair)) ||
                      IsMarkedInsideQuad(PrevHalfedge(pair)))
                    return false;
                  return true;
                });
}

void Manifold::Impl::Refine(std::function<int(vec3, vec4, vec4)> edgeDivisions,
                            bool keepInterior, ExecutionContext::Impl* ctx) {
  if (IsEmpty()) return;

  if (!ValidTangents()) {
    MakeEmpty(Error::InvalidTangents);
    return;
  }

  Manifold::Impl old = *this;
  halfedge_.MakeUnique();
  Vec<Barycentric> vertBary = Subdivide(edgeDivisions, keepInterior);
  // Cancel observed AFTER Subdivide (which is currently cancel-blind) but
  // BEFORE the no-op early-return: the user requested cancel, so honour it
  // even when Subdivide produced nothing to interpolate.
  if (IsCancelled(ctx)) {
    MakeEmpty(Error::Cancelled);
    return;
  }
  if (vertBary.size() == 0) return;

  if (old.halfedgeTangent_.size() == old.halfedge_.size()) {
    for_each_n(autoPolicy(NumTri(), 1e4), countAt(0), NumVert(),
               InterpTri({vertPos_, vertBary, &old}));
  }

  halfedgeTangent_.clear();
  if (old.halfedgeTangent_.size() == old.halfedge_.size()) {
    SetNormalsAndCoplanar();
    CalculateBBox();
  } else {
    CalculateVertNormals();
  }
  SortGeometry(ctx);
  meshRelation_.originalID = -1;
}

}  // namespace manifold
