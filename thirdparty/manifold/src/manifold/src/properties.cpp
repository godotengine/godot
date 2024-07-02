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

#include <limits>

#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;

struct FaceAreaVolume {
  VecView<const Halfedge> halfedges;
  VecView<const glm::vec3> vertPos;
  const float precision;

  thrust::pair<float, float> operator()(int face) {
    float perimeter = 0;
    glm::vec3 edge[3];
    for (int i : {0, 1, 2}) {
      const int j = (i + 1) % 3;
      edge[i] = vertPos[halfedges[3 * face + j].startVert] -
                vertPos[halfedges[3 * face + i].startVert];
      perimeter += glm::length(edge[i]);
    }
    glm::vec3 crossP = glm::cross(edge[0], edge[1]);

    float area = glm::length(crossP);
    float volume = glm::dot(crossP, vertPos[halfedges[3 * face].startVert]);

    return thrust::make_pair(area / 2.0f, volume / 6.0f);
  }
};

struct PosMin
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    if (isnan(a.x)) return b;
    if (isnan(b.x)) return a;
    return glm::min(a, b);
  }
};

struct PosMax
    : public thrust::binary_function<glm::vec3, glm::vec3, glm::vec3> {
  glm::vec3 operator()(glm::vec3 a, glm::vec3 b) {
    if (isnan(a.x)) return b;
    if (isnan(b.x)) return a;
    return glm::max(a, b);
  }
};

struct FiniteVert {
  bool operator()(glm::vec3 v) { return glm::all(glm::isfinite(v)); }
};

struct MakeMinMax {
  glm::ivec2 operator()(glm::ivec3 tri) {
    return glm::ivec2(glm::min(tri[0], glm::min(tri[1], tri[2])),
                      glm::max(tri[0], glm::max(tri[1], tri[2])));
  }
};

struct MinMax
    : public thrust::binary_function<glm::ivec2, glm::ivec2, glm::ivec2> {
  glm::ivec2 operator()(glm::ivec2 a, glm::ivec2 b) {
    a[0] = glm::min(a[0], b[0]);
    a[1] = glm::max(a[1], b[1]);
    return a;
  }
};

struct SumPair : public thrust::binary_function<thrust::pair<float, float>,
                                                thrust::pair<float, float>,
                                                thrust::pair<float, float>> {
  thrust::pair<float, float> operator()(thrust::pair<float, float> a,
                                        thrust::pair<float, float> b) {
    a.first += b.first;
    a.second += b.second;
    return a;
  }
};

struct CurvatureAngles {
  VecView<float> meanCurvature;
  VecView<float> gaussianCurvature;
  VecView<float> area;
  VecView<float> degree;
  VecView<const Halfedge> halfedge;
  VecView<const glm::vec3> vertPos;
  VecView<const glm::vec3> triNormal;

  void operator()(int tri) {
    glm::vec3 edge[3];
    glm::vec3 edgeLength(0.0);
    for (int i : {0, 1, 2}) {
      const int startVert = halfedge[3 * tri + i].startVert;
      const int endVert = halfedge[3 * tri + i].endVert;
      edge[i] = vertPos[endVert] - vertPos[startVert];
      edgeLength[i] = glm::length(edge[i]);
      edge[i] /= edgeLength[i];
      const int neighborTri = halfedge[3 * tri + i].pairedHalfedge / 3;
      const float dihedral =
          0.25 * edgeLength[i] *
          glm::asin(glm::dot(glm::cross(triNormal[tri], triNormal[neighborTri]),
                             edge[i]));
      AtomicAdd(meanCurvature[startVert], dihedral);
      AtomicAdd(meanCurvature[endVert], dihedral);
      AtomicAdd(degree[startVert], 1.0f);
    }

    glm::vec3 phi;
    phi[0] = glm::acos(-glm::dot(edge[2], edge[0]));
    phi[1] = glm::acos(-glm::dot(edge[0], edge[1]));
    phi[2] = glm::pi<float>() - phi[0] - phi[1];
    const float area3 = edgeLength[0] * edgeLength[1] *
                        glm::length(glm::cross(edge[0], edge[1])) / 6;

    for (int i : {0, 1, 2}) {
      const int vert = halfedge[3 * tri + i].startVert;
      AtomicAdd(gaussianCurvature[vert], -phi[i]);
      AtomicAdd(area[vert], area3);
    }
  }
};

struct NormalizeCurvature {
  void operator()(thrust::tuple<float&, float&, float, float> inOut) {
    float& meanCurvature = thrust::get<0>(inOut);
    float& gaussianCurvature = thrust::get<1>(inOut);
    float area = thrust::get<2>(inOut);
    float degree = thrust::get<3>(inOut);
    float factor = degree / (6 * area);
    meanCurvature *= factor;
    gaussianCurvature *= factor;
  }
};

struct UpdateProperties {
  VecView<float> properties;

  VecView<const float> oldProperties;
  VecView<const Halfedge> halfedge;
  VecView<const float> meanCurvature;
  VecView<const float> gaussianCurvature;
  const int oldNumProp;
  const int numProp;
  const int gaussianIdx;
  const int meanIdx;

  // FIXME: race condition
  void operator()(thrust::tuple<glm::ivec3&, int> inOut) {
    glm::ivec3& triProp = thrust::get<0>(inOut);
    const int tri = thrust::get<1>(inOut);

    for (const int i : {0, 1, 2}) {
      const int vert = halfedge[3 * tri + i].startVert;
      if (oldNumProp == 0) {
        triProp[i] = vert;
      }
      const int propVert = triProp[i];

      for (int p = 0; p < oldNumProp; ++p) {
        properties[numProp * propVert + p] =
            oldProperties[oldNumProp * propVert + p];
      }

      if (gaussianIdx >= 0) {
        properties[numProp * propVert + gaussianIdx] = gaussianCurvature[vert];
      }
      if (meanIdx >= 0) {
        properties[numProp * propVert + meanIdx] = meanCurvature[vert];
      }
    }
  }
};

struct CheckHalfedges {
  VecView<const Halfedge> halfedges;
  VecView<const glm::vec3> vertPos;

  bool operator()(size_t edge) {
    const Halfedge halfedge = halfedges[edge];
    if (halfedge.startVert == -1 || halfedge.endVert == -1) return true;
    if (halfedge.pairedHalfedge == -1) return false;

    if (!isfinite(vertPos[halfedge.startVert][0])) return false;
    if (!isfinite(vertPos[halfedge.endVert][0])) return false;

    const Halfedge paired = halfedges[halfedge.pairedHalfedge];
    bool good = true;
    good &= paired.pairedHalfedge == edge;
    good &= halfedge.startVert != halfedge.endVert;
    good &= halfedge.startVert == paired.endVert;
    good &= halfedge.endVert == paired.startVert;
    return good;
  }
};

struct NoDuplicates {
  VecView<const Halfedge> halfedges;

  bool operator()(int edge) {
    const Halfedge halfedge = halfedges[edge];
    if (halfedge.startVert == -1 && halfedge.endVert == -1 &&
        halfedge.pairedHalfedge == -1)
      return true;
    return halfedge.startVert != halfedges[edge + 1].startVert ||
           halfedge.endVert != halfedges[edge + 1].endVert;
  }
};

struct CheckCCW {
  VecView<const Halfedge> halfedges;
  VecView<const glm::vec3> vertPos;
  VecView<const glm::vec3> triNormal;
  const float tol;

  bool operator()(int face) {
    if (halfedges[3 * face].pairedHalfedge < 0) return true;

    const glm::mat3x2 projection = GetAxisAlignedProjection(triNormal[face]);
    glm::vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos[halfedges[3 * face + i].startVert];

    int ccw = CCW(v[0], v[1], v[2], glm::abs(tol));
    bool check = tol > 0 ? ccw >= 0 : ccw == 0;

#ifdef MANIFOLD_DEBUG
    if (tol > 0 && !check) {
      glm::vec2 v1 = v[1] - v[0];
      glm::vec2 v2 = v[2] - v[0];
      float area = v1.x * v2.y - v1.y * v2.x;
      float base2 = glm::max(glm::dot(v1, v1), glm::dot(v2, v2));
      float base = glm::sqrt(base2);
      glm::vec3 V0 = vertPos[halfedges[3 * face].startVert];
      glm::vec3 V1 = vertPos[halfedges[3 * face + 1].startVert];
      glm::vec3 V2 = vertPos[halfedges[3 * face + 2].startVert];
      glm::vec3 norm = glm::cross(V1 - V0, V2 - V0);
      printf(
          "Tri %d does not match normal, approx height = %g, base = %g\n"
          "tol = %g, area2 = %g, base2*tol2 = %g\n"
          "normal = %g, %g, %g\n"
          "norm = %g, %g, %g\nverts: %d, %d, %d\n",
          face, area / base, base, tol, area * area, base2 * tol * tol,
          triNormal[face].x, triNormal[face].y, triNormal[face].z, norm.x,
          norm.y, norm.z, halfedges[3 * face].startVert,
          halfedges[3 * face + 1].startVert, halfedges[3 * face + 2].startVert);
    }
#endif
    return check;
  }
};
}  // namespace

namespace manifold {

/**
 * Returns true if this manifold is in fact an oriented even manifold and all of
 * the data structures are consistent.
 */
bool Manifold::Impl::IsManifold() const {
  if (halfedge_.size() == 0) return true;
  auto policy = autoPolicy(halfedge_.size());

  return all_of(policy, countAt(0_z), countAt(halfedge_.size()),
                CheckHalfedges({halfedge_, vertPos_}));
}

/**
 * Returns true if this manifold is in fact an oriented 2-manifold and all of
 * the data structures are consistent.
 */
bool Manifold::Impl::Is2Manifold() const {
  if (halfedge_.size() == 0) return true;
  auto policy = autoPolicy(halfedge_.size());

  if (!IsManifold()) return false;

  Vec<Halfedge> halfedge(halfedge_);
  stable_sort(policy, halfedge.begin(), halfedge.end());

  return all_of(policy, countAt(0), countAt(2 * NumEdge() - 1),
                NoDuplicates({halfedge}));
}

/**
 * Returns true if all triangles are CCW relative to their triNormals_.
 */
bool Manifold::Impl::MatchesTriNormals() const {
  if (halfedge_.size() == 0 || faceNormal_.size() != NumTri()) return true;
  return all_of(autoPolicy(NumTri()), countAt(0), countAt(NumTri()),
                CheckCCW({halfedge_, vertPos_, faceNormal_, 2 * precision_}));
}

/**
 * Returns the number of triangles that are colinear within precision_.
 */
int Manifold::Impl::NumDegenerateTris() const {
  if (halfedge_.size() == 0 || faceNormal_.size() != NumTri()) return true;
  return count_if(
      autoPolicy(NumTri()), countAt(0), countAt(NumTri()),
      CheckCCW({halfedge_, vertPos_, faceNormal_, -1 * precision_ / 2}));
}

Properties Manifold::Impl::GetProperties() const {
  ZoneScoped;
  if (IsEmpty()) return {0, 0};
  // Kahan summation
  float area = 0;
  float volume = 0;
  float areaCompensation = 0;
  float volumeCompensation = 0;
  for (int i = 0; i < NumTri(); ++i) {
    auto [area1, volume1] =
        FaceAreaVolume({halfedge_, vertPos_, precision_})(i);
    const float t1 = area + area1;
    const float t2 = volume + volume1;
    areaCompensation += (area - t1) + area1;
    volumeCompensation += (volume - t2) + volume1;
    area = t1;
    volume = t2;
  }
  area += areaCompensation;
  volume += volumeCompensation;

  return {area, volume};
}

void Manifold::Impl::CalculateCurvature(int gaussianIdx, int meanIdx) {
  ZoneScoped;
  if (IsEmpty()) return;
  if (gaussianIdx < 0 && meanIdx < 0) return;
  Vec<float> vertMeanCurvature(NumVert(), 0);
  Vec<float> vertGaussianCurvature(NumVert(), glm::two_pi<float>());
  Vec<float> vertArea(NumVert(), 0);
  Vec<float> degree(NumVert(), 0);
  auto policy = autoPolicy(NumTri());
  for_each(policy, countAt(0), countAt(NumTri()),
           CurvatureAngles({vertMeanCurvature, vertGaussianCurvature, vertArea,
                            degree, halfedge_, vertPos_, faceNormal_}));
  for_each_n(policy,
             zip(vertMeanCurvature.begin(), vertGaussianCurvature.begin(),
                 vertArea.begin(), degree.begin()),
             NumVert(), NormalizeCurvature());

  const int oldNumProp = NumProp();
  const int numProp = glm::max(oldNumProp, glm::max(gaussianIdx, meanIdx) + 1);
  const Vec<float> oldProperties = meshRelation_.properties;
  meshRelation_.properties = Vec<float>(numProp * NumPropVert(), 0);
  meshRelation_.numProp = numProp;
  if (meshRelation_.triProperties.size() == 0) {
    meshRelation_.triProperties.resize(NumTri());
  }

  for_each_n(
      policy, zip(meshRelation_.triProperties.begin(), countAt(0)), NumTri(),
      UpdateProperties({meshRelation_.properties, oldProperties, halfedge_,
                        vertMeanCurvature, vertGaussianCurvature, oldNumProp,
                        numProp, gaussianIdx, meanIdx}));

  CreateFaces();
  Finish();
}

/**
 * Calculates the bounding box of the entire manifold, which is stored
 * internally to short-cut Boolean operations and to serve as the precision
 * range for Morton code calculation. Ignores NaNs.
 */
void Manifold::Impl::CalculateBBox() {
  auto policy = autoPolicy(NumVert());
  bBox_.min = reduce<glm::vec3>(
      policy, vertPos_.begin(), vertPos_.end(),
      glm::vec3(std::numeric_limits<float>::infinity()), PosMin());
  bBox_.max = reduce<glm::vec3>(
      policy, vertPos_.begin(), vertPos_.end(),
      glm::vec3(-std::numeric_limits<float>::infinity()), PosMax());
}

/**
 * Determines if all verts are finite. Checking just the bounding box dimensions
 * is insufficient as it ignores NaNs.
 */
bool Manifold::Impl::IsFinite() const {
  auto policy = autoPolicy(NumVert());
  return transform_reduce<bool>(policy, vertPos_.begin(), vertPos_.end(),
                                FiniteVert(), true,
                                thrust::logical_and<bool>());
}

/**
 * Checks that the input triVerts array has all indices inside bounds of the
 * vertPos_ array.
 */
bool Manifold::Impl::IsIndexInBounds(VecView<const glm::ivec3> triVerts) const {
  auto policy = autoPolicy(triVerts.size());
  glm::ivec2 minmax = transform_reduce<glm::ivec2>(
      policy, triVerts.begin(), triVerts.end(), MakeMinMax(),
      glm::ivec2(std::numeric_limits<int>::max(),
                 std::numeric_limits<int>::min()),
      MinMax());

  return minmax[0] >= 0 && minmax[1] < NumVert();
}
}  // namespace manifold
