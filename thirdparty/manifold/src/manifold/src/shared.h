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

#include "par.h"
#include "utils.h"
#include "vec_dh.h"

namespace manifold {

/** @addtogroup Private
 *  @{
 */
__host__ __device__ inline glm::vec3 SafeNormalize(glm::vec3 v) {
  v = glm::normalize(v);
  return glm::isfinite(v.x) ? v : glm::vec3(0);
}

__host__ __device__ inline int NextHalfedge(int current) {
  ++current;
  if (current % 3 == 0) current -= 3;
  return current;
}

__host__ __device__ inline glm::vec3 UVW(int vert,
                                         const glm::vec3* barycentric) {
  glm::vec3 uvw(0.0f);
  if (vert < 0) {
    uvw[vert + 3] = 1;
  } else {
    uvw = barycentric[vert];
  }
  return uvw;
}

/**
 * By using the closest axis-aligned projection to the normal instead of a
 * projection along the normal, we avoid introducing any rounding error.
 */
__host__ __device__ inline glm::mat3x2 GetAxisAlignedProjection(
    glm::vec3 normal) {
  glm::vec3 absNormal = glm::abs(normal);
  float xyzMax;
  glm::mat2x3 projection;
  if (absNormal.z > absNormal.x && absNormal.z > absNormal.y) {
    projection = glm::mat2x3(1.0f, 0.0f, 0.0f,  //
                             0.0f, 1.0f, 0.0f);
    xyzMax = normal.z;
  } else if (absNormal.y > absNormal.x) {
    projection = glm::mat2x3(0.0f, 0.0f, 1.0f,  //
                             1.0f, 0.0f, 0.0f);
    xyzMax = normal.y;
  } else {
    projection = glm::mat2x3(0.0f, 1.0f, 0.0f,  //
                             0.0f, 0.0f, 1.0f);
    xyzMax = normal.x;
  }
  if (xyzMax < 0) projection[0] *= -1.0f;
  return glm::transpose(projection);
}

__host__ __device__ inline glm::vec3 GetBarycentric(const glm::vec3& v,
                                                    const glm::mat3& triPos,
                                                    float precision) {
  const glm::mat3 edges(triPos[2] - triPos[1], triPos[0] - triPos[2],
                        triPos[1] - triPos[0]);
  const glm::vec3 d2(glm::dot(edges[0], edges[0]), glm::dot(edges[1], edges[1]),
                     glm::dot(edges[2], edges[2]));
  int longside = d2[0] > d2[1] && d2[0] > d2[2] ? 0 : d2[1] > d2[2] ? 1 : 2;
  const glm::vec3 crossP = glm::cross(edges[0], edges[1]);
  const float area2 = glm::dot(crossP, crossP);
  const float tol2 = precision * precision;
  const float vol = glm::dot(crossP, v - triPos[2]);
  if (vol * vol > area2 * tol2) return glm::vec3(NAN);

  if (d2[longside] < tol2) {  // point
    return glm::vec3(1, 0, 0);
  } else if (area2 > d2[longside] * tol2) {  // triangle
    glm::vec3 uvw(0);
    for (int i : {0, 1, 2}) {
      int j = i + 1;
      if (j > 2) j -= 3;
      uvw[i] = glm::dot(glm::cross(edges[i], v - triPos[j]), crossP);
    }
    uvw /= (uvw[0] + uvw[1] + uvw[2]);
    return uvw;
  } else {  // line
    int nextside = longside + 1;
    if (nextside > 2) nextside -= 3;
    const float alpha =
        glm::dot(v - triPos[nextside], edges[longside]) / d2[longside];
    glm::vec3 uvw(0);
    uvw[longside] = 0;
    uvw[nextside++] = 1 - alpha;
    if (nextside > 2) nextside -= 3;
    uvw[nextside] = alpha;
    return uvw;
  }
}

/**
 * This is a temporary edge strcture which only stores edges forward and
 * references the halfedge it was created from.
 */
struct TmpEdge {
  int first, second, halfedgeIdx;

  __host__ __device__ TmpEdge() {}
  __host__ __device__ TmpEdge(int start, int end, int idx) {
    first = glm::min(start, end);
    second = glm::max(start, end);
    halfedgeIdx = idx;
  }

  __host__ __device__ bool operator<(const TmpEdge& other) const {
    return first == other.first ? second < other.second : first < other.first;
  }
};

struct Halfedge2Tmp {
  __host__ __device__ void operator()(
      thrust::tuple<TmpEdge&, const Halfedge&, int> inout) {
    const Halfedge& halfedge = thrust::get<1>(inout);
    int idx = thrust::get<2>(inout);
    if (!halfedge.IsForward()) idx = -1;

    thrust::get<0>(inout) = TmpEdge(halfedge.startVert, halfedge.endVert, idx);
  }
};

struct TmpInvalid {
  __host__ __device__ bool operator()(const TmpEdge& edge) {
    return edge.halfedgeIdx < 0;
  }
};

VecDH<TmpEdge> inline CreateTmpEdges(const VecDH<Halfedge>& halfedge) {
  VecDH<TmpEdge> edges(halfedge.size());
  for_each_n(autoPolicy(edges.size()),
             zip(edges.begin(), halfedge.begin(), countAt(0)), edges.size(),
             Halfedge2Tmp());
  int numEdge =
      remove_if<decltype(edges.begin())>(
          autoPolicy(edges.size()), edges.begin(), edges.end(), TmpInvalid()) -
      edges.begin();
  ASSERT(numEdge == halfedge.size() / 2, topologyErr, "Not oriented!");
  edges.resize(numEdge);
  return edges;
}

struct ReindexEdge {
  const TmpEdge* edges;

  __host__ __device__ void operator()(int& edge) {
    edge = edges[edge].halfedgeIdx;
  }
};
/** @} */
}  // namespace manifold
