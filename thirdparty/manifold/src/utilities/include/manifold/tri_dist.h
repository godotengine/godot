// Copyright 2024 The Manifold Authors.
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

#include <array>
#include <glm/glm.hpp>

#include "manifold/common.h"

namespace manifold {

// From NVIDIA-Omniverse PhysX - BSD 3-Clause "New" or "Revised" License
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/LICENSE.md
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/sweep/GuSweepCapsuleCapsule.cpp
// With minor modifications

/**
 * Returns the distance between two line segments.
 *
 * @param[out] x Closest point on line segment pa.
 * @param[out] y Closest point on line segment qb.
 * @param[in]  p  One endpoint of the first line segment.
 * @param[in]  a  Other endpoint of the first line segment.
 * @param[in]  p  One endpoint of the second line segment.
 * @param[in]  b  Other endpoint of the second line segment.
 */
inline void EdgeEdgeDist(vec3& x, vec3& y,  // closest points
                         const vec3& p,
                         const vec3& a,  // seg 1 origin, vector
                         const vec3& q,
                         const vec3& b)  // seg 2 origin, vector
{
  const vec3 T = q - p;
  const auto ADotA = glm::dot(a, a);
  const auto BDotB = glm::dot(b, b);
  const auto ADotB = glm::dot(a, b);
  const auto ADotT = glm::dot(a, T);
  const auto BDotT = glm::dot(b, T);

  // t parameterizes ray (p, a)
  // u parameterizes ray (q, b)

  // Compute t for the closest point on ray (p, a) to ray (q, b)
  const auto Denom = ADotA * BDotB - ADotB * ADotB;

  double t;  // We will clamp result so t is on the segment (p, a)
  t = Denom != 0.0
          ? glm::clamp((ADotT * BDotB - BDotT * ADotB) / Denom, 0.0, 1.0)
          : 0.0;

  // find u for point on ray (q, b) closest to point at t
  double u;
  if (BDotB != 0.0) {
    u = (t * ADotB - BDotT) / BDotB;

    // if u is on segment (q, b), t and u correspond to closest points,
    // otherwise, clamp u, recompute and clamp t
    if (u < 0.0) {
      u = 0.0;
      t = ADotA != 0.0 ? glm::clamp(ADotT / ADotA, 0.0, 1.0) : 0.0;
    } else if (u > 1.0) {
      u = 1.0;
      t = ADotA != 0.0 ? glm::clamp((ADotB + ADotT) / ADotA, 0.0, 1.0) : 0.0;
    }
  } else {
    u = 0.0;
    t = ADotA != 0.0 ? glm::clamp(ADotT / ADotA, 0.0, 1.0) : 0.0;
  }
  x = p + a * t;
  y = q + b * u;
}

// From NVIDIA-Omniverse PhysX - BSD 3-Clause "New" or "Revised" License
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/LICENSE.md
// https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/geomutils/src/distance/GuDistanceTriangleTriangle.cpp
// With minor modifications

/**
 * Returns the minimum squared distance between two triangles.
 *
 * @param  p  First  triangle.
 * @param  q  Second triangle.
 */
inline auto DistanceTriangleTriangleSquared(const std::array<vec3, 3>& p,
                                            const std::array<vec3, 3>& q) {
  std::array<vec3, 3> Sv;
  Sv[0] = p[1] - p[0];
  Sv[1] = p[2] - p[1];
  Sv[2] = p[0] - p[2];

  std::array<vec3, 3> Tv;
  Tv[0] = q[1] - q[0];
  Tv[1] = q[2] - q[1];
  Tv[2] = q[0] - q[2];

  bool shown_disjoint = false;

  auto mindd = std::numeric_limits<double>::max();

  for (uint32_t i = 0; i < 3; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      vec3 cp;
      vec3 cq;
      EdgeEdgeDist(cp, cq, p[i], Sv[i], q[j], Tv[j]);
      const vec3 V = cq - cp;
      const auto dd = glm::dot(V, V);

      if (dd <= mindd) {
        mindd = dd;

        uint32_t id = i + 2;
        if (id >= 3) id -= 3;
        vec3 Z = p[id] - cp;
        auto a = glm::dot(Z, V);
        id = j + 2;
        if (id >= 3) id -= 3;
        Z = q[id] - cq;
        auto b = glm::dot(Z, V);

        if ((a <= 0.0) && (b >= 0.0)) {
          return glm::dot(V, V);
        };

        if (a <= 0.0)
          a = 0.0;
        else if (b > 0.0)
          b = 0.0;

        if ((mindd - a + b) > 0.0) shown_disjoint = true;
      }
    }
  }

  vec3 Sn = glm::cross(Sv[0], Sv[1]);
  auto Snl = glm::dot(Sn, Sn);

  if (Snl > 1e-15) {
    const vec3 Tp(glm::dot(p[0] - q[0], Sn), glm::dot(p[0] - q[1], Sn),
                  glm::dot(p[0] - q[2], Sn));

    int index = -1;
    if ((Tp[0] > 0.0) && (Tp[1] > 0.0) && (Tp[2] > 0.0)) {
      index = Tp[0] < Tp[1] ? 0 : 1;
      if (Tp[2] < Tp[index]) index = 2;
    } else if ((Tp[0] < 0.0) && (Tp[1] < 0.0) && (Tp[2] < 0.0)) {
      index = Tp[0] > Tp[1] ? 0 : 1;
      if (Tp[2] > Tp[index]) index = 2;
    }

    if (index >= 0) {
      shown_disjoint = true;

      const vec3& qIndex = q[index];

      vec3 V = qIndex - p[0];
      vec3 Z = glm::cross(Sn, Sv[0]);
      if (glm::dot(V, Z) > 0.0) {
        V = qIndex - p[1];
        Z = glm::cross(Sn, Sv[1]);
        if (glm::dot(V, Z) > 0.0) {
          V = qIndex - p[2];
          Z = glm::cross(Sn, Sv[2]);
          if (glm::dot(V, Z) > 0.0) {
            vec3 cp = qIndex + Sn * Tp[index] / Snl;
            vec3 cq = qIndex;
            return glm::dot(cp - cq, cp - cq);
          }
        }
      }
    }
  }

  vec3 Tn = glm::cross(Tv[0], Tv[1]);
  auto Tnl = glm::dot(Tn, Tn);

  if (Tnl > 1e-15) {
    const vec3 Sp(glm::dot(q[0] - p[0], Tn), glm::dot(q[0] - p[1], Tn),
                  glm::dot(q[0] - p[2], Tn));

    int index = -1;
    if ((Sp[0] > 0.0) && (Sp[1] > 0.0) && (Sp[2] > 0.0)) {
      index = Sp[0] < Sp[1] ? 0 : 1;
      if (Sp[2] < Sp[index]) index = 2;
    } else if ((Sp[0] < 0.0) && (Sp[1] < 0.0) && (Sp[2] < 0.0)) {
      index = Sp[0] > Sp[1] ? 0 : 1;
      if (Sp[2] > Sp[index]) index = 2;
    }

    if (index >= 0) {
      shown_disjoint = true;

      const vec3& pIndex = p[index];

      vec3 V = pIndex - q[0];
      vec3 Z = glm::cross(Tn, Tv[0]);
      if (glm::dot(V, Z) > 0.0) {
        V = pIndex - q[1];
        Z = glm::cross(Tn, Tv[1]);
        if (glm::dot(V, Z) > 0.0) {
          V = pIndex - q[2];
          Z = glm::cross(Tn, Tv[2]);
          if (glm::dot(V, Z) > 0.0) {
            vec3 cp = pIndex;
            vec3 cq = pIndex + Tn * Sp[index] / Tnl;
            return glm::dot(cp - cq, cp - cq);
          }
        }
      }
    }
  }

  return shown_disjoint ? mindd : 0.0;
};
}  // namespace manifold
