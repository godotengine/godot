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
inline void EdgeEdgeDist(glm::vec3& x, glm::vec3& y,  // closest points
                         const glm::vec3& p,
                         const glm::vec3& a,  // seg 1 origin, vector
                         const glm::vec3& q,
                         const glm::vec3& b)  // seg 2 origin, vector
{
  const glm::vec3 T = q - p;
  const float ADotA = glm::dot(a, a);
  const float BDotB = glm::dot(b, b);
  const float ADotB = glm::dot(a, b);
  const float ADotT = glm::dot(a, T);
  const float BDotT = glm::dot(b, T);

  // t parameterizes ray (p, a)
  // u parameterizes ray (q, b)

  // Compute t for the closest point on ray (p, a) to ray (q, b)
  const float Denom = ADotA * BDotB - ADotB * ADotB;

  float t;  // We will clamp result so t is on the segment (p, a)
  t = Denom != 0.0f
          ? glm::clamp((ADotT * BDotB - BDotT * ADotB) / Denom, 0.0f, 1.0f)
          : 0.0f;

  // find u for point on ray (q, b) closest to point at t
  float u;
  if (BDotB != 0.0f) {
    u = (t * ADotB - BDotT) / BDotB;

    // if u is on segment (q, b), t and u correspond to closest points,
    // otherwise, clamp u, recompute and clamp t
    if (u < 0.0f) {
      u = 0.0f;
      t = ADotA != 0.0f ? glm::clamp(ADotT / ADotA, 0.0f, 1.0f) : 0.0f;
    } else if (u > 1.0f) {
      u = 1.0f;
      t = ADotA != 0.0f ? glm::clamp((ADotB + ADotT) / ADotA, 0.0f, 1.0f)
                        : 0.0f;
    }
  } else {
    u = 0.0f;
    t = ADotA != 0.0f ? glm::clamp(ADotT / ADotA, 0.0f, 1.0f) : 0.0f;
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
inline float DistanceTriangleTriangleSquared(
    const std::array<glm::vec3, 3>& p, const std::array<glm::vec3, 3>& q) {
  std::array<glm::vec3, 3> Sv;
  Sv[0] = p[1] - p[0];
  Sv[1] = p[2] - p[1];
  Sv[2] = p[0] - p[2];

  std::array<glm::vec3, 3> Tv;
  Tv[0] = q[1] - q[0];
  Tv[1] = q[2] - q[1];
  Tv[2] = q[0] - q[2];

  bool shown_disjoint = false;

  float mindd = std::numeric_limits<float>::max();

  for (uint32_t i = 0; i < 3; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      glm::vec3 cp;
      glm::vec3 cq;
      EdgeEdgeDist(cp, cq, p[i], Sv[i], q[j], Tv[j]);
      const glm::vec3 V = cq - cp;
      const float dd = glm::dot(V, V);

      if (dd <= mindd) {
        mindd = dd;

        uint32_t id = i + 2;
        if (id >= 3) id -= 3;
        glm::vec3 Z = p[id] - cp;
        float a = glm::dot(Z, V);
        id = j + 2;
        if (id >= 3) id -= 3;
        Z = q[id] - cq;
        float b = glm::dot(Z, V);

        if ((a <= 0.0f) && (b >= 0.0f)) {
          return glm::dot(V, V);
        };

        if (a <= 0.0f)
          a = 0.0f;
        else if (b > 0.0f)
          b = 0.0f;

        if ((mindd - a + b) > 0.0f) shown_disjoint = true;
      }
    }
  }

  glm::vec3 Sn = glm::cross(Sv[0], Sv[1]);
  float Snl = glm::dot(Sn, Sn);

  if (Snl > 1e-15f) {
    const glm::vec3 Tp(glm::dot(p[0] - q[0], Sn), glm::dot(p[0] - q[1], Sn),
                       glm::dot(p[0] - q[2], Sn));

    int index = -1;
    if ((Tp[0] > 0.0f) && (Tp[1] > 0.0f) && (Tp[2] > 0.0f)) {
      index = Tp[0] < Tp[1] ? 0 : 1;
      if (Tp[2] < Tp[index]) index = 2;
    } else if ((Tp[0] < 0.0f) && (Tp[1] < 0.0f) && (Tp[2] < 0.0f)) {
      index = Tp[0] > Tp[1] ? 0 : 1;
      if (Tp[2] > Tp[index]) index = 2;
    }

    if (index >= 0) {
      shown_disjoint = true;

      const glm::vec3& qIndex = q[index];

      glm::vec3 V = qIndex - p[0];
      glm::vec3 Z = glm::cross(Sn, Sv[0]);
      if (glm::dot(V, Z) > 0.0f) {
        V = qIndex - p[1];
        Z = glm::cross(Sn, Sv[1]);
        if (glm::dot(V, Z) > 0.0f) {
          V = qIndex - p[2];
          Z = glm::cross(Sn, Sv[2]);
          if (glm::dot(V, Z) > 0.0f) {
            glm::vec3 cp = qIndex + Sn * Tp[index] / Snl;
            glm::vec3 cq = qIndex;
            return glm::dot(cp - cq, cp - cq);
          }
        }
      }
    }
  }

  glm::vec3 Tn = glm::cross(Tv[0], Tv[1]);
  float Tnl = glm::dot(Tn, Tn);

  if (Tnl > 1e-15f) {
    const glm::vec3 Sp(glm::dot(q[0] - p[0], Tn), glm::dot(q[0] - p[1], Tn),
                       glm::dot(q[0] - p[2], Tn));

    int index = -1;
    if ((Sp[0] > 0.0f) && (Sp[1] > 0.0f) && (Sp[2] > 0.0f)) {
      index = Sp[0] < Sp[1] ? 0 : 1;
      if (Sp[2] < Sp[index]) index = 2;
    } else if ((Sp[0] < 0.0f) && (Sp[1] < 0.0f) && (Sp[2] < 0.0f)) {
      index = Sp[0] > Sp[1] ? 0 : 1;
      if (Sp[2] > Sp[index]) index = 2;
    }

    if (index >= 0) {
      shown_disjoint = true;

      const glm::vec3& pIndex = p[index];

      glm::vec3 V = pIndex - q[0];
      glm::vec3 Z = glm::cross(Tn, Tv[0]);
      if (glm::dot(V, Z) > 0.0f) {
        V = pIndex - q[1];
        Z = glm::cross(Tn, Tv[1]);
        if (glm::dot(V, Z) > 0.0f) {
          V = pIndex - q[2];
          Z = glm::cross(Tn, Tv[2]);
          if (glm::dot(V, Z) > 0.0f) {
            glm::vec3 cp = pIndex;
            glm::vec3 cq = pIndex + Tn * Sp[index] / Tnl;
            return glm::dot(cp - cq, cp - cq);
          }
        }
      }
    }
  }

  return shown_disjoint ? mindd : 0.0f;
};
}  // namespace manifold
