/*
 * Agent3d.cc
 * RVO2-3D Library
 *
 * SPDX-FileCopyrightText: 2008 University of North Carolina at Chapel Hill
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <https://gamma.cs.unc.edu/RVO2/>
 */

#include "Agent3d.h"

#include <algorithm>
#include <cmath>

#include "KdTree3d.h"
#include "RVOSimulator3d.h"

namespace RVO3D {
namespace {
/**
 * @brief A sufficiently small positive number.
 */
const float RVO3D_EPSILON = 0.00001F;

/**
 * @brief Defines a directed line.
 */
class Line3D {
 public:
  /**
   * @brief Constructs a directed line.``
   */
  Line3D();

  /**
   * @brief The direction of the directed line.
   */
  Vector3 direction;

  /**
   * @brief A point on the directed line.
   */
  Vector3 point;
};

Line3D::Line3D() {}

/**
 * @brief     Solves a one-dimensional linear program on a specified line
 *            subject to linear constraints defined by planes and a spherical
 *            constraint.
 * @param[in] planes       Planes defining the linear constraints.
 * @param[in] planeNo      The plane on which the line lies.
 * @param[in] line         The line on which the one-dimensional linear program
 *                         is solved.
 * @param[in] radius       The radius of the spherical constraint.
 * @param[in] optVelocity  The optimization velocity.
 * @param[in] directionOpt True if the direction should be optimized.
 * @param[in] result       A reference to the result of the linear program.
 * @return True if successful.
 */
bool linearProgram1(const std::vector<Plane> &planes, std::size_t planeNo,
                    const Line3D &line, float radius, const Vector3 &optVelocity,
                    bool directionOpt,
                    Vector3 &result) { /* NOLINT(runtime/references) */
  const float dotProduct = line.point * line.direction;
  const float discriminant =
      dotProduct * dotProduct + radius * radius - absSq(line.point);

  if (discriminant < 0.0F) {
    /* Max speed sphere fully invalidates line. */
    return false;
  }

  const float sqrtDiscriminant = std::sqrt(discriminant);
  float tLeft = -dotProduct - sqrtDiscriminant;
  float tRight = -dotProduct + sqrtDiscriminant;

  for (std::size_t i = 0U; i < planeNo; ++i) {
    const float numerator = (planes[i].point - line.point) * planes[i].normal;
    const float denominator = line.direction * planes[i].normal;

    if (denominator * denominator <= RVO3D_EPSILON) {
      /* Lines line is (almost) parallel to plane i. */
      if (numerator > 0.0F) {
        return false;
      }

      continue;
    }

    const float t = numerator / denominator;

    if (denominator >= 0.0F) {
      /* Plane i bounds line on the left. */
      tLeft = std::max(tLeft, t);
    } else {
      /* Plane i bounds line on the right. */
      tRight = std::min(tRight, t);
    }

    if (tLeft > tRight) {
      return false;
    }
  }

  if (directionOpt) {
    /* Optimize direction. */
    if (optVelocity * line.direction > 0.0F) {
      /* Take right extreme. */
      result = line.point + tRight * line.direction;
    } else {
      /* Take left extreme. */
      result = line.point + tLeft * line.direction;
    }
  } else {
    /* Optimize closest point. */
    const float t = line.direction * (optVelocity - line.point);

    if (t < tLeft) {
      result = line.point + tLeft * line.direction;
    } else if (t > tRight) {
      result = line.point + tRight * line.direction;
    } else {
      result = line.point + t * line.direction;
    }
  }

  return true;
}

/**
 * @brief      Solves a two-dimensional linear program on a specified plane
 *             subject to linear constraints defined by planes and a spherical
 *             constraint.
 * @param[in]  planes       Planes defining the linear constraints.
 * @param[in]  planeNo      The plane on which the two-dimensional linear
 *                          program is solved.
 * @param[in]  radius       The radius of the spherical constraint.
 * @param[in]  optVelocity  The optimization velocity.
 * @param[in]  directionOpt True if the direction should be optimized.
 * @param[out] result       A reference to the result of the linear program.
 * @return     True if successful.
 */
bool linearProgram2(const std::vector<Plane> &planes, std::size_t planeNo,
                    float radius, const Vector3 &optVelocity, bool directionOpt,
                    Vector3 &result) { /* NOLINT(runtime/references) */
  const float planeDist = planes[planeNo].point * planes[planeNo].normal;
  const float planeDistSq = planeDist * planeDist;
  const float radiusSq = radius * radius;

  if (planeDistSq > radiusSq) {
    /* Max speed sphere fully invalidates plane planeNo. */
    return false;
  }

  const float planeRadiusSq = radiusSq - planeDistSq;

  const Vector3 planeCenter = planeDist * planes[planeNo].normal;

  if (directionOpt) {
    /* Project direction optVelocity on plane planeNo. */
    const Vector3 planeOptVelocity =
        optVelocity -
        (optVelocity * planes[planeNo].normal) * planes[planeNo].normal;
    const float planeOptVelocityLengthSq = absSq(planeOptVelocity);

    if (planeOptVelocityLengthSq <= RVO3D_EPSILON) {
      result = planeCenter;
    } else {
      result =
          planeCenter + std::sqrt(planeRadiusSq / planeOptVelocityLengthSq) *
                            planeOptVelocity;
    }
  } else {
    /* Project point optVelocity on plane planeNo. */
    result = optVelocity +
             ((planes[planeNo].point - optVelocity) * planes[planeNo].normal) *
                 planes[planeNo].normal;

    /* If outside planeCircle, project on planeCircle. */
    if (absSq(result) > radiusSq) {
      const Vector3 planeResult = result - planeCenter;
      const float planeResultLengthSq = absSq(planeResult);
      result = planeCenter +
               std::sqrt(planeRadiusSq / planeResultLengthSq) * planeResult;
    }
  }

  for (std::size_t i = 0U; i < planeNo; ++i) {
    if (planes[i].normal * (planes[i].point - result) > 0.0F) {
      /* Result does not satisfy constraint i. Compute new optimal result.
       * Compute intersection line of plane i and plane planeNo.
       */
      Vector3 crossProduct = cross(planes[i].normal, planes[planeNo].normal);

      if (absSq(crossProduct) <= RVO3D_EPSILON) {
        /* Planes planeNo and i are (almost) parallel, and plane i fully
         * invalidates plane planeNo.
         */
        return false;
      }

      Line3D line;
      line.direction = normalize(crossProduct);
      const Vector3 lineNormal = cross(line.direction, planes[planeNo].normal);
      line.point =
          planes[planeNo].point +
          (((planes[i].point - planes[planeNo].point) * planes[i].normal) /
           (lineNormal * planes[i].normal)) *
              lineNormal;

      if (!linearProgram1(planes, i, line, radius, optVelocity, directionOpt,
                          result)) {
        return false;
      }
    }
  }

  return true;
}

/**
 * @brief      Solves a three-dimensional linear program subject to linear
 *             constraints defined by planes and a spherical constraint.
 * @param[in]  planes       Planes defining the linear constraints.
 * @param[in]  radius       The radius of the spherical constraint.
 * @param[in]  optVelocity  The optimization velocity.
 * @param[in]  directionOpt True if the direction should be optimized.
 * @param[out] result       A reference to the result of the linear program.
 * @return     The number of the plane it fails on, and the number of planes if
 *             successful.
 */
std::size_t linearProgram3(const std::vector<Plane> &planes, float radius,
                           const Vector3 &optVelocity, bool directionOpt,
                           Vector3 &result) { /* NOLINT(runtime/references) */
  if (directionOpt) {
    /* Optimize direction. Note that the optimization velocity is of unit length
     * in this case.
     */
    result = optVelocity * radius;
  } else if (absSq(optVelocity) > radius * radius) {
    /* Optimize closest point and outside circle. */
    result = normalize(optVelocity) * radius;
  } else {
    /* Optimize closest point and inside circle. */
    result = optVelocity;
  }

  for (std::size_t i = 0U; i < planes.size(); ++i) {
    if (planes[i].normal * (planes[i].point - result) > 0.0F) {
      /* Result does not satisfy constraint i. Compute new optimal result. */
      const Vector3 tempResult = result;

      if (!linearProgram2(planes, i, radius, optVelocity, directionOpt,
                          result)) {
        result = tempResult;
        return i;
      }
    }
  }

  return planes.size();
}

/**
 * @brief      Solves a four-dimensional linear program subject to linear
 *             constraints defined by planes and a spherical constraint.
 * @param[in]  planes     Planes defining the linear constraints.
 * @param[in]  beginPlane The plane on which the three-dimensional linear
 *                        program failed.
 * @param[in]  radius     The radius of the spherical constraint.
 * @param[out] result     A reference to the result of the linear program.
 */
void linearProgram4(const std::vector<Plane> &planes, std::size_t beginPlane,
                    float radius,
                    Vector3 &result) { /* NOLINT(runtime/references) */
  float distance = 0.0F;

  for (std::size_t i = beginPlane; i < planes.size(); ++i) {
    if (planes[i].normal * (planes[i].point - result) > distance) {
      /* Result does not satisfy constraint of plane i. */
      std::vector<Plane> projPlanes;

      for (std::size_t j = 0U; j < i; ++j) {
        Plane plane;

        const Vector3 crossProduct = cross(planes[j].normal, planes[i].normal);

        if (absSq(crossProduct) <= RVO3D_EPSILON) {
          /* Plane i and plane j are (almost) parallel. */
          if (planes[i].normal * planes[j].normal > 0.0F) {
            /* Plane i and plane j point in the same direction. */
            continue;
          }

          /* Plane i and plane j point in opposite direction. */
          plane.point = 0.5F * (planes[i].point + planes[j].point);
        } else {
          /* Plane.point is point on line of intersection between plane i and
           * plane j.
           */
          const Vector3 lineNormal = cross(crossProduct, planes[i].normal);
          plane.point =
              planes[i].point +
              (((planes[j].point - planes[i].point) * planes[j].normal) /
               (lineNormal * planes[j].normal)) *
                  lineNormal;
        }

        plane.normal = normalize(planes[j].normal - planes[i].normal);
        projPlanes.push_back(plane);
      }

      const Vector3 tempResult = result;

      if (linearProgram3(projPlanes, radius, planes[i].normal, true, result) <
          projPlanes.size()) {
        /* This should in principle not happen. The result is by definition
         * already in the feasible region of this linear program. If it fails,
         * it is due to small floating point error, and the current result is
         * kept.
         */
        result = tempResult;
      }

      distance = planes[i].normal * (planes[i].point - result);
    }
  }
}
} /* namespace */

Agent3D::Agent3D()
    : id_(0U),
      maxNeighbors_(0U),
      maxSpeed_(0.0F),
      neighborDist_(0.0F),
      radius_(0.0F),
      timeHorizon_(0.0F) {}

Agent3D::~Agent3D() {}

void Agent3D::computeNeighbors(RVOSimulator3D *sim_) {
	agentNeighbors_.clear();

	if (maxNeighbors_ > 0) {
		sim_->kdTree_->computeAgentNeighbors(this, neighborDist_ * neighborDist_);
	}
}

void Agent3D::computeNewVelocity(RVOSimulator3D *sim_) {
  orcaPlanes_.clear();
  const float invTimeHorizon = 1.0F / timeHorizon_;

  /* Create agent ORCA planes. */
  for (std::size_t i = 0U; i < agentNeighbors_.size(); ++i) {
    const Agent3D *const other = agentNeighbors_[i].second;
    const Vector3 relativePosition = other->position_ - position_;
    const Vector3 relativeVelocity = velocity_ - other->velocity_;
    const float distSq = absSq(relativePosition);
    const float combinedRadius = radius_ + other->radius_;
    const float combinedRadiusSq = combinedRadius * combinedRadius;

    Plane plane;
    Vector3 u;

    if (distSq > combinedRadiusSq) {
      /* No collision. */
      const Vector3 w = relativeVelocity - invTimeHorizon * relativePosition;
      /* Vector from cutoff center to relative velocity. */
      const float wLengthSq = absSq(w);

      const float dotProduct = w * relativePosition;

      if (dotProduct < 0.0F &&
          dotProduct * dotProduct > combinedRadiusSq * wLengthSq) {
        /* Project on cut-off circle. */
        const float wLength = std::sqrt(wLengthSq);
        const Vector3 unitW = w / wLength;

        plane.normal = unitW;
        u = (combinedRadius * invTimeHorizon - wLength) * unitW;
      } else {
        /* Project on cone. */
        const float a = distSq;
        const float b = relativePosition * relativeVelocity;
        const float c = absSq(relativeVelocity) -
                        absSq(cross(relativePosition, relativeVelocity)) /
                            (distSq - combinedRadiusSq);
        const float t = (b + std::sqrt(b * b - a * c)) / a;
        const Vector3 ww = relativeVelocity - t * relativePosition;
        const float wwLength = abs(ww);
        const Vector3 unitWW = ww / wwLength;

        plane.normal = unitWW;
        u = (combinedRadius * t - wwLength) * unitWW;
      }
    } else {
      /* Collision. */
      const float invTimeStep = 1.0F / sim_->timeStep_;
      const Vector3 w = relativeVelocity - invTimeStep * relativePosition;
      const float wLength = abs(w);
      const Vector3 unitW = w / wLength;

      plane.normal = unitW;
      u = (combinedRadius * invTimeStep - wLength) * unitW;
    }

    plane.point = velocity_ + 0.5F * u;
    orcaPlanes_.push_back(plane);
  }

  const std::size_t planeFail = linearProgram3(
      orcaPlanes_, maxSpeed_, prefVelocity_, false, newVelocity_);

  if (planeFail < orcaPlanes_.size()) {
    linearProgram4(orcaPlanes_, planeFail, maxSpeed_, newVelocity_);
  }
}

void Agent3D::insertAgentNeighbor(const Agent3D *agent, float &rangeSq) {
  if (this != agent) {
    const float distSq = absSq(position_ - agent->position_);

    if (distSq < rangeSq) {
      if (agentNeighbors_.size() < maxNeighbors_) {
        agentNeighbors_.push_back(std::make_pair(distSq, agent));
      }

      std::size_t i = agentNeighbors_.size() - 1U;

      while (i != 0U && distSq < agentNeighbors_[i - 1U].first) {
        agentNeighbors_[i] = agentNeighbors_[i - 1U];
        --i;
      }

      agentNeighbors_[i] = std::make_pair(distSq, agent);

      if (agentNeighbors_.size() == maxNeighbors_) {
        rangeSq = agentNeighbors_.back().first;
      }
    }
  }
}

void Agent3D::update(RVOSimulator3D *sim_) {
  velocity_ = newVelocity_;
  position_ += velocity_ * sim_->timeStep_;
}
} /* namespace RVO3D */
