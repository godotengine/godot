/*
 * Agent2d.cpp
 * RVO2 Library
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

/**
 * @file  Agent2d.cpp
 * @brief Defines the Agent2D class.
 */

#include "Agent2d.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "KdTree2d.h"
#include "Obstacle2d.h"

namespace RVO2D {
namespace {
/**
 * @relates        Agent2D
 * @brief          Solves a one-dimensional linear program on a specified line
 *                 subject to linear constraints defined by lines and a circular
 *                 constraint.
 * @param[in]      lines        Lines defining the linear constraints.
 * @param[in]      lineNo       The specified line constraint.
 * @param[in]      radius       The radius of the circular constraint.
 * @param[in]      optVelocity  The optimization velocity.
 * @param[in]      directionOpt True if the direction should be optimized.
 * @param[in, out] result       A reference to the result of the linear program.
 * @return         True if successful.
 */
bool linearProgram1(const std::vector<Line> &lines, std::size_t lineNo,
                    float radius, const Vector2 &optVelocity, bool directionOpt,
                    Vector2 &result) { /* NOLINT(runtime/references) */
  const float dotProduct = lines[lineNo].point * lines[lineNo].direction;
  const float discriminant =
      dotProduct * dotProduct + radius * radius - absSq(lines[lineNo].point);

  if (discriminant < 0.0F) {
    /* Max speed circle fully invalidates line lineNo. */
    return false;
  }

  const float sqrtDiscriminant = std::sqrt(discriminant);
  float tLeft = -dotProduct - sqrtDiscriminant;
  float tRight = -dotProduct + sqrtDiscriminant;

  for (std::size_t i = 0U; i < lineNo; ++i) {
    const float denominator = det(lines[lineNo].direction, lines[i].direction);
    const float numerator =
        det(lines[i].direction, lines[lineNo].point - lines[i].point);

    if (std::fabs(denominator) <= RVO2D_EPSILON) {
      /* Lines lineNo and i are (almost) parallel. */
      if (numerator < 0.0F) {
        return false;
      }

      continue;
    }

    const float t = numerator / denominator;

    if (denominator >= 0.0F) {
      /* Line i bounds line lineNo on the right. */
      tRight = std::min(tRight, t);
    } else {
      /* Line i bounds line lineNo on the left. */
      tLeft = std::max(tLeft, t);
    }

    if (tLeft > tRight) {
      return false;
    }
  }

  if (directionOpt) {
    /* Optimize direction. */
    if (optVelocity * lines[lineNo].direction > 0.0F) {
      /* Take right extreme. */
      result = lines[lineNo].point + tRight * lines[lineNo].direction;
    } else {
      /* Take left extreme. */
      result = lines[lineNo].point + tLeft * lines[lineNo].direction;
    }
  } else {
    /* Optimize closest point. */
    const float t =
        lines[lineNo].direction * (optVelocity - lines[lineNo].point);

    if (t < tLeft) {
      result = lines[lineNo].point + tLeft * lines[lineNo].direction;
    } else if (t > tRight) {
      result = lines[lineNo].point + tRight * lines[lineNo].direction;
    } else {
      result = lines[lineNo].point + t * lines[lineNo].direction;
    }
  }

  return true;
}

/**
 * @relates        Agent2D
 * @brief          Solves a two-dimensional linear program subject to linear
 *                 constraints defined by lines and a circular constraint.
 * @param[in]      lines        Lines defining the linear constraints.
 * @param[in]      radius       The radius of the circular constraint.
 * @param[in]      optVelocity  The optimization velocity.
 * @param[in]      directionOpt True if the direction should be optimized.
 * @param[in, out] result       A reference to the result of the linear program.
 * @return         The number of the line it fails on, and the number of lines
 *                 if successful.
 */
std::size_t linearProgram2(const std::vector<Line> &lines, float radius,
                           const Vector2 &optVelocity, bool directionOpt,
                           Vector2 &result) { /* NOLINT(runtime/references) */
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

  for (std::size_t i = 0U; i < lines.size(); ++i) {
    if (det(lines[i].direction, lines[i].point - result) > 0.0F) {
      /* Result does not satisfy constraint i. Compute new optimal result. */
      const Vector2 tempResult = result;

      if (!linearProgram1(lines, i, radius, optVelocity, directionOpt,
                          result)) {
        result = tempResult;

        return i;
      }
    }
  }

  return lines.size();
}

/**
 * @relates        Agent2D
 * @brief          Solves a two-dimensional linear program subject to linear
 *                 constraints defined by lines and a circular constraint.
 * @param[in]      lines        Lines defining the linear constraints.
 * @param[in]      numObstLines Count of obstacle lines.
 * @param[in]      beginLine    The line on which the 2-d linear program failed.
 * @param[in]      radius       The radius of the circular constraint.
 * @param[in, out] result       A reference to the result of the linear program.
 */
void linearProgram3(const std::vector<Line> &lines, std::size_t numObstLines,
                    std::size_t beginLine, float radius,
                    Vector2 &result) { /* NOLINT(runtime/references) */
  float distance = 0.0F;

  for (std::size_t i = beginLine; i < lines.size(); ++i) {
    if (det(lines[i].direction, lines[i].point - result) > distance) {
      /* Result does not satisfy constraint of line i. */
      std::vector<Line> projLines(
          lines.begin(),
          lines.begin() + static_cast<std::ptrdiff_t>(numObstLines));

      for (std::size_t j = numObstLines; j < i; ++j) {
        Line line;

        const float determinant = det(lines[i].direction, lines[j].direction);

        if (std::fabs(determinant) <= RVO2D_EPSILON) {
          /* Line i and line j are parallel. */
          if (lines[i].direction * lines[j].direction > 0.0F) {
            /* Line i and line j point in the same direction. */
            continue;
          }

          /* Line i and line j point in opposite direction. */
          line.point = 0.5F * (lines[i].point + lines[j].point);
        } else {
          line.point = lines[i].point + (det(lines[j].direction,
                                             lines[i].point - lines[j].point) /
                                         determinant) *
                                            lines[i].direction;
        }

        line.direction = normalize(lines[j].direction - lines[i].direction);
        projLines.push_back(line);
      }

      const Vector2 tempResult = result;

      if (linearProgram2(
              projLines, radius,
              Vector2(-lines[i].direction.y(), lines[i].direction.x()), true,
              result) < projLines.size()) {
        /* This should in principle not happen. The result is by definition
         * already in the feasible region of this linear program. If it fails,
         * it is due to small floating point error, and the current result is
         * kept. */
        result = tempResult;
      }

      distance = det(lines[i].direction, lines[i].point - result);
    }
  }
}
} /* namespace */

Agent2D::Agent2D()
    : id_(0U),
      maxNeighbors_(0U),
      maxSpeed_(0.0F),
      neighborDist_(0.0F),
      radius_(0.0F),
      timeHorizon_(0.0F),
      timeHorizonObst_(0.0F) {}

Agent2D::~Agent2D() {}

void Agent2D::computeNeighbors(const KdTree2D *kdTree) {
  obstacleNeighbors_.clear();
  const float range = timeHorizonObst_ * maxSpeed_ + radius_;
  kdTree->computeObstacleNeighbors(this, range * range);

  agentNeighbors_.clear();

  if (maxNeighbors_ > 0U) {
    float rangeSq = neighborDist_ * neighborDist_;
    kdTree->computeAgentNeighbors(this, rangeSq);
  }
}

/* Search for the best new velocity. */
void Agent2D::computeNewVelocity(float timeStep) {
  orcaLines_.clear();

  const float invTimeHorizonObst = 1.0F / timeHorizonObst_;

  /* Create obstacle ORCA lines. */
  for (std::size_t i = 0U; i < obstacleNeighbors_.size(); ++i) {
    const Obstacle2D *obstacle1 = obstacleNeighbors_[i].second;
    const Obstacle2D *obstacle2 = obstacle1->next_;

    const Vector2 relativePosition1 = obstacle1->point_ - position_;
    const Vector2 relativePosition2 = obstacle2->point_ - position_;

    /* Check if velocity obstacle of obstacle is already taken care of by
     * previously constructed obstacle ORCA lines. */
    bool alreadyCovered = false;

    for (std::size_t j = 0U; j < orcaLines_.size(); ++j) {
      if (det(invTimeHorizonObst * relativePosition1 - orcaLines_[j].point,
              orcaLines_[j].direction) -
                  invTimeHorizonObst * radius_ >=
              -RVO2D_EPSILON &&
          det(invTimeHorizonObst * relativePosition2 - orcaLines_[j].point,
              orcaLines_[j].direction) -
                  invTimeHorizonObst * radius_ >=
              -RVO2D_EPSILON) {
        alreadyCovered = true;
        break;
      }
    }

    if (alreadyCovered) {
      continue;
    }

    /* Not yet covered. Check for collisions. */
    const float distSq1 = absSq(relativePosition1);
    const float distSq2 = absSq(relativePosition2);

    const float radiusSq = radius_ * radius_;

    const Vector2 obstacleVector = obstacle2->point_ - obstacle1->point_;
    const float s =
        (-relativePosition1 * obstacleVector) / absSq(obstacleVector);
    const float distSqLine = absSq(-relativePosition1 - s * obstacleVector);

    Line line;

    if (s < 0.0F && distSq1 <= radiusSq) {
      /* Collision with left vertex. Ignore if non-convex. */
      if (obstacle1->isConvex_) {
        line.point = Vector2(0.0F, 0.0F);
        line.direction =
            normalize(Vector2(-relativePosition1.y(), relativePosition1.x()));
        orcaLines_.push_back(line);
      }

      continue;
    }

    if (s > 1.0F && distSq2 <= radiusSq) {
      /* Collision with right vertex. Ignore if non-convex or if it will be
       * taken care of by neighoring obstace */
      if (obstacle2->isConvex_ &&
          det(relativePosition2, obstacle2->direction_) >= 0.0F) {
        line.point = Vector2(0.0F, 0.0F);
        line.direction =
            normalize(Vector2(-relativePosition2.y(), relativePosition2.x()));
        orcaLines_.push_back(line);
      }

      continue;
    }

    if (s >= 0.0F && s <= 1.0F && distSqLine <= radiusSq) {
      /* Collision with obstacle segment. */
      line.point = Vector2(0.0F, 0.0F);
      line.direction = -obstacle1->direction_;
      orcaLines_.push_back(line);
      continue;
    }

    /* No collision. Compute legs. When obliquely viewed, both legs can come
     * from a single vertex. Legs extend cut-off line when nonconvex vertex. */
    Vector2 leftLegDirection;
    Vector2 rightLegDirection;

    if (s < 0.0F && distSqLine <= radiusSq) {
      /* Obstacle2D viewed obliquely so that left vertex defines velocity
       * obstacle. */
      if (!obstacle1->isConvex_) {
        /* Ignore obstacle. */
        continue;
      }

      obstacle2 = obstacle1;

      const float leg1 = std::sqrt(distSq1 - radiusSq);
      leftLegDirection =
          Vector2(
              relativePosition1.x() * leg1 - relativePosition1.y() * radius_,
              relativePosition1.x() * radius_ + relativePosition1.y() * leg1) /
          distSq1;
      rightLegDirection =
          Vector2(
              relativePosition1.x() * leg1 + relativePosition1.y() * radius_,
              -relativePosition1.x() * radius_ + relativePosition1.y() * leg1) /
          distSq1;
    } else if (s > 1.0F && distSqLine <= radiusSq) {
      /* Obstacle2D viewed obliquely so that right vertex defines velocity
       * obstacle. */
      if (!obstacle2->isConvex_) {
        /* Ignore obstacle. */
        continue;
      }

      obstacle1 = obstacle2;

      const float leg2 = std::sqrt(distSq2 - radiusSq);
      leftLegDirection =
          Vector2(
              relativePosition2.x() * leg2 - relativePosition2.y() * radius_,
              relativePosition2.x() * radius_ + relativePosition2.y() * leg2) /
          distSq2;
      rightLegDirection =
          Vector2(
              relativePosition2.x() * leg2 + relativePosition2.y() * radius_,
              -relativePosition2.x() * radius_ + relativePosition2.y() * leg2) /
          distSq2;
    } else {
      /* Usual situation. */
      if (obstacle1->isConvex_) {
        const float leg1 = std::sqrt(distSq1 - radiusSq);
        leftLegDirection = Vector2(relativePosition1.x() * leg1 -
                                       relativePosition1.y() * radius_,
                                   relativePosition1.x() * radius_ +
                                       relativePosition1.y() * leg1) /
                           distSq1;
      } else {
        /* Left vertex non-convex; left leg extends cut-off line. */
        leftLegDirection = -obstacle1->direction_;
      }

      if (obstacle2->isConvex_) {
        const float leg2 = std::sqrt(distSq2 - radiusSq);
        rightLegDirection = Vector2(relativePosition2.x() * leg2 +
                                        relativePosition2.y() * radius_,
                                    -relativePosition2.x() * radius_ +
                                        relativePosition2.y() * leg2) /
                            distSq2;
      } else {
        /* Right vertex non-convex; right leg extends cut-off line. */
        rightLegDirection = obstacle1->direction_;
      }
    }

    /* Legs can never point into neighboring edge when convex vertex, take
     * cutoff-line of neighboring edge instead. If velocity projected on
     * "foreign" leg, no constraint is added. */
    const Obstacle2D *const leftNeighbor = obstacle1->previous_;

    bool isLeftLegForeign = false;
    bool isRightLegForeign = false;

    if (obstacle1->isConvex_ &&
        det(leftLegDirection, -leftNeighbor->direction_) >= 0.0F) {
      /* Left leg points into obstacle. */
      leftLegDirection = -leftNeighbor->direction_;
      isLeftLegForeign = true;
    }

    if (obstacle2->isConvex_ &&
        det(rightLegDirection, obstacle2->direction_) <= 0.0F) {
      /* Right leg points into obstacle. */
      rightLegDirection = obstacle2->direction_;
      isRightLegForeign = true;
    }

    /* Compute cut-off centers. */
    const Vector2 leftCutoff =
        invTimeHorizonObst * (obstacle1->point_ - position_);
    const Vector2 rightCutoff =
        invTimeHorizonObst * (obstacle2->point_ - position_);
    const Vector2 cutoffVector = rightCutoff - leftCutoff;

    /* Project current velocity on velocity obstacle. */

    /* Check if current velocity is projected on cutoff circles. */
    const float t =
        obstacle1 == obstacle2
            ? 0.5F
            : (velocity_ - leftCutoff) * cutoffVector / absSq(cutoffVector);
    const float tLeft = (velocity_ - leftCutoff) * leftLegDirection;
    const float tRight = (velocity_ - rightCutoff) * rightLegDirection;

    if ((t < 0.0F && tLeft < 0.0F) ||
        (obstacle1 == obstacle2 && tLeft < 0.0F && tRight < 0.0F)) {
      /* Project on left cut-off circle. */
      const Vector2 unitW = normalize(velocity_ - leftCutoff);

      line.direction = Vector2(unitW.y(), -unitW.x());
      line.point = leftCutoff + radius_ * invTimeHorizonObst * unitW;
      orcaLines_.push_back(line);
      continue;
    }

    if (t > 1.0F && tRight < 0.0F) {
      /* Project on right cut-off circle. */
      const Vector2 unitW = normalize(velocity_ - rightCutoff);

      line.direction = Vector2(unitW.y(), -unitW.x());
      line.point = rightCutoff + radius_ * invTimeHorizonObst * unitW;
      orcaLines_.push_back(line);
      continue;
    }

    /* Project on left leg, right leg, or cut-off line, whichever is closest to
     * velocity. */
    const float distSqCutoff =
        (t < 0.0F || t > 1.0F || obstacle1 == obstacle2)
            ? std::numeric_limits<float>::infinity()
            : absSq(velocity_ - (leftCutoff + t * cutoffVector));
    const float distSqLeft =
        tLeft < 0.0F
            ? std::numeric_limits<float>::infinity()
            : absSq(velocity_ - (leftCutoff + tLeft * leftLegDirection));
    const float distSqRight =
        tRight < 0.0F
            ? std::numeric_limits<float>::infinity()
            : absSq(velocity_ - (rightCutoff + tRight * rightLegDirection));

    if (distSqCutoff <= distSqLeft && distSqCutoff <= distSqRight) {
      /* Project on cut-off line. */
      line.direction = -obstacle1->direction_;
      line.point =
          leftCutoff + radius_ * invTimeHorizonObst *
                           Vector2(-line.direction.y(), line.direction.x());
      orcaLines_.push_back(line);
      continue;
    }

    if (distSqLeft <= distSqRight) {
      /* Project on left leg. */
      if (isLeftLegForeign) {
        continue;
      }

      line.direction = leftLegDirection;
      line.point =
          leftCutoff + radius_ * invTimeHorizonObst *
                           Vector2(-line.direction.y(), line.direction.x());
      orcaLines_.push_back(line);
      continue;
    }

    /* Project on right leg. */
    if (isRightLegForeign) {
      continue;
    }

    line.direction = -rightLegDirection;
    line.point =
        rightCutoff + radius_ * invTimeHorizonObst *
                          Vector2(-line.direction.y(), line.direction.x());
    orcaLines_.push_back(line);
  }

  const std::size_t numObstLines = orcaLines_.size();

  const float invTimeHorizon = 1.0F / timeHorizon_;

  /* Create agent ORCA lines. */
  for (std::size_t i = 0U; i < agentNeighbors_.size(); ++i) {
    const Agent2D *const other = agentNeighbors_[i].second;

    const Vector2 relativePosition = other->position_ - position_;
    const Vector2 relativeVelocity = velocity_ - other->velocity_;
    const float distSq = absSq(relativePosition);
    const float combinedRadius = radius_ + other->radius_;
    const float combinedRadiusSq = combinedRadius * combinedRadius;

    Line line;
    Vector2 u;

    if (distSq > combinedRadiusSq) {
      /* No collision. */
      const Vector2 w = relativeVelocity - invTimeHorizon * relativePosition;
      /* Vector from cutoff center to relative velocity. */
      const float wLengthSq = absSq(w);

      const float dotProduct = w * relativePosition;

      if (dotProduct < 0.0F &&
          dotProduct * dotProduct > combinedRadiusSq * wLengthSq) {
        /* Project on cut-off circle. */
        const float wLength = std::sqrt(wLengthSq);
        const Vector2 unitW = w / wLength;

        line.direction = Vector2(unitW.y(), -unitW.x());
        u = (combinedRadius * invTimeHorizon - wLength) * unitW;
      } else {
        /* Project on legs. */
        const float leg = std::sqrt(distSq - combinedRadiusSq);

        if (det(relativePosition, w) > 0.0F) {
          /* Project on left leg. */
          line.direction = Vector2(relativePosition.x() * leg -
                                       relativePosition.y() * combinedRadius,
                                   relativePosition.x() * combinedRadius +
                                       relativePosition.y() * leg) /
                           distSq;
        } else {
          /* Project on right leg. */
          line.direction = -Vector2(relativePosition.x() * leg +
                                        relativePosition.y() * combinedRadius,
                                    -relativePosition.x() * combinedRadius +
                                        relativePosition.y() * leg) /
                           distSq;
        }

        u = (relativeVelocity * line.direction) * line.direction -
            relativeVelocity;
      }
    } else {
      /* Collision. Project on cut-off circle of time timeStep. */
      const float invTimeStep = 1.0F / timeStep;

      /* Vector from cutoff center to relative velocity. */
      const Vector2 w = relativeVelocity - invTimeStep * relativePosition;

      const float wLength = abs(w);
      const Vector2 unitW = w / wLength;

      line.direction = Vector2(unitW.y(), -unitW.x());
      u = (combinedRadius * invTimeStep - wLength) * unitW;
    }

    line.point = velocity_ + 0.5F * u;
    orcaLines_.push_back(line);
  }

  const std::size_t lineFail =
      linearProgram2(orcaLines_, maxSpeed_, prefVelocity_, false, newVelocity_);

  if (lineFail < orcaLines_.size()) {
    linearProgram3(orcaLines_, numObstLines, lineFail, maxSpeed_, newVelocity_);
  }
}

void Agent2D::insertAgentNeighbor(const Agent2D *agent, float &rangeSq) {
	// no point processing same agent
	if (this == agent) {
		return;
	}
	// ignore other agent if layers/mask bitmasks have no matching bit
	if ((avoidance_mask_ & agent->avoidance_layers_) == 0) {
		return;
	}
	// ignore other agent if this agent is below or above
	if ((elevation_ > agent->elevation_ + agent->height_) || (elevation_ + height_ < agent->elevation_)) {
		return;
	}

	if (avoidance_priority_ > agent->avoidance_priority_) {
		return;
	}
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

void Agent2D::insertObstacleNeighbor(const Obstacle2D *obstacle, float rangeSq) {
  const Obstacle2D *const nextObstacle = obstacle->next_;

  float distSq = 0.0F;
  const float r = ((position_ - obstacle->point_) *
                   (nextObstacle->point_ - obstacle->point_)) /
                  absSq(nextObstacle->point_ - obstacle->point_);

  if (r < 0.0F) {
    distSq = absSq(position_ - obstacle->point_);
  } else if (r > 1.0F) {
    distSq = absSq(position_ - nextObstacle->point_);
  } else {
    distSq = absSq(position_ - (obstacle->point_ +
                                r * (nextObstacle->point_ - obstacle->point_)));
  }

  if (distSq < rangeSq) {
    obstacleNeighbors_.push_back(std::make_pair(distSq, obstacle));

    std::size_t i = obstacleNeighbors_.size() - 1U;

    while (i != 0U && distSq < obstacleNeighbors_[i - 1U].first) {
      obstacleNeighbors_[i] = obstacleNeighbors_[i - 1U];
      --i;
    }

    obstacleNeighbors_[i] = std::make_pair(distSq, obstacle);
  }
}

void Agent2D::update(float timeStep) {
  velocity_ = newVelocity_;
  position_ += velocity_ * timeStep;
}
} /* namespace RVO2D */
