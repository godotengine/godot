/*
 * Agent.cpp
 * RVO2-3D Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
 * <http://gamma.cs.unc.edu/RVO2/>
 */

#include "Agent.h"

#include <algorithm>
#include <cmath>

#include "Definitions.h"
#include "KdTree.h"

namespace RVO {
/**
	 * \brief   A sufficiently small positive number.
	 */
const float RVO_EPSILON = 0.00001f;

/**
	 * \brief   Defines a directed line.
	 */
class Line {
public:
    /**
		 * \brief   The direction of the directed line.
		 */
    Vector3 direction;

    /**
		 * \brief   A point on the directed line.
		 */
    Vector3 point;
};

/**
	 * \brief   Solves a one-dimensional linear program on a specified line subject to linear constraints defined by planes and a spherical constraint.
	 * \param   planes        Planes defining the linear constraints.
	 * \param   planeNo       The plane on which the line lies.
	 * \param   line          The line on which the 1-d linear program is solved
	 * \param   radius        The radius of the spherical constraint.
	 * \param   optVelocity   The optimization velocity.
	 * \param   directionOpt  True if the direction should be optimized.
	 * \param   result        A reference to the result of the linear program.
	 * \return  True if successful.
	 */
bool linearProgram1(const std::vector<Plane> &planes, size_t planeNo, const Line &line, float radius, const Vector3 &optVelocity, bool directionOpt, Vector3 &result);

/**
	 * \brief   Solves a two-dimensional linear program on a specified plane subject to linear constraints defined by planes and a spherical constraint.
	 * \param   planes        Planes defining the linear constraints.
	 * \param   planeNo       The plane on which the 2-d linear program is solved
	 * \param   radius        The radius of the spherical constraint.
	 * \param   optVelocity   The optimization velocity.
	 * \param   directionOpt  True if the direction should be optimized.
	 * \param   result        A reference to the result of the linear program.
	 * \return  True if successful.
	 */
bool linearProgram2(const std::vector<Plane> &planes, size_t planeNo, float radius, const Vector3 &optVelocity, bool directionOpt, Vector3 &result);

/**
	 * \brief   Solves a three-dimensional linear program subject to linear constraints defined by planes and a spherical constraint.
	 * \param   planes        Planes defining the linear constraints.
	 * \param   radius        The radius of the spherical constraint.
	 * \param   optVelocity   The optimization velocity.
	 * \param   directionOpt  True if the direction should be optimized.
	 * \param   result        A reference to the result of the linear program.
	 * \return  The number of the plane it fails on, and the number of planes if successful.
	 */
size_t linearProgram3(const std::vector<Plane> &planes, float radius, const Vector3 &optVelocity, bool directionOpt, Vector3 &result);

/**
	 * \brief   Solves a four-dimensional linear program subject to linear constraints defined by planes and a spherical constraint.
	 * \param   planes     Planes defining the linear constraints.
	 * \param   beginPlane The plane on which the 3-d linear program failed.
	 * \param   radius     The radius of the spherical constraint.
	 * \param   result     A reference to the result of the linear program.
	 */
void linearProgram4(const std::vector<Plane> &planes, size_t beginPlane, float radius, Vector3 &result);

Agent::Agent() :
        id_(0), maxNeighbors_(0), maxSpeed_(0.0f), neighborDist_(0.0f), radius_(0.0f), timeHorizon_(0.0f), ignore_y_(false) {}

void Agent::computeNeighbors(KdTree *kdTree_) {
    agentNeighbors_.clear();
    if (maxNeighbors_ > 0) {
        kdTree_->computeAgentNeighbors(this, neighborDist_ * neighborDist_);
    }
}

#define ABS(m_v) (((m_v) < 0) ? (-(m_v)) : (m_v))
void Agent::computeNewVelocity(float timeStep) {
    orcaPlanes_.clear();
    const float invTimeHorizon = 1.0f / timeHorizon_;

    /* Create agent ORCA planes. */
    for (size_t i = 0; i < agentNeighbors_.size(); ++i) {
        const Agent *const other = agentNeighbors_[i].second;

        Vector3 relativePosition = other->position_ - position_;
        Vector3 relativeVelocity = velocity_ - other->velocity_;
        const float combinedRadius = radius_ + other->radius_;

        // This is a Godot feature that allow the agents to avoid the collision
        // by moving only on the horizontal plane relative to the player velocity.
        if (ignore_y_) {
            // Skip if these are in two different heights
            if (ABS(relativePosition[1]) > combinedRadius * 2) {
                continue;
            }
            relativePosition[1] = 0;
            relativeVelocity[1] = 0;
        }

        const float distSq = absSq(relativePosition);
        const float combinedRadiusSq = sqr(combinedRadius);

        Plane plane;
        Vector3 u;

        if (distSq > combinedRadiusSq) {
            /* No collision. */
            const Vector3 w = relativeVelocity - invTimeHorizon * relativePosition;
            /* Vector from cutoff center to relative velocity. */
            const float wLengthSq = absSq(w);

            const float dotProduct = w * relativePosition;

            if (dotProduct < 0.0f && sqr(dotProduct) > combinedRadiusSq * wLengthSq) {
                /* Project on cut-off circle. */
                const float wLength = std::sqrt(wLengthSq);
                const Vector3 unitW = w / wLength;

                plane.normal = unitW;
                u = (combinedRadius * invTimeHorizon - wLength) * unitW;
            } else {
                /* Project on cone. */
                const float a = distSq;
                const float b = relativePosition * relativeVelocity;
                const float c = absSq(relativeVelocity) - absSq(cross(relativePosition, relativeVelocity)) / (distSq - combinedRadiusSq);
                const float t = (b + std::sqrt(sqr(b) - a * c)) / a;
                const Vector3 w = relativeVelocity - t * relativePosition;
				const float wLength = abs(w);
				const Vector3 unitW = w / wLength;

				plane.normal = unitW;
                u = (combinedRadius * t - wLength) * unitW;
			}
        } else {
            /* Collision. */
            const float invTimeStep = 1.0f / timeStep;
            const Vector3 w = relativeVelocity - invTimeStep * relativePosition;
            const float wLength = abs(w);
            const Vector3 unitW = w / wLength;

            plane.normal = unitW;
            u = (combinedRadius * invTimeStep - wLength) * unitW;
		}

        plane.point = velocity_ + 0.5f * u;
        orcaPlanes_.push_back(plane);
    }

    const size_t planeFail = linearProgram3(orcaPlanes_, maxSpeed_, prefVelocity_, false, newVelocity_);

    if (planeFail < orcaPlanes_.size()) {
        linearProgram4(orcaPlanes_, planeFail, maxSpeed_, newVelocity_);
    }

    if (ignore_y_) {
        // Not 100% necessary, but better to have.
        newVelocity_[1] = prefVelocity_[1];
    }
}

void Agent::insertAgentNeighbor(const Agent *agent, float &rangeSq) {
    if (this != agent) {
        const float distSq = absSq(position_ - agent->position_);

        if (distSq < rangeSq) {
            if (agentNeighbors_.size() < maxNeighbors_) {
                agentNeighbors_.push_back(std::make_pair(distSq, agent));
            }

            size_t i = agentNeighbors_.size() - 1;

            while (i != 0 && distSq < agentNeighbors_[i - 1].first) {
                agentNeighbors_[i] = agentNeighbors_[i - 1];
                --i;
            }

            agentNeighbors_[i] = std::make_pair(distSq, agent);

            if (agentNeighbors_.size() == maxNeighbors_) {
                rangeSq = agentNeighbors_.back().first;
			}
		}
	}
}

bool linearProgram1(const std::vector<Plane> &planes, size_t planeNo, const Line &line, float radius, const Vector3 &optVelocity, bool directionOpt, Vector3 &result) {
    const float dotProduct = line.point * line.direction;
    const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(line.point);

    if (discriminant < 0.0f) {
        /* Max speed sphere fully invalidates line. */
        return false;
    }

    const float sqrtDiscriminant = std::sqrt(discriminant);
    float tLeft = -dotProduct - sqrtDiscriminant;
    float tRight = -dotProduct + sqrtDiscriminant;

    for (size_t i = 0; i < planeNo; ++i) {
        const float numerator = (planes[i].point - line.point) * planes[i].normal;
        const float denominator = line.direction * planes[i].normal;

        if (sqr(denominator) <= RVO_EPSILON) {
            /* Lines line is (almost) parallel to plane i. */
            if (numerator > 0.0f) {
                return false;
            } else {
                continue;
			}
        }

        const float t = numerator / denominator;

        if (denominator >= 0.0f) {
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
        if (optVelocity * line.direction > 0.0f) {
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

bool linearProgram2(const std::vector<Plane> &planes, size_t planeNo, float radius, const Vector3 &optVelocity, bool directionOpt, Vector3 &result) {
    const float planeDist = planes[planeNo].point * planes[planeNo].normal;
    const float planeDistSq = sqr(planeDist);
    const float radiusSq = sqr(radius);

    if (planeDistSq > radiusSq) {
        /* Max speed sphere fully invalidates plane planeNo. */
        return false;
    }

    const float planeRadiusSq = radiusSq - planeDistSq;

    const Vector3 planeCenter = planeDist * planes[planeNo].normal;

    if (directionOpt) {
        /* Project direction optVelocity on plane planeNo. */
        const Vector3 planeOptVelocity = optVelocity - (optVelocity * planes[planeNo].normal) * planes[planeNo].normal;
        const float planeOptVelocityLengthSq = absSq(planeOptVelocity);

        if (planeOptVelocityLengthSq <= RVO_EPSILON) {
            result = planeCenter;
        } else {
            result = planeCenter + std::sqrt(planeRadiusSq / planeOptVelocityLengthSq) * planeOptVelocity;
		}
    } else {
        /* Project point optVelocity on plane planeNo. */
        result = optVelocity + ((planes[planeNo].point - optVelocity) * planes[planeNo].normal) * planes[planeNo].normal;

        /* If outside planeCircle, project on planeCircle. */
        if (absSq(result) > radiusSq) {
            const Vector3 planeResult = result - planeCenter;
            const float planeResultLengthSq = absSq(planeResult);
            result = planeCenter + std::sqrt(planeRadiusSq / planeResultLengthSq) * planeResult;
		}
    }

    for (size_t i = 0; i < planeNo; ++i) {
        if (planes[i].normal * (planes[i].point - result) > 0.0f) {
            /* Result does not satisfy constraint i. Compute new optimal result. */
            /* Compute intersection line of plane i and plane planeNo. */
            Vector3 crossProduct = cross(planes[i].normal, planes[planeNo].normal);

            if (absSq(crossProduct) <= RVO_EPSILON) {
                /* Planes planeNo and i are (almost) parallel, and plane i fully invalidates plane planeNo. */
                return false;
            }

            Line line;
            line.direction = normalize(crossProduct);
            const Vector3 lineNormal = cross(line.direction, planes[planeNo].normal);
            line.point = planes[planeNo].point + (((planes[i].point - planes[planeNo].point) * planes[i].normal) / (lineNormal * planes[i].normal)) * lineNormal;

            if (!linearProgram1(planes, i, line, radius, optVelocity, directionOpt, result)) {
                return false;
			}
		}
	}

    return true;
}

size_t linearProgram3(const std::vector<Plane> &planes, float radius, const Vector3 &optVelocity, bool directionOpt, Vector3 &result) {
    if (directionOpt) {
        /* Optimize direction. Note that the optimization velocity is of unit length in this case. */
        result = optVelocity * radius;
    } else if (absSq(optVelocity) > sqr(radius)) {
        /* Optimize closest point and outside circle. */
        result = normalize(optVelocity) * radius;
    } else {
        /* Optimize closest point and inside circle. */
        result = optVelocity;
    }

    for (size_t i = 0; i < planes.size(); ++i) {
        if (planes[i].normal * (planes[i].point - result) > 0.0f) {
            /* Result does not satisfy constraint i. Compute new optimal result. */
            const Vector3 tempResult = result;

            if (!linearProgram2(planes, i, radius, optVelocity, directionOpt, result)) {
                result = tempResult;
                return i;
			}
		}
	}

    return planes.size();
}

void linearProgram4(const std::vector<Plane> &planes, size_t beginPlane, float radius, Vector3 &result) {
    float distance = 0.0f;

    for (size_t i = beginPlane; i < planes.size(); ++i) {
        if (planes[i].normal * (planes[i].point - result) > distance) {
            /* Result does not satisfy constraint of plane i. */
            std::vector<Plane> projPlanes;

            for (size_t j = 0; j < i; ++j) {
                Plane plane;

                const Vector3 crossProduct = cross(planes[j].normal, planes[i].normal);

                if (absSq(crossProduct) <= RVO_EPSILON) {
                    /* Plane i and plane j are (almost) parallel. */
                    if (planes[i].normal * planes[j].normal > 0.0f) {
                        /* Plane i and plane j point in the same direction. */
                        continue;
                    } else {
                        /* Plane i and plane j point in opposite direction. */
                        plane.point = 0.5f * (planes[i].point + planes[j].point);
                    }
                } else {
                    /* Plane.point is point on line of intersection between plane i and plane j. */
                    const Vector3 lineNormal = cross(crossProduct, planes[i].normal);
                    plane.point = planes[i].point + (((planes[j].point - planes[i].point) * planes[j].normal) / (lineNormal * planes[j].normal)) * lineNormal;
				}

                plane.normal = normalize(planes[j].normal - planes[i].normal);
                projPlanes.push_back(plane);
            }

            const Vector3 tempResult = result;

            if (linearProgram3(projPlanes, radius, planes[i].normal, true, result) < projPlanes.size()) {
                /* This should in principle not happen.  The result is by definition already in the feasible region of this linear program. If it fails, it is due to small floating point error, and the current result is kept. */
                result = tempResult;
			}

            distance = planes[i].normal * (planes[i].point - result);
        }
	}
}
} // namespace RVO
