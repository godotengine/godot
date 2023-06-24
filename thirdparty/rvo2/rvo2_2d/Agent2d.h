/*
 * Agent2d.h
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

#ifndef RVO2D_AGENT_H_
#define RVO2D_AGENT_H_

/**
 * @file  Agent2d.h
 * @brief Declares the Agent2D class.
 */

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "Line.h"
#include "Vector2.h"

namespace RVO2D {
class KdTree2D;
class Obstacle2D;

/**
 * @brief Defines an agent in the simulation.
 */
class Agent2D {
 public:
  /**
   * @brief Constructs an agent instance.
   */
  Agent2D();

  /**
   * @brief Destroys this agent instance.
   */
  ~Agent2D();

  /**
   * @brief     Computes the neighbors of this agent.
   * @param[in] kdTree A pointer to the k-D trees for agents and static
   *                   obstacles in the simulation.
   */
  void computeNeighbors(const KdTree2D *kdTree);

  /**
   * @brief     Computes the new velocity of this agent.
   * @param[in] timeStep The time step of the simulation.
   */
  void computeNewVelocity(float timeStep);

  /**
   * @brief          Inserts an agent neighbor into the set of neighbors of this
   *                 agent.
   * @param[in]      agent   A pointer to the agent to be inserted.
   * @param[in, out] rangeSq The squared range around this agent.
   */
  void insertAgentNeighbor(const Agent2D *agent,
                           float &rangeSq); /* NOLINT(runtime/references) */

  /**
   * @brief          Inserts a static obstacle neighbor into the set of
   *                 neighbors of this agent.
   * @param[in]      obstacle The number of the static obstacle to be inserted.
   * @param[in, out] rangeSq  The squared range around this agent.
   */
  void insertObstacleNeighbor(const Obstacle2D *obstacle, float rangeSq);

  /**
   * @brief     Updates the two-dimensional position and two-dimensional
   *            velocity of this agent.
   * @param[in] timeStep The time step of the simulation.
   */
  void update(float timeStep);

  /* Not implemented. */
  Agent2D(const Agent2D &other);

  /* Not implemented. */
  Agent2D &operator=(const Agent2D &other);

  std::vector<std::pair<float, const Agent2D *> > agentNeighbors_;
  std::vector<std::pair<float, const Obstacle2D *> > obstacleNeighbors_;
  std::vector<Line> orcaLines_;
  Vector2 newVelocity_;
  Vector2 position_;
  Vector2 prefVelocity_;
  Vector2 velocity_;
  std::size_t id_;
  std::size_t maxNeighbors_;
  float maxSpeed_;
  float neighborDist_;
  float radius_;
  float timeHorizon_;
  float timeHorizonObst_;
	float height_ = 0.0;
	float elevation_ = 0.0;
	uint32_t avoidance_layers_ = 1;
	uint32_t avoidance_mask_ = 1;
	float avoidance_priority_ = 1.0;

  friend class KdTree2D;
  friend class RVOSimulator2D;
};
} /* namespace RVO */

#endif /* RVO2D_AGENT_H_ */
