/*
 * KdTree3d.h
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

#ifndef RVO3D_KD_TREE_H_
#define RVO3D_KD_TREE_H_

/**
 * @file  KdTree3d.h
 * @brief Contains the KdTree3D class.
 */

#include <cstddef>
#include <vector>

namespace RVO3D {
class Agent3D;
class RVOSimulator3D;

/**
 * @brief Defines a k-D tree for agents in the simulation.
 */
class KdTree3D {
 public:
  class AgentTreeNode;

  /**
   * @brief     Constructs a k-D tree instance.
   * @param[in] sim The simulator instance.
   */
  explicit KdTree3D(RVOSimulator3D *sim);

  /**
   * @brief Destroys this k-D tree instance.
   */
  ~KdTree3D();

  /**
   * @brief Builds an agent k-D tree.
   */
  void buildAgentTree(std::vector<Agent3D *> agents);

  /**
   * @brief     Recursive function to build a k-D tree.
   * @param[in] begin The beginning k-D tree node.
   * @param[in] end   The ending k-D tree node.
   * @param[in] node  The current k-D tree node.
   */
  void buildAgentTreeRecursive(std::size_t begin, std::size_t end,
                               std::size_t node);

  /**
   * @brief     Computes the agent neighbors of the specified agent.
   * @param[in] agent   A pointer to the agent for which agent neighbors are to
   *                    be computed.
   * @param[in] rangeSq The squared range around the agent.
   */
  void computeAgentNeighbors(Agent3D *agent, float rangeSq) const;

  /**
   * @brief         Recursive function to compute the neighbors of the specified
   *                agent.
   * @param[in]     agent   A pointer to the agent for which neighbors are to be
   *                        computed.
   * @param[in,out] rangeSq The squared range around the agent.
   * @param[in]     node    The current k-D tree node.
   */

  void queryAgentTreeRecursive(Agent3D *agent,
                               float &rangeSq, /* NOLINT(runtime/references) */
                               std::size_t node) const;

  /* Not implemented. */
  KdTree3D(const KdTree3D &other);

  /* Not implemented. */
  KdTree3D &operator=(const KdTree3D &other);

  std::vector<Agent3D *> agents_;
  std::vector<AgentTreeNode> agentTree_;
  RVOSimulator3D *sim_;

  friend class Agent3D;
  friend class RVOSimulator3D;
};
} /* namespace RVO3D */

#endif /* RVO3D_KD_TREE_H_ */
