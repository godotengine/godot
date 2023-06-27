/*
 * KdTree3d.cc
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

#include "KdTree3d.h"

#include <algorithm>
#include <utility>

#include "Agent3d.h"
#include "RVOSimulator3d.h"
#include "Vector3.h"

namespace RVO3D {
namespace {
/**
 * @brief The maximum size of a k-D leaf node.
 */
const std::size_t RVO3D_MAX_LEAF_SIZE = 10U;
} /* namespace */

/**
 * @brief Defines an agent k-D tree node.
 */
class KdTree3D::AgentTreeNode {
 public:
  /**
   * @brief Constructs an agent k-D tree node.
   */
  AgentTreeNode();

  /**
   * @brief The beginning node number.
   */
  std::size_t begin;

  /**
   * @brief The ending node number.
   */
  std::size_t end;

  /**
   * @brief The left node number.
   */
  std::size_t left;

  /**
   * @brief The right node number.
   */
  std::size_t right;

  /**
   * @brief The maximum coordinates.
   */
  Vector3 maxCoord;

  /**
   * @brief The minimum coordinates.
   */
  Vector3 minCoord;
};

KdTree3D::AgentTreeNode::AgentTreeNode()
    : begin(0U), end(0U), left(0U), right(0U) {}

KdTree3D::KdTree3D(RVOSimulator3D *sim) : sim_(sim) {}

KdTree3D::~KdTree3D() {}

void KdTree3D::buildAgentTree(std::vector<Agent3D *> agents) {
  agents_.swap(agents_);

  if (!agents_.empty()) {
    agentTree_.resize(2U * agents_.size() - 1U);
    buildAgentTreeRecursive(0U, agents_.size(), 0U);
  }
}

void KdTree3D::buildAgentTreeRecursive(std::size_t begin, std::size_t end,
                                     std::size_t node) {
  agentTree_[node].begin = begin;
  agentTree_[node].end = end;
  agentTree_[node].minCoord = agents_[begin]->position_;
  agentTree_[node].maxCoord = agents_[begin]->position_;

  for (std::size_t i = begin + 1U; i < end; ++i) {
    agentTree_[node].maxCoord[0] =
        std::max(agentTree_[node].maxCoord[0], agents_[i]->position_.x());
    agentTree_[node].minCoord[0] =
        std::min(agentTree_[node].minCoord[0], agents_[i]->position_.x());
    agentTree_[node].maxCoord[1] =
        std::max(agentTree_[node].maxCoord[1], agents_[i]->position_.y());
    agentTree_[node].minCoord[1] =
        std::min(agentTree_[node].minCoord[1], agents_[i]->position_.y());
    agentTree_[node].maxCoord[2] =
        std::max(agentTree_[node].maxCoord[2], agents_[i]->position_.z());
    agentTree_[node].minCoord[2] =
        std::min(agentTree_[node].minCoord[2], agents_[i]->position_.z());
  }

  if (end - begin > RVO3D_MAX_LEAF_SIZE) {
    /* No leaf node. */
    std::size_t coord = 0U;

    if (agentTree_[node].maxCoord[0] - agentTree_[node].minCoord[0] >
            agentTree_[node].maxCoord[1] - agentTree_[node].minCoord[1] &&
        agentTree_[node].maxCoord[0] - agentTree_[node].minCoord[0] >
            agentTree_[node].maxCoord[2] - agentTree_[node].minCoord[2]) {
      coord = 0U;
    } else if (agentTree_[node].maxCoord[1] - agentTree_[node].minCoord[1] >
               agentTree_[node].maxCoord[2] - agentTree_[node].minCoord[2]) {
      coord = 1U;
    } else {
      coord = 2U;
    }

    const float splitValue = 0.5F * (agentTree_[node].maxCoord[coord] +
                                     agentTree_[node].minCoord[coord]);

    std::size_t left = begin;

    std::size_t right = end;

    while (left < right) {
      while (left < right && agents_[left]->position_[coord] < splitValue) {
        ++left;
      }

      while (right > left &&
             agents_[right - 1U]->position_[coord] >= splitValue) {
        --right;
      }

      if (left < right) {
        std::swap(agents_[left], agents_[right - 1U]);
        ++left;
        --right;
      }
    }

    std::size_t leftSize = left - begin;

    if (leftSize == 0U) {
      ++leftSize;
      ++left;
    }

    agentTree_[node].left = node + 1U;
    agentTree_[node].right = node + 2U * leftSize;

    buildAgentTreeRecursive(begin, left, agentTree_[node].left);
    buildAgentTreeRecursive(left, end, agentTree_[node].right);
  }
}

void KdTree3D::computeAgentNeighbors(Agent3D *agent, float rangeSq) const {
  queryAgentTreeRecursive(agent, rangeSq, 0U);
}

void KdTree3D::queryAgentTreeRecursive(Agent3D *agent, float &rangeSq,
                                     std::size_t node) const {
  if (agentTree_[node].end - agentTree_[node].begin <= RVO3D_MAX_LEAF_SIZE) {
    for (std::size_t i = agentTree_[node].begin; i < agentTree_[node].end;
         ++i) {
      agent->insertAgentNeighbor(agents_[i], rangeSq);
    }
  } else {
    const float distSqLeftMinX =
        std::max(0.0F, agentTree_[agentTree_[node].left].minCoord[0] -
                           agent->position_.x());
    const float distSqLeftMaxX =
        std::max(0.0F, agent->position_.x() -
                           agentTree_[agentTree_[node].left].maxCoord[0]);
    const float distSqLeftMinY =
        std::max(0.0F, agentTree_[agentTree_[node].left].minCoord[1] -
                           agent->position_.y());
    const float distSqLeftMaxY =
        std::max(0.0F, agent->position_.y() -
                           agentTree_[agentTree_[node].left].maxCoord[1]);
    const float distSqLeftMinZ =
        std::max(0.0F, agentTree_[agentTree_[node].left].minCoord[2] -
                           agent->position_.z());
    const float distSqLeftMaxZ =
        std::max(0.0F, agent->position_.z() -
                           agentTree_[agentTree_[node].left].maxCoord[2]);

    const float distSqLeft =
        distSqLeftMinX * distSqLeftMinX + distSqLeftMaxX * distSqLeftMaxX +
        distSqLeftMinY * distSqLeftMinY + distSqLeftMaxY * distSqLeftMaxY +
        distSqLeftMinZ * distSqLeftMinZ + distSqLeftMaxZ * distSqLeftMaxZ;

    const float distSqRightMinX =
        std::max(0.0F, agentTree_[agentTree_[node].right].minCoord[0] -
                           agent->position_.x());
    const float distSqRightMaxX =
        std::max(0.0F, agent->position_.x() -
                           agentTree_[agentTree_[node].right].maxCoord[0]);
    const float distSqRightMinY =
        std::max(0.0F, agentTree_[agentTree_[node].right].minCoord[1] -
                           agent->position_.y());
    const float distSqRightMaxY =
        std::max(0.0F, agent->position_.y() -
                           agentTree_[agentTree_[node].right].maxCoord[1]);
    const float distSqRightMinZ =
        std::max(0.0F, agentTree_[agentTree_[node].right].minCoord[2] -
                           agent->position_.z());
    const float distSqRightMaxZ =
        std::max(0.0F, agent->position_.z() -
                           agentTree_[agentTree_[node].right].maxCoord[2]);

    const float distSqRight =
        distSqRightMinX * distSqRightMinX + distSqRightMaxX * distSqRightMaxX +
        distSqRightMinY * distSqRightMinY + distSqRightMaxY * distSqRightMaxY +
        distSqRightMinZ * distSqRightMinZ + distSqRightMaxZ * distSqRightMaxZ;

    if (distSqLeft < distSqRight) {
      if (distSqLeft < rangeSq) {
        queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].left);

        if (distSqRight < rangeSq) {
          queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].right);
        }
      }
    } else {
      if (distSqRight < rangeSq) {
        queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].right);

        if (distSqLeft < rangeSq) {
          queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].left);
        }
      }
    }
  }
}
} /* namespace RVO3D */
