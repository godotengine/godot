/*
 * KdTree2d.cpp
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
 * @file  KdTree2d.cpp
 * @brief Defines the KdTree2D class.
 */

#include "KdTree2d.h"

#include <algorithm>
#include <utility>

#include "Agent2d.h"
#include "Obstacle2d.h"
#include "RVOSimulator2d.h"
#include "Vector2.h"

namespace RVO2D {
namespace {
/**
 * @relates KdTree2D
 * @brief   The maximum k-D tree node leaf size.
 */
const std::size_t RVO_MAX_LEAF_SIZE = 10U;
} /* namespace */

/**
 * @brief Defines an agent k-D tree node.
 */
class KdTree2D::AgentTreeNode {
 public:
  /**
   * @brief Constructs an agent k-D tree node instance.
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
   * @brief The maximum x-coordinate.
   */
  float maxX;

  /**
   * @brief The maximum y-coordinate.
   */
  float maxY;

  /**
   * @brief The minimum x-coordinate.
   */
  float minX;

  /**
   * @brief The minimum y-coordinate.
   */
  float minY;
};

KdTree2D::AgentTreeNode::AgentTreeNode()
    : begin(0U),
      end(0U),
      left(0U),
      right(0U),
      maxX(0.0F),
      maxY(0.0F),
      minX(0.0F),
      minY(0.0F) {}

/**
 * @brief Defines an obstacle k-D tree node.
 */
class KdTree2D::ObstacleTreeNode {
 public:
  /**
   * @brief Constructs an obstacle k-D tree node instance.
   */
  ObstacleTreeNode();

  /**
   * @brief Destroys this obstacle k-D tree node instance.
   */
  ~ObstacleTreeNode();

  /**
   * @brief The obstacle number.
   */
  const Obstacle2D *obstacle;

  /**
   * @brief The left obstacle tree node.
   */
  ObstacleTreeNode *left;

  /**
   * @brief The right obstacle tree node.
   */
  ObstacleTreeNode *right;

 private:
  /* Not implemented. */
  ObstacleTreeNode(const ObstacleTreeNode &other);

  /* Not implemented. */
  ObstacleTreeNode &operator=(const ObstacleTreeNode &other);
};

KdTree2D::ObstacleTreeNode::ObstacleTreeNode()
    : obstacle(NULL), left(NULL), right(NULL) {}

KdTree2D::ObstacleTreeNode::~ObstacleTreeNode() {}

KdTree2D::KdTree2D(RVOSimulator2D *simulator)
    : obstacleTree_(NULL), simulator_(simulator) {}

KdTree2D::~KdTree2D() { deleteObstacleTree(obstacleTree_); }

void KdTree2D::buildAgentTree(std::vector<Agent2D *> agents) {
	agents_.swap(agents);

	if (!agents_.empty()) {
		agentTree_.resize(2 * agents_.size() - 1);
		buildAgentTreeRecursive(0, agents_.size(), 0);
	}
}

void KdTree2D::buildAgentTreeRecursive(std::size_t begin, std::size_t end,
                                     std::size_t node) {
  agentTree_[node].begin = begin;
  agentTree_[node].end = end;
  agentTree_[node].minX = agentTree_[node].maxX = agents_[begin]->position_.x();
  agentTree_[node].minY = agentTree_[node].maxY = agents_[begin]->position_.y();

  for (std::size_t i = begin + 1U; i < end; ++i) {
    agentTree_[node].maxX =
        std::max(agentTree_[node].maxX, agents_[i]->position_.x());
    agentTree_[node].minX =
        std::min(agentTree_[node].minX, agents_[i]->position_.x());
    agentTree_[node].maxY =
        std::max(agentTree_[node].maxY, agents_[i]->position_.y());
    agentTree_[node].minY =
        std::min(agentTree_[node].minY, agents_[i]->position_.y());
  }

  if (end - begin > RVO_MAX_LEAF_SIZE) {
    /* No leaf node. */
    const bool isVertical = agentTree_[node].maxX - agentTree_[node].minX >
                            agentTree_[node].maxY - agentTree_[node].minY;
    const float splitValue =
        0.5F * (isVertical ? agentTree_[node].maxX + agentTree_[node].minX
                           : agentTree_[node].maxY + agentTree_[node].minY);

    std::size_t left = begin;
    std::size_t right = end;

    while (left < right) {
      while (left < right &&
             (isVertical ? agents_[left]->position_.x()
                         : agents_[left]->position_.y()) < splitValue) {
        ++left;
      }

      while (right > left &&
             (isVertical ? agents_[right - 1U]->position_.x()
                         : agents_[right - 1U]->position_.y()) >= splitValue) {
        --right;
      }

      if (left < right) {
        std::swap(agents_[left], agents_[right - 1U]);
        ++left;
        --right;
      }
    }

    if (left == begin) {
      ++left;
      ++right;
    }

    agentTree_[node].left = node + 1U;
    agentTree_[node].right = node + 2U * (left - begin);

    buildAgentTreeRecursive(begin, left, agentTree_[node].left);
    buildAgentTreeRecursive(left, end, agentTree_[node].right);
  }
}

void KdTree2D::buildObstacleTree(std::vector<Obstacle2D *> obstacles) {
	deleteObstacleTree(obstacleTree_);

	obstacleTree_ = buildObstacleTreeRecursive(obstacles);
}

KdTree2D::ObstacleTreeNode *KdTree2D::buildObstacleTreeRecursive(
    const std::vector<Obstacle2D *> &obstacles) {
  if (!obstacles.empty()) {
    ObstacleTreeNode *const node = new ObstacleTreeNode();

    std::size_t optimalSplit = 0U;
    std::size_t minLeft = obstacles.size();
    std::size_t minRight = obstacles.size();

    for (std::size_t i = 0U; i < obstacles.size(); ++i) {
      std::size_t leftSize = 0U;
      std::size_t rightSize = 0U;

      const Obstacle2D *const obstacleI1 = obstacles[i];
      const Obstacle2D *const obstacleI2 = obstacleI1->next_;

      /* Compute optimal split node. */
      for (std::size_t j = 0U; j < obstacles.size(); ++j) {
        if (i != j) {
          const Obstacle2D *const obstacleJ1 = obstacles[j];
          const Obstacle2D *const obstacleJ2 = obstacleJ1->next_;

          const float j1LeftOfI = leftOf(obstacleI1->point_, obstacleI2->point_,
                                         obstacleJ1->point_);
          const float j2LeftOfI = leftOf(obstacleI1->point_, obstacleI2->point_,
                                         obstacleJ2->point_);

          if (j1LeftOfI >= -RVO2D_EPSILON && j2LeftOfI >= -RVO2D_EPSILON) {
            ++leftSize;
          } else if (j1LeftOfI <= RVO2D_EPSILON && j2LeftOfI <= RVO2D_EPSILON) {
            ++rightSize;
          } else {
            ++leftSize;
            ++rightSize;
          }

          if (std::make_pair(std::max(leftSize, rightSize),
                             std::min(leftSize, rightSize)) >=
              std::make_pair(std::max(minLeft, minRight),
                             std::min(minLeft, minRight))) {
            break;
          }
        }
      }

      if (std::make_pair(std::max(leftSize, rightSize),
                         std::min(leftSize, rightSize)) <
          std::make_pair(std::max(minLeft, minRight),
                         std::min(minLeft, minRight))) {
        minLeft = leftSize;
        minRight = rightSize;
        optimalSplit = i;
      }
    }

    /* Build split node. */
    std::vector<Obstacle2D *> leftObstacles(minLeft);
    std::vector<Obstacle2D *> rightObstacles(minRight);

    std::size_t leftCounter = 0U;
    std::size_t rightCounter = 0U;
    const std::size_t i = optimalSplit;

    const Obstacle2D *const obstacleI1 = obstacles[i];
    const Obstacle2D *const obstacleI2 = obstacleI1->next_;

    for (std::size_t j = 0U; j < obstacles.size(); ++j) {
      if (i != j) {
        Obstacle2D *const obstacleJ1 = obstacles[j];
        Obstacle2D *const obstacleJ2 = obstacleJ1->next_;

        const float j1LeftOfI =
            leftOf(obstacleI1->point_, obstacleI2->point_, obstacleJ1->point_);
        const float j2LeftOfI =
            leftOf(obstacleI1->point_, obstacleI2->point_, obstacleJ2->point_);

        if (j1LeftOfI >= -RVO2D_EPSILON && j2LeftOfI >= -RVO2D_EPSILON) {
          leftObstacles[leftCounter++] = obstacles[j];
        } else if (j1LeftOfI <= RVO2D_EPSILON && j2LeftOfI <= RVO2D_EPSILON) {
          rightObstacles[rightCounter++] = obstacles[j];
        } else {
          /* Split obstacle j. */
          const float t = det(obstacleI2->point_ - obstacleI1->point_,
                              obstacleJ1->point_ - obstacleI1->point_) /
                          det(obstacleI2->point_ - obstacleI1->point_,
                              obstacleJ1->point_ - obstacleJ2->point_);

          const Vector2 splitPoint =
              obstacleJ1->point_ +
              t * (obstacleJ2->point_ - obstacleJ1->point_);

          Obstacle2D *const newObstacle = new Obstacle2D();
          newObstacle->direction_ = obstacleJ1->direction_;
          newObstacle->point_ = splitPoint;
          newObstacle->next_ = obstacleJ2;
          newObstacle->previous_ = obstacleJ1;
          newObstacle->id_ = simulator_->obstacles_.size();
          newObstacle->isConvex_ = true;
          simulator_->obstacles_.push_back(newObstacle);

          obstacleJ1->next_ = newObstacle;
          obstacleJ2->previous_ = newObstacle;

          if (j1LeftOfI > 0.0F) {
            leftObstacles[leftCounter++] = obstacleJ1;
            rightObstacles[rightCounter++] = newObstacle;
          } else {
            rightObstacles[rightCounter++] = obstacleJ1;
            leftObstacles[leftCounter++] = newObstacle;
          }
        }
      }
    }

    node->obstacle = obstacleI1;
    node->left = buildObstacleTreeRecursive(leftObstacles);
    node->right = buildObstacleTreeRecursive(rightObstacles);

    return node;
  }

  return NULL;
}

void KdTree2D::computeAgentNeighbors(Agent2D *agent, float &rangeSq) const {
  queryAgentTreeRecursive(agent, rangeSq, 0U);
}

void KdTree2D::computeObstacleNeighbors(Agent2D *agent, float rangeSq) const {
  queryObstacleTreeRecursive(agent, rangeSq, obstacleTree_);
}

void KdTree2D::deleteObstacleTree(ObstacleTreeNode *node) {
  if (node != NULL) {
    deleteObstacleTree(node->left);
    deleteObstacleTree(node->right);
    delete node;
  }
}

void KdTree2D::queryAgentTreeRecursive(Agent2D *agent, float &rangeSq,
                                     std::size_t node) const {
  if (agentTree_[node].end - agentTree_[node].begin <= RVO_MAX_LEAF_SIZE) {
    for (std::size_t i = agentTree_[node].begin; i < agentTree_[node].end;
         ++i) {
      agent->insertAgentNeighbor(agents_[i], rangeSq);
    }
  } else {
    const float distLeftMinX = std::max(
        0.0F, agentTree_[agentTree_[node].left].minX - agent->position_.x());
    const float distLeftMaxX = std::max(
        0.0F, agent->position_.x() - agentTree_[agentTree_[node].left].maxX);
    const float distLeftMinY = std::max(
        0.0F, agentTree_[agentTree_[node].left].minY - agent->position_.y());
    const float distLeftMaxY = std::max(
        0.0F, agent->position_.y() - agentTree_[agentTree_[node].left].maxY);

    const float distSqLeft =
        distLeftMinX * distLeftMinX + distLeftMaxX * distLeftMaxX +
        distLeftMinY * distLeftMinY + distLeftMaxY * distLeftMaxY;

    const float distRightMinX = std::max(
        0.0F, agentTree_[agentTree_[node].right].minX - agent->position_.x());
    const float distRightMaxX = std::max(
        0.0F, agent->position_.x() - agentTree_[agentTree_[node].right].maxX);
    const float distRightMinY = std::max(
        0.0F, agentTree_[agentTree_[node].right].minY - agent->position_.y());
    const float distRightMaxY = std::max(
        0.0F, agent->position_.y() - agentTree_[agentTree_[node].right].maxY);

    const float distSqRight =
        distRightMinX * distRightMinX + distRightMaxX * distRightMaxX +
        distRightMinY * distRightMinY + distRightMaxY * distRightMaxY;

    if (distSqLeft < distSqRight) {
      if (distSqLeft < rangeSq) {
        queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].left);

        if (distSqRight < rangeSq) {
          queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].right);
        }
      }
    } else if (distSqRight < rangeSq) {
      queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].right);

      if (distSqLeft < rangeSq) {
        queryAgentTreeRecursive(agent, rangeSq, agentTree_[node].left);
      }
    }
  }
}

void KdTree2D::queryObstacleTreeRecursive(Agent2D *agent, float rangeSq,
                                        const ObstacleTreeNode *node) const {
  if (node != NULL) {
    const Obstacle2D *const obstacle1 = node->obstacle;
    const Obstacle2D *const obstacle2 = obstacle1->next_;

    const float agentLeftOfLine =
        leftOf(obstacle1->point_, obstacle2->point_, agent->position_);

    queryObstacleTreeRecursive(
        agent, rangeSq, agentLeftOfLine >= 0.0F ? node->left : node->right);

    const float distSqLine = agentLeftOfLine * agentLeftOfLine /
                             absSq(obstacle2->point_ - obstacle1->point_);

    if (distSqLine < rangeSq) {
      if (agentLeftOfLine < 0.0F) {
        /* Try obstacle at this node only if agent is on right side of obstacle
         * and can see obstacle. */
        agent->insertObstacleNeighbor(node->obstacle, rangeSq);
      }

      /* Try other side of line. */
      queryObstacleTreeRecursive(
          agent, rangeSq, agentLeftOfLine >= 0.0F ? node->right : node->left);
    }
  }
}

bool KdTree2D::queryVisibility(const Vector2 &vector1, const Vector2 &vector2,
                             float radius) const {
  return queryVisibilityRecursive(vector1, vector2, radius, obstacleTree_);
}

bool KdTree2D::queryVisibilityRecursive(const Vector2 &vector1,
                                      const Vector2 &vector2, float radius,
                                      const ObstacleTreeNode *node) const {
  if (node != NULL) {
    const Obstacle2D *const obstacle1 = node->obstacle;
    const Obstacle2D *const obstacle2 = obstacle1->next_;

    const float q1LeftOfI =
        leftOf(obstacle1->point_, obstacle2->point_, vector1);
    const float q2LeftOfI =
        leftOf(obstacle1->point_, obstacle2->point_, vector2);
    const float invLengthI =
        1.0F / absSq(obstacle2->point_ - obstacle1->point_);

    if (q1LeftOfI >= 0.0F && q2LeftOfI >= 0.0F) {
      return queryVisibilityRecursive(vector1, vector2, radius, node->left) &&
             ((q1LeftOfI * q1LeftOfI * invLengthI >= radius * radius &&
               q2LeftOfI * q2LeftOfI * invLengthI >= radius * radius) ||
              queryVisibilityRecursive(vector1, vector2, radius, node->right));
    }

    if (q1LeftOfI <= 0.0F && q2LeftOfI <= 0.0F) {
      return queryVisibilityRecursive(vector1, vector2, radius, node->right) &&
             ((q1LeftOfI * q1LeftOfI * invLengthI >= radius * radius &&
               q2LeftOfI * q2LeftOfI * invLengthI >= radius * radius) ||
              queryVisibilityRecursive(vector1, vector2, radius, node->left));
    }

    if (q1LeftOfI >= 0.0F && q2LeftOfI <= 0.0F) {
      /* One can see through obstacle from left to right. */
      return queryVisibilityRecursive(vector1, vector2, radius, node->left) &&
             queryVisibilityRecursive(vector1, vector2, radius, node->right);
    }

    const float point1LeftOfQ = leftOf(vector1, vector2, obstacle1->point_);
    const float point2LeftOfQ = leftOf(vector1, vector2, obstacle2->point_);
    const float invLengthQ = 1.0F / absSq(vector2 - vector1);

    return point1LeftOfQ * point2LeftOfQ >= 0.0F &&
           point1LeftOfQ * point1LeftOfQ * invLengthQ > radius * radius &&
           point2LeftOfQ * point2LeftOfQ * invLengthQ > radius * radius &&
           queryVisibilityRecursive(vector1, vector2, radius, node->left) &&
           queryVisibilityRecursive(vector1, vector2, radius, node->right);
  }

  return true;
}
} /* namespace RVO2D */
