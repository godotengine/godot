/*
 * KdTree2d.h
 * RVO2 Library
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

#ifndef RVO2D_KD_TREE_H_
#define RVO2D_KD_TREE_H_

/**
 * \file       KdTree2d.h
 * \brief      Contains the KdTree class.
 */

#include "Definitions.h"

namespace RVO2D {
	/**
	 * \brief      Defines <i>k</i>d-trees for agents and static obstacles in the
	 *             simulation.
	 */
	class KdTree2D {
	public:
		/**
		 * \brief      Defines an agent <i>k</i>d-tree node.
		 */
		class AgentTreeNode {
		public:
			/**
			 * \brief      The beginning node number.
			 */
			size_t begin;

			/**
			 * \brief      The ending node number.
			 */
			size_t end;

			/**
			 * \brief      The left node number.
			 */
			size_t left;

			/**
			 * \brief      The maximum x-coordinate.
			 */
			float maxX;

			/**
			 * \brief      The maximum y-coordinate.
			 */
			float maxY;

			/**
			 * \brief      The minimum x-coordinate.
			 */
			float minX;

			/**
			 * \brief      The minimum y-coordinate.
			 */
			float minY;

			/**
			 * \brief      The right node number.
			 */
			size_t right;
		};

		/**
		 * \brief      Defines an obstacle <i>k</i>d-tree node.
		 */
		class ObstacleTreeNode {
		public:
			/**
			 * \brief      The left obstacle tree node.
			 */
			ObstacleTreeNode *left;

			/**
			 * \brief      The obstacle number.
			 */
			const Obstacle2D *obstacle;

			/**
			 * \brief      The right obstacle tree node.
			 */
			ObstacleTreeNode *right;
		};

		/**
		 * \brief      Constructs a <i>k</i>d-tree instance.
		 * \param      sim             The simulator instance.
		 */
		explicit KdTree2D(RVOSimulator2D *sim);

		/**
		 * \brief      Destroys this kd-tree instance.
		 */
		~KdTree2D();

		/**
		 * \brief      Builds an agent <i>k</i>d-tree.
		 */
		void buildAgentTree(std::vector<Agent2D *> agents);

		void buildAgentTreeRecursive(size_t begin, size_t end, size_t node);

		/**
		 * \brief      Builds an obstacle <i>k</i>d-tree.
		 */
		void buildObstacleTree(std::vector<Obstacle2D *> obstacles);

		ObstacleTreeNode *buildObstacleTreeRecursive(const std::vector<Obstacle2D *> &
													 obstacles);

		/**
		 * \brief      Computes the agent neighbors of the specified agent.
		 * \param      agent           A pointer to the agent for which agent
		 *                             neighbors are to be computed.
		 * \param      rangeSq         The squared range around the agent.
		 */
		void computeAgentNeighbors(Agent2D *agent, float &rangeSq) const;

		/**
		 * \brief      Computes the obstacle neighbors of the specified agent.
		 * \param      agent           A pointer to the agent for which obstacle
		 *                             neighbors are to be computed.
		 * \param      rangeSq         The squared range around the agent.
		 */
		void computeObstacleNeighbors(Agent2D *agent, float rangeSq) const;

		/**
		 * \brief      Deletes the specified obstacle tree node.
		 * \param      node            A pointer to the obstacle tree node to be
		 *                             deleted.
		 */
		void deleteObstacleTree(ObstacleTreeNode *node);

		void queryAgentTreeRecursive(Agent2D *agent, float &rangeSq,
									 size_t node) const;

		void queryObstacleTreeRecursive(Agent2D *agent, float rangeSq,
										const ObstacleTreeNode *node) const;

		/**
		 * \brief      Queries the visibility between two points within a
		 *             specified radius.
		 * \param      q1              The first point between which visibility is
		 *                             to be tested.
		 * \param      q2              The second point between which visibility is
		 *                             to be tested.
		 * \param      radius          The radius within which visibility is to be
		 *                             tested.
		 * \return     True if q1 and q2 are mutually visible within the radius;
		 *             false otherwise.
		 */
		bool queryVisibility(const Vector2 &q1, const Vector2 &q2,
							 float radius) const;

		bool queryVisibilityRecursive(const Vector2 &q1, const Vector2 &q2,
									  float radius,
									  const ObstacleTreeNode *node) const;

		std::vector<Agent2D *> agents_;
		std::vector<AgentTreeNode> agentTree_;
		ObstacleTreeNode *obstacleTree_;
		RVOSimulator2D *sim_;

		static const size_t MAX_LEAF_SIZE = 10;

		friend class Agent2D;
		friend class RVOSimulator2D;
	};
}

#endif /* RVO2D_KD_TREE_H_ */
