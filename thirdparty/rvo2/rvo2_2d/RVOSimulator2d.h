/*
 * RVOSimulator2d.h
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

#ifndef RVO2D_RVO_SIMULATOR_H_
#define RVO2D_RVO_SIMULATOR_H_

/**
 * \file       RVOSimulator2d.h
 * \brief      Contains the RVOSimulator2D class.
 */

#include <cstddef>
#include <limits>
#include <vector>

#include "Vector2.h"

namespace RVO2D {
	/**
	 * \brief       Error value.
	 *
	 * A value equal to the largest unsigned integer that is returned in case
	 * of an error by functions in RVO2D::RVOSimulator2D.
	 */
	const size_t RVO2D_ERROR = std::numeric_limits<size_t>::max();

	/**
	 * \brief      Defines a directed line.
	 */
	class Line {
	public:
		/**
		 * \brief     A point on the directed line.
		 */
		Vector2 point;

		/**
		 * \brief     The direction of the directed line.
		 */
		Vector2 direction;
	};

	class Agent2D;
	class KdTree2D;
	class Obstacle2D;

	/**
	 * \brief      Defines the simulation.
	 *
	 * The main class of the library that contains all simulation functionality.
	 */
	class RVOSimulator2D {
	public:
		/**
		 * \brief      Constructs a simulator instance.
		 */
		RVOSimulator2D();

		/**
		 * \brief      Constructs a simulator instance and sets the default
		 *             properties for any new agent that is added.
		 * \param      timeStep        The time step of the simulation.
		 *                             Must be positive.
		 * \param      neighborDist    The default maximum distance (center point
		 *                             to center point) to other agents a new agent
		 *                             takes into account in the navigation. The
		 *                             larger this number, the longer he running
		 *                             time of the simulation. If the number is too
		 *                             low, the simulation will not be safe. Must be
		 *                             non-negative.
		 * \param      maxNeighbors    The default maximum number of other agents a
		 *                             new agent takes into account in the
		 *                             navigation. The larger this number, the
		 *                             longer the running time of the simulation.
		 *                             If the number is too low, the simulation
		 *                             will not be safe.
		 * \param      timeHorizon     The default minimal amount of time for which
		 *                             a new agent's velocities that are computed
		 *                             by the simulation are safe with respect to
		 *                             other agents. The larger this number, the
		 *                             sooner an agent will respond to the presence
		 *                             of other agents, but the less freedom the
		 *                             agent has in choosing its velocities.
		 *                             Must be positive.
		 * \param      timeHorizonObst The default minimal amount of time for which
		 *                             a new agent's velocities that are computed
		 *                             by the simulation are safe with respect to
		 *                             obstacles. The larger this number, the
		 *                             sooner an agent will respond to the presence
		 *                             of obstacles, but the less freedom the agent
		 *                             has in choosing its velocities.
		 *                             Must be positive.
		 * \param      radius          The default radius of a new agent.
		 *                             Must be non-negative.
		 * \param      maxSpeed        The default maximum speed of a new agent.
		 *                             Must be non-negative.
		 * \param      velocity        The default initial two-dimensional linear
		 *                             velocity of a new agent (optional).
		 */
		RVOSimulator2D(float timeStep, float neighborDist, size_t maxNeighbors,
					 float timeHorizon, float timeHorizonObst, float radius,
					 float maxSpeed, const Vector2 &velocity = Vector2());

		/**
		 * \brief      Destroys this simulator instance.
		 */
		~RVOSimulator2D();

		/**
		 * \brief      Adds a new agent with default properties to the
		 *             simulation.
		 * \param      position        The two-dimensional starting position of
		 *                             this agent.
		 * \return     The number of the agent, or RVO2D::RVO2D_ERROR when the agent
		 *             defaults have not been set.
		 */
		size_t addAgent(const Vector2 &position);

		/**
		 * \brief      Adds a new agent to the simulation.
		 * \param      position        The two-dimensional starting position of
		 *                             this agent.
		 * \param      neighborDist    The maximum distance (center point to
		 *                             center point) to other agents this agent
		 *                             takes into account in the navigation. The
		 *                             larger this number, the longer the running
		 *                             time of the simulation. If the number is too
		 *                             low, the simulation will not be safe.
		 *                             Must be non-negative.
		 * \param      maxNeighbors    The maximum number of other agents this
		 *                             agent takes into account in the navigation.
		 *                             The larger this number, the longer the
		 *                             running time of the simulation. If the
		 *                             number is too low, the simulation will not
		 *                             be safe.
		 * \param      timeHorizon     The minimal amount of time for which this
		 *                             agent's velocities that are computed by the
		 *                             simulation are safe with respect to other
		 *                             agents. The larger this number, the sooner
		 *                             this agent will respond to the presence of
		 *                             other agents, but the less freedom this
		 *                             agent has in choosing its velocities.
		 *                             Must be positive.
		 * \param      timeHorizonObst The minimal amount of time for which this
		 *                             agent's velocities that are computed by the
		 *                             simulation are safe with respect to
		 *                             obstacles. The larger this number, the
		 *                             sooner this agent will respond to the
		 *                             presence of obstacles, but the less freedom
		 *                             this agent has in choosing its velocities.
		 *                             Must be positive.
		 * \param      radius          The radius of this agent.
		 *                             Must be non-negative.
		 * \param      maxSpeed        The maximum speed of this agent.
		 *                             Must be non-negative.
		 * \param      velocity        The initial two-dimensional linear velocity
		 *                             of this agent (optional).
		 * \return     The number of the agent.
		 */
		size_t addAgent(const Vector2 &position, float neighborDist,
						size_t maxNeighbors, float timeHorizon,
						float timeHorizonObst, float radius, float maxSpeed,
						const Vector2 &velocity = Vector2());

		/**
		 * \brief      Adds a new obstacle to the simulation.
		 * \param      vertices        List of the vertices of the polygonal
		 *             obstacle in counterclockwise order.
		 * \return     The number of the first vertex of the obstacle,
		 *             or RVO2D::RVO2D_ERROR when the number of vertices is less than two.
		 * \note       To add a "negative" obstacle, e.g. a bounding polygon around
		 *             the environment, the vertices should be listed in clockwise
		 *             order.
		 */
		size_t addObstacle(const std::vector<Vector2> &vertices);

		/**
		 * \brief      Lets the simulator perform a simulation step and updates the
		 *             two-dimensional position and two-dimensional velocity of
		 *             each agent.
		 */
		void doStep();

		/**
		 * \brief      Returns the specified agent neighbor of the specified
		 *             agent.
		 * \param      agentNo         The number of the agent whose agent
		 *                             neighbor is to be retrieved.
		 * \param      neighborNo      The number of the agent neighbor to be
		 *                             retrieved.
		 * \return     The number of the neighboring agent.
		 */
		size_t getAgentAgentNeighbor(size_t agentNo, size_t neighborNo) const;

		/**
		 * \brief      Returns the maximum neighbor count of a specified agent.
		 * \param      agentNo         The number of the agent whose maximum
		 *                             neighbor count is to be retrieved.
		 * \return     The present maximum neighbor count of the agent.
		 */
		size_t getAgentMaxNeighbors(size_t agentNo) const;

		/**
		 * \brief      Returns the maximum speed of a specified agent.
		 * \param      agentNo         The number of the agent whose maximum speed
		 *                             is to be retrieved.
		 * \return     The present maximum speed of the agent.
		 */
		float getAgentMaxSpeed(size_t agentNo) const;

		/**
		 * \brief      Returns the maximum neighbor distance of a specified
		 *             agent.
		 * \param      agentNo         The number of the agent whose maximum
		 *                             neighbor distance is to be retrieved.
		 * \return     The present maximum neighbor distance of the agent.
		 */
		float getAgentNeighborDist(size_t agentNo) const;

		/**
		 * \brief      Returns the count of agent neighbors taken into account to
		 *             compute the current velocity for the specified agent.
		 * \param      agentNo         The number of the agent whose count of agent
		 *                             neighbors is to be retrieved.
		 * \return     The count of agent neighbors taken into account to compute
		 *             the current velocity for the specified agent.
		 */
		size_t getAgentNumAgentNeighbors(size_t agentNo) const;

		/**
		 * \brief      Returns the count of obstacle neighbors taken into account
		 *             to compute the current velocity for the specified agent.
		 * \param      agentNo         The number of the agent whose count of
		 *                             obstacle neighbors is to be retrieved.
		 * \return     The count of obstacle neighbors taken into account to
		 *             compute the current velocity for the specified agent.
		 */
		size_t getAgentNumObstacleNeighbors(size_t agentNo) const;


		/**
		 * \brief      Returns the count of ORCA constraints used to compute
		 *             the current velocity for the specified agent.
		 * \param      agentNo         The number of the agent whose count of ORCA
		 *                             constraints is to be retrieved.
		 * \return     The count of ORCA constraints used to compute the current
		 *             velocity for the specified agent.
		 */
		size_t getAgentNumORCALines(size_t agentNo) const;

		/**
		 * \brief      Returns the specified obstacle neighbor of the specified
		 *             agent.
		 * \param      agentNo         The number of the agent whose obstacle
		 *                             neighbor is to be retrieved.
		 * \param      neighborNo      The number of the obstacle neighbor to be
		 *                             retrieved.
		 * \return     The number of the first vertex of the neighboring obstacle
		 *             edge.
		 */
		size_t getAgentObstacleNeighbor(size_t agentNo, size_t neighborNo) const;

		/**
		 * \brief      Returns the specified ORCA constraint of the specified
		 *             agent.
		 * \param      agentNo         The number of the agent whose ORCA
		 *                             constraint is to be retrieved.
		 * \param      lineNo          The number of the ORCA constraint to be
		 *                             retrieved.
		 * \return     A line representing the specified ORCA constraint.
		 * \note       The halfplane to the left of the line is the region of
		 *             permissible velocities with respect to the specified
		 *             ORCA constraint.
		 */
		const Line &getAgentORCALine(size_t agentNo, size_t lineNo) const;

		/**
		 * \brief      Returns the two-dimensional position of a specified
		 *             agent.
		 * \param      agentNo         The number of the agent whose
		 *                             two-dimensional position is to be retrieved.
		 * \return     The present two-dimensional position of the (center of the)
		 *             agent.
		 */
		const Vector2 &getAgentPosition(size_t agentNo) const;

		/**
		 * \brief      Returns the two-dimensional preferred velocity of a
		 *             specified agent.
		 * \param      agentNo         The number of the agent whose
		 *                             two-dimensional preferred velocity is to be
		 *                             retrieved.
		 * \return     The present two-dimensional preferred velocity of the agent.
		 */
		const Vector2 &getAgentPrefVelocity(size_t agentNo) const;

		/**
		 * \brief      Returns the radius of a specified agent.
		 * \param      agentNo         The number of the agent whose radius is to
		 *                             be retrieved.
		 * \return     The present radius of the agent.
		 */
		float getAgentRadius(size_t agentNo) const;

		/**
		 * \brief      Returns the time horizon of a specified agent.
		 * \param      agentNo         The number of the agent whose time horizon
		 *                             is to be retrieved.
		 * \return     The present time horizon of the agent.
		 */
		float getAgentTimeHorizon(size_t agentNo) const;

		/**
		 * \brief      Returns the time horizon with respect to obstacles of a
		 *             specified agent.
		 * \param      agentNo         The number of the agent whose time horizon
		 *                             with respect to obstacles is to be
		 *                             retrieved.
		 * \return     The present time horizon with respect to obstacles of the
		 *             agent.
		 */
		float getAgentTimeHorizonObst(size_t agentNo) const;

		/**
		 * \brief      Returns the two-dimensional linear velocity of a
		 *             specified agent.
		 * \param      agentNo         The number of the agent whose
		 *                             two-dimensional linear velocity is to be
		 *                             retrieved.
		 * \return     The present two-dimensional linear velocity of the agent.
		 */
		const Vector2 &getAgentVelocity(size_t agentNo) const;

		/**
		 * \brief      Returns the global time of the simulation.
		 * \return     The present global time of the simulation (zero initially).
		 */
		float getGlobalTime() const;

		/**
		 * \brief      Returns the count of agents in the simulation.
		 * \return     The count of agents in the simulation.
		 */
		size_t getNumAgents() const;

		/**
		 * \brief      Returns the count of obstacle vertices in the simulation.
		 * \return     The count of obstacle vertices in the simulation.
		 */
		size_t getNumObstacleVertices() const;

		/**
		 * \brief      Returns the two-dimensional position of a specified obstacle
		 *             vertex.
		 * \param      vertexNo        The number of the obstacle vertex to be
		 *                             retrieved.
		 * \return     The two-dimensional position of the specified obstacle
		 *             vertex.
		 */
		const Vector2 &getObstacleVertex(size_t vertexNo) const;

		/**
		 * \brief      Returns the number of the obstacle vertex succeeding the
		 *             specified obstacle vertex in its polygon.
		 * \param      vertexNo        The number of the obstacle vertex whose
		 *                             successor is to be retrieved.
		 * \return     The number of the obstacle vertex succeeding the specified
		 *             obstacle vertex in its polygon.
		 */
		size_t getNextObstacleVertexNo(size_t vertexNo) const;

		/**
		 * \brief      Returns the number of the obstacle vertex preceding the
		 *             specified obstacle vertex in its polygon.
		 * \param      vertexNo        The number of the obstacle vertex whose
		 *                             predecessor is to be retrieved.
		 * \return     The number of the obstacle vertex preceding the specified
		 *             obstacle vertex in its polygon.
		 */
		size_t getPrevObstacleVertexNo(size_t vertexNo) const;

		/**
		 * \brief      Returns the time step of the simulation.
		 * \return     The present time step of the simulation.
		 */
		float getTimeStep() const;

		/**
		 * \brief      Processes the obstacles that have been added so that they
		 *             are accounted for in the simulation.
		 * \note       Obstacles added to the simulation after this function has
		 *             been called are not accounted for in the simulation.
		 */
		void processObstacles();

		/**
		 * \brief      Performs a visibility query between the two specified
		 *             points with respect to the obstacles
		 * \param      point1          The first point of the query.
		 * \param      point2          The second point of the query.
		 * \param      radius          The minimal distance between the line
		 *                             connecting the two points and the obstacles
		 *                             in order for the points to be mutually
		 *                             visible (optional). Must be non-negative.
		 * \return     A boolean specifying whether the two points are mutually
		 *             visible. Returns true when the obstacles have not been
		 *             processed.
		 */
		bool queryVisibility(const Vector2 &point1, const Vector2 &point2,
							 float radius = 0.0f) const;

		/**
		 * \brief      Sets the default properties for any new agent that is
		 *             added.
		 * \param      neighborDist    The default maximum distance (center point
		 *                             to center point) to other agents a new agent
		 *                             takes into account in the navigation. The
		 *                             larger this number, the longer he running
		 *                             time of the simulation. If the number is too
		 *                             low, the simulation will not be safe.
		 *                             Must be non-negative.
		 * \param      maxNeighbors    The default maximum number of other agents a
		 *                             new agent takes into account in the
		 *                             navigation. The larger this number, the
		 *                             longer the running time of the simulation.
		 *                             If the number is too low, the simulation
		 *                             will not be safe.
		 * \param      timeHorizon     The default minimal amount of time for which
		 *                             a new agent's velocities that are computed
		 *                             by the simulation are safe with respect to
		 *                             other agents. The larger this number, the
		 *                             sooner an agent will respond to the presence
		 *                             of other agents, but the less freedom the
		 *                             agent has in choosing its velocities.
		 *                             Must be positive.
		 * \param      timeHorizonObst The default minimal amount of time for which
		 *                             a new agent's velocities that are computed
		 *                             by the simulation are safe with respect to
		 *                             obstacles. The larger this number, the
		 *                             sooner an agent will respond to the presence
		 *                             of obstacles, but the less freedom the agent
		 *                             has in choosing its velocities.
		 *                             Must be positive.
		 * \param      radius          The default radius of a new agent.
		 *                             Must be non-negative.
		 * \param      maxSpeed        The default maximum speed of a new agent.
		 *                             Must be non-negative.
		 * \param      velocity        The default initial two-dimensional linear
		 *                             velocity of a new agent (optional).
		 */
		void setAgentDefaults(float neighborDist, size_t maxNeighbors,
							  float timeHorizon, float timeHorizonObst,
							  float radius, float maxSpeed,
							  const Vector2 &velocity = Vector2());

		/**
		 * \brief      Sets the maximum neighbor count of a specified agent.
		 * \param      agentNo         The number of the agent whose maximum
		 *                             neighbor count is to be modified.
		 * \param      maxNeighbors    The replacement maximum neighbor count.
		 */
		void setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors);

		/**
		 * \brief      Sets the maximum speed of a specified agent.
		 * \param      agentNo         The number of the agent whose maximum speed
		 *                             is to be modified.
		 * \param      maxSpeed        The replacement maximum speed. Must be
		 *                             non-negative.
		 */
		void setAgentMaxSpeed(size_t agentNo, float maxSpeed);

		/**
		 * \brief      Sets the maximum neighbor distance of a specified agent.
		 * \param      agentNo         The number of the agent whose maximum
		 *                             neighbor distance is to be modified.
		 * \param      neighborDist    The replacement maximum neighbor distance.
		 *                             Must be non-negative.
		 */
		void setAgentNeighborDist(size_t agentNo, float neighborDist);

		/**
		 * \brief      Sets the two-dimensional position of a specified agent.
		 * \param      agentNo         The number of the agent whose
		 *                             two-dimensional position is to be modified.
		 * \param      position        The replacement of the two-dimensional
		 *                             position.
		 */
		void setAgentPosition(size_t agentNo, const Vector2 &position);

		/**
		 * \brief      Sets the two-dimensional preferred velocity of a
		 *             specified agent.
		 * \param      agentNo         The number of the agent whose
		 *                             two-dimensional preferred velocity is to be
		 *                             modified.
		 * \param      prefVelocity    The replacement of the two-dimensional
		 *                             preferred velocity.
		 */
		void setAgentPrefVelocity(size_t agentNo, const Vector2 &prefVelocity);

		/**
		 * \brief      Sets the radius of a specified agent.
		 * \param      agentNo         The number of the agent whose radius is to
		 *                             be modified.
		 * \param      radius          The replacement radius.
		 *                             Must be non-negative.
		 */
		void setAgentRadius(size_t agentNo, float radius);

		/**
		 * \brief      Sets the time horizon of a specified agent with respect
		 *             to other agents.
		 * \param      agentNo         The number of the agent whose time horizon
		 *                             is to be modified.
		 * \param      timeHorizon     The replacement time horizon with respect
		 *                             to other agents. Must be positive.
		 */
		void setAgentTimeHorizon(size_t agentNo, float timeHorizon);

		/**
		 * \brief      Sets the time horizon of a specified agent with respect
		 *             to obstacles.
		 * \param      agentNo         The number of the agent whose time horizon
		 *                             with respect to obstacles is to be modified.
		 * \param      timeHorizonObst The replacement time horizon with respect to
		 *                             obstacles. Must be positive.
		 */
		void setAgentTimeHorizonObst(size_t agentNo, float timeHorizonObst);

		/**
		 * \brief      Sets the two-dimensional linear velocity of a specified
		 *             agent.
		 * \param      agentNo         The number of the agent whose
		 *                             two-dimensional linear velocity is to be
		 *                             modified.
		 * \param      velocity        The replacement two-dimensional linear
		 *                             velocity.
		 */
		void setAgentVelocity(size_t agentNo, const Vector2 &velocity);

		/**
		 * \brief      Sets the time step of the simulation.
		 * \param      timeStep        The time step of the simulation.
		 *                             Must be positive.
		 */
		void setTimeStep(float timeStep);

	public:
		std::vector<Agent2D *> agents_;
		Agent2D *defaultAgent_;
		float globalTime_;
		KdTree2D *kdTree_;
		std::vector<Obstacle2D *> obstacles_;
		float timeStep_;

		friend class Agent2D;
		friend class KdTree2D;
		friend class Obstacle2D;
	};
}

#endif /* RVO2D_RVO_SIMULATOR_H_ */
