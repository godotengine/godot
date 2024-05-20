/*
 * RVOSimulator.h
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

/**
 * \file    RVOSimulator.h
 * \brief   Contains the RVOSimulator class.
 */
#ifndef RVO3D_RVO_SIMULATOR_H_
#define RVO3D_RVO_SIMULATOR_H_

#include <cstddef>
#include <limits>
#include <vector>

#include "Vector3.h"

namespace RVO3D {
	class Agent3D;
	class KdTree3D;

	/**
	 * \brief   Error value.
	 *
	 * A value equal to the largest unsigned integer, which is returned in case of an error by functions in RVO3D::RVOSimulator.
	 */
	const size_t RVO3D_ERROR = std::numeric_limits<size_t>::max();

	/**
	 * \brief   Defines a plane.
	 */
	class Plane {
	public:
		/**
		 * \brief   A point on the plane.
		 */
		Vector3 point;

		/**
		 * \brief   The normal to the plane.
		 */
		Vector3 normal;
	};

	/**
	 * \brief  Defines the simulation.
	 *
	 * The main class of the library that contains all simulation functionality.
	 */
	class RVOSimulator3D {
	public:
		/**
		 * \brief   Constructs a simulator instance.
		 */
		RVOSimulator3D();

		/**
		 * \brief   Constructs a simulator instance and sets the default properties for any new agent that is added.
		 * \param   timeStep      The time step of the simulation. Must be positive.
		 * \param   neighborDist  The default maximum distance (center point to center point) to other agents a new agent takes into account in the navigation. The larger this number, the longer he running time of the simulation. If the number is too low, the simulation will not be safe. Must be non-negative.
		 * \param   maxNeighbors  The default maximum number of other agents a new agent takes into account in the navigation. The larger this number, the longer the running time of the simulation. If the number is too low, the simulation will not be safe.
		 * \param   timeHorizon   The default minimum amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to other agents. The larger this number, the sooner an agent will respond to the presence of other agents, but the less freedom the agent has in choosing its velocities. Must be positive.
		 * \param   radius        The default radius of a new agent. Must be non-negative.
		 * \param   maxSpeed      The default maximum speed of a new agent. Must be non-negative.
		 * \param   velocity      The default initial three-dimensional linear velocity of a new agent (optional).
		 */
		RVOSimulator3D(float timeStep, float neighborDist, size_t maxNeighbors, float timeHorizon, float radius, float maxSpeed, const Vector3 &velocity = Vector3());

		/**
		 * \brief   Destroys this simulator instance.
		 */
		~RVOSimulator3D();

		/**
		 * \brief   Adds a new agent with default properties to the simulation.
		 * \param   position  The three-dimensional starting position of this agent.
		 * \return  The number of the agent, or RVO3D::RVO3D_ERROR when the agent defaults have not been set.
		 */
		size_t addAgent(const Vector3 &position);

		/**
		 * \brief   Adds a new agent to the simulation.
		 * \param   position      The three-dimensional starting position of this agent.
		 * \param   neighborDist  The maximum distance (center point to center point) to other agents this agent takes into account in the navigation. The larger this number, the longer the running time of the simulation. If the number is too low, the simulation will not be safe. Must be non-negative.
		 * \param   maxNeighbors  The maximum number of other agents this agent takes into account in the navigation. The larger this number, the longer the running time of the simulation. If the number is too low, the simulation will not be safe.
		 * \param   timeHorizon   The minimum amount of time for which this agent's velocities that are computed by the simulation are safe with respect to other agents. The larger this number, the sooner this agent will respond to the presence of other agents, but the less freedom this agent has in choosing its velocities. Must be positive.
		 * \param   radius        The radius of this agent. Must be non-negative.
		 * \param   maxSpeed      The maximum speed of this agent. Must be non-negative.
		 * \param   velocity      The initial three-dimensional linear velocity of this agent (optional).
		 * \return  The number of the agent.
		 */
		size_t addAgent(const Vector3 &position, float neighborDist, size_t maxNeighbors, float timeHorizon, float radius, float maxSpeed, const Vector3 &velocity = Vector3());

		/**
		 * \brief   Lets the simulator perform a simulation step and updates the three-dimensional position and three-dimensional velocity of each agent.
		 */
		void doStep();

		/**
		 * \brief   Returns the specified agent neighbor of the specified agent.
		 * \param   agentNo     The number of the agent whose agent neighbor is to be retrieved.
		 * \param   neighborNo  The number of the agent neighbor to be retrieved.
		 * \return  The number of the neighboring agent.
		 */
		size_t getAgentAgentNeighbor(size_t agentNo, size_t neighborNo) const;

		/**
		 * \brief   Returns the maximum neighbor count of a specified agent.
		 * \param   agentNo  The number of the agent whose maximum neighbor count is to be retrieved.
		 * \return  The present maximum neighbor count of the agent.
		 */
		size_t getAgentMaxNeighbors(size_t agentNo) const;

		/**
		 * \brief   Returns the maximum speed of a specified agent.
		 * \param   agentNo  The number of the agent whose maximum speed is to be retrieved.
		 * \return  The present maximum speed of the agent.
		 */
		float getAgentMaxSpeed(size_t agentNo) const;

		/**
		 * \brief   Returns the maximum neighbor distance of a specified agent.
		 * \param   agentNo  The number of the agent whose maximum neighbor distance is to be retrieved.
		 * \return  The present maximum neighbor distance of the agent.
		 */
		float getAgentNeighborDist(size_t agentNo) const;

		/**
		 * \brief   Returns the count of agent neighbors taken into account to compute the current velocity for the specified agent.
		 * \param   agentNo  The number of the agent whose count of agent neighbors is to be retrieved.
		 * \return  The count of agent neighbors taken into account to compute the current velocity for the specified agent.
		 */
		size_t getAgentNumAgentNeighbors(size_t agentNo) const;

		/**
		 * \brief   Returns the count of ORCA constraints used to compute the current velocity for the specified agent.
		 * \param   agentNo  The number of the agent whose count of ORCA constraints is to be retrieved.
		 * \return  The count of ORCA constraints used to compute the current velocity for the specified agent.
		 */
		size_t getAgentNumORCAPlanes(size_t agentNo) const;

		/**
		 * \brief   Returns the specified ORCA constraint of the specified agent.
		 * \param   agentNo  The number of the agent whose ORCA constraint is to be retrieved.
		 * \param   planeNo  The number of the ORCA constraint to be retrieved.
		 * \return  A plane representing the specified ORCA constraint.
		 * \note    The halfspace to which the normal of the plane points is the region of permissible velocities with respect to the specified ORCA constraint.
		 */
		const Plane &getAgentORCAPlane(size_t agentNo, size_t planeNo) const;

		/**
		 * \brief   Returns the three-dimensional position of a specified agent.
		 * \param   agentNo  The number of the agent whose three-dimensional position is to be retrieved.
		 * \return  The present three-dimensional position of the (center of the) agent.
		 */
		const Vector3 &getAgentPosition(size_t agentNo) const;

		/**
		 * \brief   Returns the three-dimensional preferred velocity of a specified agent.
		 * \param   agentNo  The number of the agent whose three-dimensional preferred velocity is to be retrieved.
		 * \return  The present three-dimensional preferred velocity of the agent.
		 */
		const Vector3 &getAgentPrefVelocity(size_t agentNo) const;

		/**
		 * \brief   Returns the radius of a specified agent.
		 * \param   agentNo  The number of the agent whose radius is to be retrieved.
		 * \return  The present radius of the agent.
		 */
		float getAgentRadius(size_t agentNo) const;

		/**
		 * \brief   Returns the time horizon of a specified agent.
		 * \param   agentNo  The number of the agent whose time horizon is to be retrieved.
		 * \return  The present time horizon of the agent.
		 */
		float getAgentTimeHorizon(size_t agentNo) const;

		/**
		 * \brief   Returns the three-dimensional linear velocity of a specified agent.
		 * \param   agentNo  The number of the agent whose three-dimensional linear velocity is to be retrieved.
		 * \return  The present three-dimensional linear velocity of the agent.
		 */
		const Vector3 &getAgentVelocity(size_t agentNo) const;

		/**
		 * \brief   Returns the global time of the simulation.
		 * \return  The present global time of the simulation (zero initially).
		 */
		float getGlobalTime() const;

		/**
		 * \brief   Returns the count of agents in the simulation.
		 * \return  The count of agents in the simulation.
		 */
		size_t getNumAgents() const;

		/**
		 * \brief   Returns the time step of the simulation.
		 * \return  The present time step of the simulation.
		 */
		float getTimeStep() const;

		/**
		 * \brief   Removes an agent from the simulation.
		 * \param   agentNo  The number of the agent that is to be removed.
		 * \note    After the removal of the agent, the agent that previously had number getNumAgents() - 1 will now have number agentNo.
		 */
		void removeAgent(size_t agentNo);

		/**
		 * \brief   Sets the default properties for any new agent that is added.
		 * \param   neighborDist  The default maximum distance (center point to center point) to other agents a new agent takes into account in the navigation. The larger this number, the longer he running time of the simulation. If the number is too low, the simulation will not be safe. Must be non-negative.
		 * \param   maxNeighbors  The default maximum number of other agents a new agent takes into account in the navigation. The larger this number, the longer the running time of the simulation. If the number is too low, the simulation will not be safe.
		 * \param   timeHorizon   The default minimum amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to other agents. The larger this number, the sooner an agent will respond to the presence of other agents, but the less freedom the agent has in choosing its velocities. Must be positive.
		 * \param   radius        The default radius of a new agent. Must be non-negative.
		 * \param   maxSpeed      The default maximum speed of a new agent. Must be non-negative.
		 * \param   velocity      The default initial three-dimensional linear velocity of a new agent (optional).
		 */
		void setAgentDefaults(float neighborDist, size_t maxNeighbors, float timeHorizon, float radius, float maxSpeed, const Vector3 &velocity = Vector3());

		/**
		 * \brief   Sets the maximum neighbor count of a specified agent.
		 * \param   agentNo       The number of the agent whose maximum neighbor count is to be modified.
		 * \param   maxNeighbors  The replacement maximum neighbor count.
		 */
		void setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors);

		/**
		 * \brief   Sets the maximum speed of a specified agent.
		 * \param   agentNo   The number of the agent whose maximum speed is to be modified.
		 * \param   maxSpeed  The replacement maximum speed. Must be non-negative.
		 */
		void setAgentMaxSpeed(size_t agentNo, float maxSpeed);

		/**
		 * \brief   Sets the maximum neighbor distance of a specified agent.
		 * \param   agentNo       The number of the agent whose maximum neighbor distance is to be modified.
		 * \param   neighborDist  The replacement maximum neighbor distance. Must be non-negative.
		 */
		void setAgentNeighborDist(size_t agentNo, float neighborDist);

		/**
		 * \brief   Sets the three-dimensional position of a specified agent.
		 * \param   agentNo   The number of the agent whose three-dimensional position is to be modified.
		 * \param   position  The replacement of the three-dimensional position.
		 */
		void setAgentPosition(size_t agentNo, const Vector3 &position);

		/**
		 * \brief   Sets the three-dimensional preferred velocity of a specified agent.
		 * \param   agentNo       The number of the agent whose three-dimensional preferred velocity is to be modified.
		 * \param   prefVelocity  The replacement of the three-dimensional preferred velocity.
		 */
		void setAgentPrefVelocity(size_t agentNo, const Vector3 &prefVelocity);

		/**
		 * \brief   Sets the radius of a specified agent.
		 * \param   agentNo  The number of the agent whose radius is to be modified.
		 * \param   radius   The replacement radius. Must be non-negative.
		 */
		void setAgentRadius(size_t agentNo, float radius);

		/**
		 * \brief   Sets the time horizon of a specified agent with respect to other agents.
		 * \param   agentNo      The number of the agent whose time horizon is to be modified.
		 * \param   timeHorizon  The replacement time horizon with respect to other agents. Must be positive.
		 */
		void setAgentTimeHorizon(size_t agentNo, float timeHorizon);

		/**
		 * \brief   Sets the three-dimensional linear velocity of a specified agent.
		 * \param   agentNo   The number of the agent whose three-dimensional linear velocity is to be modified.
		 * \param   velocity  The replacement three-dimensional linear velocity.
		 */
		void setAgentVelocity(size_t agentNo, const Vector3 &velocity);

		/**
		 * \brief   Sets the time step of the simulation.
		 * \param   timeStep  The time step of the simulation. Must be positive.
		 */
		void setTimeStep(float timeStep);

	public:
		Agent3D *defaultAgent_;
		KdTree3D *kdTree_;
		float globalTime_;
		float timeStep_;
		std::vector<Agent3D *> agents_;

		friend class Agent3D;
		friend class KdTree3D;
	};
}

#endif
