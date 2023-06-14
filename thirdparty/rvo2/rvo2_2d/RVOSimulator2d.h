/*
 * RVOSimulator.h
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

#ifndef RVO2D_RVO_SIMULATOR_H_
#define RVO2D_RVO_SIMULATOR_H_

/**
 * @file  RVOSimulator2d.h
 * @brief Declares and defines the RVOSimulator2D class.
 */

#include <cstddef>
#include <vector>


namespace RVO2D {
class Agent2D;
class KdTree2D;
class Line;
class Obstacle2D;
class Vector2;

/**
 * @relates RVOSimulator2D
 * @brief   Error value. A value equal to the largest unsigned integer that is
 *          returned in case of an error by functions in RVO::RVOSimulator.
 */
extern const std::size_t RVO2D_ERROR;

/**
 * @brief Defines the simulation. The main class of the library that contains
 *        all simulation functionality.
 */
class RVOSimulator2D {
 public:
  /**
   * @brief Constructs a simulator instance.
   */
  RVOSimulator2D();

  /**
   * @brief     Constructs a simulator instance and sets the default
   *            properties for any new agent that is added.
   * @param[in] timeStep        The time step of the simulation. Must be
   *                            positive.
   * @param[in] neighborDist    The default maximum distance center-point to
   *                            center-point to other agents a new agent takes
   *                            into account in the navigation. The larger this
   *                            number, the longer he running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe. Must be
   *                            non-negative.
   * @param[in] maxNeighbors    The default maximum number of other agents a
   *                            new agent takes into account in the navigation.
   *                            The larger this number, the longer the running
   *                            time of the simulation. If the number is too
   *                            low, the simulation will not be safe.
   * @param[in] timeHorizon     The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to other
   *                            agents. The larger this number, the sooner an
   *                            agent will respond to the presence of other
   *                            agents, but the less freedom the agent  has in
   *                            choosing its velocities. Must be positive.
   * @param[in] timeHorizonObst The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to obstacles.
   *                            The larger this number, the sooner an agent will
   *                            respond to the presence of obstacles, but the
   *                            less freedom the agent has in choosing its
   *                            velocities. Must be positive.
   * @param[in] radius          The default radius of a new agent. Must be
   *                            non-negative.
   * @param[in] maxSpeed        The default maximum speed of a new agent. Must
   *                            be non-negative.
   */
  RVOSimulator2D(float timeStep, float neighborDist, std::size_t maxNeighbors,
               float timeHorizon, float timeHorizonObst, float radius,
               float maxSpeed);

  /**
   * @brief     Constructs a simulator instance and sets the default properties
   *            for any new agent that is added.
   * @param[in] timeStep        The time step of the simulation. Must be
   *                            positive.
   * @param[in] neighborDist    The default maximum distance center-point to
   *                            center-point to other agents a new agent takes
   *                            into account in the navigation. The larger this
   *                            number, the longer he running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe. Must be
   *                            non-negative.
   * @param[in] maxNeighbors    The default maximum number of other agents a new
   *                            agent takes into account in the navigation. The
   *                            larger this number, the longer the running time
   *                            of the simulation. If the number is too low, the
   *                            simulation will not be safe.
   * @param[in] timeHorizon     The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to other
   *                            agents. The larger this number, the sooner an
   *                            agent will respond to the presence of other
   *                            agents, but the less freedom the agent has in
   *                            choosing its velocities. Must be positive.
   * @param[in] timeHorizonObst The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to obstacles.
   *                            The larger this number, the sooner an agent will
   *                            respond to the presence of obstacles, but the
   *                            less freedom the agent has in choosing its
   *                            velocities. Must be positive.
   * @param[in] radius          The default radius of a new agent. Must be
   *                            non-negative.
   * @param[in] maxSpeed        The default maximum speed of a new agent. Must
   *                            be non-negative.
   * @param[in] velocity        The default initial two-dimensional linear
   *                            velocity of a new agent.
   */
  RVOSimulator2D(float timeStep, float neighborDist, std::size_t maxNeighbors,
               float timeHorizon, float timeHorizonObst, float radius,
               float maxSpeed, const Vector2 &velocity);

  /**
   * @brief Destroys this simulator instance.
   */
  ~RVOSimulator2D();

  /**
   * @brief     Adds a new agent with default properties to the simulation.
   * @param[in] position The two-dimensional starting position of this agent.
   * @return    The number of the agent, or RVO::RVO2D_ERROR when the agent
   *            defaults have not been set.
   */
  std::size_t addAgent(const Vector2 &position);

  /**
   * @brief     Adds a new agent to the simulation.
   * @param[in] position        The two-dimensional starting position of this
   *                            agent.
   * @param[in] neighborDist    The maximum distance center-point to
   *                            center-point to other agents this agent takes
   *                            into account in the navigation. The larger this
   *                            number, the longer the running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe. Must be
   *                            non-negative.
   * @param[in] maxNeighbors    The maximum number of other agents this agent
   *                            takes into account in the navigation. The larger
   *                            this number, the longer the running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe.
   * @param[in] timeHorizon     The minimal amount of time for which this
   *                            agent's velocities that are computed by the
   *                            simulation are safe with respect to other
   *                            agents. The larger this number, the sooner this
   *                            agent will respond to the presence of other
   *                            agents, but the less freedom this agent has in
   *                            choosing its velocities. Must be positive.
   * @param[in] timeHorizonObst The minimal amount of time for which this
   *                            agent's velocities that are computed by the
   *                            simulation are safe with respect to obstacles
   *                            The larger this number, the sooner this agent
   *                            will respond to the presence of obstacles, but
   *                            the less freedom this agent has in choosing its
   *                            velocities. Must be positive.
   * @param[in] radius          The radius of this agent. Must be non-negative.
   * @param[in] maxSpeed        The maximum speed of this agent. Must be
   *                            non-negative.
   * @return    The number of the agent.
   */
  std::size_t addAgent(const Vector2 &position, float neighborDist,
                       std::size_t maxNeighbors, float timeHorizon,
                       float timeHorizonObst, float radius, float maxSpeed);

  /**
   * @brief     Adds a new agent to the simulation.
   * @param[in] position        The two-dimensional starting position of this
   *                            agent.
   * @param[in] neighborDist    The maximum distance center-point to
   *                            center-point to other agents this agent takes
   *                            into account in the navigation. The larger this
   *                            number, the longer the running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe. Must be
   *                            non-negative.
   * @param[in] maxNeighbors    The maximum number of other agents this agent
   *                            takes into account in the navigation. The larger
   *                            this number, the longer the running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe.
   * @param[in] timeHorizon     The minimal amount of time for which this
   *                            agent's velocities that are computed by the
   *                            simulation are safe with respect to other
   *                            agents. The larger this number, the sooner this
   *                            agent will respond to the presence of other
   *                            agents, but the less freedom this agent has in
   *                            choosing its velocities. Must be positive.
   * @param[in] timeHorizonObst The minimal amount of time for which this
   *                            agent's velocities that are computed by the
   *                            simulation are safe with respect to obstacles.
   *                            The larger this number, the sooner this agent
   *                            will respond to the presence of obstacles, but
   *                            the less freedom this agent has in choosing its
   *                            velocities. Must be positive.
   * @param[in] radius          The radius of this agent. Must be non-negative.
   * @param[in] maxSpeed        The maximum speed of this agent. Must be
   *                            non-negative.
   * @param[in] velocity        The initial two-dimensional linear velocity of
   *                            this agent.
   * @return    The number of the agent.
   */
  std::size_t addAgent(const Vector2 &position, float neighborDist,
                       std::size_t maxNeighbors, float timeHorizon,
                       float timeHorizonObst, float radius, float maxSpeed,
                       const Vector2 &velocity);

  /**
   * @brief     Adds a new obstacle to the simulation.
   * @param[in] vertices List of the vertices of the polygonal obstacle in
   *                     counterclockwise order.
   * @return    The number of the first vertex of the obstacle, or
   *            RVO::RVO2D_ERROR when the number of vertices is less than two.
   * @note      To add a "negative" obstacle, e.g., a bounding polygon around
   *            the environment, the vertices should be listed in clockwise
   *            order.
   */
  std::size_t addObstacle(const std::vector<Vector2> &vertices);

  /**
   * @brief Lets the simulator perform a simulation step and updates the
   *        two-dimensional position and two-dimensional velocity of each agent.
   */
  void doStep();

  /**
   * @brief     Returns the specified agent neighbor of the specified agent.
   * @param[in] agentNo    The number of the agent whose agent neighbor is to be
   *                       retrieved.
   * @param[in] neighborNo The number of the agent neighbor to be retrieved.
   * @return    The number of the neighboring agent.
   */
  std::size_t getAgentAgentNeighbor(std::size_t agentNo,
                                    std::size_t neighborNo) const;

  /**
   * @brief     Returns the maximum neighbor count of a specified agent.
   * @param[in] agentNo The number of the agent whose maximum neighbor count is
   *                    to be retrieved.
   * @return    The present maximum neighbor count of the agent.
   */
  std::size_t getAgentMaxNeighbors(std::size_t agentNo) const;

  /**
   * @brief     Returns the maximum speed of a specified agent.
   * @param[in] agentNo The number of the agent whose maximum speed is to be
   *                    retrieved.
   * @return    The present maximum speed of the agent.
   */
  float getAgentMaxSpeed(std::size_t agentNo) const;

  /**
   * @brief     Returns the maximum neighbor distance of a specified agent.
   * @param[in] agentNo The number of the agent whose maximum neighbor distance
   *                    is to be retrieved.
   * @return    The present maximum neighbor distance of the agent.
   */
  float getAgentNeighborDist(std::size_t agentNo) const;

  /**
   * @brief     Returns the count of agent neighbors taken into account to
   *            compute the current velocity for the specified agent.
   * @param[in] agentNo The number of the agent whose count of agent neighbors
   *                    is to be retrieved.
   * @return    The count of agent neighbors taken into account to compute the
   *            current velocity for the specified agent.
   */
  std::size_t getAgentNumAgentNeighbors(std::size_t agentNo) const;

  /**
   * @brief     Returns the count of obstacle neighbors taken into account to
   *            compute the current velocity for the specified agent.
   * @param[in] agentNo The number of the agent whose count of obstacle
   *                    neighbors is to be retrieved.
   * @return    The count of obstacle neighbors taken into account to compute
   *            the current velocity for the specified agent.
   */
  std::size_t getAgentNumObstacleNeighbors(std::size_t agentNo) const;

  /**
   * @brief     Returns the count of ORCA constraints used to compute the
   *            current velocity for the specified agent.
   * @param[in] agentNo The number of the agent whose count of ORCA constraints
   *                    is to be retrieved.
   * @return    The count of ORCA constraints used to compute the current
   *            velocity for the specified agent.
   */
  std::size_t getAgentNumORCALines(std::size_t agentNo) const;

  /**
   * @brief     Returns the specified obstacle neighbor of the specified agent.
   * @param[in] agentNo    The number of the agent whose obstacle neighbor is to
   *                       be retrieved.
   * @param[in] neighborNo The number of the obstacle neighbor to be retrieved.
   * @return    The number of the first vertex of the neighboring obstacle edge.
   */
  std::size_t getAgentObstacleNeighbor(std::size_t agentNo,
                                       std::size_t neighborNo) const;

  /**
   * @brief     Returns the specified ORCA constraint of the specified agent.
   * @param[in] agentNo The number of the agent whose ORCA constraint is to be
   *                    retrieved.
   * @param[in] lineNo  The number of the ORCA constraint to be retrieved.
   * @return    A line representing the specified ORCA constraint.
   * @note      The half-plane to the left of the line is the region of
   *            permissible velocities with respect to the specified ORCA
   *            constraint.
   */
  const Line &getAgentORCALine(std::size_t agentNo, std::size_t lineNo) const;

  /**
   * @brief     Returns the two-dimensional position of a specified agent.
   * @param[in] agentNo The number of the agent whose two-dimensional position
   *                    is to be retrieved.
   * @return    The present two-dimensional position of the center of the agent.
   */
  const Vector2 &getAgentPosition(std::size_t agentNo) const;

  /**
   * @brief     Returns the two-dimensional preferred velocity of a specified
   *            agent.
   * @param[in] agentNo The number of the agent whose two-dimensional preferred
   *                    velocity is to be retrieved.
   * @return    The present two-dimensional preferred velocity of the agent.
   */
  const Vector2 &getAgentPrefVelocity(std::size_t agentNo) const;

  /**
   * @brief     Returns the radius of a specified agent.
   * @param[in] agentNo The number of the agent whose radius is to be retrieved.
   * @return    The present radius of the agent.
   */
  float getAgentRadius(std::size_t agentNo) const;

  /**
   * @brief     Returns the time horizon of a specified agent.
   * @param[in] agentNo The number of the agent whose time horizon is to be
   *                    retrieved.
   * @return    The present time horizon of the agent.
   */
  float getAgentTimeHorizon(std::size_t agentNo) const;

  /**
   * @brief     Returns the time horizon with respect to obstacles of a
   *            specified agent.
   * @param[in] agentNo The number of the agent whose time horizon with respect
   *                    to obstacles is to be retrieved.
   * @return    The present time horizon with respect to obstacles of the agent.
   */
  float getAgentTimeHorizonObst(std::size_t agentNo) const;

  /**
   * @brief     Returns the two-dimensional linear velocity of a specified
   *            agent.
   * @param[in] agentNo The number of the agent whose two-dimensional linear
   *                    velocity is to be retrieved.
   * @return    The present two-dimensional linear velocity of the agent.
   */
  const Vector2 &getAgentVelocity(std::size_t agentNo) const;

  /**
   * @brief  Returns the global time of the simulation.
   * @return The present global time of the simulation (zero initially).
   */
  float getGlobalTime() const { return globalTime_; }

  /**
   * @brief  Returns the count of agents in the simulation.
   * @return The count of agents in the simulation.
   */
  std::size_t getNumAgents() const { return agents_.size(); }

  /**
   * @brief  Returns the count of obstacle vertices in the simulation.
   * @return The count of obstacle vertices in the simulation.
   */
  std::size_t getNumObstacleVertices() const { return obstacles_.size(); }

  /**
   * @brief     Returns the two-dimensional position of a specified obstacle
   *            vertex.
   * @param[in] vertexNo The number of the obstacle vertex to be retrieved.
   * @return    The two-dimensional position of the specified obstacle vertex.
   */
  const Vector2 &getObstacleVertex(std::size_t vertexNo) const;

  /**
   * @brief     Returns the number of the obstacle vertex succeeding the
   *            specified obstacle vertex in its polygon.
   * @param[in] vertexNo The number of the obstacle vertex whose successor is to
   *                     be retrieved.
   * @return    The number of the obstacle vertex succeeding the specified
   *            obstacle vertex in its polygon.
   */
  std::size_t getNextObstacleVertexNo(std::size_t vertexNo) const;

  /**
   * @brief     Returns the number of the obstacle vertex preceding the
   *            specified obstacle vertex in its polygon.
   * @param[in] vertexNo The number of the obstacle vertex whose predecessor is
   *                     to be retrieved.
   * @return    The number of the obstacle vertex preceding the specified
   *            obstacle vertex in its polygon.
   */
  std::size_t getPrevObstacleVertexNo(std::size_t vertexNo) const;

  /**
   * @brief  Returns the time step of the simulation.
   * @return The present time step of the simulation.
   */
  float getTimeStep() const { return timeStep_; }

  /**
   * @brief Processes the obstacles that have been added so that they are
   *        accounted for in the simulation.
   * @note  Obstacles added to the simulation after this function has been
   *        called are not accounted for in the simulation.
   */
  void processObstacles();

  /**
   * @brief     Performs a visibility query between the two specified points
   *            with respect to the obstacles
   * @param[in] point1 The first point of the query.
   * @param[in] point2 The second point of the query.
   * @return    A boolean specifying whether the two points are mutually
   *            visible. Returns true when the obstacles have not been
   *            processed.
   */
  bool queryVisibility(const Vector2 &point1, const Vector2 &point2) const;

  /**
   * @brief     Performs a visibility query between the two specified points
   *            with respect to the obstacles
   * @param[in] point1 The first point of the query.
   * @param[in] point2 The second point of the query.
   * @param[in] radius The minimal distance between the line connecting the two
   *                   points and the obstacles in order for the points to be
   *                   mutually visible. Must be non-negative.
   * @return    A boolean specifying whether the two points are mutually
   *            visible. Returns true when the obstacles have not been
   *            processed.
   */
  bool queryVisibility(const Vector2 &point1, const Vector2 &point2,
                       float radius) const;

  /**
   * @brief     Sets the default properties for any new agent that is added.
   * @param[in] neighborDist    The default maximum distance center-point to
   *                            center-point to other agents a new agent takes
   *                            into account in the navigation. The larger this
   *                            number, the longer he running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe. Must be
   *                            non-negative.
   * @param[in] maxNeighbors    The default maximum number of other agents a new
   *                            agent takes into account in the navigation. The
   *                            larger this number, the longer the running time
   *                            of the simulation. If the number is too low, the
   *                            simulation will not be safe.
   * @param[in] timeHorizon     The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to other
   *                            agents. The larger this number, the sooner an
   *                            agent will respond to the presence of other
   *                            agents, but the less freedom the agent has in
   *                            choosing its velocities. Must be positive.
   * @param[in] timeHorizonObst The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to obstacles.
   *                            The larger this number, the sooner an agent will
   *                            respond to the presence of obstacles, but the
   *                            less freedom the agent has in  choosing its
   *                            velocities. Must be positive.
   * @param[in] radius          The default radius of a new agent. Must be
   *                            non-negative.
   * @param[in] maxSpeed        The default maximum speed of a new agent. Must
   *                            be non-negative.
   */
  void setAgentDefaults(float neighborDist, std::size_t maxNeighbors,
                        float timeHorizon, float timeHorizonObst, float radius,
                        float maxSpeed);

  /**
   * @brief     Sets the default properties for any new agent that is added.
   * @param[in] neighborDist    The default maximum distance center-point to
   *                            center-point to other agents a new agent takes
   *                            into account in the navigation. The larger this
   *                            number, the longer he running time of the
   *                            simulation. If the number is too low, the
   *                            simulation will not be safe. Must be
   *                            non-negative.
   * @param[in] maxNeighbors    The default maximum number of other agents a new
   *                            agent takes into account in the navigation. The
   *                            larger this number, the longer the running time
   *                            of the simulation. If the number is too low, the
   *                            simulation will not be safe.
   * @param[in] timeHorizon     The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to other
   *                            agents. The larger this number, the sooner an
   *                            agent will respond to the presence of other
   *                            agents, but the less freedom the agent has in
   *                            choosing its velocities. Must be positive.
   * @param[in] timeHorizonObst The default minimal amount of time for which a
   *                            new agent's velocities that are computed by the
   *                            simulation are safe with respect to obstacles.
   *                            The larger this number, the sooner an agent will
   *                            respond to the presence of obstacles, but the
   *                            less freedom the agent has in choosing its
   *                            velocities. Must be positive.
   * @param[in] radius          The default radius of a new agent. Must be
   *                            non-negative.
   * @param[in] maxSpeed        The default maximum speed of a new agent. Must
   *                            be non-negative.
   * @param[in] velocity        The default initial two-dimensional linear
   *                            velocity of a new agent.
   */
  void setAgentDefaults(float neighborDist, std::size_t maxNeighbors,
                        float timeHorizon, float timeHorizonObst, float radius,
                        float maxSpeed, const Vector2 &velocity);

  /**
   * @brief     Sets the maximum neighbor count of a specified agent.
   * @param[in] agentNo      The number of the agent whose maximum neighbor
   *                         count is to be modified.
   * @param[in] maxNeighbors The replacement maximum neighbor count.
   */
  void setAgentMaxNeighbors(std::size_t agentNo, std::size_t maxNeighbors);

  /**
   * @brief     Sets the maximum speed of a specified agent.
   * @param[in] agentNo  The number of the agent whose maximum speed is to be
   *                     modified.
   * @param[in] maxSpeed The replacement maximum speed. Must be non-negative.
   */
  void setAgentMaxSpeed(std::size_t agentNo, float maxSpeed);

  /**
   * @brief     Sets the maximum neighbor distance of a specified agent.
   * @param[in] agentNo      The number of the agent whose maximum neighbor
   *                         distance is to be modified.
   * @param[in] neighborDist The replacement maximum neighbor distance. Must be
   *                         non-negative.
   */
  void setAgentNeighborDist(std::size_t agentNo, float neighborDist);

  /**
   * @brief     Sets the two-dimensional position of a specified agent.
   * @param[in] agentNo  The number of the agent whose two-dimensional position
   *                     is to be modified.
   * @param[in] position The replacement of the two-dimensional position.
   */
  void setAgentPosition(std::size_t agentNo, const Vector2 &position);

  /**
   * @brief     Sets the two-dimensional preferred velocity of a specified
   *            agent.
   * @param[in] agentNo      The number of the agent whose two-dimensional
   *                         preferred velocity is to be modified.
   * @param[in] prefVelocity The replacement of the two-dimensional preferred
   *                         velocity.
   */
  void setAgentPrefVelocity(std::size_t agentNo, const Vector2 &prefVelocity);

  /**
   * @brief     Sets the radius of a specified agent.
   * @param[in] agentNo The number of the agent whose radius is to be modified.
   * @param[in] radius  The replacement radius. Must be non-negative.
   */
  void setAgentRadius(std::size_t agentNo, float radius);

  /**
   * @brief     Sets the time horizon of a specified agent with respect to other
   *            agents.
   * @param[in] agentNo     The number of the agent whose time horizon is to be
   *                        modified.
   * @param[in] timeHorizon The replacement time horizon with respect to other
   *                        agents. Must be positive.
   */
  void setAgentTimeHorizon(std::size_t agentNo, float timeHorizon);

  /**
   * @brief     Sets the time horizon of a specified agent with respect to
   *            obstacles.
   * @param[in] agentNo         The number of the agent whose time horizon with
   *                            respect to obstacles is to be modified.
   * @param[in] timeHorizonObst The replacement time horizon with respect to
   *                            obstacles. Must be positive.
   */
  void setAgentTimeHorizonObst(std::size_t agentNo, float timeHorizonObst);

  /**
   * @brief     Sets the two-dimensional linear velocity of a specified agent.
   * @param[in] agentNo  The number of the agent whose two-dimensional linear
   *                     velocity is to be modified.
   * @param[in] velocity The replacement two-dimensional linear velocity.
   */
  void setAgentVelocity(std::size_t agentNo, const Vector2 &velocity);

  /**
   * @brief     Sets the time step of the simulation.
   * @param[in] timeStep The time step of the simulation. Must be positive.
   */
  void setTimeStep(float timeStep) { timeStep_ = timeStep; }

 public:
  /* Not implemented. */
  RVOSimulator2D(const RVOSimulator2D &other);

  /* Not implemented. */
  RVOSimulator2D &operator=(const RVOSimulator2D &other);

  std::vector<Agent2D *> agents_;
  std::vector<Obstacle2D *> obstacles_;
  Agent2D *defaultAgent_;
  KdTree2D *kdTree_;
  float globalTime_;
  float timeStep_;

  friend class KdTree2D;
};
} /* namespace RVO2D */

#endif /* RVO2D_RVO_SIMULATOR_H_ */
