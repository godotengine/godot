/*
 * RVO.h
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

#ifndef RVO_RVO_H_
#define RVO_RVO_H_

#include "Vector2.h"

/**

 \file       RVO.h
 \brief      Includes all public headers in the library.

 \namespace  RVO
 \brief      Contains all classes, functions, and constants used in the library.

 \mainpage   RVO2 Library

 \author     Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, and
 Dinesh Manocha

 <b>RVO2 Library</b> is an easy-to-use C++ implementation of the
 <a href="http://gamma.cs.unc.edu/ORCA/">Optimal Reciprocal Collision Avoidance</a>
 (ORCA) formulation for multi-agent simulation. <b>RVO2 Library</b> automatically
 uses parallelism for computing the motion of the agents if your machine has
 multiple processors and your compiler supports <a href="http://www.openmp.org/">
 OpenMP</a>.

 Please follow the following steps to install and use <b>RVO2 Library</b>.

 - \subpage  whatsnew
 - \subpage  building
 - \subpage  using
 - \subpage  params

 See the documentation of the RVO::RVOSimulator class for an exhaustive list of
 public functions of <b>RVO2 Library</b>.

 <b>RVO2 Library</b>, accompanying example code, and this documentation is
 released for educational, research, and non-profit purposes under the following
 \subpage terms "terms and conditions".


 \page       whatsnew    What Is New in RVO2 Library

 \section    localca     Local Collision Avoidance

 The main difference between <b>RVO2 Library</b> and %RVO Library 1.x is the
 local collision avoidance technique used. <b>RVO2 Library</b> uses
 <a href="http://gamma.cs.unc.edu/CA/">Optimal Reciprocal Collision Avoidance</a>
 (ORCA), whereas %RVO Library 1.x uses <a href="http://gamma.cs.unc.edu/RVO/">
 Reciprocal Velocity Obstacles</a> (%RVO). For legacy reasons, and since both
 techniques are based on the same principles of reciprocal collision avoidance
 and relative velocity, we did not change the name of the library.

 A major consequence of the change of local collision avoidance technique is that
 the simulation has become much faster in <b>RVO2 Library</b>. ORCA defines
 velocity constraints with respect to other agents as half-planes, and an optimal
 velocity is efficiently found using (two-dimensional) linear programming. In
 contrast, %RVO Library 1.x uses random sampling to find a good velocity. Also,
 the behavior of the agents is smoother in <b>RVO2 Library</b>. It is proven
 mathematically that ORCA lets the velocity of agents evolve continuously over
 time, whereas %RVO Library 1.x occasionally showed oscillations and reciprocal
 dances. Furthermore, ORCA provides stronger guarantees with respect to collision
 avoidance.

 \section    global      Global Path Planning

 Local collision avoidance as provided by <b>RVO2 Library</b> should in principle
 be accompanied by global path planning that determines the preferred velocity of
 each agent in each time step of the simulation. %RVO Library 1.x has a built-in
 roadmap infrastructure to guide agents around obstacles to fixed goals.
 However, besides roadmaps, other techniques for global planning, such as
 navigation fields, cell decompositions, etc. exist. Therefore, <b>RVO2
 Library</b> does not provide global planning infrastructure anymore. Instead,
 it is the responsibility of the external application to set the preferred
 velocity of each agent ahead of each time step of the simulation. This makes the
 library more flexible to use in varying application domains. In one of the
 example applications that comes with <b>RVO2 Library</b>, we show how a roadmap
 similar to %RVO Library 1.x is used externally to guide the global navigation of
 the agents. As a consequence of this change, <b>RVO2 Library</b> does not have a
 concept of a &quot;goal position&quot; or &quot;preferred speed&quot; for each
 agent, but only relies on the preferred velocities of the agents set by the
 external application.

 \section    structure   Structure of RVO2 Library

 The structure of <b>RVO2 Library</b> is similar to that of %RVO Library 1.x.
 Users familiar with %RVO Library 1.x should find little trouble in using <b>RVO2
 Library</b>. However, <b>RVO2 Library</b> is not backwards compatible with %RVO
 Library 1.x. The main reason for this is that the ORCA technique requires
 different (and fewer) parameters to be set than %RVO. Also, the way obstacles
 are represented is different. In %RVO Library 1.x, obstacles are represented by
 an arbitrary collection of line segments. In <b>RVO2 Library</b>, obstacles are
 non-intersecting polygons, specified by lists of vertices in counterclockwise
 order. Further, in %RVO Library 1.x agents cannot be added to the simulation
 after the simulation is initialized. In <b>RVO2 Library</b> this restriction is
 removed. Obstacles still need to be processed before the simulation starts,
 though. Lastly, in %RVO Library 1.x an instance of the simulator is a singleton.
 This restriction is removed in <b>RVO2 Library</b>.

 \section    smaller     Smaller Changes

 With <b>RVO2 Library</b>, we have adopted the philosophy that anything that is
 not part of the core local collision avoidance technique is to be stripped from
 the library. Therefore, besides the roadmap infrastructure, we have also removed
 acceleration constraints of agents, orientation of agents, and the unused
 &quot;class&quot; of agents. Each of these can be implemented external of the
 library if needed. We did maintain a <i>k</i>d-tree infrastructure for
 efficiently finding other agents and obstacles nearby each agent.

 Also, <b>RVO2 Library</b> allows accessing information about the simulation,
 such as the neighbors and the collision-avoidance constraints of each agent,
 that is hidden from the user in %RVO Library 1.x.


 \page       building    Building RVO2 Library

 We assume that you have downloaded <b>RVO2 Library</b> and unpacked the ZIP
 archive to a path <tt>$RVO_ROOT</tt>.

 \section    xcode       Apple Xcode 3.x

 Open <tt>$RVO_ROOT/RVO.xcodeproj</tt> and select the <tt>%RVO</tt> target and
 a configuration (<tt>Debug</tt> or <tt>Release</tt>). A framework
 <tt>RVO.framework</tt> will be built in the default build directory, e.g. <tt>
 $RVO_ROOT/build/Release</tt>.

 \section    cmake       CMake

 Create and switch to your chosen build directory, e.g. <tt>$RVO_ROOT/build</tt>.
 Run <tt>cmake</tt> inside the build directory on the source directory, e.g.
 <tt>cmake $RVO_ROOT/src</tt>. Build files for the default generator for your
 platform will be generated in the build directory.

 \section    make        GNU Make

 Switch to the source directory <tt>$RVO_ROOT/src</tt> and run <tt>make</tt>.
 Public header files (<tt>RVO.h</tt>, <tt>RVOSimulator.h</tt>, and
 <tt>Vector2.h</tt>) will be copied to the <tt>$RVO_ROOT/include</tt> directory
 and a static library <tt>libRVO.a</tt> will be compiled into the
 <tt>$RVO_ROOT/lib</tt> directory.

 \section    visual      Microsoft Visual Studio 2008

 Open <tt>$RVO_ROOT/RVO.sln</tt> and select the <tt>%RVO</tt> project and a
 configuration (<tt>Debug</tt>, <tt>ReleaseST</tt>, or <tt>ReleaseMT</tt>).
 Public header files (<tt>RVO.h</tt>, <tt>RVOSimulator.h</tt>, and
 <tt>Vector2.h</tt>) will be copied to the <tt>$RVO_ROOT/include</tt> directory
 and a static library, e.g. <tt>RVO.lib</tt>, will be compiled into the
 <tt>$RVO_ROOT/lib</tt> directory.


 \page       using       Using RVO2 Library

 \section    structure   Structure

 A program performing an <b>RVO2 Library</b> simulation has the following global
 structure.

 \code
 #include <RVO.h>

 std::vector<RVO::Vector2> goals;

 int main()
 {
 // Create a new simulator instance.
 RVO::RVOSimulator* sim = new RVO::RVOSimulator();

 // Set up the scenario.
 setupScenario(sim);

 // Perform (and manipulate) the simulation.
 do {
 updateVisualization(sim);
 setPreferredVelocities(sim);
 sim->doStep();
 } while (!reachedGoal(sim));

 delete sim;
 }
 \endcode

 In order to use <b>RVO2 Library</b>, the user needs to include RVO.h. The first
 step is then to create an instance of RVO::RVOSimulator. Then, the process
 consists of two stages. The first stage is specifying the simulation scenario
 and its parameters. In the above example program, this is done in the method
 setupScenario(...), which we will discuss below. The second stage is the actual
 performing of the simulation.

 In the above example program, simulation steps are taken until all
 the agents have reached some predefined goals. Prior to each simulation step,
 we set the preferred velocity for each agent, i.e. the
 velocity the agent would have taken if there were no other agents around, in the
 method setPreferredVelocities(...). The simulator computes the actual velocities
 of the agents and attempts to follow the preferred velocities as closely as
 possible while guaranteeing collision avoidance at the same time. During the
 simulation, the user may want to retrieve information from the simulation for
 instance to visualize the simulation. In the above example program, this is done
 in the method updateVisualization(...), which we will discuss below. It is also
 possible to manipulate the simulation during the simulation, for instance by
 changing positions, radii, velocities, etc. of the agents.

 \section    spec        Setting up the Simulation Scenario

 A scenario that is to be simulated can be set up as follows. A scenario consists
 of two types of objects: agents and obstacles. Each of them can be manually
 specified. Agents may be added anytime before or during the simulation.
 Obstacles, however, need to be defined prior to the simulation, and
 RVO::RVOSimulator::processObstacles() need to be called in order for the
 obstacles to be accounted for in the simulation.
 The user may also want to define goal positions of the agents, or a
 roadmap to guide the agents around obstacles. This is not done in <b>RVO2
 Library</b>, but needs to be taken care of in the user's external application.

 The following example creates a scenario with four agents exchanging positions
 around a rectangular obstacle in the middle.

 \code
 void setupScenario(RVO::RVOSimulator* sim) {
 // Specify global time step of the simulation.
 sim->setTimeStep(0.25f);

 // Specify default parameters for agents that are subsequently added.
 sim->setAgentDefaults(15.0f, 10, 10.0f, 5.0f, 2.0f, 2.0f);

 // Add agents, specifying their start position.
 sim->addAgent(RVO::Vector2(-50.0f, -50.0f));
 sim->addAgent(RVO::Vector2(50.0f, -50.0f));
 sim->addAgent(RVO::Vector2(50.0f, 50.0f));
 sim->addAgent(RVO::Vector2(-50.0f, 50.0f));

 // Create goals (simulator is unaware of these).
 for (size_t i = 0; i < sim->getNumAgents(); ++i) {
 goals.push_back(-sim->getAgentPosition(i));
 }

 // Add (polygonal) obstacle(s), specifying vertices in counterclockwise order.
 std::vector<RVO::Vector2> vertices;
 vertices.push_back(RVO::Vector2(-7.0f, -20.0f));
 vertices.push_back(RVO::Vector2(7.0f, -20.0f));
 vertices.push_back(RVO::Vector2(7.0f, 20.0f));
 vertices.push_back(RVO::Vector2(-7.0f, 20.0f));

 sim->addObstacle(vertices);

 // Process obstacles so that they are accounted for in the simulation.
 sim->processObstacles();
 }
 \endcode

 See the documentation on RVO::RVOSimulator for a full overview of the
 functionality to specify scenarios.

 \section    ret         Retrieving Information from the Simulation

 During the simulation, the user can extract information from the simulation for
 instance for visualization purposes, or to determine termination conditions of
 the simulation. In the example program above, visualization is done in the
 updateVisualization(...) method. Below we give an example that simply writes
 the positions of each agent in each time step to the standard output. The
 termination condition is checked in the reachedGoal(...) method. Here we give an
 example that returns true if all agents are within one radius of their goals.

 \code
 void updateVisualization(RVO::RVOSimulator* sim) {
 // Output the current global time.
 std::cout << sim->getGlobalTime() << " ";

 // Output the position for all the agents.
 for (size_t i = 0; i < sim->getNumAgents(); ++i) {
 std::cout << sim->getAgentPosition(i) << " ";
 }

 std::cout << std::endl;
 }
 \endcode

 \code
 bool reachedGoal(RVO::RVOSimulator* sim) {
 // Check whether all agents have arrived at their goals.
 for (size_t i = 0; i < sim->getNumAgents(); ++i) {
 if (absSq(goals[i] - sim->getAgentPosition(i)) > sim->getAgentRadius(i) * sim->getAgentRadius(i)) {
 // Agent is further away from its goal than one radius.
 return false;
 }
 }
 return true;
 }
 \endcode

 Using similar functions as the ones used in this example, the user can access
 information about other parameters of the agents, as well as the global
 parameters, and the obstacles. See the documentation of the class
 RVO::RVOSimulator for an exhaustive list of public functions for retrieving
 simulation information.

 \section    manip       Manipulating the Simulation

 During the simulation, the user can manipulate the simulation, for instance by
 changing the global parameters, or changing the parameters of the agents
 (potentially causing abrupt different behavior). It is also possible to give the
 agents a new position, which make them jump through the scene.
 New agents can be added to the simulation at any time, but it is not allowed to
 add obstacles to the simulation after they have been processed by calling
 RVO::RVOSimulator::processObstacles(). Also, it is impossible to change the
 position of the vertices of the obstacles.

 See the documentation of the class RVO::RVOSimulator for an exhaustive list of
 public functions for manipulating the simulation.

 To provide global guidance to the agents, the preferred velocities of the agents
 can be changed ahead of each simulation step. In the above example program, this
 happens in the method setPreferredVelocities(...). Here we give an example that
 simply sets the preferred velocity to the unit vector towards the agent's goal
 for each agent (i.e., the preferred speed is 1.0). Note that this may not give
 convincing results with respect to global navigation around the obstacles. For
 this a roadmap or other global planning techniques may be used (see one of the
 \ref example "example programs" that accompanies <b>RVO2 Library</b>).

 \code
 void setPreferredVelocities(RVO::RVOSimulator* sim) {
 // Set the preferred velocity for each agent.
 for (size_t i = 0; i < sim->getNumAgents(); ++i) {
 if (absSq(goals[i] - sim->getAgentPosition(i)) < sim->getAgentRadius(i) * sim->getAgentRadius(i) ) {
 // Agent is within one radius of its goal, set preferred velocity to zero
 sim->setAgentPrefVelocity(i, RVO::Vector2(0.0f, 0.0f));
 } else {
 // Agent is far away from its goal, set preferred velocity as unit vector towards agent's goal.
 sim->setAgentPrefVelocity(i, normalize(goals[i] - sim->getAgentPosition(i)));
 }
 }
 }
 \endcode

 \section    example     Example Programs

 <b>RVO2 Library</b> is accompanied by three example programs, which can be found in the
 <tt>$RVO_ROOT/examples</tt> directory. The examples are named ExampleBlocks, ExampleCircle, and
 ExampleRoadmap, and contain the following demonstration scenarios:
 <table border="0" cellpadding="3" width="100%">
 <tr>
 <td valign="top" width="100"><b>ExampleBlocks</b></td>
 <td valign="top">A scenario in which 100 agents, split in four groups initially
 positioned in each of four corners of the environment, move to the
 other side of the environment through a narrow passage generated by four
 obstacles. There is no roadmap to guide the agents around the obstacles.</td>
 </tr>
 <tr>
 <td valign="top" width="100"><b>ExampleCircle</b></td>
 <td valign="top">A scenario in which 250 agents, initially positioned evenly
 distributed on a circle, move to the antipodal position on the
 circle. There are no obstacles.</td>
 </tr>
 <tr>
 <td valign="top" width="100"><b>ExampleRoadmap</b></td>
 <td valign="top">The same scenario as ExampleBlocks, but now the preferred velocities
 of the agents are determined using a roadmap guiding the agents around the obstacles.</td>
 </tr>
 </table>


 \page       params      Parameter Overview

 \section    globalp     Global Parameters

 <table border="0" cellpadding="3" width="100%">
 <tr>
 <td valign="top" width="150"><strong>Parameter</strong></td>
 <td valign="top" width="150"><strong>Type (unit)</strong></td>
 <td valign="top"><strong>Meaning</strong></td>
 </tr>
 <tr>
 <td valign="top">timeStep</td>
 <td valign="top">float (time)</td>
 <td valign="top">The time step of the simulation. Must be positive.</td>
 </tr>
 </table>

 \section    agent       Agent Parameters

 <table border="0" cellpadding="3" width="100%">
 <tr>
 <td valign="top" width="150"><strong>Parameter</strong></td>
 <td valign="top" width="150"><strong>Type (unit)</strong></td>
 <td valign="top"><strong>Meaning</strong></td>
 </tr>
 <tr>
 <td valign="top">maxNeighbors</td>
 <td valign="top">size_t</td>
 <td valign="top">The maximum number of other agents the agent takes into
 account in the navigation. The larger this number, the
 longer the running time of the simulation. If the number is
 too low, the simulation will not be safe.</td>
 </tr>
 <tr>
 <td valign="top">maxSpeed</td>
 <td valign="top">float (distance/time)</td>
 <td valign="top">The maximum speed of the agent. Must be non-negative.</td>
 </tr>
 <tr>
 <td valign="top">neighborDist</td>
 <td valign="top">float (distance)</td>
 <td valign="top">The maximum distance (center point to center point) to
 other agents the agent takes into account in the
 navigation. The larger this number, the longer the running
 time of the simulation. If the number is too low, the
 simulation will not be safe. Must be non-negative.</td>
 </tr>
 <tr>
 <td valign="top" width="150">position</td>
 <td valign="top" width="150">RVO::Vector2 (distance, distance)</td>
 <td valign="top">The current position of the agent.</td>
 </tr>
 <tr>
 <td valign="top" width="150">prefVelocity</td>
 <td valign="top" width="150">RVO::Vector2 (distance/time, distance/time)
 </td>
 <td valign="top">The current preferred velocity of the agent. This is the
 velocity the agent would take if no other agents or
 obstacles were around. The simulator computes an actual
 velocity for the agent that follows the preferred velocity
 as closely as possible, but at the same time guarantees
 collision avoidance.</td>
 </tr>
 <tr>
 <td valign="top">radius</td>
 <td valign="top">float (distance)</td>
 <td valign="top">The radius of the agent. Must be non-negative.</td>
 </tr>
 <tr>
 <td valign="top" width="150">timeHorizon</td>
 <td valign="top" width="150">float (time)</td>
 <td valign="top">The minimal amount of time for which the agent's velocities
 that are computed by the simulation are safe with respect
 to other agents. The larger this number, the sooner this
 agent will respond to the presence of other agents, but the
 less freedom the agent has in choosing its velocities.
 Must be positive. </td>
 </tr>
 <tr>
 <td valign="top">timeHorizonObst</td>
 <td valign="top">float (time)</td>
 <td valign="top">The minimal amount of time for which the agent's velocities
 that are computed by the simulation are safe with respect
 to obstacles. The larger this number, the sooner this agent
 will respond to the presence of obstacles, but the less
 freedom the agent has in choosing its velocities.
 Must be positive. </td>
 </tr>
 <tr>
 <td valign="top" width="150">velocity</td>
 <td valign="top" width="150">RVO::Vector2 (distance/time, distance/time)
 </td>
 <td valign="top">The (current) velocity of the agent.</td>
 </tr>
 </table>


 \page       terms       Terms and Conditions

 <b>RVO2 Library</b>

 Copyright 2008 University of North Carolina at Chapel Hill

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 */

#endif /* RVO_RVO_H_ */
