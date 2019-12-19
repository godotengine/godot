/*************************************************************************/
/*  rvo_space.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "rvo_space.h"

#include "rvo_agent.h"

RvoSpace::RvoSpace() {
}

bool RvoSpace::has_obstacle(RVO::Obstacle *obstacle) const {
    return std::find(obstacles.begin(), obstacles.end(), obstacle) != obstacles.end();
}

void RvoSpace::add_obstacle(RVO::Obstacle *obstacle) {
    if (!has_obstacle(obstacle)) {
        obstacles.push_back(obstacle);
        obstacles_dirty = true;
    }
}

void RvoSpace::remove_obstacle(RVO::Obstacle *obstacle) {
    auto it = std::find(obstacles.begin(), obstacles.end(), obstacle);
    if (it != obstacles.end()) {
        obstacles.erase(it);
        obstacles_dirty = true;
    }
}

bool RvoSpace::has_agent(RvoAgent *agent) const {
    return std::find(agents.begin(), agents.end(), agent->get_agent()) != agents.end();
}

void RvoSpace::add_agent(RvoAgent *agent) {
    if (!has_agent(agent)) {
        agents.push_back(agent->get_agent());
        agents_dirty = true;
    }
}

void RvoSpace::remove_agent(RvoAgent *agent) {
    remove_agent_as_controlled(agent);
    auto it = std::find(agents.begin(), agents.end(), agent->get_agent());
    if (it != agents.end()) {
        agents.erase(it);
        agents_dirty = true;
    }
}

void RvoSpace::set_agent_as_controlled(RvoAgent *agent) {
    const bool exist = std::find(controlled_agents.begin(), controlled_agents.end(), agent) != controlled_agents.end();
    if (!exist) {
        ERR_FAIL_COND(!has_agent(agent));
        controlled_agents.push_back(agent);
    }
}

void RvoSpace::remove_agent_as_controlled(RvoAgent *agent) {
    auto it = std::find(controlled_agents.begin(), controlled_agents.end(), agent);
    if (it != controlled_agents.end()) {
        controlled_agents.erase(it);
    }
}

void RvoSpace::sync() {
    if (obstacles_dirty) {
        rvo.buildObstacleTree(obstacles);
        obstacles_dirty = false;
    }

    if (agents_dirty) {
        rvo.buildAgentTree(agents);
        agents_dirty = false;
    }
}

void RvoSpace::step(real_t timestep) {
    // TODO Please do this in MT
    for (int i(0); i < static_cast<int>(controlled_agents.size()); i++) {
        controlled_agents[i]->get_agent()->computeNeighbors(&rvo);
        controlled_agents[i]->get_agent()->computeNewVelocity(timestep);
    }
}

void RvoSpace::dispatch_callbacks() {
    for (int i(0); i < static_cast<int>(controlled_agents.size()); i++) {
        controlled_agents[i]->dispatch_callback();
    }
}
