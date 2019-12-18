/*************************************************************************/
/*  rvo_collision_avoidance_server.cpp                                   */
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

#include "rvo_collision_avoidance_server.h"

RvoCollisionAvoidanceServer::RvoCollisionAvoidanceServer() :
        CollisionAvoidanceServer() {
}

RvoCollisionAvoidanceServer::~RvoCollisionAvoidanceServer() {}

RID RvoCollisionAvoidanceServer::space_create() {
    RvoSpace *space = memnew(RvoSpace);
    RID rid = space_owner.make_rid(space);
    space->set_self(rid);
    return rid;
}

RID RvoCollisionAvoidanceServer::agent_add(RID p_space) {
    RvoSpace *space = space_owner.get(p_space);
    ERR_FAIL_COND_V(space == NULL, RID());

    RvoAgent *agent = memnew(RvoAgent);
    RID rid = agent_owner.make_rid(agent);
    agent->set_self(rid);

    space->add_agent(agent->get_agent());

    return rid;
}

void RvoCollisionAvoidanceServer::agent_set_neighbor_dist(RID p_agent, real_t p_dist) {
}

void RvoCollisionAvoidanceServer::agent_set_max_neighbors(RID p_agent, int p_count) {
}

void RvoCollisionAvoidanceServer::agent_set_time_horizon(RID p_agent, real_t p_time) {
}

void RvoCollisionAvoidanceServer::agent_set_time_horizon_obs(RID p_agent, real_t p_time) {
}

void RvoCollisionAvoidanceServer::agent_set_radius(RID p_agent, real_t p_radius) {
}

void RvoCollisionAvoidanceServer::agent_set_max_speed(RID p_agent, real_t p_max_speed) {
}

void RvoCollisionAvoidanceServer::agent_set_velocity(RID p_agent, Vector2 p_velocity) {
}

void RvoCollisionAvoidanceServer::agent_set_position(RID p_agent, Vector2 p_position) {
}

void RvoCollisionAvoidanceServer::agent_set_callback(RID p_agent) {
}

RID RvoCollisionAvoidanceServer::obstacle_add(RID p_space) {
    RvoObstacle *obstacle = memnew(RvoObstacle);
    RID rid = obstacle_owner.make_rid(obstacle);
    obstacle->set_self(rid);
    return rid;
}

void RvoCollisionAvoidanceServer::free(RID p_object) {
    if (space_owner.owns(p_object)) {
        RvoSpace *obj = space_owner.get(p_object);
        space_owner.free(p_object);
        memdelete(obj);
    } else if (agent_owner.owns(p_object)) {
        RvoAgent *obj = agent_owner.get(p_object);
        agent_owner.free(p_object);
        memdelete(obj);
    } else if (obstacle_owner.owns(p_object)) {
        RvoObstacle *obj = obstacle_owner.get(p_object);
        obstacle_owner.free(p_object);
        memdelete(obj);
    } else {
        ERR_FAIL_COND("Invalid ID.");
    }
}
