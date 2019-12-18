/*************************************************************************/
/*  rvo_collision_avoidance_server.h                                     */
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

#ifndef RVO_COLLISION_AVOIDANCE_SERVER_H
#define RVO_COLLISION_AVOIDANCE_SERVER_H

#include "rvo_agent.h"
#include "rvo_obstacle.h"
#include "rvo_space.h"
#include "servers/collision_avoidance_server.h"

class RvoCollisionAvoidanceServer : public CollisionAvoidanceServer {
    mutable RID_Owner<RvoSpace> space_owner;
    mutable RID_Owner<RvoAgent> agent_owner;
    mutable RID_Owner<RvoObstacle> obstacle_owner;

public:
    RvoCollisionAvoidanceServer();
    virtual ~RvoCollisionAvoidanceServer();

    virtual RID space_create();

    virtual RID agent_add(RID p_space);
    virtual void agent_set_neighbor_dist(RID p_agent, real_t p_dist);
    virtual void agent_set_max_neighbors(RID p_agent, int p_count);
    virtual void agent_set_time_horizon(RID p_agent, real_t p_time);
    virtual void agent_set_time_horizon_obs(RID p_agent, real_t p_time);
    virtual void agent_set_radius(RID p_agent, real_t p_radius);
    virtual void agent_set_max_speed(RID p_agent, real_t p_max_speed);
    virtual void agent_set_velocity(RID p_agent, Vector2 p_velocity);
    virtual void agent_set_position(RID p_agent, Vector2 p_position);
    virtual void agent_set_callback(RID p_agent);

    virtual RID obstacle_add(RID p_space);

    virtual void free(RID p_object);
};

#endif // RVO_COLLISION_AVOIDANCE_SERVER_H
