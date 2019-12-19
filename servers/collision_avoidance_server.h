/*************************************************************************/
/*  collision_avoidance_server.h                                         */
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

#ifndef COLLISION_AVOIDANCE_SERVER_H
#define COLLISION_AVOIDANCE_SERVER_H

#include "core/object.h"
#include "core/rid.h"

class CollisionAvoidanceServer : public Object {
    GDCLASS(CollisionAvoidanceServer, Object);

    static CollisionAvoidanceServer *singleton;

protected:
    static void _bind_methods();

public:
    static CollisionAvoidanceServer *get_singleton();

    virtual RID space_create() = 0;
    virtual void space_set_active(RID p_space, bool p_active) = 0;
    virtual bool space_is_active(RID p_space) const = 0;

    /**
     * Creates and add the agent to the space.
     */
    virtual RID agent_add(RID p_space) = 0;

    /*
     * The maximum distance (center point to
     * center point) to other agents this agent
     * takes into account in the navigation. The
     * larger this number, the longer the running
     * time of the simulation. If the number is too
     * low, the simulation will not be safe.
     * Must be non-negative.
     */
    virtual void agent_set_neighbor_dist(RID p_agent, real_t p_dist) = 0;

    /**
     * The maximum number of other agents this
     * agent takes into account in the navigation.
     * The larger this number, the longer the
     * running time of the simulation. If the
     * number is too low, the simulation will not
     * be safe.
     */
    virtual void agent_set_max_neighbors(RID p_agent, int p_count) = 0;

    /**
     * The minimal amount of time for which this
     * agent's velocities that are computed by the
     * simulation are safe with respect to other
     * agents. The larger this number, the sooner
     * this agent will respond to the presence of
     * other agents, but the less freedom this
     * agent has in choosing its velocities.
     * Must be positive.
     */
    virtual void agent_set_time_horizon(RID p_agent, real_t p_time) = 0;

    /**
     * The minimal amount of time for which this
     * agent's velocities that are computed by the
     * simulation are safe with respect to
     * obstacles. The larger this number, the
     * sooner this agent will respond to the
     * presence of obstacles, but the less freedom
     * this agent has in choosing its velocities.
     * Must be positive.
     */
    virtual void agent_set_time_horizon_obs(RID p_agent, real_t p_time) = 0;

    /**
     * The radius of this agent.
     * Must be non-negative.
     */
    virtual void agent_set_radius(RID p_agent, real_t p_radius) = 0;

    /**
     * The maximum speed of this agent.
     * Must be non-negative.
     */
    virtual void agent_set_max_speed(RID p_agent, real_t p_max_speed) = 0;

    /**
     * Current velocity of the agent
     */
    virtual void agent_set_velocity(RID p_agent, Vector2 p_velocity) = 0;

    /**
     * The new target velocity.
     */
    virtual void agent_set_target_velocity(RID p_agent, Vector2 p_velocity) = 0;

    /**
     * Position of the agent in world space.
     */
    virtual void agent_set_position(RID p_agent, Vector2 p_position) = 0;

    /**
     * Callback called at the end of the RVO process
     */
    virtual void agent_set_callback(RID p_agent, Object *p_receiver, const StringName &p_method, const Variant &p_udata = Variant()) = 0;

    virtual RID obstacle_add(RID p_space) = 0;

    virtual void free(RID p_object) = 0;

    virtual void set_active(bool p_active) = 0;
    virtual void step(real_t delta_time) = 0;

    CollisionAvoidanceServer();
    virtual ~CollisionAvoidanceServer();
};

typedef CollisionAvoidanceServer *(*CreateCollisionAvoidanceServerCallback)();

class CollisionAvoidanceServerManager {
    static CreateCollisionAvoidanceServerCallback create_callback;

public:
    static void set_default_server(CreateCollisionAvoidanceServerCallback p_callback);
    static CollisionAvoidanceServer *new_default_server();
};

#endif
