/*************************************************************************/
/*  collision_avoidance_actor_2d.h                                       */
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

#ifndef COLLISION_AVOIDANCE_ACTOR_2D_H
#define COLLISION_AVOIDANCE_ACTOR_2D_H

#include "scene/2d/node_2d.h"

#include <Agent.h>

class CollisionAvoidance2D;

class CollisionAvoidanceActor2D : public Node2D {
	GDCLASS(CollisionAvoidanceActor2D, Node2D);

private:
	int id;
	RVO::Agent agent;
	CollisionAvoidance2D *parent;

protected:
	static void _bind_methods();

public:
	void set_radius(float p_radius);
	float get_radius();

	void set_obstacle_time_horizon(float p_horizon);
	float get_obstacle_time_horizon();

	void set_neighbor_time_horizon(float p_horizon);
	float get_neighbor_time_horizon();

	void set_max_neighbors(int p_neighbors);
	int get_max_neighbors();

	void set_neighbor_search_distance(float p_search_distance);
	float get_neighbor_search_distance();

	void set_max_speed(float p_speed);
	float get_max_speed();

	Transform2D get_relative_transform();

	void _notification(int p_what);

	void sync_position();
	Vector2 calculate_velocity(Vector2 p_preferred_velocity, float p_delta);

	RVO::Agent *get_agent();

	CollisionAvoidanceActor2D();
};

#endif
