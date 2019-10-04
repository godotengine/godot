/*************************************************************************/
/*  collision_avoidance_2d.h                                             */
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

#ifndef COLLISION_AVOIDANCE_2D_H
#define COLLISION_AVOIDANCE_2D_H

#include "collision_avoidance_actor_2d.h"
#include "scene/2d/node_2d.h"

#include <KdTree.h>
#include <Obstacle.h>

class CollisionAvoidance2D : public Node2D {
	GDCLASS(CollisionAvoidance2D, Node2D);

private:
	bool processed_obstacles;
	Map<int, CollisionAvoidanceActor2D *> actor_map;
	Map<int, Vector<RVO::Obstacle *> > obstacles_map;
	Rect2 bounds;
	RVO::KdTree kd_tree;
	int last_obstacle_id;
	int last_actor_id;
	bool obstacles_dirty;

	void _add_obstacle(Vector<Vector2> p_obstacle_points, int p_obstacle);

protected:
	static void _bind_methods();

public:
	RVO::KdTree *get_tree();

	void set_bounds(Rect2 p_bounds);
	Rect2 get_bounds();
	void remove_bounds();

	int add_obstacle(Vector<Vector2> p_obstacle_points);
	void move_obstacle(int p_obstacle, Vector<Vector2> p_obstacle_points);
	void remove_obstacle(int p_obstacle);

	int add_actor(CollisionAvoidanceActor2D *p_actor);
	void remove_actor(int p_id);

	void navigate_actors();

	CollisionAvoidance2D();
};

#endif
