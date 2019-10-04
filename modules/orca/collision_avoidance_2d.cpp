/*************************************************************************/
/*  collision_avoidance_2d.cpp                                           */
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

#include "collision_avoidance_2d.h"

#include <Agent.h>
#include <Definitions.h>
#include <KdTree.h>
#include <Obstacle.h>
#include <Vector2.h>

RVO::KdTree *CollisionAvoidance2D::get_tree() {
	return &kd_tree;
}

void CollisionAvoidance2D::remove_bounds() {
	remove_obstacle(0);
	bounds = Rect2();
}

void CollisionAvoidance2D::set_bounds(Rect2 p_bounds) {
	bounds = p_bounds;
	if (p_bounds.has_no_area()) {
		remove_obstacle(0);
		return;
	}
	Vector<Vector2> points = Vector<Vector2>();
	Vector2 top_left = p_bounds.get_position();
	Vector2 bottom_right = p_bounds.get_size() + p_bounds.get_position();
	if (top_left.x > bottom_right.x) {
		float swap = top_left.x;
		top_left.x = bottom_right.x;
		bottom_right.x = swap;
	}
	if (top_left.y > bottom_right.y) {
		float swap = top_left.y;
		top_left.y = bottom_right.y;
		bottom_right.y = swap;
	}

	points.push_back(top_left);
	points.push_back(Vector2(top_left.x, bottom_right.y));
	points.push_back(bottom_right);
	points.push_back(Vector2(bottom_right.x, top_left.y));

	_add_obstacle(points, 0);
}

Rect2 CollisionAvoidance2D::get_bounds() {
	return bounds;
}

void CollisionAvoidance2D::move_obstacle(int p_obstacle, Vector<Vector2> p_obstacle_points) {
	remove_obstacle(p_obstacle);
	_add_obstacle(p_obstacle_points, p_obstacle);
}

void CollisionAvoidance2D::_add_obstacle(Vector<Vector2> p_obstacle_points, int p_obstacle) {
	Vector<RVO::Obstacle *> obstacles = Vector<RVO::Obstacle *>();
	obstacles.resize(p_obstacle_points.size());

	for (int i = 0; i < p_obstacle_points.size(); i++) {
		RVO::Obstacle *point = new RVO::Obstacle();
		point->point_ = RVO::Vector2(p_obstacle_points[i].x, p_obstacle_points[i].y);

		// RVO expects the points in counterclockwise order
		obstacles.set(p_obstacle_points.size() - i - 1, point);
	}

	for (int i = 0; i < obstacles.size(); i++) {
		RVO::Obstacle *point = obstacles[i];
		point->id_ = p_obstacle;
		point->prevObstacle_ = obstacles[i > 0 ? i - 1 : obstacles.size() - 1];
		point->nextObstacle_ = obstacles[i < obstacles.size() - 1 ? i + 1 : 0];
		point->unitDir_ = RVO::normalize(point->point_ - point->nextObstacle_->point_);
		if (obstacles.size() > 2) {
			point->isConvex_ = RVO::leftOf(point->prevObstacle_->point_, point->point_, point->nextObstacle_->point_);
		} else {
			point->isConvex_ = true;
		}
	}

	obstacles_dirty = true;
	obstacles_map[p_obstacle] = obstacles;
}

int CollisionAvoidance2D::add_obstacle(Vector<Vector2> p_obstacle_points) {
	if (p_obstacle_points.size() < 2) {
		return -1;
	}

	++last_obstacle_id;

	_add_obstacle(p_obstacle_points, last_obstacle_id);

	return last_obstacle_id;
}

void CollisionAvoidance2D::remove_obstacle(int p_obstacle) {
	Vector<RVO::Obstacle *> points = obstacles_map[p_obstacle];
	for (int i = 0; i < points.size(); i++) {
		delete points[i];
	}

	obstacles_map.erase(p_obstacle);
	obstacles_dirty = true;
}

int CollisionAvoidance2D::add_actor(CollisionAvoidanceActor2D *p_actor) {
	++last_actor_id;
	actor_map[last_actor_id] = p_actor;
	return last_actor_id;
}

void CollisionAvoidance2D::remove_actor(int p_id) {
	actor_map.erase(p_id);
}

void CollisionAvoidance2D::navigate_actors() {
	auto e = actor_map.front();
	while (e != NULL) {
		e->get()->sync_position();
		e = e->next();
	}

	if (obstacles_dirty) {
		std::vector<RVO::Obstacle *> obstacles = std::vector<RVO::Obstacle *>();
		auto o = obstacles_map.front();
		while (o != NULL) {
			Vector<RVO::Obstacle *> obstacle = o->get();
			for (int i = 0; i < obstacle.size(); i++) {
				obstacles.push_back(obstacle[i]);
			}
			o = o->next();
		}

		int previous_size = obstacles.size();
		kd_tree.buildObstacleTree(obstacles);
		for (long unsigned int i = previous_size; i < obstacles.size(); i++) {
			obstacles_map[obstacles[i]->id_].push_back(obstacles[i]);
		}
		obstacles_dirty = false;
	}

	auto agents = std::vector<RVO::Agent *>();
	e = actor_map.front();
	while (e != NULL) {
		agents.push_back(e->get()->get_agent());
		e = e->next();
	}
	kd_tree.buildAgentTree(agents);
}

void CollisionAvoidance2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bounds", "bounds"), &CollisionAvoidance2D::set_bounds);
	ClassDB::bind_method(D_METHOD("get_bounds"), &CollisionAvoidance2D::get_bounds);
	ClassDB::bind_method(D_METHOD("remove_bounds"), &CollisionAvoidance2D::remove_bounds);
	ClassDB::bind_method(D_METHOD("add_obstacle", "points"), &CollisionAvoidance2D::add_obstacle);
	ClassDB::bind_method(D_METHOD("move_obstacle", "obstacle_id", "points"), &CollisionAvoidance2D::move_obstacle);
	ClassDB::bind_method(D_METHOD("remove_obstacle", "obstacle_id"), &CollisionAvoidance2D::remove_obstacle);
	ClassDB::bind_method(D_METHOD("navigate_actors"), &CollisionAvoidance2D::navigate_actors);
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "bounds"), "set_bounds", "get_bounds");
}

CollisionAvoidance2D::CollisionAvoidance2D() :
		processed_obstacles(false),
		actor_map(),
		obstacles_map(),
		bounds(),
		kd_tree(),
		last_obstacle_id(0),
		last_actor_id(0),
		obstacles_dirty(false) {
}
