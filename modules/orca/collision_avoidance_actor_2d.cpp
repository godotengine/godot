/*************************************************************************/
/*  collision_avoidance_actor_2d.cpp                                     */
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

#include "collision_avoidance_actor_2d.h"
#include "collision_avoidance_2d.h"

void CollisionAvoidanceActor2D::set_radius(float p_radius) {
	agent.radius_ = p_radius;
}
float CollisionAvoidanceActor2D::get_radius() {
	return agent.radius_;
}

void CollisionAvoidanceActor2D::set_obstacle_time_horizon(float p_horizon) {
	agent.timeHorizonObst_ = p_horizon;
}
float CollisionAvoidanceActor2D::get_obstacle_time_horizon() {
	return agent.timeHorizonObst_;
}

void CollisionAvoidanceActor2D::set_neighbor_time_horizon(float p_horizon) {
	agent.timeHorizon_ = p_horizon;
}
float CollisionAvoidanceActor2D::get_neighbor_time_horizon() {
	return agent.timeHorizon_;
}

void CollisionAvoidanceActor2D::set_max_neighbors(int p_neighbors) {
	agent.maxNeighbors_ = p_neighbors;
}
int CollisionAvoidanceActor2D::get_max_neighbors() {
	return agent.maxNeighbors_;
}

void CollisionAvoidanceActor2D::set_neighbor_search_distance(float p_search_distance) {
	agent.neighborDist_ = p_search_distance;
}
float CollisionAvoidanceActor2D::get_neighbor_search_distance() {
	return agent.neighborDist_;
}

void CollisionAvoidanceActor2D::set_max_speed(float p_speed) {
	agent.maxSpeed_ = p_speed;
}
float CollisionAvoidanceActor2D::get_max_speed() {
	return agent.maxSpeed_;
}

void CollisionAvoidanceActor2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Node2D *c = this;
			while (c) {

				parent = Object::cast_to<CollisionAvoidance2D>(c);
				if (parent) {
					id = parent->add_actor(this);
					break;
				}

				c = Object::cast_to<Node2D>(c->get_parent());
				Vector2 relative_position = get_relative_transform().get_origin();
				agent.position_ = RVO::Vector2(relative_position.x, relative_position.y);
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (parent != NULL) {
				parent->remove_actor(id);
			}
		}
	}
}

Transform2D CollisionAvoidanceActor2D::get_relative_transform() {
	if (parent == NULL) {
		return Transform2D();
	}
	Transform2D transform = get_transform();
	Node *c = get_parent();
	while (c != parent) {
		c = c->get_parent();
		transform *= Object::cast_to<Node2D>(c)->get_transform();
	}
	return transform;
}

void CollisionAvoidanceActor2D::sync_position() {
	Vector2 relative_position = get_relative_transform().get_origin();
	agent.position_ = RVO::Vector2(relative_position.x, relative_position.y);
}

Vector2 CollisionAvoidanceActor2D::calculate_velocity(Vector2 p_preferred_velocity, float p_delta) {
	if (parent == NULL) {
		return p_preferred_velocity;
	}

	agent.prefVelocity_ = RVO::Vector2(p_preferred_velocity.x, p_preferred_velocity.y);
	agent.computeNeighbors(parent->get_tree());
	agent.computeNewVelocity(p_delta);
	agent.velocity_ = agent.newVelocity_;
	return Vector2(agent.newVelocity_.x(), agent.newVelocity_.y());
}

RVO::Agent *CollisionAvoidanceActor2D::get_agent() {
	return &agent;
}

void CollisionAvoidanceActor2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CollisionAvoidanceActor2D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CollisionAvoidanceActor2D::get_radius);
	ClassDB::bind_method(D_METHOD("set_obstacle_time_horizon", "horizon"), &CollisionAvoidanceActor2D::set_obstacle_time_horizon);
	ClassDB::bind_method(D_METHOD("get_obstacle_time_horizon"), &CollisionAvoidanceActor2D::get_obstacle_time_horizon);
	ClassDB::bind_method(D_METHOD("set_neighbor_time_horizon", "horizon"), &CollisionAvoidanceActor2D::set_neighbor_time_horizon);
	ClassDB::bind_method(D_METHOD("get_neighbor_time_horizon"), &CollisionAvoidanceActor2D::get_neighbor_time_horizon);
	ClassDB::bind_method(D_METHOD("set_max_neighbors", "neighbors"), &CollisionAvoidanceActor2D::set_max_neighbors);
	ClassDB::bind_method(D_METHOD("get_max_neighbors"), &CollisionAvoidanceActor2D::get_max_neighbors);
	ClassDB::bind_method(D_METHOD("set_neighbor_search_distance", "distance"), &CollisionAvoidanceActor2D::set_neighbor_search_distance);
	ClassDB::bind_method(D_METHOD("get_neighbor_search_distance"), &CollisionAvoidanceActor2D::get_neighbor_search_distance);
	ClassDB::bind_method(D_METHOD("set_max_speed", "speed"), &CollisionAvoidanceActor2D::set_max_speed);
	ClassDB::bind_method(D_METHOD("get_max_speed"), &CollisionAvoidanceActor2D::get_max_speed);
	ClassDB::bind_method(D_METHOD("calculate_velocity", "preferred_velocity", "delta"), &CollisionAvoidanceActor2D::calculate_velocity);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_speed"), "set_max_speed", "get_max_speed");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "obstacle_time_horizon"), "set_obstacle_time_horizon", "get_obstacle_time_horizon");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "neighbor_time_horizon"), "set_neighbor_time_horizon", "get_neighbor_time_horizon");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_neighbors"), "set_max_neighbors", "get_max_neighbors");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "neighbor_search_distance"), "set_neighbor_search_distance", "get_neighbor_search_distance");

	ADD_PROPERTY_DEFAULT("max_speed", 100);
	ADD_PROPERTY_DEFAULT("radius", 5);
	ADD_PROPERTY_DEFAULT("obstacle_time_horizon", 30);
	ADD_PROPERTY_DEFAULT("neighbor_time_horizon", 10);
	ADD_PROPERTY_DEFAULT("max_neighbors", 20);
	ADD_PROPERTY_DEFAULT("neighbor_search_distance", 200);
}

CollisionAvoidanceActor2D::CollisionAvoidanceActor2D() :
		id(0),
		agent(),
		parent(NULL) {
	agent.maxSpeed_ = 100;
	agent.radius_ = 5;
	agent.timeHorizonObst_ = 30;
	agent.timeHorizon_ = 10;
	agent.maxNeighbors_ = 20;
	agent.neighborDist_ = 200;
}
