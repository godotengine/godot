/*************************************************************************/
/*  navigation_agent.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "navigation_agent.h"

#include "core/engine.h"
#include "scene/3d/navigation.h"
#include "servers/navigation_server.h"

void NavigationAgent::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_desired_distance", "desired_distance"), &NavigationAgent::set_target_desired_distance);
	ClassDB::bind_method(D_METHOD("get_target_desired_distance"), &NavigationAgent::get_target_desired_distance);

	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationAgent::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationAgent::get_radius);

	ClassDB::bind_method(D_METHOD("set_agent_height_offset", "agent_height_offset"), &NavigationAgent::set_agent_height_offset);
	ClassDB::bind_method(D_METHOD("get_agent_height_offset"), &NavigationAgent::get_agent_height_offset);

	ClassDB::bind_method(D_METHOD("set_ignore_y", "ignore"), &NavigationAgent::set_ignore_y);
	ClassDB::bind_method(D_METHOD("get_ignore_y"), &NavigationAgent::get_ignore_y);

	ClassDB::bind_method(D_METHOD("set_navigation", "navigation"), &NavigationAgent::set_navigation_node);
	ClassDB::bind_method(D_METHOD("get_navigation"), &NavigationAgent::get_navigation_node);

	ClassDB::bind_method(D_METHOD("set_neighbor_dist", "neighbor_dist"), &NavigationAgent::set_neighbor_dist);
	ClassDB::bind_method(D_METHOD("get_neighbor_dist"), &NavigationAgent::get_neighbor_dist);

	ClassDB::bind_method(D_METHOD("set_max_neighbors", "max_neighbors"), &NavigationAgent::set_max_neighbors);
	ClassDB::bind_method(D_METHOD("get_max_neighbors"), &NavigationAgent::get_max_neighbors);

	ClassDB::bind_method(D_METHOD("set_time_horizon", "time_horizon"), &NavigationAgent::set_time_horizon);
	ClassDB::bind_method(D_METHOD("get_time_horizon"), &NavigationAgent::get_time_horizon);

	ClassDB::bind_method(D_METHOD("set_max_speed", "max_speed"), &NavigationAgent::set_max_speed);
	ClassDB::bind_method(D_METHOD("get_max_speed"), &NavigationAgent::get_max_speed);

	ClassDB::bind_method(D_METHOD("set_path_max_distance", "max_speed"), &NavigationAgent::set_path_max_distance);
	ClassDB::bind_method(D_METHOD("get_path_max_distance"), &NavigationAgent::get_path_max_distance);

	ClassDB::bind_method(D_METHOD("set_target_location", "location"), &NavigationAgent::set_target_location);
	ClassDB::bind_method(D_METHOD("get_target_location"), &NavigationAgent::get_target_location);
	ClassDB::bind_method(D_METHOD("get_next_location"), &NavigationAgent::get_next_location);
	ClassDB::bind_method(D_METHOD("distance_to_target"), &NavigationAgent::distance_to_target);
	ClassDB::bind_method(D_METHOD("set_velocity", "velocity"), &NavigationAgent::set_velocity);
	ClassDB::bind_method(D_METHOD("get_nav_path"), &NavigationAgent::get_nav_path);
	ClassDB::bind_method(D_METHOD("get_nav_path_index"), &NavigationAgent::get_nav_path_index);
	ClassDB::bind_method(D_METHOD("is_target_reached"), &NavigationAgent::is_target_reached);
	ClassDB::bind_method(D_METHOD("is_target_reachable"), &NavigationAgent::is_target_reachable);
	ClassDB::bind_method(D_METHOD("is_navigation_finished"), &NavigationAgent::is_navigation_finished);
	ClassDB::bind_method(D_METHOD("get_final_location"), &NavigationAgent::get_final_location);

	ClassDB::bind_method(D_METHOD("_avoidance_done", "new_velocity"), &NavigationAgent::_avoidance_done);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "target_desired_distance", PROPERTY_HINT_RANGE, "0.1,100,0.01"), "set_target_desired_distance", "get_target_desired_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.1,100,0.01"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "agent_height_offset", PROPERTY_HINT_RANGE, "-100.0,100,0.01"), "set_agent_height_offset", "get_agent_height_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "neighbor_dist", PROPERTY_HINT_RANGE, "0.1,10000,0.01"), "set_neighbor_dist", "get_neighbor_dist");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_neighbors", PROPERTY_HINT_RANGE, "1,10000,1"), "set_max_neighbors", "get_max_neighbors");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "time_horizon", PROPERTY_HINT_RANGE, "0.01,100,0.01"), "set_time_horizon", "get_time_horizon");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_speed", PROPERTY_HINT_RANGE, "0.1,10000,0.01"), "set_max_speed", "get_max_speed");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "path_max_distance", PROPERTY_HINT_RANGE, "0.01,100,0.1"), "set_path_max_distance", "get_path_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_y"), "set_ignore_y", "get_ignore_y");

	ADD_SIGNAL(MethodInfo("path_changed"));
	ADD_SIGNAL(MethodInfo("target_reached"));
	ADD_SIGNAL(MethodInfo("navigation_finished"));
	ADD_SIGNAL(MethodInfo("velocity_computed", PropertyInfo(Variant::VECTOR3, "safe_velocity")));
}

void NavigationAgent::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			agent_parent = Object::cast_to<Spatial>(get_parent());

			NavigationServer::get_singleton()->agent_set_callback(agent, this, "_avoidance_done");

			// Search the navigation node and set it
			{
				Navigation *nav = nullptr;
				Node *p = get_parent();
				while (p != nullptr) {
					nav = Object::cast_to<Navigation>(p);
					if (nav != nullptr) {
						p = nullptr;
					} else {
						p = p->get_parent();
					}
				}

				set_navigation(nav);
			}

			set_physics_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			agent_parent = nullptr;
			set_navigation(nullptr);
			set_physics_process_internal(false);
			// Want to call ready again when the node enters the tree again. We're not using enter_tree notification because
			// the navigation map may not be ready at that time. This fixes issues with taking the agent out of the scene tree.
			request_ready();
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (agent_parent) {
				NavigationServer::get_singleton()->agent_set_position(agent, agent_parent->get_global_transform().origin);
				_check_distance_to_target();
			}
		} break;
	}
}

NavigationAgent::NavigationAgent() :
		agent_parent(nullptr),
		navigation(nullptr),
		agent(RID()),
		target_desired_distance(1.0),
		navigation_height_offset(0.0),
		path_max_distance(3.0),
		velocity_submitted(false),
		target_reached(false),
		navigation_finished(true) {
	agent = NavigationServer::get_singleton()->agent_create();
	set_neighbor_dist(50.0);
	set_max_neighbors(10);
	set_time_horizon(5.0);
	set_radius(1.0);
	set_max_speed(10.0);
	set_ignore_y(true);
}

NavigationAgent::~NavigationAgent() {
	NavigationServer::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

void NavigationAgent::set_navigation(Navigation *p_nav) {
	if (navigation == p_nav) {
		return; // Pointless
	}

	navigation = p_nav;
	NavigationServer::get_singleton()->agent_set_map(agent, navigation == nullptr ? RID() : navigation->get_rid());
}

void NavigationAgent::set_navigation_node(Node *p_nav) {
	Navigation *nav = Object::cast_to<Navigation>(p_nav);
	ERR_FAIL_NULL(nav);
	set_navigation(nav);
}

Node *NavigationAgent::get_navigation_node() const {
	return Object::cast_to<Node>(navigation);
}

void NavigationAgent::set_target_desired_distance(real_t p_dd) {
	target_desired_distance = p_dd;
}

void NavigationAgent::set_radius(real_t p_radius) {
	radius = p_radius;
	NavigationServer::get_singleton()->agent_set_radius(agent, radius);
}

void NavigationAgent::set_agent_height_offset(real_t p_hh) {
	navigation_height_offset = p_hh;
}

void NavigationAgent::set_ignore_y(bool p_ignore_y) {
	ignore_y = p_ignore_y;
	NavigationServer::get_singleton()->agent_set_ignore_y(agent, ignore_y);
}

void NavigationAgent::set_neighbor_dist(real_t p_dist) {
	neighbor_dist = p_dist;
	NavigationServer::get_singleton()->agent_set_neighbor_dist(agent, neighbor_dist);
}

void NavigationAgent::set_max_neighbors(int p_count) {
	max_neighbors = p_count;
	NavigationServer::get_singleton()->agent_set_max_neighbors(agent, max_neighbors);
}

void NavigationAgent::set_time_horizon(real_t p_time) {
	time_horizon = p_time;
	NavigationServer::get_singleton()->agent_set_time_horizon(agent, time_horizon);
}

void NavigationAgent::set_max_speed(real_t p_max_speed) {
	max_speed = p_max_speed;
	NavigationServer::get_singleton()->agent_set_max_speed(agent, max_speed);
}

void NavigationAgent::set_path_max_distance(real_t p_pmd) {
	path_max_distance = p_pmd;
}

real_t NavigationAgent::get_path_max_distance() {
	return path_max_distance;
}

void NavigationAgent::set_target_location(Vector3 p_location) {
	target_location = p_location;
	navigation_path.clear();
	target_reached = false;
	navigation_finished = false;
	update_frame_id = 0;
}

Vector3 NavigationAgent::get_target_location() const {
	return target_location;
}

Vector3 NavigationAgent::get_next_location() {
	update_navigation();
	if (navigation_path.size() == 0) {
		ERR_FAIL_COND_V(agent_parent == nullptr, Vector3());
		return agent_parent->get_global_transform().origin;
	} else {
		return navigation_path[nav_path_index] - Vector3(0, navigation_height_offset, 0);
	}
}

real_t NavigationAgent::distance_to_target() const {
	ERR_FAIL_COND_V(agent_parent == nullptr, 0.0);
	return agent_parent->get_global_transform().origin.distance_to(target_location);
}

bool NavigationAgent::is_target_reached() const {
	return target_reached;
}

bool NavigationAgent::is_target_reachable() {
	return target_desired_distance >= get_final_location().distance_to(target_location);
}

bool NavigationAgent::is_navigation_finished() {
	update_navigation();
	return navigation_finished;
}

Vector3 NavigationAgent::get_final_location() {
	update_navigation();
	if (navigation_path.size() == 0) {
		return Vector3();
	}
	return navigation_path[navigation_path.size() - 1];
}

void NavigationAgent::set_velocity(Vector3 p_velocity) {
	target_velocity = p_velocity;
	NavigationServer::get_singleton()->agent_set_target_velocity(agent, target_velocity);
	NavigationServer::get_singleton()->agent_set_velocity(agent, prev_safe_velocity);
	velocity_submitted = true;
}

void NavigationAgent::_avoidance_done(Vector3 p_new_velocity) {
	prev_safe_velocity = p_new_velocity;

	if (!velocity_submitted) {
		target_velocity = Vector3();
		return;
	}
	velocity_submitted = false;

	emit_signal("velocity_computed", p_new_velocity);
}

String NavigationAgent::get_configuration_warning() const {
	if (!Object::cast_to<Spatial>(get_parent())) {
		return TTR("The NavigationAgent can be used only under a spatial node.");
	}

	return String();
}

void NavigationAgent::update_navigation() {
	if (agent_parent == nullptr) {
		return;
	}
	if (navigation == nullptr) {
		return;
	}
	if (update_frame_id == Engine::get_singleton()->get_physics_frames()) {
		return;
	}

	update_frame_id = Engine::get_singleton()->get_physics_frames();

	Vector3 o = agent_parent->get_global_transform().origin;

	bool reload_path = false;

	if (NavigationServer::get_singleton()->agent_is_map_changed(agent)) {
		reload_path = true;
	} else if (navigation_path.size() == 0) {
		reload_path = true;
	} else {
		// Check if too far from the navigation path
		if (nav_path_index > 0) {
			Vector3 segment[2];
			segment[0] = navigation_path[nav_path_index - 1];
			segment[1] = navigation_path[nav_path_index];
			segment[0].y -= navigation_height_offset;
			segment[1].y -= navigation_height_offset;
			Vector3 p = Geometry::get_closest_point_to_segment(o, segment);
			if (o.distance_to(p) >= path_max_distance) {
				// To faraway, reload path
				reload_path = true;
			}
		}
	}

	if (reload_path) {
		navigation_path = NavigationServer::get_singleton()->map_get_path(navigation->get_rid(), o, target_location, true);
		navigation_finished = false;
		nav_path_index = 0;
		emit_signal("path_changed");
	}

	if (navigation_path.size() == 0) {
		return;
	}

	// Check if we can advance the navigation path
	if (navigation_finished == false) {
		// Advances to the next far away location.
		while (o.distance_to(navigation_path[nav_path_index] - Vector3(0, navigation_height_offset, 0)) < target_desired_distance) {
			nav_path_index += 1;
			if (nav_path_index == navigation_path.size()) {
				_check_distance_to_target();
				nav_path_index -= 1;
				navigation_finished = true;
				emit_signal("navigation_finished");
				break;
			}
		}
	}
}

void NavigationAgent::_check_distance_to_target() {
	if (!target_reached) {
		if (distance_to_target() < target_desired_distance) {
			target_reached = true;
			emit_signal("target_reached");
		}
	}
}
