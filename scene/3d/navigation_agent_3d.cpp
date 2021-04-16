/*************************************************************************/
/*  navigation_agent_3d.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "navigation_agent_3d.h"

#include "core/config/engine.h"
#include "servers/navigation_server_3d.h"

void NavigationAgent3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &NavigationAgent3D::get_rid);

	ClassDB::bind_method(D_METHOD("set_target_desired_distance", "desired_distance"), &NavigationAgent3D::set_target_desired_distance);
	ClassDB::bind_method(D_METHOD("get_target_desired_distance"), &NavigationAgent3D::get_target_desired_distance);

	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationAgent3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationAgent3D::get_radius);

	ClassDB::bind_method(D_METHOD("set_agent_height_offset", "agent_height_offset"), &NavigationAgent3D::set_agent_height_offset);
	ClassDB::bind_method(D_METHOD("get_agent_height_offset"), &NavigationAgent3D::get_agent_height_offset);

	ClassDB::bind_method(D_METHOD("set_ignore_y", "ignore"), &NavigationAgent3D::set_ignore_y);
	ClassDB::bind_method(D_METHOD("get_ignore_y"), &NavigationAgent3D::get_ignore_y);

	ClassDB::bind_method(D_METHOD("set_neighbor_dist", "neighbor_dist"), &NavigationAgent3D::set_neighbor_dist);
	ClassDB::bind_method(D_METHOD("get_neighbor_dist"), &NavigationAgent3D::get_neighbor_dist);

	ClassDB::bind_method(D_METHOD("set_max_neighbors", "max_neighbors"), &NavigationAgent3D::set_max_neighbors);
	ClassDB::bind_method(D_METHOD("get_max_neighbors"), &NavigationAgent3D::get_max_neighbors);

	ClassDB::bind_method(D_METHOD("set_time_horizon", "time_horizon"), &NavigationAgent3D::set_time_horizon);
	ClassDB::bind_method(D_METHOD("get_time_horizon"), &NavigationAgent3D::get_time_horizon);

	ClassDB::bind_method(D_METHOD("set_max_speed", "max_speed"), &NavigationAgent3D::set_max_speed);
	ClassDB::bind_method(D_METHOD("get_max_speed"), &NavigationAgent3D::get_max_speed);

	ClassDB::bind_method(D_METHOD("set_path_max_distance", "max_speed"), &NavigationAgent3D::set_path_max_distance);
	ClassDB::bind_method(D_METHOD("get_path_max_distance"), &NavigationAgent3D::get_path_max_distance);

	ClassDB::bind_method(D_METHOD("set_target_location", "location"), &NavigationAgent3D::set_target_location);
	ClassDB::bind_method(D_METHOD("get_target_location"), &NavigationAgent3D::get_target_location);
	ClassDB::bind_method(D_METHOD("get_next_location"), &NavigationAgent3D::get_next_location);
	ClassDB::bind_method(D_METHOD("distance_to_target"), &NavigationAgent3D::distance_to_target);
	ClassDB::bind_method(D_METHOD("set_velocity", "velocity"), &NavigationAgent3D::set_velocity);
	ClassDB::bind_method(D_METHOD("get_nav_path"), &NavigationAgent3D::get_nav_path);
	ClassDB::bind_method(D_METHOD("get_nav_path_index"), &NavigationAgent3D::get_nav_path_index);
	ClassDB::bind_method(D_METHOD("is_target_reached"), &NavigationAgent3D::is_target_reached);
	ClassDB::bind_method(D_METHOD("is_target_reachable"), &NavigationAgent3D::is_target_reachable);
	ClassDB::bind_method(D_METHOD("is_navigation_finished"), &NavigationAgent3D::is_navigation_finished);
	ClassDB::bind_method(D_METHOD("get_final_location"), &NavigationAgent3D::get_final_location);

	ClassDB::bind_method(D_METHOD("_avoidance_done", "new_velocity"), &NavigationAgent3D::_avoidance_done);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_desired_distance", PROPERTY_HINT_RANGE, "0.1,100,0.01"), "set_target_desired_distance", "get_target_desired_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.1,100,0.01"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_height_offset", PROPERTY_HINT_RANGE, "-100.0,100,0.01"), "set_agent_height_offset", "get_agent_height_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "neighbor_dist", PROPERTY_HINT_RANGE, "0.1,10000,0.01"), "set_neighbor_dist", "get_neighbor_dist");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_neighbors", PROPERTY_HINT_RANGE, "1,10000,1"), "set_max_neighbors", "get_max_neighbors");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_horizon", PROPERTY_HINT_RANGE, "0.01,100,0.01"), "set_time_horizon", "get_time_horizon");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_speed", PROPERTY_HINT_RANGE, "0.1,10000,0.01"), "set_max_speed", "get_max_speed");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_max_distance", PROPERTY_HINT_RANGE, "0.01,100,0.1"), "set_path_max_distance", "get_path_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_y"), "set_ignore_y", "get_ignore_y");

	ADD_SIGNAL(MethodInfo("path_changed"));
	ADD_SIGNAL(MethodInfo("target_reached"));
	ADD_SIGNAL(MethodInfo("navigation_finished"));
	ADD_SIGNAL(MethodInfo("velocity_computed", PropertyInfo(Variant::VECTOR3, "safe_velocity")));
}

void NavigationAgent3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			agent_parent = Object::cast_to<Node3D>(get_parent());
			if (agent_parent != nullptr) {
				// place agent on navigation map first or else the RVO agent callback creation fails silently later
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), agent_parent->get_world_3d()->get_navigation_map());
				NavigationServer3D::get_singleton()->agent_set_callback(agent, this, "_avoidance_done");
			}
			set_physics_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			agent_parent = nullptr;
			set_physics_process_internal(false);
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (agent_parent) {
				NavigationServer3D::get_singleton()->agent_set_position(agent, agent_parent->get_global_transform().origin);
				_check_distance_to_target();
			}
		} break;
	}
}

NavigationAgent3D::NavigationAgent3D() {
	agent = NavigationServer3D::get_singleton()->agent_create();
	set_neighbor_dist(50.0);
	set_max_neighbors(10);
	set_time_horizon(5.0);
	set_radius(1.0);
	set_max_speed(10.0);
	set_ignore_y(true);
}

NavigationAgent3D::~NavigationAgent3D() {
	NavigationServer3D::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

void NavigationAgent3D::set_target_desired_distance(real_t p_dd) {
	target_desired_distance = p_dd;
}

void NavigationAgent3D::set_radius(real_t p_radius) {
	radius = p_radius;
	NavigationServer3D::get_singleton()->agent_set_radius(agent, radius);
}

void NavigationAgent3D::set_agent_height_offset(real_t p_hh) {
	navigation_height_offset = p_hh;
}

void NavigationAgent3D::set_ignore_y(bool p_ignore_y) {
	ignore_y = p_ignore_y;
	NavigationServer3D::get_singleton()->agent_set_ignore_y(agent, ignore_y);
}

void NavigationAgent3D::set_neighbor_dist(real_t p_dist) {
	neighbor_dist = p_dist;
	NavigationServer3D::get_singleton()->agent_set_neighbor_dist(agent, neighbor_dist);
}

void NavigationAgent3D::set_max_neighbors(int p_count) {
	max_neighbors = p_count;
	NavigationServer3D::get_singleton()->agent_set_max_neighbors(agent, max_neighbors);
}

void NavigationAgent3D::set_time_horizon(real_t p_time) {
	time_horizon = p_time;
	NavigationServer3D::get_singleton()->agent_set_time_horizon(agent, time_horizon);
}

void NavigationAgent3D::set_max_speed(real_t p_max_speed) {
	max_speed = p_max_speed;
	NavigationServer3D::get_singleton()->agent_set_max_speed(agent, max_speed);
}

void NavigationAgent3D::set_path_max_distance(real_t p_pmd) {
	path_max_distance = p_pmd;
}

real_t NavigationAgent3D::get_path_max_distance() {
	return path_max_distance;
}

void NavigationAgent3D::set_target_location(Vector3 p_location) {
	target_location = p_location;
	navigation_path.clear();
	target_reached = false;
	navigation_finished = false;
	update_frame_id = 0;
}

Vector3 NavigationAgent3D::get_target_location() const {
	return target_location;
}

Vector3 NavigationAgent3D::get_next_location() {
	update_navigation();
	if (navigation_path.size() == 0) {
		ERR_FAIL_COND_V(agent_parent == nullptr, Vector3());
		return agent_parent->get_global_transform().origin;
	} else {
		return navigation_path[nav_path_index] - Vector3(0, navigation_height_offset, 0);
	}
}

real_t NavigationAgent3D::distance_to_target() const {
	ERR_FAIL_COND_V(agent_parent == nullptr, 0.0);
	return agent_parent->get_global_transform().origin.distance_to(target_location);
}

bool NavigationAgent3D::is_target_reached() const {
	return target_reached;
}

bool NavigationAgent3D::is_target_reachable() {
	return target_desired_distance >= get_final_location().distance_to(target_location);
}

bool NavigationAgent3D::is_navigation_finished() {
	update_navigation();
	return navigation_finished;
}

Vector3 NavigationAgent3D::get_final_location() {
	update_navigation();
	if (navigation_path.size() == 0) {
		return Vector3();
	}
	return navigation_path[navigation_path.size() - 1];
}

void NavigationAgent3D::set_velocity(Vector3 p_velocity) {
	target_velocity = p_velocity;
	NavigationServer3D::get_singleton()->agent_set_target_velocity(agent, target_velocity);
	NavigationServer3D::get_singleton()->agent_set_velocity(agent, prev_safe_velocity);
	velocity_submitted = true;
}

void NavigationAgent3D::_avoidance_done(Vector3 p_new_velocity) {
	prev_safe_velocity = p_new_velocity;

	if (!velocity_submitted) {
		target_velocity = Vector3();
		return;
	}
	velocity_submitted = false;

	emit_signal("velocity_computed", p_new_velocity);
}

TypedArray<String> NavigationAgent3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!Object::cast_to<Node3D>(get_parent())) {
		warnings.push_back(TTR("The NavigationAgent3D can be used only under a spatial node."));
	}

	return warnings;
}

void NavigationAgent3D::update_navigation() {
	if (agent_parent == nullptr) {
		return;
	}
	if (!agent_parent->is_inside_tree()) {
		return;
	}
	if (update_frame_id == Engine::get_singleton()->get_physics_frames()) {
		return;
	}

	update_frame_id = Engine::get_singleton()->get_physics_frames();

	Vector3 o = agent_parent->get_global_transform().origin;

	bool reload_path = false;

	if (NavigationServer3D::get_singleton()->agent_is_map_changed(agent)) {
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
			Vector3 p = Geometry3D::get_closest_point_to_segment(o, segment);
			if (o.distance_to(p) >= path_max_distance) {
				// To faraway, reload path
				reload_path = true;
			}
		}
	}

	if (reload_path) {
		navigation_path = NavigationServer3D::get_singleton()->map_get_path(agent_parent->get_world_3d()->get_navigation_map(), o, target_location, true);
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

void NavigationAgent3D::_check_distance_to_target() {
	if (!target_reached) {
		if (distance_to_target() < target_desired_distance) {
			emit_signal("target_reached");
			target_reached = true;
		}
	}
}
