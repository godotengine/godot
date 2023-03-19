/**************************************************************************/
/*  navigation_agent.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "navigation_agent.h"

#include "core/engine.h"
#include "servers/navigation_server.h"

void NavigationAgent::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &NavigationAgent::get_rid);

	ClassDB::bind_method(D_METHOD("set_avoidance_enabled", "enabled"), &NavigationAgent::set_avoidance_enabled);
	ClassDB::bind_method(D_METHOD("get_avoidance_enabled"), &NavigationAgent::get_avoidance_enabled);

	ClassDB::bind_method(D_METHOD("set_path_desired_distance", "desired_distance"), &NavigationAgent::set_path_desired_distance);
	ClassDB::bind_method(D_METHOD("get_path_desired_distance"), &NavigationAgent::get_path_desired_distance);

	ClassDB::bind_method(D_METHOD("set_target_desired_distance", "desired_distance"), &NavigationAgent::set_target_desired_distance);
	ClassDB::bind_method(D_METHOD("get_target_desired_distance"), &NavigationAgent::get_target_desired_distance);

	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationAgent::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationAgent::get_radius);

	ClassDB::bind_method(D_METHOD("set_agent_height_offset", "agent_height_offset"), &NavigationAgent::set_agent_height_offset);
	ClassDB::bind_method(D_METHOD("get_agent_height_offset"), &NavigationAgent::get_agent_height_offset);

	ClassDB::bind_method(D_METHOD("set_ignore_y", "ignore"), &NavigationAgent::set_ignore_y);
	ClassDB::bind_method(D_METHOD("get_ignore_y"), &NavigationAgent::get_ignore_y);

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

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &NavigationAgent::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &NavigationAgent::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_navigation_layer_value", "layer_number", "value"), &NavigationAgent::set_navigation_layer_value);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_value", "layer_number"), &NavigationAgent::get_navigation_layer_value);

	ClassDB::bind_method(D_METHOD("set_navigation_map", "navigation_map"), &NavigationAgent::set_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &NavigationAgent::get_navigation_map);

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

	ADD_GROUP("Pathfinding", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_location", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_target_location", "get_target_location");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "path_desired_distance", PROPERTY_HINT_RANGE, "0.1,100,0.01"), "set_path_desired_distance", "get_path_desired_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "target_desired_distance", PROPERTY_HINT_RANGE, "0.1,100,0.01"), "set_target_desired_distance", "get_target_desired_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "agent_height_offset", PROPERTY_HINT_RANGE, "-100.0,100,0.01"), "set_agent_height_offset", "get_agent_height_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "path_max_distance", PROPERTY_HINT_RANGE, "0.01,100,0.1"), "set_path_max_distance", "get_path_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");

	ADD_GROUP("Avoidance", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "avoidance_enabled"), "set_avoidance_enabled", "get_avoidance_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.1,100,0.01"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "neighbor_dist", PROPERTY_HINT_RANGE, "0.1,10000,0.01"), "set_neighbor_dist", "get_neighbor_dist");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_neighbors", PROPERTY_HINT_RANGE, "1,10000,1"), "set_max_neighbors", "get_max_neighbors");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "time_horizon", PROPERTY_HINT_RANGE, "0.01,100,0.01"), "set_time_horizon", "get_time_horizon");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_speed", PROPERTY_HINT_RANGE, "0.1,10000,0.01"), "set_max_speed", "get_max_speed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_y"), "set_ignore_y", "get_ignore_y");

	ADD_SIGNAL(MethodInfo("path_changed"));
	ADD_SIGNAL(MethodInfo("target_reached"));
	ADD_SIGNAL(MethodInfo("navigation_finished"));
	ADD_SIGNAL(MethodInfo("velocity_computed", PropertyInfo(Variant::VECTOR3, "safe_velocity")));
}

void NavigationAgent::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			// need to use POST_ENTER_TREE cause with normal ENTER_TREE not all required Nodes are ready.
			// cannot use READY as ready does not get called if Node is re-added to SceneTree
			set_agent_parent(get_parent());
			set_physics_process_internal(true);
		} break;

		case NOTIFICATION_PARENTED: {
			if (is_inside_tree() && (get_parent() != agent_parent)) {
				// only react to PARENTED notifications when already inside_tree and parent changed, e.g. users switch nodes around
				// PARENTED notification fires also when Node is added in scripts to a parent
				// this would spam transforms fails and world fails while Node is outside SceneTree
				// when node gets reparented when joining the tree POST_ENTER_TREE takes care of this
				set_agent_parent(get_parent());
				set_physics_process_internal(true);
			}
		} break;

		case NOTIFICATION_UNPARENTED: {
			// if agent has no parent no point in processing it until reparented
			set_agent_parent(nullptr);
			set_physics_process_internal(false);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			set_agent_parent(nullptr);
			set_physics_process_internal(false);
		} break;

		case NOTIFICATION_PAUSED: {
			if (agent_parent && !agent_parent->can_process()) {
				map_before_pause = NavigationServer::get_singleton()->agent_get_map(get_rid());
				NavigationServer::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (agent_parent && agent_parent->can_process() && !(map_before_pause == RID())) {
				NavigationServer::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;

		case NOTIFICATION_UNPAUSED: {
			if (agent_parent && !agent_parent->can_process()) {
				map_before_pause = NavigationServer::get_singleton()->agent_get_map(get_rid());
				NavigationServer::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (agent_parent && agent_parent->can_process() && !(map_before_pause == RID())) {
				NavigationServer::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (agent_parent) {
				if (avoidance_enabled) {
					// agent_position on NavigationServer is avoidance only and has nothing to do with pathfinding
					// no point in flooding NavigationServer queue with agent position updates that get send to the void if avoidance is not used
					NavigationServer::get_singleton()->agent_set_position(agent, agent_parent->get_global_transform().origin);
				}
				_check_distance_to_target();
			}
		} break;
	}
}

NavigationAgent::NavigationAgent() {
	agent = NavigationServer::get_singleton()->agent_create();
	NavigationServer::get_singleton()->agent_set_neighbor_dist(agent, neighbor_dist);
	NavigationServer::get_singleton()->agent_set_max_neighbors(agent, max_neighbors);
	NavigationServer::get_singleton()->agent_set_time_horizon(agent, time_horizon);
	NavigationServer::get_singleton()->agent_set_radius(agent, radius);
	NavigationServer::get_singleton()->agent_set_max_speed(agent, max_speed);
	NavigationServer::get_singleton()->agent_set_ignore_y(agent, ignore_y);
}

NavigationAgent::~NavigationAgent() {
	ERR_FAIL_NULL(NavigationServer::get_singleton());
	NavigationServer::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

void NavigationAgent::set_avoidance_enabled(bool p_enabled) {
	if (avoidance_enabled == p_enabled) {
		return;
	}

	avoidance_enabled = p_enabled;

	if (avoidance_enabled) {
		NavigationServer::get_singleton()->agent_set_callback(agent, get_instance_id(), "_avoidance_done");
	} else {
		NavigationServer::get_singleton()->agent_set_callback(agent, ObjectID(), "_avoidance_done");
	}
}

bool NavigationAgent::get_avoidance_enabled() const {
	return avoidance_enabled;
}

void NavigationAgent::set_agent_parent(Node *p_agent_parent) {
	if (agent_parent == p_agent_parent) {
		return;
	}

	// remove agent from any avoidance map before changing parent or there will be leftovers on the RVO map
	NavigationServer::get_singleton()->agent_set_callback(agent, ObjectID(), "_avoidance_done");
	if (Object::cast_to<Spatial>(p_agent_parent) != nullptr) {
		// place agent on navigation map first or else the RVO agent callback creation fails silently later
		agent_parent = Object::cast_to<Spatial>(p_agent_parent);
		if (map_override.is_valid()) {
			NavigationServer::get_singleton()->agent_set_map(get_rid(), map_override);
		} else {
			// no navigation node found in parent nodes, use default navigation map from world resource
			NavigationServer::get_singleton()->agent_set_map(get_rid(), agent_parent->get_world()->get_navigation_map());
		}

		// create new avoidance callback if enabled
		if (avoidance_enabled) {
			NavigationServer::get_singleton()->agent_set_callback(agent, get_instance_id(), "_avoidance_done");
		}
	} else {
		agent_parent = nullptr;
		NavigationServer::get_singleton()->agent_set_map(get_rid(), RID());
	}
}

void NavigationAgent::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}

	navigation_layers = p_navigation_layers;

	_request_repath();
}

uint32_t NavigationAgent::get_navigation_layers() const {
	return navigation_layers;
}

void NavigationAgent::set_navigation_layer_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Navigation layer number must be between 1 and 32 inclusive.");
	uint32_t _navigation_layers = get_navigation_layers();
	if (p_value) {
		_navigation_layers |= 1 << (p_layer_number - 1);
	} else {
		_navigation_layers &= ~(1 << (p_layer_number - 1));
	}
	set_navigation_layers(_navigation_layers);
}

bool NavigationAgent::get_navigation_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Navigation layer number must be between 1 and 32 inclusive.");
	return get_navigation_layers() & (1 << (p_layer_number - 1));
}

void NavigationAgent::set_navigation_map(RID p_navigation_map) {
	if (map_override == p_navigation_map) {
		return;
	}

	map_override = p_navigation_map;

	NavigationServer::get_singleton()->agent_set_map(agent, map_override);
	_request_repath();
}

RID NavigationAgent::get_navigation_map() const {
	if (map_override.is_valid()) {
		return map_override;
	} else if (agent_parent != nullptr) {
		return agent_parent->get_world()->get_navigation_map();
	}
	return RID();
}

void NavigationAgent::set_path_desired_distance(real_t p_dd) {
	if (Math::is_equal_approx(path_desired_distance, p_dd)) {
		return;
	}

	path_desired_distance = p_dd;
}

void NavigationAgent::set_target_desired_distance(real_t p_dd) {
	if (Math::is_equal_approx(target_desired_distance, p_dd)) {
		return;
	}

	target_desired_distance = p_dd;
}

void NavigationAgent::set_radius(real_t p_radius) {
	if (Math::is_equal_approx(radius, p_radius)) {
		return;
	}

	radius = p_radius;

	NavigationServer::get_singleton()->agent_set_radius(agent, radius);
}

void NavigationAgent::set_agent_height_offset(real_t p_hh) {
	if (Math::is_equal_approx(navigation_height_offset, p_hh)) {
		return;
	}

	navigation_height_offset = p_hh;
}

void NavigationAgent::set_ignore_y(bool p_ignore_y) {
	if (ignore_y == p_ignore_y) {
		return;
	}

	ignore_y = p_ignore_y;

	NavigationServer::get_singleton()->agent_set_ignore_y(agent, ignore_y);
}

void NavigationAgent::set_neighbor_dist(real_t p_dist) {
	if (Math::is_equal_approx(neighbor_dist, p_dist)) {
		return;
	}

	neighbor_dist = p_dist;

	NavigationServer::get_singleton()->agent_set_neighbor_dist(agent, neighbor_dist);
}

void NavigationAgent::set_max_neighbors(int p_count) {
	if (max_neighbors == p_count) {
		return;
	}

	max_neighbors = p_count;

	NavigationServer::get_singleton()->agent_set_max_neighbors(agent, max_neighbors);
}

void NavigationAgent::set_time_horizon(real_t p_time) {
	if (Math::is_equal_approx(time_horizon, p_time)) {
		return;
	}

	time_horizon = p_time;

	NavigationServer::get_singleton()->agent_set_time_horizon(agent, time_horizon);
}

void NavigationAgent::set_max_speed(real_t p_max_speed) {
	if (Math::is_equal_approx(max_speed, p_max_speed)) {
		return;
	}

	max_speed = p_max_speed;

	NavigationServer::get_singleton()->agent_set_max_speed(agent, max_speed);
}

void NavigationAgent::set_path_max_distance(real_t p_pmd) {
	if (Math::is_equal_approx(path_max_distance, p_pmd)) {
		return;
	}

	path_max_distance = p_pmd;
}

real_t NavigationAgent::get_path_max_distance() {
	return path_max_distance;
}

void NavigationAgent::set_target_location(Vector3 p_location) {
	target_location = p_location;
	_request_repath();
}

Vector3 NavigationAgent::get_target_location() const {
	return target_location;
}

Vector3 NavigationAgent::get_next_location() {
	update_navigation();
	if (navigation_path.size() == 0) {
		ERR_FAIL_COND_V_MSG(agent_parent == nullptr, Vector3(), "The agent has no parent.");
		return agent_parent->get_global_transform().origin;
	} else {
		return navigation_path[nav_path_index] - Vector3(0, navigation_height_offset, 0);
	}
}

real_t NavigationAgent::distance_to_target() const {
	ERR_FAIL_COND_V_MSG(agent_parent == nullptr, 0.0, "The agent has no parent.");
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
	String warning = Node::get_configuration_warning();

	if (!Object::cast_to<Spatial>(get_parent())) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("The NavigationAgent can be used only under a Spatial inheriting parent node.");
	}

	return warning;
}

void NavigationAgent::update_navigation() {
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
		if (map_override.is_valid()) {
			navigation_path = NavigationServer::get_singleton()->map_get_path(map_override, o, target_location, true, navigation_layers);
		} else {
			navigation_path = NavigationServer::get_singleton()->map_get_path(agent_parent->get_world()->get_navigation_map(), o, target_location, true, navigation_layers);
		}
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
		while (o.distance_to(navigation_path[nav_path_index] - Vector3(0, navigation_height_offset, 0)) < path_desired_distance) {
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

void NavigationAgent::_request_repath() {
	navigation_path.clear();
	target_reached = false;
	navigation_finished = false;
	update_frame_id = 0;
}

void NavigationAgent::_check_distance_to_target() {
	if (!target_reached) {
		if (distance_to_target() < target_desired_distance) {
			target_reached = true;
			emit_signal("target_reached");
		}
	}
}
