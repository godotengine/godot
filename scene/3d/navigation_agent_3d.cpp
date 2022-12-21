/*************************************************************************/
/*  navigation_agent_3d.cpp                                              */
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

#include "navigation_agent_3d.h"

#include "servers/navigation_server_3d.h"

void NavigationAgent3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &NavigationAgent3D::get_rid);

	ClassDB::bind_method(D_METHOD("set_avoidance_enabled", "enabled"), &NavigationAgent3D::set_avoidance_enabled);
	ClassDB::bind_method(D_METHOD("get_avoidance_enabled"), &NavigationAgent3D::get_avoidance_enabled);

	ClassDB::bind_method(D_METHOD("set_path_desired_distance", "desired_distance"), &NavigationAgent3D::set_path_desired_distance);
	ClassDB::bind_method(D_METHOD("get_path_desired_distance"), &NavigationAgent3D::get_path_desired_distance);

	ClassDB::bind_method(D_METHOD("set_target_desired_distance", "desired_distance"), &NavigationAgent3D::set_target_desired_distance);
	ClassDB::bind_method(D_METHOD("get_target_desired_distance"), &NavigationAgent3D::get_target_desired_distance);

	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationAgent3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationAgent3D::get_radius);

	ClassDB::bind_method(D_METHOD("set_agent_height_offset", "agent_height_offset"), &NavigationAgent3D::set_agent_height_offset);
	ClassDB::bind_method(D_METHOD("get_agent_height_offset"), &NavigationAgent3D::get_agent_height_offset);

	ClassDB::bind_method(D_METHOD("set_ignore_y", "ignore"), &NavigationAgent3D::set_ignore_y);
	ClassDB::bind_method(D_METHOD("get_ignore_y"), &NavigationAgent3D::get_ignore_y);

	ClassDB::bind_method(D_METHOD("set_neighbor_distance", "neighbor_distance"), &NavigationAgent3D::set_neighbor_distance);
	ClassDB::bind_method(D_METHOD("get_neighbor_distance"), &NavigationAgent3D::get_neighbor_distance);

	ClassDB::bind_method(D_METHOD("set_max_neighbors", "max_neighbors"), &NavigationAgent3D::set_max_neighbors);
	ClassDB::bind_method(D_METHOD("get_max_neighbors"), &NavigationAgent3D::get_max_neighbors);

	ClassDB::bind_method(D_METHOD("set_time_horizon", "time_horizon"), &NavigationAgent3D::set_time_horizon);
	ClassDB::bind_method(D_METHOD("get_time_horizon"), &NavigationAgent3D::get_time_horizon);

	ClassDB::bind_method(D_METHOD("set_max_speed", "max_speed"), &NavigationAgent3D::set_max_speed);
	ClassDB::bind_method(D_METHOD("get_max_speed"), &NavigationAgent3D::get_max_speed);

	ClassDB::bind_method(D_METHOD("set_path_max_distance", "max_speed"), &NavigationAgent3D::set_path_max_distance);
	ClassDB::bind_method(D_METHOD("get_path_max_distance"), &NavigationAgent3D::get_path_max_distance);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &NavigationAgent3D::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &NavigationAgent3D::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_navigation_layer_value", "layer_number", "value"), &NavigationAgent3D::set_navigation_layer_value);
	ClassDB::bind_method(D_METHOD("get_navigation_layer_value", "layer_number"), &NavigationAgent3D::get_navigation_layer_value);

	ClassDB::bind_method(D_METHOD("set_path_metadata_flags", "flags"), &NavigationAgent3D::set_path_metadata_flags);
	ClassDB::bind_method(D_METHOD("get_path_metadata_flags"), &NavigationAgent3D::get_path_metadata_flags);

	ClassDB::bind_method(D_METHOD("set_navigation_map", "navigation_map"), &NavigationAgent3D::set_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &NavigationAgent3D::get_navigation_map);

	ClassDB::bind_method(D_METHOD("set_target_location", "location"), &NavigationAgent3D::set_target_location);
	ClassDB::bind_method(D_METHOD("get_target_location"), &NavigationAgent3D::get_target_location);

	ClassDB::bind_method(D_METHOD("get_next_location"), &NavigationAgent3D::get_next_location);
	ClassDB::bind_method(D_METHOD("distance_to_target"), &NavigationAgent3D::distance_to_target);
	ClassDB::bind_method(D_METHOD("set_velocity", "velocity"), &NavigationAgent3D::set_velocity);
	ClassDB::bind_method(D_METHOD("get_current_navigation_result"), &NavigationAgent3D::get_current_navigation_result);
	ClassDB::bind_method(D_METHOD("get_current_navigation_path"), &NavigationAgent3D::get_current_navigation_path);
	ClassDB::bind_method(D_METHOD("get_current_navigation_path_index"), &NavigationAgent3D::get_current_navigation_path_index);
	ClassDB::bind_method(D_METHOD("is_target_reached"), &NavigationAgent3D::is_target_reached);
	ClassDB::bind_method(D_METHOD("is_target_reachable"), &NavigationAgent3D::is_target_reachable);
	ClassDB::bind_method(D_METHOD("is_navigation_finished"), &NavigationAgent3D::is_navigation_finished);
	ClassDB::bind_method(D_METHOD("get_final_location"), &NavigationAgent3D::get_final_location);

	ClassDB::bind_method(D_METHOD("_avoidance_done", "new_velocity"), &NavigationAgent3D::_avoidance_done);

	ADD_GROUP("Pathfinding", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_location", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_target_location", "get_target_location");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_desired_distance", PROPERTY_HINT_RANGE, "0.1,100,0.01,suffix:m"), "set_path_desired_distance", "get_path_desired_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "target_desired_distance", PROPERTY_HINT_RANGE, "0.1,100,0.01,suffix:m"), "set_target_desired_distance", "get_target_desired_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_height_offset", PROPERTY_HINT_RANGE, "-100.0,100,0.01,suffix:m"), "set_agent_height_offset", "get_agent_height_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_max_distance", PROPERTY_HINT_RANGE, "0.01,100,0.1,suffix:m"), "set_path_max_distance", "get_path_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_metadata_flags", PROPERTY_HINT_FLAGS, "Include Types,Include RIDs,Include Owners"), "set_path_metadata_flags", "get_path_metadata_flags");

	ADD_GROUP("Avoidance", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "avoidance_enabled"), "set_avoidance_enabled", "get_avoidance_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.1,100,0.01,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "neighbor_distance", PROPERTY_HINT_RANGE, "0.1,10000,0.01,suffix:m"), "set_neighbor_distance", "get_neighbor_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_neighbors", PROPERTY_HINT_RANGE, "1,10000,1"), "set_max_neighbors", "get_max_neighbors");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_horizon", PROPERTY_HINT_RANGE, "0.01,100,0.01,suffix:s"), "set_time_horizon", "get_time_horizon");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_speed", PROPERTY_HINT_RANGE, "0.1,10000,0.01,suffix:m/s"), "set_max_speed", "get_max_speed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ignore_y"), "set_ignore_y", "get_ignore_y");

	ADD_SIGNAL(MethodInfo("path_changed"));
	ADD_SIGNAL(MethodInfo("target_reached"));
	ADD_SIGNAL(MethodInfo("waypoint_reached", PropertyInfo(Variant::DICTIONARY, "details")));
	ADD_SIGNAL(MethodInfo("link_reached", PropertyInfo(Variant::DICTIONARY, "details")));
	ADD_SIGNAL(MethodInfo("navigation_finished"));
	ADD_SIGNAL(MethodInfo("velocity_computed", PropertyInfo(Variant::VECTOR3, "safe_velocity")));
}

void NavigationAgent3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			// need to use POST_ENTER_TREE cause with normal ENTER_TREE not all required Nodes are ready.
			// cannot use READY as ready does not get called if Node is readded to SceneTree
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
				map_before_pause = NavigationServer3D::get_singleton()->agent_get_map(get_rid());
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (agent_parent && agent_parent->can_process() && !(map_before_pause == RID())) {
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;

		case NOTIFICATION_UNPAUSED: {
			if (agent_parent && !agent_parent->can_process()) {
				map_before_pause = NavigationServer3D::get_singleton()->agent_get_map(get_rid());
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (agent_parent && agent_parent->can_process() && !(map_before_pause == RID())) {
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (agent_parent && target_position_submitted) {
				if (avoidance_enabled) {
					// agent_position on NavigationServer is avoidance only and has nothing to do with pathfinding
					// no point in flooding NavigationServer queue with agent position updates that get send to the void if avoidance is not used
					NavigationServer3D::get_singleton()->agent_set_position(agent, agent_parent->get_global_transform().origin);
				}
				_check_distance_to_target();
			}
		} break;
	}
}

NavigationAgent3D::NavigationAgent3D() {
	agent = NavigationServer3D::get_singleton()->agent_create();
	set_neighbor_distance(50.0);
	set_max_neighbors(10);
	set_time_horizon(5.0);
	set_radius(1.0);
	set_max_speed(10.0);
	set_ignore_y(true);

	// Preallocate query and result objects to improve performance.
	navigation_query = Ref<NavigationPathQueryParameters3D>();
	navigation_query.instantiate();

	navigation_result = Ref<NavigationPathQueryResult3D>();
	navigation_result.instantiate();
}

NavigationAgent3D::~NavigationAgent3D() {
	NavigationServer3D::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

void NavigationAgent3D::set_avoidance_enabled(bool p_enabled) {
	avoidance_enabled = p_enabled;
	if (avoidance_enabled) {
		NavigationServer3D::get_singleton()->agent_set_callback(agent, get_instance_id(), "_avoidance_done");
	} else {
		NavigationServer3D::get_singleton()->agent_set_callback(agent, ObjectID(), "_avoidance_done");
	}
}

bool NavigationAgent3D::get_avoidance_enabled() const {
	return avoidance_enabled;
}

void NavigationAgent3D::set_agent_parent(Node *p_agent_parent) {
	// remove agent from any avoidance map before changing parent or there will be leftovers on the RVO map
	NavigationServer3D::get_singleton()->agent_set_callback(agent, ObjectID(), "_avoidance_done");
	if (Object::cast_to<Node3D>(p_agent_parent) != nullptr) {
		// place agent on navigation map first or else the RVO agent callback creation fails silently later
		agent_parent = Object::cast_to<Node3D>(p_agent_parent);
		if (map_override.is_valid()) {
			NavigationServer3D::get_singleton()->agent_set_map(get_rid(), map_override);
		} else {
			NavigationServer3D::get_singleton()->agent_set_map(get_rid(), agent_parent->get_world_3d()->get_navigation_map());
		}
		// create new avoidance callback if enabled
		set_avoidance_enabled(avoidance_enabled);
	} else {
		agent_parent = nullptr;
		NavigationServer3D::get_singleton()->agent_set_map(get_rid(), RID());
	}
}

void NavigationAgent3D::set_navigation_layers(uint32_t p_navigation_layers) {
	bool navigation_layers_changed = navigation_layers != p_navigation_layers;
	navigation_layers = p_navigation_layers;
	if (navigation_layers_changed) {
		_request_repath();
	}
}

uint32_t NavigationAgent3D::get_navigation_layers() const {
	return navigation_layers;
}

void NavigationAgent3D::set_navigation_layer_value(int p_layer_number, bool p_value) {
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

bool NavigationAgent3D::get_navigation_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Navigation layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Navigation layer number must be between 1 and 32 inclusive.");
	return get_navigation_layers() & (1 << (p_layer_number - 1));
}

void NavigationAgent3D::set_path_metadata_flags(BitField<NavigationPathQueryParameters3D::PathMetadataFlags> p_path_metadata_flags) {
	if (path_metadata_flags == p_path_metadata_flags) {
		return;
	}

	path_metadata_flags = p_path_metadata_flags;
}

void NavigationAgent3D::set_navigation_map(RID p_navigation_map) {
	map_override = p_navigation_map;
	NavigationServer3D::get_singleton()->agent_set_map(agent, map_override);
	_request_repath();
}

RID NavigationAgent3D::get_navigation_map() const {
	if (map_override.is_valid()) {
		return map_override;
	} else if (agent_parent != nullptr) {
		return agent_parent->get_world_3d()->get_navigation_map();
	}
	return RID();
}

void NavigationAgent3D::set_path_desired_distance(real_t p_dd) {
	path_desired_distance = p_dd;
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

void NavigationAgent3D::set_neighbor_distance(real_t p_distance) {
	neighbor_distance = p_distance;
	NavigationServer3D::get_singleton()->agent_set_neighbor_distance(agent, neighbor_distance);
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
	target_position_submitted = true;
	_request_repath();
}

Vector3 NavigationAgent3D::get_target_location() const {
	return target_location;
}

Vector3 NavigationAgent3D::get_next_location() {
	update_navigation();

	const Vector<Vector3> &navigation_path = navigation_result->get_path();
	if (navigation_path.size() == 0) {
		ERR_FAIL_COND_V_MSG(agent_parent == nullptr, Vector3(), "The agent has no parent.");
		return agent_parent->get_global_transform().origin;
	} else {
		return navigation_path[navigation_path_index] - Vector3(0, navigation_height_offset, 0);
	}
}

real_t NavigationAgent3D::distance_to_target() const {
	ERR_FAIL_COND_V_MSG(agent_parent == nullptr, 0.0, "The agent has no parent.");
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

	const Vector<Vector3> &navigation_path = navigation_result->get_path();
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

	emit_signal(SNAME("velocity_computed"), p_new_velocity);
}

PackedStringArray NavigationAgent3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	if (!Object::cast_to<Node3D>(get_parent())) {
		warnings.push_back(RTR("The NavigationAgent3D can be used only under a Node3D inheriting parent node."));
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
	if (!target_position_submitted) {
		return;
	}
	if (update_frame_id == Engine::get_singleton()->get_physics_frames()) {
		return;
	}

	update_frame_id = Engine::get_singleton()->get_physics_frames();

	Vector3 origin = agent_parent->get_global_transform().origin;

	bool reload_path = false;

	if (NavigationServer3D::get_singleton()->agent_is_map_changed(agent)) {
		reload_path = true;
	} else if (navigation_result->get_path().size() == 0) {
		reload_path = true;
	} else {
		// Check if too far from the navigation path
		if (navigation_path_index > 0) {
			const Vector<Vector3> &navigation_path = navigation_result->get_path();

			Vector3 segment[2];
			segment[0] = navigation_path[navigation_path_index - 1];
			segment[1] = navigation_path[navigation_path_index];
			segment[0].y -= navigation_height_offset;
			segment[1].y -= navigation_height_offset;
			Vector3 p = Geometry3D::get_closest_point_to_segment(origin, segment);
			if (origin.distance_to(p) >= path_max_distance) {
				// To faraway, reload path
				reload_path = true;
			}
		}
	}

	if (reload_path) {
		navigation_query->set_start_position(origin);
		navigation_query->set_target_position(target_location);
		navigation_query->set_navigation_layers(navigation_layers);
		navigation_query->set_metadata_flags(path_metadata_flags);

		if (map_override.is_valid()) {
			navigation_query->set_map(map_override);
		} else {
			navigation_query->set_map(agent_parent->get_world_3d()->get_navigation_map());
		}

		NavigationServer3D::get_singleton()->query_path(navigation_query, navigation_result);
		navigation_finished = false;
		navigation_path_index = 0;
		emit_signal(SNAME("path_changed"));
	}

	if (navigation_result->get_path().size() == 0) {
		return;
	}

	// Check if we can advance the navigation path
	if (navigation_finished == false) {
		// Advances to the next far away location.
		const Vector<Vector3> &navigation_path = navigation_result->get_path();
		const Vector<int32_t> &navigation_path_types = navigation_result->get_path_types();
		const TypedArray<RID> &navigation_path_rids = navigation_result->get_path_rids();
		const Vector<int64_t> &navigation_path_owners = navigation_result->get_path_owner_ids();

		while (origin.distance_to(navigation_path[navigation_path_index] - Vector3(0, navigation_height_offset, 0)) < path_desired_distance) {
			Dictionary details;

			const Vector3 waypoint = navigation_path[navigation_path_index];
			details[SNAME("location")] = waypoint;

			int waypoint_type = -1;
			if (path_metadata_flags.has_flag(NavigationPathQueryParameters3D::PathMetadataFlags::PATH_METADATA_INCLUDE_TYPES)) {
				const NavigationPathQueryResult3D::PathSegmentType type = NavigationPathQueryResult3D::PathSegmentType(navigation_path_types[navigation_path_index]);

				details[SNAME("type")] = type;
				waypoint_type = type;
			}

			if (path_metadata_flags.has_flag(NavigationPathQueryParameters3D::PathMetadataFlags::PATH_METADATA_INCLUDE_RIDS)) {
				details[SNAME("rid")] = navigation_path_rids[navigation_path_index];
			}

			if (path_metadata_flags.has_flag(NavigationPathQueryParameters3D::PathMetadataFlags::PATH_METADATA_INCLUDE_OWNERS)) {
				const ObjectID waypoint_owner_id = ObjectID(navigation_path_owners[navigation_path_index]);

				// Get a reference to the owning object.
				Object *owner = nullptr;
				if (waypoint_owner_id.is_valid()) {
					owner = ObjectDB::get_instance(waypoint_owner_id);
				}

				details[SNAME("owner")] = owner;
			}

			// Emit a signal for the waypoint
			emit_signal(SNAME("waypoint_reached"), details);

			// Emit a signal if we've reached a navigation link
			if (waypoint_type == NavigationPathQueryResult3D::PATH_SEGMENT_TYPE_LINK) {
				emit_signal(SNAME("link_reached"), details);
			}

			// Move to the next waypoint on the list
			navigation_path_index += 1;

			// Check to see if we've finished our route
			if (navigation_path_index == navigation_path.size()) {
				_check_distance_to_target();
				navigation_path_index -= 1;
				navigation_finished = true;
				target_position_submitted = false;
				emit_signal(SNAME("navigation_finished"));
				break;
			}
		}
	}
}

void NavigationAgent3D::_request_repath() {
	navigation_result->reset();
	target_reached = false;
	navigation_finished = false;
	update_frame_id = 0;
}

void NavigationAgent3D::_check_distance_to_target() {
	if (!target_reached) {
		if (distance_to_target() < target_desired_distance) {
			target_reached = true;
			emit_signal(SNAME("target_reached"));
		}
	}
}
