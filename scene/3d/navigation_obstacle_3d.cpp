/**************************************************************************/
/*  navigation_obstacle_3d.cpp                                            */
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

#include "navigation_obstacle_3d.h"

#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "servers/navigation_server_3d.h"

void NavigationObstacle3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &NavigationObstacle3D::get_rid);

	ClassDB::bind_method(D_METHOD("set_navigation_map", "navigation_map"), &NavigationObstacle3D::set_navigation_map);
	ClassDB::bind_method(D_METHOD("get_navigation_map"), &NavigationObstacle3D::get_navigation_map);

	ClassDB::bind_method(D_METHOD("set_estimate_radius", "estimate_radius"), &NavigationObstacle3D::set_estimate_radius);
	ClassDB::bind_method(D_METHOD("is_radius_estimated"), &NavigationObstacle3D::is_radius_estimated);
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationObstacle3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationObstacle3D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "estimate_radius"), "set_estimate_radius", "is_radius_estimated");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.01,100,0.01"), "set_radius", "get_radius");
}

void NavigationObstacle3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "radius") {
		if (estimate_radius) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
		}
	}
}

void NavigationObstacle3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POST_ENTER_TREE: {
			set_agent_parent(get_parent());
			set_physics_process_internal(true);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			set_agent_parent(nullptr);
			set_physics_process_internal(false);
		} break;

		case NOTIFICATION_PARENTED: {
			if (is_inside_tree() && (get_parent() != parent_node3d)) {
				set_agent_parent(get_parent());
				set_physics_process_internal(true);
			}
		} break;

		case NOTIFICATION_UNPARENTED: {
			set_agent_parent(nullptr);
			set_physics_process_internal(false);
		} break;

		case NOTIFICATION_PAUSED: {
			if (parent_node3d && !parent_node3d->can_process()) {
				map_before_pause = NavigationServer3D::get_singleton()->agent_get_map(get_rid());
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (parent_node3d && parent_node3d->can_process() && !(map_before_pause == RID())) {
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;

		case NOTIFICATION_UNPAUSED: {
			if (parent_node3d && !parent_node3d->can_process()) {
				map_before_pause = NavigationServer3D::get_singleton()->agent_get_map(get_rid());
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (parent_node3d && parent_node3d->can_process() && !(map_before_pause == RID())) {
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (parent_node3d && parent_node3d->is_inside_tree()) {
				NavigationServer3D::get_singleton()->agent_set_position(agent, parent_node3d->get_global_transform().origin);

				PhysicsBody3D *rigid = Object::cast_to<PhysicsBody3D>(get_parent());
				if (rigid) {
					Vector3 v = rigid->get_linear_velocity();
					NavigationServer3D::get_singleton()->agent_set_velocity(agent, v);
					NavigationServer3D::get_singleton()->agent_set_target_velocity(agent, v);
				}
			}
		} break;
	}
}

NavigationObstacle3D::NavigationObstacle3D() {
	agent = NavigationServer3D::get_singleton()->agent_create();
	initialize_agent();
}

NavigationObstacle3D::~NavigationObstacle3D() {
	ERR_FAIL_NULL(NavigationServer3D::get_singleton());
	NavigationServer3D::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

PackedStringArray NavigationObstacle3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	if (!Object::cast_to<Node3D>(get_parent())) {
		warnings.push_back(RTR("The NavigationObstacle3D only serves to provide collision avoidance to a Node3D inheriting parent object."));
	}

	if (Object::cast_to<StaticBody3D>(get_parent())) {
		warnings.push_back(RTR("The NavigationObstacle3D is intended for constantly moving bodies like CharacterBody3D or RigidBody3D as it creates only an RVO avoidance radius and does not follow scene geometry exactly."
							   "\nNot constantly moving or complete static objects should be (re)baked to a NavigationMesh so agents can not only avoid them but also move along those objects outline at high detail"));
	}

	return warnings;
}

void NavigationObstacle3D::initialize_agent() {
	NavigationServer3D::get_singleton()->agent_set_neighbor_distance(agent, 0.0);
	NavigationServer3D::get_singleton()->agent_set_max_neighbors(agent, 0);
	NavigationServer3D::get_singleton()->agent_set_time_horizon(agent, 0.0);
	NavigationServer3D::get_singleton()->agent_set_max_speed(agent, 0.0);
}

void NavigationObstacle3D::reevaluate_agent_radius() {
	if (!estimate_radius) {
		NavigationServer3D::get_singleton()->agent_set_radius(agent, radius);
	} else if (parent_node3d && parent_node3d->is_inside_tree()) {
		NavigationServer3D::get_singleton()->agent_set_radius(agent, estimate_agent_radius());
	}
}

real_t NavigationObstacle3D::estimate_agent_radius() const {
	if (parent_node3d && parent_node3d->is_inside_tree()) {
		// Estimate the radius of this physics body
		real_t max_radius = 0.0;
		for (int i(0); i < parent_node3d->get_child_count(); i++) {
			// For each collision shape
			CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(parent_node3d->get_child(i));
			if (cs && cs->is_inside_tree()) {
				// Take the distance between the Body center to the shape center
				real_t r = cs->get_transform().origin.length();
				if (cs->get_shape().is_valid()) {
					// and add the enclosing shape radius
					r += cs->get_shape()->get_enclosing_radius();
				}
				Vector3 s = cs->get_global_transform().basis.get_scale();
				r *= MAX(s.x, MAX(s.y, s.z));
				// Takes the biggest radius
				max_radius = MAX(max_radius, r);
			} else if (cs && !cs->is_inside_tree()) {
				WARN_PRINT("A CollisionShape3D of the NavigationObstacle3D parent node was not inside the SceneTree when estimating the obstacle radius."
						   "\nMove the NavigationObstacle3D to a child position below any CollisionShape3D node of the parent node so the CollisionShape3D is already inside the SceneTree.");
			}
		}

		Vector3 s = parent_node3d->get_global_transform().basis.get_scale();
		max_radius *= MAX(s.x, MAX(s.y, s.z));

		if (max_radius > 0.0) {
			return max_radius;
		}
	}
	return 1.0; // Never a 0 radius
}

void NavigationObstacle3D::set_agent_parent(Node *p_agent_parent) {
	if (parent_node3d == p_agent_parent) {
		return;
	}

	if (Object::cast_to<Node3D>(p_agent_parent) != nullptr) {
		parent_node3d = Object::cast_to<Node3D>(p_agent_parent);
		if (map_override.is_valid()) {
			NavigationServer3D::get_singleton()->agent_set_map(get_rid(), map_override);
		} else {
			NavigationServer3D::get_singleton()->agent_set_map(get_rid(), parent_node3d->get_world_3d()->get_navigation_map());
		}
		// Need to register Callback as obstacle requires a valid Callback to be added to avoidance simulation.
		NavigationServer3D::get_singleton()->agent_set_callback(get_rid(), callable_mp(this, &NavigationObstacle3D::_avoidance_done));
		reevaluate_agent_radius();
	} else {
		parent_node3d = nullptr;
		NavigationServer3D::get_singleton()->agent_set_map(get_rid(), RID());
		NavigationServer3D::get_singleton()->agent_set_callback(agent, Callable());
	}
}

void NavigationObstacle3D::_avoidance_done(Vector3 p_new_velocity) {
	// Dummy function as obstacle requires a valid Callback to be added to avoidance simulation.
}

void NavigationObstacle3D::set_navigation_map(RID p_navigation_map) {
	if (map_override == p_navigation_map) {
		return;
	}

	map_override = p_navigation_map;

	NavigationServer3D::get_singleton()->agent_set_map(agent, map_override);
}

RID NavigationObstacle3D::get_navigation_map() const {
	if (map_override.is_valid()) {
		return map_override;
	} else if (parent_node3d != nullptr) {
		return parent_node3d->get_world_3d()->get_navigation_map();
	}
	return RID();
}

void NavigationObstacle3D::set_estimate_radius(bool p_estimate_radius) {
	if (estimate_radius == p_estimate_radius) {
		return;
	}

	estimate_radius = p_estimate_radius;

	notify_property_list_changed();
	reevaluate_agent_radius();
}

void NavigationObstacle3D::set_radius(real_t p_radius) {
	ERR_FAIL_COND_MSG(p_radius <= 0.0, "Radius must be greater than 0.");
	if (Math::is_equal_approx(radius, p_radius)) {
		return;
	}

	radius = p_radius;

	reevaluate_agent_radius();
}
