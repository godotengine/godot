/*************************************************************************/
/*  navigation_obstacle_3d.cpp                                           */
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

#include "navigation_obstacle_3d.h"

#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "servers/navigation_server_3d.h"

void NavigationObstacle3D::_bind_methods() {
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
			p_property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}
}

void NavigationObstacle3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			initialize_agent();
			parent_node3d = Object::cast_to<Node3D>(get_parent());
			if (parent_node3d != nullptr) {
				// place agent on navigation map first or else the RVO agent callback creation fails silently later
				NavigationServer3D::get_singleton()->agent_set_map(get_rid(), parent_node3d->get_world_3d()->get_navigation_map());
			}
			set_physics_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			parent_node3d = nullptr;
			set_physics_process_internal(false);
		} break;
		case NOTIFICATION_PARENTED: {
			parent_node3d = Object::cast_to<Node3D>(get_parent());
			reevaluate_agent_radius();
		} break;
		case NOTIFICATION_UNPARENTED: {
			parent_node3d = nullptr;
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (parent_node3d) {
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
}

NavigationObstacle3D::~NavigationObstacle3D() {
	NavigationServer3D::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

TypedArray<String> NavigationObstacle3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!Object::cast_to<Node3D>(get_parent())) {
		warnings.push_back(TTR("The NavigationObstacle3D only serves to provide collision avoidance to a spatial object."));
	}

	return warnings;
}

void NavigationObstacle3D::initialize_agent() {
	NavigationServer3D::get_singleton()->agent_set_neighbor_dist(agent, 0.0);
	NavigationServer3D::get_singleton()->agent_set_max_neighbors(agent, 0);
	NavigationServer3D::get_singleton()->agent_set_time_horizon(agent, 0.0);
	NavigationServer3D::get_singleton()->agent_set_max_speed(agent, 0.0);
}

void NavigationObstacle3D::reevaluate_agent_radius() {
	if (!estimate_radius) {
		NavigationServer3D::get_singleton()->agent_set_radius(agent, radius);
	} else if (parent_node3d) {
		NavigationServer3D::get_singleton()->agent_set_radius(agent, estimate_agent_radius());
	}
}

real_t NavigationObstacle3D::estimate_agent_radius() const {
	if (parent_node3d) {
		// Estimate the radius of this physics body
		real_t radius = 0.0;
		for (int i(0); i < parent_node3d->get_child_count(); i++) {
			// For each collision shape
			CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(parent_node3d->get_child(i));
			if (cs) {
				// Take the distance between the Body center to the shape center
				real_t r = cs->get_transform().origin.length();
				if (cs->get_shape().is_valid()) {
					// and add the enclosing shape radius
					r += cs->get_shape()->get_enclosing_radius();
				}
				Vector3 s = cs->get_global_transform().basis.get_scale();
				r *= MAX(s.x, MAX(s.y, s.z));
				// Takes the biggest radius
				radius = MAX(radius, r);
			}
		}

		Vector3 s = parent_node3d->get_global_transform().basis.get_scale();
		radius *= MAX(s.x, MAX(s.y, s.z));

		if (radius > 0.0) {
			return radius;
		}
	}
	return 1.0; // Never a 0 radius
}

void NavigationObstacle3D::set_estimate_radius(bool p_estimate_radius) {
	estimate_radius = p_estimate_radius;
	notify_property_list_changed();
	reevaluate_agent_radius();
}

void NavigationObstacle3D::set_radius(real_t p_radius) {
	ERR_FAIL_COND_MSG(p_radius <= 0.0, "Radius must be greater than 0.");
	radius = p_radius;
	reevaluate_agent_radius();
}
