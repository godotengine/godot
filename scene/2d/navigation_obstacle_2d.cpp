/*************************************************************************/
/*  navigation_obstacle_2d.cpp                                           */
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

#include "navigation_obstacle_2d.h"

#include "scene/2d/collision_shape_2d.h"
#include "servers/navigation_server_2d.h"

void NavigationObstacle2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_estimate_radius", "estimate_radius"), &NavigationObstacle2D::set_estimate_radius);
	ClassDB::bind_method(D_METHOD("is_radius_estimated"), &NavigationObstacle2D::is_radius_estimated);
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationObstacle2D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationObstacle2D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "estimate_radius"), "set_estimate_radius", "is_radius_estimated");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.01,500,0.01"), "set_radius", "get_radius");
}

void NavigationObstacle2D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "radius") {
		if (estimate_radius) {
			p_property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}
}

void NavigationObstacle2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			initialize_agent();
			parent_node2d = Object::cast_to<Node2D>(get_parent());
			if (parent_node2d != nullptr) {
				// place agent on navigation map first or else the RVO agent callback creation fails silently later
				NavigationServer2D::get_singleton()->agent_set_map(get_rid(), parent_node2d->get_world_2d()->get_navigation_map());
			}
			set_physics_process_internal(true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			parent_node2d = nullptr;
			set_physics_process_internal(false);
		} break;
		case NOTIFICATION_PARENTED: {
			parent_node2d = Object::cast_to<Node2D>(get_parent());
			reevaluate_agent_radius();
		} break;
		case NOTIFICATION_UNPARENTED: {
			parent_node2d = nullptr;
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (parent_node2d) {
				NavigationServer2D::get_singleton()->agent_set_position(agent, parent_node2d->get_global_position());
			}
		} break;
	}
}

NavigationObstacle2D::NavigationObstacle2D() {
	agent = NavigationServer2D::get_singleton()->agent_create();
}

NavigationObstacle2D::~NavigationObstacle2D() {
	NavigationServer2D::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

TypedArray<String> NavigationObstacle2D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!Object::cast_to<Node2D>(get_parent())) {
		warnings.push_back(TTR("The NavigationObstacle2D only serves to provide collision avoidance to a Node2D object."));
	}

	return warnings;
}

void NavigationObstacle2D::initialize_agent() {
	NavigationServer2D::get_singleton()->agent_set_neighbor_dist(agent, 0.0);
	NavigationServer2D::get_singleton()->agent_set_max_neighbors(agent, 0);
	NavigationServer2D::get_singleton()->agent_set_time_horizon(agent, 0.0);
	NavigationServer2D::get_singleton()->agent_set_max_speed(agent, 0.0);
}

void NavigationObstacle2D::reevaluate_agent_radius() {
	if (!estimate_radius) {
		NavigationServer2D::get_singleton()->agent_set_radius(agent, radius);
	} else if (parent_node2d) {
		NavigationServer2D::get_singleton()->agent_set_radius(agent, estimate_agent_radius());
	}
}

real_t NavigationObstacle2D::estimate_agent_radius() const {
	if (parent_node2d) {
		// Estimate the radius of this physics body
		real_t radius = 0.0;
		for (int i(0); i < parent_node2d->get_child_count(); i++) {
			// For each collision shape
			CollisionShape2D *cs = Object::cast_to<CollisionShape2D>(parent_node2d->get_child(i));
			if (cs) {
				// Take the distance between the Body center to the shape center
				real_t r = cs->get_transform().get_origin().length();
				if (cs->get_shape().is_valid()) {
					// and add the enclosing shape radius
					r += cs->get_shape()->get_enclosing_radius();
				}
				Size2 s = cs->get_global_scale();
				r *= MAX(s.x, s.y);
				// Takes the biggest radius
				radius = MAX(radius, r);
			}
		}
		Vector2 s = parent_node2d->get_global_scale();
		radius *= MAX(s.x, s.y);

		if (radius > 0.0) {
			return radius;
		}
	}
	return 1.0; // Never a 0 radius
}

void NavigationObstacle2D::set_estimate_radius(bool p_estimate_radius) {
	estimate_radius = p_estimate_radius;
	notify_property_list_changed();
	reevaluate_agent_radius();
}

void NavigationObstacle2D::set_radius(real_t p_radius) {
	ERR_FAIL_COND_MSG(p_radius <= 0.0, "Radius must be greater than 0.");
	radius = p_radius;
	reevaluate_agent_radius();
}
