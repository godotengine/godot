/*************************************************************************/
/*  navigation_obstacle_2d.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "scene/2d/navigation_2d.h"
#include "scene/2d/physics_body_2d.h"
#include "servers/navigation_server_2d.h"

void NavigationObstacle2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_navigation", "navigation"), &NavigationObstacle2D::set_navigation_node);
	ClassDB::bind_method(D_METHOD("get_navigation"), &NavigationObstacle2D::get_navigation_node);
}

void NavigationObstacle2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			update_agent_shape();

			// Search the navigation node and set it
			{
				Navigation2D *nav = nullptr;
				Node *p = get_parent();
				while (p != nullptr) {
					nav = Object::cast_to<Navigation2D>(p);
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
			set_navigation(nullptr);
			set_physics_process_internal(false);
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			Node2D *node = Object::cast_to<Node2D>(get_parent());
			if (node) {
				NavigationServer2D::get_singleton()->agent_set_position(agent, node->get_global_transform().get_origin());
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

void NavigationObstacle2D::set_navigation(Navigation2D *p_nav) {
	if (navigation == p_nav) {
		return; // Pointless
	}

	navigation = p_nav;
	NavigationServer2D::get_singleton()->agent_set_map(agent, navigation == nullptr ? RID() : navigation->get_rid());
}

void NavigationObstacle2D::set_navigation_node(Node *p_nav) {
	Navigation2D *nav = Object::cast_to<Navigation2D>(p_nav);
	ERR_FAIL_COND(nav == nullptr);
	set_navigation(nav);
}

Node *NavigationObstacle2D::get_navigation_node() const {
	return Object::cast_to<Node>(navigation);
}

String NavigationObstacle2D::get_configuration_warning() const {
	String warning = Node::get_configuration_warning();

	if (!Object::cast_to<Node2D>(get_parent())) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("The NavigationObstacle2D only serves to provide collision avoidance to a Node2D object.");
	}

	return warning;
}

void NavigationObstacle2D::update_agent_shape() {
	Node *node = get_parent();

	// Estimate the radius of this physics body
	real_t radius = 0.0;
	for (int i(0); i < node->get_child_count(); i++) {
		// For each collision shape
		CollisionShape2D *cs = Object::cast_to<CollisionShape2D>(node->get_child(i));
		if (cs) {
			// Take the distance between the Body center to the shape center
			real_t r = cs->get_transform().get_origin().length();
			if (cs->get_shape().is_valid()) {
				// and add the enclosing shape radius
				r += cs->get_shape()->get_enclosing_radius();
			}
			Size2 s = cs->get_global_transform().get_scale();
			r *= MAX(s.x, s.y);
			// Takes the biggest radius
			radius = MAX(radius, r);
		}
	}
	Node2D *node_2d = Object::cast_to<Node2D>(node);
	if (node_2d) {
		Vector2 s = node_2d->get_global_transform().get_scale();
		radius *= MAX(s.x, s.y);
	}

	if (radius == 0.0) {
		radius = 1.0; // Never a 0 radius
	}

	// Initialize the Agent as an object
	NavigationServer2D::get_singleton()->agent_set_neighbor_dist(agent, 0.0);
	NavigationServer2D::get_singleton()->agent_set_max_neighbors(agent, 0);
	NavigationServer2D::get_singleton()->agent_set_time_horizon(agent, 0.0);
	NavigationServer2D::get_singleton()->agent_set_radius(agent, radius);
	NavigationServer2D::get_singleton()->agent_set_max_speed(agent, 0.0);
}
