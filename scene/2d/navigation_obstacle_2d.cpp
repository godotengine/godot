/*************************************************************************/
/*  navigation_obstacle_2d.cpp                                           */
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

#include "navigation_obstacle_2d.h"

#include "scene/2d/collision_shape_2d.h"
#include "scene/2d/navigation_2d.h"
#include "scene/2d/physics_body_2d.h"
#include "servers/navigation_2d_server.h"

void NavigationObstacle2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &NavigationObstacle2D::get_rid);

	ClassDB::bind_method(D_METHOD("set_navigation", "navigation"), &NavigationObstacle2D::set_navigation_node);
	ClassDB::bind_method(D_METHOD("get_navigation"), &NavigationObstacle2D::get_navigation_node);

	ClassDB::bind_method(D_METHOD("set_estimate_radius", "estimate_radius"), &NavigationObstacle2D::set_estimate_radius);
	ClassDB::bind_method(D_METHOD("is_radius_estimated"), &NavigationObstacle2D::is_radius_estimated);
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NavigationObstacle2D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NavigationObstacle2D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "estimate_radius"), "set_estimate_radius", "is_radius_estimated");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "0.01,500,0.01"), "set_radius", "get_radius");
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
		case NOTIFICATION_ENTER_TREE: {
			parent_node2d = Object::cast_to<Node2D>(get_parent());
			reevaluate_agent_radius();

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
			request_ready(); // required to solve an issue with losing the navigation
		} break;
		case NOTIFICATION_PARENTED: {
			parent_node2d = Object::cast_to<Node2D>(get_parent());
			reevaluate_agent_radius();
		} break;
		case NOTIFICATION_UNPARENTED: {
			parent_node2d = nullptr;
		} break;
		case NOTIFICATION_PAUSED: {
			if (parent_node2d && !parent_node2d->can_process()) {
				map_before_pause = Navigation2DServer::get_singleton()->agent_get_map(get_rid());
				Navigation2DServer::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (parent_node2d && parent_node2d->can_process() && !(map_before_pause == RID())) {
				Navigation2DServer::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;
		case NOTIFICATION_UNPAUSED: {
			if (parent_node2d && !parent_node2d->can_process()) {
				map_before_pause = Navigation2DServer::get_singleton()->agent_get_map(get_rid());
				Navigation2DServer::get_singleton()->agent_set_map(get_rid(), RID());
			} else if (parent_node2d && parent_node2d->can_process() && !(map_before_pause == RID())) {
				Navigation2DServer::get_singleton()->agent_set_map(get_rid(), map_before_pause);
				map_before_pause = RID();
			}
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (parent_node2d && parent_node2d->is_inside_tree()) {
				Navigation2DServer::get_singleton()->agent_set_position(agent, parent_node2d->get_global_transform().get_origin());
			}

		} break;
	}
}

NavigationObstacle2D::NavigationObstacle2D() :
		navigation(nullptr),
		agent(RID()) {
	agent = Navigation2DServer::get_singleton()->agent_create();
	initialize_agent();
}

NavigationObstacle2D::~NavigationObstacle2D() {
	Navigation2DServer::get_singleton()->free(agent);
	agent = RID(); // Pointless
}

void NavigationObstacle2D::set_navigation(Navigation2D *p_nav) {
	if (navigation == p_nav) {
		return; // Pointless
	}

	navigation = p_nav;

	if (navigation != nullptr) {
		Navigation2DServer::get_singleton()->agent_set_map(agent, navigation->get_rid());
	} else if (parent_node2d && parent_node2d->is_inside_tree()) {
		Navigation2DServer::get_singleton()->agent_set_map(agent, parent_node2d->get_world_2d()->get_navigation_map());
	}
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
	if (!Object::cast_to<Node2D>(get_parent())) {
		return TTR("The NavigationObstacle2D only serves to provide collision avoidance to a Node2D object.");
	}

	if (Object::cast_to<StaticBody2D>(get_parent())) {
		return TTR("The NavigationObstacle2D is intended for constantly moving bodies like KinematicBody2D or RigidBody2D as it creates only an RVO avoidance radius and does not follow scene geometry exactly."
				   "\nNot constantly moving or complete static objects should be captured with a refreshed NavigationPolygon so agents can not only avoid them but also move along those objects outline at high detail");
	}

	return String();
}

void NavigationObstacle2D::initialize_agent() {
	Navigation2DServer::get_singleton()->agent_set_neighbor_dist(agent, 0.0);
	Navigation2DServer::get_singleton()->agent_set_max_neighbors(agent, 0);
	Navigation2DServer::get_singleton()->agent_set_time_horizon(agent, 0.0);
	Navigation2DServer::get_singleton()->agent_set_max_speed(agent, 0.0);
}

void NavigationObstacle2D::reevaluate_agent_radius() {
	if (!estimate_radius) {
		Navigation2DServer::get_singleton()->agent_set_radius(agent, radius);
	} else if (parent_node2d && parent_node2d->is_inside_tree()) {
		Navigation2DServer::get_singleton()->agent_set_radius(agent, estimate_agent_radius());
	}
}

real_t NavigationObstacle2D::estimate_agent_radius() const {
	if (parent_node2d && parent_node2d->is_inside_tree()) {
		// Estimate the radius of this physics body
		real_t radius = 0.0;
		for (int i(0); i < parent_node2d->get_child_count(); i++) {
			// For each collision shape
			CollisionShape2D *cs = Object::cast_to<CollisionShape2D>(parent_node2d->get_child(i));
			if (cs && cs->is_inside_tree()) {
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
			} else if (cs && !cs->is_inside_tree()) {
				WARN_PRINT("A CollisionShape2D of the NavigationObstacle2D parent node was not inside the SceneTree when estimating the obstacle radius."
						   "\nMove the NavigationObstacle2D to a child position below any CollisionShape2D node of the parent node so the CollisionShape2D is already inside the SceneTree.");
			}
		}
		Vector2 s = parent_node2d->get_global_transform().get_scale();
		radius *= MAX(s.x, s.y);

		if (radius > 0.0) {
			return radius;
		}
	}
	return 1.0; // Never a 0 radius
}

void NavigationObstacle2D::set_estimate_radius(bool p_estimate_radius) {
	estimate_radius = p_estimate_radius;
	_change_notify();
	reevaluate_agent_radius();
}

void NavigationObstacle2D::set_radius(real_t p_radius) {
	ERR_FAIL_COND_MSG(p_radius <= 0.0, "Radius must be greater than 0.");
	radius = p_radius;
	reevaluate_agent_radius();
}
