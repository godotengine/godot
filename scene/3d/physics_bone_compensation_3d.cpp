/*************************************************************************/
/*  physics_bone_compensation_3d.cpp                                     */
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

#include "physics_bone_compensation_3d.h"

#include "scene/3d/physics_body_3d.h"
#include "scene/3d/physics_bone_3d.h"

void PhysicsBoneCompensation3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			update_tracking_node();
		}
		case NOTIFICATION_EXIT_TREE: {
			break;
		}
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!tracking_node) {
				return;
			}

			PhysicsDirectBodyState3D *body = PhysicsServer3D::get_singleton()->body_get_direct_state(tracking_node->get_rid());

			const Transform current_transform = body->get_transform();
			transform_delta = last_transform.affine_inverse() * current_transform;

			Vector3 axis_unused;
			transform_delta.basis.get_rotation_axis_angle(axis_unused, rotation_delta_angle);

			last_transform = current_transform;

			Vector3 current_angular_velocity = body->get_angular_velocity();
			Vector3 current_linear_velocity = body->get_linear_velocity();

			angular_velocity_delta = current_angular_velocity - last_angular_velocity;
			linear_velocity_delta = current_linear_velocity - last_linear_velocity;

			last_angular_velocity = current_angular_velocity;
			last_linear_velocity = current_linear_velocity;
		}
	}
}

void PhysicsBoneCompensation3D::_get_property_list(List<PropertyInfo> *p_list) const {
}

void PhysicsBoneCompensation3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tracking_node_path", "path"), &PhysicsBoneCompensation3D::set_tracking_node_path);
	ClassDB::bind_method(D_METHOD("get_tracking_node_path"), &PhysicsBoneCompensation3D::get_tracking_node_path);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "tracking_node_path", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PhysicsBody3D"), "set_tracking_node_path", "get_tracking_node_path");
}

void PhysicsBoneCompensation3D::set_tracking_node_path(const NodePath &p_path) {
	if (tracking_node_path == p_path)
		return;

	tracking_node_path = p_path;
	update_tracking_node();
}

void PhysicsBoneCompensation3D::update_tracking_node() {
	Node *node = has_node(tracking_node_path) ? get_node(tracking_node_path) : nullptr;

	tracking_node = Object::cast_to<PhysicsBody3D>(node);

	set_physics_process_internal(tracking_node != nullptr);
}

PhysicsBoneCompensation3D::PhysicsBoneCompensation3D() {
}

PhysicsBoneCompensation3D::~PhysicsBoneCompensation3D() {
}
