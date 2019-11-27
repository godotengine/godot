/*************************************************************************/
/*  physics_bone_3d.cpp                                                  */
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

#include "physics_bone_3d.h"

#include "scene/3d/physics_bone_compensation_3d.h"

void PhysicsBone3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY:
			[[fallthrough]];
		case NOTIFICATION_ENTER_TREE: {
			update_skeleton_node(find_skeleton_parent());
			update_compensation_node(find_compensation_parent());
			update_bone_id();
			update_body_type();

			if (skeleton && bone_id != 1 && free_simulation) {
				const Transform gt = skeleton->get_global_transform() * skeleton->get_bone_global_pose_without_override(bone_id).translated(bone_offset);
				set_global_transform(gt);
			}

			break;
		}
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!skeleton || bone_id == -1) {
				break;
			}

			PhysicsDirectBodyState3D *state = PhysicsServer3D::get_singleton()->body_get_direct_state(get_rid());

			// If we are tracking a bone, on every physics tick we move the body based off of the skeletons bone location
			if (!free_simulation) {
				const Transform gt = skeleton->get_global_transform() * skeleton->get_bone_global_pose(bone_id).translated(bone_offset);
				state->teleport(gt);
				set_ignore_transform_notification(true);
				set_global_transform(gt);
				set_ignore_transform_notification(false);
				break;
			}

			// If we have just teleported due to an abrupt angle change of a compensation node, reset our state back to rigid mode
			// and skip the current physics frame.
			if (has_teleported) {
				PhysicsServer3D::get_singleton()->body_set_mode(get_rid(), PhysicsServer3D::BODY_MODE_RIGID);
				has_teleported = false;
				break;
			}

			// Finally, if we have a compensation node, apply the forces of that body onto ourselves.
			if (compensation_node) {
				apply_compensation(state);
			}

			break;
		}
		case NOTIFICATION_INTERNAL_PROCESS: {
			// If we are in editor, we bypass the free simulation state and update the global transform, otherwise the bones
			// would be stuck floating in the editor
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint() && skeleton && bone_id != -1) {
				const Transform gt = skeleton->get_global_transform() * skeleton->get_bone_global_pose(bone_id).translated(bone_offset);
				set_global_transform(gt);
				break;
			}
#endif

			// If we are free simulating - take the physics body's transform and translate it into a bone space global override
			if (free_simulation && skeleton && bone_id != -1) {
				skeleton->set_bone_global_pose_override(bone_id, skeleton->get_global_transform().affine_inverse() * get_global_transform().translated(-bone_offset), 1.0, true);
			}
			break;
		}
	}
}

void PhysicsBone3D::apply_compensation(PhysicsDirectBodyState3D *p_state) {
	const Vector3 &comp_origin = compensation_node->get_last_transform().origin; // the compensation nodes origin
	Transform comp_rotation = Transform(compensation_node->get_transform_delta().basis, Vector3()); // extract only the rotation transform

	// lambda to rotate our transform based off of the compensation nodes rotation
	const auto rotate_bone_fn = [&](Transform p_tform) -> Transform {
		p_tform.origin -= comp_origin;
		p_tform = comp_rotation * p_tform;
		p_tform.origin += comp_origin;
		return p_tform;
	};

	// If we need to teleport due to a drastic angle change, do that first
	if (compensation_node->get_rotation_angle_delta() >= rotation_teleport_threshold_angle) {
		const Transform rotated_gt = rotate_bone_fn(p_state->get_transform());
		p_state->teleport(rotated_gt);
		set_ignore_transform_notification(true);
		set_global_transform(rotated_gt);
		set_ignore_transform_notification(false);

		// Because we are teleporting, and other bones will *not* be caught up in this physics frame, we set the mode temporarily
		// to Kinematic. This gives all of our other constraints a frame to catch up and update before we resolved forces.
		// If you don't do this, you will see the physics bones glitch out as constraints are trying to solve between moved
		// and non-moved bones.
		PhysicsServer3D::get_singleton()->body_set_mode(get_rid(), PhysicsServer3D::BODY_MODE_KINEMATIC);
		has_teleported = true;
		return;
	}

	// Do we apply rotation compensation?
	// This directly affects the transform of the physics bones rigid body. Unlike the threshold teleport, we maintain existing
	// bone velocities.
	if (rotation_comp_amount >= CMP_EPSILON) {
		if (rotation_comp_amount <= 1 - CMP_EPSILON) {
			Quat rotation = comp_rotation.basis.get_rotation_quat();
			Quat slerped = Quat().slerp(rotation, rotation_comp_amount);
			comp_rotation.basis = Basis(slerped);
		}

		const Transform rotated_gt = rotate_bone_fn(p_state->get_transform());

		p_state->set_transform(rotated_gt);
		set_ignore_transform_notification(true);
		set_global_transform(rotated_gt);
		set_ignore_transform_notification(false);
	}

	// Are we compensating for velocity?
	if (velocity_comp_amount >= CMP_EPSILON) {
		const Vector3 linear_velocity = p_state->get_linear_velocity() + (compensation_node->get_linear_velocity_delta() * velocity_comp_amount);
		const Vector3 angular_velocity = p_state->get_angular_velocity() + (compensation_node->get_angular_velocity_delta() * velocity_comp_amount);

		p_state->set_linear_velocity(linear_velocity);
		p_state->set_angular_velocity(angular_velocity);
	}
}

void PhysicsBone3D::set_bone_name(const String &p_name) {
	bone_name = p_name;
	update_bone_id();
}

void PhysicsBone3D::set_bone_offset(const Vector3 &p_offset) {
	bone_offset = p_offset;
}

Transform PhysicsBone3D::get_p_body_global_rest() const {
	if (!skeleton || bone_id == -1)
		return Transform();

	return skeleton->get_bone_global_rest(bone_id).translated(bone_offset);
}

void PhysicsBone3D::_direct_state_changed(Object *p_state) {
	//	If we are locked to the bone, we just teleport our state
	if (!free_simulation || has_teleported) {
		return;
	}

#ifdef DEBUG_ENABLED
	state = Object::cast_to<PhysicsDirectBodyState3D>(p_state);
#else
	state = (PhysicsDirectBodyState3D *)p_state; //trust it
#endif

	RigidBody3D::_direct_state_changed(p_state);
}

void PhysicsBone3D::_get_property_list(List<PropertyInfo> *p_list) const {
	return;
}

void PhysicsBone3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &PhysicsBone3D::_direct_state_changed);

	ClassDB::bind_method(D_METHOD("get_bone_id"), &PhysicsBone3D::get_bone_id);

	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_name"), &PhysicsBone3D::set_bone_name);
	ClassDB::bind_method(D_METHOD("get_bone_name"), &PhysicsBone3D::get_bone_name);

	ClassDB::bind_method(D_METHOD("set_free_simulation", "free_sim"), &PhysicsBone3D::set_free_simulation);
	ClassDB::bind_method(D_METHOD("get_free_simulation"), &PhysicsBone3D::get_free_simulation);

	ClassDB::bind_method(D_METHOD("set_bone_offset", "offset"), &PhysicsBone3D::set_bone_offset);
	ClassDB::bind_method(D_METHOD("get_bone_offset"), &PhysicsBone3D::get_bone_offset);

	ClassDB::bind_method(D_METHOD("set_velocity_compensation_amount", "amount"), &PhysicsBone3D::set_velocity_compensation_amount);
	ClassDB::bind_method(D_METHOD("get_velocity_compensation_amount"), &PhysicsBone3D::get_velocity_compensation_amount);

	ClassDB::bind_method(D_METHOD("set_rotation_compensation_amount", "amount"), &PhysicsBone3D::set_rotation_compensation_amount);
	ClassDB::bind_method(D_METHOD("get_rotation_compensation_amount"), &PhysicsBone3D::get_rotation_compensation_amount);

	ClassDB::bind_method(D_METHOD("set_rotation_teleport_threshold_angle", "angle"), &PhysicsBone3D::set_rotation_teleport_threshold_angle);
	ClassDB::bind_method(D_METHOD("get_rotation_teleport_threshold_angle"), &PhysicsBone3D::get_rotation_teleport_threshold_angle);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "bone_name"), "set_bone_name", "get_bone_name");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "free_simulation"), "set_free_simulation", "get_free_simulation");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "bone_offset"), "set_bone_offset", "get_bone_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "velocity_compensation_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_velocity_compensation_amount", "get_velocity_compensation_amount");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation_compensation_amount", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_rotation_compensation_amount", "get_rotation_compensation_amount");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rotation_teleport_threshold_angle", PROPERTY_HINT_RANGE, "0," + rtos(Math_PI * 2.0) + ",0.01"), "set_rotation_teleport_threshold_angle", "get_rotation_teleport_threshold_angle");
}

void PhysicsBone3D::set_free_simulation(bool p_free_sim) {
	if (p_free_sim == free_simulation) {
		return;
	}

	free_simulation = p_free_sim;

	if (skeleton && bone_id != -1 && !p_free_sim) {
		skeleton->set_bone_global_pose_override(bone_id, Transform(), 0.0, false);
	}

	update_body_type();
}

PhysicsBoneCompensation3D *PhysicsBone3D::find_compensation_parent(Node *p_parent) {
	if (!p_parent) {
		return nullptr;
	}

	PhysicsBoneCompensation3D *p = Object::cast_to<PhysicsBoneCompensation3D>(p_parent);
	return p ? p : find_compensation_parent(p_parent->get_parent());
}

PhysicsBoneCompensation3D *PhysicsBone3D::find_compensation_parent() {
	return find_compensation_parent(this);
}

Skeleton3D *PhysicsBone3D::find_skeleton_parent(Node *p_parent) {
	if (!p_parent) {
		return nullptr;
	}
	Skeleton3D *s = Object::cast_to<Skeleton3D>(p_parent);
	return s ? s : find_skeleton_parent(p_parent->get_parent());
}

Skeleton3D *PhysicsBone3D::find_skeleton_parent() {
	return find_skeleton_parent(this);
}

void PhysicsBone3D::update_compensation_node(PhysicsBoneCompensation3D *p_comp_node) {
	compensation_node = p_comp_node;
}

void PhysicsBone3D::update_skeleton_node(Skeleton3D *p_skeleton) {
	skeleton = p_skeleton;
}

void PhysicsBone3D::update_bone_id() {
	if (!skeleton)
		return;

	bone_id = skeleton->find_bone(bone_name);
}

void PhysicsBone3D::update_body_type() {
	const PhysicsServer3D::BodyMode body_mode = free_simulation ? PhysicsServer3D::BODY_MODE_RIGID : PhysicsServer3D::BODY_MODE_KINEMATIC;
	PhysicsServer3D::get_singleton()->body_set_mode(get_rid(), body_mode);
}

PhysicsBone3D::PhysicsBone3D() {
	PhysicsServer3D::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");
	set_as_toplevel(true);
	set_physics_process_internal(true);
	set_process_internal(true);
}

PhysicsBone3D::~PhysicsBone3D() {
}
