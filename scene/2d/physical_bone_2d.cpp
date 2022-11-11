/*************************************************************************/
/*  physical_bone_2d.cpp                                                 */
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

#include "physical_bone_2d.h"

#include "scene/2d/joint_2d.h"

void PhysicalBone2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			_find_joint_child();
			bone_node_cache = Variant(get_node(bone_nodepath));

			// Configure joint.
			if (child_joint && auto_configure_joint) {
				_auto_configure_joint();
			}

			// Simulate physics if set.
			if (simulate_physics) {
				_start_physics_simulation();
			} else {
				_stop_physics_simulation();
			}

			set_physics_process_internal(true);
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			// Position the RigidBody in the correct position.
			if (follow_bone_when_simulating) {
				_position_at_bone2d();
			}

			// Keep the child joint in the correct position.
			if (child_joint && auto_configure_joint) {
				child_joint->set_global_position(get_global_position());
			}
		} break;
			// Currently updated using SkeletonModification2DPhysicalBones
			//case NOTIFICATION_INTERNAL_PROCESS: {
			//	// Apply the simulation result. TODO: interpolation?
			//	if (simulate_physics && !follow_bone_when_simulating) {
			//		if (parent_skeleton) {
			//			Bone2D *bone_to_use = cast_to<Bone2D>((Object*)bone_node_cache);
			//			bone_to_use->set_global_transform(get_global_transform());
			//		}
			//	}
			//} break;
	}
}

Node2D *PhysicalBone2D::get_cached_bone_node() {
	return cast_to<Node2D>((Object *)bone_node_cache);
}

void PhysicalBone2D::_position_at_bone2d() {
	// Reset to Bone2D position
	Node2D *bone_to_use = get_cached_bone_node();
	ERR_FAIL_COND(bone_to_use == nullptr);
	set_global_transform(bone_to_use->get_global_transform());
}

void PhysicalBone2D::_find_joint_child() {
	for (int i = 0; i < get_child_count(); i++) {
		Node *child_node = get_child(i);
		Joint2D *potential_joint = Object::cast_to<Joint2D>(child_node);
		if (potential_joint) {
			child_joint = potential_joint;
			break;
		}
	}
}

PackedStringArray PhysicalBone2D::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	if (!child_joint) {
		PhysicalBone2D *parent_bone = Object::cast_to<PhysicalBone2D>(get_parent());
		if (parent_bone) {
			warnings.push_back(RTR("A PhysicalBone2D node should have a Joint2D-based child node to keep bones connected! Please add a Joint2D-based node as a child to this node!"));
		}
	}

	return warnings;
}

void PhysicalBone2D::_auto_configure_joint() {
	if (!auto_configure_joint) {
		return;
	}

	if (child_joint) {
		// Node A = parent | Node B = this node
		Node *parent_node = get_parent();
		PhysicalBone2D *potential_parent_bone = Object::cast_to<PhysicalBone2D>(parent_node);

		if (potential_parent_bone) {
			child_joint->set_node_a(child_joint->get_path_to(potential_parent_bone));
			child_joint->set_node_b(child_joint->get_path_to(this));
		} else {
			WARN_PRINT("Cannot setup joint without a parent PhysicalBone2D node.");
		}

		// Place the child joint at this node's position.
		child_joint->set_global_position(get_global_position());
	}
}

void PhysicalBone2D::_start_physics_simulation() {
	if (_internal_simulate_physics) {
		return;
	}

	// Reset to Bone2D position.
	_position_at_bone2d();

	// Apply the layers and masks.
	PhysicsServer2D::get_singleton()->body_set_collision_layer(get_rid(), get_collision_layer());
	PhysicsServer2D::get_singleton()->body_set_collision_mask(get_rid(), get_collision_mask());
	PhysicsServer2D::get_singleton()->body_set_collision_priority(get_rid(), get_collision_priority());

	// Apply the correct mode.
	set_freeze_mode(RigidBody2D::FREEZE_MODE_KINEMATIC);
	set_freeze_enabled(follow_bone_when_simulating ? true : false);
	_apply_body_mode();

	_internal_simulate_physics = true;
	set_physics_process_internal(true);
}

void PhysicalBone2D::_stop_physics_simulation() {
	if (_internal_simulate_physics) {
		_internal_simulate_physics = false;

		// Reset to Bone2D position
		_position_at_bone2d();

		set_physics_process_internal(false);
		PhysicsServer2D::get_singleton()->body_set_collision_layer(get_rid(), 0);
		PhysicsServer2D::get_singleton()->body_set_collision_mask(get_rid(), 0);
		PhysicsServer2D::get_singleton()->body_set_collision_priority(get_rid(), 1.0);

		// Apply the correct mode.
		set_freeze_mode(RigidBody2D::FREEZE_MODE_STATIC);
		set_freeze_enabled(true);
		_apply_body_mode();
	}
}

Joint2D *PhysicalBone2D::get_joint() const {
	return child_joint;
}

bool PhysicalBone2D::get_auto_configure_joint() const {
	return auto_configure_joint;
}

void PhysicalBone2D::set_auto_configure_joint(bool p_auto_configure) {
	auto_configure_joint = p_auto_configure;
	_auto_configure_joint();
}

void PhysicalBone2D::set_simulate_physics(bool p_simulate) {
	if (p_simulate == simulate_physics) {
		return;
	}
	simulate_physics = p_simulate;

	if (simulate_physics) {
		_start_physics_simulation();
	} else {
		_stop_physics_simulation();
	}
}

bool PhysicalBone2D::get_simulate_physics() const {
	return simulate_physics;
}

bool PhysicalBone2D::is_simulating_physics() const {
	return _internal_simulate_physics;
}

void PhysicalBone2D::set_bone_node(const NodePath &p_nodepath) {
	bone_nodepath = p_nodepath;
	if (is_inside_tree()) {
		bone_node_cache = get_node(bone_nodepath);
	}
	notify_property_list_changed();
}

NodePath PhysicalBone2D::get_bone_node() const {
	return bone_nodepath;
}

void PhysicalBone2D::set_follow_bone_when_simulating(bool p_follow_bone) {
	follow_bone_when_simulating = p_follow_bone;
	set_freeze_enabled(follow_bone_when_simulating ? true : false);
}

bool PhysicalBone2D::get_follow_bone_when_simulating() const {
	return follow_bone_when_simulating;
}

void PhysicalBone2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_joint"), &PhysicalBone2D::get_joint);
	ClassDB::bind_method(D_METHOD("get_auto_configure_joint"), &PhysicalBone2D::get_auto_configure_joint);
	ClassDB::bind_method(D_METHOD("set_auto_configure_joint", "auto_configure_joint"), &PhysicalBone2D::set_auto_configure_joint);

	ClassDB::bind_method(D_METHOD("set_simulate_physics", "simulate_physics"), &PhysicalBone2D::set_simulate_physics);
	ClassDB::bind_method(D_METHOD("get_simulate_physics"), &PhysicalBone2D::get_simulate_physics);
	ClassDB::bind_method(D_METHOD("is_simulating_physics"), &PhysicalBone2D::is_simulating_physics);

	ClassDB::bind_method(D_METHOD("get_cached_bone_node"), &PhysicalBone2D::get_cached_bone_node);
	ClassDB::bind_method(D_METHOD("set_bone_node", "nodepath"), &PhysicalBone2D::set_bone_node);
	ClassDB::bind_method(D_METHOD("get_bone_node"), &PhysicalBone2D::get_bone_node);
	ClassDB::bind_method(D_METHOD("set_follow_bone_when_simulating", "follow_bone"), &PhysicalBone2D::set_follow_bone_when_simulating);
	ClassDB::bind_method(D_METHOD("get_follow_bone_when_simulating"), &PhysicalBone2D::get_follow_bone_when_simulating);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "bone_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node2D"), "set_bone_node", "get_bone_node");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_configure_joint"), "set_auto_configure_joint", "get_auto_configure_joint");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "simulate_physics"), "set_simulate_physics", "get_simulate_physics");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "follow_bone_when_simulating"), "set_follow_bone_when_simulating", "get_follow_bone_when_simulating");
}

PhysicalBone2D::PhysicalBone2D() {
	// Stop the RigidBody from executing its force integration.
	PhysicsServer2D::get_singleton()->body_set_collision_layer(get_rid(), 0);
	PhysicsServer2D::get_singleton()->body_set_collision_mask(get_rid(), 0);
	PhysicsServer2D::get_singleton()->body_set_mode(get_rid(), PhysicsServer2D::BodyMode::BODY_MODE_STATIC);

	child_joint = nullptr;
}

PhysicalBone2D::~PhysicalBone2D() {
}
