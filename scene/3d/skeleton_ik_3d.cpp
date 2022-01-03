/*************************************************************************/
/*  skeleton_ik_3d.cpp                                                   */
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

/**
 * @author AndreaCatania
 */

#include "skeleton_ik_3d.h"

#ifndef _3D_DISABLED

FabrikInverseKinematic::ChainItem *FabrikInverseKinematic::ChainItem::find_child(const BoneId p_bone_id) {
	for (int i = children.size() - 1; 0 <= i; --i) {
		if (p_bone_id == children[i].bone) {
			return &children.write[i];
		}
	}
	return nullptr;
}

FabrikInverseKinematic::ChainItem *FabrikInverseKinematic::ChainItem::add_child(const BoneId p_bone_id) {
	const int infant_child_id = children.size();
	children.resize(infant_child_id + 1);
	children.write[infant_child_id].bone = p_bone_id;
	children.write[infant_child_id].parent_item = this;
	return &children.write[infant_child_id];
}

/// Build a chain that starts from the root to tip
bool FabrikInverseKinematic::build_chain(Task *p_task, bool p_force_simple_chain) {
	ERR_FAIL_COND_V(-1 == p_task->root_bone, false);

	Chain &chain(p_task->chain);

	chain.tips.resize(p_task->end_effectors.size());
	chain.chain_root.bone = p_task->root_bone;
	chain.chain_root.initial_transform = p_task->skeleton->get_bone_global_pose(chain.chain_root.bone);
	chain.chain_root.current_pos = chain.chain_root.initial_transform.origin;
	chain.middle_chain_item = nullptr;

	// Holds all IDs that are composing a single chain in reverse order
	Vector<BoneId> chain_ids;
	// This is used to know the chain size
	int sub_chain_size;
	// Resize only one time in order to fit all joints for performance reason
	chain_ids.resize(p_task->skeleton->get_bone_count());

	for (int x = p_task->end_effectors.size() - 1; 0 <= x; --x) {
		const EndEffector *ee(&p_task->end_effectors[x]);
		ERR_FAIL_COND_V(p_task->root_bone >= ee->tip_bone, false);
		ERR_FAIL_INDEX_V(ee->tip_bone, p_task->skeleton->get_bone_count(), false);

		sub_chain_size = 0;
		// Picks all IDs that composing a single chain in reverse order (except the root)
		BoneId chain_sub_tip(ee->tip_bone);
		while (chain_sub_tip > p_task->root_bone) {
			chain_ids.write[sub_chain_size++] = chain_sub_tip;
			chain_sub_tip = p_task->skeleton->get_bone_parent(chain_sub_tip);
		}

		BoneId middle_chain_item_id = (BoneId)(sub_chain_size * 0.5);

		// Build chain by reading chain ids in reverse order
		// For each chain item id will be created a ChainItem if doesn't exists
		ChainItem *sub_chain(&chain.chain_root);
		for (int i = sub_chain_size - 1; 0 <= i; --i) {
			ChainItem *child_ci(sub_chain->find_child(chain_ids[i]));
			if (!child_ci) {
				child_ci = sub_chain->add_child(chain_ids[i]);

				child_ci->initial_transform = p_task->skeleton->get_bone_global_pose(child_ci->bone);
				child_ci->current_pos = child_ci->initial_transform.origin;

				if (child_ci->parent_item) {
					child_ci->length = child_ci->parent_item->current_pos.distance_to(child_ci->current_pos);
				}
			}

			sub_chain = child_ci;

			if (middle_chain_item_id == i) {
				chain.middle_chain_item = child_ci;
			}
		}

		if (!middle_chain_item_id) {
			chain.middle_chain_item = nullptr;
		}

		// Initialize current tip
		chain.tips.write[x].chain_item = sub_chain;
		chain.tips.write[x].end_effector = ee;

		if (p_force_simple_chain) {
			// NOTE:
			//	This is a "hack" that force to create only one tip per chain since the solver of multi tip (end effector)
			//	is not yet created.
			//	Remove this code when this is done
			break;
		}
	}
	return true;
}

void FabrikInverseKinematic::solve_simple(Task *p_task, bool p_solve_magnet, Vector3 p_origin_pos) {
	real_t distance_to_goal(1e4);
	real_t previous_distance_to_goal(0);
	int can_solve(p_task->max_iterations);
	while (distance_to_goal > p_task->min_distance && Math::abs(previous_distance_to_goal - distance_to_goal) > 0.005 && can_solve) {
		previous_distance_to_goal = distance_to_goal;
		--can_solve;

		solve_simple_backwards(p_task->chain, p_solve_magnet);
		solve_simple_forwards(p_task->chain, p_solve_magnet, p_origin_pos);

		distance_to_goal = p_task->chain.tips[0].end_effector->goal_transform.origin.distance_to(p_task->chain.tips[0].chain_item->current_pos);
	}
}

void FabrikInverseKinematic::solve_simple_backwards(Chain &r_chain, bool p_solve_magnet) {
	if (p_solve_magnet && !r_chain.middle_chain_item) {
		return;
	}

	Vector3 goal;
	ChainItem *sub_chain_tip;
	if (p_solve_magnet) {
		goal = r_chain.magnet_position;
		sub_chain_tip = r_chain.middle_chain_item;
	} else {
		goal = r_chain.tips[0].end_effector->goal_transform.origin;
		sub_chain_tip = r_chain.tips[0].chain_item;
	}

	while (sub_chain_tip) {
		sub_chain_tip->current_pos = goal;

		if (sub_chain_tip->parent_item) {
			// Not yet in the chain root
			// So calculate next goal location

			const Vector3 look_parent((sub_chain_tip->parent_item->current_pos - sub_chain_tip->current_pos).normalized());
			goal = sub_chain_tip->current_pos + (look_parent * sub_chain_tip->length);

			// [TODO] Constraints goes here
		}

		sub_chain_tip = sub_chain_tip->parent_item;
	}
}

void FabrikInverseKinematic::solve_simple_forwards(Chain &r_chain, bool p_solve_magnet, Vector3 p_origin_pos) {
	if (p_solve_magnet && !r_chain.middle_chain_item) {
		return;
	}

	ChainItem *sub_chain_root(&r_chain.chain_root);
	Vector3 origin = p_origin_pos;

	while (sub_chain_root) { // Reach the tip
		sub_chain_root->current_pos = origin;

		if (!sub_chain_root->children.is_empty()) {
			ChainItem &child(sub_chain_root->children.write[0]);

			// Is not tip
			// So calculate next origin location

			// Look child
			sub_chain_root->current_ori = (child.current_pos - sub_chain_root->current_pos).normalized();
			origin = sub_chain_root->current_pos + (sub_chain_root->current_ori * child.length);

			// [TODO] Constraints goes here

			if (p_solve_magnet && sub_chain_root == r_chain.middle_chain_item) {
				// In case of magnet solving this is the tip
				sub_chain_root = nullptr;
			} else {
				sub_chain_root = &child;
			}
		} else {
			// Is tip
			sub_chain_root = nullptr;
		}
	}
}

FabrikInverseKinematic::Task *FabrikInverseKinematic::create_simple_task(Skeleton3D *p_sk, BoneId root_bone, BoneId tip_bone, const Transform3D &goal_transform) {
	FabrikInverseKinematic::EndEffector ee;
	ee.tip_bone = tip_bone;

	Task *task(memnew(Task));
	task->skeleton = p_sk;
	task->root_bone = root_bone;
	task->end_effectors.push_back(ee);
	task->goal_global_transform = goal_transform;

	if (!build_chain(task)) {
		free_task(task);
		return nullptr;
	}

	return task;
}

void FabrikInverseKinematic::free_task(Task *p_task) {
	if (p_task) {
		memdelete(p_task);
	}
}

void FabrikInverseKinematic::set_goal(Task *p_task, const Transform3D &p_goal) {
	p_task->goal_global_transform = p_goal;
}

void FabrikInverseKinematic::make_goal(Task *p_task, const Transform3D &p_inverse_transf, real_t blending_delta) {
	if (blending_delta >= 0.99f) {
		// Update the end_effector (local transform) without blending
		p_task->end_effectors.write[0].goal_transform = p_inverse_transf * p_task->goal_global_transform;
	} else {
		// End effector in local transform
		const Transform3D end_effector_pose(p_task->skeleton->get_bone_global_pose_no_override(p_task->end_effectors[0].tip_bone));

		// Update the end_effector (local transform) by blending with current pose
		p_task->end_effectors.write[0].goal_transform = end_effector_pose.interpolate_with(p_inverse_transf * p_task->goal_global_transform, blending_delta);
	}
}

void FabrikInverseKinematic::solve(Task *p_task, real_t blending_delta, bool override_tip_basis, bool p_use_magnet, const Vector3 &p_magnet_position) {
	if (blending_delta <= 0.01f) {
		// Before skipping, make sure we undo the global pose overrides
		ChainItem *ci(&p_task->chain.chain_root);
		while (ci) {
			p_task->skeleton->set_bone_global_pose_override(ci->bone, ci->initial_transform, 0.0, false);

			if (!ci->children.is_empty()) {
				ci = &ci->children.write[0];
			} else {
				ci = nullptr;
			}
		}

		return; // Skip solving
	}

	// Update the initial root transform so its synced with any animation changes
	_update_chain(p_task->skeleton, &p_task->chain.chain_root);

	p_task->skeleton->set_bone_global_pose_override(p_task->chain.chain_root.bone, Transform3D(), 0.0, false);
	Vector3 origin_pos = p_task->skeleton->get_bone_global_pose(p_task->chain.chain_root.bone).origin;

	make_goal(p_task, p_task->skeleton->get_global_transform().affine_inverse(), blending_delta);

	if (p_use_magnet && p_task->chain.middle_chain_item) {
		p_task->chain.magnet_position = p_task->chain.middle_chain_item->initial_transform.origin.lerp(p_magnet_position, blending_delta);
		solve_simple(p_task, true, origin_pos);
	}
	solve_simple(p_task, false, origin_pos);

	// Assign new bone position.
	ChainItem *ci(&p_task->chain.chain_root);
	while (ci) {
		Transform3D new_bone_pose(ci->initial_transform);
		new_bone_pose.origin = ci->current_pos;

		if (!ci->children.is_empty()) {
			p_task->skeleton->update_bone_rest_forward_vector(ci->bone);
			Vector3 forward_vector = p_task->skeleton->get_bone_axis_forward_vector(ci->bone);
			// Rotate the bone towards the next bone in the chain:
			new_bone_pose.basis.rotate_to_align(forward_vector, new_bone_pose.origin.direction_to(ci->children[0].current_pos));

		} else {
			// Set target orientation to tip
			if (override_tip_basis) {
				new_bone_pose.basis = p_task->chain.tips[0].end_effector->goal_transform.basis;
			} else {
				new_bone_pose.basis = new_bone_pose.basis * p_task->chain.tips[0].end_effector->goal_transform.basis;
			}
		}

		// IK should not affect scale, so undo any scaling
		new_bone_pose.basis.orthonormalize();
		new_bone_pose.basis.scale(p_task->skeleton->get_bone_global_pose(ci->bone).basis.get_scale());

		p_task->skeleton->set_bone_global_pose_override(ci->bone, new_bone_pose, 1.0, true);

		if (!ci->children.is_empty()) {
			ci = &ci->children.write[0];
		} else {
			ci = nullptr;
		}
	}
}

void FabrikInverseKinematic::_update_chain(const Skeleton3D *p_sk, ChainItem *p_chain_item) {
	if (!p_chain_item) {
		return;
	}

	p_chain_item->initial_transform = p_sk->get_bone_global_pose_no_override(p_chain_item->bone);
	p_chain_item->current_pos = p_chain_item->initial_transform.origin;

	ChainItem *items = p_chain_item->children.ptrw();
	for (int i = 0; i < p_chain_item->children.size(); i += 1) {
		_update_chain(p_sk, items + i);
	}
}

void SkeletonIK3D::_validate_property(PropertyInfo &property) const {
	if (property.name == "root_bone" || property.name == "tip_bone") {
		if (skeleton) {
			String names("--,");
			for (int i = 0; i < skeleton->get_bone_count(); i++) {
				if (i > 0) {
					names += ",";
				}
				names += skeleton->get_bone_name(i);
			}

			property.hint = PROPERTY_HINT_ENUM;
			property.hint_string = names;
		} else {
			property.hint = PROPERTY_HINT_NONE;
			property.hint_string = "";
		}
	}

	Node::_validate_property(property);
}

void SkeletonIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_bone", "root_bone"), &SkeletonIK3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone"), &SkeletonIK3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone"), &SkeletonIK3D::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &SkeletonIK3D::get_tip_bone);

	ClassDB::bind_method(D_METHOD("set_interpolation", "interpolation"), &SkeletonIK3D::set_interpolation);
	ClassDB::bind_method(D_METHOD("get_interpolation"), &SkeletonIK3D::get_interpolation);

	ClassDB::bind_method(D_METHOD("set_target_transform", "target"), &SkeletonIK3D::set_target_transform);
	ClassDB::bind_method(D_METHOD("get_target_transform"), &SkeletonIK3D::get_target_transform);

	ClassDB::bind_method(D_METHOD("set_target_node", "node"), &SkeletonIK3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonIK3D::get_target_node);

	ClassDB::bind_method(D_METHOD("set_override_tip_basis", "override"), &SkeletonIK3D::set_override_tip_basis);
	ClassDB::bind_method(D_METHOD("is_override_tip_basis"), &SkeletonIK3D::is_override_tip_basis);

	ClassDB::bind_method(D_METHOD("set_use_magnet", "use"), &SkeletonIK3D::set_use_magnet);
	ClassDB::bind_method(D_METHOD("is_using_magnet"), &SkeletonIK3D::is_using_magnet);

	ClassDB::bind_method(D_METHOD("set_magnet_position", "local_position"), &SkeletonIK3D::set_magnet_position);
	ClassDB::bind_method(D_METHOD("get_magnet_position"), &SkeletonIK3D::get_magnet_position);

	ClassDB::bind_method(D_METHOD("get_parent_skeleton"), &SkeletonIK3D::get_parent_skeleton);
	ClassDB::bind_method(D_METHOD("is_running"), &SkeletonIK3D::is_running);

	ClassDB::bind_method(D_METHOD("set_min_distance", "min_distance"), &SkeletonIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &SkeletonIK3D::get_min_distance);

	ClassDB::bind_method(D_METHOD("set_max_iterations", "iterations"), &SkeletonIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &SkeletonIK3D::get_max_iterations);

	ClassDB::bind_method(D_METHOD("start", "one_time"), &SkeletonIK3D::start, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("stop"), &SkeletonIK3D::stop);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "root_bone"), "set_root_bone", "get_root_bone");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "tip_bone"), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "interpolation", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_interpolation", "get_interpolation");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "target"), "set_target_transform", "get_target_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_tip_basis"), "set_override_tip_basis", "is_override_tip_basis");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_magnet"), "set_use_magnet", "is_using_magnet");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "magnet"), "set_magnet_position", "get_magnet_position");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_distance"), "set_min_distance", "get_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations"), "set_max_iterations", "get_max_iterations");
}

void SkeletonIK3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			skeleton = Object::cast_to<Skeleton3D>(get_parent());
			set_process_priority(1);
			reload_chain();
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (target_node_override) {
				reload_goal();
			}

			_solve_chain();

		} break;
		case NOTIFICATION_EXIT_TREE: {
			reload_chain();
		} break;
	}
}

SkeletonIK3D::SkeletonIK3D() {
}

SkeletonIK3D::~SkeletonIK3D() {
	FabrikInverseKinematic::free_task(task);
	task = nullptr;
}

void SkeletonIK3D::set_root_bone(const StringName &p_root_bone) {
	root_bone = p_root_bone;
	reload_chain();
}

StringName SkeletonIK3D::get_root_bone() const {
	return root_bone;
}

void SkeletonIK3D::set_tip_bone(const StringName &p_tip_bone) {
	tip_bone = p_tip_bone;
	reload_chain();
}

StringName SkeletonIK3D::get_tip_bone() const {
	return tip_bone;
}

void SkeletonIK3D::set_interpolation(real_t p_interpolation) {
	interpolation = p_interpolation;
}

real_t SkeletonIK3D::get_interpolation() const {
	return interpolation;
}

void SkeletonIK3D::set_target_transform(const Transform3D &p_target) {
	target = p_target;
	reload_goal();
}

const Transform3D &SkeletonIK3D::get_target_transform() const {
	return target;
}

void SkeletonIK3D::set_target_node(const NodePath &p_node) {
	target_node_path_override = p_node;
	target_node_override = nullptr;
	reload_goal();
}

NodePath SkeletonIK3D::get_target_node() {
	return target_node_path_override;
}

void SkeletonIK3D::set_override_tip_basis(bool p_override) {
	override_tip_basis = p_override;
}

bool SkeletonIK3D::is_override_tip_basis() const {
	return override_tip_basis;
}

void SkeletonIK3D::set_use_magnet(bool p_use) {
	use_magnet = p_use;
}

bool SkeletonIK3D::is_using_magnet() const {
	return use_magnet;
}

void SkeletonIK3D::set_magnet_position(const Vector3 &p_local_position) {
	magnet_position = p_local_position;
}

const Vector3 &SkeletonIK3D::get_magnet_position() const {
	return magnet_position;
}

void SkeletonIK3D::set_min_distance(real_t p_min_distance) {
	min_distance = p_min_distance;
}

void SkeletonIK3D::set_max_iterations(int p_iterations) {
	max_iterations = p_iterations;
}

bool SkeletonIK3D::is_running() {
	return is_processing_internal();
}

void SkeletonIK3D::start(bool p_one_time) {
	if (p_one_time) {
		set_process_internal(false);

		if (target_node_override) {
			reload_goal();
		}

		_solve_chain();
	} else {
		set_process_internal(true);
	}
}

void SkeletonIK3D::stop() {
	set_process_internal(false);
	if (skeleton) {
		skeleton->clear_bones_global_pose_override();
	}
}

Transform3D SkeletonIK3D::_get_target_transform() {
	if (!target_node_override && !target_node_path_override.is_empty()) {
		target_node_override = Object::cast_to<Node3D>(get_node(target_node_path_override));
	}

	if (target_node_override && target_node_override->is_inside_tree()) {
		return target_node_override->get_global_transform();
	} else {
		return target;
	}
}

void SkeletonIK3D::reload_chain() {
	FabrikInverseKinematic::free_task(task);
	task = nullptr;

	if (!skeleton) {
		return;
	}

	task = FabrikInverseKinematic::create_simple_task(skeleton, skeleton->find_bone(root_bone), skeleton->find_bone(tip_bone), _get_target_transform());
	if (task) {
		task->max_iterations = max_iterations;
		task->min_distance = min_distance;
	}
}

void SkeletonIK3D::reload_goal() {
	if (!task) {
		return;
	}

	FabrikInverseKinematic::set_goal(task, _get_target_transform());
}

void SkeletonIK3D::_solve_chain() {
	if (!task) {
		return;
	}
	FabrikInverseKinematic::solve(task, interpolation, override_tip_basis, use_magnet, magnet_position);
}

#endif // _3D_DISABLED
