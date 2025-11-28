/**************************************************************************/
/*  iterate_ik_3d.h                                                       */
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

#pragma once

#include "scene/3d/chain_ik_3d.h"

#include "scene/resources/3d/joint_limitation_3d.h"

class IterateIK3D : public ChainIK3D {
	GDCLASS(IterateIK3D, ChainIK3D);

public:
	struct IterateIK3DJointSetting {
		// To limit rotation.
		RotationAxis rotation_axis = ROTATION_AXIS_ALL;
		Vector3 rotation_axis_vector = Vector3(1, 0, 0);
		Ref<JointLimitation3D> limitation;
		SecondaryDirection limitation_right_axis = SECONDARY_DIRECTION_NONE;
		Vector3 limitation_right_axis_vector = Vector3(1, 0, 0);
		Quaternion limitation_rotation_offset;

		// Rotation axis.
		Vector3 get_rotation_axis_vector() const {
			Vector3 ret;
			switch (rotation_axis) {
				case ROTATION_AXIS_X:
					ret = Vector3(1, 0, 0);
					break;
				case ROTATION_AXIS_Y:
					ret = Vector3(0, 1, 0);
					break;
				case ROTATION_AXIS_Z:
					ret = Vector3(0, 0, 1);
					break;
				case ROTATION_AXIS_ALL:
					ret = Vector3(0, 0, 0);
					break;
				case ROTATION_AXIS_CUSTOM:
					ret = rotation_axis_vector;
					break;
			}
			return ret;
		}

		Vector3 get_limitation_right_axis_vector() const {
			Vector3 ret;
			switch (limitation_right_axis) {
				case SECONDARY_DIRECTION_NONE:
					ret = Vector3(0, 0, 0);
					break;
				case SECONDARY_DIRECTION_PLUS_X:
					ret = Vector3(1, 0, 0);
					break;
				case SECONDARY_DIRECTION_MINUS_X:
					ret = Vector3(-1, 0, 0);
					break;
				case SECONDARY_DIRECTION_PLUS_Y:
					ret = Vector3(0, 1, 0);
					break;
				case SECONDARY_DIRECTION_MINUS_Y:
					ret = Vector3(0, -1, 0);
					break;
				case SECONDARY_DIRECTION_PLUS_Z:
					ret = Vector3(0, 0, 1);
					break;
				case SECONDARY_DIRECTION_MINUS_Z:
					ret = Vector3(0, 0, -1);
					break;
				case SECONDARY_DIRECTION_CUSTOM:
					ret = limitation_right_axis_vector;
					break;
			}
			return ret;
		}

		Quaternion get_limitation_space(const Vector3 &p_local_forward) const {
			if (limitation.is_null()) {
				return Quaternion();
			}
			return limitation->make_space(p_local_forward, get_limitation_right_axis_vector(), limitation_rotation_offset);
		}

		// Get rotation around normal vector (normal vector is rotation axis).
		Vector3 get_projected_rotation(const Quaternion &p_offset, const Vector3 &p_vector) const {
			ERR_FAIL_COND_V(rotation_axis == ROTATION_AXIS_ALL, p_vector);
			const double ALMOST_ONE = 1.0 - CMP_EPSILON;
			Vector3 axis = get_rotation_axis_vector().normalized();
			Vector3 local_vector = p_offset.xform_inv(p_vector);
			double length = local_vector.length();
			Vector3 projected = snap_vector_to_plane(axis, local_vector.normalized());
			if (!Math::is_zero_approx(length)) {
				projected = projected.normalized() * length;
			}
			if (Math::abs(local_vector.normalized().dot(axis)) > ALMOST_ONE) {
				return p_vector;
			}
			return p_offset.xform(projected);
		}

		// Get limited rotation from forward axis in local rest space.
		Vector3 get_limited_rotation(const Quaternion &p_offset, const Vector3 &p_vector, const Vector3 &p_forward) const {
			ERR_FAIL_COND_V(limitation.is_null(), p_vector);
			Vector3 local_vector = p_offset.xform_inv(p_vector);
			float length = local_vector.length();
			if (Math::is_zero_approx(length)) {
				return p_vector;
			}
			Vector3 limited = limitation->solve(p_forward, get_limitation_right_axis_vector(), limitation_rotation_offset, local_vector.normalized()) * length;
			return p_offset.xform(limited);
		}

		~IterateIK3DJointSetting() {
			limitation.unref();
		}
	};

	struct IterateIK3DSetting : public ChainIK3DSetting {
		NodePath target_node;

		LocalVector<IterateIK3DJointSetting *> joint_settings;

		bool simulated = false;

		// Make rotation as bone pose from chain coordinates.
		// p_extra is delta angle limitation.
		void cache_current_joint_rotations(Skeleton3D *p_skeleton, double p_angular_delta_limit = Math::PI) {
			Transform3D parent_gpose_tr;
			int parent = p_skeleton->get_bone_parent(root_bone.bone);
			if (parent >= 0) {
				parent_gpose_tr = p_skeleton->get_bone_global_pose(parent);
			}
			Quaternion parent_gpose = parent_gpose_tr.basis.get_rotation_quaternion();

			for (uint32_t i = 0; i < joints.size(); i++) {
				int HEAD = i;
				IKModifier3DSolverInfo *solver_info = solver_info_list[HEAD];
				if (!solver_info) {
					continue;
				}
				solver_info->current_lrest = p_skeleton->get_bone_pose(joints[HEAD].bone).basis.get_rotation_quaternion();
				solver_info->current_grest = parent_gpose * solver_info->current_lrest;
				solver_info->current_grest.normalize();
				Vector3 from = solver_info->forward_vector;
				Vector3 to = solver_info->current_grest.xform_inv(solver_info->current_vector).normalized();
				Quaternion prev = solver_info->current_lpose;
				if (joint_settings[HEAD]->rotation_axis == ROTATION_AXIS_ALL) {
					solver_info->current_lpose = solver_info->current_lrest * get_swing(Quaternion(from, to), from);
				} else {
					// To stabilize rotation path especially nearely 180deg.
					solver_info->current_lpose = solver_info->current_lrest * get_from_to_rotation_by_axis(from, to, joint_settings[HEAD]->get_rotation_axis_vector().normalized());
				}
				double diff = prev.angle_to(solver_info->current_lpose);
				if (!Math::is_zero_approx(diff)) {
					solver_info->current_lpose = prev.slerp(solver_info->current_lpose, MIN(1.0, p_angular_delta_limit / diff));
				}
				solver_info->current_gpose = parent_gpose * solver_info->current_lpose;
				solver_info->current_gpose.normalize();
				parent_gpose = solver_info->current_gpose;
			}

			// Apply back angular_delta_limit to chain coordinates.
			if (chain.is_empty()) {
				return;
			}
			chain[0] = p_skeleton->get_bone_global_pose(root_bone.bone).origin;
			for (uint32_t i = 0; i < solver_info_list.size(); i++) {
				int HEAD = i;
				int TAIL = i + 1;
				IKModifier3DSolverInfo *solver_info = solver_info_list[HEAD];
				if (!solver_info) {
					continue;
				}
				chain[TAIL] = chain[HEAD] + solver_info->current_gpose.xform(solver_info->forward_vector) * solver_info->length;
			}
			cache_current_vectors(p_skeleton);
		}

		void init_joints(Skeleton3D *p_skeleton, bool p_mutable_bone_axes) {
			chain.clear();
			bool extends_end = extend_end_bone && end_bone_length > 0;
			for (uint32_t i = 0; i < joints.size(); i++) {
				chain.push_back(p_skeleton->get_bone_global_pose(joints[i].bone).origin);
				bool last = i == joints.size() - 1;
				if (last && extends_end) {
					Vector3 axis = IKModifier3D::get_bone_axis(p_skeleton, end_bone.bone, end_bone_direction, p_mutable_bone_axes);
					if (axis.is_zero_approx()) {
						continue;
					}
					if (!solver_info_list[i]) {
						solver_info_list[i] = memnew(IKModifier3DSolverInfo);
					}
					solver_info_list[i]->forward_vector = snap_vector_to_plane(joint_settings[i]->get_rotation_axis_vector(), axis.normalized());
					solver_info_list[i]->length = end_bone_length;
					chain.push_back(p_skeleton->get_bone_global_pose(joints[i].bone).xform(axis * end_bone_length));
				} else if (!last) {
					Vector3 axis = p_skeleton->get_bone_rest(joints[i + 1].bone).origin;
					if (axis.is_zero_approx()) {
						continue;
					}
					if (!solver_info_list[i]) {
						solver_info_list[i] = memnew(IKModifier3DSolverInfo);
					}
					solver_info_list[i]->forward_vector = snap_vector_to_plane(joint_settings[i]->get_rotation_axis_vector(), axis.normalized());
					solver_info_list[i]->length = axis.length();
				}
			}
			init_current_joint_rotations(p_skeleton);
		}

		~IterateIK3DSetting() {
			for (uint32_t i = 0; i < joint_settings.size(); i++) {
				if (joint_settings[i]) {
					memdelete(joint_settings[i]);
					joint_settings[i] = nullptr;
				}
			}
			joint_settings.clear();
		}
	};

protected:
	LocalVector<IterateIK3DSetting *> iterate_settings; // For caching.

	int max_iterations = 4;
	double min_distance = 0.001; // If distance between end joint and target is less than min_distance, finish iteration.
	double min_distance_squared = min_distance * min_distance; // For cache.
	double angular_delta_limit = Math::deg_to_rad(2.0); // If the delta is too large, the results before and after iterating can change significantly, and divergence of calculations can easily occur.

	bool deterministic = false;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	static void _bind_methods();

	virtual void _validate_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const override;
	virtual void _init_joints(Skeleton3D *p_skeleton, int p_index) override;
	void _clear_joints(int p_index); // Connect signal with the IterateIK3D node so it shouldn't be included by struct IterateIK3DSetting.

	virtual void _make_simulation_dirty(int p_index) override;
	virtual void _update_bone_axis(Skeleton3D *p_skeleton, int p_index) override;

	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;
	void _process_joints(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_target_destination);
	virtual void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination);

	virtual void _set_joint_count(int p_index, int p_count) override;

	void _update_joint_limitation(int p_index, int p_joint);
	void _bind_joint_limitation(int p_index, int p_joint);
	void _unbind_joint_limitation(int p_index, int p_joint);
	void _bind_joint_limitations(int p_index);
	void _unbind_joint_limitations(int p_index);

public:
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual void set_setting_count(int p_count) override {
		_set_setting_count<IterateIK3DSetting>(p_count);
		iterate_settings = _cast_settings<IterateIK3DSetting>();
		chain_settings = _cast_settings<ChainIK3DSetting>(); // Don't forget to sync super class settings.
	}
	virtual void clear_settings() override {
		_set_setting_count<IterateIK3DSetting>(0);
		iterate_settings.clear();
		chain_settings.clear(); // Don't forget to sync super class settings.
	}

	void set_max_iterations(int p_max_iterations);
	int get_max_iterations() const;
	void set_min_distance(double p_min_distance);
	double get_min_distance() const;
	void set_angular_delta_limit(double p_angular_delta_limit);
	double get_angular_delta_limit() const;

	void set_deterministic(bool p_deterministic);
	bool is_deterministic() const;

	// Setting.
	void set_target_node(int p_index, const NodePath &p_target_node);
	NodePath get_target_node(int p_index) const;

	// Individual joints.
	void set_joint_rotation_axis(int p_index, int p_joint, RotationAxis p_axis);
	RotationAxis get_joint_rotation_axis(int p_index, int p_joint) const;
	void set_joint_rotation_axis_vector(int p_index, int p_joint, const Vector3 &p_vector);
	Vector3 get_joint_rotation_axis_vector(int p_index, int p_joint) const;
	void set_joint_limitation(int p_index, int p_joint, const Ref<JointLimitation3D> &p_limitation);
	Ref<JointLimitation3D> get_joint_limitation(int p_index, int p_joint) const;
	void set_joint_limitation_right_axis(int p_index, int p_joint, SecondaryDirection p_direction);
	SecondaryDirection get_joint_limitation_right_axis(int p_index, int p_joint) const;
	void set_joint_limitation_right_axis_vector(int p_index, int p_joint, const Vector3 &p_vector);
	Vector3 get_joint_limitation_right_axis_vector(int p_index, int p_joint) const;
	void set_joint_limitation_rotation_offset(int p_index, int p_joint, const Quaternion &p_offset);
	Quaternion get_joint_limitation_rotation_offset(int p_index, int p_joint) const;

	// Helper.
	Quaternion get_joint_limitation_space(int p_index, int p_joint, const Vector3 &p_forward) const;

#ifdef TOOLS_ENABLED
	virtual Vector3 get_bone_vector(int p_index, int p_joint) const override;
#endif // TOOLS_ENABLED

	~IterateIK3D();
};
