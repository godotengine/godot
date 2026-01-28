/**************************************************************************/
/*  two_bone_ik_3d.h                                                      */
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

#include "scene/3d/ik_modifier_3d.h"

class TwoBoneIK3D : public IKModifier3D {
	GDCLASS(TwoBoneIK3D, IKModifier3D);

public:
	struct TwoBoneIK3DSetting : public IKModifier3DSetting {
		bool joints_dirty = false;

		BoneJoint root_bone;
		BoneJoint middle_bone;
		BoneJoint end_bone;

		// To make virtual end joint.
		bool use_virtual_end = false;
		bool extend_end_bone = false;
		BoneDirection end_bone_direction = BONE_DIRECTION_FROM_PARENT;
		float end_bone_length = 0.0;

		NodePath pole_node;
		SecondaryDirection pole_direction = SECONDARY_DIRECTION_NONE; // Sort to pole target plane.
		Vector3 pole_direction_vector; // Custom vector.
		NodePath target_node;

		IKModifier3DSolverInfo *root_joint_solver_info = nullptr;
		IKModifier3DSolverInfo *mid_joint_solver_info = nullptr;
		Vector3 root_pos;
		Vector3 mid_pos;
		Vector3 end_pos;

		// To process.
		bool simulation_dirty = true;
		double cached_length_sq = 0.0;

		bool is_valid() const {
			return root_joint_solver_info && mid_joint_solver_info;
		}
		bool is_end_valid() const {
			return (!use_virtual_end && end_bone.bone != -1) || (use_virtual_end && !Math::is_zero_approx(end_bone_length));
		}
		int get_end_bone() const {
			return use_virtual_end ? middle_bone.bone : end_bone.bone; // Hack, but useful for external class such as TwoBoneIK3DGizmoPlugin.
		}

		Vector3 get_pole_direction_vector() const {
			Vector3 ret;
			switch (pole_direction) {
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
					ret = pole_direction_vector;
					break;
			}
			return ret;
		}

		void cache_current_vectors(Skeleton3D *p_skeleton) {
			if (!is_valid()) {
				return;
			}
			root_joint_solver_info->current_vector = (mid_pos - root_pos).normalized();
			mid_joint_solver_info->current_vector = (end_pos - mid_pos).normalized();
		}

		void init_current_joint_rotations(Skeleton3D *p_skeleton) {
			if (!is_valid()) {
				return;
			}

			Quaternion parent_gpose;
			int parent = p_skeleton->get_bone_parent(root_bone.bone);
			if (parent >= 0) {
				parent_gpose = p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion();
			}
			root_joint_solver_info->current_lrest = p_skeleton->get_bone_pose(root_bone.bone).basis.get_rotation_quaternion();
			root_joint_solver_info->current_grest = parent_gpose * root_joint_solver_info->current_lrest;
			root_joint_solver_info->current_grest.normalize();
			root_joint_solver_info->current_lpose = p_skeleton->get_bone_pose(root_bone.bone).basis.get_rotation_quaternion();
			root_joint_solver_info->current_gpose = parent_gpose * root_joint_solver_info->current_lpose;
			root_joint_solver_info->current_gpose.normalize();
			parent_gpose = root_joint_solver_info->current_gpose;

			// Mid joint pose is relative to the root joint pose.
			mid_joint_solver_info->current_lrest = p_skeleton->get_bone_global_pose(root_bone.bone).basis.get_rotation_quaternion().inverse() * p_skeleton->get_bone_global_pose(middle_bone.bone).basis.get_rotation_quaternion();
			mid_joint_solver_info->current_grest = parent_gpose * mid_joint_solver_info->current_lrest;
			mid_joint_solver_info->current_grest.normalize();
			mid_joint_solver_info->current_lpose = p_skeleton->get_bone_global_pose(root_bone.bone).basis.get_rotation_quaternion().inverse() * p_skeleton->get_bone_global_pose(middle_bone.bone).basis.get_rotation_quaternion();
			mid_joint_solver_info->current_gpose = parent_gpose * mid_joint_solver_info->current_lpose;
			mid_joint_solver_info->current_gpose.normalize();

			cache_current_vectors(p_skeleton);
		}

		// Make rotation as bone pose from chain coordinates.
		void cache_current_joint_rotations(Skeleton3D *p_skeleton, const Vector3 &p_pole_destination) {
			if (!is_valid()) {
				return;
			}

			Quaternion parent_gpose;
			int parent = p_skeleton->get_bone_parent(root_bone.bone);
			if (parent >= 0) {
				parent_gpose = p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion();
			}

			root_joint_solver_info->current_lrest = p_skeleton->get_bone_pose(root_bone.bone).basis.get_rotation_quaternion();
			root_joint_solver_info->current_grest = parent_gpose * root_joint_solver_info->current_lrest;
			root_joint_solver_info->current_grest.normalize();

			Vector3 from = root_joint_solver_info->forward_vector;
			Vector3 to = root_joint_solver_info->current_grest.xform_inv(root_joint_solver_info->current_vector).normalized();
			root_joint_solver_info->current_lpose = root_joint_solver_info->current_lrest * get_swing(Quaternion(from, to), from);

			root_joint_solver_info->current_gpose = parent_gpose * root_joint_solver_info->current_lpose;
			root_joint_solver_info->current_gpose.normalize();
			Quaternion root_gpose = root_joint_solver_info->current_gpose;

			// Mid joint pose is relative to the root joint pose for the case root-mid or mid-end have more than 1 joints.
			mid_joint_solver_info->current_lrest = p_skeleton->get_bone_global_pose(root_bone.bone).basis.get_rotation_quaternion().inverse() * p_skeleton->get_bone_global_pose(middle_bone.bone).basis.get_rotation_quaternion();
			mid_joint_solver_info->current_grest = root_gpose * mid_joint_solver_info->current_lrest;
			mid_joint_solver_info->current_grest.normalize();

			from = mid_joint_solver_info->forward_vector;
			to = mid_joint_solver_info->current_grest.xform_inv(mid_joint_solver_info->current_vector).normalized();
			mid_joint_solver_info->current_lpose = mid_joint_solver_info->current_lrest * get_swing(Quaternion(from, to), from);

			mid_joint_solver_info->current_gpose = root_gpose * mid_joint_solver_info->current_lpose;
			mid_joint_solver_info->current_gpose.normalize();

			bool is_pole_defined = pole_direction != SECONDARY_DIRECTION_NONE && (pole_direction != SECONDARY_DIRECTION_CUSTOM || !pole_direction_vector.is_zero_approx());
			// Fix roll to align pole vector to plane.
			if (is_pole_defined) {
				// Calc roll angles.
				Quaternion root_roll_rot = Quaternion();
				Quaternion mid_roll_rot = Quaternion();

				// Make roll to align pole_vector onto plane with selecting the point nearer pole_destination.
				Vector3 pole_dir = get_projected_normal(root_pos, end_pos, p_pole_destination);
				if (pole_dir.is_zero_approx()) {
					return;
				}
				Vector3 a = mid_joint_solver_info->current_vector.normalized(); // Global roll axis (mid forward in current pose).
				Vector3 k = mid_joint_solver_info->current_gpose.xform(get_pole_direction_vector()).normalized(); // Global pole vector.
				Vector3 n = pole_dir.cross((mid_pos - root_pos).normalized()).normalized(); // Global plane normal.

				// Guard: degenerate cases (zero or already parallel)
				if (a.is_zero_approx() || k.is_zero_approx() || n.is_zero_approx() || Math::is_zero_approx(n.dot(k))) {
					return;
				}
				// c0 cosθ + c1 sinθ + c2 = 0
				double c0 = n.dot(k - a * k.dot(a)); // n·(k⊥a)
				double c1 = n.dot(a.cross(k)); // n·(a×k)
				double c2 = n.dot(a) * k.dot(a); // (n·a)(k·a)
				double r = Math::sqrt(c0 * c0 + c1 * c1);
				double cos_arg = CLAMP(-c2 / r, -1.0, 1.0);
				double phi = Math::atan2(c1, c0);
				double acosv = Math::acos(cos_arg);

				// Two candidate angles.
				double t1 = phi + acosv;
				double t2 = phi - acosv;
				Quaternion q1(a, t1);
				Quaternion q2(a, t2);

				// Choose the one whose projected pole points closer to pole side.
				Vector3 pole_proj = snap_vector_to_plane(n, pole_dir).normalized();
				Vector3 k1p = snap_vector_to_plane(n, q1.xform(k)).normalized();
				Vector3 k2p = snap_vector_to_plane(n, q2.xform(k)).normalized();
				double s1 = pole_proj.is_zero_approx() ? Math::abs(t1) : k1p.dot(pole_proj);
				double s2 = pole_proj.is_zero_approx() ? Math::abs(t2) : k2p.dot(pole_proj);

				double t = s1 >= s2 ? t1 : t2;
				root_roll_rot = Quaternion(root_joint_solver_info->forward_vector, t);
				mid_roll_rot = Quaternion(mid_joint_solver_info->forward_vector, t);

				root_joint_solver_info->current_lpose = root_joint_solver_info->current_lpose * root_roll_rot;
				root_joint_solver_info->current_gpose = parent_gpose * root_joint_solver_info->current_lpose;
				root_joint_solver_info->current_gpose.normalize();
				root_gpose = root_joint_solver_info->current_gpose;

				mid_joint_solver_info->current_lpose = root_roll_rot.inverse() * mid_joint_solver_info->current_lpose * mid_roll_rot;
				mid_joint_solver_info->current_gpose = root_gpose * mid_joint_solver_info->current_lpose;
				mid_joint_solver_info->current_gpose.normalize();
			}
		}
	};

protected:
	LocalVector<TwoBoneIK3DSetting *> tb_settings;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	static void _bind_methods();

	virtual void _validate_bone_names() override;
	void _validate_pole_directions(Skeleton3D *p_skeleton) const;
	void _validate_pole_direction(Skeleton3D *p_skeleton, int p_index) const;

	virtual void _make_all_joints_dirty() override;
	virtual void _init_joints(Skeleton3D *p_skeleton, int p_index) override;
	void _clear_joints(int p_index);
	virtual void _update_joints(int p_index) override;
	virtual void _make_simulation_dirty(int p_index) override;
	virtual void _update_bone_axis(Skeleton3D *p_skeleton, int p_index) override;

	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;
	void _process_joints(double p_delta, Skeleton3D *p_skeleton, TwoBoneIK3DSetting *p_setting, const Vector3 &p_destination, const Vector3 &p_pole_destination);

	Transform3D _get_bone_global_rest(Skeleton3D *p_skeleton, int p_bone, int p_root) const;

public:
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual void set_setting_count(int p_count) override {
		_set_setting_count<TwoBoneIK3DSetting>(p_count);
		tb_settings = _cast_settings<TwoBoneIK3DSetting>();
	}
	virtual void clear_settings() override {
		_set_setting_count<TwoBoneIK3DSetting>(0);
		tb_settings.clear();
	}

	// Setting.
	void set_root_bone_name(int p_index, const String &p_bone_name);
	String get_root_bone_name(int p_index) const;
	void set_root_bone(int p_index, int p_bone);
	int get_root_bone(int p_index) const;

	void set_middle_bone_name(int p_index, const String &p_bone_name);
	String get_middle_bone_name(int p_index) const;
	void set_middle_bone(int p_index, int p_bone);
	int get_middle_bone(int p_index) const;

	void set_end_bone_name(int p_index, const String &p_bone_name);
	String get_end_bone_name(int p_index) const;
	void set_end_bone(int p_index, int p_bone);
	int get_end_bone(int p_index) const;

	void set_use_virtual_end(int p_index, bool p_enabled);
	bool is_using_virtual_end(int p_index) const;
	void set_extend_end_bone(int p_index, bool p_enabled);
	bool is_end_bone_extended(int p_index) const;
	void set_end_bone_direction(int p_index, BoneDirection p_bone_direction);
	BoneDirection get_end_bone_direction(int p_index) const;
	void set_end_bone_length(int p_index, float p_length);
	float get_end_bone_length(int p_index) const;

	void set_target_node(int p_index, const NodePath &p_target_node);
	NodePath get_target_node(int p_index) const;

	void set_pole_node(int p_index, const NodePath &p_pole_node);
	NodePath get_pole_node(int p_index) const;

	void set_pole_direction(int p_index, SecondaryDirection p_axis);
	SecondaryDirection get_pole_direction(int p_index) const;
	void set_pole_direction_vector(int p_index, const Vector3 &p_vector);
	Vector3 get_pole_direction_vector(int p_index) const;

	bool is_valid(int p_index) const; // Helper for editor and validation.

#ifdef TOOLS_ENABLED
	Vector3 get_root_bone_vector(int p_index) const;
	Vector3 get_middle_bone_vector(int p_index) const;
#endif // TOOLS_ENABLED

	~TwoBoneIK3D();
};
