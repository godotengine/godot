/**************************************************************************/
/*  chain_ik_3d.h                                                         */
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

class ChainIK3D : public IKModifier3D {
	GDCLASS(ChainIK3D, IKModifier3D);

public:
	struct ChainIK3DSetting : public IKModifier3DSetting {
#ifdef TOOLS_ENABLED
		// Note:
		// To cache global rest on global pose in SkeletonModifier process.
		// Since gizmo drawing might be processed after SkeletonModifier process,
		// so the gizmo which depend on modified pose is not drawn correctly.
		// Especially, limitation sphere is needed this since it bound mutable bone axis which retrieve by bone pose to the parent bone rest.
		Transform3D root_global_rest;
#endif // TOOLS_ENABLED

		BoneJoint root_bone;
		BoneJoint end_bone;

		// To make virtual end joint.
		bool extend_end_bone = false;
		BoneDirection end_bone_direction = BONE_DIRECTION_FROM_PARENT;
		float end_bone_length = 0.0;

		LocalVector<BoneJoint> joints;
		LocalVector<IKModifier3DSolverInfo *> solver_info_list;
		LocalVector<Vector3> chain;

		// Only update chain coordinates to avoid to override previous result (bone poses).
		// Chain coordinates will be converted to bone pose by child class cache_current_joint_rotations() in the end of iterating.
		void update_chain_coordinate(Skeleton3D *p_skeleton, int p_index, const Vector3 &p_position) {
			// Don't update if the position is same as the current position ().
			// But distance_squared_to() is unsuitable because converting position to rotation requires a certain level of precision.
			if (Math::is_zero_approx(chain[p_index].distance_to(p_position))) {
				return;
			}

			// Allow flipping.
			chain[p_index] = p_position;
			cache_current_vector(p_skeleton, p_index);
		}

		void update_chain_coordinate_bw(Skeleton3D *p_skeleton, int p_index, const Vector3 &p_position) {
			// Don't update if the position is same as the current position.
			// But distance_squared_to() is unsuitable because converting position to rotation requires a certain level of precision.
			if (Math::is_zero_approx(chain[p_index].distance_to(p_position))) {
				return;
			}

			// Prevent flipping from backwards.
			Vector3 result = p_position;
			int HEAD = p_index - 1;
			int TAIL = p_index;
			if (HEAD >= 0 && HEAD < (int)solver_info_list.size()) {
				IKModifier3DSolverInfo *solver_info = solver_info_list[HEAD];
				if (solver_info) {
					Vector3 old_head_to_tail = solver_info->current_vector;
					Vector3 new_head_to_tail = (result - chain[HEAD]).normalized();
					if (Math::is_equal_approx((double)old_head_to_tail.dot(new_head_to_tail), -1.0)) {
						chain[TAIL] = chain[HEAD] + old_head_to_tail * solver_info->length; // Revert.
						return; // No change, cache is not updated.
					}
				}
			}

			chain[p_index] = result;
			cache_current_vector(p_skeleton, p_index);
		}

		void update_chain_coordinate_fw(Skeleton3D *p_skeleton, int p_index, const Vector3 &p_position) {
			// Don't update if the position is same as the current position.
			// But distance_squared_to() is unsuitable because converting position to rotation requires a certain level of precision.
			if (Math::is_zero_approx(chain[p_index].distance_to(p_position))) {
				return;
			}

			// Prevent flipping from forwards.
			Vector3 result = p_position;
			int HEAD = p_index;
			int TAIL = p_index + 1;
			if (TAIL >= 0 && TAIL < (int)solver_info_list.size()) {
				IKModifier3DSolverInfo *solver_info = solver_info_list[HEAD];
				if (solver_info) {
					Vector3 old_head_to_tail = solver_info->current_vector;
					Vector3 new_head_to_tail = (chain[TAIL] - result).normalized();
					if (Math::is_equal_approx((double)old_head_to_tail.dot(new_head_to_tail), -1.0)) {
						chain[HEAD] = chain[TAIL] - old_head_to_tail * solver_info->length; // Revert.
						return; // No change, cache is not updated.
					}
				}
			}

			chain[p_index] = result;
			cache_current_vector(p_skeleton, p_index);
		}

		void cache_current_vector(Skeleton3D *p_skeleton, int p_index) {
			int cur_head = p_index - 1;
			int cur_tail = p_index;
			if (cur_head >= 0) {
				IKModifier3DSolverInfo *solver_info = solver_info_list[cur_head];
				if (solver_info) {
					solver_info->current_vector = (chain[cur_tail] - chain[cur_head]).normalized();
				}
			}
			cur_head = p_index;
			cur_tail = p_index + 1;
			if (cur_tail < (int)chain.size()) {
				IKModifier3DSolverInfo *solver_info = solver_info_list[cur_head];
				if (solver_info) {
					solver_info->current_vector = (chain[cur_tail] - chain[cur_head]).normalized();
				}
			}
		}

		void cache_current_vectors(Skeleton3D *p_skeleton) {
			for (uint32_t i = 0; i < joints.size(); i++) {
				int HEAD = i;
				int TAIL = i + 1;
				IKModifier3DSolverInfo *solver_info = solver_info_list[HEAD];
				if (!solver_info) {
					continue;
				}
				solver_info->current_vector = (chain[TAIL] - chain[HEAD]).normalized();
			}
		}

		void init_current_joint_rotations(Skeleton3D *p_skeleton) {
			if (root_bone.bone < 0) {
				return;
			}

			Quaternion parent_gpose;
			int parent = p_skeleton->get_bone_parent(root_bone.bone);
			if (parent >= 0) {
				parent_gpose = p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion();
			}

			for (uint32_t i = 0; i < joints.size(); i++) {
				IKModifier3DSolverInfo *solver_info = solver_info_list[i];
				if (!solver_info) {
					continue;
				}
				solver_info->current_lrest = p_skeleton->get_bone_pose(joints[i].bone).basis.get_rotation_quaternion();
				solver_info->current_grest = parent_gpose * solver_info->current_lrest;
				solver_info->current_grest.normalize();
				solver_info->current_lpose = p_skeleton->get_bone_pose(joints[i].bone).basis.get_rotation_quaternion();
				solver_info->current_gpose = parent_gpose * solver_info->current_lpose;
				solver_info->current_gpose.normalize();
				parent_gpose = solver_info->current_gpose;
			}

			cache_current_vectors(p_skeleton);
		}

		~ChainIK3DSetting() {
			for (uint32_t i = 0; i < solver_info_list.size(); i++) {
				if (solver_info_list[i]) {
					memdelete(solver_info_list[i]);
					solver_info_list[i] = nullptr;
				}
			}
			solver_info_list.clear();
		}
	};

protected:
#ifdef TOOLS_ENABLED
	virtual void _update_mutable_info() override;
#endif // TOOLS_ENABLED

	LocalVector<ChainIK3DSetting *> chain_settings; // For caching.

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	static void _bind_methods();

	virtual void _validate_bone_names() override;
	void _validate_axes(Skeleton3D *p_skeleton) const;
	virtual void _validate_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const;

	virtual void _make_all_joints_dirty() override;
	virtual void _update_joints(int p_index) override;
	void _set_joint_bone(int p_index, int p_joint, int p_bone);

	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;

	virtual void _set_joint_count(int p_index, int p_count);

public:
	virtual void set_setting_count(int p_count) override {
		_set_setting_count<ChainIK3DSetting>(p_count);
		chain_settings = _cast_settings<ChainIK3DSetting>();
	}
	virtual void clear_settings() override {
		_set_setting_count<ChainIK3DSetting>(0);
		chain_settings.clear();
	}

	// Setting.
	void set_root_bone_name(int p_index, const String &p_bone_name);
	String get_root_bone_name(int p_index) const;
	void set_root_bone(int p_index, int p_bone);
	int get_root_bone(int p_index) const;

	void set_end_bone_name(int p_index, const String &p_bone_name);
	String get_end_bone_name(int p_index) const;
	void set_end_bone(int p_index, int p_bone);
	int get_end_bone(int p_index) const;

	void set_extend_end_bone(int p_index, bool p_enabled);
	bool is_end_bone_extended(int p_index) const;
	void set_end_bone_direction(int p_index, BoneDirection p_bone_direction);
	BoneDirection get_end_bone_direction(int p_index) const;
	void set_end_bone_length(int p_index, float p_length);
	float get_end_bone_length(int p_index) const;

	// Individual joints.
	String get_joint_bone_name(int p_index, int p_joint) const;
	int get_joint_bone(int p_index, int p_joint) const;

	void set_joint_count(int p_index, int p_count);
	int get_joint_count(int p_index) const;

#ifdef TOOLS_ENABLED
	// Helper.
	static Transform3D get_bone_global_rest_mutable(Skeleton3D *p_skeleton, int p_bone);
	Transform3D get_chain_root_global_rest(int p_index);
	virtual Vector3 get_bone_vector(int p_index, int p_joint) const;
#endif // TOOLS_ENABLED

	~ChainIK3D();
};
