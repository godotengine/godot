/**************************************************************************/
/*  spline_ik_3d.h                                                        */
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
#include "scene/3d/path_3d.h"

class SplineIK3D : public ChainIK3D {
	GDCLASS(SplineIK3D, ChainIK3D);

public:
	struct SplineIK3DSetting : public ChainIK3DSetting {
		NodePath path_3d;
		bool tilt_enabled = true;
		int tilt_fade_in = 1;
		int tilt_fade_out = 1;

		LocalVector<double> chain_length_accum;
		LocalVector<double> twists;

		// Find the nearest point.
		int32_t find_nearest_point(const Vector3 &p_origin, double p_length_sq, const PackedVector3Array &p_points, bool p_is_path_closed, uint32_t p_first, double *r_ret) {
			ERR_FAIL_COND_V(p_first >= p_points.size(), p_points.size() - 1);
			uint32_t i = p_first;
			if (p_is_path_closed) {
				while (true) {
					if (p_origin.distance_squared_to(p_points[i]) >= p_length_sq) {
						break;
					}
					i++;
					i = Math::posmod(i, p_points.size());
					if (i == p_first) {
						i = p_points.size();
						break; // Can't found, use last.
					}
				}
			} else {
				while (i < p_points.size()) {
					if (p_origin.distance_squared_to(p_points[i]) >= p_length_sq) {
						break;
					}
					i++;
				}
			}
			if (i == 0) {
				*r_ret = 0.0;
				return p_first;
			} else if (i >= p_points.size()) {
				*r_ret = 1.0;
				return i;
			}
			*r_ret = Math::inverse_lerp((double)p_origin.distance_squared_to(p_points[i - 1]), (double)p_origin.distance_squared_to(p_points[i]), p_length_sq);
			return i - 1;
		}

		// Make rotation as bone pose from chain coordinates.
		void cache_current_joint_rotations(Skeleton3D *p_skeleton, bool p_use_tilt = false) {
			Transform3D parent_gpose_tr;
			int parent = p_skeleton->get_bone_parent(root_bone.bone);
			if (parent >= 0) {
				parent_gpose_tr = p_skeleton->get_bone_global_pose(parent);
			}
			Quaternion parent_gpose = parent_gpose_tr.basis.get_rotation_quaternion();

			if (p_use_tilt) {
				double parent_twist = 0.0;
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

					Basis b = get_swing(Quaternion(from, to), from);
					b.rotate_local(from, twists[HEAD] - parent_twist);
					parent_twist = twists[HEAD];
					solver_info->current_lpose = solver_info->current_lrest * b.get_rotation_quaternion();

					solver_info->current_gpose = parent_gpose * solver_info->current_lpose;
					solver_info->current_gpose.normalize();
					parent_gpose = solver_info->current_gpose;
				}
			} else {
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

					solver_info->current_lpose = solver_info->current_lrest * get_swing(Quaternion(from, to), from);

					solver_info->current_gpose = parent_gpose * solver_info->current_lpose;
					solver_info->current_gpose.normalize();
					parent_gpose = solver_info->current_gpose;
				}
			}

			// To update positions in preprocess of _process_joints().
			cache_current_vectors(p_skeleton);
		}
	};

protected:
	LocalVector<SplineIK3DSetting *> sp_settings; // For caching.

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	static void _bind_methods();

	virtual void _init_joints(Skeleton3D *p_skeleton, int p_index) override;
	virtual void _make_simulation_dirty(int p_index) override;
	virtual void _update_bone_axis(Skeleton3D *p_skeleton, int p_index) override;

	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;
	void _process_joints(double p_delta, Skeleton3D *p_skeleton, SplineIK3DSetting *p_setting, Ref<Curve3D> p_curve, const Transform3D &p_curve_space);

	virtual void _set_joint_count(int p_index, int p_count) override;

public:
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual void set_setting_count(int p_count) override {
		_set_setting_count<SplineIK3DSetting>(p_count);
		sp_settings = _cast_settings<SplineIK3DSetting>();
		chain_settings = _cast_settings<ChainIK3DSetting>(); // Don't forget to sync super class settings.
	}
	virtual void clear_settings() override {
		_set_setting_count<SplineIK3DSetting>(0);
		sp_settings.clear();
		chain_settings.clear(); // Don't forget to sync super class settings.
	}

	// Setting.
	void set_path_3d(int p_index, const NodePath &p_path_3d);
	NodePath get_path_3d(int p_index) const;
	void set_tilt_enabled(int p_index, bool p_enabled);
	bool is_tilt_enabled(int p_index) const;
	void set_tilt_fade_in(int p_index, int p_size);
	int get_tilt_fade_in(int p_index) const;
	void set_tilt_fade_out(int p_index, int p_size);
	int get_tilt_fade_out(int p_index) const;

	// Helper.
	double get_bezier_arc_length();

#ifdef TOOLS_ENABLED
	virtual Vector3 get_bone_vector(int p_index, int p_joint) const override;
#endif // TOOLS_ENABLED

	~SplineIK3D();
};
