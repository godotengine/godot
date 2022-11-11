/*************************************************************************/
/*  skeleton_modification_3d_ccdik.h                                     */
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

#ifndef SKELETON_MODIFICATION_3D_CCDIK_H
#define SKELETON_MODIFICATION_3D_CCDIK_H

#include "core/templates/local_vector.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

class SkeletonModification3DCCDIK : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DCCDIK, SkeletonModification3D);

private:
	enum CCDIK_Axes {
		AXIS_X,
		AXIS_Y,
		AXIS_Z
	};

	struct CCDIK_Joint_Data {
		String bone_name = "";
		mutable int bone_idx = UNCACHED_BONE_IDX;
		int ccdik_axis = 0;

		bool enable_constraint = false;
		real_t constraint_angle_min = 0;
		real_t constraint_angle_max = (2.0 * Math_PI);
		bool constraint_angles_invert = false;
	};

	LocalVector<CCDIK_Joint_Data> ccdik_data_chain;
	NodePath target_node;
	String target_bone;
	mutable Variant target_cache;

	NodePath tip_node;
	String tip_bone;
	mutable Variant tip_cache;

	bool use_high_quality_solve = true;

	void _execute_ccdik_joint(int p_joint_idx, const Transform3D &p_target_transform, const Transform3D &p_tip_transform);

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void skeleton_changed(Skeleton3D *skeleton) override;
	void execute(real_t p_delta) override;
	bool is_property_hidden(String property_name) const override;
	bool is_bone_property(String property_name) const override;
	PackedStringArray get_configuration_warnings() const override;

public:
	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;
	void set_target_bone(const String &p_target_bone);
	String get_target_bone() const;

	void set_tip_node(const NodePath &p_tip_node);
	NodePath get_tip_node() const;
	void set_tip_bone(const String &p_tip_bone);
	String get_tip_bone() const;

	void set_use_high_quality_solve(bool p_solve);
	bool get_use_high_quality_solve() const;

	String get_joint_bone(int p_joint_idx) const;
	void set_joint_bone(int p_joint_idx, String p_bone_name);
	int get_joint_ccdik_axis(int p_joint_idx) const;
	void set_joint_ccdik_axis(int p_joint_idx, int p_axis);
	bool get_joint_enable_constraint(int p_joint_idx) const;
	void set_joint_enable_constraint(int p_joint_idx, bool p_enable);
	real_t get_joint_constraint_angle_min(int p_joint_idx) const;
	void set_joint_constraint_angle_min(int p_joint_idx, real_t p_angle_min);
	real_t get_joint_constraint_angle_max(int p_joint_idx) const;
	void set_joint_constraint_angle_max(int p_joint_idx, real_t p_angle_max);
	bool get_joint_constraint_invert(int p_joint_idx) const;
	void set_joint_constraint_invert(int p_joint_idx, bool p_invert);

	int get_joint_count();
	void set_joint_count(int p_ccdik_chain_length);

	SkeletonModification3DCCDIK();
	~SkeletonModification3DCCDIK();
};

#endif // SKELETON_MODIFICATION_3D_CCDIK_H
