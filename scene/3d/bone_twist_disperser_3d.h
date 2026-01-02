/**************************************************************************/
/*  bone_twist_disperser_3d.h                                             */
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

#include "scene/3d/skeleton_modifier_3d.h"

class BoneTwistDisperser3D : public SkeletonModifier3D {
	GDCLASS(BoneTwistDisperser3D, SkeletonModifier3D);

	bool mutable_bone_axes = true;

public:
	enum DisperseMode {
		DISPERSE_MODE_EVEN,
		DISPERSE_MODE_WEIGHTED,
		DISPERSE_MODE_CUSTOM,
	};

	struct BoneJoint {
		StringName name;
		int bone = -1;
	};

	struct DisperseJointSetting {
		BoneJoint joint;
		double custom_amount = 1.0;
		// For processing.
		double amount = 1.0;
		Vector3 axis;
	};

	struct BoneTwistDisperser3DSetting {
		bool joints_dirty = false;

		DisperseMode disperse_mode = DISPERSE_MODE_EVEN;
		bool twist_from_rest = true;
		Quaternion twist_from;

		BoneJoint root_bone;
		BoneJoint end_bone;
		LocalVector<DisperseJointSetting> joints;

		bool extend_end_bone = false;
		BoneDirection end_bone_direction = BONE_DIRECTION_FROM_PARENT;

		float weight_position = 0.5;
		Ref<Curve> damping_curve;

		BoneJoint reference_bone; // To cache.

		~BoneTwistDisperser3DSetting() {
			joints.clear();
			damping_curve.unref();
		}
	};

protected:
	LocalVector<BoneTwistDisperser3DSetting *> settings;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _set_active(bool p_active) override;
	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;
	virtual void _validate_bone_names() override;

	void _make_all_joints_dirty();

	void _make_joints_dirty(int p_index);
	void _update_joints(int p_index);
	void _set_joint_bone(int p_index, int p_joint, int p_bone);

	void _update_reference_bone(int p_index);
	void _update_curve(int p_index);

	virtual void _process_modification(double p_delta) override;

public:
	void set_mutable_bone_axes(bool p_enabled);
	bool are_bone_axes_mutable() const;

	int get_setting_count() const;
	void set_setting_count(int p_count);
	void clear_settings();

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

	void set_twist_from_rest(int p_index, bool p_enabled);
	bool is_twist_from_rest(int p_index) const;
	void set_twist_from(int p_index, const Quaternion &p_from);
	Quaternion get_twist_from(int p_index) const;

	String get_reference_bone_name(int p_index) const;
	int get_reference_bone(int p_index) const;

	void set_disperse_mode(int p_index, DisperseMode p_disperse_mode);
	DisperseMode get_disperse_mode(int p_index) const;
	void set_weight_position(int p_index, float p_position);
	float get_weight_position(int p_index) const;
	void set_damping_curve(int p_index, const Ref<Curve> &p_damping_curve);
	Ref<Curve> get_damping_curve(int p_index) const;

	// Individual joints.
	String get_joint_bone_name(int p_index, int p_joint) const;
	int get_joint_bone(int p_index, int p_joint) const;

	void set_joint_twist_amount(int p_index, int p_joint, float p_amount);
	float get_joint_twist_amount(int p_index, int p_joint) const;

	void set_joint_count(int p_index, int p_count);
	int get_joint_count(int p_index) const;

	~BoneTwistDisperser3D();
};

VARIANT_ENUM_CAST(BoneTwistDisperser3D::DisperseMode);
