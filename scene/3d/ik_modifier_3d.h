/**************************************************************************/
/*  ik_modifier_3d.h                                                      */
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

class IKModifier3D : public SkeletonModifier3D {
	GDCLASS(IKModifier3D, SkeletonModifier3D);

protected:
#ifdef TOOLS_ENABLED
	bool saving = false;
#endif // TOOLS_ENABLED

	bool mutable_bone_axes = true;
	Transform3D cached_space;
	bool joints_dirty = false;

public:
	struct BoneJoint {
		StringName name;
		int bone = -1;
	};

	struct IKModifier3DSolverInfo {
		Quaternion current_lpose;
		Quaternion current_lrest;
		Quaternion current_gpose;
		Quaternion current_grest;
		Vector3 current_vector; // Global so needs xfrom_inv by gpose or grest in the process.
		Vector3 forward_vector; // Local.
		float length = 0.0;
	};

	struct IKModifier3DSetting {
		bool simulation_dirty = true;
		bool joints_dirty = false;
	};

protected:
	LocalVector<IKModifier3DSetting *> settings;

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _set_active(bool p_active) override;
	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;

	virtual void _validate_bone_names() override;

	void _rest_updated();
	virtual void _make_all_joints_dirty();
	virtual void _init_joints(Skeleton3D *p_skeleton, int p_index);
	virtual void _update_joints(int p_index);
	virtual void _make_simulation_dirty(int p_index);
	virtual void _update_bone_axis(Skeleton3D *p_skeleton, int p_index);

#ifdef TOOLS_ENABLED
	bool gizmo_dirty = false;
	void _make_gizmo_dirty();
	virtual void _update_mutable_info();
	void _redraw_gizmo();
#endif // TOOLS_ENABLED

	virtual void _process_modification(double p_delta) override;
	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta);

	template <typename T>
	void _set_setting_count(int p_count) {
		ERR_FAIL_COND(p_count < 0);
		int delta = p_count - settings.size();
		if (delta < 0) {
			for (int i = delta; i < 0; i++) {
				memdelete(static_cast<T *>(settings[settings.size() + i]));
				settings[settings.size() + i] = nullptr;
			}
		}
		settings.resize(p_count);
		delta++;
		if (delta > 1) {
			for (int i = 1; i < delta; i++) {
				settings[p_count - i] = memnew(T);
			}
		}
		notify_property_list_changed();
	}
	template <typename T>
	LocalVector<T *> _cast_settings() const {
		LocalVector<T *> result;
		for (uint32_t i = 0; i < settings.size(); i++) {
			result.push_back(static_cast<T *>(settings[i]));
		}
		return result;
	}

public:
	int get_setting_count() const;

	virtual void set_setting_count(int p_count) {
		_set_setting_count<IKModifier3DSetting>(p_count);
	}
	virtual void clear_settings() {
		_set_setting_count<IKModifier3DSetting>(0);
	}

	void set_mutable_bone_axes(bool p_enabled);
	bool are_bone_axes_mutable() const;

	// Helper.
	static Quaternion get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation);
	static Vector3 get_bone_axis(Skeleton3D *p_skeleton, int p_end_bone, BoneDirection p_direction, bool p_mutable_bone_axes);

	// To process manually.
	void reset();

	~IKModifier3D();
};
