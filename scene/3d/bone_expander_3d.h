/**************************************************************************/
/*  bone_expander_3d.h                                                    */
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

class BoneExpander3D : public SkeletonModifier3D {
	GDCLASS(BoneExpander3D, SkeletonModifier3D);

	const Vector3 DEFAULT_SCALE = Vector3(1, 1, 1);

	bool skin_dirty = false;
	bool bone_changed = false;

public:
	struct BoneExpander3DSetting {
		StringName bone_name;
		int bone = -1;
		Vector3 scale = Vector3(1, 1, 1);
	};

	struct BindInfo {
		int index = -1;
		Transform3D original_matrix;
		Vector3 apply_scale;
	};

protected:
	LocalVector<BoneExpander3DSetting> settings;
	HashMap<ObjectID, HashMap<int, BindInfo>> skin_info;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;
	virtual void _set_active(bool p_active) override;

	virtual void _process_modification(double p_delta) override;
	void _map_skin();
	void _apply_skin();
	void _restore_skin();

	void _make_skin_dirty();

	void _skinned_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new);
	void _force_render_skin(Skeleton3D *p_skeleton);

public:
	int get_setting_size();
	void set_setting_size(int p_size);
	void clear_settings();

	void set_bone_name(int p_index, const StringName &p_bone_name);
	StringName get_bone_name(int p_index) const;
	void set_bone(int p_index, int p_bone);
	int get_bone(int p_index) const;
	void set_bone_scale(int p_index, Vector3 p_scale);
	Vector3 get_bone_scale(int p_index) const;
};
