/*************************************************************************/
/*  skeleton_modification_3d.h                                           */
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

#ifndef SKELETON_MODIFICATION_3D_H
#define SKELETON_MODIFICATION_3D_H

#include "core/string/node_path.h"
#include "scene/3d/skeleton_3d.h"

class SkeletonModification3D : public Node {
	GDCLASS(SkeletonModification3D, Node);

private:
	static void _bind_methods();

	bool enabled = true;
	bool run_in_editor = true;
	bool skeleton_change_queued = true;
	mutable Variant cached_skeleton;
	mutable String bone_name_list;
	uint64_t cached_skeleton_version = 0;
	NodePath skeleton_path = NodePath("..");

protected:
	bool _cache_bone(int &bone_cache, const String &target_bone_name) const {
		if (bone_cache == UNCACHED_BONE_IDX) {
			bone_cache = resolve_bone(target_bone_name);
		}
		return bone_cache >= 0;
	}
	bool _cache_target(Variant &cache, const NodePath &target_node_path, const String &target_bone_name) const {
		if (cache.get_type() == Variant::NIL) {
			cache = resolve_target(target_node_path, target_bone_name);
		}
		return cache.get_type() == Variant::OBJECT || cache.get_type() == Variant::INT;
	}

	enum Bone_Forward_Axis {
		BONE_AXIS_X_FORWARD = 0,
		BONE_AXIS_Y_FORWARD = 1,
		BONE_AXIS_Z_FORWARD = 2,
		BONE_AXIS_NEGATIVE_X_FORWARD = 3,
		BONE_AXIS_NEGATIVE_Y_FORWARD = 4,
		BONE_AXIS_NEGATIVE_Z_FORWARD = 5,
	};

	// The forward direction vector and rest bone forward axis should be cached because they do
	// not change 99% of the time, but recalculating them can be expensive on models with many bones.
	static Bone_Forward_Axis vector_to_forward_axis(Vector3 p_rest_bone_forward_vector);

public:
	enum { UNCACHED_BONE_IDX = -2 };

	void set_enabled(bool p_enabled);
	bool get_enabled() const;
	void set_run_in_editor(bool p_enabled_in_editor);
	bool get_run_in_editor() const;

	NodePath get_skeleton_path() const;
	void set_skeleton_path(NodePath p_path);
	Skeleton3D *get_skeleton() const;

	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int32_t p_what);
	String get_bone_name_list() const;

	virtual void skeleton_changed(Skeleton3D *skeleton);
	GDVIRTUAL1(_skeleton_changed, Skeleton3D *);
	virtual void execute(real_t delta);
	GDVIRTUAL1(_execute, real_t);
	virtual bool is_bone_property(String property_name) const;
	GDVIRTUAL1R(bool, _is_bone_property, String);
	virtual bool is_property_hidden(String property_name) const;
	GDVIRTUAL1R(bool, _is_property_hidden, String);
	PackedStringArray get_configuration_warnings() const override;

	int resolve_bone(const String &target_bone_name) const;
	Variant resolve_target(const NodePath &target_node_path, const String &target_bone_name) const;
	Transform3D get_target_transform(Variant resolved_target) const;
	Quaternion get_target_quaternion(Variant resolved_target) const;
	Transform3D get_bone_transform(int bone_idx) const;
	Quaternion get_bone_quaternion(int bone_idx) const;
	Vector3 get_bone_rest_forward_vector(int p_bone);
	Basis global_pose_z_forward_to_bone_forward(Vector3 p_bone_forward_vector, Basis p_basis);

	SkeletonModification3D() {}
};

#endif // SKELETON_MODIFICATION_3D_H
