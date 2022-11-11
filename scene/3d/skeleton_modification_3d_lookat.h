/*************************************************************************/
/*  skeleton_modification_3d_lookat.h                                    */
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

#ifndef SKELETON_MODIFICATION_3D_LOOKAT_H
#define SKELETON_MODIFICATION_3D_LOOKAT_H

#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_modification_3d.h"

class SkeletonModification3DLookAt : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DLookAt, SkeletonModification3D);

public:
	enum LockRotationPlane {
		ROTATION_UNLOCKED,
		ROTATION_PLANE_X,
		ROTATION_PLANE_Y,
		ROTATION_PLANE_Z
	};

private:
	String bone_name;
	mutable int bone_idx = UNCACHED_BONE_IDX;
	NodePath target_node;
	String target_bone;
	mutable Variant target_cache;

	Vector3 additional_rotation = Vector3();
	bool lock_rotation_to_plane = false;
	LockRotationPlane lock_rotation_plane = ROTATION_UNLOCKED;

protected:
	static void _bind_methods();
	void execute(real_t delta) override;
	void skeleton_changed(Skeleton3D *skeleton) override;
	bool is_property_hidden(String property_name) const override;
	bool is_bone_property(String property_name) const override;
	PackedStringArray get_configuration_warnings() const override;

public:
	void set_bone(const String &p_name);
	String get_bone() const;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;
	void set_target_bone(const String &p_target_bone);
	String get_target_bone() const;

	void set_additional_rotation(Vector3 p_offset);
	Vector3 get_additional_rotation() const;

	void set_lock_rotation_plane(LockRotationPlane p_plane);
	LockRotationPlane get_lock_rotation_plane() const;

	SkeletonModification3DLookAt() {}
	~SkeletonModification3DLookAt() {}
};

VARIANT_ENUM_CAST(SkeletonModification3DLookAt::LockRotationPlane);

#endif // SKELETON_MODIFICATION_3D_LOOKAT_H
