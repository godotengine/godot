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

#include "scene/3d/skeleton_3d.h"
#include "scene/resources/skeleton_modification_3d.h"

#ifndef SKELETONMODIFICATION3DLOOKAT_H
#define SKELETONMODIFICATION3DLOOKAT_H

class SkeletonModification3DLookAt : public SkeletonModification3D {
	GDCLASS(SkeletonModification3DLookAt, SkeletonModification3D);

private:
	String bone_name = "";
	int bone_idx = -1;
	NodePath target_node;
	ObjectID target_node_cache;

	Vector3 additional_rotation = Vector3(1, 0, 0);
	bool lock_rotation_to_plane = false;
	int lock_rotation_plane = ROTATION_PLANE_X;

	void update_cache();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	enum ROTATION_PLANE {
		ROTATION_PLANE_X,
		ROTATION_PLANE_Y,
		ROTATION_PLANE_Z
	};

	virtual void _execute(real_t p_delta) override;
	virtual void _setup_modification(SkeletonModificationStack3D *p_stack) override;

	void set_bone_name(String p_name);
	String get_bone_name() const;

	void set_bone_index(int p_idx);
	int get_bone_index() const;

	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;

	void set_additional_rotation(Vector3 p_offset);
	Vector3 get_additional_rotation() const;

	void set_lock_rotation_to_plane(bool p_lock_to_plane);
	bool get_lock_rotation_to_plane() const;
	void set_lock_rotation_plane(int p_plane);
	int get_lock_rotation_plane() const;

	SkeletonModification3DLookAt();
	~SkeletonModification3DLookAt();
};

#endif //SKELETONMODIFICATION3DLOOKAT_H
