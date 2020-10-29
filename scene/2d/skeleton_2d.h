/*************************************************************************/
/*  skeleton_2d.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SKELETON_2D_H
#define SKELETON_2D_H

#include "scene/2d/node_2d.h"

class Skeleton2D;

class Bone2D : public Node2D {
	GDCLASS(Bone2D, Node2D);

	friend class Skeleton2D;
#ifdef TOOLS_ENABLED
	friend class AnimatedValuesBackup;
#endif

	Bone2D *parent_bone = nullptr;
	Skeleton2D *skeleton = nullptr;
	Transform2D rest;
	real_t default_length = 16.0;

	int skeleton_index = -1;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_rest(const Transform2D &p_rest);
	Transform2D get_rest() const;
	void apply_rest();
	Transform2D get_skeleton_rest() const;

	TypedArray<String> get_configuration_warnings() const override;

	void set_default_length(real_t p_length);
	real_t get_default_length() const;

	int get_index_in_skeleton() const;

	Bone2D();
};

class Skeleton2D : public Node2D {
	GDCLASS(Skeleton2D, Node2D);

	friend class Bone2D;
#ifdef TOOLS_ENABLED
	friend class AnimatedValuesBackup;
#endif

	struct Bone {
		bool operator<(const Bone &p_bone) const {
			return p_bone.bone->is_greater_than(bone);
		}
		Bone2D *bone = nullptr;
		int parent_index = 0;
		Transform2D accum_transform;
		Transform2D rest_inverse;
	};

	Vector<Bone> bones;

	bool bone_setup_dirty = true;
	void _make_bone_setup_dirty();
	void _update_bone_setup();

	bool transform_dirty = true;
	void _make_transform_dirty();
	void _update_transform();

	RID skeleton;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	int get_bone_count() const;
	Bone2D *get_bone(int p_idx);

	RID get_skeleton() const;
	Skeleton2D();
	~Skeleton2D();
};

#endif // SKELETON_2D_H
