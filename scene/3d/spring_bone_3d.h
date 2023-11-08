/**************************************************************************/
/*  spring_bone_3d.h                                                      */
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

#ifndef SPRING_BONE_3D_H
#define SPRING_BONE_3D_H

#ifndef _3D_DISABLED

#include "scene/3d/skeleton_3d.h"

class SpringBone3D : public Node {
	GDCLASS(SpringBone3D, Node);

private:
	bool enabled = true;
	NodePath skeleton_node;
	StringName bone;
	real_t stiffness = 0.75;
	real_t damping = 0.05;
	Vector3 additional_force = Vector3(0.0, 0.0, 0.0);
	real_t influence = 1.0;
	bool stretchable = false;
	StringName tail_bone;
	Vector3 tail_position_offset = Vector3(0.0, 0.0, 0.0);

	Variant skeleton_ref = Variant();
	BoneId bone_id = -1;
	BoneId bone_id_parent = -1;
	BoneId bone_id_tail = -1;
	Vector3 tail_pos;
	Vector3 prev_pos;
	Vector3 tail_dir;

protected:
	void _validate_property(PropertyInfo &p_property) const;

	static void _bind_methods();
	virtual void _notification(int p_what);

	virtual Vector3 _adjust_tail_position(const Vector3 &p_tail_position, const Vector3 &p_previous_tail_position, const Vector3 &p_bone_position, real_t p_tail_length);
	GDVIRTUAL4R(Vector3, _adjust_tail_position, Vector3, Vector3, Vector3, real_t);

public:
	SpringBone3D();
	virtual ~SpringBone3D();

	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_skeleton_node(const NodePath &p_path);
	NodePath get_skeleton_node() const;

	void set_bone(const StringName &p_bone);
	StringName get_bone() const;

	void set_stiffness(real_t p_stiffness);
	real_t get_stiffness() const;

	void set_damping(real_t p_damping);
	real_t get_damping() const;

	void set_additional_force(const Vector3 &p_additional_force);
	const Vector3 &get_additional_force() const;

	void set_influence(real_t p_influence);
	real_t get_influence() const;

	void set_stretchable(bool p_stretchable);
	bool is_stretchable() const;

	void set_tail_bone(const StringName &p_tail_bone);
	StringName get_tail_bone() const;

	void set_tail_position_offset(const Vector3 &p_tail_position_offset);
	const Vector3 &get_tail_position_offset() const;

private:
	Skeleton3D *get_skeleton() const;

	void reload_bone();
	void process_spring(real_t p_delta);
};

#endif // _3D_DISABLED

#endif // SPRING_BONE_3D_H
