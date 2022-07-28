/*************************************************************************/
/*  skeleton_modification_2d.h                                           */
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

#ifndef SKELETON_MODIFICATION_2D_H
#define SKELETON_MODIFICATION_2D_H

#include "scene/2d/skeleton_2d.h"

///////////////////////////////////////
// SkeletonModification2D
///////////////////////////////////////

class Bone2D;

class SkeletonModification2D : public Node2D {
	GDCLASS(SkeletonModification2D, Node2D);

protected:
	static void _bind_methods();

	int execution_mode = 0; // 0 = process

	bool enabled = true;
	bool is_setup = false;
	Skeleton2D *skeleton = nullptr;
	NodePath skeleton_node = NodePath("..");

	bool _print_execution_error(bool p_condition, String p_message);

	GDVIRTUAL1(_execute, double)
	GDVIRTUAL0(_draw_editor_gizmo)

public:
	NodePath get_skeleton_path() const {
		return skeleton_node;
	}

	void set_skeleton_path(NodePath p_path) {
		skeleton_node = p_path;
	}
	virtual void _draw_editor_gizmo();

	bool editor_draw_gizmo = false;
	void set_editor_draw_gizmo(bool p_draw_gizmo);
	bool get_editor_draw_gizmo() const;

	void set_enabled(bool p_enabled);
	bool get_enabled();

	void set_is_setup(bool p_setup);
	bool get_is_setup() const;

	void set_execution_mode(int p_mode);
	int get_execution_mode() const;

	float clamp_angle(float p_angle, float p_min_bound, float p_max_bound, bool p_invert_clamp = false);
	void editor_draw_angle_constraints(Bone2D *p_operation_bone, float p_min_bound, float p_max_bound, bool p_constraint_enabled, bool p_constraint_in_localspace, bool p_constraint_inverted);

	SkeletonModification2D() {}
};

#endif // SKELETON_MODIFICATION_2D_H
