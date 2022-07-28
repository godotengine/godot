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

class SkeletonModification3D : public Node3D {
	GDCLASS(SkeletonModification3D, Node3D);

protected:
	static void _bind_methods();

	int execution_mode = 0; // 0 = process

	bool enabled = true;
	bool is_setup = false;
	bool execution_error_found = false;
	Skeleton3D *skeleton = nullptr;
	NodePath skeleton_path = NodePath("..");

	bool _print_execution_error(bool p_condition, String p_message);

public:
	real_t clamp_angle(real_t p_angle, real_t p_min_bound, real_t p_max_bound, bool p_invert);

	void set_enabled(bool p_enabled);
	bool get_enabled();

	void set_execution_mode(int p_mode);
	int get_execution_mode() const;

	void set_is_setup(bool p_setup);
	bool get_is_setup() const;

	NodePath get_skeleton_path() const;
	void set_skeleton_path(NodePath p_path);

	SkeletonModification3D() {}
};

#endif // SKELETON_MODIFICATION_3D_H
