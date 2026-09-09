/**************************************************************************/
/*  animation_blend_space_2d.compat.inc                                   */
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

#ifndef DISABLE_DEPRECATED

#include "animation_blend_space_2d.h"

#include "core/object/class_db.h"

void AnimationNodeBlendSpace2D::_add_blend_point_bind_compat_110369(const Ref<AnimationRootNode> &p_node, const Vector2 &p_position, int p_at_index) {
	int n = p_at_index == -1 ? blend_points_used : p_at_index;
	while (find_blend_point_by_name(itos(n)) != -1) {
		n++;
	}
	add_blend_point(p_node, p_position, p_at_index, StringName(itos(n)));
}

void AnimationNodeBlendSpace2D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("add_blend_point", "node", "pos", "at_index"), &AnimationNodeBlendSpace2D::_add_blend_point_bind_compat_110369, DEFVAL(-1));
}

#endif // DISABLE_DEPRECATED
