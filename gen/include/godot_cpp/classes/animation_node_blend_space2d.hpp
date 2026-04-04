/**************************************************************************/
/*  animation_node_blend_space2d.hpp                                      */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/animation_root_node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AnimationNodeBlendSpace2D : public AnimationRootNode {
	GDEXTENSION_CLASS(AnimationNodeBlendSpace2D, AnimationRootNode)

public:
	enum BlendMode {
		BLEND_MODE_INTERPOLATED = 0,
		BLEND_MODE_DISCRETE = 1,
		BLEND_MODE_DISCRETE_CARRY = 2,
	};

	void add_blend_point(const Ref<AnimationRootNode> &p_node, const Vector2 &p_pos, int32_t p_at_index = -1);
	void set_blend_point_position(int32_t p_point, const Vector2 &p_pos);
	Vector2 get_blend_point_position(int32_t p_point) const;
	void set_blend_point_node(int32_t p_point, const Ref<AnimationRootNode> &p_node);
	Ref<AnimationRootNode> get_blend_point_node(int32_t p_point) const;
	void remove_blend_point(int32_t p_point);
	int32_t get_blend_point_count() const;
	void add_triangle(int32_t p_x, int32_t p_y, int32_t p_z, int32_t p_at_index = -1);
	int32_t get_triangle_point(int32_t p_triangle, int32_t p_point);
	void remove_triangle(int32_t p_triangle);
	int32_t get_triangle_count() const;
	void set_min_space(const Vector2 &p_min_space);
	Vector2 get_min_space() const;
	void set_max_space(const Vector2 &p_max_space);
	Vector2 get_max_space() const;
	void set_snap(const Vector2 &p_snap);
	Vector2 get_snap() const;
	void set_x_label(const String &p_text);
	String get_x_label() const;
	void set_y_label(const String &p_text);
	String get_y_label() const;
	void set_auto_triangles(bool p_enable);
	bool get_auto_triangles() const;
	void set_blend_mode(AnimationNodeBlendSpace2D::BlendMode p_mode);
	AnimationNodeBlendSpace2D::BlendMode get_blend_mode() const;
	void set_use_sync(bool p_enable);
	bool is_using_sync() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AnimationRootNode::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationNodeBlendSpace2D::BlendMode);

