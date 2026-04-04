/**************************************************************************/
/*  polygon2d.hpp                                                         */
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

#include <godot_cpp/classes/node2d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class Polygon2D : public Node2D {
	GDEXTENSION_CLASS(Polygon2D, Node2D)

public:
	void set_polygon(const PackedVector2Array &p_polygon);
	PackedVector2Array get_polygon() const;
	void set_uv(const PackedVector2Array &p_uv);
	PackedVector2Array get_uv() const;
	void set_color(const Color &p_color);
	Color get_color() const;
	void set_polygons(const Array &p_polygons);
	Array get_polygons() const;
	void set_vertex_colors(const PackedColorArray &p_vertex_colors);
	PackedColorArray get_vertex_colors() const;
	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;
	void set_texture_offset(const Vector2 &p_texture_offset);
	Vector2 get_texture_offset() const;
	void set_texture_rotation(float p_texture_rotation);
	float get_texture_rotation() const;
	void set_texture_scale(const Vector2 &p_texture_scale);
	Vector2 get_texture_scale() const;
	void set_invert_enabled(bool p_invert);
	bool get_invert_enabled() const;
	void set_antialiased(bool p_antialiased);
	bool get_antialiased() const;
	void set_invert_border(float p_invert_border);
	float get_invert_border() const;
	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;
	void add_bone(const NodePath &p_path, const PackedFloat32Array &p_weights);
	int32_t get_bone_count() const;
	NodePath get_bone_path(int32_t p_index) const;
	PackedFloat32Array get_bone_weights(int32_t p_index) const;
	void erase_bone(int32_t p_index);
	void clear_bones();
	void set_bone_path(int32_t p_index, const NodePath &p_path);
	void set_bone_weights(int32_t p_index, const PackedFloat32Array &p_weights);
	void set_skeleton(const NodePath &p_skeleton);
	NodePath get_skeleton() const;
	void set_internal_vertex_count(int32_t p_internal_vertex_count);
	int32_t get_internal_vertex_count() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

