/**************************************************************************/
/*  line2d.hpp                                                            */
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
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve;
class Gradient;
class Texture2D;

class Line2D : public Node2D {
	GDEXTENSION_CLASS(Line2D, Node2D)

public:
	enum LineJointMode {
		LINE_JOINT_SHARP = 0,
		LINE_JOINT_BEVEL = 1,
		LINE_JOINT_ROUND = 2,
	};

	enum LineCapMode {
		LINE_CAP_NONE = 0,
		LINE_CAP_BOX = 1,
		LINE_CAP_ROUND = 2,
	};

	enum LineTextureMode {
		LINE_TEXTURE_NONE = 0,
		LINE_TEXTURE_TILE = 1,
		LINE_TEXTURE_STRETCH = 2,
	};

	void set_points(const PackedVector2Array &p_points);
	PackedVector2Array get_points() const;
	void set_point_position(int32_t p_index, const Vector2 &p_position);
	Vector2 get_point_position(int32_t p_index) const;
	int32_t get_point_count() const;
	void add_point(const Vector2 &p_position, int32_t p_index = -1);
	void remove_point(int32_t p_index);
	void clear_points();
	void set_closed(bool p_closed);
	bool is_closed() const;
	void set_width(float p_width);
	float get_width() const;
	void set_curve(const Ref<Curve> &p_curve);
	Ref<Curve> get_curve() const;
	void set_default_color(const Color &p_color);
	Color get_default_color() const;
	void set_gradient(const Ref<Gradient> &p_color);
	Ref<Gradient> get_gradient() const;
	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;
	void set_texture_mode(Line2D::LineTextureMode p_mode);
	Line2D::LineTextureMode get_texture_mode() const;
	void set_joint_mode(Line2D::LineJointMode p_mode);
	Line2D::LineJointMode get_joint_mode() const;
	void set_begin_cap_mode(Line2D::LineCapMode p_mode);
	Line2D::LineCapMode get_begin_cap_mode() const;
	void set_end_cap_mode(Line2D::LineCapMode p_mode);
	Line2D::LineCapMode get_end_cap_mode() const;
	void set_sharp_limit(float p_limit);
	float get_sharp_limit() const;
	void set_round_precision(int32_t p_precision);
	int32_t get_round_precision() const;
	void set_antialiased(bool p_antialiased);
	bool get_antialiased() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Line2D::LineJointMode);
VARIANT_ENUM_CAST(Line2D::LineCapMode);
VARIANT_ENUM_CAST(Line2D::LineTextureMode);

