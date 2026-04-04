/**************************************************************************/
/*  curve.hpp                                                             */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve : public Resource {
	GDEXTENSION_CLASS(Curve, Resource)

public:
	enum TangentMode {
		TANGENT_FREE = 0,
		TANGENT_LINEAR = 1,
		TANGENT_MODE_COUNT = 2,
	};

	int32_t get_point_count() const;
	void set_point_count(int32_t p_count);
	int32_t add_point(const Vector2 &p_position, float p_left_tangent = 0, float p_right_tangent = 0, Curve::TangentMode p_left_mode = (Curve::TangentMode)0, Curve::TangentMode p_right_mode = (Curve::TangentMode)0);
	void remove_point(int32_t p_index);
	void clear_points();
	Vector2 get_point_position(int32_t p_index) const;
	void set_point_value(int32_t p_index, float p_y);
	int32_t set_point_offset(int32_t p_index, float p_offset);
	float sample(float p_offset) const;
	float sample_baked(float p_offset) const;
	float get_point_left_tangent(int32_t p_index) const;
	float get_point_right_tangent(int32_t p_index) const;
	Curve::TangentMode get_point_left_mode(int32_t p_index) const;
	Curve::TangentMode get_point_right_mode(int32_t p_index) const;
	void set_point_left_tangent(int32_t p_index, float p_tangent);
	void set_point_right_tangent(int32_t p_index, float p_tangent);
	void set_point_left_mode(int32_t p_index, Curve::TangentMode p_mode);
	void set_point_right_mode(int32_t p_index, Curve::TangentMode p_mode);
	float get_min_value() const;
	void set_min_value(float p_min);
	float get_max_value() const;
	void set_max_value(float p_max);
	float get_value_range() const;
	float get_min_domain() const;
	void set_min_domain(float p_min);
	float get_max_domain() const;
	void set_max_domain(float p_max);
	float get_domain_range() const;
	void clean_dupes();
	void bake();
	int32_t get_bake_resolution() const;
	void set_bake_resolution(int32_t p_resolution);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Curve::TangentMode);

