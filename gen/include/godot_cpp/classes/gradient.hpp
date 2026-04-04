/**************************************************************************/
/*  gradient.hpp                                                          */
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
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Gradient : public Resource {
	GDEXTENSION_CLASS(Gradient, Resource)

public:
	enum InterpolationMode {
		GRADIENT_INTERPOLATE_LINEAR = 0,
		GRADIENT_INTERPOLATE_CONSTANT = 1,
		GRADIENT_INTERPOLATE_CUBIC = 2,
	};

	enum ColorSpace {
		GRADIENT_COLOR_SPACE_SRGB = 0,
		GRADIENT_COLOR_SPACE_LINEAR_SRGB = 1,
		GRADIENT_COLOR_SPACE_OKLAB = 2,
	};

	void add_point(float p_offset, const Color &p_color);
	void remove_point(int32_t p_point);
	void set_offset(int32_t p_point, float p_offset);
	float get_offset(int32_t p_point);
	void reverse();
	void set_color(int32_t p_point, const Color &p_color);
	Color get_color(int32_t p_point);
	Color sample(float p_offset);
	int32_t get_point_count() const;
	void set_offsets(const PackedFloat32Array &p_offsets);
	PackedFloat32Array get_offsets() const;
	void set_colors(const PackedColorArray &p_colors);
	PackedColorArray get_colors() const;
	void set_interpolation_mode(Gradient::InterpolationMode p_interpolation_mode);
	Gradient::InterpolationMode get_interpolation_mode();
	void set_interpolation_color_space(Gradient::ColorSpace p_interpolation_color_space);
	Gradient::ColorSpace get_interpolation_color_space();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Gradient::InterpolationMode);
VARIANT_ENUM_CAST(Gradient::ColorSpace);

