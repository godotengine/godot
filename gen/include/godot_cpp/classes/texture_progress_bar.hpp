/**************************************************************************/
/*  texture_progress_bar.hpp                                              */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/range.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class TextureProgressBar : public Range {
	GDEXTENSION_CLASS(TextureProgressBar, Range)

public:
	enum FillMode {
		FILL_LEFT_TO_RIGHT = 0,
		FILL_RIGHT_TO_LEFT = 1,
		FILL_TOP_TO_BOTTOM = 2,
		FILL_BOTTOM_TO_TOP = 3,
		FILL_CLOCKWISE = 4,
		FILL_COUNTER_CLOCKWISE = 5,
		FILL_BILINEAR_LEFT_AND_RIGHT = 6,
		FILL_BILINEAR_TOP_AND_BOTTOM = 7,
		FILL_CLOCKWISE_AND_COUNTER_CLOCKWISE = 8,
	};

	void set_under_texture(const Ref<Texture2D> &p_tex);
	Ref<Texture2D> get_under_texture() const;
	void set_progress_texture(const Ref<Texture2D> &p_tex);
	Ref<Texture2D> get_progress_texture() const;
	void set_over_texture(const Ref<Texture2D> &p_tex);
	Ref<Texture2D> get_over_texture() const;
	void set_fill_mode(int32_t p_mode);
	int32_t get_fill_mode();
	void set_tint_under(const Color &p_tint);
	Color get_tint_under() const;
	void set_tint_progress(const Color &p_tint);
	Color get_tint_progress() const;
	void set_tint_over(const Color &p_tint);
	Color get_tint_over() const;
	void set_texture_progress_offset(const Vector2 &p_offset);
	Vector2 get_texture_progress_offset() const;
	void set_radial_initial_angle(float p_mode);
	float get_radial_initial_angle();
	void set_radial_center_offset(const Vector2 &p_mode);
	Vector2 get_radial_center_offset();
	void set_fill_degrees(float p_mode);
	float get_fill_degrees();
	void set_stretch_margin(Side p_margin, int32_t p_value);
	int32_t get_stretch_margin(Side p_margin) const;
	void set_nine_patch_stretch(bool p_stretch);
	bool get_nine_patch_stretch() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Range::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(TextureProgressBar::FillMode);

