/**************************************************************************/
/*  style_box_texture.hpp                                                 */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/style_box.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/rect2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class StyleBoxTexture : public StyleBox {
	GDEXTENSION_CLASS(StyleBoxTexture, StyleBox)

public:
	enum AxisStretchMode {
		AXIS_STRETCH_MODE_STRETCH = 0,
		AXIS_STRETCH_MODE_TILE = 1,
		AXIS_STRETCH_MODE_TILE_FIT = 2,
	};

	void set_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_texture() const;
	void set_texture_margin(Side p_margin, float p_size);
	void set_texture_margin_all(float p_size);
	float get_texture_margin(Side p_margin) const;
	void set_expand_margin(Side p_margin, float p_size);
	void set_expand_margin_all(float p_size);
	float get_expand_margin(Side p_margin) const;
	void set_region_rect(const Rect2 &p_region);
	Rect2 get_region_rect() const;
	void set_draw_center(bool p_enable);
	bool is_draw_center_enabled() const;
	void set_modulate(const Color &p_color);
	Color get_modulate() const;
	void set_h_axis_stretch_mode(StyleBoxTexture::AxisStretchMode p_mode);
	StyleBoxTexture::AxisStretchMode get_h_axis_stretch_mode() const;
	void set_v_axis_stretch_mode(StyleBoxTexture::AxisStretchMode p_mode);
	StyleBoxTexture::AxisStretchMode get_v_axis_stretch_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		StyleBox::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(StyleBoxTexture::AxisStretchMode);

