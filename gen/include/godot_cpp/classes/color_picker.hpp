/**************************************************************************/
/*  color_picker.hpp                                                      */
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

#include <godot_cpp/classes/v_box_container.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ColorPicker : public VBoxContainer {
	GDEXTENSION_CLASS(ColorPicker, VBoxContainer)

public:
	enum ColorModeType {
		MODE_RGB = 0,
		MODE_HSV = 1,
		MODE_RAW = 2,
		MODE_LINEAR = 2,
		MODE_OKHSL = 3,
	};

	enum PickerShapeType {
		SHAPE_HSV_RECTANGLE = 0,
		SHAPE_HSV_WHEEL = 1,
		SHAPE_VHS_CIRCLE = 2,
		SHAPE_OKHSL_CIRCLE = 3,
		SHAPE_NONE = 4,
		SHAPE_OK_HS_RECTANGLE = 5,
		SHAPE_OK_HL_RECTANGLE = 6,
	};

	void set_pick_color(const Color &p_color);
	Color get_pick_color() const;
	void set_deferred_mode(bool p_mode);
	bool is_deferred_mode() const;
	void set_color_mode(ColorPicker::ColorModeType p_color_mode);
	ColorPicker::ColorModeType get_color_mode() const;
	void set_edit_alpha(bool p_show);
	bool is_editing_alpha() const;
	void set_edit_intensity(bool p_show);
	bool is_editing_intensity() const;
	void set_can_add_swatches(bool p_enabled);
	bool are_swatches_enabled() const;
	void set_presets_visible(bool p_visible);
	bool are_presets_visible() const;
	void set_modes_visible(bool p_visible);
	bool are_modes_visible() const;
	void set_sampler_visible(bool p_visible);
	bool is_sampler_visible() const;
	void set_sliders_visible(bool p_visible);
	bool are_sliders_visible() const;
	void set_hex_visible(bool p_visible);
	bool is_hex_visible() const;
	void add_preset(const Color &p_color);
	void erase_preset(const Color &p_color);
	PackedColorArray get_presets() const;
	void add_recent_preset(const Color &p_color);
	void erase_recent_preset(const Color &p_color);
	PackedColorArray get_recent_presets() const;
	void set_picker_shape(ColorPicker::PickerShapeType p_shape);
	ColorPicker::PickerShapeType get_picker_shape() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VBoxContainer::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ColorPicker::ColorModeType);
VARIANT_ENUM_CAST(ColorPicker::PickerShapeType);

