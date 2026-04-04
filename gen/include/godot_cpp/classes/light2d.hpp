/**************************************************************************/
/*  light2d.hpp                                                           */
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
#include <godot_cpp/variant/color.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Light2D : public Node2D {
	GDEXTENSION_CLASS(Light2D, Node2D)

public:
	enum ShadowFilter {
		SHADOW_FILTER_NONE = 0,
		SHADOW_FILTER_PCF5 = 1,
		SHADOW_FILTER_PCF13 = 2,
	};

	enum BlendMode {
		BLEND_MODE_ADD = 0,
		BLEND_MODE_SUB = 1,
		BLEND_MODE_MIX = 2,
	};

	void set_enabled(bool p_enabled);
	bool is_enabled() const;
	void set_editor_only(bool p_editor_only);
	bool is_editor_only() const;
	void set_color(const Color &p_color);
	Color get_color() const;
	void set_energy(float p_energy);
	float get_energy() const;
	void set_z_range_min(int32_t p_z);
	int32_t get_z_range_min() const;
	void set_z_range_max(int32_t p_z);
	int32_t get_z_range_max() const;
	void set_layer_range_min(int32_t p_layer);
	int32_t get_layer_range_min() const;
	void set_layer_range_max(int32_t p_layer);
	int32_t get_layer_range_max() const;
	void set_item_cull_mask(int32_t p_item_cull_mask);
	int32_t get_item_cull_mask() const;
	void set_item_shadow_cull_mask(int32_t p_item_shadow_cull_mask);
	int32_t get_item_shadow_cull_mask() const;
	void set_shadow_enabled(bool p_enabled);
	bool is_shadow_enabled() const;
	void set_shadow_smooth(float p_smooth);
	float get_shadow_smooth() const;
	void set_shadow_filter(Light2D::ShadowFilter p_filter);
	Light2D::ShadowFilter get_shadow_filter() const;
	void set_shadow_color(const Color &p_shadow_color);
	Color get_shadow_color() const;
	void set_blend_mode(Light2D::BlendMode p_mode);
	Light2D::BlendMode get_blend_mode() const;
	void set_height(float p_height);
	float get_height() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node2D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Light2D::ShadowFilter);
VARIANT_ENUM_CAST(Light2D::BlendMode);

