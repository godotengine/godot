/**************************************************************************/
/*  light3d.hpp                                                           */
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
#include <godot_cpp/classes/visual_instance3d.hpp>
#include <godot_cpp/variant/color.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class Light3D : public VisualInstance3D {
	GDEXTENSION_CLASS(Light3D, VisualInstance3D)

public:
	enum Param {
		PARAM_ENERGY = 0,
		PARAM_INDIRECT_ENERGY = 1,
		PARAM_VOLUMETRIC_FOG_ENERGY = 2,
		PARAM_SPECULAR = 3,
		PARAM_RANGE = 4,
		PARAM_SIZE = 5,
		PARAM_ATTENUATION = 6,
		PARAM_SPOT_ANGLE = 7,
		PARAM_SPOT_ATTENUATION = 8,
		PARAM_SHADOW_MAX_DISTANCE = 9,
		PARAM_SHADOW_SPLIT_1_OFFSET = 10,
		PARAM_SHADOW_SPLIT_2_OFFSET = 11,
		PARAM_SHADOW_SPLIT_3_OFFSET = 12,
		PARAM_SHADOW_FADE_START = 13,
		PARAM_SHADOW_NORMAL_BIAS = 14,
		PARAM_SHADOW_BIAS = 15,
		PARAM_SHADOW_PANCAKE_SIZE = 16,
		PARAM_SHADOW_OPACITY = 17,
		PARAM_SHADOW_BLUR = 18,
		PARAM_TRANSMITTANCE_BIAS = 19,
		PARAM_INTENSITY = 20,
		PARAM_MAX = 21,
	};

	enum BakeMode {
		BAKE_DISABLED = 0,
		BAKE_STATIC = 1,
		BAKE_DYNAMIC = 2,
	};

	void set_editor_only(bool p_editor_only);
	bool is_editor_only() const;
	void set_param(Light3D::Param p_param, float p_value);
	float get_param(Light3D::Param p_param) const;
	void set_shadow(bool p_enabled);
	bool has_shadow() const;
	void set_negative(bool p_enabled);
	bool is_negative() const;
	void set_cull_mask(uint32_t p_cull_mask);
	uint32_t get_cull_mask() const;
	void set_enable_distance_fade(bool p_enable);
	bool is_distance_fade_enabled() const;
	void set_distance_fade_begin(float p_distance);
	float get_distance_fade_begin() const;
	void set_distance_fade_shadow(float p_distance);
	float get_distance_fade_shadow() const;
	void set_distance_fade_length(float p_distance);
	float get_distance_fade_length() const;
	void set_color(const Color &p_color);
	Color get_color() const;
	void set_shadow_reverse_cull_face(bool p_enable);
	bool get_shadow_reverse_cull_face() const;
	void set_shadow_caster_mask(uint32_t p_caster_mask);
	uint32_t get_shadow_caster_mask() const;
	void set_bake_mode(Light3D::BakeMode p_bake_mode);
	Light3D::BakeMode get_bake_mode() const;
	void set_projector(const Ref<Texture2D> &p_projector);
	Ref<Texture2D> get_projector() const;
	void set_temperature(float p_temperature);
	float get_temperature() const;
	Color get_correlated_color() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Light3D::Param);
VARIANT_ENUM_CAST(Light3D::BakeMode);

