/**************************************************************************/
/*  reflection_probe.hpp                                                  */
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

#include <godot_cpp/classes/visual_instance3d.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ReflectionProbe : public VisualInstance3D {
	GDEXTENSION_CLASS(ReflectionProbe, VisualInstance3D)

public:
	enum UpdateMode {
		UPDATE_ONCE = 0,
		UPDATE_ALWAYS = 1,
	};

	enum AmbientMode {
		AMBIENT_DISABLED = 0,
		AMBIENT_ENVIRONMENT = 1,
		AMBIENT_COLOR = 2,
	};

	void set_intensity(float p_intensity);
	float get_intensity() const;
	void set_blend_distance(float p_blend_distance);
	float get_blend_distance() const;
	void set_ambient_mode(ReflectionProbe::AmbientMode p_ambient);
	ReflectionProbe::AmbientMode get_ambient_mode() const;
	void set_ambient_color(const Color &p_ambient);
	Color get_ambient_color() const;
	void set_ambient_color_energy(float p_ambient_energy);
	float get_ambient_color_energy() const;
	void set_max_distance(float p_max_distance);
	float get_max_distance() const;
	void set_mesh_lod_threshold(float p_ratio);
	float get_mesh_lod_threshold() const;
	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;
	void set_origin_offset(const Vector3 &p_origin_offset);
	Vector3 get_origin_offset() const;
	void set_as_interior(bool p_enable);
	bool is_set_as_interior() const;
	void set_enable_box_projection(bool p_enable);
	bool is_box_projection_enabled() const;
	void set_enable_shadows(bool p_enable);
	bool are_shadows_enabled() const;
	void set_cull_mask(uint32_t p_layers);
	uint32_t get_cull_mask() const;
	void set_reflection_mask(uint32_t p_layers);
	uint32_t get_reflection_mask() const;
	void set_update_mode(ReflectionProbe::UpdateMode p_mode);
	ReflectionProbe::UpdateMode get_update_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(ReflectionProbe::UpdateMode);
VARIANT_ENUM_CAST(ReflectionProbe::AmbientMode);

