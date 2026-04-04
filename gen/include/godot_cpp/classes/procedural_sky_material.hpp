/**************************************************************************/
/*  procedural_sky_material.hpp                                           */
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

#include <godot_cpp/classes/material.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/color.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Texture2D;

class ProceduralSkyMaterial : public Material {
	GDEXTENSION_CLASS(ProceduralSkyMaterial, Material)

public:
	void set_sky_top_color(const Color &p_color);
	Color get_sky_top_color() const;
	void set_sky_horizon_color(const Color &p_color);
	Color get_sky_horizon_color() const;
	void set_sky_curve(float p_curve);
	float get_sky_curve() const;
	void set_sky_energy_multiplier(float p_multiplier);
	float get_sky_energy_multiplier() const;
	void set_sky_cover(const Ref<Texture2D> &p_sky_cover);
	Ref<Texture2D> get_sky_cover() const;
	void set_sky_cover_modulate(const Color &p_color);
	Color get_sky_cover_modulate() const;
	void set_ground_bottom_color(const Color &p_color);
	Color get_ground_bottom_color() const;
	void set_ground_horizon_color(const Color &p_color);
	Color get_ground_horizon_color() const;
	void set_ground_curve(float p_curve);
	float get_ground_curve() const;
	void set_ground_energy_multiplier(float p_energy);
	float get_ground_energy_multiplier() const;
	void set_sun_angle_max(float p_degrees);
	float get_sun_angle_max() const;
	void set_sun_curve(float p_curve);
	float get_sun_curve() const;
	void set_use_debanding(bool p_use_debanding);
	bool get_use_debanding() const;
	void set_energy_multiplier(float p_multiplier);
	float get_energy_multiplier() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Material::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

