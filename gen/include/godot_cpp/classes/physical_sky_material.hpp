/**************************************************************************/
/*  physical_sky_material.hpp                                             */
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

class PhysicalSkyMaterial : public Material {
	GDEXTENSION_CLASS(PhysicalSkyMaterial, Material)

public:
	void set_rayleigh_coefficient(float p_rayleigh);
	float get_rayleigh_coefficient() const;
	void set_rayleigh_color(const Color &p_color);
	Color get_rayleigh_color() const;
	void set_mie_coefficient(float p_mie);
	float get_mie_coefficient() const;
	void set_mie_eccentricity(float p_eccentricity);
	float get_mie_eccentricity() const;
	void set_mie_color(const Color &p_color);
	Color get_mie_color() const;
	void set_turbidity(float p_turbidity);
	float get_turbidity() const;
	void set_sun_disk_scale(float p_scale);
	float get_sun_disk_scale() const;
	void set_ground_color(const Color &p_color);
	Color get_ground_color() const;
	void set_energy_multiplier(float p_multiplier);
	float get_energy_multiplier() const;
	void set_use_debanding(bool p_use_debanding);
	bool get_use_debanding() const;
	void set_night_sky(const Ref<Texture2D> &p_night_sky);
	Ref<Texture2D> get_night_sky() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Material::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

