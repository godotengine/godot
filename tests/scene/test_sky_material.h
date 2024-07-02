/**************************************************************************/
/*  test_sky_material.h                                                   */
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

#ifndef TEST_SKY_MATERIAL_H
#define TEST_SKY_MATERIAL_H

#include "scene/resources/3d/sky_material.h"
#include "scene/resources/sky.h"

#include "tests/test_macros.h"

namespace TestSkyMaterial {
TEST_CASE("[ProceduralSkyMaterial][Sky] Setters and Getters") {
	Ref<ProceduralSkyMaterial> test_procedural;
	test_procedural.instantiate();

	SUBCASE("[ProceduralSkyMaterial] Set and Get Top Color") {
		Color test_color = Color(0, 0, 0);
		test_procedural->set_sky_top_color(test_color);
		CHECK(test_procedural->get_sky_top_color() == test_color);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Horizon Color") {
		Color test_color = Color(0, 0, 0);
		test_procedural->set_sky_horizon_color(test_color);
		CHECK(test_procedural->get_sky_horizon_color() == test_color);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Sky Curve") {
		float test_curve = 100;
		test_procedural->set_sky_curve(test_curve);
		CHECK(test_procedural->get_sky_curve() == test_curve);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Sky Energy Multipler") {
		float test_multipler = 100;
		test_procedural->set_sky_energy_multiplier(test_multipler);
		CHECK(test_procedural->get_sky_energy_multiplier() == test_multipler);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Sky Cover") {
		Ref<Texture2D> test_cover;
		test_cover.instantiate();
		test_procedural->set_sky_cover(test_cover);
		CHECK(test_procedural->get_sky_cover() == test_cover);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Sky Cover Modulate") {
		Color test_modulate = Color(0, 0, 0);
		test_procedural->set_sky_cover_modulate(test_modulate);
		CHECK(test_procedural->get_sky_cover_modulate() == test_modulate);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Ground Bottom Color") {
		Color test_color = Color(0, 0, 0);
		test_procedural->set_ground_bottom_color(test_color);
		CHECK(test_procedural->get_ground_bottom_color() == test_color);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Ground Horizon Color") {
		Color test_color = Color(0, 0, 0);
		test_procedural->set_ground_horizon_color(test_color);
		CHECK(test_procedural->get_ground_horizon_color() == test_color);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Sky Ground Curve") {
		float test_curve = 100;
		test_procedural->set_ground_curve(test_curve);
		CHECK(test_procedural->get_ground_curve() == test_curve);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Ground Energy Multipler") {
		float test_multipler = 100;
		test_procedural->set_ground_energy_multiplier(test_multipler);
		CHECK(test_procedural->get_ground_energy_multiplier() == test_multipler);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Sun Curve") {
		float test_curve = 100;
		test_procedural->set_sun_curve(test_curve);
		CHECK(test_procedural->get_sun_curve() == test_curve);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Ground Energy Multipler") {
		float test_multipler = 100;
		test_procedural->set_ground_energy_multiplier(test_multipler);
		CHECK(test_procedural->get_ground_energy_multiplier() == test_multipler);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Use Debanding") {
		bool test_debanding = false;
		test_procedural->set_use_debanding(test_debanding);
		CHECK(test_procedural->get_use_debanding() == test_debanding);
	}

	SUBCASE("[ProceduralSkyMaterial] Set and Get Energy Multipler") {
		float test_multipler = 100;
		test_procedural->set_energy_multiplier(test_multipler);
		CHECK(test_procedural->get_energy_multiplier() == test_multipler);
	}
}

TEST_CASE("[PanoramaSkyMaterial][Sky] Setters and Getters") {
	Ref<PanoramaSkyMaterial> test_panorama;
	test_panorama.instantiate();

	SUBCASE("[PanoramaSkyMaterial] Set And Get Panorama") {
		Ref<Texture2D> test_pano;
		test_pano.instantiate();
		test_panorama->set_panorama(test_pano);
		CHECK(test_panorama->get_panorama() == test_pano);
	}

	SUBCASE("[PanoramaSkyMaterial] Set And Get Filtering") {
		bool test_filtering = false;
		test_panorama->set_filtering_enabled(test_filtering);
		CHECK(test_panorama->is_filtering_enabled() == test_filtering);
	}

	SUBCASE("[PanoramaSkyMaterial] Check Shader Mode") {
		CHECK(test_panorama->get_shader_mode() == Shader::MODE_SKY);
	}

	SUBCASE("[PanoramaSkyMaterial] Set and Get Energy Multipler") {
		float test_multipler = 100;
		test_panorama->set_energy_multiplier(test_multipler);
		CHECK(test_panorama->get_energy_multiplier() == test_multipler);
	}
}

TEST_CASE("[PhysicalSkyMaterial][Sky] Setters and Getters") {
	Ref<PhysicalSkyMaterial> test_physical;
	test_physical.instantiate();

	SUBCASE("[PhysicalSkyMaterial] Set and Get Rayleigh Coefficient") {
		float test_rayleigh = 100;
		test_physical->set_rayleigh_coefficient(test_rayleigh);
		CHECK(test_physical->get_rayleigh_coefficient() == test_rayleigh);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Rayleigh Color") {
		Color test_color = Color(0, 0, 0);
		test_physical->set_rayleigh_color(test_color);
		CHECK(test_physical->get_rayleigh_color() == test_color);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Mie Coefficient") {
		float test_mie = 100;
		test_physical->set_mie_coefficient(test_mie);
		CHECK(test_physical->get_mie_coefficient() == test_mie);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Mie Eccentricity") {
		float test_mie = 100;
		test_physical->set_mie_eccentricity(test_mie);
		CHECK(test_physical->get_mie_eccentricity() == test_mie);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Mie Color") {
		Color test_color = Color(0, 0, 0);
		test_physical->set_mie_color(test_color);
		CHECK(test_physical->get_mie_color() == test_color);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Turbidity") {
		float test_turbidity = 100;
		test_physical->set_turbidity(test_turbidity);
		CHECK(test_physical->get_turbidity() == test_turbidity);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Sun Disk Scale") {
		float test_sun = 100;
		test_physical->set_sun_disk_scale(test_sun);
		CHECK(test_physical->get_sun_disk_scale() == test_sun);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Ground Color") {
		Color test_color = Color(0, 0, 0);
		test_physical->set_ground_color(test_color);
		CHECK(test_physical->get_ground_color() == test_color);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Energy Multipler") {
		float test_multipler = 100;
		test_physical->set_energy_multiplier(test_multipler);
		CHECK(test_physical->get_energy_multiplier() == test_multipler);
	}

	SUBCASE("[PhysicalSkyMaterial] Set and Get Use Debanding") {
		bool test_debanding = false;
		test_physical->set_use_debanding(test_debanding);
		CHECK(test_physical->get_use_debanding() == test_debanding);
	}

	SUBCASE("[PhysicalSkyMaterial] Set And Get Night Sky") {
		Ref<Texture2D> test_night;
		test_night.instantiate();
		test_physical->set_night_sky(test_night);
		CHECK(test_physical->get_night_sky() == test_night);
	}

	SUBCASE("[PhysicalSkyMaterial] Check Shader Mode") {
		CHECK(test_physical->get_shader_mode() == Shader::MODE_SKY);
	}
}
} //namespace TestSkyMaterial

#endif
