/**************************************************************************/
/*  test_decal.h                                                          */
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

#pragma once

#include "scene/3d/decal.h"

#include "tests/test_macros.h"

namespace TestDecal {
TEST_CASE("[SceneTree][Decal] Getters/Setters") {
	Decal *decal = memnew(Decal);
	AABB default_aabb{
		Vector3(-1.0f, -1.0f, -1.0f),
		Vector3(2.0f, 2.0f, 2.0f)
	};
	SUBCASE("Default values") {
		CHECK_MESSAGE(decal->get_aabb() == default_aabb, "get_aabb() returns the expected values for the default.");
	}

	SUBCASE("Set and Get Size") {
		Vector3 size = Vector3(2.0f, 3.0f, 4.0f);

		decal->set_size(size);
		AABB expected_aabb{
			Vector3(-1.0f, -1.5f, -2.0f),
			Vector3(2.0f, 3.0f, 4.0f)
		};
		CHECK_MESSAGE(decal->get_aabb() == expected_aabb, "get_aabb() returns the expected values after setting.");
		//default back the size in order to use the default_aabb for future assertions
		decal->set_size(Vector3(2.0f, 2.0f, 2.0f));
	}

	SUBCASE("Set and Get Texture") {
		Ref<Texture2D> albedo_texture;
		albedo_texture.instantiate();
		decal->set_texture(Decal::TEXTURE_ALBEDO, albedo_texture);
		CHECK_MESSAGE(decal->get_texture(Decal::TEXTURE_ALBEDO) == albedo_texture, "get_texture() returns the expected texture after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting texture.");
	}

	SUBCASE("Set and Get Emission Energy") {
		decal->set_emission_energy(2.0f);
		CHECK_MESSAGE(decal->get_emission_energy() == 2.0, "get_emission_energy() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting emission energy.");
	}

	SUBCASE("Set and Get Albedo Mix") {
		decal->set_albedo_mix(0.5);
		CHECK_MESSAGE(decal->get_albedo_mix() == 0.5f, "get_albedo_mix() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting albedo mix.");
	}

	SUBCASE("Set and Get Modulate") {
		Color new_color(0.5f, 0.5f, 0.5f, 1.0f);
		decal->set_modulate(new_color);
		CHECK_MESSAGE(decal->get_modulate() == new_color, "get_modulate() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting modulate.");
	}

	SUBCASE("Set and Get Upper Fade") {
		decal->set_upper_fade(0.4f);
		CHECK_MESSAGE(decal->get_upper_fade() == 0.4f, "get_upper_fade() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting upper fade.");
	}

	SUBCASE("Set and Get Lower Fade") {
		decal->set_lower_fade(0.2f);
		CHECK_MESSAGE(decal->get_lower_fade() == 0.2f, "get_lower_fade() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting lower fade.");
	}

	SUBCASE("Set and Get Normal Fade") {
		decal->set_normal_fade(0.1f);
		CHECK_MESSAGE(decal->get_normal_fade() == 0.1f, "get_normal_fade() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting normal fade.");
	}

	SUBCASE("Enable and Check Distance Fade") {
		decal->set_enable_distance_fade(true);
		CHECK_MESSAGE(decal->is_distance_fade_enabled() == true, "is_distance_fade_enabled() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after enabling distance fade.");
	}

	SUBCASE("Set and Get Distance Fade Begin") {
		decal->set_distance_fade_begin(50.0f);
		CHECK_MESSAGE(decal->get_distance_fade_begin() == 50.0f, "get_distance_fade_begin() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting distance fade begin.");
	}

	SUBCASE("Set and Get Distance Fade Length") {
		decal->set_distance_fade_length(15.0f);
		CHECK_MESSAGE(decal->get_distance_fade_length() == 15.0f, "get_distance_fade_length() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting distance fade length.");
	}

	SUBCASE("Set and Get Cull Mask") {
		decal->set_cull_mask(0xFF);
		CHECK_MESSAGE(decal->get_cull_mask() == 0xFF, "get_cull_mask() returns the expected value after setting.");
		CHECK_MESSAGE(default_aabb == decal->get_aabb(), "get_aabb() remains unchanged after setting cull mask.");
	}

	memdelete(decal);
}

} //namespace TestDecal
