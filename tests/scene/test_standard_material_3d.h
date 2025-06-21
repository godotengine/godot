/**************************************************************************/
/*  test_standard_material_3d.h                                           */
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

#include "scene/resources/material.h"

#include "tests/test_macros.h"

namespace TestStandardMaterial3D {

TEST_CASE("[Material][StandardMaterial3D] Constructor & default state") {
	Ref<StandardMaterial3D> mat;
	mat.instantiate();

	CHECK(mat.is_valid());
	CHECK_MESSAGE(mat->get_roughness() == doctest::Approx(1.0f), "Default roughness should be 1.0");
	CHECK_MESSAGE(mat->get_metallic() == doctest::Approx(0.0f), "Default metallic should be 0.0");
}

TEST_CASE("[Material][StandardMaterial3D] Setter & Getter logic") {
	Ref<StandardMaterial3D> mat;
	mat.instantiate();

	mat->set_roughness(0.25f);
	CHECK(mat->get_roughness() == doctest::Approx(0.25f));

	mat->set_metallic(0.9f);
	CHECK(mat->get_metallic() == doctest::Approx(0.9f));

	Color albedo(0.3, 0.6, 0.9);
	mat->set_albedo(albedo);
	CHECK(mat->get_albedo() == albedo);
}

TEST_CASE("[Material][StandardMaterial3D] Texture assignment") {
	Ref<StandardMaterial3D> mat;
	mat.instantiate();
	Ref<Texture2D> tex;
	tex.instantiate();

	mat->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, tex);
	CHECK(mat->get_texture(BaseMaterial3D::TEXTURE_ALBEDO) == tex);
}

TEST_CASE("[Material][StandardMaterial3D] Transparency mode") {
	Ref<StandardMaterial3D> mat;
	mat.instantiate();

	mat->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
	CHECK(mat->get_transparency() == BaseMaterial3D::TRANSPARENCY_ALPHA);
}

TEST_CASE("[Material][StandardMaterial3D] Enum parameter setting") {
	Ref<StandardMaterial3D> mat;
	mat.instantiate();

	mat->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
	CHECK(mat->get_transparency() == BaseMaterial3D::TRANSPARENCY_ALPHA);

	mat->set_blend_mode(BaseMaterial3D::BLEND_MODE_ADD);
	CHECK(mat->get_blend_mode() == BaseMaterial3D::BLEND_MODE_ADD);
}

TEST_CASE("[Material][StandardMaterial3D] Double assignment") {
	Ref<StandardMaterial3D> mat1;
	Ref<StandardMaterial3D> mat2;
	mat1.instantiate();
	mat2.instantiate();

	mat1->set_roughness(0.1f);
	mat2->set_roughness(0.7f);

	SUBCASE("Swap roughness") {
		float tmp = mat1->get_roughness();
		mat1->set_roughness(mat2->get_roughness());
		mat2->set_roughness(tmp);

		CHECK(mat1->get_roughness() == doctest::Approx(0.7f));
		CHECK(mat2->get_roughness() == doctest::Approx(0.1f));
	}
}

} // namespace TestStandardMaterial3D
