/**************************************************************************/
/*  test_orm_material_3d.h                                                */
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

namespace TestORMMaterial3D {

TEST_CASE("[Material][ORMMaterial3D] Constructor & default state") {
	Ref<ORMMaterial3D> mat;
	mat.instantiate();
	CHECK(mat.is_valid());
	CHECK_MESSAGE(mat->get_metallic() == doctest::Approx(0.0f), "Default metallic should be 0.0");
	CHECK_MESSAGE(mat->get_roughness() == doctest::Approx(1.0f), "Default roughness should be 1.0");
}

TEST_CASE("[Material][ORMMaterial3D] ORM texture assignment") {
	Ref<ORMMaterial3D> mat;
	mat.instantiate();
	Ref<Texture2D> tex;
	tex.instantiate();

	mat->set_texture(BaseMaterial3D::TEXTURE_ORM, tex);
	CHECK(mat->get_texture(BaseMaterial3D::TEXTURE_ORM) == tex);
}

TEST_CASE("[Material][ORMMaterial3D] Set AO texture and channel") {
	Ref<ORMMaterial3D> mat;
	mat.instantiate();
	Ref<Texture2D> tex;
	tex.instantiate();

	mat->set_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION, tex);
	mat->set_ao_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_GREEN);
	CHECK(mat->get_texture(BaseMaterial3D::TEXTURE_AMBIENT_OCCLUSION) == tex);
	CHECK(mat->get_ao_texture_channel() == BaseMaterial3D::TEXTURE_CHANNEL_GREEN);
}

TEST_CASE("[Material][ORMMaterial3D] Metallic and roughness channels") {
	Ref<ORMMaterial3D> mat;
	mat.instantiate();

	mat->set_metallic_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_RED);
	mat->set_roughness_texture_channel(BaseMaterial3D::TEXTURE_CHANNEL_BLUE);

	CHECK(mat->get_metallic_texture_channel() == BaseMaterial3D::TEXTURE_CHANNEL_RED);
	CHECK(mat->get_roughness_texture_channel() == BaseMaterial3D::TEXTURE_CHANNEL_BLUE);
}

TEST_CASE("[Material][ORMMaterial3D] Emission and refraction settings") {
	Ref<ORMMaterial3D> mat;
	mat.instantiate();

	Color emission(0.5, 0.7, 1.0);
	mat->set_emission(emission);
	CHECK(mat->get_emission() == emission);

	mat->set_refraction(0.8f);
	CHECK(mat->get_refraction() == doctest::Approx(0.8f));
}

TEST_CASE("[Material][ORMMaterial3D] AO and feature flag logic") {
	Ref<ORMMaterial3D> mat;
	mat.instantiate();

	mat->set_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION, true);
	CHECK(mat->get_feature(BaseMaterial3D::FEATURE_AMBIENT_OCCLUSION));

	mat->set_feature(BaseMaterial3D::FEATURE_HEIGHT_MAPPING, true);
	CHECK(mat->get_feature(BaseMaterial3D::FEATURE_HEIGHT_MAPPING));
}

TEST_CASE("[Material][ORMMaterial3D] Double assignment") {
	Ref<ORMMaterial3D> mat1;
	Ref<ORMMaterial3D> mat2;
	mat1.instantiate();
	mat2.instantiate();

	mat1->set_metallic(0.1f);
	mat2->set_metallic(0.7f);

	CHECK(mat1->get_metallic() == doctest::Approx(0.1f));
	CHECK(mat2->get_metallic() == doctest::Approx(0.7f));
}

} // namespace TestORMMaterial3D
