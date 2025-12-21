/**************************************************************************/
/*  test_gltf_emissive.h                                                  */
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

#include "test_gltf.h"

#ifdef TOOLS_ENABLED

namespace TestGltf {

TEST_CASE("[SceneTree][Node] GLTF emissiveTexture without emissiveFactor uses white emission") {
	init("gltf_emissive_no_factor", "res://");

	Node *loaded = gltf_import("res://emissive_no_factor.gltf");
	CHECK_MESSAGE(loaded != nullptr, "Failed to load GLB.");

	MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(loaded->find_child("Cube", true, true));
	CHECK_MESSAGE(mesh != nullptr, "Mesh not found.");

	Ref<StandardMaterial3D> mat = mesh->get_active_material(0);
	CHECK_MESSAGE(mat.is_valid(), "Material not found.");

	// Emission should be enabled.
	CHECK(mat->get_feature(BaseMaterial3D::FEATURE_EMISSION));

	// Emission operator should be MULTIPLY per glTF spec.
	CHECK(mat->get_emission_operator() == BaseMaterial3D::EMISSION_OP_MULTIPLY);

	// Without emissiveFactor, emission color should be WHITE, not BLACK.
	Color c = mat->get_emission();
	CHECK_MESSAGE(c.r > 0.9f, "Emission red should be ~1.0 when emissiveFactor is absent.");
	CHECK_MESSAGE(c.g > 0.9f, "Emission green should be ~1.0 when emissiveFactor is absent.");
	CHECK_MESSAGE(c.b > 0.9f, "Emission blue should be ~1.0 when emissiveFactor is absent.");

	memdelete(loaded);
}

} // namespace TestGltf

#endif // TOOLS_ENABLED
