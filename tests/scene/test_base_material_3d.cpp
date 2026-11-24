/**************************************************************************/
/*  test_base_material_3d.cpp                                             */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_base_material_3d)

#include "scene/resources/material.h"

class TestBaseMaterial3DInternalsAccessor {
public:
	using MaterialKey = BaseMaterial3D::MaterialKey;

	static int shader_count() {
		MutexLock lock(BaseMaterial3D::shader_map_mutex);
		return BaseMaterial3D::shader_map.size();
	}
};

namespace TestBaseMaterial3D {

using Accessor = TestBaseMaterial3DInternalsAccessor;

TEST_CASE("[BaseMaterial3D] A MaterialKey copied over other bytes matches the original") {
	// The shader map and each material keep their own copies of keys, and hash()
	// and == read every word. A copy has to carry the bits no field uses too, or
	// a copy over stale memory never matches the material's key again, and the
	// shader map never frees that shader.
	Accessor::MaterialKey key;
	key.transparency = BaseMaterial3D::TRANSPARENCY_ALPHA;
	key.cull_mode = BaseMaterial3D::CULL_DISABLED;
	key.feature_mask = 1 << BaseMaterial3D::FEATURE_EMISSION;
	key.flags = 1 << BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR;

	Accessor::MaterialKey copy;
	memset((void *)&copy, 0xff, sizeof(copy)); // Stale bytes, as in reused memory.
	copy = key;
	CHECK(copy == key);
	CHECK(Accessor::MaterialKey::hash(copy) == Accessor::MaterialKey::hash(key));
}

TEST_CASE("[SceneTree][BaseMaterial3D] Freed materials release their shaders") {
	const int shaders_before = Accessor::shader_count();
	{
		Ref<StandardMaterial3D> plain;
		plain.instantiate();
		Ref<StandardMaterial3D> changed;
		changed.instantiate();
		CHECK(plain->get_shader_rid().is_valid());
		CHECK_MESSAGE(changed->get_shader_rid() == plain->get_shader_rid(), "Materials with the same key should share a shader.");

		changed->set_transparency(BaseMaterial3D::TRANSPARENCY_ALPHA);
		changed->set_cull_mode(BaseMaterial3D::CULL_DISABLED);
		changed->set_feature(BaseMaterial3D::FEATURE_EMISSION, true);
		changed->set_flag(BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		CHECK(changed->get_shader_rid() != plain->get_shader_rid());
	}
	CHECK_MESSAGE(Accessor::shader_count() == shaders_before, "Freeing the last material using a shader should free that shader.");
}

} // namespace TestBaseMaterial3D
