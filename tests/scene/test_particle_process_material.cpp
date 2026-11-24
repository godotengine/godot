/**************************************************************************/
/*  test_particle_process_material.cpp                                    */
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

TEST_FORCE_LINK(test_particle_process_material)

#include "scene/resources/particle_process_material.h"

class TestParticleProcessMaterialInternalsAccessor {
public:
	using MaterialKey = ParticleProcessMaterial::MaterialKey;

	static int shader_count() {
		MutexLock lock(ParticleProcessMaterial::shader_map_mutex);
		return ParticleProcessMaterial::shader_map.size();
	}
};

namespace TestParticleProcessMaterial {

using Accessor = TestParticleProcessMaterialInternalsAccessor;

TEST_CASE("[ParticleProcessMaterial] A MaterialKey copied over other bytes matches the original") {
	// The shader map and each material keep their own copies of keys, and hash()
	// and == read the whole word. A copy that carried only the bytes the fields
	// use kept stale bytes beside them, never matched the material's key again,
	// and so the shader map never freed that shader.
	Accessor::MaterialKey key;
	key.emission_shape = ParticleProcessMaterial::EMISSION_SHAPE_SPHERE;
	key.turbulence_enabled = 1;

	Accessor::MaterialKey copy;
	memset((void *)&copy, 0xff, sizeof(copy)); // Stale bytes, as in reused memory.
	copy = key;
	CHECK(copy == key);
	CHECK(Accessor::MaterialKey::hash(copy) == Accessor::MaterialKey::hash(key));
}

TEST_CASE("[SceneTree][ParticleProcessMaterial] Freed materials release their shaders") {
	const int shaders_before = Accessor::shader_count();
	{
		Ref<ParticleProcessMaterial> plain;
		plain.instantiate();
		Ref<ParticleProcessMaterial> changed;
		changed.instantiate();
		CHECK(plain->get_shader_rid().is_valid());
		CHECK_MESSAGE(changed->get_shader_rid() == plain->get_shader_rid(), "Materials with the same key should share a shader.");

		changed->set_emission_shape(ParticleProcessMaterial::EMISSION_SHAPE_SPHERE);
		changed->set_particle_flag(ParticleProcessMaterial::PARTICLE_FLAG_ROTATE_Y, true);
		changed->set_turbulence_enabled(true);
		CHECK(changed->get_shader_rid() != plain->get_shader_rid());
	}
	CHECK_MESSAGE(Accessor::shader_count() == shaders_before, "Freeing the last material using a shader should free that shader.");
}

} // namespace TestParticleProcessMaterial
