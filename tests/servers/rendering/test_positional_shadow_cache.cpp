/**************************************************************************/
/*  test_positional_shadow_cache.cpp                                      */
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

TEST_FORCE_LINK(test_positional_shadow_cache)

#ifndef _3D_DISABLED

#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "servers/rendering/renderer_scene_cull.h"

namespace TestPositionalShadowCache {

TEST_CASE("[SceneTree][GeometryInstance3D] Shadow mobility") {
	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);

	CHECK_MESSAGE(mesh_instance->get_shadow_mobility() == GeometryInstance3D::SHADOW_MOBILITY_DYNAMIC, "Geometry is a dynamic shadow caster by default.");

	mesh_instance->set_shadow_mobility(GeometryInstance3D::SHADOW_MOBILITY_STATIC);
	CHECK(mesh_instance->get_shadow_mobility() == GeometryInstance3D::SHADOW_MOBILITY_STATIC);
	CHECK(int(mesh_instance->get("shadow_mobility")) == int(GeometryInstance3D::SHADOW_MOBILITY_STATIC));

	mesh_instance->set("shadow_mobility", GeometryInstance3D::SHADOW_MOBILITY_DYNAMIC);
	CHECK(mesh_instance->get_shadow_mobility() == GeometryInstance3D::SHADOW_MOBILITY_DYNAMIC);

	ERR_PRINT_OFF;
	mesh_instance->set_shadow_mobility(GeometryInstance3D::ShadowMobility(2));
	ERR_PRINT_ON;
	CHECK_MESSAGE(mesh_instance->get_shadow_mobility() == GeometryInstance3D::SHADOW_MOBILITY_DYNAMIC, "Invalid values are rejected.");

	CHECK(int(RSE::SHADOW_MOBILITY_DYNAMIC) == int(GeometryInstance3D::SHADOW_MOBILITY_DYNAMIC));
	CHECK(int(RSE::SHADOW_MOBILITY_STATIC) == int(GeometryInstance3D::SHADOW_MOBILITY_STATIC));

	memdelete(mesh_instance);
}

TEST_CASE("[SceneTree][Light3D] Shadow cache properties") {
	Light3D *lights[2] = { memnew(OmniLight3D), memnew(SpotLight3D) };

	for (Light3D *light : lights) {
		CHECK_FALSE_MESSAGE(light->is_shadow_cache_enabled(), "Shadow caching is disabled by default.");
		CHECK_MESSAGE(light->get_shadow_dynamic_update_interval() == 1, "Shadows are updated every frame by default.");

		light->set_shadow_cache_enabled(true);
		CHECK(light->is_shadow_cache_enabled());
		CHECK(bool(light->get("shadow_cache_enabled")));
		light->set("shadow_cache_enabled", false);
		CHECK_FALSE(light->is_shadow_cache_enabled());

		light->set_shadow_dynamic_update_interval(4);
		CHECK(light->get_shadow_dynamic_update_interval() == 4);
		CHECK(int(light->get("shadow_dynamic_update_interval")) == 4);

		ERR_PRINT_OFF;
		light->set_shadow_dynamic_update_interval(0);
		ERR_PRINT_ON;
		CHECK_MESSAGE(light->get_shadow_dynamic_update_interval() == 4, "Intervals lower than 1 frame are rejected.");

		memdelete(light);
	}
}

// Minimal geometry instance for the shadow caster tests. Only its address is used, it's never dereferenced.
static RenderGeometryInstance *_fake_geometry_instance(uintptr_t p_index) {
	return reinterpret_cast<RenderGeometryInstance *>(uintptr_t(0x1000) + p_index * 0x10);
}

static RendererSceneCull::Instance *_make_caster(uintptr_t p_index, bool p_static, uint32_t p_layer_mask = 1) {
	RendererSceneCull::Instance *instance = memnew(RendererSceneCull::Instance);
	instance->base_type = RSE::INSTANCE_MESH;
	instance->layer_mask = p_layer_mask;
	instance->shadow_mobility_static = p_static;

	RendererSceneCull::InstanceGeometryData *geom = memnew(RendererSceneCull::InstanceGeometryData);
	geom->geometry_instance = _fake_geometry_instance(p_index);
	geom->can_cast_shadows = true;
	geom->material_is_animated = false;
	instance->base_data = geom;
	return instance;
}

static bool _has_geometry_instance(const PagedArray<RenderGeometryInstance *> &p_array, RenderGeometryInstance *p_instance) {
	for (uint64_t i = 0; i < p_array.size(); i++) {
		if (p_array[i] == p_instance) {
			return true;
		}
	}
	return false;
}

static bool _has_instance(const PagedArray<RendererSceneCull::Instance *> &p_array, RendererSceneCull::Instance *p_instance) {
	for (uint64_t i = 0; i < p_array.size(); i++) {
		if (p_array[i] == p_instance) {
			return true;
		}
	}
	return false;
}

TEST_CASE("[RendererSceneCull] Static shadow casters are split from dynamic ones") {
	PagedArrayPool<RendererSceneCull::Instance *> instance_pool;
	PagedArrayPool<RenderGeometryInstance *> geometry_pool;

	RendererSceneCull::Instance *dynamic_a = _make_caster(0, false);
	RendererSceneCull::Instance *static_a = _make_caster(1, true);
	RendererSceneCull::Instance *dynamic_b = _make_caster(2, false);
	RendererSceneCull::Instance *static_b = _make_caster(3, true);
	RendererSceneCull::Instance *static_animated = _make_caster(4, true);
	static_cast<RendererSceneCull::InstanceGeometryData *>(static_animated->base_data)->material_is_animated = true;
	RendererSceneCull::Instance *static_hidden = _make_caster(5, true);
	static_hidden->visible = false;
	RendererSceneCull::Instance *static_other_layer = _make_caster(6, true, 2);
	RendererSceneCull::Instance *all[] = { dynamic_a, static_a, dynamic_b, static_b, static_animated, static_hidden, static_other_layer };

	PagedArray<RendererSceneCull::Instance *> casters;
	casters.set_page_pool(&instance_pool);
	RendererSceneRender::RenderShadowData shadow_data;
	shadow_data.static_instances.set_page_pool(&geometry_pool);

	SUBCASE("Updating the static cache") {
		for (RendererSceneCull::Instance *instance : all) {
			casters.push_back(instance);
		}
		RendererSceneCull::_light_instance_extract_static_shadow_casters(casters, shadow_data, true, 1);

		CHECK(shadow_data.use_static_cache);
		CHECK(shadow_data.update_static_cache);

		// Dynamic casters, and static ones with animated materials, stay in the list drawn on every update.
		CHECK(casters.size() == 3);
		CHECK(_has_instance(casters, dynamic_a));
		CHECK(_has_instance(casters, dynamic_b));
		CHECK_MESSAGE(_has_instance(casters, static_animated), "Static casters with animated materials are drawn as dynamic casters.");

		// Only the static casters that can be seen by the light are drawn into the static cache.
		CHECK(shadow_data.static_instances.size() == 2);
		CHECK(_has_geometry_instance(shadow_data.static_instances, _fake_geometry_instance(1)));
		CHECK(_has_geometry_instance(shadow_data.static_instances, _fake_geometry_instance(3)));
	}

	SUBCASE("Reusing the static cache") {
		for (RendererSceneCull::Instance *instance : all) {
			casters.push_back(instance);
		}
		RendererSceneCull::_light_instance_extract_static_shadow_casters(casters, shadow_data, false, 1);

		CHECK(shadow_data.use_static_cache);
		CHECK_FALSE(shadow_data.update_static_cache);
		CHECK(casters.size() == 3);
		CHECK_MESSAGE(shadow_data.static_instances.size() == 0, "Static casters aren't drawn when the static cache is valid.");
	}

	SUBCASE("Only dynamic casters") {
		casters.push_back(dynamic_a);
		casters.push_back(dynamic_b);
		RendererSceneCull::_light_instance_extract_static_shadow_casters(casters, shadow_data, true, 1);

		CHECK(casters.size() == 2);
		CHECK(shadow_data.static_instances.size() == 0);
	}

	casters.reset();
	shadow_data.static_instances.reset();
	for (RendererSceneCull::Instance *instance : all) {
		memdelete(instance);
	}
}

TEST_CASE("[RendererSceneCull] Static shadow cache invalidation") {
	RendererSceneCull::InstanceLightData light;
	RendererSceneCull::Instance dynamic_caster;
	RendererSceneCull::Instance static_caster;
	static_caster.shadow_mobility_static = true;

	CHECK_MESSAGE(light.is_static_shadow_dirty(), "The static cache of a new light must be drawn.");

	// Pretend the static cache was drawn and the shadow was updated.
	light.static_shadow_version_scheduled = light.static_shadow_version;
	while (light.is_shadow_dirty()) {
		light.decrement_shadow_dirty();
	}
	CHECK_FALSE(light.is_static_shadow_dirty());

	light.make_shadow_dirty_for_caster(&dynamic_caster);
	CHECK(light.is_shadow_dirty());
	CHECK_FALSE_MESSAGE(light.is_static_shadow_dirty(), "Dynamic casters don't invalidate the static cache.");

	light.make_shadow_dirty_for_caster(&static_caster);
	CHECK(light.is_shadow_dirty());
	CHECK_MESSAGE(light.is_static_shadow_dirty(), "Static casters invalidate the static cache.");

	light.static_shadow_version_scheduled = light.static_shadow_version;
	CHECK_FALSE(light.is_static_shadow_dirty());

	light.make_static_shadow_dirty();
	CHECK(light.is_shadow_dirty());
	CHECK_MESSAGE(light.is_static_shadow_dirty(), "Changes to the light invalidate the static cache.");
}

} // namespace TestPositionalShadowCache

#endif // _3D_DISABLED
