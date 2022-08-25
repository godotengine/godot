/*************************************************************************/
/*  light_storage.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "light_storage.h"
#include "core/config/project_settings.h"
#include "texture_storage.h"

using namespace RendererRD;

LightStorage *LightStorage::singleton = nullptr;

LightStorage *LightStorage::get_singleton() {
	return singleton;
}

LightStorage::LightStorage() {
	singleton = this;

	TextureStorage *texture_storage = TextureStorage::get_singleton();

	using_lightmap_array = true; // high end
	if (using_lightmap_array) {
		uint64_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);

		if (textures_per_stage <= 256) {
			lightmap_textures.resize(32);
		} else {
			lightmap_textures.resize(1024);
		}

		for (int i = 0; i < lightmap_textures.size(); i++) {
			lightmap_textures.write[i] = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE);
		}
	}

	lightmap_probe_capture_update_speed = GLOBAL_GET("rendering/lightmapping/probe_capture/update_speed");
}

LightStorage::~LightStorage() {
	singleton = nullptr;
}

/* LIGHT */

void LightStorage::_light_initialize(RID p_light, RS::LightType p_type) {
	Light light;
	light.type = p_type;

	light.param[RS::LIGHT_PARAM_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_INDIRECT_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_SPECULAR] = 0.5;
	light.param[RS::LIGHT_PARAM_RANGE] = 1.0;
	light.param[RS::LIGHT_PARAM_SIZE] = 0.0;
	light.param[RS::LIGHT_PARAM_ATTENUATION] = 1.0;
	light.param[RS::LIGHT_PARAM_SPOT_ANGLE] = 45;
	light.param[RS::LIGHT_PARAM_SPOT_ATTENUATION] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_MAX_DISTANCE] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET] = 0.1;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET] = 0.3;
	light.param[RS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET] = 0.6;
	light.param[RS::LIGHT_PARAM_SHADOW_FADE_START] = 0.8;
	light.param[RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_BIAS] = 0.02;
	light.param[RS::LIGHT_PARAM_SHADOW_OPACITY] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_BLUR] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE] = 20.0;
	light.param[RS::LIGHT_PARAM_SHADOW_VOLUMETRIC_FOG_FADE] = 0.1;
	light.param[RS::LIGHT_PARAM_TRANSMITTANCE_BIAS] = 0.05;

	light_owner.initialize_rid(p_light, light);
}

RID LightStorage::directional_light_allocate() {
	return light_owner.allocate_rid();
}

void LightStorage::directional_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_DIRECTIONAL);
}

RID LightStorage::omni_light_allocate() {
	return light_owner.allocate_rid();
}

void LightStorage::omni_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_OMNI);
}

RID LightStorage::spot_light_allocate() {
	return light_owner.allocate_rid();
}

void LightStorage::spot_light_initialize(RID p_light) {
	_light_initialize(p_light, RS::LIGHT_SPOT);
}

void LightStorage::light_free(RID p_rid) {
	light_set_projector(p_rid, RID()); //clear projector

	// delete the texture
	Light *light = light_owner.get_or_null(p_rid);
	light->dependency.deleted_notify(p_rid);
	light_owner.free(p_rid);
}

void LightStorage::light_set_color(RID p_light, const Color &p_color) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->color = p_color;
}

void LightStorage::light_set_param(RID p_light, RS::LightParam p_param, float p_value) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);
	ERR_FAIL_INDEX(p_param, RS::LIGHT_PARAM_MAX);

	if (light->param[p_param] == p_value) {
		return;
	}

	switch (p_param) {
		case RS::LIGHT_PARAM_RANGE:
		case RS::LIGHT_PARAM_SPOT_ANGLE:
		case RS::LIGHT_PARAM_SHADOW_MAX_DISTANCE:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET:
		case RS::LIGHT_PARAM_SHADOW_NORMAL_BIAS:
		case RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE:
		case RS::LIGHT_PARAM_SHADOW_BIAS: {
			light->version++;
			light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
		} break;
		case RS::LIGHT_PARAM_SIZE: {
			if ((light->param[p_param] > CMP_EPSILON) != (p_value > CMP_EPSILON)) {
				//changing from no size to size and the opposite
				light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR);
			}
		} break;
		default: {
		}
	}

	light->param[p_param] = p_value;
}

void LightStorage::light_set_shadow(RID p_light, bool p_enabled) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);
	light->shadow = p_enabled;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_set_projector(RID p_light, RID p_texture) {
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	if (light->projector == p_texture) {
		return;
	}

	if (light->type != RS::LIGHT_DIRECTIONAL && light->projector.is_valid()) {
		texture_storage->texture_remove_from_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
	}

	light->projector = p_texture;

	if (light->type != RS::LIGHT_DIRECTIONAL) {
		if (light->projector.is_valid()) {
			texture_storage->texture_add_to_decal_atlas(light->projector, light->type == RS::LIGHT_OMNI);
		}
		light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR);
	}
}

void LightStorage::light_set_negative(RID p_light, bool p_enable) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->negative = p_enable;
}

void LightStorage::light_set_cull_mask(RID p_light, uint32_t p_mask) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->cull_mask = p_mask;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->distance_fade = p_enabled;
	light->distance_fade_begin = p_begin;
	light->distance_fade_shadow = p_shadow;
	light->distance_fade_length = p_length;
}

void LightStorage::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->reverse_cull = p_enabled;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->bake_mode = p_bake_mode;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->max_sdfgi_cascade = p_cascade;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->omni_shadow_mode = p_mode;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

RS::LightOmniShadowMode LightStorage::light_omni_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_OMNI_SHADOW_CUBE);

	return light->omni_shadow_mode;
}

void LightStorage::light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_shadow_mode = p_mode;
	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_directional_set_blend_splits(RID p_light, bool p_enable) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_blend_splits = p_enable;
	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

bool LightStorage::light_directional_get_blend_splits(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, false);

	return light->directional_blend_splits;
}

void LightStorage::light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND(!light);

	light->directional_sky_mode = p_mode;
}

RS::LightDirectionalSkyMode LightStorage::light_directional_get_sky_mode(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY);

	return light->directional_sky_mode;
}

RS::LightDirectionalShadowMode LightStorage::light_directional_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL);

	return light->directional_shadow_mode;
}

uint32_t LightStorage::light_get_max_sdfgi_cascade(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->max_sdfgi_cascade;
}

RS::LightBakeMode LightStorage::light_get_bake_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, RS::LIGHT_BAKE_DISABLED);

	return light->bake_mode;
}

uint64_t LightStorage::light_get_version(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, 0);

	return light->version;
}

AABB LightStorage::light_get_aabb(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_COND_V(!light, AABB());

	switch (light->type) {
		case RS::LIGHT_SPOT: {
			float len = light->param[RS::LIGHT_PARAM_RANGE];
			float size = Math::tan(Math::deg2rad(light->param[RS::LIGHT_PARAM_SPOT_ANGLE])) * len;
			return AABB(Vector3(-size, -size, -len), Vector3(size * 2, size * 2, len));
		};
		case RS::LIGHT_OMNI: {
			float r = light->param[RS::LIGHT_PARAM_RANGE];
			return AABB(-Vector3(r, r, r), Vector3(r, r, r) * 2);
		};
		case RS::LIGHT_DIRECTIONAL: {
			return AABB();
		};
	}

	ERR_FAIL_V(AABB());
}

Dependency *LightStorage::light_get_dependency(RID p_light) const {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, nullptr);

	return &light->dependency;
}

/* REFLECTION PROBE */

RID LightStorage::reflection_probe_allocate() {
	return reflection_probe_owner.allocate_rid();
}

void LightStorage::reflection_probe_initialize(RID p_reflection_probe) {
	reflection_probe_owner.initialize_rid(p_reflection_probe, ReflectionProbe());
}

void LightStorage::reflection_probe_free(RID p_rid) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_rid);
	reflection_probe->dependency.deleted_notify(p_rid);
	reflection_probe_owner.free(p_rid);
};

void LightStorage::reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->update_mode = p_mode;
	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void LightStorage::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->intensity = p_intensity;
}

void LightStorage::reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_mode = p_mode;
}

void LightStorage::reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color = p_color;
}

void LightStorage::reflection_probe_set_ambient_energy(RID p_probe, float p_energy) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->ambient_color_energy = p_energy;
}

void LightStorage::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->max_distance = p_distance;

	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void LightStorage::reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	if (reflection_probe->extents == p_extents) {
		return;
	}
	reflection_probe->extents = p_extents;
	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void LightStorage::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->origin_offset = p_offset;
	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void LightStorage::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->interior = p_enable;
	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void LightStorage::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->box_projection = p_enable;
}

void LightStorage::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->enable_shadows = p_enable;
	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void LightStorage::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->cull_mask = p_layers;
	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

void LightStorage::reflection_probe_set_resolution(RID p_probe, int p_resolution) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);
	ERR_FAIL_COND(p_resolution < 32);

	reflection_probe->resolution = p_resolution;
}

void LightStorage::reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND(!reflection_probe);

	reflection_probe->mesh_lod_threshold = p_ratio;

	reflection_probe->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_REFLECTION_PROBE);
}

AABB LightStorage::reflection_probe_get_aabb(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, AABB());

	AABB aabb;
	aabb.position = -reflection_probe->extents;
	aabb.size = reflection_probe->extents * 2.0;

	return aabb;
}

RS::ReflectionProbeUpdateMode LightStorage::reflection_probe_get_update_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_UPDATE_ALWAYS);

	return reflection_probe->update_mode;
}

uint32_t LightStorage::reflection_probe_get_cull_mask(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->cull_mask;
}

Vector3 LightStorage::reflection_probe_get_extents(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->extents;
}

Vector3 LightStorage::reflection_probe_get_origin_offset(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Vector3());

	return reflection_probe->origin_offset;
}

bool LightStorage::reflection_probe_renders_shadows(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->enable_shadows;
}

float LightStorage::reflection_probe_get_origin_max_distance(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->max_distance;
}

float LightStorage::reflection_probe_get_mesh_lod_threshold(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->mesh_lod_threshold;
}

int LightStorage::reflection_probe_get_resolution(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->resolution;
}

float LightStorage::reflection_probe_get_intensity(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->intensity;
}

bool LightStorage::reflection_probe_is_interior(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->interior;
}

bool LightStorage::reflection_probe_is_box_projection(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, false);

	return reflection_probe->box_projection;
}

RS::ReflectionProbeAmbientMode LightStorage::reflection_probe_get_ambient_mode(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, RS::REFLECTION_PROBE_AMBIENT_DISABLED);
	return reflection_probe->ambient_mode;
}

Color LightStorage::reflection_probe_get_ambient_color(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, Color());

	return reflection_probe->ambient_color;
}
float LightStorage::reflection_probe_get_ambient_color_energy(RID p_probe) const {
	const ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_COND_V(!reflection_probe, 0);

	return reflection_probe->ambient_color_energy;
}

Dependency *LightStorage::reflection_probe_get_dependency(RID p_probe) const {
	ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_probe);
	ERR_FAIL_NULL_V(reflection_probe, nullptr);

	return &reflection_probe->dependency;
}

/* LIGHTMAP API */

RID LightStorage::lightmap_allocate() {
	return lightmap_owner.allocate_rid();
}

void LightStorage::lightmap_initialize(RID p_lightmap) {
	lightmap_owner.initialize_rid(p_lightmap, Lightmap());
}

void LightStorage::lightmap_free(RID p_rid) {
	lightmap_set_textures(p_rid, RID(), false);
	Lightmap *lightmap = lightmap_owner.get_or_null(p_rid);
	lightmap->dependency.deleted_notify(p_rid);
	lightmap_owner.free(p_rid);
}

void LightStorage::lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) {
	RendererRD::TextureStorage *texture_storage = RendererRD::TextureStorage::get_singleton();

	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	lightmap_array_version++;

	//erase lightmap users
	if (lm->light_texture.is_valid()) {
		RendererRD::TextureStorage::Texture *t = RendererRD::TextureStorage::get_singleton()->get_texture(lm->light_texture);
		if (t) {
			t->lightmap_users.erase(p_lightmap);
		}
	}

	RendererRD::TextureStorage::Texture *t = RendererRD::TextureStorage::get_singleton()->get_texture(p_light);
	lm->light_texture = p_light;
	lm->uses_spherical_harmonics = p_uses_spherical_haromics;

	RID default_2d_array = texture_storage->texture_rd_get_default(RendererRD::TextureStorage::DEFAULT_RD_TEXTURE_2D_ARRAY_WHITE);
	if (!t) {
		if (using_lightmap_array) {
			if (lm->array_index >= 0) {
				lightmap_textures.write[lm->array_index] = default_2d_array;
				lm->array_index = -1;
			}
		}

		return;
	}

	t->lightmap_users.insert(p_lightmap);

	if (using_lightmap_array) {
		if (lm->array_index < 0) {
			//not in array, try to put in array
			for (int i = 0; i < lightmap_textures.size(); i++) {
				if (lightmap_textures[i] == default_2d_array) {
					lm->array_index = i;
					break;
				}
			}
		}
		ERR_FAIL_COND_MSG(lm->array_index < 0, "Maximum amount of lightmaps in use (" + itos(lightmap_textures.size()) + ") has been exceeded, lightmap will nod display properly.");

		lightmap_textures.write[lm->array_index] = t->rd_texture;
	}
}

void LightStorage::lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->bounds = p_bounds;
}

void LightStorage::lightmap_set_probe_interior(RID p_lightmap, bool p_interior) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);
	lm->interior = p_interior;
}

void LightStorage::lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	if (p_points.size()) {
		ERR_FAIL_COND(p_points.size() * 9 != p_point_sh.size());
		ERR_FAIL_COND((p_tetrahedra.size() % 4) != 0);
		ERR_FAIL_COND((p_bsp_tree.size() % 6) != 0);
	}

	lm->points = p_points;
	lm->bsp_tree = p_bsp_tree;
	lm->point_sh = p_point_sh;
	lm->tetrahedra = p_tetrahedra;
}

PackedVector3Array LightStorage::lightmap_get_probe_capture_points(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedVector3Array());

	return lm->points;
}

PackedColorArray LightStorage::lightmap_get_probe_capture_sh(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedColorArray());
	return lm->point_sh;
}

PackedInt32Array LightStorage::lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->tetrahedra;
}

PackedInt32Array LightStorage::lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, PackedInt32Array());
	return lm->bsp_tree;
}

void LightStorage::lightmap_set_probe_capture_update_speed(float p_speed) {
	lightmap_probe_capture_update_speed = p_speed;
}

Dependency *LightStorage::lightmap_get_dependency(RID p_lightmap) const {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_NULL_V(lm, nullptr);

	return &lm->dependency;
}

void LightStorage::lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) {
	Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND(!lm);

	for (int i = 0; i < 9; i++) {
		r_sh[i] = Color(0, 0, 0, 0);
	}

	if (!lm->points.size() || !lm->bsp_tree.size() || !lm->tetrahedra.size()) {
		return;
	}

	static_assert(sizeof(Lightmap::BSP) == 24);

	const Lightmap::BSP *bsp = (const Lightmap::BSP *)lm->bsp_tree.ptr();
	int32_t node = 0;
	while (node >= 0) {
		if (Plane(bsp[node].plane[0], bsp[node].plane[1], bsp[node].plane[2], bsp[node].plane[3]).is_point_over(p_point)) {
#ifdef DEBUG_ENABLED
			ERR_FAIL_COND(bsp[node].over >= 0 && bsp[node].over < node);
#endif

			node = bsp[node].over;
		} else {
#ifdef DEBUG_ENABLED
			ERR_FAIL_COND(bsp[node].under >= 0 && bsp[node].under < node);
#endif
			node = bsp[node].under;
		}
	}

	if (node == Lightmap::BSP::EMPTY_LEAF) {
		return; //nothing could be done
	}

	node = ABS(node) - 1;

	uint32_t *tetrahedron = (uint32_t *)&lm->tetrahedra[node * 4];
	Vector3 points[4] = { lm->points[tetrahedron[0]], lm->points[tetrahedron[1]], lm->points[tetrahedron[2]], lm->points[tetrahedron[3]] };
	const Color *sh_colors[4]{ &lm->point_sh[tetrahedron[0] * 9], &lm->point_sh[tetrahedron[1] * 9], &lm->point_sh[tetrahedron[2] * 9], &lm->point_sh[tetrahedron[3] * 9] };
	Color barycentric = Geometry3D::tetrahedron_get_barycentric_coords(points[0], points[1], points[2], points[3], p_point);

	for (int i = 0; i < 4; i++) {
		float c = CLAMP(barycentric[i], 0.0, 1.0);
		for (int j = 0; j < 9; j++) {
			r_sh[j] += sh_colors[i][j] * c;
		}
	}
}

bool LightStorage::lightmap_is_interior(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, false);
	return lm->interior;
}

AABB LightStorage::lightmap_get_aabb(RID p_lightmap) const {
	const Lightmap *lm = lightmap_owner.get_or_null(p_lightmap);
	ERR_FAIL_COND_V(!lm, AABB());
	return lm->bounds;
}
