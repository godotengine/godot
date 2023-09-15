/**************************************************************************/
/*  light_storage.cpp                                                     */
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

#ifdef GLES3_ENABLED

#include "light_storage.h"
#include "config.h"
#include "texture_storage.h"

using namespace GLES3;

LightStorage *LightStorage::singleton = nullptr;

LightStorage *LightStorage::get_singleton() {
	return singleton;
}

LightStorage::LightStorage() {
	singleton = this;
}

LightStorage::~LightStorage() {
	singleton = nullptr;
}

/* Light API */

void LightStorage::_light_initialize(RID p_light, RS::LightType p_type) {
	Light light;
	light.type = p_type;

	light.param[RS::LIGHT_PARAM_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_INDIRECT_ENERGY] = 1.0;
	light.param[RS::LIGHT_PARAM_VOLUMETRIC_FOG_ENERGY] = 1.0;
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
	light.param[RS::LIGHT_PARAM_SHADOW_OPACITY] = 1.0;
	light.param[RS::LIGHT_PARAM_SHADOW_BIAS] = 0.02;
	light.param[RS::LIGHT_PARAM_SHADOW_BLUR] = 0;
	light.param[RS::LIGHT_PARAM_SHADOW_PANCAKE_SIZE] = 20.0;
	light.param[RS::LIGHT_PARAM_TRANSMITTANCE_BIAS] = 0.05;
	light.param[RS::LIGHT_PARAM_INTENSITY] = p_type == RS::LIGHT_DIRECTIONAL ? 100000.0 : 1000.0;

	light_owner.initialize_rid(p_light, light);
}

RID LightStorage::directional_light_allocate() {
	return light_owner.allocate_rid();
}

void LightStorage::directional_light_initialize(RID p_rid) {
	_light_initialize(p_rid, RS::LIGHT_DIRECTIONAL);
}

RID LightStorage::omni_light_allocate() {
	return light_owner.allocate_rid();
}

void LightStorage::omni_light_initialize(RID p_rid) {
	_light_initialize(p_rid, RS::LIGHT_OMNI);
}

RID LightStorage::spot_light_allocate() {
	return light_owner.allocate_rid();
}

void LightStorage::spot_light_initialize(RID p_rid) {
	_light_initialize(p_rid, RS::LIGHT_SPOT);
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
	ERR_FAIL_NULL(light);

	light->color = p_color;
}

void LightStorage::light_set_param(RID p_light, RS::LightParam p_param, float p_value) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);
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
	ERR_FAIL_NULL(light);
	light->shadow = p_enabled;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_set_projector(RID p_light, RID p_texture) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

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
	ERR_FAIL_NULL(light);

	light->negative = p_enable;
}

void LightStorage::light_set_cull_mask(RID p_light, uint32_t p_mask) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->cull_mask = p_mask;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->distance_fade = p_enabled;
	light->distance_fade_begin = p_begin;
	light->distance_fade_shadow = p_shadow;
	light->distance_fade_length = p_length;
}

void LightStorage::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->reverse_cull = p_enabled;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->bake_mode = p_bake_mode;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->omni_shadow_mode = p_mode;

	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

RS::LightOmniShadowMode LightStorage::light_omni_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, RS::LIGHT_OMNI_SHADOW_CUBE);

	return light->omni_shadow_mode;
}

void LightStorage::light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->directional_shadow_mode = p_mode;
	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

void LightStorage::light_directional_set_blend_splits(RID p_light, bool p_enable) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->directional_blend_splits = p_enable;
	light->version++;
	light->dependency.changed_notify(Dependency::DEPENDENCY_CHANGED_LIGHT);
}

bool LightStorage::light_directional_get_blend_splits(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, false);

	return light->directional_blend_splits;
}

void LightStorage::light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) {
	Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL(light);

	light->directional_sky_mode = p_mode;
}

RS::LightDirectionalSkyMode LightStorage::light_directional_get_sky_mode(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY);

	return light->directional_sky_mode;
}

RS::LightDirectionalShadowMode LightStorage::light_directional_get_shadow_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL);

	return light->directional_shadow_mode;
}

RS::LightBakeMode LightStorage::light_get_bake_mode(RID p_light) {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, RS::LIGHT_BAKE_DISABLED);

	return light->bake_mode;
}

uint64_t LightStorage::light_get_version(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, 0);

	return light->version;
}

uint32_t LightStorage::light_get_cull_mask(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, 0);

	return light->cull_mask;
}

AABB LightStorage::light_get_aabb(RID p_light) const {
	const Light *light = light_owner.get_or_null(p_light);
	ERR_FAIL_NULL_V(light, AABB());

	switch (light->type) {
		case RS::LIGHT_SPOT: {
			float len = light->param[RS::LIGHT_PARAM_RANGE];
			float size = Math::tan(Math::deg_to_rad(light->param[RS::LIGHT_PARAM_SPOT_ANGLE])) * len;
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

/* LIGHT INSTANCE API */

RID LightStorage::light_instance_create(RID p_light) {
	RID li = light_instance_owner.make_rid(LightInstance());

	LightInstance *light_instance = light_instance_owner.get_or_null(li);

	light_instance->self = li;
	light_instance->light = p_light;
	light_instance->light_type = light_get_type(p_light);

	return li;
}

void LightStorage::light_instance_free(RID p_light_instance) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_NULL(light_instance);
	light_instance_owner.free(p_light_instance);
}

void LightStorage::light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_NULL(light_instance);

	light_instance->transform = p_transform;
}

void LightStorage::light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) {
	LightInstance *light_instance = light_instance_owner.get_or_null(p_light_instance);
	ERR_FAIL_NULL(light_instance);

	light_instance->aabb = p_aabb;
}

void LightStorage::light_instance_set_shadow_transform(RID p_light_instance, const Projection &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale, float p_range_begin, const Vector2 &p_uv_scale) {
}

void LightStorage::light_instance_mark_visible(RID p_light_instance) {
}

/* PROBE API */

RID LightStorage::reflection_probe_allocate() {
	return RID();
}

void LightStorage::reflection_probe_initialize(RID p_rid) {
}

void LightStorage::reflection_probe_free(RID p_rid) {
}

void LightStorage::reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {
}

void LightStorage::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
}

void LightStorage::reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) {
}

void LightStorage::reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) {
}

void LightStorage::reflection_probe_set_ambient_energy(RID p_probe, float p_energy) {
}

void LightStorage::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
}

void LightStorage::reflection_probe_set_size(RID p_probe, const Vector3 &p_size) {
}

void LightStorage::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
}

void LightStorage::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
}

void LightStorage::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
}

void LightStorage::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
}

void LightStorage::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
}

void LightStorage::reflection_probe_set_resolution(RID p_probe, int p_resolution) {
}

AABB LightStorage::reflection_probe_get_aabb(RID p_probe) const {
	return AABB();
}

RS::ReflectionProbeUpdateMode LightStorage::reflection_probe_get_update_mode(RID p_probe) const {
	return RenderingServer::REFLECTION_PROBE_UPDATE_ONCE;
}

uint32_t LightStorage::reflection_probe_get_cull_mask(RID p_probe) const {
	return 0;
}

Vector3 LightStorage::reflection_probe_get_size(RID p_probe) const {
	return Vector3();
}

Vector3 LightStorage::reflection_probe_get_origin_offset(RID p_probe) const {
	return Vector3();
}

float LightStorage::reflection_probe_get_origin_max_distance(RID p_probe) const {
	return 0.0;
}

bool LightStorage::reflection_probe_renders_shadows(RID p_probe) const {
	return false;
}

void LightStorage::reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) {
}

float LightStorage::reflection_probe_get_mesh_lod_threshold(RID p_probe) const {
	return 0.0;
}

/* REFLECTION ATLAS */

RID LightStorage::reflection_atlas_create() {
	return RID();
}

void LightStorage::reflection_atlas_free(RID p_ref_atlas) {
}

int LightStorage::reflection_atlas_get_size(RID p_ref_atlas) const {
	return 0;
}

void LightStorage::reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) {
}

/* REFLECTION PROBE INSTANCE */

RID LightStorage::reflection_probe_instance_create(RID p_probe) {
	return RID();
}

void LightStorage::reflection_probe_instance_free(RID p_instance) {
}

void LightStorage::reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) {
}

void LightStorage::reflection_probe_release_atlas_index(RID p_instance) {
}

bool LightStorage::reflection_probe_instance_needs_redraw(RID p_instance) {
	return false;
}

bool LightStorage::reflection_probe_instance_has_reflection(RID p_instance) {
	return false;
}

bool LightStorage::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {
	return false;
}

Ref<RenderSceneBuffers> LightStorage::reflection_probe_atlas_get_render_buffers(RID p_reflection_atlas) {
	return Ref<RenderSceneBuffers>();
}

bool LightStorage::reflection_probe_instance_postprocess_step(RID p_instance) {
	return true;
}

/* LIGHTMAP CAPTURE */

RID LightStorage::lightmap_allocate() {
	return lightmap_owner.allocate_rid();
}

void LightStorage::lightmap_initialize(RID p_rid) {
	lightmap_owner.initialize_rid(p_rid, Lightmap());
}

void LightStorage::lightmap_free(RID p_rid) {
	Lightmap *lightmap = lightmap_owner.get_or_null(p_rid);
	lightmap->dependency.deleted_notify(p_rid);
	lightmap_owner.free(p_rid);
}

void LightStorage::lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) {
}

void LightStorage::lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) {
}

void LightStorage::lightmap_set_probe_interior(RID p_lightmap, bool p_interior) {
}

void LightStorage::lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
}

void LightStorage::lightmap_set_baked_exposure_normalization(RID p_lightmap, float p_exposure) {
}

PackedVector3Array LightStorage::lightmap_get_probe_capture_points(RID p_lightmap) const {
	return PackedVector3Array();
}

PackedColorArray LightStorage::lightmap_get_probe_capture_sh(RID p_lightmap) const {
	return PackedColorArray();
}

PackedInt32Array LightStorage::lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const {
	return PackedInt32Array();
}

PackedInt32Array LightStorage::lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const {
	return PackedInt32Array();
}

AABB LightStorage::lightmap_get_aabb(RID p_lightmap) const {
	return AABB();
}

void LightStorage::lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) {
}

bool LightStorage::lightmap_is_interior(RID p_lightmap) const {
	return false;
}

void LightStorage::lightmap_set_probe_capture_update_speed(float p_speed) {
}

float LightStorage::lightmap_get_probe_capture_update_speed() const {
	return 0;
}

/* LIGHTMAP INSTANCE */

RID LightStorage::lightmap_instance_create(RID p_lightmap) {
	return RID();
}

void LightStorage::lightmap_instance_free(RID p_lightmap) {
}

void LightStorage::lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) {
}

/* SHADOW ATLAS API */

RID LightStorage::shadow_atlas_create() {
	return RID();
}

void LightStorage::shadow_atlas_free(RID p_atlas) {
}

void LightStorage::shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits) {
}

void LightStorage::shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {
}

bool LightStorage::shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) {
	return false;
}

void LightStorage::shadow_atlas_update(RID p_atlas) {
}

void LightStorage::directional_shadow_atlas_set_size(int p_size, bool p_16_bits) {
}

int LightStorage::get_directional_light_shadow_size(RID p_light_intance) {
	return 0;
}

void LightStorage::set_directional_shadow_count(int p_count) {
}

#endif // !GLES3_ENABLED
