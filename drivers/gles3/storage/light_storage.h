/**************************************************************************/
/*  light_storage.h                                                       */
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

#ifndef LIGHT_STORAGE_GLES3_H
#define LIGHT_STORAGE_GLES3_H

#ifdef GLES3_ENABLED

#include "platform_gl.h"

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "drivers/gles3/storage/texture_storage.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/storage/light_storage.h"
#include "servers/rendering/storage/utilities.h"

namespace GLES3 {

/* LIGHT */

struct Light {
	RS::LightType type;
	float param[RS::LIGHT_PARAM_MAX];
	Color color = Color(1, 1, 1, 1);
	RID projector;
	bool shadow = false;
	bool negative = false;
	bool reverse_cull = false;
	RS::LightBakeMode bake_mode = RS::LIGHT_BAKE_DYNAMIC;
	uint32_t max_sdfgi_cascade = 2;
	uint32_t cull_mask = 0xFFFFFFFF;
	bool distance_fade = false;
	real_t distance_fade_begin = 40.0;
	real_t distance_fade_shadow = 50.0;
	real_t distance_fade_length = 10.0;
	RS::LightOmniShadowMode omni_shadow_mode = RS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID;
	RS::LightDirectionalShadowMode directional_shadow_mode = RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL;
	bool directional_blend_splits = false;
	RS::LightDirectionalSkyMode directional_sky_mode = RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY;
	uint64_t version = 0;

	Dependency dependency;
};

/* Light instance */
struct LightInstance {
	RS::LightType light_type = RS::LIGHT_DIRECTIONAL;

	AABB aabb;
	RID self;
	RID light;
	Transform3D transform;

	Vector3 light_vector;
	Vector3 spot_vector;
	float linear_att = 0.0;

	uint64_t shadow_pass = 0;
	uint64_t last_scene_pass = 0;
	uint64_t last_scene_shadow_pass = 0;
	uint64_t last_pass = 0;
	uint32_t cull_mask = 0;
	uint32_t light_directional_index = 0;

	Rect2 directional_rect;

	uint32_t gl_id = -1;

	LightInstance() {}
};

/* REFLECTION PROBE */

struct ReflectionProbe {
	RS::ReflectionProbeUpdateMode update_mode = RS::REFLECTION_PROBE_UPDATE_ONCE;
	int resolution = 256;
	float intensity = 1.0;
	RS::ReflectionProbeAmbientMode ambient_mode = RS::REFLECTION_PROBE_AMBIENT_ENVIRONMENT;
	Color ambient_color;
	float ambient_color_energy = 1.0;
	float max_distance = 0;
	Vector3 size = Vector3(20, 20, 20);
	Vector3 origin_offset;
	bool interior = false;
	bool box_projection = false;
	bool enable_shadows = false;
	uint32_t cull_mask = (1 << 20) - 1;
	float mesh_lod_threshold = 0.01;
	float baked_exposure = 1.0;

	Dependency dependency;
};

/* LIGHTMAP */

struct Lightmap {
	RID light_texture;
	bool uses_spherical_harmonics = false;
	bool interior = false;
	AABB bounds = AABB(Vector3(), Vector3(1, 1, 1));
	float baked_exposure = 1.0;
	int32_t array_index = -1; //unassigned
	PackedVector3Array points;
	PackedColorArray point_sh;
	PackedInt32Array tetrahedra;
	PackedInt32Array bsp_tree;

	struct BSP {
		static const int32_t EMPTY_LEAF = INT32_MIN;
		float plane[4];
		int32_t over = EMPTY_LEAF, under = EMPTY_LEAF;
	};

	Dependency dependency;
};

class LightStorage : public RendererLightStorage {
private:
	static LightStorage *singleton;

	/* LIGHT */
	mutable RID_Owner<Light, true> light_owner;

	/* Light instance */
	mutable RID_Owner<LightInstance> light_instance_owner;

	/* REFLECTION PROBE */
	mutable RID_Owner<ReflectionProbe, true> reflection_probe_owner;

	/* LIGHTMAP */

	Vector<RID> lightmap_textures;

	mutable RID_Owner<Lightmap, true> lightmap_owner;

public:
	static LightStorage *get_singleton();

	LightStorage();
	virtual ~LightStorage();

	/* Light API */

	Light *get_light(RID p_rid) { return light_owner.get_or_null(p_rid); };
	bool owns_light(RID p_rid) { return light_owner.owns(p_rid); };

	void _light_initialize(RID p_rid, RS::LightType p_type);

	virtual RID directional_light_allocate() override;
	virtual void directional_light_initialize(RID p_rid) override;
	virtual RID omni_light_allocate() override;
	virtual void omni_light_initialize(RID p_rid) override;
	virtual RID spot_light_allocate() override;
	virtual void spot_light_initialize(RID p_rid) override;

	virtual void light_free(RID p_rid) override;

	virtual void light_set_color(RID p_light, const Color &p_color) override;
	virtual void light_set_param(RID p_light, RS::LightParam p_param, float p_value) override;
	virtual void light_set_shadow(RID p_light, bool p_enabled) override;
	virtual void light_set_projector(RID p_light, RID p_texture) override;
	virtual void light_set_negative(RID p_light, bool p_enable) override;
	virtual void light_set_cull_mask(RID p_light, uint32_t p_mask) override;
	virtual void light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) override;
	virtual void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) override;
	virtual void light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) override;
	virtual void light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) override {}

	virtual void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) override;

	virtual void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) override;
	virtual void light_directional_set_blend_splits(RID p_light, bool p_enable) override;
	virtual bool light_directional_get_blend_splits(RID p_light) const override;
	virtual void light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) override;
	virtual RS::LightDirectionalSkyMode light_directional_get_sky_mode(RID p_light) const override;

	virtual RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) override;
	virtual RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) override;
	virtual RS::LightType light_get_type(RID p_light) const override {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, RS::LIGHT_DIRECTIONAL);

		return light->type;
	}
	virtual AABB light_get_aabb(RID p_light) const override;

	virtual float light_get_param(RID p_light, RS::LightParam p_param) override {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, 0);

		return light->param[p_param];
	}

	_FORCE_INLINE_ RID light_get_projector(RID p_light) {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, RID());

		return light->projector;
	}

	virtual Color light_get_color(RID p_light) override {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, Color());

		return light->color;
	}

	_FORCE_INLINE_ bool light_is_distance_fade_enabled(RID p_light) {
		const Light *light = light_owner.get_or_null(p_light);
		return light->distance_fade;
	}

	_FORCE_INLINE_ float light_get_distance_fade_begin(RID p_light) {
		const Light *light = light_owner.get_or_null(p_light);
		return light->distance_fade_begin;
	}

	_FORCE_INLINE_ float light_get_distance_fade_shadow(RID p_light) {
		const Light *light = light_owner.get_or_null(p_light);
		return light->distance_fade_shadow;
	}

	_FORCE_INLINE_ float light_get_distance_fade_length(RID p_light) {
		const Light *light = light_owner.get_or_null(p_light);
		return light->distance_fade_length;
	}

	virtual bool light_has_shadow(RID p_light) const override {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, RS::LIGHT_DIRECTIONAL);

		return light->shadow;
	}

	virtual bool light_has_projector(RID p_light) const override {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, RS::LIGHT_DIRECTIONAL);

		return TextureStorage::get_singleton()->owns_texture(light->projector);
	}

	_FORCE_INLINE_ bool light_is_negative(RID p_light) const {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, RS::LIGHT_DIRECTIONAL);

		return light->negative;
	}

	_FORCE_INLINE_ float light_get_transmittance_bias(RID p_light) const {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, 0.0);

		return light->param[RS::LIGHT_PARAM_TRANSMITTANCE_BIAS];
	}

	virtual bool light_get_reverse_cull_face_mode(RID p_light) const override {
		const Light *light = light_owner.get_or_null(p_light);
		ERR_FAIL_NULL_V(light, false);

		return light->reverse_cull;
	}

	virtual RS::LightBakeMode light_get_bake_mode(RID p_light) override;
	virtual uint32_t light_get_max_sdfgi_cascade(RID p_light) override { return 0; }
	virtual uint64_t light_get_version(RID p_light) const override;
	virtual uint32_t light_get_cull_mask(RID p_light) const override;

	/* LIGHT INSTANCE API */

	LightInstance *get_light_instance(RID p_rid) { return light_instance_owner.get_or_null(p_rid); };
	bool owns_light_instance(RID p_rid) { return light_instance_owner.owns(p_rid); };

	virtual RID light_instance_create(RID p_light) override;
	virtual void light_instance_free(RID p_light_instance) override;

	virtual void light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) override;
	virtual void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) override;
	virtual void light_instance_set_shadow_transform(RID p_light_instance, const Projection &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2()) override;
	virtual void light_instance_mark_visible(RID p_light_instance) override;

	_FORCE_INLINE_ RS::LightType light_instance_get_type(RID p_light_instance) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->light_type;
	}
	_FORCE_INLINE_ uint32_t light_instance_get_gl_id(RID p_light_instance) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->gl_id;
	}

	/* PROBE API */

	virtual RID reflection_probe_allocate() override;
	virtual void reflection_probe_initialize(RID p_rid) override;
	virtual void reflection_probe_free(RID p_rid) override;

	virtual void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) override;
	virtual void reflection_probe_set_intensity(RID p_probe, float p_intensity) override;
	virtual void reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) override;
	virtual void reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) override;
	virtual void reflection_probe_set_ambient_energy(RID p_probe, float p_energy) override;
	virtual void reflection_probe_set_max_distance(RID p_probe, float p_distance) override;
	virtual void reflection_probe_set_size(RID p_probe, const Vector3 &p_size) override;
	virtual void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) override;
	virtual void reflection_probe_set_as_interior(RID p_probe, bool p_enable) override;
	virtual void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) override;
	virtual void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) override;
	virtual void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) override;
	virtual void reflection_probe_set_resolution(RID p_probe, int p_resolution) override;
	virtual void reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) override;
	virtual float reflection_probe_get_mesh_lod_threshold(RID p_probe) const override;

	virtual AABB reflection_probe_get_aabb(RID p_probe) const override;
	virtual RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const override;
	virtual uint32_t reflection_probe_get_cull_mask(RID p_probe) const override;
	virtual Vector3 reflection_probe_get_size(RID p_probe) const override;
	virtual Vector3 reflection_probe_get_origin_offset(RID p_probe) const override;
	virtual float reflection_probe_get_origin_max_distance(RID p_probe) const override;
	virtual bool reflection_probe_renders_shadows(RID p_probe) const override;

	/* REFLECTION ATLAS */

	virtual RID reflection_atlas_create() override;
	virtual void reflection_atlas_free(RID p_ref_atlas) override;
	virtual int reflection_atlas_get_size(RID p_ref_atlas) const override;
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) override;

	/* REFLECTION PROBE INSTANCE */

	virtual RID reflection_probe_instance_create(RID p_probe) override;
	virtual void reflection_probe_instance_free(RID p_instance) override;
	virtual void reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) override;
	virtual void reflection_probe_release_atlas_index(RID p_instance) override;
	virtual bool reflection_probe_instance_needs_redraw(RID p_instance) override;
	virtual bool reflection_probe_instance_has_reflection(RID p_instance) override;
	virtual bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) override;
	virtual Ref<RenderSceneBuffers> reflection_probe_atlas_get_render_buffers(RID p_reflection_atlas) override;
	virtual bool reflection_probe_instance_postprocess_step(RID p_instance) override;

	/* LIGHTMAP CAPTURE */

	Lightmap *get_lightmap(RID p_rid) { return lightmap_owner.get_or_null(p_rid); };
	bool owns_lightmap(RID p_rid) { return lightmap_owner.owns(p_rid); };

	virtual RID lightmap_allocate() override;
	virtual void lightmap_initialize(RID p_rid) override;
	virtual void lightmap_free(RID p_rid) override;

	virtual void lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) override;
	virtual void lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) override;
	virtual void lightmap_set_probe_interior(RID p_lightmap, bool p_interior) override;
	virtual void lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) override;
	virtual void lightmap_set_baked_exposure_normalization(RID p_lightmap, float p_exposure) override;
	virtual PackedVector3Array lightmap_get_probe_capture_points(RID p_lightmap) const override;
	virtual PackedColorArray lightmap_get_probe_capture_sh(RID p_lightmap) const override;
	virtual PackedInt32Array lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const override;
	virtual PackedInt32Array lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const override;
	virtual AABB lightmap_get_aabb(RID p_lightmap) const override;
	virtual void lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) override;
	virtual bool lightmap_is_interior(RID p_lightmap) const override;
	virtual void lightmap_set_probe_capture_update_speed(float p_speed) override;
	virtual float lightmap_get_probe_capture_update_speed() const override;

	/* LIGHT SHADOW MAPPING */
	/*
	struct CanvasOccluder {
		RID self;

		GLuint vertex_id; // 0 means, unconfigured
		GLuint index_id; // 0 means, unconfigured
		LocalVector<Vector2> lines;
		int len;
	};

	RID_Owner<CanvasOccluder> canvas_occluder_owner;

	RID canvas_light_occluder_create();
	void canvas_light_occluder_set_polylines(RID p_occluder, const LocalVector<Vector2> &p_lines);
	*/

	/* LIGHTMAP INSTANCE */

	virtual RID lightmap_instance_create(RID p_lightmap) override;
	virtual void lightmap_instance_free(RID p_lightmap) override;
	virtual void lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) override;

	/* SHADOW ATLAS API */

	virtual RID shadow_atlas_create() override;
	virtual void shadow_atlas_free(RID p_atlas) override;
	virtual void shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits = true) override;
	virtual void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) override;
	virtual bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) override;

	virtual void shadow_atlas_update(RID p_atlas) override;

	virtual void directional_shadow_atlas_set_size(int p_size, bool p_16_bits = true) override;
	virtual int get_directional_light_shadow_size(RID p_light_intance) override;
	virtual void set_directional_shadow_count(int p_count) override;
};

} // namespace GLES3

#endif // GLES3_ENABLED

#endif // LIGHT_STORAGE_GLES3_H
