/*************************************************************************/
/*  renderer_scene_render_rd.h                                           */
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

#ifndef RENDERER_SCENE_RENDER_RD_H
#define RENDERER_SCENE_RENDER_RD_H

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/cluster_builder_rd.h"
#include "servers/rendering/renderer_rd/effects/bokeh_dof.h"
#include "servers/rendering/renderer_rd/effects/copy_effects.h"
#include "servers/rendering/renderer_rd/effects/fsr.h"
#include "servers/rendering/renderer_rd/effects/ss_effects.h"
#include "servers/rendering/renderer_rd/effects/tone_mapper.h"
#include "servers/rendering/renderer_rd/effects/vrs.h"
#include "servers/rendering/renderer_rd/environment/fog.h"
#include "servers/rendering/renderer_rd/environment/gi.h"
#include "servers/rendering/renderer_rd/environment/sky.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_scene.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering/rendering_device.h"

struct RenderDataRD {
	Ref<RenderSceneBuffersRD> render_buffers;

	Transform3D cam_transform;
	Projection cam_projection;
	Vector2 taa_jitter;
	bool cam_orthogonal = false;

	// For stereo rendering
	uint32_t view_count = 1;
	Vector3 view_eye_offset[RendererSceneRender::MAX_RENDER_VIEWS];
	Projection view_projection[RendererSceneRender::MAX_RENDER_VIEWS];

	Transform3D prev_cam_transform;
	Projection prev_cam_projection;
	Vector2 prev_taa_jitter;
	Projection prev_view_projection[RendererSceneRender::MAX_RENDER_VIEWS];

	float z_near = 0.0;
	float z_far = 0.0;

	const PagedArray<RenderGeometryInstance *> *instances = nullptr;
	const PagedArray<RID> *lights = nullptr;
	const PagedArray<RID> *reflection_probes = nullptr;
	const PagedArray<RID> *voxel_gi_instances = nullptr;
	const PagedArray<RID> *decals = nullptr;
	const PagedArray<RID> *lightmaps = nullptr;
	const PagedArray<RID> *fog_volumes = nullptr;
	RID environment;
	RID camera_attributes;
	RID shadow_atlas;
	RID reflection_atlas;
	RID reflection_probe;
	int reflection_probe_pass = 0;

	float lod_distance_multiplier = 0.0;
	Plane lod_camera_plane;
	float screen_mesh_lod_threshold = 0.0;

	RID cluster_buffer;
	uint32_t cluster_size = 0;
	uint32_t cluster_max_elements = 0;

	uint32_t directional_light_count = 0;
	bool directional_light_soft_shadows = false;

	RendererScene::RenderInfo *render_info = nullptr;
};

class RendererSceneRenderRD : public RendererSceneRender {
	friend RendererRD::SkyRD;
	friend RendererRD::GI;

protected:
	RendererRD::BokehDOF *bokeh_dof = nullptr;
	RendererRD::CopyEffects *copy_effects = nullptr;
	RendererRD::ToneMapper *tone_mapper = nullptr;
	RendererRD::FSR *fsr = nullptr;
	RendererRD::VRS *vrs = nullptr;
	double time = 0.0;
	double time_step = 0.0;

	virtual void setup_render_buffer_data(Ref<RenderSceneBuffersRD> p_render_buffers) = 0;

	void _setup_lights(RenderDataRD *p_render_data, const PagedArray<RID> &p_lights, const Transform3D &p_camera_transform, RID p_shadow_atlas, bool p_using_shadows, uint32_t &r_directional_light_count, uint32_t &r_positional_light_count, bool &r_directional_light_soft_shadows);
	void _setup_decals(const PagedArray<RID> &p_decals, const Transform3D &p_camera_inverse_xform);
	void _setup_reflections(RenderDataRD *p_render_data, const PagedArray<RID> &p_reflections, const Transform3D &p_camera_inverse_transform, RID p_environment);

	virtual void _render_scene(RenderDataRD *p_render_data, const Color &p_default_color) = 0;

	virtual void _render_shadow_begin() = 0;
	virtual void _render_shadow_append(RID p_framebuffer, const PagedArray<RenderGeometryInstance *> &p_instances, const Projection &p_projection, const Transform3D &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_use_dp, bool p_use_dp_flip, bool p_use_pancake, const Plane &p_camera_plane = Plane(), float p_lod_distance_multiplier = 0.0, float p_screen_mesh_lod_threshold = 0.0, const Rect2i &p_rect = Rect2i(), bool p_flip_y = false, bool p_clear_region = true, bool p_begin = true, bool p_end = true, RendererScene::RenderInfo *p_render_info = nullptr) = 0;
	virtual void _render_shadow_process() = 0;
	virtual void _render_shadow_end(uint32_t p_barrier = RD::BARRIER_MASK_ALL) = 0;

	virtual void _render_material(const Transform3D &p_cam_transform, const Projection &p_cam_projection, bool p_cam_orthogonal, const PagedArray<RenderGeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region, float p_exposure_normalization) = 0;
	virtual void _render_uv2(const PagedArray<RenderGeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) = 0;
	virtual void _render_sdfgi(Ref<RenderSceneBuffersRD> p_render_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const PagedArray<RenderGeometryInstance *> &p_instances, const RID &p_albedo_texture, const RID &p_emission_texture, const RID &p_emission_aniso_texture, const RID &p_geom_facing_texture, float p_exposure_normalization) = 0;
	virtual void _render_particle_collider_heightfield(RID p_fb, const Transform3D &p_cam_transform, const Projection &p_cam_projection, const PagedArray<RenderGeometryInstance *> &p_instances) = 0;

	void _debug_sdfgi_probes(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_framebuffer, uint32_t p_view_count, const Projection *p_camera_with_transforms, bool p_will_continue_color, bool p_will_continue_depth);
	void _debug_draw_cluster(Ref<RenderSceneBuffersRD> p_render_buffers);

	virtual void _base_uniforms_changed() = 0;
	virtual RID _render_buffers_get_normal_texture(Ref<RenderSceneBuffersRD> p_render_buffers) = 0;
	virtual RID _render_buffers_get_velocity_texture(Ref<RenderSceneBuffersRD> p_render_buffers) = 0;

	void _process_ssao(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_environment, RID p_normal_buffer, const Projection &p_projection);
	void _process_ssr(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_dest_framebuffer, const RID *p_normal_buffer_slices, RID p_specular_buffer, const RID *p_metallic_slices, const Color &p_metallic_mask, RID p_environment, const Projection *p_projections, const Vector3 *p_eye_offsets, bool p_use_additive);
	void _process_sss(Ref<RenderSceneBuffersRD> p_render_buffers, const Projection &p_camera);
	void _process_ssil(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_environment, RID p_normal_buffer, const Projection &p_projection, const Transform3D &p_transform);

	void _copy_framebuffer_to_ssil(Ref<RenderSceneBuffersRD> p_render_buffers);

	bool _needs_post_prepass_render(RenderDataRD *p_render_data, bool p_use_gi);
	void _post_prepass_render(RenderDataRD *p_render_data, bool p_use_gi);
	void _pre_resolve_render(RenderDataRD *p_render_data, bool p_use_gi);

	void _pre_opaque_render(RenderDataRD *p_render_data, bool p_use_ssao, bool p_use_ssil, bool p_use_gi, const RID *p_normal_roughness_slices, RID p_voxel_gi_buffer);

	void _render_buffers_copy_screen_texture(const RenderDataRD *p_render_data);
	void _render_buffers_copy_depth_texture(const RenderDataRD *p_render_data);
	void _render_buffers_post_process_and_tonemap(const RenderDataRD *p_render_data);
	void _post_process_subpass(RID p_source_texture, RID p_framebuffer, const RenderDataRD *p_render_data);
	void _disable_clear_request(const RenderDataRD *p_render_data);

	// needed for a single argument calls (material and uv2)
	PagedArrayPool<RenderGeometryInstance *> cull_argument_pool;
	PagedArray<RenderGeometryInstance *> cull_argument; //need this to exist

	RendererRD::SSEffects *ss_effects = nullptr;
	RendererRD::GI gi;
	RendererRD::SkyRD sky;

	//used for mobile renderer mostly

	typedef int32_t ForwardID;

	enum ForwardIDType {
		FORWARD_ID_TYPE_OMNI_LIGHT,
		FORWARD_ID_TYPE_SPOT_LIGHT,
		FORWARD_ID_TYPE_REFLECTION_PROBE,
		FORWARD_ID_TYPE_DECAL,
		FORWARD_ID_MAX,
	};

	virtual ForwardID _allocate_forward_id(ForwardIDType p_type) { return -1; }
	virtual void _free_forward_id(ForwardIDType p_type, ForwardID p_id) {}
	virtual void _map_forward_id(ForwardIDType p_type, ForwardID p_id, uint32_t p_index) {}
	virtual bool _uses_forward_ids() const { return false; }

	virtual void _update_shader_quality_settings() {}

private:
	RS::ViewportDebugDraw debug_draw = RS::VIEWPORT_DEBUG_DRAW_DISABLED;
	static RendererSceneRenderRD *singleton;

	/* REFLECTION ATLAS */

	struct ReflectionAtlas {
		int count = 0;
		int size = 0;

		RID reflection;
		RID depth_buffer;
		RID depth_fb;

		struct Reflection {
			RID owner;
			RendererRD::SkyRD::ReflectionData data;
			RID fbs[6];
		};

		Vector<Reflection> reflections;

		ClusterBuilderRD *cluster_builder = nullptr;
	};

	mutable RID_Owner<ReflectionAtlas> reflection_atlas_owner;

	/* REFLECTION PROBE INSTANCE */

	struct ReflectionProbeInstance {
		RID probe;
		int atlas_index = -1;
		RID atlas;

		bool dirty = true;
		bool rendering = false;
		int processing_layer = 1;
		int processing_side = 0;

		uint32_t render_step = 0;
		uint64_t last_pass = 0;
		uint32_t cull_mask = 0;

		ForwardID forward_id = -1;

		Transform3D transform;
	};

	mutable RID_Owner<ReflectionProbeInstance> reflection_probe_instance_owner;

	/* DECAL INSTANCE */

	struct DecalInstance {
		RID decal;
		Transform3D transform;
		uint32_t cull_mask = 0;
		ForwardID forward_id = -1;
	};

	mutable RID_Owner<DecalInstance> decal_instance_owner;

	/* LIGHTMAP INSTANCE */

	struct LightmapInstance {
		RID lightmap;
		Transform3D transform;
	};

	mutable RID_Owner<LightmapInstance> lightmap_instance_owner;

	/* SHADOW ATLAS */

	struct ShadowShrinkStage {
		RID texture;
		RID filter_texture;
		uint32_t size = 0;
	};

	struct ShadowAtlas {
		enum {
			QUADRANT_SHIFT = 27,
			OMNI_LIGHT_FLAG = 1 << 26,
			SHADOW_INDEX_MASK = OMNI_LIGHT_FLAG - 1,
			SHADOW_INVALID = 0xFFFFFFFF
		};

		struct Quadrant {
			uint32_t subdivision = 0;

			struct Shadow {
				RID owner;
				uint64_t version = 0;
				uint64_t fog_version = 0; // used for fog
				uint64_t alloc_tick = 0;

				Shadow() {}
			};

			Vector<Shadow> shadows;

			Quadrant() {}
		} quadrants[4];

		int size_order[4] = { 0, 1, 2, 3 };
		uint32_t smallest_subdiv = 0;

		int size = 0;
		bool use_16_bits = true;

		RID depth;
		RID fb; //for copying

		HashMap<RID, uint32_t> shadow_owners;
	};

	RID_Owner<ShadowAtlas> shadow_atlas_owner;

	void _update_shadow_atlas(ShadowAtlas *shadow_atlas);

	void _shadow_atlas_invalidate_shadow(RendererSceneRenderRD::ShadowAtlas::Quadrant::Shadow *p_shadow, RID p_atlas, RendererSceneRenderRD::ShadowAtlas *p_shadow_atlas, uint32_t p_quadrant, uint32_t p_shadow_idx);
	bool _shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow);
	bool _shadow_atlas_find_omni_shadows(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow);

	RS::ShadowQuality shadows_quality = RS::SHADOW_QUALITY_MAX; //So it always updates when first set
	RS::ShadowQuality directional_shadow_quality = RS::SHADOW_QUALITY_MAX;
	float shadows_quality_radius = 1.0;
	float directional_shadow_quality_radius = 1.0;

	float *directional_penumbra_shadow_kernel = nullptr;
	float *directional_soft_shadow_kernel = nullptr;
	float *penumbra_shadow_kernel = nullptr;
	float *soft_shadow_kernel = nullptr;
	int directional_penumbra_shadow_samples = 0;
	int directional_soft_shadow_samples = 0;
	int penumbra_shadow_samples = 0;
	int soft_shadow_samples = 0;
	RS::DecalFilter decals_filter = RS::DECAL_FILTER_LINEAR_MIPMAPS;
	RS::LightProjectorFilter light_projectors_filter = RS::LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS;

	/* DIRECTIONAL SHADOW */

	struct DirectionalShadow {
		RID depth;
		RID fb; //when renderign direct

		int light_count = 0;
		int size = 0;
		bool use_16_bits = true;
		int current_light = 0;
	} directional_shadow;

	void _update_directional_shadow_atlas();

	/* SHADOW CUBEMAPS */

	struct ShadowCubemap {
		RID cubemap;
		RID side_fb[6];
	};

	HashMap<int, ShadowCubemap> shadow_cubemaps;
	ShadowCubemap *_get_shadow_cubemap(int p_size);

	void _create_shadow_cubemaps();

	/* LIGHT INSTANCE */

	struct LightInstance {
		struct ShadowTransform {
			Projection camera;
			Transform3D transform;
			float farplane;
			float split;
			float bias_scale;
			float shadow_texel_size;
			float range_begin;
			Rect2 atlas_rect;
			Vector2 uv_scale;
		};

		RS::LightType light_type = RS::LIGHT_DIRECTIONAL;

		ShadowTransform shadow_transform[6];

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

		HashSet<RID> shadow_atlases; //shadow atlases where this light is registered

		ForwardID forward_id = -1;

		LightInstance() {}
	};

	mutable RID_Owner<LightInstance> light_instance_owner;

	/* ENVIRONMENT */

	RS::EnvironmentSSAOQuality ssao_quality = RS::ENV_SSAO_QUALITY_MEDIUM;
	bool ssao_half_size = false;
	float ssao_adaptive_target = 0.5;
	int ssao_blur_passes = 2;
	float ssao_fadeout_from = 50.0;
	float ssao_fadeout_to = 300.0;

	RS::EnvironmentSSILQuality ssil_quality = RS::ENV_SSIL_QUALITY_MEDIUM;
	bool ssil_half_size = false;
	bool ssil_using_half_size = false;
	float ssil_adaptive_target = 0.5;
	int ssil_blur_passes = 4;
	float ssil_fadeout_from = 50.0;
	float ssil_fadeout_to = 300.0;

	bool glow_bicubic_upscale = false;
	bool glow_high_quality = false;
	RS::EnvironmentSSRRoughnessQuality ssr_roughness_quality = RS::ENV_SSR_ROUGHNESS_QUALITY_LOW;

	RS::SubSurfaceScatteringQuality sss_quality = RS::SUB_SURFACE_SCATTERING_QUALITY_MEDIUM;
	float sss_scale = 0.05;
	float sss_depth_scale = 0.01;

	bool use_physical_light_units = false;

	/* Cluster builder */

	ClusterBuilderSharedDataRD cluster_builder_shared;
	ClusterBuilderRD *current_cluster_builder = nullptr;

	/* RENDER BUFFERS */

	void _allocate_luminance_textures(Ref<RenderSceneBuffersRD> rb);

	void _render_buffers_debug_draw(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_shadow_atlas, RID p_occlusion_buffer);

	/* GI */
	bool screen_space_roughness_limiter = false;
	float screen_space_roughness_limiter_amount = 0.25;
	float screen_space_roughness_limiter_limit = 0.18;

	/* Cluster */

	struct Cluster {
		/* Scene State UBO */

		// !BAS! Most data here is not just used by our clustering logic but also by other lighting implementations. Maybe rename this struct to something more appropriate

		enum {
			REFLECTION_AMBIENT_DISABLED = 0,
			REFLECTION_AMBIENT_ENVIRONMENT = 1,
			REFLECTION_AMBIENT_COLOR = 2,
		};

		struct ReflectionData {
			float box_extents[3];
			float index;
			float box_offset[3];
			uint32_t mask;
			float ambient[3]; // ambient color,
			float intensity;
			uint32_t exterior;
			uint32_t box_project;
			uint32_t ambient_mode;
			float exposure_normalization;
			float local_matrix[16]; // up to here for spot and omni, rest is for directional
		};

		struct LightData {
			float position[3];
			float inv_radius;
			float direction[3]; // in omni, x and y are used for dual paraboloid offset
			float size;

			float color[3];
			float attenuation;

			float inv_spot_attenuation;
			float cos_spot_angle;
			float specular_amount;
			float shadow_opacity;

			float atlas_rect[4]; // in omni, used for atlas uv, in spot, used for projector uv
			float shadow_matrix[16];
			float shadow_bias;
			float shadow_normal_bias;
			float transmittance_bias;
			float soft_shadow_size;
			float soft_shadow_scale;
			uint32_t mask;
			float volumetric_fog_energy;
			uint32_t bake_mode;
			float projector_rect[4];
		};

		struct DirectionalLightData {
			float direction[3];
			float energy;
			float color[3];
			float size;
			float specular;
			uint32_t mask;
			float softshadow_angle;
			float soft_shadow_scale;
			uint32_t blend_splits;
			float shadow_opacity;
			float fade_from;
			float fade_to;
			uint32_t pad[2];
			uint32_t bake_mode;
			float volumetric_fog_energy;
			float shadow_bias[4];
			float shadow_normal_bias[4];
			float shadow_transmittance_bias[4];
			float shadow_z_range[4];
			float shadow_range_begin[4];
			float shadow_split_offsets[4];
			float shadow_matrices[4][16];
			float uv_scale1[2];
			float uv_scale2[2];
			float uv_scale3[2];
			float uv_scale4[2];
		};

		struct DecalData {
			float xform[16];
			float inv_extents[3];
			float albedo_mix;
			float albedo_rect[4];
			float normal_rect[4];
			float orm_rect[4];
			float emission_rect[4];
			float modulate[4];
			float emission_energy;
			uint32_t mask;
			float upper_fade;
			float lower_fade;
			float normal_xform[12];
			float normal[3];
			float normal_fade;
		};

		template <class T>
		struct InstanceSort {
			float depth;
			T *instance = nullptr;
			bool operator<(const InstanceSort &p_sort) const {
				return depth < p_sort.depth;
			}
		};

		ReflectionData *reflections = nullptr;
		InstanceSort<ReflectionProbeInstance> *reflection_sort;
		uint32_t max_reflections;
		RID reflection_buffer;
		uint32_t max_reflection_probes_per_instance;
		uint32_t reflection_count = 0;

		DecalData *decals = nullptr;
		InstanceSort<DecalInstance> *decal_sort;
		uint32_t max_decals;
		RID decal_buffer;
		uint32_t decal_count;

		LightData *omni_lights = nullptr;
		LightData *spot_lights = nullptr;

		InstanceSort<LightInstance> *omni_light_sort;
		InstanceSort<LightInstance> *spot_light_sort;
		uint32_t max_lights;
		RID omni_light_buffer;
		RID spot_light_buffer;
		uint32_t omni_light_count = 0;
		uint32_t spot_light_count = 0;

		DirectionalLightData *directional_lights = nullptr;
		uint32_t max_directional_lights;
		RID directional_light_buffer;

	} cluster;

	struct RenderState {
		const RendererSceneRender::RenderShadowData *render_shadows = nullptr;
		int render_shadow_count = 0;
		const RendererSceneRender::RenderSDFGIData *render_sdfgi_regions = nullptr;
		int render_sdfgi_region_count = 0;
		const RendererSceneRender::RenderSDFGIUpdateData *sdfgi_update_data = nullptr;

		uint32_t voxel_gi_count = 0;

		LocalVector<int> cube_shadows;
		LocalVector<int> shadows;
		LocalVector<int> directional_shadows;

		bool depth_prepass_used; // this does not seem used anywhere...
	} render_state;

	RID shadow_sampler;

	uint64_t scene_pass = 0;
	uint64_t shadow_atlas_realloc_tolerance_msec = 500;

	/* !BAS! is this used anywhere?
	struct SDFGICosineNeighbour {
		uint32_t neighbour;
		float weight;
	};
	*/

	uint32_t max_cluster_elements = 512;

	void _render_shadow_pass(RID p_light, RID p_shadow_atlas, int p_pass, const PagedArray<RenderGeometryInstance *> &p_instances, const Plane &p_camera_plane = Plane(), float p_lod_distance_multiplier = 0, float p_screen_mesh_lod_threshold = 0.0, bool p_open_pass = true, bool p_close_pass = true, bool p_clear_region = true, RendererScene::RenderInfo *p_render_info = nullptr);

	/* Volumetric Fog */

	uint32_t volumetric_fog_size = 128;
	uint32_t volumetric_fog_depth = 128;
	bool volumetric_fog_filter_active = true;

	void _update_volumetric_fog(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_environment, const Projection &p_cam_projection, const Transform3D &p_cam_transform, const Transform3D &p_prev_cam_inv_transform, RID p_shadow_atlas, int p_directional_light_count, bool p_use_directional_shadows, int p_positional_light_count, int p_voxel_gi_count, const PagedArray<RID> &p_fog_volumes);

public:
	static RendererSceneRenderRD *get_singleton() { return singleton; }

	/* Cluster builder */
	ClusterBuilderSharedDataRD *get_cluster_builder_shared() { return &cluster_builder_shared; }

	/* GI */

	RendererRD::GI *get_gi() { return &gi; }

	/* SHADOW ATLAS API */

	virtual RID shadow_atlas_create() override;
	virtual void shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits = true) override;
	virtual void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) override;
	virtual bool shadow_atlas_update_light(RID p_atlas, RID p_light_instance, float p_coverage, uint64_t p_light_version) override;
	_FORCE_INLINE_ bool shadow_atlas_owns_light_instance(RID p_atlas, RID p_light_intance) {
		ShadowAtlas *atlas = shadow_atlas_owner.get_or_null(p_atlas);
		ERR_FAIL_COND_V(!atlas, false);
		return atlas->shadow_owners.has(p_light_intance);
	}

	_FORCE_INLINE_ RID shadow_atlas_get_texture(RID p_atlas) {
		ShadowAtlas *atlas = shadow_atlas_owner.get_or_null(p_atlas);
		ERR_FAIL_COND_V(!atlas, RID());
		return atlas->depth;
	}

	_FORCE_INLINE_ Size2i shadow_atlas_get_size(RID p_atlas) {
		ShadowAtlas *atlas = shadow_atlas_owner.get_or_null(p_atlas);
		ERR_FAIL_COND_V(!atlas, Size2i());
		return Size2(atlas->size, atlas->size);
	}

	virtual void directional_shadow_atlas_set_size(int p_size, bool p_16_bits = true) override;
	virtual int get_directional_light_shadow_size(RID p_light_intance) override;
	virtual void set_directional_shadow_count(int p_count) override;

	_FORCE_INLINE_ RID directional_shadow_get_texture() {
		return directional_shadow.depth;
	}

	_FORCE_INLINE_ Size2i directional_shadow_get_size() {
		return Size2i(directional_shadow.size, directional_shadow.size);
	}

	/* SDFGI UPDATE */

	virtual void sdfgi_update(const Ref<RenderSceneBuffers> &p_render_buffers, RID p_environment, const Vector3 &p_world_position) override;
	virtual int sdfgi_get_pending_region_count(const Ref<RenderSceneBuffers> &p_render_buffers) const override;
	virtual AABB sdfgi_get_pending_region_bounds(const Ref<RenderSceneBuffers> &p_render_buffers, int p_region) const override;
	virtual uint32_t sdfgi_get_pending_region_cascade(const Ref<RenderSceneBuffers> &p_render_buffers, int p_region) const override;
	RID sdfgi_get_ubo() const { return gi.sdfgi_ubo; }

	/* SKY API */

	virtual RID sky_allocate() override;
	virtual void sky_initialize(RID p_rid) override;

	virtual void sky_set_radiance_size(RID p_sky, int p_radiance_size) override;
	virtual void sky_set_mode(RID p_sky, RS::SkyMode p_mode) override;
	virtual void sky_set_material(RID p_sky, RID p_material) override;
	virtual Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) override;

	/* ENVIRONMENT API */

	virtual void environment_glow_set_use_bicubic_upscale(bool p_enable) override;
	virtual void environment_glow_set_use_high_quality(bool p_enable) override;

	virtual void environment_set_volumetric_fog_volume_size(int p_size, int p_depth) override;
	virtual void environment_set_volumetric_fog_filter_active(bool p_enable) override;

	virtual void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override;

	virtual void environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override;

	virtual void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) override;
	virtual void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) override;
	virtual void environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) override;

	virtual void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) override;
	RS::EnvironmentSSRRoughnessQuality environment_get_ssr_roughness_quality() const;

	virtual Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) override;

	_FORCE_INLINE_ bool is_using_physical_light_units() {
		return use_physical_light_units;
	}

	/* LIGHT INSTANCE API */

	virtual RID light_instance_create(RID p_light) override;
	virtual void light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) override;
	virtual void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) override;
	virtual void light_instance_set_shadow_transform(RID p_light_instance, const Projection &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2()) override;
	virtual void light_instance_mark_visible(RID p_light_instance) override;

	_FORCE_INLINE_ RID light_instance_get_base_light(RID p_light_instance) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->light;
	}

	_FORCE_INLINE_ Transform3D light_instance_get_base_transform(RID p_light_instance) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->transform;
	}

	_FORCE_INLINE_ Rect2 light_instance_get_shadow_atlas_rect(RID p_light_instance, RID p_shadow_atlas, Vector2i &r_omni_offset) {
		ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(p_shadow_atlas);
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		uint32_t key = shadow_atlas->shadow_owners[li->self];

		uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;
		uint32_t shadow = key & ShadowAtlas::SHADOW_INDEX_MASK;

		ERR_FAIL_COND_V(shadow >= (uint32_t)shadow_atlas->quadrants[quadrant].shadows.size(), Rect2());

		uint32_t atlas_size = shadow_atlas->size;
		uint32_t quadrant_size = atlas_size >> 1;

		uint32_t x = (quadrant & 1) * quadrant_size;
		uint32_t y = (quadrant >> 1) * quadrant_size;

		uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);
		x += (shadow % shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;
		y += (shadow / shadow_atlas->quadrants[quadrant].subdivision) * shadow_size;

		if (key & ShadowAtlas::OMNI_LIGHT_FLAG) {
			if (((shadow + 1) % shadow_atlas->quadrants[quadrant].subdivision) == 0) {
				r_omni_offset.x = 1 - int(shadow_atlas->quadrants[quadrant].subdivision);
				r_omni_offset.y = 1;
			} else {
				r_omni_offset.x = 1;
				r_omni_offset.y = 0;
			}
		}

		uint32_t width = shadow_size;
		uint32_t height = shadow_size;

		return Rect2(x / float(shadow_atlas->size), y / float(shadow_atlas->size), width / float(shadow_atlas->size), height / float(shadow_atlas->size));
	}

	_FORCE_INLINE_ Projection light_instance_get_shadow_camera(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].camera;
	}

	_FORCE_INLINE_ float light_instance_get_shadow_texel_size(RID p_light_instance, RID p_shadow_atlas) {
#ifdef DEBUG_ENABLED
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		ERR_FAIL_COND_V(!li->shadow_atlases.has(p_shadow_atlas), 0);
#endif
		ShadowAtlas *shadow_atlas = shadow_atlas_owner.get_or_null(p_shadow_atlas);
		ERR_FAIL_COND_V(!shadow_atlas, 0);
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_V(!shadow_atlas->shadow_owners.has(p_light_instance), 0);
#endif
		uint32_t key = shadow_atlas->shadow_owners[p_light_instance];

		uint32_t quadrant = (key >> ShadowAtlas::QUADRANT_SHIFT) & 0x3;

		uint32_t quadrant_size = shadow_atlas->size >> 1;

		uint32_t shadow_size = (quadrant_size / shadow_atlas->quadrants[quadrant].subdivision);

		return float(1.0) / shadow_size;
	}

	_FORCE_INLINE_ Transform3D
	light_instance_get_shadow_transform(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].transform;
	}
	_FORCE_INLINE_ float light_instance_get_shadow_bias_scale(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].bias_scale;
	}
	_FORCE_INLINE_ float light_instance_get_shadow_range(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].farplane;
	}
	_FORCE_INLINE_ float light_instance_get_shadow_range_begin(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].range_begin;
	}

	_FORCE_INLINE_ Vector2 light_instance_get_shadow_uv_scale(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].uv_scale;
	}

	_FORCE_INLINE_ Rect2 light_instance_get_directional_shadow_atlas_rect(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].atlas_rect;
	}

	_FORCE_INLINE_ float light_instance_get_directional_shadow_split(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].split;
	}

	_FORCE_INLINE_ float light_instance_get_directional_shadow_texel_size(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->shadow_transform[p_index].shadow_texel_size;
	}

	_FORCE_INLINE_ void light_instance_set_render_pass(RID p_light_instance, uint64_t p_pass) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		li->last_pass = p_pass;
	}

	_FORCE_INLINE_ uint64_t light_instance_get_render_pass(RID p_light_instance) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->last_pass;
	}

	_FORCE_INLINE_ ForwardID light_instance_get_forward_id(RID p_light_instance) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->forward_id;
	}

	_FORCE_INLINE_ RS::LightType light_instance_get_type(RID p_light_instance) {
		LightInstance *li = light_instance_owner.get_or_null(p_light_instance);
		return li->light_type;
	}

	/* FOG VOLUMES */

	virtual RID fog_volume_instance_create(RID p_fog_volume) override;
	virtual void fog_volume_instance_set_transform(RID p_fog_volume_instance, const Transform3D &p_transform) override;
	virtual void fog_volume_instance_set_active(RID p_fog_volume_instance, bool p_active) override;
	virtual RID fog_volume_instance_get_volume(RID p_fog_volume_instance) const override;
	virtual Vector3 fog_volume_instance_get_position(RID p_fog_volume_instance) const override;

	virtual RID reflection_atlas_create() override;
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) override;
	virtual int reflection_atlas_get_size(RID p_ref_atlas) const override;

	_FORCE_INLINE_ RID reflection_atlas_get_texture(RID p_ref_atlas) {
		ReflectionAtlas *atlas = reflection_atlas_owner.get_or_null(p_ref_atlas);
		ERR_FAIL_COND_V(!atlas, RID());
		return atlas->reflection;
	}

	virtual RID reflection_probe_instance_create(RID p_probe) override;
	virtual void reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) override;
	virtual void reflection_probe_release_atlas_index(RID p_instance) override;
	virtual bool reflection_probe_instance_needs_redraw(RID p_instance) override;
	virtual bool reflection_probe_instance_has_reflection(RID p_instance) override;
	virtual bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) override;
	virtual RID reflection_probe_create_framebuffer(RID p_color, RID p_depth);
	virtual bool reflection_probe_instance_postprocess_step(RID p_instance) override;

	uint32_t reflection_probe_instance_get_resolution(RID p_instance);
	RID reflection_probe_instance_get_framebuffer(RID p_instance, int p_index);
	RID reflection_probe_instance_get_depth_framebuffer(RID p_instance, int p_index);

	_FORCE_INLINE_ RID reflection_probe_instance_get_probe(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
		ERR_FAIL_COND_V(!rpi, RID());

		return rpi->probe;
	}

	_FORCE_INLINE_ ForwardID reflection_probe_instance_get_forward_id(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
		ERR_FAIL_COND_V(!rpi, 0);

		return rpi->forward_id;
	}

	_FORCE_INLINE_ void reflection_probe_instance_set_render_pass(RID p_instance, uint32_t p_render_pass) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
		ERR_FAIL_COND(!rpi);
		rpi->last_pass = p_render_pass;
	}

	_FORCE_INLINE_ uint32_t reflection_probe_instance_get_render_pass(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
		ERR_FAIL_COND_V(!rpi, 0);

		return rpi->last_pass;
	}

	_FORCE_INLINE_ Transform3D reflection_probe_instance_get_transform(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
		ERR_FAIL_COND_V(!rpi, Transform3D());

		return rpi->transform;
	}

	_FORCE_INLINE_ int reflection_probe_instance_get_atlas_index(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.get_or_null(p_instance);
		ERR_FAIL_COND_V(!rpi, -1);

		return rpi->atlas_index;
	}

	virtual RID decal_instance_create(RID p_decal) override;
	virtual void decal_instance_set_transform(RID p_decal, const Transform3D &p_transform) override;

	_FORCE_INLINE_ RID decal_instance_get_base(RID p_decal) const {
		DecalInstance *decal = decal_instance_owner.get_or_null(p_decal);
		return decal->decal;
	}

	_FORCE_INLINE_ ForwardID decal_instance_get_forward_id(RID p_decal) const {
		DecalInstance *decal = decal_instance_owner.get_or_null(p_decal);
		return decal->forward_id;
	}

	_FORCE_INLINE_ Transform3D decal_instance_get_transform(RID p_decal) const {
		DecalInstance *decal = decal_instance_owner.get_or_null(p_decal);
		return decal->transform;
	}

	virtual RID lightmap_instance_create(RID p_lightmap) override;
	virtual void lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) override;
	_FORCE_INLINE_ bool lightmap_instance_is_valid(RID p_lightmap_instance) {
		return lightmap_instance_owner.get_or_null(p_lightmap_instance) != nullptr;
	}

	_FORCE_INLINE_ RID lightmap_instance_get_lightmap(RID p_lightmap_instance) {
		LightmapInstance *li = lightmap_instance_owner.get_or_null(p_lightmap_instance);
		return li->lightmap;
	}
	_FORCE_INLINE_ Transform3D lightmap_instance_get_transform(RID p_lightmap_instance) {
		LightmapInstance *li = lightmap_instance_owner.get_or_null(p_lightmap_instance);
		return li->transform;
	}

	/* gi light probes */

	virtual RID voxel_gi_instance_create(RID p_base) override;
	virtual void voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) override;
	virtual bool voxel_gi_needs_update(RID p_probe) const override;
	virtual void voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RenderGeometryInstance *> &p_dynamic_objects) override;
	virtual void voxel_gi_set_quality(RS::VoxelGIQuality p_quality) override { gi.voxel_gi_quality = p_quality; }

	/* render buffers */

	virtual float _render_buffers_get_luminance_multiplier();
	virtual RD::DataFormat _render_buffers_get_color_format();
	virtual bool _render_buffers_can_be_storage();
	virtual Ref<RenderSceneBuffers> render_buffers_create() override;
	virtual void gi_set_use_half_resolution(bool p_enable) override;

	RID render_buffers_get_default_voxel_gi_buffer();

	virtual void update_uniform_sets(){};

	virtual void render_scene(const Ref<RenderSceneBuffers> &p_render_buffers, const CameraData *p_camera_data, const CameraData *p_prev_camera_data, const PagedArray<RenderGeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, const PagedArray<RID> &p_fog_volumes, RID p_environment, RID p_camera_attributes, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_mesh_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data = nullptr, RendererScene::RenderInfo *r_render_info = nullptr) override;

	virtual void render_material(const Transform3D &p_cam_transform, const Projection &p_cam_projection, bool p_cam_orthogonal, const PagedArray<RenderGeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) override;

	virtual void render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<RenderGeometryInstance *> &p_instances) override;

	virtual void set_scene_pass(uint64_t p_pass) override {
		scene_pass = p_pass;
	}
	_FORCE_INLINE_ uint64_t get_scene_pass() {
		return scene_pass;
	}

	virtual void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit) override;
	virtual bool screen_space_roughness_limiter_is_active() const override;
	virtual float screen_space_roughness_limiter_get_amount() const;
	virtual float screen_space_roughness_limiter_get_limit() const;

	virtual void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) override;
	RS::SubSurfaceScatteringQuality sub_surface_scattering_get_quality() const;
	virtual void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) override;

	virtual void positional_soft_shadow_filter_set_quality(RS::ShadowQuality p_quality) override;
	virtual void directional_soft_shadow_filter_set_quality(RS::ShadowQuality p_quality) override;

	virtual void decals_set_filter(RS::DecalFilter p_filter) override;
	virtual void light_projectors_set_filter(RS::LightProjectorFilter p_filter) override;

	_FORCE_INLINE_ RS::ShadowQuality shadows_quality_get() const {
		return shadows_quality;
	}
	_FORCE_INLINE_ RS::ShadowQuality directional_shadow_quality_get() const {
		return directional_shadow_quality;
	}
	_FORCE_INLINE_ float shadows_quality_radius_get() const {
		return shadows_quality_radius;
	}
	_FORCE_INLINE_ float directional_shadow_quality_radius_get() const {
		return directional_shadow_quality_radius;
	}

	_FORCE_INLINE_ float *directional_penumbra_shadow_kernel_get() {
		return directional_penumbra_shadow_kernel;
	}
	_FORCE_INLINE_ float *directional_soft_shadow_kernel_get() {
		return directional_soft_shadow_kernel;
	}
	_FORCE_INLINE_ float *penumbra_shadow_kernel_get() {
		return penumbra_shadow_kernel;
	}
	_FORCE_INLINE_ float *soft_shadow_kernel_get() {
		return soft_shadow_kernel;
	}

	_FORCE_INLINE_ int directional_penumbra_shadow_samples_get() const {
		return directional_penumbra_shadow_samples;
	}
	_FORCE_INLINE_ int directional_soft_shadow_samples_get() const {
		return directional_soft_shadow_samples;
	}
	_FORCE_INLINE_ int penumbra_shadow_samples_get() const {
		return penumbra_shadow_samples;
	}
	_FORCE_INLINE_ int soft_shadow_samples_get() const {
		return soft_shadow_samples;
	}

	_FORCE_INLINE_ RS::LightProjectorFilter light_projectors_get_filter() const {
		return light_projectors_filter;
	}
	_FORCE_INLINE_ RS::DecalFilter decals_get_filter() const {
		return decals_filter;
	}

	int get_roughness_layers() const;
	bool is_using_radiance_cubemap_array() const;

	virtual TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) override;

	virtual bool free(RID p_rid) override;

	virtual void update() override;

	virtual void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) override;
	_FORCE_INLINE_ RS::ViewportDebugDraw get_debug_draw_mode() const {
		return debug_draw;
	}

	virtual void set_time(double p_time, double p_step) override;

	RID get_reflection_probe_buffer();
	RID get_omni_light_buffer();
	RID get_spot_light_buffer();
	RID get_directional_light_buffer();
	RID get_decal_buffer();
	int get_max_directional_lights() const;

	virtual void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) override;

	virtual bool is_vrs_supported() const;
	virtual bool is_dynamic_gi_supported() const;
	virtual bool is_clustered_enabled() const;
	virtual bool is_volumetric_supported() const;
	virtual uint32_t get_max_elements() const;

	void init();

	RendererSceneRenderRD();
	~RendererSceneRenderRD();
};

#endif // RENDERER_SCENE_RENDER_RD_H
