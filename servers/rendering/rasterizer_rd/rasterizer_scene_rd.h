/*************************************************************************/
/*  rasterizer_scene_rd.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RASTERIZER_SCENE_RD_H
#define RASTERIZER_SCENE_RD_H

#include "core/local_vector.h"
#include "core/rid_owner.h"
#include "servers/rendering/rasterizer.h"
#include "servers/rendering/rasterizer_rd/light_cluster_builder.h"
#include "servers/rendering/rasterizer_rd/rasterizer_storage_rd.h"
#include "servers/rendering/rasterizer_rd/shaders/gi.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/giprobe.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/giprobe_debug.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/sdfgi_debug.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/sdfgi_debug_probes.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/sdfgi_direct_light.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/sdfgi_integrate.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/sdfgi_preprocess.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/sky.glsl.gen.h"
#include "servers/rendering/rasterizer_rd/shaders/volumetric_fog.glsl.gen.h"
#include "servers/rendering/rendering_device.h"

class RasterizerSceneRD : public RasterizerScene {
protected:
	double time;

	// Skys need less info from Directional Lights than the normal shaders
	struct SkyDirectionalLightData {
		float direction[3];
		float energy;
		float color[3];
		float size;
		uint32_t enabled;
		uint32_t pad[3];
	};

	struct SkySceneState {
		struct UBO {
			uint32_t volumetric_fog_enabled;
			float volumetric_fog_inv_length;
			float volumetric_fog_detail_spread;

			float fog_aerial_perspective;

			float fog_light_color[3];
			float fog_sun_scatter;

			uint32_t fog_enabled;
			float fog_density;

			float z_far;
			uint32_t directional_light_count;
		};

		UBO ubo;

		SkyDirectionalLightData *directional_lights;
		SkyDirectionalLightData *last_frame_directional_lights;
		uint32_t max_directional_lights;
		uint32_t last_frame_directional_light_count;
		RID directional_light_buffer;
		RID uniform_set;
		RID uniform_buffer;
		RID fog_uniform_set;
		RID default_fog_uniform_set;

		RID fog_shader;
		RID fog_material;
		RID fog_only_texture_uniform_set;
	} sky_scene_state;

	struct RenderBufferData {
		virtual void configure(RID p_color_buffer, RID p_depth_buffer, int p_width, int p_height, RS::ViewportMSAA p_msaa) = 0;
		virtual ~RenderBufferData() {}
	};
	virtual RenderBufferData *_create_render_buffer_data() = 0;

	void _setup_lights(RID *p_light_cull_result, int p_light_cull_count, const Transform &p_camera_inverse_transform, RID p_shadow_atlas, bool p_using_shadows, uint32_t &r_directional_light_count, uint32_t &r_positional_light_count);
	void _setup_decals(const RID *p_decal_instances, int p_decal_count, const Transform &p_camera_inverse_xform);
	void _setup_reflections(RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, const Transform &p_camera_inverse_transform, RID p_environment);
	void _setup_giprobes(RID p_render_buffers, const Transform &p_transform, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, uint32_t &r_gi_probes_used);

	virtual void _render_scene(RID p_render_buffer, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, int p_directional_light_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, InstanceBase **p_lightmap_cull_result, int p_lightmap_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, const Color &p_default_color) = 0;
	virtual void _render_shadow(RID p_framebuffer, InstanceBase **p_cull_result, int p_cull_count, const CameraMatrix &p_projection, const Transform &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_use_dp, bool use_dp_flip, bool p_use_pancake) = 0;
	virtual void _render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) = 0;
	virtual void _render_uv2(InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) = 0;
	virtual void _render_sdfgi(RID p_render_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, InstanceBase **p_cull_result, int p_cull_count, const RID &p_albedo_texture, const RID &p_emission_texture, const RID &p_emission_aniso_texture, const RID &p_geom_facing_texture) = 0;
	virtual void _render_particle_collider_heightfield(RID p_fb, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, InstanceBase **p_cull_result, int p_cull_count) = 0;

	virtual void _debug_giprobe(RID p_gi_probe, RenderingDevice::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha);
	void _debug_sdfgi_probes(RID p_render_buffers, RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform);

	RenderBufferData *render_buffers_get_data(RID p_render_buffers);

	virtual void _base_uniforms_changed() = 0;
	virtual void _render_buffers_uniform_set_changed(RID p_render_buffers) = 0;
	virtual RID _render_buffers_get_normal_texture(RID p_render_buffers) = 0;
	virtual RID _render_buffers_get_ambient_texture(RID p_render_buffers) = 0;
	virtual RID _render_buffers_get_reflection_texture(RID p_render_buffers) = 0;

	void _process_ssao(RID p_render_buffers, RID p_environment, RID p_normal_buffer, const CameraMatrix &p_projection);
	void _process_ssr(RID p_render_buffers, RID p_dest_framebuffer, RID p_normal_buffer, RID p_specular_buffer, RID p_metallic, const Color &p_metallic_mask, RID p_environment, const CameraMatrix &p_projection, bool p_use_additive);
	void _process_sss(RID p_render_buffers, const CameraMatrix &p_camera);

	void _setup_sky(RID p_environment, RID p_render_buffers, const CameraMatrix &p_projection, const Transform &p_transform, const Size2i p_screen_size);
	void _update_sky(RID p_environment, const CameraMatrix &p_projection, const Transform &p_transform);
	void _draw_sky(bool p_can_continue_color, bool p_can_continue_depth, RID p_fb, RID p_environment, const CameraMatrix &p_projection, const Transform &p_transform);
	void _process_gi(RID p_render_buffers, RID p_normal_roughness_buffer, RID p_ambient_buffer, RID p_reflection_buffer, RID p_gi_probe_buffer, RID p_environment, const CameraMatrix &p_projection, const Transform &p_transform, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count);

private:
	RS::ViewportDebugDraw debug_draw = RS::VIEWPORT_DEBUG_DRAW_DISABLED;
	double time_step = 0;
	static RasterizerSceneRD *singleton;

	int roughness_layers;

	RasterizerStorageRD *storage;

	struct ReflectionData {
		struct Layer {
			struct Mipmap {
				RID framebuffers[6];
				RID views[6];
				Size2i size;
			};
			Vector<Mipmap> mipmaps; //per-face view
			Vector<RID> views; // per-cubemap view
		};

		struct DownsampleLayer {
			struct Mipmap {
				RID view;
				Size2i size;
			};
			Vector<Mipmap> mipmaps;
		};

		RID radiance_base_cubemap; //cubemap for first layer, first cubemap
		RID downsampled_radiance_cubemap;
		DownsampleLayer downsampled_layer;
		RID coefficient_buffer;

		bool dirty = true;

		Vector<Layer> layers;
	};

	void _clear_reflection_data(ReflectionData &rd);
	void _update_reflection_data(ReflectionData &rd, int p_size, int p_mipmaps, bool p_use_array, RID p_base_cube, int p_base_layer, bool p_low_quality);
	void _create_reflection_fast_filter(ReflectionData &rd, bool p_use_arrays);
	void _create_reflection_importance_sample(ReflectionData &rd, bool p_use_arrays, int p_cube_side, int p_base_layer);
	void _update_reflection_mipmaps(ReflectionData &rd, int p_start, int p_end);

	/* Sky shader */

	enum SkyVersion {
		SKY_VERSION_BACKGROUND,
		SKY_VERSION_HALF_RES,
		SKY_VERSION_QUARTER_RES,
		SKY_VERSION_CUBEMAP,
		SKY_VERSION_CUBEMAP_HALF_RES,
		SKY_VERSION_CUBEMAP_QUARTER_RES,
		SKY_VERSION_MAX
	};

	struct SkyShader {
		SkyShaderRD shader;
		ShaderCompilerRD compiler;

		RID default_shader;
		RID default_material;
		RID default_shader_rd;
	} sky_shader;

	struct SkyShaderData : public RasterizerStorageRD::ShaderData {
		bool valid;
		RID version;

		RenderPipelineVertexFormatCacheRD pipelines[SKY_VERSION_MAX];
		Map<StringName, ShaderLanguage::ShaderNode::Uniform> uniforms;
		Vector<ShaderCompilerRD::GeneratedCode::Texture> texture_uniforms;

		Vector<uint32_t> ubo_offsets;
		uint32_t ubo_size;

		String path;
		String code;
		Map<StringName, RID> default_texture_params;

		bool uses_time;
		bool uses_position;
		bool uses_half_res;
		bool uses_quarter_res;
		bool uses_light;

		virtual void set_code(const String &p_Code);
		virtual void set_default_texture_param(const StringName &p_name, RID p_texture);
		virtual void get_param_list(List<PropertyInfo> *p_param_list) const;
		virtual void get_instance_param_list(List<RasterizerStorage::InstanceShaderParam> *p_param_list) const;
		virtual bool is_param_texture(const StringName &p_param) const;
		virtual bool is_animated() const;
		virtual bool casts_shadows() const;
		virtual Variant get_default_parameter(const StringName &p_parameter) const;
		SkyShaderData();
		virtual ~SkyShaderData();
	};

	RasterizerStorageRD::ShaderData *_create_sky_shader_func();
	static RasterizerStorageRD::ShaderData *_create_sky_shader_funcs() {
		return static_cast<RasterizerSceneRD *>(singleton)->_create_sky_shader_func();
	};

	struct SkyMaterialData : public RasterizerStorageRD::MaterialData {
		uint64_t last_frame;
		SkyShaderData *shader_data;
		RID uniform_buffer;
		RID uniform_set;
		Vector<RID> texture_cache;
		Vector<uint8_t> ubo_data;
		bool uniform_set_updated;

		virtual void set_render_priority(int p_priority) {}
		virtual void set_next_pass(RID p_pass) {}
		virtual void update_parameters(const Map<StringName, Variant> &p_parameters, bool p_uniform_dirty, bool p_textures_dirty);
		virtual ~SkyMaterialData();
	};

	RasterizerStorageRD::MaterialData *_create_sky_material_func(SkyShaderData *p_shader);
	static RasterizerStorageRD::MaterialData *_create_sky_material_funcs(RasterizerStorageRD::ShaderData *p_shader) {
		return static_cast<RasterizerSceneRD *>(singleton)->_create_sky_material_func(static_cast<SkyShaderData *>(p_shader));
	};

	enum SkyTextureSetVersion {
		SKY_TEXTURE_SET_BACKGROUND,
		SKY_TEXTURE_SET_HALF_RES,
		SKY_TEXTURE_SET_QUARTER_RES,
		SKY_TEXTURE_SET_CUBEMAP,
		SKY_TEXTURE_SET_CUBEMAP_HALF_RES,
		SKY_TEXTURE_SET_CUBEMAP_QUARTER_RES,
		SKY_TEXTURE_SET_MAX
	};

	enum SkySet {
		SKY_SET_UNIFORMS,
		SKY_SET_MATERIAL,
		SKY_SET_TEXTURES,
		SKY_SET_FOG,
		SKY_SET_MAX
	};

	/* SKY */
	struct Sky {
		RID radiance;
		RID half_res_pass;
		RID half_res_framebuffer;
		RID quarter_res_pass;
		RID quarter_res_framebuffer;
		Size2i screen_size;

		RID texture_uniform_sets[SKY_TEXTURE_SET_MAX];
		RID uniform_set;

		RID material;
		RID uniform_buffer;

		int radiance_size = 256;

		RS::SkyMode mode = RS::SKY_MODE_AUTOMATIC;

		ReflectionData reflection;
		bool dirty = false;
		int processing_layer = 0;
		Sky *dirty_list = nullptr;

		//State to track when radiance cubemap needs updating
		SkyMaterialData *prev_material;
		Vector3 prev_position;
		float prev_time;

		RID sdfgi_integrate_sky_uniform_set;
	};

	Sky *dirty_sky_list = nullptr;

	void _sky_invalidate(Sky *p_sky);
	void _update_dirty_skys();
	RID _get_sky_textures(Sky *p_sky, SkyTextureSetVersion p_version);

	uint32_t sky_ggx_samples_quality;
	bool sky_use_cubemap_array;

	mutable RID_Owner<Sky> sky_owner;

	/* REFLECTION ATLAS */

	struct ReflectionAtlas {
		int count = 0;
		int size = 0;

		RID reflection;
		RID depth_buffer;
		RID depth_fb;

		struct Reflection {
			RID owner;
			ReflectionData data;
			RID fbs[6];
		};

		Vector<Reflection> reflections;
	};

	RID_Owner<ReflectionAtlas> reflection_atlas_owner;

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
		uint32_t render_index = 0;

		Transform transform;
	};

	mutable RID_Owner<ReflectionProbeInstance> reflection_probe_instance_owner;

	/* DECAL INSTANCE */

	struct DecalInstance {
		RID decal;
		Transform transform;
	};

	mutable RID_Owner<DecalInstance> decal_instance_owner;

	/* GIPROBE INSTANCE */

	struct GIProbeLight {
		uint32_t type;
		float energy;
		float radius;
		float attenuation;

		float color[3];
		float spot_angle_radians;

		float position[3];
		float spot_attenuation;

		float direction[3];
		uint32_t has_shadow;
	};

	struct GIProbePushConstant {
		int32_t limits[3];
		uint32_t stack_size;

		float emission_scale;
		float propagation;
		float dynamic_range;
		uint32_t light_count;

		uint32_t cell_offset;
		uint32_t cell_count;
		float aniso_strength;
		uint32_t pad;
	};

	struct GIProbeDynamicPushConstant {
		int32_t limits[3];
		uint32_t light_count;
		int32_t x_dir[3];
		float z_base;
		int32_t y_dir[3];
		float z_sign;
		int32_t z_dir[3];
		float pos_multiplier;
		uint32_t rect_pos[2];
		uint32_t rect_size[2];
		uint32_t prev_rect_ofs[2];
		uint32_t prev_rect_size[2];
		uint32_t flip_x;
		uint32_t flip_y;
		float dynamic_range;
		uint32_t on_mipmap;
		float propagation;
		float pad[3];
	};

	struct GIProbeInstance {
		RID probe;
		RID texture;
		RID write_buffer;

		struct Mipmap {
			RID texture;
			RID uniform_set;
			RID second_bounce_uniform_set;
			RID write_uniform_set;
			uint32_t level;
			uint32_t cell_offset;
			uint32_t cell_count;
		};
		Vector<Mipmap> mipmaps;

		struct DynamicMap {
			RID texture; //color normally, or emission on first pass
			RID fb_depth; //actual depth buffer for the first pass, float depth for later passes
			RID depth; //actual depth buffer for the first pass, float depth for later passes
			RID normal; //normal buffer for the first pass
			RID albedo; //emission buffer for the first pass
			RID orm; //orm buffer for the first pass
			RID fb; //used for rendering, only valid on first map
			RID uniform_set;
			uint32_t size;
			int mipmap; // mipmap to write to, -1 if no mipmap assigned
		};

		Vector<DynamicMap> dynamic_maps;

		int slot = -1;
		uint32_t last_probe_version = 0;
		uint32_t last_probe_data_version = 0;

		//uint64_t last_pass = 0;
		uint32_t render_index = 0;

		bool has_dynamic_object_data = false;

		Transform transform;
	};

	GIProbeLight *gi_probe_lights;
	uint32_t gi_probe_max_lights;
	RID gi_probe_lights_uniform;

	enum {
		GI_PROBE_SHADER_VERSION_COMPUTE_LIGHT,
		GI_PROBE_SHADER_VERSION_COMPUTE_SECOND_BOUNCE,
		GI_PROBE_SHADER_VERSION_COMPUTE_MIPMAP,
		GI_PROBE_SHADER_VERSION_WRITE_TEXTURE,
		GI_PROBE_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING,
		GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE,
		GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_PLOT,
		GI_PROBE_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT,
		GI_PROBE_SHADER_VERSION_MAX
	};
	GiprobeShaderRD giprobe_shader;
	RID giprobe_lighting_shader_version;
	RID giprobe_lighting_shader_version_shaders[GI_PROBE_SHADER_VERSION_MAX];
	RID giprobe_lighting_shader_version_pipelines[GI_PROBE_SHADER_VERSION_MAX];

	mutable RID_Owner<GIProbeInstance> gi_probe_instance_owner;

	RS::GIProbeQuality gi_probe_quality = RS::GI_PROBE_QUALITY_HIGH;

	enum {
		GI_PROBE_DEBUG_COLOR,
		GI_PROBE_DEBUG_LIGHT,
		GI_PROBE_DEBUG_EMISSION,
		GI_PROBE_DEBUG_LIGHT_FULL,
		GI_PROBE_DEBUG_MAX
	};

	struct GIProbeDebugPushConstant {
		float projection[16];
		uint32_t cell_offset;
		float dynamic_range;
		float alpha;
		uint32_t level;
		int32_t bounds[3];
		uint32_t pad;
	};

	GiprobeDebugShaderRD giprobe_debug_shader;
	RID giprobe_debug_shader_version;
	RID giprobe_debug_shader_version_shaders[GI_PROBE_DEBUG_MAX];
	RenderPipelineVertexFormatCacheRD giprobe_debug_shader_version_pipelines[GI_PROBE_DEBUG_MAX];
	RID giprobe_debug_uniform_set;

	/* SHADOW ATLAS */

	struct ShadowShrinkStage {
		RID texture;
		RID filter_texture;
		uint32_t size;
	};

	struct ShadowAtlas {
		enum {
			QUADRANT_SHIFT = 27,
			SHADOW_INDEX_MASK = (1 << QUADRANT_SHIFT) - 1,
			SHADOW_INVALID = 0xFFFFFFFF
		};

		struct Quadrant {
			uint32_t subdivision;

			struct Shadow {
				RID owner;
				uint64_t version;
				uint64_t fog_version; // used for fog
				uint64_t alloc_tick;

				Shadow() {
					version = 0;
					fog_version = 0;
					alloc_tick = 0;
				}
			};

			Vector<Shadow> shadows;

			Quadrant() {
				subdivision = 0; //not in use
			}

		} quadrants[4];

		int size_order[4] = { 0, 1, 2, 3 };
		uint32_t smallest_subdiv = 0;

		int size = 0;

		RID depth;
		RID fb; //for copying

		Map<RID, uint32_t> shadow_owners;

		Vector<ShadowShrinkStage> shrink_stages;
	};

	RID_Owner<ShadowAtlas> shadow_atlas_owner;

	bool _shadow_atlas_find_shadow(ShadowAtlas *shadow_atlas, int *p_in_quadrants, int p_quadrant_count, int p_current_subdiv, uint64_t p_tick, int &r_quadrant, int &r_shadow);

	RS::ShadowQuality shadows_quality = RS::SHADOW_QUALITY_MAX; //So it always updates when first set
	RS::ShadowQuality directional_shadow_quality = RS::SHADOW_QUALITY_MAX;
	float shadows_quality_radius = 1.0;
	float directional_shadow_quality_radius = 1.0;

	float *directional_penumbra_shadow_kernel;
	float *directional_soft_shadow_kernel;
	float *penumbra_shadow_kernel;
	float *soft_shadow_kernel;
	int directional_penumbra_shadow_samples = 0;
	int directional_soft_shadow_samples = 0;
	int penumbra_shadow_samples = 0;
	int soft_shadow_samples = 0;

	/* DIRECTIONAL SHADOW */

	struct DirectionalShadow {
		RID depth;

		int light_count = 0;
		int size = 0;
		int current_light = 0;

		Vector<ShadowShrinkStage> shrink_stages;

	} directional_shadow;

	void _allocate_shadow_shrink_stages(RID p_base, int p_base_size, Vector<ShadowShrinkStage> &shrink_stages, uint32_t p_target_size);
	void _clear_shadow_shrink_stages(Vector<ShadowShrinkStage> &shrink_stages);

	/* SHADOW CUBEMAPS */

	struct ShadowCubemap {
		RID cubemap;
		RID side_fb[6];
	};

	Map<int, ShadowCubemap> shadow_cubemaps;
	ShadowCubemap *_get_shadow_cubemap(int p_size);

	struct ShadowMap {
		RID depth;
		RID fb;
	};

	Map<Vector2i, ShadowMap> shadow_maps;
	ShadowMap *_get_shadow_map(const Size2i &p_size);

	void _create_shadow_cubemaps();

	/* LIGHT INSTANCE */

	struct LightInstance {
		struct ShadowTransform {
			CameraMatrix camera;
			Transform transform;
			float farplane;
			float split;
			float bias_scale;
			float shadow_texel_size;
			float range_begin;
			Rect2 atlas_rect;
			Vector2 uv_scale;
		};

		RS::LightType light_type = RS::LIGHT_DIRECTIONAL;

		ShadowTransform shadow_transform[4];

		AABB aabb;
		RID self;
		RID light;
		Transform transform;

		Vector3 light_vector;
		Vector3 spot_vector;
		float linear_att = 0.0;

		uint64_t shadow_pass = 0;
		uint64_t last_scene_pass = 0;
		uint64_t last_scene_shadow_pass = 0;
		uint64_t last_pass = 0;
		uint32_t light_index = 0;
		uint32_t light_directional_index = 0;

		uint32_t current_shadow_atlas_key = 0;

		Vector2 dp;

		Rect2 directional_rect;

		Set<RID> shadow_atlases; //shadow atlases where this light is registered

		LightInstance() {}
	};

	mutable RID_Owner<LightInstance> light_instance_owner;

	/* ENVIRONMENT */

	struct Environment {
		// BG
		RS::EnvironmentBG background = RS::ENV_BG_CLEAR_COLOR;
		RID sky;
		float sky_custom_fov = 0.0;
		Basis sky_orientation;
		Color bg_color;
		float bg_energy = 1.0;
		int canvas_max_layer = 0;
		RS::EnvironmentAmbientSource ambient_source = RS::ENV_AMBIENT_SOURCE_BG;
		Color ambient_light;
		float ambient_light_energy = 1.0;
		float ambient_sky_contribution = 1.0;
		RS::EnvironmentReflectionSource reflection_source = RS::ENV_REFLECTION_SOURCE_BG;
		Color ao_color;

		/// Tonemap

		RS::EnvironmentToneMapper tone_mapper;
		float exposure = 1.0;
		float white = 1.0;
		bool auto_exposure = false;
		float min_luminance = 0.2;
		float max_luminance = 8.0;
		float auto_exp_speed = 0.2;
		float auto_exp_scale = 0.5;
		uint64_t auto_exposure_version = 0;

		// Fog
		bool fog_enabled = false;
		Color fog_light_color = Color(0.5, 0.6, 0.7);
		float fog_light_energy = 1.0;
		float fog_sun_scatter = 0.0;
		float fog_density = 0.001;
		float fog_height = 0.0;
		float fog_height_density = 0.0; //can be negative to invert effect
		float fog_aerial_perspective = 0.0;

		/// Volumetric Fog
		///
		bool volumetric_fog_enabled = false;
		float volumetric_fog_density = 0.01;
		Color volumetric_fog_light = Color(0, 0, 0);
		float volumetric_fog_light_energy = 0.0;
		float volumetric_fog_length = 64.0;
		float volumetric_fog_detail_spread = 2.0;
		RS::EnvVolumetricFogShadowFilter volumetric_fog_shadow_filter = RS::ENV_VOLUMETRIC_FOG_SHADOW_FILTER_LOW;
		float volumetric_fog_gi_inject = 0.0;

		/// Glow

		bool glow_enabled = false;
		Vector<float> glow_levels;
		float glow_intensity = 0.8;
		float glow_strength = 1.0;
		float glow_bloom = 0.0;
		float glow_mix = 0.01;
		RS::EnvironmentGlowBlendMode glow_blend_mode = RS::ENV_GLOW_BLEND_MODE_SOFTLIGHT;
		float glow_hdr_bleed_threshold = 1.0;
		float glow_hdr_luminance_cap = 12.0;
		float glow_hdr_bleed_scale = 2.0;

		/// SSAO

		bool ssao_enabled = false;
		float ssao_radius = 1;
		float ssao_intensity = 1;
		float ssao_bias = 0.01;
		float ssao_direct_light_affect = 0.0;
		float ssao_ao_channel_affect = 0.0;
		float ssao_blur_edge_sharpness = 4.0;
		RS::EnvironmentSSAOBlur ssao_blur = RS::ENV_SSAO_BLUR_3x3;

		/// SSR
		///
		bool ssr_enabled = false;
		int ssr_max_steps = 64;
		float ssr_fade_in = 0.15;
		float ssr_fade_out = 2.0;
		float ssr_depth_tolerance = 0.2;

		/// SDFGI
		bool sdfgi_enabled = false;
		RS::EnvironmentSDFGICascades sdfgi_cascades;
		float sdfgi_min_cell_size = 0.2;
		bool sdfgi_use_occlusion = false;
		bool sdfgi_use_multibounce = false;
		bool sdfgi_read_sky_light = false;
		float sdfgi_energy = 1.0;
		float sdfgi_normal_bias = 1.1;
		float sdfgi_probe_bias = 1.1;
		RS::EnvironmentSDFGIYScale sdfgi_y_scale = RS::ENV_SDFGI_Y_SCALE_DISABLED;
	};

	RS::EnvironmentSSAOQuality ssao_quality = RS::ENV_SSAO_QUALITY_MEDIUM;
	bool ssao_half_size = false;
	bool glow_bicubic_upscale = false;
	bool glow_high_quality = false;
	RS::EnvironmentSSRRoughnessQuality ssr_roughness_quality = RS::ENV_SSR_ROUGNESS_QUALITY_LOW;

	static uint64_t auto_exposure_counter;

	mutable RID_Owner<Environment> environment_owner;

	/* CAMERA EFFECTS */

	struct CameraEffects {
		bool dof_blur_far_enabled = false;
		float dof_blur_far_distance = 10;
		float dof_blur_far_transition = 5;

		bool dof_blur_near_enabled = false;
		float dof_blur_near_distance = 2;
		float dof_blur_near_transition = 1;

		float dof_blur_amount = 0.1;

		bool override_exposure_enabled = false;
		float override_exposure = 1;
	};

	RS::DOFBlurQuality dof_blur_quality = RS::DOF_BLUR_QUALITY_MEDIUM;
	RS::DOFBokehShape dof_blur_bokeh_shape = RS::DOF_BOKEH_HEXAGON;
	bool dof_blur_use_jitter = false;
	RS::SubSurfaceScatteringQuality sss_quality = RS::SUB_SURFACE_SCATTERING_QUALITY_MEDIUM;
	float sss_scale = 0.05;
	float sss_depth_scale = 0.01;

	mutable RID_Owner<CameraEffects> camera_effects_owner;

	/* RENDER BUFFERS */

	struct SDFGI;
	struct VolumetricFog;

	struct RenderBuffers {
		enum {
			MAX_GIPROBES = 8
		};

		RenderBufferData *data = nullptr;
		int width = 0, height = 0;
		RS::ViewportMSAA msaa = RS::VIEWPORT_MSAA_DISABLED;
		RS::ViewportScreenSpaceAA screen_space_aa = RS::VIEWPORT_SCREEN_SPACE_AA_DISABLED;
		bool use_debanding = false;

		RID render_target;

		uint64_t auto_exposure_version = 1;

		RID texture; //main texture for rendering to, must be filled after done rendering
		RID depth_texture; //main depth texture

		RID gi_uniform_set;
		SDFGI *sdfgi = nullptr;
		VolumetricFog *volumetric_fog = nullptr;

		//built-in textures used for ping pong image processing and blurring
		struct Blur {
			RID texture;

			struct Mipmap {
				RID texture;
				int width;
				int height;
			};

			Vector<Mipmap> mipmaps;
		};

		Blur blur[2]; //the second one starts from the first mipmap

		struct Luminance {
			Vector<RID> reduce;
			RID current;
		} luminance;

		struct SSAO {
			RID depth;
			Vector<RID> depth_slices;
			RID ao[2];
			RID ao_full; //when using half-size
		} ssao;

		struct SSR {
			RID normal_scaled;
			RID depth_scaled;
			RID blur_radius[2];
		} ssr;

		RID giprobe_textures[MAX_GIPROBES];
		RID giprobe_buffer;
	};

	RID default_giprobe_buffer;

	/* SDFGI */

	struct SDFGI {
		enum {
			MAX_CASCADES = 8,
			CASCADE_SIZE = 128,
			PROBE_DIVISOR = 16,
			ANISOTROPY_SIZE = 6,
			MAX_DYNAMIC_LIGHTS = 128,
			MAX_STATIC_LIGHTS = 1024,
			LIGHTPROBE_OCT_SIZE = 6,
			SH_SIZE = 16
		};

		struct Cascade {
			struct UBO {
				float offset[3];
				float to_cell;
				int32_t probe_offset[3];
				uint32_t pad;
			};

			//cascade blocks are full-size for volume (128^3), half size for albedo/emission
			RID sdf_tex;
			RID light_tex;
			RID light_aniso_0_tex;
			RID light_aniso_1_tex;

			RID light_data;
			RID light_aniso_0_data;
			RID light_aniso_1_data;

			struct SolidCell { // this struct is unused, but remains as reference for size
				uint32_t position;
				uint32_t albedo;
				uint32_t static_light;
				uint32_t static_light_aniso;
			};

			RID solid_cell_dispatch_buffer; //buffer for indirect compute dispatch
			RID solid_cell_buffer;

			RID lightprobe_history_tex;
			RID lightprobe_average_tex;

			float cell_size;
			Vector3i position;

			static const Vector3i DIRTY_ALL;
			Vector3i dirty_regions; //(0,0,0 is not dirty, negative is refresh from the end, DIRTY_ALL is refresh all.

			RID sdf_store_uniform_set;
			RID sdf_direct_light_uniform_set;
			RID scroll_uniform_set;
			RID scroll_occlusion_uniform_set;
			RID integrate_uniform_set;
			RID lights_buffer;
		};

		//used for rendering (voxelization)
		RID render_albedo;
		RID render_emission;
		RID render_emission_aniso;
		RID render_occlusion[8];
		RID render_geom_facing;

		RID render_sdf[2];
		RID render_sdf_half[2];

		//used for ping pong processing in cascades
		RID sdf_initialize_uniform_set;
		RID sdf_initialize_half_uniform_set;
		RID jump_flood_uniform_set[2];
		RID jump_flood_half_uniform_set[2];
		RID sdf_upscale_uniform_set;
		int upscale_jfa_uniform_set_index;
		RID occlusion_uniform_set;

		uint32_t cascade_size = 128;

		LocalVector<Cascade> cascades;

		RID lightprobe_texture;
		RID lightprobe_data;
		RID occlusion_texture;
		RID occlusion_data;
		RID ambient_texture; //integrates with volumetric fog

		RID lightprobe_history_scroll; //used for scrolling lightprobes
		RID lightprobe_average_scroll; //used for scrolling lightprobes

		uint32_t history_size = 0;
		float solid_cell_ratio = 0;
		uint32_t solid_cell_count = 0;

		RS::EnvironmentSDFGICascades cascade_mode;
		float min_cell_size = 0;
		uint32_t probe_axis_count = 0; //amount of probes per axis, this is an odd number because it encloses endpoints

		RID debug_uniform_set;
		RID debug_probes_uniform_set;
		RID cascades_ubo;

		bool uses_occlusion = false;
		bool uses_multibounce = false;
		bool reads_sky = false;
		float energy = 1.0;
		float normal_bias = 1.1;
		float probe_bias = 1.1;
		RS::EnvironmentSDFGIYScale y_scale_mode = RS::ENV_SDFGI_Y_SCALE_DISABLED;

		float y_mult = 1.0;

		uint32_t render_pass = 0;
	};

	RS::EnvironmentSDFGIRayCount sdfgi_ray_count = RS::ENV_SDFGI_RAY_COUNT_16;
	RS::EnvironmentSDFGIFramesToConverge sdfgi_frames_to_converge = RS::ENV_SDFGI_CONVERGE_IN_10_FRAMES;
	float sdfgi_solid_cell_ratio = 0.25;
	Vector3 sdfgi_debug_probe_pos;
	Vector3 sdfgi_debug_probe_dir;
	bool sdfgi_debug_probe_enabled = false;
	Vector3i sdfgi_debug_probe_index;

	struct SDGIShader {
		enum SDFGIPreprocessShaderVersion {
			PRE_PROCESS_SCROLL,
			PRE_PROCESS_SCROLL_OCCLUSION,
			PRE_PROCESS_JUMP_FLOOD_INITIALIZE,
			PRE_PROCESS_JUMP_FLOOD_INITIALIZE_HALF,
			PRE_PROCESS_JUMP_FLOOD,
			PRE_PROCESS_JUMP_FLOOD_OPTIMIZED,
			PRE_PROCESS_JUMP_FLOOD_UPSCALE,
			PRE_PROCESS_OCCLUSION,
			PRE_PROCESS_STORE,
			PRE_PROCESS_MAX
		};

		struct PreprocessPushConstant {
			int32_t scroll[3];
			int32_t grid_size;

			int32_t probe_offset[3];
			int32_t step_size;

			int32_t half_size;
			uint32_t occlusion_index;
			int32_t cascade;
			uint32_t pad;
		};

		SdfgiPreprocessShaderRD preprocess;
		RID preprocess_shader;
		RID preprocess_pipeline[PRE_PROCESS_MAX];

		struct DebugPushConstant {
			float grid_size[3];
			uint32_t max_cascades;

			int32_t screen_size[2];
			uint32_t use_occlusion;
			float y_mult;

			float cam_extent[3];
			uint32_t probe_axis_size;

			float cam_transform[16];
		};

		SdfgiDebugShaderRD debug;
		RID debug_shader;
		RID debug_shader_version;
		RID debug_pipeline;

		enum ProbeDebugMode {
			PROBE_DEBUG_PROBES,
			PROBE_DEBUG_VISIBILITY,
			PROBE_DEBUG_MAX
		};

		struct DebugProbesPushConstant {
			float projection[16];

			uint32_t band_power;
			uint32_t sections_in_band;
			uint32_t band_mask;
			float section_arc;

			float grid_size[3];
			uint32_t cascade;

			uint32_t pad;
			float y_mult;
			int32_t probe_debug_index;
			int32_t probe_axis_size;
		};

		SdfgiDebugProbesShaderRD debug_probes;
		RID debug_probes_shader;
		RID debug_probes_shader_version;

		RenderPipelineVertexFormatCacheRD debug_probes_pipeline[PROBE_DEBUG_MAX];

		struct Light {
			float color[3];
			float energy;

			float direction[3];
			uint32_t has_shadow;

			float position[3];
			float attenuation;

			uint32_t type;
			float spot_angle;
			float spot_attenuation;
			float radius;

			float shadow_color[4];
		};

		struct DirectLightPushConstant {
			float grid_size[3];
			uint32_t max_cascades;

			uint32_t cascade;
			uint32_t light_count;
			uint32_t process_offset;
			uint32_t process_increment;

			int32_t probe_axis_size;
			uint32_t multibounce;
			float y_mult;
			uint32_t pad;
		};

		enum {
			DIRECT_LIGHT_MODE_STATIC,
			DIRECT_LIGHT_MODE_DYNAMIC,
			DIRECT_LIGHT_MODE_MAX
		};
		SdfgiDirectLightShaderRD direct_light;
		RID direct_light_shader;
		RID direct_light_pipeline[DIRECT_LIGHT_MODE_MAX];

		enum {
			INTEGRATE_MODE_PROCESS,
			INTEGRATE_MODE_STORE,
			INTEGRATE_MODE_SCROLL,
			INTEGRATE_MODE_SCROLL_STORE,
			INTEGRATE_MODE_MAX
		};
		struct IntegratePushConstant {
			enum {
				SKY_MODE_DISABLED,
				SKY_MODE_COLOR,
				SKY_MODE_SKY,
			};

			float grid_size[3];
			uint32_t max_cascades;

			uint32_t probe_axis_size;
			uint32_t cascade;
			uint32_t history_index;
			uint32_t history_size;

			uint32_t ray_count;
			float ray_bias;
			int32_t image_size[2];

			int32_t world_offset[3];
			uint32_t sky_mode;

			int32_t scroll[3];
			float sky_energy;

			float sky_color[3];
			float y_mult;

			uint32_t store_ambient_texture;
			uint32_t pad[3];
		};

		SdfgiIntegrateShaderRD integrate;
		RID integrate_shader;
		RID integrate_pipeline[INTEGRATE_MODE_MAX];

		RID integrate_default_sky_uniform_set;

	} sdfgi_shader;

	void _sdfgi_erase(RenderBuffers *rb);
	int _sdfgi_get_pending_region_data(RID p_render_buffers, int p_region, Vector3i &r_local_offset, Vector3i &r_local_size, AABB &r_bounds) const;
	void _sdfgi_update_cascades(RID p_render_buffers);

	/* GI */

	struct GI {
		struct SDFGIData {
			float grid_size[3];
			uint32_t max_cascades;

			uint32_t use_occlusion;
			int32_t probe_axis_size;
			float probe_to_uvw;
			float normal_bias;

			float lightprobe_tex_pixel_size[3];
			float energy;

			float lightprobe_uv_offset[3];
			float y_mult;

			float occlusion_clamp[3];
			uint32_t pad3;

			float occlusion_renormalize[3];
			uint32_t pad4;

			float cascade_probe_size[3];
			uint32_t pad5;

			struct ProbeCascadeData {
				float position[3]; //offset of (0,0,0) in world coordinates
				float to_probe; // 1/bounds * grid_size
				int32_t probe_world_offset[3];
				float to_cell; // 1/bounds * grid_size
			};

			ProbeCascadeData cascades[SDFGI::MAX_CASCADES];
		};

		struct GIProbeData {
			float xform[16];
			float bounds[3];
			float dynamic_range;

			float bias;
			float normal_bias;
			uint32_t blend_ambient;
			uint32_t texture_slot;

			float anisotropy_strength;
			float ao;
			float ao_size;
			uint32_t mipmaps;
		};

		struct PushConstant {
			int32_t screen_size[2];
			float z_near;
			float z_far;

			float proj_info[4];

			uint32_t max_giprobes;
			uint32_t high_quality_vct;
			uint32_t use_sdfgi;
			uint32_t orthogonal;

			float ao_color[3];
			uint32_t pad;

			float cam_rotation[12];
		};

		RID sdfgi_ubo;
		enum {
			MODE_MAX = 1
		};

		GiShaderRD shader;
		RID shader_version;
		RID pipelines[MODE_MAX];
	} gi;

	bool screen_space_roughness_limiter = false;
	float screen_space_roughness_limiter_amount = 0.25;
	float screen_space_roughness_limiter_limit = 0.18;

	mutable RID_Owner<RenderBuffers> render_buffers_owner;

	void _free_render_buffer_data(RenderBuffers *rb);
	void _allocate_blur_textures(RenderBuffers *rb);
	void _allocate_luminance_textures(RenderBuffers *rb);

	void _render_buffers_debug_draw(RID p_render_buffers, RID p_shadow_atlas);
	void _render_buffers_post_process_and_tonemap(RID p_render_buffers, RID p_environment, RID p_camera_effects, const CameraMatrix &p_projection);
	void _sdfgi_debug_draw(RID p_render_buffers, const CameraMatrix &p_projection, const Transform &p_transform);

	/* Cluster */

	struct Cluster {
		/* Scene State UBO */

		struct ReflectionData { //should always be 128 bytes
			float box_extents[3];
			float index;
			float box_offset[3];
			uint32_t mask;
			float params[4]; // intensity, 0, interior , boxproject
			float ambient[3]; // ambient color,
			uint32_t ambient_mode;
			float local_matrix[16]; // up to here for spot and omni, rest is for directional
		};

		struct LightData {
			float position[3];
			float inv_radius;
			float direction[3];
			float size;
			uint16_t attenuation_energy[2]; //16 bits attenuation, then energy
			uint8_t color_specular[4]; //rgb color, a specular (8 bit unorm)
			uint16_t cone_attenuation_angle[2]; // attenuation and angle, (16bit float)
			uint8_t shadow_color_enabled[4]; //shadow rgb color, a>0.5 enabled (8bit unorm)
			float atlas_rect[4]; // in omni, used for atlas uv, in spot, used for projector uv
			float shadow_matrix[16];
			float shadow_bias;
			float shadow_normal_bias;
			float transmittance_bias;
			float soft_shadow_size;
			float soft_shadow_scale;
			uint32_t mask;
			float shadow_volumetric_fog_fade;
			uint32_t pad;
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
			uint32_t shadow_enabled;
			float fade_from;
			float fade_to;
			uint32_t pad[3];
			float shadow_volumetric_fog_fade;
			float shadow_bias[4];
			float shadow_normal_bias[4];
			float shadow_transmittance_bias[4];
			float shadow_z_range[4];
			float shadow_range_begin[4];
			float shadow_split_offsets[4];
			float shadow_matrices[4][16];
			float shadow_color1[4];
			float shadow_color2[4];
			float shadow_color3[4];
			float shadow_color4[4];
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

		ReflectionData *reflections;
		uint32_t max_reflections;
		RID reflection_buffer;
		uint32_t max_reflection_probes_per_instance;

		DecalData *decals;
		uint32_t max_decals;
		RID decal_buffer;

		LightData *lights;
		uint32_t max_lights;
		RID light_buffer;
		RID *lights_instances;
		Rect2i *lights_shadow_rect_cache;
		uint32_t lights_shadow_rect_cache_count = 0;

		DirectionalLightData *directional_lights;
		uint32_t max_directional_lights;
		RID directional_light_buffer;

		LightClusterBuilder builder;

	} cluster;

	struct VolumetricFog {
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t depth = 0;

		float length;
		float spread;

		RID light_density_map;
		RID fog_map;
		RID uniform_set;
		RID uniform_set2;
		RID sdfgi_uniform_set;
		RID sky_uniform_set;

		int last_shadow_filter = -1;
	};

	enum {
		VOLUMETRIC_FOG_SHADER_DENSITY,
		VOLUMETRIC_FOG_SHADER_DENSITY_WITH_SDFGI,
		VOLUMETRIC_FOG_SHADER_FILTER,
		VOLUMETRIC_FOG_SHADER_FOG,
		VOLUMETRIC_FOG_SHADER_MAX,
	};

	struct VolumetricFogShader {
		struct PushConstant {
			float fog_frustum_size_begin[2];
			float fog_frustum_size_end[2];

			float fog_frustum_end;
			float z_near;
			float z_far;
			uint32_t filter_axis;

			int32_t fog_volume_size[3];
			uint32_t directional_light_count;

			float light_energy[3];
			float base_density;

			float detail_spread;
			float gi_inject;
			uint32_t max_gi_probes;
			uint32_t pad;

			float cam_rotation[12];
		};

		VolumetricFogShaderRD shader;

		RID shader_version;
		RID pipelines[VOLUMETRIC_FOG_SHADER_MAX];

	} volumetric_fog;

	uint32_t volumetric_fog_depth = 128;
	uint32_t volumetric_fog_size = 128;
	bool volumetric_fog_filter_active = false;
	uint32_t volumetric_fog_directional_shadow_shrink = 512;
	uint32_t volumetric_fog_positional_shadow_shrink = 512;

	void _volumetric_fog_erase(RenderBuffers *rb);
	void _update_volumetric_fog(RID p_render_buffers, RID p_environment, const CameraMatrix &p_cam_projection, const Transform &p_cam_transform, RID p_shadow_atlas, int p_directional_light_count, bool p_use_directional_shadows, int p_positional_light_count, int p_gi_probe_count);

	RID shadow_sampler;

	uint64_t scene_pass = 0;
	uint64_t shadow_atlas_realloc_tolerance_msec = 500;

	struct SDFGICosineNeighbour {
		uint32_t neighbour;
		float weight;
	};

public:
	/* SHADOW ATLAS API */

	RID shadow_atlas_create();
	void shadow_atlas_set_size(RID p_atlas, int p_size);
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision);
	bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version);
	_FORCE_INLINE_ bool shadow_atlas_owns_light_instance(RID p_atlas, RID p_light_intance) {
		ShadowAtlas *atlas = shadow_atlas_owner.getornull(p_atlas);
		ERR_FAIL_COND_V(!atlas, false);
		return atlas->shadow_owners.has(p_light_intance);
	}

	_FORCE_INLINE_ RID shadow_atlas_get_texture(RID p_atlas) {
		ShadowAtlas *atlas = shadow_atlas_owner.getornull(p_atlas);
		ERR_FAIL_COND_V(!atlas, RID());
		return atlas->depth;
	}

	_FORCE_INLINE_ Size2i shadow_atlas_get_size(RID p_atlas) {
		ShadowAtlas *atlas = shadow_atlas_owner.getornull(p_atlas);
		ERR_FAIL_COND_V(!atlas, Size2i());
		return Size2(atlas->size, atlas->size);
	}

	void directional_shadow_atlas_set_size(int p_size);
	int get_directional_light_shadow_size(RID p_light_intance);
	void set_directional_shadow_count(int p_count);

	_FORCE_INLINE_ RID directional_shadow_get_texture() {
		return directional_shadow.depth;
	}

	_FORCE_INLINE_ Size2i directional_shadow_get_size() {
		return Size2i(directional_shadow.size, directional_shadow.size);
	}

	/* SDFGI UPDATE */

	int sdfgi_get_lightprobe_octahedron_size() const { return SDFGI::LIGHTPROBE_OCT_SIZE; }
	virtual void sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position);
	virtual int sdfgi_get_pending_region_count(RID p_render_buffers) const;
	virtual AABB sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const;
	virtual uint32_t sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const;
	virtual void sdfgi_update_probes(RID p_render_buffers, RID p_environment, const RID *p_directional_light_instances, uint32_t p_directional_light_count, const RID *p_positional_light_instances, uint32_t p_positional_light_count);
	RID sdfgi_get_ubo() const { return gi.sdfgi_ubo; }
	/* SKY API */

	RID sky_create();
	void sky_set_radiance_size(RID p_sky, int p_radiance_size);
	void sky_set_mode(RID p_sky, RS::SkyMode p_mode);
	void sky_set_material(RID p_sky, RID p_material);
	Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size);

	RID sky_get_radiance_texture_rd(RID p_sky) const;
	RID sky_get_radiance_uniform_set_rd(RID p_sky, RID p_shader, int p_set) const;
	RID sky_get_material(RID p_sky) const;

	/* ENVIRONMENT API */

	RID environment_create();

	void environment_set_background(RID p_env, RS::EnvironmentBG p_bg);
	void environment_set_sky(RID p_env, RID p_sky);
	void environment_set_sky_custom_fov(RID p_env, float p_scale);
	void environment_set_sky_orientation(RID p_env, const Basis &p_orientation);
	void environment_set_bg_color(RID p_env, const Color &p_color);
	void environment_set_bg_energy(RID p_env, float p_energy);
	void environment_set_canvas_max_layer(RID p_env, int p_max_layer);
	void environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient = RS::ENV_AMBIENT_SOURCE_BG, float p_energy = 1.0, float p_sky_contribution = 0.0, RS::EnvironmentReflectionSource p_reflection_source = RS::ENV_REFLECTION_SOURCE_BG, const Color &p_ao_color = Color());

	RS::EnvironmentBG environment_get_background(RID p_env) const;
	RID environment_get_sky(RID p_env) const;
	float environment_get_sky_custom_fov(RID p_env) const;
	Basis environment_get_sky_orientation(RID p_env) const;
	Color environment_get_bg_color(RID p_env) const;
	float environment_get_bg_energy(RID p_env) const;
	int environment_get_canvas_max_layer(RID p_env) const;
	Color environment_get_ambient_light_color(RID p_env) const;
	RS::EnvironmentAmbientSource environment_get_ambient_source(RID p_env) const;
	float environment_get_ambient_light_energy(RID p_env) const;
	float environment_get_ambient_sky_contribution(RID p_env) const;
	RS::EnvironmentReflectionSource environment_get_reflection_source(RID p_env) const;
	Color environment_get_ao_color(RID p_env) const;

	bool is_environment(RID p_env) const;

	void environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap);
	void environment_glow_set_use_bicubic_upscale(bool p_enable);
	void environment_glow_set_use_high_quality(bool p_enable);

	void environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective);
	bool environment_is_fog_enabled(RID p_env) const;
	Color environment_get_fog_light_color(RID p_env) const;
	float environment_get_fog_light_energy(RID p_env) const;
	float environment_get_fog_sun_scatter(RID p_env) const;
	float environment_get_fog_density(RID p_env) const;
	float environment_get_fog_height(RID p_env) const;
	float environment_get_fog_height_density(RID p_env) const;
	float environment_get_fog_aerial_perspective(RID p_env) const;

	void environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_light, float p_light_energy, float p_length, float p_detail_spread, float p_gi_inject, RS::EnvVolumetricFogShadowFilter p_shadow_filter);

	virtual void environment_set_volumetric_fog_volume_size(int p_size, int p_depth);
	virtual void environment_set_volumetric_fog_filter_active(bool p_enable);
	virtual void environment_set_volumetric_fog_directional_shadow_shrink_size(int p_shrink_size);
	virtual void environment_set_volumetric_fog_positional_shadow_shrink_size(int p_shrink_size);

	void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance);
	void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_bias, float p_light_affect, float p_ao_channel_affect, RS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness);
	void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size);
	bool environment_is_ssao_enabled(RID p_env) const;
	float environment_get_ssao_ao_affect(RID p_env) const;
	float environment_get_ssao_light_affect(RID p_env) const;
	bool environment_is_ssr_enabled(RID p_env) const;
	bool environment_is_sdfgi_enabled(RID p_env) const;

	virtual void environment_set_sdfgi(RID p_env, bool p_enable, RS::EnvironmentSDFGICascades p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, bool p_use_multibounce, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias);
	virtual void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count);
	virtual void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames);

	void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality);
	RS::EnvironmentSSRRoughnessQuality environment_get_ssr_roughness_quality() const;

	void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale);
	void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, RID p_ramp) {}

	virtual Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size);

	virtual RID camera_effects_create();

	virtual void camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter);
	virtual void camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape);

	virtual void camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount);
	virtual void camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure);

	RID light_instance_create(RID p_light);
	void light_instance_set_transform(RID p_light_instance, const Transform &p_transform);
	void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb);
	void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2());
	void light_instance_mark_visible(RID p_light_instance);

	_FORCE_INLINE_ RID light_instance_get_base_light(RID p_light_instance) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->light;
	}

	_FORCE_INLINE_ Transform light_instance_get_base_transform(RID p_light_instance) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->transform;
	}

	_FORCE_INLINE_ Rect2 light_instance_get_shadow_atlas_rect(RID p_light_instance, RID p_shadow_atlas) {
		ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
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

		uint32_t width = shadow_size;
		uint32_t height = shadow_size;

		return Rect2(x / float(shadow_atlas->size), y / float(shadow_atlas->size), width / float(shadow_atlas->size), height / float(shadow_atlas->size));
	}

	_FORCE_INLINE_ CameraMatrix light_instance_get_shadow_camera(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].camera;
	}

	_FORCE_INLINE_ float light_instance_get_shadow_texel_size(RID p_light_instance, RID p_shadow_atlas) {
#ifdef DEBUG_ENABLED
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		ERR_FAIL_COND_V(!li->shadow_atlases.has(p_shadow_atlas), 0);
#endif
		ShadowAtlas *shadow_atlas = shadow_atlas_owner.getornull(p_shadow_atlas);
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

	_FORCE_INLINE_ Transform
	light_instance_get_shadow_transform(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].transform;
	}
	_FORCE_INLINE_ float light_instance_get_shadow_bias_scale(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].bias_scale;
	}
	_FORCE_INLINE_ float light_instance_get_shadow_range(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].farplane;
	}
	_FORCE_INLINE_ float light_instance_get_shadow_range_begin(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].range_begin;
	}

	_FORCE_INLINE_ Vector2 light_instance_get_shadow_uv_scale(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].uv_scale;
	}

	_FORCE_INLINE_ Rect2 light_instance_get_directional_shadow_atlas_rect(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].atlas_rect;
	}

	_FORCE_INLINE_ float light_instance_get_directional_shadow_split(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].split;
	}

	_FORCE_INLINE_ float light_instance_get_directional_shadow_texel_size(RID p_light_instance, int p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->shadow_transform[p_index].shadow_texel_size;
	}

	_FORCE_INLINE_ void light_instance_set_render_pass(RID p_light_instance, uint64_t p_pass) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		li->last_pass = p_pass;
	}

	_FORCE_INLINE_ uint64_t light_instance_get_render_pass(RID p_light_instance) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->last_pass;
	}

	_FORCE_INLINE_ void light_instance_set_index(RID p_light_instance, uint32_t p_index) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		li->light_index = p_index;
	}

	_FORCE_INLINE_ uint32_t light_instance_get_index(RID p_light_instance) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->light_index;
	}

	_FORCE_INLINE_ RS::LightType light_instance_get_type(RID p_light_instance) {
		LightInstance *li = light_instance_owner.getornull(p_light_instance);
		return li->light_type;
	}

	virtual RID reflection_atlas_create();
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count);
	_FORCE_INLINE_ RID reflection_atlas_get_texture(RID p_ref_atlas) {
		ReflectionAtlas *atlas = reflection_atlas_owner.getornull(p_ref_atlas);
		ERR_FAIL_COND_V(!atlas, RID());
		return atlas->reflection;
	}

	virtual RID reflection_probe_instance_create(RID p_probe);
	virtual void reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform);
	virtual void reflection_probe_release_atlas_index(RID p_instance);
	virtual bool reflection_probe_instance_needs_redraw(RID p_instance);
	virtual bool reflection_probe_instance_has_reflection(RID p_instance);
	virtual bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas);
	virtual bool reflection_probe_instance_postprocess_step(RID p_instance);

	uint32_t reflection_probe_instance_get_resolution(RID p_instance);
	RID reflection_probe_instance_get_framebuffer(RID p_instance, int p_index);
	RID reflection_probe_instance_get_depth_framebuffer(RID p_instance, int p_index);

	_FORCE_INLINE_ RID reflection_probe_instance_get_probe(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND_V(!rpi, RID());

		return rpi->probe;
	}

	_FORCE_INLINE_ void reflection_probe_instance_set_render_index(RID p_instance, uint32_t p_render_index) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND(!rpi);
		rpi->render_index = p_render_index;
	}

	_FORCE_INLINE_ uint32_t reflection_probe_instance_get_render_index(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND_V(!rpi, 0);

		return rpi->render_index;
	}

	_FORCE_INLINE_ void reflection_probe_instance_set_render_pass(RID p_instance, uint32_t p_render_pass) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND(!rpi);
		rpi->last_pass = p_render_pass;
	}

	_FORCE_INLINE_ uint32_t reflection_probe_instance_get_render_pass(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND_V(!rpi, 0);

		return rpi->last_pass;
	}

	_FORCE_INLINE_ Transform reflection_probe_instance_get_transform(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND_V(!rpi, Transform());

		return rpi->transform;
	}

	_FORCE_INLINE_ int reflection_probe_instance_get_atlas_index(RID p_instance) {
		ReflectionProbeInstance *rpi = reflection_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND_V(!rpi, -1);

		return rpi->atlas_index;
	}

	virtual RID decal_instance_create(RID p_decal);
	virtual void decal_instance_set_transform(RID p_decal, const Transform &p_transform);

	_FORCE_INLINE_ RID decal_instance_get_base(RID p_decal) const {
		DecalInstance *decal = decal_instance_owner.getornull(p_decal);
		return decal->decal;
	}

	_FORCE_INLINE_ Transform decal_instance_get_transform(RID p_decal) const {
		DecalInstance *decal = decal_instance_owner.getornull(p_decal);
		return decal->transform;
	}

	RID gi_probe_instance_create(RID p_base);
	void gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform);
	bool gi_probe_needs_update(RID p_probe) const;
	void gi_probe_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, int p_dynamic_object_count, InstanceBase **p_dynamic_objects);

	void gi_probe_set_quality(RS::GIProbeQuality p_quality) { gi_probe_quality = p_quality; }

	_FORCE_INLINE_ uint32_t gi_probe_instance_get_slot(RID p_probe) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
		return gi_probe->slot;
	}
	_FORCE_INLINE_ RID gi_probe_instance_get_base_probe(RID p_probe) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
		return gi_probe->probe;
	}
	_FORCE_INLINE_ Transform gi_probe_instance_get_transform_to_cell(RID p_probe) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
		return storage->gi_probe_get_to_cell_xform(gi_probe->probe) * gi_probe->transform.affine_inverse();
	}

	_FORCE_INLINE_ RID gi_probe_instance_get_texture(RID p_probe) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_probe);
		return gi_probe->texture;
	}

	_FORCE_INLINE_ void gi_probe_instance_set_render_index(RID p_instance, uint32_t p_render_index) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND(!gi_probe);
		gi_probe->render_index = p_render_index;
	}

	_FORCE_INLINE_ uint32_t gi_probe_instance_get_render_index(RID p_instance) {
		GIProbeInstance *gi_probe = gi_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND_V(!gi_probe, 0);

		return gi_probe->render_index;
	}
	/*
	_FORCE_INLINE_ void gi_probe_instance_set_render_pass(RID p_instance, uint32_t p_render_pass) {
		GIProbeInstance *g_probe = gi_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND(!g_probe);
		g_probe->last_pass = p_render_pass;
	}

	_FORCE_INLINE_ uint32_t gi_probe_instance_get_render_pass(RID p_instance) {
		GIProbeInstance *g_probe = gi_probe_instance_owner.getornull(p_instance);
		ERR_FAIL_COND_V(!g_probe, 0);

		return g_probe->last_pass;
	}
*/
	RID render_buffers_create();
	void render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding);

	RID render_buffers_get_ao_texture(RID p_render_buffers);
	RID render_buffers_get_back_buffer_texture(RID p_render_buffers);
	RID render_buffers_get_gi_probe_buffer(RID p_render_buffers);
	RID render_buffers_get_default_gi_probe_buffer();

	uint32_t render_buffers_get_sdfgi_cascade_count(RID p_render_buffers) const;
	bool render_buffers_is_sdfgi_enabled(RID p_render_buffers) const;
	RID render_buffers_get_sdfgi_irradiance_probes(RID p_render_buffers) const;
	Vector3 render_buffers_get_sdfgi_cascade_offset(RID p_render_buffers, uint32_t p_cascade) const;
	Vector3i render_buffers_get_sdfgi_cascade_probe_offset(RID p_render_buffers, uint32_t p_cascade) const;
	float render_buffers_get_sdfgi_cascade_probe_size(RID p_render_buffers, uint32_t p_cascade) const;
	float render_buffers_get_sdfgi_normal_bias(RID p_render_buffers) const;
	uint32_t render_buffers_get_sdfgi_cascade_probe_count(RID p_render_buffers) const;
	uint32_t render_buffers_get_sdfgi_cascade_size(RID p_render_buffers) const;
	bool render_buffers_is_sdfgi_using_occlusion(RID p_render_buffers) const;
	float render_buffers_get_sdfgi_energy(RID p_render_buffers) const;
	RID render_buffers_get_sdfgi_occlusion_texture(RID p_render_buffers) const;

	bool render_buffers_has_volumetric_fog(RID p_render_buffers) const;
	RID render_buffers_get_volumetric_fog_texture(RID p_render_buffers);
	RID render_buffers_get_volumetric_fog_sky_uniform_set(RID p_render_buffers);
	float render_buffers_get_volumetric_fog_end(RID p_render_buffers);
	float render_buffers_get_volumetric_fog_detail_spread(RID p_render_buffers);

	void render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, RID *p_decal_cull_result, int p_decal_cull_count, InstanceBase **p_lightmap_cull_result, int p_lightmap_cull_count, RID p_environment, RID p_shadow_atlas, RID p_camera_effects, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass);

	void render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count);

	void render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region);

	void render_sdfgi(RID p_render_buffers, int p_region, InstanceBase **p_cull_result, int p_cull_count);
	void render_sdfgi_static_lights(RID p_render_buffers, uint32_t p_cascade_count, const uint32_t *p_cascade_indices, const RID **p_positional_light_cull_result, const uint32_t *p_positional_light_cull_count);

	void render_particle_collider_heightfield(RID p_collider, const Transform &p_transform, InstanceBase **p_cull_result, int p_cull_count);

	virtual void set_scene_pass(uint64_t p_pass) {
		scene_pass = p_pass;
	}
	_FORCE_INLINE_ uint64_t get_scene_pass() {
		return scene_pass;
	}

	virtual void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit);
	virtual bool screen_space_roughness_limiter_is_active() const;
	virtual float screen_space_roughness_limiter_get_amount() const;
	virtual float screen_space_roughness_limiter_get_limit() const;

	virtual void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality);
	RS::SubSurfaceScatteringQuality sub_surface_scattering_get_quality() const;
	virtual void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale);

	virtual void shadows_quality_set(RS::ShadowQuality p_quality);
	virtual void directional_shadow_quality_set(RS::ShadowQuality p_quality);
	_FORCE_INLINE_ RS::ShadowQuality shadows_quality_get() const { return shadows_quality; }
	_FORCE_INLINE_ RS::ShadowQuality directional_shadow_quality_get() const { return directional_shadow_quality; }
	_FORCE_INLINE_ float shadows_quality_radius_get() const { return shadows_quality_radius; }
	_FORCE_INLINE_ float directional_shadow_quality_radius_get() const { return directional_shadow_quality_radius; }

	_FORCE_INLINE_ float *directional_penumbra_shadow_kernel_get() { return directional_penumbra_shadow_kernel; }
	_FORCE_INLINE_ float *directional_soft_shadow_kernel_get() { return directional_soft_shadow_kernel; }
	_FORCE_INLINE_ float *penumbra_shadow_kernel_get() { return penumbra_shadow_kernel; }
	_FORCE_INLINE_ float *soft_shadow_kernel_get() { return soft_shadow_kernel; }

	_FORCE_INLINE_ int directional_penumbra_shadow_samples_get() const { return directional_penumbra_shadow_samples; }
	_FORCE_INLINE_ int directional_soft_shadow_samples_get() const { return directional_soft_shadow_samples; }
	_FORCE_INLINE_ int penumbra_shadow_samples_get() const { return penumbra_shadow_samples; }
	_FORCE_INLINE_ int soft_shadow_samples_get() const { return soft_shadow_samples; }

	int get_roughness_layers() const;
	bool is_using_radiance_cubemap_array() const;

	virtual TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size);

	virtual bool free(RID p_rid);

	virtual void update();

	virtual void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw);
	_FORCE_INLINE_ RS::ViewportDebugDraw get_debug_draw_mode() const {
		return debug_draw;
	}

	virtual void set_time(double p_time, double p_step);

	RID get_cluster_builder_texture();
	RID get_cluster_builder_indices_buffer();
	RID get_reflection_probe_buffer();
	RID get_positional_light_buffer();
	RID get_directional_light_buffer();
	RID get_decal_buffer();
	int get_max_directional_lights() const;

	void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir);

	RasterizerSceneRD(RasterizerStorageRD *p_storage);
	~RasterizerSceneRD();
};

#endif // RASTERIZER_SCENE_RD_H
