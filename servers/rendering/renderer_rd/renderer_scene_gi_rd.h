/*************************************************************************/
/*  renderer_scene_gi_rd.h                                               */
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

#ifndef RENDERING_SERVER_SCENE_GI_RD_H
#define RENDERING_SERVER_SCENE_GI_RD_H

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/renderer_scene_environment_rd.h"
#include "servers/rendering/renderer_rd/renderer_scene_sky_rd.h"
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"
#include "servers/rendering/renderer_rd/shaders/gi.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sdfgi_debug.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sdfgi_debug_probes.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sdfgi_direct_light.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sdfgi_integrate.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/sdfgi_preprocess.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/voxel_gi.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/voxel_gi_debug.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering/rendering_device.h"

// Forward declare RenderDataRD and RendererSceneRenderRD so we can pass it into some of our methods, these classes are pretty tightly bound
struct RenderDataRD;
class RendererSceneRenderRD;

class RendererSceneGIRD {
private:
	RendererStorageRD *storage;

	/* VOXEL_GI INSTANCE */

	struct VoxelGILight {
		uint32_t type;
		float energy;
		float radius;
		float attenuation;

		float color[3];
		float cos_spot_angle;

		float position[3];
		float inv_spot_attenuation;

		float direction[3];
		uint32_t has_shadow;
	};

	struct VoxelGIPushConstant {
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

	struct VoxelGIDynamicPushConstant {
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

	VoxelGILight *voxel_gi_lights = nullptr;
	uint32_t voxel_gi_max_lights = 32;
	RID voxel_gi_lights_uniform;

	enum {
		VOXEL_GI_SHADER_VERSION_COMPUTE_LIGHT,
		VOXEL_GI_SHADER_VERSION_COMPUTE_SECOND_BOUNCE,
		VOXEL_GI_SHADER_VERSION_COMPUTE_MIPMAP,
		VOXEL_GI_SHADER_VERSION_WRITE_TEXTURE,
		VOXEL_GI_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING,
		VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE,
		VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_PLOT,
		VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT,
		VOXEL_GI_SHADER_VERSION_MAX
	};

	VoxelGiShaderRD voxel_gi_shader;
	RID voxel_gi_lighting_shader_version;
	RID voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_MAX];
	RID voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_MAX];

	enum {
		VOXEL_GI_DEBUG_COLOR,
		VOXEL_GI_DEBUG_LIGHT,
		VOXEL_GI_DEBUG_EMISSION,
		VOXEL_GI_DEBUG_LIGHT_FULL,
		VOXEL_GI_DEBUG_MAX
	};

	struct VoxelGIDebugPushConstant {
		float projection[16];
		uint32_t cell_offset;
		float dynamic_range;
		float alpha;
		uint32_t level;
		int32_t bounds[3];
		uint32_t pad;
	};

	VoxelGiDebugShaderRD voxel_gi_debug_shader;
	RID voxel_gi_debug_shader_version;
	RID voxel_gi_debug_shader_version_shaders[VOXEL_GI_DEBUG_MAX];
	PipelineCacheRD voxel_gi_debug_shader_version_pipelines[VOXEL_GI_DEBUG_MAX];
	RID voxel_gi_debug_uniform_set;

	/* SDFGI */

	struct SDFGIShader {
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

		PipelineCacheRD debug_probes_pipeline[PROBE_DEBUG_MAX];

		struct Light {
			float color[3];
			float energy;

			float direction[3];
			uint32_t has_shadow;

			float position[3];
			float attenuation;

			uint32_t type;
			float cos_spot_angle;
			float inv_spot_attenuation;
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
			float bounce_feedback;
			float y_mult;
			uint32_t use_occlusion;
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

public:
	/* VOXEL_GI INSTANCE */

	//@TODO VoxelGIInstance is still directly used in the render code, we'll address this when we refactor the render code itself.

	struct VoxelGIInstance {
		// access to our containers
		RendererStorageRD *storage;
		RendererSceneGIRD *gi;

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

		Transform3D transform;

		void update(bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RendererSceneRender::GeometryInstance *> &p_dynamic_objects, RendererSceneRenderRD *p_scene_render);
		void debug(RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha);
	};

	mutable RID_Owner<VoxelGIInstance> voxel_gi_instance_owner;

	_FORCE_INLINE_ VoxelGIInstance *get_probe_instance(RID p_probe) const {
		return voxel_gi_instance_owner.get_or_null(p_probe);
	};

	_FORCE_INLINE_ RID voxel_gi_instance_get_texture(RID p_probe) {
		VoxelGIInstance *voxel_gi = get_probe_instance(p_probe);
		ERR_FAIL_COND_V(!voxel_gi, RID());
		return voxel_gi->texture;
	};

	RS::VoxelGIQuality voxel_gi_quality = RS::VOXEL_GI_QUALITY_HIGH;

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

			bool all_dynamic_lights_dirty = true;
		};

		// access to our containers
		RendererStorageRD *storage;
		RendererSceneGIRD *gi;

		// used for rendering (voxelization)
		RID render_albedo;
		RID render_emission;
		RID render_emission_aniso;
		RID render_occlusion[8];
		RID render_geom_facing;

		RID render_sdf[2];
		RID render_sdf_half[2];

		// used for ping pong processing in cascades
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
		float bounce_feedback = 0.0;
		bool reads_sky = false;
		float energy = 1.0;
		float normal_bias = 1.1;
		float probe_bias = 1.1;
		RS::EnvironmentSDFGIYScale y_scale_mode = RS::ENV_SDFGI_Y_SCALE_DISABLED;

		float y_mult = 1.0;

		uint32_t render_pass = 0;

		int32_t cascade_dynamic_light_count[SDFGI::MAX_CASCADES]; //used dynamically
		RID integrate_sky_uniform_set;

		void create(RendererSceneEnvironmentRD *p_env, const Vector3 &p_world_position, uint32_t p_requested_history_size, RendererSceneGIRD *p_gi);
		void erase();
		void update(RendererSceneEnvironmentRD *p_env, const Vector3 &p_world_position);
		void update_light();
		void update_probes(RendererSceneEnvironmentRD *p_env, RendererSceneSkyRD::Sky *p_sky);
		void store_probes();
		int get_pending_region_data(int p_region, Vector3i &r_local_offset, Vector3i &r_local_size, AABB &r_bounds) const;
		void update_cascades();

		void debug_draw(const CameraMatrix &p_projection, const Transform3D &p_transform, int p_width, int p_height, RID p_render_target, RID p_texture);
		void debug_probes(RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform);

		void pre_process_gi(const Transform3D &p_transform, RenderDataRD *p_render_data, RendererSceneRenderRD *p_scene_render);
		void render_region(RID p_render_buffers, int p_region, const PagedArray<RendererSceneRender::GeometryInstance *> &p_instances, RendererSceneRenderRD *p_scene_render);
		void render_static_lights(RID p_render_buffers, uint32_t p_cascade_count, const uint32_t *p_cascade_indices, const PagedArray<RID> *p_positional_light_cull_result, RendererSceneRenderRD *p_scene_render);
	};

	RS::EnvironmentSDFGIRayCount sdfgi_ray_count = RS::ENV_SDFGI_RAY_COUNT_16;
	RS::EnvironmentSDFGIFramesToConverge sdfgi_frames_to_converge = RS::ENV_SDFGI_CONVERGE_IN_10_FRAMES;
	RS::EnvironmentSDFGIFramesToUpdateLight sdfgi_frames_to_update_light = RS::ENV_SDFGI_UPDATE_LIGHT_IN_4_FRAMES;

	float sdfgi_solid_cell_ratio = 0.25;
	Vector3 sdfgi_debug_probe_pos;
	Vector3 sdfgi_debug_probe_dir;
	bool sdfgi_debug_probe_enabled = false;
	Vector3i sdfgi_debug_probe_index;

	/* SDFGI UPDATE */

	int sdfgi_get_lightprobe_octahedron_size() const { return SDFGI::LIGHTPROBE_OCT_SIZE; }

	/* GI */
	enum {
		MAX_VOXEL_GI_INSTANCES = 8
	};

	// Struct for use in render buffer
	struct RenderBuffersGI {
		RID voxel_gi_textures[MAX_VOXEL_GI_INSTANCES];
		RID voxel_gi_buffer;

		RID full_buffer;
		RID full_dispatch;
		RID full_mask;

		RID uniform_set;
		bool using_half_size_gi = false;
	};

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

	struct VoxelGIData {
		float xform[16]; // 64 - 64

		float bounds[3]; // 12 - 76
		float dynamic_range; // 4 - 80

		float bias; // 4 - 84
		float normal_bias; // 4 - 88
		uint32_t blend_ambient; // 4 - 92
		uint32_t mipmaps; // 4 - 96
	};

	struct PushConstant {
		int32_t screen_size[2];
		float z_near;
		float z_far;

		float proj_info[4];

		uint32_t max_voxel_gi_instances;
		uint32_t high_quality_vct;
		uint32_t orthogonal;
		uint32_t pad;

		float cam_rotation[12];
	};

	RID sdfgi_ubo;
	enum Mode {
		MODE_VOXEL_GI,
		MODE_SDFGI,
		MODE_COMBINED,
		MODE_HALF_RES_VOXEL_GI,
		MODE_HALF_RES_SDFGI,
		MODE_HALF_RES_COMBINED,
		MODE_MAX
	};

	RID default_voxel_gi_buffer;

	bool half_resolution = false;
	GiShaderRD shader;
	RID shader_version;
	RID pipelines[MODE_MAX];

	RendererSceneGIRD();
	~RendererSceneGIRD();

	void init(RendererStorageRD *p_storage, RendererSceneSkyRD *p_sky);
	void free();

	SDFGI *create_sdfgi(RendererSceneEnvironmentRD *p_env, const Vector3 &p_world_position, uint32_t p_requested_history_size);

	void setup_voxel_gi_instances(RID p_render_buffers, const Transform3D &p_transform, const PagedArray<RID> &p_voxel_gi_instances, uint32_t &r_voxel_gi_instances_used, RendererSceneRenderRD *p_scene_render);
	void process_gi(RID p_render_buffers, RID p_normal_roughness_buffer, RID p_voxel_gi_buffer, RID p_environment, const CameraMatrix &p_projection, const Transform3D &p_transform, const PagedArray<RID> &p_voxel_gi_instances, RendererSceneRenderRD *p_scene_render);

	RID voxel_gi_instance_create(RID p_base);
	void voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform);
	bool voxel_gi_needs_update(RID p_probe) const;
	void voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RendererSceneRender::GeometryInstance *> &p_dynamic_objects, RendererSceneRenderRD *p_scene_render);
	void debug_voxel_gi(RID p_voxel_gi, RD::DrawListID p_draw_list, RID p_framebuffer, const CameraMatrix &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha);
};

#endif /* !RENDERING_SERVER_SCENE_GI_RD_H */
