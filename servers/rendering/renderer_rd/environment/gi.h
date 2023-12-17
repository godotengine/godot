/**************************************************************************/
/*  gi.h                                                                  */
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

#ifndef GI_RD_H
#define GI_RD_H

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "servers/rendering/environment/renderer_gi.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/environment/sky.h"
#include "servers/rendering/renderer_rd/shaders/environment/gi.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_debug.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_debug_probes.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_direct_light.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_filter.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_integrate.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_preprocess.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/voxel_gi.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/environment/voxel_gi_debug.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_buffer_custom_data_rd.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/storage/utilities.h"

#define RB_SCOPE_GI SNAME("rbgi")
#define RB_SCOPE_HDDAGI SNAME("hddagi")

#define RB_TEX_AMBIENT SNAME("ambient")
#define RB_TEX_REFLECTION SNAME("reflection")
#define RB_TEX_AMBIENT_REFLECTION_BLEND SNAME("ambient_reflection_blend")

#define RB_TEX_AMBIENT_U32 SNAME("ambient_u32")
#define RB_TEX_REFLECTION_U32 SNAME("reflection_u32")

#define RB_TEX_REFLECTION_FILTERED SNAME("reflection_filtered")
#define RB_TEX_AMBIENT_REFLECTION_BLEND_FILTERED SNAME("ambient_reflection_blend_filtered")

#define RB_TEX_REFLECTION_U32_FILTERED SNAME("reflection_u32_filtered")

// Forward declare RenderDataRD and RendererSceneRenderRD so we can pass it into some of our methods, these classes are pretty tightly bound
class RenderDataRD;
class RendererSceneRenderRD;

namespace RendererRD {

class GI : public RendererGI {
public:
	/* VOXEL GI STORAGE */

	struct VoxelGI {
		RID octree_buffer;
		RID data_buffer;
		RID sdf_texture;

		uint32_t octree_buffer_size = 0;
		uint32_t data_buffer_size = 0;

		Vector<int> level_counts;

		int cell_count = 0;

		Transform3D to_cell_xform;
		AABB bounds;
		Vector3i octree_size;

		float dynamic_range = 2.0;
		float energy = 1.0;
		float baked_exposure = 1.0;
		float bias = 1.4;
		float normal_bias = 0.0;
		float propagation = 0.5;
		bool interior = false;
		bool use_two_bounces = true;

		uint32_t version = 1;
		uint32_t data_version = 1;

		Dependency dependency;
	};

	/* VOXEL_GI INSTANCE */

	//@TODO VoxelGIInstance is still directly used in the render code, we'll address this when we refactor the render code itself.

	struct VoxelGIInstance {
		// access to our containers
		GI *gi = nullptr;

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

		void update(bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RenderGeometryInstance *> &p_dynamic_objects);
		void debug(RD::DrawListID p_draw_list, RID p_framebuffer, const Projection &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha);
		void free_resources();
	};

private:
	static GI *singleton;

	/* VOXEL GI STORAGE */

	mutable RID_Owner<VoxelGI, true> voxel_gi_owner;

	/* VOXEL_GI INSTANCE */

	mutable RID_Owner<VoxelGIInstance> voxel_gi_instance_owner;

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

	/* HDDAGI */

	struct HDDAGIShader {
		enum HDDAGIPreprocessShaderVersion {
			PRE_PROCESS_REGION_STORE,
			PRE_PROCESS_LIGHT_STORE,
			PRE_PROCESS_LIGHT_SCROLL,
			PRE_PROCESS_OCCLUSION,
			PRE_PROCESS_OCCLUSION_STORE,
			PRE_PROCESS_LIGHTPROBE_SCROLL,
			PRE_PROCESS_LIGHTPROBE_NEIGHBOURS,
			PRE_PROCESS_LIGHTPROBE_GEOMETRY_PROXIMITY,
			PRE_PROCESS_LIGHTPROBE_UPDATE_FRAMES,
			PRE_PROCESS_MAX
		};

		struct PreprocessPushConstant {
			int32_t grid_size[3];
			uint32_t region_version;

			int32_t scroll[3];
			uint32_t cascade_count;

			int32_t offset[3];
			uint32_t probe_update_frames;

			int32_t limit[3];
			uint32_t cascade;

			int32_t region_world_pos[3];
			int32_t maximum_light_cells;

			int32_t probe_axis_size[3];
			uint32_t ray_hit_cache_frames;

			uint32_t upper_region_world_pos[3];
			int occlusion_offset;
		};

		HddagiPreprocessShaderRD preprocess;
		RID preprocess_shader;
		RID preprocess_shader_version[PRE_PROCESS_MAX];
		RID preprocess_pipeline[PRE_PROCESS_MAX];

		struct DebugPushConstant {
			float grid_size[3];
			uint32_t max_cascades;

			uint32_t screen_size;
			float esm_strength;
			float y_mult;
			float z_near;

			float inv_projection[3][4];
			float cam_basis[3][3];
			float cam_origin[3];
		};

		HddagiDebugShaderRD debug;
		RID debug_shader;
		RID debug_shader_version;
		RID debug_pipeline;

		enum ProbeDebugMode {
			PROBE_DEBUG_PROBES,
			PROBE_DEBUG_PROBES_MULTIVIEW,
			PROBE_DEBUG_OCCLUSION,
			PROBE_DEBUG_OCCLUSION_MULTIVIEW,
			PROBE_DEBUG_MAX
		};

		struct DebugProbesSceneData {
			float projection[2][16];
		};

		struct DebugProbesPushConstant {
			uint32_t band_power;
			uint32_t sections_in_band;
			uint32_t band_mask;
			float section_arc;

			float grid_size[3];
			uint32_t cascade;

			int32_t oct_size;
			float y_mult;
			int32_t probe_debug_index;
			uint32_t pad;

			int32_t probe_axis_size[3];
			uint32_t pad2;
		};

		HddagiDebugProbesShaderRD debug_probes;
		RID debug_probes_shader;
		RID debug_probes_shader_version[PROBE_DEBUG_MAX];
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
		};

		struct DirectLightPushConstant {
			int32_t grid_size[3];
			uint32_t max_cascades;

			uint32_t cascade;
			uint32_t light_count;
			uint32_t process_offset;
			uint32_t process_increment;

			float bounce_feedback;
			float y_mult;
			uint32_t use_occlusion;
			uint32_t probe_cell_size;

			int32_t probe_axis_size[3];
			uint32_t dirty_dynamic_update;
		};

		enum {
			DIRECT_LIGHT_MODE_STATIC,
			DIRECT_LIGHT_MODE_DYNAMIC,
			DIRECT_LIGHT_MODE_MAX
		};
		HddagiDirectLightShaderRD direct_light;
		RID direct_light_shader;
		RID direct_light_shader_version[DIRECT_LIGHT_MODE_MAX];
		RID direct_light_pipeline[DIRECT_LIGHT_MODE_MAX];

		enum {
			INTEGRATE_MODE_PROCESS,
			INTEGRATE_MODE_FILTER,
			INTEGRATE_MODE_CAMERA_VISIBILITY,
			INTEGRATE_MODE_MAX
		};

		struct IntegratePushConstant {
			enum {
				SKY_MODE_DISABLED,
				SKY_MODE_COLOR,
				SKY_MODE_SKY,
			};

			int32_t grid_size[3];
			uint32_t max_cascades;

			float ray_bias;
			uint32_t cascade;
			int32_t inactive_update_frames;
			uint32_t history_size;

			int32_t world_offset[3];
			uint32_t sky_mode;

			int32_t scroll[3];
			float sky_energy;

			float sky_color[3];
			float y_mult;

			uint32_t probe_axis_size[3];
			uint32_t store_ambient_texture;

			uint32_t pad[2];
			int32_t global_frame;
			uint32_t motion_accum; // Motion that happened since last update (bit 0 in X, bit 1 in Y, bit 2 in Z).
		};

		struct IntegrateCameraUBO {
			float planes[6 * 4];
			float points[8 * 4];
		};

		HddagiIntegrateShaderRD integrate;
		RID integrate_shader;
		RID integrate_shader_version[INTEGRATE_MODE_MAX];
		RID integrate_pipeline[INTEGRATE_MODE_MAX];

	} hddagi_shader;

public:
	static GI *get_singleton() { return singleton; }

	/* GI */

	enum {
		MAX_VOXEL_GI_INSTANCES = 8
	};

	// Struct for use in render buffer
	class RenderBuffersGI : public RenderBufferCustomDataRD {
		GDCLASS(RenderBuffersGI, RenderBufferCustomDataRD)

	private:
		RID voxel_gi_buffer;

	public:
		RID voxel_gi_textures[MAX_VOXEL_GI_INSTANCES];

		RID full_buffer;
		RID full_dispatch;
		RID full_mask;

		/* GI buffers */
		bool using_half_size_gi = false;

		RID scene_data_ubo;

		RID get_voxel_gi_buffer();

		virtual void configure(RenderSceneBuffersRD *p_render_buffers) override{};
		virtual void free_data() override;
	};

	/* VOXEL GI API */

	bool owns_voxel_gi(RID p_rid) { return voxel_gi_owner.owns(p_rid); };

	virtual RID voxel_gi_allocate() override;
	virtual void voxel_gi_free(RID p_voxel_gi) override;
	virtual void voxel_gi_initialize(RID p_voxel_gi) override;

	virtual void voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) override;

	virtual AABB voxel_gi_get_bounds(RID p_voxel_gi) const override;
	virtual Vector3i voxel_gi_get_octree_size(RID p_voxel_gi) const override;
	virtual Vector<uint8_t> voxel_gi_get_octree_cells(RID p_voxel_gi) const override;
	virtual Vector<uint8_t> voxel_gi_get_data_cells(RID p_voxel_gi) const override;
	virtual Vector<uint8_t> voxel_gi_get_distance_field(RID p_voxel_gi) const override;

	virtual Vector<int> voxel_gi_get_level_counts(RID p_voxel_gi) const override;
	virtual Transform3D voxel_gi_get_to_cell_xform(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) override;
	virtual float voxel_gi_get_dynamic_range(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_propagation(RID p_voxel_gi, float p_range) override;
	virtual float voxel_gi_get_propagation(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_energy(RID p_voxel_gi, float p_energy) override;
	virtual float voxel_gi_get_energy(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_baked_exposure_normalization(RID p_voxel_gi, float p_baked_exposure) override;
	virtual float voxel_gi_get_baked_exposure_normalization(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_bias(RID p_voxel_gi, float p_bias) override;
	virtual float voxel_gi_get_bias(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) override;
	virtual float voxel_gi_get_normal_bias(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) override;
	virtual bool voxel_gi_is_interior(RID p_voxel_gi) const override;

	virtual void voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) override;
	virtual bool voxel_gi_is_using_two_bounces(RID p_voxel_gi) const override;

	virtual uint32_t voxel_gi_get_version(RID p_probe) const override;
	uint32_t voxel_gi_get_data_version(RID p_probe);

	RID voxel_gi_get_octree_buffer(RID p_voxel_gi) const;
	RID voxel_gi_get_data_buffer(RID p_voxel_gi) const;

	RID voxel_gi_get_sdf_texture(RID p_voxel_gi);

	Dependency *voxel_gi_get_dependency(RID p_voxel_gi) const;

	/* VOXEL_GI INSTANCE */

	_FORCE_INLINE_ RID voxel_gi_instance_get_texture(RID p_probe) {
		VoxelGIInstance *voxel_gi = voxel_gi_instance_owner.get_or_null(p_probe);
		ERR_FAIL_NULL_V(voxel_gi, RID());
		return voxel_gi->texture;
	};

	_FORCE_INLINE_ void voxel_gi_instance_set_render_index(RID p_probe, uint32_t p_index) {
		VoxelGIInstance *voxel_gi = voxel_gi_instance_owner.get_or_null(p_probe);
		ERR_FAIL_NULL(voxel_gi);

		voxel_gi->render_index = p_index;
	};

	bool voxel_gi_instance_owns(RID p_rid) const {
		return voxel_gi_instance_owner.owns(p_rid);
	}

	void voxel_gi_instance_free(RID p_rid);

	RS::VoxelGIQuality voxel_gi_quality = RS::VOXEL_GI_QUALITY_LOW;

	/* HDDAGI */

	class HDDAGI : public RenderBufferCustomDataRD {
		GDCLASS(HDDAGI, RenderBufferCustomDataRD)

	public:
		enum {
			MAX_CASCADES = 8,
			CASCADE_SIZE = 128,
			REGION_CELLS = 8,
			MAX_DYNAMIC_LIGHTS = 128,
			MAX_STATIC_LIGHTS = 1024,
			LIGHTPROBE_OCT_SIZE = 5,
			LIGHTPROBE_HISTORY_FRAMES = 2,
			OCCLUSION_OCT_SIZE = 14,
			OCCLUSION_SUBPIXELS = 4,
			SH_SIZE = 16
		};

		struct Cascade {
			struct UBO {
				float offset[3];
				float to_cell;
				int32_t region_world_offset[3];
				uint32_t pad;
				float pad2[4];
			};

			//cascade blocks are full-size for volume (128^3), half size for albedo/emission
			float cell_size;
			Vector3i position;

			static const Vector3i DIRTY_ALL;
			Vector3i dirty_regions; //(0,0,0 is not dirty, negative is refresh from the end, DIRTY_ALL is refresh all.

			RID light_process_buffer;
			RID light_process_dispatch_buffer;
			RID light_process_dispatch_buffer_copy;

			RID light_position_bufer;

			bool static_lights_dirty = true;
			bool dynamic_lights_dirty = true;

			struct LightProcessCell { // this struct is unused, but remains as reference for size
				uint32_t position;
				uint32_t albedo;
				uint32_t emission;
				uint32_t normal;
			};

			uint32_t motion_accum = 0;
			uint16_t latest_version = 0;

			float baked_exposure_normalization = 1.0;
		};

		Vector3i cascade_size;

		// access to our containers
		GI *gi = nullptr;

		RID render_albedo; //x6, anisotropic
		RID render_aniso_normals;
		RID render_emission;
		RID render_emission_aniso;

		RID voxel_bits_tex;
		RID voxel_region_tex;
		RID voxel_disocclusion_tex;
		RID voxel_light_tex;
		RID voxel_light_tex_data;
		RID voxel_light_neighbour_data;
		RID region_version_data;

		RID light_process_buffer_render;
		RID light_process_dispatch_buffer_render;

		RID lightprobe_specular_tex;
		RID lightprobe_specular_data;
		RID lightprobe_diffuse_data;
		RID lightprobe_diffuse_tex;
		RID lightprobe_ambient_data;
		RID lightprobe_ambient_tex;
		RID lightprobe_diffuse_filter_data;
		RID lightprobe_diffuse_filter_tex;
		RID lightprobe_hit_cache_data;
		RID lightprobe_hit_cache_version_data;
		RID lightprobe_moving_average;
		RID lightprobe_moving_average_history;
		RID lightprobe_neighbour_visibility_map;
		RID lightprobe_geometry_proximity_map;
		RID lightprobe_camera_visibility_map;
		RID lightprobe_process_frame; // 28 bits is frame, upper 4 bits is frames remaining to do full updates (for having updated light when scrolling).

		Vector<RID> lightprobe_camera_buffers;

		RID occlusion_data[2];
		RID occlusion_tex[2];

		LocalVector<Cascade> cascades;

		float solid_cell_ratio = 0;
		uint32_t solid_cell_count = 0;
		uint32_t sampling_cache_buffer_cascade_size = 0;

		uint32_t update_frame = 0;
		uint32_t frames_to_converge = 6;

		bool using_probe_filter = true;
		bool using_reflection_filter = true;
		bool using_ambient_filter = true;

		int num_cascades = 6;
		float min_cell_size = 0;

		RID cascades_ubo;

		bool uses_occlusion = false;
		float bounce_feedback = 0.5;
		bool reads_sky = true;
		float energy = 1.0;
		float normal_bias = 1.1;
		float reflection_bias = 2.0;
		float probe_bias = 1.1;
		float occlusion_bias = 0.1;
		RS::EnvironmentHDDAGICascadeFormat cascade_format = RS::ENV_HDDAGI_CASCADE_FORMAT_16x8x16;

		float y_mult = 1.0;

		uint32_t version = 0;
		uint32_t render_pass = 0;

		int32_t cascade_dynamic_light_count[HDDAGI::MAX_CASCADES]; //used dynamically

		RID debug_probes_scene_data_ubo;

		virtual void configure(RenderSceneBuffersRD *p_render_buffers) override{};
		virtual void free_data() override;
		~HDDAGI();

		void create(RID p_env, const Vector3 &p_world_position, uint32_t p_requested_history_size, GI *p_gi);
		void update(RID p_env, const Vector3 &p_world_position);
		void update_light();
		void update_probes(RID p_env, RendererRD::SkyRD::Sky *p_sky, uint32_t p_view_count, const Projection *p_projections, const Vector3 *p_eye_offsets, const Transform3D &p_cam_transform);
		void store_probes();
		int get_pending_region_count() const;
		int get_pending_region_data(int p_region, Vector3i &r_local_offset, Vector3i &r_local_size, AABB &r_bounds, Vector3i &r_scroll, Vector3i &r_region_world) const;
		void update_cascades();

		RID get_lightprobe_diffuse_texture() {
			if (using_probe_filter) {
				return lightprobe_diffuse_filter_tex;
			} else {
				return lightprobe_diffuse_tex;
			}
		}

		RID get_lightprobe_specular_texture() {
			return lightprobe_specular_tex;
		}

		Vector<RID> get_lightprobe_occlusion_textures() {
			Vector<RID> ret = { occlusion_tex[0], occlusion_tex[1] };
			return ret;
		}

		void debug_draw(uint32_t p_view_count, const Projection *p_projections, const Transform3D &p_transform, int p_width, int p_height, RID p_render_target, RID p_texture, const Vector<RID> &p_texture_views);
		void debug_probes(RID p_framebuffer, const uint32_t p_view_count, const Projection *p_camera_with_transforms);

		void pre_process_gi(const Transform3D &p_transform, RenderDataRD *p_render_data);
		void render_region(Ref<RenderSceneBuffersRD> p_render_buffers, int p_region, const PagedArray<RenderGeometryInstance *> &p_instances, float p_exposure_normalization);
		void render_static_lights(RenderDataRD *p_render_data, Ref<RenderSceneBuffersRD> p_render_buffers, uint32_t p_cascade_count, const uint32_t *p_cascade_indices, const PagedArray<RID> *p_positional_light_cull_result);
	};

	RS::EnvironmentHDDAGIFramesToConverge hddagi_frames_to_converge = RS::ENV_HDDAGI_CONVERGE_IN_12_FRAMES;
	RS::EnvironmentHDDAGIFramesToUpdateLight hddagi_frames_to_update_light = RS::ENV_HDDAGI_UPDATE_LIGHT_IN_4_FRAMES;
	RS::EnvironmentHDDAGIInactiveProbeFrames inactive_probe_frames = RS::ENV_HDDAGI_INACTIVE_PROBE_4_FRAMES;

	float hddagi_solid_cell_ratio = 0.5;
	Vector3 hddagi_debug_probe_pos;
	Vector3 hddagi_debug_probe_dir;
	bool hddagi_debug_probe_enabled = false;
	Vector3i hddagi_debug_probe_index;
	uint32_t hddagi_current_version = 0;

	/* HDDAGI UPDATE */

	int hddagi_get_lightprobe_octahedron_size() const { return HDDAGI::LIGHTPROBE_OCT_SIZE; }
	int hddagi_get_occlusion_octahedron_size() const { return HDDAGI::OCCLUSION_OCT_SIZE; }

	virtual void hddagi_reset() override;

	struct HDDAGIData {
		int32_t grid_size[3];
		int32_t max_cascades;

		float normal_bias;
		float energy;
		float y_mult;
		float reflection_bias;

		int32_t probe_axis_size[3];
		float esm_strength;

		uint32_t pad3[4];

		struct ProbeCascadeData {
			float position[3]; //offset of (0,0,0) in world coordinates
			float to_probe;

			int32_t region_world_offset[3];
			float to_cell; // 1/bounds * grid_size

			uint32_t pad[3];
			float exposure_normalization;

			uint32_t pad2[4];
		};

		ProbeCascadeData cascades[HDDAGI::MAX_CASCADES];
	};

	struct VoxelGIData {
		float xform[16]; // 64 - 64

		float bounds[3]; // 12 - 76
		float dynamic_range; // 4 - 80

		float bias; // 4 - 84
		float normal_bias; // 4 - 88
		uint32_t blend_ambient; // 4 - 92
		uint32_t mipmaps; // 4 - 96

		float pad[3]; // 12 - 108
		float exposure_normalization; // 4 - 112
	};

	struct SceneData {
		float inv_projection[2][16];
		float cam_transform[16];
		float eye_offset[2][4];

		int32_t screen_size[2];
		float pad1;
		float pad2;
	};

	struct PushConstant {
		uint32_t max_voxel_gi_instances;
		uint32_t high_quality_vct;
		uint32_t orthogonal;
		uint32_t view_index;

		float proj_info[4];

		float z_near;
		float z_far;
		uint32_t pad;
		float occlusion_bias;
	};

	RID hddagi_ubo;

	enum Mode {
		MODE_VOXEL_GI,
		MODE_HDDAGI,
		MODE_COMBINED,
		MODE_HDDAGI_BLEND_AMBIENT,
		MODE_COMBINED_BLEND_AMBIENT,
		MODE_MAX
	};

	enum ShaderSpecializations {
		SHADER_SPECIALIZATION_HALF_RES = 1 << 0,
		SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX = 1 << 1,
		SHADER_SPECIALIZATION_USE_VRS = 1 << 2,
		SHADER_SPECIALIZATION_VARIATIONS = 8,
	};

	RID default_voxel_gi_buffer;

	bool half_resolution = false;
	GiShaderRD shader;
	RID shader_version;
	RID pipelines[SHADER_SPECIALIZATION_VARIATIONS][MODE_MAX];

	enum FilterMode {
		FILTER_MODE_BILATERAL,
		FILTER_MODE_BILATERAL_HALF_SIZE,
		FILTER_MODE_MAX
	};
	enum FilterShaderSpecializations {
		FILTER_SHADER_SPECIALIZATION_HALF_RES = 1 << 0,
		FILTER_SHADER_SPECIALIZATION_USE_FULL_PROJECTION_MATRIX = 1 << 1,
		FILTER_SHADER_SPECIALIZATION_VARIATIONS = 4
	};

	struct FilterPushConstant {
		uint32_t orthogonal;
		float z_near;
		float z_far;
		uint32_t view_index;

		float proj_info[4];

		int32_t filter_dir[2];
		uint32_t pad[2];
	};

	HddagiFilterShaderRD filter_shader;
	RID filter_shader_version;
	RID filter_pipelines[FILTER_SHADER_SPECIALIZATION_VARIATIONS][MODE_MAX];

	GI();
	~GI();

	void init(RendererRD::SkyRD *p_sky);
	void free();

	Ref<HDDAGI> create_hddagi(RID p_env, const Vector3 &p_world_position, uint32_t p_requested_history_size);

	void setup_voxel_gi_instances(RenderDataRD *p_render_data, Ref<RenderSceneBuffersRD> p_render_buffers, const Transform3D &p_transform, const PagedArray<RID> &p_voxel_gi_instances, uint32_t &r_voxel_gi_instances_used);
	void process_gi(Ref<RenderSceneBuffersRD> p_render_buffers, const RID *p_normal_roughness_slices, RID p_voxel_gi_buffer, RID p_environment, uint32_t p_view_count, const Projection *p_projections, const Vector3 *p_eye_offsets, const Transform3D &p_cam_transform, const PagedArray<RID> &p_voxel_gi_instances);

	RID voxel_gi_instance_create(RID p_base);
	void voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform);
	bool voxel_gi_needs_update(RID p_probe) const;
	void voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RenderGeometryInstance *> &p_dynamic_objects);
	void debug_voxel_gi(RID p_voxel_gi, RD::DrawListID p_draw_list, RID p_framebuffer, const Projection &p_camera_with_transform, bool p_lighting, bool p_emission, float p_alpha);
};

} // namespace RendererRD

#endif // GI_RD_H
