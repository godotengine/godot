/*************************************************************************/
/*  render_forward_mobile.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RENDERING_SERVER_SCENE_RENDER_FORWARD_MOBILE_H
#define RENDERING_SERVER_SCENE_RENDER_FORWARD_MOBILE_H

#include "core/templates/paged_allocator.h"
#include "servers/rendering/renderer_rd/forward_mobile/scene_shader_forward_mobile.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"

namespace RendererSceneRenderImplementation {

class RenderForwardMobile : public RendererSceneRenderRD {
	friend SceneShaderForwardMobile;

protected:
	/* Scene Shader */

	enum {
		SCENE_UNIFORM_SET = 0,
		RENDER_PASS_UNIFORM_SET = 1,
		TRANSFORMS_UNIFORM_SET = 2,
		MATERIAL_UNIFORM_SET = 3
	};

	enum {
		MAX_LIGHTMAPS = 8,
		MAX_RDL_CULL = 8, // maximum number of reflection probes, decals or lights we can cull per geometry instance
		INSTANCE_DATA_BUFFER_MIN_SIZE = 4096
	};

	enum RenderListType {
		RENDER_LIST_OPAQUE, //used for opaque objects
		RENDER_LIST_ALPHA, //used for transparent objects
		RENDER_LIST_SECONDARY, //used for shadows and other objects
		RENDER_LIST_MAX
	};

	/* Scene Shader */

	SceneShaderForwardMobile scene_shader;

	/* Render Buffer */

	struct RenderBufferDataForwardMobile : public RenderBufferData {
		RID color;
		RID depth;
		// RID normal_roughness_buffer;

		RS::ViewportMSAA msaa;
		RD::TextureSamples texture_samples;

		RID color_msaa;
		RID depth_msaa;
		// RID normal_roughness_buffer_msaa;

		RID color_fb;
		int width, height;

		void clear();
		virtual void configure(RID p_color_buffer, RID p_depth_buffer, int p_width, int p_height, RS::ViewportMSAA p_msaa);

		~RenderBufferDataForwardMobile();
	};

	virtual RenderBufferData *_create_render_buffer_data();

	/* Rendering */

	enum PassMode {
		PASS_MODE_COLOR,
		// PASS_MODE_COLOR_SPECULAR,
		PASS_MODE_COLOR_TRANSPARENT,
		PASS_MODE_SHADOW,
		PASS_MODE_SHADOW_DP,
		// PASS_MODE_DEPTH,
		// PASS_MODE_DEPTH_NORMAL_ROUGHNESS,
		// PASS_MODE_DEPTH_NORMAL_ROUGHNESS_GIPROBE,
		PASS_MODE_DEPTH_MATERIAL,
		// PASS_MODE_SDF,
	};

	struct GeometryInstanceForwardMobile;
	struct GeometryInstanceSurfaceDataCache;
	struct RenderElementInfo;

	struct RenderListParameters {
		GeometryInstanceSurfaceDataCache **elements = nullptr;
		RenderElementInfo *element_info = nullptr;
		int element_count = 0;
		bool reverse_cull = false;
		PassMode pass_mode = PASS_MODE_COLOR;
		// bool no_gi = false;
		RID render_pass_uniform_set;
		bool force_wireframe = false;
		Vector2 uv_offset;
		Plane lod_plane;
		float lod_distance_multiplier = 0.0;
		float screen_lod_threshold = 0.0;
		RD::FramebufferFormatID framebuffer_format = 0;
		uint32_t element_offset = 0;
		uint32_t barrier = RD::BARRIER_MASK_ALL;

		RenderListParameters(GeometryInstanceSurfaceDataCache **p_elements, RenderElementInfo *p_element_info, int p_element_count, bool p_reverse_cull, PassMode p_pass_mode, RID p_render_pass_uniform_set, bool p_force_wireframe = false, const Vector2 &p_uv_offset = Vector2(), const Plane &p_lod_plane = Plane(), float p_lod_distance_multiplier = 0.0, float p_screen_lod_threshold = 0.0, uint32_t p_element_offset = 0, uint32_t p_barrier = RD::BARRIER_MASK_ALL) {
			elements = p_elements;
			element_info = p_element_info;
			element_count = p_element_count;
			reverse_cull = p_reverse_cull;
			pass_mode = p_pass_mode;
			// no_gi = p_no_gi;
			render_pass_uniform_set = p_render_pass_uniform_set;
			force_wireframe = p_force_wireframe;
			uv_offset = p_uv_offset;
			lod_plane = p_lod_plane;
			lod_distance_multiplier = p_lod_distance_multiplier;
			screen_lod_threshold = p_screen_lod_threshold;
			element_offset = p_element_offset;
			barrier = p_barrier;
		}
	};

	RID _setup_render_pass_uniform_set(RenderListType p_render_list, const RenderDataRD *p_render_data, RID p_radiance_texture, bool p_use_directional_shadow_atlas = false, int p_index = 0);
	virtual void _render_scene(RenderDataRD *p_render_data, const Color &p_default_bg_color);

	virtual void _render_shadow_begin();
	virtual void _render_shadow_append(RID p_framebuffer, const PagedArray<GeometryInstance *> &p_instances, const CameraMatrix &p_projection, const Transform &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_use_dp, bool p_use_dp_flip, bool p_use_pancake, const Plane &p_camera_plane = Plane(), float p_lod_distance_multiplier = 0.0, float p_screen_lod_threshold = 0.0, const Rect2i &p_rect = Rect2i(), bool p_flip_y = false, bool p_clear_region = true, bool p_begin = true, bool p_end = true);
	virtual void _render_shadow_process();
	virtual void _render_shadow_end(uint32_t p_barrier = RD::BARRIER_MASK_ALL);

	virtual void _render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region);
	virtual void _render_uv2(const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region);
	virtual void _render_sdfgi(RID p_render_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const PagedArray<GeometryInstance *> &p_instances, const RID &p_albedo_texture, const RID &p_emission_texture, const RID &p_emission_aniso_texture, const RID &p_geom_facing_texture);
	virtual void _render_particle_collider_heightfield(RID p_fb, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, const PagedArray<GeometryInstance *> &p_instances);

	uint64_t lightmap_texture_array_version = 0xFFFFFFFF;

	virtual void _base_uniforms_changed();
	void _update_render_base_uniform_set();
	virtual RID _render_buffers_get_normal_texture(RID p_render_buffers);

	void _fill_render_list(RenderListType p_render_list, const RenderDataRD *p_render_data, PassMode p_pass_mode, bool p_append = false);
	void _fill_instance_data(RenderListType p_render_list, uint32_t p_offset = 0, int32_t p_max_elements = -1, bool p_update_buffer = true);
	// void _update_instance_data_buffer(RenderListType p_render_list);

	static RenderForwardMobile *singleton;

	void _setup_environment(const RenderDataRD *p_render_data, bool p_no_fog, const Size2i &p_screen_size, bool p_flip_y, const Color &p_default_bg_color, bool p_opaque_render_buffers = false, bool p_pancake_shadows = false, int p_index = 0);
	void _setup_lightmaps(const PagedArray<RID> &p_lightmaps, const Transform &p_cam_transform);

	RID render_base_uniform_set;
	LocalVector<RID> render_pass_uniform_sets;

	/* Light map */

	struct LightmapData {
		float normal_xform[12];
	};

	struct LightmapCaptureData {
		float sh[9 * 4];
	};

	/* Scene state */

	struct SceneState {
		// This struct is loaded into Set 1 - Binding 0, populated at start of rendering a frame, must match with shader code
		struct UBO {
			float projection_matrix[16];
			float inv_projection_matrix[16];

			float camera_matrix[16];
			float inv_camera_matrix[16];

			float viewport_size[2];
			float screen_pixel_size[2];

			float directional_penumbra_shadow_kernel[128]; //32 vec4s
			float directional_soft_shadow_kernel[128];
			float penumbra_shadow_kernel[128];
			float soft_shadow_kernel[128];

			uint32_t directional_penumbra_shadow_samples;
			uint32_t directional_soft_shadow_samples;
			uint32_t penumbra_shadow_samples;
			uint32_t soft_shadow_samples;

			float ambient_light_color_energy[4];

			float ambient_color_sky_mix;
			uint32_t use_ambient_light;
			uint32_t use_ambient_cubemap;
			uint32_t use_reflection_cubemap;

			float radiance_inverse_xform[12];

			float shadow_atlas_pixel_size[2];
			float directional_shadow_pixel_size[2];

			uint32_t directional_light_count;
			float dual_paraboloid_side;
			float z_far;
			float z_near;

			uint32_t ssao_enabled;
			float ssao_light_affect;
			float ssao_ao_affect;
			uint32_t roughness_limiter_enabled;

			float roughness_limiter_amount;
			float roughness_limiter_limit;
			uint32_t roughness_limiter_pad[2];

			float ao_color[4];

			// Fog
			uint32_t fog_enabled;
			float fog_density;
			float fog_height;
			float fog_height_density;

			float fog_light_color[3];
			float fog_sun_scatter;

			float fog_aerial_perspective;
			uint32_t material_uv2_mode;

			float time;
			float reflection_multiplier;

			uint32_t pancake_shadows;
			uint32_t pad1;
			uint32_t pad2;
			uint32_t pad3;
		};

		UBO ubo;

		LocalVector<RID> uniform_buffers;

		// !BAS! We need to change lightmaps, we're not going to do this with a buffer but pushing the used lightmap in
		LightmapData lightmaps[MAX_LIGHTMAPS];
		RID lightmap_ids[MAX_LIGHTMAPS];
		bool lightmap_has_sh[MAX_LIGHTMAPS];
		uint32_t lightmaps_used = 0;
		uint32_t max_lightmaps;
		RID lightmap_buffer;

		LightmapCaptureData *lightmap_captures;
		uint32_t max_lightmap_captures;
		RID lightmap_capture_buffer;

		bool used_screen_texture = false;
		bool used_normal_texture = false;
		bool used_depth_texture = false;
		bool used_sss = false;

		struct ShadowPass {
			uint32_t element_from;
			uint32_t element_count;
			bool flip_cull;
			PassMode pass_mode;

			RID rp_uniform_set;
			Plane camera_plane;
			float lod_distance_multiplier;
			float screen_lod_threshold;

			RID framebuffer;
			RD::InitialAction initial_depth_action;
			RD::FinalAction final_depth_action;
			Rect2i rect;
		};

		LocalVector<ShadowPass> shadow_passes;
	} scene_state;

	/* Render List */

	// !BAS! Render list can probably be reused between clustered and mobile?
	struct RenderList {
		LocalVector<GeometryInstanceSurfaceDataCache *> elements;
		LocalVector<RenderElementInfo> element_info;

		void clear() {
			elements.clear();
			element_info.clear();
		}

		//should eventually be replaced by radix

		struct SortByKey {
			_FORCE_INLINE_ bool operator()(const GeometryInstanceSurfaceDataCache *A, const GeometryInstanceSurfaceDataCache *B) const {
				return (A->sort.sort_key2 == B->sort.sort_key2) ? (A->sort.sort_key1 < B->sort.sort_key1) : (A->sort.sort_key2 < B->sort.sort_key2);
			}
		};

		void sort_by_key() {
			SortArray<GeometryInstanceSurfaceDataCache *, SortByKey> sorter;
			sorter.sort(elements.ptr(), elements.size());
		}

		void sort_by_key_range(uint32_t p_from, uint32_t p_size) {
			SortArray<GeometryInstanceSurfaceDataCache *, SortByKey> sorter;
			sorter.sort(elements.ptr() + p_from, p_size);
		}

		struct SortByDepth {
			_FORCE_INLINE_ bool operator()(const GeometryInstanceSurfaceDataCache *A, const GeometryInstanceSurfaceDataCache *B) const {
				return (A->owner->depth < B->owner->depth);
			}
		};

		void sort_by_depth() { //used for shadows

			SortArray<GeometryInstanceSurfaceDataCache *, SortByDepth> sorter;
			sorter.sort(elements.ptr(), elements.size());
		}

		struct SortByReverseDepthAndPriority {
			_FORCE_INLINE_ bool operator()(const GeometryInstanceSurfaceDataCache *A, const GeometryInstanceSurfaceDataCache *B) const {
				return (A->sort.priority == B->sort.priority) ? (A->owner->depth > B->owner->depth) : (A->sort.priority < B->sort.priority);
			}
		};

		void sort_by_reverse_depth_and_priority(bool p_alpha) { //used for alpha

			SortArray<GeometryInstanceSurfaceDataCache *, SortByReverseDepthAndPriority> sorter;
			sorter.sort(elements.ptr(), elements.size());
		}

		_FORCE_INLINE_ void add_element(GeometryInstanceSurfaceDataCache *p_element) {
			elements.push_back(p_element);
		}
	};

	struct RenderElementInfo {
		uint32_t uses_lightmap : 1;
		uint32_t lod_index : 8;
		uint32_t reserved : 23;
	};

	template <PassMode p_pass_mode>
	_FORCE_INLINE_ void _render_list_template(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element);

	void _render_list(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element);

	LocalVector<RD::DrawListID> thread_draw_lists;
	void _render_list_thread_function(uint32_t p_thread, RenderListParameters *p_params);
	void _render_list_with_threads(RenderListParameters *p_params, RID p_framebuffer, RD::InitialAction p_initial_color_action, RD::FinalAction p_final_color_action, RD::InitialAction p_initial_depth_action, RD::FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const Vector<RID> &p_storage_textures = Vector<RID>());

	uint32_t render_list_thread_threshold = 500;

	RenderList render_list[RENDER_LIST_MAX];

	/* Geometry instance */

	// check which ones of these apply, probably all except GI and SDFGI
	enum {
		INSTANCE_DATA_FLAG_USE_GI_BUFFERS = 1 << 6,
		INSTANCE_DATA_FLAG_USE_SDFGI = 1 << 7,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP_CAPTURE = 1 << 8,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP = 1 << 9,
		INSTANCE_DATA_FLAG_USE_SH_LIGHTMAP = 1 << 10,
		INSTANCE_DATA_FLAG_USE_GIPROBE = 1 << 11,
		INSTANCE_DATA_FLAG_MULTIMESH = 1 << 12,
		INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D = 1 << 13,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR = 1 << 14,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA = 1 << 15,
		INSTANCE_DATA_FLAGS_PARTICLE_TRAIL_SHIFT = 16,
		INSTANCE_DATA_FLAGS_PARTICLE_TRAIL_MASK = 0xFF,
		INSTANCE_DATA_FLAGS_NON_UNIFORM_SCALE = 1 << 24,
	};

	struct GeometryInstanceLightmapSH {
		Color sh[9];
	};

	// Cached data for drawing surfaces
	struct GeometryInstanceSurfaceDataCache {
		enum {
			FLAG_PASS_DEPTH = 1,
			FLAG_PASS_OPAQUE = 2,
			FLAG_PASS_ALPHA = 4,
			FLAG_PASS_SHADOW = 8,
			FLAG_USES_SHARED_SHADOW_MATERIAL = 128,
			FLAG_USES_SUBSURFACE_SCATTERING = 2048,
			FLAG_USES_SCREEN_TEXTURE = 4096,
			FLAG_USES_DEPTH_TEXTURE = 8192,
			FLAG_USES_NORMAL_TEXTURE = 16384,
			FLAG_USES_DOUBLE_SIDED_SHADOWS = 32768,
			FLAG_USES_PARTICLE_TRAILS = 65536,
		};

		union {
			struct {
				// !BAS! CHECK BITS!!!

				uint64_t surface_index : 10;
				uint64_t geometry_id : 32;
				uint64_t material_id_low : 16;

				uint64_t material_id_hi : 16;
				uint64_t shader_id : 32;
				uint64_t uses_lightmap : 4; // sort by lightmap id here, not whether its yes/no (is 4 bits enough?)
				uint64_t depth_layer : 4;
				uint64_t priority : 8;

				// uint64_t lod_index : 8; // no need to sort on LOD
				// uint64_t uses_forward_gi : 1; // no GI here, remove
			};
			struct {
				uint64_t sort_key1;
				uint64_t sort_key2;
			};
		} sort;

		RS::PrimitiveType primitive = RS::PRIMITIVE_MAX;
		uint32_t flags = 0;
		uint32_t surface_index = 0;
		uint32_t lod_index = 0;

		void *surface = nullptr;
		RID material_uniform_set;
		SceneShaderForwardMobile::ShaderData *shader = nullptr;

		void *surface_shadow = nullptr;
		RID material_uniform_set_shadow;
		SceneShaderForwardMobile::ShaderData *shader_shadow = nullptr;

		GeometryInstanceSurfaceDataCache *next = nullptr;
		GeometryInstanceForwardMobile *owner = nullptr;
	};

	// !BAS! GeometryInstanceForwardClustered and GeometryInstanceForwardMobile will likely have a lot of overlap
	// may need to think about making this its own class like GeometryInstanceRD?

	struct GeometryInstanceForwardMobile : public GeometryInstance {
		// setup
		uint32_t base_flags = 0;
		uint32_t flags_cache = 0;

		// this structure maps to our push constant in our shader and is populated right before our draw call
		struct PushConstant {
			float transform[16];
			uint32_t flags;
			uint32_t instance_uniforms_ofs; //base offset in global buffer for instance variables
			uint32_t gi_offset; //GI information when using lightmapping (VCT or lightmap index)
			uint32_t layer_mask = 1;
			float lightmap_uv_scale[4]; // doubles as uv_offset when needed
			uint32_t reflection_probes[2]; // packed reflection probes
			uint32_t omni_lights[2]; // packed omni lights
			uint32_t spot_lights[2]; // packed spot lights
			uint32_t decals[2]; // packed spot lights
		};

		// PushConstant push_constant; // we populate this from our instance data

		//used during rendering
		uint32_t layer_mask = 1;
		RID transforms_uniform_set;
		float depth = 0;
		bool mirror = false;
		Transform transform;
		bool store_transform_cache = true; // if true we copy our transform into our PushConstant, if false we use our transforms UBO and clear our PushConstants transform
		bool non_uniform_scale = false;
		AABB transformed_aabb; //needed for LOD
		float lod_bias = 0.0;
		float lod_model_scale = 1.0;
		int32_t shader_parameters_offset = -1;
		uint32_t instance_count = 0;
		uint32_t trail_steps = 1;
		RID mesh_instance;

		// lightmap
		uint32_t gi_offset_cache = 0; // !BAS! Should rename this to lightmap_offset_cache, in forward clustered this was shared between gi and lightmap
		uint32_t lightmap_slice_index;
		Rect2 lightmap_uv_scale;
		RID lightmap_instance;
		GeometryInstanceLightmapSH *lightmap_sh = nullptr;

		// culled light info
		uint32_t reflection_probe_count;
		RID reflection_probes[MAX_RDL_CULL];
		uint32_t omni_light_count;
		RID omni_lights[MAX_RDL_CULL];
		uint32_t spot_light_count;
		RID spot_lights[MAX_RDL_CULL];
		uint32_t decals_count;
		RID decals[MAX_RDL_CULL];

		GeometryInstanceSurfaceDataCache *surface_caches = nullptr;

		// do we use this?
		SelfList<GeometryInstanceForwardMobile> dirty_list_element;

		struct Data {
			//data used less often goes into regular heap
			RID base;
			RS::InstanceType base_type;

			RID skeleton;
			Vector<RID> surface_materials;
			RID material_override;
			AABB aabb;

			bool use_baked_light = false;
			bool cast_double_sided_shadows = false;
			// bool mirror = false; // !BAS! Does not seem used, we already have this in the main struct

			bool dirty_dependencies = false;

			RendererStorage::DependencyTracker dependency_tracker;
		};

		Data *data = nullptr;

		GeometryInstanceForwardMobile() :
				dirty_list_element(this) {}
	};

public:
	static void _geometry_instance_dependency_changed(RendererStorage::DependencyChangedNotification p_notification, RendererStorage::DependencyTracker *p_tracker);
	static void _geometry_instance_dependency_deleted(const RID &p_dependency, RendererStorage::DependencyTracker *p_tracker);

	SelfList<GeometryInstanceForwardMobile>::List geometry_instance_dirty_list;

	PagedAllocator<GeometryInstanceForwardMobile> geometry_instance_alloc;
	PagedAllocator<GeometryInstanceSurfaceDataCache> geometry_instance_surface_alloc;
	PagedAllocator<GeometryInstanceLightmapSH> geometry_instance_lightmap_sh;

	void _geometry_instance_add_surface_with_material(GeometryInstanceForwardMobile *ginstance, uint32_t p_surface, SceneShaderForwardMobile::MaterialData *p_material, uint32_t p_material_id, uint32_t p_shader_id, RID p_mesh);
	void _geometry_instance_add_surface(GeometryInstanceForwardMobile *ginstance, uint32_t p_surface, RID p_material, RID p_mesh);
	void _geometry_instance_mark_dirty(GeometryInstance *p_geometry_instance);
	void _geometry_instance_update(GeometryInstance *p_geometry_instance);
	void _update_dirty_geometry_instances();

	virtual GeometryInstance *geometry_instance_create(RID p_base);
	virtual void geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton);
	virtual void geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override);
	virtual void geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_materials);
	virtual void geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance);
	virtual void geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabb);
	virtual void geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask);
	virtual void geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias);
	virtual void geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable);
	virtual void geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable);
	virtual void geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index);
	virtual void geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9);
	virtual void geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset);
	virtual void geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable);

	virtual Transform geometry_instance_get_transform(GeometryInstance *p_instance);
	virtual AABB geometry_instance_get_aabb(GeometryInstance *p_instance);

	virtual void geometry_instance_free(GeometryInstance *p_geometry_instance);

	virtual uint32_t geometry_instance_get_pair_mask();
	virtual void geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count);
	virtual void geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count);
	virtual void geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count);
	virtual void geometry_instance_pair_gi_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_gi_probe_instances, uint32_t p_gi_probe_instance_count);

	virtual bool free(RID p_rid);

	virtual bool is_dynamic_gi_supported() const;
	virtual bool is_clustered_enabled() const;
	virtual bool is_volumetric_supported() const;
	virtual uint32_t get_max_elements() const;

	RenderForwardMobile(RendererStorageRD *p_storage);
	~RenderForwardMobile();
};
} // namespace RendererSceneRenderImplementation
#endif // !RENDERING_SERVER_SCENE_RENDER_FORWARD_MOBILE_H
