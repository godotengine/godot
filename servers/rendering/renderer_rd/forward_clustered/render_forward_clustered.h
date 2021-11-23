/*************************************************************************/
/*  render_forward_clustered.h                                           */
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

#ifndef RENDERING_SERVER_SCENE_RENDER_FORWARD_CLUSTERED_H
#define RENDERING_SERVER_SCENE_RENDER_FORWARD_CLUSTERED_H

#include "core/templates/paged_allocator.h"
#include "servers/rendering/renderer_rd/forward_clustered/scene_shader_forward_clustered.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/renderer_storage_rd.h"
#include "servers/rendering/renderer_rd/shaders/scene_forward_clustered.glsl.gen.h"

namespace RendererSceneRenderImplementation {

class RenderForwardClustered : public RendererSceneRenderRD {
	friend SceneShaderForwardClustered;

	enum {
		SCENE_UNIFORM_SET = 0,
		RENDER_PASS_UNIFORM_SET = 1,
		TRANSFORMS_UNIFORM_SET = 2,
		MATERIAL_UNIFORM_SET = 3
	};

	enum {
		SPEC_CONSTANT_SOFT_SHADOW_SAMPLES = 6,
		SPEC_CONSTANT_PENUMBRA_SHADOW_SAMPLES = 7,
		SPEC_CONSTANT_DIRECTIONAL_SOFT_SHADOW_SAMPLES = 8,
		SPEC_CONSTANT_DIRECTIONAL_PENUMBRA_SHADOW_SAMPLES = 9,
		SPEC_CONSTANT_DECAL_FILTER = 10,
		SPEC_CONSTANT_PROJECTOR_FILTER = 11,
	};

	enum {
		SDFGI_MAX_CASCADES = 8,
		MAX_VOXEL_GI_INSTANCESS = 8,
		MAX_LIGHTMAPS = 8,
		MAX_VOXEL_GI_INSTANCESS_PER_INSTANCE = 2,
		INSTANCE_DATA_BUFFER_MIN_SIZE = 4096
	};

	enum RenderListType {
		RENDER_LIST_OPAQUE, //used for opaque objects
		RENDER_LIST_ALPHA, //used for transparent objects
		RENDER_LIST_SECONDARY, //used for shadows and other objects
		RENDER_LIST_MAX

	};

	/* Scene Shader */

	SceneShaderForwardClustered scene_shader;

	/* Framebuffer */

	struct RenderBufferDataForwardClustered : public RenderBufferData {
		//for rendering, may be MSAAd

		RID color;
		RID depth;
		RID specular;
		RID normal_roughness_buffer;
		RID voxelgi_buffer;

		RS::ViewportMSAA msaa;
		RD::TextureSamples texture_samples;

		RID color_msaa;
		RID depth_msaa;
		RID specular_msaa;
		RID normal_roughness_buffer_msaa;
		RID roughness_buffer_msaa;
		RID voxelgi_buffer_msaa;

		RID depth_fb;
		RID depth_normal_roughness_fb;
		RID depth_normal_roughness_voxelgi_fb;
		RID color_fb;
		RID color_specular_fb;
		RID specular_only_fb;
		int width, height;

		RID render_sdfgi_uniform_set;
		void ensure_specular();
		void ensure_voxelgi();
		void clear();
		virtual void configure(RID p_color_buffer, RID p_depth_buffer, RID p_target_buffer, int p_width, int p_height, RS::ViewportMSAA p_msaa, uint32_t p_view_count);

		~RenderBufferDataForwardClustered();
	};

	virtual RenderBufferData *_create_render_buffer_data() override;
	void _allocate_normal_roughness_texture(RenderBufferDataForwardClustered *rb);

	RID render_base_uniform_set;
	LocalVector<RID> render_pass_uniform_sets;
	RID sdfgi_pass_uniform_set;

	uint64_t lightmap_texture_array_version = 0xFFFFFFFF;

	virtual void _base_uniforms_changed() override;
	virtual RID _render_buffers_get_normal_texture(RID p_render_buffers) override;

	bool base_uniform_set_updated = false;
	void _update_render_base_uniform_set();
	RID _setup_sdfgi_render_pass_uniform_set(RID p_albedo_texture, RID p_emission_texture, RID p_emission_aniso_texture, RID p_geom_facing_texture);
	RID _setup_render_pass_uniform_set(RenderListType p_render_list, const RenderDataRD *p_render_data, RID p_radiance_texture, bool p_use_directional_shadow_atlas = false, int p_index = 0);

	enum PassMode {
		PASS_MODE_COLOR,
		PASS_MODE_COLOR_SPECULAR,
		PASS_MODE_COLOR_TRANSPARENT,
		PASS_MODE_SHADOW,
		PASS_MODE_SHADOW_DP,
		PASS_MODE_DEPTH,
		PASS_MODE_DEPTH_NORMAL_ROUGHNESS,
		PASS_MODE_DEPTH_NORMAL_ROUGHNESS_VOXEL_GI,
		PASS_MODE_DEPTH_MATERIAL,
		PASS_MODE_SDF,
	};

	struct GeometryInstanceSurfaceDataCache;
	struct RenderElementInfo;

	struct RenderListParameters {
		GeometryInstanceSurfaceDataCache **elements = nullptr;
		RenderElementInfo *element_info = nullptr;
		int element_count = 0;
		bool reverse_cull = false;
		PassMode pass_mode = PASS_MODE_COLOR;
		bool no_gi = false;
		RID render_pass_uniform_set;
		bool force_wireframe = false;
		Vector2 uv_offset;
		Plane lod_plane;
		float lod_distance_multiplier = 0.0;
		float screen_lod_threshold = 0.0;
		RD::FramebufferFormatID framebuffer_format = 0;
		uint32_t element_offset = 0;
		uint32_t barrier = RD::BARRIER_MASK_ALL;
		bool use_directional_soft_shadow = false;

		RenderListParameters(GeometryInstanceSurfaceDataCache **p_elements, RenderElementInfo *p_element_info, int p_element_count, bool p_reverse_cull, PassMode p_pass_mode, bool p_no_gi, bool p_use_directional_soft_shadows, RID p_render_pass_uniform_set, bool p_force_wireframe = false, const Vector2 &p_uv_offset = Vector2(), const Plane &p_lod_plane = Plane(), float p_lod_distance_multiplier = 0.0, float p_screen_lod_threshold = 0.0, uint32_t p_element_offset = 0, uint32_t p_barrier = RD::BARRIER_MASK_ALL) {
			elements = p_elements;
			element_info = p_element_info;
			element_count = p_element_count;
			reverse_cull = p_reverse_cull;
			pass_mode = p_pass_mode;
			no_gi = p_no_gi;
			render_pass_uniform_set = p_render_pass_uniform_set;
			force_wireframe = p_force_wireframe;
			uv_offset = p_uv_offset;
			lod_plane = p_lod_plane;
			lod_distance_multiplier = p_lod_distance_multiplier;
			screen_lod_threshold = p_screen_lod_threshold;
			element_offset = p_element_offset;
			barrier = p_barrier;
			use_directional_soft_shadow = p_use_directional_soft_shadows;
		}
	};

	struct LightmapData {
		float normal_xform[12];
	};

	struct LightmapCaptureData {
		float sh[9 * 4];
	};

	enum {
		INSTANCE_DATA_FLAGS_NON_UNIFORM_SCALE = 1 << 5,
		INSTANCE_DATA_FLAG_USE_GI_BUFFERS = 1 << 6,
		INSTANCE_DATA_FLAG_USE_SDFGI = 1 << 7,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP_CAPTURE = 1 << 8,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP = 1 << 9,
		INSTANCE_DATA_FLAG_USE_SH_LIGHTMAP = 1 << 10,
		INSTANCE_DATA_FLAG_USE_VOXEL_GI = 1 << 11,
		INSTANCE_DATA_FLAG_MULTIMESH = 1 << 12,
		INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D = 1 << 13,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR = 1 << 14,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA = 1 << 15,
		INSTANCE_DATA_FLAGS_PARTICLE_TRAIL_SHIFT = 16,
		INSTANCE_DATA_FLAGS_PARTICLE_TRAIL_MASK = 0xFF,
		INSTANCE_DATA_FLAGS_FADE_SHIFT = 24,
		INSTANCE_DATA_FLAGS_FADE_MASK = 0xFF << INSTANCE_DATA_FLAGS_FADE_SHIFT
	};

	struct SceneState {
		// This struct is loaded into Set 1 - Binding 0, populated at start of rendering a frame, must match with shader code
		struct UBO {
			float projection_matrix[16];
			float inv_projection_matrix[16];
			float camera_matrix[16];
			float inv_camera_matrix[16];

			float viewport_size[2];
			float screen_pixel_size[2];

			uint32_t cluster_shift;
			uint32_t cluster_width;
			uint32_t cluster_type_size;
			uint32_t max_cluster_element_count_div_32;

			float directional_penumbra_shadow_kernel[128]; //32 vec4s
			float directional_soft_shadow_kernel[128];
			float penumbra_shadow_kernel[128];
			float soft_shadow_kernel[128];

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
			float opaque_prepass_threshold;
			uint32_t roughness_limiter_pad;

			float sdf_to_bounds[16];

			int32_t sdf_offset[3];
			uint32_t material_uv2_mode;

			int32_t sdf_size[3];
			uint32_t gi_upscale_for_msaa;

			uint32_t volumetric_fog_enabled;
			float volumetric_fog_inv_length;
			float volumetric_fog_detail_spread;
			uint32_t volumetric_fog_pad;

			// Fog
			uint32_t fog_enabled;
			float fog_density;
			float fog_height;
			float fog_height_density;

			float fog_light_color[3];
			float fog_sun_scatter;

			float fog_aerial_perspective;

			float time;
			float reflection_multiplier;

			uint32_t pancake_shadows;
		};

		struct PushConstant {
			uint32_t base_index; //
			uint32_t uv_offset; //packed
			uint32_t pad[2];
		};

		struct InstanceData {
			float transform[16];
			uint32_t flags;
			uint32_t instance_uniforms_ofs; //base offset in global buffer for instance variables
			uint32_t gi_offset; //GI information when using lightmapping (VCT or lightmap index)
			uint32_t layer_mask;
			float lightmap_uv_scale[4];
		};

		UBO ubo;

		LocalVector<RID> uniform_buffers;

		LightmapData lightmaps[MAX_LIGHTMAPS];
		RID lightmap_ids[MAX_LIGHTMAPS];
		bool lightmap_has_sh[MAX_LIGHTMAPS];
		uint32_t lightmaps_used = 0;
		uint32_t max_lightmaps;
		RID lightmap_buffer;

		RID instance_buffer[RENDER_LIST_MAX];
		uint32_t instance_buffer_size[RENDER_LIST_MAX] = { 0, 0, 0 };
		LocalVector<InstanceData> instance_data[RENDER_LIST_MAX];

		LightmapCaptureData *lightmap_captures;
		uint32_t max_lightmap_captures;
		RID lightmap_capture_buffer;

		RID voxelgi_ids[MAX_VOXEL_GI_INSTANCESS];
		uint32_t voxelgis_used = 0;

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

	static RenderForwardClustered *singleton;

	void _setup_environment(const RenderDataRD *p_render_data, bool p_no_fog, const Size2i &p_screen_size, bool p_flip_y, const Color &p_default_bg_color, bool p_opaque_render_buffers = false, bool p_pancake_shadows = false, int p_index = 0);
	void _setup_voxelgis(const PagedArray<RID> &p_voxelgis);
	void _setup_lightmaps(const PagedArray<RID> &p_lightmaps, const Transform3D &p_cam_transform);

	struct RenderElementInfo {
		enum { MAX_REPEATS = (1 << 20) - 1 };
		uint32_t repeat : 20;
		uint32_t uses_projector : 1;
		uint32_t uses_softshadow : 1;
		uint32_t uses_lightmap : 1;
		uint32_t uses_forward_gi : 1;
		uint32_t lod_index : 8;
	};

	template <PassMode p_pass_mode>
	_FORCE_INLINE_ void _render_list_template(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element);

	void _render_list(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element);

	LocalVector<RD::DrawListID> thread_draw_lists;
	void _render_list_thread_function(uint32_t p_thread, RenderListParameters *p_params);
	void _render_list_with_threads(RenderListParameters *p_params, RID p_framebuffer, RD::InitialAction p_initial_color_action, RD::FinalAction p_final_color_action, RD::InitialAction p_initial_depth_action, RD::FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const Vector<RID> &p_storage_textures = Vector<RID>());

	uint32_t render_list_thread_threshold = 500;

	void _update_instance_data_buffer(RenderListType p_render_list);
	void _fill_instance_data(RenderListType p_render_list, int *p_render_info = nullptr, uint32_t p_offset = 0, int32_t p_max_elements = -1, bool p_update_buffer = true);
	void _fill_render_list(RenderListType p_render_list, const RenderDataRD *p_render_data, PassMode p_pass_mode, bool p_using_sdfgi = false, bool p_using_opaque_gi = false, bool p_append = false);

	Map<Size2i, RID> sdfgi_framebuffer_size_cache;

	struct GeometryInstanceData;
	struct GeometryInstanceForwardClustered;

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
				uint64_t lod_index : 8;
				uint64_t surface_index : 8;
				uint64_t geometry_id : 32;
				uint64_t material_id_low : 16;

				uint64_t material_id_hi : 16;
				uint64_t shader_id : 32;
				uint64_t uses_softshadow : 1;
				uint64_t uses_projector : 1;
				uint64_t uses_forward_gi : 1;
				uint64_t uses_lightmap : 1;
				uint64_t depth_layer : 4;
				uint64_t priority : 8;
			};
			struct {
				uint64_t sort_key1;
				uint64_t sort_key2;
			};
		} sort;

		RS::PrimitiveType primitive = RS::PRIMITIVE_MAX;
		uint32_t flags = 0;
		uint32_t surface_index = 0;

		void *surface = nullptr;
		RID material_uniform_set;
		SceneShaderForwardClustered::ShaderData *shader = nullptr;

		void *surface_shadow = nullptr;
		RID material_uniform_set_shadow;
		SceneShaderForwardClustered::ShaderData *shader_shadow = nullptr;

		GeometryInstanceSurfaceDataCache *next = nullptr;
		GeometryInstanceForwardClustered *owner = nullptr;
	};

	struct GeometryInstanceForwardClustered : public GeometryInstance {
		//used during rendering
		bool mirror = false;
		bool non_uniform_scale = false;
		float lod_bias = 0.0;
		float lod_model_scale = 1.0;
		AABB transformed_aabb; //needed for LOD
		float depth = 0;
		uint32_t gi_offset_cache = 0;
		uint32_t flags_cache = 0;
		bool store_transform_cache = true;
		int32_t shader_parameters_offset = -1;
		uint32_t lightmap_slice_index;
		Rect2 lightmap_uv_scale;
		uint32_t layer_mask = 1;
		RID transforms_uniform_set;
		uint32_t instance_count = 0;
		uint32_t trail_steps = 1;
		RID mesh_instance;
		bool can_sdfgi = false;
		bool using_projectors = false;
		bool using_softshadows = false;
		bool fade_near = false;
		float fade_near_begin = 0;
		float fade_near_end = 0;
		bool fade_far = false;
		float fade_far_begin = 0;
		float fade_far_end = 0;
		float force_alpha = 1.0;
		float parent_fade_alpha = 1.0;

		//used during setup
		uint32_t base_flags = 0;
		Transform3D transform;
		RID voxel_gi_instances[MAX_VOXEL_GI_INSTANCESS_PER_INSTANCE];
		RID lightmap_instance;
		GeometryInstanceLightmapSH *lightmap_sh = nullptr;
		GeometryInstanceSurfaceDataCache *surface_caches = nullptr;
		SelfList<GeometryInstanceForwardClustered> dirty_list_element;

		struct Data {
			//data used less often goes into regular heap
			RID base;
			RS::InstanceType base_type;

			RID skeleton;
			Vector<RID> surface_materials;
			RID material_override;
			AABB aabb;

			bool use_dynamic_gi = false;
			bool use_baked_light = false;
			bool cast_double_sided_shadows = false;
			bool mirror = false;
			bool dirty_dependencies = false;

			RendererStorage::DependencyTracker dependency_tracker;
		};

		Data *data = nullptr;

		GeometryInstanceForwardClustered() :
				dirty_list_element(this) {}
	};

	static void _geometry_instance_dependency_changed(RendererStorage::DependencyChangedNotification p_notification, RendererStorage::DependencyTracker *p_tracker);
	static void _geometry_instance_dependency_deleted(const RID &p_dependency, RendererStorage::DependencyTracker *p_tracker);

	SelfList<GeometryInstanceForwardClustered>::List geometry_instance_dirty_list;

	PagedAllocator<GeometryInstanceForwardClustered> geometry_instance_alloc;
	PagedAllocator<GeometryInstanceSurfaceDataCache> geometry_instance_surface_alloc;
	PagedAllocator<GeometryInstanceLightmapSH> geometry_instance_lightmap_sh;

	void _geometry_instance_add_surface_with_material(GeometryInstanceForwardClustered *ginstance, uint32_t p_surface, SceneShaderForwardClustered::MaterialData *p_material, uint32_t p_material_id, uint32_t p_shader_id, RID p_mesh);
	void _geometry_instance_add_surface(GeometryInstanceForwardClustered *ginstance, uint32_t p_surface, RID p_material, RID p_mesh);
	void _geometry_instance_mark_dirty(GeometryInstance *p_geometry_instance);
	void _geometry_instance_update(GeometryInstance *p_geometry_instance);
	void _update_dirty_geometry_instances();

	/* Render List */

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

		void sort_by_reverse_depth_and_priority() { //used for alpha

			SortArray<GeometryInstanceSurfaceDataCache *, SortByReverseDepthAndPriority> sorter;
			sorter.sort(elements.ptr(), elements.size());
		}

		_FORCE_INLINE_ void add_element(GeometryInstanceSurfaceDataCache *p_element) {
			elements.push_back(p_element);
		}
	};

	RenderList render_list[RENDER_LIST_MAX];

	virtual void _update_shader_quality_settings() override;

protected:
	virtual void _render_scene(RenderDataRD *p_render_data, const Color &p_default_bg_color) override;

	virtual void _render_shadow_begin() override;
	virtual void _render_shadow_append(RID p_framebuffer, const PagedArray<GeometryInstance *> &p_instances, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_use_dp, bool p_use_dp_flip, bool p_use_pancake, const Plane &p_camera_plane = Plane(), float p_lod_distance_multiplier = 0.0, float p_screen_lod_threshold = 0.0, const Rect2i &p_rect = Rect2i(), bool p_flip_y = false, bool p_clear_region = true, bool p_begin = true, bool p_end = true, RendererScene::RenderInfo *p_render_info = nullptr) override;
	virtual void _render_shadow_process() override;
	virtual void _render_shadow_end(uint32_t p_barrier = RD::BARRIER_MASK_ALL) override;

	virtual void _render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) override;
	virtual void _render_uv2(const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) override;
	virtual void _render_sdfgi(RID p_render_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const PagedArray<GeometryInstance *> &p_instances, const RID &p_albedo_texture, const RID &p_emission_texture, const RID &p_emission_aniso_texture, const RID &p_geom_facing_texture) override;
	virtual void _render_particle_collider_heightfield(RID p_fb, const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, const PagedArray<GeometryInstance *> &p_instances) override;

public:
	_FORCE_INLINE_ virtual void update_uniform_sets() override {
		base_uniform_set_updated = true;
		_update_render_base_uniform_set();
	}

	virtual GeometryInstance *geometry_instance_create(RID p_base) override;
	virtual void geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton) override;
	virtual void geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override) override;
	virtual void geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_materials) override;
	virtual void geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance) override;
	virtual void geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabb) override;
	virtual void geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask) override;
	virtual void geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias) override;
	virtual void geometry_instance_set_fade_range(GeometryInstance *p_geometry_instance, bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) override;
	virtual void geometry_instance_set_parent_fade_alpha(GeometryInstance *p_geometry_instance, float p_alpha) override;
	virtual void geometry_instance_set_transparency(GeometryInstance *p_geometry_instance, float p_transparency) override;
	virtual void geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable) override;
	virtual void geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable) override;
	virtual void geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) override;
	virtual void geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9) override;
	virtual void geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset) override;
	virtual void geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable) override;

	virtual Transform3D geometry_instance_get_transform(GeometryInstance *p_instance) override;
	virtual AABB geometry_instance_get_aabb(GeometryInstance *p_instance) override;

	virtual void geometry_instance_free(GeometryInstance *p_geometry_instance) override;

	virtual uint32_t geometry_instance_get_pair_mask() override;
	virtual void geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count) override;
	virtual void geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) override;
	virtual void geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count) override;
	virtual void geometry_instance_pair_voxel_gi_instances(GeometryInstance *p_geometry_instance, const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) override;

	virtual void geometry_instance_set_softshadow_projector_pairing(GeometryInstance *p_geometry_instance, bool p_softshadow, bool p_projector) override;

	virtual bool free(RID p_rid) override;

	RenderForwardClustered(RendererStorageRD *p_storage);
	~RenderForwardClustered();
};
} // namespace RendererSceneRenderImplementation
#endif // !RENDERING_SERVER_SCENE_RENDER_FORWARD_CLUSTERED_H
