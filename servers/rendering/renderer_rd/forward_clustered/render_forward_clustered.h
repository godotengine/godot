/**************************************************************************/
/*  render_forward_clustered.h                                            */
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

#ifndef RENDER_FORWARD_CLUSTERED_H
#define RENDER_FORWARD_CLUSTERED_H

#include "core/templates/paged_allocator.h"
#include "servers/rendering/renderer_rd/cluster_builder_rd.h"
#include "servers/rendering/renderer_rd/effects/fsr2.h"
#include "servers/rendering/renderer_rd/effects/resolve.h"
#include "servers/rendering/renderer_rd/effects/ss_effects.h"
#include "servers/rendering/renderer_rd/effects/taa.h"
#include "servers/rendering/renderer_rd/forward_clustered/scene_shader_forward_clustered.h"
#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/shaders/forward_clustered/scene_forward_clustered.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/utilities.h"

#define RB_SCOPE_FORWARD_CLUSTERED SNAME("forward_clustered")

#define RB_TEX_SPECULAR SNAME("specular")
#define RB_TEX_SPECULAR_MSAA SNAME("specular_msaa")
#define RB_TEX_ROUGHNESS SNAME("normal_roughnesss")
#define RB_TEX_ROUGHNESS_MSAA SNAME("normal_roughnesss_msaa")
#define RB_TEX_VOXEL_GI SNAME("voxel_gi")
#define RB_TEX_VOXEL_GI_MSAA SNAME("voxel_gi_msaa")

namespace RendererSceneRenderImplementation {

class RenderForwardClustered : public RendererSceneRenderRD {
	friend SceneShaderForwardClustered;

	enum {
		SCENE_UNIFORM_SET = 0,
		RENDER_PASS_UNIFORM_SET = 1,
		TRANSFORMS_UNIFORM_SET = 2,
		MATERIAL_UNIFORM_SET = 3,
	};

	const int SAMPLERS_BINDING_FIRST_INDEX = 16;

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
		RENDER_LIST_MOTION, //used for opaque objects with motion
		RENDER_LIST_ALPHA, //used for transparent objects
		RENDER_LIST_SECONDARY, //used for shadows and other objects
		RENDER_LIST_MAX
	};

	/* Scene Shader */

	SceneShaderForwardClustered scene_shader;

	/* Framebuffer */

	class RenderBufferDataForwardClustered : public RenderBufferCustomDataRD {
		GDCLASS(RenderBufferDataForwardClustered, RenderBufferCustomDataRD)

	private:
		RenderSceneBuffersRD *render_buffers = nullptr;
		RendererRD::FSR2Context *fsr2_context = nullptr;

	public:
		ClusterBuilderRD *cluster_builder = nullptr;

		struct SSEffectsData {
			Projection last_frame_projections[RendererSceneRender::MAX_RENDER_VIEWS];
			Transform3D last_frame_transform;

			RendererRD::SSEffects::SSILRenderBuffers ssil;
			RendererRD::SSEffects::SSAORenderBuffers ssao;
			RendererRD::SSEffects::SSRRenderBuffers ssr;
		} ss_effects_data;

		enum DepthFrameBufferType {
			DEPTH_FB,
			DEPTH_FB_ROUGHNESS,
			DEPTH_FB_ROUGHNESS_VOXELGI
		};

		RID render_sdfgi_uniform_set;

		void ensure_specular();
		bool has_specular() const { return render_buffers->has_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_SPECULAR); }
		RID get_specular() const { return render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_SPECULAR); }
		RID get_specular(uint32_t p_layer) { return render_buffers->get_texture_slice(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_SPECULAR, p_layer, 0); }
		RID get_specular_msaa(uint32_t p_layer) { return render_buffers->get_texture_slice(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_SPECULAR_MSAA, p_layer, 0); }

		void ensure_normal_roughness_texture();
		bool has_normal_roughness() const { return render_buffers->has_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_ROUGHNESS); }
		RID get_normal_roughness() const { return render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_ROUGHNESS); }
		RID get_normal_roughness(uint32_t p_layer) { return render_buffers->get_texture_slice(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_ROUGHNESS, p_layer, 0); }
		RID get_normal_roughness_msaa() const { return render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_ROUGHNESS_MSAA); }
		RID get_normal_roughness_msaa(uint32_t p_layer) { return render_buffers->get_texture_slice(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_ROUGHNESS_MSAA, p_layer, 0); }

		void ensure_voxelgi();
		bool has_voxelgi() const { return render_buffers->has_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_VOXEL_GI); }
		RID get_voxelgi() const { return render_buffers->get_texture(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_VOXEL_GI); }
		RID get_voxelgi(uint32_t p_layer) { return render_buffers->get_texture_slice(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_VOXEL_GI, p_layer, 0); }
		RID get_voxelgi_msaa(uint32_t p_layer) { return render_buffers->get_texture_slice(RB_SCOPE_FORWARD_CLUSTERED, RB_TEX_VOXEL_GI_MSAA, p_layer, 0); }

		void ensure_fsr2(RendererRD::FSR2Effect *p_effect);
		RendererRD::FSR2Context *get_fsr2_context() const { return fsr2_context; }

		RID get_color_only_fb();
		RID get_color_pass_fb(uint32_t p_color_pass_flags);
		RID get_depth_fb(DepthFrameBufferType p_type = DEPTH_FB);
		RID get_specular_only_fb();
		RID get_velocity_only_fb();

		virtual void configure(RenderSceneBuffersRD *p_render_buffers) override;
		virtual void free_data() override;
	};

	virtual void setup_render_buffer_data(Ref<RenderSceneBuffersRD> p_render_buffers) override;

	enum BaseUniformSetCache {
		BASE_UNIFORM_SET_CACHE_VIEWPORT,
		BASE_UNIFORM_SET_CACHE_DEFAULT,
		BASE_UNIFORM_SET_CACHE_MAX
	};

	RID render_base_uniform_set;
	// One for custom samplers, one for default samplers.
	// Need to switch between them as default is needed for probes, shadows, materials, etc.
	RID render_base_uniform_set_cache[BASE_UNIFORM_SET_CACHE_MAX];

	uint64_t lightmap_texture_array_version_cache[BASE_UNIFORM_SET_CACHE_MAX] = { 0xFFFFFFFF, 0xFFFFFFFF };

	void _update_render_base_uniform_set(const RendererRD::MaterialStorage::Samplers &p_samplers, BaseUniformSetCache p_cache_index);
	RID _setup_sdfgi_render_pass_uniform_set(RID p_albedo_texture, RID p_emission_texture, RID p_emission_aniso_texture, RID p_geom_facing_texture);
	RID _setup_render_pass_uniform_set(RenderListType p_render_list, const RenderDataRD *p_render_data, RID p_radiance_texture, bool p_use_directional_shadow_atlas = false, int p_index = 0);

	enum PassMode {
		PASS_MODE_COLOR,
		PASS_MODE_SHADOW,
		PASS_MODE_SHADOW_DP,
		PASS_MODE_DEPTH,
		PASS_MODE_DEPTH_NORMAL_ROUGHNESS,
		PASS_MODE_DEPTH_NORMAL_ROUGHNESS_VOXEL_GI,
		PASS_MODE_DEPTH_MATERIAL,
		PASS_MODE_SDF,
	};

	enum ColorPassFlags {
		COLOR_PASS_FLAG_TRANSPARENT = 1 << 0,
		COLOR_PASS_FLAG_SEPARATE_SPECULAR = 1 << 1,
		COLOR_PASS_FLAG_MULTIVIEW = 1 << 2,
		COLOR_PASS_FLAG_MOTION_VECTORS = 1 << 3,
	};

	struct GeometryInstanceSurfaceDataCache;
	struct RenderElementInfo;

	struct RenderListParameters {
		GeometryInstanceSurfaceDataCache **elements = nullptr;
		RenderElementInfo *element_info = nullptr;
		int element_count = 0;
		bool reverse_cull = false;
		PassMode pass_mode = PASS_MODE_COLOR;
		uint32_t color_pass_flags = 0;
		bool no_gi = false;
		uint32_t view_count = 1;
		RID render_pass_uniform_set;
		bool force_wireframe = false;
		Vector2 uv_offset;
		float lod_distance_multiplier = 0.0;
		float screen_mesh_lod_threshold = 0.0;
		RD::FramebufferFormatID framebuffer_format = 0;
		uint32_t element_offset = 0;
		uint32_t barrier = RD::BARRIER_MASK_ALL_BARRIERS;
		bool use_directional_soft_shadow = false;

		RenderListParameters(GeometryInstanceSurfaceDataCache **p_elements, RenderElementInfo *p_element_info, int p_element_count, bool p_reverse_cull, PassMode p_pass_mode, uint32_t p_color_pass_flags, bool p_no_gi, bool p_use_directional_soft_shadows, RID p_render_pass_uniform_set, bool p_force_wireframe = false, const Vector2 &p_uv_offset = Vector2(), float p_lod_distance_multiplier = 0.0, float p_screen_mesh_lod_threshold = 0.0, uint32_t p_view_count = 1, uint32_t p_element_offset = 0, uint32_t p_barrier = RD::BARRIER_MASK_ALL_BARRIERS) {
			elements = p_elements;
			element_info = p_element_info;
			element_count = p_element_count;
			reverse_cull = p_reverse_cull;
			pass_mode = p_pass_mode;
			color_pass_flags = p_color_pass_flags;
			no_gi = p_no_gi;
			view_count = p_view_count;
			render_pass_uniform_set = p_render_pass_uniform_set;
			force_wireframe = p_force_wireframe;
			uv_offset = p_uv_offset;
			lod_distance_multiplier = p_lod_distance_multiplier;
			screen_mesh_lod_threshold = p_screen_mesh_lod_threshold;
			element_offset = p_element_offset;
			barrier = p_barrier;
			use_directional_soft_shadow = p_use_directional_soft_shadows;
		}
	};

	struct LightmapData {
		float normal_xform[12];
		float pad[3];
		float exposure_normalization;
	};

	struct LightmapCaptureData {
		float sh[9 * 4];
	};

	// When changing any of these enums, remember to change the corresponding enums in the shader files as well.
	enum {
		INSTANCE_DATA_FLAGS_NON_UNIFORM_SCALE = 1 << 4,
		INSTANCE_DATA_FLAG_USE_GI_BUFFERS = 1 << 5,
		INSTANCE_DATA_FLAG_USE_SDFGI = 1 << 6,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP_CAPTURE = 1 << 7,
		INSTANCE_DATA_FLAG_USE_LIGHTMAP = 1 << 8,
		INSTANCE_DATA_FLAG_USE_SH_LIGHTMAP = 1 << 9,
		INSTANCE_DATA_FLAG_USE_VOXEL_GI = 1 << 10,
		INSTANCE_DATA_FLAG_PARTICLES = 1 << 11,
		INSTANCE_DATA_FLAG_MULTIMESH = 1 << 12,
		INSTANCE_DATA_FLAG_MULTIMESH_FORMAT_2D = 1 << 13,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_COLOR = 1 << 14,
		INSTANCE_DATA_FLAG_MULTIMESH_HAS_CUSTOM_DATA = 1 << 15,
		INSTANCE_DATA_FLAGS_PARTICLE_TRAIL_SHIFT = 16,
		INSTANCE_DATA_FLAGS_PARTICLE_TRAIL_MASK = 0xFF,
		INSTANCE_DATA_FLAGS_FADE_SHIFT = 24,
		INSTANCE_DATA_FLAGS_FADE_MASK = 0xFFUL << INSTANCE_DATA_FLAGS_FADE_SHIFT
	};

	struct SceneState {
		// This struct is loaded into Set 1 - Binding 1, populated at start of rendering a frame, must match with shader code
		struct UBO {
			uint32_t cluster_shift;
			uint32_t cluster_width;
			uint32_t cluster_type_size;
			uint32_t max_cluster_element_count_div_32;

			uint32_t ss_effects_flags;
			float ssao_light_affect;
			float ssao_ao_affect;
			uint32_t pad1;

			float sdf_to_bounds[16];

			int32_t sdf_offset[3];
			uint32_t pad2;

			int32_t sdf_size[3];
			uint32_t gi_upscale_for_msaa;

			uint32_t volumetric_fog_enabled;
			float volumetric_fog_inv_length;
			float volumetric_fog_detail_spread;
			uint32_t volumetric_fog_pad;
		};

		struct PushConstant {
			uint32_t base_index; //
			uint32_t uv_offset; //packed
			uint32_t multimesh_motion_vectors_current_offset;
			uint32_t multimesh_motion_vectors_previous_offset;
		};

		struct InstanceData {
			float transform[16];
			float prev_transform[16];
			uint32_t flags;
			uint32_t instance_uniforms_ofs; //base offset in global buffer for instance variables
			uint32_t gi_offset; //GI information when using lightmapping (VCT or lightmap index)
			uint32_t layer_mask;
			float lightmap_uv_scale[4];
			float compressed_aabb_position[4];
			float compressed_aabb_size[4];
			float uv_scale[4];
		};

		UBO ubo;

		LocalVector<RID> uniform_buffers;
		LocalVector<RID> implementation_uniform_buffers;

		LightmapData lightmaps[MAX_LIGHTMAPS];
		RID lightmap_ids[MAX_LIGHTMAPS];
		bool lightmap_has_sh[MAX_LIGHTMAPS];
		uint32_t lightmaps_used = 0;
		uint32_t max_lightmaps;
		RID lightmap_buffer;

		RID instance_buffer[RENDER_LIST_MAX];
		uint32_t instance_buffer_size[RENDER_LIST_MAX] = { 0, 0, 0 };
		LocalVector<InstanceData> instance_data[RENDER_LIST_MAX];

		LightmapCaptureData *lightmap_captures = nullptr;
		uint32_t max_lightmap_captures;
		RID lightmap_capture_buffer;

		RID voxelgi_ids[MAX_VOXEL_GI_INSTANCESS];
		uint32_t voxelgis_used = 0;

		bool used_screen_texture = false;
		bool used_normal_texture = false;
		bool used_depth_texture = false;
		bool used_sss = false;
		bool used_lightmap = false;

		struct ShadowPass {
			uint32_t element_from;
			uint32_t element_count;
			bool flip_cull;
			PassMode pass_mode;

			RID rp_uniform_set;
			Plane camera_plane;
			float lod_distance_multiplier;
			float screen_mesh_lod_threshold;

			RID framebuffer;
			RD::InitialAction initial_depth_action;
			RD::FinalAction final_depth_action;
			Rect2i rect;
		};

		LocalVector<ShadowPass> shadow_passes;

	} scene_state;

	static RenderForwardClustered *singleton;

	void _setup_environment(const RenderDataRD *p_render_data, bool p_no_fog, const Size2i &p_screen_size, bool p_flip_y, const Color &p_default_bg_color, bool p_opaque_render_buffers = false, bool p_apply_alpha_multiplier = false, bool p_pancake_shadows = false, int p_index = 0);
	void _setup_voxelgis(const PagedArray<RID> &p_voxelgis);
	void _setup_lightmaps(const RenderDataRD *p_render_data, const PagedArray<RID> &p_lightmaps, const Transform3D &p_cam_transform);

	struct RenderElementInfo {
		enum { MAX_REPEATS = (1 << 20) - 1 };
		uint32_t repeat : 20;
		uint32_t uses_projector : 1;
		uint32_t uses_softshadow : 1;
		uint32_t uses_lightmap : 1;
		uint32_t uses_forward_gi : 1;
		uint32_t lod_index : 8;
	};

	template <PassMode p_pass_mode, uint32_t p_color_pass_flags = 0>
	_FORCE_INLINE_ void _render_list_template(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element);

	void _render_list(RenderingDevice::DrawListID p_draw_list, RenderingDevice::FramebufferFormatID p_framebuffer_Format, RenderListParameters *p_params, uint32_t p_from_element, uint32_t p_to_element);

	LocalVector<RD::DrawListID> thread_draw_lists;
	void _render_list_thread_function(uint32_t p_thread, RenderListParameters *p_params);
	void _render_list_with_threads(RenderListParameters *p_params, RID p_framebuffer, RD::InitialAction p_initial_color_action, RD::FinalAction p_final_color_action, RD::InitialAction p_initial_depth_action, RD::FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const Vector<RID> &p_storage_textures = Vector<RID>());

	uint32_t render_list_thread_threshold = 500;

	void _update_instance_data_buffer(RenderListType p_render_list);
	void _fill_instance_data(RenderListType p_render_list, int *p_render_info = nullptr, uint32_t p_offset = 0, int32_t p_max_elements = -1, bool p_update_buffer = true);
	void _fill_render_list(RenderListType p_render_list, const RenderDataRD *p_render_data, PassMode p_pass_mode, bool p_using_sdfgi = false, bool p_using_opaque_gi = false, bool p_using_motion_pass = false, bool p_append = false);

	HashMap<Size2i, RID> sdfgi_framebuffer_size_cache;

	struct GeometryInstanceData;
	class GeometryInstanceForwardClustered;

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
			FLAG_USES_MOTION_VECTOR = 131072,
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
		uint32_t color_pass_inclusion_mask = 0;

		void *surface = nullptr;
		RID material_uniform_set;
		SceneShaderForwardClustered::ShaderData *shader = nullptr;
		SceneShaderForwardClustered::MaterialData *material = nullptr;

		void *surface_shadow = nullptr;
		RID material_uniform_set_shadow;
		SceneShaderForwardClustered::ShaderData *shader_shadow = nullptr;

		GeometryInstanceSurfaceDataCache *next = nullptr;
		GeometryInstanceForwardClustered *owner = nullptr;
	};

	class GeometryInstanceForwardClustered : public RenderGeometryInstanceBase {
	public:
		// lightmap
		RID lightmap_instance;
		Rect2 lightmap_uv_scale;
		uint32_t lightmap_slice_index;
		GeometryInstanceLightmapSH *lightmap_sh = nullptr;

		//used during rendering

		uint32_t gi_offset_cache = 0;
		bool store_transform_cache = true;
		RID transforms_uniform_set;
		uint32_t instance_count = 0;
		uint32_t trail_steps = 1;
		bool can_sdfgi = false;
		bool using_projectors = false;
		bool using_softshadows = false;

		//used during setup
		uint64_t prev_transform_change_frame = 0xFFFFFFFF;
		bool prev_transform_dirty = true;
		Transform3D prev_transform;
		RID voxel_gi_instances[MAX_VOXEL_GI_INSTANCESS_PER_INSTANCE];
		GeometryInstanceSurfaceDataCache *surface_caches = nullptr;
		SelfList<GeometryInstanceForwardClustered> dirty_list_element;

		GeometryInstanceForwardClustered() :
				dirty_list_element(this) {}

		virtual void _mark_dirty() override;

		virtual void set_transform(const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabb) override;
		virtual void set_use_lightmap(RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) override;
		virtual void set_lightmap_capture(const Color *p_sh9) override;

		virtual void pair_light_instances(const RID *p_light_instances, uint32_t p_light_instance_count) override {}
		virtual void pair_reflection_probe_instances(const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) override {}
		virtual void pair_decal_instances(const RID *p_decal_instances, uint32_t p_decal_instance_count) override {}
		virtual void pair_voxel_gi_instances(const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) override;

		virtual void set_softshadow_projector_pairing(bool p_softshadow, bool p_projector) override;
	};

	static void _geometry_instance_dependency_changed(Dependency::DependencyChangedNotification p_notification, DependencyTracker *p_tracker);
	static void _geometry_instance_dependency_deleted(const RID &p_dependency, DependencyTracker *p_tracker);

	SelfList<GeometryInstanceForwardClustered>::List geometry_instance_dirty_list;

	PagedAllocator<GeometryInstanceForwardClustered> geometry_instance_alloc;
	PagedAllocator<GeometryInstanceSurfaceDataCache> geometry_instance_surface_alloc;
	PagedAllocator<GeometryInstanceLightmapSH> geometry_instance_lightmap_sh;

	void _geometry_instance_add_surface_with_material(GeometryInstanceForwardClustered *ginstance, uint32_t p_surface, SceneShaderForwardClustered::MaterialData *p_material, uint32_t p_material_id, uint32_t p_shader_id, RID p_mesh);
	void _geometry_instance_add_surface_with_material_chain(GeometryInstanceForwardClustered *ginstance, uint32_t p_surface, SceneShaderForwardClustered::MaterialData *p_material, RID p_mat_src, RID p_mesh);
	void _geometry_instance_add_surface(GeometryInstanceForwardClustered *ginstance, uint32_t p_surface, RID p_material, RID p_mesh);
	void _geometry_instance_update(RenderGeometryInstance *p_geometry_instance);
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

	/* Effects */

	RendererRD::Resolve *resolve_effects = nullptr;
	RendererRD::TAA *taa = nullptr;
	RendererRD::FSR2Effect *fsr2_effect = nullptr;
	RendererRD::SSEffects *ss_effects = nullptr;

	/* Cluster builder */

	ClusterBuilderSharedDataRD cluster_builder_shared;
	ClusterBuilderRD *current_cluster_builder = nullptr;

	/* SDFGI */
	void _update_sdfgi(RenderDataRD *p_render_data);

	/* Volumetric fog */
	RID shadow_sampler;

	void _update_volumetric_fog(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_environment, const Projection &p_cam_projection, const Transform3D &p_cam_transform, const Transform3D &p_prev_cam_inv_transform, RID p_shadow_atlas, int p_directional_light_count, bool p_use_directional_shadows, int p_positional_light_count, int p_voxel_gi_count, const PagedArray<RID> &p_fog_volumes);

	/* Render shadows */

	void _render_shadow_pass(RID p_light, RID p_shadow_atlas, int p_pass, const PagedArray<RenderGeometryInstance *> &p_instances, const Plane &p_camera_plane = Plane(), float p_lod_distance_multiplier = 0, float p_screen_mesh_lod_threshold = 0.0, bool p_open_pass = true, bool p_close_pass = true, bool p_clear_region = true, RenderingMethod::RenderInfo *p_render_info = nullptr, const Size2i &p_viewport_size = Size2i(1, 1));
	void _render_shadow_begin();
	void _render_shadow_append(RID p_framebuffer, const PagedArray<RenderGeometryInstance *> &p_instances, const Projection &p_projection, const Transform3D &p_transform, float p_zfar, float p_bias, float p_normal_bias, bool p_reverse_cull_face, bool p_use_dp, bool p_use_dp_flip, bool p_use_pancake, const Plane &p_camera_plane = Plane(), float p_lod_distance_multiplier = 0.0, float p_screen_mesh_lod_threshold = 0.0, const Rect2i &p_rect = Rect2i(), bool p_flip_y = false, bool p_clear_region = true, bool p_begin = true, bool p_end = true, RenderingMethod::RenderInfo *p_render_info = nullptr, const Size2i &p_viewport_size = Size2i(1, 1));
	void _render_shadow_process();
	void _render_shadow_end(uint32_t p_barrier = RD::BARRIER_MASK_ALL_BARRIERS);

	/* Render Scene */
	void _process_ssao(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_environment, const RID *p_normal_buffers, const Projection *p_projections);
	void _process_ssil(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_environment, const RID *p_normal_buffers, const Projection *p_projections, const Transform3D &p_transform);
	void _copy_framebuffer_to_ssil(Ref<RenderSceneBuffersRD> p_render_buffers);
	void _pre_opaque_render(RenderDataRD *p_render_data, bool p_use_ssao, bool p_use_ssil, bool p_use_gi, const RID *p_normal_roughness_slices, RID p_voxel_gi_buffer);
	void _process_ssr(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_dest_framebuffer, const RID *p_normal_buffer_slices, RID p_specular_buffer, const RID *p_metallic_slices, RID p_environment, const Projection *p_projections, const Vector3 *p_eye_offsets, bool p_use_additive);
	void _process_sss(Ref<RenderSceneBuffersRD> p_render_buffers, const Projection &p_camera);

	/* Debug */
	void _debug_draw_cluster(Ref<RenderSceneBuffersRD> p_render_buffers);

protected:
	/* setup */

	virtual RID _render_buffers_get_normal_texture(Ref<RenderSceneBuffersRD> p_render_buffers) override;
	virtual RID _render_buffers_get_velocity_texture(Ref<RenderSceneBuffersRD> p_render_buffers) override;

	virtual void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override;
	virtual void environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override;
	virtual void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) override;

	virtual void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) override;
	virtual void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) override;

	/* Rendering */

	virtual void _render_scene(RenderDataRD *p_render_data, const Color &p_default_bg_color) override;
	virtual void _render_buffers_debug_draw(const RenderDataRD *p_render_data) override;

	virtual void _render_material(const Transform3D &p_cam_transform, const Projection &p_cam_projection, bool p_cam_orthogonal, const PagedArray<RenderGeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region, float p_exposure_normalization) override;
	virtual void _render_uv2(const PagedArray<RenderGeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) override;
	virtual void _render_sdfgi(Ref<RenderSceneBuffersRD> p_render_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const PagedArray<RenderGeometryInstance *> &p_instances, const RID &p_albedo_texture, const RID &p_emission_texture, const RID &p_emission_aniso_texture, const RID &p_geom_facing_texture, float p_exposure_normalization) override;
	virtual void _render_particle_collider_heightfield(RID p_fb, const Transform3D &p_cam_transform, const Projection &p_cam_projection, const PagedArray<RenderGeometryInstance *> &p_instances) override;

public:
	static RenderForwardClustered *get_singleton() { return singleton; }

	ClusterBuilderSharedDataRD *get_cluster_builder_shared() { return &cluster_builder_shared; }
	RendererRD::SSEffects *get_ss_effects() { return ss_effects; }

	/* callback from updating our lighting UBOs, used to populate cluster builder */
	virtual void setup_added_reflection_probe(const Transform3D &p_transform, const Vector3 &p_half_size) override;
	virtual void setup_added_light(const RS::LightType p_type, const Transform3D &p_transform, float p_radius, float p_spot_aperture) override;
	virtual void setup_added_decal(const Transform3D &p_transform, const Vector3 &p_half_size) override;

	virtual void base_uniforms_changed() override;

	/* SDFGI UPDATE */

	virtual void sdfgi_update(const Ref<RenderSceneBuffers> &p_render_buffers, RID p_environment, const Vector3 &p_world_position) override;
	virtual int sdfgi_get_pending_region_count(const Ref<RenderSceneBuffers> &p_render_buffers) const override;
	virtual AABB sdfgi_get_pending_region_bounds(const Ref<RenderSceneBuffers> &p_render_buffers, int p_region) const override;
	virtual uint32_t sdfgi_get_pending_region_cascade(const Ref<RenderSceneBuffers> &p_render_buffers, int p_region) const override;
	RID sdfgi_get_ubo() const { return gi.sdfgi_ubo; }

	/* GEOMETRY INSTANCE */

	virtual RenderGeometryInstance *geometry_instance_create(RID p_base) override;
	virtual void geometry_instance_free(RenderGeometryInstance *p_geometry_instance) override;

	virtual uint32_t geometry_instance_get_pair_mask() override;

	virtual bool free(RID p_rid) override;

	RenderForwardClustered();
	~RenderForwardClustered();
};
} // namespace RendererSceneRenderImplementation

#endif // RENDER_FORWARD_CLUSTERED_H
