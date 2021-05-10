/*************************************************************************/
/*  rendering_server_default.h                                           */
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

#ifndef RENDERING_SERVER_DEFAULT_H
#define RENDERING_SERVER_DEFAULT_H

#include "core/math/octree.h"
#include "core/templates/command_queue_mt.h"
#include "core/templates/ordered_hash_map.h"
#include "renderer_canvas_cull.h"
#include "renderer_scene_cull.h"
#include "renderer_viewport.h"
#include "rendering_server_globals.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering_server.h"
#include "servers/server_wrap_mt_common.h"

class RenderingServerDefault : public RenderingServer {
	enum {
		MAX_INSTANCE_CULL = 8192,
		MAX_INSTANCE_LIGHTS = 4,
		LIGHT_CACHE_DIRTY = -1,
		MAX_LIGHTS_CULLED = 256,
		MAX_ROOM_CULL = 32,
		MAX_EXTERIOR_PORTALS = 128,
		MAX_LIGHT_SAMPLERS = 256,
		INSTANCE_ROOMLESS_MASK = (1 << 20)

	};

	static int changes;
	RID test_cube;

	int black_margin[4];
	RID black_image[4];

	struct FrameDrawnCallbacks {
		ObjectID object;
		StringName method;
		Variant param;
	};

	List<FrameDrawnCallbacks> frame_drawn_callbacks;

	void _draw_margins();
	static void _changes_changed() {}

	uint64_t frame_profile_frame;
	Vector<FrameProfileArea> frame_profile;

	float frame_setup_time = 0;

	//for printing
	bool print_gpu_profile = false;
	OrderedHashMap<String, float> print_gpu_profile_task_time;
	uint64_t print_frame_profile_ticks_from = 0;
	uint32_t print_frame_profile_frame_count = 0;

	mutable CommandQueueMT command_queue;

	static void _thread_callback(void *_instance);
	void _thread_loop();

	Thread::ID server_thread;
	SafeFlag exit;
	Thread thread;
	SafeFlag draw_thread_up;
	bool create_thread;

	SafeNumeric<uint64_t> draw_pending;
	void _thread_draw(bool p_swap_buffers, double frame_step);
	void _thread_flush();

	void _thread_exit();

	Mutex alloc_mutex;

	void _draw(bool p_swap_buffers, double frame_step);
	void _init();
	void _finish();

	void _free(RID p_rid);

public:
	//if editor is redrawing when it shouldn't, enable this and put a breakpoint in _changes_changed()
	//#define DEBUG_CHANGES

#ifdef DEBUG_CHANGES
	_FORCE_INLINE_ static void redraw_request() {
		changes++;
		_changes_changed();
	}

#define DISPLAY_CHANGED \
	changes++;          \
	_changes_changed();

#else
	_FORCE_INLINE_ static void redraw_request() { changes++; }
#endif

#define WRITE_ACTION redraw_request();

#ifdef DEBUG_SYNC
#define SYNC_DEBUG print_line("sync on: " + String(__FUNCTION__));
#else
#define SYNC_DEBUG
#endif

#include "servers/server_wrap_mt_common.h"

//from now on, calls forwarded to this singleton
#define ServerName RendererStorage
#define server_name RSG::storage

	/* TEXTURE API */

#define FUNCRIDTEX0(m_type)                                                                           \
	virtual RID m_type##_create() override {                                                          \
		RID ret = RSG::storage->texture_allocate();                                                   \
		if (Thread::get_caller_id() == server_thread || RSG::storage->can_create_resources_async()) { \
			RSG::storage->m_type##_initialize(ret);                                                   \
		} else {                                                                                      \
			command_queue.push(RSG::storage, &RendererStorage::m_type##_initialize, ret);             \
		}                                                                                             \
		return ret;                                                                                   \
	}

#define FUNCRIDTEX1(m_type, m_type1)                                                                  \
	virtual RID m_type##_create(m_type1 p1) override {                                                \
		RID ret = RSG::storage->texture_allocate();                                                   \
		if (Thread::get_caller_id() == server_thread || RSG::storage->can_create_resources_async()) { \
			RSG::storage->m_type##_initialize(ret, p1);                                               \
		} else {                                                                                      \
			command_queue.push(RSG::storage, &RendererStorage::m_type##_initialize, ret, p1);         \
		}                                                                                             \
		return ret;                                                                                   \
	}

#define FUNCRIDTEX2(m_type, m_type1, m_type2)                                                         \
	virtual RID m_type##_create(m_type1 p1, m_type2 p2) override {                                    \
		RID ret = RSG::storage->texture_allocate();                                                   \
		if (Thread::get_caller_id() == server_thread || RSG::storage->can_create_resources_async()) { \
			RSG::storage->m_type##_initialize(ret, p1, p2);                                           \
		} else {                                                                                      \
			command_queue.push(RSG::storage, &RendererStorage::m_type##_initialize, ret, p1, p2);     \
		}                                                                                             \
		return ret;                                                                                   \
	}

#define FUNCRIDTEX6(m_type, m_type1, m_type2, m_type3, m_type4, m_type5, m_type6)                                  \
	virtual RID m_type##_create(m_type1 p1, m_type2 p2, m_type3 p3, m_type4 p4, m_type5 p5, m_type6 p6) override { \
		RID ret = RSG::storage->texture_allocate();                                                                \
		if (Thread::get_caller_id() == server_thread || RSG::storage->can_create_resources_async()) {              \
			RSG::storage->m_type##_initialize(ret, p1, p2, p3, p4, p5, p6);                                        \
		} else {                                                                                                   \
			command_queue.push(RSG::storage, &RendererStorage::m_type##_initialize, ret, p1, p2, p3, p4, p5, p6);  \
		}                                                                                                          \
		return ret;                                                                                                \
	}

	//these go pass-through, as they can be called from any thread
	FUNCRIDTEX1(texture_2d, const Ref<Image> &)
	FUNCRIDTEX2(texture_2d_layered, const Vector<Ref<Image>> &, TextureLayeredType)
	FUNCRIDTEX6(texture_3d, Image::Format, int, int, int, bool, const Vector<Ref<Image>> &)
	FUNCRIDTEX1(texture_proxy, RID)

	//goes pass-through
	FUNC3(texture_2d_update_immediate, RID, const Ref<Image> &, int)
	//these go through command queue if they are in another thread
	FUNC3(texture_2d_update, RID, const Ref<Image> &, int)
	FUNC2(texture_3d_update, RID, const Vector<Ref<Image>> &)
	FUNC2(texture_proxy_update, RID, RID)

	//these also go pass-through
	FUNCRIDTEX0(texture_2d_placeholder)
	FUNCRIDTEX1(texture_2d_layered_placeholder, TextureLayeredType)
	FUNCRIDTEX0(texture_3d_placeholder)

	FUNC1RC(Ref<Image>, texture_2d_get, RID)
	FUNC2RC(Ref<Image>, texture_2d_layer_get, RID, int)
	FUNC1RC(Vector<Ref<Image>>, texture_3d_get, RID)

	FUNC2(texture_replace, RID, RID)

	FUNC3(texture_set_size_override, RID, int, int)
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	FUNC2(texture_bind, RID, uint32_t)
#endif

	FUNC3(texture_set_detect_3d_callback, RID, TextureDetectCallback, void *)
	FUNC3(texture_set_detect_normal_callback, RID, TextureDetectCallback, void *)
	FUNC3(texture_set_detect_roughness_callback, RID, TextureDetectRoughnessCallback, void *)

	FUNC2(texture_set_path, RID, const String &)
	FUNC1RC(String, texture_get_path, RID)
	FUNC1(texture_debug_usage, List<TextureInfo> *)

	FUNC2(texture_set_force_redraw_if_visible, RID, bool)

	/* SHADER API */

	FUNCRIDSPLIT(shader)

	FUNC2(shader_set_code, RID, const String &)
	FUNC1RC(String, shader_get_code, RID)

	FUNC2SC(shader_get_param_list, RID, List<PropertyInfo> *)

	FUNC3(shader_set_default_texture_param, RID, const StringName &, RID)
	FUNC2RC(RID, shader_get_default_texture_param, RID, const StringName &)
	FUNC2RC(Variant, shader_get_param_default, RID, const StringName &)

	FUNC1RC(ShaderNativeSourceCode, shader_get_native_source_code, RID)

	/* COMMON MATERIAL API */

	FUNCRIDSPLIT(material)

	FUNC2(material_set_shader, RID, RID)

	FUNC3(material_set_param, RID, const StringName &, const Variant &)
	FUNC2RC(Variant, material_get_param, RID, const StringName &)

	FUNC2(material_set_render_priority, RID, int)
	FUNC2(material_set_next_pass, RID, RID)

	/* MESH API */

	virtual RID mesh_create_from_surfaces(const Vector<SurfaceData> &p_surfaces, int p_blend_shape_count = 0) override {
		RID mesh = RSG::storage->mesh_allocate();

		if (Thread::get_caller_id() == server_thread || RSG::storage->can_create_resources_async()) {
			if (Thread::get_caller_id() == server_thread) {
				command_queue.flush_if_pending();
			}
			RSG::storage->mesh_initialize(mesh);
			RSG::storage->mesh_set_blend_shape_count(mesh, p_blend_shape_count);
			for (int i = 0; i < p_surfaces.size(); i++) {
				RSG::storage->mesh_add_surface(mesh, p_surfaces[i]);
			}
		} else {
			command_queue.push(RSG::storage, &RendererStorage::mesh_initialize, mesh);
			command_queue.push(RSG::storage, &RendererStorage::mesh_set_blend_shape_count, mesh, p_blend_shape_count);
			for (int i = 0; i < p_surfaces.size(); i++) {
				RSG::storage->mesh_add_surface(mesh, p_surfaces[i]);
				command_queue.push(RSG::storage, &RendererStorage::mesh_add_surface, mesh, p_surfaces[i]);
			}
		}

		return mesh;
	}

	FUNC2(mesh_set_blend_shape_count, RID, int)

	FUNCRIDSPLIT(mesh)

	FUNC2(mesh_add_surface, RID, const SurfaceData &)

	FUNC1RC(int, mesh_get_blend_shape_count, RID)

	FUNC2(mesh_set_blend_shape_mode, RID, BlendShapeMode)
	FUNC1RC(BlendShapeMode, mesh_get_blend_shape_mode, RID)

	FUNC4(mesh_surface_update_region, RID, int, int, const Vector<uint8_t> &)

	FUNC3(mesh_surface_set_material, RID, int, RID)
	FUNC2RC(RID, mesh_surface_get_material, RID, int)

	FUNC2RC(SurfaceData, mesh_get_surface, RID, int)

	FUNC1RC(int, mesh_get_surface_count, RID)

	FUNC2(mesh_set_custom_aabb, RID, const AABB &)
	FUNC1RC(AABB, mesh_get_custom_aabb, RID)

	FUNC2(mesh_set_shadow_mesh, RID, RID)

	FUNC1(mesh_clear, RID)

	/* MULTIMESH API */

	FUNCRIDSPLIT(multimesh)

	FUNC5(multimesh_allocate_data, RID, int, MultimeshTransformFormat, bool, bool)
	FUNC1RC(int, multimesh_get_instance_count, RID)

	FUNC2(multimesh_set_mesh, RID, RID)
	FUNC3(multimesh_instance_set_transform, RID, int, const Transform &)
	FUNC3(multimesh_instance_set_transform_2d, RID, int, const Transform2D &)
	FUNC3(multimesh_instance_set_color, RID, int, const Color &)
	FUNC3(multimesh_instance_set_custom_data, RID, int, const Color &)

	FUNC1RC(RID, multimesh_get_mesh, RID)
	FUNC1RC(AABB, multimesh_get_aabb, RID)

	FUNC2RC(Transform, multimesh_instance_get_transform, RID, int)
	FUNC2RC(Transform2D, multimesh_instance_get_transform_2d, RID, int)
	FUNC2RC(Color, multimesh_instance_get_color, RID, int)
	FUNC2RC(Color, multimesh_instance_get_custom_data, RID, int)

	FUNC2(multimesh_set_buffer, RID, const Vector<float> &)
	FUNC1RC(Vector<float>, multimesh_get_buffer, RID)

	FUNC2(multimesh_set_visible_instances, RID, int)
	FUNC1RC(int, multimesh_get_visible_instances, RID)

	/* IMMEDIATE API */

	FUNCRIDSPLIT(immediate)
	FUNC3(immediate_begin, RID, PrimitiveType, RID)
	FUNC2(immediate_vertex, RID, const Vector3 &)
	FUNC2(immediate_normal, RID, const Vector3 &)
	FUNC2(immediate_tangent, RID, const Plane &)
	FUNC2(immediate_color, RID, const Color &)
	FUNC2(immediate_uv, RID, const Vector2 &)
	FUNC2(immediate_uv2, RID, const Vector2 &)
	FUNC1(immediate_end, RID)
	FUNC1(immediate_clear, RID)
	FUNC2(immediate_set_material, RID, RID)
	FUNC1RC(RID, immediate_get_material, RID)

	/* SKELETON API */

	FUNCRIDSPLIT(skeleton)
	FUNC3(skeleton_allocate_data, RID, int, bool)
	FUNC1RC(int, skeleton_get_bone_count, RID)
	FUNC3(skeleton_bone_set_transform, RID, int, const Transform &)
	FUNC2RC(Transform, skeleton_bone_get_transform, RID, int)
	FUNC3(skeleton_bone_set_transform_2d, RID, int, const Transform2D &)
	FUNC2RC(Transform2D, skeleton_bone_get_transform_2d, RID, int)
	FUNC2(skeleton_set_base_transform_2d, RID, const Transform2D &)

	/* Light API */

	FUNCRIDSPLIT(directional_light)
	FUNCRIDSPLIT(omni_light)
	FUNCRIDSPLIT(spot_light)

	FUNC2(light_set_color, RID, const Color &)
	FUNC3(light_set_param, RID, LightParam, float)
	FUNC2(light_set_shadow, RID, bool)
	FUNC2(light_set_shadow_color, RID, const Color &)
	FUNC2(light_set_projector, RID, RID)
	FUNC2(light_set_negative, RID, bool)
	FUNC2(light_set_cull_mask, RID, uint32_t)
	FUNC2(light_set_reverse_cull_face_mode, RID, bool)
	FUNC2(light_set_bake_mode, RID, LightBakeMode)
	FUNC2(light_set_max_sdfgi_cascade, RID, uint32_t)

	FUNC2(light_omni_set_shadow_mode, RID, LightOmniShadowMode)

	FUNC2(light_directional_set_shadow_mode, RID, LightDirectionalShadowMode)
	FUNC2(light_directional_set_blend_splits, RID, bool)
	FUNC2(light_directional_set_sky_only, RID, bool)
	FUNC2(light_directional_set_shadow_depth_range_mode, RID, LightDirectionalShadowDepthRangeMode)

	/* PROBE API */

	FUNCRIDSPLIT(reflection_probe)

	FUNC2(reflection_probe_set_update_mode, RID, ReflectionProbeUpdateMode)
	FUNC2(reflection_probe_set_intensity, RID, float)
	FUNC2(reflection_probe_set_ambient_color, RID, const Color &)
	FUNC2(reflection_probe_set_ambient_energy, RID, float)
	FUNC2(reflection_probe_set_ambient_mode, RID, ReflectionProbeAmbientMode)
	FUNC2(reflection_probe_set_max_distance, RID, float)
	FUNC2(reflection_probe_set_extents, RID, const Vector3 &)
	FUNC2(reflection_probe_set_origin_offset, RID, const Vector3 &)
	FUNC2(reflection_probe_set_as_interior, RID, bool)
	FUNC2(reflection_probe_set_enable_box_projection, RID, bool)
	FUNC2(reflection_probe_set_enable_shadows, RID, bool)
	FUNC2(reflection_probe_set_cull_mask, RID, uint32_t)
	FUNC2(reflection_probe_set_resolution, RID, int)
	FUNC2(reflection_probe_set_lod_threshold, RID, float)

	/* DECAL API */

	FUNCRIDSPLIT(decal)

	FUNC2(decal_set_extents, RID, const Vector3 &)
	FUNC3(decal_set_texture, RID, DecalTexture, RID)
	FUNC2(decal_set_emission_energy, RID, float)
	FUNC2(decal_set_albedo_mix, RID, float)
	FUNC2(decal_set_modulate, RID, const Color &)
	FUNC2(decal_set_cull_mask, RID, uint32_t)
	FUNC4(decal_set_distance_fade, RID, bool, float, float)
	FUNC3(decal_set_fade, RID, float, float)
	FUNC2(decal_set_normal_fade, RID, float)

	/* BAKED LIGHT API */

	FUNCRIDSPLIT(gi_probe)

	FUNC8(gi_probe_allocate_data, RID, const Transform &, const AABB &, const Vector3i &, const Vector<uint8_t> &, const Vector<uint8_t> &, const Vector<uint8_t> &, const Vector<int> &)

	FUNC1RC(AABB, gi_probe_get_bounds, RID)
	FUNC1RC(Vector3i, gi_probe_get_octree_size, RID)
	FUNC1RC(Vector<uint8_t>, gi_probe_get_octree_cells, RID)
	FUNC1RC(Vector<uint8_t>, gi_probe_get_data_cells, RID)
	FUNC1RC(Vector<uint8_t>, gi_probe_get_distance_field, RID)
	FUNC1RC(Vector<int>, gi_probe_get_level_counts, RID)
	FUNC1RC(Transform, gi_probe_get_to_cell_xform, RID)

	FUNC2(gi_probe_set_dynamic_range, RID, float)
	FUNC1RC(float, gi_probe_get_dynamic_range, RID)

	FUNC2(gi_probe_set_propagation, RID, float)
	FUNC1RC(float, gi_probe_get_propagation, RID)

	FUNC2(gi_probe_set_energy, RID, float)
	FUNC1RC(float, gi_probe_get_energy, RID)

	FUNC2(gi_probe_set_ao, RID, float)
	FUNC1RC(float, gi_probe_get_ao, RID)

	FUNC2(gi_probe_set_ao_size, RID, float)
	FUNC1RC(float, gi_probe_get_ao_size, RID)

	FUNC2(gi_probe_set_bias, RID, float)
	FUNC1RC(float, gi_probe_get_bias, RID)

	FUNC2(gi_probe_set_normal_bias, RID, float)
	FUNC1RC(float, gi_probe_get_normal_bias, RID)

	FUNC2(gi_probe_set_interior, RID, bool)
	FUNC1RC(bool, gi_probe_is_interior, RID)

	FUNC2(gi_probe_set_use_two_bounces, RID, bool)
	FUNC1RC(bool, gi_probe_is_using_two_bounces, RID)

	FUNC2(gi_probe_set_anisotropy_strength, RID, float)
	FUNC1RC(float, gi_probe_get_anisotropy_strength, RID)

	/* LIGHTMAP */

	FUNCRIDSPLIT(lightmap)

	FUNC3(lightmap_set_textures, RID, RID, bool)
	FUNC2(lightmap_set_probe_bounds, RID, const AABB &)
	FUNC2(lightmap_set_probe_interior, RID, bool)
	FUNC5(lightmap_set_probe_capture_data, RID, const PackedVector3Array &, const PackedColorArray &, const PackedInt32Array &, const PackedInt32Array &)
	FUNC1RC(PackedVector3Array, lightmap_get_probe_capture_points, RID)
	FUNC1RC(PackedColorArray, lightmap_get_probe_capture_sh, RID)
	FUNC1RC(PackedInt32Array, lightmap_get_probe_capture_tetrahedra, RID)
	FUNC1RC(PackedInt32Array, lightmap_get_probe_capture_bsp_tree, RID)
	FUNC1(lightmap_set_probe_capture_update_speed, float)

	/* PARTICLES */

	FUNCRIDSPLIT(particles)

	FUNC2(particles_set_mode, RID, ParticlesMode)
	FUNC2(particles_set_emitting, RID, bool)
	FUNC1R(bool, particles_get_emitting, RID)
	FUNC2(particles_set_amount, RID, int)
	FUNC2(particles_set_lifetime, RID, float)
	FUNC2(particles_set_one_shot, RID, bool)
	FUNC2(particles_set_pre_process_time, RID, float)
	FUNC2(particles_set_explosiveness_ratio, RID, float)
	FUNC2(particles_set_randomness_ratio, RID, float)
	FUNC2(particles_set_custom_aabb, RID, const AABB &)
	FUNC2(particles_set_speed_scale, RID, float)
	FUNC2(particles_set_use_local_coordinates, RID, bool)
	FUNC2(particles_set_process_material, RID, RID)
	FUNC2(particles_set_fixed_fps, RID, int)
	FUNC2(particles_set_interpolate, RID, bool)
	FUNC2(particles_set_fractional_delta, RID, bool)
	FUNC1R(bool, particles_is_inactive, RID)
	FUNC3(particles_set_trails, RID, bool, float)
	FUNC2(particles_set_trail_bind_poses, RID, const Vector<Transform> &)

	FUNC1(particles_request_process, RID)
	FUNC1(particles_restart, RID)
	FUNC6(particles_emit, RID, const Transform &, const Vector3 &, const Color &, const Color &, uint32_t)
	FUNC2(particles_set_subemitter, RID, RID)
	FUNC2(particles_set_collision_base_size, RID, float)

	FUNC2(particles_set_transform_align, RID, RS::ParticlesTransformAlign)

	FUNC2(particles_set_draw_order, RID, RS::ParticlesDrawOrder)

	FUNC2(particles_set_draw_passes, RID, int)
	FUNC3(particles_set_draw_pass_mesh, RID, int, RID)

	FUNC1R(AABB, particles_get_current_aabb, RID)
	FUNC2(particles_set_emission_transform, RID, const Transform &)

	/* PARTICLES COLLISION */

	FUNCRIDSPLIT(particles_collision)

	FUNC2(particles_collision_set_collision_type, RID, ParticlesCollisionType)
	FUNC2(particles_collision_set_cull_mask, RID, uint32_t)
	FUNC2(particles_collision_set_sphere_radius, RID, float)
	FUNC2(particles_collision_set_box_extents, RID, const Vector3 &)
	FUNC2(particles_collision_set_attractor_strength, RID, float)
	FUNC2(particles_collision_set_attractor_directionality, RID, float)
	FUNC2(particles_collision_set_attractor_attenuation, RID, float)
	FUNC2(particles_collision_set_field_texture, RID, RID)
	FUNC1(particles_collision_height_field_update, RID)
	FUNC2(particles_collision_set_height_field_resolution, RID, ParticlesCollisionHeightfieldResolution)

#undef server_name
#undef ServerName
//from now on, calls forwarded to this singleton
#define ServerName RendererScene
#define server_name RSG::scene

	/* CAMERA API */

	FUNCRIDSPLIT(camera)
	FUNC4(camera_set_perspective, RID, float, float, float)
	FUNC4(camera_set_orthogonal, RID, float, float, float)
	FUNC5(camera_set_frustum, RID, float, Vector2, float, float)
	FUNC2(camera_set_transform, RID, const Transform &)
	FUNC2(camera_set_cull_mask, RID, uint32_t)
	FUNC2(camera_set_environment, RID, RID)
	FUNC2(camera_set_camera_effects, RID, RID)
	FUNC2(camera_set_use_vertical_aspect, RID, bool)

	/* OCCLUDER */
	FUNCRIDSPLIT(occluder)
	FUNC3(occluder_set_mesh, RID, const PackedVector3Array &, const PackedInt32Array &);

#undef server_name
#undef ServerName
//from now on, calls forwarded to this singleton
#define ServerName RendererViewport
#define server_name RSG::viewport

	/* VIEWPORT TARGET API */

	FUNCRIDSPLIT(viewport)

	FUNC2(viewport_set_use_xr, RID, bool)
	FUNC3(viewport_set_size, RID, int, int)

	FUNC2(viewport_set_active, RID, bool)
	FUNC2(viewport_set_parent_viewport, RID, RID)

	FUNC2(viewport_set_clear_mode, RID, ViewportClearMode)

	FUNC3(viewport_attach_to_screen, RID, const Rect2 &, int)
	FUNC2(viewport_set_render_direct_to_screen, RID, bool)

	FUNC2(viewport_set_update_mode, RID, ViewportUpdateMode)

	FUNC1RC(RID, viewport_get_texture, RID)

	FUNC2(viewport_set_hide_scenario, RID, bool)
	FUNC2(viewport_set_hide_canvas, RID, bool)
	FUNC2(viewport_set_disable_environment, RID, bool)

	FUNC2(viewport_attach_camera, RID, RID)
	FUNC2(viewport_set_scenario, RID, RID)
	FUNC2(viewport_attach_canvas, RID, RID)

	FUNC2(viewport_remove_canvas, RID, RID)
	FUNC3(viewport_set_canvas_transform, RID, RID, const Transform2D &)
	FUNC2(viewport_set_transparent_background, RID, bool)
	FUNC2(viewport_set_snap_2d_transforms_to_pixel, RID, bool)
	FUNC2(viewport_set_snap_2d_vertices_to_pixel, RID, bool)

	FUNC2(viewport_set_default_canvas_item_texture_filter, RID, CanvasItemTextureFilter)
	FUNC2(viewport_set_default_canvas_item_texture_repeat, RID, CanvasItemTextureRepeat)

	FUNC2(viewport_set_global_canvas_transform, RID, const Transform2D &)
	FUNC4(viewport_set_canvas_stacking, RID, RID, int, int)
	FUNC3(viewport_set_shadow_atlas_size, RID, int, bool)
	FUNC3(viewport_set_sdf_oversize_and_scale, RID, ViewportSDFOversize, ViewportSDFScale)
	FUNC3(viewport_set_shadow_atlas_quadrant_subdivision, RID, int, int)
	FUNC2(viewport_set_msaa, RID, ViewportMSAA)
	FUNC2(viewport_set_screen_space_aa, RID, ViewportScreenSpaceAA)
	FUNC2(viewport_set_use_debanding, RID, bool)
	FUNC2(viewport_set_use_occlusion_culling, RID, bool)
	FUNC1(viewport_set_occlusion_rays_per_thread, int)
	FUNC1(viewport_set_occlusion_culling_build_quality, ViewportOcclusionCullingBuildQuality)
	FUNC2(viewport_set_lod_threshold, RID, float)

	FUNC2R(int, viewport_get_render_info, RID, ViewportRenderInfo)
	FUNC2(viewport_set_debug_draw, RID, ViewportDebugDraw)

	FUNC2(viewport_set_measure_render_time, RID, bool)
	FUNC1RC(float, viewport_get_measured_render_time_cpu, RID)
	FUNC1RC(float, viewport_get_measured_render_time_gpu, RID)

	FUNC1(call_set_use_vsync, bool)

	/* ENVIRONMENT API */

#undef server_name
#undef ServerName
//from now on, calls forwarded to this singleton
#define ServerName RendererScene
#define server_name RSG::scene

	FUNC2(directional_shadow_atlas_set_size, int, bool)
	FUNC1(gi_probe_set_quality, GIProbeQuality)

	/* SKY API */

	FUNCRIDSPLIT(sky)
	FUNC2(sky_set_radiance_size, RID, int)
	FUNC2(sky_set_mode, RID, SkyMode)
	FUNC2(sky_set_material, RID, RID)
	FUNC4R(Ref<Image>, sky_bake_panorama, RID, float, bool, const Size2i &)

	FUNCRIDSPLIT(environment)

	FUNC2(environment_set_background, RID, EnvironmentBG)
	FUNC2(environment_set_sky, RID, RID)
	FUNC2(environment_set_sky_custom_fov, RID, float)
	FUNC2(environment_set_sky_orientation, RID, const Basis &)
	FUNC2(environment_set_bg_color, RID, const Color &)
	FUNC2(environment_set_bg_energy, RID, float)
	FUNC2(environment_set_canvas_max_layer, RID, int)
	FUNC7(environment_set_ambient_light, RID, const Color &, EnvironmentAmbientSource, float, float, EnvironmentReflectionSource, const Color &)

// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	FUNC2(environment_set_camera_feed_id, RID, int)
#endif
	FUNC6(environment_set_ssr, RID, bool, int, float, float, float)
	FUNC1(environment_set_ssr_roughness_quality, EnvironmentSSRRoughnessQuality)

	FUNC10(environment_set_ssao, RID, bool, float, float, float, float, float, float, float, float)
	FUNC6(environment_set_ssao_quality, EnvironmentSSAOQuality, bool, float, int, float, float)

	FUNC11(environment_set_glow, RID, bool, Vector<float>, float, float, float, float, EnvironmentGlowBlendMode, float, float, float)
	FUNC1(environment_glow_set_use_bicubic_upscale, bool)
	FUNC1(environment_glow_set_use_high_quality, bool)

	FUNC9(environment_set_tonemap, RID, EnvironmentToneMapper, float, float, bool, float, float, float, float)

	FUNC7(environment_set_adjustment, RID, bool, float, float, float, bool, RID)

	FUNC9(environment_set_fog, RID, bool, const Color &, float, float, float, float, float, float)
	FUNC10(environment_set_volumetric_fog, RID, bool, float, const Color &, float, float, float, float, bool, float)

	FUNC2(environment_set_volumetric_fog_volume_size, int, int)
	FUNC1(environment_set_volumetric_fog_filter_active, bool)

	FUNC11(environment_set_sdfgi, RID, bool, EnvironmentSDFGICascades, float, EnvironmentSDFGIYScale, bool, float, bool, float, float, float)
	FUNC1(environment_set_sdfgi_ray_count, EnvironmentSDFGIRayCount)
	FUNC1(environment_set_sdfgi_frames_to_converge, EnvironmentSDFGIFramesToConverge)
	FUNC1(environment_set_sdfgi_frames_to_update_light, EnvironmentSDFGIFramesToUpdateLight)

	FUNC3R(Ref<Image>, environment_bake_panorama, RID, bool, const Size2i &)

	FUNC3(screen_space_roughness_limiter_set_active, bool, float, float)
	FUNC1(sub_surface_scattering_set_quality, SubSurfaceScatteringQuality)
	FUNC2(sub_surface_scattering_set_scale, float, float)

	/* CAMERA EFFECTS */

	FUNCRIDSPLIT(camera_effects)

	FUNC2(camera_effects_set_dof_blur_quality, DOFBlurQuality, bool)
	FUNC1(camera_effects_set_dof_blur_bokeh_shape, DOFBokehShape)

	FUNC8(camera_effects_set_dof_blur, RID, bool, float, float, bool, float, float, float)
	FUNC3(camera_effects_set_custom_exposure, RID, bool, float)

	FUNC1(shadows_quality_set, ShadowQuality);
	FUNC1(directional_shadow_quality_set, ShadowQuality);

	/* SCENARIO API */

#undef server_name
#undef ServerName

#define ServerName RendererScene
#define server_name RSG::scene

	FUNCRIDSPLIT(scenario)

	FUNC2(scenario_set_debug, RID, ScenarioDebugMode)
	FUNC2(scenario_set_environment, RID, RID)
	FUNC2(scenario_set_camera_effects, RID, RID)
	FUNC2(scenario_set_fallback_environment, RID, RID)

	/* INSTANCING API */
	FUNCRIDSPLIT(instance)

	FUNC2(instance_set_base, RID, RID)
	FUNC2(instance_set_scenario, RID, RID)
	FUNC2(instance_set_layer_mask, RID, uint32_t)
	FUNC2(instance_set_transform, RID, const Transform &)
	FUNC2(instance_attach_object_instance_id, RID, ObjectID)
	FUNC3(instance_set_blend_shape_weight, RID, int, float)
	FUNC3(instance_set_surface_override_material, RID, int, RID)
	FUNC2(instance_set_visible, RID, bool)

	FUNC2(instance_set_custom_aabb, RID, AABB)

	FUNC2(instance_attach_skeleton, RID, RID)
	FUNC2(instance_set_exterior, RID, bool)

	FUNC2(instance_set_extra_visibility_margin, RID, real_t)

	// don't use these in a game!
	FUNC2RC(Vector<ObjectID>, instances_cull_aabb, const AABB &, RID)
	FUNC3RC(Vector<ObjectID>, instances_cull_ray, const Vector3 &, const Vector3 &, RID)
	FUNC2RC(Vector<ObjectID>, instances_cull_convex, const Vector<Plane> &, RID)

	FUNC3(instance_geometry_set_flag, RID, InstanceFlags, bool)
	FUNC2(instance_geometry_set_cast_shadows_setting, RID, ShadowCastingSetting)
	FUNC2(instance_geometry_set_material_override, RID, RID)

	FUNC5(instance_geometry_set_draw_range, RID, float, float, float, float)
	FUNC2(instance_geometry_set_as_instance_lod, RID, RID)
	FUNC4(instance_geometry_set_lightmap, RID, RID, const Rect2 &, int)
	FUNC2(instance_geometry_set_lod_bias, RID, float)

	FUNC3(instance_geometry_set_shader_parameter, RID, const StringName &, const Variant &)
	FUNC2RC(Variant, instance_geometry_get_shader_parameter, RID, const StringName &)
	FUNC2RC(Variant, instance_geometry_get_shader_parameter_default_value, RID, const StringName &)
	FUNC2C(instance_geometry_get_shader_parameter_list, RID, List<PropertyInfo> *)

	FUNC3R(TypedArray<Image>, bake_render_uv2, RID, const Vector<RID> &, const Size2i &)

	FUNC1(gi_set_use_half_resolution, bool)

#undef server_name
#undef ServerName
//from now on, calls forwarded to this singleton
#define ServerName RendererCanvasCull
#define server_name RSG::canvas

	/* CANVAS (2D) */

	FUNCRIDSPLIT(canvas)
	FUNC3(canvas_set_item_mirroring, RID, RID, const Point2 &)
	FUNC2(canvas_set_modulate, RID, const Color &)
	FUNC3(canvas_set_parent, RID, RID, float)
	FUNC1(canvas_set_disable_scale, bool)

	FUNCRIDSPLIT(canvas_texture)
	FUNC3(canvas_texture_set_channel, RID, CanvasTextureChannel, RID)
	FUNC3(canvas_texture_set_shading_parameters, RID, const Color &, float)

	FUNC2(canvas_texture_set_texture_filter, RID, CanvasItemTextureFilter)
	FUNC2(canvas_texture_set_texture_repeat, RID, CanvasItemTextureRepeat)

	FUNCRIDSPLIT(canvas_item)
	FUNC2(canvas_item_set_parent, RID, RID)

	FUNC2(canvas_item_set_default_texture_filter, RID, CanvasItemTextureFilter)
	FUNC2(canvas_item_set_default_texture_repeat, RID, CanvasItemTextureRepeat)

	FUNC2(canvas_item_set_visible, RID, bool)
	FUNC2(canvas_item_set_light_mask, RID, int)

	FUNC2(canvas_item_set_update_when_visible, RID, bool)

	FUNC2(canvas_item_set_transform, RID, const Transform2D &)
	FUNC2(canvas_item_set_clip, RID, bool)
	FUNC2(canvas_item_set_distance_field_mode, RID, bool)
	FUNC3(canvas_item_set_custom_rect, RID, bool, const Rect2 &)
	FUNC2(canvas_item_set_modulate, RID, const Color &)
	FUNC2(canvas_item_set_self_modulate, RID, const Color &)

	FUNC2(canvas_item_set_draw_behind_parent, RID, bool)

	FUNC5(canvas_item_add_line, RID, const Point2 &, const Point2 &, const Color &, float)
	FUNC5(canvas_item_add_polyline, RID, const Vector<Point2> &, const Vector<Color> &, float, bool)
	FUNC4(canvas_item_add_multiline, RID, const Vector<Point2> &, const Vector<Color> &, float)
	FUNC3(canvas_item_add_rect, RID, const Rect2 &, const Color &)
	FUNC4(canvas_item_add_circle, RID, const Point2 &, float, const Color &)
	FUNC6(canvas_item_add_texture_rect, RID, const Rect2 &, RID, bool, const Color &, bool)
	FUNC7(canvas_item_add_texture_rect_region, RID, const Rect2 &, RID, const Rect2 &, const Color &, bool, bool)
	FUNC10(canvas_item_add_nine_patch, RID, const Rect2 &, const Rect2 &, RID, const Vector2 &, const Vector2 &, NinePatchAxisMode, NinePatchAxisMode, bool, const Color &)
	FUNC6(canvas_item_add_primitive, RID, const Vector<Point2> &, const Vector<Color> &, const Vector<Point2> &, RID, float)
	FUNC5(canvas_item_add_polygon, RID, const Vector<Point2> &, const Vector<Color> &, const Vector<Point2> &, RID)
	FUNC9(canvas_item_add_triangle_array, RID, const Vector<int> &, const Vector<Point2> &, const Vector<Color> &, const Vector<Point2> &, const Vector<int> &, const Vector<float> &, RID, int)
	FUNC5(canvas_item_add_mesh, RID, const RID &, const Transform2D &, const Color &, RID)
	FUNC3(canvas_item_add_multimesh, RID, RID, RID)
	FUNC3(canvas_item_add_particles, RID, RID, RID)
	FUNC2(canvas_item_add_set_transform, RID, const Transform2D &)
	FUNC2(canvas_item_add_clip_ignore, RID, bool)
	FUNC2(canvas_item_set_sort_children_by_y, RID, bool)
	FUNC2(canvas_item_set_z_index, RID, int)
	FUNC2(canvas_item_set_z_as_relative_to_parent, RID, bool)
	FUNC3(canvas_item_set_copy_to_backbuffer, RID, bool, const Rect2 &)
	FUNC2(canvas_item_attach_skeleton, RID, RID)

	FUNC1(canvas_item_clear, RID)
	FUNC2(canvas_item_set_draw_index, RID, int)

	FUNC2(canvas_item_set_material, RID, RID)

	FUNC2(canvas_item_set_use_parent_material, RID, bool)

	FUNC6(canvas_item_set_canvas_group_mode, RID, CanvasGroupMode, float, bool, float, bool)

	FUNCRIDSPLIT(canvas_light)

	FUNC2(canvas_light_set_mode, RID, CanvasLightMode)

	FUNC2(canvas_light_attach_to_canvas, RID, RID)
	FUNC2(canvas_light_set_enabled, RID, bool)
	FUNC2(canvas_light_set_texture_scale, RID, float)
	FUNC2(canvas_light_set_transform, RID, const Transform2D &)
	FUNC2(canvas_light_set_texture, RID, RID)
	FUNC2(canvas_light_set_texture_offset, RID, const Vector2 &)
	FUNC2(canvas_light_set_color, RID, const Color &)
	FUNC2(canvas_light_set_height, RID, float)
	FUNC2(canvas_light_set_energy, RID, float)
	FUNC3(canvas_light_set_z_range, RID, int, int)
	FUNC3(canvas_light_set_layer_range, RID, int, int)
	FUNC2(canvas_light_set_item_cull_mask, RID, int)
	FUNC2(canvas_light_set_item_shadow_cull_mask, RID, int)
	FUNC2(canvas_light_set_directional_distance, RID, float)

	FUNC2(canvas_light_set_blend_mode, RID, CanvasLightBlendMode)

	FUNC2(canvas_light_set_shadow_enabled, RID, bool)
	FUNC2(canvas_light_set_shadow_filter, RID, CanvasLightShadowFilter)
	FUNC2(canvas_light_set_shadow_color, RID, const Color &)
	FUNC2(canvas_light_set_shadow_smooth, RID, float)

	FUNCRIDSPLIT(canvas_light_occluder)
	FUNC2(canvas_light_occluder_attach_to_canvas, RID, RID)
	FUNC2(canvas_light_occluder_set_enabled, RID, bool)
	FUNC2(canvas_light_occluder_set_polygon, RID, RID)
	FUNC2(canvas_light_occluder_set_as_sdf_collision, RID, bool)
	FUNC2(canvas_light_occluder_set_transform, RID, const Transform2D &)
	FUNC2(canvas_light_occluder_set_light_mask, RID, int)

	FUNCRIDSPLIT(canvas_occluder_polygon)
	FUNC3(canvas_occluder_polygon_set_shape, RID, const Vector<Vector2> &, bool)

	FUNC2(canvas_occluder_polygon_set_cull_mode, RID, CanvasOccluderPolygonCullMode)

	FUNC1(canvas_set_shadow_texture_size, int)

	/* GLOBAL VARIABLES */

#undef server_name
#undef ServerName
//from now on, calls forwarded to this singleton
#define ServerName RendererStorage
#define server_name RSG::storage

	FUNC3(global_variable_add, const StringName &, GlobalVariableType, const Variant &)
	FUNC1(global_variable_remove, const StringName &)
	FUNC0RC(Vector<StringName>, global_variable_get_list)
	FUNC2(global_variable_set, const StringName &, const Variant &)
	FUNC2(global_variable_set_override, const StringName &, const Variant &)
	FUNC1RC(GlobalVariableType, global_variable_get_type, const StringName &)
	FUNC1RC(Variant, global_variable_get, const StringName &)

	FUNC1(global_variables_load_settings, bool)
	FUNC0(global_variables_clear)

#undef server_name
#undef ServerName
#undef WRITE_ACTION
#undef SYNC_DEBUG

	/* BLACK BARS */

	virtual void black_bars_set_margins(int p_left, int p_top, int p_right, int p_bottom) override;
	virtual void black_bars_set_images(RID p_left, RID p_top, RID p_right, RID p_bottom) override;

	/* FREE */

	virtual void free(RID p_rid) override {
		if (Thread::get_caller_id() == server_thread) {
			command_queue.flush_if_pending();
			_free(p_rid);
		} else {
			command_queue.push(this, &RenderingServerDefault::_free, p_rid);
		}
	}

	/* EVENT QUEUING */

	virtual void request_frame_drawn_callback(Object *p_where, const StringName &p_method, const Variant &p_userdata) override;

	virtual void draw(bool p_swap_buffers, double frame_step) override;
	virtual void sync() override;
	virtual bool has_changed() const override;
	virtual void init() override;
	virtual void finish() override;

	/* STATUS INFORMATION */

	virtual uint64_t get_render_info(RenderInfo p_info) override;
	virtual String get_video_adapter_name() const override;
	virtual String get_video_adapter_vendor() const override;

	virtual void set_frame_profiling_enabled(bool p_enable) override;
	virtual Vector<FrameProfileArea> get_frame_profile() override;
	virtual uint64_t get_frame_profile_frame() override;

	virtual RID get_test_cube() override;

	/* TESTING */

	virtual float get_frame_setup_time_cpu() const override;

	virtual void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter = true) override;
	virtual void set_default_clear_color(const Color &p_color) override;

	virtual bool has_feature(Features p_feature) const override;

	virtual bool has_os_feature(const String &p_feature) const override;
	virtual void set_debug_generate_wireframes(bool p_generate) override;

	virtual bool is_low_end() const override;

	virtual void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) override;

	virtual void set_print_gpu_profile(bool p_enable) override;

	RenderingServerDefault(bool p_create_thread = false);
	~RenderingServerDefault();
};

#endif
