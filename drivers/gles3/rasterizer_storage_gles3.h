/*************************************************************************/
/*  rasterizer_storage_gles3.h                                           */
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

#ifndef RASTERIZER_STORAGE_OPENGL_H
#define RASTERIZER_STORAGE_OPENGL_H

#ifdef GLES3_ENABLED

#include "core/templates/local_vector.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_storage.h"
#include "servers/rendering/shader_compiler.h"
#include "servers/rendering/shader_language.h"
#include "storage/canvas_texture_storage.h"
#include "storage/config.h"
#include "storage/material_storage.h"
#include "storage/render_target_storage.h"
#include "storage/texture_storage.h"

// class RasterizerCanvasGLES3;
// class RasterizerSceneGLES3;

class RasterizerStorageGLES3 : public RendererStorage {
public:
	// RasterizerCanvasGLES3 *canvas;
	// RasterizerSceneGLES3 *scene;

	static GLuint system_fbo;

	GLES3::Config *config;

	struct Resources {
		GLuint white_tex;
		GLuint black_tex;
		GLuint normal_tex;
		GLuint aniso_tex;

		GLuint mipmap_blur_fbo;
		GLuint mipmap_blur_color;

		GLuint radical_inverse_vdc_cache_tex;
		bool use_rgba_2d_shadows;

		GLuint quadie;

		size_t skeleton_transform_buffer_size;
		GLuint skeleton_transform_buffer;
		LocalVector<float> skeleton_transform_cpu_buffer;

	} resources;

	struct Info {
		uint64_t texture_mem = 0;
		uint64_t vertex_mem = 0;

		struct Render {
			uint32_t object_count;
			uint32_t draw_call_count;
			uint32_t material_switch_count;
			uint32_t surface_switch_count;
			uint32_t shader_rebind_count;
			uint32_t vertices_count;
			uint32_t _2d_item_count;
			uint32_t _2d_draw_call_count;

			void reset() {
				object_count = 0;
				draw_call_count = 0;
				material_switch_count = 0;
				surface_switch_count = 0;
				shader_rebind_count = 0;
				vertices_count = 0;
				_2d_item_count = 0;
				_2d_draw_call_count = 0;
			}
		} render, render_final, snap;

		Info() {
			render.reset();
			render_final.reset();
		}

	} info;

	void bind_quad_array() const;

	/////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////API////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////

public:
	/* SKY API */
	// not sure if used in godot 4?
	struct Sky {
		RID self;
		RID panorama;
		GLuint radiance;
		int radiance_size;
	};

	mutable RID_PtrOwner<Sky> sky_owner;

	RID sky_create();
	void sky_set_texture(RID p_sky, RID p_panorama, int p_radiance_size);

	/* Light API */

	RID directional_light_allocate() override;
	void directional_light_initialize(RID p_rid) override;
	RID omni_light_allocate() override;
	void omni_light_initialize(RID p_rid) override;
	RID spot_light_allocate() override;
	void spot_light_initialize(RID p_rid) override;
	RID reflection_probe_allocate() override;
	void reflection_probe_initialize(RID p_rid) override;

	void light_set_color(RID p_light, const Color &p_color) override;
	void light_set_param(RID p_light, RS::LightParam p_param, float p_value) override;
	void light_set_shadow(RID p_light, bool p_enabled) override;
	void light_set_projector(RID p_light, RID p_texture) override;
	void light_set_negative(RID p_light, bool p_enable) override;
	void light_set_cull_mask(RID p_light, uint32_t p_mask) override;
	void light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) override;
	void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) override;
	void light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) override;
	void light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) override;

	void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) override;

	void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) override;
	void light_directional_set_blend_splits(RID p_light, bool p_enable) override;
	bool light_directional_get_blend_splits(RID p_light) const override;
	void light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) override;
	RS::LightDirectionalSkyMode light_directional_get_sky_mode(RID p_light) const override;

	RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) override;
	RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) override;

	bool light_has_shadow(RID p_light) const override;
	bool light_has_projector(RID p_light) const override;

	RS::LightType light_get_type(RID p_light) const override;
	AABB light_get_aabb(RID p_light) const override;
	float light_get_param(RID p_light, RS::LightParam p_param) override;
	Color light_get_color(RID p_light) override;
	RS::LightBakeMode light_get_bake_mode(RID p_light) override;
	uint32_t light_get_max_sdfgi_cascade(RID p_light) override;
	uint64_t light_get_version(RID p_light) const override;

	/* PROBE API */

	void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) override;
	void reflection_probe_set_intensity(RID p_probe, float p_intensity) override;
	void reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) override;
	void reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) override;
	void reflection_probe_set_ambient_energy(RID p_probe, float p_energy) override;
	void reflection_probe_set_max_distance(RID p_probe, float p_distance) override;
	void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) override;
	void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) override;
	void reflection_probe_set_as_interior(RID p_probe, bool p_enable) override;
	void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) override;
	void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) override;
	void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) override;
	void reflection_probe_set_resolution(RID p_probe, int p_resolution) override;
	void reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) override;
	float reflection_probe_get_mesh_lod_threshold(RID p_probe) const override;

	AABB reflection_probe_get_aabb(RID p_probe) const override;
	RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const override;
	uint32_t reflection_probe_get_cull_mask(RID p_probe) const override;
	Vector3 reflection_probe_get_extents(RID p_probe) const override;
	Vector3 reflection_probe_get_origin_offset(RID p_probe) const override;
	float reflection_probe_get_origin_max_distance(RID p_probe) const override;
	bool reflection_probe_renders_shadows(RID p_probe) const override;

	void base_update_dependency(RID p_base, DependencyTracker *p_instance) override;

	/* VOXEL GI API */

	RID voxel_gi_allocate() override;
	void voxel_gi_initialize(RID p_rid) override;
	void voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) override;

	AABB voxel_gi_get_bounds(RID p_voxel_gi) const override;
	Vector3i voxel_gi_get_octree_size(RID p_voxel_gi) const override;
	Vector<uint8_t> voxel_gi_get_octree_cells(RID p_voxel_gi) const override;
	Vector<uint8_t> voxel_gi_get_data_cells(RID p_voxel_gi) const override;
	Vector<uint8_t> voxel_gi_get_distance_field(RID p_voxel_gi) const override;

	Vector<int> voxel_gi_get_level_counts(RID p_voxel_gi) const override;
	Transform3D voxel_gi_get_to_cell_xform(RID p_voxel_gi) const override;

	void voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) override;
	float voxel_gi_get_dynamic_range(RID p_voxel_gi) const override;

	void voxel_gi_set_propagation(RID p_voxel_gi, float p_range) override;
	float voxel_gi_get_propagation(RID p_voxel_gi) const override;

	void voxel_gi_set_energy(RID p_voxel_gi, float p_range) override;
	float voxel_gi_get_energy(RID p_voxel_gi) const override;

	void voxel_gi_set_bias(RID p_voxel_gi, float p_range) override;
	float voxel_gi_get_bias(RID p_voxel_gi) const override;

	void voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) override;
	float voxel_gi_get_normal_bias(RID p_voxel_gi) const override;

	void voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) override;
	bool voxel_gi_is_interior(RID p_voxel_gi) const override;

	void voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) override;
	bool voxel_gi_is_using_two_bounces(RID p_voxel_gi) const override;

	void voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) override;
	float voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const override;

	uint32_t voxel_gi_get_version(RID p_voxel_gi) override;

	/* LIGHTMAP CAPTURE */
	RID lightmap_allocate() override;
	void lightmap_initialize(RID p_rid) override;
	void lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) override;
	void lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) override;
	void lightmap_set_probe_interior(RID p_lightmap, bool p_interior) override;
	void lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) override;
	PackedVector3Array lightmap_get_probe_capture_points(RID p_lightmap) const override;
	PackedColorArray lightmap_get_probe_capture_sh(RID p_lightmap) const override;
	PackedInt32Array lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const override;
	PackedInt32Array lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const override;
	AABB lightmap_get_aabb(RID p_lightmap) const override;
	void lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) override;
	bool lightmap_is_interior(RID p_lightmap) const override;
	void lightmap_set_probe_capture_update_speed(float p_speed) override;
	float lightmap_get_probe_capture_update_speed() const override;

	/* OCCLUDER */

	void occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices);

	/* PARTICLES */

	RID particles_allocate() override;
	void particles_initialize(RID p_rid) override;
	void particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) override;
	void particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) override;
	void particles_set_emitting(RID p_particles, bool p_emitting) override;
	void particles_set_amount(RID p_particles, int p_amount) override;
	void particles_set_lifetime(RID p_particles, double p_lifetime) override;
	void particles_set_one_shot(RID p_particles, bool p_one_shot) override;
	void particles_set_pre_process_time(RID p_particles, double p_time) override;
	void particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) override;
	void particles_set_randomness_ratio(RID p_particles, real_t p_ratio) override;
	void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) override;
	void particles_set_speed_scale(RID p_particles, double p_scale) override;
	void particles_set_use_local_coordinates(RID p_particles, bool p_enable) override;
	void particles_set_process_material(RID p_particles, RID p_material) override;
	RID particles_get_process_material(RID p_particles) const override;
	void particles_set_fixed_fps(RID p_particles, int p_fps) override;
	void particles_set_interpolate(RID p_particles, bool p_enable) override;
	void particles_set_fractional_delta(RID p_particles, bool p_enable) override;
	void particles_set_subemitter(RID p_particles, RID p_subemitter_particles) override;
	void particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) override;
	void particles_set_collision_base_size(RID p_particles, real_t p_size) override;

	void particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) override;

	void particles_set_trails(RID p_particles, bool p_enable, double p_length) override;
	void particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) override;

	void particles_restart(RID p_particles) override;

	void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) override;

	void particles_set_draw_passes(RID p_particles, int p_count) override;
	void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) override;

	void particles_request_process(RID p_particles) override;
	AABB particles_get_current_aabb(RID p_particles) override;
	AABB particles_get_aabb(RID p_particles) const override;

	void particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) override;

	bool particles_get_emitting(RID p_particles) override;
	int particles_get_draw_passes(RID p_particles) const override;
	RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const override;

	void particles_add_collision(RID p_particles, RID p_instance) override;
	void particles_remove_collision(RID p_particles, RID p_instance) override;

	void particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) override;

	void update_particles() override;
	bool particles_is_inactive(RID p_particles) const override;

	/* PARTICLES COLLISION */

	RID particles_collision_allocate() override;
	void particles_collision_initialize(RID p_rid) override;
	void particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) override;
	void particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) override;
	void particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) override;
	void particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) override;
	void particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) override;
	void particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) override;
	void particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) override;
	void particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) override;
	void particles_collision_height_field_update(RID p_particles_collision) override;
	void particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) override;
	AABB particles_collision_get_aabb(RID p_particles_collision) const override;
	bool particles_collision_is_heightfield(RID p_particles_collision) const override;
	RID particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const override;

	RID particles_collision_instance_create(RID p_collision) override;
	void particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) override;
	void particles_collision_instance_set_active(RID p_collision_instance, bool p_active) override;

	/* FOG VOLUMES */

	RID fog_volume_allocate() override;
	void fog_volume_initialize(RID p_rid) override;

	void fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) override;
	void fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) override;
	void fog_volume_set_material(RID p_fog_volume, RID p_material) override;
	AABB fog_volume_get_aabb(RID p_fog_volume) const override;
	RS::FogVolumeShape fog_volume_get_shape(RID p_fog_volume) const override;

	/* VISIBILITY NOTIFIER */
	RID visibility_notifier_allocate() override;
	void visibility_notifier_initialize(RID p_notifier) override;
	void visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) override;
	void visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) override;

	AABB visibility_notifier_get_aabb(RID p_notifier) const override;
	void visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) override;

	// RENDER TARGET

	mutable RID_PtrOwner<GLES3::RenderTarget> render_target_owner;

	void _render_target_clear(GLES3::RenderTarget *rt);
	void _render_target_allocate(GLES3::RenderTarget *rt);
	void _set_current_render_target(RID p_render_target);

	RID render_target_create() override;
	void render_target_set_position(RID p_render_target, int p_x, int p_y) override;
	void render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) override;
	Size2i render_target_get_size(RID p_render_target);
	RID render_target_get_texture(RID p_render_target) override;
	void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) override;

	void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) override;
	bool render_target_was_used(RID p_render_target) override;
	void render_target_clear_used(RID p_render_target);
	void render_target_set_msaa(RID p_render_target, RS::ViewportMSAA p_msaa);
	void render_target_set_use_fxaa(RID p_render_target, bool p_fxaa);
	void render_target_set_use_debanding(RID p_render_target, bool p_debanding);

	// new
	void render_target_set_as_unused(RID p_render_target) override {
		render_target_clear_used(p_render_target);
	}

	void render_target_request_clear(RID p_render_target, const Color &p_clear_color) override;
	bool render_target_is_clear_requested(RID p_render_target) override;
	Color render_target_get_clear_request_color(RID p_render_target) override;
	void render_target_disable_clear_request(RID p_render_target) override;
	void render_target_do_clear_request(RID p_render_target) override;

	void render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) override;
	Rect2i render_target_get_sdf_rect(RID p_render_target) const override;
	void render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) override;

	// access from canvas
	//	GLES3::RenderTarget * render_target_get(RID p_render_target);

	/* CANVAS SHADOW */

	struct CanvasLightShadow {
		RID self;
		int size;
		int height;
		GLuint fbo;
		GLuint depth;
		GLuint distance; //for older devices
	};

	RID_PtrOwner<CanvasLightShadow> canvas_light_shadow_owner;

	RID canvas_light_shadow_buffer_create(int p_width);

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

	RS::InstanceType get_base_type(RID p_rid) const override;

	bool free(RID p_rid) override;

	struct Frame {
		GLES3::RenderTarget *current_rt;

		// these 2 may have been superseded by the equivalents in the render target.
		// these may be able to be removed.
		bool clear_request;
		Color clear_request_color;

		float time;
		float delta;
		uint64_t count;

		Frame() {
			//			current_rt = nullptr;
			//			clear_request = false;
		}
	} frame;

	void initialize();
	void finalize();

	void _copy_screen();

	void update_memory_info() override;
	uint64_t get_rendering_info(RS::RenderingInfo p_info) override;

	bool has_os_feature(const String &p_feature) const override;

	void update_dirty_resources() override;

	void set_debug_generate_wireframes(bool p_generate) override;

	//	void render_info_begin_capture() override;
	//	void render_info_end_capture() override;
	//	int get_captured_render_info(RS::RenderInfo p_info) override;

	//	int get_render_info(RS::RenderInfo p_info) override;
	String get_video_adapter_name() const override;
	String get_video_adapter_vendor() const override;
	RenderingDevice::DeviceType get_video_adapter_type() const override;

	void capture_timestamps_begin() override {}
	void capture_timestamp(const String &p_name) override {}
	uint32_t get_captured_timestamps_count() const override {
		return 0;
	}
	uint64_t get_captured_timestamps_frame() const override {
		return 0;
	}
	uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const override {
		return 0;
	}
	uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const override {
		return 0;
	}
	String get_captured_timestamp_name(uint32_t p_index) const override {
		return String();
	}

	// make access easier to these
	struct Dimensions {
		// render target
		int rt_width;
		int rt_height;

		// window
		int win_width;
		int win_height;
		Dimensions() {
			rt_width = 0;
			rt_height = 0;
			win_width = 0;
			win_height = 0;
		}
	} _dims;

	void buffer_orphan_and_upload(unsigned int p_buffer_size, unsigned int p_offset, unsigned int p_data_size, const void *p_data, GLenum p_target = GL_ARRAY_BUFFER, GLenum p_usage = GL_DYNAMIC_DRAW, bool p_optional_orphan = false) const;
	bool safe_buffer_sub_data(unsigned int p_total_buffer_size, GLenum p_target, unsigned int p_offset, unsigned int p_data_size, const void *p_data, unsigned int &r_offset_after) const;

	void bind_framebuffer(GLuint framebuffer) {
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	}

	void bind_framebuffer_system() {
		glBindFramebuffer(GL_FRAMEBUFFER, RasterizerStorageGLES3::system_fbo);
	}

	RasterizerStorageGLES3();
	~RasterizerStorageGLES3();
};

inline bool RasterizerStorageGLES3::safe_buffer_sub_data(unsigned int p_total_buffer_size, GLenum p_target, unsigned int p_offset, unsigned int p_data_size, const void *p_data, unsigned int &r_offset_after) const {
	r_offset_after = p_offset + p_data_size;
#ifdef DEBUG_ENABLED
	// we are trying to write across the edge of the buffer
	if (r_offset_after > p_total_buffer_size) {
		return false;
	}
#endif
	glBufferSubData(p_target, p_offset, p_data_size, p_data);
	return true;
}

// standardize the orphan / upload in one place so it can be changed per platform as necessary, and avoid future
// bugs causing pipeline stalls
inline void RasterizerStorageGLES3::buffer_orphan_and_upload(unsigned int p_buffer_size, unsigned int p_offset, unsigned int p_data_size, const void *p_data, GLenum p_target, GLenum p_usage, bool p_optional_orphan) const {
	// Orphan the buffer to avoid CPU/GPU sync points caused by glBufferSubData
	// Was previously #ifndef GLES_OVER_GL however this causes stalls on desktop mac also (and possibly other)
	if (!p_optional_orphan || (config->should_orphan)) {
		glBufferData(p_target, p_buffer_size, nullptr, p_usage);
#ifdef RASTERIZER_EXTRA_CHECKS
		// fill with garbage off the end of the array
		if (p_buffer_size) {
			unsigned int start = p_offset + p_data_size;
			unsigned int end = start + 1024;
			if (end < p_buffer_size) {
				uint8_t *garbage = (uint8_t *)alloca(1024);
				for (int n = 0; n < 1024; n++) {
					garbage[n] = Math::random(0, 255);
				}
				glBufferSubData(p_target, start, 1024, garbage);
			}
		}
#endif
	}
	glBufferSubData(p_target, p_offset, p_data_size, p_data);
}

#endif // GLES3_ENABLED

#endif // RASTERIZER_STORAGE_OPENGL_H
