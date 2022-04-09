/*************************************************************************/
/*  rasterizer_storage_gles3.cpp                                         */
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

#include "rasterizer_storage_gles3.h"

#ifdef GLES3_ENABLED

#include "core/config/project_settings.h"
#include "core/math/transform_3d.h"
// #include "rasterizer_canvas_gles3.h"
#include "rasterizer_scene_gles3.h"
#include "servers/rendering/shader_language.h"

GLuint RasterizerStorageGLES3::system_fbo = 0;

void RasterizerStorageGLES3::bind_quad_array() const {
	//glBindBuffer(GL_ARRAY_BUFFER, resources.quadie);
	//glVertexAttribPointer(RS::ARRAY_VERTEX, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, 0);
	//glVertexAttribPointer(RS::ARRAY_TEX_UV, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, CAST_INT_TO_UCHAR_PTR(8));

	//glEnableVertexAttribArray(RS::ARRAY_VERTEX);
	//glEnableVertexAttribArray(RS::ARRAY_TEX_UV);
}

RID RasterizerStorageGLES3::sky_create() {
	Sky *sky = memnew(Sky);
	sky->radiance = 0;
	return sky_owner.make_rid(sky);
}

void RasterizerStorageGLES3::sky_set_texture(RID p_sky, RID p_panorama, int p_radiance_size) {
}

/* Light API */

RID RasterizerStorageGLES3::directional_light_allocate() {
	return RID();
}

void RasterizerStorageGLES3::directional_light_initialize(RID p_rid) {
}

RID RasterizerStorageGLES3::omni_light_allocate() {
	return RID();
}

void RasterizerStorageGLES3::omni_light_initialize(RID p_rid) {
}

RID RasterizerStorageGLES3::spot_light_allocate() {
	return RID();
}

void RasterizerStorageGLES3::spot_light_initialize(RID p_rid) {
}

RID RasterizerStorageGLES3::reflection_probe_allocate() {
	return RID();
}

void RasterizerStorageGLES3::reflection_probe_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::light_set_color(RID p_light, const Color &p_color) {
}

void RasterizerStorageGLES3::light_set_param(RID p_light, RS::LightParam p_param, float p_value) {
}

void RasterizerStorageGLES3::light_set_shadow(RID p_light, bool p_enabled) {
}

void RasterizerStorageGLES3::light_set_projector(RID p_light, RID p_texture) {
}

void RasterizerStorageGLES3::light_set_negative(RID p_light, bool p_enable) {
}

void RasterizerStorageGLES3::light_set_cull_mask(RID p_light, uint32_t p_mask) {
}

void RasterizerStorageGLES3::light_set_distance_fade(RID p_light, bool p_enabled, float p_begin, float p_shadow, float p_length) {
}

void RasterizerStorageGLES3::light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) {
}

void RasterizerStorageGLES3::light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) {
}

void RasterizerStorageGLES3::light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) {
}

void RasterizerStorageGLES3::light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) {
}

void RasterizerStorageGLES3::light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) {
}

void RasterizerStorageGLES3::light_directional_set_blend_splits(RID p_light, bool p_enable) {
}

bool RasterizerStorageGLES3::light_directional_get_blend_splits(RID p_light) const {
	return false;
}

void RasterizerStorageGLES3::light_directional_set_sky_mode(RID p_light, RS::LightDirectionalSkyMode p_mode) {
}

RS::LightDirectionalSkyMode RasterizerStorageGLES3::light_directional_get_sky_mode(RID p_light) const {
	return RS::LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY;
}

RS::LightDirectionalShadowMode RasterizerStorageGLES3::light_directional_get_shadow_mode(RID p_light) {
	return RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL;
}

RS::LightOmniShadowMode RasterizerStorageGLES3::light_omni_get_shadow_mode(RID p_light) {
	return RS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID;
}

bool RasterizerStorageGLES3::light_has_shadow(RID p_light) const {
	return false;
}

bool RasterizerStorageGLES3::light_has_projector(RID p_light) const {
	return false;
}

RS::LightType RasterizerStorageGLES3::light_get_type(RID p_light) const {
	return RS::LIGHT_OMNI;
}

AABB RasterizerStorageGLES3::light_get_aabb(RID p_light) const {
	return AABB();
}

float RasterizerStorageGLES3::light_get_param(RID p_light, RS::LightParam p_param) {
	return 0.0;
}

Color RasterizerStorageGLES3::light_get_color(RID p_light) {
	return Color();
}

RS::LightBakeMode RasterizerStorageGLES3::light_get_bake_mode(RID p_light) {
	return RS::LIGHT_BAKE_DISABLED;
}

uint32_t RasterizerStorageGLES3::light_get_max_sdfgi_cascade(RID p_light) {
	return 0;
}

uint64_t RasterizerStorageGLES3::light_get_version(RID p_light) const {
	return 0;
}

/* PROBE API */

void RasterizerStorageGLES3::reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) {
}

void RasterizerStorageGLES3::reflection_probe_set_intensity(RID p_probe, float p_intensity) {
}

void RasterizerStorageGLES3::reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) {
}

void RasterizerStorageGLES3::reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) {
}

void RasterizerStorageGLES3::reflection_probe_set_ambient_energy(RID p_probe, float p_energy) {
}

void RasterizerStorageGLES3::reflection_probe_set_max_distance(RID p_probe, float p_distance) {
}

void RasterizerStorageGLES3::reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) {
}

void RasterizerStorageGLES3::reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) {
}

void RasterizerStorageGLES3::reflection_probe_set_as_interior(RID p_probe, bool p_enable) {
}

void RasterizerStorageGLES3::reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) {
}

void RasterizerStorageGLES3::reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) {
}

void RasterizerStorageGLES3::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) {
}

void RasterizerStorageGLES3::reflection_probe_set_resolution(RID p_probe, int p_resolution) {
}

AABB RasterizerStorageGLES3::reflection_probe_get_aabb(RID p_probe) const {
	return AABB();
}

RS::ReflectionProbeUpdateMode RasterizerStorageGLES3::reflection_probe_get_update_mode(RID p_probe) const {
	return RenderingServer::REFLECTION_PROBE_UPDATE_ONCE;
}

uint32_t RasterizerStorageGLES3::reflection_probe_get_cull_mask(RID p_probe) const {
	return 0;
}

Vector3 RasterizerStorageGLES3::reflection_probe_get_extents(RID p_probe) const {
	return Vector3();
}

Vector3 RasterizerStorageGLES3::reflection_probe_get_origin_offset(RID p_probe) const {
	return Vector3();
}

float RasterizerStorageGLES3::reflection_probe_get_origin_max_distance(RID p_probe) const {
	return 0.0;
}

bool RasterizerStorageGLES3::reflection_probe_renders_shadows(RID p_probe) const {
	return false;
}

void RasterizerStorageGLES3::base_update_dependency(RID p_base, DependencyTracker *p_instance) {
}

void RasterizerStorageGLES3::reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) {
}

float RasterizerStorageGLES3::reflection_probe_get_mesh_lod_threshold(RID p_probe) const {
	return 0.0;
}

/* VOXEL GI API */

RID RasterizerStorageGLES3::voxel_gi_allocate() {
	return RID();
}

void RasterizerStorageGLES3::voxel_gi_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) {
}

AABB RasterizerStorageGLES3::voxel_gi_get_bounds(RID p_voxel_gi) const {
	return AABB();
}

Vector3i RasterizerStorageGLES3::voxel_gi_get_octree_size(RID p_voxel_gi) const {
	return Vector3i();
}

Vector<uint8_t> RasterizerStorageGLES3::voxel_gi_get_octree_cells(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<uint8_t> RasterizerStorageGLES3::voxel_gi_get_data_cells(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<uint8_t> RasterizerStorageGLES3::voxel_gi_get_distance_field(RID p_voxel_gi) const {
	return Vector<uint8_t>();
}

Vector<int> RasterizerStorageGLES3::voxel_gi_get_level_counts(RID p_voxel_gi) const {
	return Vector<int>();
}

Transform3D RasterizerStorageGLES3::voxel_gi_get_to_cell_xform(RID p_voxel_gi) const {
	return Transform3D();
}

void RasterizerStorageGLES3::voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_dynamic_range(RID p_voxel_gi) const {
	return 0;
}

void RasterizerStorageGLES3::voxel_gi_set_propagation(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_propagation(RID p_voxel_gi) const {
	return 0;
}

void RasterizerStorageGLES3::voxel_gi_set_energy(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_energy(RID p_voxel_gi) const {
	return 0.0;
}

void RasterizerStorageGLES3::voxel_gi_set_bias(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_bias(RID p_voxel_gi) const {
	return 0.0;
}

void RasterizerStorageGLES3::voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) {
}

float RasterizerStorageGLES3::voxel_gi_get_normal_bias(RID p_voxel_gi) const {
	return 0.0;
}

void RasterizerStorageGLES3::voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) {
}

bool RasterizerStorageGLES3::voxel_gi_is_interior(RID p_voxel_gi) const {
	return false;
}

void RasterizerStorageGLES3::voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) {
}

bool RasterizerStorageGLES3::voxel_gi_is_using_two_bounces(RID p_voxel_gi) const {
	return false;
}

void RasterizerStorageGLES3::voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) {
}

float RasterizerStorageGLES3::voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const {
	return 0;
}

uint32_t RasterizerStorageGLES3::voxel_gi_get_version(RID p_voxel_gi) {
	return 0;
}

/* LIGHTMAP CAPTURE */
RID RasterizerStorageGLES3::lightmap_allocate() {
	return RID();
}

void RasterizerStorageGLES3::lightmap_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) {
}

void RasterizerStorageGLES3::lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) {
}

void RasterizerStorageGLES3::lightmap_set_probe_interior(RID p_lightmap, bool p_interior) {
}

void RasterizerStorageGLES3::lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
}

PackedVector3Array RasterizerStorageGLES3::lightmap_get_probe_capture_points(RID p_lightmap) const {
	return PackedVector3Array();
}

PackedColorArray RasterizerStorageGLES3::lightmap_get_probe_capture_sh(RID p_lightmap) const {
	return PackedColorArray();
}

PackedInt32Array RasterizerStorageGLES3::lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const {
	return PackedInt32Array();
}

PackedInt32Array RasterizerStorageGLES3::lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const {
	return PackedInt32Array();
}

AABB RasterizerStorageGLES3::lightmap_get_aabb(RID p_lightmap) const {
	return AABB();
}

void RasterizerStorageGLES3::lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) {
}

bool RasterizerStorageGLES3::lightmap_is_interior(RID p_lightmap) const {
	return false;
}

void RasterizerStorageGLES3::lightmap_set_probe_capture_update_speed(float p_speed) {
}

float RasterizerStorageGLES3::lightmap_get_probe_capture_update_speed() const {
	return 0;
}

/* OCCLUDER */

void RasterizerStorageGLES3::occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) {
}

/* PARTICLES */

RID RasterizerStorageGLES3::particles_allocate() {
	return RID();
}

void RasterizerStorageGLES3::particles_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) {
}

void RasterizerStorageGLES3::particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) {
}

void RasterizerStorageGLES3::particles_set_emitting(RID p_particles, bool p_emitting) {
}

void RasterizerStorageGLES3::particles_set_amount(RID p_particles, int p_amount) {
}

void RasterizerStorageGLES3::particles_set_lifetime(RID p_particles, double p_lifetime) {
}

void RasterizerStorageGLES3::particles_set_one_shot(RID p_particles, bool p_one_shot) {
}

void RasterizerStorageGLES3::particles_set_pre_process_time(RID p_particles, double p_time) {
}

void RasterizerStorageGLES3::particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) {
}

void RasterizerStorageGLES3::particles_set_randomness_ratio(RID p_particles, real_t p_ratio) {
}

void RasterizerStorageGLES3::particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) {
}

void RasterizerStorageGLES3::particles_set_speed_scale(RID p_particles, double p_scale) {
}

void RasterizerStorageGLES3::particles_set_use_local_coordinates(RID p_particles, bool p_enable) {
}

void RasterizerStorageGLES3::particles_set_process_material(RID p_particles, RID p_material) {
}
RID RasterizerStorageGLES3::particles_get_process_material(RID p_particles) const {
	return RID();
}

void RasterizerStorageGLES3::particles_set_fixed_fps(RID p_particles, int p_fps) {
}

void RasterizerStorageGLES3::particles_set_interpolate(RID p_particles, bool p_enable) {
}

void RasterizerStorageGLES3::particles_set_fractional_delta(RID p_particles, bool p_enable) {
}

void RasterizerStorageGLES3::particles_set_subemitter(RID p_particles, RID p_subemitter_particles) {
}

void RasterizerStorageGLES3::particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) {
}

void RasterizerStorageGLES3::particles_set_collision_base_size(RID p_particles, real_t p_size) {
}

void RasterizerStorageGLES3::particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) {
}

void RasterizerStorageGLES3::particles_set_trails(RID p_particles, bool p_enable, double p_length) {
}

void RasterizerStorageGLES3::particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) {
}

void RasterizerStorageGLES3::particles_restart(RID p_particles) {
}

void RasterizerStorageGLES3::particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) {
}

void RasterizerStorageGLES3::particles_set_draw_passes(RID p_particles, int p_count) {
}

void RasterizerStorageGLES3::particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) {
}

void RasterizerStorageGLES3::particles_request_process(RID p_particles) {
}

AABB RasterizerStorageGLES3::particles_get_current_aabb(RID p_particles) {
	return AABB();
}

AABB RasterizerStorageGLES3::particles_get_aabb(RID p_particles) const {
	return AABB();
}

void RasterizerStorageGLES3::particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) {
}

bool RasterizerStorageGLES3::particles_get_emitting(RID p_particles) {
	return false;
}

int RasterizerStorageGLES3::particles_get_draw_passes(RID p_particles) const {
	return 0;
}

RID RasterizerStorageGLES3::particles_get_draw_pass_mesh(RID p_particles, int p_pass) const {
	return RID();
}

void RasterizerStorageGLES3::particles_add_collision(RID p_particles, RID p_instance) {
}

void RasterizerStorageGLES3::particles_remove_collision(RID p_particles, RID p_instance) {
}

void RasterizerStorageGLES3::particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) {
}

void RasterizerStorageGLES3::update_particles() {
}

bool RasterizerStorageGLES3::particles_is_inactive(RID p_particles) const {
	return false;
}

/* PARTICLES COLLISION */

RID RasterizerStorageGLES3::particles_collision_allocate() {
	return RID();
}

void RasterizerStorageGLES3::particles_collision_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) {
}

void RasterizerStorageGLES3::particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) {
}

void RasterizerStorageGLES3::particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) {
}

void RasterizerStorageGLES3::particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) {
}

void RasterizerStorageGLES3::particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) {
}

void RasterizerStorageGLES3::particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) {
}

void RasterizerStorageGLES3::particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) {
}

void RasterizerStorageGLES3::particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) {
}

void RasterizerStorageGLES3::particles_collision_height_field_update(RID p_particles_collision) {
}

void RasterizerStorageGLES3::particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) {
}

AABB RasterizerStorageGLES3::particles_collision_get_aabb(RID p_particles_collision) const {
	return AABB();
}

bool RasterizerStorageGLES3::particles_collision_is_heightfield(RID p_particles_collision) const {
	return false;
}

RID RasterizerStorageGLES3::particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const {
	return RID();
}

RID RasterizerStorageGLES3::particles_collision_instance_create(RID p_collision) {
	return RID();
}

void RasterizerStorageGLES3::particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) {
}

void RasterizerStorageGLES3::particles_collision_instance_set_active(RID p_collision_instance, bool p_active) {
}

RID RasterizerStorageGLES3::fog_volume_allocate() {
	return RID();
}

void RasterizerStorageGLES3::fog_volume_initialize(RID p_rid) {
}

void RasterizerStorageGLES3::fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) {
}

void RasterizerStorageGLES3::fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) {
}

void RasterizerStorageGLES3::fog_volume_set_material(RID p_fog_volume, RID p_material) {
}

AABB RasterizerStorageGLES3::fog_volume_get_aabb(RID p_fog_volume) const {
	return AABB();
}

RS::FogVolumeShape RasterizerStorageGLES3::fog_volume_get_shape(RID p_fog_volume) const {
	return RS::FOG_VOLUME_SHAPE_BOX;
}

/* VISIBILITY NOTIFIER */
RID RasterizerStorageGLES3::visibility_notifier_allocate() {
	return RID();
}

void RasterizerStorageGLES3::visibility_notifier_initialize(RID p_notifier) {
}

void RasterizerStorageGLES3::visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) {
}

void RasterizerStorageGLES3::visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) {
}

AABB RasterizerStorageGLES3::visibility_notifier_get_aabb(RID p_notifier) const {
	return AABB();
}

void RasterizerStorageGLES3::visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) {
}

/* RENDER TARGET */

void RasterizerStorageGLES3::_set_current_render_target(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);

	if (rt) {
		if (rt->allocate_is_dirty) {
			rt->allocate_is_dirty = false;
			_render_target_allocate(rt);
		}

		frame.current_rt = rt;
		ERR_FAIL_COND(!rt);
		frame.clear_request = false;

		glViewport(0, 0, rt->width, rt->height);

		_dims.rt_width = rt->width;
		_dims.rt_height = rt->height;
		_dims.win_width = rt->width;
		_dims.win_height = rt->height;

	} else {
		frame.current_rt = nullptr;
		frame.clear_request = false;
		bind_framebuffer_system();
	}
}

void RasterizerStorageGLES3::_render_target_allocate(GLES3::RenderTarget *rt) {
	// do not allocate a render target with no size
	if (rt->width <= 0 || rt->height <= 0) {
		return;
	}

	// do not allocate a render target that is attached to the screen
	if (rt->flags[RENDER_TARGET_DIRECT_TO_SCREEN]) {
		rt->fbo = RasterizerStorageGLES3::system_fbo;
		return;
	}

	GLuint color_internal_format;
	GLuint color_format;
	GLuint color_type = GL_UNSIGNED_BYTE;
	Image::Format image_format;

	if (rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
#ifdef GLES_OVER_GL
		color_internal_format = GL_RGBA8;
#else
		color_internal_format = GL_RGBA;
#endif
		color_format = GL_RGBA;
		image_format = Image::FORMAT_RGBA8;
	} else {
#ifdef GLES_OVER_GL
		color_internal_format = GL_RGB8;
#else
		color_internal_format = GL_RGB;
#endif
		color_format = GL_RGB;
		image_format = Image::FORMAT_RGB8;
	}

	rt->used_dof_blur_near = false;
	rt->mip_maps_allocated = false;

	{
		/* Front FBO */

		GLES3::Texture *texture = GLES3::TextureStorage::get_singleton()->get_texture(rt->texture);
		ERR_FAIL_COND(!texture);

		// framebuffer
		glGenFramebuffers(1, &rt->fbo);
		bind_framebuffer(rt->fbo);

		// color
		glGenTextures(1, &rt->color);
		glBindTexture(GL_TEXTURE_2D, rt->color);

		glTexImage2D(GL_TEXTURE_2D, 0, color_internal_format, rt->width, rt->height, 0, color_format, color_type, nullptr);

		if (texture->flags & GLES3::TEXTURE_FLAG_FILTER) {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		} else {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->color, 0);

		// depth

		if (config->support_depth_texture) {
			glGenTextures(1, &rt->depth);
			glBindTexture(GL_TEXTURE_2D, rt->depth);
			glTexImage2D(GL_TEXTURE_2D, 0, config->depth_internalformat, rt->width, rt->height, 0, GL_DEPTH_COMPONENT, config->depth_type, nullptr);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rt->depth, 0);
		} else {
			glGenRenderbuffers(1, &rt->depth);
			glBindRenderbuffer(GL_RENDERBUFFER, rt->depth);

			glRenderbufferStorage(GL_RENDERBUFFER, config->depth_buffer_internalformat, rt->width, rt->height);

			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);
		}

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			glDeleteFramebuffers(1, &rt->fbo);
			if (config->support_depth_texture) {
				glDeleteTextures(1, &rt->depth);
			} else {
				glDeleteRenderbuffers(1, &rt->depth);
			}

			glDeleteTextures(1, &rt->color);
			rt->fbo = 0;
			rt->width = 0;
			rt->height = 0;
			rt->color = 0;
			rt->depth = 0;
			texture->tex_id = 0;
			texture->active = false;
			WARN_PRINT("Could not create framebuffer!!");
			return;
		}

		texture->format = image_format;
		texture->gl_format_cache = color_format;
		texture->gl_type_cache = GL_UNSIGNED_BYTE;
		texture->gl_internal_format_cache = color_internal_format;
		texture->tex_id = rt->color;
		texture->width = rt->width;
		texture->alloc_width = rt->width;
		texture->height = rt->height;
		texture->alloc_height = rt->height;
		texture->active = true;

		GLES3::TextureStorage::get_singleton()->texture_set_flags(rt->texture, texture->flags);
	}

	/* BACK FBO */
	/* For MSAA */

#ifndef JAVASCRIPT_ENABLED
	if (rt->msaa >= RS::VIEWPORT_MSAA_2X && rt->msaa <= RS::VIEWPORT_MSAA_8X) {
		rt->multisample_active = true;

		static const int msaa_value[] = { 0, 2, 4, 8, 16 };
		int msaa = msaa_value[rt->msaa];

		int max_samples = 0;
		glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
		if (msaa > max_samples) {
			WARN_PRINT("MSAA must be <= GL_MAX_SAMPLES, falling-back to GL_MAX_SAMPLES = " + itos(max_samples));
			msaa = max_samples;
		}

		//regular fbo
		glGenFramebuffers(1, &rt->multisample_fbo);
		bind_framebuffer(rt->multisample_fbo);

		glGenRenderbuffers(1, &rt->multisample_depth);
		glBindRenderbuffer(GL_RENDERBUFFER, rt->multisample_depth);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, config->depth_buffer_internalformat, rt->width, rt->height);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->multisample_depth);

		glGenRenderbuffers(1, &rt->multisample_color);
		glBindRenderbuffer(GL_RENDERBUFFER, rt->multisample_color);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, color_internal_format, rt->width, rt->height);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rt->multisample_color);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			// Delete allocated resources and default to no MSAA
			WARN_PRINT_ONCE("Cannot allocate back framebuffer for MSAA");
			printf("err status: %x\n", status);
			rt->multisample_active = false;

			glDeleteFramebuffers(1, &rt->multisample_fbo);
			rt->multisample_fbo = 0;

			glDeleteRenderbuffers(1, &rt->multisample_depth);
			rt->multisample_depth = 0;

			glDeleteRenderbuffers(1, &rt->multisample_color);
			rt->multisample_color = 0;
		}

		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		bind_framebuffer(0);

	} else
#endif // JAVASCRIPT_ENABLED
	{
		rt->multisample_active = false;
	}

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// copy texscreen buffers
	//	if (!(rt->flags[RendererStorage::RENDER_TARGET_NO_SAMPLING])) {
	if (true) {
		glGenTextures(1, &rt->copy_screen_effect.color);
		glBindTexture(GL_TEXTURE_2D, rt->copy_screen_effect.color);

		if (rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rt->width, rt->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		} else {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rt->width, rt->height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glGenFramebuffers(1, &rt->copy_screen_effect.fbo);
		bind_framebuffer(rt->copy_screen_effect.fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->copy_screen_effect.color, 0);

		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			_render_target_clear(rt);
			ERR_FAIL_COND(status != GL_FRAMEBUFFER_COMPLETE);
		}
	}

	// Allocate mipmap chains for post_process effects
	//	if (!rt->flags[RendererStorage::RENDER_TARGET_NO_3D] && rt->width >= 2 && rt->height >= 2) {
	if (rt->width >= 2 && rt->height >= 2) {
		for (int i = 0; i < 2; i++) {
			ERR_FAIL_COND(rt->mip_maps[i].sizes.size());
			int w = rt->width;
			int h = rt->height;

			if (i > 0) {
				w >>= 1;
				h >>= 1;
			}

			int level = 0;
			int fb_w = w;
			int fb_h = h;

			while (true) {
				GLES3::RenderTarget::MipMaps::Size mm;
				mm.width = w;
				mm.height = h;
				rt->mip_maps[i].sizes.push_back(mm);

				w >>= 1;
				h >>= 1;

				if (w < 2 || h < 2) {
					break;
				}

				level++;
			}

			GLsizei width = fb_w;
			GLsizei height = fb_h;

			if (config->render_to_mipmap_supported) {
				glGenTextures(1, &rt->mip_maps[i].color);
				glBindTexture(GL_TEXTURE_2D, rt->mip_maps[i].color);

				for (int l = 0; l < level + 1; l++) {
					glTexImage2D(GL_TEXTURE_2D, l, color_internal_format, width, height, 0, color_format, color_type, nullptr);
					width = MAX(1, (width / 2));
					height = MAX(1, (height / 2));
				}
#ifdef GLES_OVER_GL
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, level);
#endif
			} else {
				// Can't render to specific levels of a mipmap in ES 2.0 or Webgl so create a texture for each level
				for (int l = 0; l < level + 1; l++) {
					glGenTextures(1, &rt->mip_maps[i].sizes.write[l].color);
					glBindTexture(GL_TEXTURE_2D, rt->mip_maps[i].sizes[l].color);
					glTexImage2D(GL_TEXTURE_2D, 0, color_internal_format, width, height, 0, color_format, color_type, nullptr);
					width = MAX(1, (width / 2));
					height = MAX(1, (height / 2));
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
				}
			}

			glDisable(GL_SCISSOR_TEST);
			glColorMask(1, 1, 1, 1);
			glDepthMask(GL_TRUE);

			for (int j = 0; j < rt->mip_maps[i].sizes.size(); j++) {
				GLES3::RenderTarget::MipMaps::Size &mm = rt->mip_maps[i].sizes.write[j];

				glGenFramebuffers(1, &mm.fbo);
				bind_framebuffer(mm.fbo);

				if (config->render_to_mipmap_supported) {
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->mip_maps[i].color, j);
				} else {
					glBindTexture(GL_TEXTURE_2D, rt->mip_maps[i].sizes[j].color);
					glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->mip_maps[i].sizes[j].color, 0);
				}

				bool used_depth = false;
				if (j == 0 && i == 0) { //use always
					if (config->support_depth_texture) {
						glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rt->depth, 0);
					} else {
						glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);
					}
					used_depth = true;
				}

				GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
				if (status != GL_FRAMEBUFFER_COMPLETE) {
					WARN_PRINT_ONCE("Cannot allocate mipmaps for 3D post processing effects");
					bind_framebuffer_system();
					return;
				}

				glClearColor(1.0, 0.0, 1.0, 0.0);
				glClear(GL_COLOR_BUFFER_BIT);
				if (used_depth) {
					glClearDepth(1.0);
					glClear(GL_DEPTH_BUFFER_BIT);
				}
			}

			rt->mip_maps[i].levels = level;

			if (config->render_to_mipmap_supported) {
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}
		}
		rt->mip_maps_allocated = true;
	}

	bind_framebuffer_system();
}

void RasterizerStorageGLES3::_render_target_clear(GLES3::RenderTarget *rt) {
	// there is nothing to clear when DIRECT_TO_SCREEN is used
	if (rt->flags[RENDER_TARGET_DIRECT_TO_SCREEN]) {
		return;
	}

	if (rt->fbo) {
		glDeleteFramebuffers(1, &rt->fbo);
		glDeleteTextures(1, &rt->color);
		rt->fbo = 0;
	}

	if (rt->external.fbo != 0) {
		// free this
		glDeleteFramebuffers(1, &rt->external.fbo);

		// clean up our texture
		GLES3::Texture *t = GLES3::TextureStorage::get_singleton()->get_texture(rt->external.texture);
		t->alloc_height = 0;
		t->alloc_width = 0;
		t->width = 0;
		t->height = 0;
		t->active = false;
		GLES3::TextureStorage::get_singleton()->texture_free(rt->external.texture);
		memdelete(t);

		rt->external.fbo = 0;
	}

	if (rt->depth) {
		if (config->support_depth_texture) {
			glDeleteTextures(1, &rt->depth);
		} else {
			glDeleteRenderbuffers(1, &rt->depth);
		}

		rt->depth = 0;
	}

	GLES3::Texture *tex = GLES3::TextureStorage::get_singleton()->get_texture(rt->texture);
	tex->alloc_height = 0;
	tex->alloc_width = 0;
	tex->width = 0;
	tex->height = 0;
	tex->active = false;

	if (rt->copy_screen_effect.color) {
		glDeleteFramebuffers(1, &rt->copy_screen_effect.fbo);
		rt->copy_screen_effect.fbo = 0;

		glDeleteTextures(1, &rt->copy_screen_effect.color);
		rt->copy_screen_effect.color = 0;
	}

	for (int i = 0; i < 2; i++) {
		if (rt->mip_maps[i].sizes.size()) {
			for (int j = 0; j < rt->mip_maps[i].sizes.size(); j++) {
				glDeleteFramebuffers(1, &rt->mip_maps[i].sizes[j].fbo);
				glDeleteTextures(1, &rt->mip_maps[i].sizes[j].color);
			}

			glDeleteTextures(1, &rt->mip_maps[i].color);
			rt->mip_maps[i].sizes.clear();
			rt->mip_maps[i].levels = 0;
			rt->mip_maps[i].color = 0;
		}
	}

	if (rt->multisample_active) {
		glDeleteFramebuffers(1, &rt->multisample_fbo);
		rt->multisample_fbo = 0;

		glDeleteRenderbuffers(1, &rt->multisample_depth);
		rt->multisample_depth = 0;

		glDeleteRenderbuffers(1, &rt->multisample_color);

		rt->multisample_color = 0;
	}
}

RID RasterizerStorageGLES3::render_target_create() {
	GLES3::RenderTarget *rt = memnew(GLES3::RenderTarget);
	GLES3::Texture *t = memnew(GLES3::Texture);

	t->type = RenderingDevice::TEXTURE_TYPE_2D;
	t->flags = 0;
	t->width = 0;
	t->height = 0;
	t->alloc_height = 0;
	t->alloc_width = 0;
	t->format = Image::FORMAT_R8;
	t->target = GL_TEXTURE_2D;
	t->gl_format_cache = 0;
	t->gl_internal_format_cache = 0;
	t->gl_type_cache = 0;
	t->data_size = 0;
	t->total_data_size = 0;
	t->ignore_mipmaps = false;
	t->compressed = false;
	t->mipmaps = 1;
	t->active = true;
	t->tex_id = 0;
	t->render_target = rt;

	rt->texture = GLES3::TextureStorage::get_singleton()->make_rid(t);
	return render_target_owner.make_rid(rt);
}

void RasterizerStorageGLES3::render_target_set_position(RID p_render_target, int p_x, int p_y) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->x = p_x;
	rt->y = p_y;
}

void RasterizerStorageGLES3::render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	if (p_width == rt->width && p_height == rt->height) {
		return;
	}

	_render_target_clear(rt);

	rt->width = p_width;
	rt->height = p_height;

	// print_line("render_target_set_size " + itos(p_render_target.get_id()) + ", w " + itos(p_width) + " h " + itos(p_height));

	rt->allocate_is_dirty = true;
	//_render_target_allocate(rt);
}

// TODO: convert to Size2i internally
Size2i RasterizerStorageGLES3::render_target_get_size(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Size2());

	return Size2i(rt->width, rt->height);
}

RID RasterizerStorageGLES3::render_target_get_texture(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, RID());

	if (rt->external.fbo == 0) {
		return rt->texture;
	} else {
		return rt->external.texture;
	}
}

void RasterizerStorageGLES3::render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	if (p_texture_id == 0) {
		if (rt->external.fbo != 0) {
			// free this
			glDeleteFramebuffers(1, &rt->external.fbo);

			// and this
			if (rt->external.depth != 0) {
				glDeleteRenderbuffers(1, &rt->external.depth);
			}

			// clean up our texture
			GLES3::Texture *t = GLES3::TextureStorage::get_singleton()->get_texture(rt->external.texture);
			t->alloc_height = 0;
			t->alloc_width = 0;
			t->width = 0;
			t->height = 0;
			t->active = false;
			GLES3::TextureStorage::get_singleton()->texture_free(rt->external.texture);
			memdelete(t);

			rt->external.fbo = 0;
			rt->external.color = 0;
			rt->external.depth = 0;
		}
	} else {
		GLES3::Texture *t;

		if (rt->external.fbo == 0) {
			// create our fbo
			glGenFramebuffers(1, &rt->external.fbo);
			bind_framebuffer(rt->external.fbo);

			// allocate a texture
			t = memnew(GLES3::Texture);

			t->type = RenderingDevice::TEXTURE_TYPE_2D;
			t->flags = 0;
			t->width = 0;
			t->height = 0;
			t->alloc_height = 0;
			t->alloc_width = 0;
			t->format = Image::FORMAT_RGBA8;
			t->target = GL_TEXTURE_2D;
			t->gl_format_cache = 0;
			t->gl_internal_format_cache = 0;
			t->gl_type_cache = 0;
			t->data_size = 0;
			t->compressed = false;
			t->srgb = false;
			t->total_data_size = 0;
			t->ignore_mipmaps = false;
			t->mipmaps = 1;
			t->active = true;
			t->tex_id = 0;
			t->render_target = rt;

			rt->external.texture = GLES3::TextureStorage::get_singleton()->make_rid(t);

		} else {
			// bind our frame buffer
			bind_framebuffer(rt->external.fbo);

			// find our texture
			t = GLES3::TextureStorage::get_singleton()->get_texture(rt->external.texture);
		}

		// set our texture
		t->tex_id = p_texture_id;
		rt->external.color = p_texture_id;

		// size shouldn't be different
		t->width = rt->width;
		t->height = rt->height;
		t->alloc_height = rt->width;
		t->alloc_width = rt->height;

		// Switch our texture on our frame buffer
		{
			// set our texture as the destination for our framebuffer
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, p_texture_id, 0);

			// seeing we're rendering into this directly, better also use our depth buffer, just use our existing one :)
			if (config->support_depth_texture) {
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rt->depth, 0);
			} else {
				glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);
			}
		}

		// check status and unbind
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		bind_framebuffer_system();

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			printf("framebuffer fail, status: %x\n", status);
		}

		ERR_FAIL_COND(status != GL_FRAMEBUFFER_COMPLETE);
	}
}

void RasterizerStorageGLES3::render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	// When setting DIRECT_TO_SCREEN, you need to clear before the value is set, but allocate after as
	// those functions change how they operate depending on the value of DIRECT_TO_SCREEN
	if (p_flag == RENDER_TARGET_DIRECT_TO_SCREEN && p_value != rt->flags[RENDER_TARGET_DIRECT_TO_SCREEN]) {
		_render_target_clear(rt);
		rt->flags[p_flag] = p_value;
		_render_target_allocate(rt);
	}

	rt->flags[p_flag] = p_value;

	switch (p_flag) {
		case RENDER_TARGET_TRANSPARENT:
			/*
		case RENDER_TARGET_HDR:
		case RENDER_TARGET_NO_3D:
		case RENDER_TARGET_NO_SAMPLING:
		case RENDER_TARGET_NO_3D_EFFECTS: */
			{
				//must reset for these formats
				_render_target_clear(rt);
				_render_target_allocate(rt);
			}
			break;
		default: {
		}
	}
}

bool RasterizerStorageGLES3::render_target_was_used(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);

	return rt->used_in_frame;
}

void RasterizerStorageGLES3::render_target_clear_used(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->used_in_frame = false;
}

void RasterizerStorageGLES3::render_target_set_msaa(RID p_render_target, RS::ViewportMSAA p_msaa) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	if (rt->msaa == p_msaa) {
		return;
	}

	_render_target_clear(rt);
	rt->msaa = p_msaa;
	_render_target_allocate(rt);
}

//RasterizerStorageGLES3::GLES3::RenderTarget * RasterizerStorageGLES3::render_target_get(RID p_render_target)
//{
//	return render_target_owner.get_or_null(p_render_target);
//}

void RasterizerStorageGLES3::render_target_set_use_fxaa(RID p_render_target, bool p_fxaa) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->use_fxaa = p_fxaa;
}

void RasterizerStorageGLES3::render_target_set_use_debanding(RID p_render_target, bool p_debanding) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);

	if (p_debanding) {
		WARN_PRINT_ONCE("Debanding is not supported in the OpenGL backend. Switch to the Vulkan backend and make sure HDR is enabled.");
	}

	rt->use_debanding = p_debanding;
}

void RasterizerStorageGLES3::render_target_request_clear(RID p_render_target, const Color &p_clear_color) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = true;
	rt->clear_color = p_clear_color;

	//	ERR_FAIL_COND(!frame.current_rt);
	//	frame.clear_request = true;
	//	frame.clear_request_color = p_color;
}

bool RasterizerStorageGLES3::render_target_is_clear_requested(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, false);
	return rt->clear_requested;
}
Color RasterizerStorageGLES3::render_target_get_clear_request_color(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND_V(!rt, Color());
	return rt->clear_color;
}

void RasterizerStorageGLES3::render_target_disable_clear_request(RID p_render_target) {
	GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_render_target);
	ERR_FAIL_COND(!rt);
	rt->clear_requested = false;
}

void RasterizerStorageGLES3::render_target_do_clear_request(RID p_render_target) {
}

void RasterizerStorageGLES3::render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) {
}

Rect2i RasterizerStorageGLES3::render_target_get_sdf_rect(RID p_render_target) const {
	return Rect2i();
}

void RasterizerStorageGLES3::render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) {
}

/* CANVAS SHADOW */

RID RasterizerStorageGLES3::canvas_light_shadow_buffer_create(int p_width) {
	CanvasLightShadow *cls = memnew(CanvasLightShadow);

	if (p_width > config->max_texture_size) {
		p_width = config->max_texture_size;
	}

	cls->size = p_width;
	cls->height = 16;

	glActiveTexture(GL_TEXTURE0);

	glGenFramebuffers(1, &cls->fbo);
	bind_framebuffer(cls->fbo);

	glGenRenderbuffers(1, &cls->depth);
	glBindRenderbuffer(GL_RENDERBUFFER, cls->depth);
	glRenderbufferStorage(GL_RENDERBUFFER, config->depth_buffer_internalformat, cls->size, cls->height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, cls->depth);

	glGenTextures(1, &cls->distance);
	glBindTexture(GL_TEXTURE_2D, cls->distance);
	if (config->use_rgba_2d_shadows) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cls->size, cls->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	} else {
#ifdef GLES_OVER_GL
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, cls->size, cls->height, 0, _RED_OES, GL_FLOAT, nullptr);
#else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_FLOAT, cls->size, cls->height, 0, _RED_OES, GL_FLOAT, NULL);
#endif
	}

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cls->distance, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	//printf("errnum: %x\n",status);
	bind_framebuffer_system();

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		memdelete(cls);
		ERR_FAIL_COND_V(status != GL_FRAMEBUFFER_COMPLETE, RID());
	}

	return canvas_light_shadow_owner.make_rid(cls);
}

/* LIGHT SHADOW MAPPING */
/*

RID RasterizerStorageGLES3::canvas_light_occluder_create() {
	CanvasOccluder *co = memnew(CanvasOccluder);
	co->index_id = 0;
	co->vertex_id = 0;
	co->len = 0;

	return canvas_occluder_owner.make_rid(co);
}

void RasterizerStorageGLES3::canvas_light_occluder_set_polylines(RID p_occluder, const PoolVector<Vector2> &p_lines) {
	CanvasOccluder *co = canvas_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!co);

	co->lines = p_lines;

	if (p_lines.size() != co->len) {
		if (co->index_id) {
			glDeleteBuffers(1, &co->index_id);
		} if (co->vertex_id) {
			glDeleteBuffers(1, &co->vertex_id);
		}

		co->index_id = 0;
		co->vertex_id = 0;
		co->len = 0;
	}

	if (p_lines.size()) {
		PoolVector<float> geometry;
		PoolVector<uint16_t> indices;
		int lc = p_lines.size();

		geometry.resize(lc * 6);
		indices.resize(lc * 3);

		PoolVector<float>::Write vw = geometry.write();
		PoolVector<uint16_t>::Write iw = indices.write();

		PoolVector<Vector2>::Read lr = p_lines.read();

		const int POLY_HEIGHT = 16384;

		for (int i = 0; i < lc / 2; i++) {
			vw[i * 12 + 0] = lr[i * 2 + 0].x;
			vw[i * 12 + 1] = lr[i * 2 + 0].y;
			vw[i * 12 + 2] = POLY_HEIGHT;

			vw[i * 12 + 3] = lr[i * 2 + 1].x;
			vw[i * 12 + 4] = lr[i * 2 + 1].y;
			vw[i * 12 + 5] = POLY_HEIGHT;

			vw[i * 12 + 6] = lr[i * 2 + 1].x;
			vw[i * 12 + 7] = lr[i * 2 + 1].y;
			vw[i * 12 + 8] = -POLY_HEIGHT;

			vw[i * 12 + 9] = lr[i * 2 + 0].x;
			vw[i * 12 + 10] = lr[i * 2 + 0].y;
			vw[i * 12 + 11] = -POLY_HEIGHT;

			iw[i * 6 + 0] = i * 4 + 0;
			iw[i * 6 + 1] = i * 4 + 1;
			iw[i * 6 + 2] = i * 4 + 2;

			iw[i * 6 + 3] = i * 4 + 2;
			iw[i * 6 + 4] = i * 4 + 3;
			iw[i * 6 + 5] = i * 4 + 0;
		}

		//if same buffer len is being set, just use BufferSubData to avoid a pipeline flush

		if (!co->vertex_id) {
			glGenBuffers(1, &co->vertex_id);
			glBindBuffer(GL_ARRAY_BUFFER, co->vertex_id);
			glBufferData(GL_ARRAY_BUFFER, lc * 6 * sizeof(real_t), vw.ptr(), GL_STATIC_DRAW);
		} else {
			glBindBuffer(GL_ARRAY_BUFFER, co->vertex_id);
			glBufferSubData(GL_ARRAY_BUFFER, 0, lc * 6 * sizeof(real_t), vw.ptr());
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind

		if (!co->index_id) {
			glGenBuffers(1, &co->index_id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, co->index_id);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, lc * 3 * sizeof(uint16_t), iw.ptr(), GL_DYNAMIC_DRAW);
		} else {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, co->index_id);
			glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, lc * 3 * sizeof(uint16_t), iw.ptr());
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind

		co->len = lc;
	}
}
*/

RS::InstanceType RasterizerStorageGLES3::get_base_type(RID p_rid) const {
	return RS::INSTANCE_NONE;

	/*
	if (mesh_owner.owns(p_rid)) {
		return RS::INSTANCE_MESH;
	} else if (light_owner.owns(p_rid)) {
		return RS::INSTANCE_LIGHT;
	} else if (multimesh_owner.owns(p_rid)) {
		return RS::INSTANCE_MULTIMESH;
	} else if (immediate_owner.owns(p_rid)) {
		return RS::INSTANCE_IMMEDIATE;
	} else if (reflection_probe_owner.owns(p_rid)) {
		return RS::INSTANCE_REFLECTION_PROBE;
	} else if (lightmap_capture_data_owner.owns(p_rid)) {
		return RS::INSTANCE_LIGHTMAP_CAPTURE;
	} else {
		return RS::INSTANCE_NONE;
	}
*/
}

bool RasterizerStorageGLES3::free(RID p_rid) {
	if (render_target_owner.owns(p_rid)) {
		GLES3::RenderTarget *rt = render_target_owner.get_or_null(p_rid);
		_render_target_clear(rt);

		GLES3::Texture *t = GLES3::TextureStorage::get_singleton()->get_texture(rt->texture);
		if (t) {
			GLES3::TextureStorage::get_singleton()->texture_free(rt->texture);
			memdelete(t);
		}
		render_target_owner.free(p_rid);
		memdelete(rt);

		return true;
	} else if (GLES3::TextureStorage::get_singleton()->owns_texture(p_rid)) {
		GLES3::TextureStorage::get_singleton()->texture_free(p_rid);
		return true;
	} else if (GLES3::CanvasTextureStorage::get_singleton()->owns_canvas_texture(p_rid)) {
		GLES3::CanvasTextureStorage::get_singleton()->canvas_texture_free(p_rid);
		return true;
	} else if (sky_owner.owns(p_rid)) {
		Sky *sky = sky_owner.get_or_null(p_rid);
		sky_set_texture(p_rid, RID(), 256);
		sky_owner.free(p_rid);
		memdelete(sky);

		return true;
	} else if (GLES3::MaterialStorage::get_singleton()->owns_shader(p_rid)) {
		GLES3::MaterialStorage::get_singleton()->shader_free(p_rid);
		return true;
	} else if (GLES3::MaterialStorage::get_singleton()->owns_material(p_rid)) {
		GLES3::MaterialStorage::get_singleton()->material_free(p_rid);
		return true;
	} else {
		return false;
	}
	/*
	} else if (skeleton_owner.owns(p_rid)) {
		Skeleton *s = skeleton_owner.get_or_null(p_rid);

		if (s->update_list.in_list()) {
			skeleton_update_list.remove(&s->update_list);
		}

		for (Set<InstanceBaseDependency *>::Element *E = s->instances.front(); E; E = E->next()) {
			E->get()->skeleton = RID();
		}

		skeleton_allocate(p_rid, 0, false);

		if (s->tex_id) {
			glDeleteTextures(1, &s->tex_id);
		}

		skeleton_owner.free(p_rid);
		memdelete(s);

		return true;
	} else if (mesh_owner.owns(p_rid)) {
		Mesh *mesh = mesh_owner.get_or_null(p_rid);

		mesh->instance_remove_deps();
		mesh_clear(p_rid);

		while (mesh->multimeshes.first()) {
			MultiMesh *multimesh = mesh->multimeshes.first()->self();
			multimesh->mesh = RID();
			multimesh->dirty_aabb = true;

			mesh->multimeshes.remove(mesh->multimeshes.first());

			if (!multimesh->update_list.in_list()) {
				multimesh_update_list.add(&multimesh->update_list);
			}
		}

		mesh_owner.free(p_rid);
		memdelete(mesh);

		return true;
	} else if (multimesh_owner.owns(p_rid)) {
		MultiMesh *multimesh = multimesh_owner.get_or_null(p_rid);
		multimesh->instance_remove_deps();

		if (multimesh->mesh.is_valid()) {
			Mesh *mesh = mesh_owner.get_or_null(multimesh->mesh);
			if (mesh) {
				mesh->multimeshes.remove(&multimesh->mesh_list);
			}
		}

		multimesh_allocate(p_rid, 0, RS::MULTIMESH_TRANSFORM_3D, RS::MULTIMESH_COLOR_NONE);

		update_dirty_multimeshes();

		multimesh_owner.free(p_rid);
		memdelete(multimesh);

		return true;
	} else if (immediate_owner.owns(p_rid)) {
		Immediate *im = immediate_owner.get_or_null(p_rid);
		im->instance_remove_deps();

		immediate_owner.free(p_rid);
		memdelete(im);

		return true;
	} else if (light_owner.owns(p_rid)) {
		Light *light = light_owner.get_or_null(p_rid);
		light->instance_remove_deps();

		light_owner.free(p_rid);
		memdelete(light);

		return true;
	} else if (reflection_probe_owner.owns(p_rid)) {
		// delete the texture
		ReflectionProbe *reflection_probe = reflection_probe_owner.get_or_null(p_rid);
		reflection_probe->instance_remove_deps();

		reflection_probe_owner.free(p_rid);
		memdelete(reflection_probe);

		return true;
	} else if (lightmap_capture_data_owner.owns(p_rid)) {
		// delete the texture
		LightmapCapture *lightmap_capture = lightmap_capture_data_owner.get_or_null(p_rid);
		lightmap_capture->instance_remove_deps();

		lightmap_capture_data_owner.free(p_rid);
		memdelete(lightmap_capture);
		return true;

	} else if (canvas_occluder_owner.owns(p_rid)) {
		CanvasOccluder *co = canvas_occluder_owner.get_or_null(p_rid);
		if (co->index_id) {
			glDeleteBuffers(1, &co->index_id);
		}
		if (co->vertex_id) {
			glDeleteBuffers(1, &co->vertex_id);
		}

		canvas_occluder_owner.free(p_rid);
		memdelete(co);

		return true;

	} else if (canvas_light_shadow_owner.owns(p_rid)) {
		CanvasLightShadow *cls = canvas_light_shadow_owner.get_or_null(p_rid);
		glDeleteFramebuffers(1, &cls->fbo);
		glDeleteRenderbuffers(1, &cls->depth);
		glDeleteTextures(1, &cls->distance);
		canvas_light_shadow_owner.free(p_rid);
		memdelete(cls);

		return true;
		*/
}

bool RasterizerStorageGLES3::has_os_feature(const String &p_feature) const {
	if (p_feature == "s3tc") {
		return config->s3tc_supported;
	}

	if (p_feature == "etc") {
		return config->etc_supported;
	}

	if (p_feature == "skinning_fallback") {
		return config->use_skeleton_software;
	}

	return false;
}

////////////////////////////////////////////

void RasterizerStorageGLES3::set_debug_generate_wireframes(bool p_generate) {
}

//void RasterizerStorageGLES3::render_info_begin_capture() {
//	info.snap = info.render;
//}

//void RasterizerStorageGLES3::render_info_end_capture() {
//	info.snap.object_count = info.render.object_count - info.snap.object_count;
//	info.snap.draw_call_count = info.render.draw_call_count - info.snap.draw_call_count;
//	info.snap.material_switch_count = info.render.material_switch_count - info.snap.material_switch_count;
//	info.snap.surface_switch_count = info.render.surface_switch_count - info.snap.surface_switch_count;
//	info.snap.shader_rebind_count = info.render.shader_rebind_count - info.snap.shader_rebind_count;
//	info.snap.vertices_count = info.render.vertices_count - info.snap.vertices_count;
//	info.snap._2d_item_count = info.render._2d_item_count - info.snap._2d_item_count;
//	info.snap._2d_draw_call_count = info.render._2d_draw_call_count - info.snap._2d_draw_call_count;
//}

//int RasterizerStorageGLES3::get_captured_render_info(RS::RenderInfo p_info) {
//	switch (p_info) {
//		case RS::INFO_OBJECTS_IN_FRAME: {
//			return info.snap.object_count;
//		} break;
//		case RS::INFO_VERTICES_IN_FRAME: {
//			return info.snap.vertices_count;
//		} break;
//		case RS::INFO_MATERIAL_CHANGES_IN_FRAME: {
//			return info.snap.material_switch_count;
//		} break;
//		case RS::INFO_SHADER_CHANGES_IN_FRAME: {
//			return info.snap.shader_rebind_count;
//		} break;
//		case RS::INFO_SURFACE_CHANGES_IN_FRAME: {
//			return info.snap.surface_switch_count;
//		} break;
//		case RS::INFO_DRAW_CALLS_IN_FRAME: {
//			return info.snap.draw_call_count;
//		} break;
//			/*
//		case RS::INFO_2D_ITEMS_IN_FRAME: {
//			return info.snap._2d_item_count;
//		} break;
//		case RS::INFO_2D_DRAW_CALLS_IN_FRAME: {
//			return info.snap._2d_draw_call_count;
//		} break;
//			*/
//		default: {
//			return get_render_info(p_info);
//		}
//	}
//}

//int RasterizerStorageGLES3::get_render_info(RS::RenderInfo p_info) {
//	switch (p_info) {
//		case RS::INFO_OBJECTS_IN_FRAME:
//			return info.render_final.object_count;
//		case RS::INFO_VERTICES_IN_FRAME:
//			return info.render_final.vertices_count;
//		case RS::INFO_MATERIAL_CHANGES_IN_FRAME:
//			return info.render_final.material_switch_count;
//		case RS::INFO_SHADER_CHANGES_IN_FRAME:
//			return info.render_final.shader_rebind_count;
//		case RS::INFO_SURFACE_CHANGES_IN_FRAME:
//			return info.render_final.surface_switch_count;
//		case RS::INFO_DRAW_CALLS_IN_FRAME:
//			return info.render_final.draw_call_count;
//			/*
//		case RS::INFO_2D_ITEMS_IN_FRAME:
//			return info.render_final._2d_item_count;
//		case RS::INFO_2D_DRAW_CALLS_IN_FRAME:
//			return info.render_final._2d_draw_call_count;
//*/
//		case RS::INFO_USAGE_VIDEO_MEM_TOTAL:
//			return 0; //no idea
//		case RS::INFO_VIDEO_MEM_USED:
//			return info.vertex_mem + info.texture_mem;
//		case RS::INFO_TEXTURE_MEM_USED:
//			return info.texture_mem;
//		case RS::INFO_VERTEX_MEM_USED:
//			return info.vertex_mem;
//		default:
//			return 0; //no idea either
//	}
//}

String RasterizerStorageGLES3::get_video_adapter_name() const {
	return (const char *)glGetString(GL_RENDERER);
}

String RasterizerStorageGLES3::get_video_adapter_vendor() const {
	return (const char *)glGetString(GL_VENDOR);
}

RenderingDevice::DeviceType RasterizerStorageGLES3::get_video_adapter_type() const {
	return RenderingDevice::DeviceType::DEVICE_TYPE_OTHER;
}

void RasterizerStorageGLES3::initialize() {
	RasterizerStorageGLES3::system_fbo = 0;
	config = GLES3::Config::get_singleton();
	config->initialize();

	//determine formats for depth textures (or renderbuffers)
	if (config->support_depth_texture) {
		// Will use texture for depth
		// have to manually see if we can create a valid framebuffer texture using UNSIGNED_INT,
		// as there is no extension to test for this.
		GLuint fbo;
		glGenFramebuffers(1, &fbo);
		bind_framebuffer(fbo);
		GLuint depth;
		glGenTextures(1, &depth);
		glBindTexture(GL_TEXTURE_2D, depth);
		glTexImage2D(GL_TEXTURE_2D, 0, config->depth_internalformat, 32, 32, 0, GL_DEPTH_COMPONENT, config->depth_type, nullptr);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		bind_framebuffer_system();
		glDeleteFramebuffers(1, &fbo);
		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(1, &depth);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			// If it fails, test to see if it supports a framebuffer texture using UNSIGNED_SHORT
			// This is needed because many OSX devices don't support either UNSIGNED_INT or UNSIGNED_SHORT
#ifdef GLES_OVER_GL
			config->depth_internalformat = GL_DEPTH_COMPONENT16;
#else
			// OES_depth_texture extension only specifies GL_DEPTH_COMPONENT.
			config->depth_internalformat = GL_DEPTH_COMPONENT;
#endif
			config->depth_type = GL_UNSIGNED_SHORT;

			glGenFramebuffers(1, &fbo);
			bind_framebuffer(fbo);

			glGenTextures(1, &depth);
			glBindTexture(GL_TEXTURE_2D, depth);
			glTexImage2D(GL_TEXTURE_2D, 0, config->depth_internalformat, 32, 32, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);

			status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			if (status != GL_FRAMEBUFFER_COMPLETE) {
				//if it fails again depth textures aren't supported, use rgba shadows and renderbuffer for depth
				config->support_depth_texture = false;
				config->use_rgba_3d_shadows = true;
			}

			bind_framebuffer_system();
			glDeleteFramebuffers(1, &fbo);
			glBindTexture(GL_TEXTURE_2D, 0);
			glDeleteTextures(1, &depth);
		}
	}

	//picky requirements for these
	config->support_shadow_cubemaps = config->support_depth_texture && config->support_write_depth && config->support_depth_cubemaps;

	frame.count = 0;
	frame.delta = 0;
	frame.current_rt = nullptr;
	frame.clear_request = false;

	// the use skeleton software path should be used if either float texture is not supported,
	// OR max_vertex_texture_image_units is zero
	config->use_skeleton_software = (config->float_texture_supported == false) || (config->max_vertex_texture_image_units == 0);

	{
		// quad for copying stuff

		glGenBuffers(1, &resources.quadie);
		glBindBuffer(GL_ARRAY_BUFFER, resources.quadie);
		{
			const float qv[16] = {
				-1,
				-1,
				0,
				0,
				-1,
				1,
				0,
				1,
				1,
				1,
				1,
				1,
				1,
				-1,
				1,
				0,
			};

			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 16, qv, GL_STATIC_DRAW);
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	{
		//default textures

		glGenTextures(1, &resources.white_tex);
		unsigned char whitetexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i++) {
			whitetexdata[i] = 255;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.white_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, whitetexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenTextures(1, &resources.black_tex);
		unsigned char blacktexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i++) {
			blacktexdata[i] = 0;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.black_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, blacktexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenTextures(1, &resources.normal_tex);
		unsigned char normaltexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i += 3) {
			normaltexdata[i + 0] = 128;
			normaltexdata[i + 1] = 128;
			normaltexdata[i + 2] = 255;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.normal_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, normaltexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		glGenTextures(1, &resources.aniso_tex);
		unsigned char anisotexdata[8 * 8 * 3];
		for (int i = 0; i < 8 * 8 * 3; i += 3) {
			anisotexdata[i + 0] = 255;
			anisotexdata[i + 1] = 128;
			anisotexdata[i + 2] = 0;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.aniso_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, anisotexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	// skeleton buffer
	{
		resources.skeleton_transform_buffer_size = 0;
		glGenBuffers(1, &resources.skeleton_transform_buffer);
	}

	// radical inverse vdc cache texture
	// used for cubemap filtering
	if (true /*||config->float_texture_supported*/) { //uint8 is similar and works everywhere
		glGenTextures(1, &resources.radical_inverse_vdc_cache_tex);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, resources.radical_inverse_vdc_cache_tex);

		uint8_t radical_inverse[512];

		for (uint32_t i = 0; i < 512; i++) {
			uint32_t bits = i;

			bits = (bits << 16) | (bits >> 16);
			bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
			bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
			bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
			bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);

			float value = float(bits) * 2.3283064365386963e-10;
			radical_inverse[i] = uint8_t(CLAMP(value * 255.0, 0, 255));
		}

		glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 512, 1, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, radical_inverse);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //need this for proper sampling

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	{
		glGenFramebuffers(1, &resources.mipmap_blur_fbo);
		glGenTextures(1, &resources.mipmap_blur_color);
	}

#ifdef GLES_OVER_GL
	//this needs to be enabled manually in OpenGL 2.1

	if (config->extensions.has("GL_ARB_seamless_cube_map")) {
		glEnable(_EXT_TEXTURE_CUBE_MAP_SEAMLESS);
	}
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif
}

void RasterizerStorageGLES3::finalize() {
}

void RasterizerStorageGLES3::_copy_screen() {
	bind_quad_array();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

void RasterizerStorageGLES3::update_memory_info() {
}

uint64_t RasterizerStorageGLES3::get_rendering_info(RS::RenderingInfo p_info) {
	return 0;
}

void RasterizerStorageGLES3::update_dirty_resources() {
	GLES3::MaterialStorage::get_singleton()->update_dirty_shaders();
	GLES3::MaterialStorage::get_singleton()->update_dirty_materials();
	//	update_dirty_skeletons();
	//	update_dirty_multimeshes();
}

RasterizerStorageGLES3::RasterizerStorageGLES3() {
	RasterizerStorageGLES3::system_fbo = 0;
}

RasterizerStorageGLES3::~RasterizerStorageGLES3() {
}

#endif // GLES3_ENABLED
