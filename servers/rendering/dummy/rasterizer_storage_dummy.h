/*************************************************************************/
/*  rasterizer_storage_dummy.h                                           */
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

#ifndef RASTERIZER_STORAGE_DUMMY_H
#define RASTERIZER_STORAGE_DUMMY_H

#include "servers/rendering/renderer_storage.h"
#include "storage/texture_storage.h"

class RasterizerStorageDummy : public RendererStorage {
public:
	void base_update_dependency(RID p_base, DependencyTracker *p_instance) override {}

	/* VOXEL GI API */

	RID voxel_gi_allocate() override { return RID(); }
	void voxel_gi_initialize(RID p_rid) override {}
	void voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) override {}

	AABB voxel_gi_get_bounds(RID p_voxel_gi) const override { return AABB(); }
	Vector3i voxel_gi_get_octree_size(RID p_voxel_gi) const override { return Vector3i(); }
	Vector<uint8_t> voxel_gi_get_octree_cells(RID p_voxel_gi) const override { return Vector<uint8_t>(); }
	Vector<uint8_t> voxel_gi_get_data_cells(RID p_voxel_gi) const override { return Vector<uint8_t>(); }
	Vector<uint8_t> voxel_gi_get_distance_field(RID p_voxel_gi) const override { return Vector<uint8_t>(); }

	Vector<int> voxel_gi_get_level_counts(RID p_voxel_gi) const override { return Vector<int>(); }
	Transform3D voxel_gi_get_to_cell_xform(RID p_voxel_gi) const override { return Transform3D(); }

	void voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_dynamic_range(RID p_voxel_gi) const override { return 0; }

	void voxel_gi_set_propagation(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_propagation(RID p_voxel_gi) const override { return 0; }

	void voxel_gi_set_energy(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_energy(RID p_voxel_gi) const override { return 0.0; }

	void voxel_gi_set_bias(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_bias(RID p_voxel_gi) const override { return 0.0; }

	void voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_normal_bias(RID p_voxel_gi) const override { return 0.0; }

	void voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) override {}
	bool voxel_gi_is_interior(RID p_voxel_gi) const override { return false; }

	void voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) override {}
	bool voxel_gi_is_using_two_bounces(RID p_voxel_gi) const override { return false; }

	void voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) override {}
	float voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const override { return 0; }

	uint32_t voxel_gi_get_version(RID p_voxel_gi) override { return 0; }

	/* OCCLUDER */

	void occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) {}

	/* PARTICLES */

	RID particles_allocate() override { return RID(); }
	void particles_initialize(RID p_rid) override {}
	void particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) override {}
	void particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) override {}
	void particles_set_emitting(RID p_particles, bool p_emitting) override {}
	void particles_set_amount(RID p_particles, int p_amount) override {}
	void particles_set_lifetime(RID p_particles, double p_lifetime) override {}
	void particles_set_one_shot(RID p_particles, bool p_one_shot) override {}
	void particles_set_pre_process_time(RID p_particles, double p_time) override {}
	void particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) override {}
	void particles_set_randomness_ratio(RID p_particles, real_t p_ratio) override {}
	void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) override {}
	void particles_set_speed_scale(RID p_particles, double p_scale) override {}
	void particles_set_use_local_coordinates(RID p_particles, bool p_enable) override {}
	void particles_set_process_material(RID p_particles, RID p_material) override {}
	RID particles_get_process_material(RID p_particles) const override { return RID(); }
	void particles_set_fixed_fps(RID p_particles, int p_fps) override {}
	void particles_set_interpolate(RID p_particles, bool p_enable) override {}
	void particles_set_fractional_delta(RID p_particles, bool p_enable) override {}
	void particles_set_subemitter(RID p_particles, RID p_subemitter_particles) override {}
	void particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) override {}
	void particles_set_collision_base_size(RID p_particles, real_t p_size) override {}

	void particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) override {}

	void particles_set_trails(RID p_particles, bool p_enable, double p_length) override {}
	void particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) override {}

	void particles_restart(RID p_particles) override {}

	void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) override {}

	void particles_set_draw_passes(RID p_particles, int p_count) override {}
	void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) override {}

	void particles_request_process(RID p_particles) override {}
	AABB particles_get_current_aabb(RID p_particles) override { return AABB(); }
	AABB particles_get_aabb(RID p_particles) const override { return AABB(); }

	void particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) override {}

	bool particles_get_emitting(RID p_particles) override { return false; }
	int particles_get_draw_passes(RID p_particles) const override { return 0; }
	RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const override { return RID(); }

	void particles_add_collision(RID p_particles, RID p_instance) override {}
	void particles_remove_collision(RID p_particles, RID p_instance) override {}

	void particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) override {}

	void update_particles() override {}

	/* PARTICLES COLLISION */

	RID particles_collision_allocate() override { return RID(); }
	void particles_collision_initialize(RID p_rid) override {}
	void particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) override {}
	void particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) override {}
	void particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) override {}
	void particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) override {}
	void particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) override {}
	void particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) override {}
	void particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) override {}
	void particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) override {}
	void particles_collision_height_field_update(RID p_particles_collision) override {}
	void particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) override {}
	AABB particles_collision_get_aabb(RID p_particles_collision) const override { return AABB(); }
	bool particles_collision_is_heightfield(RID p_particles_collision) const override { return false; }
	RID particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const override { return RID(); }

	RID particles_collision_instance_create(RID p_collision) override { return RID(); }
	void particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) override {}
	void particles_collision_instance_set_active(RID p_collision_instance, bool p_active) override {}

	bool particles_is_inactive(RID p_particles) const override { return false; }

	/* FOG VOLUMES */

	RID fog_volume_allocate() override { return RID(); }
	void fog_volume_initialize(RID p_rid) override {}

	void fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) override {}
	void fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) override {}
	void fog_volume_set_material(RID p_fog_volume, RID p_material) override {}
	AABB fog_volume_get_aabb(RID p_fog_volume) const override { return AABB(); }
	RS::FogVolumeShape fog_volume_get_shape(RID p_fog_volume) const override { return RS::FOG_VOLUME_SHAPE_BOX; }

	/* VISIBILITY NOTIFIER */
	virtual RID visibility_notifier_allocate() override { return RID(); }
	virtual void visibility_notifier_initialize(RID p_notifier) override {}
	virtual void visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) override {}
	virtual void visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) override {}

	virtual AABB visibility_notifier_get_aabb(RID p_notifier) const override { return AABB(); }
	virtual void visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) override {}

	/* STORAGE */

	RS::InstanceType get_base_type(RID p_rid) const override { return RS::INSTANCE_NONE; }
	bool free(RID p_rid) override {
		if (RendererDummy::TextureStorage::get_singleton()->owns_texture(p_rid)) {
			RendererDummy::TextureStorage::get_singleton()->texture_free(p_rid);
			return true;
		}
		return false;
	}

	virtual void update_memory_info() override {}
	virtual uint64_t get_rendering_info(RS::RenderingInfo p_info) override { return 0; }

	bool has_os_feature(const String &p_feature) const override {
		return p_feature == "rgtc" || p_feature == "bptc" || p_feature == "s3tc" || p_feature == "etc" || p_feature == "etc2";
	}

	void update_dirty_resources() override {}

	void set_debug_generate_wireframes(bool p_generate) override {}

	String get_video_adapter_name() const override { return String(); }
	String get_video_adapter_vendor() const override { return String(); }
	RenderingDevice::DeviceType get_video_adapter_type() const override { return RenderingDevice::DeviceType::DEVICE_TYPE_OTHER; }

	static RendererStorage *base_singleton;

	void capture_timestamps_begin() override {}
	void capture_timestamp(const String &p_name) override {}
	uint32_t get_captured_timestamps_count() const override { return 0; }
	uint64_t get_captured_timestamps_frame() const override { return 0; }
	uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const override { return 0; }
	uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const override { return 0; }
	String get_captured_timestamp_name(uint32_t p_index) const override { return String(); }

	RasterizerStorageDummy() {}
	~RasterizerStorageDummy() {}
};

#endif // !RASTERIZER_STORAGE_DUMMY_H
