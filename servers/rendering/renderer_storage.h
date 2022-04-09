/*************************************************************************/
/*  renderer_storage.h                                                   */
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

#ifndef RENDERINGSERVERSTORAGE_H
#define RENDERINGSERVERSTORAGE_H

#include "servers/rendering_server.h"

class RendererStorage {
	Color default_clear_color;

public:
	enum DependencyChangedNotification {
		DEPENDENCY_CHANGED_AABB,
		DEPENDENCY_CHANGED_MATERIAL,
		DEPENDENCY_CHANGED_MESH,
		DEPENDENCY_CHANGED_MULTIMESH,
		DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES,
		DEPENDENCY_CHANGED_PARTICLES,
		DEPENDENCY_CHANGED_DECAL,
		DEPENDENCY_CHANGED_SKELETON_DATA,
		DEPENDENCY_CHANGED_SKELETON_BONES,
		DEPENDENCY_CHANGED_LIGHT,
		DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR,
		DEPENDENCY_CHANGED_REFLECTION_PROBE,
	};

	struct DependencyTracker;

	struct Dependency {
		void changed_notify(DependencyChangedNotification p_notification);
		void deleted_notify(const RID &p_rid);

		~Dependency();

	private:
		friend struct DependencyTracker;
		Map<DependencyTracker *, uint32_t> instances;
	};

	struct DependencyTracker {
		void *userdata = nullptr;
		typedef void (*ChangedCallback)(DependencyChangedNotification, DependencyTracker *);
		typedef void (*DeletedCallback)(const RID &, DependencyTracker *);

		ChangedCallback changed_callback = nullptr;
		DeletedCallback deleted_callback = nullptr;

		void update_begin() { // call before updating dependencies
			instance_version++;
		}

		void update_dependency(Dependency *p_dependency) { //called internally, can't be used directly, use update functions in Storage
			dependencies.insert(p_dependency);
			p_dependency->instances[this] = instance_version;
		}

		void update_end() { //call after updating dependencies
			List<Pair<Dependency *, Map<DependencyTracker *, uint32_t>::Element *>> to_clean_up;
			for (Set<Dependency *>::Element *E = dependencies.front(); E; E = E->next()) {
				Dependency *dep = E->get();
				Map<DependencyTracker *, uint32_t>::Element *F = dep->instances.find(this);
				ERR_CONTINUE(!F);
				if (F->get() != instance_version) {
					Pair<Dependency *, Map<DependencyTracker *, uint32_t>::Element *> p;
					p.first = dep;
					p.second = F;
					to_clean_up.push_back(p);
				}
			}

			while (to_clean_up.size()) {
				to_clean_up.front()->get().first->instances.erase(to_clean_up.front()->get().second);
				dependencies.erase(to_clean_up.front()->get().first);
				to_clean_up.pop_front();
			}
		}

		void clear() { // clear all dependencies
			for (Set<Dependency *>::Element *E = dependencies.front(); E; E = E->next()) {
				Dependency *dep = E->get();
				dep->instances.erase(this);
			}
			dependencies.clear();
		}

		~DependencyTracker() { clear(); }

	private:
		friend struct Dependency;
		uint32_t instance_version = 0;
		Set<Dependency *> dependencies;
	};

	virtual void base_update_dependency(RID p_base, DependencyTracker *p_instance) = 0;

	/* VOXEL GI API */

	virtual RID voxel_gi_allocate() = 0;
	virtual void voxel_gi_initialize(RID p_rid) = 0;

	virtual void voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) = 0;

	virtual AABB voxel_gi_get_bounds(RID p_voxel_gi) const = 0;
	virtual Vector3i voxel_gi_get_octree_size(RID p_voxel_gi) const = 0;
	virtual Vector<uint8_t> voxel_gi_get_octree_cells(RID p_voxel_gi) const = 0;
	virtual Vector<uint8_t> voxel_gi_get_data_cells(RID p_voxel_gi) const = 0;
	virtual Vector<uint8_t> voxel_gi_get_distance_field(RID p_voxel_gi) const = 0;

	virtual Vector<int> voxel_gi_get_level_counts(RID p_voxel_gi) const = 0;
	virtual Transform3D voxel_gi_get_to_cell_xform(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) = 0;
	virtual float voxel_gi_get_dynamic_range(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_propagation(RID p_voxel_gi, float p_range) = 0;
	virtual float voxel_gi_get_propagation(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_energy(RID p_voxel_gi, float p_energy) = 0;
	virtual float voxel_gi_get_energy(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_bias(RID p_voxel_gi, float p_bias) = 0;
	virtual float voxel_gi_get_bias(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) = 0;
	virtual float voxel_gi_get_normal_bias(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) = 0;
	virtual bool voxel_gi_is_interior(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) = 0;
	virtual bool voxel_gi_is_using_two_bounces(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) = 0;
	virtual float voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const = 0;

	virtual uint32_t voxel_gi_get_version(RID p_probe) = 0;

	/* PARTICLES */

	virtual RID particles_allocate() = 0;
	virtual void particles_initialize(RID p_rid) = 0;
	virtual void particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) = 0;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting) = 0;
	virtual bool particles_get_emitting(RID p_particles) = 0;

	virtual void particles_set_amount(RID p_particles, int p_amount) = 0;
	virtual void particles_set_lifetime(RID p_particles, double p_lifetime) = 0;
	virtual void particles_set_one_shot(RID p_particles, bool p_one_shot) = 0;
	virtual void particles_set_pre_process_time(RID p_particles, double p_time) = 0;
	virtual void particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) = 0;
	virtual void particles_set_randomness_ratio(RID p_particles, real_t p_ratio) = 0;
	virtual void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) = 0;
	virtual void particles_set_speed_scale(RID p_particles, double p_scale) = 0;
	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_process_material(RID p_particles, RID p_material) = 0;
	virtual RID particles_get_process_material(RID p_particles) const = 0;
	virtual void particles_set_fixed_fps(RID p_particles, int p_fps) = 0;
	virtual void particles_set_interpolate(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_fractional_delta(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_collision_base_size(RID p_particles, real_t p_size) = 0;

	virtual void particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) = 0;

	virtual void particles_set_trails(RID p_particles, bool p_enable, double p_length) = 0;
	virtual void particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) = 0;

	virtual void particles_restart(RID p_particles) = 0;
	virtual void particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) = 0;
	virtual void particles_set_subemitter(RID p_particles, RID p_subemitter_particles) = 0;

	virtual bool particles_is_inactive(RID p_particles) const = 0;

	virtual void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) = 0;

	virtual void particles_set_draw_passes(RID p_particles, int p_count) = 0;
	virtual void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) = 0;

	virtual void particles_request_process(RID p_particles) = 0;
	virtual AABB particles_get_current_aabb(RID p_particles) = 0;
	virtual AABB particles_get_aabb(RID p_particles) const = 0;

	virtual void particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) = 0;

	virtual int particles_get_draw_passes(RID p_particles) const = 0;
	virtual RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const = 0;

	virtual void particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) = 0;

	virtual void particles_add_collision(RID p_particles, RID p_particles_collision_instance) = 0;
	virtual void particles_remove_collision(RID p_particles, RID p_particles_collision_instance) = 0;

	virtual void particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) = 0;

	virtual void update_particles() = 0;

	/* PARTICLES COLLISION */

	virtual RID particles_collision_allocate() = 0;
	virtual void particles_collision_initialize(RID p_rid) = 0;

	virtual void particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) = 0;
	virtual void particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) = 0;
	virtual void particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) = 0; //for spheres
	virtual void particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) = 0; //for non-spheres
	virtual void particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) = 0;
	virtual void particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) = 0;
	virtual void particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) = 0;
	virtual void particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) = 0; //for SDF and vector field, heightfield is dynamic
	virtual void particles_collision_height_field_update(RID p_particles_collision) = 0; //for SDF and vector field
	virtual void particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) = 0; //for SDF and vector field
	virtual AABB particles_collision_get_aabb(RID p_particles_collision) const = 0;
	virtual bool particles_collision_is_heightfield(RID p_particles_collision) const = 0;
	virtual RID particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const = 0;

	/* FOG VOLUMES */

	virtual RID fog_volume_allocate() = 0;
	virtual void fog_volume_initialize(RID p_rid) = 0;

	virtual void fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) = 0;
	virtual void fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) = 0;
	virtual void fog_volume_set_material(RID p_fog_volume, RID p_material) = 0;
	virtual AABB fog_volume_get_aabb(RID p_fog_volume) const = 0;
	virtual RS::FogVolumeShape fog_volume_get_shape(RID p_fog_volume) const = 0;

	/* VISIBILITY NOTIFIER */

	virtual RID visibility_notifier_allocate() = 0;
	virtual void visibility_notifier_initialize(RID p_notifier) = 0;
	virtual void visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) = 0;
	virtual void visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) = 0;

	virtual AABB visibility_notifier_get_aabb(RID p_notifier) const = 0;
	virtual void visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) = 0;

	//used from 2D and 3D
	virtual RID particles_collision_instance_create(RID p_collision) = 0;
	virtual void particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) = 0;
	virtual void particles_collision_instance_set_active(RID p_collision_instance, bool p_active) = 0;

	virtual RS::InstanceType get_base_type(RID p_rid) const = 0;
	virtual bool free(RID p_rid) = 0;

	virtual bool has_os_feature(const String &p_feature) const = 0;

	virtual void update_dirty_resources() = 0;

	virtual void set_debug_generate_wireframes(bool p_generate) = 0;

	virtual void update_memory_info() = 0;

	virtual uint64_t get_rendering_info(RS::RenderingInfo p_info) = 0;
	virtual String get_video_adapter_name() const = 0;
	virtual String get_video_adapter_vendor() const = 0;
	virtual RenderingDevice::DeviceType get_video_adapter_type() const = 0;

	static RendererStorage *base_singleton;

	void set_default_clear_color(const Color &p_color) {
		default_clear_color = p_color;
	}

	Color get_default_clear_color() const {
		return default_clear_color;
	}
#define TIMESTAMP_BEGIN()                             \
	{                                                 \
		if (RSG::storage->capturing_timestamps)       \
			RSG::storage->capture_timestamps_begin(); \
	}

#define RENDER_TIMESTAMP(m_text)                     \
	{                                                \
		if (RSG::storage->capturing_timestamps)      \
			RSG::storage->capture_timestamp(m_text); \
	}

	bool capturing_timestamps = false;

	virtual void capture_timestamps_begin() = 0;
	virtual void capture_timestamp(const String &p_name) = 0;
	virtual uint32_t get_captured_timestamps_count() const = 0;
	virtual uint64_t get_captured_timestamps_frame() const = 0;
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const = 0;
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const = 0;
	virtual String get_captured_timestamp_name(uint32_t p_index) const = 0;

	RendererStorage();
	virtual ~RendererStorage() {}
};

#endif // RENDERINGSERVERSTORAGE_H
