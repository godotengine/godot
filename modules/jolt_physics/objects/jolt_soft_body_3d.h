/**************************************************************************/
/*  jolt_soft_body_3d.h                                                   */
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

#pragma once

#include "../capabilities/soft_body_cap_state.h"
#include "jolt_object_3d.h"

#include "core/variant/typed_array.h"
#include "servers/physics_3d/physics_server_3d_enums.h"

#include <Jolt/Jolt.h>

#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>

class JoltArea3D;
class JoltSpace3D;
class PhysicsServer3DRenderingServerHandler;

class JoltSoftBody3D final : public JoltObject3D {
	HashSet<int> pinned_vertices;

	JoltSoftBodyCapState cap_state;
#ifdef TESTS_ENABLED
	bool test_skip_skin_call = false;
	bool test_without_volume_constraints = false;
	mutable uint64_t test_renderer_reads = 0;
	uint64_t test_body_creations = 0;
#endif

	LocalVector<int> mesh_to_physics;
	LocalVector<JoltArea3D *> areas;
	LocalVector<Vector3> normals;
	LocalVector<RID> exceptions;

	RID mesh;
	bool owner_separate_thread = false;
	bool cap_faulted = false;
	Transform3D reference_transform;
	struct BindingCache {
		RID mesh;
		PackedVector3Array vertices;
		PackedInt32Array indices;
		Transform3D frame;
	};
	BindingCache binding_cache;
	JPH::Ref<JPH::SoftBodySharedSettings> prepared_settings;
	LocalVector<const struct CapabilitySpec *> active_capabilities;

	bool _prepare_capability(JoltSoftBodyCapState &r_state, BindingCache &r_cache, JPH::Ref<JPH::SoftBodySharedSettings> &r_settings, LocalVector<int> &r_map, float p_mass, int p_precision, float p_stiffness, const HashSet<int> &p_pins, String *r_error) const;
	bool _thread_preflight(const JoltSoftBodyCapState &p_state, bool p_transition, String *r_error) const;
	bool _typed_preflight(float p_mass, int p_precision, float p_stiffness, const HashSet<int> &p_pins);
	void _invalidate_prepared() { prepared_settings = nullptr; }

	JPH::SoftBodyCreationSettings *jolt_settings = new JPH::SoftBodyCreationSettings();

	float mass = 1.0f;
	float pressure = 0.0f;
	float linear_damping = 0.01f;
	float stiffness_coefficient = 0.5f;
	float shrinking_factor = 0.0f;

	int simulation_precision = 5;

	virtual JPH::BroadPhaseLayer _get_broad_phase_layer() const override;
	virtual JPH::ObjectLayer _get_object_layer() const override;

	virtual void _space_changing() override;
	virtual void _space_changed() override;

	virtual void _add_to_space() override;

	bool _build_mesh_geometry(JPH::SoftBodySharedSettings &r_settings);
	JPH::SoftBodySharedSettings *_create_shared_settings();

	void _apply_environmental_forces(float p_step);

	void _update_mass();
	void _update_pressure();
	void _update_damping();
	void _update_simulation_precision();
	void _update_group_filter();

	void _try_rebuild();

	void _mesh_changed();
	void _simulation_precision_changed();
	void _mass_changed();
	void _pressure_changed();
	void _damping_changed();
	void _pins_changed();
	void _vertices_changed();
	void _exceptions_changed();
	void _motion_changed();
	void _transform_changed();
	void _areas_changed();

public:
#ifdef TESTS_ENABLED
	JoltSoftBodyCapProbe::BuildCounts get_cap_build_counts_for_tests() const {
		auto counts = cap_state.test_counts;
		counts.renderer_reads = test_renderer_reads;
		counts.body_creations = test_body_creations;
		return counts;
	}
	void record_solver_step_for_tests() { ++cap_state.test_counts.solver_steps; }
	void record_removal_during_traversal_for_tests() { ++cap_state.test_counts.removals_during_traversal; }
	void skip_next_skin_call_for_tests() { test_skip_skin_call = true; }
	void set_volume_control_for_tests() { test_without_volume_constraints = true; }
#endif
	JoltSoftBody3D();
	virtual ~JoltSoftBody3D() override;

	void add_collision_exception(const RID &p_excepted_body);
	void remove_collision_exception(const RID &p_excepted_body);
	bool has_collision_exception(const RID &p_excepted_body) const;

	const LocalVector<RID> &get_collision_exceptions() const { return exceptions; }

	void add_area(JoltArea3D *p_area);
	void remove_area(JoltArea3D *p_area);

	virtual bool can_interact_with(const JoltBody3D &p_other) const override;
	virtual bool can_interact_with(const JoltSoftBody3D &p_other) const override;
	virtual bool can_interact_with(const JoltArea3D &p_other) const override;

	virtual bool reports_contacts() const override { return false; }

	virtual Vector3 get_velocity_at_position(const Vector3 &p_position) const override;

	virtual void pre_step(float p_step) override;
	void remove_faulted_live_body();

	void set_mesh(const RID &p_mesh);
	void set_owner_separate_thread(bool p_value) { owner_separate_thread = p_value; }

	bool is_pickable() const { return pickable; }
	void set_pickable(bool p_enabled) { pickable = p_enabled; }

	bool is_sleeping() const;
	void set_is_sleeping(bool p_enabled);

	bool is_sleep_allowed() const;
	void set_is_sleep_allowed(bool p_enabled);

	void put_to_sleep() { set_is_sleeping(true); }
	void wake_up() { set_is_sleeping(false); }

	int get_simulation_precision() const { return simulation_precision; }
	void set_simulation_precision(int p_precision);

	float get_mass() const { return mass; }
	void set_mass(float p_mass);

	float get_stiffness_coefficient() const;
	void set_stiffness_coefficient(float p_coefficient);

	float get_shrinking_factor() const;
	void set_shrinking_factor(float p_shrinking_factor);

	float get_pressure() const { return pressure; }
	void set_pressure(float p_pressure);

	float get_linear_damping() const { return linear_damping; }
	void set_linear_damping(float p_damping);

	float get_drag() const;
	void set_drag(float p_drag);

	Variant get_state(PS3DE::BodyState p_state) const;
	void set_state(PS3DE::BodyState p_state, const Variant &p_value);

	Transform3D get_transform() const;
	void set_transform(const Transform3D &p_transform);

	AABB get_bounds() const;

	void update_rendering_server(PhysicsServer3DRenderingServerHandler *p_rendering_server_handler);

	Vector3 get_vertex_position(int p_index);
	void set_vertex_position(int p_index, const Vector3 &p_position);

	bool set_extra_property(const StringName &p_name, const Variant &p_value);
	Variant get_extra_property(const StringName &p_name) const;
	TypedArray<Dictionary> get_extra_property_list() const;

	void clear_extra_properties() { cap_state.clear(); }

	void pin_vertex(int p_index);
	void unpin_vertex(int p_index);

	void unpin_all_vertices();

	bool is_vertex_pinned(int p_index) const;

	void apply_vertex_impulse(int p_index, const Vector3 &p_impulse);
	void apply_vertex_force(int p_index, const Vector3 &p_force);
	void apply_central_impulse(const Vector3 &p_impulse);
	void apply_central_force(const Vector3 &p_force);
};
