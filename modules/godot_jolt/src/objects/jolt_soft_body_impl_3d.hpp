#pragma once

#include "objects/jolt_object_impl_3d.hpp"

class JoltSpace3D;

class JoltSoftBodyImpl3D final : public JoltObjectImpl3D {
	struct Shared {
		LocalVector<int32_t> mesh_to_physics;

		JPH::Ref<JPH::SoftBodySharedSettings> settings = new JPH::SoftBodySharedSettings();

		int32_t ref_count = 1;
	};

public:
	JoltSoftBodyImpl3D();

	~JoltSoftBodyImpl3D() override;

	void add_collision_exception(const RID& p_excepted_body);

	void remove_collision_exception(const RID& p_excepted_body);

	bool has_collision_exception(const RID& p_excepted_body) const;

	TypedArray<RID> get_collision_exceptions() const;

	bool can_interact_with(const JoltBodyImpl3D& p_other) const override;

	bool can_interact_with(const JoltSoftBodyImpl3D& p_other) const override;

	bool can_interact_with(const JoltAreaImpl3D& p_other) const override;

	bool reports_contacts() const override { return false; }

	Vector3 get_velocity_at_position(const Vector3& p_position) const override;

	void set_mesh(const RID& p_mesh);

	bool is_pickable() const { return pickable; }

	void set_pickable(bool p_enabled) { pickable = p_enabled; }

	bool is_sleeping() const;

	void set_is_sleeping(bool p_enabled);

	void put_to_sleep() { set_is_sleeping(true); }

	void wake_up() { set_is_sleeping(false); }

	int32_t get_simulation_precision() const { return simulation_precision; }

	void set_simulation_precision(int32_t p_precision);

	float get_mass() const { return mass; }

	void set_mass(float p_mass);

	float get_stiffness_coefficient() const;

	void set_stiffness_coefficient(float p_coefficient);

	float get_pressure() const { return pressure; }

	void set_pressure(float p_pressure);

	float get_linear_damping() const { return linear_damping; }

	void set_linear_damping(float p_damping);

	float get_drag() const;

	void set_drag(float p_drag);

	Variant get_state(PhysicsServer3D::BodyState p_state) const;

	void set_state(PhysicsServer3D::BodyState p_state, const Variant& p_value);

	Transform3D get_transform() const;

	void set_transform(const Transform3D& p_transform);

	AABB get_bounds() const;

	void update_rendering_server(PhysicsServer3DRenderingServerHandler* p_rendering_server_handler);

	Vector3 get_vertex_position(int32_t p_index);

	void set_vertex_position(int32_t p_index, const Vector3& p_position);

	void pin_vertex(int32_t p_index);

	void unpin_vertex(int32_t p_index);

	void unpin_all_vertices();

	bool is_vertex_pinned(int32_t p_index) const;

	String to_string() const;

private:
	JPH::BroadPhaseLayer _get_broad_phase_layer() const override;

	JPH::ObjectLayer _get_object_layer() const override;

	void _space_changing() override;

	void _space_changed() override;

	void _add_to_space() override;

	bool _ref_shared_data();

	void _deref_shared_data();

	void _update_mass();

	void _update_pressure();

	void _update_damping();

	void _update_simulation_precision();

	void _update_group_filter();

	void _try_rebuild();

	void _mesh_changed();

	void _pressure_changed();

	void _damping_changed();

	void _pins_changed();

	void _exceptions_changed();

	inline static HashMap<RID, Shared> mesh_to_shared;

	HashSet<int32_t> pinned_vertices;

	LocalVector<RID> exceptions;

	LocalVector<Vector3> normals;

	const Shared* shared = nullptr;

	RID mesh;

	JPH::SoftBodyCreationSettings* jolt_settings = new JPH::SoftBodyCreationSettings();

	float mass = 0.0f;

	float pressure = 0.0f;

	float linear_damping = 0.01f;

	float stiffness_coefficient = 0.5f;

	int32_t simulation_precision = 5;
};
