#pragma once

#include "objects/jolt_physics_direct_body_state_3d.hpp"
#include "objects/jolt_shaped_object_impl_3d.hpp"

class JoltAreaImpl3D;
class JoltJointImpl3D;
class JoltSoftBodyImpl3D;

class JoltBodyImpl3D final : public JoltShapedObjectImpl3D {
public:
	using DampMode = PhysicsServer3D::BodyDampMode;

	struct Contact {
		float depth = 0.0f;

		int32_t shape_index = 0;

		int32_t collider_shape_index = 0;

		ObjectID collider_id;

		RID collider_rid;

		Vector3 normal;

		Vector3 position;

		Vector3 collider_position;

		Vector3 velocity;

		Vector3 collider_velocity;

		Vector3 impulse;
	};

	JoltBodyImpl3D();

	~JoltBodyImpl3D() override;

	void set_transform(Transform3D p_transform);

	Variant get_state(PhysicsServer3D::BodyState p_state) const;

	void set_state(PhysicsServer3D::BodyState p_state, const Variant& p_value);

	Variant get_param(PhysicsServer3D::BodyParameter p_param) const;

	void set_param(PhysicsServer3D::BodyParameter p_param, const Variant& p_value);

	bool has_state_sync_callback() const { return body_state_callback.is_valid(); }

	void set_state_sync_callback(const Callable& p_callback) { body_state_callback = p_callback; }

	bool has_custom_integration_callback() const { return custom_integration_callback.is_valid(); }

	void set_custom_integration_callback(const Callable& p_callback, const Variant& p_userdata) {
		custom_integration_callback = p_callback;
		custom_integration_userdata = p_userdata;
	}

	bool has_custom_integrator() const { return custom_integrator; }

	void set_custom_integrator(bool p_enabled);

	bool is_sleeping() const;

	void set_is_sleeping(bool p_enabled);

	void put_to_sleep() { set_is_sleeping(true); }

	void wake_up() { set_is_sleeping(false); }

	bool can_sleep() const;

	void set_can_sleep(bool p_enabled);

	Basis get_principal_inertia_axes() const;

	Vector3 get_inverse_inertia() const;

	Basis get_inverse_inertia_tensor() const;

	void set_linear_velocity(const Vector3& p_velocity);

	void set_angular_velocity(const Vector3& p_velocity);

	void set_axis_velocity(const Vector3& p_axis_velocity);

	Vector3 get_velocity_at_position(const Vector3& p_position) const override;

	bool has_custom_center_of_mass() const override { return custom_center_of_mass; }

	Vector3 get_center_of_mass_custom() const override { return center_of_mass_custom; }

	void set_center_of_mass_custom(const Vector3& p_center_of_mass);

	int32_t get_max_contacts_reported() const { return contacts.size(); }

	void set_max_contacts_reported(int32_t p_count);

	int32_t get_contact_count() const { return contact_count; }

	const Contact& get_contact(int32_t p_index) { return contacts[p_index]; }

	bool reports_contacts() const override { return !contacts.is_empty(); }

	bool reports_all_kinematic_contacts() const;

	void add_contact(
		const JoltBodyImpl3D* p_collider,
		float p_depth,
		int32_t p_shape_index,
		int32_t p_collider_shape_index,
		const Vector3& p_normal,
		const Vector3& p_position,
		const Vector3& p_collider_position,
		const Vector3& p_velocity,
		const Vector3& p_collider_velocity,
		const Vector3& p_impulse
	);

	void reset_mass_properties();

	void apply_force(const Vector3& p_force, const Vector3& p_position);

	void apply_central_force(const Vector3& p_force);

	void apply_impulse(const Vector3& p_impulse, const Vector3& p_position);

	void apply_central_impulse(const Vector3& p_impulse);

	void apply_torque(const Vector3& p_torque);

	void apply_torque_impulse(const Vector3& p_impulse);

	void add_constant_central_force(const Vector3& p_force);

	void add_constant_force(const Vector3& p_force, const Vector3& p_position);

	void add_constant_torque(const Vector3& p_torque);

	Vector3 get_constant_force() const;

	void set_constant_force(const Vector3& p_force);

	Vector3 get_constant_torque() const;

	void set_constant_torque(const Vector3& p_torque);

	Vector3 get_linear_surface_velocity() const { return linear_surface_velocity; }

	Vector3 get_angular_surface_velocity() const { return angular_surface_velocity; }

	void add_collision_exception(const RID& p_excepted_body);

	void remove_collision_exception(const RID& p_excepted_body);

	bool has_collision_exception(const RID& p_excepted_body) const;

	TypedArray<RID> get_collision_exceptions() const;

	void add_area(JoltAreaImpl3D* p_area);

	void remove_area(JoltAreaImpl3D* p_area);

	void add_joint(JoltJointImpl3D* p_joint);

	void remove_joint(JoltJointImpl3D* p_joint);

	void call_queries(JPH::Body& p_jolt_body);

	void pre_step(float p_step, JPH::Body& p_jolt_body) override;

	void move_kinematic(float p_step, JPH::Body& p_jolt_body);

	JoltPhysicsDirectBodyState3D* get_direct_state();

	PhysicsServer3D::BodyMode get_mode() const { return mode; }

	void set_mode(PhysicsServer3D::BodyMode p_mode);

	bool is_static() const { return mode == PhysicsServer3D::BODY_MODE_STATIC; }

	bool is_kinematic() const { return mode == PhysicsServer3D::BODY_MODE_KINEMATIC; }

	bool is_rigid_free() const { return mode == PhysicsServer3D::BODY_MODE_RIGID; }

	bool is_rigid_linear() const { return mode == PhysicsServer3D::BODY_MODE_RIGID_LINEAR; }

	bool is_rigid() const { return is_rigid_free() || is_rigid_linear(); }

	bool is_ccd_enabled() const;

	void set_ccd_enabled(bool p_enabled);

	float get_mass() const { return mass; }

	void set_mass(float p_mass);

	Vector3 get_inertia() const { return inertia; }

	void set_inertia(const Vector3& p_inertia);

	float get_bounce() const;

	void set_bounce(float p_bounce);

	float get_friction() const;

	void set_friction(float p_friction);

	float get_gravity_scale() const { return gravity_scale; };

	void set_gravity_scale(float p_scale);

	Vector3 get_gravity() const { return gravity; }

	float get_linear_damp() const { return linear_damp; }

	void set_linear_damp(float p_damp);

	float get_angular_damp() const { return angular_damp; }

	void set_angular_damp(float p_damp);

	float get_total_linear_damp() const { return total_linear_damp; }

	float get_total_angular_damp() const { return total_angular_damp; }

	float get_collision_priority() const { return collision_priority; }

	void set_collision_priority(float p_priority) { collision_priority = p_priority; }

	DampMode get_linear_damp_mode() const { return linear_damp_mode; }

	void set_linear_damp_mode(DampMode p_mode) { linear_damp_mode = p_mode; }

	DampMode get_angular_damp_mode() const { return angular_damp_mode; }

	void set_angular_damp_mode(DampMode p_mode) { angular_damp_mode = p_mode; }

	bool is_axis_locked(PhysicsServer3D::BodyAxis p_axis) const;

	void set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_enabled);

	bool are_axes_locked() const { return locked_axes != 0; }

	bool can_interact_with(const JoltBodyImpl3D& p_other) const override;

	bool can_interact_with(const JoltSoftBodyImpl3D& p_other) const override;

	bool can_interact_with(const JoltAreaImpl3D& p_other) const override;

private:
	JPH::BroadPhaseLayer _get_broad_phase_layer() const override;

	JPH::ObjectLayer _get_object_layer() const override;

	JPH::EMotionType _get_motion_type() const override;

	void _add_to_space() override;

	void _integrate_forces(float p_step, JPH::Body& p_jolt_body);

	void _pre_step_static(float p_step, JPH::Body& p_jolt_body);

	void _pre_step_rigid(float p_step, JPH::Body& p_jolt_body);

	void _pre_step_kinematic(float p_step, JPH::Body& p_jolt_body);

	JPH::EAllowedDOFs _calculate_allowed_dofs() const;

	JPH::MassProperties _calculate_mass_properties(const JPH::Shape& p_shape) const;

	JPH::MassProperties _calculate_mass_properties() const;

	void _update_mass_properties();

	void _update_gravity(JPH::Body& p_jolt_body);

	void _update_damp();

	void _update_kinematic_transform();

	void _update_group_filter();

	void _update_joint_constraints();

	void _update_possible_kinematic_contacts();

	void _destroy_joint_constraints();

	void _mode_changed();

	void _shapes_built() override;

	void _space_changing() override;

	void _space_changed() override;

	void _areas_changed();

	void _joints_changed();

	void _transform_changed();

	void _motion_changed();

	void _exceptions_changed();

	void _axis_lock_changed();

	void _contact_reporting_changed();

	LocalVector<RID> exceptions;

	LocalVector<Contact> contacts;

	LocalVector<JoltAreaImpl3D*> areas;

	LocalVector<JoltJointImpl3D*> joints;

	Variant custom_integration_userdata;

	Transform3D kinematic_transform;

	Vector3 inertia;

	Vector3 center_of_mass_custom;

	Vector3 constant_force;

	Vector3 constant_torque;

	Vector3 linear_surface_velocity;

	Vector3 angular_surface_velocity;

	Vector3 gravity;

	Callable body_state_callback;

	Callable custom_integration_callback;

	JoltPhysicsDirectBodyState3D* direct_state = nullptr;

	PhysicsServer3D::BodyMode mode = PhysicsServer3D::BODY_MODE_RIGID;

	DampMode linear_damp_mode = PhysicsServer3D::BODY_DAMP_MODE_COMBINE;

	DampMode angular_damp_mode = PhysicsServer3D::BODY_DAMP_MODE_COMBINE;

	float mass = 1.0f;

	float linear_damp = 0.0f;

	float angular_damp = 0.0f;

	float total_linear_damp = 0.0f;

	float total_angular_damp = 0.0f;

	float gravity_scale = 1.0f;

	float collision_priority = 1.0f;

	int32_t contact_count = 0;

	uint32_t locked_axes = 0;

	bool sync_state = false;

	bool custom_center_of_mass = false;

	bool custom_integrator = false;
};
