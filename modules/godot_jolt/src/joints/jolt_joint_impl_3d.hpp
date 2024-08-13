#pragma once
#include "../common.h"
#include "misc/error_macros.hpp"
class JoltBodyImpl3D;
class JoltSpace3D;

class JoltJointImpl3D {
public:
	JoltJointImpl3D() = default;

	JoltJointImpl3D(
		const JoltJointImpl3D& p_old_joint,
		JoltBodyImpl3D* p_body_a,
		JoltBodyImpl3D* p_body_b,
		const Transform3D& p_local_ref_a,
		const Transform3D& p_local_ref_b
	);

	virtual ~JoltJointImpl3D();

	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_TYPE_MAX; }

	RID get_rid() const { return rid; }

	void set_rid(const RID& p_rid) { rid = p_rid; }

	JoltSpace3D* get_space() const;

	JPH::Constraint* get_jolt_ref() const { return jolt_ref; }

	bool is_enabled() const { return enabled; }

	void set_enabled(bool p_enabled);

	int32_t get_solver_priority() const;

	void set_solver_priority(int32_t p_priority);

	int32_t get_solver_velocity_iterations() const { return velocity_iterations; }

	void set_solver_velocity_iterations(int32_t p_iterations);

	int32_t get_solver_position_iterations() const { return position_iterations; }

	void set_solver_position_iterations(int32_t p_iterations);

	bool is_collision_disabled() const { return collision_disabled; }

	void set_collision_disabled(bool p_disabled);

	void destroy();

	virtual void rebuild() { }

protected:
	void _shift_reference_frames(
		const Vector3& p_linear_shift,
		const Vector3& p_angular_shift,
		Transform3D& p_shifted_ref_a,
		Transform3D& p_shifted_ref_b
	);

	void _wake_up_bodies();

	void _update_enabled();

	void _update_iterations();

	void _enabled_changed();

	void _iterations_changed();

	String _bodies_to_string() const;

	bool enabled = true;

	bool collision_disabled = false;

	int32_t velocity_iterations = 0;

	int32_t position_iterations = 0;

	JPH::Ref<JPH::Constraint> jolt_ref;

	JoltBodyImpl3D* body_a = nullptr;

	JoltBodyImpl3D* body_b = nullptr;

	RID rid;

	Transform3D local_ref_a;

	Transform3D local_ref_b;
};
