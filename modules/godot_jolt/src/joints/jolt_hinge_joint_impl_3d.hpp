#pragma once

#include "joints/jolt_joint_impl_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"

class JoltHingeJointImpl3D final : public JoltJointImpl3D {
	using Parameter = PhysicsServer3D::HingeJointParam;

	using JoltParameter = JoltPhysicsServer3D::HingeJointParamJolt;

	using Flag = PhysicsServer3D::HingeJointFlag;

	using JoltFlag = JoltPhysicsServer3D::HingeJointFlagJolt;

public:
	JoltHingeJointImpl3D(
		const JoltJointImpl3D& p_old_joint,
		JoltBodyImpl3D* p_body_a,
		JoltBodyImpl3D* p_body_b,
		const Transform3D& p_local_ref_a,
		const Transform3D& p_local_ref_b
	);

	PhysicsServer3D::JointType get_type() const override {
		return PhysicsServer3D::JOINT_TYPE_HINGE;
	}

	double get_param(Parameter p_param) const;

	void set_param(Parameter p_param, double p_value);

	double get_jolt_param(JoltParameter p_param) const;

	void set_jolt_param(JoltParameter p_param, double p_value);

	bool get_flag(Flag p_flag) const;

	void set_flag(Flag p_flag, bool p_enabled);

	bool get_jolt_flag(JoltFlag p_flag) const;

	void set_jolt_flag(JoltFlag p_flag, bool p_enabled);

	float get_applied_force() const;

	float get_applied_torque() const;

	void rebuild() override;

private:
	JPH::Constraint* _build_hinge(
		JPH::Body* p_jolt_body_a,
		JPH::Body* p_jolt_body_b,
		const Transform3D& p_shifted_ref_a,
		const Transform3D& p_shifted_ref_b,
		float p_limit
	) const;

	JPH::Constraint* _build_fixed(
		JPH::Body* p_jolt_body_a,
		JPH::Body* p_jolt_body_b,
		const Transform3D& p_shifted_ref_a,
		const Transform3D& p_shifted_ref_b
	) const;

	bool _is_sprung() const { return limit_spring_enabled && limit_spring_frequency > 0.0; }

	bool _is_fixed() const { return limits_enabled && limit_lower == limit_upper && !_is_sprung(); }

	void _update_motor_state();

	void _update_motor_velocity();

	void _update_motor_limit();

	void _limits_changed();

	void _limit_spring_changed();

	void _motor_state_changed();

	void _motor_speed_changed();

	void _motor_limit_changed();

	double limit_lower = 0.0;

	double limit_upper = 0.0;

	double limit_spring_frequency = 0.0;

	double limit_spring_damping = 0.0;

	double motor_target_speed = 0.0f;

	double motor_max_torque = 0.0;

	bool limits_enabled = false;

	bool limit_spring_enabled = false;

	bool motor_enabled = false;
};
