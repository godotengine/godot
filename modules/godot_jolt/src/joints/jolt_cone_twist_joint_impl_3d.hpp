#pragma once

#include "joints/jolt_joint_impl_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"

class JoltConeTwistJointImpl3D final : public JoltJointImpl3D {
	using Parameter = PhysicsServer3D::ConeTwistJointParam;

	using JoltParameter = JoltPhysicsServer3D::ConeTwistJointParamJolt;

	using JoltFlag = JoltPhysicsServer3D::ConeTwistJointFlagJolt;

public:
	JoltConeTwistJointImpl3D(
		const JoltJointImpl3D& p_old_joint,
		JoltBodyImpl3D* p_body_a,
		JoltBodyImpl3D* p_body_b,
		const Transform3D& p_local_ref_a,
		const Transform3D& p_local_ref_b
	);

	PhysicsServer3D::JointType get_type() const override {
		return PhysicsServer3D::JOINT_TYPE_CONE_TWIST;
	}

	double get_param(PhysicsServer3D::ConeTwistJointParam p_param) const;

	void set_param(PhysicsServer3D::ConeTwistJointParam p_param, double p_value);

	double get_jolt_param(JoltParameter p_param) const;

	void set_jolt_param(JoltParameter p_param, double p_value);

	bool get_jolt_flag(JoltFlag p_flag) const;

	void set_jolt_flag(JoltFlag p_flag, bool p_enabled);

	float get_applied_force() const;

	float get_applied_torque() const;

	void rebuild() override;

private:
	JPH::Constraint* _build_swing_twist(
		JPH::Body* p_jolt_body_a,
		JPH::Body* p_jolt_body_b,
		const Transform3D& p_shifted_ref_a,
		const Transform3D& p_shifted_ref_b,
		float p_swing_limit_span,
		float p_twist_limit_span
	) const;

	void _update_swing_motor_state();

	void _update_twist_motor_state();

	void _update_motor_velocity();

	void _update_swing_motor_limit();

	void _update_twist_motor_limit();

	void _limits_changed();

	void _swing_motor_state_changed();

	void _twist_motor_state_changed();

	void _motor_velocity_changed();

	void _swing_motor_limit_changed();

	void _twist_motor_limit_changed();

	double swing_limit_span = 0.0;

	double twist_limit_span = 0.0;

	double swing_motor_target_speed_y = 0.0;

	double swing_motor_target_speed_z = 0.0;

	double twist_motor_target_speed = 0.0;

	double swing_motor_max_torque = 0.0;

	double twist_motor_max_torque = 0.0;

	bool swing_limit_enabled = true;

	bool twist_limit_enabled = true;

	bool swing_motor_enabled = false;

	bool twist_motor_enabled = false;
};
