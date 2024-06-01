#pragma once

#include "joints/jolt_joint_impl_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"

class JoltGeneric6DOFJointImpl3D final : public JoltJointImpl3D {
	using Axis = Vector3::Axis;

	using JoltAxis = JPH::SixDOFConstraintSettings::EAxis;

	using Param = PhysicsServer3D::G6DOFJointAxisParam;

	using JoltParam = JoltPhysicsServer3D::G6DOFJointAxisParamJolt;

	using Flag = PhysicsServer3D::G6DOFJointAxisFlag;

	using JoltFlag = JoltPhysicsServer3D::G6DOFJointAxisFlagJolt;

	enum {
		AXIS_LINEAR_X = JoltAxis::TranslationX,
		AXIS_LINEAR_Y = JoltAxis::TranslationY,
		AXIS_LINEAR_Z = JoltAxis::TranslationZ,
		AXIS_ANGULAR_X = JoltAxis::RotationX,
		AXIS_ANGULAR_Y = JoltAxis::RotationY,
		AXIS_ANGULAR_Z = JoltAxis::RotationZ,
		AXIS_COUNT = JoltAxis::Num,
		AXES_LINEAR = AXIS_LINEAR_X,
		AXES_ANGULAR = AXIS_ANGULAR_X
	};

public:
	JoltGeneric6DOFJointImpl3D(
		const JoltJointImpl3D& p_old_joint,
		JoltBodyImpl3D* p_body_a,
		JoltBodyImpl3D* p_body_b,
		const Transform3D& p_local_ref_a,
		const Transform3D& p_local_ref_b
	);

	PhysicsServer3D::JointType get_type() const override {
		return PhysicsServer3D::JOINT_TYPE_6DOF;
	}

	double get_param(Axis p_axis, Param p_param) const;

	void set_param(Axis p_axis, Param p_param, double p_value);

	bool get_flag(Axis p_axis, Flag p_flag) const;

	void set_flag(Axis p_axis, Flag p_flag, bool p_enabled);

	double get_jolt_param(Axis p_axis, JoltParam p_param) const;

	void set_jolt_param(Axis p_axis, JoltParam p_param, double p_value);

	bool get_jolt_flag(Axis p_axis, JoltFlag p_flag) const;

	void set_jolt_flag(Axis p_axis, JoltFlag p_flag, bool p_enabled);

	float get_applied_force() const;

	float get_applied_torque() const;

	void rebuild() override;

private:
	JPH::Constraint* _build_6dof(
		JPH::Body* p_jolt_body_a,
		JPH::Body* p_jolt_body_b,
		const Transform3D& p_shifted_ref_a,
		const Transform3D& p_shifted_ref_b
	) const;

	void _update_limit_spring_parameters(int32_t p_axis);

	void _update_motor_state(int32_t p_axis);

	void _update_motor_velocity(int32_t p_axis);

	void _update_motor_limit(int32_t p_axis);

	void _update_spring_parameters(int32_t p_axis);

	void _update_spring_equilibrium(int32_t p_axis);

	void _limits_changed();

	void _limit_spring_parameters_changed(int32_t p_axis);

	void _motor_state_changed(int32_t p_axis);

	void _motor_speed_changed(int32_t p_axis);

	void _motor_limit_changed(int32_t p_axis);

	void _spring_state_changed(int32_t p_axis);

	void _spring_parameters_changed(int32_t p_axis);

	void _spring_equilibrium_changed(int32_t p_axis);

	double limit_lower[AXIS_COUNT] = {};

	double limit_upper[AXIS_COUNT] = {};

	double limit_spring_frequency[AXIS_COUNT] = {};

	double limit_spring_damping[AXIS_COUNT] = {};

	double motor_speed[AXIS_COUNT] = {};

	double motor_limit[AXIS_COUNT] = {};

	double spring_stiffness[AXIS_COUNT] = {};

	double spring_frequency[AXIS_COUNT] = {};

	double spring_damping[AXIS_COUNT] = {};

	double spring_equilibrium[AXIS_COUNT] = {};

	bool limit_enabled[AXIS_COUNT] = {};

	bool limit_spring_enabled[AXIS_COUNT] = {};

	bool motor_enabled[AXIS_COUNT] = {};

	bool spring_enabled[AXIS_COUNT] = {};

	bool spring_use_frequency[AXIS_COUNT] = {};
};
