#pragma once

#include "joints/jolt_joint_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"

class JoltHingeJoint3D final : public JoltJoint3D {
	GDCLASS(JoltHingeJoint3D, JoltJoint3D)

public:
	enum Param {
		PARAM_LIMIT_UPPER = PhysicsServer3D::HINGE_JOINT_LIMIT_UPPER,
		PARAM_LIMIT_LOWER = PhysicsServer3D::HINGE_JOINT_LIMIT_LOWER,
		PARAM_LIMIT_SPRING_FREQUENCY = JoltPhysicsServer3D::HINGE_JOINT_LIMIT_SPRING_FREQUENCY,
		PARAM_LIMIT_SPRING_DAMPING = JoltPhysicsServer3D::HINGE_JOINT_LIMIT_SPRING_DAMPING,
		PARAM_MOTOR_TARGET_VELOCITY = PhysicsServer3D::HINGE_JOINT_MOTOR_TARGET_VELOCITY,
		PARAM_MOTOR_MAX_TORQUE = JoltPhysicsServer3D::HINGE_JOINT_MOTOR_MAX_TORQUE
	};

	enum Flag {
		FLAG_USE_LIMIT = PhysicsServer3D::HINGE_JOINT_FLAG_USE_LIMIT,
		FLAG_USE_LIMIT_SPRING = JoltPhysicsServer3D::HINGE_JOINT_FLAG_USE_LIMIT_SPRING,
		FLAG_ENABLE_MOTOR = PhysicsServer3D::HINGE_JOINT_FLAG_ENABLE_MOTOR,
	};

private:
	static void _bind_methods();

public:
	bool get_limit_enabled() const { return limit_enabled; }

	void set_limit_enabled(bool p_enabled);

	double get_limit_upper() const { return limit_upper; }

	void set_limit_upper(double p_value);

	double get_limit_lower() const { return limit_lower; }

	void set_limit_lower(double p_value);

	bool get_limit_spring_enabled() const { return limit_spring_enabled; }

	void set_limit_spring_enabled(bool p_enabled);

	double get_limit_spring_frequency() const { return limit_spring_frequency; }

	void set_limit_spring_frequency(double p_value);

	double get_limit_spring_damping() const { return limit_spring_damping; }

	void set_limit_spring_damping(double p_value);

	bool get_motor_enabled() const { return motor_enabled; }

	void set_motor_enabled(bool p_enabled);

	double get_motor_target_velocity() const { return motor_target_velocity; }

	void set_motor_target_velocity(double p_value);

	double get_motor_max_torque() const { return motor_max_torque; }

	void set_motor_max_torque(double p_value);

	float get_applied_force() const;

	float get_applied_torque() const;

private:
	void _configure(PhysicsBody3D* p_body_a, PhysicsBody3D* p_body_b) override;

	void _update_param(Param p_param);

	void _update_jolt_param(Param p_param);

	void _update_flag(Flag p_flag);

	void _update_jolt_flag(Flag p_flag);

	void _param_changed(Param p_param);

	void _flag_changed(Flag p_flag);

	double limit_upper = 0.0;

	double limit_lower = 0.0;

	double limit_spring_frequency = 0.0;

	double limit_spring_damping = 0.0;

	double motor_target_velocity = 0.0;

	double motor_max_torque = INFINITY;

	bool limit_enabled = false;

	bool limit_spring_enabled = false;

	bool motor_enabled = false;
};
