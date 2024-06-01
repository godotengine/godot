#pragma once

#include "joints/jolt_joint_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"

class JoltGeneric6DOFJoint3D final : public JoltJoint3D {
	GDCLASS(JoltGeneric6DOFJoint3D, JoltJoint3D)

public:
	// clang-format off

	using Axis = Vector3::Axis;

	enum Param {
		PARAM_LINEAR_LIMIT_UPPER = PhysicsServer3D::G6DOF_JOINT_LINEAR_UPPER_LIMIT,
		PARAM_LINEAR_LIMIT_LOWER = PhysicsServer3D::G6DOF_JOINT_LINEAR_LOWER_LIMIT,
		PARAM_LINEAR_LIMIT_SPRING_FREQUENCY = JoltPhysicsServer3D::G6DOF_JOINT_LINEAR_LIMIT_SPRING_FREQUENCY,
		PARAM_LINEAR_LIMIT_SPRING_DAMPING = JoltPhysicsServer3D::G6DOF_JOINT_LINEAR_LIMIT_SPRING_DAMPING,
		PARAM_LINEAR_MOTOR_TARGET_VELOCITY = PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_TARGET_VELOCITY,
		PARAM_LINEAR_MOTOR_MAX_FORCE = PhysicsServer3D::G6DOF_JOINT_LINEAR_MOTOR_FORCE_LIMIT,
		PARAM_LINEAR_SPRING_FREQUENCY = JoltPhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_FREQUENCY,
		PARAM_LINEAR_SPRING_DAMPING = JoltPhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_DAMPING,
		PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT = JoltPhysicsServer3D::G6DOF_JOINT_LINEAR_SPRING_EQUILIBRIUM_POINT,
		PARAM_ANGULAR_LIMIT_UPPER = PhysicsServer3D::G6DOF_JOINT_ANGULAR_UPPER_LIMIT,
		PARAM_ANGULAR_LIMIT_LOWER = PhysicsServer3D::G6DOF_JOINT_ANGULAR_LOWER_LIMIT,
		PARAM_ANGULAR_MOTOR_TARGET_VELOCITY = PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY,
		PARAM_ANGULAR_MOTOR_MAX_TORQUE = PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT,
		PARAM_ANGULAR_SPRING_FREQUENCY = JoltPhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_FREQUENCY,
		PARAM_ANGULAR_SPRING_DAMPING = JoltPhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_DAMPING,
		PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT = JoltPhysicsServer3D::G6DOF_JOINT_ANGULAR_SPRING_EQUILIBRIUM_POINT,
	};

	enum Flag {
		FLAG_ENABLE_LINEAR_LIMIT = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT,
		FLAG_ENABLE_LINEAR_LIMIT_SPRING = JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT_SPRING,
		FLAG_ENABLE_LINEAR_MOTOR = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR,
		FLAG_ENABLE_LINEAR_SPRING = JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING,
		FLAG_ENABLE_ANGULAR_LIMIT = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT,
		FLAG_ENABLE_ANGULAR_MOTOR = PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_MOTOR,
		FLAG_ENABLE_ANGULAR_SPRING = JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING,

		// HACK(mihe): These ones are left out of most places where the other flags are are used,
		// due to not being user-facing as far as this node is concerned, and are mostly just here
		// to simplify things a bit.
		FLAG_ENABLE_LINEAR_SPRING_FREQUENCY = JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING_FREQUENCY,
		FLAG_ENABLE_ANGULAR_SPRING_FREQUENCY = JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING_FREQUENCY
	};

	// clang-format on

private:
	static constexpr Axis AXIS_X = Axis::AXIS_X;

	static constexpr Axis AXIS_Y = Axis::AXIS_Y;

	static constexpr Axis AXIS_Z = Axis::AXIS_Z;

	static constexpr int32_t AXIS_COUNT = 3;

	static void _bind_methods();

public:
	JoltGeneric6DOFJoint3D();

	double get_param(Axis p_axis, Param p_param) const;

	void set_param(Axis p_axis, Param p_param, double p_value);

	double get_param_x(Param p_param) const { return get_param(AXIS_X, p_param); }

	void set_param_x(Param p_param, double p_value) { set_param(AXIS_X, p_param, p_value); }

	double get_param_y(Param p_param) const { return get_param(AXIS_Y, p_param); }

	void set_param_y(Param p_param, double p_value) { set_param(AXIS_Y, p_param, p_value); }

	double get_param_z(Param p_param) const { return get_param(AXIS_Z, p_param); }

	void set_param_z(Param p_param, double p_value) { set_param(AXIS_Z, p_param, p_value); }

	bool get_flag(Axis p_axis, Flag p_flag) const;

	void set_flag(Axis p_axis, Flag p_flag, bool p_enabled);

	bool get_flag_x(Flag p_flag) const { return get_flag(AXIS_X, p_flag); }

	void set_flag_x(Flag p_flag, bool p_enabled) { set_flag(AXIS_X, p_flag, p_enabled); }

	bool get_flag_y(Flag p_flag) const { return get_flag(AXIS_Y, p_flag); }

	void set_flag_y(Flag p_flag, bool p_enabled) { set_flag(AXIS_Y, p_flag, p_enabled); }

	bool get_flag_z(Flag p_flag) const { return get_flag(AXIS_Z, p_flag); }

	void set_flag_z(Flag p_flag, bool p_enabled) { set_flag(AXIS_Z, p_flag, p_enabled); }

	// clang-format off

	double get_linear_limit_x_upper() const { return get_param_x(PARAM_LINEAR_LIMIT_UPPER); }

	void set_linear_limit_x_upper(double p_value) { return set_param_x(PARAM_LINEAR_LIMIT_UPPER, p_value); }

	double get_linear_limit_x_lower() const { return get_param_x(PARAM_LINEAR_LIMIT_LOWER); }

	void set_linear_limit_x_lower(double p_value) { return set_param_x(PARAM_LINEAR_LIMIT_LOWER, p_value); }

	double get_linear_limit_y_upper() const { return get_param_y(PARAM_LINEAR_LIMIT_UPPER); }

	void set_linear_limit_y_upper(double p_value) { return set_param_y(PARAM_LINEAR_LIMIT_UPPER, p_value); }

	double get_linear_limit_y_lower() const { return get_param_y(PARAM_LINEAR_LIMIT_LOWER); }

	void set_linear_limit_y_lower(double p_value) { return set_param_y(PARAM_LINEAR_LIMIT_LOWER, p_value); }

	double get_linear_limit_z_upper() const { return get_param_z(PARAM_LINEAR_LIMIT_UPPER); }

	void set_linear_limit_z_upper(double p_value) { return set_param_z(PARAM_LINEAR_LIMIT_UPPER, p_value); }

	double get_linear_limit_z_lower() const { return get_param_z(PARAM_LINEAR_LIMIT_LOWER); }

	void set_linear_limit_z_lower(double p_value) { return set_param_z(PARAM_LINEAR_LIMIT_LOWER, p_value); }

	double get_linear_limit_spring_x_frequency() const { return get_param_x(PARAM_LINEAR_LIMIT_SPRING_FREQUENCY); }

	void set_linear_limit_spring_x_frequency(double p_value) { return set_param_x(PARAM_LINEAR_LIMIT_SPRING_FREQUENCY, p_value); }

	double get_linear_limit_spring_x_damping() const { return get_param_x(PARAM_LINEAR_LIMIT_SPRING_DAMPING); }

	void set_linear_limit_spring_x_damping(double p_value) { return set_param_x(PARAM_LINEAR_LIMIT_SPRING_DAMPING, p_value); }

	double get_linear_limit_spring_y_frequency() const { return get_param_y(PARAM_LINEAR_LIMIT_SPRING_FREQUENCY); }

	void set_linear_limit_spring_y_frequency(double p_value) { return set_param_y(PARAM_LINEAR_LIMIT_SPRING_FREQUENCY, p_value); }

	double get_linear_limit_spring_y_damping() const { return get_param_y(PARAM_LINEAR_LIMIT_SPRING_DAMPING); }

	void set_linear_limit_spring_y_damping(double p_value) { return set_param_y(PARAM_LINEAR_LIMIT_SPRING_DAMPING, p_value); }

	double get_linear_limit_spring_z_frequency() const { return get_param_z(PARAM_LINEAR_LIMIT_SPRING_FREQUENCY); }

	void set_linear_limit_spring_z_frequency(double p_value) { return set_param_z(PARAM_LINEAR_LIMIT_SPRING_FREQUENCY, p_value); }

	double get_linear_limit_spring_z_damping() const { return get_param_z(PARAM_LINEAR_LIMIT_SPRING_DAMPING); }

	void set_linear_limit_spring_z_damping(double p_value) { return set_param_z(PARAM_LINEAR_LIMIT_SPRING_DAMPING, p_value); }

	double get_linear_motor_x_target_velocity() const { return get_param_x(PARAM_LINEAR_MOTOR_TARGET_VELOCITY); }

	void set_linear_motor_x_target_velocity(double p_value) { return set_param_x(PARAM_LINEAR_MOTOR_TARGET_VELOCITY, p_value); }

	double get_linear_motor_x_max_force() const { return get_param_x(PARAM_LINEAR_MOTOR_MAX_FORCE); }

	void set_linear_motor_x_max_force(double p_value) { return set_param_x(PARAM_LINEAR_MOTOR_MAX_FORCE, p_value); }

	double get_linear_motor_y_target_velocity() const { return get_param_y(PARAM_LINEAR_MOTOR_TARGET_VELOCITY); }

	void set_linear_motor_y_target_velocity(double p_value) { return set_param_y(PARAM_LINEAR_MOTOR_TARGET_VELOCITY, p_value); }

	double get_linear_motor_y_max_force() const { return get_param_y(PARAM_LINEAR_MOTOR_MAX_FORCE); }

	void set_linear_motor_y_max_force(double p_value) { return set_param_y(PARAM_LINEAR_MOTOR_MAX_FORCE, p_value); }

	double get_linear_motor_z_target_velocity() const { return get_param_z(PARAM_LINEAR_MOTOR_TARGET_VELOCITY); }

	void set_linear_motor_z_target_velocity(double p_value) { return set_param_z(PARAM_LINEAR_MOTOR_TARGET_VELOCITY, p_value); }

	double get_linear_motor_z_max_force() const { return get_param_z(PARAM_LINEAR_MOTOR_MAX_FORCE); }

	void set_linear_motor_z_max_force(double p_value) { return set_param_z(PARAM_LINEAR_MOTOR_MAX_FORCE, p_value); }

	double get_linear_spring_x_frequency() const { return get_param_x(PARAM_LINEAR_SPRING_FREQUENCY); }

	void set_linear_spring_x_frequency(double p_value) { return set_param_x(PARAM_LINEAR_SPRING_FREQUENCY, p_value); }

	double get_linear_spring_x_damping() const { return get_param_x(PARAM_LINEAR_SPRING_DAMPING); }

	void set_linear_spring_x_damping(double p_value) { return set_param_x(PARAM_LINEAR_SPRING_DAMPING, p_value); }

	double get_linear_spring_x_equilibrium_point() const { return get_param_x(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT); }

	void set_linear_spring_x_equilibrium_point(double p_value) { return set_param_x(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT, p_value); }

	double get_linear_spring_y_frequency() const { return get_param_y(PARAM_LINEAR_SPRING_FREQUENCY); }

	void set_linear_spring_y_frequency(double p_value) { return set_param_y(PARAM_LINEAR_SPRING_FREQUENCY, p_value); }

	double get_linear_spring_y_damping() const { return get_param_y(PARAM_LINEAR_SPRING_DAMPING); }

	void set_linear_spring_y_damping(double p_value) { return set_param_y(PARAM_LINEAR_SPRING_DAMPING, p_value); }

	double get_linear_spring_y_equilibrium_point() const { return get_param_y(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT); }

	void set_linear_spring_y_equilibrium_point(double p_value) { return set_param_y(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT, p_value); }

	double get_linear_spring_z_frequency() const { return get_param_z(PARAM_LINEAR_SPRING_FREQUENCY); }

	void set_linear_spring_z_frequency(double p_value) { return set_param_z(PARAM_LINEAR_SPRING_FREQUENCY, p_value); }

	double get_linear_spring_z_damping() const { return get_param_z(PARAM_LINEAR_SPRING_DAMPING); }

	void set_linear_spring_z_damping(double p_value) { return set_param_z(PARAM_LINEAR_SPRING_DAMPING, p_value); }

	double get_linear_spring_z_equilibrium_point() const { return get_param_z(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT); }

	void set_linear_spring_z_equilibrium_point(double p_value) { return set_param_z(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT, p_value); }

	bool get_linear_limit_x_enabled() const { return get_flag_x(FLAG_ENABLE_LINEAR_LIMIT); }

	void set_linear_limit_x_enabled(bool p_enabled) { return set_flag_x(FLAG_ENABLE_LINEAR_LIMIT, p_enabled); }

	bool get_linear_limit_y_enabled() const { return get_flag_y(FLAG_ENABLE_LINEAR_LIMIT); }

	void set_linear_limit_y_enabled(bool p_enabled) { return set_flag_y(FLAG_ENABLE_LINEAR_LIMIT, p_enabled); }

	bool get_linear_limit_z_enabled() const { return get_flag_z(FLAG_ENABLE_LINEAR_LIMIT); }

	void set_linear_limit_z_enabled(bool p_enabled) { return set_flag_z(FLAG_ENABLE_LINEAR_LIMIT, p_enabled); }

	bool get_linear_limit_spring_x_enabled() const { return get_flag_x(FLAG_ENABLE_LINEAR_LIMIT_SPRING); }

	void set_linear_limit_spring_x_enabled(bool p_enabled) { return set_flag_x(FLAG_ENABLE_LINEAR_LIMIT_SPRING, p_enabled); }

	bool get_linear_limit_spring_y_enabled() const { return get_flag_y(FLAG_ENABLE_LINEAR_LIMIT_SPRING); }

	void set_linear_limit_spring_y_enabled(bool p_enabled) { return set_flag_y(FLAG_ENABLE_LINEAR_LIMIT_SPRING, p_enabled); }

	bool get_linear_limit_spring_z_enabled() const { return get_flag_z(FLAG_ENABLE_LINEAR_LIMIT_SPRING); }

	void set_linear_limit_spring_z_enabled(bool p_enabled) { return set_flag_z(FLAG_ENABLE_LINEAR_LIMIT_SPRING, p_enabled); }

	bool get_linear_motor_x_enabled() const { return get_flag_x(FLAG_ENABLE_LINEAR_MOTOR); }

	void set_linear_motor_x_enabled(bool p_enabled) { return set_flag_x(FLAG_ENABLE_LINEAR_MOTOR, p_enabled); }

	bool get_linear_motor_y_enabled() const { return get_flag_y(FLAG_ENABLE_LINEAR_MOTOR); }

	void set_linear_motor_y_enabled(bool p_enabled) { return set_flag_y(FLAG_ENABLE_LINEAR_MOTOR, p_enabled); }

	bool get_linear_motor_z_enabled() const { return get_flag_z(FLAG_ENABLE_LINEAR_MOTOR); }

	void set_linear_motor_z_enabled(bool p_enabled) { return set_flag_z(FLAG_ENABLE_LINEAR_MOTOR, p_enabled); }

	bool get_linear_spring_x_enabled() const { return get_flag_x(FLAG_ENABLE_LINEAR_SPRING); }

	void set_linear_spring_x_enabled(bool p_enabled) { return set_flag_x(FLAG_ENABLE_LINEAR_SPRING, p_enabled); }

	bool get_linear_spring_y_enabled() const { return get_flag_y(FLAG_ENABLE_LINEAR_SPRING); }

	void set_linear_spring_y_enabled(bool p_enabled) { return set_flag_y(FLAG_ENABLE_LINEAR_SPRING, p_enabled); }

	bool get_linear_spring_z_enabled() const { return get_flag_z(FLAG_ENABLE_LINEAR_SPRING); }

	void set_linear_spring_z_enabled(bool p_enabled) { return set_flag_z(FLAG_ENABLE_LINEAR_SPRING, p_enabled); }

	double get_angular_limit_x_upper() const { return get_param_x(PARAM_ANGULAR_LIMIT_UPPER); }

	void set_angular_limit_x_upper(double p_value) { return set_param_x(PARAM_ANGULAR_LIMIT_UPPER, p_value); }

	double get_angular_limit_x_lower() const { return get_param_x(PARAM_ANGULAR_LIMIT_LOWER); }

	void set_angular_limit_x_lower(double p_value) { return set_param_x(PARAM_ANGULAR_LIMIT_LOWER, p_value); }

	double get_angular_limit_y_upper() const { return get_param_y(PARAM_ANGULAR_LIMIT_UPPER); }

	void set_angular_limit_y_upper(double p_value) { return set_param_y(PARAM_ANGULAR_LIMIT_UPPER, p_value); }

	double get_angular_limit_y_lower() const { return get_param_y(PARAM_ANGULAR_LIMIT_LOWER); }

	void set_angular_limit_y_lower(double p_value) { return set_param_y(PARAM_ANGULAR_LIMIT_LOWER, p_value); }

	double get_angular_limit_z_upper() const { return get_param_z(PARAM_ANGULAR_LIMIT_UPPER); }

	void set_angular_limit_z_upper(double p_value) { return set_param_z(PARAM_ANGULAR_LIMIT_UPPER, p_value); }

	double get_angular_limit_z_lower() const { return get_param_z(PARAM_ANGULAR_LIMIT_LOWER); }

	void set_angular_limit_z_lower(double p_value) { return set_param_z(PARAM_ANGULAR_LIMIT_LOWER, p_value); }

	double get_angular_motor_x_target_velocity() const { return get_param_x(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY); }

	void set_angular_motor_x_target_velocity(double p_value) { return set_param_x(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, p_value); }

	double get_angular_motor_x_max_torque() const { return get_param_x(PARAM_ANGULAR_MOTOR_MAX_TORQUE); }

	void set_angular_motor_x_max_torque(double p_value) { return set_param_x(PARAM_ANGULAR_MOTOR_MAX_TORQUE, p_value); }

	double get_angular_motor_y_target_velocity() const { return get_param_y(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY); }

	void set_angular_motor_y_target_velocity(double p_value) { return set_param_y(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, p_value); }

	double get_angular_motor_y_max_torque() const { return get_param_y(PARAM_ANGULAR_MOTOR_MAX_TORQUE); }

	void set_angular_motor_y_max_torque(double p_value) { return set_param_y(PARAM_ANGULAR_MOTOR_MAX_TORQUE, p_value); }

	double get_angular_motor_z_target_velocity() const { return get_param_z(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY); }

	void set_angular_motor_z_target_velocity(double p_value) { return set_param_z(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, p_value); }

	double get_angular_motor_z_max_torque() const { return get_param_z(PARAM_ANGULAR_MOTOR_MAX_TORQUE); }

	void set_angular_motor_z_max_torque(double p_value) { return set_param_z(PARAM_ANGULAR_MOTOR_MAX_TORQUE, p_value); }

	double get_angular_spring_x_frequency() const { return get_param_x(PARAM_ANGULAR_SPRING_FREQUENCY); }

	void set_angular_spring_x_frequency(double p_value) { return set_param_x(PARAM_ANGULAR_SPRING_FREQUENCY, p_value); }

	double get_angular_spring_x_damping() const { return get_param_x(PARAM_ANGULAR_SPRING_DAMPING); }

	void set_angular_spring_x_damping(double p_value) { return set_param_x(PARAM_ANGULAR_SPRING_DAMPING, p_value); }

	double get_angular_spring_x_equilibrium_point() const { return get_param_x(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT); }

	void set_angular_spring_x_equilibrium_point(double p_value) { return set_param_x(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, p_value); }

	double get_angular_spring_y_frequency() const { return get_param_y(PARAM_ANGULAR_SPRING_FREQUENCY); }

	void set_angular_spring_y_frequency(double p_value) { return set_param_y(PARAM_ANGULAR_SPRING_FREQUENCY, p_value); }

	double get_angular_spring_y_damping() const { return get_param_y(PARAM_ANGULAR_SPRING_DAMPING); }

	void set_angular_spring_y_damping(double p_value) { return set_param_y(PARAM_ANGULAR_SPRING_DAMPING, p_value); }

	double get_angular_spring_y_equilibrium_point() const { return get_param_y(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT); }

	void set_angular_spring_y_equilibrium_point(double p_value) { return set_param_y(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, p_value); }

	double get_angular_spring_z_frequency() const { return get_param_z(PARAM_ANGULAR_SPRING_FREQUENCY); }

	void set_angular_spring_z_frequency(double p_value) { return set_param_z(PARAM_ANGULAR_SPRING_FREQUENCY, p_value); }

	double get_angular_spring_z_damping() const { return get_param_z(PARAM_ANGULAR_SPRING_DAMPING); }

	void set_angular_spring_z_damping(double p_value) { return set_param_z(PARAM_ANGULAR_SPRING_DAMPING, p_value); }

	double get_angular_spring_z_equilibrium_point() const { return get_param_z(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT); }

	void set_angular_spring_z_equilibrium_point(double p_value) { return set_param_z(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, p_value); }

	bool get_angular_limit_x_enabled() const { return get_flag_x(FLAG_ENABLE_ANGULAR_LIMIT); }

	void set_angular_limit_x_enabled(bool p_enabled) { return set_flag_x(FLAG_ENABLE_ANGULAR_LIMIT, p_enabled); }

	bool get_angular_limit_y_enabled() const { return get_flag_y(FLAG_ENABLE_ANGULAR_LIMIT); }

	void set_angular_limit_y_enabled(bool p_enabled) { return set_flag_y(FLAG_ENABLE_ANGULAR_LIMIT, p_enabled); }

	bool get_angular_limit_z_enabled() const { return get_flag_z(FLAG_ENABLE_ANGULAR_LIMIT); }

	void set_angular_limit_z_enabled(bool p_enabled) { return set_flag_z(FLAG_ENABLE_ANGULAR_LIMIT, p_enabled); }

	bool get_angular_motor_x_enabled() const { return get_flag_x(FLAG_ENABLE_ANGULAR_MOTOR); }

	void set_angular_motor_x_enabled(bool p_enabled) { return set_flag_x(FLAG_ENABLE_ANGULAR_MOTOR, p_enabled); }

	bool get_angular_motor_y_enabled() const { return get_flag_y(FLAG_ENABLE_ANGULAR_MOTOR); }

	void set_angular_motor_y_enabled(bool p_enabled) { return set_flag_y(FLAG_ENABLE_ANGULAR_MOTOR, p_enabled); }

	bool get_angular_motor_z_enabled() const { return get_flag_z(FLAG_ENABLE_ANGULAR_MOTOR); }

	void set_angular_motor_z_enabled(bool p_enabled) { return set_flag_z(FLAG_ENABLE_ANGULAR_MOTOR, p_enabled); }

	bool get_angular_spring_x_enabled() const { return get_flag_x(FLAG_ENABLE_ANGULAR_SPRING); }

	void set_angular_spring_x_enabled(bool p_enabled) { return set_flag_x(FLAG_ENABLE_ANGULAR_SPRING, p_enabled); }

	bool get_angular_spring_y_enabled() const { return get_flag_y(FLAG_ENABLE_ANGULAR_SPRING); }

	void set_angular_spring_y_enabled(bool p_enabled) { return set_flag_y(FLAG_ENABLE_ANGULAR_SPRING, p_enabled); }

	bool get_angular_spring_z_enabled() const { return get_flag_z(FLAG_ENABLE_ANGULAR_SPRING); }

	void set_angular_spring_z_enabled(bool p_enabled) { return set_flag_z(FLAG_ENABLE_ANGULAR_SPRING, p_enabled); }

	// clang-format on

	float get_applied_force() const;

	float get_applied_torque() const;

private:
	void _configure(PhysicsBody3D* p_body_a, PhysicsBody3D* p_body_b) override;

	const double* _get_param_ptr(Axis p_axis, Param p_param) const;

	double* _get_param_ptr(Axis p_axis, Param p_param);

	const bool* _get_flag_ptr(Axis p_axis, Flag p_flag) const;

	bool* _get_flag_ptr(Axis p_axis, Flag p_flag);

	void _update_param(Axis p_axis, Param p_param, double p_value);

	void _update_param(Axis p_axis, Param p_param);

	void _update_jolt_param(Axis p_axis, Param p_param, double p_value);

	void _update_jolt_param(Axis p_axis, Param p_param);

	void _update_flag(Axis p_axis, Flag p_flag, bool p_enabled);

	void _update_flag(Axis p_axis, Flag p_flag);

	void _update_jolt_flag(Axis p_axis, Flag p_flag, bool p_enabled);

	void _update_jolt_flag(Axis p_axis, Flag p_flag);

	void _param_changed(Axis p_axis, Param p_param);

	void _flag_changed(Axis p_axis, Flag p_flag);

	double linear_limit_upper[AXIS_COUNT] = {};

	double linear_limit_lower[AXIS_COUNT] = {};

	double linear_limit_spring_frequency[AXIS_COUNT] = {};

	double linear_limit_spring_damping[AXIS_COUNT] = {};

	double linear_motor_target_velocity[AXIS_COUNT] = {};

	double linear_motor_max_force[AXIS_COUNT] = {};

	double linear_spring_frequency[AXIS_COUNT] = {};

	double linear_spring_damping[AXIS_COUNT] = {};

	double linear_spring_equilibrium_point[AXIS_COUNT] = {};

	double angular_limit_upper[AXIS_COUNT] = {};

	double angular_limit_lower[AXIS_COUNT] = {};

	double angular_motor_target_velocity[AXIS_COUNT] = {};

	double angular_motor_max_torque[AXIS_COUNT] = {};

	double angular_spring_frequency[AXIS_COUNT] = {};

	double angular_spring_damping[AXIS_COUNT] = {};

	double angular_spring_equilibrium_point[AXIS_COUNT] = {};

	bool linear_limit_enabled[AXIS_COUNT] = {};

	bool linear_limit_spring_enabled[AXIS_COUNT] = {};

	bool linear_motor_enabled[AXIS_COUNT] = {};

	bool linear_spring_enabled[AXIS_COUNT] = {};

	bool angular_limit_enabled[AXIS_COUNT] = {};

	bool angular_motor_enabled[AXIS_COUNT] = {};

	bool angular_spring_enabled[AXIS_COUNT] = {};
};

VARIANT_ENUM_CAST(JoltGeneric6DOFJoint3D::Param)
VARIANT_ENUM_CAST(JoltGeneric6DOFJoint3D::Flag)
