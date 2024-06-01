#include "jolt_generic_6dof_joint.hpp"

namespace {

using ServerAxis = Vector3::Axis;
using ServerParam = PhysicsServer3D::G6DOFJointAxisParam;
using ServerParamJolt = JoltPhysicsServer3D::G6DOFJointAxisParamJolt;
using ServerFlag = JoltPhysicsServer3D::G6DOFJointAxisFlag;
using ServerFlagJolt = JoltPhysicsServer3D::G6DOFJointAxisFlagJolt;

} // namespace

void JoltGeneric6DOFJoint3D::_bind_methods() {
	BIND_METHOD(JoltGeneric6DOFJoint3D, get_param_x, "param");
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_param_x, "param", "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_param_y, "param");
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_param_y, "param", "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_param_z, "param");
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_param_z, "param", "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_flag_x, "flag");
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_flag_x, "flag", "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_flag_y, "flag");
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_flag_y, "flag", "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_flag_z, "flag");
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_flag_z, "flag", "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_x_upper);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_x_upper, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_x_lower);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_x_lower, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_y_upper);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_y_upper, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_y_lower);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_y_lower, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_z_upper);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_z_upper, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_z_lower);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_z_lower, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_x_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_x_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_x_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_x_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_y_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_y_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_y_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_y_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_z_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_z_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_z_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_z_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_x_target_velocity);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_x_target_velocity, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_x_max_force);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_x_max_force, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_y_target_velocity);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_y_target_velocity, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_y_max_force);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_y_max_force, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_z_target_velocity);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_z_target_velocity, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_z_max_force);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_z_max_force, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_x_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_x_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_x_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_x_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_x_equilibrium_point);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_x_equilibrium_point, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_y_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_y_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_y_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_y_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_y_equilibrium_point);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_y_equilibrium_point, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_z_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_z_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_z_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_z_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_z_equilibrium_point);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_z_equilibrium_point, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_x_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_x_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_y_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_y_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_z_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_z_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_x_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_x_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_y_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_y_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_limit_spring_z_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_limit_spring_z_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_x_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_x_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_y_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_y_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_motor_z_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_motor_z_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_x_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_x_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_y_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_y_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_linear_spring_z_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_linear_spring_z_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_x_upper);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_x_upper, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_x_lower);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_x_lower, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_y_upper);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_y_upper, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_y_lower);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_y_lower, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_z_upper);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_z_upper, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_z_lower);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_z_lower, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_x_target_velocity);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_x_target_velocity, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_x_max_torque);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_x_max_torque, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_y_target_velocity);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_y_target_velocity, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_y_max_torque);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_y_max_torque, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_z_target_velocity);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_z_target_velocity, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_z_max_torque);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_z_max_torque, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_x_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_x_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_x_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_x_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_x_equilibrium_point);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_x_equilibrium_point, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_y_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_y_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_y_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_y_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_y_equilibrium_point);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_y_equilibrium_point, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_z_frequency);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_z_frequency, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_z_damping);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_z_damping, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_z_equilibrium_point);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_z_equilibrium_point, "value");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_x_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_x_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_y_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_y_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_limit_z_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_limit_z_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_x_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_x_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_y_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_y_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_motor_z_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_motor_z_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_x_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_x_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_y_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_y_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_angular_spring_z_enabled);
	BIND_METHOD(JoltGeneric6DOFJoint3D, set_angular_spring_z_enabled, "enabled");

	BIND_METHOD(JoltGeneric6DOFJoint3D, get_applied_force);
	BIND_METHOD(JoltGeneric6DOFJoint3D, get_applied_torque);

	ADD_GROUP("Linear Limit", "linear_limit_");

	BIND_SUBPROPERTY("linear_limit_x", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_limit_x", "upper", Variant::FLOAT, "suffix:m");
	BIND_SUBPROPERTY("linear_limit_x", "lower", Variant::FLOAT, "suffix:m");

	BIND_SUBPROPERTY("linear_limit_y", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_limit_y", "upper", Variant::FLOAT, "suffix:m");
	BIND_SUBPROPERTY("linear_limit_y", "lower", Variant::FLOAT, "suffix:m");

	BIND_SUBPROPERTY("linear_limit_z", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_limit_z", "upper", Variant::FLOAT, "suffix:m");
	BIND_SUBPROPERTY("linear_limit_z", "lower", Variant::FLOAT, "suffix:m");

	ADD_GROUP("Linear Limit Spring", "linear_limit_spring_");

	BIND_SUBPROPERTY("linear_limit_spring_x", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_limit_spring_x", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("linear_limit_spring_x", "damping", Variant::FLOAT);

	BIND_SUBPROPERTY("linear_limit_spring_y", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_limit_spring_y", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("linear_limit_spring_y", "damping", Variant::FLOAT);

	BIND_SUBPROPERTY("linear_limit_spring_z", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_limit_spring_z", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("linear_limit_spring_z", "damping", Variant::FLOAT);

	ADD_GROUP("Linear Motor", "linear_motor_");

	BIND_SUBPROPERTY("linear_motor_x", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_motor_x", "target_velocity", Variant::FLOAT, "suffix:m/s");
	BIND_SUBPROPERTY("linear_motor_x", "max_force", Variant::FLOAT, U"suffix:kg⋅m/s² (N)");

	BIND_SUBPROPERTY("linear_motor_y", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_motor_y", "target_velocity", Variant::FLOAT, "suffix:m/s");
	BIND_SUBPROPERTY("linear_motor_y", "max_force", Variant::FLOAT, U"suffix:kg⋅m/s² (N)");

	BIND_SUBPROPERTY("linear_motor_z", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_motor_z", "target_velocity", Variant::FLOAT, "suffix:m/s");
	BIND_SUBPROPERTY("linear_motor_z", "max_force", Variant::FLOAT, U"suffix:kg⋅m/s² (N)");

	ADD_GROUP("Linear Spring", "linear_spring_");

	BIND_SUBPROPERTY("linear_spring_x", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_spring_x", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("linear_spring_x", "damping", Variant::FLOAT);
	BIND_SUBPROPERTY("linear_spring_x", "equilibrium_point", Variant::FLOAT, "suffix:m");

	BIND_SUBPROPERTY("linear_spring_y", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_spring_y", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("linear_spring_y", "damping", Variant::FLOAT);
	BIND_SUBPROPERTY("linear_spring_y", "equilibrium_point", Variant::FLOAT, "suffix:m");

	BIND_SUBPROPERTY("linear_spring_z", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("linear_spring_z", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("linear_spring_z", "damping", Variant::FLOAT);
	BIND_SUBPROPERTY("linear_spring_z", "equilibrium_point", Variant::FLOAT, "suffix:m");

	ADD_GROUP("Angular Limit", "angular_limit_");

	BIND_SUBPROPERTY("angular_limit_x", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_limit_x", "upper", Variant::FLOAT, "radians");
	BIND_SUBPROPERTY("angular_limit_x", "lower", Variant::FLOAT, "radians");

	BIND_SUBPROPERTY("angular_limit_y", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_limit_y", "upper", Variant::FLOAT, "radians");
	BIND_SUBPROPERTY("angular_limit_y", "lower", Variant::FLOAT, "radians");

	BIND_SUBPROPERTY("angular_limit_z", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_limit_z", "upper", Variant::FLOAT, "radians");
	BIND_SUBPROPERTY("angular_limit_z", "lower", Variant::FLOAT, "radians");

	ADD_GROUP("Angular Motor", "angular_motor_");

	BIND_SUBPROPERTY("angular_motor_x", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_motor_x", "target_velocity", Variant::FLOAT, U"radians,suffix:°/s");
	BIND_SUBPROPERTY("angular_motor_x", "max_torque", Variant::FLOAT, U"suffix:kg⋅m²/s² (Nm)");

	BIND_SUBPROPERTY("angular_motor_y", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_motor_y", "target_velocity", Variant::FLOAT, U"radians,suffix:°/s");
	BIND_SUBPROPERTY("angular_motor_y", "max_torque", Variant::FLOAT, U"suffix:kg⋅m²/s² (Nm)");

	BIND_SUBPROPERTY("angular_motor_z", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_motor_z", "target_velocity", Variant::FLOAT, U"radians,suffix:°/s");
	BIND_SUBPROPERTY("angular_motor_z", "max_torque", Variant::FLOAT, U"suffix:kg⋅m²/s² (Nm)");

	ADD_GROUP("Angular Spring", "angular_spring_");

	BIND_SUBPROPERTY("angular_spring_x", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_spring_x", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("angular_spring_x", "damping", Variant::FLOAT);
	BIND_SUBPROPERTY("angular_spring_x", "equilibrium_point", Variant::FLOAT, "radians");

	BIND_SUBPROPERTY("angular_spring_y", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_spring_y", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("angular_spring_y", "damping", Variant::FLOAT);
	BIND_SUBPROPERTY("angular_spring_y", "equilibrium_point", Variant::FLOAT, "radians");

	BIND_SUBPROPERTY("angular_spring_z", "enabled", Variant::BOOL);
	BIND_SUBPROPERTY("angular_spring_z", "frequency", Variant::FLOAT, "suffix:hz");
	BIND_SUBPROPERTY("angular_spring_z", "damping", Variant::FLOAT);
	BIND_SUBPROPERTY("angular_spring_z", "equilibrium_point", Variant::FLOAT, "radians");

	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_SPRING_FREQUENCY);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_LIMIT_SPRING_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_MOTOR_MAX_FORCE);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_SPRING_FREQUENCY);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_SPRING_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_MOTOR_MAX_TORQUE);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_SPRING_FREQUENCY);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_SPRING_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT);

	BIND_ENUM_CONSTANT(FLAG_ENABLE_LINEAR_LIMIT);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_LINEAR_LIMIT_SPRING);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_LINEAR_MOTOR);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_LINEAR_SPRING);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_ANGULAR_LIMIT);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_ANGULAR_MOTOR);
	BIND_ENUM_CONSTANT(FLAG_ENABLE_ANGULAR_SPRING);
}

JoltGeneric6DOFJoint3D::JoltGeneric6DOFJoint3D() {
	std::fill_n(linear_limit_spring_frequency, AXIS_COUNT, 0.0);
	std::fill_n(linear_limit_spring_damping, AXIS_COUNT, 0.0);
	std::fill_n(linear_motor_max_force, AXIS_COUNT, INFINITY);
	std::fill_n(linear_spring_frequency, AXIS_COUNT, 0.0);
	std::fill_n(linear_spring_damping, AXIS_COUNT, 0.0);
	std::fill_n(angular_motor_max_torque, AXIS_COUNT, INFINITY);
	std::fill_n(angular_spring_frequency, AXIS_COUNT, 0.0);
	std::fill_n(angular_spring_damping, AXIS_COUNT, 0.0);
	std::fill_n(linear_limit_enabled, AXIS_COUNT, true);
	std::fill_n(angular_limit_enabled, AXIS_COUNT, true);
}

double JoltGeneric6DOFJoint3D::get_param(Axis p_axis, Param p_param) const {
	const double* value = _get_param_ptr(p_axis, p_param);
	QUIET_FAIL_NULL_D(value);

	return *value;
}

void JoltGeneric6DOFJoint3D::set_param(Axis p_axis, Param p_param, double p_value) {
	double* value = _get_param_ptr(p_axis, p_param);
	QUIET_FAIL_NULL(value);

	if (*value == p_value) {
		return;
	}

	*value = p_value;

	_param_changed(p_axis, p_param);
}

bool JoltGeneric6DOFJoint3D::get_flag(Axis p_axis, Flag p_flag) const {
	const bool* value = _get_flag_ptr(p_axis, p_flag);
	QUIET_FAIL_NULL_D(value);

	return *value;
}

void JoltGeneric6DOFJoint3D::set_flag(Axis p_axis, Flag p_flag, bool p_enabled) {
	bool* value = _get_flag_ptr(p_axis, p_flag);
	QUIET_FAIL_NULL(value);

	if (*value == p_enabled) {
		return;
	}

	*value = p_enabled;

	_flag_changed(p_axis, p_flag);
}

float JoltGeneric6DOFJoint3D::get_applied_force() const {
	JoltPhysicsServer3D* server = _get_jolt_physics_server();
	QUIET_FAIL_NULL_D(server);

	return server->generic_6dof_joint_get_applied_force(rid);
}

float JoltGeneric6DOFJoint3D::get_applied_torque() const {
	JoltPhysicsServer3D* server = _get_jolt_physics_server();
	QUIET_FAIL_NULL_D(server);

	return server->generic_6dof_joint_get_applied_torque(rid);
}

void JoltGeneric6DOFJoint3D::_configure(PhysicsBody3D* p_body_a, PhysicsBody3D* p_body_b) {
	PhysicsServer3D* server = _get_physics_server();
	ERR_FAIL_NULL(server);

	server->joint_make_generic_6dof(
		rid,
		p_body_a->get_rid(),
		_get_body_local_transform(*p_body_a),
		p_body_b != nullptr ? p_body_b->get_rid() : RID(),
		p_body_b != nullptr ? _get_body_local_transform(*p_body_b)
							: get_global_transform().orthonormalized()
	);

	for (int32_t i = 0; i < AXIS_COUNT; ++i) {
		const auto axis = (Axis)i;

		_update_param(axis, PARAM_LINEAR_LIMIT_UPPER);
		_update_param(axis, PARAM_LINEAR_LIMIT_LOWER);
		_update_param(axis, PARAM_LINEAR_MOTOR_TARGET_VELOCITY);
		_update_param(axis, PARAM_LINEAR_MOTOR_MAX_FORCE);
		_update_param(axis, PARAM_LINEAR_SPRING_DAMPING);
		_update_param(axis, PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT);
		_update_param(axis, PARAM_ANGULAR_LIMIT_UPPER);
		_update_param(axis, PARAM_ANGULAR_LIMIT_LOWER);
		_update_param(axis, PARAM_ANGULAR_MOTOR_TARGET_VELOCITY);
		_update_param(axis, PARAM_ANGULAR_MOTOR_MAX_TORQUE);
		_update_param(axis, PARAM_ANGULAR_SPRING_DAMPING);
		_update_param(axis, PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT);

		_update_jolt_param(axis, PARAM_LINEAR_LIMIT_SPRING_FREQUENCY);
		_update_jolt_param(axis, PARAM_LINEAR_LIMIT_SPRING_DAMPING);
		_update_jolt_param(axis, PARAM_LINEAR_SPRING_FREQUENCY);
		_update_jolt_param(axis, PARAM_ANGULAR_SPRING_FREQUENCY);

		_update_flag(axis, FLAG_ENABLE_LINEAR_LIMIT);
		_update_flag(axis, FLAG_ENABLE_LINEAR_MOTOR);
		_update_flag(axis, FLAG_ENABLE_LINEAR_SPRING);
		_update_flag(axis, FLAG_ENABLE_ANGULAR_LIMIT);
		_update_flag(axis, FLAG_ENABLE_ANGULAR_MOTOR);
		_update_flag(axis, FLAG_ENABLE_ANGULAR_SPRING);

		_update_jolt_flag(axis, FLAG_ENABLE_LINEAR_LIMIT_SPRING);
		_update_jolt_flag(axis, FLAG_ENABLE_LINEAR_SPRING_FREQUENCY, true);
		_update_jolt_flag(axis, FLAG_ENABLE_ANGULAR_SPRING_FREQUENCY, true);
	}
}

const double* JoltGeneric6DOFJoint3D::_get_param_ptr(Axis p_axis, Param p_param) const {
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
	return const_cast<JoltGeneric6DOFJoint3D*>(this)->_get_param_ptr(p_axis, p_param);
}

double* JoltGeneric6DOFJoint3D::_get_param_ptr(Axis p_axis, Param p_param) {
	switch (p_param) {
		case PARAM_LINEAR_LIMIT_UPPER: {
			return &linear_limit_upper[p_axis];
		}
		case PARAM_LINEAR_LIMIT_LOWER: {
			return &linear_limit_lower[p_axis];
		}
		case PARAM_LINEAR_LIMIT_SPRING_FREQUENCY: {
			return &linear_limit_spring_frequency[p_axis];
		}
		case PARAM_LINEAR_LIMIT_SPRING_DAMPING: {
			return &linear_limit_spring_damping[p_axis];
		}
		case PARAM_LINEAR_MOTOR_TARGET_VELOCITY: {
			return &linear_motor_target_velocity[p_axis];
		}
		case PARAM_LINEAR_MOTOR_MAX_FORCE: {
			return &linear_motor_max_force[p_axis];
		}
		case PARAM_LINEAR_SPRING_FREQUENCY: {
			return &linear_spring_frequency[p_axis];
		}
		case PARAM_LINEAR_SPRING_DAMPING: {
			return &linear_spring_damping[p_axis];
		}
		case PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT: {
			return &linear_spring_equilibrium_point[p_axis];
		}
		case PARAM_ANGULAR_LIMIT_UPPER: {
			return &angular_limit_upper[p_axis];
		}
		case PARAM_ANGULAR_LIMIT_LOWER: {
			return &angular_limit_lower[p_axis];
		}
		case PARAM_ANGULAR_MOTOR_TARGET_VELOCITY: {
			return &angular_motor_target_velocity[p_axis];
		}
		case PARAM_ANGULAR_MOTOR_MAX_TORQUE: {
			return &angular_motor_max_torque[p_axis];
		}
		case PARAM_ANGULAR_SPRING_FREQUENCY: {
			return &angular_spring_frequency[p_axis];
		}
		case PARAM_ANGULAR_SPRING_DAMPING: {
			return &angular_spring_damping[p_axis];
		}
		case PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT: {
			return &angular_spring_equilibrium_point[p_axis];
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled parameter: '%d'", p_param));
		}
	}
}

const bool* JoltGeneric6DOFJoint3D::_get_flag_ptr(Axis p_axis, Flag p_flag) const {
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
	return const_cast<JoltGeneric6DOFJoint3D*>(this)->_get_flag_ptr(p_axis, p_flag);
}

bool* JoltGeneric6DOFJoint3D::_get_flag_ptr(Axis p_axis, Flag p_flag) {
	switch (p_flag) {
		case FLAG_ENABLE_LINEAR_LIMIT: {
			return &linear_limit_enabled[p_axis];
		}
		case FLAG_ENABLE_LINEAR_LIMIT_SPRING: {
			return &linear_limit_spring_enabled[p_axis];
		}
		case FLAG_ENABLE_LINEAR_MOTOR: {
			return &linear_motor_enabled[p_axis];
		}
		case FLAG_ENABLE_LINEAR_SPRING: {
			return &linear_spring_enabled[p_axis];
		}
		case FLAG_ENABLE_ANGULAR_LIMIT: {
			return &angular_limit_enabled[p_axis];
		}
		case FLAG_ENABLE_ANGULAR_MOTOR: {
			return &angular_motor_enabled[p_axis];
		}
		case FLAG_ENABLE_ANGULAR_SPRING: {
			return &angular_spring_enabled[p_axis];
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled flag: '%d'", p_flag));
		}
	}
}

void JoltGeneric6DOFJoint3D::_update_param(Axis p_axis, Param p_param, double p_value) {
	QUIET_FAIL_COND(_is_invalid());

	PhysicsServer3D* server = _get_physics_server();
	ERR_FAIL_NULL(server);

	server->generic_6dof_joint_set_param(rid, p_axis, (ServerParam)p_param, p_value);
}

void JoltGeneric6DOFJoint3D::_update_param(Axis p_axis, Param p_param) {
	const double* value = _get_param_ptr(p_axis, p_param);
	ERR_FAIL_NULL(value);

	_update_param(p_axis, p_param, *value);
}

void JoltGeneric6DOFJoint3D::_update_jolt_param(Axis p_axis, Param p_param, double p_value) {
	QUIET_FAIL_COND(_is_invalid());

	JoltPhysicsServer3D* server = _get_jolt_physics_server();
	QUIET_FAIL_NULL(server);

	server->generic_6dof_joint_set_jolt_param(rid, p_axis, (ServerParamJolt)p_param, p_value);
}

void JoltGeneric6DOFJoint3D::_update_jolt_param(Axis p_axis, Param p_param) {
	const double* value = _get_param_ptr(p_axis, p_param);
	ERR_FAIL_NULL(value);

	_update_jolt_param(p_axis, p_param, *value);
}

void JoltGeneric6DOFJoint3D::_update_flag(Axis p_axis, Flag p_flag, bool p_enabled) {
	QUIET_FAIL_COND(_is_invalid());

	PhysicsServer3D* server = _get_physics_server();
	ERR_FAIL_NULL(server);

	server->generic_6dof_joint_set_flag(rid, p_axis, (ServerFlag)p_flag, p_enabled);
}

void JoltGeneric6DOFJoint3D::_update_flag(Axis p_axis, Flag p_flag) {
	const bool* value = _get_flag_ptr(p_axis, p_flag);
	QUIET_FAIL_NULL(value);

	_update_flag(p_axis, p_flag, *value);
}

void JoltGeneric6DOFJoint3D::_update_jolt_flag(Axis p_axis, Flag p_flag, bool p_enabled) {
	QUIET_FAIL_COND(_is_invalid());

	JoltPhysicsServer3D* server = _get_jolt_physics_server();
	QUIET_FAIL_NULL(server);

	server->generic_6dof_joint_set_jolt_flag(rid, p_axis, (ServerFlagJolt)p_flag, p_enabled);
}

void JoltGeneric6DOFJoint3D::_update_jolt_flag(Axis p_axis, Flag p_flag) {
	const bool* value = _get_flag_ptr(p_axis, p_flag);
	QUIET_FAIL_NULL(value);

	_update_jolt_flag(p_axis, p_flag, *value);
}

void JoltGeneric6DOFJoint3D::_param_changed(Axis p_axis, Param p_param) {
	switch (p_param) {
		case PARAM_LINEAR_LIMIT_UPPER:
		case PARAM_LINEAR_LIMIT_LOWER:
		case PARAM_LINEAR_MOTOR_TARGET_VELOCITY:
		case PARAM_LINEAR_MOTOR_MAX_FORCE:
		case PARAM_LINEAR_SPRING_DAMPING:
		case PARAM_LINEAR_SPRING_EQUILIBRIUM_POINT:
		case PARAM_ANGULAR_LIMIT_UPPER:
		case PARAM_ANGULAR_LIMIT_LOWER:
		case PARAM_ANGULAR_MOTOR_TARGET_VELOCITY:
		case PARAM_ANGULAR_MOTOR_MAX_TORQUE:
		case PARAM_ANGULAR_SPRING_DAMPING:
		case PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT: {
			_update_param(p_axis, p_param);
		} break;

		case PARAM_LINEAR_LIMIT_SPRING_FREQUENCY:
		case PARAM_LINEAR_LIMIT_SPRING_DAMPING:
		case PARAM_LINEAR_SPRING_FREQUENCY:
		case PARAM_ANGULAR_SPRING_FREQUENCY: {
			_update_jolt_param(p_axis, p_param);
		} break;

		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		} break;
	}
}

void JoltGeneric6DOFJoint3D::_flag_changed(Axis p_axis, Flag p_flag) {
	switch (p_flag) {
		case FLAG_ENABLE_LINEAR_LIMIT:
		case FLAG_ENABLE_LINEAR_MOTOR:
		case FLAG_ENABLE_LINEAR_SPRING:
		case FLAG_ENABLE_ANGULAR_LIMIT:
		case FLAG_ENABLE_ANGULAR_MOTOR:
		case FLAG_ENABLE_ANGULAR_SPRING: {
			_update_flag(p_axis, p_flag);
		} break;

		case FLAG_ENABLE_LINEAR_LIMIT_SPRING: {
			_update_jolt_flag(p_axis, p_flag);
		} break;

		default: {
			ERR_FAIL_MSG(vformat("Unhandled flag: '%d'", p_flag));
		} break;
	}
}
