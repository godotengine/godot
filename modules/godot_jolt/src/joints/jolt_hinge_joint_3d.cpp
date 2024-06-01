#include "jolt_hinge_joint_3d.hpp"

#include "servers/jolt_physics_server_3d.hpp"

namespace {

using ServerParam = PhysicsServer3D::HingeJointParam;
using ServerFlag = PhysicsServer3D::HingeJointFlag;
using ServerParamJolt = JoltPhysicsServer3D::HingeJointParamJolt;
using ServerFlagJolt = JoltPhysicsServer3D::HingeJointFlagJolt;

} // namespace

void JoltHingeJoint3D::_bind_methods() {
	BIND_METHOD(JoltHingeJoint3D, get_limit_enabled);
	BIND_METHOD(JoltHingeJoint3D, set_limit_enabled, "enabled");

	BIND_METHOD(JoltHingeJoint3D, get_limit_upper);
	BIND_METHOD(JoltHingeJoint3D, set_limit_upper, "value");

	BIND_METHOD(JoltHingeJoint3D, get_limit_lower);
	BIND_METHOD(JoltHingeJoint3D, set_limit_lower, "value");

	BIND_METHOD(JoltHingeJoint3D, get_limit_spring_enabled);
	BIND_METHOD(JoltHingeJoint3D, set_limit_spring_enabled, "enabled");

	BIND_METHOD(JoltHingeJoint3D, get_limit_spring_frequency);
	BIND_METHOD(JoltHingeJoint3D, set_limit_spring_frequency, "value");

	BIND_METHOD(JoltHingeJoint3D, get_limit_spring_damping);
	BIND_METHOD(JoltHingeJoint3D, set_limit_spring_damping, "value");

	BIND_METHOD(JoltHingeJoint3D, get_motor_enabled);
	BIND_METHOD(JoltHingeJoint3D, set_motor_enabled, "enabled");

	BIND_METHOD(JoltHingeJoint3D, get_motor_target_velocity);
	BIND_METHOD(JoltHingeJoint3D, set_motor_target_velocity, "value");

	BIND_METHOD(JoltHingeJoint3D, get_motor_max_torque);
	BIND_METHOD(JoltHingeJoint3D, set_motor_max_torque, "value");

	BIND_METHOD(JoltHingeJoint3D, get_applied_force);
	BIND_METHOD(JoltHingeJoint3D, get_applied_torque);

	ADD_GROUP("Limit", "limit_");

	BIND_PROPERTY("limit_enabled", Variant::BOOL);
	BIND_PROPERTY_RANGED("limit_upper", Variant::FLOAT, "-180,180,0.1,radians");
	BIND_PROPERTY_RANGED("limit_lower", Variant::FLOAT, "-180,180,0.1,radians");

	ADD_GROUP("Limit Spring", "limit_spring_");

	BIND_PROPERTY("limit_spring_enabled", Variant::BOOL);
	BIND_PROPERTY("limit_spring_frequency", Variant::FLOAT, "suffix:hz");
	BIND_PROPERTY("limit_spring_damping", Variant::FLOAT);

	ADD_GROUP("Motor", "motor_");

	BIND_PROPERTY("motor_enabled", Variant::BOOL);
	BIND_PROPERTY("motor_target_velocity", Variant::FLOAT, U"radians,suffix:°/s");
	BIND_PROPERTY("motor_max_torque", Variant::FLOAT, U"suffix:kg⋅m²/s² (Nm)");
}

void JoltHingeJoint3D::set_limit_enabled(bool p_enabled) {
	if (limit_enabled == p_enabled) {
		return;
	}

	limit_enabled = p_enabled;

	_flag_changed(FLAG_USE_LIMIT);
}

void JoltHingeJoint3D::set_limit_upper(double p_value) {
	if (limit_upper == p_value) {
		return;
	}

	limit_upper = p_value;

	_param_changed(PARAM_LIMIT_UPPER);
}

void JoltHingeJoint3D::set_limit_lower(double p_value) {
	if (limit_lower == p_value) {
		return;
	}

	limit_lower = p_value;

	_param_changed(PARAM_LIMIT_LOWER);
}

void JoltHingeJoint3D::set_limit_spring_enabled(bool p_enabled) {
	if (limit_spring_enabled == p_enabled) {
		return;
	}

	limit_spring_enabled = p_enabled;

	_flag_changed(FLAG_USE_LIMIT_SPRING);
}

void JoltHingeJoint3D::set_limit_spring_frequency(double p_value) {
	if (limit_spring_frequency == p_value) {
		return;
	}

	limit_spring_frequency = p_value;

	_param_changed(PARAM_LIMIT_SPRING_FREQUENCY);
}

void JoltHingeJoint3D::set_limit_spring_damping(double p_value) {
	if (limit_spring_damping == p_value) {
		return;
	}

	limit_spring_damping = p_value;

	_param_changed(PARAM_LIMIT_SPRING_DAMPING);
}

void JoltHingeJoint3D::set_motor_enabled(bool p_enabled) {
	if (motor_enabled == p_enabled) {
		return;
	}

	motor_enabled = p_enabled;

	_flag_changed(FLAG_ENABLE_MOTOR);
}

void JoltHingeJoint3D::set_motor_target_velocity(double p_value) {
	if (motor_target_velocity == p_value) {
		return;
	}

	motor_target_velocity = p_value;

	_param_changed(PARAM_MOTOR_TARGET_VELOCITY);
}

void JoltHingeJoint3D::set_motor_max_torque(double p_value) {
	if (motor_max_torque == p_value) {
		return;
	}

	motor_max_torque = p_value;

	_param_changed(PARAM_MOTOR_MAX_TORQUE);
}

float JoltHingeJoint3D::get_applied_force() const {
	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL_D(physics_server);

	return physics_server->hinge_joint_get_applied_force(rid);
}

float JoltHingeJoint3D::get_applied_torque() const {
	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL_D(physics_server);

	return physics_server->hinge_joint_get_applied_torque(rid);
}

void JoltHingeJoint3D::_configure(PhysicsBody3D* p_body_a, PhysicsBody3D* p_body_b) {
	PhysicsServer3D* physics_server = _get_physics_server();
	ERR_FAIL_NULL(physics_server);

	physics_server->joint_make_hinge(
		rid,
		p_body_a->get_rid(),
		_get_body_local_transform(*p_body_a),
		p_body_b != nullptr ? p_body_b->get_rid() : RID(),
		p_body_b != nullptr ? _get_body_local_transform(*p_body_b)
							: get_global_transform().orthonormalized()
	);

	_update_param(PARAM_LIMIT_UPPER);
	_update_param(PARAM_LIMIT_LOWER);
	_update_param(PARAM_MOTOR_TARGET_VELOCITY);

	_update_jolt_param(PARAM_LIMIT_SPRING_FREQUENCY);
	_update_jolt_param(PARAM_LIMIT_SPRING_DAMPING);
	_update_jolt_param(PARAM_MOTOR_MAX_TORQUE);

	_update_flag(FLAG_USE_LIMIT);
	_update_flag(FLAG_ENABLE_MOTOR);

	_update_jolt_flag(FLAG_USE_LIMIT_SPRING);
}

void JoltHingeJoint3D::_update_param(Param p_param) {
	QUIET_FAIL_COND(_is_invalid());

	PhysicsServer3D* physics_server = _get_physics_server();
	ERR_FAIL_NULL(physics_server);

	double* value = nullptr;

	switch (p_param) {
		case PARAM_LIMIT_UPPER: {
			value = &limit_upper;
		} break;
		case PARAM_LIMIT_LOWER: {
			value = &limit_lower;
		} break;
		case PARAM_MOTOR_TARGET_VELOCITY: {
			value = &motor_target_velocity;
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		} break;
	}

	physics_server->hinge_joint_set_param(rid, ServerParam(p_param), *value);
}

void JoltHingeJoint3D::_update_jolt_param(Param p_param) {
	QUIET_FAIL_COND(_is_invalid());

	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL(physics_server);

	double* value = nullptr;

	switch (p_param) {
		case PARAM_LIMIT_SPRING_FREQUENCY: {
			value = &limit_spring_frequency;
		} break;
		case PARAM_LIMIT_SPRING_DAMPING: {
			value = &limit_spring_damping;
		} break;
		case PARAM_MOTOR_MAX_TORQUE: {
			value = &motor_max_torque;
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		} break;
	}

	physics_server->hinge_joint_set_jolt_param(rid, ServerParamJolt(p_param), *value);
}

void JoltHingeJoint3D::_update_flag(Flag p_flag) {
	QUIET_FAIL_COND(_is_invalid());

	PhysicsServer3D* physics_server = _get_physics_server();
	ERR_FAIL_NULL(physics_server);

	bool* value = nullptr;

	switch (p_flag) {
		case FLAG_USE_LIMIT: {
			value = &limit_enabled;
		} break;
		case FLAG_ENABLE_MOTOR: {
			value = &motor_enabled;
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled flag: '%d'", p_flag));
		} break;
	}

	physics_server->hinge_joint_set_flag(rid, ServerFlag(p_flag), *value);
}

void JoltHingeJoint3D::_update_jolt_flag(Flag p_flag) {
	QUIET_FAIL_COND(_is_invalid());

	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL(physics_server);

	bool* value = nullptr;

	switch (p_flag) {
		case FLAG_USE_LIMIT_SPRING: {
			value = &limit_spring_enabled;
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled flag: '%d'", p_flag));
		} break;
	}

	physics_server->hinge_joint_set_jolt_flag(rid, ServerFlagJolt(p_flag), *value);
}

void JoltHingeJoint3D::_param_changed(Param p_param) {
	switch (p_param) {
		case PARAM_LIMIT_UPPER:
		case PARAM_LIMIT_LOWER:
		case PARAM_MOTOR_TARGET_VELOCITY: {
			_update_param(p_param);
		} break;
		case PARAM_LIMIT_SPRING_FREQUENCY:
		case PARAM_LIMIT_SPRING_DAMPING:
		case PARAM_MOTOR_MAX_TORQUE: {
			_update_jolt_param(p_param);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		} break;
	}
}

void JoltHingeJoint3D::_flag_changed(Flag p_flag) {
	switch (p_flag) {
		case FLAG_USE_LIMIT:
		case FLAG_ENABLE_MOTOR: {
			_update_flag(p_flag);
		} break;
		case FLAG_USE_LIMIT_SPRING: {
			_update_jolt_flag(p_flag);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled flag: '%d'", p_flag));
		} break;
	}
}
