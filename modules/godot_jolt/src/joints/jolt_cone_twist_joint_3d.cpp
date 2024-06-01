#include "jolt_cone_twist_joint_3d.hpp"

#include "servers/jolt_physics_server_3d.hpp"

namespace {

using ServerParam = PhysicsServer3D::ConeTwistJointParam;
using ServerParamJolt = JoltPhysicsServer3D::ConeTwistJointParamJolt;
using ServerFlagJolt = JoltPhysicsServer3D::ConeTwistJointFlagJolt;

} // namespace

void JoltConeTwistJoint3D::_bind_methods() {
	BIND_METHOD(JoltConeTwistJoint3D, get_swing_limit_enabled);
	BIND_METHOD(JoltConeTwistJoint3D, set_swing_limit_enabled, "enabled");

	BIND_METHOD(JoltConeTwistJoint3D, get_twist_limit_enabled);
	BIND_METHOD(JoltConeTwistJoint3D, set_twist_limit_enabled, "enabled");

	BIND_METHOD(JoltConeTwistJoint3D, get_swing_limit_span);
	BIND_METHOD(JoltConeTwistJoint3D, set_swing_limit_span, "value");

	BIND_METHOD(JoltConeTwistJoint3D, get_twist_limit_span);
	BIND_METHOD(JoltConeTwistJoint3D, set_twist_limit_span, "value");

	BIND_METHOD(JoltConeTwistJoint3D, get_swing_motor_enabled);
	BIND_METHOD(JoltConeTwistJoint3D, set_swing_motor_enabled, "enabled");

	BIND_METHOD(JoltConeTwistJoint3D, get_twist_motor_enabled);
	BIND_METHOD(JoltConeTwistJoint3D, set_twist_motor_enabled, "enabled");

	BIND_METHOD(JoltConeTwistJoint3D, get_swing_motor_target_velocity_y);
	BIND_METHOD(JoltConeTwistJoint3D, set_swing_motor_target_velocity_y, "value");

	BIND_METHOD(JoltConeTwistJoint3D, get_swing_motor_target_velocity_z);
	BIND_METHOD(JoltConeTwistJoint3D, set_swing_motor_target_velocity_z, "value");

	BIND_METHOD(JoltConeTwistJoint3D, get_twist_motor_target_velocity);
	BIND_METHOD(JoltConeTwistJoint3D, set_twist_motor_target_velocity, "value");

	BIND_METHOD(JoltConeTwistJoint3D, get_swing_motor_max_torque);
	BIND_METHOD(JoltConeTwistJoint3D, set_swing_motor_max_torque, "value");

	BIND_METHOD(JoltConeTwistJoint3D, get_twist_motor_max_torque);
	BIND_METHOD(JoltConeTwistJoint3D, set_twist_motor_max_torque, "value");

	BIND_METHOD(JoltConeTwistJoint3D, get_applied_force);
	BIND_METHOD(JoltConeTwistJoint3D, get_applied_torque);

	ADD_GROUP("Swing Limit", "swing_limit_");

	BIND_PROPERTY("swing_limit_enabled", Variant::BOOL);
	BIND_PROPERTY_RANGED("swing_limit_span", Variant::FLOAT, "-180,180,0.1,radians");

	ADD_GROUP("Twist Limit", "twist_limit_");

	BIND_PROPERTY("twist_limit_enabled", Variant::BOOL);
	BIND_PROPERTY_RANGED("twist_limit_span", Variant::FLOAT, "-180,180,0.1,radians");

	ADD_GROUP("Swing Motor", "swing_motor_");

	BIND_PROPERTY("swing_motor_enabled", Variant::BOOL);
	BIND_PROPERTY("swing_motor_target_velocity_y", Variant::FLOAT, U"radians,suffix:°/s");
	BIND_PROPERTY("swing_motor_target_velocity_z", Variant::FLOAT, U"radians,suffix:°/s");
	BIND_PROPERTY("swing_motor_max_torque", Variant::FLOAT, U"suffix:kg⋅m²/s² (Nm)");

	ADD_GROUP("Twist Motor", "twist_motor_");

	BIND_PROPERTY("twist_motor_enabled", Variant::BOOL);
	BIND_PROPERTY("twist_motor_target_velocity", Variant::FLOAT, U"radians,suffix:°/s");
	BIND_PROPERTY("twist_motor_max_torque", Variant::FLOAT, U"suffix:kg⋅m²/s² (Nm)");
}

void JoltConeTwistJoint3D::set_swing_limit_enabled(bool p_enabled) {
	if (swing_limit_enabled == p_enabled) {
		return;
	}

	swing_limit_enabled = p_enabled;

	_flag_changed(FLAG_USE_SWING_LIMIT);
}

void JoltConeTwistJoint3D::set_twist_limit_enabled(bool p_enabled) {
	if (twist_limit_enabled == p_enabled) {
		return;
	}

	twist_limit_enabled = p_enabled;

	_flag_changed(FLAG_USE_TWIST_LIMIT);
}

void JoltConeTwistJoint3D::set_swing_limit_span(double p_value) {
	if (swing_limit_span == p_value) {
		return;
	}

	swing_limit_span = p_value;

	_param_changed(PARAM_SWING_LIMIT_SPAN);
}

void JoltConeTwistJoint3D::set_twist_limit_span(double p_value) {
	if (twist_limit_span == p_value) {
		return;
	}

	twist_limit_span = p_value;

	_param_changed(PARAM_TWIST_LIMIT_SPAN);
}

void JoltConeTwistJoint3D::set_swing_motor_enabled(bool p_enabled) {
	if (swing_motor_enabled == p_enabled) {
		return;
	}

	swing_motor_enabled = p_enabled;

	_flag_changed(FLAG_ENABLE_SWING_MOTOR);
}

void JoltConeTwistJoint3D::set_twist_motor_enabled(bool p_enabled) {
	if (twist_motor_enabled == p_enabled) {
		return;
	}

	twist_motor_enabled = p_enabled;

	_flag_changed(FLAG_ENABLE_TWIST_MOTOR);
}

void JoltConeTwistJoint3D::set_swing_motor_target_velocity_y(double p_value) {
	if (swing_motor_target_velocity_y == p_value) {
		return;
	}

	swing_motor_target_velocity_y = p_value;

	_param_changed(PARAM_SWING_MOTOR_TARGET_VELOCITY_Y);
}

void JoltConeTwistJoint3D::set_swing_motor_target_velocity_z(double p_value) {
	if (swing_motor_target_velocity_z == p_value) {
		return;
	}

	swing_motor_target_velocity_z = p_value;

	_param_changed(PARAM_SWING_MOTOR_TARGET_VELOCITY_Z);
}

void JoltConeTwistJoint3D::set_twist_motor_target_velocity(double p_value) {
	if (twist_motor_target_velocity == p_value) {
		return;
	}

	twist_motor_target_velocity = p_value;

	_param_changed(PARAM_TWIST_MOTOR_TARGET_VELOCITY);
}

void JoltConeTwistJoint3D::set_swing_motor_max_torque(double p_value) {
	if (swing_motor_max_torque == p_value) {
		return;
	}

	swing_motor_max_torque = p_value;

	_param_changed(PARAM_SWING_MOTOR_MAX_TORQUE);
}

void JoltConeTwistJoint3D::set_twist_motor_max_torque(double p_value) {
	if (twist_motor_max_torque == p_value) {
		return;
	}

	twist_motor_max_torque = p_value;

	_param_changed(PARAM_TWIST_MOTOR_MAX_TORQUE);
}

float JoltConeTwistJoint3D::get_applied_force() const {
	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL_D(physics_server);

	return physics_server->cone_twist_joint_get_applied_force(rid);
}

float JoltConeTwistJoint3D::get_applied_torque() const {
	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL_D(physics_server);

	return physics_server->cone_twist_joint_get_applied_torque(rid);
}

void JoltConeTwistJoint3D::_configure(PhysicsBody3D* p_body_a, PhysicsBody3D* p_body_b) {
	PhysicsServer3D* physics_server = _get_physics_server();
	ERR_FAIL_NULL(physics_server);

	physics_server->joint_make_cone_twist(
		rid,
		p_body_a->get_rid(),
		_get_body_local_transform(*p_body_a),
		p_body_b != nullptr ? p_body_b->get_rid() : RID(),
		p_body_b != nullptr ? _get_body_local_transform(*p_body_b)
							: get_global_transform().orthonormalized()
	);

	_update_param(PARAM_SWING_LIMIT_SPAN);
	_update_param(PARAM_TWIST_LIMIT_SPAN);

	_update_jolt_param(PARAM_SWING_MOTOR_TARGET_VELOCITY_Y);
	_update_jolt_param(PARAM_SWING_MOTOR_TARGET_VELOCITY_Z);
	_update_jolt_param(PARAM_TWIST_MOTOR_TARGET_VELOCITY);
	_update_jolt_param(PARAM_SWING_MOTOR_MAX_TORQUE);
	_update_jolt_param(PARAM_TWIST_MOTOR_MAX_TORQUE);

	_update_jolt_flag(FLAG_USE_SWING_LIMIT);
	_update_jolt_flag(FLAG_USE_TWIST_LIMIT);
	_update_jolt_flag(FLAG_ENABLE_SWING_MOTOR);
	_update_jolt_flag(FLAG_ENABLE_TWIST_MOTOR);
}

void JoltConeTwistJoint3D::_update_param(Param p_param) {
	QUIET_FAIL_COND(_is_invalid());

	PhysicsServer3D* physics_server = _get_physics_server();
	ERR_FAIL_NULL(physics_server);

	double* value = nullptr;

	switch (p_param) {
		case PARAM_SWING_LIMIT_SPAN: {
			value = &swing_limit_span;
		} break;
		case PARAM_TWIST_LIMIT_SPAN: {
			value = &twist_limit_span;
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		} break;
	}

	physics_server->cone_twist_joint_set_param(rid, ServerParam(p_param), *value);
}

void JoltConeTwistJoint3D::_update_jolt_param(Param p_param) {
	QUIET_FAIL_COND(_is_invalid());

	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL(physics_server);

	double* value = nullptr;

	switch (p_param) {
		case PARAM_SWING_MOTOR_TARGET_VELOCITY_Y: {
			value = &swing_motor_target_velocity_y;
		} break;
		case PARAM_SWING_MOTOR_TARGET_VELOCITY_Z: {
			value = &swing_motor_target_velocity_z;
		} break;
		case PARAM_TWIST_MOTOR_TARGET_VELOCITY: {
			value = &twist_motor_target_velocity;
		} break;
		case PARAM_SWING_MOTOR_MAX_TORQUE: {
			value = &swing_motor_max_torque;
		} break;
		case PARAM_TWIST_MOTOR_MAX_TORQUE: {
			value = &twist_motor_max_torque;
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		} break;
	}

	physics_server->cone_twist_joint_set_jolt_param(rid, ServerParamJolt(p_param), *value);
}

void JoltConeTwistJoint3D::_update_jolt_flag(Flag p_flag) {
	QUIET_FAIL_COND(_is_invalid());

	JoltPhysicsServer3D* physics_server = _get_jolt_physics_server();
	QUIET_FAIL_NULL(physics_server);

	bool* value = nullptr;

	switch (p_flag) {
		case FLAG_USE_SWING_LIMIT: {
			value = &swing_limit_enabled;
		} break;
		case FLAG_USE_TWIST_LIMIT: {
			value = &twist_limit_enabled;
		} break;
		case FLAG_ENABLE_SWING_MOTOR: {
			value = &swing_motor_enabled;
		} break;
		case FLAG_ENABLE_TWIST_MOTOR: {
			value = &twist_motor_enabled;
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled flag: '%d'", p_flag));
		} break;
	}

	physics_server->cone_twist_joint_set_jolt_flag(rid, ServerFlagJolt(p_flag), *value);
}

void JoltConeTwistJoint3D::_param_changed(Param p_param) {
	switch (p_param) {
		case PARAM_SWING_LIMIT_SPAN:
		case PARAM_TWIST_LIMIT_SPAN: {
			_update_param(p_param);
		} break;
		case PARAM_SWING_MOTOR_TARGET_VELOCITY_Y:
		case PARAM_SWING_MOTOR_TARGET_VELOCITY_Z:
		case PARAM_TWIST_MOTOR_TARGET_VELOCITY:
		case PARAM_SWING_MOTOR_MAX_TORQUE:
		case PARAM_TWIST_MOTOR_MAX_TORQUE: {
			_update_jolt_param(p_param);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		} break;
	}
}

void JoltConeTwistJoint3D::_flag_changed(Flag p_flag) {
	switch (p_flag) {
		case FLAG_USE_SWING_LIMIT:
		case FLAG_USE_TWIST_LIMIT:
		case FLAG_ENABLE_SWING_MOTOR:
		case FLAG_ENABLE_TWIST_MOTOR: {
			_update_jolt_flag(p_flag);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled flag: '%d'", p_flag));
		} break;
	}
}
