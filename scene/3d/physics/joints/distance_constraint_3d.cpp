#include "distance_constraint_3d.h"

void DistanceConstraint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_param", "param", "value"), &DistanceConstraint3D::set_param);
	ClassDB::bind_method(D_METHOD("get_param", "param"), &DistanceConstraint3D::get_param);
	ClassDB::bind_method(D_METHOD("set_point_param", "point", "value"), &DistanceConstraint3D::set_point_param);
	ClassDB::bind_method(D_METHOD("get_point_param", "point"), &DistanceConstraint3D::get_point_param);
	ClassDB::bind_method(D_METHOD("get_global_point", "point"), &DistanceConstraint3D::get_global_point);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spring/frequency"), "set_param", "get_param", PARAM_LIMITS_SPRING_FREQUENCY);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "spring/damping"), "set_param", "get_param", PARAM_LIMITS_SPRING_DAMPING);

	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "distance/min"), "set_param", "get_param", PARAM_DISTANCE_MIN);
	ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "distance/max"), "set_param", "get_param", PARAM_DISTANCE_MAX);

	ADD_PROPERTYI(PropertyInfo(Variant::VECTOR3, "anchor/a"), "set_point_param", "get_point_param", POINT_PARAM_A);
	ADD_PROPERTYI(PropertyInfo(Variant::VECTOR3, "anchor/b"), "set_point_param", "get_point_param", POINT_PARAM_B);

	BIND_ENUM_CONSTANT(PARAM_LIMITS_SPRING_FREQUENCY);
	BIND_ENUM_CONSTANT(PARAM_LIMITS_SPRING_DAMPING);
	BIND_ENUM_CONSTANT(PARAM_DISTANCE_MIN);
	BIND_ENUM_CONSTANT(PARAM_DISTANCE_MAX);
	BIND_ENUM_CONSTANT(PARAM_MAX);

	BIND_ENUM_CONSTANT(POINT_PARAM_A);
	BIND_ENUM_CONSTANT(POINT_PARAM_B);
}

JoltPhysicsServer3D *DistanceConstraint3D::_get_jolt_physics_server() {
	JoltPhysicsServer3D *physics_server = JoltPhysicsServer3D::get_singleton();

	if (unlikely(physics_server == nullptr)) {
		ERR_PRINT_ONCE(
				"DistanceConstraint3D was unable to retrieve the Jolt-based physics server. "
				"Make sure that you have 'Jolt Physics' set as the currently active physics engine. ");
	}

	return physics_server;
}

void DistanceConstraint3D::set_param(Param p_param, real_t p_value) {
	ERR_FAIL_INDEX(p_param, PARAM_MAX);
	params[p_param] = p_value;
	if (is_configured()) {
		_get_jolt_physics_server()->distance_constraint_set_jolt_param(get_rid(), JoltPhysicsServer3D::DistanceConstraintParamJolt(p_param), p_value);
	}
}

real_t DistanceConstraint3D::get_param(Param p_param) const {
	ERR_FAIL_INDEX_V(p_param, PARAM_MAX, 0);
	return params[p_param];
}

void DistanceConstraint3D::set_point_param(PointParam p_param, const Vector3 &p_value) {
	ERR_FAIL_INDEX(p_param, POINT_PARAM_MAX);
	point_params[p_param] = p_value;
	if (is_configured()) {
		_disconnect_signals();
		_update_joint();
	}
}

Vector3 DistanceConstraint3D::get_point_param(PointParam p_param) const {
	ERR_FAIL_INDEX_V(p_param, POINT_PARAM_MAX, Vector3());
	return point_params[p_param];
}

Vector3 DistanceConstraint3D::get_global_point(PointParam p_param) const {
	const PhysicsBody3D *body = _get_body_from_param(p_param);
	const Vector3 local_point = get_point_param(p_param);
	if (body == nullptr) {
		return to_global(local_point);
	}
	return body->to_global(local_point);
}

PhysicsBody3D *DistanceConstraint3D::_get_body_from_param(PointParam p_param) const {
	const NodePath node_path = p_param == POINT_PARAM_A ? get_node_a() : get_node_b();
	Node *node = get_node_or_null(node_path);
	return Object::cast_to<PhysicsBody3D>(node);
}

void DistanceConstraint3D::_configure_joint(RID p_joint, PhysicsBody3D *p_body_a, PhysicsBody3D *p_body_b) {
	JoltPhysicsServer3D *physics_server = _get_jolt_physics_server();
	ERR_FAIL_NULL(physics_server);

	const bool are_bodies_switched = _get_body_from_param(POINT_PARAM_A) == nullptr;

	const Vector3 global_position = are_bodies_switched ? get_global_point(POINT_PARAM_A) : get_global_point(POINT_PARAM_B);
	const Vector3 point_a = get_point_param(POINT_PARAM_A);
	const Vector3 point_b = get_point_param(POINT_PARAM_B);
	const Vector3 p_body_a_point = are_bodies_switched ? point_b : point_a;
	const Vector3 p_body_b_point = are_bodies_switched ? point_a : point_b;

	physics_server->joint_make_distance_constraint(
			p_joint,
			p_body_a->get_rid(),
			p_body_a_point,
			p_body_b != nullptr ? p_body_b->get_rid() : RID(),
			p_body_b != nullptr ? p_body_b_point : global_position);

	for (int i = 0; i < PARAM_MAX; i++) {
		physics_server->distance_constraint_set_jolt_param(p_joint, JoltPhysicsServer3D::DistanceConstraintParamJolt(i), params[i]);
	}
}

PackedStringArray DistanceConstraint3D::get_configuration_warnings() const {
	PackedStringArray warnings = Joint3D::get_configuration_warnings();

	JoltPhysicsServer3D *physics_server = JoltPhysicsServer3D::get_singleton();
	if (!physics_server) {
		warnings.push_back(RTR("DistanceConstraint3D is only compatible with Jolt Physics. Please change your Physics Engine in Project Settings."));
	}
	return warnings;
}

DistanceConstraint3D::DistanceConstraint3D() {
	params[PARAM_LIMITS_SPRING_FREQUENCY] = 0.0;
	params[PARAM_LIMITS_SPRING_DAMPING] = 0.0;
	params[PARAM_DISTANCE_MIN] = 0.0;
	params[PARAM_DISTANCE_MAX] = INFINITY;

	point_params[POINT_PARAM_A] = Vector3(0.0, 0.0, 0.0);
	point_params[POINT_PARAM_B] = Vector3(0.0, 0.0, 0.0);
}