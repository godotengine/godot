//
// Created by amara on 16/11/2021.
//

#include "l_physics_body.h"

#include "../lilyphys_server.h"
#include "../l_body_state.h"

LPhysicsBody::LPhysicsBody() : LCollisionObject(Type::TYPE_BODY) {
    rid = LilyphysServer::get_singleton()->create_physics_body();
    LilyphysServer::get_singleton()->set_integration_callback(rid, this, "_state_changed");
}

void LPhysicsBody::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_state_changed"), &LPhysicsBody::_state_changed);

    ClassDB::bind_method(D_METHOD("set_acceleration", "acceleration"), &LPhysicsBody::set_acceleration);
    ClassDB::bind_method(D_METHOD("set_angular_damping", "angular_damping"), &LPhysicsBody::set_angular_damping);
    ClassDB::bind_method(D_METHOD("set_angular_velocity", "angular_velocity"), &LPhysicsBody::set_angular_velocity);
    ClassDB::bind_method(D_METHOD("set_inverse_inertia_tensor", "inverse_inertia_tensor"), &LPhysicsBody::set_inverse_inertia_tensor);
    ClassDB::bind_method(D_METHOD("set_inverse_mass", "inverse_mass"), &LPhysicsBody::set_inverse_mass);
    ClassDB::bind_method(D_METHOD("set_mass", "mass"), &LPhysicsBody::set_mass);
    ClassDB::bind_method(D_METHOD("set_linear_damping", "linear_damping"), &LPhysicsBody::set_linear_damping);
    ClassDB::bind_method(D_METHOD("set_velocity", "velocity"), &LPhysicsBody::set_velocity);

    ClassDB::bind_method(D_METHOD("get_acceleration"), &LPhysicsBody::get_acceleration);
    ClassDB::bind_method(D_METHOD("get_angular_damping"), &LPhysicsBody::get_angular_damping);
    ClassDB::bind_method(D_METHOD("get_angular_velocity"), &LPhysicsBody::get_angular_velocity);
    ClassDB::bind_method(D_METHOD("get_inverse_inertia_tensor"), &LPhysicsBody::get_inverse_inertia_tensor);
    ClassDB::bind_method(D_METHOD("get_inverse_mass"), &LPhysicsBody::get_inverse_mass);
    ClassDB::bind_method(D_METHOD("get_mass"), &LPhysicsBody::get_mass);
    ClassDB::bind_method(D_METHOD("get_linear_damping"), &LPhysicsBody::get_linear_damping);
    ClassDB::bind_method(D_METHOD("get_velocity"), &LPhysicsBody::get_velocity);

    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "acceleration"), "set_acceleration", "get_acceleration");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_damping"), "set_angular_damping", "get_angular_damping");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
    ADD_PROPERTY(PropertyInfo(Variant::BASIS, "inverse_inertia_tensor"), "set_inverse_inertia_tensor", "get_inverse_inertia_tensor");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "inverse_mass"), "set_inverse_mass", "get_inverse_mass");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "mass"), "set_mass", "get_mass");
    ADD_PROPERTY(PropertyInfo(Variant::REAL, "linear_damping"), "set_linear_damping", "get_linear_damping");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "velocity"), "set_velocity", "get_velocity");
}

void LPhysicsBody::_state_changed(Object *p_state) {
    LBodyState *state = Object::cast_to<LBodyState>(p_state);
    ERR_FAIL_COND_MSG(!state, "Method '_state_changed' must receive a valid LBodyState object as argument");

    set_ignore_transform_notification(true);
    set_global_transform(state->get_transform());
    set_ignore_transform_notification(false);
}

void LPhysicsBody::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE:
        case NOTIFICATION_TRANSFORM_CHANGED:
        LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::TRANSFORM, get_global_transform());
        break;
    }
}

real_t LPhysicsBody::get_inverse_mass() const {
    return inverse_mass;
}

void LPhysicsBody::set_inverse_mass(real_t p_inverse_mass) {
    inverse_mass = p_inverse_mass;
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::INVERSE_MASS, inverse_mass);
    _change_notify("inverse_mass");
    _change_notify("mass");
}

real_t LPhysicsBody::get_linear_damping() const {
    return linear_damping;
}

void LPhysicsBody::set_linear_damping(real_t p_linear_damping) {
    linear_damping = p_linear_damping;
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::LINEAR_DAMPING, linear_damping);
}

real_t LPhysicsBody::get_angular_damping() const {
    return angular_damping;
}

void LPhysicsBody::set_angular_damping(real_t p_angular_damping) {
    angular_damping = p_angular_damping;
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::ANGULAR_DAMPING, angular_damping);
}

const Vector3 &LPhysicsBody::get_velocity() const {
    return velocity;
}

void LPhysicsBody::set_velocity(const Vector3 &p_velocity) {
    velocity = p_velocity;
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::VELOCITY, velocity);
}

const Vector3 &LPhysicsBody::get_acceleration() const {
    return acceleration;
}

void LPhysicsBody::set_acceleration(const Vector3 &p_acceleration) {
    acceleration = p_acceleration;
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::ACCELERATION, acceleration);
}

const Vector3 &LPhysicsBody::get_angular_velocity() const {
    return angular_velocity;
}

void LPhysicsBody::set_angular_velocity(const Vector3 &p_angular_velocity) {
    angular_velocity = p_angular_velocity;
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::ANGULAR_VELOCITY, angular_velocity);
}

const Basis &LPhysicsBody::get_inverse_inertia_tensor() const {
    return inverse_inertia_tensor;
}

void LPhysicsBody::set_inverse_inertia_tensor(const Basis &p_inverse_inertia_tensor) {
    inverse_inertia_tensor = p_inverse_inertia_tensor;
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::INV_INERTIA_TENSOR, inverse_inertia_tensor);
}

void LPhysicsBody::set_mass(real_t p_mass) {
    set_inverse_mass(1.0f / p_mass);
}

real_t LPhysicsBody::get_mass() const {
    return (1.0f / get_inverse_mass());
}