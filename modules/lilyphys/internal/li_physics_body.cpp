//
// Created by amara on 16/11/2021.
//

#include "li_physics_body.h"
#include "../l_body_state.h"

LIPhysicsBody::LIPhysicsBody() : LICollisionObject(TYPE_BODY) {

}

const Transform &LIPhysicsBody::get_transform() const {
    return transform;
}

real_t LIPhysicsBody::get_inverse_mass() const {
    return inverse_mass;
}

real_t LIPhysicsBody::get_linear_damping() const {
    return linear_damping;
}

real_t LIPhysicsBody::get_angular_damping() const {
    return angular_damping;
}

const Vector3 &LIPhysicsBody::get_velocity() const {
    return velocity;
}

const Vector3 &LIPhysicsBody::get_acceleration() const {
    return acceleration;
}

const Vector3 &LIPhysicsBody::get_last_acceleration() const {
    return last_acceleration;
}

const Vector3 &LIPhysicsBody::get_rotation() const {
    return rotation;
}

const Vector3 &LIPhysicsBody::get_force_accum() const {
    return force_accum;
}

const Vector3 &LIPhysicsBody::get_torque_accum() const {
    return torque_accum;
}

const Basis &LIPhysicsBody::get_inv_inertia_tensor() const {
    return inv_inertia_tensor;
}

void LIPhysicsBody::set_integration_callback(ObjectID p_id, const StringName &p_method, const Variant &p_user_data = Variant()) {
    callback = {p_id, p_method, p_user_data};
}

void LIPhysicsBody::perform_callback() {
    LBodyState *dbs = LBodyState::singleton;
    dbs->body = this;
    Variant v = dbs;
    Object *obj = ObjectDB::get_instance(callback.id);
    if (!obj) {
        set_integration_callback(0, StringName());
    } else {
        const Variant *vp[2] = { &v, &callback.user_data };

        Variant::CallError ce;
        int argc = (callback.user_data.get_type() == Variant::NIL) ? 1 : 2;
        obj->call(callback.method, vp, argc, ce);
    }
}
