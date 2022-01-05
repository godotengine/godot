//
// Created by amara on 05/01/2022.
//

#include "l_static_body.h"

#include "../l_body_state.h"

LStaticBody::LStaticBody() : LCollisionObject(TYPE_BODY) {
    rid = LilyphysServer::get_singleton()->create_physics_body(false);
    LilyphysServer::get_singleton()->set_integration_callback(rid, this, "_state_changed");
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::INVERSE_MASS, 0.0f);
}

void LStaticBody::_state_changed(Object *p_state) {
    LBodyState *state = Object::cast_to<LBodyState>(p_state);
    ERR_FAIL_COND_MSG(!state, "Method '_state_changed' must receive a valid LBodyState object as argument");

    set_ignore_transform_notification(true);
    set_global_transform(state->get_transform());
    set_ignore_transform_notification(false);
}

void LStaticBody::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_state_changed"), &LStaticBody::_state_changed);
}

void LStaticBody::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE:
        case NOTIFICATION_TRANSFORM_CHANGED:
            LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::TRANSFORM, get_global_transform());
            break;
    }
}