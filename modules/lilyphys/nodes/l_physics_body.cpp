//
// Created by amara on 16/11/2021.
//

#include "l_physics_body.h"

#include "../lilyphys_server.h"
#include "../l_body_state.h"

LPhysicsBody::LPhysicsBody() {
    rid = LilyphysServer::get_singleton()->create_physics_body();
    LilyphysServer::get_singleton()->set_integration_callback(rid, this, "_state_changed");
}

void LPhysicsBody::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_state_changed"), &LPhysicsBody::_state_changed);
}

void LPhysicsBody::_state_changed(Object *p_state) {
    LBodyState *state = Object::cast_to<LBodyState>(p_state);
    ERR_FAIL_COND_MSG(!state, "Method '_state_changed' must receive a valid LBodyState object as argument");

    set_ignore_transform_notification(true);
    set_global_transform(state->get_transform());
    set_ignore_transform_notification(false);
    linear_velocity = state->get_velocity();
    angular_velocity = state->get_velocity();
}

void LPhysicsBody::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE:
        case NOTIFICATION_TRANSFORM_CHANGED:
        LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::TRANSFORM, get_global_transform());
        break;
    }
}
