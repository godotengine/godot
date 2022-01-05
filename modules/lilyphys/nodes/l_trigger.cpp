//
// Created by amara on 05/01/2022.
//

#include "l_trigger.h"

#include "../l_body_state.h"

LTrigger::LTrigger() : LCollisionObject(TYPE_BODY) {
    rid = LilyphysServer::get_singleton()->create_physics_body(false);
    LilyphysServer::get_singleton()->set_integration_callback(rid, this, "_state_changed");
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::INVERSE_MASS, 0.0f);
    LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::TRIGGER, true);
}

void LTrigger::_state_changed(Object *p_state) {
    LBodyState *state = Object::cast_to<LBodyState>(p_state);
    ERR_FAIL_COND_MSG(!state, "Method '_state_changed' must receive a valid LBodyState object as argument");

    set_ignore_transform_notification(true);
    set_global_transform(state->get_transform());
    set_ignore_transform_notification(false);

    Array collisions = LilyphysServer::get_singleton()->get_body_collisions(get_rid());
    Set<RID> detected_collisions;
    for (int i = 0; i < collisions.size(); i++) {
        LCollision* collision = Object::cast_to<LCollision>(collisions[i]);
        ERR_FAIL_COND(!collision)
        // Get the RID and path.
        RID other = collision->get_body_0();
        NodePath other_path = collision->get_body0_path();
        if (other == get_rid()) {
            other = collision->get_body_1();
            other_path = collision->get_body1_path();
        }
        // Check if the body just entered.
        if (!body_register.has(other)) {
            emit_signal("body_entered", other_path);
        }
        body_register[other] = other_path;
        detected_collisions.insert(other);
    }
    for (Map<RID, NodePath>::Element* e = body_register.front(); e; e = e->next()) {
        if (!detected_collisions.find(e->key())) {
            emit_signal("body_exited", e->get());
            body_register.erase(e);
        }
    }
}

void LTrigger::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_state_changed"), &LTrigger::_state_changed);
    ADD_SIGNAL(MethodInfo("body_entered", PropertyInfo(Variant::NODE_PATH, "other")));
    ADD_SIGNAL(MethodInfo("body_exited", PropertyInfo(Variant::NODE_PATH, "other")));
}

void LTrigger::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE:
        case NOTIFICATION_TRANSFORM_CHANGED:
            LilyphysServer::get_singleton()->set_physics_body_parameter(rid, LPhysicsBodyPropertyType::TRANSFORM, get_global_transform());
            break;
    }
}