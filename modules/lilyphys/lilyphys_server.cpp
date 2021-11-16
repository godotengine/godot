//
// Created by amara on 19/10/2021.
//

#include "lilyphys_server.h"
#include "l_body_state.h"

LilyphysServer *LilyphysServer::singleton = nullptr;

void LilyphysServer::_bind_methods() {

}

LilyphysServer *LilyphysServer::get_singleton() {
    return singleton;
}

LilyphysServer::LilyphysServer() {
    singleton = this;
}

void LilyphysServer::init() {
    print_line("LilyphysServer has been created!");
    LBodyState::singleton = memnew(LBodyState);
}

void LilyphysServer::finish() {
    memdelete(LBodyState::singleton);
}

void LilyphysServer::step(float p_step) {
    if (!active) {
        return;
    }

    // Do the integration step for all physics bodies.
    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
        LIPhysicsBody *object = body_owner.get(e->get());
        Transform trans = object->get_transform();
        //trans.origin.y += p_step;
        trans.rotate(Vector3(0, 1, 0), p_step * 0.5 * Math_PI);
        object->set_transform(trans);
    }

    // Send the physics state to the nodes.
    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
        LIPhysicsBody *object = body_owner.get(e->get());
        object->perform_callback();
    }
}

RID LilyphysServer::create_physics_body() {
    LIPhysicsBody* object = memnew(LIPhysicsBody);
    RID rid = body_owner.make_rid(object);
    bodies.insert(rid);
    object->set_self(rid);
    return rid;
}

void LilyphysServer::set_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method,
                                              const Variant &p_user_data) {
    LIPhysicsBody *body = body_owner.get(p_body);
    ERR_FAIL_COND(!body);
    body->set_integration_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(0), p_method, p_user_data);
}
