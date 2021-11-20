//
// Created by amara on 19/10/2021.
//

#include "lilyphys_server.h"
#include "l_body_state.h"
#include "core/project_settings.h"
#include "internal/li_physics_body.h"

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
    active = true;
    LBodyState::singleton = memnew(LBodyState);

    // Set initial gravity settings.
    GLOBAL_DEF("lilyphys/forces/gravity", Vector3(0.0f, -9.81f, 0.0f));
    // Create gravity.
    LIGravity *gravity_generator = memnew(LIGravity);
    gravity = generator_owner.make_rid(gravity_generator);
    generators.insert(gravity);
}

void LilyphysServer::finish() {
    memdelete(LBodyState::singleton);
}

void LilyphysServer::step(float p_step) {
    if (!active) {
        return;
    }

    // Clear accumulators on all objects.
    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
        body_owner.get(e->get())->clear_accumulators();
    }

    // Update all forces.
    for (List<Registration>::Element *E=registry.front(); E; E=E->next()) {
        generator_owner.get(E->get().generator)->update_force(body_owner.get(E->get().body), p_step);
    }

    // Do the integration step for all physics bodies.
    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
        LIPhysicsBody *object = body_owner.get(e->get());
        object->integrate(p_step);
        //Transform trans = object->get_transform();
        // SPEEEEEEEEEN
        //trans.rotate(Vector3(0, 1, 0), p_step * 0.5 * Math_PI);
        //object->set_transform(trans);
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
    // Automatically register the default gravity force.
    register_generator(rid, gravity);
    return rid;
}

void LilyphysServer::set_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method,
                                              const Variant &p_user_data) {
    LIPhysicsBody *body = body_owner.get(p_body);
    ERR_FAIL_COND(!body);
    body->set_integration_callback(p_receiver ? p_receiver->get_instance_id() : ObjectID(0), p_method, p_user_data);
}

void LilyphysServer::register_generator(RID p_body, RID p_generator) {
    registry.push_back({p_body, p_generator});
}

void LilyphysServer::unregister_generator(RID p_body, RID p_generator) {
    for(List<Registration>::Element *E=registry.front(); E; E=E->next()) {
        if (E->get().body == p_body && E->get().generator == p_generator) {
            registry.erase(E);
        }
    }
}

void LilyphysServer::clear_registry() {
    for(List<Registration>::Element *E=registry.front(); E; E=E->next()) {
        registry.erase(E);
    }
}

void LilyphysServer::set_physics_body_parameter(RID rid, LPhysicsBodyPropertyType type, const Variant& value) {
    LIPhysicsBody *body = body_owner.get(rid);
    ERR_FAIL_COND(!body);
    body->set_property(type, value);
}
