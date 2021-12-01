//
// Created by amara on 19/10/2021.
//

#include "lilyphys_server.h"

#include "l_body_state.h"
#include "internal/li_physics_body.h"
#include "internal/li_spring.h"

#include "core/project_settings.h"

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
    }

    // We should really...REALLY get broad phase collision...... this hurts my SOUL !
    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
        for (Set<RID>::Element *f = bodies.front(); f; f = f->next()) {
            if (e->get() != f->get()) {
                CollisionResult result = solver.check_collision(body_owner.get(e->get()), body_owner.get(f->get()));
//                if (result.intersect) {
//                    LIPhysicsBody *object = body_owner.get(e->get());
//                }
            }
        }
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
    // Spring test code.
//    if (bodies.size() == 2) {
//        LISpring* spring_gen = memnew(LISpring(Vector3(0.5f, 0.5f, 0.5f), rid, Vector3(0.5f, 0.5f, 0.5f), 0.1f, 2.0f));
//        spring = generator_owner.make_rid(spring_gen);
//        generators.insert(spring);
//        register_generator(body_owner.get(bodies.front()->get())->get_self(), spring);
//    }
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

LIPhysicsBody *LilyphysServer::get_physics_body(RID p_rid) {
    LIPhysicsBody *body = body_owner.get(p_rid);
    ERR_FAIL_COND_V_MSG(!body, nullptr, "RID does not correspond with valid physics body.")
    return body;
}

RID LilyphysServer::create_shape(LShapeType p_type) {
    LIShape* shape;
    switch (p_type) {
        case SHAPE_BOX:
            shape = memnew(LIBoxShape);
            break;
    }
    ERR_FAIL_COND_V(!shape, RID());
    RID id = shape_owner.make_rid(shape);
    shape->set_self(id);

    return id;
}

void LilyphysServer::shape_set_data(RID p_id, const Variant &p_data) {
    LIShape* shape = shape_owner.get(p_id);
    ERR_FAIL_COND(!shape);
    shape->set_data(p_data);
}

Variant LilyphysServer::shape_get_data(RID p_id) {
    LIShape* shape = shape_owner.get(p_id);
    ERR_FAIL_COND_V_MSG(!shape, Variant(), "RID does not correspond with valid shape.");
    return shape->get_data();
}

size_t LilyphysServer::physics_body_add_shape(RID p_body, RID p_shape) {
    LIPhysicsBody* body = body_owner.get(p_body);
    ERR_FAIL_COND_V(!shape_owner.owns(p_shape), 0);
    ERR_FAIL_COND_V(!body, 0);
    return body->add_shape(p_shape);
}

void LilyphysServer::physics_body_remove_shape(RID p_body, size_t p_id) {
    LIPhysicsBody* body = body_owner.get(p_body);
    ERR_FAIL_COND(!body);
    body->remove_shape(p_id);
}

void LilyphysServer::physics_body_shape_set_disabled(RID p_body, size_t p_id, bool p_disabled) {
    LIPhysicsBody* body = body_owner.get(p_body);
    ERR_FAIL_COND(!body);
    body->set_shape_disabled(p_id, p_disabled);
}

void LilyphysServer::physics_body_shape_set_transform(RID p_body, size_t p_id, const Transform &p_transform) {
    LIPhysicsBody* body = body_owner.get(p_body);
    ERR_FAIL_COND(!body);
    body->set_shape_transform(p_id, p_transform);
}

bool LilyphysServer::physics_body_get_shape_exists(RID p_body, size_t p_id) {
    LIPhysicsBody* body = body_owner.get(p_body);
    ERR_FAIL_COND_V(!body, false);
    return body->get_shape_exists(p_id);
}

void LilyphysServer::free(RID p_rid) {
    if (shape_owner.owns(p_rid)) {
        LIShape* shape = shape_owner.get(p_rid);
        while (shape->get_owners().size()) {
            LICollisionObject *so = shape->get_owners().front()->key();
            so->remove_shape(shape->get_self());
        }
        shape_owner.free(p_rid);
        memdelete(shape);
    }
    else if (body_owner.owns(p_rid)) {
        LIPhysicsBody* body = body_owner.get(p_rid);
        body->clear_shapes();
        body_owner.free(p_rid);
        memdelete(body);
    }
    else {
        ERR_FAIL_MSG("Invalid rid trying to be freed.")
    }
}

void LilyphysServer::shape_remove_owner(RID p_shape, LICollisionObject *owner) {
    LIShape* shape = shape_owner.get(p_shape);
    ERR_FAIL_COND(!shape);
    shape->remove_owner(owner);
}

void LilyphysServer::shape_add_owner(RID p_shape, LICollisionObject *owner) {
    LIShape* shape = shape_owner.get(p_shape);
    ERR_FAIL_COND(!shape);
    shape->add_owner(owner);
}

Vector3 LilyphysServer::shape_get_support(RID p_id, Vector3 p_direction) {
    LIShape* shape = shape_owner.get(p_id);
    ERR_FAIL_COND_V(!shape, Vector3(0, 0, 0));
    return shape->get_support(p_direction);
}
