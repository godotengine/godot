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

void LilyphysServer::clear_registry_for_body(RID p_body) {
    for(List<Registration>::Element *E=registry.front(); E; E=E->next()) {
        if (E->get().body == p_body) {
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
        case SHAPE_SPHERE:
            shape = memnew(LISphereShape);
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
        while (!shape->get_owners().empty()) {
            LICollisionObject *so = shape->get_owners().front()->key();
            so->remove_shape(shape->get_self());
        }
        shape_owner.free(p_rid);
        shapes.erase(p_rid);
        memdelete(shape);
    }
    else if (body_owner.owns(p_rid)) {
        LIPhysicsBody* body = body_owner.get(p_rid);
        clear_registry_for_body(p_rid);
        body->clear_shapes();
        body_owner.free(p_rid);
        bodies.erase(p_rid);
        memdelete(body);
    }
    else {
        ERR_FAIL_MSG("Invalid rid trying to be freed.");
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

void LilyphysServer::find_all_active_bodies() {
    active_bodies.clear();
    for (Set<RID>::Element *E = bodies.front(); E; E = E->next()) {
        if (body_owner.get(E->get())->is_active()) {
            active_bodies.insert(E->get());
        }
    }
}

void LilyphysServer::copy_all_current_state_to_old() {
    for (Set<RID>::Element *E = bodies.front(); E; E = E->next()) {
        LIPhysicsBody* body = body_owner.get(E->get());
        if (body->is_active() && body->get_velocity_changed()) {
            body->copy_current_state_to_old();
        }
    }
}

void LilyphysServer::restore_all_state() {
    for (Set<RID>::Element *E = bodies.front(); E; E = E->next()) {
        LIPhysicsBody* body = body_owner.get(E->get());
        if (body->is_active() && body->get_velocity_changed()) {
            body->restore_old_state();
        }
    }
}

void LilyphysServer::clear_all_accumulators() {
    for (Set<RID>::Element *E = bodies.front(); E; E = E->next()) {
        body_owner.get(E->get())->clear_accumulators();
    }
}

void LilyphysServer::detect_all_collisions(float p_step) {
    copy_all_current_state_to_old();

    integrate_all_bodies(p_step);
    update_all_positions(p_step);

    collisions.clear();
    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
        body_owner.get(e->get())->clear_collisions();
    }

    // We should really...REALLY get broad phase collision...... this hurts my SOUL !
    Set<RID>::Element *e = bodies.front();
    detect_collision(e->get());

//    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
//        detect_collision(e->get());
//    }

    restore_all_state();
}

void LilyphysServer::detect_collision(RID p_body) {
    LIPhysicsBody* body = body_owner.get(p_body);
    for (Set<RID>::Element *f = bodies.front(); f; f = f->next()) {
        if (p_body != f->get()) {
            List<RID> results = solver.check_collision(body, body_owner.get(f->get()), collision_result_owner);
            body->add_collisions(results);
            body_owner.get(f->get())->add_collisions(results);
            body_owner.get(f->get())->add_collisions(results);
            for (int i = 0; i < results.size(); i++) {
                collisions.push_back(results[i]);
            }
        }
    }
}

// Gets run once per collision, so we precalculate everything possible.
void LilyphysServer::preprocess_collision(float p_step, CollisionResult* p_result) {
    LIPhysicsBody* body0 = body_owner.get(p_result->body0);
    LIPhysicsBody* body1 = body_owner.get(p_result->body1);

    // Only process collisions between two LIPhysicsBodies, NOT areas!
    if (!body0 || !body1) {
        return;
    }

    p_result->satisfied = false;

    // Normalized direction vector from body0 to body1;
    const Vector3 N = body0->get_transform().origin.direction_to(body1->get_transform().origin);
    const float timescale = p_step * (float)num_penetration_relaxation_timesteps;
    p_result->R0 = p_result->pos - body0->get_transform().origin;
    p_result->R1 = p_result->pos - body1->get_transform().origin;

    if (!body0->has_finite_mass()) {
        p_result->denominator = 0.0f;
    }
    else {
        p_result->denominator = body0->get_inverse_mass() + N.dot(body0->get_global_inv_inertia_tensor().xform(
                p_result->R0.cross(N)).cross(p_result->R0));
    }

    if (body1->has_finite_mass()) {
        p_result->denominator += body1->get_inverse_mass() + N.dot(body1->get_global_inv_inertia_tensor().xform(
                p_result->R1.cross(N)).cross(p_result->R1));
    }

    if (p_result->denominator < FLOAT_TINY) {
        p_result->denominator = FLOAT_TINY;
    }

    if (p_result->depth > allowed_penetration) {
        p_result->min_separation_velocity = (p_result->depth - allowed_penetration) / timescale;
    }
    else {
        float approachScale = -0.1f * (p_result->depth - allowed_penetration) / (FLOAT_TINY + allowed_penetration);
        CLAMP(approachScale, FLOAT_TINY, 1.0f);
        p_result->min_separation_velocity = approachScale * (p_result->depth - allowed_penetration) / MAX(p_step, FLOAT_TINY);
    }
    if (p_result->min_separation_velocity > max_velocity_magnitude) {
        p_result->min_separation_velocity = max_velocity_magnitude;
    }

}

bool LilyphysServer::process_collision(float p_step, CollisionResult* p_result, bool p_first_contact) {
    //p_result->satisfied = true;

    LIPhysicsBody* body0 = body_owner.get(p_result->body0);
    LIPhysicsBody* body1 = body_owner.get(p_result->body1);

    // Only process collisions between two LIPhysicsBodies, NOT areas! Also, no collisions with both immovable objects!
    if (!body0 || !body1 || (!body0->has_finite_mass() && !body1->has_finite_mass())) {
        return false;
    }

    // Make sure the one with a finite mass is body0.
    if (!body0->has_finite_mass()) {
        body1 = body0;
        body0 = body_owner.get(p_result->body1);
        p_result->dir = -p_result->dir;
    }

    if (!body1->has_finite_mass()) {
        //Vector3 R0 = p_result->pos - body0->get_transform().origin;
        Vector3 R0 = body0->get_transform().inverse().xform(p_result->pos);
        Vector3 n = p_result->dir;
        real_t V01 = (body0->get_velocity() + (body0->get_angular_velocity().cross(R0))).dot(n);

        real_t i = ((-(1 + 0.5f) * V01)) /
                (body0->get_inverse_mass() + (body0->get_inv_inertia_tensor().xform(R0.cross(n)).cross(R0)).dot(n));

        body0->set_velocity(body0->get_velocity() + (i * n) / body0->get_mass());
        body0->set_angular_velocity(body0->get_angular_velocity() + body0->get_inv_inertia_tensor().xform(R0.cross(i * n)));
    }

    return true;
}

void LilyphysServer::handle_all_constraints(float p_step, int p_iterations, bool p_force_inelastic) {
    int orig_collision_count = collisions.size();

    // TODO: Prepare all constraints.

    // Prepare all collisions.
    if (p_force_inelastic) {
        for (List<RID>::Element *E = collisions.front(); E; E = E->next()) {
            CollisionResult* result = collision_result_owner.get(E->get());
            ERR_FAIL_COND(!result);
            preprocess_collision(p_step, result);
            result->restitution = 0.0f;
            result->satisfied = false;
        }
    }
    else {
        for (List<RID>::Element *E = collisions.front(); E; E = E->next()) {
            CollisionResult* result = collision_result_owner.get(E->get());
            ERR_FAIL_COND(!result);
            preprocess_collision(p_step, result);
        }
    }

    static int dir = 1;
    for (int step = 0; step < p_iterations; step++) {
        //bool got_one = false;
        int num_collisions = collisions.size();
        dir = !dir;

        for (int i = dir ? 0 : num_collisions - 1; i >= 0 && i < num_collisions; dir ? i++ : i--) {
            CollisionResult* result = collision_result_owner.get(collisions[i]);
            ERR_FAIL_COND(!result);
            //if (!result->satisfied) {
                // Do something different if force inelastic? (Process contact function instead?)
                //got_one |=
                        process_collision(p_step, result, step == 0);
                //result->satisfied = false;
            //}
        }

        // TODO: Apply constraints

        // TODO: Try to activate all frozen bodies.
        try_activate_all_frozen_bodies();

        // Number of collisions might have increased when bodies get activated.
        num_collisions = collisions.size();

        // Preprocess any new collisions.
        if (p_force_inelastic) {
            for (int i = orig_collision_count; i < num_collisions; i++) {
                CollisionResult* result = collision_result_owner.get(collisions[i]);
                ERR_FAIL_COND(!result);
                preprocess_collision(p_step, result);
                result->restitution = 0.0f;
                result->satisfied = false;
            }
        }
        else {
            for (int i = orig_collision_count; i < num_collisions; i++) {
                CollisionResult* result = collision_result_owner.get(collisions[i]);
                ERR_FAIL_COND(!result);
                preprocess_collision(p_step, result);
            }
        }
        orig_collision_count = num_collisions;

//        if (!got_one) {
//            break;
//        }
    }
}

void LilyphysServer::integrate_all_bodies(float p_step) {
    for (Set<RID>::Element *E = active_bodies.front(); E; E = E->next()) {
        LIPhysicsBody* body = body_owner.get(E->get());
        ERR_FAIL_COND(!body);
        if (body->get_velocity_changed()) {
            body->integrate_velocity(p_step);
        }
    }
}

void LilyphysServer::try_freeze_all_bodies() {

}

void LilyphysServer::update_all_positions(float p_step) {
    for (Set<RID>::Element *E = active_bodies.front(); E; E = E->next()) {
        body_owner.get(E->get())->update_position(p_step);
    }
}

void LilyphysServer::perform_all_callbacks() {
    // Send the physics state to the nodes.
    for (Set<RID>::Element *e = bodies.front(); e; e = e->next()) {
        LIPhysicsBody *object = body_owner.get(e->get());
        ERR_FAIL_COND(!object);
        object->perform_callback();
    }
}

void LilyphysServer::step(float p_step) {
    if (!active) {
        return;
    }

    if (step_through && !do_step) {
        return;
    }
    do_step = false;

    free_queue();

    clear_all_accumulators();

    // Update all forces.
    for (List<Registration>::Element *E=registry.front(); E; E=E->next()) {
        generator_owner.get(E->get().generator)->update_force(body_owner.get(E->get().body), p_step);
    }

    find_all_active_bodies();

    detect_all_collisions(p_step);

    handle_all_constraints(p_step, 1, false);

    integrate_all_bodies(p_step);

    //handle_all_constraints(p_step, collision_iterations, true);

    update_all_positions(p_step);

    perform_all_callbacks();


}

void LilyphysServer::try_activate_all_frozen_bodies() {
    for (Set<RID>::Element *E = bodies.front(); E; E = E->next()) {
        // TODO: Set criteria for activating bodies.
        LIPhysicsBody* body = body_owner.get(E->get());
        if (!body->is_active()) {
            activate_body(E->get());
        }
    }
}

void LilyphysServer::activate_body(RID p_body) {
    LIPhysicsBody* body = body_owner.get(p_body);
    if (body->is_active() || !body->has_finite_mass()) {
        return;
    }

    body->set_active(true);
    active_bodies.insert(p_body);

    detect_collision(p_body);
    // TODO: Check if we also shouldn't activate other objects. Doesn't matter now since everything is activated.
}

void LilyphysServer::queue_free_rid(RID p_rid) {
    deletion_queue.insert(p_rid);
}

void LilyphysServer::free_queue() {
    for (Set<RID>::Element *E = deletion_queue.front(); E; E = E->next()) {
        free(E->get());
        deletion_queue.erase(E);
    }
}

void LilyphysServer::set_collision_satisfied(RID p_collision, bool p_satisfied) {
    CollisionResult* result = collision_result_owner.get(p_collision);
    ERR_FAIL_COND(!result);
    result->satisfied = p_satisfied;
}

Array LilyphysServer::get_body_collisions(RID p_body) {
    Array array{};
    LIPhysicsBody* body = body_owner.get(p_body);
    ERR_FAIL_COND_V(!body, array);
    List<RID> collisions = body->get_collisions();
    for (List<RID>::Element *E = collisions.front(); E; E = E->next()) {
        CollisionResult* result = collision_result_owner.get(E->get());
        ERR_FAIL_COND_V(!result, Array{});
        // Sooooo where does this get deleted?
        LCollision* collision = memnew(LCollision);
        collision->init(result->dir, result->pos, result->depth, result->body0, result->body1, result->shape_transform);
        array.push_back(collision);
    }
    return array;
}

/*
 * void LilyphysServer::handle_all_constraints(float p_step, int p_iterations, bool p_force_inelastic) {
    int orig_collision_count = collisions.size();

    // TODO: Prepare all constraints.

    // Prepare all collisions.
    if (p_force_inelastic) {
        for (List<RID>::Element *E = collisions.front(); E; E = E->next()) {
            CollisionResult* result = collision_result_owner.get(E->get());
            ERR_FAIL_COND(!result);
            preprocess_collision(p_step, result);
            result->restitution = 0.0f;
            result->satisfied = false;
        }
    }
    else {
        for (List<RID>::Element *E = collisions.front(); E; E = E->next()) {
            CollisionResult* result = collision_result_owner.get(E->get());
            ERR_FAIL_COND(!result);
            preprocess_collision(p_step, result);
        }
    }

    static int dir = 1;
    for (int step = 0; step < p_iterations; step++) {
        bool got_one = false;
        int num_collisions = collisions.size();
        dir = !dir;

        for (int i = dir ? 0 : num_collisions - 1; i >= 0 && i < num_collisions; dir ? i++ : i--) {
            CollisionResult* result = collision_result_owner.get(collisions[i]);
            ERR_FAIL_COND(!result);
            if (!result->satisfied) {
                // Do something different if force inelastic? (Process contact function instead?)
                got_one |= process_collision(p_step, result, step == 0);
                result->satisfied = false;
            }
        }

        // TODO: Apply constraints

        // TODO: Try to activate all frozen bodies.
        try_activate_all_frozen_bodies();

        // Number of collisions might have increased when bodies get activated.
        num_collisions = collisions.size();

        // Preprocess any new collisions.
        if (p_force_inelastic) {
            for (int i = orig_collision_count; i < num_collisions; i++) {
                CollisionResult* result = collision_result_owner.get(collisions[i]);
                ERR_FAIL_COND(!result);
                preprocess_collision(p_step, result);
                result->restitution = 0.0f;
                result->satisfied = false;
            }
        }
        else {
            for (int i = orig_collision_count; i < num_collisions; i++) {
                CollisionResult* result = collision_result_owner.get(collisions[i]);
                ERR_FAIL_COND(!result);
                preprocess_collision(p_step, result);
            }
        }
        orig_collision_count = num_collisions;

        if (!got_one) {
            break;
        }
    }
}
 */