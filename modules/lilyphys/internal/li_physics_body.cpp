//
// Created by amara on 16/11/2021.
//

#include "li_physics_body.h"

#include "../l_body_state.h"
#include "../lilyphys_server.h"

#include "core/math/math_funcs.h"
#include "../l_collision_solver.h"

LIPhysicsBody::LIPhysicsBody() : LICollisionObject(TYPE_BODY) {
    velocity_changed = true;
    set_activity_threshold(0.5f, 30.0f);
    set_deactivation_threshold(0.2f, 0.2f);
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

const Vector3 &LIPhysicsBody::get_angular_velocity() const {
    return angular_velocity;
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

bool LIPhysicsBody::has_finite_mass() const {
    return !(Math::is_equal_approx(get_inverse_mass(), 0) || immovable || trigger);
}

void LIPhysicsBody::add_force(const Vector3 &p_force) {
    force_accum += p_force;
    velocity_changed = true;
}

void LIPhysicsBody::integrate_velocity(float p_step) {
    last_acceleration = acceleration;
    last_acceleration += force_accum * inverse_mass;

    Basis iit_world = transform.get_basis() * inv_inertia_tensor;
    Vector3 angular_acceleration = iit_world.xform(torque_accum);

    velocity += last_acceleration * p_step;
    angular_velocity += angular_acceleration * p_step;

    // This appears to be how GDPhysics does it.
    // I think this might be more correct than doing pow(linear_damping, p_step) instead.
    real_t damp = 1.0f - p_step * linear_damping;
    real_t angular_damp = 1.0f - p_step * angular_damping;

    velocity *= damp;
    angular_velocity *= angular_damp;
}

void LIPhysicsBody::update_position(float p_step) {
    transform.origin += velocity * p_step + aux_velocity * p_step;
    transform.basis *= Basis(angular_velocity * p_step) * Basis(aux_angular_velocity * p_step);
    transform.orthonormalize();
    global_inverse_inertia_tensor = transform.basis * inv_inertia_tensor * transform.basis.transposed();
}

void LIPhysicsBody::clear_accumulators() {
    force_accum = {0, 0, 0};
    torque_accum = {0, 0, 0};
}

void LIPhysicsBody::set_inertia_tensor(const Basis& tensor) {
    inv_inertia_tensor = tensor.inverse();
}

void LIPhysicsBody::set_mass(const real_t& mass) {
    inverse_mass = 1.0f / mass;
}

real_t LIPhysicsBody::get_mass() const {
    return 1.0f / inverse_mass;
}

void LIPhysicsBody::set_property(const LPhysicsBodyPropertyType type, const Variant& value) {
    switch (type) {
        case LPhysicsBodyPropertyType::ACCELERATION:
            acceleration = value;
            break;
        case LPhysicsBodyPropertyType::ANGULAR_DAMPING:
            angular_damping = value;
            break;
        case LPhysicsBodyPropertyType::INVERSE_MASS:
            inverse_mass = value;
            break;
        case LPhysicsBodyPropertyType::INV_INERTIA_TENSOR:
            inv_inertia_tensor = value;
            break;
        case LPhysicsBodyPropertyType::LINEAR_DAMPING:
            linear_damping = value;
            break;
        case LPhysicsBodyPropertyType::ANGULAR_VELOCITY:
            angular_velocity = value;
            break;
        case LPhysicsBodyPropertyType::TRANSFORM:
            transform = value;
            break;
        case LPhysicsBodyPropertyType::VELOCITY:
            velocity = value;
            break;
        case LPhysicsBodyPropertyType::TRIGGER:
            trigger = value;
            break;
        case LPhysicsBodyPropertyType::NODE_PATH:
            node_path = value;
            break;
    }
}

void LIPhysicsBody::add_force_at_point(const Vector3 &p_force, const Vector3 &p_point) {
    Vector3 point = p_point;
    point -= transform.origin;
    force_accum += p_force;
    torque_accum += point.cross(p_force);
    velocity_changed = true;
}

void LIPhysicsBody::add_force_at_body_point(const Vector3 &p_force, const Vector3 &p_point) {
    add_force_at_point(p_force, to_global(p_point));
}

Vector3 LIPhysicsBody::to_global(const Vector3 &p_vector) const {
    return transform.xform(p_vector);
}

void LIPhysicsBody::copy_current_state_to_old() {
    old_transform = transform;
    old_velocity = velocity;
    old_angular_velocity = angular_velocity;
}

void LIPhysicsBody::restore_old_state() {
    transform = old_transform;
    velocity = old_velocity;
    angular_velocity = old_angular_velocity;
}

void LIPhysicsBody::add_collisions(const List<RID> &p_collisions) {
    for (int i = 0; i < p_collisions.size(); i++) {
        collisions.push_back(p_collisions[i]);
    }
}

void LIPhysicsBody::clear_collisions() {
    collisions.clear();
}

const Basis &LIPhysicsBody::get_global_inv_inertia_tensor() const {
    return global_inverse_inertia_tensor;
}

void LIPhysicsBody::apply_global_impulse(const Vector3 &p_impulse, const Vector3 &p_delta) {
    if (!has_finite_mass()) {
        return;
    }
    velocity += p_impulse * inverse_mass;
    angular_velocity += global_inverse_inertia_tensor.xform(p_delta.cross(p_impulse));
    velocity_changed = true;
}

void LIPhysicsBody::apply_negative_global_impulse(const Vector3 &p_impulse, const Vector3 &p_delta) {
    if (!has_finite_mass()) {
        return;
    }
    velocity += p_impulse * -inverse_mass;
    angular_velocity -= global_inverse_inertia_tensor.xform(p_delta.cross(p_impulse));
    velocity_changed = true;
}

void LIPhysicsBody::set_collisions_unsatisfied() {
    for (int i = 0; i < collisions.size(); i++) {
        LilyphysServer::get_singleton()->set_collision_satisfied(collisions[i], false);
    }
}

void LIPhysicsBody::set_velocity(const Vector3 &p_velocity) {
    velocity = p_velocity;
    velocity_changed = true;
}

void LIPhysicsBody::set_angular_velocity(const Vector3 &p_velocity) {
    angular_velocity = p_velocity;
    velocity_changed = true;
}

void LIPhysicsBody::recalculate_inertia_tensor() {
    // Get one of the shapes we've got and use it to calculate an inertia tensor.....yeah.
    if (shapes.empty() || !has_finite_mass()) {
        Basis tensor{};
        tensor.set_zero();
        set_inertia_tensor(tensor);
        return;
    }

    RID shape = shapes.back()->get().shape;
    Basis tensor = LilyphysServer::get_singleton()->shape_get_inertia_tensor(shape, get_mass());
    inv_inertia_tensor = tensor;
}

void LIPhysicsBody::set_shape_transform(size_t p_id, const Transform &p_transform) {
    LICollisionObject::set_shape_transform(p_id, p_transform);
    recalculate_inertia_tensor();
}

void LIPhysicsBody::set_immovable(const bool &p_immovable) {
    immovable = p_immovable;
}

void LIPhysicsBody::set_aux_velocity(const Vector3 &p_velocity) {
    aux_velocity = p_velocity;
    velocity_changed = true;
}

void LIPhysicsBody::set_aux_angular_velocity(const Vector3 &p_velocity) {
    aux_velocity = p_velocity;
    velocity_changed = true;
}

const Vector3 &LIPhysicsBody::get_aux_velocity() const {
    return aux_velocity;
}

const Vector3 &LIPhysicsBody::get_aux_angular_velocity() const {
    return aux_angular_velocity;
}

void LIPhysicsBody::set_temp_immovable() {
    orig_immovable = immovable;
    immovable = true;
}

void LIPhysicsBody::restore_temp_immovable() {
    immovable = orig_immovable;
}

bool LIPhysicsBody::get_should_be_active() {
    //print_line(vformat("Velocity: %s, %s", velocity.length_squared(), Math::rad2deg(angular_velocity.length_squared())));
    //print_line(vformat("Thresholds: %s, %s", sq_velocity_activity_threshold, sq_angular_velocity_activity_threshold));
    return (
            (((velocity.length_squared() > sq_velocity_activity_threshold) ||
            (angular_velocity.length_squared() > sq_angular_velocity_activity_threshold))
            && !trigger)
            );
}

void LIPhysicsBody::set_activity_threshold(real_t p_velocity, real_t p_angular_velocity) {
    sq_velocity_activity_threshold = Math::pow(p_velocity, 2.0f);
    sq_angular_velocity_activity_threshold = Math::pow(Math::deg2rad(p_angular_velocity), 2.0f);
}

void LIPhysicsBody::try_to_freeze(real_t p_step) {
    if (!has_finite_mass() || !is_active()) {
        return;
    }

    if ((transform.origin - last_position_for_deactivation).length_squared() > sq_delta_pos_threshold) {
        last_position_for_deactivation = transform.origin;
        inactive_time = 0.0f;
        return;
    }

    Basis delta_mat = transform.basis - last_orientation_for_deactivation;
    if ( (delta_mat.get_column(0).length_squared() > sq_delta_orient_threshold) ||
         (delta_mat.get_column(1).length_squared() > sq_delta_orient_threshold) ||
         (delta_mat.get_column(2).length_squared() > sq_delta_orient_threshold) ) {
        last_orientation_for_deactivation = transform.basis;
        inactive_time = 0.0f;
        return;
    }

    if (get_should_be_active()) {
        return;
    }

    inactive_time += p_step;
    //print_line(vformat("Adding to inactive time. (%s)", inactive_time));

    if (inactive_time > deactivation_time) {
        last_orientation_for_deactivation = transform.basis;
        last_position_for_deactivation = transform.origin;
        set_active(false);
    }
}

void LIPhysicsBody::set_deactivation_threshold(real_t p_pos_threshold, real_t p_orient_threshold) {
    sq_delta_pos_threshold = Math::pow(p_pos_threshold, 2.0f);
    sq_delta_orient_threshold = Math::pow(p_orient_threshold, 2.0f);
}

void LIPhysicsBody::damp_for_deactivation() {
    real_t frac = inactive_time / deactivation_time;
    const real_t r = 0.5;
    if (frac < r) {
        return;
    }

    real_t scale = 1.0f - ((frac - r) / 1.0f - r);
    scale = CLAMP(scale, 0.0f, 1.0f);
    velocity *= scale;
    angular_velocity *= scale;
}
