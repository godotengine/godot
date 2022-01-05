//
// Created by amara on 16/11/2021.
//

#ifndef LILYPHYS_LI_PHYSICS_BODY_H
#define LILYPHYS_LI_PHYSICS_BODY_H

#include "li_collision_object.h"
#include "core/math/transform.h"
#include "core/object.h"
#include "core/vector.h"

class CollisionResult;

enum LPhysicsBodyPropertyType {
    TRANSFORM,
    INVERSE_MASS,
    LINEAR_DAMPING,
    ANGULAR_DAMPING,
    VELOCITY,
    ACCELERATION,
    ANGULAR_VELOCITY,
    INV_INERTIA_TENSOR,
    TRIGGER,
    NODE_PATH
};

class LIPhysicsBody : public LICollisionObject {
private:
    Transform old_transform;
    Vector3 old_velocity;
    Vector3 old_angular_velocity;
    Vector3 velocity;
    Vector3 angular_velocity;
    Vector3 aux_velocity;
    Vector3 aux_angular_velocity;

    real_t inverse_mass = 10;
    real_t linear_damping = 0.8;
    real_t angular_damping = 0.8;
    Vector3 acceleration;
    Vector3 last_acceleration;
    Vector3 force_accum;
    Vector3 torque_accum;
    // Deactivation stuff.
    real_t sq_velocity_activity_threshold;
    real_t sq_angular_velocity_activity_threshold;
    Vector3 last_position_for_deactivation;
    Basis last_orientation_for_deactivation;
    real_t sq_delta_pos_threshold;
    real_t sq_delta_orient_threshold;
    real_t inactive_time = 0.0f;
    real_t deactivation_time = 1.0f;

    Basis inv_inertia_tensor;
    Basis global_inverse_inertia_tensor;
    struct IntegrationCallback {
        ObjectID id;
        StringName method;
        Variant user_data;
    };
    IntegrationCallback callback;
    bool active = true;
    bool velocity_changed;
    bool orig_immovable = false;
    bool immovable = false;
    bool should_process_shock = true;
    bool trigger = false;
    List<RID> collisions;
public:
    LIPhysicsBody();
    void set_integration_callback(ObjectID p_id, const StringName& p_method, const Variant& p_user_data);
    void perform_callback();
    bool has_finite_mass() const;
    void add_force(const Vector3 &p_force);
    void add_force_at_point(const Vector3& p_force, const Vector3& p_point);
    void add_force_at_body_point(const Vector3& p_force, const Vector3& p_point);
    void apply_global_impulse(const Vector3& p_impulse, const Vector3& p_delta);
    void apply_negative_global_impulse(const Vector3& p_impulse, const Vector3& p_delta);
    void set_collisions_unsatisfied();
    void integrate_velocity(float p_step);
    void update_position(float p_step);
    void clear_accumulators();
    Vector3 to_global(const Vector3& p_vector) const;
    void copy_current_state_to_old();
    void restore_old_state();
    void recalculate_inertia_tensor();
    void set_shape_transform(size_t p_id, const Transform &p_transform);
    void set_temp_immovable();
    void restore_temp_immovable();
    void set_activity_threshold(real_t p_velocity, real_t p_angular_velocity);
    void set_deactivation_threshold(real_t p_pos_threshold, real_t p_orient_threshold);
    void try_to_freeze(real_t p_step);
    void damp_for_deactivation();

    real_t get_inverse_mass() const;
    real_t get_mass() const;
    real_t get_linear_damping() const;
    real_t get_angular_damping() const;
    const Vector3 &get_velocity() const;
    const Vector3 &get_acceleration() const;
    const Vector3 &get_last_acceleration() const;
    const Vector3 &get_angular_velocity() const;
    const Vector3 &get_aux_velocity() const;
    const Vector3 &get_aux_angular_velocity() const;
    const Vector3 &get_force_accum() const;
    const Vector3 &get_torque_accum() const;
    const Basis &get_inv_inertia_tensor() const;
    const Basis &get_global_inv_inertia_tensor() const;
    bool is_active() const { return active; }
    bool get_velocity_changed() const { return velocity_changed; }
    bool get_immovable() const { return immovable; }
    List<RID> get_collisions() const { return collisions; }
    bool get_should_process_shock() const { return should_process_shock; }
    bool get_should_be_active();
    bool is_trigger() const { return trigger; }

    void clear_velocity_changed() { velocity_changed = false; }
    void set_active(bool p_active) { active = p_active; }
    void add_collisions(const List<RID>& p_collisions);
    void clear_collisions();
    void set_inertia_tensor(const Basis& tensor);
    void set_mass(const real_t &mass);
    void set_property(LPhysicsBodyPropertyType type, const Variant& value);
    void set_velocity(const Vector3& p_velocity);
    void set_aux_velocity(const Vector3& p_velocity);
    void set_angular_velocity(const Vector3& p_velocity);
    void set_aux_angular_velocity(const Vector3& p_velocity);
    void set_immovable(const bool& p_immovable);
    void set_trigger(const bool& p_trigger) { trigger = p_trigger; }
};


#endif //LILYPHYS_LI_PHYSICS_BODY_H
