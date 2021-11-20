//
// Created by amara on 16/11/2021.
//

#ifndef LILYPHYS_LI_PHYSICS_BODY_H
#define LILYPHYS_LI_PHYSICS_BODY_H

#include "li_collision_object.h"
#include "core/math/transform.h"
#include "core/object.h"

enum LPhysicsBodyPropertyType {
    TRANSFORM,
    INVERSE_MASS,
    LINEAR_DAMPING,
    ANGULAR_DAMPING,
    VELOCITY,
    ACCELERATION,
    ROTATION,
    INV_INERTIA_TENSOR
};

class LIPhysicsBody : public LICollisionObject {
private:
    Transform transform;
    real_t inverse_mass;
    real_t linear_damping;
    real_t angular_damping;
    Vector3 velocity;
    Vector3 acceleration;
    Vector3 last_acceleration;
    // Rotation is the angular velocity, not the orientation.
    Vector3 rotation;
    Vector3 force_accum;
    Vector3 torque_accum;
    Basis inv_inertia_tensor;
    struct IntegrationCallback {
        ObjectID id;
        StringName method;
        Variant user_data;
    };
    IntegrationCallback callback;
public:
    LIPhysicsBody();
    void set_transform(const Transform& p_transform) { transform = p_transform; };
    void set_integration_callback(ObjectID p_id, const StringName& p_method, const Variant& p_user_data);
    void perform_callback();
    bool has_finite_mass() const;
    void add_force(const Vector3 &p_force);
    void integrate(float p_step);
    void clear_accumulators();

    const Transform &get_transform() const;
    real_t get_inverse_mass() const;
    real_t get_mass() const;
    real_t get_linear_damping() const;
    real_t get_angular_damping() const;
    const Vector3 &get_velocity() const;
    const Vector3 &get_acceleration() const;
    const Vector3 &get_last_acceleration() const;
    const Vector3 &get_rotation() const;
    const Vector3 &get_force_accum() const;
    const Vector3 &get_torque_accum() const;
    const Basis &get_inv_inertia_tensor() const;

    void set_inertia_tensor(const Basis& tensor);
    void set_mass(const real_t &mass);
    void set_property(LPhysicsBodyPropertyType type, const Variant& value);
};


#endif //LILYPHYS_LI_PHYSICS_BODY_H
