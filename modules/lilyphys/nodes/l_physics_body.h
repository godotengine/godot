//
// Created by amara on 16/11/2021.
//

#ifndef LILYPHYS_L_PHYSICS_BODY_H
#define LILYPHYS_L_PHYSICS_BODY_H

#include "l_collision_object.h"

class LPhysicsBody : public LCollisionObject {
GDCLASS(LPhysicsBody, LCollisionObject);
private:
    real_t inverse_mass = 10;
    real_t linear_damping = 0.8;
    real_t angular_damping = 0.8;
    Vector3 velocity{};
    Vector3 acceleration{};
    Vector3 angular_velocity{};
    Basis inverse_inertia_tensor{};
    bool ignore_mass = false;
    bool start_deactivated = false;
protected:
    static void _bind_methods();
    void _notification(int p_what);
public:
    LPhysicsBody();
    void _state_changed(Object *p_state);

    real_t get_inverse_mass() const;
    void set_inverse_mass(real_t p_inverse_mass);
    real_t get_linear_damping() const;
    void set_linear_damping(real_t p_linear_damping);
    real_t get_angular_damping() const;
    void set_angular_damping(real_t p_angular_damping);
    const Vector3 &get_velocity() const;
    void set_velocity(const Vector3 &p_velocity);
    const Vector3 &get_acceleration() const;
    void set_acceleration(const Vector3 &p_acceleration);
    const Vector3 &get_angular_velocity() const;
    void set_angular_velocity(const Vector3 &p_angular_velocity);
    const Basis &get_inverse_inertia_tensor() const;
    void set_inverse_inertia_tensor(const Basis &p_inverse_inertia_tensor);
    void set_mass(real_t p_mass);
    real_t get_mass() const;
    void set_start_deactivated(const bool& p_start_deactivated) { start_deactivated = p_start_deactivated; }
    bool get_start_deactivated() const { return start_deactivated; }
};


#endif //LILYPHYS_L_PHYSICS_BODY_H
