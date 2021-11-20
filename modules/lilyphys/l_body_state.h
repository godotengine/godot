//
// Created by amara on 16/11/2021.
//

#ifndef LILYPHYS_L_BODY_STATE_H
#define LILYPHYS_L_BODY_STATE_H

#include "core/object.h"
#include "internal/li_physics_body.h"

class LBodyState : public Object {
    GDCLASS(LBodyState, Object);
public:
    friend LIPhysicsBody;
    static LBodyState* singleton;
    Transform get_transform() const;
    real_t get_inverse_mass() const;
    real_t get_linear_damping() const;
    real_t get_angular_damping() const;
    Vector3 get_velocity() const;
    Vector3 get_acceleration() const;
    Vector3 get_last_acceleration() const;
    Vector3 get_rotation() const;
    Vector3 get_force_accum() const;
    Vector3 get_torque_accum() const;
    Basis get_inv_inertia_tensor() const;
private:
    LIPhysicsBody* body;
};


#endif //LILYPHYS_L_BODY_STATE_H
