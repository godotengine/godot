//
// Created by amara on 16/11/2021.
//

#include "l_body_state.h"

LBodyState* LBodyState::singleton = NULL;

Transform LBodyState::get_transform() const {
    return body->get_transform();
}

real_t LBodyState::get_inverse_mass() const {
    return body->get_inverse_mass();
}

real_t LBodyState::get_linear_damping() const {
    return body->get_linear_damping();
}

real_t LBodyState::get_angular_damping() const {
    return body->get_angular_damping();
}

Vector3 LBodyState::get_velocity() const {
    return body->get_velocity();
}

Vector3 LBodyState::get_acceleration() const {
    return body->get_acceleration();
}

Vector3 LBodyState::get_last_acceleration() const {
    return body->get_last_acceleration();
}

Vector3 LBodyState::get_rotation() const {
    return body->get_rotation();
}

Vector3 LBodyState::get_force_accum() const {
    return body->get_force_accum();
}

Vector3 LBodyState::get_torque_accum() const {
    return body->get_torque_accum();
}

Basis LBodyState::get_inv_inertia_tensor() const {
    return body->get_inv_inertia_tensor();
}
