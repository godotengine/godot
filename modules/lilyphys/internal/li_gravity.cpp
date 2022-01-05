//
// Created by amara on 17/11/2021.
//

#include "li_gravity.h"

#include "../lilyphys_server.h"
#include "core/project_settings.h"

void LIGravity::update_force(LIPhysicsBody* p_body, real_t p_delta) {
    if (!p_body->has_finite_mass()) {
        return;
    }
    p_body->add_force(Vector3(ProjectSettings::get_singleton()->get_setting("lilyphys/forces/gravity")) * p_body->get_mass());
}

Vector3 LIGravity::get_gravity() const {
    return Vector3(ProjectSettings::get_singleton()->get_setting("lilyphys/forces/gravity"));
}
