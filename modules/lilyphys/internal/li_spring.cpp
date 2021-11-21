//
// Created by amara on 21/11/2021.
//

#include "li_spring.h"

#include "../lilyphys_server.h"

void LISpring::update_force(LIPhysicsBody *p_body, real_t p_delta) {
    LIPhysicsBody* other_body = LilyphysServer::get_singleton()->get_physics_body(other);
    if (!other_body) {
        return;
    }
    Vector3 lws = p_body->to_global(connection_point);
    Vector3 ows = other_body->to_global(other_connection_point);
    Vector3 force = lws - ows;
    real_t magnitude = force.length();
    magnitude = abs(magnitude - rest_length);
    magnitude *= spring_constant;
    force = force.normalized();
    force *= -magnitude;
    p_body->add_force_at_point(force, lws);
}
