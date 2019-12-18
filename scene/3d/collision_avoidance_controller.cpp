/*************************************************************************/
/*  collision_avoidance_controller.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "collision_avoidance_controller.h"

#include "scene/3d/physics_body.h"
#include "servers/collision_avoidance_server.h"

void CollisionAvoidanceController::_bind_methods() {
}

void CollisionAvoidanceController::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_READY: {
            ERR_FAIL_COND(agent.is_valid());
            PhysicsBody *parent = Object::cast_to<PhysicsBody>(get_parent());
            if (parent) {
                agent = CollisionAvoidanceServer::get_singleton()->agent_add(parent->get_world()->get_collision_avoidance_space());
            }
        } break;
        case NOTIFICATION_EXIT_TREE: {
            if (agent.is_valid()) {
                CollisionAvoidanceServer::get_singleton()->free(agent);
                agent = RID();
            }
        } break;
    }
}

CollisionAvoidanceController::CollisionAvoidanceController() :
        agent(RID()) {
}

String CollisionAvoidanceController::get_configuration_warning() const {
    if (!Object::cast_to<PhysicsBody>(get_parent())) {
        return TTR("CollisionAvoidanceController only serves to provide collision avoidance to a physics object. Please only use it as a child of RigidBody, KinematicBody.");
    }

    return String();
}
