/*************************************************************************/
/*  collision_avoidance_server.cpp                                       */
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

#include "collision_avoidance_server.h"

CollisionAvoidanceServer *CollisionAvoidanceServer::singleton = NULL;

void CollisionAvoidanceServer::_bind_methods() {

    ClassDB::bind_method(D_METHOD("space_create"), &CollisionAvoidanceServer::space_create);

    ClassDB::bind_method(D_METHOD("agent_add", "space"), &CollisionAvoidanceServer::agent_add);

    ClassDB::bind_method(D_METHOD("obstacle_add", "space"), &CollisionAvoidanceServer::obstacle_add);

    ClassDB::bind_method(D_METHOD("free", "object"), &CollisionAvoidanceServer::free);
}

CollisionAvoidanceServer *CollisionAvoidanceServer::get_singleton() {
    return singleton;
}

CollisionAvoidanceServer::CollisionAvoidanceServer() {
    ERR_FAIL_COND(singleton != NULL);
    singleton = this;
}

CollisionAvoidanceServer::~CollisionAvoidanceServer() {
    singleton = NULL;
}

CreateCollisionAvoidanceServerCallback CollisionAvoidanceServerManager::create_callback = NULL;

void CollisionAvoidanceServerManager::set_default_server(CreateCollisionAvoidanceServerCallback p_callback) {
    create_callback = p_callback;
}

CollisionAvoidanceServer *CollisionAvoidanceServerManager::new_default_server() {
    ERR_FAIL_COND_V(create_callback == NULL, NULL);
    return create_callback();
}
