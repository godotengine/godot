/*************************************************************************/
/*  physics.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/physics.h"

#include "core/engine.h"
#include "core/message_queue.h"
#include "scene/main/scene_tree.h"

Physics *Physics::singleton = NULL;

Physics::Physics() {
	ERR_FAIL_COND(singleton != NULL);
	singleton = this;
}

Physics::~Physics() {
	singleton = NULL;
}

Physics *Physics::get_singleton() {
	return singleton;
}

void Physics::_bind_methods() {
	ClassDB::bind_method(D_METHOD("simulate"), &Physics::simulate);
}

void Physics::simulate() {
	MessageQueue::get_singleton()->flush();

	// Copy from Nodes to the objects inside the physics server.
	SceneTree::get_singleton()->flush_transform_notifications();

	// Run the physics tick.
	int physics_fps = Engine::get_singleton()->get_iterations_per_second();
	float frame_slice = 1.0 / physics_fps;
	float time_scale = Engine::get_singleton()->get_time_scale();
	PhysicsServer::get_singleton()->step(frame_slice * time_scale);
	Physics2DServer::get_singleton()->step(frame_slice * time_scale);

	MessageQueue::get_singleton()->flush();

	// Update the colliding bodies and areas.
	PhysicsServer::get_singleton()->flush_queries();
	Physics2DServer::get_singleton()->flush_queries();
}
