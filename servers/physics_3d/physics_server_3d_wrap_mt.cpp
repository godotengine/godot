/*************************************************************************/
/*  physics_server_3d_wrap_mt.cpp                                        */
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

#include "physics_server_3d_wrap_mt.h"

#include "core/os/os.h"

void PhysicsServer3DWrapMT::thread_exit() {
	exit = true;
}

void PhysicsServer3DWrapMT::thread_step(real_t p_delta) {
	physics_3d_server->step(p_delta);
	step_sem.post();
}

void PhysicsServer3DWrapMT::_thread_callback(void *_instance) {
	PhysicsServer3DWrapMT *vsmt = reinterpret_cast<PhysicsServer3DWrapMT *>(_instance);

	vsmt->thread_loop();
}

void PhysicsServer3DWrapMT::thread_loop() {
	server_thread = Thread::get_caller_id();

	physics_3d_server->init();

	exit = false;
	step_thread_up = true;
	while (!exit) {
		// flush commands one by one, until exit is requested
		command_queue.wait_and_flush_one();
	}

	command_queue.flush_all(); // flush all

	physics_3d_server->finish();
}

/* EVENT QUEUING */

void PhysicsServer3DWrapMT::step(real_t p_step) {
	if (create_thread) {
		command_queue.push(this, &PhysicsServer3DWrapMT::thread_step, p_step);
	} else {
		command_queue.flush_all(); //flush all pending from other threads
		physics_3d_server->step(p_step);
	}
}

void PhysicsServer3DWrapMT::sync() {
	if (create_thread) {
		if (first_frame) {
			first_frame = false;
		} else {
			step_sem.wait(); //must not wait if a step was not issued
		}
	}
	physics_3d_server->sync();
}

void PhysicsServer3DWrapMT::flush_queries() {
	physics_3d_server->flush_queries();
}

void PhysicsServer3DWrapMT::end_sync() {
	physics_3d_server->end_sync();
}

void PhysicsServer3DWrapMT::init() {
	if (create_thread) {
		//OS::get_singleton()->release_rendering_thread();
		thread.start(_thread_callback, this);
		while (!step_thread_up) {
			OS::get_singleton()->delay_usec(1000);
		}
	} else {
		physics_3d_server->init();
	}
}

void PhysicsServer3DWrapMT::finish() {
	if (thread.is_started()) {
		command_queue.push(this, &PhysicsServer3DWrapMT::thread_exit);
		thread.wait_to_finish();
	} else {
		physics_3d_server->finish();
	}
}

PhysicsServer3DWrapMT::PhysicsServer3DWrapMT(PhysicsServer3D *p_contained, bool p_create_thread) :
		command_queue(p_create_thread) {
	physics_3d_server = p_contained;
	create_thread = p_create_thread;
	step_pending = 0;
	step_thread_up = false;

	pool_max_size = GLOBAL_GET("memory/limits/multithreaded_server/rid_pool_prealloc");

	if (!p_create_thread) {
		server_thread = Thread::get_caller_id();
	} else {
		server_thread = 0;
	}

	main_thread = Thread::get_caller_id();
	first_frame = true;
}

PhysicsServer3DWrapMT::~PhysicsServer3DWrapMT() {
	memdelete(physics_3d_server);
	//finish();
}
