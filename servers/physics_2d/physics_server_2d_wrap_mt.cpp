/*************************************************************************/
/*  physics_server_2d_wrap_mt.cpp                                        */
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

#include "physics_server_2d_wrap_mt.h"

#include "core/os/os.h"

void PhysicsServer2DWrapMT::thread_exit() {
	exit.set();
}

void PhysicsServer2DWrapMT::thread_step(real_t p_delta) {
	physics_2d_server->step(p_delta);
	step_sem.post();
}

void PhysicsServer2DWrapMT::_thread_callback(void *_instance) {
	PhysicsServer2DWrapMT *vsmt = reinterpret_cast<PhysicsServer2DWrapMT *>(_instance);

	vsmt->thread_loop();
}

void PhysicsServer2DWrapMT::thread_loop() {
	server_thread = Thread::get_caller_id();

	physics_2d_server->init();

	exit.clear();
	step_thread_up.set();
	while (!exit.is_set()) {
		// flush commands one by one, until exit is requested
		command_queue.wait_and_flush_one();
	}

	command_queue.flush_all(); // flush all

	physics_2d_server->finish();
}

/* EVENT QUEUING */

void PhysicsServer2DWrapMT::step(real_t p_step) {
	if (create_thread) {
		command_queue.push(this, &PhysicsServer2DWrapMT::thread_step, p_step);
	} else {
		command_queue.flush_all(); //flush all pending from other threads
		physics_2d_server->step(p_step);
	}
}

void PhysicsServer2DWrapMT::sync() {
	if (create_thread) {
		if (first_frame) {
			first_frame = false;
		} else {
			step_sem.wait(); //must not wait if a step was not issued
		}
	}
	physics_2d_server->sync();
}

void PhysicsServer2DWrapMT::flush_queries() {
	physics_2d_server->flush_queries();
}

void PhysicsServer2DWrapMT::end_sync() {
	physics_2d_server->end_sync();
}

void PhysicsServer2DWrapMT::init() {
	if (create_thread) {
		//OS::get_singleton()->release_rendering_thread();
		thread.start(_thread_callback, this);
		while (!step_thread_up.is_set()) {
			OS::get_singleton()->delay_usec(1000);
		}
	} else {
		physics_2d_server->init();
	}
}

void PhysicsServer2DWrapMT::finish() {
	if (thread.is_started()) {
		command_queue.push(this, &PhysicsServer2DWrapMT::thread_exit);
		thread.wait_to_finish();
	} else {
		physics_2d_server->finish();
	}
}

PhysicsServer2DWrapMT::PhysicsServer2DWrapMT(PhysicsServer2D *p_contained, bool p_create_thread) :
		command_queue(p_create_thread) {
	physics_2d_server = p_contained;
	create_thread = p_create_thread;
	step_pending = 0;

	pool_max_size = GLOBAL_GET("memory/limits/multithreaded_server/rid_pool_prealloc");

	if (!p_create_thread) {
		server_thread = Thread::get_caller_id();
	} else {
		server_thread = 0;
	}

	main_thread = Thread::get_caller_id();
	first_frame = true;
}

PhysicsServer2DWrapMT::~PhysicsServer2DWrapMT() {
	memdelete(physics_2d_server);
	//finish();
}
