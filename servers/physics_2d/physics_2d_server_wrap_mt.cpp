/*************************************************************************/
/*  physics_2d_server_wrap_mt.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "physics_2d_server_wrap_mt.h"

#include "core/os/os.h"

void Physics2DServerWrapMT::thread_exit() {
	exit.set();
}

void Physics2DServerWrapMT::thread_step(real_t p_delta) {
	physics_2d_server->step(p_delta);
	step_sem.post();
}

void Physics2DServerWrapMT::_thread_callback(void *_instance) {
	Physics2DServerWrapMT *vsmt = reinterpret_cast<Physics2DServerWrapMT *>(_instance);

	vsmt->thread_loop();
}

void Physics2DServerWrapMT::thread_loop() {
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

void Physics2DServerWrapMT::step(real_t p_step) {
	if (create_thread) {
		command_queue.push(this, &Physics2DServerWrapMT::thread_step, p_step);
	} else {
		command_queue.flush_all(); //flush all pending from other threads
		physics_2d_server->step(p_step);
	}
}

void Physics2DServerWrapMT::sync() {
	if (create_thread) {
		if (first_frame) {
			first_frame = false;
		} else {
			step_sem.wait(); //must not wait if a step was not issued
		}
	}

	physics_2d_server->sync();
}

void Physics2DServerWrapMT::flush_queries() {
	physics_2d_server->flush_queries();
}

void Physics2DServerWrapMT::end_sync() {
	physics_2d_server->end_sync();
}

void Physics2DServerWrapMT::init() {
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

void Physics2DServerWrapMT::finish() {
	if (create_thread) {
		command_queue.push(this, &Physics2DServerWrapMT::thread_exit);
		thread.wait_to_finish();
	} else {
		physics_2d_server->finish();
	}

	line_shape_free_cached_ids();
	ray_shape_free_cached_ids();
	segment_shape_free_cached_ids();
	circle_shape_free_cached_ids();
	rectangle_shape_free_cached_ids();
	capsule_shape_free_cached_ids();
	convex_polygon_shape_free_cached_ids();
	concave_polygon_shape_free_cached_ids();

	space_free_cached_ids();
	area_free_cached_ids();
	body_free_cached_ids();
}

Physics2DServerWrapMT::Physics2DServerWrapMT(Physics2DServer *p_contained, bool p_create_thread) :
		command_queue(p_create_thread) {
	physics_2d_server = p_contained;
	create_thread = p_create_thread;

	pool_max_size = GLOBAL_GET("memory/limits/multithreaded_server/rid_pool_prealloc");

	if (!p_create_thread) {
		server_thread = Thread::get_caller_id();
	} else {
		server_thread = 0;
	}

	main_thread = Thread::get_caller_id();
	first_frame = true;
}

Physics2DServerWrapMT::~Physics2DServerWrapMT() {
	memdelete(physics_2d_server);
	//finish();
}
