/*************************************************************************/
/*  visual_server_wrap_mt.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "visual_server_wrap_mt.h"
#include "os/os.h"
#include "project_settings.h"

void VisualServerWrapMT::thread_exit() {

	exit = true;
}

void VisualServerWrapMT::thread_draw() {

	if (!atomic_decrement(&draw_pending)) {

		visual_server->draw();
	}
}

void VisualServerWrapMT::thread_flush() {

	atomic_decrement(&draw_pending);
}

void VisualServerWrapMT::_thread_callback(void *_instance) {

	VisualServerWrapMT *vsmt = reinterpret_cast<VisualServerWrapMT *>(_instance);

	vsmt->thread_loop();
}

void VisualServerWrapMT::thread_loop() {

	server_thread = Thread::get_caller_id();

	OS::get_singleton()->make_rendering_thread();

	visual_server->init();

	exit = false;
	draw_thread_up = true;
	while (!exit) {
		// flush commands one by one, until exit is requested
		command_queue.wait_and_flush_one();
	}

	command_queue.flush_all(); // flush all

	visual_server->finish();
}

/* EVENT QUEUING */

void VisualServerWrapMT::sync() {

	if (create_thread) {

		atomic_increment(&draw_pending);
		command_queue.push_and_sync(this, &VisualServerWrapMT::thread_flush);
	} else {

		command_queue.flush_all(); //flush all pending from other threads
	}
}

void VisualServerWrapMT::draw(bool p_swap_buffers) {

	if (create_thread) {

		atomic_increment(&draw_pending);
		command_queue.push(this, &VisualServerWrapMT::thread_draw);
	} else {

		visual_server->draw(p_swap_buffers);
	}
}

void VisualServerWrapMT::init() {

	if (create_thread) {

		print_line("CREATING RENDER THREAD");
		OS::get_singleton()->release_rendering_thread();
		if (create_thread) {
			thread = Thread::create(_thread_callback, this);
			print_line("STARTING RENDER THREAD");
		}
		while (!draw_thread_up) {
			OS::get_singleton()->delay_usec(1000);
		}
		print_line("DONE RENDER THREAD");
	} else {

		visual_server->init();
	}
}

void VisualServerWrapMT::finish() {

	if (thread) {

		command_queue.push(this, &VisualServerWrapMT::thread_exit);
		Thread::wait_to_finish(thread);
		memdelete(thread);

		thread = NULL;
	} else {
		visual_server->finish();
	}

	texture_free_cached_ids();
	shader_free_cached_ids();
	material_free_cached_ids();
	mesh_free_cached_ids();
	multimesh_free_cached_ids();
	immediate_free_cached_ids();
	skeleton_free_cached_ids();
	directional_light_free_cached_ids();
	omni_light_free_cached_ids();
	spot_light_free_cached_ids();
	reflection_probe_free_cached_ids();
	gi_probe_free_cached_ids();
	particles_free_cached_ids();
	camera_free_cached_ids();
	viewport_free_cached_ids();
	environment_free_cached_ids();
	scenario_free_cached_ids();
	instance_free_cached_ids();
	canvas_free_cached_ids();
	canvas_item_free_cached_ids();
	canvas_light_occluder_free_cached_ids();
	canvas_occluder_polygon_free_cached_ids();
}

void VisualServerWrapMT::set_use_vsync_callback(bool p_enable) {

	singleton_mt->call_set_use_vsync(p_enable);
}

VisualServerWrapMT *VisualServerWrapMT::singleton_mt = NULL;

VisualServerWrapMT::VisualServerWrapMT(VisualServer *p_contained, bool p_create_thread) :
		command_queue(p_create_thread) {

	singleton_mt = this;
	OS::switch_vsync_function = set_use_vsync_callback; //as this goes to another thread, make sure it goes properly

	visual_server = p_contained;
	create_thread = p_create_thread;
	thread = NULL;
	draw_pending = 0;
	draw_thread_up = false;
	alloc_mutex = Mutex::create();
	pool_max_size = GLOBAL_GET("memory/limits/multithreaded_server/rid_pool_prealloc");

	if (!p_create_thread) {
		server_thread = Thread::get_caller_id();
	} else {
		server_thread = 0;
	}
}

VisualServerWrapMT::~VisualServerWrapMT() {

	memdelete(visual_server);
	memdelete(alloc_mutex);
	//finish();
}
