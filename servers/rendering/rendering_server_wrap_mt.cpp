/*************************************************************************/
/*  rendering_server_wrap_mt.cpp                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rendering_server_wrap_mt.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "servers/display_server.h"

void RenderingServerWrapMT::thread_exit() {
	exit = true;
}

void RenderingServerWrapMT::thread_draw(bool p_swap_buffers, double frame_step) {
	if (!atomic_decrement(&draw_pending)) {
		rendering_server->draw(p_swap_buffers, frame_step);
	}
}

void RenderingServerWrapMT::thread_flush() {
	atomic_decrement(&draw_pending);
}

void RenderingServerWrapMT::_thread_callback(void *_instance) {
	RenderingServerWrapMT *vsmt = reinterpret_cast<RenderingServerWrapMT *>(_instance);

	vsmt->thread_loop();
}

void RenderingServerWrapMT::thread_loop() {
	server_thread = Thread::get_caller_id();

	DisplayServer::get_singleton()->make_rendering_thread();

	rendering_server->init();

	exit = false;
	draw_thread_up = true;
	while (!exit) {
		// flush commands one by one, until exit is requested
		command_queue.wait_and_flush_one();
	}

	command_queue.flush_all(); // flush all

	rendering_server->finish();
}

/* EVENT QUEUING */

void RenderingServerWrapMT::sync() {
	if (create_thread) {
		atomic_increment(&draw_pending);
		command_queue.push_and_sync(this, &RenderingServerWrapMT::thread_flush);
	} else {
		command_queue.flush_all(); //flush all pending from other threads
	}
}

void RenderingServerWrapMT::draw(bool p_swap_buffers, double frame_step) {
	if (create_thread) {
		atomic_increment(&draw_pending);
		command_queue.push(this, &RenderingServerWrapMT::thread_draw, p_swap_buffers, frame_step);
	} else {
		rendering_server->draw(p_swap_buffers, frame_step);
	}
}

void RenderingServerWrapMT::init() {
	if (create_thread) {
		print_verbose("RenderingServerWrapMT: Creating render thread");
		DisplayServer::get_singleton()->release_rendering_thread();
		if (create_thread) {
			thread = Thread::create(_thread_callback, this);
			print_verbose("RenderingServerWrapMT: Starting render thread");
		}
		while (!draw_thread_up) {
			OS::get_singleton()->delay_usec(1000);
		}
		print_verbose("RenderingServerWrapMT: Finished render thread");
	} else {
		rendering_server->init();
	}
}

void RenderingServerWrapMT::finish() {
	sky_free_cached_ids();
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
	lightmap_free_cached_ids();
	particles_free_cached_ids();
	particles_collision_free_cached_ids();
	camera_free_cached_ids();
	viewport_free_cached_ids();
	environment_free_cached_ids();
	camera_effects_free_cached_ids();
	scenario_free_cached_ids();
	instance_free_cached_ids();
	canvas_free_cached_ids();
	canvas_item_free_cached_ids();
	canvas_light_occluder_free_cached_ids();
	canvas_occluder_polygon_free_cached_ids();

	if (thread) {
		command_queue.push(this, &RenderingServerWrapMT::thread_exit);
		Thread::wait_to_finish(thread);
		memdelete(thread);

		thread = nullptr;
	} else {
		rendering_server->finish();
	}
}

void RenderingServerWrapMT::set_use_vsync_callback(bool p_enable) {
	singleton_mt->call_set_use_vsync(p_enable);
}

RenderingServerWrapMT *RenderingServerWrapMT::singleton_mt = nullptr;

RenderingServerWrapMT::RenderingServerWrapMT(RenderingServer *p_contained, bool p_create_thread) :
		command_queue(p_create_thread) {
	singleton_mt = this;
	DisplayServer::switch_vsync_function = set_use_vsync_callback; //as this goes to another thread, make sure it goes properly

	rendering_server = p_contained;
	create_thread = p_create_thread;
	thread = nullptr;
	draw_pending = 0;
	draw_thread_up = false;
	pool_max_size = GLOBAL_GET("memory/limits/multithreaded_server/rid_pool_prealloc");

	if (!p_create_thread) {
		server_thread = Thread::get_caller_id();
	} else {
		server_thread = 0;
	}
}

RenderingServerWrapMT::~RenderingServerWrapMT() {
	memdelete(rendering_server);
	//finish();
}
