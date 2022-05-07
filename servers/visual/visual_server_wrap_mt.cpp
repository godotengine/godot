/*************************************************************************/
/*  visual_server_wrap_mt.cpp                                            */
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

#include "visual_server_wrap_mt.h"
#include "core/os/os.h"
#include "core/project_settings.h"

void VisualServerWrapMT::thread_exit() {
	exit.set();
}

void VisualServerWrapMT::thread_draw(bool p_swap_buffers, double frame_step) {
	if (!draw_pending.decrement()) {
		visual_server->draw(p_swap_buffers, frame_step);
	}
}

void VisualServerWrapMT::thread_flush() {
	draw_pending.decrement();
}

void VisualServerWrapMT::_thread_callback(void *_instance) {
	VisualServerWrapMT *vsmt = reinterpret_cast<VisualServerWrapMT *>(_instance);

	vsmt->thread_loop();
}

void VisualServerWrapMT::thread_loop() {
	server_thread = Thread::get_caller_id();

	OS::get_singleton()->make_rendering_thread();

	visual_server->init();

	exit.clear();
	draw_thread_up.set();
	while (!exit.is_set()) {
		// flush commands one by one, until exit is requested
		command_queue.wait_and_flush_one();
	}

	command_queue.flush_all(); // flush all

	visual_server->finish();
}

/* EVENT QUEUING */

void VisualServerWrapMT::sync() {
	if (create_thread) {
		draw_pending.increment();
		command_queue.push_and_sync(this, &VisualServerWrapMT::thread_flush);
	} else {
		command_queue.flush_all(); //flush all pending from other threads
	}
}

void VisualServerWrapMT::draw(bool p_swap_buffers, double frame_step) {
	if (create_thread) {
		draw_pending.increment();
		command_queue.push(this, &VisualServerWrapMT::thread_draw, p_swap_buffers, frame_step);
	} else {
		visual_server->draw(p_swap_buffers, frame_step);
	}
}

void VisualServerWrapMT::init() {
	if (create_thread) {
		print_verbose("VisualServerWrapMT: Creating render thread");
		OS::get_singleton()->release_rendering_thread();
		if (create_thread) {
			thread.start(_thread_callback, this);
			print_verbose("VisualServerWrapMT: Starting render thread");
		}
		while (!draw_thread_up.is_set()) {
			OS::get_singleton()->delay_usec(1000);
		}
		print_verbose("VisualServerWrapMT: Finished render thread");
	} else {
		visual_server->init();
	}
}

void VisualServerWrapMT::finish() {
	if (create_thread) {
		command_queue.push(this, &VisualServerWrapMT::thread_exit);
		thread.wait_to_finish();
	} else {
		visual_server->finish();
	}

	texture_free_cached_ids();
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
	lightmap_capture_free_cached_ids();
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
	room_free_cached_ids();
	roomgroup_free_cached_ids();
	portal_free_cached_ids();
	ghost_free_cached_ids();
	occluder_instance_free_cached_ids();
	occluder_resource_free_cached_ids();
}

void VisualServerWrapMT::set_use_vsync_callback(bool p_enable) {
	singleton_mt->call_set_use_vsync(p_enable);
}

VisualServerWrapMT *VisualServerWrapMT::singleton_mt = nullptr;

VisualServerWrapMT::VisualServerWrapMT(VisualServer *p_contained, bool p_create_thread) :
		command_queue(p_create_thread) {
	singleton_mt = this;
	OS::switch_vsync_function = set_use_vsync_callback; //as this goes to another thread, make sure it goes properly

	visual_server = p_contained;
	create_thread = p_create_thread;
	pool_max_size = GLOBAL_GET("memory/limits/multithreaded_server/rid_pool_prealloc");

	if (!p_create_thread) {
		server_thread = Thread::get_caller_id();
	} else {
		server_thread = 0;
	}
}

VisualServerWrapMT::~VisualServerWrapMT() {
	memdelete(visual_server);
	//finish();
}
