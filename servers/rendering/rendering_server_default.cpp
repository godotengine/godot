/*************************************************************************/
/*  rendering_server_default.cpp                                         */
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

#include "rendering_server_default.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"
#include "core/templates/sort_array.h"
#include "renderer_canvas_cull.h"
#include "renderer_scene_cull.h"
#include "rendering_server_globals.h"

// careful, these may run in different threads than the rendering server

int RenderingServerDefault::changes = 0;

/* FREE */

void RenderingServerDefault::_free(RID p_rid) {
	if (unlikely(p_rid.is_null())) {
		return;
	}
	if (RSG::storage->free(p_rid)) {
		return;
	}
	if (RSG::canvas->free(p_rid)) {
		return;
	}
	if (RSG::viewport->free(p_rid)) {
		return;
	}
	if (RSG::scene->free(p_rid)) {
		return;
	}
}

/* EVENT QUEUING */

void RenderingServerDefault::request_frame_drawn_callback(const Callable &p_callable) {
	frame_drawn_callbacks.push_back(p_callable);
}

void RenderingServerDefault::_draw(bool p_swap_buffers, double frame_step) {
	//needs to be done before changes is reset to 0, to not force the editor to redraw
	RS::get_singleton()->emit_signal(SNAME("frame_pre_draw"));

	changes = 0;

	RSG::rasterizer->begin_frame(frame_step);

	TIMESTAMP_BEGIN()

	uint64_t time_usec = OS::get_singleton()->get_ticks_usec();

	RSG::scene->update(); //update scenes stuff before updating instances

	frame_setup_time = double(OS::get_singleton()->get_ticks_usec() - time_usec) / 1000.0;

	RSG::storage->update_particles(); //need to be done after instances are updated (colliders and particle transforms), and colliders are rendered

	RSG::scene->render_probes();

	RSG::viewport->draw_viewports();
	RSG::canvas_render->update();

	RSG::rasterizer->end_frame(p_swap_buffers);

	RSG::canvas->update_visibility_notifiers();
	RSG::scene->update_visibility_notifiers();

	while (frame_drawn_callbacks.front()) {
		Callable c = frame_drawn_callbacks.front()->get();
		Variant result;
		Callable::CallError ce;
		c.call(nullptr, 0, result, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			String err = Variant::get_callable_error_text(c, nullptr, 0, ce);
			ERR_PRINT("Error calling frame drawn function: " + err);
		}

		frame_drawn_callbacks.pop_front();
	}
	RS::get_singleton()->emit_signal(SNAME("frame_post_draw"));

	if (RSG::storage->get_captured_timestamps_count()) {
		Vector<FrameProfileArea> new_profile;
		if (RSG::storage->capturing_timestamps) {
			new_profile.resize(RSG::storage->get_captured_timestamps_count());
		}

		uint64_t base_cpu = RSG::storage->get_captured_timestamp_cpu_time(0);
		uint64_t base_gpu = RSG::storage->get_captured_timestamp_gpu_time(0);
		for (uint32_t i = 0; i < RSG::storage->get_captured_timestamps_count(); i++) {
			uint64_t time_cpu = RSG::storage->get_captured_timestamp_cpu_time(i);
			uint64_t time_gpu = RSG::storage->get_captured_timestamp_gpu_time(i);

			String name = RSG::storage->get_captured_timestamp_name(i);

			if (name.begins_with("vp_")) {
				RSG::viewport->handle_timestamp(name, time_cpu, time_gpu);
			}

			if (RSG::storage->capturing_timestamps) {
				new_profile.write[i].gpu_msec = double((time_gpu - base_gpu) / 1000) / 1000.0;
				new_profile.write[i].cpu_msec = double(time_cpu - base_cpu) / 1000.0;
				new_profile.write[i].name = RSG::storage->get_captured_timestamp_name(i);
			}
		}

		frame_profile = new_profile;
	}

	frame_profile_frame = RSG::storage->get_captured_timestamps_frame();

	if (print_gpu_profile) {
		if (print_frame_profile_ticks_from == 0) {
			print_frame_profile_ticks_from = OS::get_singleton()->get_ticks_usec();
		}
		double total_time = 0.0;

		for (int i = 0; i < frame_profile.size() - 1; i++) {
			String name = frame_profile[i].name;
			if (name[0] == '<' || name[0] == '>') {
				continue;
			}

			double time = frame_profile[i + 1].gpu_msec - frame_profile[i].gpu_msec;

			if (name[0] != '<' && name[0] != '>') {
				if (print_gpu_profile_task_time.has(name)) {
					print_gpu_profile_task_time[name] += time;
				} else {
					print_gpu_profile_task_time[name] = time;
				}
			}
		}

		if (frame_profile.size()) {
			total_time = frame_profile[frame_profile.size() - 1].gpu_msec;
		}

		uint64_t ticks_elapsed = OS::get_singleton()->get_ticks_usec() - print_frame_profile_ticks_from;
		print_frame_profile_frame_count++;
		if (ticks_elapsed > 1000000) {
			print_line("GPU PROFILE (total " + rtos(total_time) + "ms): ");

			float print_threshold = 0.01;
			for (OrderedHashMap<String, float>::Element E = print_gpu_profile_task_time.front(); E; E = E.next()) {
				double time = E.value() / double(print_frame_profile_frame_count);
				if (time > print_threshold) {
					print_line("\t-" + E.key() + ": " + rtos(time) + "ms");
				}
			}
			print_gpu_profile_task_time.clear();
			print_frame_profile_ticks_from = OS::get_singleton()->get_ticks_usec();
			print_frame_profile_frame_count = 0;
		}
	}

	RSG::storage->update_memory_info();
}

double RenderingServerDefault::get_frame_setup_time_cpu() const {
	return frame_setup_time;
}

bool RenderingServerDefault::has_changed() const {
	return changes > 0;
}

void RenderingServerDefault::_init() {
	RSG::rasterizer->initialize();
}

void RenderingServerDefault::_finish() {
	if (test_cube.is_valid()) {
		free(test_cube);
	}

	RSG::rasterizer->finalize();
}

void RenderingServerDefault::init() {
	if (create_thread) {
		print_verbose("RenderingServerWrapMT: Creating render thread");
		DisplayServer::get_singleton()->release_rendering_thread();
		if (create_thread) {
			thread.start(_thread_callback, this);
			print_verbose("RenderingServerWrapMT: Starting render thread");
		}
		while (!draw_thread_up.is_set()) {
			OS::get_singleton()->delay_usec(1000);
		}
		print_verbose("RenderingServerWrapMT: Finished render thread");
	} else {
		_init();
	}
}

void RenderingServerDefault::finish() {
	if (create_thread) {
		command_queue.push(this, &RenderingServerDefault::_thread_exit);
		thread.wait_to_finish();
	} else {
		_finish();
	}
}

/* STATUS INFORMATION */

uint64_t RenderingServerDefault::get_rendering_info(RenderingInfo p_info) {
	if (p_info == RENDERING_INFO_TOTAL_OBJECTS_IN_FRAME) {
		return RSG::viewport->get_total_objects_drawn();
	} else if (p_info == RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME) {
		return RSG::viewport->get_total_vertices_drawn();
	} else if (p_info == RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME) {
		return RSG::viewport->get_total_draw_calls_used();
	}
	return RSG::storage->get_rendering_info(p_info);
}

String RenderingServerDefault::get_video_adapter_name() const {
	return RSG::storage->get_video_adapter_name();
}

String RenderingServerDefault::get_video_adapter_vendor() const {
	return RSG::storage->get_video_adapter_vendor();
}

RenderingDevice::DeviceType RenderingServerDefault::get_video_adapter_type() const {
	return RSG::storage->get_video_adapter_type();
}

void RenderingServerDefault::set_frame_profiling_enabled(bool p_enable) {
	RSG::storage->capturing_timestamps = p_enable;
}

uint64_t RenderingServerDefault::get_frame_profile_frame() {
	return frame_profile_frame;
}

Vector<RenderingServer::FrameProfileArea> RenderingServerDefault::get_frame_profile() {
	return frame_profile;
}

/* TESTING */

void RenderingServerDefault::set_boot_image(const Ref<Image> &p_image, const Color &p_color, RenderingServer::SplashStretchMode p_stretch_mode, bool p_use_filter) {
	redraw_request();
	RSG::rasterizer->set_boot_image(p_image, p_color, p_stretch_mode, p_use_filter);
}

void RenderingServerDefault::set_default_clear_color(const Color &p_color) {
	RSG::viewport->set_default_clear_color(p_color);
}

bool RenderingServerDefault::has_feature(Features p_feature) const {
	return false;
}

void RenderingServerDefault::sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) {
	RSG::scene->sdfgi_set_debug_probe_select(p_position, p_dir);
}

void RenderingServerDefault::set_print_gpu_profile(bool p_enable) {
	RSG::storage->capturing_timestamps = p_enable;
	print_gpu_profile = p_enable;
}

RID RenderingServerDefault::get_test_cube() {
	if (!test_cube.is_valid()) {
		test_cube = _make_test_cube();
	}
	return test_cube;
}

bool RenderingServerDefault::has_os_feature(const String &p_feature) const {
	return RSG::storage->has_os_feature(p_feature);
}

void RenderingServerDefault::set_debug_generate_wireframes(bool p_generate) {
	RSG::storage->set_debug_generate_wireframes(p_generate);
}

bool RenderingServerDefault::is_low_end() const {
	// FIXME: Commented out when rebasing vulkan branch on master,
	// causes a crash, it seems rasterizer is not initialized yet the
	// first time it's called.
	//return RSG::rasterizer->is_low_end();
	return false;
}

void RenderingServerDefault::_thread_exit() {
	exit.set();
}

void RenderingServerDefault::_thread_draw(bool p_swap_buffers, double frame_step) {
	if (!draw_pending.decrement()) {
		_draw(p_swap_buffers, frame_step);
	}
}

void RenderingServerDefault::_thread_flush() {
	draw_pending.decrement();
}

void RenderingServerDefault::_thread_callback(void *_instance) {
	RenderingServerDefault *vsmt = reinterpret_cast<RenderingServerDefault *>(_instance);

	vsmt->_thread_loop();
}

void RenderingServerDefault::_thread_loop() {
	server_thread = Thread::get_caller_id();

	DisplayServer::get_singleton()->make_rendering_thread();

	_init();

	draw_thread_up.set();
	while (!exit.is_set()) {
		// flush commands one by one, until exit is requested
		command_queue.wait_and_flush();
	}

	command_queue.flush_all(); // flush all

	_finish();
}

/* EVENT QUEUING */

void RenderingServerDefault::sync() {
	if (create_thread) {
		draw_pending.increment();
		command_queue.push_and_sync(this, &RenderingServerDefault::_thread_flush);
	} else {
		command_queue.flush_all(); //flush all pending from other threads
	}
}

void RenderingServerDefault::draw(bool p_swap_buffers, double frame_step) {
	if (create_thread) {
		draw_pending.increment();
		command_queue.push(this, &RenderingServerDefault::_thread_draw, p_swap_buffers, frame_step);
	} else {
		_draw(p_swap_buffers, frame_step);
	}
}

RenderingServerDefault::RenderingServerDefault(bool p_create_thread) :
		command_queue(p_create_thread) {
	create_thread = p_create_thread;

	if (!p_create_thread) {
		server_thread = Thread::get_caller_id();
	} else {
		server_thread = 0;
	}

	RSG::threaded = p_create_thread;
	RSG::canvas = memnew(RendererCanvasCull);
	RSG::viewport = memnew(RendererViewport);
	RendererSceneCull *sr = memnew(RendererSceneCull);
	RSG::scene = sr;
	RSG::rasterizer = RendererCompositor::create();
	RSG::storage = RSG::rasterizer->get_storage();
	RSG::canvas_render = RSG::rasterizer->get_canvas();
	sr->set_scene_render(RSG::rasterizer->get_scene());

	frame_profile_frame = 0;
}

RenderingServerDefault::~RenderingServerDefault() {
	memdelete(RSG::canvas);
	memdelete(RSG::viewport);
	memdelete(RSG::rasterizer);
	memdelete(RSG::scene);
}
