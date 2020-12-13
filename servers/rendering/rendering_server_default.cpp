/*************************************************************************/
/*  rendering_server_default.cpp                                         */
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

#include "rendering_server_default.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"
#include "core/templates/sort_array.h"
#include "renderer_canvas_cull.h"
#include "renderer_scene_cull.h"
#include "rendering_server_globals.h"

// careful, these may run in different threads than the visual server

int RenderingServerDefault::changes = 0;

/* BLACK BARS */

void RenderingServerDefault::black_bars_set_margins(int p_left, int p_top, int p_right, int p_bottom) {
	black_margin[MARGIN_LEFT] = p_left;
	black_margin[MARGIN_TOP] = p_top;
	black_margin[MARGIN_RIGHT] = p_right;
	black_margin[MARGIN_BOTTOM] = p_bottom;
}

void RenderingServerDefault::black_bars_set_images(RID p_left, RID p_top, RID p_right, RID p_bottom) {
	black_image[MARGIN_LEFT] = p_left;
	black_image[MARGIN_TOP] = p_top;
	black_image[MARGIN_RIGHT] = p_right;
	black_image[MARGIN_BOTTOM] = p_bottom;
}

void RenderingServerDefault::_draw_margins() {
	RSG::canvas_render->draw_window_margins(black_margin, black_image);
};

/* FREE */

void RenderingServerDefault::free(RID p_rid) {
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

void RenderingServerDefault::request_frame_drawn_callback(Object *p_where, const StringName &p_method, const Variant &p_userdata) {
	ERR_FAIL_NULL(p_where);
	FrameDrawnCallbacks fdc;
	fdc.object = p_where->get_instance_id();
	fdc.method = p_method;
	fdc.param = p_userdata;

	frame_drawn_callbacks.push_back(fdc);
}

void RenderingServerDefault::draw(bool p_swap_buffers, double frame_step) {
	//needs to be done before changes is reset to 0, to not force the editor to redraw
	RS::get_singleton()->emit_signal("frame_pre_draw");

	changes = 0;

	RSG::rasterizer->begin_frame(frame_step);

	TIMESTAMP_BEGIN()

	RSG::scene->update(); //update scenes stuff before updating instances

	RSG::storage->update_particles(); //need to be done after instances are updated (colliders and particle transforms), and colliders are rendered

	RSG::scene->render_probes();
	RSG::viewport->draw_viewports();
	RSG::canvas_render->update();

	_draw_margins();
	RSG::rasterizer->end_frame(p_swap_buffers);

	while (frame_drawn_callbacks.front()) {
		Object *obj = ObjectDB::get_instance(frame_drawn_callbacks.front()->get().object);
		if (obj) {
			Callable::CallError ce;
			const Variant *v = &frame_drawn_callbacks.front()->get().param;
			obj->call(frame_drawn_callbacks.front()->get().method, &v, 1, ce);
			if (ce.error != Callable::CallError::CALL_OK) {
				String err = Variant::get_call_error_text(obj, frame_drawn_callbacks.front()->get().method, &v, 1, ce);
				ERR_PRINT("Error calling frame drawn function: " + err);
			}
		}

		frame_drawn_callbacks.pop_front();
	}
	RS::get_singleton()->emit_signal("frame_post_draw");

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
				new_profile.write[i].gpu_msec = float((time_gpu - base_gpu) / 1000) / 1000.0;
				new_profile.write[i].cpu_msec = float(time_cpu - base_cpu) / 1000.0;
				new_profile.write[i].name = RSG::storage->get_captured_timestamp_name(i);
			}
		}

		frame_profile = new_profile;
	}

	frame_profile_frame = RSG::storage->get_captured_timestamps_frame();
}

void RenderingServerDefault::sync() {
}

bool RenderingServerDefault::has_changed() const {
	return changes > 0;
}

void RenderingServerDefault::init() {
	RSG::rasterizer->initialize();
}

void RenderingServerDefault::finish() {
	if (test_cube.is_valid()) {
		free(test_cube);
	}

	RSG::rasterizer->finalize();
}

/* STATUS INFORMATION */

int RenderingServerDefault::get_render_info(RenderInfo p_info) {
	return RSG::storage->get_render_info(p_info);
}

String RenderingServerDefault::get_video_adapter_name() const {
	return RSG::storage->get_video_adapter_name();
}

String RenderingServerDefault::get_video_adapter_vendor() const {
	return RSG::storage->get_video_adapter_vendor();
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

void RenderingServerDefault::set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {
	redraw_request();
	RSG::rasterizer->set_boot_image(p_image, p_color, p_scale, p_use_filter);
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

void RenderingServerDefault::call_set_use_vsync(bool p_enable) {
	DisplayServer::get_singleton()->_set_use_vsync(p_enable);
}

bool RenderingServerDefault::is_low_end() const {
	// FIXME: Commented out when rebasing vulkan branch on master,
	// causes a crash, it seems rasterizer is not initialized yet the
	// first time it's called.
	//return RSG::rasterizer->is_low_end();
	return false;
}

RenderingServerDefault::RenderingServerDefault() {
	RSG::canvas = memnew(RendererCanvasCull);
	RSG::viewport = memnew(RendererViewport);
	RendererSceneCull *sr = memnew(RendererSceneCull);
	RSG::scene = sr;
	RSG::rasterizer = RendererCompositor::create();
	RSG::storage = RSG::rasterizer->get_storage();
	RSG::canvas_render = RSG::rasterizer->get_canvas();
	sr->scene_render = RSG::rasterizer->get_scene();

	frame_profile_frame = 0;

	for (int i = 0; i < 4; i++) {
		black_margin[i] = 0;
		black_image[i] = RID();
	}
}

RenderingServerDefault::~RenderingServerDefault() {
	memdelete(RSG::canvas);
	memdelete(RSG::viewport);
	memdelete(RSG::rasterizer);
	memdelete(RSG::scene);
}
