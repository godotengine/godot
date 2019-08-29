/*************************************************************************/
/*  editor_live_view.cpp                                                 */
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

#include "editor/editor_live_view.h"
#include "core/os/shared_memory.h"
#include "editor/editor_scale.h"
#include "editor/script_editor_debugger.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"

constexpr int HEADER_SIZE = 16; // Enough for metadata and for 64-bit alignment

void EditorLiveView::_update_processing_state() {

	bool editor_focus_ok = editor_focused || !stop_while_unfocused;
	bool can_update = editor_focus_ok && visible;

	bool was_processing = is_processing();
	set_process(running && can_update);

	if (was_processing != is_processing()) {
		can_request = is_processing();
	}

	float alpha = running && !can_update ? 0.5f : 1.0f;
	texture_rect->set_self_modulate(Color(1, 1, 1, alpha));

	if (running && !editor_focus_ok) {

		label->set_text(TTR("Waiting for editor focus"));
		label->set_visible(true);
	} else {

		label->set_visible(false);
	}
}

void EditorLiveView::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_PROCESS: {

			if (!can_request) {
				break;
			}
			if (!fb_data->is_open()) {
				break;
			}

			uint32_t time = OS::get_singleton()->get_ticks_msec();
			if (time - last_frame_time < min_frame_interval) {
				break;
			}

			int sx = texture_rect->get_size().x;
			int sy = texture_rect->get_size().y;
			if (sx == 0 || sy == 0) {
				break;
			}

			ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
			if (sed) {
				Array msg;
				msg.push_back("request_framebuffer");
				msg.push_back(sx);
				msg.push_back(sy);
				if (sed->send_message(msg)) {
					last_frame_time = time;
					can_request = false;
				}
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {

			visible = is_visible_in_tree();
			_update_processing_state();

		} break;

		case NOTIFICATION_WM_FOCUS_IN: {

			editor_focused = true;
			_update_processing_state();

		} break;

		case NOTIFICATION_WM_FOCUS_OUT: {

			editor_focused = false;
			_update_processing_state();

		} break;
	}
}

void EditorLiveView::start() {

	stop_while_unfocused = EDITOR_GET("run/live_view/stop_while_editor_unfocused");

	// We need this because, although the editor is about to lose its focus due to the
	// game being about to run, it's currently focused and that would make auto-switch
	// kick in even if stop-while-editor-unfocused is enabled. Therefore, we need to
	// know if the editor being focused is because it regained focus
	editor_focused = false;

	int max_fps = EDITOR_GET("run/live_view/fps_limit");
	min_frame_interval = max_fps == 0 ? 0 : 1000 / max_fps;
	last_frame_time = 0;

	texture_rect->set_texture(Ref<Texture>());

	fb_data->open();

	running = true;
	_update_processing_state();
}

void EditorLiveView::stop() {

	fb_data->close();

	running = false;
	_update_processing_state();
}

void EditorLiveView::refresh() {

	if (!is_processing()) {
		return;
	}
	if (!fb_data->is_open()) {
		return;
	}

	bool could_refresh = false;

	const uint32_t *data = reinterpret_cast<uint32_t *>(fb_data->begin_access());
	if (data && data != SharedMemory::UNSIZED) {

		int width = data[0];
		int height = data[1];

		// Check if the other side couldn't send image data
		if (width && height) {

			Image::Format format = (Image::Format)data[2];

			bool recreate = !fb_img.is_valid() ||
							fb_img->get_width() != width ||
							fb_img->get_height() != height ||
							fb_img->get_format() != format;
			if (recreate) {
				fb_img.instance();
				fb_img->create(width, height, false, format);
			}

			{
				PoolByteArray::Write w = fb_img->get_data_ref().write();
				memcpy(w.ptr(), reinterpret_cast<const uint8_t *>(data) + HEADER_SIZE, fb_img->get_data_ref().size());
			}

			Ref<ImageTexture> tex = texture_rect->get_texture();
			if (recreate || !tex.is_valid()) {
				tex.instance();
				tex->create(width, height, format, Texture::FLAG_FILTER | Texture::FLAG_VIDEO_SURFACE);
				texture_rect->set_texture(tex);
			}

			tex->set_data(fb_img);
			could_refresh = true;
		}

		fb_data->end_access();
	}

	if (!could_refresh) {
		texture_rect->set_texture(Ref<Texture>());
	}

	can_request = true;
}

EditorLiveView::EditorLiveView() :
		stop_while_unfocused(false),
		visible(true),
		editor_focused(true),
		running(false),
		can_request(false),
		min_frame_interval(0),
		last_frame_time(0) {

	fb_data = SharedMemory::create("godot_live_view");

	VBoxContainer *vb = this;

	vb->set_custom_minimum_size(Size2(200 * EDSCALE, 0));

	Label *title = memnew(Label);
	title->set_text(TTR("Live View:"));
	vb->add_child(title);

	texture_rect = memnew(TextureRect);
	texture_rect->set_h_size_flags(SIZE_EXPAND_FILL);
	texture_rect->set_v_size_flags(SIZE_EXPAND_FILL);
	texture_rect->set_expand(true);
	texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	texture_rect->set_flip_v(true);
	vb->add_child(texture_rect);

	label = memnew(Label);
	label->set_anchors_preset(PRESET_WIDE);
	label->set_h_size_flags(SIZE_EXPAND_FILL);
	label->set_v_size_flags(SIZE_EXPAND_FILL);
	label->set_align(Label::ALIGN_CENTER);
	label->set_valign(Label::VALIGN_CENTER);
	label->set_autowrap(true);
	texture_rect->add_child(label);
}

EditorLiveView::~EditorLiveView() {

	memdelete(fb_data);
}

void LiveViewDebugHelper::handle_request_framebuffer(int p_target_w, int p_target_h) {

	uint32_t *fb_data_mem = reinterpret_cast<uint32_t *>(fb_data->begin_access());
	if (!fb_data_mem) {
		return;
	}

	Ref<ViewportTexture> vp_texture = SceneTree::get_singleton()->get_root()->get_texture();
	int vp_w = vp_texture->get_width();
	int vp_h = vp_texture->get_height();

	bool can_transfer = p_target_w > 0 && p_target_h > 0 && vp_w > 0 && vp_h > 0;
	if (!can_transfer) {

		// At least reply with zeroes so the recipient knows it can request again

		if (fb_data_mem == SharedMemory::UNSIZED || fb_data->get_size() < HEADER_SIZE) {
			fb_data_mem = reinterpret_cast<uint32_t *>(fb_data->set_size(HEADER_SIZE));
			if (!fb_data_mem) {
				return;
			}
		}

		fb_data_mem[0] = 0;
		fb_data_mem[1] = 0;
	} else {

		Ref<Image> vp_image = vp_texture->get_data();

		// Decide the output size that fits the target region while keeping the viewport aspect ratio
		// and never bigger than it
		float vp_aspect = (float)vp_w / vp_h;
		float target_aspect = (float)p_target_w / p_target_h;
		int out_w;
		int out_h;
		if (target_aspect >= vp_aspect) {
			out_h = MIN(vp_h, p_target_h);
			out_w = vp_aspect * out_h;
		} else {
			out_w = MIN(vp_w, p_target_w);
			out_h = out_w / vp_aspect;
		}

		Image::Format format = vp_image->get_format();
		fb_data_mem = reinterpret_cast<uint32_t *>(fb_data->set_size(HEADER_SIZE + Image::get_image_data_size(out_w, out_h, format)));
		if (!fb_data_mem) {
			return;
		}

		fb_data_mem[0] = out_w;
		fb_data_mem[1] = out_h;
		fb_data_mem[2] = format;
		{
			// Resize the image directly to shared memory

			PoolByteArray::Read r = vp_image->get_data().read();

			const uint8_t *src = r.ptr();
			uint8_t *dst = reinterpret_cast<uint8_t *>(fb_data_mem) + HEADER_SIZE;
			int ps = Image::get_format_pixel_size(format);

			Image::scale_nearest_raw(ps, src, dst, vp_w, vp_h, out_w, out_h);
		}
	}

	fb_data->end_access();

	ScriptDebugger::get_singleton()->send_message("framebuffer", Array());
}

LiveViewDebugHelper::LiveViewDebugHelper() {

	fb_data = SharedMemory::create("godot_live_view");
	fb_data->open();
}

LiveViewDebugHelper::~LiveViewDebugHelper() {

	fb_data->close();
	memdelete(fb_data);
}
