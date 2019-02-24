/*************************************************************************/
/*  live_view.cpp                                                        */
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

#include "editor/live_view.h"
#include "core/os/shared_mem_access.h"
#include "editor/script_editor_debugger.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"

void LiveViewDock::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_PROCESS: {

			if (!is_visible_in_tree()) {
				waiting = false;
				break;
			}
			if (waiting) {
				break;
			}

			ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
			if (sed) {
				Array msg;
				msg.push_back("request_framebuffer");
				if (sed->send_message(msg)) {
					waiting = true;
				}
			}
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {

			if (!is_visible_in_tree()) {
				waiting = false;
			}
		} break;
	}
}

void LiveViewDock::start() {

	fb_data->open

			set_process(true);
	waiting = false;
}

void LiveViewDock::stop() {

	fb_data->close();

	set_process(false);
	waiting = false;
}

void LiveViewDock::update() {

	if (!waiting) {
		return;
	}

	if (!fb_data->is_open()) {
		if (fb_data->open() != OK) {
			return;
		}
	}

	const uint32_t *data = (uint32_t *)fb_data->lock();
	if (data) {

		int width = data[0];
		int height = data[1];
		Image::Format format = (Image::Format)data[2];

		int data_size = data[3];
		PoolByteArray pixel_data;
		pixel_data.resize(data_size);
		{
			PoolByteArray::Write w = pixel_data.write();
			memcpy(w.ptr(), &data[4], data_size);
		}

		Ref<Image> img = memnew(Image);
		img->create(width, height, false, format, pixel_data);

		Ref<ImageTexture> tex = get_texture();
		// Create if no texture or recreate if size or format changed
		if (!tex.is_valid() ||
				tex->get_width() != width ||
				tex->get_height() != height ||
				tex->get_data()->get_format() != format) {

			tex = Ref<ImageTexture>(memnew(ImageTexture));
			tex->create(width, height, format, Texture::FLAG_FILTER);
			set_texture(tex);
		}

		tex->set_data(img);

		fb_data->unlock();
	}

	waiting = false;
}

LiveViewDock::LiveViewDock() :
		waiting(false),
		fb_data(SharedMemAccess::create("godot_live_view")) {

	fb_data->open(1);
	fb_data->close();

	set_expand(true);
	set_stretch_mode(STRETCH_KEEP_ASPECT_CENTERED);

	set_name("LiveView");
}

LiveViewDock::~LiveViewDock() {

	memdelete(fb_data);
}

void LiveViewDebugHelper::_debugger_request_framebuffer(void *user_data) {

	Ref<ViewportTexture> vp_texture = SceneTree::get_singleton()->get_root()->get_texture();
	Ref<Image> framebuffer = vp_texture->get_data();

	SharedMemAccess *fb_data = SharedMemAccess::create("godot_live_view");
	fb_data->

			uint32_t data_size = framebuffer->get_data().size();

	Array result;
	result.append(framebuffer->get_width());
	result.append(framebuffer->get_height());
	result.append(framebuffer->get_format());
	result.append(framebuffer->get_data().size());
	result.append(framebuffer->get_data());

	ScriptDebugger::get_singleton()->send_message("framebuffer", result);
}
