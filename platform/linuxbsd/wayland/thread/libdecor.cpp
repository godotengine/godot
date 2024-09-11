/**************************************************************************/
/*  libdecor.cpp                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "wayland/wayland_thread.h"

#ifdef LIBDECOR_ENABLED
void WaylandThread::libdecor_on_error(struct libdecor *context, enum libdecor_error error, const char *message) {
	ERR_PRINT(vformat("libdecor error %d: %s", error, message));
}

// NOTE: This is pretty much a reimplementation of _xdg_surface_on_configure
// and _xdg_toplevel_on_configure. Libdecor really likes wrapping everything,
// forcing us to do stuff like this.
void WaylandThread::libdecor_frame_on_configure(struct libdecor_frame *frame, struct libdecor_configuration *configuration, void *user_data) {
	WindowState *ws = (WindowState *)user_data;
	ERR_FAIL_NULL(ws);

	int width = 0;
	int height = 0;

	ws->pending_libdecor_configuration = configuration;

	if (!libdecor_configuration_get_content_size(configuration, frame, &width, &height)) {
		// The configuration doesn't have a size. We'll use the one already set in the window.
		width = ws->rect.size.width;
		height = ws->rect.size.height;
	}

	ERR_FAIL_COND_MSG(width == 0 || height == 0, "Window has invalid size.");

	libdecor_window_state window_state = LIBDECOR_WINDOW_STATE_NONE;

	// Expect the window to be in a plain state. It will get properly set if the
	// compositor reports otherwise below.
	ws->mode = DisplayServer::WINDOW_MODE_WINDOWED;
	ws->suspended = false;

	if (libdecor_configuration_get_window_state(configuration, &window_state)) {
		if (window_state & LIBDECOR_WINDOW_STATE_MAXIMIZED) {
			ws->mode = DisplayServer::WINDOW_MODE_MAXIMIZED;
		}

		if (window_state & LIBDECOR_WINDOW_STATE_FULLSCREEN) {
			ws->mode = DisplayServer::WINDOW_MODE_FULLSCREEN;
		}

		if (window_state & LIBDECOR_WINDOW_STATE_SUSPENDED) {
			ws->suspended = true;
		}
	}

	window_state_update_size(ws, width, height);

	DEBUG_LOG_WAYLAND_THREAD(vformat("libdecor frame on configure rect %s", ws->rect));
}

void WaylandThread::libdecor_frame_on_close(struct libdecor_frame *frame, void *user_data) {
	WindowState *ws = (WindowState *)user_data;
	ERR_FAIL_NULL(ws);

	Ref<WindowEventMessage> winevent_msg;
	winevent_msg.instantiate();
	winevent_msg->event = DisplayServer::WINDOW_EVENT_CLOSE_REQUEST;

	ws->wayland_thread->push_message(winevent_msg);

	DEBUG_LOG_WAYLAND_THREAD("libdecor frame on close");
}

void WaylandThread::libdecor_frame_on_commit(struct libdecor_frame *frame, void *user_data) {
	// We're skipping this as we don't really care about libdecor's commit for
	// atomicity reasons. See `_frame_wl_callback_on_done` for more info.

	DEBUG_LOG_WAYLAND_THREAD("libdecor frame on commit");
}

void WaylandThread::libdecor_frame_on_dismiss_popup(struct libdecor_frame *frame, const char *seat_name, void *user_data) {
}
#endif // LIBDECOR_ENABLED
