/**************************************************************************/
/*  surface.cpp                                                           */
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

void WaylandThread::_wl_surface_on_enter(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
	if (!wl_output || !wl_proxy_is_godot((struct wl_proxy *)wl_output)) {
		// This won't have the right data bound to it. Not worth it and would probably
		// just break everything.
		return;
	}

	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	DEBUG_LOG_WAYLAND_THREAD(vformat("Window entered output %x.\n", (size_t)wl_output));

	ws->wl_outputs.insert(wl_output);

	// Workaround for buffer scaling as there's no guaranteed way of knowing the
	// preferred scale.
	// TODO: Skip this branch for newer `wl_surface`s once we add support for
	// `wl_surface::preferred_buffer_scale`
	if (ws->preferred_fractional_scale == 0) {
		window_state_update_size(ws, ws->rect.size.width, ws->rect.size.height);
	}
}

void WaylandThread::_wl_surface_on_leave(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
	if (!wl_output || !wl_proxy_is_godot((struct wl_proxy *)wl_output)) {
		// This won't have the right data bound to it. Not worth it and would probably
		// just break everything.
		return;
	}

	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	ws->wl_outputs.erase(wl_output);

	DEBUG_LOG_WAYLAND_THREAD(vformat("Window left output %x.\n", (size_t)wl_output));
}

// TODO: Add support to this event.
void WaylandThread::_wl_surface_on_preferred_buffer_scale(void *data, struct wl_surface *wl_surface, int32_t factor) {
}

// TODO: Add support to this event.
void WaylandThread::_wl_surface_on_preferred_buffer_transform(void *data, struct wl_surface *wl_surface, uint32_t transform) {
}
