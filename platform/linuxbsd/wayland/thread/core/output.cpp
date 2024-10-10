/**************************************************************************/
/*  output.cpp                                                            */
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

void WaylandThread::_wl_output_on_geometry(void *data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char *make, const char *model, int32_t transform) {
	ScreenState *ss = (ScreenState *)data;
	ERR_FAIL_NULL(ss);

	ss->pending_data.position.x = x;

	ss->pending_data.position.x = x;
	ss->pending_data.position.y = y;

	ss->pending_data.physical_size.width = physical_width;
	ss->pending_data.physical_size.height = physical_height;

	ss->pending_data.make.parse_utf8(make);
	ss->pending_data.model.parse_utf8(model);

	// `wl_output::done` is a version 2 addition. We'll directly update the data
	// for compatibility.
	if (wl_output_get_version(wl_output) == 1) {
		ss->data = ss->pending_data;
	}
}

void WaylandThread::_wl_output_on_mode(void *data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {
	ScreenState *ss = (ScreenState *)data;
	ERR_FAIL_NULL(ss);

	ss->pending_data.size.width = width;
	ss->pending_data.size.height = height;

	ss->pending_data.refresh_rate = refresh ? refresh / 1000.0f : -1;

	// `wl_output::done` is a version 2 addition. We'll directly update the data
	// for compatibility.
	if (wl_output_get_version(wl_output) == 1) {
		ss->data = ss->pending_data;
	}
}

// NOTE: The following `wl_output` events are only for version 2 onwards, so we
// can assume that they're "atomic" (i.e. rely on the `wl_output::done` event).

void WaylandThread::_wl_output_on_done(void *data, struct wl_output *wl_output) {
	ScreenState *ss = (ScreenState *)data;
	ERR_FAIL_NULL(ss);

	ss->data = ss->pending_data;

	ss->wayland_thread->_update_scale(ss->data.scale);

	DEBUG_LOG_WAYLAND_THREAD(vformat("Output %x done.", (size_t)wl_output));
}

void WaylandThread::_wl_output_on_scale(void *data, struct wl_output *wl_output, int32_t factor) {
	ScreenState *ss = (ScreenState *)data;
	ERR_FAIL_NULL(ss);

	ss->pending_data.scale = factor;

	DEBUG_LOG_WAYLAND_THREAD(vformat("Output %x scale %d", (size_t)wl_output, factor));
}

void WaylandThread::_wl_output_on_name(void *data, struct wl_output *wl_output, const char *name) {
}

void WaylandThread::_wl_output_on_description(void *data, struct wl_output *wl_output, const char *description) {
}
