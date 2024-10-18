/**************************************************************************/
/*  xdg-shell.cpp                                                         */
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

void WaylandThread::_xdg_wm_base_on_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}

void WaylandThread::_xdg_surface_on_configure(void *data, struct xdg_surface *xdg_surface, uint32_t serial) {
	xdg_surface_ack_configure(xdg_surface, serial);

	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	DEBUG_LOG_WAYLAND_THREAD(vformat("xdg surface on configure width %d height %d", ws->rect.size.width, ws->rect.size.height));
}

void WaylandThread::_xdg_toplevel_on_configure(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states) {
	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	// Expect the window to be in a plain state. It will get properly set if the
	// compositor reports otherwise below.
	ws->mode = DisplayServer::WINDOW_MODE_WINDOWED;
	ws->suspended = false;

	uint32_t *state = nullptr;
	wl_array_for_each(state, states) {
		switch (*state) {
			case XDG_TOPLEVEL_STATE_MAXIMIZED: {
				ws->mode = DisplayServer::WINDOW_MODE_MAXIMIZED;
			} break;

			case XDG_TOPLEVEL_STATE_FULLSCREEN: {
				ws->mode = DisplayServer::WINDOW_MODE_FULLSCREEN;
			} break;

			case XDG_TOPLEVEL_STATE_SUSPENDED: {
				ws->suspended = true;
			} break;

			default: {
				// We don't care about the other states (for now).
			} break;
		}
	}

	if (width != 0 && height != 0) {
		window_state_update_size(ws, width, height);
	}

	DEBUG_LOG_WAYLAND_THREAD(vformat("XDG toplevel on configure width %d height %d.", width, height));
}

void WaylandThread::_xdg_toplevel_on_close(void *data, struct xdg_toplevel *xdg_toplevel) {
	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	Ref<WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_CLOSE_REQUEST;
	ws->wayland_thread->push_message(msg);
}

void WaylandThread::_xdg_toplevel_on_configure_bounds(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height) {
}

void WaylandThread::_xdg_toplevel_on_wm_capabilities(void *data, struct xdg_toplevel *xdg_toplevel, struct wl_array *capabilities) {
	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	ws->can_maximize = false;
	ws->can_fullscreen = false;
	ws->can_minimize = false;

	uint32_t *capability = nullptr;
	wl_array_for_each(capability, capabilities) {
		switch (*capability) {
			case XDG_TOPLEVEL_WM_CAPABILITIES_MAXIMIZE: {
				ws->can_maximize = true;
			} break;
			case XDG_TOPLEVEL_WM_CAPABILITIES_FULLSCREEN: {
				ws->can_fullscreen = true;
			} break;

			case XDG_TOPLEVEL_WM_CAPABILITIES_MINIMIZE: {
				ws->can_minimize = true;
			} break;

			default: {
			} break;
		}
	}
}

void WaylandThread::_xdg_exported_on_exported(void *data, zxdg_exported_v1 *exported, const char *handle) {
	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	ws->exported_handle = vformat("wayland:%s", String::utf8(handle));
}

void WaylandThread::_xdg_toplevel_decoration_on_configure(void *data, struct zxdg_toplevel_decoration_v1 *xdg_toplevel_decoration, uint32_t mode) {
	if (mode == ZXDG_TOPLEVEL_DECORATION_V1_MODE_CLIENT_SIDE) {
#ifdef LIBDECOR_ENABLED
		WARN_PRINT_ONCE("Native client side decorations are not yet supported without libdecor!");
#else
		WARN_PRINT_ONCE("Native client side decorations are not yet supported!");
#endif // LIBDECOR_ENABLED
	}
}
