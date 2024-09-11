/**************************************************************************/
/*  seat.cpp                                                              */
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

void WaylandThread::_wl_seat_on_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities) {
	SeatState *ss = (SeatState *)data;

	ERR_FAIL_NULL(ss);

	// TODO: Handle touch.

	// Pointer handling.
	if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
		if (!ss->wl_pointer) {
			ss->cursor_surface = wl_compositor_create_surface(ss->registry->wl_compositor);
			wl_surface_commit(ss->cursor_surface);

			ss->wl_pointer = wl_seat_get_pointer(wl_seat);
			wl_pointer_add_listener(ss->wl_pointer, &wl_pointer_listener, ss);

			if (ss->registry->wp_relative_pointer_manager) {
				ss->wp_relative_pointer = zwp_relative_pointer_manager_v1_get_relative_pointer(ss->registry->wp_relative_pointer_manager, ss->wl_pointer);
				zwp_relative_pointer_v1_add_listener(ss->wp_relative_pointer, &wp_relative_pointer_listener, ss);
			}

			if (ss->registry->wp_pointer_gestures) {
				ss->wp_pointer_gesture_pinch = zwp_pointer_gestures_v1_get_pinch_gesture(ss->registry->wp_pointer_gestures, ss->wl_pointer);
				zwp_pointer_gesture_pinch_v1_add_listener(ss->wp_pointer_gesture_pinch, &wp_pointer_gesture_pinch_listener, ss);
			}

			// TODO: Constrain new pointers if the global mouse mode is constrained.
		}
	} else {
		if (ss->cursor_frame_callback) {
			// Just in case. I got bitten by weird race-like conditions already.
			wl_callback_set_user_data(ss->cursor_frame_callback, nullptr);

			wl_callback_destroy(ss->cursor_frame_callback);
			ss->cursor_frame_callback = nullptr;
		}

		if (ss->cursor_surface) {
			wl_surface_destroy(ss->cursor_surface);
			ss->cursor_surface = nullptr;
		}

		if (ss->wl_pointer) {
			wl_pointer_destroy(ss->wl_pointer);
			ss->wl_pointer = nullptr;
		}

		if (ss->wp_relative_pointer) {
			zwp_relative_pointer_v1_destroy(ss->wp_relative_pointer);
			ss->wp_relative_pointer = nullptr;
		}

		if (ss->wp_confined_pointer) {
			zwp_confined_pointer_v1_destroy(ss->wp_confined_pointer);
			ss->wp_confined_pointer = nullptr;
		}

		if (ss->wp_locked_pointer) {
			zwp_locked_pointer_v1_destroy(ss->wp_locked_pointer);
			ss->wp_locked_pointer = nullptr;
		}
	}

	// Keyboard handling.
	if (capabilities & WL_SEAT_CAPABILITY_KEYBOARD) {
		if (!ss->wl_keyboard) {
			ss->xkb_context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
			ERR_FAIL_NULL(ss->xkb_context);

			ss->wl_keyboard = wl_seat_get_keyboard(wl_seat);
			wl_keyboard_add_listener(ss->wl_keyboard, &wl_keyboard_listener, ss);
		}
	} else {
		if (ss->xkb_context) {
			xkb_context_unref(ss->xkb_context);
			ss->xkb_context = nullptr;
		}

		if (ss->wl_keyboard) {
			wl_keyboard_destroy(ss->wl_keyboard);
			ss->wl_keyboard = nullptr;
		}
	}
}

void WaylandThread::_wl_seat_on_name(void *data, struct wl_seat *wl_seat, const char *name) {
}
