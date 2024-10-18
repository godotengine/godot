/**************************************************************************/
/*  text-input.cpp                                                        */
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

void WaylandThread::_wp_fractional_scale_on_preferred_scale(void *data, struct wp_fractional_scale_v1 *wp_fractional_scale_v1, uint32_t scale) {
	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	ws->preferred_fractional_scale = (double)scale / 120;

	window_state_update_size(ws, ws->rect.size.width, ws->rect.size.height);
}

void WaylandThread::_wp_relative_pointer_on_relative_motion(void *data, struct zwp_relative_pointer_v1 *wp_relative_pointer, uint32_t uptime_hi, uint32_t uptime_lo, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t dx_unaccel, wl_fixed_t dy_unaccel) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	PointerData &pd = ss->pointer_data_buffer;

	WindowState *ws = wl_surface_get_window_state(ss->pointed_surface);
	ERR_FAIL_NULL(ws);

	pd.relative_motion.x = wl_fixed_to_double(dx);
	pd.relative_motion.y = wl_fixed_to_double(dy);

	pd.relative_motion *= window_state_get_scale_factor(ws);

	pd.relative_motion_time = uptime_lo;
}

void WaylandThread::_wp_pointer_gesture_pinch_on_begin(void *data, struct zwp_pointer_gesture_pinch_v1 *wp_pointer_gesture_pinch_v1, uint32_t serial, uint32_t time, struct wl_surface *surface, uint32_t fingers) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (fingers == 2) {
		ss->old_pinch_scale = wl_fixed_from_int(1);
		ss->active_gesture = Gesture::MAGNIFY;
	}
}

void WaylandThread::_wp_pointer_gesture_pinch_on_update(void *data, struct zwp_pointer_gesture_pinch_v1 *wp_pointer_gesture_pinch_v1, uint32_t time, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t scale, wl_fixed_t rotation) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread *wayland_thread = ss->wayland_thread;
	ERR_FAIL_NULL(wayland_thread);

	PointerData &pd = ss->pointer_data_buffer;

	if (ss->active_gesture == Gesture::MAGNIFY) {
		Ref<InputEventMagnifyGesture> mg;
		mg.instantiate();

		mg->set_window_id(DisplayServer::MAIN_WINDOW_ID);

		// Set all pressed modifiers.
		mg->set_shift_pressed(ss->shift_pressed);
		mg->set_ctrl_pressed(ss->ctrl_pressed);
		mg->set_alt_pressed(ss->alt_pressed);
		mg->set_meta_pressed(ss->meta_pressed);

		mg->set_position(pd.position);

		wl_fixed_t scale_delta = scale - ss->old_pinch_scale;
		mg->set_factor(1 + wl_fixed_to_double(scale_delta));

		Ref<InputEventMessage> magnify_msg;
		magnify_msg.instantiate();
		magnify_msg->event = mg;

		// Since Wayland allows only one gesture at a time and godot instead expects
		// both of them, we'll have to create two separate input events: one for
		// magnification and one for panning.

		Ref<InputEventPanGesture> pg;
		pg.instantiate();

		pg->set_window_id(DisplayServer::MAIN_WINDOW_ID);

		// Set all pressed modifiers.
		pg->set_shift_pressed(ss->shift_pressed);
		pg->set_ctrl_pressed(ss->ctrl_pressed);
		pg->set_alt_pressed(ss->alt_pressed);
		pg->set_meta_pressed(ss->meta_pressed);

		pg->set_position(pd.position);
		pg->set_delta(Vector2(wl_fixed_to_double(dx), wl_fixed_to_double(dy)));

		Ref<InputEventMessage> pan_msg;
		pan_msg.instantiate();
		pan_msg->event = pg;

		wayland_thread->push_message(magnify_msg);
		wayland_thread->push_message(pan_msg);

		ss->old_pinch_scale = scale;
	}
}

void WaylandThread::_wp_pointer_gesture_pinch_on_end(void *data, struct zwp_pointer_gesture_pinch_v1 *wp_pointer_gesture_pinch_v1, uint32_t serial, uint32_t time, int32_t cancelled) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->active_gesture = Gesture::NONE;
}

void WaylandThread::_wp_text_input_on_enter(void *data, struct zwp_text_input_v3 *wp_text_input_v3, struct wl_surface *surface) {
	SeatState *ss = (SeatState *)data;
	if (!ss) {
		return;
	}

	ss->ime_enabled = true;
}

void WaylandThread::_wp_text_input_on_leave(void *data, struct zwp_text_input_v3 *wp_text_input_v3, struct wl_surface *surface) {
	SeatState *ss = (SeatState *)data;
	if (!ss) {
		return;
	}

	ss->ime_enabled = false;
	ss->ime_active = false;
	ss->ime_text = String();
	ss->ime_text_commit = String();
	ss->ime_cursor = Vector2i();

	Ref<IMEUpdateEventMessage> msg;
	msg.instantiate();
	msg->text = String();
	msg->selection = Vector2i();
	ss->wayland_thread->push_message(msg);
}

void WaylandThread::_wp_text_input_on_preedit_string(void *data, struct zwp_text_input_v3 *wp_text_input_v3, const char *text, int32_t cursor_begin, int32_t cursor_end) {
	SeatState *ss = (SeatState *)data;
	if (!ss) {
		return;
	}

	ss->ime_text = String::utf8(text);

	// Convert cursor positions from UTF-8 to UTF-32 offset.
	int32_t cursor_begin_utf32 = 0;
	int32_t cursor_end_utf32 = 0;
	for (int i = 0; i < ss->ime_text.length(); i++) {
		uint32_t c = ss->ime_text[i];
		if (c <= 0x7f) { // 7 bits.
			cursor_begin -= 1;
			cursor_end -= 1;
		} else if (c <= 0x7ff) { // 11 bits
			cursor_begin -= 2;
			cursor_end -= 2;
		} else if (c <= 0xffff) { // 16 bits
			cursor_begin -= 3;
			cursor_end -= 3;
		} else if (c <= 0x001fffff) { // 21 bits
			cursor_begin -= 4;
			cursor_end -= 4;
		} else if (c <= 0x03ffffff) { // 26 bits
			cursor_begin -= 5;
			cursor_end -= 5;
		} else if (c <= 0x7fffffff) { // 31 bits
			cursor_begin -= 6;
			cursor_end -= 6;
		} else {
			cursor_begin -= 1;
			cursor_end -= 1;
		}
		if (cursor_begin == 0) {
			cursor_begin_utf32 = i + 1;
		}
		if (cursor_end == 0) {
			cursor_end_utf32 = i + 1;
		}
		if (cursor_begin <= 0 && cursor_end <= 0) {
			break;
		}
	}
	ss->ime_cursor = Vector2i(cursor_begin_utf32, cursor_end_utf32 - cursor_begin_utf32);
}

void WaylandThread::_wp_text_input_on_commit_string(void *data, struct zwp_text_input_v3 *wp_text_input_v3, const char *text) {
	SeatState *ss = (SeatState *)data;
	if (!ss) {
		return;
	}

	ss->ime_text_commit = String::utf8(text);
}

void WaylandThread::_wp_text_input_on_delete_surrounding_text(void *data, struct zwp_text_input_v3 *wp_text_input_v3, uint32_t before_length, uint32_t after_length) {
	// Not implemented.
}

void WaylandThread::_wp_text_input_on_done(void *data, struct zwp_text_input_v3 *wp_text_input_v3, uint32_t serial) {
	SeatState *ss = (SeatState *)data;
	if (!ss) {
		return;
	}

	if (!ss->ime_text_commit.is_empty()) {
		Ref<IMECommitEventMessage> msg;
		msg.instantiate();
		msg->text = ss->ime_text_commit;
		ss->wayland_thread->push_message(msg);
	} else if (!ss->ime_text.is_empty()) {
		Ref<IMEUpdateEventMessage> msg;
		msg.instantiate();
		msg->text = ss->ime_text;
		msg->selection = ss->ime_cursor;
		ss->wayland_thread->push_message(msg);
	}
	ss->ime_text = String();
	ss->ime_text_commit = String();
	ss->ime_cursor = Vector2i();
}
