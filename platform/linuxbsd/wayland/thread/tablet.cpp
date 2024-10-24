/**************************************************************************/
/*  tablet.cpp                                                            */
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

// FIXME: Does this cause issues with *BSDs?
#include <linux/input-event-codes.h>

void WaylandThread::_wp_tablet_seat_on_tablet_added(void *data, struct zwp_tablet_seat_v2 *wp_tablet_seat_v2, struct zwp_tablet_v2 *id) {
}

void WaylandThread::_wp_tablet_seat_on_tool_added(void *data, struct zwp_tablet_seat_v2 *wp_tablet_seat_v2, struct zwp_tablet_tool_v2 *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	TabletToolState *state = memnew(TabletToolState);
	state->wl_seat = ss->wl_seat;

	wl_proxy_tag_godot((struct wl_proxy *)id);
	zwp_tablet_tool_v2_add_listener(id, &wp_tablet_tool_listener, state);

	ss->tablet_tools.push_back(id);
}

void WaylandThread::_wp_tablet_seat_on_pad_added(void *data, struct zwp_tablet_seat_v2 *wp_tablet_seat_v2, struct zwp_tablet_pad_v2 *id) {
}

void WaylandThread::_wp_tablet_tool_on_type(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t tool_type) {
	TabletToolState *state = wp_tablet_tool_get_state(wp_tablet_tool_v2);

	if (state && tool_type == ZWP_TABLET_TOOL_V2_TYPE_ERASER) {
		state->is_eraser = true;
	}
}

void WaylandThread::_wp_tablet_tool_on_hardware_serial(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t hardware_serial_hi, uint32_t hardware_serial_lo) {
}

void WaylandThread::_wp_tablet_tool_on_hardware_id_wacom(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t hardware_id_hi, uint32_t hardware_id_lo) {
}

void WaylandThread::_wp_tablet_tool_on_capability(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t capability) {
}

void WaylandThread::_wp_tablet_tool_on_done(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2) {
}

void WaylandThread::_wp_tablet_tool_on_removed(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	SeatState *ss = wl_seat_get_seat_state(ts->wl_seat);
	if (!ss) {
		return;
	}

	List<struct zwp_tablet_tool_v2 *>::Element *E = ss->tablet_tools.find(wp_tablet_tool_v2);

	if (E && E->get()) {
		struct zwp_tablet_tool_v2 *tool = E->get();
		TabletToolState *state = wp_tablet_tool_get_state(tool);
		if (state) {
			memdelete(state);
		}

		zwp_tablet_tool_v2_destroy(tool);
		ss->tablet_tools.erase(E);
	}
}

void WaylandThread::_wp_tablet_tool_on_proximity_in(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t serial, struct zwp_tablet_v2 *tablet, struct wl_surface *surface) {
	if (!surface || !wl_proxy_is_godot((struct wl_proxy *)surface)) {
		// We're probably on a decoration or something.
		return;
	}

	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	SeatState *ss = wl_seat_get_seat_state(ts->wl_seat);
	if (!ss) {
		return;
	}

	WaylandThread *wayland_thread = ss->wayland_thread;
	ERR_FAIL_NULL(wayland_thread);

	ts->data_pending.proximity_serial = serial;
	ts->data_pending.proximal_surface = surface;
	ts->last_surface = surface;

	Ref<WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_MOUSE_ENTER;
	wayland_thread->push_message(msg);

	DEBUG_LOG_WAYLAND_THREAD("Tablet tool entered window.");
}

void WaylandThread::_wp_tablet_tool_on_proximity_out(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts || !ts->data_pending.proximal_surface) {
		// Not our stuff, we don't care.
		return;
	}

	SeatState *ss = wl_seat_get_seat_state(ts->wl_seat);
	if (!ss) {
		return;
	}

	WaylandThread *wayland_thread = ss->wayland_thread;
	ERR_FAIL_NULL(wayland_thread);

	ts->data_pending.proximal_surface = nullptr;

	Ref<WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_MOUSE_EXIT;

	wayland_thread->push_message(msg);

	DEBUG_LOG_WAYLAND_THREAD("Tablet tool left window.");
}

void WaylandThread::_wp_tablet_tool_on_down(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t serial) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	TabletToolData &td = ts->data_pending;

	td.pressed_button_mask.set_flag(mouse_button_to_mask(MouseButton::LEFT));
	td.last_button_pressed = MouseButton::LEFT;
	td.double_click_begun = true;

	// The protocol doesn't cover this, but we can use this funky hack to make
	// double clicking work.
	td.button_time = OS::get_singleton()->get_ticks_msec();
}

void WaylandThread::_wp_tablet_tool_on_up(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	TabletToolData &td = ts->data_pending;

	td.pressed_button_mask.clear_flag(mouse_button_to_mask(MouseButton::LEFT));

	// The protocol doesn't cover this, but we can use this funky hack to make
	// double clicking work.
	td.button_time = OS::get_singleton()->get_ticks_msec();
}

void WaylandThread::_wp_tablet_tool_on_motion(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, wl_fixed_t x, wl_fixed_t y) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	if (!ts->data_pending.proximal_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	WindowState *ws = wl_surface_get_window_state(ts->data_pending.proximal_surface);
	ERR_FAIL_NULL(ws);

	TabletToolData &td = ts->data_pending;

	double scale_factor = window_state_get_scale_factor(ws);

	td.position.x = wl_fixed_to_double(x);
	td.position.y = wl_fixed_to_double(y);
	td.position *= scale_factor;

	td.motion_time = OS::get_singleton()->get_ticks_msec();
}

void WaylandThread::_wp_tablet_tool_on_pressure(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t pressure) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	ts->data_pending.pressure = pressure;
}

void WaylandThread::_wp_tablet_tool_on_distance(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t distance) {
	// Unsupported
}

void WaylandThread::_wp_tablet_tool_on_tilt(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, wl_fixed_t tilt_x, wl_fixed_t tilt_y) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	TabletToolData &td = ts->data_pending;

	td.tilt.x = wl_fixed_to_double(tilt_x);
	td.tilt.y = wl_fixed_to_double(tilt_y);
}

void WaylandThread::_wp_tablet_tool_on_rotation(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, wl_fixed_t degrees) {
	// Unsupported.
}

void WaylandThread::_wp_tablet_tool_on_slider(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, int32_t position) {
	// Unsupported.
}

void WaylandThread::_wp_tablet_tool_on_wheel(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, wl_fixed_t degrees, int32_t clicks) {
	// TODO
}

void WaylandThread::_wp_tablet_tool_on_button(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t serial, uint32_t button, uint32_t state) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	TabletToolData &td = ts->data_pending;

	MouseButton mouse_button = MouseButton::NONE;

	if (button == BTN_STYLUS) {
		mouse_button = MouseButton::LEFT;
	}

	if (button == BTN_STYLUS2) {
		mouse_button = MouseButton::RIGHT;
	}

	if (mouse_button != MouseButton::NONE) {
		MouseButtonMask mask = mouse_button_to_mask(mouse_button);

		if (state == ZWP_TABLET_TOOL_V2_BUTTON_STATE_PRESSED) {
			td.pressed_button_mask.set_flag(mask);
			td.last_button_pressed = mouse_button;
			td.double_click_begun = true;
		} else {
			td.pressed_button_mask.clear_flag(mask);
		}

		// The protocol doesn't cover this, but we can use this funky hack to make
		// double clicking work.
		td.button_time = OS::get_singleton()->get_ticks_msec();
	}
}

void WaylandThread::_wp_tablet_tool_on_frame(void *data, struct zwp_tablet_tool_v2 *wp_tablet_tool_v2, uint32_t time) {
	TabletToolState *ts = wp_tablet_tool_get_state(wp_tablet_tool_v2);
	if (!ts) {
		return;
	}

	SeatState *ss = wl_seat_get_seat_state(ts->wl_seat);
	if (!ss) {
		return;
	}

	WaylandThread *wayland_thread = ss->wayland_thread;
	ERR_FAIL_NULL(wayland_thread);

	TabletToolData &old_td = ts->data;
	TabletToolData &td = ts->data_pending;

	if (old_td.position != td.position || old_td.tilt != td.tilt || old_td.pressure != td.pressure) {
		Ref<InputEventMouseMotion> mm;
		mm.instantiate();

		mm->set_window_id(DisplayServer::MAIN_WINDOW_ID);

		// Set all pressed modifiers.
		mm->set_shift_pressed(ss->shift_pressed);
		mm->set_ctrl_pressed(ss->ctrl_pressed);
		mm->set_alt_pressed(ss->alt_pressed);
		mm->set_meta_pressed(ss->meta_pressed);

		mm->set_button_mask(td.pressed_button_mask);

		mm->set_position(td.position);
		mm->set_global_position(td.position);

		// NOTE: The Godot API expects normalized values and we store them raw,
		// straight from the compositor, so we have to normalize them here.

		// According to the tablet proto spec, tilt is expressed in degrees relative
		// to the Z axis of the tablet, so it shouldn't go over 90 degrees either way,
		// I think. We'll clamp it just in case.
		td.tilt = td.tilt.clampf(-90, 90);

		mm->set_tilt(td.tilt / 90);

		// The tablet proto spec explicitly says that pressure is defined as a value
		// between 0 to 65535.
		mm->set_pressure(td.pressure / (float)65535);

		mm->set_pen_inverted(ts->is_eraser);

		mm->set_relative(td.position - old_td.position);
		mm->set_relative_screen_position(mm->get_relative());

		Vector2 pos_delta = td.position - old_td.position;
		uint32_t time_delta = td.motion_time - old_td.motion_time;
		mm->set_velocity((Vector2)pos_delta / time_delta);

		Ref<InputEventMessage> inputev_msg;
		inputev_msg.instantiate();

		inputev_msg->event = mm;

		wayland_thread->push_message(inputev_msg);
	}

	if (old_td.pressed_button_mask != td.pressed_button_mask) {
		BitField<MouseButtonMask> pressed_mask_delta = BitField<MouseButtonMask>((int64_t)old_td.pressed_button_mask ^ (int64_t)td.pressed_button_mask);

		for (MouseButton test_button : { MouseButton::LEFT, MouseButton::RIGHT }) {
			MouseButtonMask test_button_mask = mouse_button_to_mask(test_button);

			if (pressed_mask_delta.has_flag(test_button_mask)) {
				Ref<InputEventMouseButton> mb;
				mb.instantiate();

				// Set all pressed modifiers.
				mb->set_shift_pressed(ss->shift_pressed);
				mb->set_ctrl_pressed(ss->ctrl_pressed);
				mb->set_alt_pressed(ss->alt_pressed);
				mb->set_meta_pressed(ss->meta_pressed);

				mb->set_window_id(DisplayServer::MAIN_WINDOW_ID);
				mb->set_position(td.position);
				mb->set_global_position(td.position);

				mb->set_button_mask(td.pressed_button_mask);
				mb->set_button_index(test_button);
				mb->set_pressed(td.pressed_button_mask.has_flag(test_button_mask));

				// We have to set the last position pressed here as we can't take for
				// granted what the individual events might have seen due to them not having
				// a garaunteed order.
				if (mb->is_pressed()) {
					td.last_pressed_position = td.position;
				}

				if (old_td.double_click_begun && mb->is_pressed() && td.last_button_pressed == old_td.last_button_pressed && (td.button_time - old_td.button_time) < 400 && Vector2(td.last_pressed_position).distance_to(Vector2(old_td.last_pressed_position)) < 5) {
					td.double_click_begun = false;
					mb->set_double_click(true);
				}

				Ref<InputEventMessage> msg;
				msg.instantiate();

				msg->event = mb;

				wayland_thread->push_message(msg);
			}
		}
	}

	old_td = td;
}
