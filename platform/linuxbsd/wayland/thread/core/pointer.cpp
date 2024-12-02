/**************************************************************************/
/*  pointer.cpp                                                           */
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

void WaylandThread::_wl_pointer_on_enter(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	if (!surface || !wl_proxy_is_godot((struct wl_proxy *)surface)) {
		return;
	}

	DEBUG_LOG_WAYLAND_THREAD("Pointing window.");

	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ERR_FAIL_NULL(ss->cursor_surface);
	ss->pointer_enter_serial = serial;
	ss->pointed_surface = surface;
	ss->last_pointed_surface = surface;

	seat_state_update_cursor(ss);

	Ref<WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_MOUSE_ENTER;

	ss->wayland_thread->push_message(msg);
}

void WaylandThread::_wl_pointer_on_leave(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface) {
	if (!surface || !wl_proxy_is_godot((struct wl_proxy *)surface)) {
		return;
	}

	DEBUG_LOG_WAYLAND_THREAD("Left window.");

	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread *wayland_thread = ss->wayland_thread;
	ERR_FAIL_NULL(wayland_thread);

	ss->pointed_surface = nullptr;

	ss->pointer_data_buffer.pressed_button_mask.clear();

	Ref<WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_MOUSE_EXIT;

	wayland_thread->push_message(msg);
}

void WaylandThread::_wl_pointer_on_motion(void *data, struct wl_pointer *wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	WindowState *ws = wl_surface_get_window_state(ss->pointed_surface);
	ERR_FAIL_NULL(ws);

	PointerData &pd = ss->pointer_data_buffer;

	// TODO: Scale only when sending the Wayland message.
	pd.position.x = wl_fixed_to_double(surface_x);
	pd.position.y = wl_fixed_to_double(surface_y);

	pd.position *= window_state_get_scale_factor(ws);

	pd.motion_time = time;
}

void WaylandThread::_wl_pointer_on_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	PointerData &pd = ss->pointer_data_buffer;

	MouseButton button_pressed = MouseButton::NONE;

	switch (button) {
		case BTN_LEFT:
			button_pressed = MouseButton::LEFT;
			break;

		case BTN_MIDDLE:
			button_pressed = MouseButton::MIDDLE;
			break;

		case BTN_RIGHT:
			button_pressed = MouseButton::RIGHT;
			break;

		case BTN_EXTRA:
			button_pressed = MouseButton::MB_XBUTTON1;
			break;

		case BTN_SIDE:
			button_pressed = MouseButton::MB_XBUTTON2;
			break;

		default: {
		}
	}

	MouseButtonMask mask = mouse_button_to_mask(button_pressed);

	if (state & WL_POINTER_BUTTON_STATE_PRESSED) {
		pd.pressed_button_mask.set_flag(mask);
		pd.last_button_pressed = button_pressed;
		pd.double_click_begun = true;
	} else {
		pd.pressed_button_mask.clear_flag(mask);
	}

	pd.button_time = time;
	pd.button_serial = serial;
}

void WaylandThread::_wl_pointer_on_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	PointerData &pd = ss->pointer_data_buffer;

	switch (axis) {
		case WL_POINTER_AXIS_VERTICAL_SCROLL: {
			pd.scroll_vector.y = wl_fixed_to_double(value);
		} break;

		case WL_POINTER_AXIS_HORIZONTAL_SCROLL: {
			pd.scroll_vector.x = wl_fixed_to_double(value);
		} break;
	}

	pd.button_time = time;
}

void WaylandThread::_wl_pointer_on_frame(void *data, struct wl_pointer *wl_pointer) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	WaylandThread *wayland_thread = ss->wayland_thread;
	ERR_FAIL_NULL(wayland_thread);

	wayland_thread->_set_current_seat(ss->wl_seat);

	PointerData &old_pd = ss->pointer_data;
	PointerData &pd = ss->pointer_data_buffer;

	if (old_pd.motion_time != pd.motion_time || old_pd.relative_motion_time != pd.relative_motion_time) {
		Ref<InputEventMouseMotion> mm;
		mm.instantiate();

		// Set all pressed modifiers.
		mm->set_shift_pressed(ss->shift_pressed);
		mm->set_ctrl_pressed(ss->ctrl_pressed);
		mm->set_alt_pressed(ss->alt_pressed);
		mm->set_meta_pressed(ss->meta_pressed);

		mm->set_window_id(DisplayServer::MAIN_WINDOW_ID);
		mm->set_button_mask(pd.pressed_button_mask);
		mm->set_position(pd.position);
		mm->set_global_position(pd.position);

		Vector2 pos_delta = pd.position - old_pd.position;

		if (old_pd.relative_motion_time != pd.relative_motion_time) {
			uint32_t time_delta = pd.relative_motion_time - old_pd.relative_motion_time;

			mm->set_relative(pd.relative_motion);
			mm->set_velocity((Vector2)pos_delta / time_delta);
		} else {
			// The spec includes the possibility of having motion events without an
			// associated relative motion event. If that's the case, fallback to a
			// simple delta of the position. The captured mouse won't report the
			// relative speed anymore though.
			uint32_t time_delta = pd.motion_time - old_pd.motion_time;

			mm->set_relative(pd.position - old_pd.position);
			mm->set_velocity((Vector2)pos_delta / time_delta);
		}
		mm->set_relative_screen_position(mm->get_relative());
		mm->set_screen_velocity(mm->get_velocity());

		Ref<InputEventMessage> msg;
		msg.instantiate();

		msg->event = mm;

		wayland_thread->push_message(msg);
	}

	if (pd.discrete_scroll_vector_120 - old_pd.discrete_scroll_vector_120 != Vector2i()) {
		// This is a discrete scroll (eg. from a scroll wheel), so we'll just emit
		// scroll wheel buttons.
		if (pd.scroll_vector.y != 0) {
			MouseButton button = pd.scroll_vector.y > 0 ? MouseButton::WHEEL_DOWN : MouseButton::WHEEL_UP;
			pd.pressed_button_mask.set_flag(mouse_button_to_mask(button));
		}

		if (pd.scroll_vector.x != 0) {
			MouseButton button = pd.scroll_vector.x > 0 ? MouseButton::WHEEL_RIGHT : MouseButton::WHEEL_LEFT;
			pd.pressed_button_mask.set_flag(mouse_button_to_mask(button));
		}
	} else {
		if (pd.scroll_vector - old_pd.scroll_vector != Vector2()) {
			// This is a continuous scroll, so we'll emit a pan gesture.
			Ref<InputEventPanGesture> pg;
			pg.instantiate();

			// Set all pressed modifiers.
			pg->set_shift_pressed(ss->shift_pressed);
			pg->set_ctrl_pressed(ss->ctrl_pressed);
			pg->set_alt_pressed(ss->alt_pressed);
			pg->set_meta_pressed(ss->meta_pressed);

			pg->set_position(pd.position);

			pg->set_window_id(DisplayServer::MAIN_WINDOW_ID);

			pg->set_delta(pd.scroll_vector);

			Ref<InputEventMessage> msg;
			msg.instantiate();

			msg->event = pg;

			wayland_thread->push_message(msg);
		}
	}

	if (old_pd.pressed_button_mask != pd.pressed_button_mask) {
		BitField<MouseButtonMask> pressed_mask_delta = old_pd.pressed_button_mask ^ pd.pressed_button_mask;

		const MouseButton buttons_to_test[] = {
			MouseButton::LEFT,
			MouseButton::MIDDLE,
			MouseButton::RIGHT,
			MouseButton::WHEEL_UP,
			MouseButton::WHEEL_DOWN,
			MouseButton::WHEEL_LEFT,
			MouseButton::WHEEL_RIGHT,
			MouseButton::MB_XBUTTON1,
			MouseButton::MB_XBUTTON2,
		};

		for (MouseButton test_button : buttons_to_test) {
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
				mb->set_position(pd.position);
				mb->set_global_position(pd.position);

				if (test_button == MouseButton::WHEEL_UP || test_button == MouseButton::WHEEL_DOWN) {
					// If this is a discrete scroll, specify how many "clicks" it did for this
					// pointer frame.
					mb->set_factor(Math::abs(pd.discrete_scroll_vector_120.y / (float)120));
				}

				if (test_button == MouseButton::WHEEL_RIGHT || test_button == MouseButton::WHEEL_LEFT) {
					// If this is a discrete scroll, specify how many "clicks" it did for this
					// pointer frame.
					mb->set_factor(fabs(pd.discrete_scroll_vector_120.x / (float)120));
				}

				mb->set_button_mask(pd.pressed_button_mask);

				mb->set_button_index(test_button);
				mb->set_pressed(pd.pressed_button_mask.has_flag(test_button_mask));

				// We have to set the last position pressed here as we can't take for
				// granted what the individual events might have seen due to them not having
				// a guaranteed order.
				if (mb->is_pressed()) {
					pd.last_pressed_position = pd.position;
				}

				if (old_pd.double_click_begun && mb->is_pressed() && pd.last_button_pressed == old_pd.last_button_pressed && (pd.button_time - old_pd.button_time) < 400 && Vector2(old_pd.last_pressed_position).distance_to(Vector2(pd.last_pressed_position)) < 5) {
					pd.double_click_begun = false;
					mb->set_double_click(true);
				}

				Ref<InputEventMessage> msg;
				msg.instantiate();

				msg->event = mb;

				wayland_thread->push_message(msg);

				// Send an event resetting immediately the wheel key.
				// Wayland specification defines axis_stop events as optional and says to
				// treat all axis events as unterminated. As such, we have to manually do
				// it ourselves.
				if (test_button == MouseButton::WHEEL_UP || test_button == MouseButton::WHEEL_DOWN || test_button == MouseButton::WHEEL_LEFT || test_button == MouseButton::WHEEL_RIGHT) {
					// FIXME: This is ugly, I can't find a clean way to clone an InputEvent.
					// This works for now, despite being horrible.
					Ref<InputEventMouseButton> wh_up;
					wh_up.instantiate();

					wh_up->set_window_id(DisplayServer::MAIN_WINDOW_ID);
					wh_up->set_position(pd.position);
					wh_up->set_global_position(pd.position);

					// We have to unset the button to avoid it getting stuck.
					pd.pressed_button_mask.clear_flag(test_button_mask);
					wh_up->set_button_mask(pd.pressed_button_mask);

					wh_up->set_button_index(test_button);
					wh_up->set_pressed(false);

					Ref<InputEventMessage> msg_up;
					msg_up.instantiate();
					msg_up->event = wh_up;
					wayland_thread->push_message(msg_up);
				}
			}
		}
	}

	// Reset the scroll vectors as we already handled them.
	pd.scroll_vector = Vector2();
	pd.discrete_scroll_vector_120 = Vector2i();

	// Update the data all getters read. Wayland's specification requires us to do
	// this, since all pointer actions are sent in individual events.
	old_pd = pd;
}

void WaylandThread::_wl_pointer_on_axis_source(void *data, struct wl_pointer *wl_pointer, uint32_t axis_source) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	ss->pointer_data_buffer.scroll_type = axis_source;
}

void WaylandThread::_wl_pointer_on_axis_stop(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis) {
}

// NOTE: This event is deprecated since version 8 and superseded by
// `wl_pointer::axis_value120`. This thus converts the data to its
// fraction-of-120 format.
void WaylandThread::_wl_pointer_on_axis_discrete(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	PointerData &pd = ss->pointer_data_buffer;

	// NOTE: We can allow ourselves to not accumulate this data (and thus just
	// assign it) as the spec guarantees only one event per axis type.

	if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pd.discrete_scroll_vector_120.y = discrete * 120;
	}

	if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pd.discrete_scroll_vector_120.x = discrete * 120;
	}
}

// Supersedes `wl_pointer::axis_discrete` Since version 8.
void WaylandThread::_wl_pointer_on_axis_value120(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t value120) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (!ss->pointed_surface) {
		// We're probably on a decoration or some other third-party thing.
		return;
	}

	PointerData &pd = ss->pointer_data_buffer;

	if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pd.discrete_scroll_vector_120.y += value120;
	}

	if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pd.discrete_scroll_vector_120.x += value120;
	}
}

// TODO: Add support to this event.
void WaylandThread::_wl_pointer_on_axis_relative_direction(void *data, struct wl_pointer *wl_pointer, uint32_t axis, uint32_t direction) {
}
