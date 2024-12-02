/**************************************************************************/
/*  wayland_thread.cpp                                                    */
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

#include "wayland_thread.h"

#ifdef WAYLAND_ENABLED

// For the actual polling thread.
#include <poll.h>

#include <sys/mman.h>

// Sets up an `InputEventKey` and returns whether it has any meaningful value.
bool WaylandThread::_seat_state_configure_key_event(SeatState &p_ss, Ref<InputEventKey> p_event, xkb_keycode_t p_keycode, bool p_pressed) {
	// TODO: Handle keys that release multiple symbols?
	Key keycode = KeyMappingXKB::get_keycode(xkb_state_key_get_one_sym(p_ss.xkb_state, p_keycode));
	Key physical_keycode = KeyMappingXKB::get_scancode(p_keycode);
	KeyLocation key_location = KeyMappingXKB::get_location(p_keycode);

	if (physical_keycode == Key::NONE) {
		return false;
	}

	if (keycode == Key::NONE) {
		keycode = physical_keycode;
	}

	if (keycode >= Key::A + 32 && keycode <= Key::Z + 32) {
		keycode -= 'a' - 'A';
	}

	p_event->set_window_id(DisplayServer::MAIN_WINDOW_ID);

	// Set all pressed modifiers.
	p_event->set_shift_pressed(p_ss.shift_pressed);
	p_event->set_ctrl_pressed(p_ss.ctrl_pressed);
	p_event->set_alt_pressed(p_ss.alt_pressed);
	p_event->set_meta_pressed(p_ss.meta_pressed);

	p_event->set_pressed(p_pressed);
	p_event->set_keycode(keycode);
	p_event->set_physical_keycode(physical_keycode);
	p_event->set_location(key_location);

	uint32_t unicode = xkb_state_key_get_utf32(p_ss.xkb_state, p_keycode);

	if (unicode != 0) {
		p_event->set_key_label(fix_key_label(unicode, keycode));
	} else {
		p_event->set_key_label(keycode);
	}

	if (p_pressed) {
		p_event->set_unicode(fix_unicode(unicode));
	}

	// Taken from DisplayServerX11.
	if (p_event->get_keycode() == Key::BACKTAB) {
		// Make it consistent across platforms.
		p_event->set_keycode(Key::TAB);
		p_event->set_physical_keycode(Key::TAB);
		p_event->set_shift_pressed(true);
	}

	return true;
}

void WaylandThread::_set_current_seat(struct wl_seat *p_seat) {
	if (p_seat == wl_seat_current) {
		return;
	}

	SeatState *old_state = wl_seat_get_seat_state(wl_seat_current);

	if (old_state) {
		seat_state_unlock_pointer(old_state);
	}

	SeatState *new_state = wl_seat_get_seat_state(p_seat);
	seat_state_unlock_pointer(new_state);

	wl_seat_current = p_seat;
	pointer_set_constraint(pointer_constraint);
}

// Returns whether it loaded the theme or not.
bool WaylandThread::_load_cursor_theme(int p_cursor_size) {
	if (wl_cursor_theme) {
		wl_cursor_theme_destroy(wl_cursor_theme);
		wl_cursor_theme = nullptr;
	}

	if (cursor_theme_name.is_empty()) {
		cursor_theme_name = "default";
	}

	print_verbose(vformat("Loading cursor theme \"%s\" size %d.", cursor_theme_name, p_cursor_size));

	wl_cursor_theme = wl_cursor_theme_load(cursor_theme_name.utf8().get_data(), p_cursor_size, registry.wl_shm);

	ERR_FAIL_NULL_V_MSG(wl_cursor_theme, false, "Can't load any cursor theme.");

	static const char *cursor_names[] = {
		"left_ptr",
		"xterm",
		"hand2",
		"cross",
		"watch",
		"left_ptr_watch",
		"fleur",
		"dnd-move",
		"crossed_circle",
		"v_double_arrow",
		"h_double_arrow",
		"size_bdiag",
		"size_fdiag",
		"move",
		"row_resize",
		"col_resize",
		"question_arrow"
	};

	static const char *cursor_names_fallback[] = {
		nullptr,
		nullptr,
		"pointer",
		"cross",
		"wait",
		"progress",
		"grabbing",
		"hand1",
		"forbidden",
		"ns-resize",
		"ew-resize",
		"fd_double_arrow",
		"bd_double_arrow",
		"fleur",
		"sb_v_double_arrow",
		"sb_h_double_arrow",
		"help"
	};

	for (int i = 0; i < DisplayServer::CURSOR_MAX; i++) {
		struct wl_cursor *cursor = wl_cursor_theme_get_cursor(wl_cursor_theme, cursor_names[i]);

		if (!cursor && cursor_names_fallback[i]) {
			cursor = wl_cursor_theme_get_cursor(wl_cursor_theme, cursor_names_fallback[i]);
		}

		if (cursor && cursor->image_count > 0) {
			wl_cursors[i] = cursor;
		} else {
			wl_cursors[i] = nullptr;
			print_verbose("Failed loading cursor: " + String(cursor_names[i]));
		}
	}

	return true;
}

void WaylandThread::_update_scale(int p_scale) {
	if (p_scale <= cursor_scale) {
		return;
	}

	print_verbose(vformat("Bumping cursor scale to %d", p_scale));

	// There's some display that's bigger than the cache, let's update it.
	cursor_scale = p_scale;

	if (wl_cursor_theme == nullptr) {
		// Ugh. Either we're still initializing (this must've been called from the
		// first roundtrips) or we had some error while doing so. We'll trust that it
		// will be updated for us if needed.
		return;
	}

	int cursor_size = unscaled_cursor_size * p_scale;

	if (_load_cursor_theme(cursor_size)) {
		for (struct wl_seat *wl_seat : registry.wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			seat_state_update_cursor(ss);
		}
	}
}

void WaylandThread::_frame_wl_callback_on_done(void *data, struct wl_callback *wl_callback, uint32_t callback_data) {
	wl_callback_destroy(wl_callback);

	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);
	ERR_FAIL_NULL(ws->wayland_thread);
	ERR_FAIL_NULL(ws->wl_surface);

	ws->wayland_thread->set_frame();

	ws->frame_callback = wl_surface_frame(ws->wl_surface),
	wl_callback_add_listener(ws->frame_callback, &frame_wl_callback_listener, ws);

	if (ws->wl_surface && ws->buffer_scale_changed) {
		// NOTE: We're only now setting the buffer scale as the idea is to get this
		// data committed together with the new frame, all by the rendering driver.
		// This is important because we might otherwise set an invalid combination of
		// buffer size and scale (e.g. odd size and 2x scale). We're pretty much
		// guaranteed to get a proper buffer in the next render loop as the rescaling
		// method also informs the engine of a "window rect change", triggering
		// rendering if needed.
		wl_surface_set_buffer_scale(ws->wl_surface, window_state_get_preferred_buffer_scale(ws));
	}
}

void WaylandThread::_cursor_frame_callback_on_done(void *data, struct wl_callback *wl_callback, uint32_t time_ms) {
	wl_callback_destroy(wl_callback);

	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->cursor_frame_callback = nullptr;

	ss->cursor_time_ms = time_ms;

	seat_state_update_cursor(ss);
}

// NOTE: This must be started after a valid wl_display is loaded.
void WaylandThread::_poll_events_thread(void *p_data) {
	ThreadData *data = (ThreadData *)p_data;
	ERR_FAIL_NULL(data);
	ERR_FAIL_NULL(data->wl_display);

	struct pollfd poll_fd;
	poll_fd.fd = wl_display_get_fd(data->wl_display);
	poll_fd.events = POLLIN | POLLHUP;

	while (true) {
		// Empty the event queue while it's full.
		while (wl_display_prepare_read(data->wl_display) != 0) {
			// We aren't using wl_display_dispatch(), instead "manually" handling events
			// through wl_display_dispatch_pending so that we can use a global mutex and
			// be sure that this and the main thread won't race over stuff, as long as
			// the main thread locks it too.
			//
			// Note that the main thread can still call wl_display_roundtrip as that
			// method directly handles all events, effectively bypassing this polling
			// loop and thus the mutex locking, avoiding a deadlock.
			MutexLock mutex_lock(data->mutex);

			if (wl_display_dispatch_pending(data->wl_display) == -1) {
				// Oh no. We'll check and handle any display error below.
				break;
			}
		}

		int werror = wl_display_get_error(data->wl_display);

		if (werror) {
			if (werror == EPROTO) {
				struct wl_interface *wl_interface = nullptr;
				uint32_t id = 0;

				int error_code = wl_display_get_protocol_error(data->wl_display, (const struct wl_interface **)&wl_interface, &id);
				CRASH_NOW_MSG(vformat("Wayland protocol error %d on interface %s@%d.", error_code, wl_interface ? wl_interface->name : "unknown", id));
			} else {
				CRASH_NOW_MSG(vformat("Wayland client error code %d.", werror));
			}
		}

		wl_display_flush(data->wl_display);

		// Wait for the event file descriptor to have new data.
		poll(&poll_fd, 1, -1);

		if (data->thread_done.is_set()) {
			wl_display_cancel_read(data->wl_display);
			break;
		}

		if (poll_fd.revents | POLLIN) {
			// Load the queues with fresh new data.
			wl_display_read_events(data->wl_display);
		} else {
			// Oh well... Stop signaling that we want to read.
			wl_display_cancel_read(data->wl_display);
		}

		// The docs advise to redispatch unconditionally and it looks like that if we
		// don't do this we can't catch protocol errors, which is bad.
		MutexLock mutex_lock(data->mutex);
		wl_display_dispatch_pending(data->wl_display);
	}
}

struct wl_display *WaylandThread::get_wl_display() const {
	return wl_display;
}

// NOTE: Stuff like libdecor can (and will) register foreign proxies which
// aren't formatted as we like. This method is needed to detect whether a proxy
// has our tag. Also, be careful! The proxy has to be manually tagged or it
// won't be recognized.
bool WaylandThread::wl_proxy_is_godot(struct wl_proxy *p_proxy) {
	ERR_FAIL_NULL_V(p_proxy, false);

	return wl_proxy_get_tag(p_proxy) == &proxy_tag;
}

void WaylandThread::wl_proxy_tag_godot(struct wl_proxy *p_proxy) {
	ERR_FAIL_NULL(p_proxy);

	wl_proxy_set_tag(p_proxy, &proxy_tag);
}

// Returns the wl_surface's `WindowState`, otherwise `nullptr`.
// NOTE: This will fail if the surface isn't tagged as ours.
WaylandThread::WindowState *WaylandThread::wl_surface_get_window_state(struct wl_surface *p_surface) {
	if (p_surface && wl_proxy_is_godot((wl_proxy *)p_surface)) {
		return (WindowState *)wl_surface_get_user_data(p_surface);
	}

	return nullptr;
}

// Returns the wl_outputs's `ScreenState`, otherwise `nullptr`.
// NOTE: This will fail if the output isn't tagged as ours.
WaylandThread::ScreenState *WaylandThread::wl_output_get_screen_state(struct wl_output *p_output) {
	if (p_output && wl_proxy_is_godot((wl_proxy *)p_output)) {
		return (ScreenState *)wl_output_get_user_data(p_output);
	}

	return nullptr;
}

// Returns the wl_seat's `SeatState`, otherwise `nullptr`.
// NOTE: This will fail if the output isn't tagged as ours.
WaylandThread::SeatState *WaylandThread::wl_seat_get_seat_state(struct wl_seat *p_seat) {
	if (p_seat && wl_proxy_is_godot((wl_proxy *)p_seat)) {
		return (SeatState *)wl_seat_get_user_data(p_seat);
	}

	return nullptr;
}

// Returns the wp_tablet_tool's `TabletToolState`, otherwise `nullptr`.
// NOTE: This will fail if the output isn't tagged as ours.
WaylandThread::TabletToolState *WaylandThread::wp_tablet_tool_get_state(struct zwp_tablet_tool_v2 *p_tool) {
	if (p_tool && wl_proxy_is_godot((wl_proxy *)p_tool)) {
		return (TabletToolState *)zwp_tablet_tool_v2_get_user_data(p_tool);
	}

	return nullptr;
}
// Returns the wl_data_offer's `OfferState`, otherwise `nullptr`.
// NOTE: This will fail if the output isn't tagged as ours.
WaylandThread::OfferState *WaylandThread::wl_data_offer_get_offer_state(struct wl_data_offer *p_offer) {
	if (p_offer && wl_proxy_is_godot((wl_proxy *)p_offer)) {
		return (OfferState *)wl_data_offer_get_user_data(p_offer);
	}

	return nullptr;
}

// Returns the wl_data_offer's `OfferState`, otherwise `nullptr`.
// NOTE: This will fail if the output isn't tagged as ours.
WaylandThread::OfferState *WaylandThread::wp_primary_selection_offer_get_offer_state(struct zwp_primary_selection_offer_v1 *p_offer) {
	if (p_offer && wl_proxy_is_godot((wl_proxy *)p_offer)) {
		return (OfferState *)zwp_primary_selection_offer_v1_get_user_data(p_offer);
	}

	return nullptr;
}

// This is implemented as a method because this is the simplest way of
// accounting for dynamic output scale changes.
int WaylandThread::window_state_get_preferred_buffer_scale(WindowState *p_ws) {
	ERR_FAIL_NULL_V(p_ws, 1);

	if (p_ws->preferred_fractional_scale > 0) {
		// We're scaling fractionally. Per spec, the buffer scale is always 1.
		return 1;
	}

	if (p_ws->wl_outputs.is_empty()) {
		DEBUG_LOG_WAYLAND_THREAD("Window has no output associated, returning buffer scale of 1.");
		return 1;
	}

	// TODO: Cache value?
	int max_size = 1;

	// ================================ IMPORTANT =================================
	// NOTE: Due to a Godot limitation, we can't really rescale the whole UI yet.
	// Because of this reason, all platforms have resorted to forcing the highest
	// scale possible of a system on any window, despite of what screen it's onto.
	// On this backend everything's already in place for dynamic window scale
	// handling, but in the meantime we'll just select the biggest _global_ output.
	// To restore dynamic scale selection, simply iterate over `p_ws->wl_outputs`
	// instead.
	for (struct wl_output *wl_output : p_ws->registry->wl_outputs) {
		ScreenState *ss = wl_output_get_screen_state(wl_output);

		if (ss && ss->pending_data.scale > max_size) {
			// NOTE: For some mystical reason, wl_output.done is emitted _after_ windows
			// get resized but the scale event gets sent _before_ that. I'm still leaning
			// towards the idea that rescaling when a window gets a resolution change is a
			// pretty good approach, but this means that we'll have to use the screen data
			// before it's "committed".
			// FIXME: Use the committed data. Somehow.
			max_size = ss->pending_data.scale;
		}
	}

	return max_size;
}

double WaylandThread::window_state_get_scale_factor(WindowState *p_ws) {
	ERR_FAIL_NULL_V(p_ws, 1);

	if (p_ws->fractional_scale > 0) {
		// The fractional scale amount takes priority.
		return p_ws->fractional_scale;
	}

	return p_ws->buffer_scale;
}

void WaylandThread::window_state_update_size(WindowState *p_ws, int p_width, int p_height) {
	ERR_FAIL_NULL(p_ws);

	int preferred_buffer_scale = window_state_get_preferred_buffer_scale(p_ws);
	bool using_fractional = p_ws->preferred_fractional_scale > 0;

	// If neither is true we no-op.
	bool scale_changed = false;
	bool size_changed = false;

	if (p_ws->rect.size.width != p_width || p_ws->rect.size.height != p_height) {
		p_ws->rect.size.width = p_width;
		p_ws->rect.size.height = p_height;

		size_changed = true;
	}

	if (using_fractional && p_ws->fractional_scale != p_ws->preferred_fractional_scale) {
		p_ws->fractional_scale = p_ws->preferred_fractional_scale;
		scale_changed = true;
	}

	if (p_ws->buffer_scale != preferred_buffer_scale) {
		// The buffer scale is always important, even if we use frac scaling.
		p_ws->buffer_scale = preferred_buffer_scale;
		p_ws->buffer_scale_changed = true;

		if (!using_fractional) {
			// We don't bother updating everything else if it's turned on though.
			scale_changed = true;
		}
	}

	if (p_ws->wl_surface && (size_changed || scale_changed)) {
		if (p_ws->wp_viewport) {
			wp_viewport_set_destination(p_ws->wp_viewport, p_width, p_height);
		}

		if (p_ws->xdg_surface) {
			xdg_surface_set_window_geometry(p_ws->xdg_surface, 0, 0, p_width, p_height);
		}
	}

#ifdef LIBDECOR_ENABLED
	if (p_ws->libdecor_frame) {
		struct libdecor_state *state = libdecor_state_new(p_width, p_height);
		libdecor_frame_commit(p_ws->libdecor_frame, state, p_ws->pending_libdecor_configuration);
		libdecor_state_free(state);
		p_ws->pending_libdecor_configuration = nullptr;
	}
#endif

	if (size_changed || scale_changed) {
		Size2i scaled_size = scale_vector2i(p_ws->rect.size, window_state_get_scale_factor(p_ws));

		if (using_fractional) {
			DEBUG_LOG_WAYLAND_THREAD(vformat("Resizing the window from %s to %s (fractional scale x%f).", p_ws->rect.size, scaled_size, p_ws->fractional_scale));
		} else {
			DEBUG_LOG_WAYLAND_THREAD(vformat("Resizing the window from %s to %s (buffer scale x%d).", p_ws->rect.size, scaled_size, p_ws->buffer_scale));
		}

		// FIXME: Actually resize the hint instead of centering it.
		p_ws->wayland_thread->pointer_set_hint(scaled_size / 2);

		Ref<WindowRectMessage> rect_msg;
		rect_msg.instantiate();
		rect_msg->rect = p_ws->rect;
		rect_msg->rect.size = scaled_size;
		p_ws->wayland_thread->push_message(rect_msg);
	}

	if (scale_changed) {
		Ref<WindowEventMessage> dpi_msg;
		dpi_msg.instantiate();
		dpi_msg->event = DisplayServer::WINDOW_EVENT_DPI_CHANGE;
		p_ws->wayland_thread->push_message(dpi_msg);
	}
}

// Scales a vector according to wp_fractional_scale's rules, where coordinates
// must be scaled with away from zero half-rounding.
Vector2i WaylandThread::scale_vector2i(const Vector2i &p_vector, double p_amount) {
	// This snippet is tiny, I know, but this is done a lot.
	int x = round(p_vector.x * p_amount);
	int y = round(p_vector.y * p_amount);

	return Vector2i(x, y);
}

void WaylandThread::seat_state_unlock_pointer(SeatState *p_ss) {
	ERR_FAIL_NULL(p_ss);

	if (p_ss->wl_pointer == nullptr) {
		return;
	}

	if (p_ss->wp_locked_pointer) {
		zwp_locked_pointer_v1_destroy(p_ss->wp_locked_pointer);
		p_ss->wp_locked_pointer = nullptr;
	}

	if (p_ss->wp_confined_pointer) {
		zwp_confined_pointer_v1_destroy(p_ss->wp_confined_pointer);
		p_ss->wp_confined_pointer = nullptr;
	}
}

void WaylandThread::seat_state_lock_pointer(SeatState *p_ss) {
	ERR_FAIL_NULL(p_ss);

	if (p_ss->wl_pointer == nullptr) {
		return;
	}

	if (registry.wp_pointer_constraints == nullptr) {
		return;
	}

	if (p_ss->wp_locked_pointer == nullptr) {
		struct wl_surface *locked_surface = p_ss->last_pointed_surface;

		if (locked_surface == nullptr) {
			locked_surface = window_get_wl_surface(DisplayServer::MAIN_WINDOW_ID);
		}

		ERR_FAIL_NULL(locked_surface);

		p_ss->wp_locked_pointer = zwp_pointer_constraints_v1_lock_pointer(registry.wp_pointer_constraints, locked_surface, p_ss->wl_pointer, nullptr, ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
	}
}

void WaylandThread::seat_state_set_hint(SeatState *p_ss, int p_x, int p_y) {
	if (p_ss->wp_locked_pointer == nullptr) {
		return;
	}

	zwp_locked_pointer_v1_set_cursor_position_hint(p_ss->wp_locked_pointer, wl_fixed_from_int(p_x), wl_fixed_from_int(p_y));
}

void WaylandThread::seat_state_confine_pointer(SeatState *p_ss) {
	ERR_FAIL_NULL(p_ss);

	if (p_ss->wl_pointer == nullptr) {
		return;
	}

	if (registry.wp_pointer_constraints == nullptr) {
		return;
	}

	if (p_ss->wp_confined_pointer == nullptr) {
		struct wl_surface *confined_surface = p_ss->last_pointed_surface;

		if (confined_surface == nullptr) {
			confined_surface = window_get_wl_surface(DisplayServer::MAIN_WINDOW_ID);
		}

		ERR_FAIL_NULL(confined_surface);

		p_ss->wp_confined_pointer = zwp_pointer_constraints_v1_confine_pointer(registry.wp_pointer_constraints, confined_surface, p_ss->wl_pointer, nullptr, ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
	}
}

void WaylandThread::seat_state_update_cursor(SeatState *p_ss) {
	ERR_FAIL_NULL(p_ss);

	WaylandThread *thread = p_ss->wayland_thread;
	ERR_FAIL_NULL(p_ss->wayland_thread);

	if (!p_ss->wl_pointer || !p_ss->cursor_surface) {
		return;
	}

	// NOTE: Those values are valid by default and will hide the cursor when
	// unchanged.
	struct wl_buffer *cursor_buffer = nullptr;
	uint32_t hotspot_x = 0;
	uint32_t hotspot_y = 0;
	int scale = 1;

	if (thread->cursor_visible) {
		DisplayServer::CursorShape shape = thread->cursor_shape;

		struct CustomCursor *custom_cursor = thread->custom_cursors.getptr(shape);

		if (custom_cursor) {
			cursor_buffer = custom_cursor->wl_buffer;
			hotspot_x = custom_cursor->hotspot.x;
			hotspot_y = custom_cursor->hotspot.y;

			// We can't really reasonably scale custom cursors, so we'll let the
			// compositor do it for us (badly).
			scale = 1;
		} else {
			struct wl_cursor *wl_cursor = thread->wl_cursors[shape];

			if (!wl_cursor) {
				return;
			}

			int frame_idx = 0;

			if (wl_cursor->image_count > 1) {
				// The cursor is animated.
				frame_idx = wl_cursor_frame(wl_cursor, p_ss->cursor_time_ms);

				if (!p_ss->cursor_frame_callback) {
					// Since it's animated, we'll re-update it the next frame.
					p_ss->cursor_frame_callback = wl_surface_frame(p_ss->cursor_surface);
					wl_callback_add_listener(p_ss->cursor_frame_callback, &cursor_frame_callback_listener, p_ss);
				}
			}

			struct wl_cursor_image *wl_cursor_image = wl_cursor->images[frame_idx];

			scale = thread->cursor_scale;

			cursor_buffer = wl_cursor_image_get_buffer(wl_cursor_image);

			// As the surface's buffer is scaled (thus the surface is smaller) and the
			// hotspot must be expressed in surface-local coordinates, we need to scale
			// it down accordingly.
			hotspot_x = wl_cursor_image->hotspot_x / scale;
			hotspot_y = wl_cursor_image->hotspot_y / scale;
		}
	}

	wl_pointer_set_cursor(p_ss->wl_pointer, p_ss->pointer_enter_serial, p_ss->cursor_surface, hotspot_x, hotspot_y);
	wl_surface_set_buffer_scale(p_ss->cursor_surface, scale);
	wl_surface_attach(p_ss->cursor_surface, cursor_buffer, 0, 0);
	wl_surface_damage_buffer(p_ss->cursor_surface, 0, 0, INT_MAX, INT_MAX);

	wl_surface_commit(p_ss->cursor_surface);
}

void WaylandThread::seat_state_echo_keys(SeatState *p_ss) {
	ERR_FAIL_NULL(p_ss);

	if (p_ss->wl_keyboard == nullptr) {
		return;
	}

	// TODO: Comment and document out properly this block of code.
	// In short, this implements key repeating.
	if (p_ss->repeat_key_delay_msec && p_ss->repeating_keycode != XKB_KEYCODE_INVALID) {
		uint64_t current_ticks = OS::get_singleton()->get_ticks_msec();
		uint64_t delayed_start_ticks = p_ss->last_repeat_start_msec + p_ss->repeat_start_delay_msec;

		if (p_ss->last_repeat_msec < delayed_start_ticks) {
			p_ss->last_repeat_msec = delayed_start_ticks;
		}

		if (current_ticks >= delayed_start_ticks) {
			uint64_t ticks_delta = current_ticks - p_ss->last_repeat_msec;

			int keys_amount = (ticks_delta / p_ss->repeat_key_delay_msec);

			for (int i = 0; i < keys_amount; i++) {
				Ref<InputEventKey> k;
				k.instantiate();

				if (!_seat_state_configure_key_event(*p_ss, k, p_ss->repeating_keycode, true)) {
					continue;
				}

				k->set_echo(true);

				Input::get_singleton()->parse_input_event(k);
			}

			p_ss->last_repeat_msec += ticks_delta - (ticks_delta % p_ss->repeat_key_delay_msec);
		}
	}
}

void WaylandThread::push_message(Ref<Message> message) {
	messages.push_back(message);
}

bool WaylandThread::has_message() {
	return messages.front() != nullptr;
}

Ref<WaylandThread::Message> WaylandThread::pop_message() {
	if (messages.front() != nullptr) {
		Ref<Message> msg = messages.front()->get();
		messages.pop_front();
		return msg;
	}

	// This method should only be called if `has_messages` returns true but if
	// that isn't the case we'll just return an invalid `Ref`. After all, due to
	// its `InputEvent`-like interface, we still have to dynamically cast and check
	// the `Ref`'s validity anyways.
	return Ref<Message>();
}

void WaylandThread::window_create(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
	// TODO: Implement multi-window support.
	WindowState &ws = main_window;

	ws.registry = &registry;
	ws.wayland_thread = this;

	ws.rect.size.width = p_width;
	ws.rect.size.height = p_height;

	ws.wl_surface = wl_compositor_create_surface(registry.wl_compositor);
	wl_proxy_tag_godot((struct wl_proxy *)ws.wl_surface);
	wl_surface_add_listener(ws.wl_surface, &wl_surface_listener, &ws);

	if (registry.wp_viewporter) {
		ws.wp_viewport = wp_viewporter_get_viewport(registry.wp_viewporter, ws.wl_surface);

		if (registry.wp_fractional_scale_manager) {
			ws.wp_fractional_scale = wp_fractional_scale_manager_v1_get_fractional_scale(registry.wp_fractional_scale_manager, ws.wl_surface);
			wp_fractional_scale_v1_add_listener(ws.wp_fractional_scale, &wp_fractional_scale_listener, &ws);
		}
	}

	bool decorated = false;

#ifdef LIBDECOR_ENABLED
	if (!decorated && libdecor_context) {
		ws.libdecor_frame = libdecor_decorate(libdecor_context, ws.wl_surface, (struct libdecor_frame_interface *)&libdecor_frame_interface, &ws);
		libdecor_frame_map(ws.libdecor_frame);

		decorated = true;
	}
#endif

	if (!decorated) {
		// libdecor has failed loading or is disabled, we shall handle xdg_toplevel
		// creation and decoration ourselves (and by decorating for now I just mean
		// asking for SSDs and hoping for the best).
		ws.xdg_surface = xdg_wm_base_get_xdg_surface(registry.xdg_wm_base, ws.wl_surface);
		xdg_surface_add_listener(ws.xdg_surface, &xdg_surface_listener, &ws);

		ws.xdg_toplevel = xdg_surface_get_toplevel(ws.xdg_surface);
		xdg_toplevel_add_listener(ws.xdg_toplevel, &xdg_toplevel_listener, &ws);

		if (registry.xdg_decoration_manager) {
			ws.xdg_toplevel_decoration = zxdg_decoration_manager_v1_get_toplevel_decoration(registry.xdg_decoration_manager, ws.xdg_toplevel);
			zxdg_toplevel_decoration_v1_add_listener(ws.xdg_toplevel_decoration, &xdg_toplevel_decoration_listener, &ws);

			decorated = true;
		}
	}

	ws.frame_callback = wl_surface_frame(ws.wl_surface);
	wl_callback_add_listener(ws.frame_callback, &frame_wl_callback_listener, &ws);

	if (registry.xdg_exporter) {
		ws.xdg_exported = zxdg_exporter_v1_export(registry.xdg_exporter, ws.wl_surface);
		zxdg_exported_v1_add_listener(ws.xdg_exported, &xdg_exported_listener, &ws);
	}

	wl_surface_commit(ws.wl_surface);

	// Wait for the surface to be configured before continuing.
	wl_display_roundtrip(wl_display);
}

struct wl_surface *WaylandThread::window_get_wl_surface(DisplayServer::WindowID p_window_id) const {
	// TODO: Use window IDs for multiwindow support.
	const WindowState &ws = main_window;

	return ws.wl_surface;
}

void WaylandThread::window_set_max_size(DisplayServer::WindowID p_window_id, const Size2i &p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	Vector2i logical_max_size = p_size / window_state_get_scale_factor(&ws);

	if (ws.wl_surface && ws.xdg_toplevel) {
		xdg_toplevel_set_max_size(ws.xdg_toplevel, logical_max_size.width, logical_max_size.height);
	}

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		libdecor_frame_set_max_content_size(ws.libdecor_frame, logical_max_size.width, logical_max_size.height);
	}

	// FIXME: I'm not sure whether we have to commit the surface for this to apply.
#endif
}

void WaylandThread::window_set_min_size(DisplayServer::WindowID p_window_id, const Size2i &p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	Size2i logical_min_size = p_size / window_state_get_scale_factor(&ws);

	if (ws.wl_surface && ws.xdg_toplevel) {
		xdg_toplevel_set_min_size(ws.xdg_toplevel, logical_min_size.width, logical_min_size.height);
	}

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		libdecor_frame_set_min_content_size(ws.libdecor_frame, logical_min_size.width, logical_min_size.height);
	}

	// FIXME: I'm not sure whether we have to commit the surface for this to apply.
#endif
}

bool WaylandThread::window_can_set_mode(DisplayServer::WindowID p_window_id, DisplayServer::WindowMode p_window_mode) const {
	// TODO: Use window IDs for multiwindow support.
	const WindowState &ws = main_window;

	switch (p_window_mode) {
		case DisplayServer::WINDOW_MODE_WINDOWED: {
			// Looks like it's guaranteed.
			return true;
		};

		case DisplayServer::WINDOW_MODE_MINIMIZED: {
#ifdef LIBDECOR_ENABLED
			if (ws.libdecor_frame) {
				return libdecor_frame_has_capability(ws.libdecor_frame, LIBDECOR_ACTION_MINIMIZE);
			}
#endif // LIBDECOR_ENABLED

			return ws.can_minimize;
		};

		case DisplayServer::WINDOW_MODE_MAXIMIZED: {
			// NOTE: libdecor doesn't seem to have a maximize capability query?
			// The fact that there's a fullscreen one makes me suspicious.
			return ws.can_maximize;
		};

		case DisplayServer::WINDOW_MODE_FULLSCREEN: {
#ifdef LIBDECOR_ENABLED
			if (ws.libdecor_frame) {
				return libdecor_frame_has_capability(ws.libdecor_frame, LIBDECOR_ACTION_FULLSCREEN);
			}
#endif // LIBDECOR_ENABLED

			return ws.can_fullscreen;
		};

		case DisplayServer::WINDOW_MODE_EXCLUSIVE_FULLSCREEN: {
			// I'm not really sure but from what I can find Wayland doesn't really have
			// the concept of exclusive fullscreen.
			// TODO: Discuss whether to fallback to regular fullscreen or not.
			return false;
		};
	}

	return false;
}

void WaylandThread::window_try_set_mode(DisplayServer::WindowID p_window_id, DisplayServer::WindowMode p_window_mode) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	if (ws.mode == p_window_mode) {
		return;
	}

	// Don't waste time with hidden windows and whatnot. Behave like it worked.
#ifdef LIBDECOR_ENABLED
	if ((!ws.wl_surface || !ws.xdg_toplevel) && !ws.libdecor_frame) {
#else
	if (!ws.wl_surface || !ws.xdg_toplevel) {
#endif // LIBDECOR_ENABLED
		ws.mode = p_window_mode;
		return;
	}

	// Return back to a windowed state so that we can apply what the user asked.
	switch (ws.mode) {
		case DisplayServer::WINDOW_MODE_WINDOWED: {
			// Do nothing.
		} break;

		case DisplayServer::WINDOW_MODE_MINIMIZED: {
			// We can't do much according to the xdg_shell protocol. I have no idea
			// whether this implies that we should return or who knows what. For now
			// we'll do nothing.
			// TODO: Test this properly.
		} break;

		case DisplayServer::WINDOW_MODE_MAXIMIZED: {
			// Try to unmaximize. This isn't garaunteed to work actually, so we'll have
			// to check whether something changed.
			if (ws.xdg_toplevel) {
				xdg_toplevel_unset_maximized(ws.xdg_toplevel);
			}

#ifdef LIBDECOR_ENABLED
			if (ws.libdecor_frame) {
				libdecor_frame_unset_maximized(ws.libdecor_frame);
			}
#endif // LIBDECOR_ENABLED
		} break;

		case DisplayServer::WINDOW_MODE_FULLSCREEN:
		case DisplayServer::WINDOW_MODE_EXCLUSIVE_FULLSCREEN: {
			// Same thing as above, unset fullscreen and check later if it worked.
			if (ws.xdg_toplevel) {
				xdg_toplevel_unset_fullscreen(ws.xdg_toplevel);
			}

#ifdef LIBDECOR_ENABLED
			if (ws.libdecor_frame) {
				libdecor_frame_unset_fullscreen(ws.libdecor_frame);
			}
#endif // LIBDECOR_ENABLED
		} break;
	}

	// Wait for a configure event and hope that something changed.
	wl_display_roundtrip(wl_display);

	if (ws.mode != DisplayServer::WINDOW_MODE_WINDOWED) {
		// The compositor refused our "normalization" request. It'd be useless or
		// unpredictable to attempt setting a new state. We're done.
		return;
	}

	// Ask the compositor to set the state indicated by the new mode.
	switch (p_window_mode) {
		case DisplayServer::WINDOW_MODE_WINDOWED: {
			// Do nothing. We're already windowed.
		} break;

		case DisplayServer::WINDOW_MODE_MINIMIZED: {
			if (!window_can_set_mode(p_window_id, p_window_mode)) {
				// Minimization is special (read below). Better not mess with it if the
				// compositor explicitly announces that it doesn't support it.
				break;
			}

			if (ws.xdg_toplevel) {
				xdg_toplevel_set_minimized(ws.xdg_toplevel);
			}

#ifdef LIBDECOR_ENABLED
			if (ws.libdecor_frame) {
				libdecor_frame_set_minimized(ws.libdecor_frame);
			}
#endif // LIBDECOR_ENABLED
	   // We have no way to actually detect this state, so we'll have to report it
	   // manually to the engine (hoping that it worked). In the worst case it'll
	   // get reset by the next configure event.
			ws.mode = DisplayServer::WINDOW_MODE_MINIMIZED;
		} break;

		case DisplayServer::WINDOW_MODE_MAXIMIZED: {
			if (ws.xdg_toplevel) {
				xdg_toplevel_set_maximized(ws.xdg_toplevel);
			}

#ifdef LIBDECOR_ENABLED
			if (ws.libdecor_frame) {
				libdecor_frame_set_maximized(ws.libdecor_frame);
			}
#endif // LIBDECOR_ENABLED
		} break;

		case DisplayServer::WINDOW_MODE_FULLSCREEN: {
			if (ws.xdg_toplevel) {
				xdg_toplevel_set_fullscreen(ws.xdg_toplevel, nullptr);
			}

#ifdef LIBDECOR_ENABLED
			if (ws.libdecor_frame) {
				libdecor_frame_set_fullscreen(ws.libdecor_frame, nullptr);
			}
#endif // LIBDECOR_ENABLED
		} break;

		default: {
		} break;
	}
}

void WaylandThread::window_set_borderless(DisplayServer::WindowID p_window_id, bool p_borderless) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	if (ws.xdg_toplevel_decoration) {
		if (p_borderless) {
			// We implement borderless windows by simply asking the compositor to let
			// us handle decorations (we don't).
			zxdg_toplevel_decoration_v1_set_mode(ws.xdg_toplevel_decoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_CLIENT_SIDE);
		} else {
			zxdg_toplevel_decoration_v1_set_mode(ws.xdg_toplevel_decoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
		}
	}

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		bool visible_current = libdecor_frame_is_visible(ws.libdecor_frame);
		bool visible_target = !p_borderless;

		// NOTE: We have to do this otherwise we trip on a libdecor bug where it's
		// possible to destroy the frame more than once, by setting the visibility
		// to false multiple times and thus crashing.
		if (visible_current != visible_target) {
			print_verbose(vformat("Setting libdecor frame visibility to %d", visible_target));
			libdecor_frame_set_visibility(ws.libdecor_frame, visible_target);
		}
	}
#endif // LIBDECOR_ENABLED
}

void WaylandThread::window_set_title(DisplayServer::WindowID p_window_id, const String &p_title) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		libdecor_frame_set_title(ws.libdecor_frame, p_title.utf8());
	}
#endif // LIBDECOR_ENABLE

	if (ws.xdg_toplevel) {
		xdg_toplevel_set_title(ws.xdg_toplevel, p_title.utf8());
	}
}

void WaylandThread::window_set_app_id(DisplayServer::WindowID p_window_id, const String &p_app_id) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		libdecor_frame_set_app_id(ws.libdecor_frame, p_app_id.utf8());
		return;
	}
#endif // LIBDECOR_ENABLED

	if (ws.xdg_toplevel) {
		xdg_toplevel_set_app_id(ws.xdg_toplevel, p_app_id.utf8());
		return;
	}
}

DisplayServer::WindowMode WaylandThread::window_get_mode(DisplayServer::WindowID p_window_id) const {
	// TODO: Use window IDs for multiwindow support.
	const WindowState &ws = main_window;

	return ws.mode;
}

void WaylandThread::window_request_attention(DisplayServer::WindowID p_window_id) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	if (registry.xdg_activation) {
		// Window attention requests are done through the XDG activation protocol.
		xdg_activation_token_v1 *xdg_activation_token = xdg_activation_v1_get_activation_token(registry.xdg_activation);
		xdg_activation_token_v1_add_listener(xdg_activation_token, &xdg_activation_token_listener, &ws);
		xdg_activation_token_v1_commit(xdg_activation_token);
	}
}

void WaylandThread::window_set_idle_inhibition(DisplayServer::WindowID p_window_id, bool p_enable) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	if (p_enable) {
		if (ws.registry->wp_idle_inhibit_manager && !ws.wp_idle_inhibitor) {
			ERR_FAIL_NULL(ws.wl_surface);
			ws.wp_idle_inhibitor = zwp_idle_inhibit_manager_v1_create_inhibitor(ws.registry->wp_idle_inhibit_manager, ws.wl_surface);
		}
	} else {
		if (ws.wp_idle_inhibitor) {
			zwp_idle_inhibitor_v1_destroy(ws.wp_idle_inhibitor);
			ws.wp_idle_inhibitor = nullptr;
		}
	}
}

bool WaylandThread::window_get_idle_inhibition(DisplayServer::WindowID p_window_id) const {
	// TODO: Use window IDs for multiwindow support.
	const WindowState &ws = main_window;

	return ws.wp_idle_inhibitor != nullptr;
}

WaylandThread::ScreenData WaylandThread::screen_get_data(int p_screen) const {
	ERR_FAIL_INDEX_V(p_screen, registry.wl_outputs.size(), ScreenData());

	return wl_output_get_screen_state(registry.wl_outputs.get(p_screen))->data;
}

int WaylandThread::get_screen_count() const {
	return registry.wl_outputs.size();
}

DisplayServer::WindowID WaylandThread::pointer_get_pointed_window_id() const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss) {
		WindowState *ws = wl_surface_get_window_state(ss->pointed_surface);

		if (ws) {
			return ws->id;
		}
	}

	return DisplayServer::INVALID_WINDOW_ID;
}

void WaylandThread::pointer_set_constraint(PointerConstraint p_constraint) {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss) {
		seat_state_unlock_pointer(ss);

		if (p_constraint == PointerConstraint::LOCKED) {
			seat_state_lock_pointer(ss);
		} else if (p_constraint == PointerConstraint::CONFINED) {
			seat_state_confine_pointer(ss);
		}
	}

	pointer_constraint = p_constraint;
}

void WaylandThread::pointer_set_hint(const Point2i &p_hint) {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);
	if (!ss) {
		return;
	}

	WindowState *ws = wl_surface_get_window_state(ss->pointed_surface);

	int hint_x = 0;
	int hint_y = 0;

	if (ws) {
		// NOTE: It looks like it's not really recommended to convert from
		// "godot-space" to "wayland-space" and in general I received mixed feelings
		// discussing about this. I'm not really sure about the maths behind this but,
		// oh well, we're setting a cursor hint. ¯\_(ツ)_/¯
		// See: https://oftc.irclog.whitequark.org/wayland/2023-08-23#1692756914-1692816818
		hint_x = round(p_hint.x / window_state_get_scale_factor(ws));
		hint_y = round(p_hint.y / window_state_get_scale_factor(ws));
	}

	if (ss) {
		seat_state_set_hint(ss, hint_x, hint_y);
	}
}

WaylandThread::PointerConstraint WaylandThread::pointer_get_constraint() const {
	return pointer_constraint;
}

BitField<MouseButtonMask> WaylandThread::pointer_get_button_mask() const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss) {
		return ss->pointer_data.pressed_button_mask;
	}

	return BitField<MouseButtonMask>();
}

Error WaylandThread::init() {
#ifdef SOWRAP_ENABLED
#ifdef DEBUG_ENABLED
	int dylibloader_verbose = 1;
#else
	int dylibloader_verbose = 0;
#endif // DEBUG_ENABLED

	if (initialize_wayland_client(dylibloader_verbose) != 0) {
		WARN_PRINT("Can't load the Wayland client library.");
		return ERR_CANT_CREATE;
	}

	if (initialize_wayland_cursor(dylibloader_verbose) != 0) {
		WARN_PRINT("Can't load the Wayland cursor library.");
		return ERR_CANT_CREATE;
	}

	if (initialize_xkbcommon(dylibloader_verbose) != 0) {
		WARN_PRINT("Can't load the XKBcommon library.");
		return ERR_CANT_CREATE;
	}
#endif // SOWRAP_ENABLED

	KeyMappingXKB::initialize();

	wl_display = wl_display_connect(nullptr);
	ERR_FAIL_NULL_V_MSG(wl_display, ERR_CANT_CREATE, "Can't connect to a Wayland display.");

	thread_data.wl_display = wl_display;

	events_thread.start(_poll_events_thread, &thread_data);

	wl_registry = wl_display_get_registry(wl_display);

	ERR_FAIL_NULL_V_MSG(wl_registry, ERR_UNAVAILABLE, "Can't obtain the Wayland registry global.");

	registry.wayland_thread = this;

	wl_registry_add_listener(wl_registry, &wl_registry_listener, &registry);

	// Wait for registry to get notified from the compositor.
	wl_display_roundtrip(wl_display);

	ERR_FAIL_NULL_V_MSG(registry.wl_shm, ERR_UNAVAILABLE, "Can't obtain the Wayland shared memory global.");
	ERR_FAIL_NULL_V_MSG(registry.wl_compositor, ERR_UNAVAILABLE, "Can't obtain the Wayland compositor global.");
	ERR_FAIL_NULL_V_MSG(registry.xdg_wm_base, ERR_UNAVAILABLE, "Can't obtain the Wayland XDG shell global.");

	if (!registry.xdg_decoration_manager) {
#ifdef LIBDECOR_ENABLED
		WARN_PRINT("Can't obtain the XDG decoration manager. Libdecor will be used for drawing CSDs, if available.");
#else
		WARN_PRINT("Can't obtain the XDG decoration manager. Decorations won't show up.");
#endif // LIBDECOR_ENABLED
	}

	if (!registry.xdg_activation) {
		WARN_PRINT("Can't obtain the XDG activation global. Attention requesting won't work!");
	}

#ifndef DBUS_ENABLED
	if (!registry.wp_idle_inhibit_manager) {
		WARN_PRINT("Can't obtain the idle inhibition manager. The screen might turn off even after calling screen_set_keep_on()!");
	}
#endif // DBUS_ENABLED

	// Wait for seat capabilities.
	wl_display_roundtrip(wl_display);

#ifdef LIBDECOR_ENABLED
	bool libdecor_found = true;

#ifdef SOWRAP_ENABLED
	if (initialize_libdecor(dylibloader_verbose) != 0) {
		libdecor_found = false;
	}
#endif // SOWRAP_ENABLED

	if (libdecor_found) {
		libdecor_context = libdecor_new(wl_display, (struct libdecor_interface *)&libdecor_interface);
	} else {
		print_verbose("libdecor not found. Client-side decorations disabled.");
	}
#endif // LIBDECOR_ENABLED

	cursor_theme_name = OS::get_singleton()->get_environment("XCURSOR_THEME");

	unscaled_cursor_size = OS::get_singleton()->get_environment("XCURSOR_SIZE").to_int();
	if (unscaled_cursor_size <= 0) {
		print_verbose("Detected invalid cursor size preference, defaulting to 24.");
		unscaled_cursor_size = 24;
	}

	// NOTE: The scale is useful here as it might've been updated by _update_scale.
	bool cursor_theme_loaded = _load_cursor_theme(unscaled_cursor_size * cursor_scale);

	if (!cursor_theme_loaded) {
		return ERR_CANT_CREATE;
	}

	// Update the cursor.
	cursor_set_shape(DisplayServer::CURSOR_ARROW);

	initialized = true;
	return OK;
}

void WaylandThread::cursor_set_visible(bool p_visible) {
	cursor_visible = p_visible;

	for (struct wl_seat *wl_seat : registry.wl_seats) {
		SeatState *ss = wl_seat_get_seat_state(wl_seat);
		ERR_FAIL_NULL(ss);

		seat_state_update_cursor(ss);
	}
}

void WaylandThread::cursor_set_shape(DisplayServer::CursorShape p_cursor_shape) {
	cursor_shape = p_cursor_shape;

	for (struct wl_seat *wl_seat : registry.wl_seats) {
		SeatState *ss = wl_seat_get_seat_state(wl_seat);
		ERR_FAIL_NULL(ss);

		seat_state_update_cursor(ss);
	}
}

void WaylandThread::cursor_shape_set_custom_image(DisplayServer::CursorShape p_cursor_shape, Ref<Image> p_image, const Point2i &p_hotspot) {
	ERR_FAIL_COND(!p_image.is_valid());

	Size2i image_size = p_image->get_size();

	// NOTE: The stride is the width of the image in bytes.
	unsigned int image_stride = image_size.width * 4;
	unsigned int data_size = image_stride * image_size.height;

	// We need a shared memory object file descriptor in order to create a
	// wl_buffer through wl_shm.
	int fd = WaylandThread::_allocate_shm_file(data_size);
	ERR_FAIL_COND(fd == -1);

	CustomCursor &cursor = custom_cursors[p_cursor_shape];
	cursor.hotspot = p_hotspot;

	if (cursor.wl_buffer) {
		// Clean up the old Wayland buffer.
		wl_buffer_destroy(cursor.wl_buffer);
	}

	if (cursor.buffer_data) {
		// Clean up the old buffer data.
		munmap(cursor.buffer_data, cursor.buffer_data_size);
	}

	cursor.buffer_data = (uint32_t *)mmap(nullptr, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	cursor.buffer_data_size = data_size;

	// Create the Wayland buffer.
	struct wl_shm_pool *wl_shm_pool = wl_shm_create_pool(registry.wl_shm, fd, data_size);
	// TODO: Make sure that WL_SHM_FORMAT_ARGB8888 format is supported. It
	// technically isn't garaunteed to be supported, but I think that'd be a
	// pretty unlikely thing to stumble upon.
	cursor.wl_buffer = wl_shm_pool_create_buffer(wl_shm_pool, 0, image_size.width, image_size.height, image_stride, WL_SHM_FORMAT_ARGB8888);
	wl_shm_pool_destroy(wl_shm_pool);

	// Fill the cursor buffer with the image data.
	for (unsigned int index = 0; index < (unsigned int)(image_size.width * image_size.height); index++) {
		int row_index = floor(index / image_size.width);
		int column_index = (index % int(image_size.width));

		cursor.buffer_data[index] = p_image->get_pixel(column_index, row_index).to_argb32();

		// Wayland buffers, unless specified, require associated alpha, so we'll just
		// associate the alpha in-place.
		uint8_t *pixel_data = (uint8_t *)&cursor.buffer_data[index];
		pixel_data[0] = pixel_data[0] * pixel_data[3] / 255;
		pixel_data[1] = pixel_data[1] * pixel_data[3] / 255;
		pixel_data[2] = pixel_data[2] * pixel_data[3] / 255;
	}
}

void WaylandThread::cursor_shape_clear_custom_image(DisplayServer::CursorShape p_cursor_shape) {
	if (custom_cursors.has(p_cursor_shape)) {
		CustomCursor cursor = custom_cursors[p_cursor_shape];
		custom_cursors.erase(p_cursor_shape);

		if (cursor.wl_buffer) {
			wl_buffer_destroy(cursor.wl_buffer);
		}

		if (cursor.buffer_data) {
			munmap(cursor.buffer_data, cursor.buffer_data_size);
		}
	}
}

void WaylandThread::window_set_ime_active(const bool p_active, DisplayServer::WindowID p_window_id) {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss && ss->wp_text_input && ss->ime_enabled) {
		if (p_active) {
			ss->ime_active = true;
			zwp_text_input_v3_enable(ss->wp_text_input);
			zwp_text_input_v3_set_cursor_rectangle(ss->wp_text_input, ss->ime_rect.position.x, ss->ime_rect.position.y, ss->ime_rect.size.x, ss->ime_rect.size.y);
		} else {
			ss->ime_active = false;
			ss->ime_text = String();
			ss->ime_text_commit = String();
			ss->ime_cursor = Vector2i();
			zwp_text_input_v3_disable(ss->wp_text_input);
		}
		zwp_text_input_v3_commit(ss->wp_text_input);
	}
}

void WaylandThread::window_set_ime_position(const Point2i &p_pos, DisplayServer::WindowID p_window_id) {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss && ss->wp_text_input && ss->ime_enabled) {
		ss->ime_rect = Rect2i(p_pos, Size2i(1, 10));
		zwp_text_input_v3_set_cursor_rectangle(ss->wp_text_input, ss->ime_rect.position.x, ss->ime_rect.position.y, ss->ime_rect.size.x, ss->ime_rect.size.y);
		zwp_text_input_v3_commit(ss->wp_text_input);
	}
}

int WaylandThread::keyboard_get_layout_count() const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss && ss->xkb_keymap) {
		return xkb_keymap_num_layouts(ss->xkb_keymap);
	}

	return 0;
}

int WaylandThread::keyboard_get_current_layout_index() const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss) {
		return ss->current_layout_index;
	}

	return 0;
}

void WaylandThread::keyboard_set_current_layout_index(int p_index) {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss) {
		ss->current_layout_index = p_index;
	}
}

String WaylandThread::keyboard_get_layout_name(int p_index) const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss && ss->xkb_keymap) {
		String ret;
		ret.parse_utf8(xkb_keymap_layout_get_name(ss->xkb_keymap, p_index));

		return ret;
	}

	return "";
}

Key WaylandThread::keyboard_get_key_from_physical(Key p_key) const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss && ss->xkb_state) {
		xkb_keycode_t xkb_keycode = KeyMappingXKB::get_xkb_keycode(p_key);
		return KeyMappingXKB::get_keycode(xkb_state_key_get_one_sym(ss->xkb_state, xkb_keycode));
	}

	return Key::NONE;
}

void WaylandThread::keyboard_echo_keys() {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss) {
		seat_state_echo_keys(ss);
	}
}

void WaylandThread::selection_set_text(const String &p_text) {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (registry.wl_data_device_manager == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't set selection, wl_data_device_manager global not available.");
	}

	if (ss == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't set selection, current seat not set.");
		return;
	}

	if (ss->wl_data_device == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't set selection, seat doesn't have wl_data_device.");
	}

	ss->selection_data = p_text.to_utf8_buffer();

	if (ss->wl_data_source_selection == nullptr) {
		ss->wl_data_source_selection = wl_data_device_manager_create_data_source(registry.wl_data_device_manager);
		wl_data_source_add_listener(ss->wl_data_source_selection, &wl_data_source_listener, ss);
		wl_data_source_offer(ss->wl_data_source_selection, "text/plain;charset=utf-8");
		wl_data_source_offer(ss->wl_data_source_selection, "text/plain");

		// TODO: Implement a good way of getting the latest serial from the user.
		wl_data_device_set_selection(ss->wl_data_device, ss->wl_data_source_selection, MAX(ss->pointer_data.button_serial, ss->last_key_pressed_serial));
	}

	// Wait for the message to get to the server before continuing, otherwise the
	// clipboard update might come with a delay.
	wl_display_roundtrip(wl_display);
}

bool WaylandThread::selection_has_mime(const String &p_mime) const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't get selection, current seat not set.");
		return false;
	}

	OfferState *os = wl_data_offer_get_offer_state(ss->wl_data_offer_selection);
	if (!os) {
		return false;
	}

	return os->mime_types.has(p_mime);
}

Vector<uint8_t> WaylandThread::selection_get_mime(const String &p_mime) const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);
	if (ss == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't get selection, current seat not set.");
		return Vector<uint8_t>();
	}

	if (ss->wl_data_source_selection) {
		// We have a source so the stuff we're pasting is ours. We'll have to pass the
		// data directly or we'd stall waiting for Godot (ourselves) to send us the
		// data :P

		OfferState *os = wl_data_offer_get_offer_state(ss->wl_data_offer_selection);
		ERR_FAIL_NULL_V(os, Vector<uint8_t>());

		if (os->mime_types.has(p_mime)) {
			// All righty, we're offering this type. Let's just return the data as is.
			return ss->selection_data;
		}

		// ... we don't offer that type. Oh well.
		return Vector<uint8_t>();
	}

	return _wl_data_offer_read(wl_display, p_mime.utf8(), ss->wl_data_offer_selection);
}

bool WaylandThread::primary_has_mime(const String &p_mime) const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (ss == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't get selection, current seat not set.");
		return false;
	}

	OfferState *os = wp_primary_selection_offer_get_offer_state(ss->wp_primary_selection_offer);
	if (!os) {
		return false;
	}

	return os->mime_types.has(p_mime);
}

Vector<uint8_t> WaylandThread::primary_get_mime(const String &p_mime) const {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);
	if (ss == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't get primary, current seat not set.");
		return Vector<uint8_t>();
	}

	if (ss->wp_primary_selection_source) {
		// We have a source so the stuff we're pasting is ours. We'll have to pass the
		// data directly or we'd stall waiting for Godot (ourselves) to send us the
		// data :P

		OfferState *os = wp_primary_selection_offer_get_offer_state(ss->wp_primary_selection_offer);
		ERR_FAIL_NULL_V(os, Vector<uint8_t>());

		if (os->mime_types.has(p_mime)) {
			// All righty, we're offering this type. Let's just return the data as is.
			return ss->selection_data;
		}

		// ... we don't offer that type. Oh well.
		return Vector<uint8_t>();
	}

	return _wp_primary_selection_offer_read(wl_display, p_mime.utf8(), ss->wp_primary_selection_offer);
}

void WaylandThread::primary_set_text(const String &p_text) {
	SeatState *ss = wl_seat_get_seat_state(wl_seat_current);

	if (registry.wp_primary_selection_device_manager == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't set primary, protocol not available");
		return;
	}

	if (ss == nullptr) {
		DEBUG_LOG_WAYLAND_THREAD("Couldn't set primary, current seat not set.");
		return;
	}

	ss->primary_data = p_text.to_utf8_buffer();

	if (ss->wp_primary_selection_source == nullptr) {
		ss->wp_primary_selection_source = zwp_primary_selection_device_manager_v1_create_source(registry.wp_primary_selection_device_manager);
		zwp_primary_selection_source_v1_add_listener(ss->wp_primary_selection_source, &wp_primary_selection_source_listener, ss);
		zwp_primary_selection_source_v1_offer(ss->wp_primary_selection_source, "text/plain;charset=utf-8");
		zwp_primary_selection_source_v1_offer(ss->wp_primary_selection_source, "text/plain");
	}

	// TODO: Implement a good way of getting the latest serial from the user.
	zwp_primary_selection_device_v1_set_selection(ss->wp_primary_selection_device, ss->wp_primary_selection_source, MAX(ss->pointer_data.button_serial, ss->last_key_pressed_serial));

	// Wait for the message to get to the server before continuing, otherwise the
	// clipboard update might come with a delay.
	wl_display_roundtrip(wl_display);
}

void WaylandThread::commit_surfaces() {
	wl_surface_commit(main_window.wl_surface);
}

void WaylandThread::set_frame() {
	frame = true;
}

bool WaylandThread::get_reset_frame() {
	bool old_frame = frame;
	frame = false;

	return old_frame;
}

// Dispatches events until a frame event is received, a window is reported as
// suspended or the timeout expires.
bool WaylandThread::wait_frame_suspend_ms(int p_timeout) {
	if (main_window.suspended) {
		// The window is suspended! The compositor is telling us _explicitly_ that we
		// don't need to draw, without letting us guess through the frame event's
		// timing and stuff like that. Our job here is done.
		return false;
	}

	if (frame) {
		// We already have a frame! Probably it got there while the caller locked :D
		frame = false;
		return true;
	}

	struct pollfd poll_fd;
	poll_fd.fd = wl_display_get_fd(wl_display);
	poll_fd.events = POLLIN | POLLHUP;

	int begin_ms = OS::get_singleton()->get_ticks_msec();
	int remaining_ms = p_timeout;

	while (remaining_ms > 0) {
		// Empty the event queue while it's full.
		while (wl_display_prepare_read(wl_display) != 0) {
			if (wl_display_dispatch_pending(wl_display) == -1) {
				// Oh no. We'll check and handle any display error below.
				break;
			}

			if (main_window.suspended) {
				return false;
			}

			if (frame) {
				// We had a frame event in the queue :D
				frame = false;
				return true;
			}
		}

		int werror = wl_display_get_error(wl_display);

		if (werror) {
			if (werror == EPROTO) {
				struct wl_interface *wl_interface = nullptr;
				uint32_t id = 0;

				int error_code = wl_display_get_protocol_error(wl_display, (const struct wl_interface **)&wl_interface, &id);
				CRASH_NOW_MSG(vformat("Wayland protocol error %d on interface %s@%d.", error_code, wl_interface ? wl_interface->name : "unknown", id));
			} else {
				CRASH_NOW_MSG(vformat("Wayland client error code %d.", werror));
			}
		}

		wl_display_flush(wl_display);

		// Wait for the event file descriptor to have new data.
		poll(&poll_fd, 1, remaining_ms);

		if (poll_fd.revents | POLLIN) {
			// Load the queues with fresh new data.
			wl_display_read_events(wl_display);
		} else {
			// Oh well... Stop signaling that we want to read.
			wl_display_cancel_read(wl_display);

			// We've got no new events :(
			// We won't even bother with checking the frame flag.
			return false;
		}

		// Let's try dispatching now...
		wl_display_dispatch_pending(wl_display);

		if (main_window.suspended) {
			return false;
		}

		if (frame) {
			frame = false;
			return true;
		}

		remaining_ms -= OS::get_singleton()->get_ticks_msec() - begin_ms;
	}

	DEBUG_LOG_WAYLAND_THREAD("Frame timeout.");
	return false;
}

bool WaylandThread::is_suspended() const {
	return main_window.suspended;
}

void WaylandThread::destroy() {
	if (!initialized) {
		return;
	}

	if (wl_display && events_thread.is_started()) {
		thread_data.thread_done.set();

		// By sending a roundtrip message we're unblocking the polling thread so that
		// it can realize that it's done and also handle every event that's left.
		wl_display_roundtrip(wl_display);

		events_thread.wait_to_finish();
	}

	if (main_window.wp_fractional_scale) {
		wp_fractional_scale_v1_destroy(main_window.wp_fractional_scale);
	}

	if (main_window.wp_viewport) {
		wp_viewport_destroy(main_window.wp_viewport);
	}

	if (main_window.frame_callback) {
		wl_callback_destroy(main_window.frame_callback);
	}

#ifdef LIBDECOR_ENABLED
	if (main_window.libdecor_frame) {
		libdecor_frame_close(main_window.libdecor_frame);
	}
#endif // LIBDECOR_ENABLED

	if (main_window.xdg_toplevel) {
		xdg_toplevel_destroy(main_window.xdg_toplevel);
	}

	if (main_window.xdg_surface) {
		xdg_surface_destroy(main_window.xdg_surface);
	}

	if (main_window.wl_surface) {
		wl_surface_destroy(main_window.wl_surface);
	}

	for (struct wl_seat *wl_seat : registry.wl_seats) {
		SeatState *ss = wl_seat_get_seat_state(wl_seat);
		ERR_FAIL_NULL(ss);

		wl_seat_destroy(wl_seat);

		xkb_context_unref(ss->xkb_context);
		xkb_state_unref(ss->xkb_state);
		xkb_keymap_unref(ss->xkb_keymap);

		if (ss->wl_keyboard) {
			wl_keyboard_destroy(ss->wl_keyboard);
		}

		if (ss->keymap_buffer) {
			munmap((void *)ss->keymap_buffer, ss->keymap_buffer_size);
		}

		if (ss->wl_pointer) {
			wl_pointer_destroy(ss->wl_pointer);
		}

		if (ss->cursor_frame_callback) {
			// We don't need to set a null userdata for safety as the thread is done.
			wl_callback_destroy(ss->cursor_frame_callback);
		}

		if (ss->cursor_surface) {
			wl_surface_destroy(ss->cursor_surface);
		}

		if (ss->wl_data_device) {
			wl_data_device_destroy(ss->wl_data_device);
		}

		if (ss->wp_relative_pointer) {
			zwp_relative_pointer_v1_destroy(ss->wp_relative_pointer);
		}

		if (ss->wp_locked_pointer) {
			zwp_locked_pointer_v1_destroy(ss->wp_locked_pointer);
		}

		if (ss->wp_confined_pointer) {
			zwp_confined_pointer_v1_destroy(ss->wp_confined_pointer);
		}

		if (ss->wp_tablet_seat) {
			zwp_tablet_seat_v2_destroy(ss->wp_tablet_seat);
		}

		for (struct zwp_tablet_tool_v2 *tool : ss->tablet_tools) {
			TabletToolState *state = wp_tablet_tool_get_state(tool);
			if (state) {
				memdelete(state);
			}

			zwp_tablet_tool_v2_destroy(tool);
		}

		memdelete(ss);
	}

	for (struct wl_output *wl_output : registry.wl_outputs) {
		ERR_FAIL_NULL(wl_output);

		memdelete(wl_output_get_screen_state(wl_output));
		wl_output_destroy(wl_output);
	}

	if (wl_cursor_theme) {
		wl_cursor_theme_destroy(wl_cursor_theme);
	}

	if (registry.wp_idle_inhibit_manager) {
		zwp_idle_inhibit_manager_v1_destroy(registry.wp_idle_inhibit_manager);
	}

	if (registry.wp_pointer_constraints) {
		zwp_pointer_constraints_v1_destroy(registry.wp_pointer_constraints);
	}

	if (registry.wp_pointer_gestures) {
		zwp_pointer_gestures_v1_destroy(registry.wp_pointer_gestures);
	}

	if (registry.wp_relative_pointer_manager) {
		zwp_relative_pointer_manager_v1_destroy(registry.wp_relative_pointer_manager);
	}

	if (registry.xdg_activation) {
		xdg_activation_v1_destroy(registry.xdg_activation);
	}

	if (registry.xdg_decoration_manager) {
		zxdg_decoration_manager_v1_destroy(registry.xdg_decoration_manager);
	}

	if (registry.wp_fractional_scale_manager) {
		wp_fractional_scale_manager_v1_destroy(registry.wp_fractional_scale_manager);
	}

	if (registry.wp_viewporter) {
		wp_viewporter_destroy(registry.wp_viewporter);
	}

	if (registry.xdg_wm_base) {
		xdg_wm_base_destroy(registry.xdg_wm_base);
	}

	if (registry.xdg_exporter) {
		zxdg_exporter_v1_destroy(registry.xdg_exporter);
	}

	if (registry.wl_shm) {
		wl_shm_destroy(registry.wl_shm);
	}

	if (registry.wl_compositor) {
		wl_compositor_destroy(registry.wl_compositor);
	}

	if (wl_registry) {
		wl_registry_destroy(wl_registry);
	}

	if (wl_display) {
		wl_display_disconnect(wl_display);
	}
}

#endif // WAYLAND_ENABLED
