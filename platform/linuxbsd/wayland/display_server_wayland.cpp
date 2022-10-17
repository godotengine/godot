/*************************************************************************/
/*  display_server_wayland.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "display_server_wayland.h"

#ifdef WAYLAND_ENABLED

#ifdef VULKAN_ENABLED
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#endif

#if defined(GLES3_ENABLED)
#include "drivers/gles3/rasterizer_gles3.h"
#endif

#define DISPLAY_SERVER_WAYLAND_DEBUG_LOGS_ENABLED
#ifdef DISPLAY_SERVER_WAYLAND_DEBUG_LOGS_ENABLED
#define DEBUG_LOG_WAYLAND(...) print_verbose(__VA_ARGS__)
#else
#define DEBUG_LOG_WAYLAND(...)
#endif

// Implementation specific methods.

void DisplayServerWayland::_poll_events_thread(void *p_wls) {
	WaylandState *wls = (WaylandState *)p_wls;

	struct pollfd poll_fd;
	poll_fd.fd = wl_display_get_fd(wls->wl_display);
	poll_fd.events = POLLIN | POLLHUP;

	while (true) {
		// Empty the event queue while it's full.
		while (wl_display_prepare_read(wls->wl_display) != 0) {
			MutexLock mutex_lock(wls->mutex);
			wl_display_dispatch_pending(wls->wl_display);
		}

		if (wl_display_flush(wls->wl_display) == -1) {
			if (errno != EAGAIN) {
				print_error(vformat("Error %d while flushing the Wayland display.", errno));
				wls->events_thread_done.set();
			}
		}

		// Wait for the event file descriptor to have new data.
		poll(&poll_fd, 1, -1);

		if (wls->events_thread_done.is_set()) {
			wl_display_cancel_read(wls->wl_display);
			break;
		}

		if (poll_fd.revents | POLLIN) {
			wl_display_read_events(wls->wl_display);
		} else {
			wl_display_cancel_read(wls->wl_display);
		}
	}
}

String DisplayServerWayland::_string_read_fd(int fd) {
	// This is pretty much an arbitrary size.
	uint32_t chunk_size = 2048;

	LocalVector<uint8_t> data;
	data.resize(chunk_size);

	uint32_t bytes_read = 0;

	while (true) {
		ssize_t last_bytes_read = read(fd, data.ptr() + bytes_read, chunk_size);
		if (last_bytes_read < 0) {
			ERR_PRINT(vformat("Read error %d.", errno));
			return "";
		}

		if (last_bytes_read == 0) {
			// We're done, we've reached the EOF.
			DEBUG_LOG_WAYLAND(vformat("Done reading %d bytes.", bytes_read));
			close(fd);
			break;
		}

		DEBUG_LOG_WAYLAND(vformat("Read chunk of %d bytes.", last_bytes_read));

		bytes_read += last_bytes_read;

		// Increase the buffer size by one chunk in preparation of the next read.
		data.resize(bytes_read + chunk_size);
	}

	String ret;
	ret.parse_utf8((const char *)data.ptr(), bytes_read);
	return ret;
}

String DisplayServerWayland::_wl_data_offer_read(wl_data_offer *wl_data_offer) const {
	if (!wl_data_offer) {
		return "";
	}

	int fds[2];
	if (pipe(fds) == 0) {
		// This function expects to return a string, so we can only ask for a MIME of
		// "text/plain"
		wl_data_offer_receive(wl_data_offer, "text/plain", fds[1]);

		// Wait for the compositor to know about the pipe.
		wl_display_roundtrip(wls.wl_display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		return _string_read_fd(fds[0]);
	}

	return "";
}

String DisplayServerWayland::_wp_primary_selection_offer_read(zwp_primary_selection_offer_v1 *wp_primary_selection_offer) const {
	if (!wp_primary_selection_offer) {
		return "";
	}

	int fds[2];
	if (pipe(fds) == 0) {
		// This function expects to return a string, so we can only ask for a MIME of
		// "text/plain"
		zwp_primary_selection_offer_v1_receive(wp_primary_selection_offer, "text/plain", fds[1]);

		// Wait for the compositor to know about the pipe.
		wl_display_roundtrip(wls.wl_display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		return _string_read_fd(fds[0]);
	}

	return "";
}

// Sets a given seat state as its own wayland state's current seat and makes
// sure that any old seat gets reset.
void DisplayServerWayland::_seat_state_set_current(SeatState &p_ss) {
	WaylandState *wls = p_ss.wls;
	ERR_FAIL_NULL(wls);

	if (wls->current_seat) {
		// There was an older seat there, clean up its state
		if (wls->current_seat == &p_ss) {
			return;
		}

		// Unlock the pointer if it's locked.
		if (wls->current_seat->wp_locked_pointer) {
			zwp_locked_pointer_v1_destroy(wls->current_seat->wp_locked_pointer);
			wls->current_seat->wp_locked_pointer = nullptr;
		}

		// Unconfine the pointer if it's confined.
		if (wls->current_seat->wp_confined_pointer) {
			zwp_confined_pointer_v1_destroy(wls->current_seat->wp_confined_pointer);
			wls->current_seat->wp_confined_pointer = nullptr;
		}

		_seat_state_override_cursor_shape(*wls->current_seat, CURSOR_ARROW);
	}

	wls->current_seat = &p_ss;

	_wayland_state_update_cursor(*wls);
}

void DisplayServerWayland::_seat_state_override_cursor_shape(SeatState &p_ss, CursorShape p_shape) {
	if (!p_ss.wl_pointer) {
		return;
	}

	ERR_FAIL_NULL(p_ss.wls);

	struct wl_buffer *cursor_buffer = nullptr;
	Point2i hotspot;

	if (!p_ss.wls->custom_cursors.has(p_shape)) {
		// Select the default cursor data.
		struct wl_cursor_image *cursor_image = p_ss.wls->cursor_images[p_shape];

		if (!cursor_image) {
			// TODO: Error out?
			return;
		}

		cursor_buffer = p_ss.wls->cursor_bufs[p_shape];
		hotspot.x = cursor_image->hotspot_x;
		hotspot.y = cursor_image->hotspot_y;
	} else {
		// Select the custom cursor data.
		CustomWaylandCursor &custom_cursor = p_ss.wls->custom_cursors[p_shape];

		cursor_buffer = custom_cursor.wl_buffer;
		hotspot = custom_cursor.hotspot;
	}

	// Update the cursor's hotspot.
	wl_pointer_set_cursor(p_ss.wl_pointer, 0, p_ss.cursor_surface, hotspot.x, hotspot.y);

	// Attach the new cursor's buffer and damage it.
	wl_surface_attach(p_ss.cursor_surface, cursor_buffer, 0, 0);
	wl_surface_damage_buffer(p_ss.cursor_surface, 0, 0, INT_MAX, INT_MAX);

	// Commit everything.
	wl_surface_commit(p_ss.cursor_surface);
}

void DisplayServerWayland::_wayland_state_update_cursor(WaylandState &p_wls) {
	if (!p_wls.current_seat || !p_wls.current_seat->wl_pointer) {
		return;
	}

	SeatState &ss = *p_wls.current_seat;

	ERR_FAIL_NULL(ss.cursor_surface);

	struct wl_pointer *wp = ss.wl_pointer;
	struct zwp_pointer_constraints_v1 *pc = p_wls.globals.wp_pointer_constraints;

	// We want to change the location of these pointers so we get their reference.
	struct zwp_locked_pointer_v1 *&lp = ss.wp_locked_pointer;
	struct zwp_confined_pointer_v1 *&cp = ss.wp_confined_pointer;

	// All modes but `MOUSE_MODE_VISIBLE` and `MOUSE_MODE_CONFINED` are hidden.
	if (p_wls.mouse_mode != MOUSE_MODE_VISIBLE && p_wls.mouse_mode != MOUSE_MODE_CONFINED) {
		// Reset the cursor's hotspot.
		wl_pointer_set_cursor(ss.wl_pointer, 0, ss.cursor_surface, 0, 0);

		// Unmap the cursor.
		wl_surface_attach(ss.cursor_surface, nullptr, 0, 0);

		wl_surface_commit(ss.cursor_surface);
	} else {
		// Update the cursor shape.
		_seat_state_override_cursor_shape(ss, p_wls.cursor_shape);
	}

	// Constrain/Free pointer movement depending on its mode.
	switch (p_wls.mouse_mode) {
		// Unconstrained pointer.
		case MOUSE_MODE_VISIBLE:
		case MOUSE_MODE_HIDDEN: {
			if (lp) {
				zwp_locked_pointer_v1_destroy(lp);
				lp = nullptr;
			}

			if (cp) {
				zwp_confined_pointer_v1_destroy(cp);
				cp = nullptr;
			}
		} break;

		// Locked pointer.
		case MOUSE_MODE_CAPTURED: {
			if (!lp) {
				WindowData &wd = p_wls.windows[MAIN_WINDOW_ID];
				lp = zwp_pointer_constraints_v1_lock_pointer(pc, wd.wl_surface, wp, nullptr, ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);

				// Center the cursor on unlock.
				wl_fixed_t unlock_x = wl_fixed_from_int(wd.rect.size.width / 2);
				wl_fixed_t unlock_y = wl_fixed_from_int(wd.rect.size.height / 2);

				zwp_locked_pointer_v1_set_cursor_position_hint(lp, unlock_x, unlock_y);
			}
		} break;

		// Confined pointer.
		case MOUSE_MODE_CONFINED:
		case MOUSE_MODE_CONFINED_HIDDEN: {
			if (!cp) {
				WindowData &wd = p_wls.windows[MAIN_WINDOW_ID];
				cp = zwp_pointer_constraints_v1_confine_pointer(pc, wd.wl_surface, wp, nullptr, ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
			}
		}
	}
}

// Get the "global" position of a point in the window's local coordinate space.
Point2i DisplayServerWayland::_wayland_state_point_window_to_global(const WaylandState &wls, WindowID p_window, Point2i p_position) {
	while (p_window != INVALID_WINDOW_ID) {
		if (!wls.windows.has(p_window)) {
			break;
		}

		const WindowData &wd = wls.windows[p_window];

		p_position += wd.rect.position;
		p_window = wd.parent;
	}

	return p_position;
}

void DisplayServerWayland::dispatch_input_events(const Ref<InputEvent> &p_event) {
	((DisplayServerWayland *)(get_singleton()))->_dispatch_input_event(p_event);
}

void DisplayServerWayland::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	Variant ev = p_event;
	Variant *evp = &ev;
	Variant ret;
	Callable::CallError ce;

	// Close the topmost popup menu if the user clicks outside of it or its safe
	// rect.
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && wls.popup_list.size() > 0) {
		WindowID &front_id = wls.popup_list.front()->get();

		ERR_FAIL_COND(!wls.windows.has(front_id));
		WindowData &wd = wls.windows[front_id];

		if (!wd.rect.has_point(mb->get_global_position()) && !wd.safe_rect.has_point(mb->get_global_position())) {
			_send_window_event(front_id, WINDOW_EVENT_CLOSE_REQUEST);
			return;
		}
	}

	Ref<InputEventFromWindow> event_from_window = p_event;

	if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
		// Send to a single window.
		if (wls.windows.has(event_from_window->get_window_id())) {
			Callable callable = wls.windows[event_from_window->get_window_id()].input_event_callback;
			if (callable.is_valid()) {
				callable.callp((const Variant **)&evp, 1, ret, ce);
			}
		}
	} else {
		// Send to all windows.
		for (KeyValue<WindowID, WindowData> &E : wls.windows) {
			Callable callable = E.value.input_event_callback;
			if (callable.is_valid()) {
				callable.callp((const Variant **)&evp, 1, ret, ce);
			}
		}
	}
}

void DisplayServerWayland::_get_key_modifier_state(SeatState &p_seat, Ref<InputEventWithModifiers> p_event) {
	p_event->set_shift_pressed(p_seat.shift_pressed);
	p_event->set_ctrl_pressed(p_seat.ctrl_pressed);
	p_event->set_alt_pressed(p_seat.alt_pressed);
	p_event->set_meta_pressed(p_seat.meta_pressed);
}

// Sets up an `InputEventKey` and returns whether it has any meaningful value.
bool DisplayServerWayland::_seat_state_configure_key_event(SeatState &p_ss, Ref<InputEventKey> p_event, xkb_keycode_t p_keycode, bool p_pressed) {
	// TODO: Handle keys that release multiple symbols?
	Key keycode = KeyMappingXKB::get_keycode(xkb_state_key_get_one_sym(p_ss.xkb_state, p_keycode));
	Key physical_keycode = KeyMappingXKB::get_scancode(p_keycode);

	if (physical_keycode == Key::NONE) {
		return false;
	}

	if (keycode == Key::NONE) {
		keycode = physical_keycode;
	}

	if (keycode >= Key::A + 32 && keycode <= Key::Z + 32) {
		keycode -= 'a' - 'A';
	}

	if (p_ss.keyboard_focused_window_id != INVALID_WINDOW_ID) {
		p_event->set_window_id(p_ss.keyboard_focused_window_id);
	}

	// Set all pressed modifiers.
	_get_key_modifier_state(p_ss, p_event);

	p_event->set_keycode(keycode);
	p_event->set_physical_keycode(physical_keycode);
	p_event->set_unicode(xkb_state_key_get_utf32(p_ss.xkb_state, p_keycode));
	p_event->set_pressed(p_pressed);

	// Taken from DisplayServerX11.
	if (p_event->get_keycode() == Key::BACKTAB) {
		// Make it consistent across platforms.
		p_event->set_keycode(Key::TAB);
		p_event->set_physical_keycode(Key::TAB);
		p_event->set_shift_pressed(true);
	}

	return true;
}

DisplayServer::WindowID DisplayServerWayland::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	MutexLock mutex_lock(wls.mutex);

	WindowID id = wls.window_id_counter++;

	WindowData &wd = wls.windows[id];

	wd.id = id;
	wd.wls = &wls;

	wd.mode = p_mode;
	wd.flags = p_flags;
	wd.vsync_mode = p_vsync_mode;
	wd.rect = p_rect;

	wd.title = "Godot";

	return id;
}

// Method which forces the modesetting logic without any fancy no-op case
// useful in internal Wayland window handling.
void DisplayServerWayland::_window_data_set_mode(WindowData &p_wd, WindowMode p_mode) {
	if (!p_wd.wl_surface || !p_wd.xdg_toplevel) {
		// TODO: Ask whether this is the right behaviour.

		// Don't waste time with popups and whatnot. Behave like it worked.
		p_wd.mode = p_mode;
		return;
	}

	// Return back to a windowed state so that we can apply what the user asked.
	switch (p_wd.mode) {
		case WINDOW_MODE_WINDOWED: {
			// Do nothing.
		} break;

		case WINDOW_MODE_MINIMIZED: {
			// We can't do much according to the xdg_shell protocol. I have no idea
			// whether this implies that we should return or who knows what. For now
			// we'll do nothing.
			// TODO: Test this properly.
		} break;

		case WINDOW_MODE_MAXIMIZED: {
			// Try to unmaximize. This isn't garaunteed to work actually, so we'll have
			// to check whether something changed.
			xdg_toplevel_unset_maximized(p_wd.xdg_toplevel);
		} break;

		case WINDOW_MODE_FULLSCREEN:
		case WINDOW_MODE_EXCLUSIVE_FULLSCREEN: {
			// Same thing as above, unset fullscreen and check later if it worked.
			xdg_toplevel_unset_fullscreen(p_wd.xdg_toplevel);
		} break;
	}

	// Wait for a configure event and hope that something changed.
	wl_display_roundtrip(wls.wl_display);

	if (p_wd.mode != WINDOW_MODE_WINDOWED) {
		// The compositor refused our "normalization" request. It'd be useless or
		// unpredictable to attempt setting a new state. We're done.
		return;
	}

	// Ask the compositor to set the state indicated by the new mode.
	switch (p_mode) {
		case WINDOW_MODE_WINDOWED: {
			// Do nothing. We're already windowed.
		} break;

		case WINDOW_MODE_MINIMIZED: {
			xdg_toplevel_set_minimized(p_wd.xdg_toplevel);

			// We have no way to actually detect this state, so we'll have to report it
			// manually to the engine (hoping that it worked). In the worst case it'll
			// get reset by the next configure event.
			p_wd.mode = WINDOW_MODE_MINIMIZED;
		} break;

		case WINDOW_MODE_MAXIMIZED: {
			xdg_toplevel_set_maximized(p_wd.xdg_toplevel);
		} break;

		case WINDOW_MODE_FULLSCREEN:
		case WINDOW_MODE_EXCLUSIVE_FULLSCREEN: {
			xdg_toplevel_set_fullscreen(p_wd.xdg_toplevel, nullptr);
		}
	}
}

void DisplayServerWayland::_send_window_event(WindowID p_window, WindowEvent p_event) {
	ERR_FAIL_COND(!wls.windows.has(p_window));

	WindowData &wd = wls.windows[p_window];

	if (wd.window_event_callback.is_valid()) {
		Variant var_event = Variant(p_event);
		Variant *arg = &var_event;

		Variant ret;
		Callable::CallError ce;

		wd.window_event_callback.callp((const Variant **)&arg, 1, ret, ce);
	}
}

// Based on the wayland book's shared memory boilerplate (PD/CC0).
// See: https://wayland-book.com/surfaces/shared-memory.html
int DisplayServerWayland::_allocate_shm_file(size_t size) {
	int retries = 100;

	do {
		// Generate a random name.
		char name[] = "/wl_shm-godot-XXXXXX";
		for (long unsigned int i = sizeof(name) - 7; i < sizeof(name) - 1; i++) {
			name[i] = Math::random('A', 'Z');
		}

		// Try to open a shared memory object with that name.
		int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fd >= 0) {
			// Success, unlink its name as we just need the file descriptor.
			shm_unlink(name);

			// Resize the file to the requested length.
			int ret;
			do {
				ret = ftruncate(fd, size);
			} while (ret < 0 && errno == EINTR);

			if (ret < 0) {
				close(fd);
				return -1;
			}

			return fd;
		}

		retries--;
	} while (retries > 0 && errno == EEXIST);

	return -1;
}

void DisplayServerWayland::_wl_registry_on_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version) {
	WaylandState *wls = (WaylandState *)data;
	ERR_FAIL_NULL(wls);

	WaylandGlobals &globals = wls->globals;

	if (strcmp(interface, wl_shm_interface.name) == 0) {
		globals.wl_shm = (struct wl_shm *)wl_registry_bind(wl_registry, name, &wl_shm_interface, 1);
		globals.wl_shm_name = name;
		return;
	}

	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		globals.wl_compositor = (struct wl_compositor *)wl_registry_bind(wl_registry, name, &wl_compositor_interface, 4);
		globals.wl_compositor_name = name;
		return;
	}

	if (strcmp(interface, wl_subcompositor_interface.name) == 0) {
		globals.wl_subcompositor = (struct wl_subcompositor *)wl_registry_bind(wl_registry, name, &wl_subcompositor_interface, 1);
		globals.wl_subcompositor_name = name;
		return;
	}

	if (strcmp(interface, wl_data_device_manager_interface.name) == 0) {
		globals.wl_data_device_manager = (struct wl_data_device_manager *)wl_registry_bind(wl_registry, name, &wl_data_device_manager_interface, 3);
		globals.wl_data_device_manager_name = name;

		for (SeatState &ss : wls->seats) {
			if (!ss.wl_data_device) {
				ss.wl_data_device = wl_data_device_manager_get_data_device(wls->globals.wl_data_device_manager, ss.wl_seat);
				wl_data_device_add_listener(ss.wl_data_device, &wl_data_device_listener, &ss);
			}
		}
		return;
	}

	if (strcmp(interface, zwp_primary_selection_device_manager_v1_interface.name) == 0) {
		globals.wp_primary_selection_device_manager = (struct zwp_primary_selection_device_manager_v1 *)wl_registry_bind(wl_registry, name, &zwp_primary_selection_device_manager_v1_interface, 1);

		for (SeatState &ss : wls->seats) {
			if (!ss.wp_primary_selection_device) {
				ss.wp_primary_selection_device = zwp_primary_selection_device_manager_v1_get_device(wls->globals.wp_primary_selection_device_manager, ss.wl_seat);
				zwp_primary_selection_device_v1_add_listener(ss.wp_primary_selection_device, &wp_primary_selection_device_listener, &ss);
			}
		}
	}

	if (strcmp(interface, wl_output_interface.name) == 0) {
		// The screen listener requires a pointer for its state data. For this
		// reason, to get one that points to a variable that can live outside of this
		// scope, we push a default `ScreenData` in `wls->screens` and get the
		// address of this new element.
		ScreenData *sd = &wls->screens.push_back({})->get();
		sd->wl_output = (struct wl_output *)wl_registry_bind(wl_registry, name, &wl_output_interface, 2);
		sd->wl_output_name = name;

		wl_output_add_listener(sd->wl_output, &wl_output_listener, sd);
		return;
	}

	if (strcmp(interface, wl_seat_interface.name) == 0) {
		// The seat listener requires a pointer for its state data. For this reason,
		// to get one that points to a variable that can live outside of this scope,
		// we push a default `SeatState` in `wls->seats` and get the address of this
		// new element.
		SeatState *ss = &wls->seats.push_back({})->get();
		ss->wls = wls;
		ss->wl_seat = (struct wl_seat *)wl_registry_bind(wl_registry, name, &wl_seat_interface, 5);
		ss->wl_seat_name = name;

		if (!ss->wl_data_device && globals.wl_data_device_manager) {
			ss->wl_data_device = wl_data_device_manager_get_data_device(globals.wl_data_device_manager, ss->wl_seat);
			wl_data_device_add_listener(ss->wl_data_device, &wl_data_device_listener, ss);
		}

		if (!ss->wp_primary_selection_device && globals.wp_primary_selection_device_manager) {
			ss->wp_primary_selection_device = zwp_primary_selection_device_manager_v1_get_device(wls->globals.wp_primary_selection_device_manager, ss->wl_seat);
			zwp_primary_selection_device_v1_add_listener(ss->wp_primary_selection_device, &wp_primary_selection_device_listener, ss);
		}

		wl_seat_add_listener(ss->wl_seat, &wl_seat_listener, ss);
		return;
	}

	if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		globals.xdg_wm_base = (struct xdg_wm_base *)wl_registry_bind(wl_registry, name, &xdg_wm_base_interface, 2);
		globals.xdg_wm_base_name = name;
		return;
	}

	if (strcmp(interface, zxdg_decoration_manager_v1_interface.name) == 0) {
		globals.xdg_decoration_manager = (struct zxdg_decoration_manager_v1 *)wl_registry_bind(wl_registry, name, &zxdg_decoration_manager_v1_interface, 1);
		globals.xdg_decoration_manager_name = name;
		return;
	}

	if (strcmp(interface, xdg_activation_v1_interface.name) == 0) {
		globals.xdg_activation = (struct xdg_activation_v1 *)wl_registry_bind(wl_registry, name, &xdg_activation_v1_interface, 1);
		globals.xdg_activation_name = name;
		return;
	}

	if (strcmp(interface, zwp_pointer_constraints_v1_interface.name) == 0) {
		globals.wp_pointer_constraints = (struct zwp_pointer_constraints_v1 *)wl_registry_bind(wl_registry, name, &zwp_pointer_constraints_v1_interface, 1);
		globals.wp_pointer_constraints_name = name;
		return;
	}

	if (strcmp(interface, zwp_relative_pointer_manager_v1_interface.name) == 0) {
		globals.wp_relative_pointer_manager = (struct zwp_relative_pointer_manager_v1 *)wl_registry_bind(wl_registry, name, &zwp_relative_pointer_manager_v1_interface, 1);
		globals.wp_relative_pointer_manager_name = name;
		return;
	}

	if (strcmp(interface, zwp_idle_inhibit_manager_v1_interface.name) == 0) {
		globals.wp_idle_inhibit_manager = (struct zwp_idle_inhibit_manager_v1 *)wl_registry_bind(wl_registry, name, &zwp_idle_inhibit_manager_v1_interface, 1);
		globals.wp_idle_inhibit_manager_name = name;
		return;
	}
}

void DisplayServerWayland::_wl_registry_on_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name) {
	WaylandState *wls = (WaylandState *)data;
	ERR_FAIL_NULL(wls);

	WaylandGlobals &globals = wls->globals;

	if (name == globals.wl_shm_name) {
		if (globals.wl_shm) {
			wl_shm_destroy(globals.wl_shm);
		}
		globals.wl_shm = nullptr;
		globals.wl_shm_name = 0;
		return;
	}

	if (name == globals.wl_compositor_name) {
		if (globals.wl_compositor) {
			wl_compositor_destroy(globals.wl_compositor);
		}
		globals.wl_compositor = nullptr;
		globals.wl_compositor_name = 0;
		return;
	}

	if (name == globals.wl_subcompositor_name) {
		if (globals.wl_subcompositor) {
			wl_subcompositor_destroy(globals.wl_subcompositor);
		}
		globals.wl_subcompositor = nullptr;
		globals.wl_subcompositor_name = 0;
		return;
	}
	if (name == globals.wl_data_device_manager_name) {
		if (globals.wl_data_device_manager) {
			wl_data_device_manager_destroy(globals.wl_data_device_manager);
		}
		globals.wl_data_device_manager = nullptr;
		globals.wl_data_device_manager_name = 0;

		// Destroy any seat data device that's there.
		for (SeatState &ss : wls->seats) {
			if (ss.wl_data_device) {
				wl_data_device_destroy(ss.wl_data_device);
			}
			ss.wl_data_device = nullptr;
		}
	}

	if (name == globals.xdg_wm_base_name) {
		if (globals.xdg_wm_base) {
			xdg_wm_base_destroy(globals.xdg_wm_base);
		}
		globals.xdg_wm_base = nullptr;
		globals.xdg_wm_base_name = 0;
		return;
	}

	if (name == globals.xdg_decoration_manager_name) {
		if (globals.xdg_decoration_manager) {
			zxdg_decoration_manager_v1_destroy(globals.xdg_decoration_manager);
		}
		globals.xdg_decoration_manager = nullptr;
		globals.xdg_decoration_manager_name = 0;
		return;
	}

	if (name == globals.xdg_activation_name) {
		if (globals.xdg_activation) {
			xdg_activation_v1_destroy(globals.xdg_activation);
		}
		globals.xdg_activation = nullptr;
		globals.xdg_activation_name = 0;
		return;
	}

	if (name == globals.wp_pointer_constraints_name) {
		if (globals.wp_pointer_constraints) {
			zwp_pointer_constraints_v1_destroy(globals.wp_pointer_constraints);
		}
		globals.wp_pointer_constraints = nullptr;
		globals.wp_pointer_constraints_name = 0;
		return;
	}

	if (name == globals.wp_relative_pointer_manager_name) {
		if (globals.wp_relative_pointer_manager) {
			zwp_relative_pointer_manager_v1_destroy(globals.wp_relative_pointer_manager);
		}
		globals.wp_relative_pointer_manager = nullptr;
		globals.wp_relative_pointer_manager_name = 0;
		return;
	}

	if (name == globals.wp_idle_inhibit_manager_name) {
		if (globals.wp_idle_inhibit_manager) {
			zwp_idle_inhibit_manager_v1_destroy(globals.wp_idle_inhibit_manager);
		}
		globals.wp_idle_inhibit_manager = nullptr;
		globals.wp_idle_inhibit_manager_name = 0;
		return;
	}

	{
		// FIXME: This is a very bruteforce approach.
		List<ScreenData>::Element *it = wls->screens.front();
		while (it) {
			// Iterate through all of the screens to find if any got removed.
			ScreenData &sd = it->get();

			if (sd.wl_output_name == name) {
				if (sd.wl_output) {
					wl_output_destroy(sd.wl_output);
				}
				wls->screens.erase(it);
				return;
			}

			it = it->next();
		}
	}

	{
		// FIXME: This is a very bruteforce approach.
		List<SeatState>::Element *it = wls->seats.front();
		while (it) {
			// Iterate through all of the seats to find if any got removed.
			SeatState &ss = it->get();

			if (ss.wl_seat_name == name) {
				if (ss.wl_data_device) {
					wl_data_device_destroy(ss.wl_data_device);
				}

				if (ss.wl_seat) {
					wl_seat_destroy(ss.wl_seat);
				}

				wls->seats.erase(it);
				return;
			}

			it = it->next();
		}
	}
}

void DisplayServerWayland::_wl_surface_on_enter(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
}

void DisplayServerWayland::_wl_surface_on_leave(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
}

void DisplayServerWayland::_wl_output_on_geometry(void *data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char *make, const char *model, int32_t transform) {
	ScreenData *sd = (ScreenData *)data;
	ERR_FAIL_NULL(sd);

	sd->position.x = x;
	sd->position.y = y;

	sd->physical_size.width = physical_width;
	sd->physical_size.height = physical_height;

	sd->make.parse_utf8(make);
	sd->model.parse_utf8(model);
}

void DisplayServerWayland::_wl_output_on_mode(void *data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {
	ScreenData *sd = (ScreenData *)data;
	ERR_FAIL_NULL(sd);

	sd->size.width = width;
	sd->size.height = height;

	sd->refresh_rate = refresh ? refresh / 1000.0f : -1;
}

void DisplayServerWayland::_wl_output_on_done(void *data, struct wl_output *wl_output) {
	// TODO: "Atomic" output property change handling?
}

void DisplayServerWayland::_wl_output_on_scale(void *data, struct wl_output *wl_output, int32_t factor) {
	ScreenData *sd = (ScreenData *)data;
	ERR_FAIL_NULL(sd);

	sd->scale = factor;
}

void DisplayServerWayland::_wl_output_on_name(void *data, struct wl_output *wl_output, const char *name) {
}

void DisplayServerWayland::_wl_output_on_description(void *data, struct wl_output *wl_output, const char *description) {
}

void DisplayServerWayland::_wl_seat_on_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	// TODO: Handle touch.

	// Pointer handling.
	if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
		ss->cursor_surface = wl_compositor_create_surface(wls->globals.wl_compositor);

		ss->wl_pointer = wl_seat_get_pointer(wl_seat);
		wl_pointer_add_listener(ss->wl_pointer, &wl_pointer_listener, ss);

		ss->wp_relative_pointer = zwp_relative_pointer_manager_v1_get_relative_pointer(wls->globals.wp_relative_pointer_manager, ss->wl_pointer);
		zwp_relative_pointer_v1_add_listener(ss->wp_relative_pointer, &wp_relative_pointer_listener, ss);
	} else {
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
		ss->xkb_context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
		ERR_FAIL_NULL(ss->xkb_context);

		ss->wl_keyboard = wl_seat_get_keyboard(wl_seat);
		wl_keyboard_add_listener(ss->wl_keyboard, &wl_keyboard_listener, ss);
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

void DisplayServerWayland::_wl_seat_on_name(void *data, struct wl_seat *wl_seat, const char *name) {
}

void DisplayServerWayland::_wl_pointer_on_enter(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	// Restore the cursor with our own cursor surface.
	_seat_state_override_cursor_shape(*ss, wls->cursor_shape);

	PointerData &pd = ss->pointer_data_buffer;

	pd.pointed_window_id = INVALID_WINDOW_ID;

	for (KeyValue<WindowID, WindowData> &E : wls->windows) {
		WindowData &wd = E.value;

		if (wd.wl_surface == surface) {
			pd.pointed_window_id = E.key;
			break;
		}
	}

	ERR_FAIL_COND_MSG(pd.pointed_window_id == INVALID_WINDOW_ID, "Cursor focused to an invalid window.");

	DEBUG_LOG_WAYLAND(vformat("Pointing window %d", pd.pointed_window_id));

	Ref<WaylandWindowEventMessage> msg;
	msg.instantiate();
	msg->event = WINDOW_EVENT_MOUSE_ENTER;
	msg->id = pd.pointed_window_id;

	wls->message_queue.push_back(msg);
}

void DisplayServerWayland::_wl_pointer_on_leave(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	PointerData &pd = ss->pointer_data_buffer;

	if (pd.pointed_window_id != INVALID_WINDOW_ID) {
		Ref<WaylandWindowEventMessage> msg;
		msg.instantiate();
		msg->event = WINDOW_EVENT_MOUSE_EXIT;
		msg->id = pd.pointed_window_id;

		wls->message_queue.push_back(msg);
	}

	pd.pointed_window_id = INVALID_WINDOW_ID;

	DEBUG_LOG_WAYLAND("Left a surface");
}

void DisplayServerWayland::_wl_pointer_on_motion(void *data, struct wl_pointer *wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	PointerData &pd = ss->pointer_data_buffer;

	pd.position.x = wl_fixed_to_int(surface_x);
	pd.position.y = wl_fixed_to_int(surface_y);

	pd.motion_time = time;
}

void DisplayServerWayland::_wl_pointer_on_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

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

	if (state & WL_POINTER_BUTTON_STATE_PRESSED) {
		pd.pressed_button_mask |= mouse_button_to_mask(button_pressed);
		pd.last_button_pressed = button_pressed;
	} else {
		pd.pressed_button_mask &= ~mouse_button_to_mask(button_pressed);
	}

	pd.button_time = time;
	pd.button_serial = serial;
}

void DisplayServerWayland::_wl_pointer_on_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	PointerData &pd = ss->pointer_data_buffer;

	MouseButton button_pressed = MouseButton::NONE;

	switch (axis) {
		case WL_POINTER_AXIS_VERTICAL_SCROLL: {
			button_pressed = value >= 0 ? MouseButton::WHEEL_DOWN : MouseButton::WHEEL_UP;
			pd.scroll_vector.y = wl_fixed_to_double(value);
		} break;

		case WL_POINTER_AXIS_HORIZONTAL_SCROLL: {
			button_pressed = value >= 0 ? MouseButton::WHEEL_RIGHT : MouseButton::WHEEL_LEFT;
			pd.scroll_vector.x = wl_fixed_to_double(value);
		} break;
	}

	// These buttons will get unpressed when the event is sent.
	pd.pressed_button_mask |= mouse_button_to_mask(button_pressed);
	pd.last_button_pressed = button_pressed;

	pd.button_time = time;
}

void DisplayServerWayland::_wl_pointer_on_frame(void *data, struct wl_pointer *wl_pointer) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	_seat_state_set_current(*ss);

	PointerData &old_pd = ss->pointer_data;
	PointerData &pd = ss->pointer_data_buffer;

	if (pd.pointed_window_id != INVALID_WINDOW_ID) {
		if (old_pd.motion_time != pd.motion_time || old_pd.relative_motion_time != pd.relative_motion_time) {
			Ref<InputEventMouseMotion> mm;
			mm.instantiate();

			// Set all pressed modifiers.
			_get_key_modifier_state(*ss, mm);

			mm->set_window_id(pd.pointed_window_id);
			mm->set_button_mask(pd.pressed_button_mask);
			mm->set_position(pd.position);
			mm->set_global_position(_wayland_state_point_window_to_global(*wls, pd.pointed_window_id, pd.position));

			// FIXME: I'm not sure whether accessing the Input singleton like this might
			// give problems.
			Input::get_singleton()->set_mouse_position(pd.position);
			mm->set_velocity(Input::get_singleton()->get_last_mouse_velocity());

			if (old_pd.relative_motion_time != pd.relative_motion_time) {
				mm->set_relative(pd.relative_motion);
			} else {
				// The spec includes the possibility of having motion events without an
				// associated relative motion event. If that's the case, fallback to a
				// simple delta of the position.
				mm->set_relative(pd.position - old_pd.position);
			}

			Ref<WaylandInputEventMessage> msg;
			msg.instantiate();

			msg->event = mm;

			wls->message_queue.push_back(msg);
		}

		if (old_pd.pressed_button_mask != pd.pressed_button_mask) {
			MouseButton pressed_mask_delta = old_pd.pressed_button_mask ^ pd.pressed_button_mask;

			// This is the cleanest and simplest approach I could find to avoid writing the same code 7 times.
			for (MouseButton test_button : { MouseButton::LEFT, MouseButton::MIDDLE, MouseButton::RIGHT,
						 MouseButton::WHEEL_UP, MouseButton::WHEEL_DOWN, MouseButton::WHEEL_LEFT,
						 MouseButton::WHEEL_RIGHT, MouseButton::MB_XBUTTON1, MouseButton::MB_XBUTTON2 }) {
				MouseButton test_button_mask = mouse_button_to_mask(test_button);
				if ((pressed_mask_delta & test_button_mask) != MouseButton::NONE) {
					Ref<InputEventMouseButton> mb;
					mb.instantiate();

					// Set all pressed modifiers.
					_get_key_modifier_state(*ss, mb);

					mb->set_window_id(pd.pointed_window_id);
					mb->set_position(pd.position);
					mb->set_global_position(_wayland_state_point_window_to_global(*wls, pd.pointed_window_id, pd.position));

					mb->set_button_mask(pd.pressed_button_mask);

					mb->set_button_index(test_button);
					mb->set_pressed((pd.pressed_button_mask & test_button_mask) != MouseButton::NONE);

					if (pd.last_button_pressed == old_pd.last_button_pressed && (pd.button_time - old_pd.button_time) < 400 && Vector2(pd.position).distance_to(Vector2(old_pd.position)) < 5) {
						mb->set_double_click(true);
					}

					if (test_button == MouseButton::WHEEL_UP || test_button == MouseButton::WHEEL_DOWN) {
						mb->set_factor(abs(pd.scroll_vector.y));
					}

					if (test_button == MouseButton::WHEEL_RIGHT || test_button == MouseButton::WHEEL_LEFT) {
						mb->set_factor(abs(pd.scroll_vector.x));
					}

					Ref<WaylandInputEventMessage> msg;
					msg.instantiate();

					msg->event = mb;

					wls->message_queue.push_back(msg);

					// Send an event resetting immediately the wheel key.
					// Wayland specification defines axis_stop events as optional and says to
					// treat all axis events as unterminated. As such, we have to manually do
					// it ourselves.
					if (test_button == MouseButton::WHEEL_UP || test_button == MouseButton::WHEEL_DOWN || test_button == MouseButton::WHEEL_LEFT || test_button == MouseButton::WHEEL_RIGHT) {
						// FIXME: This is ugly, I can't find a clean way to clone an InputEvent.
						// This works for now, despite being horrible.
						Ref<InputEventMouseButton> wh_up;
						wh_up.instantiate();

						wh_up->set_window_id(pd.pointed_window_id);
						wh_up->set_position(pd.position);
						wh_up->set_global_position(_wayland_state_point_window_to_global(*wls, pd.pointed_window_id, pd.position));

						// We have to unset the button to avoid it getting stuck.
						pd.pressed_button_mask &= ~test_button_mask;
						wh_up->set_button_mask(pd.pressed_button_mask);

						wh_up->set_button_index(test_button);
						wh_up->set_pressed(false);

						Ref<WaylandInputEventMessage> msg_up;
						msg_up.instantiate();
						msg_up->event = wh_up;
						wls->message_queue.push_back(msg_up);
					}
				}
			}
		}
	}

	// Update the data all getters read. Wayland's specification requires us to do
	// this, since all pointer actions are sent in individual events.
	old_pd = pd;
}

void DisplayServerWayland::_wl_pointer_on_axis_source(void *data, struct wl_pointer *wl_pointer, uint32_t axis_source) {
}

void DisplayServerWayland::_wl_pointer_on_axis_stop(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis) {
}

void DisplayServerWayland::_wl_pointer_on_axis_discrete(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete) {
}

void DisplayServerWayland::_wl_pointer_on_axis_value120(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t value120) {
}

void DisplayServerWayland::_wl_keyboard_on_keymap(void *data, struct wl_keyboard *wl_keyboard, uint32_t format, int32_t fd, uint32_t size) {
	ERR_FAIL_COND_MSG(format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1, "Unsupported keymap format announced from the Wayland compositor.");

	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	if (ss->keymap_buffer) {
		// We have already a mapped buffer, so we unmap it. There's no need to reset
		// its pointer or size, as we're gonna set them below.
		munmap((void *)ss->keymap_buffer, ss->keymap_buffer_size);
		ss->keymap_buffer = nullptr;
	}

	ss->keymap_buffer = (const char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	ss->keymap_buffer_size = size;

	xkb_keymap_unref(ss->xkb_keymap);
	ss->xkb_keymap = xkb_keymap_new_from_string(ss->xkb_context, ss->keymap_buffer,
			XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);

	xkb_state_unref(ss->xkb_state);
	ss->xkb_state = xkb_state_new(ss->xkb_keymap);
}

void DisplayServerWayland::_wl_keyboard_on_enter(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	_seat_state_set_current(*ss);

	for (KeyValue<WindowID, WindowData> &E : wls->windows) {
		WindowData &wd = E.value;

		if (wd.wl_surface == surface) {
			ss->keyboard_focused_window_id = E.key;
			break;
		}
	}

	Ref<WaylandWindowEventMessage> msg;
	msg.instantiate();
	msg->id = ss->keyboard_focused_window_id;
	msg->event = WINDOW_EVENT_FOCUS_IN;
	wls->message_queue.push_back(msg);
}

void DisplayServerWayland::_wl_keyboard_on_leave(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	Ref<WaylandWindowEventMessage> msg;
	msg.instantiate();

	msg->id = ss->keyboard_focused_window_id;
	msg->event = WINDOW_EVENT_FOCUS_OUT;

	wls->message_queue.push_back(msg);

	ss->keyboard_focused_window_id = INVALID_WINDOW_ID;
	ss->repeating_keycode = XKB_KEYCODE_INVALID;
}

void DisplayServerWayland::_wl_keyboard_on_key(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	_seat_state_set_current(*ss);

	// We have to add 8 to the scancode to get an XKB-compatible keycode.
	xkb_keycode_t xkb_keycode = key + 8;

	bool pressed = state & WL_KEYBOARD_KEY_STATE_PRESSED;

	if (pressed) {
		if (xkb_keymap_key_repeats(ss->xkb_keymap, xkb_keycode)) {
			ss->last_repeat_start_msec = OS::get_singleton()->get_ticks_msec();
			ss->repeating_keycode = xkb_keycode;
		}

		ss->last_key_pressed_serial = serial;
	} else if (ss->repeating_keycode == xkb_keycode) {
		ss->repeating_keycode = XKB_KEYCODE_INVALID;
	}

	Ref<InputEventKey> k;
	k.instantiate();

	if (!_seat_state_configure_key_event(*ss, k, xkb_keycode, pressed)) {
		return;
	}

	Ref<WaylandInputEventMessage> msg;
	msg.instantiate();
	msg->event = k;
	wls->message_queue.push_back(msg);
}

void DisplayServerWayland::_wl_keyboard_on_modifiers(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	_seat_state_set_current(*ss);

	xkb_state_update_mask(ss->xkb_state, mods_depressed, mods_latched, mods_locked, ss->current_layout_index, ss->current_layout_index, group);

	ss->shift_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_SHIFT, XKB_STATE_MODS_DEPRESSED);
	ss->ctrl_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_CTRL, XKB_STATE_MODS_DEPRESSED);
	ss->alt_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_ALT, XKB_STATE_MODS_DEPRESSED);
	ss->meta_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_LOGO, XKB_STATE_MODS_DEPRESSED);
}

void DisplayServerWayland::_wl_keyboard_on_repeat_info(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->repeat_key_delay_msec = 1000 / rate;
	ss->repeat_start_delay_msec = delay;
}

void DisplayServerWayland::_wl_data_device_on_data_offer(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id) {
	ERR_FAIL_NULL(data);

	wl_data_offer_add_listener(id, &wl_data_offer_listener, data);
}

void DisplayServerWayland::_wl_data_device_on_enter(void *data, struct wl_data_device *wl_data_device, uint32_t serial, struct wl_surface *surface, wl_fixed_t x, wl_fixed_t y, struct wl_data_offer *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	for (KeyValue<WindowID, WindowData> &E : ss->wls->windows) {
		WindowData &wd = E.value;

		if (wd.wl_surface == surface) {
			ss->dnd_current_window_id = E.key;
			break;
		}
	}

	ss->dnd_enter_serial = serial;

	wl_data_offer_set_actions(id, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);

	ss->wl_data_offer_dnd = id;
}

void DisplayServerWayland::_wl_data_device_on_leave(void *data, struct wl_data_device *wl_data_device) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->dnd_current_window_id = INVALID_WINDOW_ID;

	if (ss->wl_data_offer_dnd) {
		wl_data_offer_destroy(ss->wl_data_offer_dnd);
		ss->wl_data_offer_dnd = nullptr;
	}
}

void DisplayServerWayland::_wl_data_device_on_motion(void *data, struct wl_data_device *wl_data_device, uint32_t time, wl_fixed_t x, wl_fixed_t y) {
}

void DisplayServerWayland::_wl_data_device_on_drop(void *data, struct wl_data_device *wl_data_device) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	int fds[2];
	if (pipe(fds) == 0) {
		wl_data_offer_receive(ss->wl_data_offer_dnd, "text/uri-list", fds[1]);

		// Let the compositor know about the pipe, but don't handle any event. For
		// some cursed reason both leave and drop events are released at the same
		// time, although both of them clean up the offer. I'm still not sure why.
		wl_display_flush(ss->wls->wl_display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		Ref<WaylandDropFilesEventMessage> msg;
		msg.instantiate();

		msg->id = ss->dnd_current_window_id;

		msg->files = _string_read_fd(fds[0]).split("\n", false);
		for (int i = 0; i < msg->files.size(); i++) {
			msg->files.write[i] = msg->files[i].replace("file://", "").uri_decode().strip_edges();
		}

		ss->wls->message_queue.push_back(msg);
	}

	wl_data_offer_finish(ss->wl_data_offer_dnd);
	wl_data_offer_destroy(ss->wl_data_offer_dnd);
	ss->wl_data_offer_dnd = nullptr;
}

void DisplayServerWayland::_wl_data_device_on_selection(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (ss->wl_data_offer_selection) {
		wl_data_offer_destroy(ss->wl_data_offer_selection);
	}

	ss->wl_data_offer_selection = id;
}

void DisplayServerWayland::_wl_data_offer_on_offer(void *data, struct wl_data_offer *wl_data_offer, const char *mime_type) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (strcmp(mime_type, "text/uri-list") == 0) {
		wl_data_offer_accept(wl_data_offer, ss->dnd_enter_serial, mime_type);
	}
}

void DisplayServerWayland::_wl_data_offer_on_source_actions(void *data, struct wl_data_offer *wl_data_offer, uint32_t source_actions) {
}

void DisplayServerWayland::_wl_data_offer_on_action(void *data, struct wl_data_offer *wl_data_offer, uint32_t dnd_action) {
}

void DisplayServerWayland::_wl_data_source_on_target(void *data, struct wl_data_source *wl_data_source, const char *mime_type) {
}

void DisplayServerWayland::_wl_data_source_on_send(void *data, struct wl_data_source *wl_data_source, const char *mime_type, int32_t fd) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	Vector<uint8_t> *data_to_send = nullptr;

	if (wl_data_source == ss->wl_data_source_selection) {
		data_to_send = &ss->selection_data;
		DEBUG_LOG_WAYLAND("Clipboard: requested selection.");
	}

	if (data_to_send) {
		ssize_t written_bytes = 0;

		if (strcmp(mime_type, "text/plain") == 0) {
			written_bytes = write(fd, data_to_send->ptr(), data_to_send->size());
		}

		if (written_bytes > 0) {
			DEBUG_LOG_WAYLAND(vformat("Clipboard: sent %d bytes.", written_bytes));
		} else if (written_bytes == 0) {
			DEBUG_LOG_WAYLAND("Clipboard: no bytes sent.");
		} else {
			ERR_PRINT(vformat("Clipboard: write error %d.", errno));
		}
	}

	close(fd);
}

void DisplayServerWayland::_wl_data_source_on_cancelled(void *data, struct wl_data_source *wl_data_source) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	wl_data_source_destroy(wl_data_source);

	if (wl_data_source == ss->wl_data_source_selection) {
		ss->wl_data_source_selection = nullptr;

		ss->selection_data.clear();

		DEBUG_LOG_WAYLAND("Clipboard: selection set by another program.");
		return;
	}
}

void DisplayServerWayland::_wl_data_source_on_dnd_drop_performed(void *data, struct wl_data_source *wl_data_source) {
}

void DisplayServerWayland::_wl_data_source_on_dnd_finished(void *data, struct wl_data_source *wl_data_source) {
}

void DisplayServerWayland::_wl_data_source_on_action(void *data, struct wl_data_source *wl_data_source, uint32_t dnd_action) {
}

void DisplayServerWayland::_xdg_wm_base_on_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}

void DisplayServerWayland::_xdg_surface_on_configure(void *data, struct xdg_surface *xdg_surface, uint32_t serial) {
	xdg_surface_ack_configure(xdg_surface, serial);

	WindowData *wd = (WindowData *)data;
	ERR_FAIL_NULL(wd);

	WaylandState *wls = (WaylandState *)wd->wls;
	ERR_FAIL_NULL(wls);

	xdg_surface_set_window_geometry(wd->xdg_surface, 0, 0, wd->rect.size.width, wd->rect.size.height);

	Ref<WaylandWindowRectMessage> msg;
	msg.instantiate();
	msg->id = wd->id;
	msg->rect = wd->rect;
	wls->message_queue.push_back(msg);

	DEBUG_LOG_WAYLAND(vformat("xdg surface id %d on configure rect %s", wd->id, wd->rect));
}

void DisplayServerWayland::_xdg_toplevel_on_configure(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states) {
	WindowData *wd = (WindowData *)data;
	ERR_FAIL_NULL(wd);

	WaylandState *wls = wd->wls;

	if (width != 0 && height != 0) {
		wd->rect.size.width = width;
		wd->rect.size.height = height;
	}

	if (wls->current_seat && wls->current_seat->wp_locked_pointer && wd->id == wls->current_seat->pointer_data.pointed_window_id) {
		// Since the cursor's currently locked and the window's rect might have
		// changed, we have to recenter the position hint to ensure that the cursor
		// stays centered on unlock.
		wl_fixed_t unlock_x = wl_fixed_from_int(width / 2);
		wl_fixed_t unlock_y = wl_fixed_from_int(height / 2);

		zwp_locked_pointer_v1_set_cursor_position_hint(wls->current_seat->wp_locked_pointer, unlock_x, unlock_y);
	}

	// Expect the window to be in windowed mode. The mode will get overridden if
	// the compositor reports otherwise.
	wd->mode = WINDOW_MODE_WINDOWED;

	uint32_t *state = nullptr;
	wl_array_for_each(state, states) {
		switch (*state) {
			case XDG_TOPLEVEL_STATE_MAXIMIZED: {
				wd->mode = WINDOW_MODE_MAXIMIZED;
			} break;

			case XDG_TOPLEVEL_STATE_FULLSCREEN: {
				wd->mode = WINDOW_MODE_FULLSCREEN;
			} break;

			default: {
				// We don't care about the other states (for now).
			} break;
		}
	}

	DEBUG_LOG_WAYLAND(vformat("xdg toplevel on configure width %d height %d", width, height));
}

void DisplayServerWayland::_xdg_toplevel_on_close(void *data, struct xdg_toplevel *xdg_toplevel) {
	WindowData *wd = (WindowData *)data;
	ERR_FAIL_NULL(wd);

	WaylandState *wls = (WaylandState *)wd->wls;
	ERR_FAIL_NULL(wls);

	Ref<WaylandWindowEventMessage> msg;
	msg.instantiate();
	msg->id = wd->id;
	msg->event = WINDOW_EVENT_CLOSE_REQUEST;
	wls->message_queue.push_back(msg);
}

void DisplayServerWayland::_xdg_toplevel_on_configure_bounds(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height) {
}

void DisplayServerWayland::_xdg_toplevel_on_wm_capabilities(void *data, struct xdg_toplevel *xdg_toplevel, struct wl_array *capabilities) {
}

void DisplayServerWayland::_xdg_toplevel_decoration_on_configure(void *data, struct zxdg_toplevel_decoration_v1 *xdg_toplevel_decoration, uint32_t mode) {
	if (mode == ZXDG_TOPLEVEL_DECORATION_V1_MODE_CLIENT_SIDE) {
		WARN_PRINT_ONCE("Client side decorations are not yet supported!");
	}
}

void DisplayServerWayland::_xdg_activation_token_on_done(void *data, struct xdg_activation_token_v1 *xdg_activation_token, const char *token) {
	WindowData *wd = (WindowData *)data;
	ERR_FAIL_NULL(wd);

	ERR_FAIL_NULL(wd->wl_surface);

	xdg_activation_v1_activate(wd->wls->globals.xdg_activation, token, wd->wl_surface);

	xdg_activation_token_v1_destroy(xdg_activation_token);

	DEBUG_LOG_WAYLAND(vformat("Received activation token and requested window activation for %d", wd->id));
}

void DisplayServerWayland::_wp_relative_pointer_on_relative_motion(void *data, struct zwp_relative_pointer_v1 *wp_relative_pointer, uint32_t uptime_hi, uint32_t uptime_lo, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t dx_unaccel, wl_fixed_t dy_unaccel) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	PointerData &pd = ss->pointer_data_buffer;

	pd.relative_motion.x = wl_fixed_to_double(dx);
	pd.relative_motion.y = wl_fixed_to_double(dy);

	pd.relative_motion_time = uptime_lo;
}

void DisplayServerWayland::_wp_primary_selection_device_on_data_offer(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *offer) {
	// This method is purposely left unimplemented as we don't care about the
	// offered MIME type, as we only want `text/plain` data.

	// TODO: Perhaps we could try to detect other text types such as `TEXT`?
}

void DisplayServerWayland::_wp_primary_selection_device_on_selection(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (ss->wp_primary_selection_offer) {
		zwp_primary_selection_offer_v1_destroy(ss->wp_primary_selection_offer);
	}

	ss->wp_primary_selection_offer = id;
}

void DisplayServerWayland::_wp_primary_selection_source_on_send(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1, const char *mime_type, int32_t fd) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	Vector<uint8_t> *data_to_send = nullptr;

	if (wp_primary_selection_source_v1 == ss->wp_primary_selection_source) {
		data_to_send = &ss->primary_data;
		DEBUG_LOG_WAYLAND("Clipboard: requested primary selection.");
	}

	if (data_to_send) {
		ssize_t written_bytes = 0;

		if (strcmp(mime_type, "text/plain") == 0) {
			written_bytes = write(fd, data_to_send->ptr(), data_to_send->size());
		}

		if (written_bytes > 0) {
			DEBUG_LOG_WAYLAND(vformat("Clipboard: sent %d bytes.", written_bytes));
		} else if (written_bytes == 0) {
			DEBUG_LOG_WAYLAND("Clipboard: no bytes sent.");
		} else {
			ERR_PRINT(vformat("Clipboard: write error %d.", errno));
		}
	}

	close(fd);
}

void DisplayServerWayland::_wp_primary_selection_source_on_cancelled(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (wp_primary_selection_source_v1 == ss->wp_primary_selection_source) {
		zwp_primary_selection_source_v1_destroy(ss->wp_primary_selection_source);
		ss->wp_primary_selection_source = nullptr;

		ss->primary_data.clear();

		DEBUG_LOG_WAYLAND("Clipboard: primary selection set by another program.");
		return;
	}
}

// Interface mthods

bool DisplayServerWayland::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_MOUSE:
		case FEATURE_CLIPBOARD:
		case FEATURE_CURSOR_SHAPE:
		case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_SWAP_BUFFERS:
		case FEATURE_KEEP_SCREEN_ON:
		case FEATURE_CLIPBOARD_PRIMARY: {
			return true;
		} break;

		default: {
			return false;
		}
	}
}

String DisplayServerWayland::get_name() const {
	return "Wayland";
}

#ifdef SPEECHD_ENABLED

bool DisplayServerWayland::tts_is_speaking() const {
	ERR_FAIL_COND_V(!tts, false);
	return tts->is_speaking();
}

bool DisplayServerWayland::tts_is_paused() const {
	ERR_FAIL_COND_V(!tts, false);
	return tts->is_paused();
}

TypedArray<Dictionary> DisplayServerWayland::tts_get_voices() const {
	ERR_FAIL_COND_V(!tts, TypedArray<Dictionary>());
	return tts->get_voices();
}

void DisplayServerWayland::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	ERR_FAIL_COND(!tts);
	tts->speak(p_text, p_voice, p_volume, p_pitch, p_rate, p_utterance_id, p_interrupt);
}

void DisplayServerWayland::tts_pause() {
	ERR_FAIL_COND(!tts);
	tts->pause();
}

void DisplayServerWayland::tts_resume() {
	ERR_FAIL_COND(!tts);
	tts->resume();
}

void DisplayServerWayland::tts_stop() {
	ERR_FAIL_COND(!tts);
	tts->stop();
}

#endif

#ifdef DBUS_ENABLED

bool DisplayServerWayland::is_dark_mode_supported() const {
	return portal_desktop->is_supported();
}

bool DisplayServerWayland::is_dark_mode() const {
	switch (portal_desktop->get_appearance_color_scheme()) {
		case 1:
			// Prefers dark theme.
			return true;
		case 2:
			// Prefers light theme.
			return false;
		default:
			// Preference unknown.
			return false;
	}
}

#endif

void DisplayServerWayland::mouse_set_mode(MouseMode p_mode) {
	if (p_mode == wls.mouse_mode) {
		return;
	}

	MutexLock mutex_lock(wls.mutex);

	wls.mouse_mode = p_mode;

	_wayland_state_update_cursor(wls);
}

DisplayServerWayland::MouseMode DisplayServerWayland::mouse_get_mode() const {
	return wls.mouse_mode;
}

void DisplayServerWayland::warp_mouse(const Point2i &p_to) {
	// NOTE: This is hacked together as for some reason the pointer constraints
	// protocol doesn't implement pointer warping (not even in the window). This
	// isn't efficient *at all* and perhaps there could be better behaviours in
	// the pointer capturing logic in general, but this will do for now.
	MutexLock mutex_lock(wls.mutex);

	if (wls.current_seat) {
		MouseMode old_mouse_mode = wls.mouse_mode;

		mouse_set_mode(MOUSE_MODE_CAPTURED);

		// Override the hint set by MOUSE_MODE_CAPTURED with the requested one.
		zwp_locked_pointer_v1_set_cursor_position_hint(wls.current_seat->wp_locked_pointer, wl_fixed_from_int(p_to.x), wl_fixed_from_int(p_to.y));
		// Committing the surface is required to set the hint instantly.
		wl_surface_commit(wls.windows[MAIN_WINDOW_ID].wl_surface);

		mouse_set_mode(old_mouse_mode);
	}
}

Point2i DisplayServerWayland::mouse_get_position() const {
	MutexLock mutex_lock(wls.mutex);

	if (wls.current_seat) {
		return _wayland_state_point_window_to_global(wls, wls.current_seat->pointer_data.pointed_window_id, wls.current_seat->pointer_data.position);
	}

	return Point2i();
}

MouseButton DisplayServerWayland::mouse_get_button_state() const {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return MouseButton::NONE;
	}

	return wls.current_seat->pointer_data.pressed_button_mask;
}

// NOTE: According to the Wayland specification, this method will only do
// anything if the user has interacted with the application by sending a
// "recent enough" input event.
// TODO: Add this limitation to the documentation.
void DisplayServerWayland::clipboard_set(const String &p_text) {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return;
	}

	SeatState &ss = *wls.current_seat;

	if (!wls.current_seat->wl_data_source_selection) {
		ss.wl_data_source_selection = wl_data_device_manager_create_data_source(wls.globals.wl_data_device_manager);
		wl_data_source_add_listener(ss.wl_data_source_selection, &wl_data_source_listener, wls.current_seat);
		wl_data_source_offer(ss.wl_data_source_selection, "text/plain");
	}

	// TODO: Implement a good way of getting the latest serial from the user.
	wl_data_device_set_selection(ss.wl_data_device, ss.wl_data_source_selection, MAX(ss.pointer_data.button_serial, ss.last_key_pressed_serial));

	// Wait for the message to get to the server before continuing, otherwise the
	// clipboard update might come with a delay.
	wl_display_roundtrip(wls.wl_display);

	ss.selection_data = p_text.to_utf8_buffer();
}

String DisplayServerWayland::clipboard_get() const {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return String();
	}

	return _wl_data_offer_read(wls.current_seat->wl_data_offer_selection);
}

void DisplayServerWayland::clipboard_set_primary(const String &p_text) {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return;
	}

	SeatState &ss = *wls.current_seat;

	if (!wls.current_seat->wp_primary_selection_source) {
		ss.wp_primary_selection_source = zwp_primary_selection_device_manager_v1_create_source(wls.globals.wp_primary_selection_device_manager);
		zwp_primary_selection_source_v1_add_listener(ss.wp_primary_selection_source, &wp_primary_selection_source_listener, wls.current_seat);
		zwp_primary_selection_source_v1_offer(ss.wp_primary_selection_source, "text/plain");
	}

	// TODO: Implement a good way of getting the latest serial from the user.
	zwp_primary_selection_device_v1_set_selection(ss.wp_primary_selection_device, ss.wp_primary_selection_source, MAX(ss.pointer_data.button_serial, ss.last_key_pressed_serial));

	// Wait for the message to get to the server before continuing, otherwise the
	// clipboard update might come with a delay.
	wl_display_roundtrip(wls.wl_display);

	ss.primary_data = p_text.to_utf8_buffer();
}

String DisplayServerWayland::clipboard_get_primary() const {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return String();
	}

	return _wp_primary_selection_offer_read(wls.current_seat->wp_primary_selection_offer);
}

int DisplayServerWayland::get_screen_count() const {
	MutexLock mutex_lock(wls.mutex);
	return wls.screens.size();
}

Point2i DisplayServerWayland::screen_get_position(int p_screen) const {
	MutexLock mutex_lock(wls.mutex);

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	ERR_FAIL_INDEX_V(p_screen, (int)wls.screens.size(), Point2i());

	return wls.screens[p_screen].position;
}

Size2i DisplayServerWayland::screen_get_size(int p_screen) const {
	MutexLock mutex_lock(wls.mutex);

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	ERR_FAIL_INDEX_V(p_screen, (int)wls.screens.size(), Size2i());

	return wls.screens[p_screen].size;
}

Rect2i DisplayServerWayland::screen_get_usable_rect(int p_screen) const {
	// Unsupported on wayland.
	return Rect2i(Point2i(), screen_get_size(p_screen));
}

int DisplayServerWayland::screen_get_dpi(int p_screen) const {
	MutexLock mutex_lock(wls.mutex);

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	// Invalid screen?
	ERR_FAIL_INDEX_V(p_screen, get_screen_count(), 0);

	const ScreenData &sd = wls.screens[p_screen];

	int width_mm = sd.physical_size.width;
	int height_mm = sd.physical_size.height;

	double xdpi = (width_mm ? sd.size.width / (double)width_mm * 25.4 : 0);
	double ydpi = (height_mm ? sd.size.height / (double)height_mm * 25.4 : 0);

	if (xdpi || ydpi) {
		return (xdpi + ydpi) / (xdpi && ydpi ? 2 : 1);
	}

	// Could not get DPI.
	return 96;
}

float DisplayServerWayland::screen_get_refresh_rate(int p_screen) const {
	MutexLock mutex_lock(wls.mutex);

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	ERR_FAIL_INDEX_V(p_screen, (int)wls.screens.size(), -1);

	return wls.screens[p_screen].refresh_rate;
}

void DisplayServerWayland::screen_set_keep_on(bool p_enable) {
	MutexLock mutex_lock(wls.mutex);

	if (screen_is_kept_on() == p_enable) {
		return;
	}

	if (p_enable) {
		if (wls.globals.wp_idle_inhibit_manager && !wls.wp_idle_inhibitor) {
			ERR_FAIL_COND(!wls.windows.has(MAIN_WINDOW_ID));
			ERR_FAIL_COND(!wls.windows[MAIN_WINDOW_ID].wl_surface);
			wls.wp_idle_inhibitor = zwp_idle_inhibit_manager_v1_create_inhibitor(wls.globals.wp_idle_inhibit_manager, wls.windows[MAIN_WINDOW_ID].wl_surface);
		}
	} else {
		if (wls.wp_idle_inhibitor) {
			zwp_idle_inhibitor_v1_destroy(wls.wp_idle_inhibitor);
			wls.wp_idle_inhibitor = nullptr;
		}
	}

#ifdef DBUS_ENABLED
	if (screensaver) {
		if (p_enable) {
			screensaver->inhibit();
		} else {
			screensaver->uninhibit();
		}

		screensaver_inhibited = p_enable;
	}
#endif
}

bool DisplayServerWayland::screen_is_kept_on() const {
#ifdef DBUS_ENABLED
	return wls.wp_idle_inhibitor != nullptr || screensaver_inhibited;
#endif

	return wls.wp_idle_inhibitor != nullptr;
}

Vector<DisplayServer::WindowID> DisplayServerWayland::get_window_list() const {
	MutexLock mutex_lock(wls.mutex);

	Vector<int> ret;
	for (const KeyValue<WindowID, WindowData> &E : wls.windows) {
		ret.push_back(E.key);
	}

	return ret;
}

DisplayServer::WindowID DisplayServerWayland::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	return _create_window(p_mode, p_vsync_mode, p_flags, p_rect);
}

void DisplayServerWayland::_show_window(DisplayServer::WindowID p_id) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_id));
	WindowData &wd = wls.windows[p_id];

	if (!wd.visible) {
		DEBUG_LOG_WAYLAND(vformat("Showing window %d", p_id));

		wd.wl_surface = wl_compositor_create_surface(wls.globals.wl_compositor);
		wl_surface_add_listener(wd.wl_surface, &wl_surface_listener, &wd);

		wd.xdg_surface = xdg_wm_base_get_xdg_surface(wls.globals.xdg_wm_base, wd.wl_surface);
		xdg_surface_add_listener(wd.xdg_surface, &xdg_surface_listener, &wd);

		wd.xdg_toplevel = xdg_surface_get_toplevel(wd.xdg_surface);

		if (!window_get_flag(WINDOW_FLAG_BORDERLESS, p_id) && wls.globals.xdg_decoration_manager) {
			wd.xdg_toplevel_decoration = zxdg_decoration_manager_v1_get_toplevel_decoration(wls.globals.xdg_decoration_manager, wd.xdg_toplevel);
			zxdg_toplevel_decoration_v1_add_listener(wd.xdg_toplevel_decoration, &xdg_toplevel_decoration_listener, &wd);

			zxdg_toplevel_decoration_v1_set_mode(wd.xdg_toplevel_decoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
		}

		xdg_toplevel_add_listener(wd.xdg_toplevel, &xdg_toplevel_listener, &wd);

		if (wd.parent != INVALID_WINDOW_ID) {
			ERR_FAIL_COND(!wls.windows.has(wd.parent));

			WindowData &parent_wd = wls.windows[wd.parent];

			if (parent_wd.xdg_toplevel) {
				xdg_toplevel_set_parent(wd.xdg_toplevel, parent_wd.xdg_toplevel);
			}
		}

		if (wd.title.utf8().ptr()) {
			xdg_toplevel_set_title(wd.xdg_toplevel, wd.title.utf8().ptr());
		}

		if (window_get_flag(WINDOW_FLAG_POPUP, p_id)) {
			if (wls.popup_list.size() > 0 && wls.popup_list.front()->get() != wd.parent) {
				// Delete the whole popup tree, we're creating a new one with a different root.
				_send_window_event(wls.popup_list.front()->get(), WINDOW_EVENT_CLOSE_REQUEST);
				wls.popup_list.clear();
			}

			wls.popup_list.push_back(p_id);

			// Focus the popup.
			if (wls.current_seat) {
				wls.current_seat->keyboard_focused_window_id = p_id;
			}
		}

		wl_surface_commit(wd.wl_surface);

		// Wait for the surface to be configured before continuing.
		wl_display_roundtrip(wls.wl_display);

#ifdef VULKAN_ENABLED
		// Since `VulkanContextWayland::window_create` automatically assigns a buffer
		// to the `wl_surface` and doing so instantly maps it, moving this method here
		// is the only solution I can think of to implement this method properly.
		if (wls.context_vulkan) {
			Error err = wls.context_vulkan->window_create(p_id, wd.vsync_mode, wls.wl_display, wd.wl_surface, wd.rect.size.width, wd.rect.size.height);
			ERR_FAIL_COND_MSG(err == ERR_CANT_CREATE, "Can't show a Vulkan window.");
		}
#endif

#ifdef GLES3_ENABLED
		if (wls.gl_manager) {
			Error err = wls.gl_manager->window_create(p_id, wls.wl_display, wd.wl_surface, wd.rect.size.width, wd.rect.size.height);
			ERR_FAIL_COND_MSG(err == ERR_CANT_CREATE, "Can't show a GLES3 window.");
		}
#endif
		wd.visible = true;

		if (wd.xdg_toplevel) {
			// Actually try to apply any mode the window has now that it's visible.
			_window_data_set_mode(wd, wd.mode);
		}
	}
}

void DisplayServerWayland::delete_sub_window(DisplayServer::WindowID p_id) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_id));

	ERR_FAIL_COND_MSG(p_id == MAIN_WINDOW_ID, "Main window can't be deleted.");

	WindowData &wd = wls.windows[p_id];

	if (window_get_flag(WINDOW_FLAG_BORDERLESS, p_id)) {
		DEBUG_LOG_WAYLAND(vformat("Destroying popup %d.", p_id));
		wls.popup_list.erase(p_id);
	} else {
		DEBUG_LOG_WAYLAND(vformat("Destroying window %d.", p_id));
	}

	while (wd.children.size()) {
		// Unparent all children of the window.
		window_set_transient(*wd.children.begin(), INVALID_WINDOW_ID);
	}

	if (wd.parent != INVALID_WINDOW_ID) {
		window_set_transient(p_id, INVALID_WINDOW_ID);
	}

#ifdef VULKAN_ENABLED
	if (wls.context_vulkan && wd.visible) {
		wls.context_vulkan->window_destroy(p_id);
	}
#endif

#ifdef GLES3_ENABLED
	if (wls.gl_manager && wd.visible) {
		wls.gl_manager->window_destroy(p_id);
	}
#endif

	if (wd.xdg_toplevel_decoration) {
		zxdg_toplevel_decoration_v1_destroy(wd.xdg_toplevel_decoration);
	}

	if (wd.xdg_toplevel) {
		xdg_toplevel_destroy(wd.xdg_toplevel);
	}

	if (wd.xdg_surface) {
		xdg_surface_destroy(wd.xdg_surface);
	}

	if (wd.wl_surface) {
		wl_surface_destroy(wd.wl_surface);
	}

	wls.windows.erase(p_id);
}

DisplayServer::WindowID DisplayServerWayland::get_window_at_screen_position(const Point2i &p_position) const {
	// Standard Wayland APIs don't support this.
	return MAIN_WINDOW_ID;
}

void DisplayServerWayland::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	wd.instance_id = p_instance;
}

ObjectID DisplayServerWayland::window_get_attached_instance_id(WindowID p_window) const {
	ERR_FAIL_COND_V(!wls.windows.has(p_window), ObjectID());
	const WindowData &wd = wls.windows[p_window];
	return wd.instance_id;
}

void DisplayServerWayland::window_set_title(const String &p_title, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));

	WindowData &wd = wls.windows[p_window];

	wd.title = p_title;

	if (wd.xdg_toplevel) {
		xdg_toplevel_set_title(wd.xdg_toplevel, p_title.utf8().get_data());
	}
}

void DisplayServerWayland::window_set_mouse_passthrough(const Vector<Vector2> &p_region, DisplayServer::WindowID p_window) {
	// TODO
	DEBUG_LOG_WAYLAND(vformat("wayland stub window_set_mouse_passthrough region %s window %d", p_region, p_window));
}

void DisplayServerWayland::window_set_rect_changed_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	wd.rect_changed_callback = p_callable;
}

void DisplayServerWayland::window_set_window_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	wd.window_event_callback = p_callable;
}

void DisplayServerWayland::window_set_input_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	wd.input_event_callback = p_callable;
}

void DisplayServerWayland::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];
	wd.input_text_callback = p_callable;
}

void DisplayServerWayland::window_set_drop_files_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	wd.drop_files_callback = p_callable;
}

int DisplayServerWayland::window_get_current_screen(DisplayServer::WindowID p_window) const {
	// Standard Wayland APIs don't support getting the screen of a window.
	return 0;
}

void DisplayServerWayland::window_set_current_screen(int p_screen, DisplayServer::WindowID p_window) {
	// Standard Wayland APIs don't support setting the screen of a window.
}

Point2i DisplayServerWayland::window_get_position(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND_V(!wls.windows.has(p_window), Point2i());

	if (wls.windows[p_window].xdg_toplevel) {
		// We can't know the position of toplevels with the standard protocol.
		return Point2i();
	} else {
		return wls.windows[p_window].rect.position;
	}
}

void DisplayServerWayland::window_set_position(const Point2i &p_position, DisplayServer::WindowID p_window) {
	// Setting the position of a non borderless window is not supported.
}

void DisplayServerWayland::window_set_max_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));

	DEBUG_LOG_WAYLAND(vformat("window id %d max size set to %s", p_window, p_size));

	if (p_size.x < 0 || p_size.y < 0) {
		ERR_FAIL_MSG("Maximum window size can't be negative!");
	}

	WindowData &wd = wls.windows[p_window];
	if ((p_size != Size2i()) && ((p_size.x < wd.min_size.x) || (p_size.y < wd.min_size.y))) {
		ERR_PRINT("Maximum window size can't be smaller than minimum window size!");
		return;
	}

	wd.max_size = p_size;

	if (wd.wl_surface && wd.xdg_toplevel) {
		xdg_toplevel_set_max_size(wd.xdg_toplevel, p_size.width, p_size.height);
		wl_surface_commit(wd.wl_surface);
	}
}

Size2i DisplayServerWayland::window_get_max_size(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND_V(!wls.windows.has(p_window), Size2i());

	return wls.windows[p_window].max_size;
}

void DisplayServerWayland::gl_window_make_current(DisplayServer::WindowID p_window_id) {
#if defined(GLES3_ENABLED)
	if (wls.gl_manager) {
		wls.gl_manager->window_make_current(p_window_id);
	}
#endif
}

// TODO: Make accurate with the X11 implementation.
void DisplayServerWayland::window_set_transient(DisplayServer::WindowID p_window, DisplayServer::WindowID p_parent) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(p_window == p_parent);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	ERR_FAIL_COND(wd.parent == p_parent);

	struct xdg_toplevel *parent_toplevel = nullptr;

	// Unset the window's parent if there's already one.
	if (wd.parent != INVALID_WINDOW_ID) {
		ERR_FAIL_COND(!wls.windows.has(wd.parent));
		WindowData &old_parent_wd = wls.windows[wd.parent];

		ERR_FAIL_COND(!old_parent_wd.children.has(p_window));
		old_parent_wd.children.erase(p_window);

		wd.parent = INVALID_WINDOW_ID;
	}

	if (p_parent != INVALID_WINDOW_ID) {
		ERR_FAIL_COND(!wls.windows.has(p_parent));
		WindowData &parent_wd = wls.windows[p_parent];

		parent_wd.children.insert(p_window);
		wd.parent = p_parent;

		ERR_FAIL_COND_MSG(!window_get_flag(WINDOW_FLAG_BORDERLESS, p_window) && window_get_flag(WINDOW_FLAG_BORDERLESS, p_parent), "Toplevels can't be parented to a popup.");
	}

	if (wd.xdg_toplevel) {
		xdg_toplevel_set_parent(wd.xdg_toplevel, parent_toplevel);
	}
}

void DisplayServerWayland::window_set_min_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));

	DEBUG_LOG_WAYLAND(vformat("window id %d minsize set to %s", p_window, p_size));

	WindowData &wd = wls.windows[p_window];

	if (p_size.x < 0 || p_size.y < 0) {
		ERR_FAIL_MSG("Minimum window size can't be negative!");
	}

	if ((p_size != Size2i()) && (wd.max_size != Size2i()) && ((p_size.x > wd.max_size.x) || (p_size.y > wd.max_size.y))) {
		ERR_PRINT("Minimum window size can't be larger than maximum window size!");
		return;
	}

	wd.min_size = p_size;

	if (wd.wl_surface && wd.xdg_toplevel) {
		xdg_toplevel_set_min_size(wd.xdg_toplevel, p_size.width, p_size.height);
		wl_surface_commit(wd.wl_surface);
	}
}

Size2i DisplayServerWayland::window_get_min_size(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND_V(!wls.windows.has(p_window), Size2i());

	return wls.windows[p_window].min_size;
}

void DisplayServerWayland::window_set_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	wd.rect.size = p_size;

	if (wd.xdg_surface) {
		xdg_surface_set_window_geometry(wd.xdg_surface, 0, 0, wd.rect.size.width, wd.rect.size.height);
	}

#ifdef VULKAN_ENABLED
	if (wd.visible && wls.context_vulkan) {
		wls.context_vulkan->window_resize(p_window, wd.rect.size.width, wd.rect.size.height);
	}
#endif

#ifdef GLES3_ENABLED
	if (wd.visible && wls.gl_manager) {
		wls.gl_manager->window_resize(p_window, wd.rect.size.width, wd.rect.size.height);
	}
#endif
}

Size2i DisplayServerWayland::window_get_size(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND_V(!wls.windows.has(p_window), Size2i());

	return wls.windows[p_window].rect.size;
}

Size2i DisplayServerWayland::window_get_real_size(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND_V(!wls.windows.has(p_window), Size2i());

	// I don't think there's a way of actually knowing the window size in wayland,
	// other than the one requested by the compositor, which happens to be
	// the one the windows always uses anyway.
	return wls.windows[p_window].rect.size;
}

void DisplayServerWayland::window_set_mode(WindowMode p_mode, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));

	WindowData &wd = wls.windows[p_window];

	if (wd.mode == p_mode) {
		return;
	}

	_window_data_set_mode(wd, p_mode);
}

DisplayServer::WindowMode DisplayServerWayland::window_get_mode(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND_V(!wls.windows.has(p_window), WINDOW_MODE_WINDOWED);

	return wls.windows[p_window].mode;
}

bool DisplayServerWayland::window_is_maximize_allowed(DisplayServer::WindowID p_window) const {
	// TODO
	DEBUG_LOG_WAYLAND(vformat("wayland stub window_is_maximize_allowed window %d, returning false", p_window));
	return false;
}

void DisplayServerWayland::window_set_flag(WindowFlags p_flag, bool p_enabled, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	DEBUG_LOG_WAYLAND(vformat("Window %d set flag %d", p_window, p_flag));

	switch (p_flag) {
		case WINDOW_FLAG_BORDERLESS: {
			if (wls.globals.xdg_decoration_manager && wd.xdg_toplevel_decoration) {
				if (p_enabled) {
					zxdg_toplevel_decoration_v1_set_mode(wd.xdg_toplevel_decoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_CLIENT_SIDE);
				} else {
					zxdg_toplevel_decoration_v1_set_mode(wd.xdg_toplevel_decoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
				}
			}
		} break;

		default: {
		}
	}

	if (p_enabled) {
		wd.flags |= 1 << p_flag;
	} else {
		wd.flags &= ~(1 << p_flag);
	}
}

bool DisplayServerWayland::window_get_flag(WindowFlags p_flag, DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND_V(!wls.windows.has(p_window), false);

	return wls.windows[p_window].flags & (1 << p_flag);
}

void DisplayServerWayland::window_request_attention(DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_window));

	if (wls.globals.xdg_activation) {
		// Window attention requests are done through the XDG activation protocol.
		xdg_activation_token_v1 *xdg_activation_token = xdg_activation_v1_get_activation_token(wls.globals.xdg_activation);
		xdg_activation_token_v1_add_listener(xdg_activation_token, &xdg_activation_token_listener, &wls.windows[p_window]);
		xdg_activation_token_v1_commit(xdg_activation_token);

		DEBUG_LOG_WAYLAND(vformat("Requested activation token for window %d", p_window));
	}
}

void DisplayServerWayland::window_move_to_foreground(DisplayServer::WindowID p_window) {
	// Standard Wayland APIs don't support this.
}

bool DisplayServerWayland::window_can_draw(DisplayServer::WindowID p_window) const {
	// TODO: Implement this. For now a simple return true will work tough
	return true;
}

bool DisplayServerWayland::can_any_window_draw() const {
	// TODO: Implement this. For now a simple return true will work tough
	return true;
}

void DisplayServerWayland::window_set_ime_active(const bool p_active, DisplayServer::WindowID p_window) {
	// TODO
	DEBUG_LOG_WAYLAND(vformat("wayland stub window_set_ime_active active %s window %d", p_active ? "true" : "false", p_window));
}

void DisplayServerWayland::window_set_ime_position(const Point2i &p_pos, DisplayServer::WindowID p_window) {
	// TODO
	DEBUG_LOG_WAYLAND(vformat("wayland stub window_set_ime_position pos %s window %d", p_pos, p_window));
}

void DisplayServerWayland::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, DisplayServer::WindowID p_window) {
	// Right now Wayland doesn't support anything more than full VSync.
	// See: https://gitlab.freedesktop.org/wayland/wayland-protocols/-/merge_requests/65
}

DisplayServer::VSyncMode DisplayServerWayland::window_get_vsync_mode(DisplayServer::WindowID p_vsync_mode) const {
	// TODO: Figure out whether it is possible to disable VSync with Wayland
	// (doubt it) or handle any other mode.
	return VSYNC_ENABLED;
}

void DisplayServerWayland::cursor_set_shape(CursorShape p_shape) {
	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	MutexLock mutex_lock(wls.mutex);

	if (p_shape == wls.cursor_shape) {
		return;
	}

	wls.cursor_shape = p_shape;

	_wayland_state_update_cursor(wls);
}

DisplayServerWayland::CursorShape DisplayServerWayland::cursor_get_shape() const {
	MutexLock mutex_lock(wls.mutex);

	return wls.cursor_shape;
}

void DisplayServerWayland::cursor_set_custom_image(const Ref<Resource> &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	MutexLock mutex_lock(wls.mutex);

	if (p_cursor.is_valid()) {
		HashMap<CursorShape, CustomWaylandCursor>::Iterator cursor_c = wls.custom_cursors.find(p_shape);

		if (cursor_c) {
			if (cursor_c->value.cursor_rid == p_cursor->get_rid() && cursor_c->value.hotspot == p_hotspot) {
				cursor_set_shape(p_shape);
				return;
			}

			wls.custom_cursors.erase(p_shape);
		}

		Ref<Texture2D> texture = p_cursor;
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Ref<Image> image;
		Size2i texture_size;
		Rect2i atlas_rect;

		ERR_FAIL_COND(!texture.is_valid());

		image = texture->get_image();

		ERR_FAIL_COND(!image.is_valid());

		if (!image.is_valid() && atlas_texture.is_valid()) {
			texture = atlas_texture->get_atlas();

			atlas_rect.size.width = texture->get_width();
			atlas_rect.size.height = texture->get_height();
			atlas_rect.position.x = atlas_texture->get_region().position.x;
			atlas_rect.position.y = atlas_texture->get_region().position.y;

			texture_size.width = atlas_texture->get_region().size.x;
			texture_size.height = atlas_texture->get_region().size.y;
		} else if (image.is_valid()) {
			texture_size.width = texture->get_width();
			texture_size.height = texture->get_height();
		}

		ERR_FAIL_COND(!texture.is_valid());
		ERR_FAIL_COND(p_hotspot.x < 0 || p_hotspot.y < 0);
		// NOTE: The Wayland protocol says nothing about cursor size limits, yet if
		// the texture is larger than 256x256 it won't show at least on sway.
		ERR_FAIL_COND(texture_size.width > 256 || texture_size.height > 256);
		ERR_FAIL_COND(p_hotspot.x > texture_size.width || p_hotspot.y > texture_size.height);
		ERR_FAIL_COND(texture_size.height == 0 || texture_size.width == 0);

		// NOTE: The stride is the width of the image in bytes.
		unsigned int texture_stride = texture_size.width * 4;
		unsigned int data_size = texture_stride * texture_size.height;

		// We need a shared memory object file descriptor in order to create a
		// wl_buffer through wl_shm.
		int fd = _allocate_shm_file(data_size);
		ERR_FAIL_COND(fd == -1);

		CustomWaylandCursor &cursor = wls.custom_cursors[p_shape];
		cursor.cursor_rid = p_cursor->get_rid();
		cursor.hotspot = p_hotspot;

		if (cursor.buffer_data) {
			// Clean up the old buffer data.
			munmap(cursor.buffer_data, cursor.buffer_data_size);
		}

		cursor.buffer_data = (uint32_t *)mmap(NULL, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

		if (cursor.wl_buffer) {
			// Clean up the old Wayland buffer.
			wl_buffer_destroy(cursor.wl_buffer);
		}

		// Create the Wayland buffer.
		struct wl_shm_pool *wl_shm_pool = wl_shm_create_pool(wls.globals.wl_shm, fd, texture_size.height * data_size);
		// TODO: Make sure that WL_SHM_FORMAT_ARGB8888 format is supported. It
		// technically isn't garaunteed to be supported, but I think that'd be a
		// pretty unlikely thing to stumble upon.
		cursor.wl_buffer = wl_shm_pool_create_buffer(wl_shm_pool, 0, texture_size.width, texture_size.height, texture_stride, WL_SHM_FORMAT_ARGB8888);
		wl_shm_pool_destroy(wl_shm_pool);

		// Fill the cursor buffer with the texture data.
		for (unsigned int index = 0; index < (unsigned int)(texture_size.width * texture_size.height); index++) {
			int row_index = floor(index / texture_size.width) + atlas_rect.position.y;
			int column_index = (index % int(texture_size.width)) + atlas_rect.position.x;

			if (atlas_texture.is_valid()) {
				column_index = MIN(column_index, atlas_rect.size.width - 1);
				row_index = MIN(row_index, atlas_rect.size.height - 1);
			}

			cursor.buffer_data[index] = image->get_pixel(column_index, row_index).to_argb32();
		}
	} else {
		// Reset to default system cursor.
		if (wls.custom_cursors.has(p_shape)) {
			wls.custom_cursors.erase(p_shape);
		}
	}

	_wayland_state_update_cursor(wls);
}

int DisplayServerWayland::keyboard_get_layout_count() const {
	MutexLock mutex_lock(wls.mutex);

	if (wls.current_seat && wls.current_seat->xkb_keymap) {
		return xkb_keymap_num_layouts(wls.current_seat->xkb_keymap);
	}

	return 0;
}

int DisplayServerWayland::keyboard_get_current_layout() const {
	MutexLock mutex_lock(wls.mutex);

	if (wls.current_seat) {
		return wls.current_seat->current_layout_index;
	}

	return 0;
}

void DisplayServerWayland::keyboard_set_current_layout(int p_index) {
	MutexLock mutex_lock(wls.mutex);

	if (wls.current_seat) {
		wls.current_seat->current_layout_index = p_index;
	}
}

String DisplayServerWayland::keyboard_get_layout_language(int p_index) const {
	// xkbcommon exposes only the layout's name, which looks like it overlaps with
	// its language.
	return keyboard_get_layout_name(p_index);
}

String DisplayServerWayland::keyboard_get_layout_name(int p_index) const {
	MutexLock mutex_lock(wls.mutex);

	String ret;

	if (wls.current_seat && wls.current_seat->xkb_keymap) {
		ret.parse_utf8(xkb_keymap_layout_get_name(wls.current_seat->xkb_keymap, p_index));
	}

	return ret;
}

Key DisplayServerWayland::keyboard_get_keycode_from_physical(Key p_keycode) const {
	MutexLock mutex_lock(wls.mutex);

	xkb_keycode_t xkb_keycode = KeyMappingXKB::get_xkb_keycode(p_keycode);

	Key key = Key::NONE;

	if (wls.current_seat && wls.current_seat->xkb_state) {
		// NOTE: Be aware that this method will always return something, even if this
		// line might never be executed if the current seat doesn't have a keyboard.
		key = KeyMappingXKB::get_keycode(xkb_state_key_get_one_sym(wls.current_seat->xkb_state, xkb_keycode));
	}

	// If not found, fallback to QWERTY.
	// This should match the behavior of the event pump.
	if (key == Key::NONE) {
		return p_keycode;
	}

	if (key >= Key::A + 32 && key <= Key::Z + 32) {
		key -= 'a' - 'A';
	}

	// Make it consistent with the keys returned by `Input`.
	if (key == Key::BACKTAB) {
		key = Key::TAB;
	}

	return key;
}

void DisplayServerWayland::process_events() {
	MutexLock mutex_lock(wls.mutex);

	int werror = wl_display_get_error(wls.wl_display);

	if (werror) {
		if (werror == EPROTO) {
			struct wl_interface *wl_interface = nullptr;
			uint32_t id = 0;

			int error_code = wl_display_get_protocol_error(wls.wl_display, (const struct wl_interface **)&wl_interface, &id);
			print_error(vformat("Wayland protocol error %d on interface %s@%d.", error_code, wl_interface ? wl_interface->name : "unknown", id));
		} else {
			print_error(vformat("Wayland client error code %d.", werror));
		}
	}

	while (wls.message_queue.front()) {
		Ref<WaylandMessage> msg = wls.message_queue.front()->get();

		Ref<WaylandWindowRectMessage> winrect_msg = msg;

		if (winrect_msg.is_valid() && wls.windows.has(winrect_msg->id)) {
			WindowID id = winrect_msg->id;
			Rect2i rect = winrect_msg->rect;
			WindowData &wd = wls.windows[id];

			if (wd.visible && wd.rect == rect) {
				// Resizing is very costly, do it only if this is the last rect update.
#ifdef VULKAN_ENABLED
				if (wls.context_vulkan) {
					wls.context_vulkan->window_resize(id, rect.size.width, rect.size.height);
				}
#endif

#ifdef GLES3_ENABLED
				if (wls.gl_manager) {
					wls.gl_manager->window_resize(id, rect.size.width, rect.size.height);
				}
#endif
			}

			if (wd.rect_changed_callback.is_valid()) {
				Variant var_rect = Variant(rect);
				Variant *arg = &var_rect;

				Variant ret;
				Callable::CallError ce;

				wd.rect_changed_callback.callp((const Variant **)&arg, 1, ret, ce);
			}
		}

		Ref<WaylandWindowEventMessage> winev_msg = msg;

		if (winev_msg.is_valid() && wls.windows.has(winev_msg->id)) {
			_send_window_event(winev_msg->id, winev_msg->event);
		}

		Ref<WaylandInputEventMessage> inputev_msg = msg;

		if (inputev_msg.is_valid()) {
			Input::get_singleton()->parse_input_event(inputev_msg->event);
		}

		Ref<WaylandDropFilesEventMessage> dropfiles_msg = msg;

		if (dropfiles_msg.is_valid() && wls.windows.has(dropfiles_msg->id)) {
			WindowData wd = wls.windows[dropfiles_msg->id];

			if (wd.drop_files_callback.is_valid()) {
				Variant var_files = dropfiles_msg->files;
				Variant *arg = &var_files;

				Variant ret;
				Callable::CallError ce;

				wd.drop_files_callback.callp((const Variant **)&arg, 1, ret, ce);
			}
		}

		wls.message_queue.pop_front();
	}

	if (!wls.current_seat) {
		return;
	}

	SeatState &seat = *wls.current_seat;

	if (seat.repeat_key_delay_msec && seat.repeating_keycode != XKB_KEYCODE_INVALID) {
		uint64_t current_ticks = OS::get_singleton()->get_ticks_msec();
		uint64_t delayed_start_ticks = seat.last_repeat_start_msec + seat.repeat_start_delay_msec;

		if (seat.last_repeat_msec < delayed_start_ticks) {
			seat.last_repeat_msec = delayed_start_ticks;
		}

		if (current_ticks >= delayed_start_ticks) {
			uint64_t ticks_delta = current_ticks - seat.last_repeat_msec;

			int keys_amount = (ticks_delta / seat.repeat_key_delay_msec);

			for (int i = 0; i < keys_amount; i++) {
				Ref<InputEventKey> k;
				k.instantiate();

				if (!_seat_state_configure_key_event(seat, k, seat.repeating_keycode, true)) {
					continue;
				}

				k->set_echo(true);

				Input::get_singleton()->parse_input_event(k);
			}

			seat.last_repeat_msec += ticks_delta - (ticks_delta % seat.repeat_key_delay_msec);
		}
	}

	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerWayland::release_rendering_thread() {
#if defined(GLES3_ENABLED)
	if (wls.gl_manager) {
		wls.gl_manager->release_current();
	}
#endif
}

void DisplayServerWayland::make_rendering_thread() {
#if defined(GLES3_ENABLED)
	if (wls.gl_manager) {
		wls.gl_manager->make_current();
	}
#endif
}

void DisplayServerWayland::swap_buffers() {
#if defined(GLES3_ENABLED)
	if (wls.gl_manager) {
		wls.gl_manager->swap_buffers();
	}
#endif
}

Vector<String> DisplayServerWayland::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif

#ifdef GLES3_ENABLED
	drivers.push_back("opengl3");
#endif

	return drivers;
}

DisplayServer *DisplayServerWayland::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerWayland(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
	if (r_error != OK) {
		ERR_PRINT("Can't create the Wayland display server.");

		memdelete(ds);
	}
	return ds;
}

DisplayServerWayland::DisplayServerWayland(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	r_error = ERR_UNAVAILABLE;

	wls.wl_display = wl_display_connect(nullptr);

	ERR_FAIL_COND_MSG(!wls.wl_display, "Can't connect to a Wayland display.");

	events_thread.start(_poll_events_thread, &wls);

	wls.wl_registry = wl_display_get_registry(wls.wl_display);

	ERR_FAIL_COND_MSG(!wls.wl_registry, "Can't obtain the Wayland registry global.");

	wl_registry_add_listener(wls.wl_registry, &wl_registry_listener, &wls);

	// Wait for globals to get notified from the compositor.
	wl_display_roundtrip(wls.wl_display);

	// TODO: Perhaps gracefully handle missing protocols when possible?
	ERR_FAIL_COND_MSG(!wls.globals.wl_shm, "Can't obtain the Wayland shared memory global.");
	ERR_FAIL_COND_MSG(!wls.globals.wl_compositor, "Can't obtain the Wayland compositor global.");
	ERR_FAIL_COND_MSG(!wls.globals.wl_subcompositor, "Can't obtain the Wayland subcompositor global.");
	ERR_FAIL_COND_MSG(!wls.globals.wl_data_device_manager, "Can't obtain the Wayland data device manager global.");
	ERR_FAIL_COND_MSG(!wls.globals.wp_pointer_constraints, "Can't obtain the Wayland pointer constraints global.");
	ERR_FAIL_COND_MSG(!wls.globals.xdg_wm_base, "Can't obtain the Wayland XDG shell global.");

	if (!wls.globals.xdg_decoration_manager) {
		WARN_PRINT("Can't obtain the XDG decoration manager. Probably this compositor handles only CSDs, which aren't supported yet!");
	}

	if (!wls.globals.xdg_activation) {
		WARN_PRINT("Can't obtain the XDG activation global. Attention requesting won't work!");
	}

#ifndef DBUS_ENABLED
	if (!wls.globals.wp_idle_inhibit_manager) {
		WARN_PRINT("Can't obtain the idle inhibition manager. The screen might turn off even after calling screen_set_keep_on()!");
	}
#endif

	// Input.
	Input::get_singleton()->set_event_dispatch_function(dispatch_input_events);

	// Wait for seat capabilities.
	wl_display_roundtrip(wls.wl_display);

	xdg_wm_base_add_listener(wls.globals.xdg_wm_base, &xdg_wm_base_listener, nullptr);

#ifdef SPEECHD_ENABLED
	// Init TTS
	tts = memnew(TTS_Linux);
#endif

#if defined(VULKAN_ENABLED)
	if (p_rendering_driver == "vulkan") {
		wls.context_vulkan = memnew(VulkanContextWayland);

		if (wls.context_vulkan->initialize() != OK) {
			memdelete(wls.context_vulkan);
			wls.context_vulkan = nullptr;
			r_error = ERR_CANT_CREATE;
			ERR_FAIL_MSG("Could not initialize Vulkan.");
		}
	}
#endif

#if defined(GLES3_ENABLED)
	if (p_rendering_driver == "opengl3") {
		wls.gl_manager = memnew(GLManagerWayland);

		if (wls.gl_manager->initialize() != OK) {
			memdelete(wls.gl_manager);
			wls.gl_manager = nullptr;
			r_error = ERR_CANT_CREATE;
			ERR_FAIL_MSG("Could not initialize GLES3.");
		}

		RasterizerGLES3::make_current();
	}
#endif

	int64_t cursor_size = OS::get_singleton()->get_environment("XCURSOR_SIZE").to_int();
	if (cursor_size <= 0) {
		print_verbose("Detected invalid cursor size preference, defaulting to 24.");
		cursor_size = 24;
	}

	print_verbose(vformat("Loading default cursor theme size %d.", cursor_size));
	wls.wl_cursor_theme = wl_cursor_theme_load(nullptr, cursor_size, wls.globals.wl_shm);

	ERR_FAIL_NULL_MSG(wls.wl_cursor_theme, "Can't find a cursor theme.");

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

	for (int i = 0; i < CURSOR_MAX; i++) {
		struct wl_cursor *cursor = wl_cursor_theme_get_cursor(wls.wl_cursor_theme, cursor_names[i]);

		if (!cursor && cursor_names_fallback[i]) {
			cursor = wl_cursor_theme_get_cursor(wls.wl_cursor_theme, cursor_names[i]);
		}

		if (cursor && cursor->image_count > 0) {
			wls.cursor_images[i] = cursor->images[0];
			wls.cursor_bufs[i] = wl_cursor_image_get_buffer(cursor->images[0]);
		} else {
			wls.cursor_images[i] = nullptr;
			wls.cursor_bufs[i] = nullptr;
			print_verbose("Failed loading cursor: " + String(cursor_names[i]));
		}
	}

	cursor_set_shape(CURSOR_BUSY);

	WindowID main_window_id = _create_window(p_mode, p_vsync_mode, p_flags, Rect2i(Point2i(0, 0), p_resolution));
	_show_window(main_window_id);

#ifdef VULKAN_ENABLED
	if (p_rendering_driver == "vulkan") {
		wls.rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		wls.rendering_device_vulkan->initialize(wls.context_vulkan);

		RendererCompositorRD::make_current();
	}
#endif

#ifdef DBUS_ENABLED
	portal_desktop = memnew(FreeDesktopPortalDesktop);
	screensaver = memnew(FreeDesktopScreenSaver);
#endif

	r_error = OK;
}

DisplayServerWayland::~DisplayServerWayland() {
	if (wls.wl_display && events_thread.is_started()) {
		wls.events_thread_done.set();

		// Wait for any Wayland event to be handled and unblock the polling thread.
		wl_display_roundtrip(wls.wl_display);

		events_thread.wait_to_finish();
	}

	// Destroy all windows.
	for (KeyValue<WindowID, WindowData> &E : wls.windows) {
		WindowID id = E.key;
		WindowData &wd = E.value;

#ifdef VULKAN_ENABLED
		if (wls.context_vulkan && wd.visible) {
			wls.context_vulkan->window_destroy(id);
		}
#endif

#ifdef GLES3_ENABLED
		if (wls.gl_manager && wd.visible) {
			wls.gl_manager->window_destroy(id);
		}
#endif
		if (wd.xdg_toplevel) {
			xdg_toplevel_destroy(wd.xdg_toplevel);
		}

		if (wd.xdg_surface) {
			xdg_surface_destroy(wd.xdg_surface);
		}

		if (wd.wl_surface) {
			wl_surface_destroy(wd.wl_surface);
		}
	}

	for (SeatState &seat : wls.seats) {
		wl_seat_destroy(seat.wl_seat);

		xkb_context_unref(seat.xkb_context);
		xkb_state_unref(seat.xkb_state);
		xkb_keymap_unref(seat.xkb_keymap);

		if (seat.wl_keyboard) {
			wl_keyboard_destroy(seat.wl_keyboard);
		}

		if (seat.keymap_buffer) {
			munmap((void *)seat.keymap_buffer, seat.keymap_buffer_size);
		}

		if (seat.wl_pointer) {
			wl_pointer_destroy(seat.wl_pointer);
		}

		if (seat.cursor_surface) {
			wl_surface_destroy(seat.cursor_surface);
		}

		if (seat.wl_data_device) {
			wl_data_device_destroy(seat.wl_data_device);
		}

		if (seat.wp_relative_pointer) {
			zwp_relative_pointer_v1_destroy(seat.wp_relative_pointer);
		}

		if (seat.wp_locked_pointer) {
			zwp_locked_pointer_v1_destroy(seat.wp_locked_pointer);
		}

		if (seat.wp_confined_pointer) {
			zwp_confined_pointer_v1_destroy(seat.wp_confined_pointer);
		}
	}

	for (ScreenData &screen : wls.screens) {
		if (screen.wl_output) {
			wl_output_destroy(screen.wl_output);
		}
	}

	if (wls.wl_cursor_theme) {
		wl_cursor_theme_destroy(wls.wl_cursor_theme);
	}

	if (wls.globals.wp_idle_inhibit_manager) {
		zwp_idle_inhibit_manager_v1_destroy(wls.globals.wp_idle_inhibit_manager);
	}

	if (wls.globals.wp_pointer_constraints) {
		zwp_pointer_constraints_v1_destroy(wls.globals.wp_pointer_constraints);
	}

	if (wls.globals.wp_relative_pointer_manager) {
		zwp_relative_pointer_manager_v1_destroy(wls.globals.wp_relative_pointer_manager);
	}

	if (wls.globals.xdg_activation) {
		xdg_activation_v1_destroy(wls.globals.xdg_activation);
	}

	if (wls.globals.xdg_decoration_manager) {
		zxdg_decoration_manager_v1_destroy(wls.globals.xdg_decoration_manager);
	}

	if (wls.globals.xdg_wm_base) {
		xdg_wm_base_destroy(wls.globals.xdg_wm_base);
	}

	if (wls.globals.wl_shm) {
		wl_shm_destroy(wls.globals.wl_shm);
	}

	if (wls.globals.wl_subcompositor) {
		wl_subcompositor_destroy(wls.globals.wl_subcompositor);
	}

	if (wls.globals.wl_compositor) {
		wl_compositor_destroy(wls.globals.wl_compositor);
	}

	if (wls.wl_registry) {
		wl_registry_destroy(wls.wl_registry);
	}

	if (wls.wl_display) {
		wl_display_disconnect(wls.wl_display);
	}

	// Destroy all drivers.
#ifdef VULKAN_ENABLED
	if (wls.rendering_device_vulkan) {
		wls.rendering_device_vulkan->finalize();
		memdelete(wls.rendering_device_vulkan);
	}

	if (wls.context_vulkan) {
		memdelete(wls.context_vulkan);
	}
#endif

#ifdef SPEECHD_ENABLED
	if (tts) {
		memdelete(tts);
	}
#endif

#ifdef DBUS_ENABLED
	if (portal_desktop) {
		memdelete(portal_desktop);
		memdelete(screensaver);
	}
#endif
}

void DisplayServerWayland::register_wayland_driver() {
	register_create_function("wayland", create_func, get_rendering_drivers_func);
}

#endif //WAYLAND_ENABLED
