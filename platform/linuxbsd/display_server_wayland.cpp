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

// Implementation specific methods.

void DisplayServerWayland::_poll_events_thread(void *p_wls) {
	WaylandState *wls = (WaylandState *)p_wls;

	struct pollfd poll_fd;
	poll_fd.fd = wl_display_get_fd(wls->display);
	poll_fd.events = POLLIN | POLLHUP;

	while (true) {
		// Empty the event queue while it's full.
		while (wl_display_prepare_read(wls->display) != 0) {
			MutexLock mutex_lock(wls->mutex);
			wl_display_dispatch_pending(wls->display);
		}

		errno = 0;
		if (wl_display_flush(wls->display) == -1) {
			if (errno != EAGAIN) {
				print_error(vformat("Error %d while flushing the Wayland display.", errno));
				wls->events_thread_done.set();
			}
		}

		// Wait for the event file descriptor to have new data.
		poll(&poll_fd, 1, -1);

		if (wls->events_thread_done.is_set()) {
			wl_display_cancel_read(wls->display);
			break;
		}

		if (poll_fd.revents | POLLIN) {
			wl_display_read_events(wls->display);
		} else {
			wl_display_cancel_read(wls->display);
		}
	}
}

String DisplayServerWayland::read_data_control_offer(zwlr_data_control_offer_v1 *wlr_data_control_offer) const {
	if (!wlr_data_control_offer) {
		// The clipboard's empty, return an empty string.
		return "";
	}

	int fds[2];
	if (pipe(fds) == 0) {
		// This function expects to return a string, so we can only ask for a MIME of
		// "text/plain"
		zwlr_data_control_offer_v1_receive(wlr_data_control_offer, "text/plain", fds[1]);

		// Wait for the compositor to know about the pipe.
		wl_display_roundtrip(wls.display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		// This is pretty much an arbitrary size.
		uint32_t chunk_size = 2048;

		LocalVector<uint8_t> data;
		data.resize(chunk_size);

		uint32_t bytes_read = 0;

		while (true) {
			ssize_t last_bytes_read = read(fds[0], data.ptr() + bytes_read, chunk_size);
			if (last_bytes_read < 0) {
				ERR_PRINT(vformat("Clipboard: read error %d.", errno));
			}

			if (last_bytes_read == 0) {
				// We're done, we've reached the EOF.
				print_verbose(vformat("Clipboard: done reading %d bytes.", bytes_read));
				close(fds[0]);
				break;
			}

			print_verbose(vformat("Clipboard: read chunk of %d bytes.", last_bytes_read));

			bytes_read += last_bytes_read;

			// Increase the buffer size by one chunk in preparation of the next read.
			data.resize(bytes_read + chunk_size);
		}

		String ret;
		ret.parse_utf8((const char *)data.ptr(), bytes_read);
		return ret;
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

	struct wl_cursor_image *cursor_image = p_ss.wls->cursor_images[p_shape];

	if (!cursor_image) {
		return;
	}

	// Update the cursor's hotspot.
	wl_pointer_set_cursor(p_ss.wl_pointer, 0, p_ss.cursor_surface, cursor_image->hotspot_x, cursor_image->hotspot_y);

	// Attach the new cursor's buffer and damage it.
	wl_surface_attach(p_ss.cursor_surface, p_ss.wls->cursor_bufs[p_shape], 0, 0);
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

void DisplayServerWayland::dispatch_input_events(const Ref<InputEvent> &p_event) {
	((DisplayServerWayland *)(get_singleton()))->_dispatch_input_event(p_event);
}

void DisplayServerWayland::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	Variant ev = p_event;
	Variant *evp = &ev;
	Variant ret;
	Callable::CallError ce;

	Ref<InputEventFromWindow> event_from_window = p_event;
	if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
		// Send to a single window.
		if (wls.windows.has(event_from_window->get_window_id())) {
			Callable callable = wls.windows[event_from_window->get_window_id()].input_event_callback;
			if (callable.is_valid()) {
				callable.call((const Variant **)&evp, 1, ret, ce);
			}
		}
	} else {
		// Send to all windows.
		for (KeyValue<WindowID, WindowData> &E : wls.windows) {
			Callable callable = E.value.input_event_callback;
			if (callable.is_valid()) {
				callable.call((const Variant **)&evp, 1, ret, ce);
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

DisplayServerWayland::WindowID DisplayServerWayland::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	MutexLock mutex_lock(wls.mutex);

	WindowID id = wls.window_id_counter++;

	WindowData &wd = wls.windows[id];

	wd.id = id;
	wd.wls = &wls;

	wd.flags = p_flags;
	wd.vsync_mode = p_vsync_mode;
	wd.rect = p_rect;

	wd.title = "Godot";

	return id;
}

void DisplayServerWayland::_send_window_event(WindowID p_window, WindowEvent p_event) {
	ERR_FAIL_COND(!wls.windows.has(p_window));
	WindowData &wd = wls.windows[p_window];

	if (wd.window_event_callback.is_valid()) {
		Variant var_event = Variant(p_event);
		Variant *arg = &var_event;

		Variant ret;
		Callable::CallError ce;

		wd.window_event_callback.call((const Variant **)&arg, 1, ret, ce);
	}
}

void DisplayServerWayland::_delete_window(WindowID p_window) {
	WindowData &wd = wls.windows[p_window];

	if (window_get_flag(WINDOW_FLAG_POPUP, p_window)) {
		print_verbose(vformat("Deleting popup %d.", p_window));
	} else {
		print_verbose(vformat("Deleting window %d.", p_window));
	}

	while (wd.children.size()) {
		// Unparent all children of the window.
		window_set_transient(wd.children.front()->get(), INVALID_WINDOW_ID);
	}

	if (wd.parent != INVALID_WINDOW_ID) {
		window_set_transient(p_window, INVALID_WINDOW_ID);
	}

#ifdef VULKAN_ENABLED
	if (wls.context_vulkan && wd.visible) {
		wls.context_vulkan->window_destroy(p_window);
		wd.visible = false;
	}
#endif

	if (wd.xdg_popup) {
		xdg_popup_destroy(wd.xdg_popup);
		wd.xdg_popup = nullptr;
	}

	if (wd.xdg_toplevel) {
		xdg_toplevel_destroy(wd.xdg_toplevel);
		wd.xdg_toplevel = nullptr;
	}

	if (wd.xdg_surface) {
		xdg_surface_destroy(wd.xdg_surface);
		wd.xdg_surface = nullptr;
	}

	if (wd.wl_surface) {
		wl_surface_destroy(wd.wl_surface);
		wd.wl_surface = nullptr;
	}

	wls.windows.erase(p_window);
}

void DisplayServerWayland::_wl_registry_on_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version) {
	WaylandState *wls = (WaylandState *)data;

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

	if (strcmp(interface, wl_output_interface.name) == 0) {
		ScreenData *sd = memnew(ScreenData);
		sd->wl_output = (struct wl_output *)wl_registry_bind(wl_registry, name, &wl_output_interface, 3);
		sd->wl_output_name = name;

		wls->screens.push_back(sd);

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
		ss->wl_seat = (struct wl_seat *)wl_registry_bind(wl_registry, name, &wl_seat_interface, 7);
		ss->wl_seat_name = name;

		wl_seat_add_listener(ss->wl_seat, &wl_seat_listener, ss);
		return;
	}

	if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		globals.xdg_wm_base = (struct xdg_wm_base *)wl_registry_bind(wl_registry, name, &xdg_wm_base_interface, 2);
		globals.xdg_wm_base_name = name;
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

	if (strcmp(interface, zwlr_data_control_manager_v1_interface.name) == 0) {
		globals.wlr_data_control_manager = (struct zwlr_data_control_manager_v1 *)wl_registry_bind(wl_registry, name, &zwlr_data_control_manager_v1_interface, 2);
		globals.wlr_data_control_manager_name = name;
		return;
	}
}

void DisplayServerWayland::_wl_registry_on_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name) {
	WaylandState *wls = (WaylandState *)data;

	WaylandGlobals &globals = wls->globals;

	if (name == globals.wl_shm_name) {
		wl_shm_destroy(globals.wl_shm);
		globals.wl_shm = nullptr;
		return;
	}

	if (name == globals.wl_compositor_name) {
		wl_compositor_destroy(globals.wl_compositor);
		globals.wl_compositor = nullptr;
		return;
	}

	if (name == globals.xdg_wm_base_name) {
		xdg_wm_base_destroy(globals.xdg_wm_base);
		globals.xdg_wm_base = nullptr;
		return;
	}

	if (name == globals.wp_pointer_constraints_name) {
		zwp_pointer_constraints_v1_destroy(globals.wp_pointer_constraints);
		globals.wp_pointer_constraints = nullptr;
		return;
	}

	if (name == globals.wp_relative_pointer_manager_name) {
		zwp_relative_pointer_manager_v1_destroy(globals.wp_relative_pointer_manager);
		globals.wp_relative_pointer_manager = nullptr;
		return;
	}

	if (name == globals.wlr_data_control_manager_name) {
		zwlr_data_control_manager_v1_destroy(globals.wlr_data_control_manager);
		globals.wlr_data_control_manager = nullptr;
	}

	// FIXME: This is a very bruteforce approach.
	for (int i = 0; i < (int)wls->screens.size(); i++) {
		// Iterate through all of the screens to find if any got removed.
		ScreenData *sd = wls->screens[i];

		if (sd->wl_output_name == name) {
			wl_output_destroy(sd->wl_output);
			memfree(wls->screens[i]);
			wls->screens.remove_at(i);
			return;
		}
	}

	List<SeatState>::Element *it = wls->seats.front();
	while (it) {
		// Iterate through all of the seats to find if any got removed.
		SeatState &ss = it->get();

		if (ss.wl_seat_name == name) {
			wl_seat_destroy(ss.wl_seat);
			wls->seats.erase(it);
			return;
		}

		it = it->next();
	}
}

void DisplayServerWayland::_wl_surface_on_enter(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
}

void DisplayServerWayland::_wl_surface_on_leave(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
}

void DisplayServerWayland::_wl_output_on_geometry(void *data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char *make, const char *model, int32_t transform) {
	ScreenData *sd = (ScreenData *)data;

	sd->position.x = x;
	sd->position.y = y;

	sd->physical_size.width = physical_width;
	sd->physical_size.height = physical_height;

	sd->make.parse_utf8(make);
	sd->model.parse_utf8(model);
}

void DisplayServerWayland::_wl_output_on_mode(void *data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {
	ScreenData *sd = (ScreenData *)data;

	sd->size.width = width;
	sd->size.height = height;

	sd->refresh_rate = refresh ? refresh / 1000.0f : -1;
}

void DisplayServerWayland::_wl_output_on_done(void *data, struct wl_output *wl_output) {
}

void DisplayServerWayland::_wl_output_on_scale(void *data, struct wl_output *wl_output, int32_t factor) {
	ScreenData *sd = (ScreenData *)data;

	sd->scale = factor;
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
		// DEBUG
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
			free(ss->xkb_context);
			ss->xkb_context = nullptr;
		}

		if (ss->wl_keyboard) {
			wl_keyboard_destroy(ss->wl_keyboard);
			ss->wl_keyboard = nullptr;
		}
	}
}

void DisplayServerWayland::_wl_seat_on_name(void *data, struct wl_seat *wl_seat, const char *name) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	ss->wlr_data_control_device = zwlr_data_control_manager_v1_get_data_device(wls->globals.wlr_data_control_manager, ss->wl_seat);
	zwlr_data_control_device_v1_add_listener(ss->wlr_data_control_device, &wlr_data_control_device_listener, ss);
}

void DisplayServerWayland::_wl_pointer_on_enter(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	PointerData &pd = ss->pointer_data_buffer;
	struct wl_cursor_image *cursor_image = wls->cursor_images[wls->cursor_shape];

	wl_pointer_set_cursor(ss->wl_pointer, serial, ss->cursor_surface, cursor_image->hotspot_x, cursor_image->hotspot_y);

	wl_surface_commit(ss->cursor_surface);

	pd.pointed_window_id = INVALID_WINDOW_ID;

	for (KeyValue<WindowID, WindowData> &E : wls->windows) {
		WindowData &wd = E.value;

		if (wd.wl_surface == surface) {
			pd.pointed_window_id = E.key;
			break;
		}
	}

	ERR_FAIL_COND_MSG(pd.pointed_window_id == INVALID_WINDOW_ID, "Cursor focused to an invalid window.");

	print_verbose(vformat("Pointing window %d", pd.pointed_window_id));

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

	print_verbose("Left a surface");
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

		// TODO: Handle more buttons.
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
			// FIXME: We're lying! With Wayland we can only know the position of the
			// mouse in our windows and nowhere else!
			mm->set_global_position(pd.position);

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
						 MouseButton::WHEEL_RIGHT }) {
				MouseButton test_button_mask = mouse_button_to_mask(test_button);
				if ((pressed_mask_delta & test_button_mask) != MouseButton::NONE) {
					Ref<InputEventMouseButton> mb;
					mb.instantiate();

					// Set all pressed modifiers.
					_get_key_modifier_state(*ss, mb);

					mb->set_window_id(pd.pointed_window_id);
					mb->set_position(pd.position);
					// FIXME: We're lying!
					mb->set_global_position(pd.position);
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
						// FIXME: We're lying!
						wh_up->set_global_position(pd.position);

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

void DisplayServerWayland::_xdg_wm_base_on_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}

// wlr-protocols event handlers.

void DisplayServerWayland::_wlr_data_control_device_on_data_offer(void *data, struct zwlr_data_control_device_v1 *wlr_data_control_device, struct zwlr_data_control_offer_v1 *id) {
	// This method is purposely left unimplemented as we don't care about the
	// offered MIME type, as we only want `text/plain` data.

	// TODO: Perhaps we could try to detect other text types such as `TEXT`?
}

void DisplayServerWayland::_wlr_data_control_device_on_selection(void *data, struct zwlr_data_control_device_v1 *wlr_data_control_device, struct zwlr_data_control_offer_v1 *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->selection_data_control_offer = id;
}

void DisplayServerWayland::_wlr_data_control_device_on_finished(void *data, struct zwlr_data_control_device_v1 *wlr_data_control_device) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->wlr_data_control_device = nullptr;
}

void DisplayServerWayland::_wlr_data_control_device_on_primary_selection(void *data, struct zwlr_data_control_device_v1 *wlr_data_control_device, struct zwlr_data_control_offer_v1 *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->primary_data_control_offer = id;
}

void DisplayServerWayland::_wlr_data_control_source_on_send(void *data, struct zwlr_data_control_source_v1 *wlr_data_control_source, const char *mime_type, int32_t fd) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	Vector<uint8_t> *data_to_send = nullptr;

	if (wlr_data_control_source == ss->selection_data_control_source) {
		data_to_send = &ss->selection_data;
		print_verbose("Clipboard: requested selection.");
	} else if (wlr_data_control_source == ss->primary_data_control_source) {
		data_to_send = &ss->primary_data;
		print_verbose("Clipboard: requested primary selection.");
	} else {
		return;
	}

	if (data_to_send) {
		ssize_t written_bytes = 0;

		if (strcmp(mime_type, "text/plain") == 0) {
			written_bytes = write(fd, data_to_send->ptr(), data_to_send->size());
		}

		if (written_bytes > 0) {
			print_verbose(vformat("Clipboard: sent %d bytes.", written_bytes));
		} else if (written_bytes == 0) {
			print_verbose("Clipboard: no bytes sent.");
		} else {
			ERR_PRINT(vformat("Clipboard: write error %d.", errno));
		}
	}

	close(fd);
}

void DisplayServerWayland::_wlr_data_control_source_on_cancelled(void *data, struct zwlr_data_control_source_v1 *wlr_data_control_source) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (wlr_data_control_source == ss->selection_data_control_source) {
		zwlr_data_control_source_v1_destroy(ss->selection_data_control_source);
		ss->selection_data_control_source = nullptr;

		ss->selection_data.clear();

		print_verbose("Clipboard: selection set by another program.");
		return;
	}

	if (wlr_data_control_source == ss->primary_data_control_source) {
		zwlr_data_control_source_v1_destroy(ss->primary_data_control_source);
		ss->primary_data_control_source = nullptr;

		ss->primary_data.clear();

		print_verbose("Clipboard: primary selection set by another program.");
		return;
	}
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
}

void DisplayServerWayland::_xdg_toplevel_on_configure(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states) {
	WindowData *wd = (WindowData *)data;
	ERR_FAIL_NULL(wd);

	if (width != 0 && height != 0) {
		wd->rect.size.width = width;
		wd->rect.size.height = height;
	}
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

void DisplayServerWayland::_xdg_popup_on_configure(void *data, struct xdg_popup *xdg_popup, int32_t x, int32_t y, int32_t width, int32_t height) {
	WindowData *wd = (WindowData *)data;
	ERR_FAIL_NULL(wd);

	wd->rect.position.x = x;
	wd->rect.position.y = y;
	wd->rect.size.width = width;
	wd->rect.size.height = height;

	// DEBUG
	print_verbose("xdg popup on configure");
}

void DisplayServerWayland::_xdg_popup_on_popup_done(void *data, struct xdg_popup *xdg_popup) {
	WindowData *wd = (WindowData *)data;
	ERR_FAIL_NULL(wd);

	WaylandState *wls = (WaylandState *)wd->wls;
	ERR_FAIL_NULL(wls);

	Ref<WaylandWindowEventMessage> msg;
	msg.instantiate();
	msg->id = wd->id;
	msg->event = WINDOW_EVENT_CLOSE_REQUEST;
	wls->message_queue.push_back(msg);

	// DEBUG
	print_verbose(vformat("Window %d xdg popup on popup done", wd->id));
}

void DisplayServerWayland::_xdg_popup_on_repositioned(void *data, struct xdg_popup *xdg_popup, uint32_t token) {
	// DEBUG
	print_verbose("xdg popup on repositioned");
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

// Interface mthods

bool DisplayServerWayland::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_MOUSE:
		case FEATURE_SUBWINDOWS:
			return true;
		default: {
		}
	}

	return false;
}

String DisplayServerWayland::get_name() const {
	return "Wayland";
}

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
	// TODO
	print_verbose("wayland stub warp_mouse");
}

Point2i DisplayServerWayland::mouse_get_position() const {
	MutexLock mutex_lock(wls.mutex);

	Point2i position;

	if (wls.current_seat) {
		position = wls.current_seat->pointer_data.position;

		WindowID window_id = wls.current_seat->pointer_data.pointed_window_id;

		while (window_id != INVALID_WINDOW_ID) {
			position += wls.windows[window_id].rect.position;
			window_id = wls.windows[window_id].parent;
		}
	}

	return position;
}

MouseButton DisplayServerWayland::mouse_get_button_state() const {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return MouseButton::NONE;
	}

	return wls.current_seat->pointer_data.pressed_button_mask;
}

void DisplayServerWayland::clipboard_set(const String &p_text) {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return;
	}

	SeatState &ss = *wls.current_seat;

	if (!wls.current_seat->selection_data_control_source) {
		ss.selection_data_control_source = zwlr_data_control_manager_v1_create_data_source(wls.globals.wlr_data_control_manager);
		zwlr_data_control_source_v1_add_listener(ss.selection_data_control_source, &wlr_data_control_source_listener, &wls);
		zwlr_data_control_source_v1_offer(ss.selection_data_control_source, "text/plain");

		zwlr_data_control_device_v1_set_selection(ss.wlr_data_control_device, ss.selection_data_control_source);

		// Wait for the message to get to the server before continuing, otherwise the
		// clipboard update might come with a delay.
		wl_display_roundtrip(wls.display);
	}

	ss.selection_data = p_text.to_utf8_buffer();
}

String DisplayServerWayland::clipboard_get() const {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return String();
	}

	return read_data_control_offer(wls.current_seat->selection_data_control_offer);
}

void DisplayServerWayland::clipboard_set_primary(const String &p_text) {
	MutexLock mutex_lock(wls.mutex);

	if (!wls.current_seat) {
		return;
	}

	SeatState &ss = *wls.current_seat;

	if (!ss.primary_data_control_source) {
		ss.primary_data_control_source = zwlr_data_control_manager_v1_create_data_source(wls.globals.wlr_data_control_manager);
		zwlr_data_control_source_v1_add_listener(ss.primary_data_control_source, &wlr_data_control_source_listener, &wls);
		zwlr_data_control_source_v1_offer(ss.primary_data_control_source, "text/plain");

		zwlr_data_control_device_v1_set_primary_selection(ss.wlr_data_control_device, ss.primary_data_control_source);

		// Wait for the message to get to the server before continuing, otherwise the
		// clipboard update might come with a delay.
		wl_display_roundtrip(wls.display);
	}

	ss.primary_data = p_text.to_utf8_buffer();
}

String DisplayServerWayland::clipboard_get_primary() const {
	if (!wls.current_seat) {
		return String();
	}

	return read_data_control_offer(wls.current_seat->primary_data_control_offer);
}

int DisplayServerWayland::get_screen_count() const {
	MutexLock mutex_lock(wls.mutex);
	return wls.screens.size();
}

Point2i DisplayServerWayland::screen_get_position(int p_screen) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_INDEX_V(p_screen, (int)wls.screens.size(), Point2i());

	return wls.screens[p_screen]->position;
}

Size2i DisplayServerWayland::screen_get_size(int p_screen) const {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_INDEX_V(p_screen, (int)wls.screens.size(), Size2i());

	return wls.screens[p_screen]->size;
}

Rect2i DisplayServerWayland::screen_get_usable_rect(int p_screen) const {
	// TODO
	print_verbose("wayland stub screen_get_usable_rect");
	return Rect2i(0, 0, 1920, 1080);
}

int DisplayServerWayland::screen_get_dpi(int p_screen) const {
	MutexLock mutex_lock(wls.mutex);

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	// Invalid screen?
	ERR_FAIL_INDEX_V(p_screen, get_screen_count(), 0);

	ScreenData &sd = *wls.screens[p_screen];

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

	return wls.screens[p_screen]->refresh_rate;
}

bool DisplayServerWayland::screen_is_touchscreen(int p_screen) const {
	// TODO
	print_verbose("wayland stub screen_is_touchscreen");
	return false;
}

#if defined(DBUS_ENABLED)

void DisplayServerWayland::screen_set_keep_on(bool p_enable) {
	// TODO
	print_verbose("wayland stub screen_set_keep_on");
}

bool DisplayServerWayland::screen_is_kept_on() const {
	// TODO
	print_verbose("wayland stub screen_is_kept_on");
	return false;
}

#endif

Vector<DisplayServer::WindowID> DisplayServerWayland::get_window_list() const {
	// TODO
	print_verbose("wayland stub get_window_list");

	return Vector<DisplayServer::WindowID>();
}

DisplayServer::WindowID DisplayServerWayland::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	return _create_window(p_mode, p_vsync_mode, p_flags, p_rect);
}

void DisplayServerWayland::show_window(DisplayServer::WindowID p_id) {
	MutexLock mutex_lock(wls.mutex);

	WindowData &wd = wls.windows[p_id];

	ERR_FAIL_COND(!wls.windows.has(p_id));

	if (!wd.visible) {
		wd.visible = true;

		print_verbose(vformat("Showing window %d", p_id));

		wd.wl_surface = wl_compositor_create_surface(wls.globals.wl_compositor);
		wl_surface_add_listener(wd.wl_surface, &wl_surface_listener, &wd);

		wd.xdg_surface = xdg_wm_base_get_xdg_surface(wls.globals.xdg_wm_base, wd.wl_surface);
		xdg_surface_add_listener(wd.xdg_surface, &xdg_surface_listener, &wd);

		if (window_get_flag(WINDOW_FLAG_POPUP, p_id)) {
			ERR_FAIL_COND_MSG(wd.parent == INVALID_WINDOW_ID, "Popups must have a parent.");

			WindowData &parent_wd = wls.windows[wd.parent];

			struct xdg_positioner *xdg_positioner = xdg_wm_base_create_positioner(wls.globals.xdg_wm_base);
			xdg_positioner_set_size(xdg_positioner, wd.rect.size.x, wd.rect.size.y);
			xdg_positioner_set_anchor(xdg_positioner, XDG_POSITIONER_ANCHOR_TOP_LEFT);
			xdg_positioner_set_gravity(xdg_positioner, XDG_POSITIONER_GRAVITY_BOTTOM_RIGHT);
			xdg_positioner_set_anchor_rect(xdg_positioner, 0, 0, parent_wd.rect.size.width, parent_wd.rect.size.height);
			xdg_positioner_set_offset(xdg_positioner, wd.rect.position.x - parent_wd.rect.position.x, wd.rect.position.y - parent_wd.rect.position.y);

			print_verbose(vformat("Created popup at %s", wd.rect.position));

			wd.xdg_popup = xdg_surface_get_popup(wd.xdg_surface, parent_wd.xdg_surface, xdg_positioner);
			xdg_popup_add_listener(wd.xdg_popup, &xdg_popup_listener, &wd);

			xdg_positioner_destroy(xdg_positioner);
		} else {
			wd.xdg_toplevel = xdg_surface_get_toplevel(wd.xdg_surface);
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
		}

		wl_surface_commit(wd.wl_surface);

#ifdef VULKAN_ENABLED
		// Since `VulkanContextWayland::window_create` automatically assigns a buffer
		// to the `wl_surface` and doing so instantly maps it, moving this method here
		// is the only solution I can think of to implement this method properly.
		if (wls.context_vulkan) {
			wls.context_vulkan->window_create(p_id, wd.vsync_mode, wls.display, wd.wl_surface, wd.rect.size.width, wd.rect.size.height);
		}
#endif
	}
}

void DisplayServerWayland::delete_sub_window(DisplayServer::WindowID p_id) {
	MutexLock mutex_lock(wls.mutex);

	ERR_FAIL_COND(!wls.windows.has(p_id));
	ERR_FAIL_COND_MSG(p_id == MAIN_WINDOW_ID, "Main window can't be deleted.");

	_delete_window(p_id);
}

DisplayServer::WindowID DisplayServerWayland::get_window_at_screen_position(const Point2i &p_position) const {
	// TODO
	print_verbose("wayland stub get_window_at_screen_position");
	return WindowID(0);
}

void DisplayServerWayland::window_attach_instance_id(ObjectID p_instance, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_attach_instance_id");
}

ObjectID DisplayServerWayland::window_get_attached_instance_id(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_attached_instance_id");
	return ObjectID();
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
	print_verbose("wayland stub window_set_mouse_passthrough");
}

void DisplayServerWayland::window_set_rect_changed_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	WindowData &wd = wls.windows[p_window];
	wd.rect_changed_callback = p_callable;
}

void DisplayServerWayland::window_set_window_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	WindowData &wd = wls.windows[p_window];
	wd.window_event_callback = p_callable;
}

void DisplayServerWayland::window_set_input_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	WindowData &wd = wls.windows[p_window];
	wd.input_event_callback = p_callable;
}

void DisplayServerWayland::window_set_input_text_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_input_text_callback");
}

void DisplayServerWayland::window_set_drop_files_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_drop_files_callback");
}

int DisplayServerWayland::window_get_current_screen(DisplayServer::WindowID p_window) const {
	// TODO: Implement this somehow.
	// I've tried to do it before, but since we can only register window
	// entering/leaving from a screen, it would be too complex for the little
	// accuracy and usefulness we would get from it, as such, I've purposely left
	// this method as a stub, for now.
	return 0;
}

void DisplayServerWayland::window_set_current_screen(int p_screen, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_current_screen");
}

Point2i DisplayServerWayland::window_get_position(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	if (wls.windows[p_window].xdg_toplevel) {
		return Point2i();
	} else {
		return wls.windows[p_window].rect.position;
	}
}

void DisplayServerWayland::window_set_position(const Point2i &p_position, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	WindowData &wd = wls.windows[p_window];

	wd.rect.position = p_position;
}

void DisplayServerWayland::window_set_max_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_max_size");
}

Size2i DisplayServerWayland::window_get_max_size(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_max_size");
	return Size2i(1920, 1080);
}

void DisplayServerWayland::gl_window_make_current(DisplayServer::WindowID p_window_id) {
	// TODO
	print_verbose("wayland stub gl_window_make_current");
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

		ERR_FAIL_COND_MSG(!window_get_flag(WINDOW_FLAG_POPUP, p_window) && window_get_flag(WINDOW_FLAG_POPUP, p_parent), "Toplevels can't be parented to a popup.");
	}

	if (wd.xdg_toplevel) {
		xdg_toplevel_set_parent(wd.xdg_toplevel, parent_toplevel);
	}
}

void DisplayServerWayland::window_set_min_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_min_size");
}

Size2i DisplayServerWayland::window_get_min_size(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_min_size");
	return Size2i(0, 0);
}

void DisplayServerWayland::window_set_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

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
}

Size2i DisplayServerWayland::window_get_size(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	return wls.windows[p_window].rect.size;
}

Size2i DisplayServerWayland::window_get_real_size(DisplayServer::WindowID p_window) const {
	MutexLock mutex_lock(wls.mutex);

	// I don't think there's a way of actually knowing the window size in wayland,
	// other than the one requested by the compositor, which happens to be
	// the one the windows always uses anyway.
	return wls.windows[p_window].rect.size;
}

void DisplayServerWayland::window_set_mode(WindowMode p_mode, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_mode");
}

DisplayServerWayland::WindowMode DisplayServerWayland::window_get_mode(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_mode");
	return WINDOW_MODE_WINDOWED;
}

bool DisplayServerWayland::window_is_maximize_allowed(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_is_maximize_allowed");
	return false;
}

void DisplayServerWayland::window_set_flag(WindowFlags p_flag, bool p_enabled, DisplayServer::WindowID p_window) {
	MutexLock mutex_lock(wls.mutex);

	WindowData &wd = wls.windows[p_window];

	print_verbose(vformat("Window %d set flag %d", p_window, p_flag));

	switch (p_flag) {
		case WINDOW_FLAG_POPUP: {
			ERR_FAIL_COND_MSG(wd.visible, "Popup flag can't changed while window is opened.");
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
	// TODO
	print_verbose("wayland stub window_request_attention");
}

void DisplayServerWayland::window_move_to_foreground(DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_move_to_foreground");
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
	print_verbose("wayland stub window_set_ime_active");
}

void DisplayServerWayland::window_set_ime_position(const Point2i &p_pos, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_ime_position");
}

void DisplayServerWayland::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, DisplayServer::WindowID p_window) {
	// TODO: Figure out whether it is possible to disable VSync with Wayland
	// (doubt it) or handle any other mode.
	print_verbose("wayland stub window_set_vsync_mode");
}

DisplayServer::VSyncMode DisplayServerWayland::window_get_vsync_mode(DisplayServer::WindowID p_vsync_mode) const {
	// TODO: Figure out whether it is possible to disable VSync with Wayland
	// (doubt it) or handle any other mode.
	return VSYNC_ENABLED;
}

void DisplayServerWayland::cursor_set_shape(CursorShape p_shape) {
	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (p_shape == wls.cursor_shape) {
		return;
	}

	MutexLock mutex_lock(wls.mutex);

	wls.cursor_shape = p_shape;

	_wayland_state_update_cursor(wls);
}

DisplayServerWayland::CursorShape DisplayServerWayland::cursor_get_shape() const {
	MutexLock mutex_lock(wls.mutex);

	return wls.cursor_shape;
}

void DisplayServerWayland::cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	// TODO
	print_verbose("wayland stub cursor_set_custom_image");
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

	int werror = wl_display_get_error(wls.display);

	if (werror) {
		if (werror == EPROTO) {
			struct wl_interface *wl_interface = nullptr;
			uint32_t id = 0;

			int error_code = wl_display_get_protocol_error(wls.display, (const struct wl_interface **)&wl_interface, &id);
			print_error(vformat("Wayland protocol error %d on interface %s@%d", error_code, wl_interface ? wl_interface->name : "unknown", id));
		} else {
			print_error(vformat("Wayland client error code %d", werror));
		}
	}

	while (wls.message_queue.front()) {
		Ref<WaylandMessage> msg = wls.message_queue.front()->get();

		Ref<WaylandWindowRectMessage> winrect_msg = msg;

		if (winrect_msg.is_valid() && wls.windows.has(winrect_msg->id)) {
			WindowID id = winrect_msg->id;
			Rect2i rect = winrect_msg->rect;
			WindowData &wd = wls.windows[id];

#ifdef VULKAN_ENABLED
			if (wd.visible && wls.context_vulkan) {
				wls.context_vulkan->window_resize(id, rect.size.width, rect.size.height);
			}
#endif

			if (wd.rect_changed_callback.is_valid()) {
				Variant var_rect = Variant(rect);
				Variant *arg = &var_rect;

				Variant ret;
				Callable::CallError ce;

				wd.rect_changed_callback.call((const Variant **)&arg, 1, ret, ce);
			}

			// TODO: It would probably be better to do this Wayland-side.
			if (wls.current_seat && wls.current_seat->wp_locked_pointer && id == wls.current_seat->pointer_data.pointed_window_id) {
				// Since the window changes size, we have to reset its position hint, to
				// successfully have a centered cursor on unlock.
				wl_fixed_t unlock_x = wl_fixed_from_int(rect.size.width / 2);
				wl_fixed_t unlock_y = wl_fixed_from_int(rect.size.height / 2);

				zwp_locked_pointer_v1_set_cursor_position_hint(wls.current_seat->wp_locked_pointer, unlock_x, unlock_y);
			}
		}

		Ref<WaylandWindowEventMessage> winev_msg = msg;

		if (winev_msg.is_valid() && wls.windows.has(winev_msg->id)) {
			_send_window_event(winev_msg->id, winev_msg->event);
		}

		Ref<WaylandInputEventMessage> inputev_msg = msg;

		if (inputev_msg.is_valid()) {
			Ref<InputEvent> ev = inputev_msg->event;

			Input::get_singleton()->parse_input_event(inputev_msg->event);
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
	// TODO
	print_verbose("wayland stub release_rendering_thread");
}

void DisplayServerWayland::make_rendering_thread() {
	// TODO
	print_verbose("wayland stub make_rendering_thread");
}

void DisplayServerWayland::swap_buffers() {
	// TODO
	print_verbose("wayland stub swap_buffers");
}

void DisplayServerWayland::set_context(Context p_context) {
	// TODO
	print_verbose("wayland stub set_context");
}

void DisplayServerWayland::set_native_icon(const String &p_filename) {
	// TODO
	print_verbose("wayland stub set_native_icon");
}

void DisplayServerWayland::set_icon(const Ref<Image> &p_icon) {
	// TODO
	print_verbose("wayland stub set_icon");
}

Vector<String> DisplayServerWayland::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif

	// TODO
	/*
	 * #ifdef GLES3_ENABLED
	 * 	drivers.push_back("opengl3");
	 * #endif
	 */

	return drivers;
}

DisplayServer *DisplayServerWayland::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerWayland(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported Vulkan or OpenGL versions.\n"
								   "Please update your drivers or if you have a very old or integrated GPU, upgrade it.\n"
								   "If you have updated your graphics drivers recently, try rebooting.",
				"Unable to initialize Video driver");
	}
	return ds;
}

DisplayServerWayland::DisplayServerWayland(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	wls.display = wl_display_connect(nullptr);

	// TODO: Better error handling.
	ERR_FAIL_NULL(wls.display);

	wls.registry = wl_display_get_registry(wls.display);

	// TODO: Better error handling.
	ERR_FAIL_NULL(wls.display);

	wl_registry_add_listener(wls.registry, &wl_registry_listener, &wls);

	// Wait for globals to get notified from the compositor.
	wl_display_roundtrip(wls.display);

	// TODO: Perhaps gracefully handle missing protocols when possible?
	// TODO: Split this huge check into something more manageble.
	ERR_FAIL_COND(!wls.globals.wl_shm || !wls.globals.wl_compositor || !wls.globals.wp_pointer_constraints || !wls.globals.wlr_data_control_manager || !wls.globals.xdg_wm_base);

	// Input.
	Input::get_singleton()->set_event_dispatch_function(dispatch_input_events);

	// Wait for seat capabilities.
	wl_display_roundtrip(wls.display);

	xdg_wm_base_add_listener(wls.globals.xdg_wm_base, &xdg_wm_base_listener, nullptr);

#if defined(VULKAN_ENABLED)
	if (p_rendering_driver == "vulkan") {
		wls.context_vulkan = memnew(VulkanContextWayland);

		if (wls.context_vulkan->initialize() != OK) {
			memdelete(wls.context_vulkan);
			wls.context_vulkan = nullptr;
			r_error = ERR_CANT_CREATE;
			ERR_FAIL_MSG("Could not initialize Vulkan");
		}
	}
#endif

	// FIXME: We should get the cursor size from the user, somehow.
	wls.wl_cursor_theme = wl_cursor_theme_load(nullptr, 24, wls.globals.wl_shm);

	ERR_FAIL_NULL(wls.wl_cursor_theme);

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

	WindowID main_window_id = _create_window(p_mode, p_vsync_mode, p_flags, screen_get_usable_rect());
	show_window(main_window_id);

#ifdef VULKAN_ENABLED
	if (p_rendering_driver == "vulkan") {
		wls.rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		wls.rendering_device_vulkan->initialize(wls.context_vulkan);

		RendererCompositorRD::make_current();
	}

	r_error = OK;
#endif

	events_thread.start(_poll_events_thread, &wls);
}

DisplayServerWayland::~DisplayServerWayland() {
	wls.events_thread_done.set();

	// Wait for any Wayland events to be handled and unblock the polling thread.
	wl_display_roundtrip(wls.display);

	events_thread.wait_to_finish();

	// Destroy all windows.
	for (KeyValue<WindowID, WindowData> &E : wls.windows) {
		WindowID id = E.key;
		_delete_window(id);
	}

	for (SeatState &seat : wls.seats) {
		munmap((void *)seat.keymap_buffer, seat.keymap_buffer_size);
	}

	// Free all screens.
	for (int i = 0; i < (int)wls.screens.size(); i++) {
		memfree(wls.screens[i]);
		wls.screens[i] = nullptr;
	}

	wl_display_disconnect(wls.display);

	// Destroy all drivers.
#ifdef VULKAN_ENABLED
	if (wls.rendering_device_vulkan) {
		wls.rendering_device_vulkan->finalize();
		memdelete(wls.rendering_device_vulkan);
		wls.rendering_device_vulkan = nullptr;
	}

	if (wls.context_vulkan) {
		memdelete(wls.context_vulkan);
		wls.context_vulkan = nullptr;
	}
#endif
}

void DisplayServerWayland::register_wayland_driver() {
	register_create_function("wayland", create_func, get_rendering_drivers_func);
}

#endif //WAYLAND_ENABLED
