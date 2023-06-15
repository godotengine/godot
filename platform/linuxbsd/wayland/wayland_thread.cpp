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

// TODO: Get rid of this.
#include "display_server_wayland.h"

#ifdef WAYLAND_ENABLED

// Read the content pointed by fd into a string.
String WaylandThread::_string_read_fd(int fd) {
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

// Based on the wayland book's shared memory boilerplate (PD/CC0).
// See: https://wayland-book.com/surfaces/shared-memory.html
int WaylandThread::_allocate_shm_file(size_t size) {
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

// Read the content of a "text/plain" wl_data_offer.
String WaylandThread::_wl_data_offer_read(struct wl_display *p_display, struct wl_data_offer *p_offer) {
	if (!p_offer) {
		return "";
	}

	int fds[2];
	if (pipe(fds) == 0) {
		// This function expects to return a string, so we can only ask for a MIME of
		// "text/plain"
		wl_data_offer_receive(p_offer, "text/plain", fds[1]);

		// Wait for the compositor to know about the pipe.
		wl_display_roundtrip(p_display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		return _string_read_fd(fds[0]);
	}

	return "";
}

// Read the content of a "text/plain" wp_primary_selection_offer.
String WaylandThread::_wp_primary_selection_offer_read(struct wl_display *p_display, struct zwp_primary_selection_offer_v1 *p_offer) {
	if (!p_offer) {
		return "";
	}

	int fds[2];
	if (pipe(fds) == 0) {
		// This function expects to return a string, so we can only ask for a MIME of
		// "text/plain"
		zwp_primary_selection_offer_v1_receive(p_offer, "text/plain", fds[1]);

		// Wait for the compositor to know about the pipe.
		wl_display_roundtrip(p_display);

		// Close the write end of the pipe, which we don't need and would otherwise
		// just stall our next `read`s.
		close(fds[1]);

		return _string_read_fd(fds[0]);
	}

	return "";
}

void WaylandThread::_seat_state_set_current(WaylandThread::SeatState &p_ss) {
	WaylandThread::WaylandState *wls = p_ss.wls;
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

		// Free the pointer if it's confined.
		if (wls->current_seat->wp_confined_pointer) {
			zwp_confined_pointer_v1_destroy(wls->current_seat->wp_confined_pointer);
			wls->current_seat->wp_confined_pointer = nullptr;
		}
	}

	wls->current_seat = &p_ss;

	_wayland_state_update_cursor(*wls);
}

// Sets up an `InputEventKey` and returns whether it has any meaningful value.
bool WaylandThread::_seat_state_configure_key_event(WaylandThread::SeatState &p_ss, Ref<InputEventKey> p_event, xkb_keycode_t p_keycode, bool p_pressed) {
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

	p_event->set_window_id(DisplayServer::MAIN_WINDOW_ID);

	// Set all pressed modifiers.
	p_event->set_shift_pressed(p_ss.shift_pressed);
	p_event->set_ctrl_pressed(p_ss.ctrl_pressed);
	p_event->set_alt_pressed(p_ss.alt_pressed);
	p_event->set_meta_pressed(p_ss.meta_pressed);

	p_event->set_pressed(p_pressed);
	p_event->set_keycode(keycode);
	p_event->set_physical_keycode(physical_keycode);

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

// TODO: Port this method properly with WaylandThread semantics.
void WaylandThread::_wayland_state_update_cursor(WaylandThread::WaylandState &p_wls) {
	if (!p_wls.current_seat || !p_wls.current_seat->wl_pointer) {
		return;
	}

	WaylandThread::SeatState &ss = *p_wls.current_seat;

	ERR_FAIL_NULL(ss.cursor_surface);

	struct wl_pointer *wp = ss.wl_pointer;
	struct zwp_pointer_constraints_v1 *pc = p_wls.globals.wp_pointer_constraints;

	// In order to change the address of the WaylandThread::SeatState's pointers we need to get
	// their reference first.
	struct zwp_locked_pointer_v1 *&lp = ss.wp_locked_pointer;
	struct zwp_confined_pointer_v1 *&cp = ss.wp_confined_pointer;

	// All modes but `MOUSE_MODE_VISIBLE` and `MOUSE_MODE_CONFINED` are hidden.
	if (p_wls.mouse_mode != DisplayServer::MOUSE_MODE_VISIBLE && p_wls.mouse_mode != DisplayServer::MOUSE_MODE_CONFINED) {
		// Reset the cursor's hotspot.
		wl_pointer_set_cursor(ss.wl_pointer, ss.pointer_enter_serial, ss.cursor_surface, 0, 0);

		// Unmap the cursor.
		wl_surface_attach(ss.cursor_surface, nullptr, 0, 0);

		wl_surface_commit(ss.cursor_surface);
	} else {
		// Update the cursor shape.
		if (!ss.wl_pointer) {
			return;
		}

		struct wl_buffer *cursor_buffer = nullptr;
		Point2i hotspot;

		if (!p_wls.custom_cursors.has(p_wls.cursor_shape)) {
			// Select the default cursor data.
			struct wl_cursor_image *cursor_image = p_wls.cursor_images[p_wls.cursor_shape];

			if (!cursor_image) {
				// TODO: Error out?
				return;
			}

			cursor_buffer = p_wls.cursor_bufs[p_wls.cursor_shape];
			hotspot.x = cursor_image->hotspot_x;
			hotspot.y = cursor_image->hotspot_y;
		} else {
			// Select the custom cursor data.
			WaylandThread::CustomWaylandCursor &custom_cursor = p_wls.custom_cursors[p_wls.cursor_shape];

			cursor_buffer = custom_cursor.wl_buffer;
			hotspot = custom_cursor.hotspot;
		}

		// Update the cursor's hotspot.
		wl_pointer_set_cursor(ss.wl_pointer, ss.pointer_enter_serial, ss.cursor_surface, hotspot.x, hotspot.y);

		// Attach the new cursor's buffer and damage it.
		wl_surface_attach(ss.cursor_surface, cursor_buffer, 0, 0);
		wl_surface_damage_buffer(ss.cursor_surface, 0, 0, INT_MAX, INT_MAX);

		// Commit everything.
		wl_surface_commit(ss.cursor_surface);
	}

	struct wl_surface *wl_surface = p_wls.wayland_thread->window_get_wl_surface(DisplayServer::MAIN_WINDOW_ID);
	WaylandThread::WindowState *ws = p_wls.wayland_thread->wl_surface_get_window_state(wl_surface);

	// Constrain/Free pointer movement depending on its mode.
	switch (p_wls.mouse_mode) {
		// Unconstrained pointer.
		case DisplayServer::MOUSE_MODE_VISIBLE:
		case DisplayServer::MOUSE_MODE_HIDDEN: {
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
		case DisplayServer::MOUSE_MODE_CAPTURED: {
			if (!lp) {
				Rect2i logical_rect = ws->rect;

				lp = zwp_pointer_constraints_v1_lock_pointer(pc, wl_surface, wp, nullptr, ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);

				// Center the cursor on unlock.
				wl_fixed_t unlock_x = wl_fixed_from_int(logical_rect.size.width / 2);
				wl_fixed_t unlock_y = wl_fixed_from_int(logical_rect.size.height / 2);

				zwp_locked_pointer_v1_set_cursor_position_hint(lp, unlock_x, unlock_y);
			}
		} break;

		// Confined pointer.
		case DisplayServer::MOUSE_MODE_CONFINED:
		case DisplayServer::MOUSE_MODE_CONFINED_HIDDEN: {
			if (!cp) {
				cp = zwp_pointer_constraints_v1_confine_pointer(pc, wl_surface, wp, nullptr, ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
			}
		}
	}
}

void WaylandThread::_wl_registry_on_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version) {
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
			// Initialize the data device for all current seats.
			if (ss.wl_seat && !ss.wl_data_device && globals.wl_data_device_manager) {
				ss.wl_data_device = wl_data_device_manager_get_data_device(wls->globals.wl_data_device_manager, ss.wl_seat);
				wl_data_device_add_listener(ss.wl_data_device, &wl_data_device_listener, &ss);
			}
		}
		return;
	}

	if (strcmp(interface, zwp_primary_selection_device_manager_v1_interface.name) == 0) {
		globals.wp_primary_selection_device_manager = (struct zwp_primary_selection_device_manager_v1 *)wl_registry_bind(wl_registry, name, &zwp_primary_selection_device_manager_v1_interface, 1);

		for (SeatState &ss : wls->seats) {
			if (!ss.wp_primary_selection_device && globals.wp_primary_selection_device_manager) {
				// Initialize the primary selection device for all current seats.
				ss.wp_primary_selection_device = zwp_primary_selection_device_manager_v1_get_device(wls->globals.wp_primary_selection_device_manager, ss.wl_seat);
				zwp_primary_selection_device_v1_add_listener(ss.wp_primary_selection_device, &wp_primary_selection_device_listener, &ss);
			}
		}
	}

	if (strcmp(interface, wl_output_interface.name) == 0) {
		struct wl_output *wl_output = (struct wl_output *)wl_registry_bind(wl_registry, name, &wl_output_interface, 2);

		globals.wl_outputs.push_back(wl_output);

		ScreenState *ss = memnew(ScreenState);
		ss->wl_output_name = name;

		wl_proxy_tag_godot((struct wl_proxy *)wl_output);
		wl_output_add_listener(wl_output, &wl_output_listener, ss);
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
			// Initialize the data device for this new seat.
			ss->wl_data_device = wl_data_device_manager_get_data_device(globals.wl_data_device_manager, ss->wl_seat);
			wl_data_device_add_listener(ss->wl_data_device, &wl_data_device_listener, ss);
		}

		if (!ss->wp_primary_selection_device && globals.wp_primary_selection_device_manager) {
			// Initialize the primary selection device for this new seat.
			ss->wp_primary_selection_device = zwp_primary_selection_device_manager_v1_get_device(wls->globals.wp_primary_selection_device_manager, ss->wl_seat);
			zwp_primary_selection_device_v1_add_listener(ss->wp_primary_selection_device, &wp_primary_selection_device_listener, ss);
		}

		if (!ss->wp_tablet_seat && globals.wp_tablet_manager) {
			// Get this own seat's tablet handle.
			ss->wp_tablet_seat = zwp_tablet_manager_v2_get_tablet_seat(globals.wp_tablet_manager, ss->wl_seat);
			zwp_tablet_seat_v2_add_listener(ss->wp_tablet_seat, &wp_tablet_seat_listener, ss);
		}

		wl_seat_add_listener(ss->wl_seat, &wl_seat_listener, ss);
		return;
	}

	if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		globals.xdg_wm_base = (struct xdg_wm_base *)wl_registry_bind(wl_registry, name, &xdg_wm_base_interface, MAX(2, MIN(5, (int)version)));
		globals.xdg_wm_base_name = name;

		xdg_wm_base_add_listener(globals.xdg_wm_base, &xdg_wm_base_listener, nullptr);
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

	if (strcmp(interface, zwp_pointer_gestures_v1_interface.name) == 0) {
		globals.wp_pointer_gestures = (struct zwp_pointer_gestures_v1 *)wl_registry_bind(wl_registry, name, &zwp_pointer_gestures_v1_interface, 1);
		globals.wp_pointer_gestures_name = name;
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

	if (strcmp(interface, zwp_tablet_manager_v2_interface.name) == 0) {
		globals.wp_tablet_manager = (struct zwp_tablet_manager_v2 *)wl_registry_bind(wl_registry, name, &zwp_tablet_manager_v2_interface, 1);
		globals.wp_tablet_manager_name = name;

		for (SeatState &ss : wls->seats) {
			if (ss.wl_seat) {
				ss.wp_tablet_seat = zwp_tablet_manager_v2_get_tablet_seat(globals.wp_tablet_manager, ss.wl_seat);
				zwp_tablet_seat_v2_add_listener(ss.wp_tablet_seat, &wp_tablet_seat_listener, &ss);
			}
		}

		return;
	}
}

void WaylandThread::_wl_registry_on_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name) {
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

	if (name == globals.wp_pointer_gestures_name) {
		if (globals.wp_pointer_gestures) {
			zwp_pointer_gestures_v1_destroy(globals.wp_pointer_gestures);
		}
		globals.wp_pointer_gestures = nullptr;
		globals.wp_pointer_gestures_name = 0;
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

	if (name == globals.wp_tablet_manager_name) {
		zwp_tablet_manager_v2_destroy(globals.wp_tablet_manager);

		for (SeatState &ss : wls->seats) {
			{
				// Let's destroy all tablet tools.
				List<struct zwp_tablet_tool_v2 *>::Element *it = ss.tablet_tools.front();

				while (it) {
					zwp_tablet_tool_v2_destroy(it->get());
					it = it->next();
				}
			}
		}
	}

	{
		// FIXME: This is a very bruteforce approach.
		List<struct wl_output *>::Element *it = globals.wl_outputs.front();
		while (it) {
			// Iterate through all of the screens to find if any got removed.
			struct wl_output *wl_output = it->get();
			ERR_FAIL_NULL(wl_output);

			ScreenState *ss = wl_output_get_screen_state(wl_output);

			if (ss->wl_output_name == name) {
				globals.wl_outputs.erase(it);

				memdelete(ss);
				wl_output_destroy(wl_output);

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

				if (ss.wp_tablet_seat) {
					zwp_tablet_seat_v2_destroy(ss.wp_tablet_seat);

					for (struct zwp_tablet_tool_v2 *tool : ss.tablet_tools) {
						zwp_tablet_tool_v2_destroy(tool);
					}
				}

				{
					// Let's destroy all tools.
					for (struct zwp_tablet_tool_v2 *tool : ss.tablet_tools) {
						zwp_tablet_tool_v2_destroy(tool);
					}
				}

				wls->seats.erase(it);
				return;
			}

			it = it->next();
		}
	}
}

void WaylandThread::_wl_surface_on_enter(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	// TODO: Handle multiple outputs?

	ws->wl_output = wl_output;
}

void WaylandThread::_wl_surface_on_leave(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {
}

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
}

void WaylandThread::_wl_output_on_mode(void *data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh) {
	ScreenState *ss = (ScreenState *)data;
	ERR_FAIL_NULL(ss);

	ss->pending_data.size.width = width;
	ss->pending_data.size.height = height;

	ss->pending_data.refresh_rate = refresh ? refresh / 1000.0f : -1;
}

void WaylandThread::_wl_output_on_done(void *data, struct wl_output *wl_output) {
	ScreenState *ss = (ScreenState *)data;
	ERR_FAIL_NULL(ss);

	ss->data = ss->pending_data;

	DEBUG_LOG_WAYLAND(vformat("Output %x done.", (size_t)wl_output));
}

void WaylandThread::_wl_output_on_scale(void *data, struct wl_output *wl_output, int32_t factor) {
	ScreenState *ss = (ScreenState *)data;
	ERR_FAIL_NULL(ss);

	ss->pending_data.scale = factor;

	DEBUG_LOG_WAYLAND(vformat("Output %x scale %d", (size_t)wl_output, factor));
}

void WaylandThread::_wl_output_on_name(void *data, struct wl_output *wl_output, const char *name) {
}

void WaylandThread::_wl_output_on_description(void *data, struct wl_output *wl_output, const char *description) {
}

void WaylandThread::_xdg_wm_base_on_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}

void WaylandThread::_xdg_surface_on_configure(void *data, struct xdg_surface *xdg_surface, uint32_t serial) {
	xdg_surface_ack_configure(xdg_surface, serial);

	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	int scale = WaylandThread::window_state_calculate_scale(ws);

	Rect2i scaled_rect = ws->rect;
	scaled_rect.size *= scale;

	if (ws->wl_surface) {
		wl_surface_set_buffer_scale(ws->wl_surface, scale);
	}

	if (ws->xdg_surface) {
		xdg_surface_set_window_geometry(ws->xdg_surface, 0, 0, scaled_rect.size.width, scaled_rect.size.height);
	}

#if 0
	// TODO: Port to WaylandThread with wl_pointer user data.
	//WaylandState *wls = ws->wls;
	if (wls->current_seat && wls->current_seat->wp_locked_pointer) {
		// Since the cursor's currently locked and the window's rect might have
		// changed, we have to recenter the position hint to ensure that the cursor
		// stays centered on unlock.
		wl_fixed_t unlock_x = wl_fixed_from_int(width / 2);
		wl_fixed_t unlock_y = wl_fixed_from_int(height / 2);

		zwp_locked_pointer_v1_set_cursor_position_hint(wls->current_seat->wp_locked_pointer, unlock_x, unlock_y);
	}
#endif

	Ref<WindowRectMessage> msg;
	msg.instantiate();
	msg->rect = scaled_rect;

	ws->wayland_thread->push_message(msg);

	DEBUG_LOG_WAYLAND(vformat("xdg surface on configure rect %s", ws->rect));
}

void WaylandThread::_xdg_toplevel_on_configure(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states) {
	WindowState *ws = (WindowState *)data;
	ERR_FAIL_NULL(ws);

	if (width != 0 && height != 0) {
		ws->rect.size.width = width;
		ws->rect.size.height = height;
	}

	// Expect the window to be in windowed mode. The mode will get overridden if
	// the compositor reports otherwise.
	ws->mode = DisplayServer::WINDOW_MODE_WINDOWED;

	uint32_t *state = nullptr;
	wl_array_for_each(state, states) {
		switch (*state) {
			case XDG_TOPLEVEL_STATE_MAXIMIZED: {
				ws->mode = DisplayServer::WINDOW_MODE_MAXIMIZED;
			} break;

			case XDG_TOPLEVEL_STATE_FULLSCREEN: {
				ws->mode = DisplayServer::WINDOW_MODE_FULLSCREEN;
			} break;

			default: {
				// We don't care about the other states (for now).
			} break;
		}
	}

	DEBUG_LOG_WAYLAND(vformat("XDG toplevel on configure width %d height %d.", width, height));
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
			}; break;
			case XDG_TOPLEVEL_WM_CAPABILITIES_FULLSCREEN: {
				ws->can_fullscreen = true;
			}; break;

			case XDG_TOPLEVEL_WM_CAPABILITIES_MINIMIZE: {
				ws->can_minimize = true;
			}; break;

			default: {
			}; break;
		}
	}
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

#ifdef LIBDECOR_ENABLED
void WaylandThread::libdecor_on_error(struct libdecor *context, enum libdecor_error error, const char *message) {
	ERR_PRINT(vformat("libdecor error %d: %s", error, message));
}

// NOTE: This is pretty much a reimplementation of _xdg_surface_on_configure
// and _xdg_toplevel_on_configure. Libdecor really likes wrapping everything,
// forcing us to do stuff like this.
void WaylandThread::libdecor_frame_on_configure(struct libdecor_frame *frame, struct libdecor_configuration *configuration, void *user_data) {
	WindowState *ws = (WindowState *)user_data;
	ERR_FAIL_NULL(ws);

	int width = 0;
	int height = 0;

	if (!libdecor_configuration_get_content_size(configuration, frame, &width, &height)) {
		// The configuration doesn't have a size. We'll use the one already set in the window.
		width = ws->rect.size.width;
		height = ws->rect.size.height;
	} else {
		// The configuration has a size, let's update the window rect.
		ws->rect.size.width = width;
		ws->rect.size.height = height;
	}

	libdecor_window_state window_state = LIBDECOR_WINDOW_STATE_NONE;

	// Expect the window to be in windowed mode. The mode will get overridden if
	// the compositor reports otherwise.
	ws->mode = DisplayServer::WINDOW_MODE_WINDOWED;

	if (libdecor_configuration_get_window_state(configuration, &window_state)) {
		switch (window_state) {
			case LIBDECOR_WINDOW_STATE_MAXIMIZED: {
				ws->mode = DisplayServer::WINDOW_MODE_MAXIMIZED;
			} break;

			case LIBDECOR_WINDOW_STATE_FULLSCREEN: {
				ws->mode = DisplayServer::WINDOW_MODE_FULLSCREEN;
			} break;

			default: {
				// We don't care about the other states (for now).
			} break;
		}
	}

	int scale = WaylandThread::window_state_calculate_scale(ws);

	Rect2i scaled_rect = ws->rect;
	scaled_rect.size *= scale;

	if (ws->wl_surface) {
		wl_surface_set_buffer_scale(ws->wl_surface, scale);
		wl_surface_commit(ws->wl_surface);
	}

	if (ws->libdecor_frame) {
		struct libdecor_state *state = libdecor_state_new(scaled_rect.size.width, scaled_rect.size.height);
		libdecor_frame_commit(ws->libdecor_frame, state, configuration);
		libdecor_state_free(state);
	}

	Ref<WindowRectMessage> winrect_msg;
	winrect_msg.instantiate();
	winrect_msg->rect = scaled_rect;
	ws->wayland_thread->push_message(winrect_msg);

	DEBUG_LOG_WAYLAND("libdecor frame on configure");
}

void WaylandThread::libdecor_frame_on_close(struct libdecor_frame *frame, void *user_data) {
	WindowState *ws = (WindowState *)user_data;
	ERR_FAIL_NULL(ws);

	Ref<WindowEventMessage> winevent_msg;
	winevent_msg.instantiate();
	winevent_msg->event = DisplayServer::WINDOW_EVENT_CLOSE_REQUEST;

	ws->wayland_thread->push_message(winevent_msg);

	DEBUG_LOG_WAYLAND("libdecor frame on close");
}

void WaylandThread::libdecor_frame_on_commit(struct libdecor_frame *frame, void *user_data) {
	WindowState *ws = (WindowState *)user_data;
	ERR_FAIL_NULL(ws);

	wl_surface_commit(ws->wl_surface);

	DEBUG_LOG_WAYLAND("libdecor frame on commit");
}

void WaylandThread::libdecor_frame_on_dismiss_popup(struct libdecor_frame *frame, const char *seat_name, void *user_data) {
}
#endif // LIBDECOR_ENABLED

void WaylandThread::_wl_seat_on_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;

	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	// TODO: Handle touch.

	// Pointer handling.
	if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
		WaylandThread::WaylandGlobals &globals = wls->globals;

		ss->cursor_surface = wl_compositor_create_surface(globals.wl_compositor);

		ss->wl_pointer = wl_seat_get_pointer(wl_seat);
		wl_pointer_add_listener(ss->wl_pointer, &wl_pointer_listener, ss);

		ss->wp_relative_pointer = zwp_relative_pointer_manager_v1_get_relative_pointer(globals.wp_relative_pointer_manager, ss->wl_pointer);
		zwp_relative_pointer_v1_add_listener(ss->wp_relative_pointer, &wp_relative_pointer_listener, ss);

		if (globals.wp_pointer_gestures) {
			ss->wp_pointer_gesture_pinch = zwp_pointer_gestures_v1_get_pinch_gesture(globals.wp_pointer_gestures, ss->wl_pointer);
			zwp_pointer_gesture_pinch_v1_add_listener(ss->wp_pointer_gesture_pinch, &wp_pointer_gesture_pinch_listener, ss);
		}
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

void WaylandThread::_wl_seat_on_name(void *data, struct wl_seat *wl_seat, const char *name) {
}

void WaylandThread::_wl_pointer_on_enter(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	// Make sure the cursor shows its assigned surface.
	_wayland_state_update_cursor(*wls);

	ss->pointer_enter_serial = serial;

	ss->window_pointed = true;
	ss->pointed_surface = surface;

	Ref<WaylandThread::WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_MOUSE_ENTER;

	wls->wayland_thread->push_message(msg);

	DEBUG_LOG_WAYLAND("Pointing window.");
}

void WaylandThread::_wl_pointer_on_leave(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	ss->window_pointed = false;
	ss->pointed_surface = nullptr;

	Ref<WaylandThread::WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_MOUSE_EXIT;

	wls->wayland_thread->push_message(msg);

	DEBUG_LOG_WAYLAND("Left window.");
}

void WaylandThread::_wl_pointer_on_motion(void *data, struct wl_pointer *wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WindowState *ws = WaylandThread::wl_surface_get_window_state(ss->pointed_surface);
	ERR_FAIL_NULL(ws);

	int scale = WaylandThread::window_state_calculate_scale(ws);

	WaylandThread::PointerData &pd = ss->pointer_data_buffer;

	pd.position.x = wl_fixed_to_int(surface_x) * scale;
	pd.position.y = wl_fixed_to_int(surface_y) * scale;

	pd.motion_time = time;
}

void WaylandThread::_wl_pointer_on_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	WaylandThread::PointerData &pd = ss->pointer_data_buffer;

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
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::PointerData &pd = ss->pointer_data_buffer;

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
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	_seat_state_set_current(*ss);

	WaylandThread::PointerData &old_pd = ss->pointer_data;
	WaylandThread::PointerData &pd = ss->pointer_data_buffer;

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

		Ref<WaylandThread::InputEventMessage> msg;
		msg.instantiate();

		msg->event = mm;

		wls->wayland_thread->push_message(msg);
	}

	if (pd.discrete_scroll_vector - old_pd.discrete_scroll_vector != Vector2i()) {
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

			Ref<WaylandThread::InputEventMessage> msg;
			msg.instantiate();

			msg->event = pg;

			wls->wayland_thread->push_message(msg);
		}
	}

	if (old_pd.pressed_button_mask != pd.pressed_button_mask) {
		BitField<MouseButtonMask> pressed_mask_delta = BitField<MouseButtonMask>((uint32_t)old_pd.pressed_button_mask ^ (uint32_t)pd.pressed_button_mask);

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
					mb->set_factor(abs(pd.discrete_scroll_vector.y));
				}

				if (test_button == MouseButton::WHEEL_RIGHT || test_button == MouseButton::WHEEL_LEFT) {
					// If this is a discrete scroll, specify how many "clicks" it did for this
					// pointer frame.
					mb->set_factor(abs(pd.discrete_scroll_vector.x));
				}

				mb->set_button_mask(pd.pressed_button_mask);

				mb->set_button_index(test_button);
				mb->set_pressed(pd.pressed_button_mask.has_flag(test_button_mask));

				// We have to set the last position pressed here as we can't take for
				// granted what the individual events might have seen due to them not having
				// a garaunteed order.
				if (mb->is_pressed()) {
					pd.last_pressed_position = pd.position;
				}

				if (old_pd.double_click_begun && mb->is_pressed() && pd.last_button_pressed == old_pd.last_button_pressed && (pd.button_time - old_pd.button_time) < 400 && Vector2(old_pd.last_pressed_position).distance_to(Vector2(pd.last_pressed_position)) < 5) {
					pd.double_click_begun = false;
					mb->set_double_click(true);
				}

				Ref<WaylandThread::InputEventMessage> msg;
				msg.instantiate();

				msg->event = mb;

				wls->wayland_thread->push_message(msg);

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

					Ref<WaylandThread::InputEventMessage> msg_up;
					msg_up.instantiate();
					msg_up->event = wh_up;
					wls->wayland_thread->push_message(msg_up);
				}
			}
		}
	}

	// Reset the scroll vectors as we already handled them.
	pd.scroll_vector = Vector2();
	pd.discrete_scroll_vector = Vector2();

	// Update the data all getters read. Wayland's specification requires us to do
	// this, since all pointer actions are sent in individual events.
	old_pd = pd;
}

void WaylandThread::_wl_pointer_on_axis_source(void *data, struct wl_pointer *wl_pointer, uint32_t axis_source) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	ss->pointer_data_buffer.scroll_type = axis_source;
}

void WaylandThread::_wl_pointer_on_axis_stop(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis) {
}

void WaylandThread::_wl_pointer_on_axis_discrete(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	WaylandThread::PointerData &pd = ss->pointer_data_buffer;

	if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pd.discrete_scroll_vector.y = discrete;
	}

	if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pd.discrete_scroll_vector.x = discrete;
	}
}

// TODO: Add support to this event.
void WaylandThread::_wl_pointer_on_axis_value120(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t value120) {
}

void WaylandThread::_wl_keyboard_on_keymap(void *data, struct wl_keyboard *wl_keyboard, uint32_t format, int32_t fd, uint32_t size) {
	ERR_FAIL_COND_MSG(format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1, "Unsupported keymap format announced from the Wayland compositor.");

	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
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

void WaylandThread::_wl_keyboard_on_enter(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	_seat_state_set_current(*ss);

	Ref<WaylandThread::WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_FOCUS_IN;
	wls->wayland_thread->push_message(msg);
}

void WaylandThread::_wl_keyboard_on_leave(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	ss->repeating_keycode = XKB_KEYCODE_INVALID;

	Ref<WaylandThread::WindowEventMessage> msg;
	msg.instantiate();
	msg->event = DisplayServer::WINDOW_EVENT_FOCUS_OUT;
	wls->wayland_thread->push_message(msg);
}

void WaylandThread::_wl_keyboard_on_key(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
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

	Ref<WaylandThread::InputEventMessage> msg;
	msg.instantiate();
	msg->event = k;
	wls->wayland_thread->push_message(msg);
}

void WaylandThread::_wl_keyboard_on_modifiers(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	_seat_state_set_current(*ss);

	xkb_state_update_mask(ss->xkb_state, mods_depressed, mods_latched, mods_locked, ss->current_layout_index, ss->current_layout_index, group);

	ss->shift_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_SHIFT, XKB_STATE_MODS_DEPRESSED);
	ss->ctrl_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_CTRL, XKB_STATE_MODS_DEPRESSED);
	ss->alt_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_ALT, XKB_STATE_MODS_DEPRESSED);
	ss->meta_pressed = xkb_state_mod_name_is_active(ss->xkb_state, XKB_MOD_NAME_LOGO, XKB_STATE_MODS_DEPRESSED);
}

void WaylandThread::_wl_keyboard_on_repeat_info(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->repeat_key_delay_msec = 1000 / rate;
	ss->repeat_start_delay_msec = delay;
}

void WaylandThread::_wl_data_device_on_data_offer(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id) {
	ERR_FAIL_NULL(data);

	wl_data_offer_add_listener(id, &wl_data_offer_listener, data);
}

void WaylandThread::_wl_data_device_on_enter(void *data, struct wl_data_device *wl_data_device, uint32_t serial, struct wl_surface *surface, wl_fixed_t x, wl_fixed_t y, struct wl_data_offer *id) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->dnd_enter_serial = serial;

	wl_data_offer_set_actions(id, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);
}

void WaylandThread::_wl_data_device_on_leave(void *data, struct wl_data_device *wl_data_device) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (ss->wl_data_offer_dnd) {
		wl_data_offer_destroy(ss->wl_data_offer_dnd);
		ss->wl_data_offer_dnd = nullptr;
	}
}

void WaylandThread::_wl_data_device_on_motion(void *data, struct wl_data_device *wl_data_device, uint32_t time, wl_fixed_t x, wl_fixed_t y) {
}

void WaylandThread::_wl_data_device_on_drop(void *data, struct wl_data_device *wl_data_device) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ERR_FAIL_NULL(ss->wl_data_offer_dnd);

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

		Ref<WaylandThread::DropFilesEventMessage> msg;
		msg.instantiate();

		msg->files = _string_read_fd(fds[0]).split("\r\n", false);
		for (int i = 0; i < msg->files.size(); i++) {
			msg->files.write[i] = msg->files[i].replace("file://", "").uri_decode();
		}

		ss->wls->wayland_thread->push_message(msg);
	}

	wl_data_offer_finish(ss->wl_data_offer_dnd);
	wl_data_offer_destroy(ss->wl_data_offer_dnd);
	ss->wl_data_offer_dnd = nullptr;
}

void WaylandThread::_wl_data_device_on_selection(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (ss->wl_data_offer_selection) {
		wl_data_offer_destroy(ss->wl_data_offer_selection);
	}

	ss->wl_data_offer_selection = id;
}

void WaylandThread::_wl_data_offer_on_offer(void *data, struct wl_data_offer *wl_data_offer, const char *mime_type) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (strcmp(mime_type, "text/uri-list") == 0) {
		ss->wl_data_offer_dnd = wl_data_offer;
		wl_data_offer_accept(wl_data_offer, ss->dnd_enter_serial, mime_type);
	}
}

void WaylandThread::_wl_data_offer_on_source_actions(void *data, struct wl_data_offer *wl_data_offer, uint32_t source_actions) {
}

void WaylandThread::_wl_data_offer_on_action(void *data, struct wl_data_offer *wl_data_offer, uint32_t dnd_action) {
}

void WaylandThread::_wl_data_source_on_target(void *data, struct wl_data_source *wl_data_source, const char *mime_type) {
}

void WaylandThread::_wl_data_source_on_send(void *data, struct wl_data_source *wl_data_source, const char *mime_type, int32_t fd) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
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

void WaylandThread::_wl_data_source_on_cancelled(void *data, struct wl_data_source *wl_data_source) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	wl_data_source_destroy(wl_data_source);

	if (wl_data_source == ss->wl_data_source_selection) {
		ss->wl_data_source_selection = nullptr;

		ss->selection_data.clear();

		DEBUG_LOG_WAYLAND("Clipboard: selection set by another program.");
		return;
	}
}

void WaylandThread::_wl_data_source_on_dnd_drop_performed(void *data, struct wl_data_source *wl_data_source) {
}

void WaylandThread::_wl_data_source_on_dnd_finished(void *data, struct wl_data_source *wl_data_source) {
}

void WaylandThread::_wl_data_source_on_action(void *data, struct wl_data_source *wl_data_source, uint32_t dnd_action) {
}

void WaylandThread::_wp_relative_pointer_on_relative_motion(void *data, struct zwp_relative_pointer_v1 *wp_relative_pointer, uint32_t uptime_hi, uint32_t uptime_lo, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t dx_unaccel, wl_fixed_t dy_unaccel) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::PointerData &pd = ss->pointer_data_buffer;

	pd.relative_motion.x = wl_fixed_to_double(dx);
	pd.relative_motion.y = wl_fixed_to_double(dy);

	pd.relative_motion_time = uptime_lo;
}

void WaylandThread::_wp_pointer_gesture_pinch_on_begin(void *data, struct zwp_pointer_gesture_pinch_v1 *zwp_pointer_gesture_pinch_v1, uint32_t serial, uint32_t time, struct wl_surface *surface, uint32_t fingers) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (fingers == 2) {
		ss->old_pinch_scale = wl_fixed_from_int(1);
		ss->active_gesture = Gesture::MAGNIFY;
	}
}

void WaylandThread::_wp_pointer_gesture_pinch_on_update(void *data, struct zwp_pointer_gesture_pinch_v1 *zwp_pointer_gesture_pinch_v1, uint32_t time, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t scale, wl_fixed_t rotation) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	WaylandThread::PointerData &pd = ss->pointer_data_buffer;

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

		Ref<WaylandThread::InputEventMessage> magnify_msg;
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

		Ref<WaylandThread::InputEventMessage> pan_msg;
		pan_msg.instantiate();
		pan_msg->event = pg;

		wls->wayland_thread->push_message(magnify_msg);
		wls->wayland_thread->push_message(pan_msg);

		ss->old_pinch_scale = scale;
	}
}

void WaylandThread::_wp_pointer_gesture_pinch_on_end(void *data, struct zwp_pointer_gesture_pinch_v1 *zwp_pointer_gesture_pinch_v1, uint32_t serial, uint32_t time, int32_t cancelled) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->active_gesture = Gesture::NONE;
}

void WaylandThread::_wp_primary_selection_device_on_data_offer(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *offer) {
	// This method is purposely left unimplemented as we don't care about the
	// offered MIME type, as we only want `text/plain` data.

	// TODO: Perhaps we could try to detect other text types such as `TEXT`?
}

void WaylandThread::_wp_primary_selection_device_on_selection(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *id) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (ss->wp_primary_selection_offer) {
		zwp_primary_selection_offer_v1_destroy(ss->wp_primary_selection_offer);
	}

	ss->wp_primary_selection_offer = id;
}

void WaylandThread::_wp_primary_selection_source_on_send(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1, const char *mime_type, int32_t fd) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
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

void WaylandThread::_wp_primary_selection_source_on_cancelled(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (wp_primary_selection_source_v1 == ss->wp_primary_selection_source) {
		zwp_primary_selection_source_v1_destroy(ss->wp_primary_selection_source);
		ss->wp_primary_selection_source = nullptr;

		ss->primary_data.clear();

		DEBUG_LOG_WAYLAND("Clipboard: primary selection set by another program.");
		return;
	}
}

void WaylandThread::_wp_tablet_seat_on_tablet_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_v2 *id) {
	DEBUG_LOG_WAYLAND(vformat("wp tablet seat %x on tablet %x added", (size_t)zwp_tablet_seat_v2, (size_t)id));
}

void WaylandThread::_wp_tablet_seat_on_tool_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_tool_v2 *id) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->tablet_tools.push_back(id);

	zwp_tablet_tool_v2_add_listener(id, &wp_tablet_tool_listener, ss);

	DEBUG_LOG_WAYLAND(vformat("wp tablet seat %x on tool %x added", (size_t)zwp_tablet_seat_v2, (size_t)id));
}

void WaylandThread::_wp_tablet_seat_on_pad_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_pad_v2 *id) {
	DEBUG_LOG_WAYLAND(vformat("wp tablet seat %x on pad %x added", (size_t)zwp_tablet_seat_v2, (size_t)id));
}

void WaylandThread::_wp_tablet_tool_on_type(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t tool_type) {
	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on type %d", (size_t)zwp_tablet_tool_v2, tool_type));
}

void WaylandThread::_wp_tablet_tool_on_hardware_serial(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t hardware_serial_hi, uint32_t hardware_serial_lo) {
	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on hardware serial %x%x", (size_t)zwp_tablet_tool_v2, hardware_serial_hi, hardware_serial_lo));
}

void WaylandThread::_wp_tablet_tool_on_hardware_id_wacom(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t hardware_id_hi, uint32_t hardware_id_lo) {
	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on hardware id wacom hardware id %x%x", (size_t)zwp_tablet_tool_v2, hardware_id_hi, hardware_id_lo));
}

void WaylandThread::_wp_tablet_tool_on_capability(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t capability) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (capability == ZWP_TABLET_TOOL_V2_TYPE_ERASER) {
		ss->tablet_tool_data_buffer.is_eraser = true;
	}

	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on capability %d", (size_t)zwp_tablet_tool_v2, capability));
}

void WaylandThread::_wp_tablet_tool_on_done(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2) {
	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on done", (size_t)zwp_tablet_tool_v2));
}

void WaylandThread::_wp_tablet_tool_on_removed(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	List<struct zwp_tablet_tool_v2 *>::Element *it = ss->tablet_tools.front();

	while (it) {
		struct zwp_tablet_tool_v2 *tool = it->get();

		if (tool == zwp_tablet_tool_v2) {
			zwp_tablet_tool_v2_destroy(tool);
			ss->tablet_tools.erase(it);
			break;
		}
	}

	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on removed", (size_t)zwp_tablet_tool_v2));
}

void WaylandThread::_wp_tablet_tool_on_proximity_in(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t serial, struct zwp_tablet_v2 *tablet, struct wl_surface *surface) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	ss->tablet_tool_data_buffer.in_proximity = true;
	ss->pointer_enter_serial = serial;

	DEBUG_LOG_WAYLAND("Tablet tool entered window.");

	if (!ss->window_pointed) {
		Ref<WaylandThread::WindowEventMessage> msg;
		msg.instantiate();
		msg->event = DisplayServer::WINDOW_EVENT_MOUSE_ENTER;

		wls->wayland_thread->push_message(msg);
	}

	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on proximity in serial %d tablet %x surface %x", (size_t)zwp_tablet_tool_v2, serial, (size_t)tablet, (size_t)surface));
}

void WaylandThread::_wp_tablet_tool_on_proximity_out(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	ss->tablet_tool_data_buffer.in_proximity = false;

	DEBUG_LOG_WAYLAND("Tablet tool left window.");

	if (!ss->window_pointed) {
		Ref<WaylandThread::WindowEventMessage> msg;
		msg.instantiate();
		msg->event = DisplayServer::WINDOW_EVENT_MOUSE_EXIT;

		wls->wayland_thread->push_message(msg);
	}
	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on proximity out", (size_t)zwp_tablet_tool_v2));
}

void WaylandThread::_wp_tablet_tool_on_down(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t serial) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::TabletToolData &td = ss->tablet_tool_data_buffer;

	td.touching = true;
	td.pressed_button_mask.set_flag(mouse_button_to_mask(MouseButton::LEFT));
	td.last_button_pressed = MouseButton::LEFT;
	td.double_click_begun = true;

	// The protocol doesn't cover this, but we can use this funky hack to make
	// double clicking work.
	td.button_time = OS::get_singleton()->get_ticks_msec();

	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on down serial %x", (size_t)zwp_tablet_tool_v2, serial));
}

void WaylandThread::_wp_tablet_tool_on_up(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->tablet_tool_data_buffer.touching = false;
	ss->tablet_tool_data_buffer.pressed_button_mask.clear_flag(mouse_button_to_mask(MouseButton::LEFT));

	// The protocol doesn't cover this, but we can use this funky hack to make
	// double clicking work.
	ss->tablet_tool_data_buffer.button_time = OS::get_singleton()->get_ticks_msec();

	DEBUG_LOG_WAYLAND(vformat("wp tablet tool %x on up", (size_t)zwp_tablet_tool_v2));
}

void WaylandThread::_wp_tablet_tool_on_motion(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t x, wl_fixed_t y) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WindowState *ws = WaylandThread::wl_surface_get_window_state(ss->pointed_surface);
	ERR_FAIL_NULL(ws);

	int scale = WaylandThread::window_state_calculate_scale(ws);

	ss->tablet_tool_data_buffer.position.x = wl_fixed_to_double(x) * scale;
	ss->tablet_tool_data_buffer.position.y = wl_fixed_to_double(y) * scale;
}

void WaylandThread::_wp_tablet_tool_on_pressure(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t pressure) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->tablet_tool_data_buffer.pressure = pressure;
}

void WaylandThread::_wp_tablet_tool_on_distance(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t distance) {
	// Unsupported
}

void WaylandThread::_wp_tablet_tool_on_tilt(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t tilt_x, wl_fixed_t tilt_y) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->tablet_tool_data_buffer.tilt.x = wl_fixed_to_double(tilt_x);
	ss->tablet_tool_data_buffer.tilt.y = wl_fixed_to_double(tilt_y);
}

void WaylandThread::_wp_tablet_tool_on_rotation(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t degrees) {
	// Unsupported.
}

void WaylandThread::_wp_tablet_tool_on_slider(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, int32_t position) {
	// Unsupported.
}

void WaylandThread::_wp_tablet_tool_on_wheel(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t degrees, int32_t clicks) {
	// TODO
}

void WaylandThread::_wp_tablet_tool_on_button(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t serial, uint32_t button, uint32_t state) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::TabletToolData &td = ss->tablet_tool_data_buffer;

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

void WaylandThread::_wp_tablet_tool_on_frame(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t time) {
	WaylandThread::SeatState *ss = (WaylandThread::SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread::WaylandState *wls = ss->wls;
	ERR_FAIL_NULL(wls);

	_seat_state_set_current(*ss);

	WaylandThread::TabletToolData &old_td = ss->tablet_tool_data;
	WaylandThread::TabletToolData &td = ss->tablet_tool_data_buffer;

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
		// to the Z axis of the tablet, so it shouldn't go over 90 degrees, I think.
		// TODO: Investigate whether the tilt can go over 90 degrees (it shouldn't).
		mm->set_tilt(td.tilt / 90);

		// The tablet proto spec explicitly says that pressure is defined as a value
		// between 0 to 65535.
		mm->set_pressure(td.pressure / (float)65535);

		// FIXME: Tool handling is broken.
		mm->set_pen_inverted(td.is_eraser);

		mm->set_relative(td.position - old_td.position);

		// FIXME: I'm not sure whether accessing the Input singleton like this might
		// give problems.
		Input::get_singleton()->set_mouse_position(td.position);
		mm->set_velocity(Input::get_singleton()->get_last_mouse_velocity());

		Ref<WaylandThread::InputEventMessage> inputev_msg;
		inputev_msg.instantiate();

		inputev_msg->event = mm;

		wls->wayland_thread->push_message(inputev_msg);
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

				Ref<WaylandThread::InputEventMessage> msg;
				msg.instantiate();

				msg->event = mb;

				wls->wayland_thread->push_message(msg);
			}
		}
	}

	old_td = td;
}

void WaylandThread::_xdg_activation_token_on_done(void *data, struct xdg_activation_token_v1 *xdg_activation_token, const char *token) {
#if 0
	// TODO: Port to `WaylandThread` API.
	WindowState *ws = (WindowState *)state;
	ERR_FAIL_NULL(ws);

	ERR_FAIL_NULL(ws->wl_surface);

	xdg_activation_v1_activate(wls->globals.xdg_activation, token, ws->wl_surface);

	xdg_activation_token_v1_destroy(xdg_activation_token);

	DEBUG_LOG_WAYLAND(vformat("Received activation token and requested window activation."));
#endif
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

			wl_display_dispatch_pending(data->wl_display);
		}

		if (wl_display_flush(data->wl_display) == -1) {
			if (errno != EAGAIN) {
				print_error(vformat("Error %d while flushing the Wayland display.", errno));
				data->thread_done.set();
			}
		}

		// Wait for the event file descriptor to have new data.
		poll(&poll_fd, 1, -1);

		if (data->thread_done.is_set()) {
			wl_display_cancel_read(data->wl_display);
			break;
		}

		if (poll_fd.revents | POLLIN) {
			wl_display_read_events(data->wl_display);
		} else {
			wl_display_cancel_read(data->wl_display);
		}
	}
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
WaylandThread::ScreenState *WaylandThread::wl_output_get_screen_state(struct wl_output *p_output) {
	if (p_output && wl_proxy_is_godot((wl_proxy *)p_output)) {
		return (ScreenState *)wl_output_get_user_data(p_output);
	}

	return nullptr;
}

// NOTE: This method is the simplest way of accounting for dynamic output scale
// changes.
int WaylandThread::window_state_calculate_scale(WindowState *p_ws) {
	// TODO: Handle multiple screens (eg. two screens: one scale 2, one scale 1).

	// TODO: Cache value?
	ScreenState *ss = wl_output_get_screen_state(p_ws->wl_output);

	if (ss) {
		// NOTE: For some mystical reason, wl_output.done is emitted _after_ windows
		// get resized but the scale event gets sent _before_ that. I'm still leaning
		// towards the idea that rescaling when a window gets a resolution change is a
		// pretty good approach, but this means that we'll have to use the screen data
		// before it's "committed".
		// FIXME: Use the commited data.
		return ss->pending_data.scale;
	}

	return 1;
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

// TODO: Finish splitting.
void WaylandThread::window_create(DisplayServer::WindowID p_window_id, int p_width, int p_height) {
	// TODO: Implement multi-window support.
	WindowState &ws = main_window;

	ws.globals = &wls->globals;
	ws.wayland_thread = this;

	ws.rect.size.width = p_width;
	ws.rect.size.height = p_height;

	ws.wl_surface = wl_compositor_create_surface(wls->globals.wl_compositor);

	wl_proxy_tag_godot((struct wl_proxy *)ws.wl_surface);
	wl_surface_add_listener(ws.wl_surface, &wl_surface_listener, &ws);

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
		ws.xdg_surface = xdg_wm_base_get_xdg_surface(wls->globals.xdg_wm_base, ws.wl_surface);
		xdg_surface_add_listener(ws.xdg_surface, &xdg_surface_listener, &ws);

		ws.xdg_toplevel = xdg_surface_get_toplevel(ws.xdg_surface);
		xdg_toplevel_add_listener(ws.xdg_toplevel, &xdg_toplevel_listener, &ws);

		ws.xdg_toplevel_decoration = zxdg_decoration_manager_v1_get_toplevel_decoration(wls->globals.xdg_decoration_manager, ws.xdg_toplevel);
		zxdg_toplevel_decoration_v1_add_listener(ws.xdg_toplevel_decoration, &xdg_toplevel_decoration_listener, &ws);

		decorated = true;
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

void WaylandThread::window_resize(DisplayServer::WindowID p_window_id, Size2i p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	int scale = window_state_calculate_scale(&ws);

	ws.rect.size = p_size / scale;

	if (ws.wl_surface) {
		wl_surface_set_buffer_scale(ws.wl_surface, scale);

		if (ws.xdg_surface) {
			xdg_surface_set_window_geometry(ws.xdg_surface, 0, 0, p_size.width, p_size.height);
		}

		wl_surface_commit(ws.wl_surface);
	}

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		struct libdecor_state *state = libdecor_state_new(p_size.width, p_size.height);
		// I'm not sure whether we can just pass null here.
		libdecor_frame_commit(ws.libdecor_frame, state, nullptr);
		libdecor_state_free(state);
	}
#endif
}

void WaylandThread::window_set_max_size(DisplayServer::WindowID p_window_id, Size2i p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	Size2i logical_max_size = p_size / window_state_calculate_scale(&ws);

	if (ws.wl_surface && ws.xdg_toplevel) {
		xdg_toplevel_set_max_size(ws.xdg_toplevel, logical_max_size.width, logical_max_size.height);
		wl_surface_commit(ws.wl_surface);
	}

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		libdecor_frame_set_max_content_size(ws.libdecor_frame, logical_max_size.width, logical_max_size.height);
	}

	// FIXME: I'm not sure whether we have to commit the surface for this to apply.
#endif
}

void WaylandThread::window_set_min_size(DisplayServer::WindowID p_window_id, Size2i p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	Size2i logical_min_size = p_size / window_state_calculate_scale(&ws);

	if (ws.wl_surface && ws.xdg_toplevel) {
		xdg_toplevel_set_min_size(ws.xdg_toplevel, logical_min_size.width, logical_min_size.height);
		wl_surface_commit(ws.wl_surface);
	}

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		libdecor_frame_set_min_content_size(ws.libdecor_frame, logical_min_size.width, logical_min_size.height);
	}

	// FIXME: I'm not sure whether we have to commit the surface for this to apply.
#endif
}

bool WaylandThread::window_can_set_mode(DisplayServer::WindowID p_window_id_id, DisplayServer::WindowMode p_mode) const {
	// TODO: Use window IDs for multiwindow support.
	const WindowState &ws = main_window;

	switch (p_mode) {
		case DisplayServer::WINDOW_MODE_WINDOWED: {
			// Looks like it's guaranteed.
			return true;
		};

		case DisplayServer::WINDOW_MODE_MINIMIZED: {
			return ws.can_minimize;
		};

		case DisplayServer::WINDOW_MODE_MAXIMIZED: {
			return ws.can_maximize;
		};

		case DisplayServer::WINDOW_MODE_FULLSCREEN: {
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
		libdecor_frame_set_visibility(ws.libdecor_frame, !p_borderless);
	}
#endif
}

void WaylandThread::window_set_title(DisplayServer::WindowID p_window_id, String p_title) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame && p_title.utf8().ptr()) {
		libdecor_frame_set_title(ws.libdecor_frame, p_title.utf8().ptr());
	}
#endif // LIBDECOR_ENABLE

	if (ws.xdg_toplevel && p_title.utf8().ptr()) {
		xdg_toplevel_set_title(ws.xdg_toplevel, p_title.utf8().ptr());
	}
}

void WaylandThread::window_set_app_id(DisplayServer::WindowID p_window_id, String p_app_id) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

#ifdef LIBDECOR_ENABLED
	if (ws.libdecor_frame) {
		libdecor_frame_set_app_id(ws.libdecor_frame, p_app_id.utf8().ptrw());
		return;
	}
#endif

	if (ws.xdg_toplevel) {
		xdg_toplevel_set_app_id(ws.xdg_toplevel, p_app_id.utf8().ptrw());
		return;
	}
}

void WaylandThread::window_request_attention(DisplayServer::WindowID p_window_id) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	if (wls->globals.xdg_activation) {
		// Window attention requests are done through the XDG activation protocol.
		xdg_activation_token_v1 *xdg_activation_token = xdg_activation_v1_get_activation_token(wls->globals.xdg_activation);
		xdg_activation_token_v1_add_listener(xdg_activation_token, &xdg_activation_token_listener, &ws);
		xdg_activation_token_v1_commit(xdg_activation_token);
	}
}

void WaylandThread::window_set_idle_inhibition(DisplayServer::WindowID p_window_id, bool p_enable) {
	// TODO: Use window IDs for multiwindow support.
	WindowState &ws = main_window;

	if (p_enable) {
		if (ws.globals->wp_idle_inhibit_manager && !ws.wp_idle_inhibitor) {
			ERR_FAIL_COND(!ws.wl_surface);
			ws.wp_idle_inhibitor = zwp_idle_inhibit_manager_v1_create_inhibitor(ws.globals->wp_idle_inhibit_manager, ws.wl_surface);
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
	ERR_FAIL_INDEX_V(p_screen, wls->globals.wl_outputs.size(), ScreenData());

	return wl_output_get_screen_state(wls->globals.wl_outputs[p_screen])->data;
}

int WaylandThread::get_screen_count() const {
	return wls->globals.wl_outputs.size();
}

Error WaylandThread::init(WaylandThread::WaylandState &p_wls) {
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
	ERR_FAIL_COND_V_MSG(!wl_display, ERR_CANT_CREATE, "Can't connect to a Wayland display.");

	thread_data.wl_display = wl_display;

	// FIXME: Get rid of this.
	{
		wls = &p_wls;
		wls->wl_display = wl_display;
	}

	events_thread.start(_poll_events_thread, &thread_data);

	wls->wl_registry = wl_display_get_registry(wl_display);

	ERR_FAIL_COND_V_MSG(!wls->wl_registry, ERR_UNAVAILABLE, "Can't obtain the Wayland registry global.");

	wl_registry_add_listener(wls->wl_registry, &wl_registry_listener, wls);

	// Wait for globals to get notified from the compositor.
	wl_display_roundtrip(wl_display);

	WaylandGlobals &globals = wls->globals;

	ERR_FAIL_COND_V_MSG(!globals.wl_shm, ERR_UNAVAILABLE, "Can't obtain the Wayland shared memory global.");
	ERR_FAIL_COND_V_MSG(!globals.wl_compositor, ERR_UNAVAILABLE, "Can't obtain the Wayland compositor global.");
	ERR_FAIL_COND_V_MSG(!globals.wl_subcompositor, ERR_UNAVAILABLE, "Can't obtain the Wayland subcompositor global.");
	ERR_FAIL_COND_V_MSG(!globals.wl_data_device_manager, ERR_UNAVAILABLE, "Can't obtain the Wayland data device manager global.");
	ERR_FAIL_COND_V_MSG(!globals.wp_pointer_constraints, ERR_UNAVAILABLE, "Can't obtain the Wayland pointer constraints global.");
	ERR_FAIL_COND_V_MSG(!globals.xdg_wm_base, ERR_UNAVAILABLE, "Can't obtain the Wayland XDG shell global.");

	if (!globals.xdg_decoration_manager) {
#ifdef LIBDECOR_ENABLED
		WARN_PRINT("Can't obtain the XDG decoration manager. Libdecor will be used for drawing CSDs, if available.");
#else
		WARN_PRINT("Can't obtain the XDG decoration manager. Decorations won't show up.");
#endif // LIBDECOR_ENABLED
	}

	if (!globals.xdg_activation) {
		WARN_PRINT("Can't obtain the XDG activation global. Attention requesting won't work!");
	}

#ifndef DBUS_ENABLED
	if (!globals.wp_idle_inhibit_manager) {
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

	initialized = true;
	return OK;
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

	// TODO: Remove this.
	if (!wls) {
		return;
	}

	if (main_window.xdg_toplevel) {
		xdg_toplevel_destroy(main_window.xdg_toplevel);
	}

	if (main_window.xdg_surface) {
		xdg_surface_destroy(main_window.xdg_surface);
	}

	if (main_window.wl_surface) {
		wl_surface_destroy(main_window.wl_surface);
	}

	for (SeatState &seat : wls->seats) {
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

		if (seat.wp_tablet_seat) {
			zwp_tablet_seat_v2_destroy(seat.wp_tablet_seat);
		}

		for (struct zwp_tablet_tool_v2 *tool : seat.tablet_tools) {
			zwp_tablet_tool_v2_destroy(tool);
		}
	}

	for (struct wl_output *wl_output : wls->globals.wl_outputs) {
		ERR_FAIL_NULL(wl_output);

		memdelete(wl_output_get_screen_state(wl_output));
		wl_output_destroy(wl_output);
	}

	if (wls->wl_cursor_theme) {
		wl_cursor_theme_destroy(wls->wl_cursor_theme);
	}

	if (wls->globals.wp_idle_inhibit_manager) {
		zwp_idle_inhibit_manager_v1_destroy(wls->globals.wp_idle_inhibit_manager);
	}

	if (wls->globals.wp_pointer_constraints) {
		zwp_pointer_constraints_v1_destroy(wls->globals.wp_pointer_constraints);
	}

	if (wls->globals.wp_pointer_gestures) {
		zwp_pointer_gestures_v1_destroy(wls->globals.wp_pointer_gestures);
	}

	if (wls->globals.wp_relative_pointer_manager) {
		zwp_relative_pointer_manager_v1_destroy(wls->globals.wp_relative_pointer_manager);
	}

	if (wls->globals.xdg_activation) {
		xdg_activation_v1_destroy(wls->globals.xdg_activation);
	}

	if (wls->globals.xdg_decoration_manager) {
		zxdg_decoration_manager_v1_destroy(wls->globals.xdg_decoration_manager);
	}

	if (wls->globals.xdg_wm_base) {
		xdg_wm_base_destroy(wls->globals.xdg_wm_base);
	}

	if (wls->globals.wl_shm) {
		wl_shm_destroy(wls->globals.wl_shm);
	}

	if (wls->globals.wl_subcompositor) {
		wl_subcompositor_destroy(wls->globals.wl_subcompositor);
	}

	if (wls->globals.wl_compositor) {
		wl_compositor_destroy(wls->globals.wl_compositor);
	}

	if (wls->wl_registry) {
		wl_registry_destroy(wls->wl_registry);
	}

	if (wl_display) {
		wl_display_disconnect(wl_display);
	}
}

#endif // WAYLAND_ENABLED
