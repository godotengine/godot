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

// FIXME: This is an hack while I refactor everything. This is supposed to be
// separate from `DisplayServerWayland`.
using WaylandThread = DisplayServerWayland::WaylandThread;

// NOTE: Stuff like libdecor can (and will) register foreign proxies which
// aren't formatted as we like. This method is needed to detect whether a proxy
// has our tag. Also, be careful! The proxy has to be manually tagged or it
// won't be recognized.
bool WaylandThread::_wl_proxy_is_godot(struct wl_proxy *p_proxy) {
	ERR_FAIL_NULL_V(p_proxy, false);

	return wl_proxy_get_tag(p_proxy) == &proxy_tag;
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
		// The screen listener requires a pointer for its state data. For this
		// reason, to get one that points to a variable that can live outside of this
		// scope, we push a default `ScreenData` in `wls->screens` and get the
		// address of this new element.
		ScreenData *sd = &wls->screens.push_back({})->get();
		sd->wl_output = (struct wl_output *)wl_registry_bind(wl_registry, name, &wl_output_interface, 2);
		sd->wl_output_name = name;
		sd->wls = wls;

		wl_proxy_set_tag((struct wl_proxy *)sd->wl_output, &proxy_tag);
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
	WindowData *wd = (WindowData *)data;
	ScreenData *sd = (ScreenData *)wl_output_get_user_data(wl_output);

	if (!_wl_proxy_is_godot((struct wl_proxy *)wl_output)) {
		return;
	}

	wd->scale = sd->scale;

	wl_surface_set_buffer_scale(wl_surface, wd->scale);
	wl_surface_commit(wl_surface);
}

void WaylandThread::_wl_surface_on_leave(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output) {}

// NOTE: This must be started after a valid wl_display is loaded.
void WaylandThread::_poll_events_thread(void *p_data) {
	ThreadData *data = (ThreadData *)p_data;

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

// TODO: Finish splitting.
void WaylandThread::window_create(DisplayServer::Context p_context) {
	WindowData &wd = wls->main_window;

	wd.wl_surface = wl_compositor_create_surface(wls->globals.wl_compositor);
	wl_surface_add_listener(wd.wl_surface, &wl_surface_listener, &wd);

	bool decorated = false;

	// TODO: Put into `DisplayServerWayland` somehow.
	String app_id = _get_app_id_from_context(p_context);

#ifdef LIBDECOR_ENABLED
	if (!decorated && wls->libdecor_context) {
		wd.libdecor_frame = libdecor_decorate(wls->libdecor_context, wd.wl_surface, (struct libdecor_frame_interface *)&libdecor_frame_interface, &wd);

		libdecor_frame_set_max_content_size(wd.libdecor_frame, wd.max_size.width, wd.max_size.height);
		libdecor_frame_set_min_content_size(wd.libdecor_frame, wd.min_size.width, wd.max_size.height);
		libdecor_frame_set_app_id(wd.libdecor_frame, app_id.utf8().ptrw());

		libdecor_frame_map(wd.libdecor_frame);

		decorated = true;
	}
#endif

	if (!decorated) {
		// libdecor has failed loading or is disabled, we shall handle xdg_toplevel
		// creation and decoration ourselves (and by decorating for now I just mean
		// asking for SSDs and hoping for the best).
		wd.xdg_surface = xdg_wm_base_get_xdg_surface(wls->globals.xdg_wm_base, wd.wl_surface);
		xdg_surface_add_listener(wd.xdg_surface, &xdg_surface_listener, &wd);

		wd.xdg_toplevel = xdg_surface_get_toplevel(wd.xdg_surface);
		xdg_toplevel_add_listener(wd.xdg_toplevel, &xdg_toplevel_listener, &wd);

		xdg_toplevel_set_max_size(wd.xdg_toplevel, wd.max_size.width, wd.max_size.height);
		xdg_toplevel_set_min_size(wd.xdg_toplevel, wd.min_size.width, wd.min_size.height);
		xdg_toplevel_set_app_id(wd.xdg_toplevel, app_id.utf8().ptrw());

		wd.xdg_toplevel_decoration = zxdg_decoration_manager_v1_get_toplevel_decoration(wls->globals.xdg_decoration_manager, wd.xdg_toplevel);
		zxdg_toplevel_decoration_v1_add_listener(wd.xdg_toplevel_decoration, &xdg_toplevel_decoration_listener, &wd);

		decorated = true;
	}

	wl_surface_commit(wd.wl_surface);

	// Wait for the surface to be configured before continuing.
	wl_display_roundtrip(wls->wl_display);
}

void WaylandThread::window_resize(Size2i p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowData &wd = wls->main_window;

	wd.logical_rect.size = p_size / wd.scale;

	if (wd.wl_surface && wd.xdg_surface) {
		xdg_surface_set_window_geometry(wd.xdg_surface, 0, 0, p_size.width, p_size.height);
		wl_surface_commit(wd.wl_surface);
	}

#ifdef LIBDECOR_ENABLED
	if (wd.libdecor_frame) {
		struct libdecor_state *state = libdecor_state_new(p_size.width, p_size.height);
		// I'm not sure whether we can just pass null here.
		libdecor_frame_commit(wd.libdecor_frame, state, nullptr);
		libdecor_state_free(state);
	}
#endif
}

void WaylandThread::window_set_max_size(Size2i p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowData &wd = wls->main_window;

	Size2i logical_max_size = p_size / wd.scale;

	if (wd.wl_surface && wd.xdg_toplevel) {
		xdg_toplevel_set_max_size(wd.xdg_toplevel, logical_max_size.width, logical_max_size.height);
		wl_surface_commit(wd.wl_surface);
	}

#ifdef LIBDECOR_ENABLED
	if (wd.libdecor_frame) {
		libdecor_frame_set_max_content_size(wd.libdecor_frame, logical_max_size.width, logical_max_size.height);
	}

	// FIXME: I'm not sure whether we have to commit the surface for this to apply.
#endif
}

void WaylandThread::window_set_min_size(Size2i p_size) {
	// TODO: Use window IDs for multiwindow support.
	WindowData &wd = wls->main_window;

	Size2i logical_min_size = p_size / wd.scale;

	if (wd.wl_surface && wd.xdg_toplevel) {
		xdg_toplevel_set_min_size(wd.xdg_toplevel, logical_min_size.width, logical_min_size.height);
		wl_surface_commit(wd.wl_surface);
	}

#ifdef LIBDECOR_ENABLED
	if (wd.libdecor_frame) {
		libdecor_frame_set_min_content_size(wd.libdecor_frame, logical_min_size.width, logical_min_size.height);
	}

	// FIXME: I'm not sure whether we have to commit the surface for this to apply.
#endif
}

void WaylandThread::window_set_borderless(bool p_borderless) {
	// TODO: Use window IDs for multiwindow support.
	WindowData &wd = wls->main_window;

	if (wls->globals.xdg_decoration_manager && wd.xdg_toplevel_decoration) {
		if (p_borderless) {
			// We implement borderless windows by simply asking the compositor to let
			// us handle decorations (we don't).
			zxdg_toplevel_decoration_v1_set_mode(wd.xdg_toplevel_decoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_CLIENT_SIDE);
		} else {
			zxdg_toplevel_decoration_v1_set_mode(wd.xdg_toplevel_decoration, ZXDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
		}
	}

#ifdef LIBDECOR_ENABLED
	if (wd.libdecor_frame) {
		libdecor_frame_set_visibility(wd.libdecor_frame, !p_borderless);
	}
#endif
}

void WaylandThread::window_set_title(String p_title) {
	// TODO: Use window IDs for multiwindow support.
	WindowData &wd = wls->main_window;

#ifdef LIBDECOR_ENABLED
	if (wd.libdecor_frame && p_title.utf8().ptr()) {
		libdecor_frame_set_title(wd.libdecor_frame, wd.title.utf8().ptr());
	}
#endif // LIBDECOR_ENABLE

	if (wd.xdg_toplevel && p_title.utf8().ptr()) {
		xdg_toplevel_set_title(wd.xdg_toplevel, wd.title.utf8().ptr());
	}
}

void WaylandThread::window_request_attention() {
	// TODO: Use window IDs for multiwindow support.
	WindowData &wd = wls->main_window;

	if (wls->globals.xdg_activation) {
		// Window attention requests are done through the XDG activation protocol.
		xdg_activation_token_v1 *xdg_activation_token = xdg_activation_v1_get_activation_token(wls->globals.xdg_activation);
		xdg_activation_token_v1_add_listener(xdg_activation_token, &xdg_activation_token_listener, &wd);
		xdg_activation_token_v1_commit(xdg_activation_token);
	}
}

void WaylandThread::init(DisplayServerWayland::WaylandState &p_wls) {
#ifdef SOWRAP_ENABLED
#ifdef DEBUG_ENABLED
	int dylibloader_verbose = 1;
#else
	int dylibloader_verbose = 0;
#endif // DEBUG_ENABLED

	if (initialize_wayland_client(dylibloader_verbose) != 0) {
		WARN_PRINT("Can't load the Wayland client library.");
		return;
	}

	if (initialize_wayland_cursor(dylibloader_verbose) != 0) {
		WARN_PRINT("Can't load the Wayland cursor library.");
		return;
	}

	if (initialize_xkbcommon(dylibloader_verbose) != 0) {
		WARN_PRINT("Can't load the XKBcommon library.");
		return;
	}
#endif // SOWRAP_ENABLED

	KeyMappingXKB::initialize();

	// FIXME: Get rid of this.
	wls = &p_wls;

	wls->wl_display = wl_display_connect(nullptr);
	thread_data.wl_display = wls->wl_display;

	ERR_FAIL_COND_MSG(!wls->wl_display, "Can't connect to a Wayland display.");

	events_thread.start(_poll_events_thread, &thread_data);

	wls->wl_registry = wl_display_get_registry(wls->wl_display);

	ERR_FAIL_COND_MSG(!wls->wl_registry, "Can't obtain the Wayland registry global.");

	wl_registry_add_listener(wls->wl_registry, &wl_registry_listener, wls);

	// Wait for globals to get notified from the compositor.
	wl_display_roundtrip(wls->wl_display);

	WaylandGlobals &globals = wls->globals;

	ERR_FAIL_COND_MSG(!globals.wl_shm, "Can't obtain the Wayland shared memory global.");
	ERR_FAIL_COND_MSG(!globals.wl_compositor, "Can't obtain the Wayland compositor global.");
	ERR_FAIL_COND_MSG(!globals.wl_subcompositor, "Can't obtain the Wayland subcompositor global.");
	ERR_FAIL_COND_MSG(!globals.wl_data_device_manager, "Can't obtain the Wayland data device manager global.");
	ERR_FAIL_COND_MSG(!globals.wp_pointer_constraints, "Can't obtain the Wayland pointer constraints global.");
	ERR_FAIL_COND_MSG(!globals.xdg_wm_base, "Can't obtain the Wayland XDG shell global.");

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
	wl_display_roundtrip(wls->wl_display);

#ifdef LIBDECOR_ENABLED
	bool libdecor_found = true;

#ifdef SOWRAP_ENABLED
	if (initialize_libdecor(dylibloader_verbose) != 0) {
		libdecor_found = false;
	}
#endif // SOWRAP_ENABLED

	if (libdecor_found) {
		wls->libdecor_context = libdecor_new(wls->wl_display, (struct libdecor_interface *)&libdecor_interface);
	} else {
		print_verbose("libdecor not found. Client-side decorations disabled.");
	}
#endif // LIBDECOR_ENABLED
}

void WaylandThread::destroy() {
	if (wls->wl_display && events_thread.is_started()) {
		thread_data.thread_done.set();

		// By sending a roundtrip message we're unblocking the polling thread so that
		// it can realize that it's done and also handle every event that's left.
		wl_display_roundtrip(wls->wl_display);

		events_thread.wait_to_finish();
	}

	if (wls->main_window.xdg_toplevel) {
		xdg_toplevel_destroy(wls->main_window.xdg_toplevel);
	}

	if (wls->main_window.xdg_surface) {
		xdg_surface_destroy(wls->main_window.xdg_surface);
	}

	if (wls->main_window.wl_egl_window) {
		wl_egl_window_destroy(wls->main_window.wl_egl_window);
	}

	if (wls->main_window.wl_surface) {
		wl_surface_destroy(wls->main_window.wl_surface);
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

	for (ScreenData &screen : wls->screens) {
		if (screen.wl_output) {
			wl_output_destroy(screen.wl_output);
		}
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

	if (wls->wl_display) {
		wl_display_disconnect(wls->wl_display);
	}
}

#endif // WAYLAND_ENABLED
