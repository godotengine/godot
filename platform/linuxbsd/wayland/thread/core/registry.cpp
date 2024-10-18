/**************************************************************************/
/*  registry.cpp                                                          */
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

void WaylandThread::_wl_registry_on_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version) {
	RegistryState *registry = (RegistryState *)data;
	ERR_FAIL_NULL(registry);

	if (strcmp(interface, wl_shm_interface.name) == 0) {
		registry->wl_shm = (struct wl_shm *)wl_registry_bind(wl_registry, name, &wl_shm_interface, 1);
		registry->wl_shm_name = name;
		return;
	}

	if (strcmp(interface, zxdg_exporter_v1_interface.name) == 0) {
		registry->xdg_exporter = (struct zxdg_exporter_v1 *)wl_registry_bind(wl_registry, name, &zxdg_exporter_v1_interface, 1);
		registry->xdg_exporter_name = name;
		return;
	}

	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		registry->wl_compositor = (struct wl_compositor *)wl_registry_bind(wl_registry, name, &wl_compositor_interface, CLAMP((int)version, 1, 6));
		registry->wl_compositor_name = name;
		return;
	}

	if (strcmp(interface, wl_data_device_manager_interface.name) == 0) {
		registry->wl_data_device_manager = (struct wl_data_device_manager *)wl_registry_bind(wl_registry, name, &wl_data_device_manager_interface, CLAMP((int)version, 1, 3));
		registry->wl_data_device_manager_name = name;

		// This global creates some seat data. Let's do that for the ones already available.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (ss->wl_data_device == nullptr) {
				ss->wl_data_device = wl_data_device_manager_get_data_device(registry->wl_data_device_manager, wl_seat);
				wl_data_device_add_listener(ss->wl_data_device, &wl_data_device_listener, ss);
			}
		}
		return;
	}

	if (strcmp(interface, wl_output_interface.name) == 0) {
		struct wl_output *wl_output = (struct wl_output *)wl_registry_bind(wl_registry, name, &wl_output_interface, CLAMP((int)version, 1, 4));
		wl_proxy_tag_godot((struct wl_proxy *)wl_output);

		registry->wl_outputs.push_back(wl_output);

		ScreenState *ss = memnew(ScreenState);
		ss->wl_output_name = name;
		ss->wayland_thread = registry->wayland_thread;

		wl_proxy_tag_godot((struct wl_proxy *)wl_output);
		wl_output_add_listener(wl_output, &wl_output_listener, ss);
		return;
	}

	if (strcmp(interface, wl_seat_interface.name) == 0) {
		struct wl_seat *wl_seat = (struct wl_seat *)wl_registry_bind(wl_registry, name, &wl_seat_interface, CLAMP((int)version, 1, 9));
		wl_proxy_tag_godot((struct wl_proxy *)wl_seat);

		SeatState *ss = memnew(SeatState);
		ss->wl_seat = wl_seat;
		ss->wl_seat_name = name;

		ss->registry = registry;
		ss->wayland_thread = registry->wayland_thread;

		// Some extra stuff depends on other globals. We'll initialize them if the
		// globals are already there, otherwise we'll have to do that once and if they
		// get announced.
		//
		// NOTE: Don't forget to also bind/destroy with the respective global.
		if (!ss->wl_data_device && registry->wl_data_device_manager) {
			// Clipboard & DnD.
			ss->wl_data_device = wl_data_device_manager_get_data_device(registry->wl_data_device_manager, wl_seat);
			wl_data_device_add_listener(ss->wl_data_device, &wl_data_device_listener, ss);
		}

		if (!ss->wp_primary_selection_device && registry->wp_primary_selection_device_manager) {
			// Primary selection.
			ss->wp_primary_selection_device = zwp_primary_selection_device_manager_v1_get_device(registry->wp_primary_selection_device_manager, wl_seat);
			zwp_primary_selection_device_v1_add_listener(ss->wp_primary_selection_device, &wp_primary_selection_device_listener, ss);
		}

		if (!ss->wp_tablet_seat && registry->wp_tablet_manager) {
			// Tablet.
			ss->wp_tablet_seat = zwp_tablet_manager_v2_get_tablet_seat(registry->wp_tablet_manager, wl_seat);
			zwp_tablet_seat_v2_add_listener(ss->wp_tablet_seat, &wp_tablet_seat_listener, ss);
		}

		if (!ss->wp_text_input && registry->wp_text_input_manager) {
			// IME.
			ss->wp_text_input = zwp_text_input_manager_v3_get_text_input(registry->wp_text_input_manager, wl_seat);
			zwp_text_input_v3_add_listener(ss->wp_text_input, &wp_text_input_listener, ss);
		}

		registry->wl_seats.push_back(wl_seat);

		wl_seat_add_listener(wl_seat, &wl_seat_listener, ss);

		if (registry->wayland_thread->wl_seat_current == nullptr) {
			registry->wayland_thread->_set_current_seat(wl_seat);
		}

		return;
	}

	if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		registry->xdg_wm_base = (struct xdg_wm_base *)wl_registry_bind(wl_registry, name, &xdg_wm_base_interface, CLAMP((int)version, 1, 6));
		registry->xdg_wm_base_name = name;

		xdg_wm_base_add_listener(registry->xdg_wm_base, &xdg_wm_base_listener, nullptr);
		return;
	}

	if (strcmp(interface, wp_viewporter_interface.name) == 0) {
		registry->wp_viewporter = (struct wp_viewporter *)wl_registry_bind(wl_registry, name, &wp_viewporter_interface, 1);
		registry->wp_viewporter_name = name;
	}

	if (strcmp(interface, wp_fractional_scale_manager_v1_interface.name) == 0) {
		registry->wp_fractional_scale_manager = (struct wp_fractional_scale_manager_v1 *)wl_registry_bind(wl_registry, name, &wp_fractional_scale_manager_v1_interface, 1);
		registry->wp_fractional_scale_manager_name = name;

		// NOTE: We're not mapping the fractional scale object here because this is
		// supposed to be a "startup global". If for some reason this isn't true (who
		// knows), add a conditional branch for creating the add-on object.
	}

	if (strcmp(interface, zxdg_decoration_manager_v1_interface.name) == 0) {
		registry->xdg_decoration_manager = (struct zxdg_decoration_manager_v1 *)wl_registry_bind(wl_registry, name, &zxdg_decoration_manager_v1_interface, 1);
		registry->xdg_decoration_manager_name = name;
		return;
	}

	if (strcmp(interface, xdg_activation_v1_interface.name) == 0) {
		registry->xdg_activation = (struct xdg_activation_v1 *)wl_registry_bind(wl_registry, name, &xdg_activation_v1_interface, 1);
		registry->xdg_activation_name = name;
		return;
	}

	if (strcmp(interface, zwp_primary_selection_device_manager_v1_interface.name) == 0) {
		registry->wp_primary_selection_device_manager = (struct zwp_primary_selection_device_manager_v1 *)wl_registry_bind(wl_registry, name, &zwp_primary_selection_device_manager_v1_interface, 1);

		// This global creates some seat data. Let's do that for the ones already available.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (!ss->wp_primary_selection_device && registry->wp_primary_selection_device_manager) {
				ss->wp_primary_selection_device = zwp_primary_selection_device_manager_v1_get_device(registry->wp_primary_selection_device_manager, wl_seat);
				zwp_primary_selection_device_v1_add_listener(ss->wp_primary_selection_device, &wp_primary_selection_device_listener, ss);
			}
		}
	}

	if (strcmp(interface, zwp_relative_pointer_manager_v1_interface.name) == 0) {
		registry->wp_relative_pointer_manager = (struct zwp_relative_pointer_manager_v1 *)wl_registry_bind(wl_registry, name, &zwp_relative_pointer_manager_v1_interface, 1);
		registry->wp_relative_pointer_manager_name = name;
		return;
	}

	if (strcmp(interface, zwp_pointer_constraints_v1_interface.name) == 0) {
		registry->wp_pointer_constraints = (struct zwp_pointer_constraints_v1 *)wl_registry_bind(wl_registry, name, &zwp_pointer_constraints_v1_interface, 1);
		registry->wp_pointer_constraints_name = name;
		return;
	}

	if (strcmp(interface, zwp_pointer_gestures_v1_interface.name) == 0) {
		registry->wp_pointer_gestures = (struct zwp_pointer_gestures_v1 *)wl_registry_bind(wl_registry, name, &zwp_pointer_gestures_v1_interface, 1);
		registry->wp_pointer_gestures_name = name;
		return;
	}

	if (strcmp(interface, zwp_idle_inhibit_manager_v1_interface.name) == 0) {
		registry->wp_idle_inhibit_manager = (struct zwp_idle_inhibit_manager_v1 *)wl_registry_bind(wl_registry, name, &zwp_idle_inhibit_manager_v1_interface, 1);
		registry->wp_idle_inhibit_manager_name = name;
		return;
	}

	if (strcmp(interface, zwp_tablet_manager_v2_interface.name) == 0) {
		registry->wp_tablet_manager = (struct zwp_tablet_manager_v2 *)wl_registry_bind(wl_registry, name, &zwp_tablet_manager_v2_interface, 1);
		registry->wp_tablet_manager_name = name;

		// This global creates some seat data. Let's do that for the ones already available.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			ss->wp_tablet_seat = zwp_tablet_manager_v2_get_tablet_seat(registry->wp_tablet_manager, wl_seat);
			zwp_tablet_seat_v2_add_listener(ss->wp_tablet_seat, &wp_tablet_seat_listener, ss);
		}

		return;
	}

	if (strcmp(interface, zwp_text_input_manager_v3_interface.name) == 0) {
		registry->wp_text_input_manager = (struct zwp_text_input_manager_v3 *)wl_registry_bind(wl_registry, name, &zwp_text_input_manager_v3_interface, 1);
		registry->wp_text_input_manager_name = name;

		// This global creates some seat data. Let's do that for the ones already available.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			ss->wp_text_input = zwp_text_input_manager_v3_get_text_input(registry->wp_text_input_manager, wl_seat);
			zwp_text_input_v3_add_listener(ss->wp_text_input, &wp_text_input_listener, ss);
		}

		return;
	}
}

void WaylandThread::_wl_registry_on_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name) {
	RegistryState *registry = (RegistryState *)data;
	ERR_FAIL_NULL(registry);

	if (name == registry->wl_shm_name) {
		if (registry->wl_shm) {
			wl_shm_destroy(registry->wl_shm);
			registry->wl_shm = nullptr;
		}

		registry->wl_shm_name = 0;

		return;
	}

	if (name == registry->xdg_exporter_name) {
		if (registry->xdg_exporter) {
			zxdg_exporter_v1_destroy(registry->xdg_exporter);
			registry->xdg_exporter = nullptr;
		}

		registry->xdg_exporter_name = 0;

		return;
	}

	if (name == registry->wl_compositor_name) {
		if (registry->wl_compositor) {
			wl_compositor_destroy(registry->wl_compositor);
			registry->wl_compositor = nullptr;
		}

		registry->wl_compositor_name = 0;

		return;
	}

	if (name == registry->wl_data_device_manager_name) {
		if (registry->wl_data_device_manager) {
			wl_data_device_manager_destroy(registry->wl_data_device_manager);
			registry->wl_data_device_manager = nullptr;
		}

		registry->wl_data_device_manager_name = 0;

		// This global is used to create some seat data. Let's clean it.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (ss->wl_data_device) {
				wl_data_device_destroy(ss->wl_data_device);
				ss->wl_data_device = nullptr;
			}

			ss->wl_data_device = nullptr;
		}

		return;
	}

	if (name == registry->xdg_wm_base_name) {
		if (registry->xdg_wm_base) {
			xdg_wm_base_destroy(registry->xdg_wm_base);
			registry->xdg_wm_base = nullptr;
		}

		registry->xdg_wm_base_name = 0;

		return;
	}

	if (name == registry->wp_viewporter_name) {
		WindowState *ws = &registry->wayland_thread->main_window;

		if (registry->wp_viewporter) {
			wp_viewporter_destroy(registry->wp_viewporter);
			registry->wp_viewporter = nullptr;
		}

		if (ws->wp_viewport) {
			wp_viewport_destroy(ws->wp_viewport);
			ws->wp_viewport = nullptr;
		}

		registry->wp_viewporter_name = 0;

		return;
	}

	if (name == registry->wp_fractional_scale_manager_name) {
		WindowState *ws = &registry->wayland_thread->main_window;

		if (registry->wp_fractional_scale_manager) {
			wp_fractional_scale_manager_v1_destroy(registry->wp_fractional_scale_manager);
			registry->wp_fractional_scale_manager = nullptr;
		}

		if (ws->wp_fractional_scale) {
			wp_fractional_scale_v1_destroy(ws->wp_fractional_scale);
			ws->wp_fractional_scale = nullptr;
		}

		registry->wp_fractional_scale_manager_name = 0;
	}

	if (name == registry->xdg_decoration_manager_name) {
		if (registry->xdg_decoration_manager) {
			zxdg_decoration_manager_v1_destroy(registry->xdg_decoration_manager);
			registry->xdg_decoration_manager = nullptr;
		}

		registry->xdg_decoration_manager_name = 0;

		return;
	}

	if (name == registry->xdg_activation_name) {
		if (registry->xdg_activation) {
			xdg_activation_v1_destroy(registry->xdg_activation);
			registry->xdg_activation = nullptr;
		}

		registry->xdg_activation_name = 0;

		return;
	}

	if (name == registry->wp_primary_selection_device_manager_name) {
		if (registry->wp_primary_selection_device_manager) {
			zwp_primary_selection_device_manager_v1_destroy(registry->wp_primary_selection_device_manager);
			registry->wp_primary_selection_device_manager = nullptr;
		}

		registry->wp_primary_selection_device_manager_name = 0;

		// This global is used to create some seat data. Let's clean it.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (ss->wp_primary_selection_device) {
				zwp_primary_selection_device_v1_destroy(ss->wp_primary_selection_device);
				ss->wp_primary_selection_device = nullptr;
			}

			if (ss->wp_primary_selection_source) {
				zwp_primary_selection_source_v1_destroy(ss->wp_primary_selection_source);
				ss->wp_primary_selection_source = nullptr;
			}

			if (ss->wp_primary_selection_offer) {
				memfree(wp_primary_selection_offer_get_offer_state(ss->wp_primary_selection_offer));
				zwp_primary_selection_offer_v1_destroy(ss->wp_primary_selection_offer);
				ss->wp_primary_selection_offer = nullptr;
			}
		}

		return;
	}

	if (name == registry->wp_relative_pointer_manager_name) {
		if (registry->wp_relative_pointer_manager) {
			zwp_relative_pointer_manager_v1_destroy(registry->wp_relative_pointer_manager);
			registry->wp_relative_pointer_manager = nullptr;
		}

		registry->wp_relative_pointer_manager_name = 0;

		// This global is used to create some seat data. Let's clean it.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (ss->wp_relative_pointer) {
				zwp_relative_pointer_v1_destroy(ss->wp_relative_pointer);
				ss->wp_relative_pointer = nullptr;
			}
		}

		return;
	}

	if (name == registry->wp_pointer_constraints_name) {
		if (registry->wp_pointer_constraints) {
			zwp_pointer_constraints_v1_destroy(registry->wp_pointer_constraints);
			registry->wp_pointer_constraints = nullptr;
		}

		registry->wp_pointer_constraints_name = 0;

		// This global is used to create some seat data. Let's clean it.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (ss->wp_relative_pointer) {
				zwp_relative_pointer_v1_destroy(ss->wp_relative_pointer);
				ss->wp_relative_pointer = nullptr;
			}

			if (ss->wp_locked_pointer) {
				zwp_locked_pointer_v1_destroy(ss->wp_locked_pointer);
				ss->wp_locked_pointer = nullptr;
			}

			if (ss->wp_confined_pointer) {
				zwp_confined_pointer_v1_destroy(ss->wp_confined_pointer);
				ss->wp_confined_pointer = nullptr;
			}
		}

		return;
	}

	if (name == registry->wp_pointer_gestures_name) {
		if (registry->wp_pointer_gestures) {
			zwp_pointer_gestures_v1_destroy(registry->wp_pointer_gestures);
		}

		registry->wp_pointer_gestures = nullptr;
		registry->wp_pointer_gestures_name = 0;

		// This global is used to create some seat data. Let's clean it.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (ss->wp_pointer_gesture_pinch) {
				zwp_pointer_gesture_pinch_v1_destroy(ss->wp_pointer_gesture_pinch);
				ss->wp_pointer_gesture_pinch = nullptr;
			}
		}

		return;
	}

	if (name == registry->wp_idle_inhibit_manager_name) {
		if (registry->wp_idle_inhibit_manager) {
			zwp_idle_inhibit_manager_v1_destroy(registry->wp_idle_inhibit_manager);
			registry->wp_idle_inhibit_manager = nullptr;
		}

		registry->wp_idle_inhibit_manager_name = 0;

		return;
	}

	if (name == registry->wp_tablet_manager_name) {
		if (registry->wp_tablet_manager) {
			zwp_tablet_manager_v2_destroy(registry->wp_tablet_manager);
			registry->wp_tablet_manager = nullptr;
		}

		registry->wp_tablet_manager_name = 0;

		// This global is used to create some seat data. Let's clean it.
		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			for (struct zwp_tablet_tool_v2 *tool : ss->tablet_tools) {
				TabletToolState *state = wp_tablet_tool_get_state(tool);
				if (state) {
					memdelete(state);
				}

				zwp_tablet_tool_v2_destroy(tool);
			}

			ss->tablet_tools.clear();
		}

		return;
	}

	if (name == registry->wp_text_input_manager_name) {
		if (registry->wp_text_input_manager) {
			zwp_text_input_manager_v3_destroy(registry->wp_text_input_manager);
			registry->wp_text_input_manager = nullptr;
		}

		registry->wp_text_input_manager_name = 0;

		for (struct wl_seat *wl_seat : registry->wl_seats) {
			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			zwp_text_input_v3_destroy(ss->wp_text_input);
			ss->wp_text_input = nullptr;
		}

		return;
	}

	{
		// Iterate through all of the seats to find if any got removed.
		List<struct wl_seat *>::Element *E = registry->wl_seats.front();
		while (E) {
			struct wl_seat *wl_seat = E->get();
			List<struct wl_seat *>::Element *N = E->next();

			SeatState *ss = wl_seat_get_seat_state(wl_seat);
			ERR_FAIL_NULL(ss);

			if (ss->wl_seat_name == name) {
				if (wl_seat) {
					wl_seat_destroy(wl_seat);
				}

				if (ss->wl_data_device) {
					wl_data_device_destroy(ss->wl_data_device);
				}

				if (ss->wp_tablet_seat) {
					zwp_tablet_seat_v2_destroy(ss->wp_tablet_seat);

					for (struct zwp_tablet_tool_v2 *tool : ss->tablet_tools) {
						TabletToolState *state = wp_tablet_tool_get_state(tool);
						if (state) {
							memdelete(state);
						}

						zwp_tablet_tool_v2_destroy(tool);
					}
				}

				memdelete(ss);

				registry->wl_seats.erase(E);
				return;
			}

			E = N;
		}
	}

	{
		// Iterate through all of the outputs to find if any got removed.
		// FIXME: This is a very bruteforce approach.
		List<struct wl_output *>::Element *it = registry->wl_outputs.front();
		while (it) {
			// Iterate through all of the screens to find if any got removed.
			struct wl_output *wl_output = it->get();
			ERR_FAIL_NULL(wl_output);

			ScreenState *ss = wl_output_get_screen_state(wl_output);

			if (ss->wl_output_name == name) {
				registry->wl_outputs.erase(it);

				memdelete(ss);
				wl_output_destroy(wl_output);

				return;
			}

			it = it->next();
		}
	}
}
