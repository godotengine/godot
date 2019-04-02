/*************************************************************************/
/*  os_wayland.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "os_wayland.h"
#include "core/os/input.h"
#include "drivers/dummy/rasterizer_dummy.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "key_mapping_xkb.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"
#include <limits.h>
#include <linux/input-event-codes.h>
#include <sys/mman.h>

void OS_Wayland::registry_global(void *data, struct wl_registry *registry,
		uint32_t id, const char *interface, uint32_t version) {
	OS_Wayland *d_wl = (OS_Wayland *)data;

	if (strcmp(interface, wl_shm_interface.name) == 0) {
		d_wl->shm = (struct wl_shm *)wl_registry_bind(
				registry, id, &wl_shm_interface, 1);
	} else if (strcmp(interface, wl_compositor_interface.name) == 0) {
		d_wl->compositor = (struct wl_compositor *)wl_registry_bind(
				registry, id, &wl_compositor_interface, 3);
	} else if (strcmp(interface, wl_seat_interface.name) == 0) {
		d_wl->seat = (struct wl_seat *)wl_registry_bind(
				registry, id, &wl_seat_interface, 1);
		wl_seat_add_listener(d_wl->seat, &d_wl->seat_listener, d_wl);
	} else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		d_wl->xdgbase = (struct xdg_wm_base *)wl_registry_bind(
				registry, id, &xdg_wm_base_interface, 1);
	} else if (strcmp(interface, wl_output_interface.name) == 0) {
		struct wl_output *wl_output = (struct wl_output *)wl_registry_bind(
				registry, id, &wl_output_interface, 3);
		WaylandOutput *output = new WaylandOutput(d_wl, wl_output);
		d_wl->outputs.push_back(output);
		wl_output_add_listener(wl_output, &d_wl->output_listener, output);
	} else if (strcmp(interface, zwp_pointer_constraints_v1_interface.name) == 0) {
		d_wl->pointer_constraints = (struct zwp_pointer_constraints_v1 *)wl_registry_bind(
				registry, id, &zwp_pointer_constraints_v1_interface, 1);
	} else if (strcmp(interface, zwp_relative_pointer_manager_v1_interface.name) == 0) {
		d_wl->relative_pointer_manager = (struct zwp_relative_pointer_manager_v1 *)wl_registry_bind(
				registry, id, &zwp_relative_pointer_manager_v1_interface, 1);
	}
}

void OS_Wayland::registry_global_remove(void *data,
		struct wl_registry *wl_registry, uint32_t name) {
	// This space deliberately left blank
}

void OS_Wayland::update_scale_factor() {
	int max = 1;
	for (int i = 0; i < outputs.size(); ++i) {
		WaylandOutput *output = outputs[i];
		if (output->entered && output->scale > max) {
			max = output->scale;
		}
	}
	if (!is_hidpi_allowed()) {
		max = 1;
	}
	scale_factor = max;
	wl_surface_set_buffer_scale(surface, max);
	Size2 size = get_window_size();
	context_gl_egl->set_window_size(size.width * max, size.height * max);
	wl_egl_window_resize(egl_window, size.width * max, size.height * max, 0, 0);
}

void OS_Wayland::surface_enter(void *data, struct wl_surface *wl_surface,
		struct wl_output *wl_output) {
	WaylandOutput *output = (WaylandOutput *)wl_output_get_user_data(wl_output);
	output->entered = true;
	output->d_wl->update_scale_factor();
}

void OS_Wayland::surface_leave(void *data, struct wl_surface *wl_surface,
		struct wl_output *wl_output) {
	WaylandOutput *output = (WaylandOutput *)wl_output_get_user_data(wl_output);
	output->entered = false;
	output->d_wl->update_scale_factor();
}

void OS_Wayland::xdg_toplevel_configure(void *data,
		struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height,
		struct wl_array *states) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (!d_wl->context_gl_egl) {
		return;
	}
	d_wl->context_gl_egl->set_window_size(
			width * d_wl->scale_factor, height * d_wl->scale_factor);
	wl_egl_window_resize(d_wl->egl_window,
			width * d_wl->scale_factor, height * d_wl->scale_factor, 0, 0);
	d_wl->maximized = d_wl->fullscreen = d_wl->resizing =
			d_wl->activated = false;
	void *p;
	wl_array_for_each(p, states) {
		uint32_t *state = (uint32_t *)p;
		switch (*state) {
			case XDG_TOPLEVEL_STATE_MAXIMIZED:
				d_wl->maximized = true;
				break;
			case XDG_TOPLEVEL_STATE_FULLSCREEN:
				d_wl->fullscreen = true;
				break;
			case XDG_TOPLEVEL_STATE_RESIZING:
				d_wl->resizing = true;
				break;
			case XDG_TOPLEVEL_STATE_ACTIVATED:
				d_wl->activated = true;
				break;
			default:
				// This space deliberately left blank
				break;
		}
	}
}

void OS_Wayland::set_window_fullscreen(bool p_enabled) {
	if (!p_enabled) {
		xdg_toplevel_unset_fullscreen(xdgtoplevel);
		return;
	}
	if (desired_output != NULL) {
		xdg_toplevel_set_fullscreen(xdgtoplevel, desired_output->output);
	} else {
		xdg_toplevel_set_fullscreen(xdgtoplevel, NULL);
	}
}

void OS_Wayland::xdg_toplevel_close(void *data,
		struct xdg_toplevel *xdg_toplevel) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	d_wl->main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
}

void OS_Wayland::xdg_surface_configure(void *data,
		struct xdg_surface *xdg_surface, uint32_t serial) {
	xdg_surface_ack_configure(xdg_surface, serial);
}

void OS_Wayland::xdg_wm_base_ping(void *data,
		struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}

void OS_Wayland::seat_name(void *data,
		struct wl_seat *wl_seat, const char *name) {
	// This space deliberately left blank
}

void OS_Wayland::seat_capabilities(void *data, struct wl_seat *wl_seat,
		uint32_t capabilities) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
		d_wl->mouse_pointer = wl_seat_get_pointer(wl_seat);
		wl_pointer_add_listener(d_wl->mouse_pointer,
				&d_wl->pointer_listener, d_wl);
		if (d_wl->relative_pointer_manager) {
			d_wl->relative_pointer =
					zwp_relative_pointer_manager_v1_get_relative_pointer(
							d_wl->relative_pointer_manager, d_wl->mouse_pointer);
			zwp_relative_pointer_v1_add_listener(d_wl->relative_pointer,
					&d_wl->relative_pointer_listener, d_wl);
		}
	}
	if (capabilities & WL_SEAT_CAPABILITY_KEYBOARD) {
		struct wl_keyboard *keyboard;
		keyboard = wl_seat_get_keyboard(wl_seat);
		wl_keyboard_add_listener(keyboard, &d_wl->keyboard_listener, d_wl);
	}
	if (capabilities & WL_SEAT_CAPABILITY_TOUCH) {
		// TODO: touch support
	}
}

void OS_Wayland::pointer_enter(void *data,
		struct wl_pointer *wl_pointer, uint32_t serial,
		struct wl_surface *surface,
		wl_fixed_t surface_x, wl_fixed_t surface_y) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	Ref<InputEventMouseMotion> mm;
	mm.instance();
	float x = (float)wl_fixed_to_double(surface_x);
	float y = (float)wl_fixed_to_double(surface_y);
	Vector2 myVec = Vector2();
	myVec.x = x * d_wl->scale_factor;
	myVec.y = y * d_wl->scale_factor;
	d_wl->_mouse_pos = myVec;
	mm->set_position(d_wl->_mouse_pos);
	mm->set_global_position(d_wl->_mouse_pos);
	d_wl->input->parse_input_event(mm);
	if (d_wl->cursor_want != d_wl->cursor_have) {
		struct wl_surface *cursor_surface = NULL;
		if (d_wl->cursor_want != CURSOR_MAX) {
			cursor_surface = d_wl->cursor_surfaces[d_wl->cursor_want];
		}
		wl_pointer_set_cursor(d_wl->mouse_pointer, serial, cursor_surface, 0, 0);
		d_wl->cursor_have = d_wl->cursor_want;
		d_wl->cursor_serial = serial;
	}
}

void OS_Wayland::pointer_leave(void *data,
		struct wl_pointer *wl_pointer, uint32_t serial,
		struct wl_surface *surface) {
	// This space intentionally left blank
}

void OS_Wayland::pointer_motion(void *data,
		struct wl_pointer *wl_pointer, uint32_t time,
		wl_fixed_t surface_x, wl_fixed_t surface_y) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	Ref<InputEventMouseMotion> mm;
	mm.instance();
	float x = (float)wl_fixed_to_double(surface_x);
	float y = (float)wl_fixed_to_double(surface_y);
	Vector2 myVec = Vector2();
	myVec.x = x * d_wl->scale_factor;
	myVec.y = y * d_wl->scale_factor;
	mm->set_relative(myVec - d_wl->_mouse_pos);
	d_wl->_mouse_pos = myVec;
	mm->set_position(d_wl->_mouse_pos);
	mm->set_global_position(d_wl->_mouse_pos);
	d_wl->input->parse_input_event(mm);
}

void OS_Wayland::pointer_button(void *data,
		struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time,
		uint32_t button, uint32_t state) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	Ref<InputEventMouseButton> mb;
	mb.instance();
	mb->set_pressed(state == WL_POINTER_BUTTON_STATE_PRESSED);
	switch (button) {
		case BTN_LEFT:
			mb->set_button_index(BUTTON_LEFT);
			break;
		case BTN_RIGHT:
			mb->set_button_index(BUTTON_RIGHT);
			break;
		case BTN_MIDDLE:
			mb->set_button_index(BUTTON_MIDDLE);
			break;
		default:
			break;
	}
	mb->set_position(d_wl->_mouse_pos);
	mb->set_global_position(d_wl->_mouse_pos);
	d_wl->input->parse_input_event(mb);

	uint64_t diff = OS::get_singleton()->get_ticks_usec() / 1000 - d_wl->last_click_ms;
	if (mb->get_button_index() == d_wl->last_click_button_index) {
		auto dist = Point2(d_wl->last_click_pos).distance_to(d_wl->_mouse_pos);
		if (diff < 400 && dist < 5) {
			d_wl->last_click_ms = 0;
			d_wl->last_click_pos = Point2(-100, -100);
			d_wl->last_click_button_index = -1;
			mb->set_doubleclick(true);
		}
	} else if (mb->get_button_index() < 4 || mb->get_button_index() > 7) {
		d_wl->last_click_button_index = mb->get_button_index();
	}
	if (!mb->is_doubleclick()) {
		d_wl->last_click_ms += diff;
		d_wl->last_click_pos = d_wl->_mouse_pos;
	}
	d_wl->cursor_serial = serial;
}

void OS_Wayland::pointer_axis(void *data,
		struct wl_pointer *wl_pointer, uint32_t time,
		uint32_t axis, wl_fixed_t value) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	Ref<InputEventMouseButton> mb;
	mb.instance();
	double factor = wl_fixed_to_double(value);
	switch (axis) {
		case WL_POINTER_AXIS_VERTICAL_SCROLL:
			if (factor > 0) {
				mb->set_button_index(BUTTON_WHEEL_DOWN);
			} else {
				mb->set_button_index(BUTTON_WHEEL_UP);
			}
			break;
		case WL_POINTER_AXIS_HORIZONTAL_SCROLL:
			if (factor > 0) {
				mb->set_button_index(BUTTON_WHEEL_LEFT);
			} else {
				mb->set_button_index(BUTTON_WHEEL_RIGHT);
			}
			break;
		default:
			return; // Unknown axis
	}
	mb->set_position(d_wl->_mouse_pos);
	mb->set_global_position(d_wl->_mouse_pos);

	mb->set_pressed(true);
	d_wl->input->parse_input_event(mb);

	mb->set_pressed(false);
	d_wl->input->parse_input_event(mb);
}

void OS_Wayland::pointer_frame(void *data,
		struct wl_pointer *wl_pointer) {
	// TODO: Build GD input events over the course of several WL events, then
	// submit here
}

void OS_Wayland::pointer_axis_source(void *data,
		struct wl_pointer *wl_pointer, uint32_t axis_source) {
	// This space deliberately left blank
}

void OS_Wayland::pointer_axis_stop(void *data,
		struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis) {
	// This space deliberately left blank
}

void OS_Wayland::pointer_axis_discrete(void *data,
		struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete) {
	// This space deliberately left blank
}

void OS_Wayland::relative_pointer_relative_motion(void *data,
		struct zwp_relative_pointer_v1 *zwp_relative_pointer_v1,
		uint32_t utime_hi, uint32_t utime_lo,
		wl_fixed_t dx, wl_fixed_t dy,
		wl_fixed_t dx_unaccel, wl_fixed_t dy_unaccel) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (d_wl->get_mouse_mode() != MOUSE_MODE_CAPTURED) {
		return;
	}
	Ref<InputEventMouseMotion> mm;
	mm.instance();
	float x = (float)wl_fixed_to_double(dx);
	float y = (float)wl_fixed_to_double(dy);
	Vector2 myVec = Vector2();
	myVec.x = x;
	myVec.y = y;
	mm->set_relative(myVec);
	d_wl->_mouse_pos += myVec;
	mm->set_position(d_wl->_mouse_pos);
	mm->set_global_position(d_wl->_mouse_pos);
	d_wl->input->parse_input_event(mm);
	if (d_wl->cursor_want != d_wl->cursor_have) {
		struct wl_surface *cursor_surface = NULL;
		if (d_wl->cursor_want != CURSOR_MAX) {
			cursor_surface = d_wl->cursor_surfaces[d_wl->cursor_want];
		}
		wl_pointer_set_cursor(d_wl->mouse_pointer, d_wl->cursor_serial,
				cursor_surface, 0, 0);
		d_wl->cursor_have = d_wl->cursor_want;
	}
}

void OS_Wayland::_set_modifier_for_event(Ref<InputEventWithModifiers> ev) {
	ev->set_control(xkb_state_mod_name_is_active(xkb_state,
							XKB_MOD_NAME_CTRL, XKB_STATE_MODS_EFFECTIVE) > 0);
	ev->set_alt(xkb_state_mod_name_is_active(xkb_state,
						XKB_MOD_NAME_ALT, XKB_STATE_MODS_EFFECTIVE) > 0);
	ev->set_metakey(xkb_state_mod_name_is_active(xkb_state,
							XKB_MOD_NAME_LOGO, XKB_STATE_MODS_EFFECTIVE) > 0);
	ev->set_shift(xkb_state_mod_name_is_active(xkb_state,
						  XKB_MOD_NAME_SHIFT, XKB_STATE_MODS_EFFECTIVE) > 0);
}

void OS_Wayland::keyboard_keymap(void *data, struct wl_keyboard *wl_keyboard,
		uint32_t format, int32_t fd, uint32_t size) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	char *keymap_str = (char *)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
	xkb_keymap_unref(d_wl->xkb_keymap);
	d_wl->xkb_keymap = xkb_keymap_new_from_string(d_wl->xkb_context, keymap_str,
			XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
	munmap(keymap_str, size);
	close(fd);
	xkb_state_unref(d_wl->xkb_state);
	d_wl->xkb_state = xkb_state_new(d_wl->xkb_keymap);
}

void OS_Wayland::keyboard_enter(void *data, struct wl_keyboard *wl_keyboard,
		uint32_t serial, struct wl_surface *surface, struct wl_array *keys) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (d_wl->main_loop) {
		d_wl->main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
	}
}

void OS_Wayland::keyboard_leave(void *data, struct wl_keyboard *wl_keyboard,
		uint32_t serial, struct wl_surface *surface) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (d_wl->main_loop) {
		d_wl->main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	}
}

void OS_Wayland::keyboard_key(void *data, struct wl_keyboard *wl_keyboard,
		uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	xkb_keysym_t keysym = xkb_state_key_get_one_sym(d_wl->xkb_state, key + 8);
	uint32_t utf32 = xkb_keysym_to_utf32(keysym);
	Ref<InputEventKey> ev;
	ev.instance();
	ev->set_pressed(state == WL_KEYBOARD_KEY_STATE_PRESSED);
	ev->set_unicode(utf32);
	ev->set_scancode(KeyMappingXKB::get_keycode(keysym));
	d_wl->_set_modifier_for_event(ev);
	d_wl->input->parse_input_event(ev);
}

void OS_Wayland::keyboard_modifier(void *data, struct wl_keyboard *wl_keyboard,
		uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched,
		uint32_t mods_locked, uint32_t group) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	xkb_state_update_mask(d_wl->xkb_state, mods_depressed, mods_latched,
			mods_locked, 0, 0, group);
}

void OS_Wayland::keyboard_repeat_info(void *data,
		struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay) {
	// TODO
}

void OS_Wayland::output_geometry(void *data, struct wl_output *output,
		int32_t x, int32_t y,
		int32_t physical_width, int32_t physical_height,
		int32_t subpixel, const char *make, const char *model,
		int32_t transform) {
	// This space deliberately left blank
}

void OS_Wayland::output_mode(void *data, struct wl_output *wl_output,
		uint32_t flags, int32_t width, int32_t height, int32_t refresh) {
	WaylandOutput *output = (WaylandOutput *)data;
	VideoMode mode = VideoMode(width, height);
	output->modes.push_back(mode);
	if (flags & WL_OUTPUT_MODE_CURRENT) {
		output->current_video_mode = mode;
	}
}

OS::VideoMode OS_Wayland::get_video_mode(int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}
	return outputs[p_screen]->current_video_mode;
}

void OS_Wayland::get_fullscreen_mode_list(
		List<VideoMode> *p_list, int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}
	// Modesetting is not allowed on Wayland, so we only show the current mode
	p_list->push_back(outputs[p_screen]->current_video_mode);
}

void OS_Wayland::output_done(void *data, struct wl_output *output) {
	// This space deliberately left blank
}

void OS_Wayland::output_scale(void *data,
		struct wl_output *wl_output, int32_t factor) {
	WaylandOutput *output = (WaylandOutput *)data;
	output->scale = factor;
}

int OS_Wayland::get_current_screen() const {
	// Note: technically we can be on several screens at once.
	for (int i = 0; i < outputs.size(); ++i) {
		if (outputs[i]->entered) {
			return i;
		}
	}
	return -1;
}

void OS_Wayland::set_current_screen(int p_screen) {
	if (p_screen < 0 || p_screen >= outputs.size()) {
		return;
	}
	// Note, we can only choose an output if the window is full screen on
	// Wayland. We stash the output the user wants to use in case they
	// enter fullscreen later.
	desired_output = outputs[p_screen];
	if (!is_window_fullscreen()) {
		return;
	}
	xdg_toplevel_set_fullscreen(xdgtoplevel, desired_output->output);
}

Size2 OS_Wayland::get_screen_size(int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}
	if (p_screen < 0 || p_screen >= outputs.size()) {
		return Size2(0, 0);
	}
	VideoMode current = outputs[p_screen]->current_video_mode;
	return Size2(current.width, current.height);
}

int OS_Wayland::get_screen_dpi(int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}
	if (p_screen < 0 || p_screen >= outputs.size()) {
		return 72;
	}
	// 192 is the magic value at which the godot editor automatically scales
	// itself up. 96 is 192/2.
	return outputs[p_screen]->scale * 96;
}

OS_GenericUnix *OS_Wayland::detect() {
	if (!getenv("WAYLAND_DISPLAY") && !getenv("WAYLAND_SOCKET")) {
		return NULL;
	}
	return new OS_Wayland();
}

void OS_Wayland::_initialize_wl_display() {
	display = wl_display_connect(NULL);
	if (display == NULL) {
		print_line("Error: unable to connect to Wayland display");
		exit(1);
	}

	struct wl_registry *wl_registry = wl_display_get_registry(display);
	wl_registry_add_listener(wl_registry, &registry_listener, this);

	wl_display_dispatch(display);
	wl_display_roundtrip(display);
}

Error OS_Wayland::initialize_display(
		const VideoMode &p_desired, int p_video_driver) {
	_initialize_wl_display();

	main_loop = NULL;

	xkb_context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);

	if (compositor == NULL || xdgbase == NULL) {
		print_verbose("Error: Wayland compositor is missing required globals");
		exit(1);
	}

	surface = wl_compositor_create_surface(compositor);
	if (surface == NULL) {
		print_verbose("Error creating Wayland surface");
		exit(1);
	}
	wl_surface_add_listener(surface, &surface_listener, this);
	confine_region = wl_compositor_create_region(compositor);
	wl_region_add(confine_region, 0, 0, INT32_MAX, INT32_MAX);

	egl_window = wl_egl_window_create(surface,
			p_desired.width, p_desired.height);

	xdgsurface = xdg_wm_base_get_xdg_surface(xdgbase, surface);
	xdg_surface_add_listener(xdgsurface, &xdg_surface_listener, this);

	xdgtoplevel = xdg_surface_get_toplevel(xdgsurface);
	xdg_toplevel_add_listener(xdgtoplevel, &xdg_toplevel_listener, this);
	xdg_toplevel_set_title(xdgtoplevel, "Godot");
	wl_surface_commit(surface);

	xdg_wm_base_add_listener(xdgbase, &xdg_wm_base_listener, this);
	wl_display_roundtrip(display);

	if (egl_window == EGL_NO_SURFACE) {
		print_verbose("Error: unable to create EGL window");
		exit(1);
	}

	//TODO: check for possible context types
	ContextGL_EGL::Driver context_type = ContextGL_EGL::Driver::GLES_3_0;

	bool gl_initialization_error = false;
	while (!context_gl_egl) {
		EGLNativeDisplayType n_disp = (EGLNativeDisplayType)display;
		EGLNativeWindowType n_wind = (EGLNativeWindowType)egl_window;
		context_gl_egl = memnew(ContextGL_EGL(n_disp, n_wind, p_desired, context_type));
		if (context_gl_egl->initialize() != OK) {
			memdelete(context_gl_egl);
			context_gl_egl = NULL;
			if (GLOBAL_GET("rendering/quality/driver/driver_fallback") == "Best") {
				if (p_video_driver == VIDEO_DRIVER_GLES2) {
					gl_initialization_error = true;
					break;
				}

				p_video_driver = VIDEO_DRIVER_GLES2;
				context_type = ContextGL_EGL::GLES_2_0;
			} else {
				gl_initialization_error = true;
				break;
			}
		}
	}

	while (true) {
		if (context_type == ContextGL_EGL::GLES_3_0) {
			if (RasterizerGLES3::is_viable() == OK) {
				RasterizerGLES3::register_config();
				RasterizerGLES3::make_current();
				break;
			} else {
				if (GLOBAL_GET("rendering/quality/driver/driver_fallback") == "Best") {
					p_video_driver = VIDEO_DRIVER_GLES2;
					context_type = ContextGL_EGL::GLES_2_0;
					continue;
				} else {
					gl_initialization_error = true;
					break;
				}
			}
		}

		if (context_type == ContextGL_EGL::GLES_2_0) {
			if (RasterizerGLES2::is_viable() == OK) {
				RasterizerGLES2::register_config();
				RasterizerGLES2::make_current();
				break;
			} else {
				gl_initialization_error = true;
				break;
			}
		}
	}

	video_driver_index = p_video_driver;

	if (gl_initialization_error) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported OpenGL versions.\n"
								   "Please update your drivers or if you have a very old or integrated GPU upgrade it.",
				"Unable to initialize Video driver");
		return ERR_UNAVAILABLE;
	}

	glClearColor(1.0, 1.0, 0.0, 0.1);
	glClear(GL_COLOR_BUFFER_BIT);
	glFlush();

	swap_buffers();
	wl_display_dispatch(display);
	wl_display_roundtrip(display);

	visual_server = memnew(VisualServerRaster);

	visual_server->init();

	input = memnew(InputDefault);

	wl_display_dispatch(display);
	wl_display_roundtrip(display);

	// TODO: HiDPI cursors
	struct wl_cursor_theme *cursor_theme =
			wl_cursor_theme_load("default", 16, shm);
	if (cursor_theme) {
		for (int i = 0; i < CURSOR_MAX; i++) {
			static const char *cursor_names[] = {
				"left_ptr",
				"xterm",
				"hand2",
				"cross",
				"watch",
				"left_ptr_watch",
				"fleur",
				"hand1",
				"X_cursor",
				"sb_v_double_arrow",
				"sb_h_double_arrow",
				"size_bdiag",
				"size_fdiag",
				"hand1",
				"sb_v_double_arrow",
				"sb_h_double_arrow",
				"question_arrow"
			};

			struct wl_cursor *cur =
					wl_cursor_theme_get_cursor(cursor_theme, cursor_names[i]);
			if (!cur) {
				// left_ptr is baked into wayland-cursor and guaranteed to work
				cur = wl_cursor_theme_get_cursor(cursor_theme, "left_ptr");
			}
			// TODO: Animated cursors?
			struct wl_cursor_image *image = cur->images[0];
			struct wl_buffer *buffer = wl_cursor_image_get_buffer(image);
			struct wl_surface *surface =
					wl_compositor_create_surface(compositor);
			if (!buffer || !surface) {
				continue;
			}
			wl_surface_attach(surface, buffer,
					-image->hotspot_x, -image->hotspot_y);
			wl_surface_commit(surface);
			cursor_surfaces[i] = surface;
		}
		wl_display_dispatch(display);
		wl_display_roundtrip(display);
		wl_cursor_theme_destroy(cursor_theme);
	}

	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server,
				get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}

	return Error::OK;
}

void OS_Wayland::finalize_display() {
	delete_main_loop();
	wl_display_disconnect(display);
	for (int i = 0; i < outputs.size(); ++i) {
		free(outputs[i]);
	}
}

void OS_Wayland::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

void OS_Wayland::delete_main_loop() {
	if (main_loop)
		memdelete(main_loop);
	main_loop = NULL;
}

MainLoop *OS_Wayland::get_main_loop() const {
	return main_loop;
}

void OS_Wayland::set_mouse_mode(MouseMode p_mode) {
	struct wl_surface *cursor_surface = NULL;
	switch (p_mode) {
		case MOUSE_MODE_CONFINED:
			if (pointer_constraints != NULL && confined_pointer == NULL) {
				if (locked_pointer != NULL) {
					zwp_locked_pointer_v1_destroy(locked_pointer);
					locked_pointer = NULL;
				}
				confined_pointer = zwp_pointer_constraints_v1_confine_pointer(
						pointer_constraints, surface, mouse_pointer, confine_region,
						ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
			}
			/* Fallthrough */
		case MOUSE_MODE_VISIBLE:
			cursor_want = cursor_saved;
			if (cursor_want != CURSOR_MAX) {
				cursor_surface = cursor_surfaces[cursor_want];
			}
			// Best effort
			wl_pointer_set_cursor(mouse_pointer, cursor_serial, cursor_surface, 0, 0);
			break;
		case MOUSE_MODE_CAPTURED:
			if (pointer_constraints != NULL && locked_pointer == NULL) {
				if (confined_pointer != NULL) {
					zwp_confined_pointer_v1_destroy(confined_pointer);
					confined_pointer = NULL;
				}
				locked_pointer = zwp_pointer_constraints_v1_lock_pointer(
						pointer_constraints, surface, mouse_pointer, confine_region,
						ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT);
			}
			/* fallthrough */
		case MOUSE_MODE_HIDDEN:
			cursor_saved = cursor_want;
			cursor_want = CURSOR_MAX;
			// Best effort
			wl_pointer_set_cursor(mouse_pointer, cursor_serial, NULL, 0, 0);
			break;
	}
	mouse_mode = p_mode;
}

OS::MouseMode OS_Wayland::get_mouse_mode() const {
	return mouse_mode;
}

Point2 OS_Wayland::get_mouse_position() const {
	return _mouse_pos;
}

int OS_Wayland::get_mouse_button_state() const {
	print_line("not implemented (OS_Wayland): get_mouse_button_state");
	return 0;
}

void OS_Wayland::set_window_title(const String &p_title) {
	xdg_toplevel_set_title(xdgtoplevel, p_title.utf8().ptr());
}

Size2 OS_Wayland::get_window_size() const {
	return Size2(context_gl_egl->get_window_width(),
			context_gl_egl->get_window_height());
}

bool OS_Wayland::get_window_per_pixel_transparency_enabled() const {
	return false;
}

void OS_Wayland::set_window_per_pixel_transparency_enabled(bool p_enabled) {
	print_line("not implemented (OS_Wayland): set_window_per_pixel_transparency_enabled");
}

int OS_Wayland::get_video_driver_count() const {
	return 2;
}

const char *OS_Wayland::get_video_driver_name(int p_driver) const {
	switch (p_driver) {
		case VIDEO_DRIVER_GLES2:
			return "GLES2";
		case VIDEO_DRIVER_GLES3:
			return "GLES3";
		default:
			return "INVALID VIDEO DRIVER";
	}
}

int OS_Wayland::get_current_video_driver() const {
	return video_driver_index;
}

String OS_Wayland::get_name() {
	return String("Wayland");
}

bool OS_Wayland::can_draw() const {
	wl_display_dispatch_pending(display);
	return true;
}

void OS_Wayland::set_cursor_shape(CursorShape p_shape) {
	cursor_want = p_shape;
	// Make best-effort attempt to set it now
	wl_pointer_set_cursor(mouse_pointer, cursor_serial,
			cursor_surfaces[cursor_want], 0, 0);
}

void OS_Wayland::set_custom_mouse_cursor(const RES &p_cursor,
		CursorShape p_shape, const Vector2 &p_hotspot) {
	// TODO
}

void OS_Wayland::swap_buffers() {
#if defined(OPENGL_ENABLED)
	context_gl_egl->swap_buffers();
#endif
}

void OS_Wayland::set_icon(const Ref<Image> &p_icon) {
}

void OS_Wayland::process_events() {
	wl_display_dispatch(display);
}

OS_Wayland::WaylandOutput::WaylandOutput(
		OS_Wayland *d_wl, struct wl_output *output) {
	wl_output_set_user_data(output, this);
	this->d_wl = d_wl;
	this->output = output;
	this->scale = 1;
	this->entered = false;
}
