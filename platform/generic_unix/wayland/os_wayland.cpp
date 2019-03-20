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
#include "drivers/dummy/rasterizer_dummy.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"
#include <linux/input-event-codes.h>

void OS_Wayland::registry_global(void *data, struct wl_registry *registry,
		uint32_t id, const char *interface, uint32_t version) {
	OS_Wayland *d_wl = (OS_Wayland *)data;

	if (strcmp(interface, "wl_compositor") == 0) {
		d_wl->compositor = (wl_compositor *)wl_registry_bind(
				registry, id, &wl_compositor_interface, 1);
	}
	else if (strcmp(interface, "wl_seat") == 0) {
		d_wl->seat = (wl_seat *)wl_registry_bind(
				registry, id, &wl_seat_interface, 1);
		wl_seat_add_listener(d_wl->seat, &d_wl->seat_listener, d_wl);
	}
	else if (strcmp(interface, "xdg_wm_base") == 0) {
		d_wl->xdgbase = (xdg_wm_base *)wl_registry_bind(
				registry, id, &xdg_wm_base_interface, 1);
	}
}

void OS_Wayland::registry_global_remove(void *data,
		struct wl_registry *wl_registry, uint32_t name) {
	// This space deliberately left blank
}

void OS_Wayland::xdg_toplevel_configure_handler(void *data,
		struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height,
		struct wl_array *states) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	d_wl->context_gl_egl->set_window_size(width, height);
	wl_egl_window_resize(d_wl->egl_window, width, height, 0, 0);
}

void OS_Wayland::xdg_toplevel_close_handler(void *data,
		struct xdg_toplevel *xdg_toplevel) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	d_wl->main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
}

void OS_Wayland::xdg_surface_configure_handler(void *data,
		struct xdg_surface *xdg_surface, uint32_t serial) {
	// TODO: surface configure
	xdg_surface_ack_configure(xdg_surface, serial);
}

void OS_Wayland::xdg_ping_handler(void *data,
		struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}

void OS_Wayland::seat_name_handler(void *data,
		struct wl_seat *wl_seat, const char *name) {
	// This space deliberately left blank
}

void OS_Wayland::seat_capabilities_handler(void *data, struct wl_seat *wl_seat,
		uint32_t capabilities) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
		d_wl->mouse_pointer = wl_seat_get_pointer(wl_seat);
		wl_pointer_add_listener(d_wl->mouse_pointer, &d_wl->pointer_listener, d_wl);
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

void OS_Wayland::pointer_enter_handler(void *data,
		struct wl_pointer *wl_pointer, uint32_t serial,
		struct wl_surface *surface,
		wl_fixed_t surface_x, wl_fixed_t surface_y) {
}

void OS_Wayland::pointer_leave_handler(void *data,
		struct wl_pointer *wl_pointer, uint32_t serial,
		struct wl_surface *surface) {
}

void OS_Wayland::pointer_motion_handler(void *data,
		struct wl_pointer *wl_pointer, uint32_t time,
		wl_fixed_t surface_x, wl_fixed_t surface_y) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	Ref<InputEventMouseMotion> mm;
	mm.instance();
	float x = (float)wl_fixed_to_double(surface_x);
	float y = (float)wl_fixed_to_double(surface_y);
	Vector2 myVec = Vector2();
	myVec.x = x;
	myVec.y = y;
	d_wl->_mouse_pos = myVec;
	mm->set_position(d_wl->_mouse_pos);
	mm->set_global_position(d_wl->_mouse_pos);
	d_wl->input->parse_input_event(mm);
}

void OS_Wayland::pointer_button_handler(void *data,
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
}

void OS_Wayland::pointer_axis_handler(void *data,
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

void OS_Wayland::pointer_frame_handler(void *data,
		struct wl_pointer *wl_pointer) {
	// TODO: Build GD input events over the course of several WL events, then
	// submit here
}

void OS_Wayland::pointer_axis_source_handler(void *data,
		struct wl_pointer *wl_pointer, uint32_t axis_source) {
	// This space deliberately left blank
}

void OS_Wayland::pointer_axis_stop_handler(void *data,
		struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis) {
	// This space deliberately left blank
}

void OS_Wayland::pointer_axis_discrete_handler(void *data,
		struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete) {
	// This space deliberately left blank
}

void OS_Wayland::keyboard_keymap_handler(void *data,
		struct wl_keyboard *wl_keyboard, uint32_t format, int32_t fd, uint32_t size) {
	// TODO
}

void OS_Wayland::keyboard_enter_handler(void *data,
		struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (d_wl->main_loop) {
		d_wl->main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
	}
}

void OS_Wayland::keyboard_leave_handler(void *data,
		struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface) {
	OS_Wayland *d_wl = (OS_Wayland *)data;
	if (d_wl->main_loop) {
		d_wl->main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	}
}

void OS_Wayland::keyboard_key_handler(void *data,
		struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
	// TODO
}

void OS_Wayland::keyboard_modifier_handler(void *data,
		struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
	// TODO
}

void OS_Wayland::keyboard_repeat_info_handler(void *data,
		struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay) {
	// TODO
}

void OS_Wayland::_initialize_wl_display() {
	display = NULL;
	display = wl_display_connect(NULL);
	if (display == NULL) {
		print_line("Can't connect to wayland display !?\n");
		exit(1);
	}

	struct wl_registry *wl_registry = wl_display_get_registry(display);
	wl_registry_add_listener(wl_registry, &registry_listener, this);

	wl_display_dispatch(display);
	wl_display_roundtrip(display);
}

Error OS_Wayland::initialize_display(const VideoMode &p_desired, int p_video_driver) {
	_initialize_wl_display();

	main_loop = NULL;

	if (compositor == NULL || xdgbase == NULL) {
		print_verbose("Error: Wayland compositor is missing required globals");
		exit(1);
	}

	surface = wl_compositor_create_surface(compositor);
	if (surface == NULL) {
		print_verbose("Error creating Wayland surface");
		exit(1);
	}

	egl_window = wl_egl_window_create(surface,
			p_desired.width, p_desired.height);

	xdgsurface = xdg_wm_base_get_xdg_surface(xdgbase, surface);
	xdg_surface_add_listener(xdgsurface, &xdg_surface_listener, NULL);

	xdgtoplevel = xdg_surface_get_toplevel(xdgsurface);
	xdg_toplevel_add_listener(xdgtoplevel, &xdg_toplevel_listener, this);
	xdg_toplevel_set_title(xdgtoplevel, "Godot");
	wl_surface_commit(surface);

	xdg_wm_base_add_listener(xdgbase, &xdg_wm_base_listener, NULL);
	wl_display_roundtrip(display);

	if (egl_window == EGL_NO_SURFACE) {
		print_verbose("Error: unable to create EGL window");
		exit(1);
	}

	context_gl_egl = NULL;
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

	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server,
					get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}

	return Error::OK;
}

void OS_Wayland::finalize_display() {
	wl_display_disconnect(display);
	print_line("not implemented (OS_Wayland): finalize_display");
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

void OS_Wayland::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
	print_line("not implemented (OS_Wayland): set_video_mode");
}

OS::VideoMode OS_Wayland::get_video_mode(int p_screen) const {
	print_line("not implemented (OS_Wayland): get_video_mode");
	return VideoMode();
}

void OS_Wayland::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
	print_line("not implemented (OS_Wayland): get_fullscreen_mode_list");
}

Size2 OS_Wayland::get_window_size() const {
	//print_line("not implemented (OS_Wayland): get_mouse_position");
	return Size2(context_gl_egl->get_window_width(), context_gl_egl->get_window_height());
}

bool OS_Wayland::get_window_per_pixel_transparency_enabled() const {
	print_line("not implemented (OS_Wayland): get_window_per_pixel_transparency_enabled");
	return false;
}

void OS_Wayland::set_window_per_pixel_transparency_enabled(bool p_enabled) {
	print_line("not implemented (OS_Wayland): set_window_per_pixel_transparency_enabled");
}

int OS_Wayland::get_video_driver_count() const {
	print_line("not implemented (OS_Wayland): get_video_driver_count");
	return 0;
}

const char *OS_Wayland::get_video_driver_name(int p_driver) const {
	print_line("not implemented (OS_Wayland): get_video_driver_name");
	return "";
}

int OS_Wayland::get_current_video_driver() const {
	print_line("not implemented (OS_Wayland): get_current_video_driver");
	return 0;
}

String OS_Wayland::get_name() {
	return String("Wayland");
}

bool OS_Wayland::can_draw() const {
	wl_display_dispatch_pending(display);
	return true;
}

void OS_Wayland::set_cursor_shape(CursorShape p_shape) {
	// TODO
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
