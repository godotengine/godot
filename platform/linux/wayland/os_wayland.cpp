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
#include <linux/input-event-codes.h>
//
//
// Wayland events
//
//
#define DISPLAY_WL (OS_Wayland *)OS_Wayland::get_singleton()

void OS_Wayland::global_registry_handler(void *data, struct wl_registry *registry, uint32_t id, const char *interface, uint32_t version) {
	OS_Wayland *d_wl = DISPLAY_WL;

	printf("Got a registry event for %s id %d\n", interface, id);

	if (strcmp(interface, "wl_compositor") == 0) {

		d_wl->compositor = (wl_compositor *)wl_registry_bind(registry, id, &wl_compositor_interface, 1);
	}

	else if (strcmp(interface, "wl_seat") == 0) {

		d_wl->seat = (wl_seat *)wl_registry_bind(registry, id, &wl_seat_interface, 1);
		wl_seat_add_listener(d_wl->seat, &d_wl->seat_listener, NULL);
	}

	else if (strcmp(interface, "xdg_wm_base") == 0) {

		d_wl->xdgbase = (xdg_wm_base *)wl_registry_bind(registry, id, &xdg_wm_base_interface, 1);
	}
}

void OS_Wayland::global_registry_remover(void *data, struct wl_registry *wl_registry, uint32_t name) {
}

void OS_Wayland::xdg_toplevel_configure_handler(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states) {

	printf("configure: %dx%d\n", width, height);
}

void OS_Wayland::xdg_toplevel_close_handler(void *data, struct xdg_toplevel *xdg_toplevel) {

	printf("close\n");
}

void OS_Wayland::xdg_surface_configure_handler(void *data, struct xdg_surface *xdg_surface, uint32_t serial) {
	printf("configure surface: %d", serial);
	xdg_surface_ack_configure(xdg_surface, serial);
}
void OS_Wayland::xdg_ping_handler(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
	printf("ping-pong\n");
}

void OS_Wayland::seat_name_handler(void *data, struct wl_seat *wl_seat, const char *name) {
}
void OS_Wayland::seat_capabilities_handler(void *data, struct wl_seat *wl_seat, uint32_t capabilities) {
	OS_Wayland *d_wl = DISPLAY_WL;
	if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
		print_verbose("pointer!!!");
		d_wl->mouse_pointer = wl_seat_get_pointer(wl_seat);
		wl_pointer_add_listener(d_wl->mouse_pointer, &d_wl->pointer_listener, NULL);
	}
	if (capabilities & WL_SEAT_CAPABILITY_KEYBOARD) {
		print_verbose("keyboard!!!");
		struct wl_keyboard *keyboard;
		keyboard = wl_seat_get_keyboard(wl_seat);
		wl_keyboard_add_listener(keyboard, &d_wl->keyboard_listener, NULL);
	}
	if (capabilities & WL_SEAT_CAPABILITY_TOUCH) {
		print_verbose("touch!!!");
	}
}
void OS_Wayland::pointer_enter_handler(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	print_verbose("pointer entered");
}
void OS_Wayland::pointer_leave_handler(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface) {
	print_verbose("pointer left");
}
void OS_Wayland::pointer_motion_handler(void *data, struct wl_pointer *wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	OS_Wayland *d_wl = DISPLAY_WL;
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
void OS_Wayland::pointer_button_handler(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
	OS_Wayland *d_wl = DISPLAY_WL;
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
void OS_Wayland::pointer_axis_handler(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {
}
void OS_Wayland::pointer_frame_handler(void *data, struct wl_pointer *wl_pointer) {
}
void OS_Wayland::pointer_axis_source_handler(void *data, struct wl_pointer *wl_pointer, uint32_t axis_source) {
}
void OS_Wayland::pointer_axis_stop_handler(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis) {
}
void OS_Wayland::pointer_axis_discrete_handler(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete) {
}

void OS_Wayland::keyboard_keymap_handler(void *data, struct wl_keyboard *wl_keyboard, uint32_t format, int32_t fd, uint32_t size) {
}
void OS_Wayland::keyboard_enter_handler(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys) {
}
void OS_Wayland::keyboard_leave_handler(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface) {
}
void OS_Wayland::keyboard_key_handler(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
}
void OS_Wayland::keyboard_modifier_handler(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
}
void OS_Wayland::keyboard_repeat_info_handler(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay) {
}

//
//
// end of wayland events
//
//

//OS_Wayland
void OS_Wayland::_get_server_refs() {
	// server stuff getten
	display = NULL;
	display = wl_display_connect(NULL);
	if (display == NULL) {
		print_line("Can't connect to wayland display !?\n");
		exit(1);
	}
	struct wl_registry *wl_registry = wl_display_get_registry(display);
	wl_registry_add_listener(wl_registry, &registry_listener, NULL);

	// This call the attached listener global_registry_handler
	wl_display_dispatch(display);
	wl_display_roundtrip(display);
}

Error OS_Wayland::initialize_display(const VideoMode &p_desired, int p_video_driver) {
	_get_server_refs();
	// If at this point, global_registry_handler didn't set the
	// compositor, nor the shell, bailout !
	if (compositor == NULL || xdgbase == NULL) {
		print_verbose("No compositor !? No Shell !! There's NOTHING in here !\n");
		exit(1);
	} else {
		print_verbose("Okay, we got a compositor and a shell... That's something !\n");
		// ESContext.native_display = display;
	}
	// create window
	surface = wl_compositor_create_surface(compositor);
	if (surface == NULL) {
		print_verbose("No Compositor surface ! Yay....\n");
		exit(1);
	} else
		print_verbose("Got a compositor surface !\n");

	xdgsurface = xdg_wm_base_get_xdg_surface(xdgbase, surface);
	xdg_surface_add_listener(xdgsurface, &xdg_surface_listener, NULL);
	// shell_surface = wl_shell_get_shell_surface(shell, surface);

	xdgtoplevel = xdg_surface_get_toplevel(xdgsurface);
	xdg_toplevel_add_listener(xdgtoplevel, &xdg_toplevel_listener, NULL);
	xdg_toplevel_set_title(xdgtoplevel, "Godot");
	wl_surface_commit(surface);

	// wait for the "initial" set of globals to appear

	xdg_wm_base_add_listener(xdgbase, &xdg_wm_base_listener, NULL);
	wl_display_roundtrip(display);
	//make opaque
	// region = wl_compositor_create_region(compositor);
	// wl_region_add(region, 0, 0, p_desired.width, p_desired.height);
	// wl_surface_set_opaque_region(surface, region);

	//wl_display_dispatch(display);
	struct wl_egl_window *egl_window = wl_egl_window_create(surface, p_desired.width, p_desired.height);

	if (egl_window == EGL_NO_SURFACE) {
		print_verbose("No window !?\n");
		exit(1);
	} else
		print_verbose("Window created !\n");

	context_gl_egl = NULL;
	ContextGL_EGL::Driver context_type = ContextGL_EGL::Driver::GLES_3_0; //TODO: check for possible context types

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

	// video_driver_index = p_video_driver;

	// context_gl->set_use_vsync(current_videomode.use_vsync);

	//#endif

	//VISUAL SERVER
	visual_server = memnew(VisualServerRaster);

	visual_server->init();

	input = memnew(InputDefault);

	// if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {

	// 	visual_server = memnew(VisualServerWrapMT(visual_server, get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	// }
	// ESContext.window_width = width;
	// ESContext.window_height = height;
	// ESContext.native_window = egl_window;

	//INPUT

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
	print_line("not implemented (OS_Wayland): delete_main_loop");
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
	xdg_toplevel_set_title(xdgtoplevel, (char *)p_title.c_str());
	print_line("not implemented (OS_Wayland): set_window_title" + p_title);
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
	print_line("not implemented (OS_Wayland): get_name");
	return String("");
}
bool OS_Wayland::can_draw() const {
	wl_display_dispatch_pending(display);
	return true;
}
void OS_Wayland::set_cursor_shape(CursorShape p_shape) {
	print_line("not implemented (OS_Wayland): set_cursor_shape");
}
void OS_Wayland::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	print_line("not implemented (OS_Wayland): set_custom_mouse_cursor");
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