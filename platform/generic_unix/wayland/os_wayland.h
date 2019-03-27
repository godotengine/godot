/*************************************************************************/
/*  os_wayland.h                                                         */
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
/**
	@author Timo Kandra <toger5@hotmail.de>
*/
#include "core/os/input.h"
#include "joypad_linux.h"
#include "main/input_default.h"
#include "os_genericunix.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"

#include <wayland-client-protocol.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include <wayland-server.h>
#include <xkbcommon/xkbcommon.h>

#include "context_egl_wayland.h"
#include "wayland_protocol/xdg-shell.h"
#undef CursorShape

class OS_Wayland : public OS_GenericUnix {
private:
	class WaylandOutput {
	public:
		WaylandOutput();
		WaylandOutput(OS_Wayland *d_wl, struct wl_output *output);

		OS_Wayland *d_wl;
		struct wl_output *output;
		int scale;
		bool entered;
	};

	Vector<WaylandOutput *> outputs = {};

	Vector2 _mouse_pos;

	MainLoop *main_loop;
	InputDefault *input;
	VisualServer *visual_server;
	VideoMode current_videomode;
	ContextGL_EGL *context_gl_egl = NULL;
	int scale_factor = 1;

	void _initialize_wl_display();

	struct wl_compositor *compositor = NULL;

	struct wl_display *display = NULL;
	struct wl_surface *surface;
	struct wl_egl_window *egl_window;
	struct wl_region *region;
	struct xdg_wm_base *xdgbase;
	struct xdg_surface *xdgsurface;
	struct xdg_toplevel *xdgtoplevel;
	struct wl_seat *seat;
	struct wl_pointer *mouse_pointer;

	struct xkb_context *xkb_context = NULL;
	struct xkb_keymap *xkb_keymap = NULL;
	struct xkb_state *xkb_state = NULL;

	void _set_modifier_for_event(Ref<InputEventWithModifiers> ev);

	static void registry_global(void *data, struct wl_registry *registry, uint32_t id, const char *interface, uint32_t version);
	static void registry_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name);

	const struct wl_registry_listener registry_listener = {
		&registry_global,
		&registry_global_remove,
	};

	void update_scale_factor();

	static void surface_enter(void *data, struct wl_surface *wl_surface,
			struct wl_output *output);
	static void surface_leave(void *data, struct wl_surface *wl_surface,
			struct wl_output *output);

	const struct wl_surface_listener surface_listener = {
		.enter = surface_enter,
		.leave = surface_leave,
	};

	static void xdg_toplevel_configure(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states);
	static void xdg_toplevel_close(void *data, struct xdg_toplevel *xdg_toplevel);

	const struct xdg_toplevel_listener xdg_toplevel_listener = {
		.configure = &xdg_toplevel_configure,
		.close = &xdg_toplevel_close,
	};

	static void xdg_surface_configure(void *data, struct xdg_surface *xdg_surface, uint32_t serial);

	const struct xdg_surface_listener xdg_surface_listener = {
		.configure = &xdg_surface_configure,
	};

	static void xdg_wm_base_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial);

	const struct xdg_wm_base_listener xdg_wm_base_listener = {
		.ping = &xdg_wm_base_ping,
	};

	static void seat_name(void *data, struct wl_seat *wl_seat, const char *name);
	static void seat_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities);

	const struct wl_seat_listener seat_listener = {
		&seat_capabilities,
		&seat_name,
	};

	static void pointer_enter(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y);
	static void pointer_leave(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface);
	static void pointer_motion(void *data, struct wl_pointer *wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y);
	static void pointer_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state);
	static void pointer_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value);
	static void pointer_frame(void *data, struct wl_pointer *wl_pointer);
	static void pointer_axis_source(void *data, struct wl_pointer *wl_pointer, uint32_t axis_source);
	static void pointer_axis_stop(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis);
	static void pointer_axis_discrete(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete);

	const struct wl_pointer_listener pointer_listener = {
		&pointer_enter,
		&pointer_leave,
		&pointer_motion,
		&pointer_button,
		&pointer_axis,
		&pointer_frame,
		&pointer_axis_source,
		&pointer_axis_stop,
		&pointer_axis_discrete,
	};

	static void keyboard_keymap(void *data, struct wl_keyboard *wl_keyboard, uint32_t format, int32_t fd, uint32_t size);
	static void keyboard_enter(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys);
	static void keyboard_leave(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface);
	static void keyboard_key(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state);
	static void keyboard_modifier(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group);
	static void keyboard_repeat_info(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay);

	const struct wl_keyboard_listener keyboard_listener = {
		&keyboard_keymap,
		&keyboard_enter,
		&keyboard_leave,
		&keyboard_key,
		&keyboard_modifier,
		&keyboard_repeat_info,
	};

	static void output_geometry(void *data, struct wl_output *output,
			int32_t x, int32_t y,
			int32_t physical_width, int32_t physical_height,
			int32_t subpixel, const char *make, const char *model,
			int32_t transform);
	static void output_mode(void *data, struct wl_output *output,
			uint32_t flags, int32_t width, int32_t height, int32_t refresh);
	static void output_done(void *data, struct wl_output *output);
	static void output_scale(void *data,
			struct wl_output *output, int32_t factor);

	const struct wl_output_listener output_listener = {
		&output_geometry,
		&output_mode,
		&output_done,
		&output_scale,
	};

protected:
	Error initialize_display(const VideoMode &p_desired, int p_video_driver);
	void finalize_display();

	void set_main_loop(MainLoop *p_main_loop);
	void delete_main_loop();

public:
	MainLoop *get_main_loop() const;

	// virtual void set_mouse_mode(MouseMode p_mode);
	// virtual MouseMode get_mouse_mode() const;

	// virtual void warp_mouse_position(const Point2 &p_to) {}
	Point2 get_mouse_position() const;
	int get_mouse_button_state() const;
	void set_window_title(const String &p_title);

	// virtual void set_clipboard(const String &p_text);
	// virtual String get_clipboard() const;

	void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	VideoMode get_video_mode(int p_screen = 0) const;
	void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual int get_screen_count() const { return outputs.size(); }
	// virtual int get_current_screen() const { return 0; }
	// virtual void set_current_screen(int p_screen) {}
	// virtual Point2 get_screen_position(int p_screen = -1) const { return Point2(); }
	virtual Size2 get_screen_size(int p_screen = -1) const { return get_window_size(); }
	virtual int get_screen_dpi(int p_screen = -1) const { return 72; }

	Size2 get_window_size() const;
	virtual Size2 get_real_window_size() const { return get_window_size(); }
	virtual void set_window_size(const Size2 p_size) {}
	// virtual void set_window_fullscreen(bool p_enabled) {}
	// virtual bool is_window_fullscreen() const { return true; }

	// virtual void set_window_resizable(bool p_enabled) {}
	// virtual bool is_window_resizable() const { return false; }

	// virtual void set_window_minimized(bool p_enabled) {}
	// virtual bool is_window_minimized() const { return false; }

	// virtual void set_window_maximized(bool p_enabled) {}
	// virtual bool is_window_maximized() const { return true; }

	// virtual void set_window_always_on_top(bool p_enabled) {}
	// virtual bool is_window_always_on_top() const { return false; }

	// virtual Rect2 get_window_safe_area() const;

	// virtual void set_borderless_window(bool p_borderless) {}
	// virtual bool get_borderless_window() { return 0; }

	bool get_window_per_pixel_transparency_enabled() const;
	void set_window_per_pixel_transparency_enabled(bool p_enabled);

	int get_video_driver_count() const;
	const char *get_video_driver_name(int p_driver) const;
	int get_current_video_driver() const;

	// virtual void set_keep_screen_on(bool p_enabled);
	// virtual bool is_keep_screen_on() const;

	String get_name();
	bool can_draw() const;

	void set_cursor_shape(CursorShape p_shape);
	void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);

	virtual void swap_buffers();

	virtual void set_icon(const Ref<Image> &p_icon);

	// virtual bool is_joy_known(int p_device);
	// virtual String get_joy_guid(int p_device) const;

	virtual void process_events();
};
