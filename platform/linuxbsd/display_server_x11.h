/*************************************************************************/
/*  display_server_x11.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DISPLAY_SERVER_X11_H
#define DISPLAY_SERVER_X11_H

#ifdef X11_ENABLED

#include "servers/display_server.h"

#include "core/input/input.h"
#include "core/local_vector.h"
#include "drivers/alsa/audio_driver_alsa.h"
#include "drivers/alsamidi/midi_driver_alsamidi.h"
#include "drivers/pulseaudio/audio_driver_pulseaudio.h"
#include "drivers/unix/os_unix.h"
#include "joypad_linux.h"
#include "servers/audio_server.h"
#include "servers/rendering/rasterizer.h"
#include "servers/rendering_server.h"

#if defined(OPENGL_ENABLED)
#include "context_gl_x11.h"
#endif

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/linuxbsd/vulkan_context_x11.h"
#endif

#include <X11/Xcursor/Xcursor.h>
#include <X11/Xlib.h>
#include <X11/extensions/XInput2.h>
#include <X11/extensions/Xrandr.h>
#include <X11/keysym.h>

// Hints for X11 fullscreen
typedef struct {
	unsigned long flags;
	unsigned long functions;
	unsigned long decorations;
	long inputMode;
	unsigned long status;
} Hints;

typedef struct _xrr_monitor_info {
	Atom name;
	Bool primary;
	Bool automatic;
	int noutput;
	int x;
	int y;
	int width;
	int height;
	int mwidth;
	int mheight;
	RROutput *outputs;
} xrr_monitor_info;

#undef CursorShape

class DisplayServerX11 : public DisplayServer {
	//No need to register, it's platform-specific and nothing is added
	//GDCLASS(DisplayServerX11, DisplayServer)

	_THREAD_SAFE_CLASS_

	Atom wm_delete;
	Atom xdnd_enter;
	Atom xdnd_position;
	Atom xdnd_status;
	Atom xdnd_action_copy;
	Atom xdnd_drop;
	Atom xdnd_finished;
	Atom xdnd_selection;
	Atom xdnd_aware;
	Atom requested;
	int xdnd_version;

#if defined(OPENGL_ENABLED)
	ContextGL_X11 *context_gles2;
#endif
#if defined(VULKAN_ENABLED)
	VulkanContextX11 *context_vulkan;
	RenderingDeviceVulkan *rendering_device_vulkan;
#endif

	struct WindowData {
		Window x11_window;
		::XIC xic;

		Size2i min_size;
		Size2i max_size;
		Point2i position;
		Size2i size;
		Point2i im_position;
		bool im_active = false;
		Callable rect_changed_callback;
		Callable event_callback;
		Callable input_event_callback;
		Callable input_text_callback;
		Callable drop_files_callback;

		WindowID transient_parent = INVALID_WINDOW_ID;
		Set<WindowID> transient_children;

		ObjectID instance_id;

		bool menu_type = false;
		bool no_focus = false;

		//better to guess on the fly, given WM can change it
		//WindowMode mode;
		bool fullscreen = false; //OS can't exit from this mode
		bool on_top = false;
		bool borderless = false;
		bool resize_disabled = false;
		Vector2i last_position_before_fs;
		bool focused = false;
		bool minimized = false;

		unsigned int focus_order = 0;
	};

	Map<WindowID, WindowData> windows;

	WindowID window_id_counter = MAIN_WINDOW_ID;
	WindowID _create_window(WindowMode p_mode, uint32_t p_flags, const Rect2i &p_rect);

	String internal_clipboard;
	Window xdnd_source_window;
	::Display *x11_display;
	char *xmbstring;
	int xmblen;
	unsigned long last_timestamp;
	::Time last_keyrelease_time;
	::XIM xim;
	::XIMStyle xim_style;
	static void _xim_destroy_callback(::XIM im, ::XPointer client_data,
			::XPointer call_data);

	Point2i last_mouse_pos;
	bool last_mouse_pos_valid;
	Point2i last_click_pos;
	uint64_t last_click_ms;
	int last_click_button_index;
	uint32_t last_button_state;
	bool app_focused = false;
	uint64_t time_since_no_focus = 0;

	struct {
		int opcode;
		Vector<int> touch_devices;
		Map<int, Vector2> absolute_devices;
		Map<int, Vector2> pen_pressure_range;
		Map<int, Vector2> pen_tilt_x_range;
		Map<int, Vector2> pen_tilt_y_range;
		XIEventMask all_event_mask;
		Map<int, Vector2> state;
		double pressure;
		bool pressure_supported;
		Vector2 tilt;
		Vector2 mouse_pos_to_filter;
		Vector2 relative_motion;
		Vector2 raw_pos;
		Vector2 old_raw_pos;
		::Time last_relative_time;
	} xi;

	bool _refresh_device_info();

	unsigned int _get_mouse_button_state(unsigned int p_x11_button, int p_x11_type);
	void _get_key_modifier_state(unsigned int p_x11_state, Ref<InputEventWithModifiers> state);
	void _flush_mouse_motion();

	MouseMode mouse_mode;
	Point2i center;

	void _handle_key_event(WindowID p_window, XKeyEvent *p_event, LocalVector<XEvent> &p_events, uint32_t &p_event_index, bool p_echo = false);
	void _handle_selection_request_event(XSelectionRequestEvent *p_event);

	String _clipboard_get_impl(Atom p_source, Window x11_window, Atom target) const;
	String _clipboard_get(Atom p_source, Window x11_window) const;

	//bool minimized;
	//bool window_has_focus;
	bool do_mouse_warp;

	const char *cursor_theme;
	int cursor_size;
	XcursorImage *img[CURSOR_MAX];
	Cursor cursors[CURSOR_MAX];
	Cursor null_cursor;
	CursorShape current_cursor;
	Map<CursorShape, Vector<Variant>> cursors_cache;

	bool layered_window;

	String rendering_driver;
	//bool window_focused;
	//void set_wm_border(bool p_enabled);
	void set_wm_fullscreen(bool p_enabled);
	void set_wm_above(bool p_enabled);

	typedef xrr_monitor_info *(*xrr_get_monitors_t)(Display *dpy, Window window, Bool get_active, int *nmonitors);
	typedef void (*xrr_free_monitors_t)(xrr_monitor_info *monitors);
	xrr_get_monitors_t xrr_get_monitors;
	xrr_free_monitors_t xrr_free_monitors;
	void *xrandr_handle;
	Bool xrandr_ext_ok;

	struct Property {
		unsigned char *data;
		int format, nitems;
		Atom type;
	};
	static Property _read_property(Display *p_display, Window p_window, Atom p_property);

	void _update_real_mouse_position(const WindowData &wd);
	bool _window_maximize_check(WindowID p_window, const char *p_atom_name) const;
	void _update_size_hints(WindowID p_window);
	void _set_wm_fullscreen(WindowID p_window, bool p_enabled);
	void _set_wm_maximized(WindowID p_window, bool p_enabled);

	void _update_context(WindowData &wd);

	Context context = CONTEXT_ENGINE;

	void _send_window_event(const WindowData &wd, WindowEvent p_event);
	static void _dispatch_input_events(const Ref<InputEvent> &p_event);
	void _dispatch_input_event(const Ref<InputEvent> &p_event);

	mutable Mutex events_mutex;
	Thread *events_thread = nullptr;
	bool events_thread_done = false;
	LocalVector<XEvent> polled_events;
	static void _poll_events_thread(void *ud);
	void _poll_events();

	static Bool _predicate_all_events(Display *display, XEvent *event, XPointer arg);
	static Bool _predicate_clipboard_selection(Display *display, XEvent *event, XPointer arg);

protected:
	void _window_changed(XEvent *event);

public:
	virtual bool has_feature(Feature p_feature) const;
	virtual String get_name() const;

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual void mouse_set_mode(MouseMode p_mode);
	virtual MouseMode mouse_get_mode() const;

	virtual void mouse_warp_to_position(const Point2i &p_to);
	virtual Point2i mouse_get_position() const;
	virtual Point2i mouse_get_absolute_position() const;
	virtual int mouse_get_button_state() const;

	virtual void clipboard_set(const String &p_text);
	virtual String clipboard_get() const;

	virtual int get_screen_count() const;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual bool screen_is_touchscreen(int p_screen = SCREEN_OF_MAIN_WINDOW) const;

	virtual Vector<DisplayServer::WindowID> get_window_list() const;

	virtual WindowID create_sub_window(WindowMode p_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i());
	virtual void show_window(WindowID p_id);
	virtual void delete_sub_window(WindowID p_id);

	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID);
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window = MAIN_WINDOW_ID);

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);

	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID);

	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID);

	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_transient(WindowID p_window, WindowID p_parent);

	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID);
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID);
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID);

	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID);

	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual bool can_any_window_draw() const;

	virtual void window_set_ime_active(const bool p_active, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_ime_position(const Point2i &p_pos, WindowID p_window = MAIN_WINDOW_ID);

	virtual void cursor_set_shape(CursorShape p_shape);
	virtual CursorShape cursor_get_shape() const;
	virtual void cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);

	virtual int keyboard_get_layout_count() const;
	virtual int keyboard_get_current_layout() const;
	virtual void keyboard_set_current_layout(int p_index);
	virtual String keyboard_get_layout_language(int p_index) const;
	virtual String keyboard_get_layout_name(int p_index) const;

	virtual void process_events();

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual void set_context(Context p_context);

	virtual void set_native_icon(const String &p_filename);
	virtual void set_icon(const Ref<Image> &p_icon);

	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	static Vector<String> get_rendering_drivers_func();

	static void register_x11_driver();

	DisplayServerX11(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerX11();
};

#endif // X11 enabled

#endif // DISPLAY_SERVER_X11_H
