/*************************************************************************/
/*  display_server.h                                                     */
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

#ifndef DISPLAY_SERVER_H
#define DISPLAY_SERVER_H

#include "core/input/input.h"
#include "core/io/resource.h"
#include "core/os/os.h"
#include "core/variant/callable.h"

class Texture2D;

class DisplayServer : public Object {
	GDCLASS(DisplayServer, Object)

	static DisplayServer *singleton;
	static bool hidpi_allowed;

public:
	_FORCE_INLINE_ static DisplayServer *get_singleton() {
		return singleton;
	}

	enum WindowMode {
		WINDOW_MODE_WINDOWED,
		WINDOW_MODE_MINIMIZED,
		WINDOW_MODE_MAXIMIZED,
		WINDOW_MODE_FULLSCREEN
	};

	// Keep the VSyncMode enum values in sync with the `display/window/vsync/vsync_mode`
	// project setting hint.
	enum VSyncMode {
		VSYNC_DISABLED,
		VSYNC_ENABLED,
		VSYNC_ADAPTIVE,
		VSYNC_MAILBOX
	};

	enum HandleType {
		DISPLAY_HANDLE,
		WINDOW_HANDLE,
		WINDOW_VIEW,
	};

	typedef DisplayServer *(*CreateFunction)(const String &, WindowMode, VSyncMode, uint32_t, const Size2i &, Error &r_error);
	typedef Vector<String> (*GetRenderingDriversFunction)();

private:
	static void _input_set_mouse_mode(Input::MouseMode p_mode);
	static Input::MouseMode _input_get_mouse_mode();
	static void _input_warp(const Vector2 &p_to_pos);
	static Input::CursorShape _input_get_current_cursor_shape();
	static void _input_set_custom_mouse_cursor_func(const RES &, Input::CursorShape, const Vector2 &p_hostspot);

protected:
	static void _bind_methods();

	enum {
		MAX_SERVERS = 64
	};

	struct DisplayServerCreate {
		const char *name;
		CreateFunction create_function;
		GetRenderingDriversFunction get_rendering_drivers_function;
	};

	static DisplayServerCreate server_create_functions[MAX_SERVERS];
	static int server_create_count;

	friend class RendererViewport;

public:
	enum Feature {
		FEATURE_GLOBAL_MENU,
		FEATURE_SUBWINDOWS,
		FEATURE_TOUCHSCREEN,
		FEATURE_MOUSE,
		FEATURE_MOUSE_WARP,
		FEATURE_CLIPBOARD,
		FEATURE_VIRTUAL_KEYBOARD,
		FEATURE_CURSOR_SHAPE,
		FEATURE_CUSTOM_CURSOR_SHAPE,
		FEATURE_NATIVE_DIALOG,
		FEATURE_IME,
		FEATURE_WINDOW_TRANSPARENCY,
		FEATURE_HIDPI,
		FEATURE_ICON,
		FEATURE_NATIVE_ICON,
		FEATURE_ORIENTATION,
		FEATURE_SWAP_BUFFERS,
		FEATURE_KEEP_SCREEN_ON,
		FEATURE_CLIPBOARD_PRIMARY,
	};

	virtual bool has_feature(Feature p_feature) const = 0;
	virtual String get_name() const = 0;

	virtual void global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag = Variant());
	virtual void global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag = Variant());
	virtual void global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu);
	virtual void global_menu_add_separator(const String &p_menu_root);

	virtual bool global_menu_is_item_checked(const String &p_menu_root, int p_idx) const;
	virtual bool global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const;
	virtual Callable global_menu_get_item_callback(const String &p_menu_root, int p_idx);
	virtual Variant global_menu_get_item_tag(const String &p_menu_root, int p_idx);
	virtual String global_menu_get_item_text(const String &p_menu_root, int p_idx);
	virtual String global_menu_get_item_submenu(const String &p_menu_root, int p_idx);

	virtual void global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked);
	virtual void global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable);
	virtual void global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback);
	virtual void global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag);
	virtual void global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text);
	virtual void global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu);

	virtual int global_menu_get_item_count(const String &p_menu_root) const;

	virtual void global_menu_remove_item(const String &p_menu_root, int p_idx);
	virtual void global_menu_clear(const String &p_menu_root);

	enum MouseMode {
		MOUSE_MODE_VISIBLE,
		MOUSE_MODE_HIDDEN,
		MOUSE_MODE_CAPTURED,
		MOUSE_MODE_CONFINED,
		MOUSE_MODE_CONFINED_HIDDEN,
	};

	virtual void mouse_set_mode(MouseMode p_mode);
	virtual MouseMode mouse_get_mode() const;

	virtual void mouse_warp_to_position(const Point2i &p_to);
	virtual Point2i mouse_get_position() const;
	virtual MouseButton mouse_get_button_state() const;

	virtual void clipboard_set(const String &p_text);
	virtual String clipboard_get() const;
	virtual bool clipboard_has() const;
	virtual void clipboard_set_primary(const String &p_text);
	virtual String clipboard_get_primary() const;

	enum {
		SCREEN_OF_MAIN_WINDOW = -1
	};

	virtual int get_screen_count() const = 0;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual float screen_get_scale(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual float screen_get_max_scale() const {
		float scale = 1.f;
		int screen_count = get_screen_count();
		for (int i = 0; i < screen_count; i++) {
			scale = fmax(scale, screen_get_scale(i));
		}
		return scale;
	}
	virtual bool screen_is_touchscreen(int p_screen = SCREEN_OF_MAIN_WINDOW) const;

	// Keep the ScreenOrientation enum values in sync with the `display/window/handheld/orientation`
	// project setting hint.
	enum ScreenOrientation {
		SCREEN_LANDSCAPE,
		SCREEN_PORTRAIT,
		SCREEN_REVERSE_LANDSCAPE,
		SCREEN_REVERSE_PORTRAIT,
		SCREEN_SENSOR_LANDSCAPE,
		SCREEN_SENSOR_PORTRAIT,
		SCREEN_SENSOR,
	};

	virtual void screen_set_orientation(ScreenOrientation p_orientation, int p_screen = SCREEN_OF_MAIN_WINDOW);
	virtual ScreenOrientation screen_get_orientation(int p_screen = SCREEN_OF_MAIN_WINDOW) const;

	virtual void screen_set_keep_on(bool p_enable); //disable screensaver
	virtual bool screen_is_kept_on() const;
	enum {
		MAIN_WINDOW_ID = 0,
		INVALID_WINDOW_ID = -1
	};

	typedef int WindowID;

	virtual Vector<DisplayServer::WindowID> get_window_list() const = 0;

	enum WindowFlags {
		WINDOW_FLAG_RESIZE_DISABLED,
		WINDOW_FLAG_BORDERLESS,
		WINDOW_FLAG_ALWAYS_ON_TOP,
		WINDOW_FLAG_TRANSPARENT,
		WINDOW_FLAG_NO_FOCUS,
		WINDOW_FLAG_MAX,
	};

	// Separate enum otherwise we get warnings in switches not handling all values.
	enum WindowFlagsBit {
		WINDOW_FLAG_RESIZE_DISABLED_BIT = (1 << WINDOW_FLAG_RESIZE_DISABLED),
		WINDOW_FLAG_BORDERLESS_BIT = (1 << WINDOW_FLAG_BORDERLESS),
		WINDOW_FLAG_ALWAYS_ON_TOP_BIT = (1 << WINDOW_FLAG_ALWAYS_ON_TOP),
		WINDOW_FLAG_TRANSPARENT_BIT = (1 << WINDOW_FLAG_TRANSPARENT),
		WINDOW_FLAG_NO_FOCUS_BIT = (1 << WINDOW_FLAG_NO_FOCUS)
	};

	virtual WindowID create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i());
	virtual void show_window(WindowID p_id);
	virtual void delete_sub_window(WindowID p_id);

	virtual int64_t window_get_native_handle(HandleType p_handle_type, WindowID p_window = MAIN_WINDOW_ID) const;

	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const = 0;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const = 0;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) = 0;

	enum WindowEvent {
		WINDOW_EVENT_MOUSE_ENTER,
		WINDOW_EVENT_MOUSE_EXIT,
		WINDOW_EVENT_FOCUS_IN,
		WINDOW_EVENT_FOCUS_OUT,
		WINDOW_EVENT_CLOSE_REQUEST,
		WINDOW_EVENT_GO_BACK_REQUEST,
		WINDOW_EVENT_DPI_CHANGE,
	};
	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) = 0;

	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) = 0;

	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID) = 0;

	virtual void window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window = MAIN_WINDOW_ID);

	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const = 0;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID) = 0;

	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const = 0;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID) = 0;

	virtual void window_set_transient(WindowID p_window, WindowID p_parent) = 0;

	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const = 0;

	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const = 0;

	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const = 0;
	virtual Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const = 0; // FIXME: Find clearer name for this.

	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const = 0;

	virtual void window_set_vsync_mode(VSyncMode p_vsync_mode, WindowID p_window = MAIN_WINDOW_ID);
	virtual VSyncMode window_get_vsync_mode(WindowID p_window) const;

	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const = 0;

	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const = 0;

	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID) = 0;
	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID) = 0;

	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const = 0;

	virtual bool can_any_window_draw() const = 0;

	virtual void window_set_ime_active(const bool p_active, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_ime_position(const Point2i &p_pos, WindowID p_window = MAIN_WINDOW_ID);

	// necessary for GL focus, may be able to use one of the existing functions for this, not sure yet
	virtual void gl_window_make_current(DisplayServer::WindowID p_window_id);

	virtual Point2i ime_get_selection() const;
	virtual String ime_get_text() const;

	virtual void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void virtual_keyboard_hide();

	// returns height of the currently shown virtual keyboard (0 if keyboard is hidden)
	virtual int virtual_keyboard_get_height() const;

	enum CursorShape {
		CURSOR_ARROW,
		CURSOR_IBEAM,
		CURSOR_POINTING_HAND,
		CURSOR_CROSS,
		CURSOR_WAIT,
		CURSOR_BUSY,
		CURSOR_DRAG,
		CURSOR_CAN_DROP,
		CURSOR_FORBIDDEN,
		CURSOR_VSIZE,
		CURSOR_HSIZE,
		CURSOR_BDIAGSIZE,
		CURSOR_FDIAGSIZE,
		CURSOR_MOVE,
		CURSOR_VSPLIT,
		CURSOR_HSPLIT,
		CURSOR_HELP,
		CURSOR_MAX
	};
	virtual void cursor_set_shape(CursorShape p_shape);
	virtual CursorShape cursor_get_shape() const;
	virtual void cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape = CURSOR_ARROW, const Vector2 &p_hotspot = Vector2());

	virtual bool get_swap_cancel_ok();

	virtual void enable_for_stealing_focus(OS::ProcessID pid);

	virtual Error dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback);
	virtual Error dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback);

	virtual int keyboard_get_layout_count() const;
	virtual int keyboard_get_current_layout() const;
	virtual void keyboard_set_current_layout(int p_index);
	virtual String keyboard_get_layout_language(int p_index) const;
	virtual String keyboard_get_layout_name(int p_index) const;
	virtual Key keyboard_get_keycode_from_physical(Key p_keycode) const;

	virtual int tablet_get_driver_count() const { return 1; };
	virtual String tablet_get_driver_name(int p_driver) const { return "default"; };
	virtual String tablet_get_current_driver() const { return "default"; };
	virtual void tablet_set_current_driver(const String &p_driver){};

	virtual void process_events() = 0;

	virtual void force_process_and_drop_events();

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual void set_native_icon(const String &p_filename);
	virtual void set_icon(const Ref<Image> &p_icon);

	enum Context {
		CONTEXT_EDITOR,
		CONTEXT_PROJECTMAN,
		CONTEXT_ENGINE,
	};

	virtual void set_context(Context p_context);

	static void register_create_function(const char *p_name, CreateFunction p_function, GetRenderingDriversFunction p_get_drivers);
	static int get_create_function_count();
	static const char *get_create_function_name(int p_index);
	static Vector<String> get_create_function_rendering_drivers(int p_index);
	static DisplayServer *create(int p_index, const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);

	DisplayServer();
	~DisplayServer();
};

VARIANT_ENUM_CAST(DisplayServer::WindowEvent)
VARIANT_ENUM_CAST(DisplayServer::Feature)
VARIANT_ENUM_CAST(DisplayServer::MouseMode)
VARIANT_ENUM_CAST(DisplayServer::ScreenOrientation)
VARIANT_ENUM_CAST(DisplayServer::WindowMode)
VARIANT_ENUM_CAST(DisplayServer::WindowFlags)
VARIANT_ENUM_CAST(DisplayServer::HandleType)
VARIANT_ENUM_CAST(DisplayServer::CursorShape)
VARIANT_ENUM_CAST(DisplayServer::VSyncMode)

#endif // DISPLAY_SERVER_H
