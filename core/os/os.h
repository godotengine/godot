/*************************************************************************/
/*  os.h                                                                 */
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

#ifndef OS_H
#define OS_H

#include "core/engine.h"
#include "core/image.h"
#include "core/io/logger.h"
#include "core/list.h"
#include "core/os/main_loop.h"
#include "core/ustring.h"
#include "core/vector.h"

#include <stdarg.h>
#include <stdlib.h>

class OS {
	static OS *singleton;
	static uint64_t target_ticks;
	String _execpath;
	List<String> _cmdline;
	bool _keep_screen_on;
	bool low_processor_usage_mode;
	int low_processor_usage_mode_sleep_usec;
	bool _verbose_stdout;
	bool _debug_stdout;
	String _local_clipboard;
	uint64_t _msec_splash;
	bool _no_window;
	int _exit_code = EXIT_FAILURE; // unexpected exit is marked as failure
	bool _is_custom_exit_code = false;
	int _orientation;
	bool _allow_hidpi;
	bool _allow_layered;
	bool _use_vsync;
	bool _vsync_via_compositor;
	bool _delta_smoothing_enabled;

	char *last_error;

	void *_stack_bottom;

	CompositeLogger *_logger;

	bool restart_on_exit;
	List<String> restart_commandline;

protected:
	void _set_logger(CompositeLogger *p_logger);

public:
	typedef void (*ImeCallback)(void *p_inp, String p_text, Point2 p_selection);
	typedef bool (*HasServerFeatureCallback)(const String &p_feature);

	enum PowerState {
		POWERSTATE_UNKNOWN, /**< cannot determine power status */
		POWERSTATE_ON_BATTERY, /**< Not plugged in, running on the battery */
		POWERSTATE_NO_BATTERY, /**< Plugged in, no battery available */
		POWERSTATE_CHARGING, /**< Plugged in, charging battery */
		POWERSTATE_CHARGED /**< Plugged in, battery charged */
	};

	enum RenderThreadMode {

		RENDER_THREAD_UNSAFE,
		RENDER_THREAD_SAFE,
		RENDER_SEPARATE_THREAD
	};
	struct VideoMode {
		int width, height;
		bool fullscreen;
		bool resizable;
		bool borderless_window;
		bool maximized;
		bool always_on_top;
		bool use_vsync;
		bool vsync_via_compositor;
		bool layered;
		float get_aspect() const { return (float)width / (float)height; }
		VideoMode(int p_width = 1024, int p_height = 600, bool p_fullscreen = false, bool p_resizable = true, bool p_borderless_window = false, bool p_maximized = false, bool p_always_on_top = false, bool p_use_vsync = false, bool p_vsync_via_compositor = false) {
			width = p_width;
			height = p_height;
			fullscreen = p_fullscreen;
			resizable = p_resizable;
			borderless_window = p_borderless_window;
			maximized = p_maximized;
			always_on_top = p_always_on_top;
			use_vsync = p_use_vsync;
			vsync_via_compositor = p_vsync_via_compositor;
			layered = false;
		}
	};

protected:
	friend class Main;

	HasServerFeatureCallback has_server_feature_callback;
	RenderThreadMode _render_thread_mode;

	// functions used by main to initialize/deinitialize the OS
	void add_logger(Logger *p_logger);

	virtual void initialize_core() = 0;
	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) = 0;

	virtual void set_main_loop(MainLoop *p_main_loop) = 0;
	virtual void delete_main_loop() = 0;

	virtual void finalize() = 0;
	virtual void finalize_core() = 0;

	virtual void set_cmdline(const char *p_execpath, const List<String> &p_args);

	virtual bool _check_internal_feature_support(const String &p_feature) = 0;

public:
	typedef int64_t ProcessID;

	static OS *get_singleton();

	virtual void global_menu_add_item(const String &p_menu, const String &p_label, const Variant &p_signal, const Variant &p_meta){};
	virtual void global_menu_add_separator(const String &p_menu){};
	virtual void global_menu_remove_item(const String &p_menu, int p_idx){};
	virtual void global_menu_clear(const String &p_menu){};

	void print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, Logger::ErrorType p_type = Logger::ERR_ERROR);
	void print(const char *p_format, ...) _PRINTF_FORMAT_ATTRIBUTE_2_3;
	void printerr(const char *p_format, ...) _PRINTF_FORMAT_ATTRIBUTE_2_3;

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!") = 0;
	virtual String get_stdin_string(bool p_block = true) = 0;

	enum MouseMode {
		MOUSE_MODE_VISIBLE,
		MOUSE_MODE_HIDDEN,
		MOUSE_MODE_CAPTURED,
		MOUSE_MODE_CONFINED
	};

	virtual void set_mouse_mode(MouseMode p_mode);
	virtual MouseMode get_mouse_mode() const;

	virtual void warp_mouse_position(const Point2 &p_to) {}
	virtual Point2 get_mouse_position() const = 0;
	virtual int get_mouse_button_state() const = 0;
	virtual void set_window_title(const String &p_title) = 0;
	virtual void set_window_mouse_passthrough(const PoolVector2Array &p_region){};

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0) = 0;
	virtual VideoMode get_video_mode(int p_screen = 0) const = 0;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const = 0;

	enum VideoDriver {
		VIDEO_DRIVER_GLES3,
		VIDEO_DRIVER_GLES2,
		VIDEO_DRIVER_MAX,
	};

	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;
	virtual int get_current_video_driver() const = 0;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual int get_tablet_driver_count() const { return 0; };
	virtual String get_tablet_driver_name(int p_driver) const { return ""; };
	virtual String get_current_tablet_driver() const { return ""; };
	virtual void set_current_tablet_driver(const String &p_driver){};

	virtual PoolStringArray get_connected_midi_inputs();
	virtual void open_midi_inputs();
	virtual void close_midi_inputs();

	virtual int get_screen_count() const { return 1; }
	virtual int get_current_screen() const { return 0; }
	virtual void set_current_screen(int p_screen) {}
	virtual Point2 get_screen_position(int p_screen = -1) const { return Point2(); }
	virtual Size2 get_screen_size(int p_screen = -1) const { return get_window_size(); }
	virtual int get_screen_dpi(int p_screen = -1) const { return 72; }
	virtual float get_screen_scale(int p_screen = -1) const { return 1.0; }
	virtual float get_screen_max_scale() const { return 1.0; };
	virtual Point2 get_window_position() const { return Vector2(); }
	virtual void set_window_position(const Point2 &p_position) {}
	virtual Size2 get_max_window_size() const { return Size2(); };
	virtual Size2 get_min_window_size() const { return Size2(); };
	virtual Size2 get_window_size() const = 0;
	virtual Size2 get_real_window_size() const { return get_window_size(); }
	virtual void set_min_window_size(const Size2 p_size) {}
	virtual void set_max_window_size(const Size2 p_size) {}
	virtual void set_window_size(const Size2 p_size) {}
	virtual void set_window_fullscreen(bool p_enabled) {}
	virtual bool is_window_fullscreen() const { return true; }
	virtual void set_window_resizable(bool p_enabled) {}
	virtual bool is_window_resizable() const { return false; }
	virtual void set_window_minimized(bool p_enabled) {}
	virtual bool is_window_minimized() const { return false; }
	virtual void set_window_maximized(bool p_enabled) {}
	virtual bool is_window_maximized() const { return true; }
	virtual void set_window_always_on_top(bool p_enabled) {}
	virtual bool is_window_always_on_top() const { return false; }
	virtual bool is_window_focused() const { return true; }
	virtual void set_console_visible(bool p_enabled) {}
	virtual bool is_console_visible() const { return false; }
	virtual void request_attention() {}
	virtual void center_window();

	// Returns internal pointers and handles.
	// While exposed to GDScript this is mostly to give GDNative plugins access to this information.
	// Note that whether a valid handle is returned depends on whether it applies to the given
	// platform and often to the chosen render driver.
	// NULL will be returned if a handle is not available.

	enum HandleType {
		APPLICATION_HANDLE, // HINSTANCE, NSApplication*, UIApplication*, JNIEnv* ...
		DISPLAY_HANDLE, // X11::Display* ...
		WINDOW_HANDLE, // HWND, X11::Window*, NSWindow*, UIWindow*, Android activity ...
		WINDOW_VIEW, // HDC, NSView*, UIView*, Android surface ...
		OPENGL_CONTEXT, // HGLRC, X11::GLXContext, NSOpenGLContext*, EGLContext* ...
	};

	virtual void *get_native_handle(int p_handle_type) { return nullptr; };

	// Returns window area free of hardware controls and other obstacles.
	// The application should use this to determine where to place UI elements.
	//
	// Keep in mind the area returned is in window coordinates rather than
	// viewport coordinates - you should perform the conversion on your own.
	//
	// The maximum size of the area is Rect2(0, 0, window_size.width, window_size.height).
	virtual Rect2 get_window_safe_area() const {
		Size2 window_size = get_window_size();
		return Rect2(0, 0, window_size.width, window_size.height);
	}

	virtual void set_borderless_window(bool p_borderless) {}
	virtual bool get_borderless_window() { return false; }

	virtual bool get_window_per_pixel_transparency_enabled() const { return false; }
	virtual void set_window_per_pixel_transparency_enabled(bool p_enabled) {}

	virtual void set_ime_active(const bool p_active) {}
	virtual void set_ime_position(const Point2 &p_pos) {}
	virtual Point2 get_ime_selection() const { return Point2(); }
	virtual String get_ime_text() const { return String(); }

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false) { return ERR_UNAVAILABLE; }
	virtual Error close_dynamic_library(void *p_library_handle) { return ERR_UNAVAILABLE; }
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false) { return ERR_UNAVAILABLE; }

	virtual void set_keep_screen_on(bool p_enabled);
	virtual bool is_keep_screen_on() const;
	virtual void set_low_processor_usage_mode(bool p_enabled);
	virtual bool is_in_low_processor_usage_mode() const;
	virtual void set_low_processor_usage_mode_sleep_usec(int p_usec);
	virtual int get_low_processor_usage_mode_sleep_usec() const;

	virtual String get_executable_path() const;
	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking = true, ProcessID *r_child_id = nullptr, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr) = 0;
	virtual Error kill(const ProcessID &p_pid) = 0;
	virtual int get_process_id() const;
	virtual void vibrate_handheld(int p_duration_ms = 500);

	virtual Error shell_open(String p_uri);
	virtual Error set_cwd(const String &p_cwd);

	virtual bool has_environment(const String &p_var) const = 0;
	virtual String get_environment(const String &p_var) const = 0;
	virtual bool set_environment(const String &p_var, const String &p_value) const = 0;

	virtual String get_name() const = 0;
	virtual List<String> get_cmdline_args() const { return _cmdline; }
	virtual String get_model_name() const;

	void ensure_user_data_dir();

	virtual MainLoop *get_main_loop() const = 0;

	virtual void yield();

	enum Weekday {
		DAY_SUNDAY,
		DAY_MONDAY,
		DAY_TUESDAY,
		DAY_WEDNESDAY,
		DAY_THURSDAY,
		DAY_FRIDAY,
		DAY_SATURDAY
	};

	enum Month {
		/// Start at 1 to follow Windows SYSTEMTIME structure
		/// https://msdn.microsoft.com/en-us/library/windows/desktop/ms724950(v=vs.85).aspx
		MONTH_JANUARY = 1,
		MONTH_FEBRUARY,
		MONTH_MARCH,
		MONTH_APRIL,
		MONTH_MAY,
		MONTH_JUNE,
		MONTH_JULY,
		MONTH_AUGUST,
		MONTH_SEPTEMBER,
		MONTH_OCTOBER,
		MONTH_NOVEMBER,
		MONTH_DECEMBER
	};

	struct Date {
		int year;
		Month month;
		int day;
		Weekday weekday;
		bool dst;
	};

	struct Time {
		int hour;
		int min;
		int sec;
	};

	struct TimeZoneInfo {
		int bias;
		String name;
	};

	virtual Date get_date(bool local = false) const = 0;
	virtual Time get_time(bool local = false) const = 0;
	virtual TimeZoneInfo get_time_zone_info() const = 0;
	virtual String get_iso_date_time(bool local = false) const;
	virtual uint64_t get_unix_time() const;
	virtual uint64_t get_system_time_secs() const;
	virtual uint64_t get_system_time_msecs() const;

	virtual void delay_usec(uint32_t p_usec) const = 0;
	virtual void add_frame_delay(bool p_can_draw);

	virtual uint64_t get_ticks_usec() const = 0;
	uint32_t get_ticks_msec() const;
	uint64_t get_splash_tick_msec() const;

	virtual bool can_draw() const = 0;

	virtual bool is_userfs_persistent() const { return true; }

	bool is_stdout_verbose() const;
	bool is_stdout_debug_enabled() const;

	virtual void disable_crash_handler() {}
	virtual bool is_disable_crash_handler() const { return false; }
	virtual void initialize_debugging() {}

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

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_input_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void hide_virtual_keyboard();

	// returns height of the currently shown virtual keyboard (0 if keyboard is hidden)
	virtual int get_virtual_keyboard_height() const;

	virtual void set_cursor_shape(CursorShape p_shape);
	virtual CursorShape get_cursor_shape() const;
	virtual void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);

	virtual bool get_swap_ok_cancel() { return false; }
	virtual void dump_memory_to_file(const char *p_file);
	virtual void dump_resources_to_file(const char *p_file);
	virtual void print_resources_in_use(bool p_short = false);
	virtual void print_all_resources(String p_to_file = "");

	virtual uint64_t get_static_memory_usage() const;
	virtual uint64_t get_static_memory_peak_usage() const;
	virtual uint64_t get_dynamic_memory_usage() const;
	virtual uint64_t get_free_static_memory() const;

	RenderThreadMode get_render_thread_mode() const { return _render_thread_mode; }

	virtual String get_locale() const;
	String get_locale_language() const;

	String get_safe_dir_name(const String &p_dir_name, bool p_allow_dir_separator = false) const;
	virtual String get_godot_dir_name() const;

	virtual String get_data_path() const;
	virtual String get_config_path() const;
	virtual String get_cache_path() const;
	virtual String get_bundle_resource_dir() const;
	virtual String get_bundle_icon_path() const;

	virtual String get_user_data_dir() const;
	virtual String get_resource_dir() const;

	enum SystemDir {
		SYSTEM_DIR_DESKTOP,
		SYSTEM_DIR_DCIM,
		SYSTEM_DIR_DOCUMENTS,
		SYSTEM_DIR_DOWNLOADS,
		SYSTEM_DIR_MOVIES,
		SYSTEM_DIR_MUSIC,
		SYSTEM_DIR_PICTURES,
		SYSTEM_DIR_RINGTONES,
	};

	virtual String get_system_dir(SystemDir p_dir, bool p_shared_storage = true) const;

	virtual Error move_to_trash(const String &p_path) { return FAILED; }

	virtual void set_no_window_mode(bool p_enable);
	virtual bool is_no_window_mode_enabled() const;

	virtual bool has_touchscreen_ui_hint() const;

	enum ScreenOrientation {

		SCREEN_LANDSCAPE,
		SCREEN_PORTRAIT,
		SCREEN_REVERSE_LANDSCAPE,
		SCREEN_REVERSE_PORTRAIT,
		SCREEN_SENSOR_LANDSCAPE,
		SCREEN_SENSOR_PORTRAIT,
		SCREEN_SENSOR,
	};

	virtual void set_screen_orientation(ScreenOrientation p_orientation);
	virtual ScreenOrientation get_screen_orientation() const;
	ScreenOrientation get_screen_orientation_from_string(const String &p_orientation) const;

	virtual void enable_for_stealing_focus(ProcessID pid) {}
	virtual void move_window_to_foreground() {}

	virtual void debug_break();

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual void set_native_icon(const String &p_filename);
	virtual void set_icon(const Ref<Image> &p_icon);

	virtual int get_exit_code() const;
	virtual void set_exit_code(int p_code);
	virtual bool is_custom_exit_code();

	virtual int get_processor_count() const;

	virtual String get_unique_id() const;

	virtual Error native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track);
	virtual bool native_video_is_playing() const;
	virtual void native_video_pause();
	virtual void native_video_unpause();
	virtual void native_video_stop();

	virtual bool can_use_threads() const;

	virtual Error dialog_show(String p_title, String p_description, Vector<String> p_buttons, Object *p_obj, String p_callback);
	virtual Error dialog_input_text(String p_title, String p_description, String p_partial, Object *p_obj, String p_callback);

	enum LatinKeyboardVariant {
		LATIN_KEYBOARD_QWERTY,
		LATIN_KEYBOARD_QWERTZ,
		LATIN_KEYBOARD_AZERTY,
		LATIN_KEYBOARD_QZERTY,
		LATIN_KEYBOARD_DVORAK,
		LATIN_KEYBOARD_NEO,
		LATIN_KEYBOARD_COLEMAK,
	};

	virtual LatinKeyboardVariant get_latin_keyboard_variant() const;

	virtual int keyboard_get_layout_count() const;
	virtual int keyboard_get_current_layout() const;
	virtual void keyboard_set_current_layout(int p_index);
	virtual String keyboard_get_layout_language(int p_index) const;
	virtual String keyboard_get_layout_name(int p_index) const;

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;

	enum EngineContext {
		CONTEXT_EDITOR,
		CONTEXT_PROJECTMAN,
		CONTEXT_ENGINE,
	};

	virtual void set_context(int p_context);

	//amazing hack because OpenGL needs this to be set on a separate thread..
	//also core can't access servers, so a callback must be used
	typedef void (*SwitchVSyncCallbackInThread)(bool);

	static SwitchVSyncCallbackInThread switch_vsync_function;
	void set_use_vsync(bool p_enable);
	bool is_vsync_enabled() const;

	//real, actual overridable function to switch vsync, which needs to be called from graphics thread if needed
	virtual void _set_use_vsync(bool p_enable) {}

	void set_vsync_via_compositor(bool p_enable);
	bool is_vsync_via_compositor_enabled() const;

	void set_delta_smoothing(bool p_enabled);
	bool is_delta_smoothing_enabled() const;

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	virtual void force_process_input(){};
	bool has_feature(const String &p_feature);

	void set_has_server_feature_callback(HasServerFeatureCallback p_callback);

	bool is_layered_allowed() const { return _allow_layered; }
	bool is_hidpi_allowed() const { return _allow_hidpi; }

	void set_restart_on_exit(bool p_restart, const List<String> &p_restart_arguments);
	bool is_restart_on_exit_set() const;
	List<String> get_restart_on_exit_arguments() const;

	virtual bool request_permission(const String &p_name) { return true; }
	virtual bool request_permissions() { return true; }
	virtual Vector<String> get_granted_permissions() const { return Vector<String>(); }

	virtual void process_and_drop_events() {}
	OS();
	virtual ~OS();
};

VARIANT_ENUM_CAST(OS::PowerState);

#endif
