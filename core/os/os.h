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

#include "core/config/engine.h"
#include "core/io/image.h"
#include "core/io/logger.h"
#include "core/os/main_loop.h"
#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "core/templates/vector.h"

#include <stdarg.h>
#include <stdlib.h>

class OS {
	static OS *singleton;
	static uint64_t target_ticks;
	String _execpath;
	List<String> _cmdline;
	bool _keep_screen_on = true; // set default value to true, because this had been true before godot 2.0.
	bool low_processor_usage_mode = false;
	int low_processor_usage_mode_sleep_usec = 10000;
	bool _verbose_stdout = false;
	bool _debug_stdout = false;
	bool _single_window = false;
	String _local_clipboard;
	int _exit_code = EXIT_FAILURE; // unexpected exit is marked as failure
	int _orientation;
	bool _allow_hidpi = false;
	bool _allow_layered = false;
	bool _stdout_enabled = true;
	bool _stderr_enabled = true;

	char *last_error;

	CompositeLogger *_logger = nullptr;

	bool restart_on_exit = false;
	List<String> restart_commandline;

	// for the user interface we keep a record of the current display driver
	// so we can retrieve the rendering drivers available
	int _display_driver_id = -1;
	String _current_rendering_driver_name = "";

protected:
	void _set_logger(CompositeLogger *p_logger);

public:
	typedef void (*ImeCallback)(void *p_inp, String p_text, Point2 p_selection);
	typedef bool (*HasServerFeatureCallback)(const String &p_feature);

	enum RenderThreadMode {
		RENDER_THREAD_UNSAFE,
		RENDER_THREAD_SAFE,
		RENDER_SEPARATE_THREAD
	};

	enum RenderMainThreadMode {
		RENDER_MAIN_THREAD_ONLY,
		RENDER_ANY_THREAD,
	};

protected:
	friend class Main;
	// Needed by tests to setup command-line args.
	friend int test_main(int argc, char *argv[]);

	HasServerFeatureCallback has_server_feature_callback = nullptr;
	RenderThreadMode _render_thread_mode = RENDER_THREAD_SAFE;
	RenderMainThreadMode _render_main_thread_mode = RENDER_ANY_THREAD;

	// Functions used by Main to initialize/deinitialize the OS.
	void add_logger(Logger *p_logger);

	virtual void initialize() = 0;
	virtual void initialize_joypads() = 0;

	void set_current_rendering_driver_name(String p_driver_name) { _current_rendering_driver_name = p_driver_name; }
	void set_display_driver_id(int p_display_driver_id) { _display_driver_id = p_display_driver_id; }

	virtual void set_main_loop(MainLoop *p_main_loop) = 0;
	virtual void delete_main_loop() = 0;

	virtual void finalize() = 0;
	virtual void finalize_core() = 0;

	virtual void set_cmdline(const char *p_execpath, const List<String> &p_args);

	virtual bool _check_internal_feature_support(const String &p_feature) = 0;

public:
	typedef int64_t ProcessID;

	static OS *get_singleton();

	String get_current_rendering_driver_name() const { return _current_rendering_driver_name; }
	int get_display_driver_id() const { return _display_driver_id; }

	void print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify = false, Logger::ErrorType p_type = Logger::ERR_ERROR);
	void print(const char *p_format, ...) _PRINTF_FORMAT_ATTRIBUTE_2_3;
	void printerr(const char *p_format, ...) _PRINTF_FORMAT_ATTRIBUTE_2_3;

	virtual String get_stdin_string(bool p_block = true) = 0;

	virtual PackedStringArray get_connected_midi_inputs();
	virtual void open_midi_inputs();
	virtual void close_midi_inputs();

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false) { return ERR_UNAVAILABLE; }
	virtual Error close_dynamic_library(void *p_library_handle) { return ERR_UNAVAILABLE; }
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false) { return ERR_UNAVAILABLE; }

	virtual void set_low_processor_usage_mode(bool p_enabled);
	virtual bool is_in_low_processor_usage_mode() const;
	virtual void set_low_processor_usage_mode_sleep_usec(int p_usec);
	virtual int get_low_processor_usage_mode_sleep_usec() const;

	virtual String get_executable_path() const;
	virtual Error execute(const String &p_path, const List<String> &p_arguments, String *r_pipe = nullptr, int *r_exitcode = nullptr, bool read_stderr = false, Mutex *p_pipe_mutex = nullptr, bool p_open_console = false) = 0;
	virtual Error create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id = nullptr, bool p_open_console = false) = 0;
	virtual Error create_instance(const List<String> &p_arguments, ProcessID *r_child_id = nullptr) { return create_process(get_executable_path(), p_arguments, r_child_id); };
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

	bool is_layered_allowed() const { return _allow_layered; }
	bool is_hidpi_allowed() const { return _allow_hidpi; }

	void ensure_user_data_dir();

	virtual MainLoop *get_main_loop() const = 0;

	virtual void yield();

	enum Weekday : uint8_t {
		WEEKDAY_SUNDAY,
		WEEKDAY_MONDAY,
		WEEKDAY_TUESDAY,
		WEEKDAY_WEDNESDAY,
		WEEKDAY_THURSDAY,
		WEEKDAY_FRIDAY,
		WEEKDAY_SATURDAY,
	};

	enum Month : uint8_t {
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
		MONTH_DECEMBER,
	};

	struct Date {
		int64_t year;
		Month month;
		uint8_t day;
		Weekday weekday;
		bool dst;
	};

	struct Time {
		uint8_t hour;
		uint8_t minute;
		uint8_t second;
	};

	struct TimeZoneInfo {
		int bias;
		String name;
	};

	virtual Date get_date(bool p_utc = false) const = 0;
	virtual Time get_time(bool p_utc = false) const = 0;
	virtual TimeZoneInfo get_time_zone_info() const = 0;
	virtual double get_unix_time() const;

	virtual void delay_usec(uint32_t p_usec) const = 0;
	virtual void add_frame_delay(bool p_can_draw);

	virtual uint64_t get_ticks_usec() const = 0;
	uint64_t get_ticks_msec() const;

	virtual bool is_userfs_persistent() const { return true; }

	bool is_stdout_verbose() const;
	bool is_stdout_debug_enabled() const;

	bool is_stdout_enabled() const;
	bool is_stderr_enabled() const;
	void set_stdout_enabled(bool p_enabled);
	void set_stderr_enabled(bool p_enabled);

	virtual bool is_single_window() const;

	virtual void disable_crash_handler() {}
	virtual bool is_disable_crash_handler() const { return false; }
	virtual void initialize_debugging() {}

	virtual void dump_memory_to_file(const char *p_file);
	virtual void dump_resources_to_file(const char *p_file);
	virtual void print_resources_in_use(bool p_short = false);
	virtual void print_all_resources(String p_to_file = "");

	virtual uint64_t get_static_memory_usage() const;
	virtual uint64_t get_static_memory_peak_usage() const;
	virtual uint64_t get_free_static_memory() const;

	RenderThreadMode get_render_thread_mode() const { return _render_thread_mode; }
	RenderMainThreadMode get_render_main_thread_mode() const { return _render_main_thread_mode; }
	void set_render_main_thread_mode(RenderMainThreadMode p_thread_mode) { _render_main_thread_mode = p_thread_mode; }

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

	virtual void debug_break();

	virtual int get_exit_code() const;
	// `set_exit_code` should only be used from `SceneTree` (or from a similar
	// level, e.g. from the `Main::start` if leaving without creating a `SceneTree`).
	// For other components, `SceneTree.quit()` should be used instead.
	virtual void set_exit_code(int p_code);

	virtual int get_processor_count() const;
	virtual int get_default_thread_pool_size() const { return get_processor_count(); }

	virtual String get_unique_id() const;

	virtual bool can_use_threads() const;

	bool has_feature(const String &p_feature);

	void set_has_server_feature_callback(HasServerFeatureCallback p_callback);

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

#endif // OS_H
