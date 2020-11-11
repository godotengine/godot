/*************************************************************************/
/*  os_javascript.h                                                      */
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

#ifndef OS_JAVASCRIPT_H
#define OS_JAVASCRIPT_H

#include "audio_driver_javascript.h"
#include "drivers/unix/os_unix.h"
#include "main/input_default.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"

#include <emscripten/html5.h>

class OS_JavaScript : public OS_Unix {
private:
	VideoMode video_mode;
	Vector2 windowed_size;
	bool window_maximized;
	bool entering_fullscreen;
	bool just_exited_fullscreen;
	bool transparency_enabled;

	EMSCRIPTEN_WEBGL_CONTEXT_HANDLE webgl_ctx;

	InputDefault *input;
	Ref<InputEventKey> deferred_key_event;
	CursorShape cursor_shape;
	Point2 touches[32];

	char canvas_id[256];
	bool cursor_inside_canvas;
	Point2i last_click_pos;
	double last_click_ms;
	int last_click_button_index;

	int last_width;
	int last_height;

	MainLoop *main_loop;
	int video_driver_index;
	AudioDriverJavaScript *audio_driver_javascript;
	VisualServer *visual_server;

	bool swap_ok_cancel;
	bool idb_available;
	bool idb_needs_sync;
	bool idb_is_syncing;

	static Point2 compute_position_in_canvas(int x, int y);
	static EM_BOOL fullscreen_change_callback(int p_event_type, const EmscriptenFullscreenChangeEvent *p_event, void *p_user_data);

	static EM_BOOL keydown_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data);
	static EM_BOOL keypress_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data);
	static EM_BOOL keyup_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data);

	static EM_BOOL mousemove_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data);
	static EM_BOOL mouse_button_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data);

	static EM_BOOL wheel_callback(int p_event_type, const EmscriptenWheelEvent *p_event, void *p_user_data);

	static EM_BOOL touch_press_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data);
	static EM_BOOL touchmove_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data);

	static EM_BOOL gamepad_change_callback(int p_event_type, const EmscriptenGamepadEvent *p_event, void *p_user_data);
	void process_joypads();

	static void file_access_close_callback(const String &p_file, int p_flags);

	static void request_quit_callback();
	static void drop_files_callback(char **p_filev, int p_filec);
	static void send_notification_callback(int p_notification);
	static void fs_sync_callback();
	static void update_clipboard_callback(const char *p_text);

protected:
	void resume_audio();

	virtual int get_current_video_driver() const;

	virtual void initialize_core();
	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();

	virtual bool _check_internal_feature_support(const String &p_feature);

public:
	bool check_size_force_redraw();

	// Override return type to make writing static callbacks less tedious.
	static OS_JavaScript *get_singleton();

	virtual bool get_swap_ok_cancel();
	virtual void swap_buffers();
	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual void set_window_size(const Size2);
	virtual Size2 get_window_size() const;
	virtual void set_window_maximized(bool p_enabled);
	virtual bool is_window_maximized() const;
	virtual void set_window_fullscreen(bool p_enabled);
	virtual bool is_window_fullscreen() const;
	virtual Size2 get_screen_size(int p_screen = -1) const;

	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;
	virtual void set_cursor_shape(CursorShape p_shape);
	virtual void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);
	virtual void set_mouse_mode(MouseMode p_mode);
	virtual MouseMode get_mouse_mode() const;

	virtual bool get_window_per_pixel_transparency_enabled() const;
	virtual void set_window_per_pixel_transparency_enabled(bool p_enabled);

	virtual bool has_touchscreen_ui_hint() const;

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;

	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	virtual MainLoop *get_main_loop() const;
	bool main_loop_iterate();

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking = true, ProcessID *r_child_id = NULL, String *r_pipe = NULL, int *r_exitcode = NULL, bool read_stderr = false, Mutex *p_pipe_mutex = NULL);
	virtual Error kill(const ProcessID &p_pid);
	virtual int get_process_id() const;

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");
	virtual void set_window_title(const String &p_title);
	virtual void set_icon(const Ref<Image> &p_icon);
	String get_executable_path() const;
	virtual Error shell_open(String p_uri);
	virtual String get_name() const;
	virtual void add_frame_delay(bool p_can_draw) {}
	virtual bool can_draw() const;

	virtual String get_cache_path() const;
	virtual String get_config_path() const;
	virtual String get_data_path() const;
	virtual String get_user_data_dir() const;

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	virtual bool is_userfs_persistent() const;

	OS_JavaScript();
};

#endif
