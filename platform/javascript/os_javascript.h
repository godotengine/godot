/*************************************************************************/
/*  os_javascript.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "audio_server_javascript.h"
#include "drivers/unix/os_unix.h"
#include "javascript_eval.h"
#include "main/input_default.h"
#include "os/input.h"
#include "os/main_loop.h"
#include "power_javascript.h"
#include "servers/audio_server.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "servers/visual/rasterizer.h"

#include <emscripten/html5.h>

typedef String (*GetDataDirFunc)();

class OS_JavaScript : public OS_Unix {

	bool idbfs_available;
	int64_t time_to_save_sync;
	int64_t last_sync_time;

	VisualServer *visual_server;
	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;
	AudioDriverJavaScript audio_driver_javascript;
	const char *gl_extensions;

	InputDefault *input;
	Vector2 windowed_size;
	bool window_maximized;
	bool soft_fs_enabled;
	bool canvas_size_adjustment_requested;
	VideoMode video_mode;
	CursorShape cursor_shape;
	MainLoop *main_loop;

	GetDataDirFunc get_data_dir_func;

	PowerJavascript *power_manager;

#ifdef JAVASCRIPT_EVAL_ENABLED
	JavaScript *javascript_eval;
#endif

	static void _close_notification_funcs(const String &p_file, int p_flags);

	void process_joypads();

	void set_css_cursor(const char *);
	const char *get_css_cursor() const;

public:
	// functions used by main to initialize/deintialize the OS
	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;

	virtual VideoMode get_default_video_mode() const;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual void initialize_logger();
	virtual void initialize_core();
	virtual void initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();

	typedef int64_t ProcessID;

	//static OS* get_singleton();

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual void set_mouse_mode(MouseMode p_mode);
	virtual MouseMode get_mouse_mode() const;
	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	//virtual void set_clipboard(const String& p_text);
	//virtual String get_clipboard() const;

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual Size2 get_screen_size(int p_screen = -1) const;

	virtual void set_window_size(const Size2);
	virtual Size2 get_window_size() const;
	virtual void set_window_maximized(bool p_enabled);
	virtual bool is_window_maximized() const { return window_maximized; }
	virtual void set_window_fullscreen(bool p_enable);
	virtual bool is_window_fullscreen() const;

	void request_canvas_size_adjustment();

	virtual String get_name();
	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	virtual bool is_userfs_persistent() const;

	virtual void set_cursor_shape(CursorShape p_shape);

	void main_loop_begin();
	bool main_loop_iterate();
	void main_loop_request_quit();
	void main_loop_end();
	void main_loop_focusout();
	void main_loop_focusin();

	virtual bool has_touchscreen_ui_hint() const;

	void set_opengl_extensions(const char *p_gl_extensions);

	virtual Error shell_open(String p_uri);
	virtual String get_data_dir() const;
	String get_executable_path() const;
	virtual String get_resource_dir() const;

	void process_accelerometer(const Vector3 &p_accelerometer);
	void push_input(const Ref<InputEvent> &p_ev);

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;
	bool joy_connection_changed(int p_type, const EmscriptenGamepadEvent *p_event);

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	virtual bool _check_internal_feature_support(const String &p_feature);

	void set_idbfs_available(bool p_idbfs_available);

	OS_JavaScript(const char *p_execpath, GetDataDirFunc p_get_data_dir_func);
	~OS_JavaScript();
};

#endif
