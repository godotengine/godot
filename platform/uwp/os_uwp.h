/*************************************************************************/
/*  os_uwp.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef OS_UWP_H
#define OS_UWP_H

#include "context_egl_uwp.h"
#include "core/math/transform_2d.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "core/ustring.h"
#include "drivers/xaudio2/audio_driver_xaudio2.h"
#include "joypad_uwp.h"
#include "main/input_default.h"
#include "power_uwp.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"

#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

class OS_UWP : public OS {
public:
	struct KeyEvent {
		enum MessageType {
			KEY_EVENT_MESSAGE,
			CHAR_EVENT_MESSAGE
		};

		bool alt, shift, control;
		MessageType type;
		bool pressed;
		unsigned int scancode;
		unsigned int physical_scancode;
		unsigned int unicode;
		bool echo;
		CorePhysicalKeyStatus status;
	};

private:
	enum {
		JOYPADS_MAX = 8,
		JOY_AXIS_COUNT = 6,
		MAX_JOY_AXIS = 32768, // I've no idea
		KEY_EVENT_BUFFER_SIZE = 512
	};

	FILE *stdo;

	KeyEvent key_event_buffer[KEY_EVENT_BUFFER_SIZE];
	int key_event_pos;

	uint64_t ticks_start;
	uint64_t ticks_per_second;

	bool minimized;
	bool old_invalid;
	bool outside;
	int old_x, old_y;
	Point2i center;
	VisualServer *visual_server;
	int pressrc;

	ContextEGL_UWP *gl_context;
	Windows::UI::Core::CoreWindow ^ window;

	VideoMode video_mode;
	int video_driver_index;

	MainLoop *main_loop;

	AudioDriverXAudio2 audio_driver;

	PowerUWP *power_manager;

	MouseMode mouse_mode;
	bool alt_mem;
	bool gr_mem;
	bool shift_mem;
	bool control_mem;
	bool meta_mem;
	bool force_quit;
	uint32_t last_button_state;

	CursorShape cursor_shape;

	InputDefault *input;

	JoypadUWP ^ joypad;

	Windows::System::Display::DisplayRequest ^ display_request;

	void _post_dpad(DWORD p_dpad, int p_device, bool p_pressed);

	void _drag_event(int idx, UINT uMsg, WPARAM wParam, LPARAM lParam);
	void _touch_event(int idx, UINT uMsg, WPARAM wParam, LPARAM lParam);

	ref class ManagedType {
	public:
		property bool alert_close_handle;
		property Platform::String ^ clipboard;
		void alert_close(Windows::UI::Popups::IUICommand ^ command);
		void on_clipboard_changed(Platform::Object ^ sender, Platform::Object ^ ev);
		void update_clipboard();
		void on_accelerometer_reading_changed(Windows::Devices::Sensors::Accelerometer ^ sender, Windows::Devices::Sensors::AccelerometerReadingChangedEventArgs ^ args);
		void on_magnetometer_reading_changed(Windows::Devices::Sensors::Magnetometer ^ sender, Windows::Devices::Sensors::MagnetometerReadingChangedEventArgs ^ args);
		void on_gyroscope_reading_changed(Windows::Devices::Sensors::Gyrometer ^ sender, Windows::Devices::Sensors::GyrometerReadingChangedEventArgs ^ args);

		/** clang-format breaks this, it does not understand this token. */
		/* clang-format off */
	internal:
		ManagedType() { alert_close_handle = false; }
		property OS_UWP* os;
		/* clang-format on */
	};
	ManagedType ^ managed_object;
	Windows::Devices::Sensors::Accelerometer ^ accelerometer;
	Windows::Devices::Sensors::Magnetometer ^ magnetometer;
	Windows::Devices::Sensors::Gyrometer ^ gyrometer;

	// functions used by main to initialize/deinitialize the OS
protected:
	virtual int get_video_driver_count() const;
	virtual int get_current_video_driver() const;

	virtual void initialize_core();
	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();
	virtual void finalize_core();

	void process_events();

	void process_key_events();

public:
	// Event to send to the app wrapper
	HANDLE mouse_mode_changed;

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");
	String get_stdin_string(bool p_block);

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;
	virtual Size2 get_window_size() const;
	virtual void set_window_size(const Size2 p_size);
	virtual void set_window_fullscreen(bool p_enabled);
	virtual bool is_window_fullscreen() const;
	virtual void set_keep_screen_on(bool p_enabled);

	virtual MainLoop *get_main_loop() const;

	virtual String get_name() const;

	virtual Date get_date(bool utc) const;
	virtual Time get_time(bool utc) const;
	virtual TimeZoneInfo get_time_zone_info() const;
	virtual uint64_t get_unix_time() const;

	virtual bool can_draw() const;
	virtual Error set_cwd(const String &p_cwd);

	virtual void delay_usec(uint32_t p_usec) const;
	virtual uint64_t get_ticks_usec() const;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking = true, ProcessID *r_child_id = NULL, String *r_pipe = NULL, int *r_exitcode = NULL, bool read_stderr = false, Mutex *p_pipe_mutex = NULL);
	virtual Error kill(const ProcessID &p_pid);

	virtual bool has_environment(const String &p_var) const;
	virtual String get_environment(const String &p_var) const;
	virtual bool set_environment(const String &p_var, const String &p_value) const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	void set_cursor_shape(CursorShape p_shape);
	CursorShape get_cursor_shape() const;
	virtual void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);
	void set_icon(const Ref<Image> &p_icon);

	virtual String get_executable_path() const;

	virtual String get_locale() const;

	virtual void move_window_to_foreground();
	virtual String get_user_data_dir() const;

	virtual bool _check_internal_feature_support(const String &p_feature);

	void set_window(Windows::UI::Core::CoreWindow ^ p_window);
	void screen_size_changed();

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual bool has_touchscreen_ui_hint() const;

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_input_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void hide_virtual_keyboard();

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);
	virtual Error close_dynamic_library(void *p_library_handle);
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual Error shell_open(String p_uri);

	void run();

	virtual bool get_swap_ok_cancel() { return true; }

	void input_event(const Ref<InputEvent> &p_event);

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	void queue_key_event(KeyEvent &p_event);

	OS_UWP();
	~OS_UWP();
};

#endif
