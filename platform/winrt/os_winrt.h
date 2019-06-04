/*************************************************************************/
/*  os_winrt.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef OSWinrt_H
#define OSWinrt_H

#include "os/input.h"
#include "os/os.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"

#include "servers/audio/audio_driver_dummy.h"
#include "servers/audio/audio_server_sw.h"
#include "servers/audio/sample_manager_sw.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "servers/spatial_sound/spatial_sound_server_sw.h"
#include "servers/spatial_sound_2d/spatial_sound_2d_server_sw.h"

#include "gl_context_egl.h"

#include "core/math/math_2d.h"
#include "core/ustring.h"

#include <io.h>

#include "main/input_default.h"
#include <fcntl.h>
#include <stdio.h>

#include "audio_driver_winrt.h"
#include "joystick_winrt.h"

#include <windows.h>

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class OSWinrt : public OS {

public:
	struct KeyEvent {

		enum MessageType {
			KEY_EVENT_MESSAGE,
			CHAR_EVENT_MESSAGE
		};

		InputModifierState mod_state;
		MessageType type;
		bool pressed;
		unsigned int scancode;
		unsigned int unicode;
		bool echo;
		CorePhysicalKeyStatus status;
	};

private:
	enum {
		JOYSTICKS_MAX = 8,
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
	unsigned int last_id;
	VisualServer *visual_server;
	Rasterizer *rasterizer;
	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;
	int pressrc;

	ContextEGL *gl_context;

	VideoMode video_mode;

	MainLoop *main_loop;

	AudioDriverWinRT audio_driver;
	AudioServerSW *audio_server;
	SampleManagerMallocSW *sample_manager;
	SpatialSoundServerSW *spatial_sound_server;
	SpatialSound2DServerSW *spatial_sound_2d_server;

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

	JoystickWinrt ^ joystick;

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

		internal : ManagedType() { alert_close_handle = false; }
		property OSWinrt *os;
	};
	ManagedType ^ managed_object;
	Windows::Devices::Sensors::Accelerometer ^ accelerometer;
	Windows::Devices::Sensors::Magnetometer ^ magnetometer;
	Windows::Devices::Sensors::Gyrometer ^ gyrometer;

	// functions used by main to initialize/deintialize the OS
protected:
	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;

	virtual VideoMode get_default_video_mode() const;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual void initialize_core();
	virtual void initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();
	virtual void finalize_core();

	void process_events();

	void process_key_events();

public:
	void print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, ErrorType p_type);

	// Event to send to the app wrapper
	HANDLE mouse_mode_changed;

	virtual void vprint(const char *p_format, va_list p_list, bool p_stderr = false);
	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");
	String get_stdin_string(bool p_block);

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	virtual Point2 get_mouse_pos() const;
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

	virtual String get_name();

	virtual Date get_date(bool utc) const;
	virtual Time get_time(bool utc) const;
	virtual TimeZoneInfo get_time_zone_info() const;
	virtual uint64_t get_unix_time() const;

	virtual bool can_draw() const;
	virtual Error set_cwd(const String &p_cwd);

	virtual void delay_usec(uint32_t p_usec) const;
	virtual uint64_t get_ticks_usec() const;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id = NULL, String *r_pipe = NULL, int *r_exitcode = NULL, bool read_stderr = false);
	virtual Error kill(const ProcessID &p_pid);

	virtual bool has_environment(const String &p_var) const;
	virtual String get_environment(const String &p_var) const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	void set_cursor_shape(CursorShape p_shape);
	void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);
	void set_icon(const Image &p_icon);

	virtual String get_executable_path() const;

	virtual String get_locale() const;

	virtual void move_window_to_foreground();
	virtual String get_data_dir() const;

	void set_gl_context(ContextEGL *p_context);
	void screen_size_changed();

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual bool has_touchscreen_ui_hint() const;

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2());
	virtual void hide_virtual_keyboard();

	virtual Error shell_open(String p_uri);

	void run();

	virtual bool get_swap_ok_cancel() { return true; }

	void input_event(InputEvent &p_event);

	void queue_key_event(KeyEvent &p_event);

	OSWinrt();
	~OSWinrt();
};

#endif
