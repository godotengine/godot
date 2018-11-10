/*************************************************************************/
/*  os_windows.h                                                         */
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

#ifndef OS_WINDOWS_H
#define OS_WINDOWS_H
#include "context_gl_win.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "crash_handler_win.h"
#include "drivers/rtaudio/audio_driver_rtaudio.h"
#include "drivers/wasapi/audio_driver_wasapi.h"
#include "drivers/winmidi/win_midi.h"
#include "power_windows.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"
#ifdef XAUDIO2_ENABLED
#include "drivers/xaudio2/audio_driver_xaudio2.h"
#endif
#include "drivers/unix/ip_unix.h"
#include "key_mapping_win.h"
#include "main/input_default.h"

#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <windows.h>
#include <windowsx.h>

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class JoypadWindows;
class OS_Windows : public OS {

	enum {
		KEY_EVENT_BUFFER_SIZE = 512
	};

	FILE *stdo;

	struct KeyEvent {

		bool alt, shift, control, meta;
		UINT uMsg;
		WPARAM wParam;
		LPARAM lParam;
	};

	KeyEvent key_event_buffer[KEY_EVENT_BUFFER_SIZE];
	int key_event_pos;

	uint64_t ticks_start;
	uint64_t ticks_per_second;

	bool old_invalid;
	bool outside;
	int old_x, old_y;
	Point2i center;
#if defined(OPENGL_ENABLED)
	ContextGL_Win *gl_context;
#endif
	VisualServer *visual_server;
	int pressrc;
	HDC hDC; // Private GDI Device Context
	HINSTANCE hInstance; // Holds The Instance Of The Application
	HWND hWnd;

	HBITMAP hBitmap; //DIB section for layered window
	uint8_t *dib_data;
	Size2 dib_size;
	HDC hDC_dib;
	bool layered_window;

	uint32_t move_timer_id;

	HCURSOR hCursor;

	Size2 window_rect;
	VideoMode video_mode;
	bool preserve_window_size = false;

	MainLoop *main_loop;

	WNDPROC user_proc;

	// IME
	HIMC im_himc;
	Vector2 im_position;

	MouseMode mouse_mode;
	bool alt_mem;
	bool gr_mem;
	bool shift_mem;
	bool control_mem;
	bool meta_mem;
	bool force_quit;
	bool window_has_focus;
	uint32_t last_button_state;
	bool use_raw_input;

	HCURSOR cursors[CURSOR_MAX] = { NULL };
	CursorShape cursor_shape;

	InputDefault *input;
	JoypadWindows *joypad;
	Map<int, Vector2> touch_state;

	PowerWindows *power_manager;

	int video_driver_index;
#ifdef WASAPI_ENABLED
	AudioDriverWASAPI driver_wasapi;
#endif
#ifdef RTAUDIO_ENABLED
	AudioDriverRtAudio driver_rtaudio;
#endif
#ifdef XAUDIO2_ENABLED
	AudioDriverXAudio2 driver_xaudio2;
#endif
#ifdef WINMIDI_ENABLED
	MIDIDriverWinMidi driver_midi;
#endif

	CrashHandler crash_handler;

	void _drag_event(float p_x, float p_y, int idx);
	void _touch_event(bool p_pressed, float p_x, float p_y, int idx);

	void _update_window_style(bool repaint = true);

	// functions used by main to initialize/deinitialize the OS
protected:
	virtual int get_current_video_driver() const;

	virtual void initialize_core();
	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();
	virtual void finalize_core();

	void process_events();
	void process_key_events();

	struct ProcessInfo {

		STARTUPINFO si;
		PROCESS_INFORMATION pi;
	};
	Map<ProcessID, ProcessInfo> *process_map;

	bool pre_fs_valid;
	RECT pre_fs_rect;
	bool maximized;
	bool minimized;
	bool borderless;

public:
	LRESULT WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");
	String get_stdin_string(bool p_block);

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	virtual void warp_mouse_position(const Point2 &p_to);
	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual int get_screen_count() const;
	virtual int get_current_screen() const;
	virtual void set_current_screen(int p_screen);
	virtual Point2 get_screen_position(int p_screen = -1) const;
	virtual Size2 get_screen_size(int p_screen = -1) const;
	virtual int get_screen_dpi(int p_screen = -1) const;

	virtual Point2 get_window_position() const;
	virtual void set_window_position(const Point2 &p_position);
	virtual Size2 get_window_size() const;
	virtual Size2 get_real_window_size() const;
	virtual void set_window_size(const Size2 p_size);
	virtual void set_window_fullscreen(bool p_enabled);
	virtual bool is_window_fullscreen() const;
	virtual void set_window_resizable(bool p_enabled);
	virtual bool is_window_resizable() const;
	virtual void set_window_minimized(bool p_enabled);
	virtual bool is_window_minimized() const;
	virtual void set_window_maximized(bool p_enabled);
	virtual bool is_window_maximized() const;
	virtual void set_window_always_on_top(bool p_enabled);
	virtual bool is_window_always_on_top() const;
	virtual void request_attention();

	virtual void set_borderless_window(bool p_borderless);
	virtual bool get_borderless_window();

	virtual bool get_window_per_pixel_transparency_enabled() const;
	virtual void set_window_per_pixel_transparency_enabled(bool p_enabled);

	virtual uint8_t *get_layered_buffer_data();
	virtual Size2 get_layered_buffer_size();
	virtual void swap_layered_buffer();

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);
	virtual Error close_dynamic_library(void *p_library_handle);
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual MainLoop *get_main_loop() const;

	virtual String get_name();

	virtual Date get_date(bool utc) const;
	virtual Time get_time(bool utc) const;
	virtual TimeZoneInfo get_time_zone_info() const;
	virtual uint64_t get_unix_time() const;
	virtual uint64_t get_system_time_secs() const;

	virtual bool can_draw() const;
	virtual Error set_cwd(const String &p_cwd);

	virtual void delay_usec(uint32_t p_usec) const;
	virtual uint64_t get_ticks_usec() const;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id = NULL, String *r_pipe = NULL, int *r_exitcode = NULL, bool read_stderr = false);
	virtual Error kill(const ProcessID &p_pid);
	virtual int get_process_id() const;

	virtual bool has_environment(const String &p_var) const;
	virtual String get_environment(const String &p_var) const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	void set_cursor_shape(CursorShape p_shape);
	virtual void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);
	void GetMaskBitmaps(HBITMAP hSourceBitmap, COLORREF clrTransparent, OUT HBITMAP &hAndMaskBitmap, OUT HBITMAP &hXorMaskBitmap);
	void set_icon(const Ref<Image> &p_icon);

	virtual String get_executable_path() const;

	virtual String get_locale() const;

	virtual int get_processor_count() const;

	virtual LatinKeyboardVariant get_latin_keyboard_variant() const;

	virtual void enable_for_stealing_focus(ProcessID pid);
	virtual void move_window_to_foreground();

	virtual String get_config_path() const;
	virtual String get_data_path() const;
	virtual String get_cache_path() const;
	virtual String get_godot_dir_name() const;

	virtual String get_system_dir(SystemDir p_dir) const;
	virtual String get_user_data_dir() const;

	virtual String get_unique_id() const;

	virtual void set_ime_active(const bool p_active);
	virtual void set_ime_position(const Point2 &p_pos);

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual Error shell_open(String p_uri);

	void run();

	virtual bool get_swap_ok_cancel() { return true; }

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;

	virtual void _set_use_vsync(bool p_enable);
	//virtual bool is_vsync_enabled() const;

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	virtual bool _check_internal_feature_support(const String &p_feature);

	void disable_crash_handler();
	bool is_disable_crash_handler() const;
	virtual void initialize_debugging();

	void force_process_input();

	virtual Error move_to_trash(const String &p_path);

	OS_Windows(HINSTANCE _hInstance);
	~OS_Windows();
};

#endif
