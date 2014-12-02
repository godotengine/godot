/*************************************************************************/
/*  os_windows.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#define WINVER 0x0500

#include "os/input.h"
#include "os/os.h"
#include "context_gl_win.h"
#include "servers/visual_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/physics/physics_server_sw.h"

#include "servers/audio/audio_server_sw.h"
#include "servers/audio/sample_manager_sw.h"
#include "drivers/rtaudio/audio_driver_rtaudio.h"
#include "servers/spatial_sound/spatial_sound_server_sw.h"
#include "servers/spatial_sound_2d/spatial_sound_2d_server_sw.h"
#include "drivers/unix/ip_unix.h"
#include "servers/physics_2d/physics_2d_server_sw.h"


#include <windows.h>

#include "key_mapping_win.h"
#include <windowsx.h>
#include <io.h>

#include <fcntl.h>
#include <stdio.h>
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class OS_Windows : public OS {

	enum {
		JOYSTICKS_MAX = 8,
		JOY_AXIS_COUNT = 6,
		MAX_JOY_AXIS = 32768, // I've no idea
		KEY_EVENT_BUFFER_SIZE=512
	};

	FILE *stdo;


	struct KeyEvent {

		InputModifierState mod_state;
		UINT uMsg;
		WPARAM	wParam;
		LPARAM	lParam;

	};

	KeyEvent key_event_buffer[KEY_EVENT_BUFFER_SIZE];
	int key_event_pos;


	uint64_t ticks_start;
	uint64_t ticks_per_second;

	bool minimized;
        bool old_invalid;
        bool outside;
	int old_x,old_y;
	Point2i center;
	unsigned int last_id;
#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED) || defined(GLES2_ENABLED)
	ContextGL_Win *gl_context;
#endif
	VisualServer *visual_server;
	Rasterizer *rasterizer;
	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;
	int pressrc;
	HDC		hDC;	// Private GDI Device Context
	HINSTANCE	hInstance;		// Holds The Instance Of The Application
	HWND hWnd;

	struct Joystick {

		int id;
		bool attached;

		DWORD last_axis[JOY_AXIS_COUNT];
		DWORD last_buttons;
		DWORD last_pov;
		String name;

		Joystick() {
			id = -1;
			attached = false;
			for (int i=0; i<JOY_AXIS_COUNT; i++) {

				last_axis[i] = 0;
			};
			last_buttons = 0;
			last_pov = 0;
		};
	};

	List<Joystick> joystick_change_queue;
	int joystick_count;
	Joystick joysticks[JOYSTICKS_MAX];
	
	VideoMode video_mode;

	MainLoop *main_loop;

	WNDPROC user_proc;

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

#ifdef RTAUDIO_ENABLED
	AudioDriverRtAudio driver_rtaudio;
#endif

	void _post_dpad(DWORD p_dpad, int p_device, bool p_pressed);

	void _drag_event(int p_x, int p_y, int idx);
	void _touch_event(bool p_pressed, int p_x, int p_y, int idx);

	// functions used by main to initialize/deintialize the OS
protected:	
	virtual int get_video_driver_count() const;
	virtual const char * get_video_driver_name(int p_driver) const;
	
	virtual VideoMode get_default_video_mode() const;
	
	virtual int get_audio_driver_count() const;
	virtual const char * get_audio_driver_name(int p_driver) const;
	
	virtual void initialize_core();
	virtual void initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver);
	
	virtual void set_main_loop( MainLoop * p_main_loop );    
	virtual void delete_main_loop();
	
	virtual void finalize();
	virtual void finalize_core();
	
	void process_events();

	void probe_joysticks();
	void process_joysticks();
	void process_key_events();
	
	struct ProcessInfo {

		STARTUPINFO si;
		PROCESS_INFORMATION pi;
	};
	Map<ProcessID, ProcessInfo>* process_map;

public:
	LRESULT WndProc(HWND	hWnd,UINT uMsg,	WPARAM	wParam,	LPARAM	lParam);


	void print_error(const char* p_function,const char* p_file,int p_line,const char *p_code,const char*p_rationale,ErrorType p_type);

	virtual void vprint(const char *p_format, va_list p_list, bool p_stderr=false);
	virtual void alert(const String& p_alert,const String& p_title="ALERT!");
	String get_stdin_string(bool p_block);

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	virtual void warp_mouse_pos(const Point2& p_to);
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String& p_title);
	
	virtual void set_video_mode(const VideoMode& p_video_mode,int p_screen=0);
	virtual VideoMode get_video_mode(int p_screen=0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen=0) const;

	virtual MainLoop *get_main_loop() const;

	virtual String get_name();
	
	virtual Date get_date() const;
	virtual Time get_time() const;
	virtual uint64_t get_unix_time() const;

	virtual bool can_draw() const;
	virtual Error set_cwd(const String& p_cwd);

	virtual void delay_usec(uint32_t p_usec) const; 
	virtual uint64_t get_ticks_usec() const;

	virtual Error execute(const String& p_path, const List<String>& p_arguments,bool p_blocking,ProcessID *r_child_id=NULL,String* r_pipe=NULL,int *r_exitcode=NULL);
	virtual Error kill(const ProcessID& p_pid);

	virtual bool has_environment(const String& p_var) const;
	virtual String get_environment(const String& p_var) const;

	virtual void set_clipboard(const String& p_text);
	virtual String get_clipboard() const;

	void set_cursor_shape(CursorShape p_shape);
	void set_icon(const Image& p_icon);

	virtual String get_executable_path() const;

	virtual String get_locale() const;

	virtual void move_window_to_foreground();
	virtual String get_data_dir() const;
	virtual String get_system_dir(SystemDir p_dir) const;


	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual Error shell_open(String p_uri);

	void run();

	virtual bool get_swap_ok_cancel() { return true; }

	OS_Windows(HINSTANCE _hInstance);	
	~OS_Windows();

};

#endif
