/*************************************************************************/
/*  os_x11.h                                                             */
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
#ifndef OS_X11_H
#define OS_X11_H


#include "os/input.h"
#include "drivers/unix/os_unix.h"
#include "context_gl_x11.h"
#include "servers/visual_server.h"
#include "servers/visual/visual_server_wrap_mt.h"
#include "servers/visual/rasterizer.h"
#include "servers/physics_server.h"
#include "servers/audio/audio_server_sw.h"
#include "servers/audio/sample_manager_sw.h"
#include "servers/spatial_sound/spatial_sound_server_sw.h"
#include "servers/spatial_sound_2d/spatial_sound_2d_server_sw.h"
#include "drivers/rtaudio/audio_driver_rtaudio.h"
#include "drivers/alsa/audio_driver_alsa.h"
#include "drivers/pulseaudio/audio_driver_pulseaudio.h"
#include "servers/physics_2d/physics_2d_server_sw.h"

#include <X11/keysym.h>
#include <X11/Xlib.h>
#include <X11/Xcursor/Xcursor.h>

// Hints for X11 fullscreen
typedef struct {
	unsigned long flags;
	unsigned long functions;
	unsigned long decorations;
	long inputMode;
	unsigned long status;
} Hints;

#undef CursorShape
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class OS_X11 : public OS_Unix {

	Atom wm_delete;
#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
	ContextGL_X11 *context_gl;
#endif
	Rasterizer *rasterizer;
	VisualServer *visual_server;
	VideoMode current_videomode;
	List<String> args;
	Window x11_window;
	MainLoop *main_loop;	
	::Display* x11_display;
	char *xmbstring;
	int xmblen;
	unsigned long last_timestamp;
	::Time last_keyrelease_time;
	::XIC xic;
	::XIM xim;
	::XIMStyle xim_style;
	Point2i last_mouse_pos;
	bool last_mouse_pos_valid;
	Point2i last_click_pos;
	uint64_t last_click_ms;
	unsigned int event_id;
	uint32_t last_button_state;
	
	PhysicsServer *physics_server;
	unsigned int get_mouse_button_state(unsigned int p_x11_state);
	InputModifierState get_key_modifier_state(unsigned int p_x11_state);
	Physics2DServer *physics_2d_server;

	MouseMode mouse_mode;
	Point2i center;
	
	void handle_key_event(XKeyEvent *p_event,bool p_echo=false);
	void process_xevents();
	virtual void delete_main_loop();
	IP_Unix *ip_unix;

	AudioServerSW *audio_server;
	SampleManagerMallocSW *sample_manager;
	SpatialSoundServerSW *spatial_sound_server;
	SpatialSound2DServerSW *spatial_sound_2d_server;

	bool force_quit;
	bool minimized;
	int dpad_last[2];


	const char *cursor_theme;
	int cursor_size;
	Cursor cursors[CURSOR_MAX];
	Cursor null_cursor;
	CursorShape current_cursor;

	InputDefault *input;

#ifdef RTAUDIO_ENABLED
	AudioDriverRtAudio driver_rtaudio;
#endif

#ifdef ALSA_ENABLED
	AudioDriverALSA driver_alsa;
#endif

#ifdef PULSEAUDIO_ENABLED
	AudioDriverPulseAudio driver_pulseaudio;
#endif

	enum {
		JOYSTICKS_MAX = 8,
		MAX_JOY_AXIS = 32768, // I've no idea
	};

	struct Joystick {

		int fd;
		int last_axis[JOY_AXIS_MAX];

		Joystick() {
			fd = -1;
			for (int i=0; i<JOY_AXIS_MAX; i++) {

				last_axis[i] = 0;
			};
		};
	};

	Atom net_wm_icon;


	int joystick_count;
	Joystick joysticks[JOYSTICKS_MAX];


protected:

	virtual int get_video_driver_count() const;
	virtual const char * get_video_driver_name(int p_driver) const;	
	virtual VideoMode get_default_video_mode() const;

    virtual int get_audio_driver_count() const;
    virtual const char * get_audio_driver_name(int p_driver) const;

	virtual void initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver);	
	virtual void finalize();

	virtual void set_main_loop( MainLoop * p_main_loop );    

	void probe_joystick(int p_id = -1);
	void process_joysticks();
	void close_joystick(int p_id = -1);

public:

	virtual String get_name();

	virtual void set_cursor_shape(CursorShape p_shape);

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	virtual void warp_mouse_pos(const Point2& p_to);
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String& p_title);

	virtual void set_icon(const Image& p_icon);

	virtual MainLoop *get_main_loop() const;
	
	virtual bool can_draw() const;

	virtual void set_clipboard(const String& p_text);
	virtual String get_clipboard() const;

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual String get_system_dir(SystemDir p_dir) const;

	virtual Error shell_open(String p_uri);

	virtual void set_video_mode(const VideoMode& p_video_mode,int p_screen=0);
	virtual VideoMode get_video_mode(int p_screen=0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen=0) const;

	virtual void move_window_to_foreground();

	void run();

	OS_X11();
};

#endif
