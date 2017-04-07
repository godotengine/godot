/*************************************************************************/
/*  os_x11.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef OS_X11_H
#define OS_X11_H

#include "context_gl_x11.h"
#include "drivers/unix/os_unix.h"
#include "os/input.h"
#include "servers/visual_server.h"
//#include "servers/visual/visual_server_wrap_mt.h"
#include "drivers/alsa/audio_driver_alsa.h"
#include "drivers/pulseaudio/audio_driver_pulseaudio.h"
#include "drivers/rtaudio/audio_driver_rtaudio.h"
#include "joypad_linux.h"
#include "main/input_default.h"
#include "power_x11.h"
#include "servers/audio/audio_driver_dummy.h"
#include "servers/audio_server.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "servers/physics_2d/physics_2d_server_wrap_mt.h"
#include "servers/physics_server.h"
#include "servers/visual/rasterizer.h"

#include <X11/Xcursor/Xcursor.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>
#include <X11/keysym.h>

// Hints for X11 fullscreen
typedef struct {
	unsigned long flags;
	unsigned long functions;
	unsigned long decorations;
	long inputMode;
	unsigned long status;
} Hints;

typedef struct _xrr_monitor_info {
	Atom name;
	Bool primary;
	Bool automatic;
	int noutput;
	int x;
	int y;
	int width;
	int height;
	int mwidth;
	int mheight;
	RROutput *outputs;
} xrr_monitor_info;

#undef CursorShape
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class OS_X11 : public OS_Unix {

	Atom wm_delete;
	Atom xdnd_enter;
	Atom xdnd_position;
	Atom xdnd_status;
	Atom xdnd_action_copy;
	Atom xdnd_drop;
	Atom xdnd_finished;
	Atom xdnd_selection;
	Atom requested;

	int xdnd_version;

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
	ContextGL_X11 *context_gl;
#endif
	//Rasterizer *rasterizer;
	VisualServer *visual_server;
	VideoMode current_videomode;
	List<String> args;
	Window x11_window;
	Window xdnd_source_window;
	MainLoop *main_loop;
	::Display *x11_display;
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
	uint32_t last_button_state;

	PhysicsServer *physics_server;
	unsigned int get_mouse_button_state(unsigned int p_x11_state);
	InputModifierState get_key_modifier_state(unsigned int p_x11_state);
	Physics2DServer *physics_2d_server;

	MouseMode mouse_mode;
	Point2i center;

	void handle_key_event(XKeyEvent *p_event, bool p_echo = false);
	void process_xevents();
	virtual void delete_main_loop();
	IP_Unix *ip_unix;

	bool force_quit;
	bool minimized;
	bool window_has_focus;
	bool do_mouse_warp;

	const char *cursor_theme;
	int cursor_size;
	XcursorImage *img[CURSOR_MAX];
	Cursor cursors[CURSOR_MAX];
	Cursor null_cursor;
	CursorShape current_cursor;

	InputDefault *input;

#ifdef JOYDEV_ENABLED
	JoypadLinux *joypad;
#endif

#ifdef RTAUDIO_ENABLED
	AudioDriverRtAudio driver_rtaudio;
#endif

#ifdef ALSA_ENABLED
	AudioDriverALSA driver_alsa;
#endif

#ifdef PULSEAUDIO_ENABLED
	AudioDriverPulseAudio driver_pulseaudio;
#endif
	AudioDriverDummy driver_dummy;

	Atom net_wm_icon;

	PowerX11 *power_manager;

	int audio_driver_index;
	unsigned int capture_idle;
	bool maximized;
	//void set_wm_border(bool p_enabled);
	void set_wm_fullscreen(bool p_enabled);

	typedef xrr_monitor_info *(*xrr_get_monitors_t)(Display *dpy, Window window, Bool get_active, int *nmonitors);
	typedef void (*xrr_free_monitors_t)(xrr_monitor_info *monitors);
	xrr_get_monitors_t xrr_get_monitors;
	xrr_free_monitors_t xrr_free_monitors;
	void *xrandr_handle;
	Bool xrandr_ext_ok;

protected:
	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;
	virtual VideoMode get_default_video_mode() const;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual void initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);
	virtual void finalize();

	virtual void set_main_loop(MainLoop *p_main_loop);

public:
	virtual String get_name();

	virtual void set_cursor_shape(CursorShape p_shape);

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	virtual void warp_mouse_pos(const Point2 &p_to);
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	virtual void set_icon(const Image &p_icon);

	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual String get_system_dir(SystemDir p_dir) const;

	virtual Error shell_open(String p_uri);

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual int get_screen_count() const;
	virtual int get_current_screen() const;
	virtual void set_current_screen(int p_screen);
	virtual Point2 get_screen_position(int p_screen = 0) const;
	virtual Size2 get_screen_size(int p_screen = 0) const;
	virtual int get_screen_dpi(int p_screen = 0) const;
	virtual Point2 get_window_position() const;
	virtual void set_window_position(const Point2 &p_position);
	virtual Size2 get_window_size() const;
	virtual void set_window_size(const Size2 p_size);
	virtual void set_window_fullscreen(bool p_enabled);
	virtual bool is_window_fullscreen() const;
	virtual void set_window_resizable(bool p_enabled);
	virtual bool is_window_resizable() const;
	virtual void set_window_minimized(bool p_enabled);
	virtual bool is_window_minimized() const;
	virtual void set_window_maximized(bool p_enabled);
	virtual bool is_window_maximized() const;
	virtual void request_attention();

	virtual void move_window_to_foreground();
	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;

	virtual void set_context(int p_context);

	virtual void set_use_vsync(bool p_enable);
	virtual bool is_vsync_enabled() const;

	virtual PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	void run();

	OS_X11();
};

#endif
