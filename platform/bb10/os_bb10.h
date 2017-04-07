/*************************************************************************/
/*  os_bb10.h                                                            */
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
#ifndef OS_BB10_H
#define OS_BB10_H

#include "audio_driver_bb10.h"
#include "drivers/unix/os_unix.h"
#include "main/input_default.h"
#include "os/input.h"
#include "os/main_loop.h"
#include "payment_service.h"
#include "power_bb10.h"
#include "servers/audio_server.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "servers/visual/rasterizer.h"

#include <bps/event.h>
#include <screen/screen.h>
#include <stdint.h>
#include <sys/platform.h>

class OSBB10 : public OS_Unix {

	screen_context_t screen_cxt;
	float fullscreen_mixer_volume;
	float fullscreen_stream_volume;

	Rasterizer *rasterizer;
	VisualServer *visual_server;
	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;
	AudioDriverBB10 *audio_driver;
	PowerBB10 *power_manager;

#ifdef PAYMENT_SERVICE_ENABLED
	PaymentService *payment_service;
#endif

	VideoMode default_videomode;
	MainLoop *main_loop;

	void process_events();

	void _resize(bps_event_t *event);
	void handle_screen_event(bps_event_t *event);
	void handle_accelerometer();

	int last_touch_x[16];
	int last_touch_y[16];

	bool accel_supported;
	float pitch;
	float roll;

	bool minimized;
	bool fullscreen;
	bool flip_accelerometer;
	String data_dir;

	InputDefault *input;

public:
	// functions used by main to initialize/deintialize the OS
	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;

	virtual VideoMode get_default_video_mode() const;

	virtual String get_data_dir() const;

	virtual int get_audio_driver_count() const;
	virtual const char *get_audio_driver_name(int p_driver) const;

	virtual void initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();

	typedef int64_t ProcessID;

	static OS *get_singleton();

	virtual void set_mouse_show(bool p_show);
	virtual void set_mouse_grab(bool p_grab);
	virtual bool is_mouse_grab_enabled() const;
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	//virtual void set_clipboard(const String& p_text);
	//virtual String get_clipboard() const;

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect);
	virtual void hide_virtual_keyboard();

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual Size2 get_window_size() const;
	virtual String get_name();
	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	virtual void set_cursor_shape(CursorShape p_shape);

	virtual bool has_touchscreen_ui_hint() const;

	virtual Error shell_open(String p_uri);

	void run();

	virtual PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	OSBB10();
	~OSBB10();
};

#endif
