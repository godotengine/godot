/*************************************************************************/
/*  os_server.h                                                          */
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
#ifndef OS_SERVER_H
#define OS_SERVER_H

#include "../x11/power_x11.h"
#include "drivers/rtaudio/audio_driver_rtaudio.h"
#include "drivers/unix/os_unix.h"
#include "main/input_default.h"
#include "servers/audio/audio_driver_dummy.h"
#include "servers/audio_server.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "servers/physics_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"

//bitch
#undef CursorShape
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class OS_Server : public OS_Unix {

	//Rasterizer *rasterizer;
	VisualServer *visual_server;
	VideoMode current_videomode;
	List<String> args;
	MainLoop *main_loop;

	AudioDriverDummy driver_dummy;
	bool grab;

	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;

	virtual void delete_main_loop();
	IP_Unix *ip_unix;

	bool force_quit;

	InputDefault *input;

	PowerX11 *power_manager;

protected:
	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;
	virtual VideoMode get_default_video_mode() const;

	virtual void initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);
	virtual void finalize();

	virtual void set_main_loop(MainLoop *p_main_loop);

public:
	virtual String get_name();

	virtual void set_cursor_shape(CursorShape p_shape);

	virtual void set_mouse_show(bool p_show);
	virtual void set_mouse_grab(bool p_grab);
	virtual bool is_mouse_grab_enabled() const;
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual Size2 get_window_size() const;

	virtual void move_window_to_foreground();

	void run();

	virtual PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	OS_Server();
};

#endif
