/*************************************************************************/
/*  os_nacl.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef OS_NACL_H
#define OS_NACL_H

#include "core/os/os.h"

#include "servers/visual_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/spatial_sound/spatial_sound_server_sw.h"
#include "servers/spatial_sound_2d/spatial_sound_2d_server_sw.h"
#include "servers/audio/audio_server_sw.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "audio_driver_nacl.h"
#include "os/input.h"


#include <ppapi/cpp/input_event.h>

class OSNacl : OS {

	uint64_t ticks_start;

protected:

	enum {
		MAX_EVENTS = 64,
	};

	MainLoop *main_loop;

	Rasterizer *rasterizer;
	VisualServer *visual_server;
	PhysicsServer *physics_server;
	SpatialSoundServerSW *spatial_sound_server;

	AudioServerSW *audio_server;
	SampleManagerMallocSW *sample_manager;
	SpatialSound2DServerSW *spatial_sound_2d_server;
	Physics2DServer *physics_2d_server;
	AudioDriverNacl* audio_driver;


	// functions used by main to initialize/deintialize the OS
	virtual int get_video_driver_count() const;
	virtual const char * get_video_driver_name(int p_driver) const;

	virtual VideoMode get_default_video_mode() const;

	virtual int get_audio_driver_count() const;
	virtual const char * get_audio_driver_name(int p_driver) const;

	void vprint(const char* p_format, va_list p_list, bool p_stderr);

	virtual void initialize_core();
	virtual void initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver);

	virtual void set_main_loop( MainLoop * p_main_loop );
	virtual void delete_main_loop();

	virtual void finalize();
	virtual void finalize_core();

	int mouse_last_x, mouse_last_y;

	InputEvent event_queue[MAX_EVENTS];
	int event_count;
	void queue_event(const InputEvent& p_event);

	int event_id;
	uint32_t mouse_mask;

	uint32_t last_scancode;

	bool minimized;

	VideoMode video_mode;

    InputDefault *input;

public:

	void add_package(String p_name, Vector<uint8_t> p_data);

	void handle_event(const pp::InputEvent& p_event);

	virtual void alert(const String& p_alert,const String& p_title);
	virtual String get_stdin_string(bool p_block);

	virtual void set_mouse_show(bool p_show);
	virtual void set_mouse_grab(bool p_grab);
	virtual bool is_mouse_grab_enabled() const;
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String& p_title);

	virtual void set_video_mode(const VideoMode& p_video_mode,int p_screen);
	virtual VideoMode get_video_mode(int p_screen) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen) const;

    virtual Error execute(const String& p_path, const List<String>& p_arguments,bool p_blocking,ProcessID *r_child_id=NULL,String* r_pipe=NULL,int *r_exitcode=NULL);
	virtual Error kill(const ProcessID& p_pid);

	virtual bool has_environment(const String& p_var) const;
	virtual String get_environment(const String& p_var) const;

	virtual void set_cursor_shape(CursorShape p_shape);

	virtual String get_name();

	virtual MainLoop *get_main_loop() const;

	virtual Date get_date(bool utc) const;
	virtual Time get_time(bool utc) const;

	virtual void delay_usec(uint32_t p_usec) const;
	virtual uint64_t get_ticks_usec() const;

    virtual String get_resource_dir() const;

	virtual bool can_draw() const;

	bool iterate();

	OSNacl();
	~OSNacl();
};

#endif // OS_NACL_H
