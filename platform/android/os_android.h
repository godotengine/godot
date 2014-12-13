/*************************************************************************/
/*  os_android.h                                                         */
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
#ifndef OS_ANDROID_H
#define OS_ANDROID_H

#include "os/input.h"
#include "drivers/unix/os_unix.h"
#include "os/main_loop.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/spatial_sound/spatial_sound_server_sw.h"
#include "servers/spatial_sound_2d/spatial_sound_2d_server_sw.h"
#include "servers/audio/audio_server_sw.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "servers/visual/rasterizer.h"


#ifdef ANDROID_NATIVE_ACTIVITY

#include <android/sensor.h>
#include <android/log.h>
#include <android_native_app_glue.h>

#else


#endif

#include "audio_driver_jandroid.h"
#include "audio_driver_opensl.h"

typedef void (*GFXInitFunc)(void *ud,bool gl2);
typedef int (*OpenURIFunc)(const String&);
typedef String (*GetDataDirFunc)();
typedef String (*GetLocaleFunc)();
typedef String (*GetModelFunc)();
typedef String (*GetUniqueIDFunc)();
typedef void (*ShowVirtualKeyboardFunc)(const String&);
typedef void (*HideVirtualKeyboardFunc)();
typedef void (*SetScreenOrientationFunc)(int);
typedef String (*GetSystemDirFunc)(int);

typedef void (*VideoPlayFunc)(const String&);
typedef bool (*VideoIsPlayingFunc)();
typedef void (*VideoPauseFunc)();
typedef void (*VideoStopFunc)();

class OS_Android : public OS_Unix {
public:

	struct TouchPos {
		int id;
		Point2 pos;
	};

private:

	Vector<TouchPos> touch;

	Point2 last_mouse;
	unsigned int last_id;
	GFXInitFunc gfx_init_func;
	void*gfx_init_ud;

	bool use_gl2;
	bool use_reload_hooks;
	bool use_apk_expansion;

	Rasterizer *rasterizer;
	VisualServer *visual_server;
	AudioServerSW *audio_server;
	SampleManagerMallocSW *sample_manager;
	SpatialSoundServerSW *spatial_sound_server;
	SpatialSound2DServerSW *spatial_sound_2d_server;
	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;

#if 0
	AudioDriverAndroid audio_driver_android;
#else
	AudioDriverOpenSL audio_driver_android;
#endif

	const char* gl_extensions;

	InputDefault *input;
	VideoMode default_videomode;
	MainLoop * main_loop;

	OpenURIFunc open_uri_func;
	GetDataDirFunc get_data_dir_func;
	GetLocaleFunc get_locale_func;
	GetModelFunc get_model_func;
	ShowVirtualKeyboardFunc show_virtual_keyboard_func;
	HideVirtualKeyboardFunc hide_virtual_keyboard_func;
	SetScreenOrientationFunc set_screen_orientation_func;
	GetUniqueIDFunc get_unique_id_func;
	GetSystemDirFunc get_system_dir_func;

	VideoPlayFunc video_play_func;
	VideoIsPlayingFunc video_is_playing_func;
	VideoPauseFunc video_pause_func;
	VideoStopFunc video_stop_func;

public:

	// functions used by main to initialize/deintialize the OS
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


	typedef int64_t ProcessID;

	static OS* get_singleton();

	virtual void vprint(const char* p_format, va_list p_list, bool p_stderr=false);
	virtual void print(const char *p_format, ... );
	virtual void alert(const String& p_alert);


	virtual void set_mouse_show(bool p_show);
	virtual void set_mouse_grab(bool p_grab);
	virtual bool is_mouse_grab_enabled() const;
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String& p_title);

	//virtual void set_clipboard(const String& p_text);
	//virtual String get_clipboard() const;

	virtual void set_video_mode(const VideoMode& p_video_mode,int p_screen=0);
	virtual VideoMode get_video_mode(int p_screen=0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen=0) const;

	virtual String get_name();
	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	virtual void set_cursor_shape(CursorShape p_shape);

	void main_loop_begin();
	bool main_loop_iterate();
	void main_loop_request_quit();
	void main_loop_end();
	void main_loop_focusout();
	void main_loop_focusin();

	virtual bool has_touchscreen_ui_hint() const;

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String& p_existing_text,const Rect2& p_screen_rect=Rect2());
	virtual void hide_virtual_keyboard();

	void set_opengl_extensions(const char* p_gl_extensions);
	void set_display_size(Size2 p_size);

	void reload_gfx();

	void set_need_reload_hooks(bool p_needs_them);
	virtual void set_screen_orientation(ScreenOrientation p_orientation);

	virtual Error shell_open(String p_uri);
	virtual String get_data_dir() const;
	virtual String get_resource_dir() const;
	virtual String get_locale() const;
	virtual String get_model_name() const;

	virtual String get_unique_ID() const;

	virtual String get_system_dir(SystemDir p_dir) const;


	void process_accelerometer(const Vector3& p_accelerometer);
	void process_touch(int p_what,int p_pointer, const Vector<TouchPos>& p_points);
	void process_event(InputEvent p_event);
	void init_video_mode(int p_video_width,int p_video_height);

	virtual Error native_video_play(String p_path, float p_volume);
	virtual bool native_video_is_playing();
	virtual void native_video_pause();
	virtual void native_video_stop();

	OS_Android(GFXInitFunc p_gfx_init_func,void*p_gfx_init_ud, OpenURIFunc p_open_uri_func, GetDataDirFunc p_get_data_dir_func,GetLocaleFunc p_get_locale_func,GetModelFunc p_get_model_func, ShowVirtualKeyboardFunc p_show_vk, HideVirtualKeyboardFunc p_hide_vk,  SetScreenOrientationFunc p_screen_orient,GetUniqueIDFunc p_get_unique_id,GetSystemDirFunc p_get_sdir_func, VideoPlayFunc p_video_play_func, VideoIsPlayingFunc p_video_is_playing_func, VideoPauseFunc p_video_pause_func, VideoStopFunc p_video_stop_func,bool p_use_apk_expansion);
	~OS_Android();

};

#endif
