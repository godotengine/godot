#ifndef OS_FLASH_H
#define OS_FLASH_H

#include "os/input.h"
#include "drivers/unix/os_unix.h"
#include "os/input.h"
#include "servers/visual_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/spatial_sound/spatial_sound_server_sw.h"
#include "servers/spatial_sound_2d/spatial_sound_2d_server_sw.h"
#include "servers/audio/audio_server_sw.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "main/input_default.h"

class OSFlash : public OS_Unix {

	VideoMode default_videomode;
	MainLoop * main_loop;
	InputDefault *input;
	Rasterizer *rasterizer;
	VisualServer *visual_server;
	AudioDriverSW* audio_driver;
	AudioServerSW *audio_server;
	SampleManagerMallocSW *sample_manager;
	SpatialSoundServerSW *spatial_sound_server;
	SpatialSound2DServerSW *spatial_sound_2d_server;
	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;

public:

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

	virtual void set_mouse_show(bool p_show);
	virtual void set_mouse_grab(bool p_grab);
	virtual bool is_mouse_grab_enabled() const;
	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String& p_title);

	//virtual void set_clipboard(const String& p_text);
	//virtual String get_clipboard() const;


	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String& p_existing_text,const Rect2& p_screen_rect);
	virtual void hide_virtual_keyboard();

	virtual void set_video_mode(const VideoMode& p_video_mode,int p_screen=0);
	virtual VideoMode get_video_mode(int p_screen=0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen=0) const;

	virtual String get_name();
	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	virtual void set_cursor_shape(CursorShape p_shape);

	virtual bool has_touchscreen_ui_hint() const;

	virtual void yield();

	virtual Error shell_open(String p_uri);

	bool iterate();
};

#endif
