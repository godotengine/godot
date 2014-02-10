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
#include "audio_driver_android.h"

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


	Rasterizer *rasterizer;
	VisualServer *visual_server;
//	AudioDriverPSP audio_driver_psp;
	AudioServerSW *audio_server;
	SampleManagerMallocSW *sample_manager;
	SpatialSoundServerSW *spatial_sound_server;
	SpatialSound2DServerSW *spatial_sound_2d_server;
	PhysicsServer *physics_server;
	Physics2DServer *physics_2d_server;
	AudioDriverAndroid audio_driver_android;
	InputDefault *input;

	VideoMode default_videomode;
	MainLoop * main_loop;
public:


	void initialize_core();

	// functions used by main to initialize/deintialize the OS
	virtual int get_video_driver_count() const;
	virtual const char * get_video_driver_name(int p_driver) const;

	virtual VideoMode get_default_video_mode() const;

	virtual int get_audio_driver_count() const;
	virtual const char * get_audio_driver_name(int p_driver) const;

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

	virtual void set_screen_orientation(ScreenOrientation p_orientation);

	virtual void set_video_mode(const VideoMode& p_video_mode,int p_screen=0);
	virtual VideoMode get_video_mode(int p_screen=0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen=0) const;

	virtual String get_name();
	virtual MainLoop *get_main_loop() const;

	virtual bool can_draw() const;

	virtual void set_cursor_shape(CursorShape p_shape);

	void main_loop_begin();
	bool main_loop_iterate();
	void main_loop_end();

	void process_touch(int p_what,int p_pointer, const Vector<TouchPos>& p_points);
	OS_Android(int p_video_width,int p_video_height);
	~OS_Android();

};

#endif
