#ifndef OS_HAIKU_H
#define OS_HAIKU_H

#include "os/os.h"
#include "drivers/unix/os_unix.h"


class OS_Haiku : public OS_Unix {
private:
	virtual void delete_main_loop();

protected:
	virtual int get_video_driver_count() const;
	virtual const char* get_video_driver_name(int p_driver) const;	
	virtual VideoMode get_default_video_mode() const;

	virtual void initialize(const VideoMode& p_desired, int p_video_driver, int p_audio_driver);
	virtual void finalize();

	virtual void set_main_loop(MainLoop* p_main_loop);

public:
	OS_Haiku();
	void run();

	virtual String get_name();

	virtual MainLoop* get_main_loop() const;
	virtual bool can_draw() const;

	virtual Point2 get_mouse_pos() const;
	virtual int get_mouse_button_state() const;
	virtual void set_cursor_shape(CursorShape p_shape);

	virtual void set_window_title(const String& p_title);
	virtual Size2 get_window_size() const;

	virtual void set_video_mode(const VideoMode& p_video_mode, int p_screen=0);
	virtual VideoMode get_video_mode(int p_screen=0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen=0) const;
};

#endif
