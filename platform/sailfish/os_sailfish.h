#ifndef OS_SAILFISH_H
#define OS_SAILFISH_H

#include "os/input.h"
#include "drivers/unix/os_unix.h"
#include "servers/visual_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/physics_server.h"
#include "servers/physics_2d/physics_2d_server_sw.h"
#include "drivers/pulseaudio/audio_driver_pulseaudio.h"
#include "servers/audio/audio_driver_dummy.h"
#include "main/input_default.h"

#include <audioresource.h>

#include <QGuiApplication>
#include <QOpenGLWindow>
#include <QOpenGLContext>

class OS_Sailfish : public OS_Unix, QObject {
public:
	QGuiApplication* application;
	QOpenGLWindow* window;
	bool is_audio_resource_acquired;

private:
	MainLoop* main_loop;
	VideoMode current_video_mode;
	Rasterizer* rasterizer;
	VisualServer* visual_server;
	PhysicsServer* physics_server;
	Physics2DServer* physics_2d_server;
	InputDefault* input;
	audioresource_t* audio_resource;

#if defined(OPENGL_ENABLED)
	QOpenGLContext* openglContext;
#endif

#ifdef PULSEAUDIO_ENABLED
	AudioDriverPulseAudio driver_pulseaudio;
#endif

	AudioDriverDummy driver_dummy;

	int last_button_state;
	Point2i last_mouse_position;
	bool last_mouse_position_valid;

	virtual void delete_main_loop();

	int get_mouse_button_state(Qt::MouseButtons p_buttons);
	void get_key_modifier_state(Qt::KeyboardModifiers p_modifiers, Ref<InputEventWithModifiers> state);

protected:

	virtual int get_video_driver_count() const;
	virtual const char* get_video_driver_name(int p_driver) const;
	virtual VideoMode get_default_video_mode() const;

	virtual int get_audio_driver_count() const;
	virtual const char* get_audio_driver_name(int p_driver) const;

	virtual void initialize(const VideoMode& p_desired, int p_video_driver, int p_audio_driver);
	virtual void finalize();

	virtual void set_main_loop(MainLoop* p_main_loop);

	bool eventFilter(QObject* obj, QEvent* event);

public:

	OS_Sailfish();
	virtual String get_name();

	virtual void set_cursor_shape(CursorShape p_shape);

	// void set_mouse_mode(MouseMode p_mode);
	// MouseMode get_mouse_mode() const;
	//
	// virtual void warp_mouse_pos(const Point2& p_to);
	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String& p_title);
	//
	// virtual void set_icon(const Image& p_icon);
	//
	virtual MainLoop* get_main_loop() const;

	virtual bool can_draw() const;

	// virtual void set_clipboard(const String& p_text);
	// virtual String get_clipboard() const;
	//
	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();
	//
	// virtual String get_system_dir(SystemDir p_dir) const;
	//
	// virtual Error shell_open(String p_uri);
	//
	virtual void set_video_mode(const VideoMode& p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen=0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode>* p_list,int p_screen=0) const;
	//
	virtual int get_screen_count() const;
	virtual int get_current_screen() const;
	// virtual void set_current_screen(int p_screen);
	// virtual Point2 get_screen_position(int p_screen=0) const;
	// virtual Size2 get_screen_size(int p_screen=0) const;
	// virtual int get_screen_dpi(int p_screen=0) const;
	// virtual Point2 get_window_position() const;
	// virtual void set_window_position(const Point2& p_position);
	virtual Size2 get_window_size() const;
	// virtual void set_window_size(const Size2 p_size);
	// virtual void set_window_fullscreen(bool p_enabled);
	// virtual bool is_window_fullscreen() const;
	// virtual void set_window_resizable(bool p_enabled);
	// virtual bool is_window_resizable() const;
	// virtual void set_window_minimized(bool p_enabled);
	// virtual bool is_window_minimized() const;
	// virtual void set_window_maximized(bool p_enabled);
	// virtual bool is_window_maximized() const;
	// virtual void request_attention();
	//
	// virtual void move_window_to_foreground();
	//
	// virtual bool is_joy_known(int p_device);
	// virtual String get_joy_guid(int p_device) const;
	//
	// virtual void set_context(int p_context);
	//
	// virtual void set_use_vsync(bool p_enable);
	// virtual bool is_vsync_enabled() const;

	void run();

	//
	void start_audio_driver();
	void stop_audio_driver();
};

#endif
