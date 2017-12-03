/*************************************************************************/
/*  os_iphone.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifdef IPHONE_ENABLED

#ifndef OS_IPHONE_H
#define OS_IPHONE_H

#include "drivers/coreaudio/audio_driver_coreaudio.h"
#include "drivers/unix/os_unix.h"
#include "os/input.h"

#include "game_center.h"
#include "icloud.h"
#include "in_app_store.h"
#include "main/input_default.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"

class OSIPhone : public OS_Unix {

public:
	enum Orientations {
		PortraitDown,
		PortraitUp,
		LandscapeLeft,
		LandscapeRight,
	};

private:
	enum {
		MAX_MOUSE_COUNT = 8,
		MAX_EVENTS = 64,
	};

	static HashMap<String, void *> dynamic_symbol_lookup_table;
	friend void register_dynamic_symbol(char *name, void *address);

	uint8_t supported_orientations;

	VisualServer *visual_server;

	AudioDriverCoreAudio audio_driver;

#ifdef GAME_CENTER_ENABLED
	GameCenter *game_center;
#endif
#ifdef STOREKIT_ENABLED
	InAppStore *store_kit;
#endif
#ifdef ICLOUD_ENABLED
	ICloud *icloud;
#endif

	MainLoop *main_loop;

	VideoMode video_mode;

	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;

	virtual void initialize_core();
	virtual void initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual MainLoop *get_main_loop() const;

	virtual void delete_main_loop();

	virtual void finalize();

	struct MouseList {

		bool pressed[MAX_MOUSE_COUNT];
		MouseList() {
			for (int i = 0; i < MAX_MOUSE_COUNT; i++)
				pressed[i] = false;
		};
	};

	MouseList mouse_list;

	Vector3 last_accel;

	Ref<InputEvent> event_queue[MAX_EVENTS];
	int event_count;
	void queue_event(const Ref<InputEvent> &p_event);

	String data_dir;
	String unique_ID;
	String locale_code;

	InputDefault *input;

	int virtual_keyboard_height;

public:
	bool iterate();

	uint8_t get_orientations() const;

	void mouse_button(int p_idx, int p_x, int p_y, bool p_pressed, bool p_doubleclick, bool p_use_as_mouse);
	void mouse_move(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y, bool p_use_as_mouse);
	void touches_cancelled();
	void key(uint32_t p_key, bool p_pressed);
	void set_virtual_keyboard_height(int p_height);

	int set_base_framebuffer(int p_fb);

	void update_gravity(float p_x, float p_y, float p_z);
	void update_accelerometer(float p_x, float p_y, float p_z);
	void update_magnetometer(float p_x, float p_y, float p_z);
	void update_gyroscope(float p_x, float p_y, float p_z);

	int get_unused_joy_id();
	void joy_connection_changed(int p_idx, bool p_connected, String p_name);
	void joy_button(int p_device, int p_button, bool p_pressed);
	void joy_axis(int p_device, int p_axis, const InputDefault::JoyAxis &p_value);

	static OSIPhone *get_singleton();

	virtual void set_mouse_show(bool p_show);
	virtual void set_mouse_grab(bool p_grab);
	virtual bool is_mouse_grab_enabled() const;
	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);
	virtual Error close_dynamic_library(void *p_library_handle);
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual void set_keep_screen_on(bool p_enabled);

	virtual bool can_draw() const;

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2());
	virtual void hide_virtual_keyboard();
	virtual int get_virtual_keyboard_height() const;

	virtual void set_cursor_shape(CursorShape p_shape);

	virtual Size2 get_window_size() const;

	virtual bool has_touchscreen_ui_hint() const;

	void set_data_dir(String p_dir);

	virtual String get_name();

	Error shell_open(String p_uri);

	String get_user_data_dir() const;

	void set_locale(String p_locale);
	String get_locale() const;

	void set_unique_id(String p_ID);
	String get_unique_id() const;

	virtual Error native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track);
	virtual bool native_video_is_playing() const;
	virtual void native_video_pause();
	virtual void native_video_unpause();
	virtual void native_video_focus_out();
	virtual void native_video_stop();

	virtual bool _check_internal_feature_support(const String &p_feature);
	OSIPhone(int width, int height, String p_data_dir);
	~OSIPhone();
};

#endif // OS_IPHONE_H

#endif
