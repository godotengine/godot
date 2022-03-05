/*************************************************************************/
/*  os_iphone.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/os/input.h"
#include "drivers/coreaudio/audio_driver_coreaudio.h"
#include "drivers/unix/os_unix.h"
#include "joypad_iphone.h"

#include "ios.h"
#include "main/input_default.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"

class OSIPhone : public OS_Unix {
private:
	static HashMap<String, void *> dynamic_symbol_lookup_table;
	friend void register_dynamic_symbol(char *name, void *address);

	VisualServer *visual_server;

	AudioDriverCoreAudio audio_driver;

	iOS *ios;

	JoypadIPhone *joypad_iphone;

	MainLoop *main_loop;

	VideoMode video_mode;

	EAGLContext *offscreen_gl_context;

	virtual int get_video_driver_count() const;
	virtual const char *get_video_driver_name(int p_driver) const;

	virtual int get_current_video_driver() const;

	virtual void initialize_core();
	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual MainLoop *get_main_loop() const;

	virtual void delete_main_loop();

	virtual void finalize();

	void perform_event(const Ref<InputEvent> &p_event);

	void set_data_dir(String p_dir);

	String data_dir;
	String cache_dir;

	InputDefault *input;

	int virtual_keyboard_height = 0;

	int video_driver_index;

	bool is_focused = false;

public:
	static OSIPhone *get_singleton();

	OSIPhone(String p_data_dir, String p_cache_dir);
	~OSIPhone();

	bool iterate();

	void start();

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);
	virtual Error close_dynamic_library(void *p_library_handle);
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual String get_name() const;
	virtual String get_model_name() const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	Error shell_open(String p_uri);

	String get_user_data_dir() const;
	String get_cache_path() const;

	String get_locale() const;

	String get_unique_id() const;
	virtual String get_processor_name() const;

	virtual void vibrate_handheld(int p_duration_ms = 500);

	virtual bool _check_internal_feature_support(const String &p_feature);

	virtual int get_screen_dpi(int p_screen = -1) const;
	virtual float get_screen_refresh_rate(int p_screen = -1) const;

	void pencil_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_doubleclick);
	void touch_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_doubleclick);
	void pencil_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y, float p_force);
	void touch_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y);
	void touches_cancelled(int p_idx);
	void pencil_cancelled(int p_idx);
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
	void joy_axis(int p_device, int p_axis, float p_value);

	virtual void set_mouse_show(bool p_show);
	virtual void set_mouse_grab(bool p_grab);
	virtual bool is_mouse_grab_enabled() const;
	virtual Point2 get_mouse_position() const;
	virtual int get_mouse_button_state() const;

	virtual void set_window_title(const String &p_title);

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;

	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	void set_offscreen_gl_context(EAGLContext *p_context);
	virtual bool is_offscreen_gl_available() const;
	virtual void set_offscreen_gl_current(bool p_current);

	virtual void set_keep_screen_on(bool p_enabled);

	virtual bool can_draw() const;

	virtual bool has_virtual_keyboard() const;
	virtual void show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_input_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void hide_virtual_keyboard();
	virtual int get_virtual_keyboard_height() const;

	virtual Size2 get_window_size() const;
	virtual Rect2 get_window_safe_area() const;

	virtual bool has_touchscreen_ui_hint() const;

	virtual Error native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track);
	virtual bool native_video_is_playing() const;
	virtual void native_video_pause();
	virtual void native_video_unpause();
	virtual void native_video_focus_out();
	virtual void native_video_stop();

	void on_focus_out();
	void on_focus_in();
};

#endif // OS_IPHONE_H

#endif
