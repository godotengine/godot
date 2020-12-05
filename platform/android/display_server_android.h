/*************************************************************************/
/*  display_server_android.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DISPLAY_SERVER_ANDROID_H
#define DISPLAY_SERVER_ANDROID_H

#include "servers/display_server.h"

#if defined(VULKAN_ENABLED)
class VulkanContextAndroid;
class RenderingDeviceVulkan;
#endif

class DisplayServerAndroid : public DisplayServer {
public:
	struct TouchPos {
		int id;
		Point2 pos;
	};

	enum {
		JOY_EVENT_BUTTON = 0,
		JOY_EVENT_AXIS = 1,
		JOY_EVENT_HAT = 2
	};

	struct JoypadEvent {
		int device;
		int type;
		int index;
		bool pressed;
		float value;
		int hat;
	};

private:
	String rendering_driver;

	bool alt_mem = false;
	bool shift_mem = false;
	bool control_mem = false;
	bool meta_mem = false;

	int buttons_state;

	bool keep_screen_on;

	Vector<TouchPos> touch;
	Point2 hover_prev_pos; // needed to calculate the relative position on hover events
	Point2 scroll_prev_pos; // needed to calculate the relative position on scroll events

#if defined(VULKAN_ENABLED)
	VulkanContextAndroid *context_vulkan;
	RenderingDeviceVulkan *rendering_device_vulkan;
#endif

	ObjectID window_attached_instance_id;

	Callable window_event_callback;
	Callable input_event_callback;
	Callable input_text_callback;

	void _window_callback(const Callable &p_callable, const Variant &p_arg) const;

	static void _dispatch_input_events(const Ref<InputEvent> &p_event);

	void _set_key_modifier_state(Ref<InputEventWithModifiers> ev);

	static int _button_index_from_mask(int button_mask);

	static int _android_button_mask_to_godot_button_mask(int android_button_mask);

	void _wheel_button_click(int event_buttons_mask, const Ref<InputEventMouseButton> &ev, int wheel_button, float factor);

public:
	static DisplayServerAndroid *get_singleton();

	virtual bool has_feature(Feature p_feature) const;
	virtual String get_name() const;

	virtual void clipboard_set(const String &p_text);
	virtual String clipboard_get() const;

	virtual void screen_set_keep_on(bool p_enable);
	virtual bool screen_is_kept_on() const;

	virtual void screen_set_orientation(ScreenOrientation p_orientation, int p_screen = SCREEN_OF_MAIN_WINDOW);
	virtual ScreenOrientation screen_get_orientation(int p_screen = SCREEN_OF_MAIN_WINDOW) const;

	virtual int get_screen_count() const;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual bool screen_is_touchscreen(int p_screen = SCREEN_OF_MAIN_WINDOW) const;

	virtual void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void virtual_keyboard_hide();
	virtual int virtual_keyboard_get_height() const;

	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);

	void send_window_event(WindowEvent p_event) const;
	void send_input_event(const Ref<InputEvent> &p_event) const;
	void send_input_text(const String &p_text) const;

	virtual Vector<WindowID> get_window_list() const;
	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const;
	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID);
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID);
	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID);
	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_transient(WindowID p_window, WindowID p_parent);
	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID);
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID);
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID);
	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual bool can_any_window_draw() const;

	virtual void alert(const String &p_alert, const String &p_title);

	virtual void process_events();

	void process_accelerometer(const Vector3 &p_accelerometer);
	void process_gravity(const Vector3 &p_gravity);
	void process_magnetometer(const Vector3 &p_magnetometer);
	void process_gyroscope(const Vector3 &p_gyroscope);
	void process_touch(int p_event, int p_pointer, const Vector<TouchPos> &p_points);
	void process_hover(int p_type, Point2 p_pos);
	void process_mouse_event(int event_action, int event_android_buttons_mask, Point2 event_pos, float event_vertical_factor = 0, float event_horizontal_factor = 0);
	void process_double_tap(int event_android_button_mask, Point2 p_pos);
	void process_scroll(Point2 p_pos);
	void process_joy_event(JoypadEvent p_event);
	void process_key_event(int p_keycode, int p_scancode, int p_unicode_char, bool p_pressed);

	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	static Vector<String> get_rendering_drivers_func();
	static void register_android_driver();

	void reset_window();

	virtual Point2i mouse_get_position() const;
	virtual int mouse_get_button_state() const;

	DisplayServerAndroid(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerAndroid();
};

#endif // DISPLAY_SERVER_ANDROID_H
