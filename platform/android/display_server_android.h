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
	// Touch pointer info constants.
	// Note: These values must match the one in org.godotengine.godot.input.GodotInputHandler.java
	enum {
		// Must match org.godotengine.godot.input.GodotInputHandler#POINTER_INFO_ID_OFFSET
		TOUCH_POINTER_INFO_ID_OFFSET = 0,
		// Must match org.godotengine.godot.input.GodotInputHandler#POINTER_INFO_TOOL_TYPE_OFFSET
		TOUCH_POINTER_INFO_TOOL_TYPE_OFFSET = 1,
		// Must match org.godotengine.godot.input.GodotInputHandler#POINTER_INFO_SIZE
		TOUCH_POINTER_INFO_SIZE = 2
	};

	// Android MotionEvent's ACTION_* constants.
	enum {
		ACTION_BUTTON_PRESS = 11,
		ACTION_BUTTON_RELEASE = 12,
		ACTION_CANCEL = 3,
		ACTION_DOWN = 0,
		ACTION_MOVE = 2,
		ACTION_POINTER_DOWN = 5,
		ACTION_POINTER_UP = 6,
		ACTION_UP = 1
	};

	// Android MotionEvent's TOOL_TYPE_* constants.
	enum {
		TOOL_TYPE_UNKNOWN = 0,
		TOOL_TYPE_FINGER = 1,
		TOOL_TYPE_STYLUS = 2,
		TOOL_TYPE_MOUSE = 3,
		TOOL_TYPE_ERASER = 4
	};

	// Android MotionEvent's BUTTON_* constants.
	enum {
		BUTTON_PRIMARY = 1,
		BUTTON_SECONDARY = 2,
		BUTTON_TERTIARY = 4,
		BUTTON_BACK = 8,
		BUTTON_FORWARD = 16,
		BUTTON_STYLUS_PRIMARY = 32,
		BUTTON_STYLUS_SECONDARY = 64
	};

	struct TouchPos {
		int id;
		Point2 pos;
		int tool_type;
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
	void send_touch_event(TouchPos touch_pos, int android_motion_event_action_button);
	void release_touch_event(TouchPos touch_pos, int android_motion_event_action_button, bool update_last_mouse_buttons_mask = false);
	void release_touches(int android_motion_event_action_button, bool update_last_mouse_buttons_mask = false);
	int get_mouse_button_index(int android_motion_event_button_state);
	inline bool is_mouse_pointer(TouchPos touch_pos) const;
	inline bool is_mouse_pointer(int tool_type) const;

	Vector<TouchPos> touch;
	// Needed to calculate the relative position on hover events
	Point2 hover_prev_pos = Point2();
	// This is specific to the Godot engine and should not be confused with Android's MotionEvent#getButtonState()
	int last_mouse_buttons_mask = 0;
	Point2 last_mouse_position = Point2();

	String rendering_driver;

	bool keep_screen_on;

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

	virtual Point2i mouse_get_position() const;
	virtual int mouse_get_button_state() const;

	virtual void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), int p_max_length = -1);
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
	void process_touch(int motion_event_action, int motion_event_action_button, int p_pointer, const Vector<TouchPos> &p_points);
	void process_hover(int tool_type, int p_type, Point2 p_pos);
	void process_double_tap(int tool_type, int android_motion_event_button_state, Point2 p_pos);
	void process_scroll(int tool_type, Point2 start, Point2 end, Vector2 scroll_delta);
	void process_joy_event(JoypadEvent p_event);
	void process_key_event(int p_keycode, int p_scancode, int p_unicode_char, bool p_pressed);

	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	static Vector<String> get_rendering_drivers_func();
	static void register_android_driver();

	DisplayServerAndroid(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerAndroid();
};

#endif // DISPLAY_SERVER_ANDROID_H
