/*************************************************************************/
/*  display_server_javascript.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DISPLAY_SERVER_JAVASCRIPT_H
#define DISPLAY_SERVER_JAVASCRIPT_H

#include "servers/display_server.h"

#include <emscripten.h>
#include <emscripten/html5.h>

class DisplayServerJavaScript : public DisplayServer {
private:
	WindowMode window_mode = WINDOW_MODE_WINDOWED;
	ObjectID window_attached_instance_id = {};

	Callable window_event_callback;
	Callable input_event_callback;
	Callable input_text_callback;
	Callable drop_files_callback;

	String clipboard;
	Ref<InputEventKey> deferred_key_event;
	Point2 touches[32];

	char canvas_id[256] = { 0 };
	bool cursor_inside_canvas = true;
	CursorShape cursor_shape = CURSOR_ARROW;
	Point2i last_click_pos = Point2(-100, -100); // TODO check this again.
	double last_click_ms = 0;
	int last_click_button_index = -1;

	bool swap_cancel_ok = false;

	// utilities
	static Point2 compute_position_in_canvas(int p_x, int p_y);
	static void focus_canvas();
	static bool is_canvas_focused();
	template <typename T>
	static void dom2godot_mod(T *emscripten_event_ptr, Ref<InputEventWithModifiers> godot_event);
	static Ref<InputEventKey> setup_key_event(const EmscriptenKeyboardEvent *emscripten_event);
	static const char *godot2dom_cursor(DisplayServer::CursorShape p_shape);

	// events
	static EM_BOOL fullscreen_change_callback(int p_event_type, const EmscriptenFullscreenChangeEvent *p_event, void *p_user_data);

	static EM_BOOL keydown_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data);
	static EM_BOOL keypress_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data);
	static EM_BOOL keyup_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data);

	static void vk_input_text_callback(const char *p_text, int p_cursor);

	static EM_BOOL mousemove_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data);
	static EM_BOOL mouse_button_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data);

	static EM_BOOL wheel_callback(int p_event_type, const EmscriptenWheelEvent *p_event, void *p_user_data);

	static EM_BOOL touch_press_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data);
	static EM_BOOL touchmove_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data);

	static void gamepad_callback(int p_index, int p_connected, const char *p_id, const char *p_guid);
	void process_joypads();

	static Vector<String> get_rendering_drivers_func();
	static DisplayServer *create_func(const String &p_rendering_driver, DisplayServer::WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);

	static void _dispatch_input_event(const Ref<InputEvent> &p_event);

	static void request_quit_callback();
	static void update_clipboard_callback(const char *p_text);
	static void send_window_event_callback(int p_notification);
	static void drop_files_js_callback(char **p_filev, int p_filec);

protected:
	int get_current_video_driver() const;

public:
	// Override return type to make writing static callbacks less tedious.
	static DisplayServerJavaScript *get_singleton();

	// utilities
	bool check_size_force_redraw();

	// from DisplayServer
	void alert(const String &p_alert, const String &p_title = "ALERT!") override;
	bool has_feature(Feature p_feature) const override;
	String get_name() const override;

	// cursor
	void cursor_set_shape(CursorShape p_shape) override;
	CursorShape cursor_get_shape() const override;
	void cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape = CURSOR_ARROW, const Vector2 &p_hotspot = Vector2()) override;

	// mouse
	void mouse_set_mode(MouseMode p_mode) override;
	MouseMode mouse_get_mode() const override;

	// touch
	bool screen_is_touchscreen(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	// clipboard
	void clipboard_set(const String &p_text) override;
	String clipboard_get() const override;

	// screen
	int get_screen_count() const override;
	Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	float screen_get_scale(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_input_length = -1, int p_cursor_start = -1, int p_cursor_end = -1) override;
	void virtual_keyboard_hide() override;

	// windows
	Vector<DisplayServer::WindowID> get_window_list() const override;
	WindowID get_window_at_screen_position(const Point2i &p_position) const override;

	void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID) override;
	ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const override;

	void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID) override;

	int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const override;
	void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID) override;

	Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const override;
	void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID) override;

	void window_set_transient(WindowID p_window, WindowID p_parent) override;

	void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const override;

	bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const override;

	void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID) override;
	bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const override;

	void window_request_attention(WindowID p_window = MAIN_WINDOW_ID) override;
	void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID) override;

	bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const override;

	bool can_any_window_draw() const override;

	// events
	void process_events() override;

	// icon
	void set_icon(const Ref<Image> &p_icon) override;

	// others
	bool get_swap_cancel_ok() override;
	void swap_buffers() override;

	static void register_javascript_driver();
	DisplayServerJavaScript(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerJavaScript();
};

#endif // DISPLAY_SERVER_JAVASCRIPT_H
