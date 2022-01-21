/*************************************************************************/
/*  display_server_javascript.h                                          */
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

#ifndef DISPLAY_SERVER_JAVASCRIPT_H
#define DISPLAY_SERVER_JAVASCRIPT_H

#include "servers/display_server.h"

#include <emscripten.h>
#include <emscripten/html5.h>

class DisplayServerJavaScript : public DisplayServer {
private:
	struct JSTouchEvent {
		uint32_t identifier[32] = { 0 };
		double coords[64] = { 0 };
	};
	JSTouchEvent touch_event;

	struct JSKeyEvent {
		char code[32] = { 0 };
		char key[32] = { 0 };
		uint8_t modifiers[4] = { 0 };
	};
	JSKeyEvent key_event;

#ifdef GLES3_ENABLED
	EMSCRIPTEN_WEBGL_CONTEXT_HANDLE webgl_ctx = 0;
#endif

	WindowMode window_mode = WINDOW_MODE_WINDOWED;
	ObjectID window_attached_instance_id = {};

	Callable window_event_callback;
	Callable input_event_callback;
	Callable input_text_callback;
	Callable drop_files_callback;

	String clipboard;
	Point2 touches[32];

	char canvas_id[256] = { 0 };
	bool cursor_inside_canvas = true;
	CursorShape cursor_shape = CURSOR_ARROW;
	Point2i last_click_pos = Point2(-100, -100); // TODO check this again.
	uint64_t last_click_ms = 0;
	MouseButton last_click_button_index = MouseButton::NONE;

	bool swap_cancel_ok = false;

	// utilities
	static void dom2godot_mod(Ref<InputEventWithModifiers> ev, int p_mod);
	static const char *godot2dom_cursor(DisplayServer::CursorShape p_shape);

	// events
	static void fullscreen_change_callback(int p_fullscreen);
	static int mouse_button_callback(int p_pressed, int p_button, double p_x, double p_y, int p_modifiers);
	static void mouse_move_callback(double p_x, double p_y, double p_rel_x, double p_rel_y, int p_modifiers);
	static int mouse_wheel_callback(double p_delta_x, double p_delta_y);
	static void touch_callback(int p_type, int p_count);
	static void key_callback(int p_pressed, int p_repeat, int p_modifiers);
	static void vk_input_text_callback(const char *p_text, int p_cursor);
	static void gamepad_callback(int p_index, int p_connected, const char *p_id, const char *p_guid);
	void process_joypads();

	static Vector<String> get_rendering_drivers_func();
	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_window_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);

	static void _dispatch_input_event(const Ref<InputEvent> &p_event);

	static void request_quit_callback();
	static void window_blur_callback();
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
	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;

	// cursor
	virtual void cursor_set_shape(CursorShape p_shape) override;
	virtual CursorShape cursor_get_shape() const override;
	virtual void cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape = CURSOR_ARROW, const Vector2 &p_hotspot = Vector2()) override;

	// mouse
	virtual void mouse_set_mode(MouseMode p_mode) override;
	virtual MouseMode mouse_get_mode() const override;
	virtual Point2i mouse_get_position() const override;

	// touch
	virtual bool screen_is_touchscreen(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	// clipboard
	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;

	// screen
	virtual int get_screen_count() const override;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_scale(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	virtual void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), bool p_multiline = false, int p_max_input_length = -1, int p_cursor_start = -1, int p_cursor_end = -1) override;
	virtual void virtual_keyboard_hide() override;

	// windows
	virtual Vector<DisplayServer::WindowID> get_window_list() const override;
	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const override;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_transient(WindowID p_window, WindowID p_parent) override;

	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID) override;

	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool can_any_window_draw() const override;

	// events
	virtual void process_events() override;

	// icon
	virtual void set_icon(const Ref<Image> &p_icon) override;

	// others
	virtual bool get_swap_cancel_ok() override;
	virtual void swap_buffers() override;

	static void register_javascript_driver();
	DisplayServerJavaScript(const String &p_rendering_driver, WindowMode p_window_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Size2i &p_resolution, Error &r_error);
	~DisplayServerJavaScript();
};

#endif // DISPLAY_SERVER_JAVASCRIPT_H
