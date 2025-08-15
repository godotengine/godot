/**************************************************************************/
/*  display_server_openharmony.h                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "servers/display_server.h"

#include <inputmethod/inputmethod_controller_capi.h>

class RenderingContextDriver;
class RenderingDevice;

class DisplayServerOpenHarmony : public DisplayServer {
	String rendering_driver;
	RenderingContextDriver *rendering_context = nullptr;
	RenderingDevice *rendering_device = nullptr;
	ObjectID window_attached_instance_id;

	bool ime_active = false;
	VirtualKeyboardType keyboard_type = KEYBOARD_TYPE_DEFAULT;
	InputMethod_KeyboardStatus keyboard_status = IME_KEYBOARD_STATUS_NONE;
	InputMethod_TextEditorProxy *text_editor_proxy = nullptr;
	InputMethod_AttachOptions *attach_options = nullptr;
	InputMethod_InputMethodProxy *input_method_proxy = nullptr;

	Callable window_event_callback;
	Callable window_resize_callback;
	Callable input_event_callback;
	Callable input_text_callback;

	void _window_callback(const Callable &p_callable, const Variant &p_arg, bool p_deferred = false) const;
	static void _dispatch_input_events(const Ref<InputEvent> &p_event);

	static void _get_text_config(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_TextConfig *p_text_config);
	static void _insert_text(InputMethod_TextEditorProxy *p_text_editor_proxy, const char16_t *p_text, size_t length);
	static void _delete_forward(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t length);
	static void _delete_backward(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t length);
	static void _send_keyboard_status(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_KeyboardStatus keyboard_status);
	static void _send_enter_key(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_EnterKeyType enter_key_type);
	static void _move_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_Direction direction);
	static void _handle_set_selection(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t start, int32_t end);
	static void _handle_extend_action(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_ExtendAction action);
	static void _get_left_text_of_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t number, char16_t *p_text, size_t *p_length);
	static void _get_right_text_of_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t number, char16_t *p_text, size_t *p_length);
	static int32_t _get_text_index_at_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy);
	static int32_t _receive_private_command(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_PrivateCommand *p_command[], size_t length);
	static int32_t _set_preview_text(InputMethod_TextEditorProxy *p_text_editor_proxy, const char16_t *p_text, size_t length, int32_t start, int32_t end);
	static void _finish_text_preview(InputMethod_TextEditorProxy *p_text_editor_proxy);

	static void _input_text_key(Key p_key, char32_t p_char, Key p_unshifted, Key p_physical, int p_modifier, bool p_pressed, KeyLocation p_location);

public:
	static DisplayServerOpenHarmony *get_singleton();
	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error);
	static Vector<String> get_rendering_drivers_func();
	static void register_openharmony_driver();

	DisplayServerOpenHarmony(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error);
	~DisplayServerOpenHarmony();

	void send_input_event(const Ref<InputEvent> &p_event) const;
	void resize_window(uint32_t p_width, uint32_t p_height);
	void send_window_event(DisplayServer::WindowEvent p_event) const;

	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;
	virtual int get_screen_count() const override;
	virtual int get_primary_screen() const override;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_scale(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_refresh_rate(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual bool is_touchscreen_available() const override;

	virtual Point2i mouse_get_position() const override { return Point2i(); }
	virtual void mouse_set_mode(MouseMode p_mode) override {}
	virtual MouseMode mouse_get_mode() const override { return MOUSE_MODE_VISIBLE; }
	virtual void mouse_set_mode_override(MouseMode p_mode) override {}
	virtual MouseMode mouse_get_mode_override() const override { return MOUSE_MODE_VISIBLE; }
	virtual void mouse_set_mode_override_enabled(bool p_override_enabled) override {}
	virtual bool mouse_is_mode_override_enabled() const override { return false; }
	virtual void warp_mouse(const Point2i &p_position) override {}
	virtual BitField<MouseButtonMask> mouse_get_button_state() const override { return BitField<MouseButtonMask>(0); }

	virtual void screen_set_orientation(ScreenOrientation p_orientation, int p_screen = SCREEN_OF_MAIN_WINDOW) override;
	virtual ScreenOrientation screen_get_orientation(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;

	virtual void screen_set_keep_on(bool p_enable) override;
	virtual bool screen_is_kept_on() const override;

	virtual void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), VirtualKeyboardType p_type = KEYBOARD_TYPE_DEFAULT, int p_max_length = -1, int p_cursor_start = -1, int p_cursor_end = -1) override;
	virtual void virtual_keyboard_hide() override;
	virtual int virtual_keyboard_get_height() const override;

	virtual void window_set_ime_active(const bool p_active, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_ime_position(const Point2i &p_pos, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual Vector<WindowID> get_window_list() const override;
	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const override;
	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual Point2i window_get_position_with_decorations(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_transient(WindowID p_window, WindowID p_parent) override;
	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual Size2i window_get_size_with_decorations(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_vsync_mode(VSyncMode p_vsync_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual VSyncMode window_get_vsync_mode(WindowID p_window) const override;
	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_is_focused(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual bool can_any_window_draw() const override;
	virtual void process_events() override;
};
