/*************************************************************************/
/*  display_server.cpp                                                   */
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

#include "display_server.h"

#include "core/input/input.h"
#include "scene/resources/texture.h"
#include "servers/display_server_headless.h"

DisplayServer *DisplayServer::singleton = nullptr;

bool DisplayServer::hidpi_allowed = false;

DisplayServer::DisplayServerCreate DisplayServer::server_create_functions[DisplayServer::MAX_SERVERS] = {
	{ "headless", &DisplayServerHeadless::create_func, &DisplayServerHeadless::get_rendering_drivers_func }
};

int DisplayServer::server_create_count = 1;

void DisplayServer::global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_add_separator(const String &p_menu_root) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback) {
	WARN_PRINT("Global menus not supported by this display server.");
}

bool DisplayServer::global_menu_is_item_checked(const String &p_menu_root, int p_idx) const {
	WARN_PRINT("Global menus not supported by this display server.");
	return false;
}

bool DisplayServer::global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const {
	WARN_PRINT("Global menus not supported by this display server.");
	return false;
}

Callable DisplayServer::global_menu_get_item_callback(const String &p_menu_root, int p_idx) {
	WARN_PRINT("Global menus not supported by this display server.");
	return Callable();
}

Variant DisplayServer::global_menu_get_item_tag(const String &p_menu_root, int p_idx) {
	WARN_PRINT("Global menus not supported by this display server.");
	return Variant();
}

String DisplayServer::global_menu_get_item_text(const String &p_menu_root, int p_idx) {
	WARN_PRINT("Global menus not supported by this display server.");
	return String();
}

String DisplayServer::global_menu_get_item_submenu(const String &p_menu_root, int p_idx) {
	WARN_PRINT("Global menus not supported by this display server.");
	return String();
}

void DisplayServer::global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu) {
	WARN_PRINT("Global menus not supported by this display server.");
}

int DisplayServer::global_menu_get_item_count(const String &p_menu_root) const {
	WARN_PRINT("Global menus not supported by this display server.");
	return 0;
}

void DisplayServer::global_menu_remove_item(const String &p_menu_root, int p_idx) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::global_menu_clear(const String &p_menu_root) {
	WARN_PRINT("Global menus not supported by this display server.");
}

void DisplayServer::mouse_set_mode(MouseMode p_mode) {
	WARN_PRINT("Mouse is not supported by this display server.");
}

DisplayServer::MouseMode DisplayServer::mouse_get_mode() const {
	return MOUSE_MODE_VISIBLE;
}

void DisplayServer::mouse_warp_to_position(const Point2i &p_to) {
	WARN_PRINT("Mouse warping is not supported by this display server.");
}

Point2i DisplayServer::mouse_get_absolute_position() const {
	ERR_FAIL_V_MSG(Point2i(), "Mouse is not supported by this display server.");
}

Point2i DisplayServer::mouse_get_position() const {
	ERR_FAIL_V_MSG(Point2i(), "Mouse is not supported by this display server.");
}

MouseButton DisplayServer::mouse_get_button_state() const {
	ERR_FAIL_V_MSG(MOUSE_BUTTON_NONE, "Mouse is not supported by this display server.");
}

void DisplayServer::clipboard_set(const String &p_text) {
	WARN_PRINT("Clipboard is not supported by this display server.");
}

String DisplayServer::clipboard_get() const {
	ERR_FAIL_V_MSG(String(), "Clipboard is not supported by this display server.");
}

void DisplayServer::clipboard_set_primary(const String &p_text) {
	WARN_PRINT("Primary clipboard is not supported by this display server.");
}

String DisplayServer::clipboard_get_primary() const {
	ERR_FAIL_V_MSG(String(), "Primary clipboard is not supported by this display server.");
}

void DisplayServer::screen_set_orientation(ScreenOrientation p_orientation, int p_screen) {
	WARN_PRINT("Orientation not supported by this display server.");
}

DisplayServer::ScreenOrientation DisplayServer::screen_get_orientation(int p_screen) const {
	return SCREEN_LANDSCAPE;
}

float DisplayServer::screen_get_scale(int p_screen) const {
	return 1.0f;
};

bool DisplayServer::screen_is_touchscreen(int p_screen) const {
	//return false;
	return Input::get_singleton() && Input::get_singleton()->is_emulating_touch_from_mouse();
}

void DisplayServer::screen_set_keep_on(bool p_enable) {
	WARN_PRINT("Keeping screen on not supported by this display server.");
}

bool DisplayServer::screen_is_kept_on() const {
	return false;
}

DisplayServer::WindowID DisplayServer::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	ERR_FAIL_V_MSG(INVALID_WINDOW_ID, "Sub-windows not supported by this display server.");
}

void DisplayServer::show_window(WindowID p_id) {
	ERR_FAIL_MSG("Sub-windows not supported by this display server.");
}

void DisplayServer::delete_sub_window(WindowID p_id) {
	ERR_FAIL_MSG("Sub-windows not supported by this display server.");
}

void DisplayServer::window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window) {
	ERR_FAIL_MSG("Mouse passthrough not supported by this display server.");
}

void DisplayServer::window_set_ime_active(const bool p_active, WindowID p_window) {
	WARN_PRINT("IME not supported by this display server.");
}

void DisplayServer::window_set_ime_position(const Point2i &p_pos, WindowID p_window) {
	WARN_PRINT("IME not supported by this display server.");
}

Point2i DisplayServer::ime_get_selection() const {
	ERR_FAIL_V_MSG(Point2i(), "IME or NOTIFICATION_WM_IME_UPDATE not supported by this display server.");
}

String DisplayServer::ime_get_text() const {
	ERR_FAIL_V_MSG(String(), "IME or NOTIFICATION_WM_IME_UPDATEnot supported by this display server.");
}

void DisplayServer::console_set_visible(bool p_enabled) {
	WARN_PRINT("Console window not supported by this display server.");
}

bool DisplayServer::is_console_visible() const {
	return false;
}

void DisplayServer::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, bool p_multiline, int p_max_length, int p_cursor_start, int p_cursor_end) {
	WARN_PRINT("Virtual keyboard not supported by this display server.");
}

void DisplayServer::virtual_keyboard_hide() {
	WARN_PRINT("Virtual keyboard not supported by this display server.");
}

// returns height of the currently shown keyboard (0 if keyboard is hidden)
int DisplayServer::virtual_keyboard_get_height() const {
	ERR_FAIL_V_MSG(0, "Virtual keyboard not supported by this display server.");
}

void DisplayServer::cursor_set_shape(CursorShape p_shape) {
	WARN_PRINT("Cursor shape not supported by this display server.");
}

DisplayServer::CursorShape DisplayServer::cursor_get_shape() const {
	return CURSOR_ARROW;
}

void DisplayServer::cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	WARN_PRINT("Custom cursor shape not supported by this display server.");
}

bool DisplayServer::get_swap_cancel_ok() {
	return false;
}

void DisplayServer::enable_for_stealing_focus(OS::ProcessID pid) {
}

Error DisplayServer::dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback) {
	WARN_PRINT("Native dialogs not supported by this display server.");
	return OK;
}

Error DisplayServer::dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback) {
	WARN_PRINT("Native dialogs not supported by this display server.");
	return OK;
}

int DisplayServer::keyboard_get_layout_count() const {
	return 0;
}

int DisplayServer::keyboard_get_current_layout() const {
	return -1;
}

void DisplayServer::keyboard_set_current_layout(int p_index) {
}

String DisplayServer::keyboard_get_layout_language(int p_index) const {
	return "";
}

String DisplayServer::keyboard_get_layout_name(int p_index) const {
	return "Not supported";
}

Key DisplayServer::keyboard_get_keycode_from_physical(Key p_keycode) const {
	ERR_FAIL_V_MSG(p_keycode, "Not supported by this display server.");
}

void DisplayServer::force_process_and_drop_events() {
}

void DisplayServer::release_rendering_thread() {
	WARN_PRINT("Rendering thread not supported by this display server.");
}

void DisplayServer::make_rendering_thread() {
	WARN_PRINT("Rendering thread not supported by this display server.");
}

void DisplayServer::swap_buffers() {
	WARN_PRINT("Swap buffers not supported by this display server.");
}

void DisplayServer::set_native_icon(const String &p_filename) {
	WARN_PRINT("Native icon not supported by this display server.");
}

void DisplayServer::set_icon(const Ref<Image> &p_icon) {
	WARN_PRINT("Icon not supported by this display server.");
}

void DisplayServer::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	WARN_PRINT("Changing the VSync mode is not supported by this display server.");
}

DisplayServer::VSyncMode DisplayServer::window_get_vsync_mode(WindowID p_window) const {
	WARN_PRINT("Changing the VSync mode is not supported by this display server.");
	return VSyncMode::VSYNC_ENABLED;
}

void DisplayServer::set_context(Context p_context) {
}

void DisplayServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_feature", "feature"), &DisplayServer::has_feature);
	ClassDB::bind_method(D_METHOD("get_name"), &DisplayServer::get_name);

	ClassDB::bind_method(D_METHOD("global_menu_add_item", "menu_root", "label", "callback", "tag"), &DisplayServer::global_menu_add_item, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("global_menu_add_check_item", "menu_root", "label", "callback", "tag"), &DisplayServer::global_menu_add_check_item, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("global_menu_add_submenu_item", "menu_root", "label", "submenu"), &DisplayServer::global_menu_add_submenu_item);
	ClassDB::bind_method(D_METHOD("global_menu_add_separator", "menu_root"), &DisplayServer::global_menu_add_separator);

	ClassDB::bind_method(D_METHOD("global_menu_is_item_checked", "menu_root", "idx"), &DisplayServer::global_menu_is_item_checked);
	ClassDB::bind_method(D_METHOD("global_menu_is_item_checkable", "menu_root", "idx"), &DisplayServer::global_menu_is_item_checkable);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_callback", "menu_root", "idx"), &DisplayServer::global_menu_get_item_callback);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_tag", "menu_root", "idx"), &DisplayServer::global_menu_get_item_tag);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_text", "menu_root", "idx"), &DisplayServer::global_menu_get_item_text);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_submenu", "menu_root", "idx"), &DisplayServer::global_menu_get_item_submenu);

	ClassDB::bind_method(D_METHOD("global_menu_set_item_checked", "menu_root", "idx", "checked"), &DisplayServer::global_menu_set_item_checked);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_checkable", "menu_root", "idx", "checkable"), &DisplayServer::global_menu_set_item_checkable);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_callback", "menu_root", "idx", "callback"), &DisplayServer::global_menu_set_item_callback);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_tag", "menu_root", "idx", "tag"), &DisplayServer::global_menu_set_item_tag);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_text", "menu_root", "idx", "text"), &DisplayServer::global_menu_set_item_text);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_submenu", "menu_root", "idx", "submenu"), &DisplayServer::global_menu_set_item_submenu);

	ClassDB::bind_method(D_METHOD("global_menu_remove_item", "menu_root", "idx"), &DisplayServer::global_menu_remove_item);
	ClassDB::bind_method(D_METHOD("global_menu_clear", "menu_root"), &DisplayServer::global_menu_clear);

	ClassDB::bind_method(D_METHOD("mouse_set_mode", "mouse_mode"), &DisplayServer::mouse_set_mode);
	ClassDB::bind_method(D_METHOD("mouse_get_mode"), &DisplayServer::mouse_get_mode);

	ClassDB::bind_method(D_METHOD("mouse_warp_to_position", "position"), &DisplayServer::mouse_warp_to_position);
	ClassDB::bind_method(D_METHOD("mouse_get_position"), &DisplayServer::mouse_get_position);
	ClassDB::bind_method(D_METHOD("mouse_get_absolute_position"), &DisplayServer::mouse_get_absolute_position);
	ClassDB::bind_method(D_METHOD("mouse_get_button_state"), &DisplayServer::mouse_get_button_state);

	ClassDB::bind_method(D_METHOD("clipboard_set", "clipboard"), &DisplayServer::clipboard_set);
	ClassDB::bind_method(D_METHOD("clipboard_get"), &DisplayServer::clipboard_get);
	ClassDB::bind_method(D_METHOD("clipboard_set_primary", "clipboard_primary"), &DisplayServer::clipboard_set_primary);
	ClassDB::bind_method(D_METHOD("clipboard_get_primary"), &DisplayServer::clipboard_get_primary);

	ClassDB::bind_method(D_METHOD("get_screen_count"), &DisplayServer::get_screen_count);
	ClassDB::bind_method(D_METHOD("screen_get_position", "screen"), &DisplayServer::screen_get_position, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_size", "screen"), &DisplayServer::screen_get_size, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_usable_rect", "screen"), &DisplayServer::screen_get_usable_rect, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_dpi", "screen"), &DisplayServer::screen_get_dpi, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_scale", "screen"), &DisplayServer::screen_get_scale, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_is_touchscreen", "screen"), &DisplayServer::screen_is_touchscreen, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_max_scale"), &DisplayServer::screen_get_max_scale);

	ClassDB::bind_method(D_METHOD("screen_set_orientation", "orientation", "screen"), &DisplayServer::screen_set_orientation, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_orientation", "screen"), &DisplayServer::screen_get_orientation, DEFVAL(SCREEN_OF_MAIN_WINDOW));

	ClassDB::bind_method(D_METHOD("screen_set_keep_on", "enable"), &DisplayServer::screen_set_keep_on);
	ClassDB::bind_method(D_METHOD("screen_is_kept_on"), &DisplayServer::screen_is_kept_on);

	ClassDB::bind_method(D_METHOD("get_window_list"), &DisplayServer::get_window_list);
	ClassDB::bind_method(D_METHOD("get_window_at_screen_position", "position"), &DisplayServer::get_window_at_screen_position);

	ClassDB::bind_method(D_METHOD("create_sub_window", "mode", "vsync_mode", "flags", "rect"), &DisplayServer::create_sub_window, DEFVAL(Rect2i()));
	ClassDB::bind_method(D_METHOD("delete_sub_window", "window_id"), &DisplayServer::delete_sub_window);

	ClassDB::bind_method(D_METHOD("window_set_title", "title", "window_id"), &DisplayServer::window_set_title, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_mouse_passthrough", "region", "window_id"), &DisplayServer::window_set_mouse_passthrough, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_current_screen", "window_id"), &DisplayServer::window_get_current_screen, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_current_screen", "screen", "window_id"), &DisplayServer::window_set_current_screen, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_position", "window_id"), &DisplayServer::window_get_position, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_position", "position", "window_id"), &DisplayServer::window_set_position, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_size", "window_id"), &DisplayServer::window_get_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_size", "size", "window_id"), &DisplayServer::window_set_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_rect_changed_callback", "callback", "window_id"), &DisplayServer::window_set_rect_changed_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_window_event_callback", "callback", "window_id"), &DisplayServer::window_set_window_event_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_input_event_callback", "callback", "window_id"), &DisplayServer::window_set_input_event_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_input_text_callback", "callback", "window_id"), &DisplayServer::window_set_input_text_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_drop_files_callback", "callback", "window_id"), &DisplayServer::window_set_drop_files_callback, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_attach_instance_id", "instance_id", "window_id"), &DisplayServer::window_attach_instance_id, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_attached_instance_id", "window_id"), &DisplayServer::window_get_attached_instance_id, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_max_size", "window_id"), &DisplayServer::window_get_max_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_max_size", "max_size", "window_id"), &DisplayServer::window_set_max_size, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_min_size", "window_id"), &DisplayServer::window_get_min_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_min_size", "min_size", "window_id"), &DisplayServer::window_set_min_size, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_real_size", "window_id"), &DisplayServer::window_get_real_size, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_mode", "window_id"), &DisplayServer::window_get_mode, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_mode", "mode", "window_id"), &DisplayServer::window_set_mode, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_set_flag", "flag", "enabled", "window_id"), &DisplayServer::window_set_flag, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_flag", "flag", "window_id"), &DisplayServer::window_get_flag, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_request_attention", "window_id"), &DisplayServer::window_request_attention, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_move_to_foreground", "window_id"), &DisplayServer::window_move_to_foreground, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_can_draw", "window_id"), &DisplayServer::window_can_draw, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_set_transient", "window_id", "parent_window_id"), &DisplayServer::window_set_transient);

	ClassDB::bind_method(D_METHOD("window_set_ime_active", "active", "window_id"), &DisplayServer::window_set_ime_active, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_ime_position", "position", "window_id"), &DisplayServer::window_set_ime_position, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_set_vsync_mode", "vsync_mode", "window_id"), &DisplayServer::window_set_vsync_mode, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_vsync_mode", "window_id"), &DisplayServer::window_get_vsync_mode, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("ime_get_selection"), &DisplayServer::ime_get_selection);
	ClassDB::bind_method(D_METHOD("ime_get_text"), &DisplayServer::ime_get_text);

	ClassDB::bind_method(D_METHOD("console_set_visible", "console_visible"), &DisplayServer::console_set_visible);
	ClassDB::bind_method(D_METHOD("is_console_visible"), &DisplayServer::is_console_visible);

	ClassDB::bind_method(D_METHOD("virtual_keyboard_show", "existing_text", "position", "multiline", "max_length", "cursor_start", "cursor_end"), &DisplayServer::virtual_keyboard_show, DEFVAL(Rect2i()), DEFVAL(false), DEFVAL(-1), DEFVAL(-1), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("virtual_keyboard_hide"), &DisplayServer::virtual_keyboard_hide);

	ClassDB::bind_method(D_METHOD("virtual_keyboard_get_height"), &DisplayServer::virtual_keyboard_get_height);

	ClassDB::bind_method(D_METHOD("cursor_set_shape", "shape"), &DisplayServer::cursor_set_shape);
	ClassDB::bind_method(D_METHOD("cursor_get_shape"), &DisplayServer::cursor_get_shape);
	ClassDB::bind_method(D_METHOD("cursor_set_custom_image", "cursor", "shape", "hotspot"), &DisplayServer::cursor_set_custom_image, DEFVAL(CURSOR_ARROW), DEFVAL(Vector2()));

	ClassDB::bind_method(D_METHOD("get_swap_cancel_ok"), &DisplayServer::get_swap_cancel_ok);

	ClassDB::bind_method(D_METHOD("enable_for_stealing_focus", "process_id"), &DisplayServer::enable_for_stealing_focus);

	ClassDB::bind_method(D_METHOD("dialog_show", "title", "description", "buttons", "callback"), &DisplayServer::dialog_show);
	ClassDB::bind_method(D_METHOD("dialog_input_text", "title", "description", "existing_text", "callback"), &DisplayServer::dialog_input_text);

	ClassDB::bind_method(D_METHOD("keyboard_get_layout_count"), &DisplayServer::keyboard_get_layout_count);
	ClassDB::bind_method(D_METHOD("keyboard_get_current_layout"), &DisplayServer::keyboard_get_current_layout);
	ClassDB::bind_method(D_METHOD("keyboard_set_current_layout", "index"), &DisplayServer::keyboard_set_current_layout);
	ClassDB::bind_method(D_METHOD("keyboard_get_layout_language", "index"), &DisplayServer::keyboard_get_layout_language);
	ClassDB::bind_method(D_METHOD("keyboard_get_layout_name", "index"), &DisplayServer::keyboard_get_layout_name);
	ClassDB::bind_method(D_METHOD("keyboard_get_keycode_from_physical", "keycode"), &DisplayServer::keyboard_get_keycode_from_physical);

	ClassDB::bind_method(D_METHOD("process_events"), &DisplayServer::process_events);
	ClassDB::bind_method(D_METHOD("force_process_and_drop_events"), &DisplayServer::force_process_and_drop_events);

	ClassDB::bind_method(D_METHOD("set_native_icon", "filename"), &DisplayServer::set_native_icon);
	ClassDB::bind_method(D_METHOD("set_icon", "image"), &DisplayServer::set_icon);

	ClassDB::bind_method(D_METHOD("tablet_get_driver_count"), &DisplayServer::tablet_get_driver_count);
	ClassDB::bind_method(D_METHOD("tablet_get_driver_name", "idx"), &DisplayServer::tablet_get_driver_name);
	ClassDB::bind_method(D_METHOD("tablet_get_current_driver"), &DisplayServer::tablet_get_current_driver);
	ClassDB::bind_method(D_METHOD("tablet_set_current_driver", "name"), &DisplayServer::tablet_set_current_driver);

	BIND_ENUM_CONSTANT(FEATURE_GLOBAL_MENU);
	BIND_ENUM_CONSTANT(FEATURE_SUBWINDOWS);
	BIND_ENUM_CONSTANT(FEATURE_TOUCHSCREEN);
	BIND_ENUM_CONSTANT(FEATURE_MOUSE);
	BIND_ENUM_CONSTANT(FEATURE_MOUSE_WARP);
	BIND_ENUM_CONSTANT(FEATURE_CLIPBOARD);
	BIND_ENUM_CONSTANT(FEATURE_VIRTUAL_KEYBOARD);
	BIND_ENUM_CONSTANT(FEATURE_CURSOR_SHAPE);
	BIND_ENUM_CONSTANT(FEATURE_CUSTOM_CURSOR_SHAPE);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_DIALOG);
	BIND_ENUM_CONSTANT(FEATURE_CONSOLE_WINDOW);
	BIND_ENUM_CONSTANT(FEATURE_IME);
	BIND_ENUM_CONSTANT(FEATURE_WINDOW_TRANSPARENCY);
	BIND_ENUM_CONSTANT(FEATURE_HIDPI);
	BIND_ENUM_CONSTANT(FEATURE_ICON);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_ICON);
	BIND_ENUM_CONSTANT(FEATURE_ORIENTATION);
	BIND_ENUM_CONSTANT(FEATURE_SWAP_BUFFERS);
	BIND_ENUM_CONSTANT(FEATURE_CLIPBOARD_PRIMARY);

	BIND_ENUM_CONSTANT(MOUSE_MODE_VISIBLE);
	BIND_ENUM_CONSTANT(MOUSE_MODE_HIDDEN);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CAPTURED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED_HIDDEN);

	BIND_CONSTANT(SCREEN_OF_MAIN_WINDOW);
	BIND_CONSTANT(MAIN_WINDOW_ID);
	BIND_CONSTANT(INVALID_WINDOW_ID);

	BIND_ENUM_CONSTANT(SCREEN_LANDSCAPE);
	BIND_ENUM_CONSTANT(SCREEN_PORTRAIT);
	BIND_ENUM_CONSTANT(SCREEN_REVERSE_LANDSCAPE);
	BIND_ENUM_CONSTANT(SCREEN_REVERSE_PORTRAIT);
	BIND_ENUM_CONSTANT(SCREEN_SENSOR_LANDSCAPE);
	BIND_ENUM_CONSTANT(SCREEN_SENSOR_PORTRAIT);
	BIND_ENUM_CONSTANT(SCREEN_SENSOR);

	BIND_ENUM_CONSTANT(CURSOR_ARROW);
	BIND_ENUM_CONSTANT(CURSOR_IBEAM);
	BIND_ENUM_CONSTANT(CURSOR_POINTING_HAND);
	BIND_ENUM_CONSTANT(CURSOR_CROSS);
	BIND_ENUM_CONSTANT(CURSOR_WAIT);
	BIND_ENUM_CONSTANT(CURSOR_BUSY);
	BIND_ENUM_CONSTANT(CURSOR_DRAG);
	BIND_ENUM_CONSTANT(CURSOR_CAN_DROP);
	BIND_ENUM_CONSTANT(CURSOR_FORBIDDEN);
	BIND_ENUM_CONSTANT(CURSOR_VSIZE);
	BIND_ENUM_CONSTANT(CURSOR_HSIZE);
	BIND_ENUM_CONSTANT(CURSOR_BDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_FDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_MOVE);
	BIND_ENUM_CONSTANT(CURSOR_VSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HELP);
	BIND_ENUM_CONSTANT(CURSOR_MAX);

	BIND_ENUM_CONSTANT(WINDOW_MODE_WINDOWED);
	BIND_ENUM_CONSTANT(WINDOW_MODE_MINIMIZED);
	BIND_ENUM_CONSTANT(WINDOW_MODE_MAXIMIZED);
	BIND_ENUM_CONSTANT(WINDOW_MODE_FULLSCREEN);

	BIND_ENUM_CONSTANT(WINDOW_FLAG_RESIZE_DISABLED);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_BORDERLESS);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_ALWAYS_ON_TOP);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_TRANSPARENT);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_NO_FOCUS);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_MAX);

	BIND_ENUM_CONSTANT(WINDOW_EVENT_MOUSE_ENTER);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_MOUSE_EXIT);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_FOCUS_IN);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_FOCUS_OUT);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_CLOSE_REQUEST);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_GO_BACK_REQUEST);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_DPI_CHANGE);

	BIND_ENUM_CONSTANT(VSYNC_DISABLED);
	BIND_ENUM_CONSTANT(VSYNC_ENABLED);
	BIND_ENUM_CONSTANT(VSYNC_ADAPTIVE);
	BIND_ENUM_CONSTANT(VSYNC_MAILBOX);
}

void DisplayServer::register_create_function(const char *p_name, CreateFunction p_function, GetRenderingDriversFunction p_get_drivers) {
	ERR_FAIL_COND(server_create_count == MAX_SERVERS);
	// Headless display server is always last
	server_create_functions[server_create_count] = server_create_functions[server_create_count - 1];
	server_create_functions[server_create_count - 1].name = p_name;
	server_create_functions[server_create_count - 1].create_function = p_function;
	server_create_functions[server_create_count - 1].get_rendering_drivers_function = p_get_drivers;
	server_create_count++;
}

int DisplayServer::get_create_function_count() {
	return server_create_count;
}

const char *DisplayServer::get_create_function_name(int p_index) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	return server_create_functions[p_index].name;
}

Vector<String> DisplayServer::get_create_function_rendering_drivers(int p_index) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, Vector<String>());
	return server_create_functions[p_index].get_rendering_drivers_function();
}

DisplayServer *DisplayServer::create(int p_index, const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	return server_create_functions[p_index].create_function(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error);
}

void DisplayServer::_input_set_mouse_mode(Input::MouseMode p_mode) {
	singleton->mouse_set_mode(MouseMode(p_mode));
}

Input::MouseMode DisplayServer::_input_get_mouse_mode() {
	return Input::MouseMode(singleton->mouse_get_mode());
}

void DisplayServer::_input_warp(const Vector2 &p_to_pos) {
	singleton->mouse_warp_to_position(p_to_pos);
}

Input::CursorShape DisplayServer::_input_get_current_cursor_shape() {
	return (Input::CursorShape)singleton->cursor_get_shape();
}

void DisplayServer::_input_set_custom_mouse_cursor_func(const RES &p_image, Input::CursorShape p_shape, const Vector2 &p_hostspot) {
	singleton->cursor_set_custom_image(p_image, (CursorShape)p_shape, p_hostspot);
}

DisplayServer::DisplayServer() {
	singleton = this;
	Input::set_mouse_mode_func = _input_set_mouse_mode;
	Input::get_mouse_mode_func = _input_get_mouse_mode;
	Input::warp_mouse_func = _input_warp;
	Input::get_current_cursor_shape_func = _input_get_current_cursor_shape;
	Input::set_custom_mouse_cursor_func = _input_set_custom_mouse_cursor_func;
}

DisplayServer::~DisplayServer() {
	singleton = nullptr;
}
