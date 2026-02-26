/**************************************************************************/
/*  display_server_android.cpp                                            */
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

#include "display_server_android.h"

#include "java_godot_io_wrapper.h"
#include "java_godot_wrapper.h"
#include "os_android.h"
#include "tts_android.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/input/input_event.h"
#include "servers/display/native_menu.h"

#if defined(RD_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"

#if defined(VULKAN_ENABLED)
#include "rendering_context_driver_vulkan_android.h"
#endif
#endif

#ifdef GLES3_ENABLED
#include "drivers/gles3/rasterizer_gles3.h"

#include <EGL/egl.h>
#endif

#if defined(RD_ENABLED)
static RenderingContextDriver *rendering_context_global = nullptr;
static bool rendering_context_global_checked = false;
#endif

DisplayServerAndroid *DisplayServerAndroid::get_singleton() {
	return static_cast<DisplayServerAndroid *>(DisplayServer::get_singleton());
}

bool DisplayServerAndroid::has_feature(DisplayServerEnums::Feature p_feature) const {
	switch (p_feature) {
#ifndef DISABLE_DEPRECATED
		case DisplayServerEnums::FEATURE_GLOBAL_MENU: {
			return (native_menu && native_menu->has_feature(NativeMenu::FEATURE_GLOBAL_MENU));
		} break;
#endif
		case DisplayServerEnums::FEATURE_CURSOR_SHAPE:
		//case DisplayServerEnums::FEATURE_CUSTOM_CURSOR_SHAPE:
		//case DisplayServerEnums::FEATURE_HIDPI:
		//case DisplayServerEnums::FEATURE_ICON:
		//case DisplayServerEnums::FEATURE_IME:
		case DisplayServerEnums::FEATURE_MOUSE:
		//case DisplayServerEnums::FEATURE_MOUSE_WARP:
		case DisplayServerEnums::FEATURE_NATIVE_DIALOG:
		case DisplayServerEnums::FEATURE_NATIVE_DIALOG_INPUT:
		case DisplayServerEnums::FEATURE_NATIVE_DIALOG_FILE:
		//case DisplayServerEnums::FEATURE_NATIVE_DIALOG_FILE_EXTRA:
		case DisplayServerEnums::FEATURE_NATIVE_DIALOG_FILE_MIME:
		//case DisplayServerEnums::FEATURE_NATIVE_ICON:
		case DisplayServerEnums::FEATURE_WINDOW_TRANSPARENCY:
		case DisplayServerEnums::FEATURE_CLIPBOARD:
		case DisplayServerEnums::FEATURE_KEEP_SCREEN_ON:
		case DisplayServerEnums::FEATURE_ORIENTATION:
		case DisplayServerEnums::FEATURE_TOUCHSCREEN:
		case DisplayServerEnums::FEATURE_VIRTUAL_KEYBOARD:
		case DisplayServerEnums::FEATURE_TEXT_TO_SPEECH:
			return true;
		default:
			return false;
	}
}

String DisplayServerAndroid::get_name() const {
	return "Android";
}

bool DisplayServerAndroid::tts_is_speaking() const {
	return TTS_Android::is_speaking();
}

bool DisplayServerAndroid::tts_is_paused() const {
	return TTS_Android::is_paused();
}

TypedArray<Dictionary> DisplayServerAndroid::tts_get_voices() const {
	return TTS_Android::get_voices();
}

void DisplayServerAndroid::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int64_t p_utterance_id, bool p_interrupt) {
	TTS_Android::speak(p_text, p_voice, p_volume, p_pitch, p_rate, p_utterance_id, p_interrupt);
}

void DisplayServerAndroid::tts_pause() {
	TTS_Android::pause();
}

void DisplayServerAndroid::tts_resume() {
	TTS_Android::resume();
}

void DisplayServerAndroid::tts_stop() {
	TTS_Android::stop();
}

bool DisplayServerAndroid::is_dark_mode_supported() const {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, false);

	return godot_java->is_dark_mode_supported();
}

bool DisplayServerAndroid::is_dark_mode() const {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, false);

	return godot_java->is_dark_mode();
}

void DisplayServerAndroid::set_system_theme_change_callback(const Callable &p_callable) {
	system_theme_changed = p_callable;
}

void DisplayServerAndroid::emit_system_theme_changed() {
	if (system_theme_changed.is_valid()) {
		system_theme_changed.call_deferred();
	}
}

void DisplayServerAndroid::set_hardware_keyboard_connection_change_callback(const Callable &p_callable) {
	hardware_keyboard_connection_changed = p_callable;
}

void DisplayServerAndroid::emit_hardware_keyboard_connection_changed(bool p_connected) {
	if (hardware_keyboard_connection_changed.is_valid()) {
		hardware_keyboard_connection_changed.call_deferred(p_connected);
	}
}

void DisplayServerAndroid::clipboard_set(const String &p_text) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL(godot_java);

	if (godot_java->has_set_clipboard()) {
		godot_java->set_clipboard(p_text);
	} else {
		DisplayServer::clipboard_set(p_text);
	}
}

String DisplayServerAndroid::clipboard_get() const {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, String());

	if (godot_java->has_get_clipboard()) {
		return godot_java->get_clipboard();
	} else {
		return DisplayServer::clipboard_get();
	}
}

bool DisplayServerAndroid::clipboard_has() const {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, false);

	if (godot_java->has_has_clipboard()) {
		return godot_java->has_clipboard();
	} else {
		return DisplayServer::clipboard_has();
	}
}

Error DisplayServerAndroid::dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, FAILED);
	dialog_callback = p_callback;
	return godot_java->show_dialog(p_title, p_description, p_buttons);
}

void DisplayServerAndroid::emit_dialog_callback(int p_button_index) {
	if (dialog_callback.is_valid()) {
		dialog_callback.call_deferred(p_button_index);
	}
}

Error DisplayServerAndroid::dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, FAILED);
	input_dialog_callback = p_callback;
	return godot_java->show_input_dialog(p_title, p_description, p_partial);
}

void DisplayServerAndroid::emit_input_dialog_callback(String p_text) {
	if (input_dialog_callback.is_valid()) {
		input_dialog_callback.call_deferred(p_text);
	}
}

Error DisplayServerAndroid::file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, DisplayServerEnums::FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback, DisplayServerEnums::WindowID p_window_id) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, FAILED);
	file_picker_callback = p_callback;
	return godot_java->show_file_picker(p_current_directory, p_filename, p_mode, p_filters);
}

void DisplayServerAndroid::emit_file_picker_callback(bool p_ok, const Vector<String> &p_selected_paths) {
	if (file_picker_callback.is_valid()) {
		file_picker_callback.call_deferred(p_ok, p_selected_paths, 0);
	}
}

Color DisplayServerAndroid::get_accent_color() const {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, Color(0, 0, 0, 0));
	return godot_java->get_accent_color();
}

Color DisplayServerAndroid::get_base_color() const {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL_V(godot_java, Color(0, 0, 0, 0));
	return godot_java->get_base_color();
}

TypedArray<Rect2> DisplayServerAndroid::get_display_cutouts() const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, Array());
	return godot_io_java->get_display_cutouts();
}

Rect2i DisplayServerAndroid::get_display_safe_area() const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, Rect2i());
	return godot_io_java->get_display_safe_area();
}

void DisplayServerAndroid::screen_set_keep_on(bool p_enable) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL(godot_java);

	godot_java->set_keep_screen_on(p_enable);
	keep_screen_on = p_enable;
}

bool DisplayServerAndroid::screen_is_kept_on() const {
	return keep_screen_on;
}

void DisplayServerAndroid::screen_set_orientation(DisplayServerEnums::ScreenOrientation p_orientation, int p_screen) {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX(p_screen, screen_count);

	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL(godot_io_java);

	godot_io_java->set_screen_orientation(p_orientation);
}

DisplayServerEnums::ScreenOrientation DisplayServerAndroid::screen_get_orientation(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, DisplayServerEnums::SCREEN_LANDSCAPE);

	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, DisplayServerEnums::SCREEN_LANDSCAPE);

	const int orientation = godot_io_java->get_screen_orientation();
	ERR_FAIL_INDEX_V_MSG(orientation, 7, DisplayServerEnums::SCREEN_LANDSCAPE, "Unrecognized screen orientation");
	return (DisplayServerEnums::ScreenOrientation)orientation;
}

int DisplayServerAndroid::get_display_rotation() const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, 0);

	return godot_io_java->get_display_rotation();
}

int DisplayServerAndroid::get_screen_count() const {
	return 1;
}

int DisplayServerAndroid::get_primary_screen() const {
	return 0;
}

Point2i DisplayServerAndroid::screen_get_position(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Point2i());

	return Point2i(0, 0);
}

Size2i DisplayServerAndroid::screen_get_size(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Size2i());

	return OS_Android::get_singleton()->get_display_size();
}

Rect2i DisplayServerAndroid::screen_get_usable_rect(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Rect2i());

	Size2i display_size = OS_Android::get_singleton()->get_display_size();
	return Rect2i(0, 0, display_size.width, display_size.height);
}

int DisplayServerAndroid::screen_get_dpi(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 160);

	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, 160);

	return godot_io_java->get_screen_dpi();
}

float DisplayServerAndroid::screen_get_scale(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 1.0f);

	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, 1.0f);

	float screen_scale = godot_io_java->get_scaled_density();

	// Update the scale to avoid cropping.
	Size2i screen_size = screen_get_size(p_screen);
	if (screen_size != Size2i()) {
		float width_scale = screen_size.width / (float)OS_Android::DEFAULT_WINDOW_WIDTH;
		float height_scale = screen_size.height / (float)OS_Android::DEFAULT_WINDOW_HEIGHT;
		screen_scale = MIN(screen_scale, MIN(width_scale, height_scale));
	}

	return screen_scale;
}

float DisplayServerAndroid::screen_get_refresh_rate(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, SCREEN_REFRESH_RATE_FALLBACK);

	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	if (!godot_io_java) {
		ERR_PRINT("An error occurred while trying to get the screen refresh rate.");
		return SCREEN_REFRESH_RATE_FALLBACK;
	}

	return godot_io_java->get_screen_refresh_rate(SCREEN_REFRESH_RATE_FALLBACK);
}

bool DisplayServerAndroid::is_touchscreen_available() const {
	return true;
}

void DisplayServerAndroid::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, DisplayServerEnums::VirtualKeyboardType p_type, int p_max_length, int p_cursor_start, int p_cursor_end) {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL(godot_io_java);

	if (godot_io_java->has_vk()) {
		godot_io_java->show_vk(p_existing_text, (int)p_type, p_max_length, p_cursor_start, p_cursor_end);
	} else {
		ERR_PRINT("Virtual keyboard not available");
	}
}

void DisplayServerAndroid::virtual_keyboard_hide() {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL(godot_io_java);

	if (godot_io_java->has_vk()) {
		godot_io_java->hide_vk();
	} else {
		ERR_PRINT("Virtual keyboard not available");
	}
}

int DisplayServerAndroid::virtual_keyboard_get_height() const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, 0);

	return godot_io_java->get_vk_height();
}

bool DisplayServerAndroid::has_hardware_keyboard() const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, false);

	return godot_io_java->has_hardware_keyboard();
}

void DisplayServerAndroid::window_set_window_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	window_event_callback = p_callable;
}

void DisplayServerAndroid::window_set_input_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerAndroid::window_set_input_text_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	input_text_callback = p_callable;
}

void DisplayServerAndroid::window_set_rect_changed_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	rect_changed_callback = p_callable;
}

void DisplayServerAndroid::window_set_drop_files_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

template <typename... Args>
void DisplayServerAndroid::_window_callback(const Callable &p_callable, bool p_deferred, const Args &...p_rest_args) const {
	if (p_callable.is_valid()) {
		if (p_deferred) {
			p_callable.call_deferred(p_rest_args...);
		} else {
			p_callable.call(p_rest_args...);
		}
	}
}

void DisplayServerAndroid::send_window_event(DisplayServerEnums::WindowEvent p_event, bool p_deferred) const {
	_window_callback(window_event_callback, p_deferred, int(p_event));
}

void DisplayServerAndroid::send_input_event(const Ref<InputEvent> &p_event) const {
	_window_callback(input_event_callback, false, p_event);
}

void DisplayServerAndroid::send_input_text(const String &p_text) const {
	_window_callback(input_text_callback, false, p_text, false);
}

void DisplayServerAndroid::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	DisplayServerAndroid::get_singleton()->send_input_event(p_event);
}

Vector<DisplayServerEnums::WindowID> DisplayServerAndroid::get_window_list() const {
	Vector<DisplayServerEnums::WindowID> ret;
	ret.push_back(DisplayServerEnums::MAIN_WINDOW_ID);
	return ret;
}

DisplayServerEnums::WindowID DisplayServerAndroid::get_window_at_screen_position(const Point2i &p_position) const {
	return DisplayServerEnums::MAIN_WINDOW_ID;
}

int64_t DisplayServerAndroid::window_get_native_handle(DisplayServerEnums::HandleType p_handle_type, DisplayServerEnums::WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != DisplayServerEnums::MAIN_WINDOW_ID, 0);
	switch (p_handle_type) {
		case DisplayServerEnums::WINDOW_HANDLE: {
			return reinterpret_cast<int64_t>(static_cast<OS_Android *>(OS::get_singleton())->get_godot_java()->get_activity());
		}
		case DisplayServerEnums::WINDOW_VIEW: {
			return 0; // Not supported.
		}
#ifdef GLES3_ENABLED
		case DisplayServerEnums::DISPLAY_HANDLE: {
			if (rendering_driver == "opengl3") {
				return reinterpret_cast<int64_t>(eglGetCurrentDisplay());
			}
			return 0;
		}
		case DisplayServerEnums::OPENGL_CONTEXT: {
			if (rendering_driver == "opengl3") {
				return reinterpret_cast<int64_t>(eglGetCurrentContext());
			}
			return 0;
		}
		case DisplayServerEnums::EGL_DISPLAY: {
			// @todo Find a way to get this from the Java side.
			return 0;
		}
		case DisplayServerEnums::EGL_CONFIG: {
			// @todo Find a way to get this from the Java side.
			return 0;
		}
#endif
		default: {
			return 0;
		}
	}
}

void DisplayServerAndroid::window_attach_instance_id(ObjectID p_instance, DisplayServerEnums::WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerAndroid::window_get_attached_instance_id(DisplayServerEnums::WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerAndroid::window_set_title(const String &p_title, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

int DisplayServerAndroid::window_get_current_screen(DisplayServerEnums::WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != DisplayServerEnums::MAIN_WINDOW_ID, DisplayServerEnums::INVALID_SCREEN);
	return 0;
}

void DisplayServerAndroid::window_set_current_screen(int p_screen, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

Point2i DisplayServerAndroid::window_get_position(DisplayServerEnums::WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerAndroid::window_get_position_with_decorations(DisplayServerEnums::WindowID p_window) const {
	return Point2i();
}

void DisplayServerAndroid::window_set_position(const Point2i &p_position, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

void DisplayServerAndroid::window_set_transient(DisplayServerEnums::WindowID p_window, DisplayServerEnums::WindowID p_parent) {
	// Not supported on Android.
}

void DisplayServerAndroid::window_set_max_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

Size2i DisplayServerAndroid::window_get_max_size(DisplayServerEnums::WindowID p_window) const {
	return Size2i();
}

void DisplayServerAndroid::window_set_min_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

Size2i DisplayServerAndroid::window_get_min_size(DisplayServerEnums::WindowID p_window) const {
	return Size2i();
}

void DisplayServerAndroid::window_set_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

Size2i DisplayServerAndroid::window_get_size(DisplayServerEnums::WindowID p_window) const {
	return OS_Android::get_singleton()->get_display_size();
}

Size2i DisplayServerAndroid::window_get_size_with_decorations(DisplayServerEnums::WindowID p_window) const {
	return OS_Android::get_singleton()->get_display_size();
}

void DisplayServerAndroid::window_set_mode(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::WindowID p_window) {
	OS_Android::get_singleton()->get_godot_java()->enable_immersive_mode(p_mode == DisplayServerEnums::WINDOW_MODE_FULLSCREEN || p_mode == DisplayServerEnums::WINDOW_MODE_EXCLUSIVE_FULLSCREEN);
}

DisplayServerEnums::WindowMode DisplayServerAndroid::window_get_mode(DisplayServerEnums::WindowID p_window) const {
	if (OS_Android::get_singleton()->get_godot_java()->is_in_immersive_mode()) {
		return DisplayServerEnums::WINDOW_MODE_FULLSCREEN;
	} else {
		return DisplayServerEnums::WINDOW_MODE_MAXIMIZED;
	}
}

bool DisplayServerAndroid::window_is_maximize_allowed(DisplayServerEnums::WindowID p_window) const {
	return false;
}

void DisplayServerAndroid::window_set_flag(DisplayServerEnums::WindowFlags p_flag, bool p_enabled, DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

bool DisplayServerAndroid::window_get_flag(DisplayServerEnums::WindowFlags p_flag, DisplayServerEnums::WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != DisplayServerEnums::MAIN_WINDOW_ID, false);
	switch (p_flag) {
		case DisplayServerEnums::WindowFlags::WINDOW_FLAG_TRANSPARENT:
			return is_window_transparency_available();

		default:
			return false;
	}
}

void DisplayServerAndroid::window_request_attention(DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

void DisplayServerAndroid::window_move_to_foreground(DisplayServerEnums::WindowID p_window) {
	// Not supported on Android.
}

bool DisplayServerAndroid::window_is_focused(DisplayServerEnums::WindowID p_window) const {
	return true;
}

bool DisplayServerAndroid::window_can_draw(DisplayServerEnums::WindowID p_window) const {
	return true;
}

bool DisplayServerAndroid::can_any_window_draw() const {
	return true;
}

void DisplayServerAndroid::window_set_color(const Color &p_color) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_NULL(godot_java);
	godot_java->set_window_color(p_color);
}

void DisplayServerAndroid::process_events() {
	Input::get_singleton()->flush_buffered_events();
}

Vector<String> DisplayServerAndroid::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef GLES3_ENABLED
	drivers.push_back("opengl3");
#endif
#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif

	return drivers;
}

DisplayServer *DisplayServerAndroid::create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerAndroid(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error));
	if (r_error != OK) {
		if (p_rendering_driver == "vulkan") {
			OS::get_singleton()->alert(
					"Your device seems not to support the required Vulkan version.\n\n"
					"Please try exporting your game using the 'gl_compatibility' renderer.",
					"Unable to initialize Vulkan video driver");
		} else {
			OS::get_singleton()->alert(
					"Your device seems not to support the required OpenGL ES 3.0 version.",
					"Unable to initialize OpenGL video driver");
		}
	}
	return ds;
}

void DisplayServerAndroid::register_android_driver() {
	register_create_function("android", create_func, get_rendering_drivers_func);
}

#ifdef VULKAN_ENABLED
bool DisplayServerAndroid::check_vulkan_global_context(bool p_vulkan_requirements_met) {
	if (!rendering_context_global_checked) {
		bool fallback_to_opengl3 = GLOBAL_GET("rendering/rendering_device/fallback_to_opengl3");
		Error err = ERR_CANT_CREATE;
		if (p_vulkan_requirements_met) {
			rendering_context_global = memnew(RenderingContextDriverVulkanAndroid);
			err = rendering_context_global->initialize();
		}

		if (err != OK) {
			if (rendering_context_global != nullptr) {
				memdelete(rendering_context_global);
				rendering_context_global = nullptr;
			}

#if defined(GLES3_ENABLED)
			if (fallback_to_opengl3) {
				WARN_PRINT("Your device does not seem to support Vulkan, switching to OpenGL 3.");
				OS::get_singleton()->set_current_rendering_driver_name("opengl3", OS::RENDERING_SOURCE_FALLBACK);
				OS::get_singleton()->set_current_rendering_method("gl_compatibility", OS::RENDERING_SOURCE_FALLBACK);
			} else
#endif
			{
				ERR_PRINT("Failed to initialize Vulkan context.");
			}
		}

		rendering_context_global_checked = true;
	}

	return rendering_context_global != nullptr;
}

void DisplayServerAndroid::free_vulkan_global_context() {
	if (rendering_context_global != nullptr) {
		memdelete(rendering_context_global);
		rendering_context_global = nullptr;
		rendering_context_global_checked = false;
	}
}
#endif

void DisplayServerAndroid::reset_window() {
#if defined(RD_ENABLED)
	if (rendering_context_global) {
		if (rendering_device) {
			rendering_device->screen_free(DisplayServerEnums::MAIN_WINDOW_ID);
		}

		DisplayServerEnums::VSyncMode last_vsync_mode = rendering_context_global->window_get_vsync_mode(DisplayServerEnums::MAIN_WINDOW_ID);
		rendering_context_global->window_destroy(DisplayServerEnums::MAIN_WINDOW_ID);

		union {
#ifdef VULKAN_ENABLED
			RenderingContextDriverVulkanAndroid::WindowPlatformData vulkan;
#endif
		} wpd;
#ifdef VULKAN_ENABLED
		if (rendering_driver == "vulkan") {
			ANativeWindow *native_window = OS_Android::get_singleton()->get_native_window();
			ERR_FAIL_NULL(native_window);
			wpd.vulkan.window = native_window;
		}
#endif

		if (rendering_context_global->window_create(DisplayServerEnums::MAIN_WINDOW_ID, &wpd) != OK) {
			ERR_PRINT(vformat("Failed to reset %s window.", rendering_driver));
			return;
		}

		Size2i display_size = OS_Android::get_singleton()->get_display_size();
		rendering_context_global->window_set_size(DisplayServerEnums::MAIN_WINDOW_ID, display_size.width, display_size.height);
		rendering_context_global->window_set_vsync_mode(DisplayServerEnums::MAIN_WINDOW_ID, last_vsync_mode);

		if (rendering_device) {
			rendering_device->screen_create(DisplayServerEnums::MAIN_WINDOW_ID);
		}
	}
#endif
}

void DisplayServerAndroid::notify_surface_changed(int p_width, int p_height) {
	if (rect_changed_callback.is_valid()) {
		rect_changed_callback.call(Rect2i(0, 0, p_width, p_height));
	}
}

void DisplayServerAndroid::notify_application_paused() {
#if defined(RD_ENABLED)
	if (rendering_device) {
		rendering_device->update_pipeline_cache();
	}
#endif // defined(RD_ENABLED)
}

DisplayServerAndroid::DisplayServerAndroid(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	rendering_driver = p_rendering_driver;

	keep_screen_on = GLOBAL_GET("display/window/energy_saving/keep_screen_on");

	native_menu = memnew(NativeMenu);

#ifdef VULKAN_ENABLED
	if (rendering_driver == "vulkan") {
		check_vulkan_global_context(true);
		if (rendering_context_global == nullptr) {
			ERR_PRINT("Can't initialize display server with Vulkan driver because no Vulkan context is available.");
			r_error = ERR_UNAVAILABLE;
			return;
		}

		ANativeWindow *native_window = OS_Android::get_singleton()->get_native_window();
		ERR_FAIL_NULL(native_window);

		RenderingContextDriverVulkanAndroid::WindowPlatformData wpd;
		wpd.window = native_window;

		if (rendering_context_global->window_create(DisplayServerEnums::MAIN_WINDOW_ID, &wpd) != OK) {
			ERR_PRINT(vformat("Failed to create %s window.", rendering_driver));
			r_error = ERR_UNAVAILABLE;
			return;
		}

		Size2i display_size = OS_Android::get_singleton()->get_display_size();
		rendering_context_global->window_set_size(DisplayServerEnums::MAIN_WINDOW_ID, display_size.width, display_size.height);
		rendering_context_global->window_set_vsync_mode(DisplayServerEnums::MAIN_WINDOW_ID, p_vsync_mode);

		rendering_device = memnew(RenderingDevice);
		if (rendering_device->initialize(rendering_context_global, DisplayServerEnums::MAIN_WINDOW_ID) != OK) {
			rendering_device = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}

		rendering_device->screen_create(DisplayServerEnums::MAIN_WINDOW_ID);

		RendererCompositorRD::make_current();
	}
#endif

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		RasterizerGLES3::make_current(false);
	}
#endif

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerAndroid::~DisplayServerAndroid() {
	if (native_menu) {
		memdelete(native_menu);
		native_menu = nullptr;
	}

#if defined(RD_ENABLED)
	if (rendering_device) {
		memdelete(rendering_device);
	}

	free_vulkan_global_context();
#endif
}

void DisplayServerAndroid::process_accelerometer(const Vector3 &p_accelerometer) {
	Input::get_singleton()->set_accelerometer(p_accelerometer);
}

void DisplayServerAndroid::process_gravity(const Vector3 &p_gravity) {
	Input::get_singleton()->set_gravity(p_gravity);
}

void DisplayServerAndroid::process_magnetometer(const Vector3 &p_magnetometer) {
	Input::get_singleton()->set_magnetometer(p_magnetometer);
}

void DisplayServerAndroid::process_gyroscope(const Vector3 &p_gyroscope) {
	Input::get_singleton()->set_gyroscope(p_gyroscope);
}

void DisplayServerAndroid::_mouse_update_mode() {
	DisplayServerEnums::MouseMode wanted_mouse_mode = mouse_mode_override_enabled
			? mouse_mode_override
			: mouse_mode_base;

	if (!OS_Android::get_singleton()->get_godot_java()->get_godot_view()->can_update_pointer_icon() || !OS_Android::get_singleton()->get_godot_java()->get_godot_view()->can_capture_pointer()) {
		return;
	}
	if (mouse_mode == wanted_mouse_mode) {
		return;
	}

	if (wanted_mouse_mode == DisplayServerEnums::MouseMode::MOUSE_MODE_HIDDEN) {
		OS_Android::get_singleton()->get_godot_java()->get_godot_view()->set_pointer_icon(CURSOR_TYPE_NULL);
	} else {
		cursor_set_shape(cursor_shape);
	}

	if (wanted_mouse_mode == DisplayServerEnums::MouseMode::MOUSE_MODE_CAPTURED) {
		OS_Android::get_singleton()->get_godot_java()->get_godot_view()->request_pointer_capture();
	} else {
		OS_Android::get_singleton()->get_godot_java()->get_godot_view()->release_pointer_capture();
	}

	mouse_mode = wanted_mouse_mode;
}

void DisplayServerAndroid::mouse_set_mode(DisplayServerEnums::MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode, DisplayServerEnums::MouseMode::MOUSE_MODE_MAX);
	if (p_mode == mouse_mode_base) {
		return;
	}
	mouse_mode_base = p_mode;
	_mouse_update_mode();
}

DisplayServerEnums::MouseMode DisplayServerAndroid::mouse_get_mode() const {
	return mouse_mode;
}

void DisplayServerAndroid::mouse_set_mode_override(DisplayServerEnums::MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode, DisplayServerEnums::MouseMode::MOUSE_MODE_MAX);
	if (p_mode == mouse_mode_override) {
		return;
	}
	mouse_mode_override = p_mode;
	_mouse_update_mode();
}

DisplayServerEnums::MouseMode DisplayServerAndroid::mouse_get_mode_override() const {
	return mouse_mode_override;
}

void DisplayServerAndroid::mouse_set_mode_override_enabled(bool p_override_enabled) {
	mouse_mode_override_enabled = p_override_enabled;
	_mouse_update_mode();
}

bool DisplayServerAndroid::mouse_is_mode_override_enabled() const {
	return mouse_mode_override_enabled;
}

Point2i DisplayServerAndroid::mouse_get_position() const {
	return Input::get_singleton()->get_mouse_position();
}

BitField<MouseButtonMask> DisplayServerAndroid::mouse_get_button_state() const {
	return Input::get_singleton()->get_mouse_button_mask();
}

void DisplayServerAndroid::_cursor_set_shape_helper(DisplayServerEnums::CursorShape p_shape, bool force) {
	if (!OS_Android::get_singleton()->get_godot_java()->get_godot_view()->can_update_pointer_icon()) {
		return;
	}
	if (cursor_shape == p_shape && !force) {
		return;
	}

	cursor_shape = p_shape;

	if (mouse_mode == DisplayServerEnums::MouseMode::MOUSE_MODE_VISIBLE || mouse_mode == DisplayServerEnums::MouseMode::MOUSE_MODE_CONFINED) {
		OS_Android::get_singleton()->get_godot_java()->get_godot_view()->set_pointer_icon(android_cursors[cursor_shape]);
	}
}

void DisplayServerAndroid::cursor_set_shape(DisplayServerEnums::CursorShape p_shape) {
	ERR_FAIL_INDEX(p_shape, DisplayServerEnums::CURSOR_MAX);
	_cursor_set_shape_helper(p_shape);
}

DisplayServerEnums::CursorShape DisplayServerAndroid::cursor_get_shape() const {
	return cursor_shape;
}

void DisplayServerAndroid::cursor_set_custom_image(const Ref<Resource> &p_cursor, DisplayServerEnums::CursorShape p_shape, const Vector2 &p_hotspot) {
	ERR_FAIL_INDEX(p_shape, DisplayServerEnums::CURSOR_MAX);
	String cursor_path = p_cursor.is_valid() ? p_cursor->get_path() : "";
	if (!cursor_path.is_empty()) {
		cursor_path = ProjectSettings::get_singleton()->globalize_path(cursor_path);
	}
	OS_Android::get_singleton()->get_godot_java()->get_godot_view()->configure_pointer_icon(android_cursors[cursor_shape], cursor_path, p_hotspot);
	_cursor_set_shape_helper(p_shape, true);
}

void DisplayServerAndroid::window_set_vsync_mode(DisplayServerEnums::VSyncMode p_vsync_mode, DisplayServerEnums::WindowID p_window) {
#if defined(RD_ENABLED)
	if (rendering_context_global) {
		rendering_context_global->window_set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServerEnums::VSyncMode DisplayServerAndroid::window_get_vsync_mode(DisplayServerEnums::WindowID p_window) const {
#if defined(RD_ENABLED)
	if (rendering_context_global) {
		return rendering_context_global->window_get_vsync_mode(p_window);
	}
#endif
	return DisplayServerEnums::VSYNC_ENABLED;
}

void DisplayServerAndroid::reset_swap_buffers_flag() {
	swap_buffers_flag = false;
}

bool DisplayServerAndroid::should_swap_buffers() const {
	return swap_buffers_flag;
}

void DisplayServerAndroid::swap_buffers() {
	swap_buffers_flag = true;
}

void DisplayServerAndroid::set_native_icon(const String &p_filename) {
	// NOT SUPPORTED
}

void DisplayServerAndroid::set_icon(const Ref<Image> &p_icon) {
	// NOT SUPPORTED
}

bool DisplayServerAndroid::is_window_transparency_available() const {
	return GLOBAL_GET_CACHED(bool, "display/window/per_pixel_transparency/allowed");
}
