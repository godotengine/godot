/**************************************************************************/
/*  display_server_switch.cpp                                             */
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

#include "display_server_switch.h"

#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "main/main.h"
#include "scene/resources/texture.h"

#include "drivers/gles3/rasterizer_gles3.h"

#include "switch_wrapper.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

bool DisplayServerSwitch::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_TOUCHSCREEN:
		case FEATURE_SWAP_BUFFERS:
			return true;
		default: {
		}
	}
	return false;
}

String DisplayServerSwitch::get_name() const {
	return "NVN";
}

int DisplayServerSwitch::get_screen_count() const {
	return 1;
}

int DisplayServerSwitch::get_primary_screen() const {
	return 0;
}

int DisplayServerSwitch::get_keyboard_focus_screen() const {
	return get_primary_screen();
}

Point2i DisplayServerSwitch::screen_get_position(int p_screen) const {
	return Point2i(0, 0);
}

Size2i DisplayServerSwitch::screen_get_size(int p_screen) const {
	if (_window) {
		unsigned int w, h;
		nwindowGetDimensions(_window, &w, &h);
		return Size2i(w, h);
	}
	return Size2i(0, 0);
}

Rect2i DisplayServerSwitch::screen_get_usable_rect(int p_screen) const {
	Size2i size = screen_get_size();
	return Rect2i(0, 0, size[0], size[1]);
}

int DisplayServerSwitch::screen_get_dpi(int p_screen) const {
	return 236;
}

float DisplayServerSwitch::screen_get_refresh_rate(int p_screen) const {
	return SCREEN_REFRESH_RATE_FALLBACK;
}

int64_t DisplayServerSwitch::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	return 0;
}

void DisplayServerSwitch::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
}

ObjectID DisplayServerSwitch::window_get_attached_instance_id(WindowID p_window) const {
	return ObjectID();
}

Vector<DisplayServer::WindowID> DisplayServerSwitch::get_window_list() const {
	Vector<DisplayServer::WindowID> list;
	list.push_back(0);
	return list;
}

DisplayServer::WindowID DisplayServerSwitch::get_window_at_screen_position(const Point2i &p_position) const {
	return INVALID_WINDOW_ID;
}

void DisplayServerSwitch::window_set_title(const String &p_title, WindowID p_window) {
}

void DisplayServerSwitch::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
}

void DisplayServerSwitch::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
}

void DisplayServerSwitch::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
}

void DisplayServerSwitch::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
}

void DisplayServerSwitch::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
}

int DisplayServerSwitch::window_get_current_screen(WindowID p_window) const {
	return 0;
}

void DisplayServerSwitch::gl_window_make_current(DisplayServer::WindowID p_window_id) {
	if (_gl_manager) {
		_gl_manager->make_current();
	}
}

void DisplayServerSwitch::window_set_current_screen(int p_screen, WindowID p_window) {
	// always on the current screen
}

void DisplayServerSwitch::window_set_transient(WindowID p_window, WindowID p_parent) {
}

Point2i DisplayServerSwitch::window_get_position(WindowID p_window) const {
	return screen_get_position();
}

Point2i DisplayServerSwitch::window_get_position_with_decorations(WindowID p_window) const {
	return window_get_position();
}

void DisplayServerSwitch::window_set_position(const Point2i &p_position, WindowID p_window) {
}

void DisplayServerSwitch::window_set_max_size(const Size2i p_size, WindowID p_window) {
}

Size2i DisplayServerSwitch::window_get_max_size(WindowID p_window) const {
	return screen_get_size();
}

void DisplayServerSwitch::window_set_min_size(const Size2i p_size, WindowID p_window) {
}

Size2i DisplayServerSwitch::window_get_min_size(WindowID p_window) const {
	return screen_get_size();
}

void DisplayServerSwitch::window_set_size(const Size2i p_size, WindowID p_window) {
}

Size2i DisplayServerSwitch::window_get_size(WindowID p_window) const {
	return screen_get_size();
}

Size2i DisplayServerSwitch::window_get_size_with_decorations(WindowID p_window) const {
	return window_get_size();
}

bool DisplayServerSwitch::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerSwitch::window_set_mode(WindowMode p_mode, WindowID p_window) {
}

DisplayServer::WindowMode DisplayServerSwitch::window_get_mode(WindowID p_window) const {
	return WINDOW_MODE_EXCLUSIVE_FULLSCREEN;
}

void DisplayServerSwitch::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
}

bool DisplayServerSwitch::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED:
		case WINDOW_FLAG_BORDERLESS:
		case WINDOW_FLAG_ALWAYS_ON_TOP:
			return true;
		case WINDOW_FLAG_TRANSPARENT:
		case WINDOW_FLAG_NO_FOCUS:
		case WINDOW_FLAG_MOUSE_PASSTHROUGH:
		case WINDOW_FLAG_POPUP:
			return false;
		default:
			return false;
	}

	return false;
}

void DisplayServerSwitch::window_request_attention(WindowID p_window) {
}

void DisplayServerSwitch::window_move_to_foreground(WindowID p_window) {
}

bool DisplayServerSwitch::window_is_focused(WindowID p_window) const {
}

bool DisplayServerSwitch::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerSwitch::can_any_window_draw() const {
	return true;
}

int DisplayServerSwitch::keyboard_get_layout_count() const {
	return 1;
}

int DisplayServerSwitch::keyboard_get_current_layout() const {
	return 0;
}

void DisplayServerSwitch::keyboard_set_current_layout(int p_index) {
}

String DisplayServerSwitch::keyboard_get_layout_language(int p_index) const {
	//TODO:vrince doable
	return "en";
}

String DisplayServerSwitch::keyboard_get_layout_name(int p_index) const {
	return "EN";
}

Key DisplayServerSwitch::keyboard_get_keycode_from_physical(Key p_keycode) const {
	//TODO:vrince doable
	return Key::NONE;
}

void DisplayServerSwitch::process_events() {
	// input
	// keyboard
	// pads

	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerSwitch::release_rendering_thread() {
	if (_gl_manager) {
		_gl_manager->release_current();
	}
}

void DisplayServerSwitch::make_rendering_thread() {
	if (_gl_manager) {
		_gl_manager->make_current();
	}
}

void DisplayServerSwitch::swap_buffers() {
	if (_gl_manager) {
		_gl_manager->swap_buffers();
	}
}

void DisplayServerSwitch::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	if (_gl_manager) {
		_gl_manager->set_use_vsync(p_vsync_mode == DisplayServer::VSYNC_ENABLED);
	}
}

DisplayServer::VSyncMode DisplayServerSwitch::window_get_vsync_mode(WindowID p_window) const {
	if (_gl_manager) {
		return _gl_manager->is_using_vsync() ? DisplayServer::VSYNC_ENABLED : DisplayServer::VSYNC_DISABLED;
	}
}

Vector<String> DisplayServerSwitch::get_rendering_drivers_func() {
	Vector<String> drivers;
	drivers.push_back("opengl3");
	return drivers;
}

DisplayServer *DisplayServerSwitch::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerSwitch(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert(
				"Make sure you are un full memory mode (not applet mode).",
				"Unable to initialize OpenGL video driver");
	}
	return ds;
}

DisplayServerSwitch::DisplayServerSwitch(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error) {
	OS::get_singleton()->print("DisplayServerSwitch\n");

	r_error = FAILED;
	_window = nwindowGetDefault();

	//Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	// Initialize context and rendering device.
	_gl_manager = memnew(GLManagerSwitch());

	if (!nwindowIsValid(_window)) {
		_window = nullptr;
		ERR_FAIL_MSG("Windows is not valid");
		return;
	}

	unsigned int w, h;
	nwindowGetDimensions(_window, &w, &h);
	printf("nwindows size (%d,%d)\n", w, h);

	if (_gl_manager->initialize(_window) != OK) {
		memdelete(_gl_manager);
		_window = nullptr;
		_gl_manager = nullptr;
		ERR_FAIL_MSG("Cannot initialize GL");
		return;
	}

	RasterizerGLES3::make_current();
	RasterizerGLES3::make_current();

	// plug nvwindows id
	// WindowID main_window = _create_window(p_mode, p_vsync_mode, p_flags, Rect2i(window_position, p_resolution));

	r_error = OK;
}

DisplayServerSwitch::~DisplayServerSwitch() {
	if (_gl_manager) {
		_gl_manager->cleanup();
	}

	if (_gl_manager) {
		memdelete(_gl_manager);
		_gl_manager = nullptr;
	}
}

void DisplayServerSwitch::register_NVN_driver() {
	register_create_function("NVN", create_func, get_rendering_drivers_func);
}
