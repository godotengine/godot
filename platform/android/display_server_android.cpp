/*************************************************************************/
/*  display_server_android.cpp                                           */
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

#include "display_server_android.h"

#include "android_keys_utils.h"
#include "core/project_settings.h"
#include "java_godot_io_wrapper.h"
#include "java_godot_wrapper.h"
#include "os_android.h"

#if defined(OPENGL_ENABLED)
#include "drivers/gles2/rasterizer_gles2.h"
#endif
#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/android/vulkan/vulkan_context_android.h"
#include "servers/rendering/rasterizer_rd/rasterizer_rd.h"
#endif

DisplayServerAndroid *DisplayServerAndroid::get_singleton() {
	return (DisplayServerAndroid *)DisplayServer::get_singleton();
}

bool DisplayServerAndroid::has_feature(Feature p_feature) const {
	switch (p_feature) {
		//case FEATURE_CONSOLE_WINDOW:
		//case FEATURE_CURSOR_SHAPE:
		//case FEATURE_CUSTOM_CURSOR_SHAPE:
		//case FEATURE_GLOBAL_MENU:
		//case FEATURE_HIDPI:
		//case FEATURE_ICON:
		//case FEATURE_IME:
		//case FEATURE_MOUSE:
		//case FEATURE_MOUSE_WARP:
		//case FEATURE_NATIVE_DIALOG:
		//case FEATURE_NATIVE_ICON:
		//case FEATURE_NATIVE_VIDEO:
		//case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_CLIPBOARD:
		case FEATURE_KEEP_SCREEN_ON:
		case FEATURE_ORIENTATION:
		case FEATURE_TOUCHSCREEN:
		case FEATURE_VIRTUAL_KEYBOARD:
			return true;
		default:
			return false;
	}
}

String DisplayServerAndroid::get_name() const {
	return "Android";
}

void DisplayServerAndroid::clipboard_set(const String &p_text) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_COND(!godot_java);

	if (godot_java->has_set_clipboard()) {
		godot_java->set_clipboard(p_text);
	} else {
		DisplayServer::clipboard_set(p_text);
	}
}

String DisplayServerAndroid::clipboard_get() const {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_COND_V(!godot_java, String());

	if (godot_java->has_get_clipboard()) {
		return godot_java->get_clipboard();
	} else {
		return DisplayServer::clipboard_get();
	}
}

void DisplayServerAndroid::screen_set_keep_on(bool p_enable) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_COND(!godot_java);

	godot_java->set_keep_screen_on(p_enable);
	keep_screen_on = p_enable;
}

bool DisplayServerAndroid::screen_is_kept_on() const {
	return keep_screen_on;
}

void DisplayServerAndroid::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_COND(!godot_io_java);

	godot_io_java->set_screen_orientation(p_orientation);
}

DisplayServer::ScreenOrientation DisplayServerAndroid::screen_get_orientation(int p_screen) const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_COND_V(!godot_io_java, SCREEN_LANDSCAPE);

	return (ScreenOrientation)godot_io_java->get_screen_orientation();
}

int DisplayServerAndroid::get_screen_count() const {
	return 1;
}

Point2i DisplayServerAndroid::screen_get_position(int p_screen) const {
	return Point2i(0, 0);
}

Size2i DisplayServerAndroid::screen_get_size(int p_screen) const {
	return OS_Android::get_singleton()->get_display_size();
}

Rect2i DisplayServerAndroid::screen_get_usable_rect(int p_screen) const {
	Size2i display_size = OS_Android::get_singleton()->get_display_size();
	return Rect2i(0, 0, display_size.width, display_size.height);
}

int DisplayServerAndroid::screen_get_dpi(int p_screen) const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_COND_V(!godot_io_java, 0);

	return godot_io_java->get_screen_dpi();
}

bool DisplayServerAndroid::screen_is_touchscreen(int p_screen) const {
	return true;
}

void DisplayServerAndroid::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, int p_max_length) {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_COND(!godot_io_java);

	if (godot_io_java->has_vk()) {
		godot_io_java->show_vk(p_existing_text, p_max_length);
	} else {
		ERR_PRINT("Virtual keyboard not available");
	}
}

void DisplayServerAndroid::virtual_keyboard_hide() {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_COND(!godot_io_java);

	if (godot_io_java->has_vk()) {
		godot_io_java->hide_vk();
	} else {
		ERR_PRINT("Virtual keyboard not available");
	}
}

int DisplayServerAndroid::virtual_keyboard_get_height() const {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_COND_V(!godot_io_java, 0);

	return godot_io_java->get_vk_height();
}

void DisplayServerAndroid::window_set_window_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	window_event_callback = p_callable;
}

void DisplayServerAndroid::window_set_input_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerAndroid::window_set_input_text_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	input_text_callback = p_callable;
}

void DisplayServerAndroid::window_set_rect_changed_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

void DisplayServerAndroid::window_set_drop_files_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

void DisplayServerAndroid::_window_callback(const Callable &p_callable, const Variant &p_arg) const {
	if (!p_callable.is_null()) {
		const Variant *argp = &p_arg;
		Variant ret;
		Callable::CallError ce;
		p_callable.call((const Variant **)&argp, 1, ret, ce);
	}
}

void DisplayServerAndroid::send_window_event(DisplayServer::WindowEvent p_event) const {
	_window_callback(window_event_callback, int(p_event));
}

void DisplayServerAndroid::send_input_event(const Ref<InputEvent> &p_event) const {
	_window_callback(input_event_callback, p_event);
}

void DisplayServerAndroid::send_input_text(const String &p_text) const {
	_window_callback(input_text_callback, p_text);
}

void DisplayServerAndroid::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	DisplayServerAndroid::get_singleton()->send_input_event(p_event);
}

Vector<DisplayServer::WindowID> DisplayServerAndroid::get_window_list() const {
	Vector<WindowID> ret;
	ret.push_back(MAIN_WINDOW_ID);
	return ret;
}

DisplayServer::WindowID DisplayServerAndroid::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

void DisplayServerAndroid::window_attach_instance_id(ObjectID p_instance, DisplayServer::WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerAndroid::window_get_attached_instance_id(DisplayServer::WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerAndroid::window_set_title(const String &p_title, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

int DisplayServerAndroid::window_get_current_screen(DisplayServer::WindowID p_window) const {
	return SCREEN_OF_MAIN_WINDOW;
}

void DisplayServerAndroid::window_set_current_screen(int p_screen, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

Point2i DisplayServerAndroid::window_get_position(DisplayServer::WindowID p_window) const {
	return Point2i();
}

void DisplayServerAndroid::window_set_position(const Point2i &p_position, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

void DisplayServerAndroid::window_set_transient(DisplayServer::WindowID p_window, DisplayServer::WindowID p_parent) {
	// Not supported on Android.
}

void DisplayServerAndroid::window_set_max_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

Size2i DisplayServerAndroid::window_get_max_size(DisplayServer::WindowID p_window) const {
	return Size2i();
}

void DisplayServerAndroid::window_set_min_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

Size2i DisplayServerAndroid::window_get_min_size(DisplayServer::WindowID p_window) const {
	return Size2i();
}

void DisplayServerAndroid::window_set_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

Size2i DisplayServerAndroid::window_get_size(DisplayServer::WindowID p_window) const {
	return OS_Android::get_singleton()->get_display_size();
}

Size2i DisplayServerAndroid::window_get_real_size(DisplayServer::WindowID p_window) const {
	return OS_Android::get_singleton()->get_display_size();
}

void DisplayServerAndroid::window_set_mode(DisplayServer::WindowMode p_mode, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

DisplayServer::WindowMode DisplayServerAndroid::window_get_mode(DisplayServer::WindowID p_window) const {
	return WINDOW_MODE_FULLSCREEN;
}

bool DisplayServerAndroid::window_is_maximize_allowed(DisplayServer::WindowID p_window) const {
	return false;
}

void DisplayServerAndroid::window_set_flag(DisplayServer::WindowFlags p_flag, bool p_enabled, DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

bool DisplayServerAndroid::window_get_flag(DisplayServer::WindowFlags p_flag, DisplayServer::WindowID p_window) const {
	return false;
}

void DisplayServerAndroid::window_request_attention(DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

void DisplayServerAndroid::window_move_to_foreground(DisplayServer::WindowID p_window) {
	// Not supported on Android.
}

bool DisplayServerAndroid::window_can_draw(DisplayServer::WindowID p_window) const {
	return true;
}

bool DisplayServerAndroid::can_any_window_draw() const {
	return true;
}

void DisplayServerAndroid::alert(const String &p_alert, const String &p_title) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_COND(!godot_java);

	godot_java->alert(p_alert, p_title);
}

void DisplayServerAndroid::process_events() {
	// Nothing to do
}

Vector<String> DisplayServerAndroid::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef OPENGL_ENABLED
	drivers.push_back("opengl");
#endif
#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif

	return drivers;
}

DisplayServer *DisplayServerAndroid::create_func(const String &p_rendering_driver, DisplayServer::WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	return memnew(DisplayServerAndroid(p_rendering_driver, p_mode, p_flags, p_resolution, r_error));
}

void DisplayServerAndroid::register_android_driver() {
	register_create_function("android", create_func, get_rendering_drivers_func);
}

DisplayServerAndroid::DisplayServerAndroid(const String &p_rendering_driver, DisplayServer::WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	rendering_driver = p_rendering_driver;

	// TODO: rendering_driver is broken, change when different drivers are supported again
	rendering_driver = "vulkan";

	keep_screen_on = GLOBAL_GET("display/window/energy_saving/keep_screen_on");

#if defined(OPENGL_ENABLED)
	if (rendering_driver == "opengl") {
		bool gl_initialization_error = false;

		if (RasterizerGLES2::is_viable() == OK) {
			RasterizerGLES2::register_config();
			RasterizerGLES2::make_current();
		} else {
			gl_initialization_error = true;
		}

		if (gl_initialization_error) {
			OS::get_singleton()->alert("Your device does not support any of the supported OpenGL versions.\n"
									   "Please try updating your Android version.",
					"Unable to initialize video driver");
			return;
		}
	}
#endif

#if defined(VULKAN_ENABLED)
	context_vulkan = nullptr;
	rendering_device_vulkan = nullptr;

	if (rendering_driver == "vulkan") {
		ANativeWindow *native_window = OS_Android::get_singleton()->get_native_window();
		ERR_FAIL_COND(!native_window);

		context_vulkan = memnew(VulkanContextAndroid);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			ERR_FAIL_MSG("Failed to initialize Vulkan context");
		}

		Size2i display_size = OS_Android::get_singleton()->get_display_size();
		if (context_vulkan->window_create(native_window, display_size.width, display_size.height) == -1) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			ERR_FAIL_MSG("Failed to create Vulkan window.");
		}

		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RasterizerRD::make_current();
	}
#endif

	InputFilter::get_singleton()->set_event_dispatch_function(_dispatch_input_events);
}

DisplayServerAndroid::~DisplayServerAndroid() {
#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		if (rendering_device_vulkan) {
			rendering_device_vulkan->finalize();
			memdelete(rendering_device_vulkan);
		}

		if (context_vulkan) {
			memdelete(context_vulkan);
		}
	}
#endif
}

Point2i DisplayServerAndroid::mouse_get_position() const {

	return last_mouse_position;
}

int DisplayServerAndroid::mouse_get_button_state() const {

	return last_mouse_buttons_mask;
}

void DisplayServerAndroid::process_joy_event(DisplayServerAndroid::JoypadEvent p_event) {
	switch (p_event.type) {
		case JOY_EVENT_BUTTON:
			InputFilter::get_singleton()->joy_button(p_event.device, p_event.index, p_event.pressed);
			break;
		case JOY_EVENT_AXIS:
			InputFilter::JoyAxis value;
			value.min = -1;
			value.value = p_event.value;
			InputFilter::get_singleton()->joy_axis(p_event.device, p_event.index, value);
			break;
		case JOY_EVENT_HAT:
			InputFilter::get_singleton()->joy_hat(p_event.device, p_event.hat);
			break;
		default:
			return;
	}
}

void DisplayServerAndroid::process_key_event(int p_keycode, int p_scancode, int p_unicode_char, bool p_pressed) {
	Ref<InputEventKey> ev;
	ev.instance();
	int val = p_unicode_char;
	int keycode = android_get_keysym(p_keycode);
	int phy_keycode = android_get_keysym(p_scancode);
	ev->set_keycode(keycode);
	ev->set_physical_keycode(phy_keycode);
	ev->set_unicode(val);
	ev->set_pressed(p_pressed);

	if (val == '\n') {
		ev->set_keycode(KEY_ENTER);
	} else if (val == 61448) {
		ev->set_keycode(KEY_BACKSPACE);
		ev->set_unicode(KEY_BACKSPACE);
	} else if (val == 61453) {
		ev->set_keycode(KEY_ENTER);
		ev->set_unicode(KEY_ENTER);
	} else if (p_keycode == 4) {
		OS_Android::get_singleton()->main_loop_request_go_back();
	}

	InputFilter::get_singleton()->parse_input_event(ev);
}

int DisplayServerAndroid::get_mouse_button_index(int android_motion_event_button_state) {
	if ((android_motion_event_button_state & BUTTON_PRIMARY) != 0) {
		return BUTTON_LEFT;
	}
	if ((android_motion_event_button_state & BUTTON_SECONDARY) != 0) {
		return BUTTON_RIGHT;
	}
	if ((android_motion_event_button_state & BUTTON_TERTIARY) != 0) {
		return BUTTON_MIDDLE;
	}
	if ((android_motion_event_button_state & BUTTON_FORWARD) != 0) {
		return BUTTON_XBUTTON1;
	}
	if ((android_motion_event_button_state & BUTTON_BACK) != 0) {
		return BUTTON_XBUTTON2;
	}

	return 0;
}

bool DisplayServerAndroid::is_mouse_pointer(DisplayServerAndroid::TouchPos touch_pos) const {
	return is_mouse_pointer(touch_pos.tool_type);
}

bool DisplayServerAndroid::is_mouse_pointer(int tool_type) const {
	return tool_type == TOOL_TYPE_MOUSE;
}

void DisplayServerAndroid::send_touch_event(DisplayServerAndroid::TouchPos touch_pos, int android_motion_event_action_button) {
	if (is_mouse_pointer(touch_pos)) {
		int button_index = get_mouse_button_index(android_motion_event_action_button);
		last_mouse_buttons_mask |= button_index;
		last_mouse_position = touch_pos.pos;

		Ref<InputEventMouseButton> ev;
		ev.instance();
		ev->set_pressed(true);
		ev->set_position(touch_pos.pos);
		ev->set_global_position(touch_pos.pos);
		ev->set_button_index(button_index);
		ev->set_button_mask(last_mouse_buttons_mask);
		InputFilter::get_singleton()->parse_input_event(ev);
	} else {
		Ref<InputEventScreenTouch> ev;
		ev.instance();
		ev->set_index(touch_pos.id);
		ev->set_pressed(true);
		ev->set_position(touch_pos.pos);
		InputFilter::get_singleton()->parse_input_event(ev);
	}
}

void DisplayServerAndroid::release_touch_event(TouchPos touch_pos, int android_motion_event_action_button, bool update_last_mouse_buttons_mask) {
	if (is_mouse_pointer(touch_pos)) {

		int button_index = get_mouse_button_index(android_motion_event_action_button);
		last_mouse_position = touch_pos.pos;
		if (update_last_mouse_buttons_mask) {
			last_mouse_buttons_mask &= ~button_index;
		}

		Ref<InputEventMouseButton> ev;
		ev.instance();
		ev->set_pressed(false);
		ev->set_position(touch_pos.pos);
		ev->set_global_position(touch_pos.pos);
		ev->set_button_index(button_index);
		ev->set_button_mask(last_mouse_buttons_mask);
		InputFilter::get_singleton()->parse_input_event(ev);

	} else {
		Ref<InputEventScreenTouch> ev;
		ev.instance();
		ev->set_index(touch_pos.id);
		ev->set_pressed(false);
		ev->set_position(touch_pos.pos);
		InputFilter::get_singleton()->parse_input_event(ev);
	}
}

void DisplayServerAndroid::release_touches(int android_motion_event_action_button, bool update_last_mouse_buttons_mask) {
	if (touch.size()) {
		//end all if exist
		for (int i = 0; i < touch.size(); i++) {

			TouchPos touch_pos = touch[i];
			release_touch_event(touch_pos, android_motion_event_action_button, update_last_mouse_buttons_mask);
		}
		touch.clear();
	}
}

void DisplayServerAndroid::process_touch(int motion_event_action, int motion_event_action_button, int p_pointer, const Vector<TouchPos> &p_points) {

	switch (motion_event_action) {
		case ACTION_DOWN:
		case ACTION_BUTTON_PRESS: { // gesture begin

			// end all if exist
			release_touches(motion_event_action_button);

			touch.resize(p_points.size());
			for (int i = 0; i < p_points.size(); i++) {
				touch.write[i].id = p_points[i].id;
				touch.write[i].pos = p_points[i].pos;
				touch.write[i].tool_type = p_points[i].tool_type;
			}

			//send touch
			for (int i = 0; i < touch.size(); i++) {

				TouchPos touch_pos = touch[i];
				send_touch_event(touch_pos, motion_event_action_button);
			}

		} break;

		case ACTION_MOVE: { // motion

			ERR_FAIL_COND(touch.size() != p_points.size());

			for (int i = 0; i < touch.size(); i++) {

				int idx = -1;
				for (int j = 0; j < p_points.size(); j++) {

					if (touch[i].id == p_points[j].id) {
						idx = j;
						break;
					}
				}

				ERR_CONTINUE(idx == -1);

				if (touch[i].pos == p_points[idx].pos)
					continue; //no move unncesearily

				TouchPos touch_pos = p_points[idx];
				if (is_mouse_pointer(touch_pos)) {
					Ref<InputEventMouseMotion> ev;
					ev.instance();
					ev->set_position(touch_pos.pos);
					ev->set_global_position(touch_pos.pos);
					ev->set_relative(touch_pos.pos - touch[i].pos);
					ev->set_button_mask(last_mouse_buttons_mask);
					InputFilter::get_singleton()->parse_input_event(ev);
				} else {
					Ref<InputEventScreenDrag> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_position(p_points[idx].pos);
					ev->set_relative(p_points[idx].pos - touch[i].pos);
					InputFilter::get_singleton()->parse_input_event(ev);
				}
				touch.write[i].pos = p_points[idx].pos;
				touch.write[i].tool_type = p_points[idx].tool_type;
			}

		} break;

		case ACTION_UP:
		case ACTION_BUTTON_RELEASE: {
			release_touches(motion_event_action_button, true);
		} break;

		case ACTION_CANCEL: {
			release_touches(motion_event_action_button);
			// Reset the `last_mouse_buttons_mask` if the tool type is TOOL_TYPE_MOUSE or TOOL_TYPE_STYLUS.
			for (int i = 0; i < p_points.size(); i++) {
				if (is_mouse_pointer(p_points[i])) {
					last_mouse_buttons_mask = 0;
					break;
				}
			}
		} break;

		case ACTION_POINTER_DOWN: { // add touch

			for (int i = 0; i < p_points.size(); i++) {
				if (p_points[i].id == p_pointer) {
					TouchPos tp = p_points[i];
					touch.push_back(tp);

					send_touch_event(tp, motion_event_action_button);

					break;
				}
			}
		} break;

		case ACTION_POINTER_UP: { // remove touch

			for (int i = 0; i < touch.size(); i++) {
				if (touch[i].id == p_pointer) {

					release_touch_event(touch[i], motion_event_action_button, true);
					touch.remove(i);

					break;
				}
			}
		} break;
	}
}

void DisplayServerAndroid::process_hover(int tool_type, int p_type, Point2 p_pos) {
	// https://developer.android.com/reference/android/view/MotionEvent.html#ACTION_HOVER_ENTER
	switch (p_type) {
		case 7: // hover move
		case 9: // hover enter
		case 10: { // hover exit
			Ref<InputEventMouseMotion> ev;
			ev.instance();
			ev->set_position(p_pos);
			ev->set_global_position(p_pos);
			ev->set_relative(p_pos - hover_prev_pos);
			if (is_mouse_pointer(tool_type)) {
				ev->set_button_mask(last_mouse_buttons_mask);
				last_mouse_position = p_pos;
			}
			InputFilter::get_singleton()->parse_input_event(ev);
			hover_prev_pos = p_pos;
		} break;
	}
}

void DisplayServerAndroid::process_double_tap(int tool_type, int android_motion_event_button_state, Point2 p_pos) {
	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_position(p_pos);
	ev->set_global_position(p_pos);
	ev->set_doubleclick(true);
	ev->set_pressed(false);

	if (is_mouse_pointer(tool_type)) {
		// Since this is a double tap, only one of the mouse button was active.
		int button_index = get_mouse_button_index(android_motion_event_button_state);
		ev->set_button_index(button_index);
		ev->set_button_mask(last_mouse_buttons_mask);
		last_mouse_position = p_pos;
	}

	InputFilter::get_singleton()->parse_input_event(ev);
}

void DisplayServerAndroid::process_scroll(int tool_type, Point2 start, Point2 end, Vector2 scroll_delta) {
	if (is_mouse_pointer(tool_type)) {
		// For a mouse scroll, `start` and `end` are the same.
		Point2 event_position = start;

		// Identify the BUTTON_WHEEL index.
		int button_index;
		float scroll_factor;
		if (abs(scroll_delta.x) > abs(scroll_delta.y)) {
			// Horizontal scroll
			button_index = (scroll_delta.x > 0.0f) ? BUTTON_WHEEL_RIGHT : BUTTON_WHEEL_LEFT;
			scroll_factor = scroll_delta.x;
		} else {
			// Vertical scroll
			button_index = (scroll_delta.y > 0.0f) ? BUTTON_WHEEL_UP : BUTTON_WHEEL_DOWN;
			scroll_factor = scroll_delta.y;
		}

		// We dispatch pressed and released events for mouse scrolling
		Ref<InputEventMouseButton> ev;
		ev.instance();
		ev->set_position(event_position);
		ev->set_global_position(event_position);
		ev->set_button_index(button_index);
		ev->set_factor(scroll_factor);
		ev->set_pressed(true);
		ev->set_button_mask(last_mouse_buttons_mask | button_index);
		InputFilter::get_singleton()->parse_input_event(ev);

		// Send release event
		ev->set_pressed(false);
		ev->set_button_mask(last_mouse_buttons_mask & ~button_index);
		InputFilter::get_singleton()->parse_input_event(ev);

		last_mouse_position = event_position;
	} else {
		Ref<InputEventPanGesture> ev;
		ev.instance();
		ev->set_position(end);
		ev->set_delta(scroll_delta);
		InputFilter::get_singleton()->parse_input_event(ev);
	}
}

void DisplayServerAndroid::process_accelerometer(const Vector3 &p_accelerometer) {
	InputFilter::get_singleton()->set_accelerometer(p_accelerometer);
}

void DisplayServerAndroid::process_gravity(const Vector3 &p_gravity) {
	InputFilter::get_singleton()->set_gravity(p_gravity);
}

void DisplayServerAndroid::process_magnetometer(const Vector3 &p_magnetometer) {
	InputFilter::get_singleton()->set_magnetometer(p_magnetometer);
}

void DisplayServerAndroid::process_gyroscope(const Vector3 &p_gyroscope) {
	InputFilter::get_singleton()->set_gyroscope(p_gyroscope);
}
