/*************************************************************************/
/*  os_android.cpp                                                       */
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

#include "os_android.h"

#include "core/project_settings.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "file_access_android.h"
#include "main/main.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"

#include "dir_access_jandroid.h"
#include "file_access_jandroid.h"
#include "net_socket_android.h"

#include <android/input.h>
#include <core/os/keyboard.h>
#include <dlfcn.h>

#include "android_keys_utils.h"
#include "java_godot_io_wrapper.h"
#include "java_godot_wrapper.h"

class AndroidLogger : public Logger {
public:
	virtual void logv(const char *p_format, va_list p_list, bool p_err) {
		__android_log_vprint(p_err ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "godot", p_format, p_list);
	}

	virtual ~AndroidLogger() {}
};

int OS_Android::get_video_driver_count() const {

	return 2;
}

const char *OS_Android::get_video_driver_name(int p_driver) const {

	switch (p_driver) {
		case VIDEO_DRIVER_GLES3:
			return "GLES3";
		case VIDEO_DRIVER_GLES2:
			return "GLES2";
	}
	ERR_FAIL_V_MSG(NULL, "Invalid video driver index: " + itos(p_driver) + ".");
}
int OS_Android::get_audio_driver_count() const {

	return 1;
}

const char *OS_Android::get_audio_driver_name(int p_driver) const {

	return "Android";
}

void OS_Android::initialize_core() {

	OS_Unix::initialize_core();

	if (use_apk_expansion)
		FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_RESOURCES);
	else {
#ifdef USE_JAVA_FILE_ACCESS
		FileAccess::make_default<FileAccessJAndroid>(FileAccess::ACCESS_RESOURCES);
#else
		FileAccess::make_default<FileAccessAndroid>(FileAccess::ACCESS_RESOURCES);
#endif
	}
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_FILESYSTEM);
	if (use_apk_expansion)
		DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_RESOURCES);
	else
		DirAccess::make_default<DirAccessJAndroid>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_FILESYSTEM);

	NetSocketAndroid::make_default();
}

void OS_Android::set_opengl_extensions(const char *p_gl_extensions) {

	ERR_FAIL_COND(!p_gl_extensions);
	gl_extensions = p_gl_extensions;
}

int OS_Android::get_current_video_driver() const {
	return video_driver_index;
}

Error OS_Android::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	bool use_gl3 = godot_java->get_gles_version_code() >= 0x00030000;
	use_gl3 = use_gl3 && (GLOBAL_GET("rendering/quality/driver/driver_name") == "GLES3");
	bool gl_initialization_error = false;

	while (true) {
		if (use_gl3) {
			if (RasterizerGLES3::is_viable() == OK) {
				godot_java->gfx_init(false);
				RasterizerGLES3::register_config();
				RasterizerGLES3::make_current();
				break;
			} else {
				if (GLOBAL_GET("rendering/quality/driver/fallback_to_gles2")) {
					p_video_driver = VIDEO_DRIVER_GLES2;
					use_gl3 = false;
					continue;
				} else {
					gl_initialization_error = true;
					break;
				}
			}
		} else {
			if (RasterizerGLES2::is_viable() == OK) {
				godot_java->gfx_init(true);
				RasterizerGLES2::register_config();
				RasterizerGLES2::make_current();
				break;
			} else {
				gl_initialization_error = true;
				break;
			}
		}
	}

	if (gl_initialization_error) {
		OS::get_singleton()->alert("Your device does not support any of the supported OpenGL versions.\n"
								   "Please try updating your Android version.",
				"Unable to initialize Video driver");
		return ERR_UNAVAILABLE;
	}

	video_driver_index = p_video_driver;

	visual_server = memnew(VisualServerRaster);
	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, false));
	}

	visual_server->init();

	AudioDriverManager::initialize(p_audio_driver);

	input = memnew(InputDefault);
	input->set_fallback_mapping(godot_java->get_input_fallback_mapping());

	//power_manager = memnew(PowerAndroid);

	return OK;
}

void OS_Android::set_main_loop(MainLoop *p_main_loop) {

	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

void OS_Android::delete_main_loop() {

	memdelete(main_loop);
}

void OS_Android::finalize() {

	memdelete(input);
}

GodotJavaWrapper *OS_Android::get_godot_java() {
	return godot_java;
}

GodotIOJavaWrapper *OS_Android::get_godot_io_java() {
	return godot_io_java;
}

void OS_Android::alert(const String &p_alert, const String &p_title) {

	//print("ALERT: %s\n", p_alert.utf8().get_data());
	godot_java->alert(p_alert, p_title);
}

bool OS_Android::request_permission(const String &p_name) {

	return godot_java->request_permission(p_name);
}

bool OS_Android::request_permissions() {

	return godot_java->request_permissions();
}

Vector<String> OS_Android::get_granted_permissions() const {

	return godot_java->get_granted_permissions();
}

Error OS_Android::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	p_library_handle = dlopen(p_path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_COND_V_MSG(!p_library_handle, ERR_CANT_OPEN, "Can't open dynamic library: " + p_path + ", error: " + dlerror() + ".");
	return OK;
}

void OS_Android::set_mouse_show(bool p_show) {

	//android has no mouse...
}

void OS_Android::set_mouse_grab(bool p_grab) {

	//it really has no mouse...!
}

bool OS_Android::is_mouse_grab_enabled() const {

	//*sigh* technology has evolved so much since i was a kid..
	return false;
}

Point2 OS_Android::get_mouse_position() const {
	return hover_prev_pos;
}

int OS_Android::get_mouse_button_state() const {
	return buttons_state;
}

void OS_Android::set_window_title(const String &p_title) {
	//This queries/updates the currently connected devices/joypads
	//Set_window_title is called when initializing the main loop (main.cpp)
	//therefore this place is found to be suitable (I found no better).
	godot_java->init_input_devices();
}

void OS_Android::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
}

OS::VideoMode OS_Android::get_video_mode(int p_screen) const {

	return default_videomode;
}

void OS_Android::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {

	p_list->push_back(default_videomode);
}

void OS_Android::set_keep_screen_on(bool p_enabled) {
	OS::set_keep_screen_on(p_enabled);

	godot_java->set_keep_screen_on(p_enabled);
}

Size2 OS_Android::get_window_size() const {

	return Vector2(default_videomode.width, default_videomode.height);
}

Rect2 OS_Android::get_window_safe_area() const {
	int xywh[4];
	godot_io_java->get_window_safe_area(xywh);
	return Rect2(xywh[0], xywh[1], xywh[2], xywh[3]);
}

String OS_Android::get_name() const {

	return "Android";
}

MainLoop *OS_Android::get_main_loop() const {

	return main_loop;
}

bool OS_Android::can_draw() const {

	return true; //always?
}

void OS_Android::main_loop_begin() {

	if (main_loop)
		main_loop->init();
}

bool OS_Android::main_loop_iterate() {

	if (!main_loop)
		return false;
	return Main::iteration();
}

void OS_Android::main_loop_end() {

	if (main_loop)
		main_loop->finish();
}

void OS_Android::main_loop_focusout() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	audio_driver_android.set_pause(true);
}

void OS_Android::main_loop_focusin() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
	audio_driver_android.set_pause(false);
}

void OS_Android::process_joy_event(OS_Android::JoypadEvent p_event) {

	switch (p_event.type) {
		case JOY_EVENT_BUTTON:
			input->joy_button(p_event.device, p_event.index, p_event.pressed);
			break;
		case JOY_EVENT_AXIS:
			InputDefault::JoyAxis value;
			value.min = -1;
			value.value = p_event.value;
			input->joy_axis(p_event.device, p_event.index, value);
			break;
		case JOY_EVENT_HAT:
			input->joy_hat(p_event.device, p_event.hat);
			break;
		default:
			return;
	}
}

void OS_Android::_set_key_modifier_state(Ref<InputEventWithModifiers> ev) const {
	ev->set_shift(shift_mem);
	ev->set_alt(alt_mem);
	ev->set_metakey(meta_mem);
	ev->set_control(control_mem);
}

void OS_Android::process_event(Ref<InputEvent> p_event) {

	input->parse_input_event(p_event);
}

void OS_Android::process_key_event(int p_scancode, int p_unicode_char, bool p_pressed) {

	Ref<InputEventKey> ev;
	ev.instance();
	int val = p_unicode_char;
	unsigned int scancode = android_get_keysym(p_scancode);

	switch (scancode) {
		case KEY_SHIFT: {
			shift_mem = p_pressed;
		} break;
		case KEY_ALT: {
			alt_mem = p_pressed;
		} break;
		case KEY_CONTROL: {
			control_mem = p_pressed;
		} break;
		case KEY_META: {
			meta_mem = p_pressed;
		} break;
	}

	ev->set_scancode(scancode);
	ev->set_unicode(val);
	ev->set_pressed(p_pressed);

	_set_key_modifier_state(ev);

	if (val == '\n') {
		ev->set_scancode(KEY_ENTER);
	} else if (val == 61448) {
		ev->set_scancode(KEY_BACKSPACE);
		ev->set_unicode(KEY_BACKSPACE);
	} else if (val == 61453) {
		ev->set_scancode(KEY_ENTER);
		ev->set_unicode(KEY_ENTER);
	} else if (p_scancode == 4) {
		main_loop_request_go_back();
	}

	input->parse_input_event(ev);
}

void OS_Android::process_touch(int p_event, int p_pointer, const Vector<TouchPos> &p_points) {

	switch (p_event) {
		case AMOTION_EVENT_ACTION_DOWN: { //gesture begin
			if (touch.size()) {
				//end all if exist
				for (int i = 0; i < touch.size(); i++) {
					Ref<InputEventScreenTouch> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					input->parse_input_event(ev);
				}
			}

			touch.resize(p_points.size());
			for (int i = 0; i < p_points.size(); i++) {
				touch.write[i].id = p_points[i].id;
				touch.write[i].pos = p_points[i].pos;
			}

			//send touch
			for (int i = 0; i < touch.size(); i++) {
				Ref<InputEventScreenTouch> ev;
				ev.instance();
				ev->set_index(touch[i].id);
				ev->set_pressed(true);
				ev->set_position(touch[i].pos);
				input->parse_input_event(ev);
			}

		} break;
		case AMOTION_EVENT_ACTION_MOVE: { //motion
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

				Ref<InputEventScreenDrag> ev;
				ev.instance();
				ev->set_index(touch[i].id);
				ev->set_position(p_points[idx].pos);
				ev->set_relative(p_points[idx].pos - touch[i].pos);
				input->parse_input_event(ev);
				touch.write[i].pos = p_points[idx].pos;
			}

		} break;
		case AMOTION_EVENT_ACTION_CANCEL:
		case AMOTION_EVENT_ACTION_UP: { //release
			if (touch.size()) {
				//end all if exist
				for (int i = 0; i < touch.size(); i++) {
					Ref<InputEventScreenTouch> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					input->parse_input_event(ev);
				}
				touch.clear();
			}
		} break;
		case AMOTION_EVENT_ACTION_POINTER_DOWN: { // add touch
			for (int i = 0; i < p_points.size(); i++) {
				if (p_points[i].id == p_pointer) {
					TouchPos tp = p_points[i];
					touch.push_back(tp);

					Ref<InputEventScreenTouch> ev;
					ev.instance();

					ev->set_index(tp.id);
					ev->set_pressed(true);
					ev->set_position(tp.pos);
					input->parse_input_event(ev);

					break;
				}
			}
		} break;
		case AMOTION_EVENT_ACTION_POINTER_UP: { // remove touch
			for (int i = 0; i < touch.size(); i++) {
				if (touch[i].id == p_pointer) {
					Ref<InputEventScreenTouch> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					input->parse_input_event(ev);
					touch.remove(i);

					break;
				}
			}
		} break;
	}
}

void OS_Android::process_hover(int p_type, Point2 p_pos) {
	// https://developer.android.com/reference/android/view/MotionEvent.html#ACTION_HOVER_ENTER
	switch (p_type) {
		case AMOTION_EVENT_ACTION_HOVER_MOVE: // hover move
		case AMOTION_EVENT_ACTION_HOVER_ENTER: // hover enter
		case AMOTION_EVENT_ACTION_HOVER_EXIT: { // hover exit
			Ref<InputEventMouseMotion> ev;
			ev.instance();
			_set_key_modifier_state(ev);
			ev->set_position(p_pos);
			ev->set_global_position(p_pos);
			ev->set_relative(p_pos - hover_prev_pos);
			input->parse_input_event(ev);
			hover_prev_pos = p_pos;
		} break;
	}
}

void OS_Android::process_mouse_event(int event_action, int event_android_buttons_mask, Point2 event_pos, float event_vertical_factor, float event_horizontal_factor) {
	int event_buttons_mask = _android_button_mask_to_godot_button_mask(event_android_buttons_mask);
	switch (event_action) {
		case AMOTION_EVENT_ACTION_BUTTON_PRESS:
		case AMOTION_EVENT_ACTION_BUTTON_RELEASE: {
			Ref<InputEventMouseButton> ev;
			ev.instance();
			_set_key_modifier_state(ev);
			ev->set_position(event_pos);
			ev->set_global_position(event_pos);
			ev->set_pressed(event_action == AMOTION_EVENT_ACTION_BUTTON_PRESS);
			int changed_button_mask = buttons_state ^ event_buttons_mask;

			buttons_state = event_buttons_mask;

			ev->set_button_index(_button_index_from_mask(changed_button_mask));
			ev->set_button_mask(event_buttons_mask);
			input->parse_input_event(ev);
		} break;

		case AMOTION_EVENT_ACTION_MOVE: {
			Ref<InputEventMouseMotion> ev;
			ev.instance();
			_set_key_modifier_state(ev);
			ev->set_position(event_pos);
			ev->set_global_position(event_pos);
			ev->set_relative(event_pos - hover_prev_pos);
			ev->set_button_mask(event_buttons_mask);
			input->parse_input_event(ev);
			hover_prev_pos = event_pos;
		} break;
		case AMOTION_EVENT_ACTION_SCROLL: {
			Ref<InputEventMouseButton> ev;
			ev.instance();
			_set_key_modifier_state(ev);
			ev->set_position(event_pos);
			ev->set_global_position(event_pos);
			ev->set_pressed(true);
			buttons_state = event_buttons_mask;
			if (event_vertical_factor > 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_UP, event_vertical_factor);
			} else if (event_vertical_factor < 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_DOWN, -event_vertical_factor);
			}

			if (event_horizontal_factor > 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_RIGHT, event_horizontal_factor);
			} else if (event_horizontal_factor < 0) {
				_wheel_button_click(event_buttons_mask, ev, BUTTON_WHEEL_LEFT, -event_horizontal_factor);
			}
		} break;
	}
}

void OS_Android::_wheel_button_click(int event_buttons_mask, const Ref<InputEventMouseButton> &ev, int wheel_button, float factor) {
	Ref<InputEventMouseButton> evd = ev->duplicate();
	evd->set_button_index(wheel_button);
	evd->set_button_mask(event_buttons_mask ^ (1 << (wheel_button - 1)));
	evd->set_factor(factor);
	input->parse_input_event(evd);
	Ref<InputEventMouseButton> evdd = evd->duplicate();
	evdd->set_pressed(false);
	evdd->set_button_mask(event_buttons_mask);
	input->parse_input_event(evdd);
}

void OS_Android::process_double_tap(int event_android_button_mask, Point2 p_pos) {
	int event_button_mask = _android_button_mask_to_godot_button_mask(event_android_button_mask);
	Ref<InputEventMouseButton> ev;
	ev.instance();
	_set_key_modifier_state(ev);
	ev->set_position(p_pos);
	ev->set_global_position(p_pos);
	ev->set_pressed(event_button_mask != 0);
	ev->set_button_index(_button_index_from_mask(event_button_mask));
	ev->set_button_mask(event_button_mask);
	ev->set_doubleclick(true);
	input->parse_input_event(ev);
}

void OS_Android::process_scroll(Point2 p_pos) {
	Ref<InputEventPanGesture> ev;
	ev.instance();
	_set_key_modifier_state(ev);
	ev->set_position(p_pos);
	ev->set_delta(p_pos - scroll_prev_pos);
	input->parse_input_event(ev);
	scroll_prev_pos = p_pos;
}

int OS_Android::_button_index_from_mask(int button_mask) {
	switch (button_mask) {
		case BUTTON_MASK_LEFT:
			return BUTTON_LEFT;
		case BUTTON_MASK_RIGHT:
			return BUTTON_RIGHT;
		case BUTTON_MASK_MIDDLE:
			return BUTTON_MIDDLE;
		case BUTTON_MASK_XBUTTON1:
			return BUTTON_XBUTTON1;
		case BUTTON_MASK_XBUTTON2:
			return BUTTON_XBUTTON2;
		default:
			return 0;
	}
}

int OS_Android::_android_button_mask_to_godot_button_mask(int android_button_mask) {
	int godot_button_mask = 0;
	if (android_button_mask & AMOTION_EVENT_BUTTON_PRIMARY) {
		godot_button_mask |= BUTTON_MASK_LEFT;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_SECONDARY) {
		godot_button_mask |= BUTTON_MASK_RIGHT;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_TERTIARY) {
		godot_button_mask |= BUTTON_MASK_MIDDLE;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_BACK) {
		godot_button_mask |= BUTTON_MASK_XBUTTON1;
	}
	if (android_button_mask & AMOTION_EVENT_BUTTON_SECONDARY) {
		godot_button_mask |= BUTTON_MASK_XBUTTON2;
	}

	return godot_button_mask;
}

void OS_Android::process_accelerometer(const Vector3 &p_accelerometer) {

	input->set_accelerometer(p_accelerometer);
}

void OS_Android::process_gravity(const Vector3 &p_gravity) {

	input->set_gravity(p_gravity);
}

void OS_Android::process_magnetometer(const Vector3 &p_magnetometer) {

	input->set_magnetometer(p_magnetometer);
}

void OS_Android::process_gyroscope(const Vector3 &p_gyroscope) {

	input->set_gyroscope(p_gyroscope);
}

bool OS_Android::has_touchscreen_ui_hint() const {

	return true;
}

bool OS_Android::has_virtual_keyboard() const {

	return true;
}

int OS_Android::get_virtual_keyboard_height() const {
	return godot_io_java->get_vk_height();

	// ERR_PRINT("Cannot obtain virtual keyboard height.");
	// return 0;
}

void OS_Android::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect, bool p_multiline, int p_max_input_length, int p_cursor_start, int p_cursor_end) {

	if (godot_io_java->has_vk()) {
		godot_io_java->show_vk(p_existing_text, p_multiline, p_max_input_length, p_cursor_start, p_cursor_end);
	} else {

		ERR_PRINT("Virtual keyboard not available");
	};
}

void OS_Android::hide_virtual_keyboard() {

	if (godot_io_java->has_vk()) {

		godot_io_java->hide_vk();
	} else {

		ERR_PRINT("Virtual keyboard not available");
	};
}

void OS_Android::init_video_mode(int p_video_width, int p_video_height) {

	default_videomode.width = p_video_width;
	default_videomode.height = p_video_height;
	default_videomode.fullscreen = true;
	default_videomode.resizable = false;
}

void OS_Android::main_loop_request_go_back() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_GO_BACK_REQUEST);
}

void OS_Android::set_display_size(Size2 p_size) {

	default_videomode.width = p_size.x;
	default_videomode.height = p_size.y;
}

Error OS_Android::shell_open(String p_uri) {

	return godot_io_java->open_uri(p_uri);
}

String OS_Android::get_resource_dir() const {

	return "/"; //android has its own filesystem for resources inside the APK
}

String OS_Android::get_locale() const {

	String locale = godot_io_java->get_locale();
	if (locale != "") {
		return locale;
	}

	return OS_Unix::get_locale();
}

void OS_Android::set_clipboard(const String &p_text) {

	// DO we really need the fallback to OS_Unix here?!
	if (godot_java->has_set_clipboard()) {
		godot_java->set_clipboard(p_text);
	} else {
		OS_Unix::set_clipboard(p_text);
	}
}

String OS_Android::get_clipboard() const {

	// DO we really need the fallback to OS_Unix here?!
	if (godot_java->has_get_clipboard()) {
		return godot_java->get_clipboard();
	}

	return OS_Unix::get_clipboard();
}

String OS_Android::get_model_name() const {

	String model = godot_io_java->get_model();
	if (model != "")
		return model;

	return OS_Unix::get_model_name();
}

int OS_Android::get_screen_dpi(int p_screen) const {

	return godot_io_java->get_screen_dpi();
}

String OS_Android::get_user_data_dir() const {

	if (data_dir_cache != String())
		return data_dir_cache;

	String data_dir = godot_io_java->get_user_data_dir();
	if (data_dir != "") {

		//store current dir
		char real_current_dir_name[2048];
		getcwd(real_current_dir_name, 2048);

		//go to data dir
		chdir(data_dir.utf8().get_data());

		//get actual data dir, so we resolve potential symlink (Android 6.0+ seems to use symlink)
		char data_current_dir_name[2048];
		getcwd(data_current_dir_name, 2048);

		//cache by parsing utf8
		data_dir_cache.parse_utf8(data_current_dir_name);

		//restore original dir so we don't mess things up
		chdir(real_current_dir_name);

		return data_dir_cache;
	}

	return ".";
}

void OS_Android::set_screen_orientation(ScreenOrientation p_orientation) {

	godot_io_java->set_screen_orientation(p_orientation);
}

OS::ScreenOrientation OS_Android::get_screen_orientation() const {
	const int orientation = godot_io_java->get_screen_orientation();
	return OS::ScreenOrientation(orientation);
}

String OS_Android::get_unique_id() const {

	String unique_id = godot_io_java->get_unique_id();
	if (unique_id != "")
		return unique_id;

	return OS::get_unique_id();
}

Error OS_Android::native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {
	// FIXME: Add support for volume, audio and subtitle tracks

	godot_io_java->play_video(p_path);
	return OK;
}

bool OS_Android::native_video_is_playing() const {

	return godot_io_java->is_video_playing();
}

void OS_Android::native_video_pause() {

	godot_io_java->pause_video();
}

String OS_Android::get_system_dir(SystemDir p_dir) const {

	return godot_io_java->get_system_dir(p_dir);
}

void OS_Android::native_video_stop() {

	godot_io_java->stop_video();
}

void OS_Android::set_context_is_16_bits(bool p_is_16) {

	//use_16bits_fbo = p_is_16;
	//if (rasterizer)
	//	rasterizer->set_force_16_bits_fbo(p_is_16);
}

void OS_Android::joy_connection_changed(int p_device, bool p_connected, String p_name) {
	return input->joy_connection_changed(p_device, p_connected, p_name, "");
}

bool OS_Android::is_joy_known(int p_device) {
	return input->is_joy_mapped(p_device);
}

String OS_Android::get_joy_guid(int p_device) const {
	return input->get_joy_guid_remapped(p_device);
}

void OS_Android::vibrate_handheld(int p_duration_ms) {
	godot_java->vibrate(p_duration_ms);
}

bool OS_Android::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "mobile") {
		//TODO support etc2 only if GLES3 driver is selected
		return true;
	}
#if defined(__aarch64__)
	if (p_feature == "arm64-v8a") {
		return true;
	}
#elif defined(__ARM_ARCH_7A__)
	if (p_feature == "armeabi-v7a" || p_feature == "armeabi") {
		return true;
	}
#elif defined(__arm__)
	if (p_feature == "armeabi") {
		return true;
	}
#endif
	return false;
}

OS_Android::OS_Android(GodotJavaWrapper *p_godot_java, GodotIOJavaWrapper *p_godot_io_java, bool p_use_apk_expansion) {

	use_apk_expansion = p_use_apk_expansion;
	default_videomode.width = 800;
	default_videomode.height = 600;
	default_videomode.fullscreen = true;
	default_videomode.resizable = false;
	buttons_state = 0;

	main_loop = NULL;
	gl_extensions = NULL;
	//rasterizer = NULL;
	use_gl2 = false;

	godot_java = p_godot_java;
	godot_io_java = p_godot_io_java;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(AndroidLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	AudioDriverManager::add_driver(&audio_driver_android);
}

OS_Android::~OS_Android() {
}
