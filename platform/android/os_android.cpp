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

#include "core/io/file_access_buffered_fa.h"
#include "core/project_settings.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "file_access_android.h"
#include "main/main.h"
#include "platform/android/display_server_android.h"

#include "dir_access_jandroid.h"
#include "file_access_jandroid.h"
#include "net_socket_android.h"

#include <dlfcn.h>

#include "java_godot_io_wrapper.h"
#include "java_godot_wrapper.h"

class AndroidLogger : public Logger {
public:
	virtual void logv(const char *p_format, va_list p_list, bool p_err) {
		__android_log_vprint(p_err ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "godot", p_format, p_list);
	}

	virtual ~AndroidLogger() {}
};

void OS_Android::initialize_core() {
	OS_Unix::initialize_core();

	if (use_apk_expansion)
		FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_RESOURCES);
	else {
#ifdef USE_JAVA_FILE_ACCESS
		FileAccess::make_default<FileAccessBufferedFA<FileAccessJAndroid>>(FileAccess::ACCESS_RESOURCES);
#else
		//FileAccess::make_default<FileAccessBufferedFA<FileAccessAndroid> >(FileAccess::ACCESS_RESOURCES);
		FileAccess::make_default<FileAccessAndroid>(FileAccess::ACCESS_RESOURCES);
#endif
	}
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_FILESYSTEM);
	//FileAccessBufferedFA<FileAccessUnix>::make_default();
	if (use_apk_expansion)
		DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_RESOURCES);
	else
		DirAccess::make_default<DirAccessJAndroid>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_FILESYSTEM);

	NetSocketAndroid::make_default();
}

void OS_Android::initialize() {
	initialize_core();
}

void OS_Android::initialize_joypads() {
	Input::get_singleton()->set_fallback_mapping(godot_java->get_input_fallback_mapping());

	// This queries/updates the currently connected devices/joypads.
	godot_java->init_input_devices();
}

void OS_Android::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

void OS_Android::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
		main_loop = nullptr;
	}
}

void OS_Android::finalize() {
}

OS_Android *OS_Android::get_singleton() {
	return (OS_Android *)OS::get_singleton();
}

GodotJavaWrapper *OS_Android::get_godot_java() {
	return godot_java;
}

GodotIOJavaWrapper *OS_Android::get_godot_io_java() {
	return godot_io_java;
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

String OS_Android::get_name() const {
	return "Android";
}

MainLoop *OS_Android::get_main_loop() const {
	return main_loop;
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
	DisplayServerAndroid::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
	audio_driver_android.set_pause(true);
}

void OS_Android::main_loop_focusin() {
	DisplayServerAndroid::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_IN);
	audio_driver_android.set_pause(false);
}

<<<<<<< HEAD
=======
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

void OS_Android::process_event(Ref<InputEvent> p_event) {

	input->parse_input_event(p_event);
}

void OS_Android::process_touch(int p_what, int p_pointer, const Vector<TouchPos> &p_points) {

	switch (p_what) {
		case 0: { //gesture begin

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
		case 1: { //motion

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
		case 2: { //release

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
		case 3: { // add touch

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
		case 4: { // remove touch

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
		case 7: // hover move
		case 9: // hover enter
		case 10: { // hover exit
			Ref<InputEventMouseMotion> ev;
			ev.instance();
			ev->set_position(p_pos);
			ev->set_global_position(p_pos);
			ev->set_relative(p_pos - hover_prev_pos);
			input->parse_input_event(ev);
			hover_prev_pos = p_pos;
		} break;
	}
}

void OS_Android::process_double_tap(Point2 p_pos) {
	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_position(p_pos);
	ev->set_global_position(p_pos);
	ev->set_pressed(false);
	ev->set_doubleclick(true);
	input->parse_input_event(ev);
}

void OS_Android::process_scroll(Point2 p_pos) {
	Ref<InputEventPanGesture> ev;
	ev.instance();
	ev->set_position(p_pos);
	ev->set_delta(p_pos - scroll_prev_pos);
	input->parse_input_event(ev);
	scroll_prev_pos = p_pos;
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

void OS_Android::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect, int p_max_input_length, int p_cursor_start, int p_cursor_end) {

	if (godot_io_java->has_vk()) {
		godot_io_java->show_vk(p_existing_text, p_max_input_length, p_cursor_start, p_cursor_end);
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

>>>>>>> master
void OS_Android::main_loop_request_go_back() {
	DisplayServerAndroid::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_GO_BACK_REQUEST);
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

String OS_Android::get_model_name() const {
	String model = godot_io_java->get_model();
	if (model != "")
		return model;

	return OS_Unix::get_model_name();
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

String OS_Android::get_unique_id() const {
	String unique_id = godot_io_java->get_unique_id();
	if (unique_id != "")
		return unique_id;

	return OS::get_unique_id();
}

String OS_Android::get_system_dir(SystemDir p_dir) const {
	return godot_io_java->get_system_dir(p_dir);
}

void OS_Android::set_display_size(const Size2i &p_size) {
	display_size = p_size;
}

Size2i OS_Android::get_display_size() const {
	return display_size;
}

void OS_Android::set_context_is_16_bits(bool p_is_16) {
#if defined(OPENGL_ENABLED)
	//use_16bits_fbo = p_is_16;
	//if (rasterizer)
	//	rasterizer->set_force_16_bits_fbo(p_is_16);
#endif
}

void OS_Android::set_opengl_extensions(const char *p_gl_extensions) {
#if defined(OPENGL_ENABLED)
	ERR_FAIL_COND(!p_gl_extensions);
	gl_extensions = p_gl_extensions;
#endif
}

void OS_Android::set_native_window(ANativeWindow *p_native_window) {
#if defined(VULKAN_ENABLED)
	native_window = p_native_window;
#endif
}

ANativeWindow *OS_Android::get_native_window() const {
#if defined(VULKAN_ENABLED)
	return native_window;
#else
	return nullptr;
#endif
}

void OS_Android::vibrate_handheld(int p_duration_ms) {
	godot_java->vibrate(p_duration_ms);
}

bool OS_Android::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "mobile") {
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
	display_size.width = 800;
	display_size.height = 600;

	use_apk_expansion = p_use_apk_expansion;

	main_loop = nullptr;

#if defined(OPENGL_ENABLED)
	gl_extensions = nullptr;
	use_gl2 = false;
#endif

#if defined(VULKAN_ENABLED)
	native_window = nullptr;
#endif

	godot_java = p_godot_java;
	godot_io_java = p_godot_io_java;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(AndroidLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	AudioDriverManager::add_driver(&audio_driver_android);

	DisplayServerAndroid::register_android_driver();
}

OS_Android::~OS_Android() {
}
