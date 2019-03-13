/*************************************************************************/
/*  os_android.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
	ERR_EXPLAIN("Invalid video driver index " + itos(p_driver));
	ERR_FAIL_V(NULL);
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
		FileAccess::make_default<FileAccessBufferedFA<FileAccessJAndroid> >(FileAccess::ACCESS_RESOURCES);
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
	input->set_fallback_mapping("Default Android Gamepad");

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

Error OS_Android::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	p_library_handle = dlopen(p_path.utf8().get_data(), RTLD_NOW);
	if (!p_library_handle) {
		ERR_EXPLAIN("Can't open dynamic library: " + p_path + ". Error: " + dlerror());
		ERR_FAIL_V(ERR_CANT_OPEN);
	}
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

	return Point2();
}

int OS_Android::get_mouse_button_state() const {

	return 0;
}

void OS_Android::set_window_title(const String &p_title) {
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

String OS_Android::get_name() {

	return "Android";
}

MainLoop *OS_Android::get_main_loop() const {

	return main_loop;
}

bool OS_Android::can_draw() const {

	return true; //always?
}

void OS_Android::set_cursor_shape(CursorShape p_shape) {

	//android really really really has no mouse.. how amazing..
}

void OS_Android::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
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

void OS_Android::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect) {

	if (godot_io_java->has_vk()) {
		godot_io_java->show_vk(p_existing_text);
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
