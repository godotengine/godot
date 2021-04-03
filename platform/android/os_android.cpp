/*************************************************************************/
/*  os_android.cpp                                                       */
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

#include "os_android.h"

#include "core/config/project_settings.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"
#include "platform/android/display_server_android.h"

#include "dir_access_jandroid.h"
#include "file_access_android.h"
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
		FileAccess::make_default<FileAccessAndroid>(FileAccess::ACCESS_RESOURCES);
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
		main_loop->initialize();
}

bool OS_Android::main_loop_iterate() {
	if (!main_loop)
		return false;
	DisplayServerAndroid::get_singleton()->process_events();
	return Main::iteration();
}

void OS_Android::main_loop_end() {
	if (main_loop)
		main_loop->finalize();
}

void OS_Android::main_loop_focusout() {
	DisplayServerAndroid::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_OUT);
	audio_driver_android.set_pause(true);
}

void OS_Android::main_loop_focusin() {
	DisplayServerAndroid::get_singleton()->send_window_event(DisplayServer::WINDOW_EVENT_FOCUS_IN);
	audio_driver_android.set_pause(false);
}

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

void OS_Android::start_gps_tracker(int p_time_ms, int p_distance) {
	godot_java->start_gps_tracker(p_time_ms, p_distance);
}

void OS_Android::stop_gps_tracker() {
	godot_java->stop_gps_tracker();
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
	use_16bits_fbo = false;
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
