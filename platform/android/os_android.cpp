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

String _remove_symlink(const String &dir) {
	// Workaround for Android 6.0+ using a symlink.
	// Save the current directory.
	char current_dir_name[2048];
	getcwd(current_dir_name, 2048);
	// Change directory to the external data directory.
	chdir(dir.utf8().get_data());
	// Get the actual directory without the potential symlink.
	char dir_name_wihout_symlink[2048];
	getcwd(dir_name_wihout_symlink, 2048);
	// Convert back to a String.
	String dir_without_symlink(dir_name_wihout_symlink);
	// Restore original current directory.
	chdir(current_dir_name);
	return dir_without_symlink;
}

class AndroidLogger : public Logger {
public:
	virtual void logv(const char *p_format, va_list p_list, bool p_err) {
		__android_log_vprint(p_err ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "godot", p_format, p_list);
	}

	virtual ~AndroidLogger() {}
};

void OS_Android::alert(const String &p_alert, const String &p_title) {
	GodotJavaWrapper *godot_java = OS_Android::get_singleton()->get_godot_java();
	ERR_FAIL_COND(!godot_java);

	godot_java->alert(p_alert, p_title);
}

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

Error OS_Android::shell_open(String p_uri) {
	return godot_io_java->open_uri(p_uri);
}

String OS_Android::get_resource_dir() const {
	return "/"; //android has its own filesystem for resources inside the APK
}

String OS_Android::get_locale() const {
	String locale = godot_io_java->get_locale();
	if (!locale.is_empty()) {
		return locale;
	}

	return OS_Unix::get_locale();
}

String OS_Android::get_model_name() const {
	String model = godot_io_java->get_model();
	if (!model.is_empty())
		return model;

	return OS_Unix::get_model_name();
}

String OS_Android::get_data_path() const {
	return get_user_data_dir();
}

String OS_Android::get_user_data_dir() const {
	if (!data_dir_cache.is_empty())
		return data_dir_cache;

	String data_dir = godot_io_java->get_user_data_dir();
	if (!data_dir.is_empty()) {
		data_dir_cache = _remove_symlink(data_dir);
		return data_dir_cache;
	}
	return ".";
}

String OS_Android::get_cache_path() const {
	if (!cache_dir_cache.is_empty())
		return cache_dir_cache;

	String cache_dir = godot_io_java->get_cache_dir();
	if (!cache_dir.is_empty()) {
		cache_dir_cache = _remove_symlink(cache_dir);
		return cache_dir_cache;
	}
	return ".";
}

String OS_Android::get_unique_id() const {
	String unique_id = godot_io_java->get_unique_id();
	if (!unique_id.is_empty())
		return unique_id;

	return OS::get_unique_id();
}

String OS_Android::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
	return godot_io_java->get_system_dir(p_dir, p_shared_storage);
}

void OS_Android::set_display_size(const Size2i &p_size) {
	display_size = p_size;
}

Size2i OS_Android::get_display_size() const {
	return display_size;
}

void OS_Android::set_opengl_extensions(const char *p_gl_extensions) {
#if defined(GLES3_ENABLED)
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

#if defined(GLES3_ENABLED)
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
