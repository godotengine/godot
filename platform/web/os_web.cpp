/**************************************************************************/
/*  os_web.cpp                                                            */
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

#include "os_web.h"

#include "api/javascript_bridge_singleton.h"
#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/config_file.h"
#include "core/io/file_access.h"
#include "core/io/file_access_pack.h"
#include "core/io/json.h"
#include "core/os/main_loop.h"
#include "core/profiling/profiling.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"

#include "modules/modules_enabled.gen.h" // For websocket.

#include "display_server_web.h"
#include "godot_js.h"
#include "ip_web.h"
#include "net_socket_web.h"

#include <dlfcn.h>
#include <emscripten.h>
#include <cstdlib>

void OS_Web::alert(const String &p_alert, const String &p_title) {
	godot_js_display_alert(p_alert.utf8().get_data());
}

// Lifecycle
void OS_Web::initialize() {
	OS_Unix::initialize_core();
	IPWeb::make_default();
	NetSocketWeb::make_default();
	DisplayServerWeb::register_web_driver();
}

void OS_Web::resume_audio() {
	AudioDriverWeb::resume();
}

void OS_Web::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

MainLoop *OS_Web::get_main_loop() const {
	return main_loop;
}

void OS_Web::fs_sync_callback() {
	get_singleton()->idb_is_syncing = false;
}

bool OS_Web::main_loop_iterate() {
	GodotProfileFrameMark;
	GodotProfileZone("OS_Web::main_loop_iterate");
	if (is_userfs_persistent() && idb_needs_sync && !idb_is_syncing) {
		idb_is_syncing = true;
		idb_needs_sync = false;
		godot_js_os_fs_sync(&fs_sync_callback);
	}

	DisplayServer::get_singleton()->process_events();

	return Main::iteration();
}

void OS_Web::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}
	main_loop = nullptr;
}

void OS_Web::finalize() {
	delete_main_loop();
	for (AudioDriverWeb *driver : audio_drivers) {
		memdelete(driver);
	}
	audio_drivers.clear();
}

// Miscellaneous

Error OS_Web::execute(const String &p_path, const List<String> &p_arguments, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex, bool p_open_console) {
	return create_process(p_path, p_arguments);
}

Dictionary OS_Web::execute_with_pipe(const String &p_path, const List<String> &p_arguments, bool p_blocking) {
	ERR_FAIL_V_MSG(Dictionary(), "OS::execute_with_pipe is not available on the Web platform.");
}

Error OS_Web::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id, bool p_open_console) {
	Array args;
	for (const String &E : p_arguments) {
		args.push_back(E);
	}
	String json_args = Variant(args).to_json_string();
	int failed = godot_js_os_execute(json_args.utf8().get_data());
	ERR_FAIL_COND_V_MSG(failed, ERR_UNAVAILABLE, "OS::execute() or create_process() must be implemented in Web via 'engine.setOnExecute' if required.");
	return OK;
}

Error OS_Web::kill(const ProcessID &p_pid) {
	ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "OS::kill() is not available on the Web platform.");
}

int OS_Web::get_process_id() const {
	return 0;
}

bool OS_Web::is_process_running(const ProcessID &p_pid) const {
	return false;
}

int OS_Web::get_process_exit_code(const ProcessID &p_pid) const {
	return -1;
}

int OS_Web::get_processor_count() const {
	return godot_js_os_hw_concurrency_get();
}

String OS_Web::get_unique_id() const {
	ERR_FAIL_V_MSG("", "OS::get_unique_id() is not available on the Web platform.");
}

int OS_Web::get_default_thread_pool_size() const {
#ifdef THREADS_ENABLED
	return godot_js_os_thread_pool_size_get();
#else // No threads.
	return 1;
#endif
}

Error OS_Web::async_pck_install_file(const String &p_path) const {
	String path = ResourceUID::ensure_path(p_path);
	ERR_FAIL_COND_V_MSG(!path.begins_with("res://"), ERR_FILE_BAD_PATH, vformat(R"*(Not able to install "%s" from a ".asyncpck".)*", path));

	if (FileAccess::exists(path)) {
		return OK;
	}

	Error err;
	String pck_path = async_pck_get_async_pck_path(path, &err);
	if (err != OK) {
		return err;
	}

	String pck_base_dir = pck_path.get_base_dir();
	err = static_cast<Error>(godot_js_os_asyncpck_install_file(pck_base_dir.utf8().get_data(), path.utf8().get_data()));
	return err;
}

Dictionary OS_Web::async_pck_install_file_get_status(const String &p_path) const {
	String path = ResourceUID::ensure_path(p_path);

	if (FileAccess::exists(path)) {
		Dictionary status;
		status["files"] = Dictionary();
		status["size"] = 0;
		status["progress"] = 0;
		status["progress_ratio"] = 1;
		status["status"] = "STATUS_INSTALLED";
		return status;
		return Dictionary();
	}

	Error err;
	String pck_path = async_pck_get_async_pck_path(path, &err);
	if (err != OK) {
		Dictionary status;
		status["files"] = Dictionary();
		status["size"] = 0;
		status["progress"] = 0;
		status["progress_ratio"] = 1;
		status["status"] = "STATUS_ERROR";
		status["errors"] = Dictionary();
		return status;
	}

	String pck_base_dir = pck_path.get_base_dir();

	int32_t status_text_length = 0;
	char *status_text_ptr = godot_js_os_asyncpck_install_file_get_status(pck_base_dir.utf8().get_data(), path.utf8().get_data(), &status_text_length);
	if (status_text_ptr == nullptr || status_text_length <= 0) {
		Dictionary status;
		status["files"] = Dictionary();
		status["size"] = 0;
		status["progress"] = 0;
		status["progress_ratio"] = 1;
		status["status"] = "STATUS_ERROR";
		status["errors"] = Dictionary();
		return status;
	}
	Dictionary status = JSON::parse_string(String::utf8(status_text_ptr, status_text_length));
	return status;
}

bool OS_Web::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "web") {
		return true;
	}

	if (p_feature == "web_extensions") {
#ifdef WEB_DLINK_ENABLED
		return true;
#else
		return false;
#endif
	}
	if (p_feature == "web_noextensions") {
#ifdef WEB_DLINK_ENABLED
		return false;
#else
		return true;
#endif
	}

	if (godot_js_os_has_feature(p_feature.utf8().get_data())) {
		return true;
	}
	return false;
}

String OS_Web::get_executable_path() const {
	return OS::get_executable_path();
}

Error OS_Web::shell_open(const String &p_uri) {
	// Open URI in a new tab, browser will deal with it by protocol.
	godot_js_os_shell_open(p_uri.utf8().get_data());
	return OK;
}

String OS_Web::get_name() const {
	return "Web";
}

void OS_Web::add_frame_delay(bool p_can_draw, bool p_wake_for_events) {
#ifndef PROXY_TO_PTHREAD_ENABLED
	OS::add_frame_delay(p_can_draw, p_wake_for_events);
#endif
}

void OS_Web::vibrate_handheld(int p_duration_ms, float p_amplitude) {
	godot_js_input_vibrate_handheld(p_duration_ms);
}

String OS_Web::get_user_data_dir(const String &p_user_dir) const {
	String userfs = "/userfs";
	return userfs.path_join(p_user_dir).replace_char('\\', '/');
}

String OS_Web::get_cache_path() const {
	return "/home/web_user/.cache";
}

String OS_Web::get_config_path() const {
	return "/home/web_user/.config";
}

String OS_Web::get_data_path() const {
	return "/home/web_user/.local/share";
}

void OS_Web::file_access_close_callback(const String &p_file, int p_flags) {
	OS_Web *os = OS_Web::get_singleton();
	if (!(os->is_userfs_persistent() && (p_flags & FileAccess::WRITE))) {
		return; // FS persistence is not working or we are not writing.
	}
	bool is_file_persistent = p_file.begins_with("/userfs");
#ifdef TOOLS_ENABLED
	// Hack for editor persistence (can we track).
	is_file_persistent = is_file_persistent || p_file.begins_with("/home/web_user/");
#endif
	if (is_file_persistent) {
		os->idb_needs_sync = true;
	}
}

void OS_Web::dir_access_remove_callback(const String &p_file) {
	OS_Web *os = OS_Web::get_singleton();
	bool is_file_persistent = p_file.begins_with("/userfs");
#ifdef TOOLS_ENABLED
	// Hack for editor persistence (can we track).
	is_file_persistent = is_file_persistent || p_file.begins_with("/home/web_user/");
#endif
	if (is_file_persistent) {
		os->idb_needs_sync = true;
	}
}

void OS_Web::update_pwa_state_callback() {
	if (OS_Web::get_singleton()) {
		OS_Web::get_singleton()->pwa_is_waiting = true;
	}
	if (JavaScriptBridge::get_singleton()) {
		JavaScriptBridge::get_singleton()->emit_signal("pwa_update_available");
	}
}

char *OS_Web::get_config_as_json_callback(const char *p_config_file_data_ptr) {
	ERR_FAIL_NULL_V(p_config_file_data_ptr, nullptr);
	String config_file_data_as_string = String::utf8(p_config_file_data_ptr);
	ERR_FAIL_COND_V(config_file_data_as_string.is_empty(), nullptr);

	Ref<ConfigFile> config_file;
	config_file.instantiate();
	config_file->parse(config_file_data_as_string);

	Dictionary json_config_file_data;
	for (const String &section : config_file->get_sections()) {
		Dictionary sectionData;
		for (const String &key : config_file->get_section_keys(section)) {
			sectionData[key] = config_file->get_value(section, key);
		}
		json_config_file_data[section] = sectionData;
	}
	String json_config_file_data_as_string = JSON::stringify(json_config_file_data, String(" ").repeat(2));
	size_t json_config_file_data_len = json_config_file_data_as_string.utf8().size();
	char *returned_json_config_file_data_ptr = (char *)memalloc(sizeof(char) * json_config_file_data_len);
	ERR_FAIL_NULL_V(returned_json_config_file_data_ptr, nullptr);
	memcpy(returned_json_config_file_data_ptr, json_config_file_data_as_string.utf8().ptr(), json_config_file_data_len);
	godot_js_string *js_string = (godot_js_string *)memalloc(sizeof(godot_js_string));
	js_string->length = json_config_file_data_len;
	js_string->data = returned_json_config_file_data_ptr;
	ERR_FAIL_COND_V(js_string->data != returned_json_config_file_data_ptr, nullptr);
	return returned_json_config_file_data_ptr;
}

void OS_Web::force_fs_sync() {
	if (is_userfs_persistent()) {
		idb_needs_sync = true;
	}
}

Error OS_Web::pwa_update() {
	return godot_js_pwa_update() ? FAILED : OK;
}

bool OS_Web::is_userfs_persistent() const {
	return idb_available;
}

Error OS_Web::open_dynamic_library(const String &p_path, void *&p_library_handle, GDExtensionData *p_data) {
	String path = p_path.get_file();
	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_NULL_V_MSG(p_library_handle, ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, dlerror()));

	if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
		*p_data->r_resolved_path = path;
	}

	return OK;
}

OS_Web *OS_Web::get_singleton() {
	return static_cast<OS_Web *>(OS::get_singleton());
}

void OS_Web::initialize_joypads() {
}

OS_Web::OS_Web() {
	char locale_ptr[16];
	godot_js_config_locale_get(locale_ptr, 16);
	setenv("LANG", locale_ptr, true);

	godot_js_pwa_cb(&OS_Web::update_pwa_state_callback);
	godot_js_runtime_set_get_config_file_as_json_cb(&OS_Web::get_config_as_json_callback);

	if (AudioDriverWeb::is_available()) {
		audio_drivers.push_back(memnew(AudioDriverWorklet));
		audio_drivers.push_back(memnew(AudioDriverScriptProcessor));
	}
	for (AudioDriverWeb *audio_driver : audio_drivers) {
		AudioDriverManager::add_driver(audio_driver);
	}

	idb_available = godot_js_os_fs_is_persistent();

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	FileAccessUnix::close_notification_func = file_access_close_callback;
	DirAccessUnix::remove_notification_func = dir_access_remove_callback;
}
