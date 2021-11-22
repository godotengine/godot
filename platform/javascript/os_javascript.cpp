/*************************************************************************/
/*  os_javascript.cpp                                                    */
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

#include "os_javascript.h"

#include "core/debugger/engine_debugger.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"
#include "platform/javascript/display_server_javascript.h"

#include "modules/modules_enabled.gen.h" // For websocket.
#ifdef MODULE_WEBSOCKET_ENABLED
#include "modules/websocket/remote_debugger_peer_websocket.h"
#endif

#include <dlfcn.h>
#include <emscripten.h>
#include <stdlib.h>

#include "godot_js.h"

void OS_JavaScript::alert(const String &p_alert, const String &p_title) {
	godot_js_display_alert(p_alert.utf8().get_data());
}

// Lifecycle
void OS_JavaScript::initialize() {
	OS_Unix::initialize_core();
	DisplayServerJavaScript::register_javascript_driver();

#ifdef MODULE_WEBSOCKET_ENABLED
	EngineDebugger::register_uri_handler("ws://", RemoteDebuggerPeerWebSocket::create);
	EngineDebugger::register_uri_handler("wss://", RemoteDebuggerPeerWebSocket::create);
#endif
}

void OS_JavaScript::resume_audio() {
	AudioDriverJavaScript::resume();
}

void OS_JavaScript::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

MainLoop *OS_JavaScript::get_main_loop() const {
	return main_loop;
}

void OS_JavaScript::fs_sync_callback() {
	get_singleton()->idb_is_syncing = false;
}

bool OS_JavaScript::main_loop_iterate() {
	if (is_userfs_persistent() && idb_needs_sync && !idb_is_syncing) {
		idb_is_syncing = true;
		idb_needs_sync = false;
		godot_js_os_fs_sync(&fs_sync_callback);
	}

	DisplayServer::get_singleton()->process_events();

	return Main::iteration();
}

void OS_JavaScript::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}
	main_loop = nullptr;
}

void OS_JavaScript::finalize() {
	delete_main_loop();
	for (AudioDriverJavaScript *driver : audio_drivers) {
		memdelete(driver);
	}
	audio_drivers.clear();
}

// Miscellaneous

Error OS_JavaScript::execute(const String &p_path, const List<String> &p_arguments, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex) {
	return create_process(p_path, p_arguments);
}

Error OS_JavaScript::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id) {
	Array args;
	for (const String &E : p_arguments) {
		args.push_back(E);
	}
	String json_args = Variant(args).to_json_string();
	int failed = godot_js_os_execute(json_args.utf8().get_data());
	ERR_FAIL_COND_V_MSG(failed, ERR_UNAVAILABLE, "OS::execute() or create_process() must be implemented in JavaScript via 'engine.setOnExecute' if required.");
	return OK;
}

Error OS_JavaScript::kill(const ProcessID &p_pid) {
	ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "OS::kill() is not available on the HTML5 platform.");
}

int OS_JavaScript::get_process_id() const {
	ERR_FAIL_V_MSG(0, "OS::get_process_id() is not available on the HTML5 platform.");
}

int OS_JavaScript::get_processor_count() const {
	return godot_js_os_hw_concurrency_get();
}

bool OS_JavaScript::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "html5" || p_feature == "web") {
		return true;
	}

#ifdef JAVASCRIPT_EVAL_ENABLED
	if (p_feature == "javascript") {
		return true;
	}
#endif
#ifndef NO_THREADS
	if (p_feature == "threads") {
		return true;
	}
#endif
#if WASM_GDNATIVE
	if (p_feature == "wasm32") {
		return true;
	}
#endif

	return false;
}

String OS_JavaScript::get_executable_path() const {
	return OS::get_executable_path();
}

Error OS_JavaScript::shell_open(String p_uri) {
	// Open URI in a new tab, browser will deal with it by protocol.
	godot_js_os_shell_open(p_uri.utf8().get_data());
	return OK;
}

String OS_JavaScript::get_name() const {
	return "HTML5";
}

String OS_JavaScript::get_user_data_dir() const {
	return "/userfs";
};

String OS_JavaScript::get_cache_path() const {
	return "/home/web_user/.cache";
}

String OS_JavaScript::get_config_path() const {
	return "/home/web_user/.config";
}

String OS_JavaScript::get_data_path() const {
	return "/home/web_user/.local/share";
}

void OS_JavaScript::file_access_close_callback(const String &p_file, int p_flags) {
	OS_JavaScript *os = OS_JavaScript::get_singleton();
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

bool OS_JavaScript::is_userfs_persistent() const {
	return idb_available;
}

Error OS_JavaScript::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	String path = p_path.get_file();
	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_COND_V_MSG(!p_library_handle, ERR_CANT_OPEN, "Can't open dynamic library: " + p_path + ". Error: " + dlerror());
	return OK;
}

OS_JavaScript *OS_JavaScript::get_singleton() {
	return static_cast<OS_JavaScript *>(OS::get_singleton());
}

void OS_JavaScript::initialize_joypads() {
}

OS_JavaScript::OS_JavaScript() {
	char locale_ptr[16];
	godot_js_config_locale_get(locale_ptr, 16);
	setenv("LANG", locale_ptr, true);

	if (AudioDriverJavaScript::is_available()) {
#ifdef NO_THREADS
		audio_drivers.push_back(memnew(AudioDriverScriptProcessor));
#endif
		audio_drivers.push_back(memnew(AudioDriverWorklet));
	}
	for (int i = 0; i < audio_drivers.size(); i++) {
		AudioDriverManager::add_driver(audio_drivers[i]);
	}

	idb_available = godot_js_os_fs_is_persistent();

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	FileAccessUnix::close_notification_func = file_access_close_callback;
}
