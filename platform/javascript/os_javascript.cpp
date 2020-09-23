/*************************************************************************/
/*  os_javascript.cpp                                                    */
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

#include "os_javascript.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/file_access_buffered_fa.h"
#include "core/io/json.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"
#include "modules/modules_enabled.gen.h"
#include "platform/javascript/display_server_javascript.h"

#ifdef MODULE_WEBSOCKET_ENABLED
#include "modules/websocket/remote_debugger_peer_websocket.h"
#endif

#include <emscripten.h>
#include <stdlib.h>

// Lifecycle
void OS_JavaScript::initialize() {
	OS_Unix::initialize_core();
	FileAccess::make_default<FileAccessBufferedFA<FileAccessUnix>>(FileAccess::ACCESS_RESOURCES);
	DisplayServerJavaScript::register_javascript_driver();

#ifdef MODULE_WEBSOCKET_ENABLED
	EngineDebugger::register_uri_handler("ws://", RemoteDebuggerPeerWebSocket::create);
	EngineDebugger::register_uri_handler("wss://", RemoteDebuggerPeerWebSocket::create);
#endif
}

void OS_JavaScript::resume_audio() {
	if (audio_driver_javascript) {
		audio_driver_javascript->resume();
	}
}

void OS_JavaScript::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

MainLoop *OS_JavaScript::get_main_loop() const {
	return main_loop;
}

extern "C" EMSCRIPTEN_KEEPALIVE void _idb_synced() {
	OS_JavaScript::get_singleton()->idb_is_syncing = false;
}

bool OS_JavaScript::main_loop_iterate() {
	if (is_userfs_persistent() && idb_needs_sync && !idb_is_syncing) {
		idb_is_syncing = true;
		idb_needs_sync = false;
		/* clang-format off */
		EM_ASM({
			FS.syncfs(function(error) {
				if (error) {
					err('Failed to save IDB file system: ' + error.message);
				}
				ccall("_idb_synced", 'void', [], []);
			});
		});
		/* clang-format on */
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

void OS_JavaScript::finalize_async() {
	finalizing = true;
	if (audio_driver_javascript) {
		audio_driver_javascript->finish_async();
	}
}

void OS_JavaScript::finalize() {
	delete_main_loop();
	if (audio_driver_javascript) {
		memdelete(audio_driver_javascript);
		audio_driver_javascript = nullptr;
	}
}

// Miscellaneous

Error OS_JavaScript::execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex) {
	Array args;
	for (const List<String>::Element *E = p_arguments.front(); E; E = E->next()) {
		args.push_back(E->get());
	}
	String json_args = JSON::print(args);
	/* clang-format off */
	int failed = EM_ASM_INT({
		const json_args = UTF8ToString($0);
		const args = JSON.parse(json_args);
		if (Module["onExecute"]) {
			Module["onExecute"](args);
			return 0;
		}
		return 1;
	}, json_args.utf8().get_data());
	/* clang-format on */
	ERR_FAIL_COND_V_MSG(failed, ERR_UNAVAILABLE, "OS::execute() must be implemented in JavaScript via 'engine.setOnExecute' if required.");
	return OK;
}

Error OS_JavaScript::kill(const ProcessID &p_pid) {
	ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "OS::kill() is not available on the HTML5 platform.");
}

int OS_JavaScript::get_process_id() const {
	ERR_FAIL_V_MSG(0, "OS::get_process_id() is not available on the HTML5 platform.");
}

bool OS_JavaScript::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "HTML5" || p_feature == "web")
		return true;

#ifdef JAVASCRIPT_EVAL_ENABLED
	if (p_feature == "JavaScript")
		return true;
#endif

	return false;
}

String OS_JavaScript::get_executable_path() const {
	return OS::get_executable_path();
}

Error OS_JavaScript::shell_open(String p_uri) {
	// Open URI in a new tab, browser will deal with it by protocol.
	/* clang-format off */
	EM_ASM_({
		window.open(UTF8ToString($0), '_blank');
	}, p_uri.utf8().get_data());
	/* clang-format on */
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

void OS_JavaScript::set_idb_available(bool p_idb_available) {
	idb_available = p_idb_available;
}

bool OS_JavaScript::is_userfs_persistent() const {
	return idb_available;
}

OS_JavaScript *OS_JavaScript::get_singleton() {
	return static_cast<OS_JavaScript *>(OS::get_singleton());
}

void OS_JavaScript::initialize_joypads() {
}

OS_JavaScript::OS_JavaScript() {
	if (AudioDriverJavaScript::is_available()) {
		audio_driver_javascript = memnew(AudioDriverJavaScript);
		AudioDriverManager::add_driver(audio_driver_javascript);
	}

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	FileAccessUnix::close_notification_func = file_access_close_callback;
}
