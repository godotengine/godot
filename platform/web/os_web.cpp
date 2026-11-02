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
#include "display_server_web.h"
#include "godot_js.h"
#include "ip_web.h"
#include "net_socket_web.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/os/main_loop.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/profiling/profiling.h"
#include "core/templates/hash_map.h"
#include "core/templates/safe_refcount.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"

#include <dlfcn.h>
#include <emscripten.h>
#ifdef THREADS_ENABLED
#include <emscripten/threading.h>
#endif

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
	// 2dog: the callbacks reach this instance's IDB sync state; the next lifetime installs its own.
	FileAccessUnix::close_notification_func = nullptr;
	DirAccessUnix::remove_notification_func = nullptr;
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

bool OS_Web::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "web") {
		return true;
	}

	// 2dog: static library builds load GDExtensions through the side module loader.
	if (p_feature == "web_extensions") {
#if defined(WEB_DLINK_ENABLED) || defined(WEB_SIDE_MODULE_LOADER_ENABLED)
		return true;
#else
		return false;
#endif
	}
	if (p_feature == "web_noextensions") {
#if defined(WEB_DLINK_ENABLED) || defined(WEB_SIDE_MODULE_LOADER_ENABLED)
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

#ifdef WEB_SIDE_MODULE_LOADER_ENABLED
// 2dog: load GDExtension side modules from the exported pack through Godot's file system.
static String _side_module_path(const String &p_path) {
	if (FileAccess::exists(p_path)) {
		return p_path;
	}
	const String local = ProjectSettings::get_singleton()->localize_path(p_path);
	if (FileAccess::exists(local)) {
		return local;
	}
	// 2dog: restore res:// when globalize_path has no resource directory for a packed project.
	if (p_path.is_relative_path() && FileAccess::exists("res://" + p_path)) {
		return "res://" + p_path;
	}
	return String();
}

static String _side_module_error() {
	char *error = godot_js_dylink_error();
	const String message = String::utf8(error);
	free(error);
	return message;
}

#ifdef THREADS_ENABLED
// 2dog: wasm function tables are per thread, so every thread instantiates each side module at the memory and table
// layout its first loader chose. Entries are append-only and never unloaded; the count publishes them.
struct WebSideModule {
	char *name = nullptr;
	uint8_t *bytes = nullptr;
	int size = 0;
	char *layout = nullptr;
};

static constexpr int WEB_SIDE_MODULE_MAX = 256;
static WebSideModule web_side_modules[WEB_SIDE_MODULE_MAX];
static SafeNumeric<int> web_side_module_count;
static thread_local int web_side_modules_synced = 0;
static thread_local bool web_side_modules_syncing = false;

static Mutex &_side_module_mutex() {
	static Mutex mutex;
	return mutex;
}

// 2dog: called at thread start (library_godot_dylink.js) and after every futex wait.
extern "C" EMSCRIPTEN_KEEPALIVE void godot_web_sync_side_modules() {
	if (web_side_modules_syncing) {
		return;
	}
#ifdef PROXY_TO_PTHREAD_ENABLED
	// 2dog: the page thread never runs extension code (Godot runs on workers), so it skips the synchronous compiles.
	if (emscripten_is_main_browser_thread()) {
		return;
	}
#endif
	const int count = web_side_module_count.get();
	if (web_side_modules_synced >= count) {
		return;
	}
	web_side_modules_syncing = true;
	for (int i = web_side_modules_synced; i < count; i++) {
		const WebSideModule &module = web_side_modules[i];
		// 2dog: failures are reported by the loader; a thread without the module fails only if it calls into it.
		godot_js_dylink_thread_load(i, module.name, module.bytes, module.size, module.layout);
	}
	web_side_modules_synced = count;
	web_side_modules_syncing = false;
}

// 2dog: threads that were blocked while a module loaded catch up when they wake, before running more code, like
// emscripten's dlsync. Threaded 2dog hosts link with --wrap=emscripten_futex_wait.
extern "C" int __real_emscripten_futex_wait(volatile void *p_addr, uint32_t p_val, double p_max_wait_ms);
extern "C" int __wrap_emscripten_futex_wait(volatile void *p_addr, uint32_t p_val, double p_max_wait_ms) {
	const int ret = __real_emscripten_futex_wait(p_addr, p_val, p_max_wait_ms);
	godot_web_sync_side_modules();
	return ret;
}

static Error _load_side_module(const String &p_path, const Vector<uint8_t> &p_bytes, int &r_handle) {
	const CharString name = p_path.get_file().utf8();
	MutexLock lock(_side_module_mutex());
	// 2dog: dependencies resolve by name among the modules this thread has loaded.
	godot_web_sync_side_modules();
	const int count = web_side_module_count.get();
	for (int i = 0; i < count; i++) {
		if (strcmp(web_side_modules[i].name, name.get_data()) == 0) {
			r_handle = i + 1;
			return OK;
		}
	}
	ERR_FAIL_COND_V_MSG(count == WEB_SIDE_MODULE_MAX, ERR_OUT_OF_MEMORY, vformat("Can't open dynamic library: %s. Too many side modules.", p_path));

	// 2dog: threads load from these bytes whenever they catch up, so they are kept.
	WebSideModule module;
	module.name = strdup(name.get_data());
	module.size = p_bytes.size();
	module.bytes = static_cast<uint8_t *>(malloc(module.size));
	memcpy(module.bytes, p_bytes.ptr(), module.size);
	module.layout = godot_js_dylink_thread_load(count, module.name, module.bytes, module.size, nullptr);
	if (module.layout == nullptr) {
		free(module.name);
		free(module.bytes);
		ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, _side_module_error()));
	}
	web_side_modules[count] = module;
	web_side_modules_synced = count + 1;
	web_side_module_count.set(count + 1);
	r_handle = count + 1;
	return OK;
}

static void _close_side_module(int) {
	// 2dog: shared modules stay loaded: wasm instances cannot unload, and other threads may still run their code.
}
#else
static Error _load_side_module(const String &p_path, const Vector<uint8_t> &p_bytes, int &r_handle) {
	r_handle = godot_js_dylink_open(p_path.get_file().utf8().get_data(), p_bytes.ptr(), p_bytes.size());
	ERR_FAIL_COND_V_MSG(r_handle == 0, ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, _side_module_error()));
	return OK;
}

static void _close_side_module(int p_handle) {
	godot_js_dylink_close(p_handle);
}
#endif

static Error _open_side_module(const String &p_path, int &r_handle) {
	const String path = _side_module_path(p_path);
	ERR_FAIL_COND_V_MSG(path.is_empty(), ERR_FILE_NOT_FOUND, vformat("Can't open dynamic library, file not found: '%s'.", p_path));
	const Vector<uint8_t> bytes = FileAccess::get_file_as_bytes(path);
	ERR_FAIL_COND_V_MSG(bytes.is_empty(), ERR_CANT_OPEN, vformat("Can't read dynamic library: '%s'.", path));
	return _load_side_module(path, bytes, r_handle);
}

// 2dog: close dependencies with their owning library handle.
static HashMap<int, Vector<int>> &_side_module_dependencies() {
	static HashMap<int, Vector<int>> dependencies;
	return dependencies;
}

Error OS_Web::open_dynamic_library(const String &p_path, void *&p_library_handle, GDExtensionData *p_data) {
	Vector<int> dependency_handles;
	Error err = OK;
	if (p_data != nullptr && p_data->library_dependencies != nullptr) {
		// 2dog: load dependencies first so dylink needed entries resolve by file name.
		for (const String &dependency : *p_data->library_dependencies) {
			int dependency_handle = 0;
			err = _open_side_module(dependency, dependency_handle);
			if (err != OK) {
				break;
			}
			dependency_handles.push_back(dependency_handle);
		}
	}

	int handle = 0;
	if (err == OK) {
		err = _open_side_module(p_path, handle);
	}
	if (err != OK) {
		for (int dependency_handle : dependency_handles) {
			_close_side_module(dependency_handle);
		}
		return err;
	}
	if (!dependency_handles.is_empty()) {
		_side_module_dependencies()[handle].append_array(dependency_handles);
	}
	p_library_handle = reinterpret_cast<void *>(static_cast<intptr_t>(handle));

	if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
		*p_data->r_resolved_path = _side_module_path(p_path);
	}

	return OK;
}

Error OS_Web::close_dynamic_library(void *p_library_handle) {
	const int handle = static_cast<int>(reinterpret_cast<intptr_t>(p_library_handle));
	_close_side_module(handle);
	if (const Vector<int> *dependencies = _side_module_dependencies().getptr(handle)) {
		for (int dependency_handle : *dependencies) {
			_close_side_module(dependency_handle);
		}
		_side_module_dependencies().erase(handle);
	}
	return OK;
}

Error OS_Web::get_dynamic_library_symbol_handle(void *p_library_handle, const String &p_name, void *&p_symbol_handle, bool p_optional) {
#ifdef THREADS_ENABLED
	godot_web_sync_side_modules();
#endif
	p_symbol_handle =godot_js_dylink_symbol(static_cast<int>(reinterpret_cast<intptr_t>(p_library_handle)), p_name.utf8().get_data());
	if (p_symbol_handle == nullptr) {
		ERR_FAIL_COND_V_MSG(!p_optional, ERR_CANT_RESOLVE, vformat("Can't resolve symbol %s. Error: %s.", p_name, _side_module_error()));
		return ERR_CANT_RESOLVE;
	}
	return OK;
}
#else
Error OS_Web::open_dynamic_library(const String &p_path, void *&p_library_handle, GDExtensionData *p_data) {
	String path = p_path.get_file();
	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_NULL_V_MSG(p_library_handle, ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, dlerror()));

	if (p_data != nullptr && p_data->r_resolved_path != nullptr) {
		*p_data->r_resolved_path = path;
	}

	return OK;
}
#endif

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
