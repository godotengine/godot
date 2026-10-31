/**************************************************************************/
/*  os_sdl3.cpp                                                           */
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

#include "os_sdl3.h"

#include "core/version_generated.gen.h"

#include "core/object/object.h"
#include "core/os/main_loop.h"
#include "core/variant/variant.h"
#include "display_server_sdl3.h"
#include "drivers/sdl/joypad_sdl.h"

#include <SDL3/SDL.h>

#ifdef WINDOWS_ENABLED
#include "drivers/windows/dir_access_windows.h"
#include "drivers/windows/file_access_windows.h"
#include "drivers/windows/file_access_windows_pipe.h"
#include "drivers/windows/ip_windows.h"
#include "drivers/windows/net_socket_winsock.h"
#else
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "drivers/unix/file_access_unix_pipe.h"
#ifndef UNIX_SOCKET_UNAVAILABLE
#include "drivers/unix/ip_unix.h"
#include "drivers/unix/net_socket_unix.h"
#endif
#endif

#include <cstdio>
#include <time.h>

#ifdef WINDOWS_ENABLED
#include <direct.h>
#include <windows.h>
#include <bcrypt.h>
#endif

static String _as_string(const char *p_sdl_owned) {
	String result = String::utf8(p_sdl_owned);
	SDL_free(const_cast<char *>(p_sdl_owned));
	return result;
}

static String _sdl3_pref_path(const String &p_app_name) {
	CharString app = p_app_name.utf8();
	// SDL ends the path with a separator; DirAccessUnix compares user:// against this as a prefix and
	// the slashless form it derives would fail that check.
	String path = _as_string(SDL_GetPrefPath(nullptr, app.get_data()));
	while (path.length() > 1 && (path.ends_with("/") || path.ends_with("\\"))) {
		path = path.left(-1);
	}
	return path;
}

OS_SDL3::OS_SDL3() {
}

OS_SDL3::~OS_SDL3() {
}

void OS_SDL3::initialize() {
	// The display server performs the one SDL_Init (video + events + gamepad);
	// OS must not, because headless/import instances never create one.
	DisplayServerSDL3::register_sdl3_driver();

#ifdef WINDOWS_ENABLED
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_FILESYSTEM);
	FileAccess::make_default<FileAccessWindowsPipe>(FileAccess::ACCESS_PIPE);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_FILESYSTEM);
	NetSocketWinSock::make_default();
	IPWindows::make_default();
#else
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_FILESYSTEM);
	FileAccess::make_default<FileAccessUnixPipe>(FileAccess::ACCESS_PIPE);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_FILESYSTEM);
#ifndef UNIX_SOCKET_UNAVAILABLE
	NetSocketUnix::make_default();
	IPUnix::make_default();
#endif
#endif

	// main.cpp does not call this for libgodot hosts on every platform, and the
	// joypad driver owns no timer of its own.
	initialize_joypads();
}

void OS_SDL3::initialize_joypads() {
	if (joypad_sdl != nullptr) {
		return;
	}
	joypad_sdl = memnew(JoypadSDL());
	if (joypad_sdl->initialize() != OK) {
		ERR_PRINT("Couldn't initialize SDL joypad input driver.");
		memdelete(joypad_sdl);
		joypad_sdl = nullptr;
	}
}

void OS_SDL3::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

void OS_SDL3::delete_main_loop() {
	memdelete(main_loop);
	main_loop = nullptr;
}

void OS_SDL3::finalize() {
	if (joypad_sdl) {
		memdelete(joypad_sdl);
		joypad_sdl = nullptr;
	}
}

void OS_SDL3::finalize_core() {
	for (const KeyValue<ProcessID, SDL_Process *> &kv : processes) {
		SDL_DestroyProcess(kv.value);
	}
	processes.clear();
#ifdef WINDOWS_ENABLED
	// make_default() asserts it has not run yet, so a second engine lifetime needs these.
	FileAccessWindows::finalize();
	NetSocketWinSock::cleanup();
#endif
}

bool OS_SDL3::_check_internal_feature_support(const String &p_feature) {
	return p_feature == get_identifier();
}

Vector<String> OS_SDL3::get_video_adapter_driver_info() const {
	// shortcut: no adapter driver strings yet; the RD backend reports its own.
	return Vector<String>();
}

String OS_SDL3::get_stdin_string(int64_t p_buffer_size) {
	Vector<uint8_t> data;
	data.resize(p_buffer_size);
	if (fgets((char *)data.ptrw(), data.size(), stdin)) {
		return String::utf8((char *)data.ptr()).replace("\r\n", "\n").rstrip("\n");
	}
	return String();
}

PackedByteArray OS_SDL3::get_stdin_buffer(int64_t p_buffer_size) {
	Vector<uint8_t> data;
	data.resize(p_buffer_size);
	const size_t read = fread(data.ptrw(), 1, data.size(), stdin);
	data.resize(read);
	return data;
}

Error OS_SDL3::get_entropy(uint8_t *r_buffer, int p_bytes) {
	// SDL_GetRandom is not cryptographic, so use the platform CSPRNG directly.
	ERR_FAIL_COND_V(p_bytes < 0, ERR_INVALID_PARAMETER);
	int left = p_bytes;
	int ofs = 0;
	while (left > 0) {
#ifdef WINDOWS_ENABLED
		// BCryptGenRandom takes a ULONG length, so large requests are chunked.
		const int chunk = MIN(left, 256);
		if (BCryptGenRandom(nullptr, r_buffer + ofs, (ULONG)chunk, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
			return FAILED;
		}
#else
		const int chunk = MIN(left, 256);
		if (getentropy(r_buffer + ofs, (size_t)chunk) != 0) {
			return FAILED;
		}
#endif
		left -= chunk;
		ofs += chunk;
	}
	return OK;
}

// p_capture routes stdout to a pipe SDL_ReadProcess can drain; p_merge_stderr folds stderr into it.
static SDL_Process *_spawn(const String &p_path, const List<String> &p_arguments, bool p_capture, bool p_merge_stderr) {
	// SDL_CreateProcess reads these during the call, but the CharStrings must stay
	// alive until the pointers are handed over: arg.utf8() returns a temporary
	// whose buffer dies at the end of its statement.
	LocalVector<CharString> storage;
	storage.push_back(p_path.utf8());
	for (const String &arg : p_arguments) {
		storage.push_back(arg.utf8());
	}

	LocalVector<const char *> argv;
	for (const CharString &arg : storage) {
		argv.push_back(arg.get_data());
	}
	argv.push_back(nullptr);

	SDL_PropertiesID props = SDL_CreateProperties();
	SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, (void *)argv.ptr());
	if (p_capture) {
		SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
		SDL_SetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_TO_STDOUT_BOOLEAN, p_merge_stderr);
	}
	SDL_Process *proc = SDL_CreateProcessWithProperties(props);
	SDL_DestroyProperties(props);
	if (proc == nullptr) {
		ERR_PRINT(vformat("SDL_CreateProcess failed: %s", SDL_GetError()));
	}
	return proc;
}

Error OS_SDL3::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id, bool p_open_console) {
	SDL_Process *proc = _spawn(p_path, p_arguments, false, false);
	if (proc == nullptr) {
		return FAILED;
	}

	ProcessID pid = (ProcessID)SDL_GetNumberProperty(SDL_GetProcessProperties(proc), SDL_PROP_PROCESS_PID_NUMBER, 0);
	processes.insert(pid, proc);
	if (r_child_id) {
		*r_child_id = pid;
	}
	return OK;
}

// Removes the process from the bookkeeping. Its exit code is not kept: callers
// that need one read get_process_exit_code() before the process goes away.
void OS_SDL3::_release_process(ProcessID p_pid) {
	SDL_Process **proc = processes.getptr(p_pid);
	if (proc == nullptr) {
		return;
	}
	SDL_DestroyProcess(*proc);
	processes.erase(p_pid);
}

Error OS_SDL3::execute(const String &p_path, const List<String> &p_arguments, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex, bool p_open_console) {
	if (r_pipe != nullptr) {
		SDL_Process *proc = _spawn(p_path, p_arguments, true, read_stderr);
		if (proc == nullptr) {
			return FAILED;
		}
		// Blocks until the child exits, so the exit code comes with the output.
		size_t length = 0;
		int exit_code = -1;
		void *data = SDL_ReadProcess(proc, &length, &exit_code);
		if (data != nullptr) {
			if (p_pipe_mutex) {
				p_pipe_mutex->lock();
			}
			*r_pipe += String::utf8((const char *)data, (int)length);
			if (p_pipe_mutex) {
				p_pipe_mutex->unlock();
			}
			SDL_free(data);
		}
		SDL_DestroyProcess(proc);
		if (r_exitcode) {
			*r_exitcode = exit_code;
		}
		return OK;
	}
	ProcessID pid;
	Error err = create_process(p_path, p_arguments, &pid, p_open_console);
	if (err != OK) {
		return err;
	}
	while (is_process_running(pid)) {
		delay_usec(1000);
	}
	// Read the code before releasing: the handle carries it.
	const int exit_code = get_process_exit_code(pid);
	_release_process(pid);
	if (r_exitcode) {
		*r_exitcode = exit_code;
	}
	return OK;
}

Error OS_SDL3::kill(const ProcessID &p_pid) {
	SDL_Process **proc = processes.getptr(p_pid);
	if (proc == nullptr) {
		return ERR_INVALID_PARAMETER;
	}
	// Reap before destroying: a child that ignores the signal would otherwise
	// become a zombie with its handle dropped.
	SDL_KillProcess(*proc, true);
	SDL_WaitProcess(*proc, true, nullptr);
	_release_process(p_pid);
	return OK;
}

bool OS_SDL3::is_process_running(const ProcessID &p_pid) const {
	SDL_Process *const *proc = processes.getptr(p_pid);
	if (proc == nullptr) {
		return false;
	}
	return !SDL_WaitProcess(*proc, false, nullptr);
}

int OS_SDL3::get_process_exit_code(const ProcessID &p_pid) const {
	SDL_Process *const *proc = processes.getptr(p_pid);
	if (proc == nullptr) {
		return -1;
	}
	int exit_code = 0;
	if (SDL_WaitProcess(*proc, false, &exit_code)) {
		return exit_code;
	}
	return -1;
}

bool OS_SDL3::has_environment(const String &p_var) const {
	CharString name = p_var.utf8();
	return SDL_GetEnvironmentVariable(SDL_GetEnvironment(), name.get_data()) != nullptr;
}

String OS_SDL3::get_environment(const String &p_var) const {
	CharString name = p_var.utf8();
	const char *value = SDL_GetEnvironmentVariable(SDL_GetEnvironment(), name.get_data());
	return value != nullptr ? String::utf8(value) : String();
}

void OS_SDL3::set_environment(const String &p_var, const String &p_value) const {
	CharString name = p_var.utf8();
	CharString value = p_value.utf8();
	SDL_SetEnvironmentVariable(SDL_GetEnvironment(), name.get_data(), value.get_data(), true);
}

void OS_SDL3::unset_environment(const String &p_var) const {
	CharString name = p_var.utf8();
	SDL_UnsetEnvironmentVariable(SDL_GetEnvironment(), name.get_data());
}

String OS_SDL3::get_name() const {
	return "SDL3";
}

String OS_SDL3::get_distribution_name() const {
	return get_name();
}

String OS_SDL3::get_version() const {
	return SDL_GetRevision();
}

MainLoop *OS_SDL3::get_main_loop() const {
	return main_loop;
}

String OS_SDL3::get_executable_path() const {
	// SDL owns this string for the life of the process; freeing it leaves SDL's
	// internal cache dangling, so it must not go through _as_string().
	return String::utf8(SDL_GetBasePath());
}

OS::DateTime OS_SDL3::get_datetime(bool utc) const {
	DateTime result;
	SDL_Time now = 0;
	if (SDL_GetCurrentTime(&now)) {
		SDL_DateTime dt;
		if (SDL_TimeToDateTime(now, &dt, !utc)) {
			result.year = dt.year;
			result.month = (Month)dt.month;
			result.day = dt.day;
			result.weekday = (Weekday)dt.day_of_week;
			result.hour = dt.hour;
			result.minute = dt.minute;
			result.second = dt.second;
			return result;
		}
	}
	return result;
}

OS::TimeZoneInfo OS_SDL3::get_time_zone_info() const {
	TimeZoneInfo info;
	SDL_Time now = 0;
	if (SDL_GetCurrentTime(&now)) {
		SDL_DateTime dt;
		if (SDL_TimeToDateTime(now, &dt, true)) {
			// SDL reports seconds east of UTC; Godot's bias is minutes where east
			// is negative, matching Windows' TIME_ZONE_INFORMATION and OS_Unix.
			info.bias = -dt.utc_offset / 60;
		}
	}
	// shortcut: SDL exposes no time zone name, so this stays empty, add a lookup
	// when a host needs TZ names.
	return info;
}

void OS_SDL3::delay_usec(uint32_t p_usec) const {
	SDL_DelayNS((uint64_t)p_usec * 1000);
}

uint64_t OS_SDL3::get_ticks_usec() const {
	return SDL_GetTicksNS() / 1000;
}


String OS_SDL3::get_user_data_dir(const String &p_user_dir) const {
	return _sdl3_pref_path(get_godot_dir_name().path_join(p_user_dir));
}

String OS_SDL3::get_temp_path() const {
	return _sdl3_pref_path("temp");
}

String OS_SDL3::get_cache_path() const {
	return _sdl3_pref_path("cache");
}

String OS_SDL3::get_config_path() const {
	return _sdl3_pref_path("config");
}

String OS_SDL3::get_data_path() const {
	return _sdl3_pref_path("data");
}

String OS_SDL3::get_locale() const {
	int count = 0;
	SDL_Locale **locales = SDL_GetPreferredLocales(&count);
	// "en" matches OS_Unix when the environment names no usable locale (LANG=C, unset).
	String locale = "en";
	if (locales != nullptr && count > 0 && locales[0]->language != nullptr) {
		locale = String::utf8(locales[0]->language);
		if (locales[0]->country != nullptr) {
			locale += "_" + String::utf8(locales[0]->country);
		}
	}
	SDL_free(locales);
	return locale;
}

Error OS_SDL3::set_cwd(const String &p_cwd) {
#ifdef WINDOWS_ENABLED
	// SDL has a getter but no setter; the host only needs this on desktop.
	if (_wchdir((const wchar_t *)p_cwd.utf16().get_data()) != 0) {
		return ERR_CANT_OPEN;
	}
#else
	if (chdir(p_cwd.utf8().get_data()) != 0) {
		return ERR_CANT_OPEN;
	}
#endif
	cwd = p_cwd;
	return OK;
}

String OS_SDL3::get_cwd() const {
	if (cwd.is_empty()) {
		return _as_string(SDL_GetCurrentDirectory());
	}
	return cwd;
}

int OS_SDL3::get_processor_count() const {
	const int count = SDL_GetNumLogicalCPUCores();
	return count > 0 ? count : 1;
}
