/*************************************************************************/
/*  platform.cpp                                                         */
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

#include "platform.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/midi_driver.h"
#include "core/version_generated.gen.h"
#include "servers/audio_server.h"

#include <stdarg.h>

Platform *Platform::singleton = nullptr;
uint64_t Platform::target_ticks = 0;

Platform *Platform::get_singleton() {
	return singleton;
}

uint32_t Platform::get_ticks_msec() const {
	return get_ticks_usec() / 1000;
}

String Platform::get_iso_date_time(bool local) const {
	Platform::Date date = get_date(local);
	Platform::Time time = get_time(local);

	String timezone;
	if (!local) {
		TimeZoneInfo zone = get_time_zone_info();
		if (zone.bias >= 0) {
			timezone = "+";
		}
		timezone = timezone + itos(zone.bias / 60).pad_zeros(2) + itos(zone.bias % 60).pad_zeros(2);
	} else {
		timezone = "Z";
	}

	return itos(date.year).pad_zeros(2) +
		   "-" +
		   itos(date.month).pad_zeros(2) +
		   "-" +
		   itos(date.day).pad_zeros(2) +
		   "T" +
		   itos(time.hour).pad_zeros(2) +
		   ":" +
		   itos(time.min).pad_zeros(2) +
		   ":" +
		   itos(time.sec).pad_zeros(2) +
		   timezone;
}

double Platform::get_unix_time() const {
	return 0;
}

void Platform::debug_break() {
	// something
}

void Platform::_set_logger(CompositeLogger *p_logger) {
	if (_logger) {
		memdelete(_logger);
	}
	_logger = p_logger;
}

void Platform::add_logger(Logger *p_logger) {
	if (!_logger) {
		Vector<Logger *> loggers;
		loggers.push_back(p_logger);
		_logger = memnew(CompositeLogger(loggers));
	} else {
		_logger->add_logger(p_logger);
	}
}

void Platform::print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, Logger::ErrorType p_type) {
	if (!_stderr_enabled) {
		return;
	}

	_logger->log_error(p_function, p_file, p_line, p_code, p_rationale, p_type);
}

void Platform::print(const char *p_format, ...) {
	if (!_stdout_enabled) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	_logger->logv(p_format, argp, false);

	va_end(argp);
}

void Platform::printerr(const char *p_format, ...) {
	if (!_stderr_enabled) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	_logger->logv(p_format, argp, true);

	va_end(argp);
}

void Platform::set_low_processor_usage_mode(bool p_enabled) {
	low_processor_usage_mode = p_enabled;
}

bool Platform::is_in_low_processor_usage_mode() const {
	return low_processor_usage_mode;
}

void Platform::set_low_processor_usage_mode_sleep_usec(int p_usec) {
	low_processor_usage_mode_sleep_usec = p_usec;
}

int Platform::get_low_processor_usage_mode_sleep_usec() const {
	return low_processor_usage_mode_sleep_usec;
}

String Platform::get_executable_path() const {
	return _execpath;
}

int Platform::get_process_id() const {
	return -1;
}

void Platform::vibrate_handheld(int p_duration_ms) {
	WARN_PRINT("vibrate_handheld() only works with Android and iOS");
}

bool Platform::is_stdout_verbose() const {
	return _verbose_stdout;
}

bool Platform::is_stdout_debug_enabled() const {
	return _debug_stdout;
}

bool Platform::is_stdout_enabled() const {
	return _stdout_enabled;
}

bool Platform::is_stderr_enabled() const {
	return _stderr_enabled;
}

void Platform::set_stdout_enabled(bool p_enabled) {
	_stdout_enabled = p_enabled;
}

void Platform::set_stderr_enabled(bool p_enabled) {
	_stderr_enabled = p_enabled;
}

void Platform::dump_memory_to_file(const char *p_file) {
	//Memory::dump_static_mem_to_file(p_file);
}

static FileAccess *_PrintResourceFile = nullptr;

static void _print_resource(Object *p_obj) {
	Resource *res = Object::cast_to<Resource>(p_obj);
	if (!res) {
		return;
	}

	String str = itos(res->get_instance_id()) + String(res->get_class()) + ":" + String(res->get_name()) + " - " + res->get_path();
	if (_PrintResourceFile) {
		_PrintResourceFile->store_line(str);
	} else {
		print_line(str);
	}
}

void Platform::print_all_resources(String p_to_file) {
	ERR_FAIL_COND(p_to_file != "" && _PrintResourceFile);
	if (p_to_file != "") {
		Error err;
		_PrintResourceFile = FileAccess::open(p_to_file, FileAccess::WRITE, &err);
		if (err != OK) {
			_PrintResourceFile = nullptr;
			ERR_FAIL_MSG("Can't print all resources to file: " + String(p_to_file) + ".");
		}
	}

	ObjectDB::debug_objects(_print_resource);

	if (p_to_file != "") {
		if (_PrintResourceFile) {
			memdelete(_PrintResourceFile);
		}
		_PrintResourceFile = nullptr;
	}
}

void Platform::print_resources_in_use(bool p_short) {
	ResourceCache::dump(nullptr, p_short);
}

void Platform::dump_resources_to_file(const char *p_file) {
	ResourceCache::dump(p_file);
}

void Platform::set_no_window_mode(bool p_enable) {
	_no_window = p_enable;
}

bool Platform::is_no_window_mode_enabled() const {
	return _no_window;
}

int Platform::get_exit_code() const {
	return _exit_code;
}

void Platform::set_exit_code(int p_code) {
	_exit_code = p_code;
}

String Platform::get_locale() const {
	return "en";
}

// Helper function to ensure that a dir name/path will be valid on the Platform
String Platform::get_safe_dir_name(const String &p_dir_name, bool p_allow_dir_separator) const {
	Vector<String> invalid_chars = String(": * ? \" < > |").split(" ");
	if (p_allow_dir_separator) {
		// Dir separators are allowed, but disallow ".." to avoid going up the filesystem
		invalid_chars.push_back("..");
	} else {
		invalid_chars.push_back("/");
	}

	String safe_dir_name = p_dir_name.replace("\\", "/").strip_edges();
	for (int i = 0; i < invalid_chars.size(); i++) {
		safe_dir_name = safe_dir_name.replace(invalid_chars[i], "-");
	}
	return safe_dir_name;
}

// Path to data, config, cache, etc. Platform-specific folders

// Get properly capitalized engine name for system paths
String Platform::get_godot_dir_name() const {
	// Default to lowercase, so only override when different case is needed
	return String(VERSION_SHORT_NAME).to_lower();
}

// Platform equivalent of XDG_DATA_HOME
String Platform::get_data_path() const {
	return ".";
}

// Platform equivalent of XDG_CONFIG_HOME
String Platform::get_config_path() const {
	return ".";
}

// Platform equivalent of XDG_CACHE_HOME
String Platform::get_cache_path() const {
	return ".";
}

// Path to macOS .app bundle resources
String Platform::get_bundle_resource_dir() const {
	return ".";
}

// Platform specific path for user://
String Platform::get_user_data_dir() const {
	return ".";
}

// Absolute path to res://
String Platform::get_resource_dir() const {
	return ProjectSettings::get_singleton()->get_resource_path();
}

// Access system-specific dirs like Documents, Downloads, etc.
String Platform::get_system_dir(SystemDir p_dir) const {
	return ".";
}

Error Platform::shell_open(String p_uri) {
	return ERR_UNAVAILABLE;
}

// implement these with the canvas?

uint64_t Platform::get_static_memory_usage() const {
	return Memory::get_mem_usage();
}

uint64_t Platform::get_static_memory_peak_usage() const {
	return Memory::get_mem_max_usage();
}

Error Platform::set_cwd(const String &p_cwd) {
	return ERR_CANT_OPEN;
}

uint64_t Platform::get_free_static_memory() const {
	return Memory::get_mem_available();
}

void Platform::yield() {
}

void Platform::ensure_user_data_dir() {
	String dd = get_user_data_dir();
	DirAccess *da = DirAccess::open(dd);
	if (da) {
		memdelete(da);
		return;
	}

	da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = da->make_dir_recursive(dd);
	ERR_FAIL_COND_MSG(err != OK, "Error attempting to create data dir: " + dd + ".");

	memdelete(da);
}

String Platform::get_model_name() const {
	return "GenericDevice";
}

void Platform::set_cmdline(const char *p_execpath, const List<String> &p_args) {
	_execpath = p_execpath;
	_cmdline = p_args;
}

String Platform::get_unique_id() const {
	ERR_FAIL_V("");
}

int Platform::get_processor_count() const {
	return 1;
}

bool Platform::can_use_threads() const {
#ifdef NO_THREADS
	return false;
#else
	return true;
#endif
}

void Platform::set_has_server_feature_callback(HasServerFeatureCallback p_callback) {
	has_server_feature_callback = p_callback;
}

bool Platform::has_feature(const String &p_feature) {
	if (p_feature == get_name()) {
		return true;
	}
#ifdef DEBUG_ENABLED
	if (p_feature == "debug") {
		return true;
	}
#else
	if (p_feature == "release")
		return true;
#endif
#ifdef TOOLS_ENABLED
	if (p_feature == "editor") {
		return true;
	}
#else
	if (p_feature == "standalone")
		return true;
#endif

	if (sizeof(void *) == 8 && p_feature == "64") {
		return true;
	}
	if (sizeof(void *) == 4 && p_feature == "32") {
		return true;
	}
#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__)
	if (p_feature == "x86_64") {
		return true;
	}
#elif (defined(__i386) || defined(__i386__))
	if (p_feature == "x86") {
		return true;
	}
#elif defined(__aarch64__)
	if (p_feature == "arm64") {
		return true;
	}
#elif defined(__arm__)
#if defined(__ARM_ARCH_7A__)
	if (p_feature == "armv7a" || p_feature == "armv7") {
		return true;
	}
#endif
#if defined(__ARM_ARCH_7S__)
	if (p_feature == "armv7s" || p_feature == "armv7") {
		return true;
	}
#endif
	if (p_feature == "arm") {
		return true;
	}
#endif

	if (_check_internal_feature_support(p_feature)) {
		return true;
	}

	if (has_server_feature_callback && has_server_feature_callback(p_feature)) {
		return true;
	}

	if (ProjectSettings::get_singleton()->has_custom_feature(p_feature)) {
		return true;
	}

	return false;
}

void Platform::set_restart_on_exit(bool p_restart, const List<String> &p_restart_arguments) {
	restart_on_exit = p_restart;
	restart_commandline = p_restart_arguments;
}

bool Platform::is_restart_on_exit_set() const {
	return restart_on_exit;
}

List<String> Platform::get_restart_on_exit_arguments() const {
	return restart_commandline;
}

PackedStringArray Platform::get_connected_midi_inputs() {
	if (MIDIDriver::get_singleton()) {
		return MIDIDriver::get_singleton()->get_connected_inputs();
	}

	PackedStringArray list;
	ERR_FAIL_V_MSG(list, vformat("MIDI input isn't supported on %s.", Platform::get_singleton()->get_name()));
}

void Platform::open_midi_inputs() {
	if (MIDIDriver::get_singleton()) {
		MIDIDriver::get_singleton()->open();
	} else {
		ERR_PRINT(vformat("MIDI input isn't supported on %s.", Platform::get_singleton()->get_name()));
	}
}

void Platform::close_midi_inputs() {
	if (MIDIDriver::get_singleton()) {
		MIDIDriver::get_singleton()->close();
	} else {
		ERR_PRINT(vformat("MIDI input isn't supported on %s.", Platform::get_singleton()->get_name()));
	}
}

void Platform::add_frame_delay(bool p_can_draw) {
	const uint32_t frame_delay = Engine::get_singleton()->get_frame_delay();
	if (frame_delay) {
		// Add fixed frame delay to decrease CPU/GPU usage. This doesn't take
		// the actual frame time into account.
		// Due to the high fluctuation of the actual sleep duration, it's not recommended
		// to use this as a FPS limiter.
		delay_usec(frame_delay * 1000);
	}

	// Add a dynamic frame delay to decrease CPU/GPU usage. This takes the
	// previous frame time into account for a smoother result.
	uint64_t dynamic_delay = 0;
	if (is_in_low_processor_usage_mode() || !p_can_draw) {
		dynamic_delay = get_low_processor_usage_mode_sleep_usec();
	}
	const int target_fps = Engine::get_singleton()->get_target_fps();
	if (target_fps > 0 && !Engine::get_singleton()->is_editor_hint()) {
		// Override the low processor usage mode sleep delay if the target FPS is lower.
		dynamic_delay = MAX(dynamic_delay, (uint64_t)(1000000 / target_fps));
	}

	if (dynamic_delay > 0) {
		target_ticks += dynamic_delay;
		uint64_t current_ticks = get_ticks_usec();

		if (current_ticks < target_ticks) {
			delay_usec(target_ticks - current_ticks);
		}

		current_ticks = get_ticks_usec();
		target_ticks = MIN(MAX(target_ticks, current_ticks - dynamic_delay), current_ticks + dynamic_delay);
	}
}

Platform::Platform() {
	singleton = this;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));
}

Platform::~Platform() {
	memdelete(_logger);
	singleton = nullptr;
}
