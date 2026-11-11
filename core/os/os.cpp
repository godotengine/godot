/**************************************************************************/
/*  os.cpp                                                                */
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

#include "os.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/json.h"
#include "core/os/midi_driver.h"
#include "core/version_generated.gen.h"

#include <cstdarg>
#include <cstdio>

#ifdef MINGW_ENABLED
#define MINGW_STDTHREAD_REDUNDANCY_WARNING
#include <thirdparty/mingw-std-threads/mingw.thread.h>
#define THREADING_NAMESPACE mingw_stdthread
#else
#include <thread>
#define THREADING_NAMESPACE std
#endif

OS *OS::singleton = nullptr;
uint64_t OS::target_ticks = 0;

OS *OS::get_singleton() {
	return singleton;
}

bool OS::prefer_meta_over_ctrl() {
#if defined(MACOS_ENABLED) || defined(APPLE_EMBEDDED_ENABLED)
	return true;
#elif defined(WEB_ENABLED)
	return singleton->has_feature("web_macos") || singleton->has_feature("web_ios");
#else
	return false;
#endif
}

uint64_t OS::get_ticks_msec() const {
	return get_ticks_usec() / 1000ULL;
}

double OS::get_unix_time() const {
	return 0;
}

void OS::_set_logger(CompositeLogger *p_logger) {
	memdelete(_logger);
	_logger = p_logger;
}

void OS::add_logger(Logger *p_logger) {
	if (!_logger) {
		Vector<Logger *> loggers;
		loggers.push_back(p_logger);
		_logger = memnew(CompositeLogger(loggers));
	} else {
		_logger->add_logger(p_logger);
	}
}

String OS::get_identifier() const {
	return get_name().to_lower();
}

void OS::print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, Logger::ErrorType p_type, const Vector<Ref<ScriptBacktrace>> &p_script_backtraces) {
	if (!_stderr_enabled) {
		return;
	}

	if (_logger) {
		_logger->log_error(p_function, p_file, p_line, p_code, p_rationale, p_editor_notify, p_type, p_script_backtraces);
	}
}

void OS::print(const char *p_format, ...) {
	if (!_stdout_enabled) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	if (_logger) {
		_logger->logv(p_format, argp, false);
	}

	va_end(argp);
}

void OS::print_rich(const char *p_format, ...) {
	if (!_stdout_enabled) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	if (_logger) {
		_logger->logv(p_format, argp, false);
	}

	va_end(argp);
}

void OS::printerr(const char *p_format, ...) {
	if (!_stderr_enabled) {
		return;
	}

	va_list argp;
	va_start(argp, p_format);

	if (_logger) {
		_logger->logv(p_format, argp, true);
	}

	va_end(argp);
}

void OS::alert(const String &p_alert, const String &p_title) {
	fprintf(stderr, "%s: %s\n", p_title.utf8().get_data(), p_alert.utf8().get_data());
}

void OS::set_low_processor_usage_mode(bool p_enabled) {
	low_processor_usage_mode = p_enabled;
}

bool OS::is_in_low_processor_usage_mode() const {
	return low_processor_usage_mode;
}

void OS::set_low_processor_usage_mode_sleep_usec(int p_usec) {
	low_processor_usage_mode_sleep_usec = p_usec;
}

int OS::get_low_processor_usage_mode_sleep_usec() const {
	return low_processor_usage_mode_sleep_usec;
}

void OS::set_delta_smoothing(bool p_enabled) {
	_delta_smoothing_enabled = p_enabled;
}

bool OS::is_delta_smoothing_enabled() const {
	return _delta_smoothing_enabled;
}

String OS::get_executable_path() const {
	return _execpath;
}

int OS::get_process_id() const {
	return -1;
}

bool OS::is_stdout_verbose() const {
	return _verbose_stdout;
}

bool OS::is_stdout_debug_enabled() const {
	return _debug_stdout;
}

bool OS::is_stdout_enabled() const {
	return _stdout_enabled;
}

bool OS::is_stderr_enabled() const {
	return _stderr_enabled;
}

void OS::set_stdout_enabled(bool p_enabled) {
	_stdout_enabled = p_enabled;
}

void OS::set_stderr_enabled(bool p_enabled) {
	_stderr_enabled = p_enabled;
}

String OS::multibyte_to_string(const String &p_encoding, const PackedByteArray &p_array) const {
	return String();
}

PackedByteArray OS::string_to_multibyte(const String &p_encoding, const String &p_string) const {
	return PackedByteArray();
}

int OS::get_exit_code() const {
	return _exit_code;
}

void OS::set_exit_code(int p_code) {
	_exit_code = p_code;
}

String OS::get_locale() const {
	return "en";
}

// Non-virtual helper to extract the 2 or 3-letter language code from
// `get_locale()` in a way that's consistent for all platforms.
String OS::get_locale_language() const {
	return get_locale().left(3).remove_char('_');
}

// Embedded PCK offset.
uint64_t OS::get_embedded_pck_offset() const {
	return 0;
}

// Default boot screen rect scale mode is "Keep Aspect Centered"
Rect2 OS::calculate_boot_screen_rect(const Size2 &p_window_size, const Size2 &p_imgrect_size) const {
	Rect2 screenrect;
	if (p_window_size.width > p_window_size.height) {
		// Scale horizontally.
		screenrect.size.y = p_window_size.height;
		screenrect.size.x = p_imgrect_size.x * p_window_size.height / p_imgrect_size.y;
		screenrect.position.x = (p_window_size.width - screenrect.size.x) / 2;
	} else {
		// Scale vertically.
		screenrect.size.x = p_window_size.width;
		screenrect.size.y = p_imgrect_size.y * p_window_size.width / p_imgrect_size.x;
		screenrect.position.y = (p_window_size.height - screenrect.size.y) / 2;
	}
	return screenrect;
}

// Helper function to ensure that a dir name/path will be valid on the OS
String OS::get_safe_dir_name(const String &p_dir_name, bool p_allow_paths) const {
	String safe_dir_name = p_dir_name;
	Vector<String> invalid_chars = String(": * ? \" < > |").split(" ");
	if (p_allow_paths) {
		// Dir separators are allowed, but disallow ".." to avoid going up the filesystem
		invalid_chars.push_back("..");
		safe_dir_name = safe_dir_name.replace_char('\\', '/').replace("//", "/").strip_edges();
	} else {
		invalid_chars.push_back("/");
		invalid_chars.push_back("\\");
		safe_dir_name = safe_dir_name.strip_edges();

		// These directory names are invalid.
		if (safe_dir_name == ".") {
			safe_dir_name = "dot";
		} else if (safe_dir_name == "..") {
			safe_dir_name = "twodots";
		}
	}

	for (int i = 0; i < invalid_chars.size(); i++) {
		safe_dir_name = safe_dir_name.replace(invalid_chars[i], "-");
	}

	// Trim trailing periods from the returned value as it's not valid for folder names on Windows.
	// This check is still applied on non-Windows platforms so the returned value is consistent across platforms.
	return safe_dir_name.rstrip(".");
}

// Path to data, config, cache, etc. OS-specific folders

// Get properly capitalized engine name for system paths
String OS::get_godot_dir_name() const {
	// Default to lowercase, so only override when different case is needed
	return String(GODOT_VERSION_SHORT_NAME).to_lower();
}

// OS equivalent of XDG_DATA_HOME
String OS::get_data_path() const {
	return ".";
}

// OS equivalent of XDG_CONFIG_HOME
String OS::get_config_path() const {
	return ".";
}

// OS equivalent of XDG_CACHE_HOME
String OS::get_cache_path() const {
	return ".";
}

String OS::get_temp_path() const {
	return ".";
}

// Path to macOS .app bundle resources
String OS::get_bundle_resource_dir() const {
	return ".";
}

// Path to macOS .app bundle embedded icon (.icns file).
String OS::get_bundle_icon_path() const {
	return String();
}

// Name of macOS .app bundle embedded icon (Liquid Glass asset name).
String OS::get_bundle_icon_name() const {
	return String();
}

// OS specific path for user://
String OS::get_user_data_dir(const String &p_user_dir) const {
	return ".";
}

String OS::get_user_data_dir() const {
	String appname = get_safe_dir_name(GLOBAL_GET("application/config/name"));
	if (!appname.is_empty()) {
		bool use_custom_dir = GLOBAL_GET("application/config/use_custom_user_dir");
		if (use_custom_dir) {
			String custom_dir = get_safe_dir_name(GLOBAL_GET("application/config/custom_user_dir_name"), true);
			if (custom_dir.is_empty()) {
				custom_dir = appname;
			}
			return get_user_data_dir(custom_dir);
		} else {
			return get_user_data_dir(get_godot_dir_name().path_join("app_userdata").path_join(appname));
		}
	} else {
		return get_user_data_dir(get_godot_dir_name().path_join("app_userdata").path_join("[unnamed project]"));
	}
}

// Absolute path to res://
String OS::get_resource_dir() const {
	return ProjectSettings::get_singleton()->get_resource_path();
}

// Access system-specific dirs like Documents, Downloads, etc.
String OS::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
	return ".";
}

String OS::expand_path(const String &p_path) const {
	return p_path;
}

void OS::create_lock_file() {
	if (Engine::get_singleton()->is_recovery_mode_hint()) {
		return;
	}

	String lock_file_path = get_user_data_dir().path_join(".recovery_mode_lock");
	Ref<FileAccess> lock_file = FileAccess::open(lock_file_path, FileAccess::WRITE);
	if (lock_file.is_valid()) {
		lock_file->close();
	}
}

void OS::remove_lock_file() {
	String lock_file_path = get_user_data_dir().path_join(".recovery_mode_lock");
	DirAccess::remove_absolute(lock_file_path);
}

Error OS::shell_open(const String &p_uri) {
	return ERR_UNAVAILABLE;
}

Error OS::shell_show_in_file_manager(String p_path, bool p_open_folder) {
	p_path = p_path.trim_prefix("file://");

	if (!DirAccess::dir_exists_absolute(p_path)) {
		p_path = p_path.get_base_dir();
	}

	p_path = String("file://") + p_path;

	return shell_open(p_path);
}
// implement these with the canvas?

uint64_t OS::get_static_memory_usage() const {
	return Memory::get_mem_usage();
}

uint64_t OS::get_static_memory_peak_usage() const {
	return Memory::get_mem_max_usage();
}

Error OS::set_cwd(const String &p_cwd) {
	return ERR_CANT_OPEN;
}

String OS::get_cwd() const {
	return ".";
}

Dictionary OS::get_memory_info() const {
	Dictionary meminfo;

	meminfo["physical"] = -1;
	meminfo["free"] = -1;
	meminfo["available"] = -1;
	meminfo["stack"] = -1;

	return meminfo;
}

void OS::yield() {
}

void OS::ensure_user_data_dir() {
	String dd = get_user_data_dir();
	if (DirAccess::exists(dd)) {
		return;
	}

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = da->make_dir_recursive(dd);
	ERR_FAIL_COND_MSG(err != OK, vformat("Error attempting to create data dir: %s.", dd));
}

String OS::get_model_name() const {
	return "GenericDevice";
}

void OS::set_cmdline(const char *p_execpath, const List<String> &p_args, const List<String> &p_user_args) {
	_execpath = String::utf8(p_execpath);
	_cmdline = p_args;
	_user_args = p_user_args;
}

String OS::get_unique_id() const {
	return "";
}

int OS::get_processor_count() const {
	return THREADING_NAMESPACE::thread::hardware_concurrency();
}

String OS::get_processor_name() const {
	return "";
}

void OS::set_has_server_feature_callback(HasServerFeatureCallback p_callback) {
	has_server_feature_callback = p_callback;
}

bool OS::has_feature(const String &p_feature) {
	// Feature tags are always lowercase for consistency.
	if (p_feature == get_identifier()) {
		return true;
	}

	if (p_feature == "movie") {
		return _writing_movie;
	}

#ifdef DEBUG_ENABLED
	if (p_feature == "debug") {
		return true;
	}
#endif // DEBUG_ENABLED

#ifdef TOOLS_ENABLED
	if (p_feature == "editor") {
		return true;
	}
	if (p_feature == "editor_hint") {
		return _in_editor;
	} else if (p_feature == "editor_runtime") {
		return !_in_editor;
	} else if (p_feature == "embedded_in_editor") {
		return _embedded_in_editor;
	}
#else
	if (p_feature == "template") {
		return true;
	}
#ifdef DEBUG_ENABLED
	if (p_feature == "template_debug") {
		return true;
	}
#else
	if (p_feature == "template_release" || p_feature == "release") {
		return true;
	}
#endif // DEBUG_ENABLED
#endif // TOOLS_ENABLED

#ifdef REAL_T_IS_DOUBLE
	if (p_feature == "double") {
		return true;
	}
#else
	if (p_feature == "single") {
		return true;
	}
#endif // REAL_T_IS_DOUBLE

	if (sizeof(void *) == 8 && p_feature == "64") {
		return true;
	}
	if (sizeof(void *) == 4 && p_feature == "32") {
		return true;
	}
#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(__i386) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
#if defined(MACOS_ENABLED)
	if (p_feature == "universal") {
		return true;
	}
#endif
	if (p_feature == "x86_64") {
		return true;
	}
#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
	if (p_feature == "x86_32") {
		return true;
	}
#endif
	if (p_feature == "x86") {
		return true;
	}
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(MACOS_ENABLED)
	if (p_feature == "universal") {
		return true;
	}
#endif
	if (p_feature == "arm64") {
		return true;
	}
#elif defined(__arm__) || defined(_M_ARM)
	if (p_feature == "arm32") {
		return true;
	}
#endif
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
#elif defined(__riscv)
#if __riscv_xlen == 8
	if (p_feature == "rv64") {
		return true;
	}
#endif
	if (p_feature == "riscv") {
		return true;
	}
#elif defined(__powerpc__)
#if defined(__powerpc64__)
	if (p_feature == "ppc64") {
		return true;
	}
#endif
	if (p_feature == "ppc") {
		return true;
	}
#elif defined(__wasm__)
#if defined(__wasm64__)
	if (p_feature == "wasm64") {
		return true;
	}
#elif defined(__wasm32__)
	if (p_feature == "wasm32") {
		return true;
	}
#endif
	if (p_feature == "wasm") {
		return true;
	}
#elif defined(__loongarch64)
	if (p_feature == "loongarch64") {
		return true;
	}
#endif

#if defined(IOS_SIMULATOR) || defined(VISIONOS_SIMULATOR)
	if (p_feature == "simulator") {
		return true;
	}
#endif

	if (p_feature == "threads") {
#ifdef THREADS_ENABLED
		return true;
#else
		return false;
#endif
	}
	if (p_feature == "nothreads") {
#ifdef THREADS_ENABLED
		return false;
#else
		return true;
#endif
	}

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

bool OS::is_sandboxed() const {
	return false;
}

void OS::set_restart_on_exit(bool p_restart, const List<String> &p_restart_arguments) {
	restart_on_exit = p_restart;
	restart_commandline = p_restart_arguments;
}

bool OS::is_restart_on_exit_set() const {
	return restart_on_exit;
}

List<String> OS::get_restart_on_exit_arguments() const {
	return List<String>(restart_commandline);
}

PackedStringArray OS::get_connected_midi_inputs() {
	if (MIDIDriver::get_singleton()) {
		return MIDIDriver::get_singleton()->get_connected_inputs();
	}

	PackedStringArray list;
	ERR_FAIL_V_MSG(list, vformat("MIDI input isn't supported on %s.", OS::get_singleton()->get_name()));
}

void OS::open_midi_inputs() {
	if (MIDIDriver::get_singleton()) {
		MIDIDriver::get_singleton()->open();
	} else {
		ERR_PRINT(vformat("MIDI input isn't supported on %s.", OS::get_singleton()->get_name()));
	}
}

void OS::close_midi_inputs() {
	if (MIDIDriver::get_singleton()) {
		MIDIDriver::get_singleton()->close();
	} else {
		ERR_PRINT(vformat("MIDI input isn't supported on %s.", OS::get_singleton()->get_name()));
	}
}

uint64_t OS::get_frame_delay(bool p_can_draw) const {
	const uint32_t frame_delay = Engine::get_singleton()->get_frame_delay();

	// Add a dynamic frame delay to decrease CPU/GPU usage. This takes the
	// previous frame time into account for a smoother result.
	uint64_t dynamic_delay = 0;
	if (is_in_low_processor_usage_mode() || !p_can_draw) {
		dynamic_delay = get_low_processor_usage_mode_sleep_usec();
	}
	const int max_fps = Engine::get_singleton()->get_max_fps();
	if (max_fps > 0 && !Engine::get_singleton()->is_editor_hint()) {
		// Override the low processor usage mode sleep delay if the target FPS is lower.
		dynamic_delay = MAX(dynamic_delay, (uint64_t)(1000000 / max_fps));
	}

	return frame_delay + dynamic_delay;
}

void OS::add_frame_delay(bool p_can_draw, bool p_wake_for_events) {
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
	const int max_fps = Engine::get_singleton()->get_max_fps();
	if (max_fps > 0 && !Engine::get_singleton()->is_editor_hint()) {
		// Override the low processor usage mode sleep delay if the target FPS is lower.
		dynamic_delay = MAX(dynamic_delay, (uint64_t)(1000000 / max_fps));
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

Error OS::setup_remote_filesystem(const String &p_server_host, int p_port, const String &p_password, String &r_project_path) {
	return default_rfs.synchronize_with_server(p_server_host, p_port, p_password, r_project_path);
}

OS::PreferredTextureFormat OS::get_preferred_texture_format() const {
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
	return PREFERRED_TEXTURE_FORMAT_ETC2_ASTC; // By rule, ARM hardware uses ETC texture compression.
#elif defined(__x86_64__) || defined(_M_X64) || defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
	return PREFERRED_TEXTURE_FORMAT_S3TC_BPTC; // By rule, X86 hardware prefers S3TC and derivatives.
#else
	return PREFERRED_TEXTURE_FORMAT_S3TC_BPTC; // Override in platform if needed.
#endif
}

void OS::set_use_benchmark(bool p_use_benchmark) {
	use_benchmark = p_use_benchmark;
}

bool OS::is_use_benchmark_set() {
	return use_benchmark;
}

void OS::set_benchmark_file(const String &p_benchmark_file) {
	benchmark_file = p_benchmark_file;
}

String OS::get_benchmark_file() {
	return benchmark_file;
}

void OS::benchmark_begin_measure(const String &p_context, const String &p_what) {
#ifdef TOOLS_ENABLED
	Pair<String, String> mark_key(p_context, p_what);
	ERR_FAIL_COND_MSG(benchmark_marks_from.has(mark_key), vformat("Benchmark key '%s:%s' already exists.", p_context, p_what));

	benchmark_marks_from[mark_key] = OS::get_singleton()->get_ticks_usec();
#endif
}
void OS::benchmark_end_measure(const String &p_context, const String &p_what) {
#ifdef TOOLS_ENABLED
	Pair<String, String> mark_key(p_context, p_what);
	ERR_FAIL_COND_MSG(!benchmark_marks_from.has(mark_key), vformat("Benchmark key '%s:%s' doesn't exist.", p_context, p_what));

	uint64_t total = OS::get_singleton()->get_ticks_usec() - benchmark_marks_from[mark_key];
	double total_f = double(total) / double(1000000);
	benchmark_marks_final[mark_key] = total_f;
#endif
}

void OS::benchmark_dump() {
#ifdef TOOLS_ENABLED
	if (!use_benchmark) {
		return;
	}

	if (!benchmark_file.is_empty()) {
		Ref<FileAccess> f = FileAccess::open(benchmark_file, FileAccess::WRITE);
		if (f.is_valid()) {
			Dictionary benchmark_marks;
			for (const KeyValue<Pair<String, String>, double> &E : benchmark_marks_final) {
				const String mark_key = vformat("[%s] %s", E.key.first, E.key.second);
				benchmark_marks[mark_key] = E.value;
			}

			Ref<JSON> json;
			json.instantiate();
			f->store_string(json->stringify(benchmark_marks, "\t", false, true));
		}
	} else {
		HashMap<String, String> results;
		for (const KeyValue<Pair<String, String>, double> &E : benchmark_marks_final) {
			if (E.key.first == "Startup" && !results.has(E.key.first)) {
				results.insert(E.key.first, "", true); // Hack to make sure "Startup" always comes first.
			}

			results[E.key.first] += vformat("\t\t- %s: %.3f msec.\n", E.key.second, (E.value * 1000));
		}

		print_line("BENCHMARK:");
		for (const KeyValue<String, String> &E : results) {
			print_line(vformat("\t[%s]\n%s", E.key, E.value));
		}
	}
#endif
}

Vector<String> OS::_read_elf_debug_id(const Ref<FileAccess> &p_f) const {
	// Read program architecture bits from class field.
	int bits = p_f->get_8() * 32;

	// Get info about the section header table.
	int64_t section_table_pos;
	int64_t section_header_size;
	if (bits == 32) {
		section_header_size = 40;
		p_f->seek(0x20);
		section_table_pos = p_f->get_32();
		p_f->seek(0x30);
	} else { // 64
		section_header_size = 64;
		p_f->seek(0x28);
		section_table_pos = p_f->get_64();
		p_f->seek(0x3c);
	}
	int num_sections = p_f->get_16();
	int string_section_idx = p_f->get_16();

	// Load the strings table.
	uint8_t *strings;
	{
		// Jump to the strings section header.
		p_f->seek(section_table_pos + string_section_idx * section_header_size);

		// Read strings data size and offset.
		int64_t string_data_pos;
		int64_t string_data_size;
		if (bits == 32) {
			p_f->seek(p_f->get_position() + 0x10);
			string_data_pos = p_f->get_32();
			string_data_size = p_f->get_32();
		} else { // 64
			p_f->seek(p_f->get_position() + 0x18);
			string_data_pos = p_f->get_64();
			string_data_size = p_f->get_64();
		}
		ERR_FAIL_COND_V(string_data_pos == 0 || string_data_size == 0, Vector<String>());

		// Read strings data.
		p_f->seek(string_data_pos);
		strings = (uint8_t *)memalloc(string_data_size);
		ERR_FAIL_NULL_V(strings, Vector<String>());
		p_f->get_buffer(strings, string_data_size);
	}

	Vector<String> ids;
	// Search for the ".note.gnu.build-id" section.
	for (int i = 0; i < num_sections; ++i) {
		int64_t section_header_pos = section_table_pos + i * section_header_size;
		p_f->seek(section_header_pos);

		uint32_t name_offset = p_f->get_32();
		if (strcmp((char *)strings + name_offset, ".note.gnu.build-id") == 0) {
			// ".note.gnu.build-id" section found.

			int64_t link_start = 0;
			if (bits == 32) {
				p_f->seek(section_header_pos + 0x10);
				link_start = p_f->get_32();
			} else { // 64
				p_f->seek(section_header_pos + 0x18);
				link_start = p_f->get_64();
			}
			p_f->seek(link_start);

			uint32_t name_size = p_f->get_32();
			uint32_t data_size = p_f->get_32();
			p_f->get_32(); // Type, skip.
			if (data_size == 0) {
				continue;
			}
			p_f->seek(p_f->get_position() + name_size);
			// 160-bit SHA-1.
			PackedByteArray id = p_f->get_buffer(data_size);
			ids.push_back(String::hex_encode_buffer(id.ptr(), id.size()));
			break;
		}
	}

	memfree(strings);
	return ids;
}

Vector<String> OS::_read_pe_debug_id(const Ref<FileAccess> &p_f) const {
	// Jump to the PE header and check the magic number.
	{
		p_f->seek(0x3c);
		uint32_t pe_pos = p_f->get_32();

		p_f->seek(pe_pos);
		uint32_t magic = p_f->get_32();
		ERR_FAIL_COND_V(magic != 0x00004550, Vector<String>());
	}

	// Process header.
	int num_sections = 0;

	int64_t header_pos = p_f->get_position();
	int64_t opt_header_pos = 0;
	{
		p_f->seek(header_pos + 2);
		num_sections = p_f->get_16();
		p_f->seek(header_pos + 16);
		uint16_t opt_header_size = p_f->get_16();
		opt_header_pos = p_f->get_position() + 2;
		// Skip rest of header + optional header to go to the section headers.
		p_f->seek(opt_header_pos + opt_header_size);
	}

	// Search for the ".buildid" section.
	int64_t dbg_off = 0;
	int64_t dbg_size = 0;
	int64_t section_table_pos = p_f->get_position();
	for (int i = 0; i < num_sections; i++) {
		int64_t section_header_pos = section_table_pos + i * 40;
		p_f->seek(section_header_pos);

		uint8_t section_name[9];
		p_f->get_buffer(section_name, 8);
		section_name[8] = 0x00;
		if (strcmp((char *)section_name, ".buildid") == 0) {
			p_f->seek(section_table_pos + i * 40 + 16);
			dbg_size = p_f->get_32();
			p_f->seek(section_table_pos + i * 40 + 20);
			dbg_off = p_f->get_32();
			break;
		}
	}

	Vector<String> ids;
	for (uint32_t pos = dbg_off; pos < dbg_off + dbg_size; pos += 28) {
		p_f->seek(pos + 12);
		uint32_t rec_type = p_f->get_32();
		uint32_t rec_size = p_f->get_32();
		p_f->get_32();
		uint32_t rec_offset = p_f->get_32();
		if (rec_type == 0x00000002) {
			// TYPE_CODEVIEW
			p_f->seek(rec_offset);
			uint32_t cv_signature = p_f->get_32();
			uint32_t cv_size = 0;
			if (cv_signature == 0x53445352) {
				// RSDS
				cv_size = rec_size - 4;
			} else {
				continue;
			}
			if (cv_size == 21) {
				// UUID + 1
				uint32_t uuid_1 = p_f->get_32();
				uint16_t uuid_2 = p_f->get_16();
				uint16_t uuid_3 = p_f->get_16();
				uint16_t uuid_4 = p_f->get_16();
				PackedByteArray uuid_5 = p_f->get_buffer(6);
				uint32_t uuid_6 = p_f->get_32();
				ids.push_back(vformat("%08x-%04x-%04x-%04x-%s-%x", uuid_1, uuid_2, uuid_3, uuid_4, String::hex_encode_buffer(uuid_5.ptr(), uuid_5.size()), uuid_6));
			} else {
				// Unknown ID type, read as is.
				PackedByteArray id = p_f->get_buffer(cv_size);
				ids.push_back(String::hex_encode_buffer(id.ptr(), id.size()));
			}
			break;
		}
	}

	p_f->close();
	return ids;
}

Vector<String> OS::_read_macho_debug_id(const Ref<FileAccess> &p_f) const {
	Vector<uint64_t> offsets;
	Vector<String> ids;

	p_f->seek(0);
	uint32_t magic = p_f->get_32();
	if (magic == 0xbebafeca || magic == 0xcafebabe || magic == 0xbfbafeca || magic == 0xcafebabf) {
		// LipO.
		bool swap = (magic == 0xbebafeca || magic == 0xbfbafeca);
		bool is64 = (magic == 0xbfbafeca || magic == 0xcafebabf);
		uint32_t fat_archs = p_f->get_32();
		if (swap) {
			fat_archs = BSWAP32(fat_archs);
		}
		for (uint32_t i = 0; i < fat_archs; i++) {
			p_f->seek(p_f->get_position() + 2 * 4);
			if (is64) {
				uint64_t fat_offset = p_f->get_64();
				if (swap) {
					fat_offset = BSWAP64(fat_offset);
				}
				offsets.push_back(fat_offset);
				p_f->seek(p_f->get_position() + 4 * 4);
			} else {
				uint32_t fat_offset = p_f->get_32();
				if (swap) {
					fat_offset = BSWAP32(fat_offset);
				}
				offsets.push_back(fat_offset);
				p_f->seek(p_f->get_position() + 2 * 4);
			}
		}
	} else {
		offsets.push_back(0);
	}
	for (uint32_t i = 0; i < offsets.size(); i++) {
		// MachO.
		p_f->seek(offsets[i]);
		magic = p_f->get_32();
		if (magic != 0xcefaedfe && magic != 0xfeedface && magic != 0xcffaedfe && magic != 0xfeedfacf) {
			continue;
		}

		// Read MachO header.
		bool swap = (magic == 0xcffaedfe || magic == 0xcefaedfe);
		bool is64 = (magic == 0xcffaedfe || magic == 0xfeedfacf);
		p_f->seek(offsets[i] + 4 * 4);
		uint32_t num_cmd = p_f->get_32();
		if (swap) {
			num_cmd = BSWAP32(num_cmd);
		}
		if (is64) { // Skip rest of header.
			p_f->seek(offsets[i] + 8 * 4);
		} else {
			p_f->seek(offsets[i] + 7 * 4);
		}

		// Read load commands.
		for (uint32_t j = 0; j < num_cmd; j++) {
			uint64_t ps = p_f->get_position();
			uint32_t cmd = p_f->get_32();
			uint32_t cmdsize = p_f->get_32();
			if (swap) {
				cmd = BSWAP32(cmd);
				cmdsize = BSWAP32(cmdsize);
			}
			if (cmd == 0x0000001b) {
				// LC_UUID.
				if (cmdsize == 24) {
					// UUID.
					uint32_t uuid_1 = BSWAP32(p_f->get_32());
					uint16_t uuid_2 = BSWAP16(p_f->get_16());
					uint16_t uuid_3 = BSWAP16(p_f->get_16());
					uint16_t uuid_4 = BSWAP16(p_f->get_16());
					PackedByteArray uuid_5 = p_f->get_buffer(6);
					ids.push_back(vformat("%08x-%04x-%04x-%04x-%s", uuid_1, uuid_2, uuid_3, uuid_4, String::hex_encode_buffer(uuid_5.ptr(), uuid_5.size())));
				} else {
					// Unknown ID type, read as is.
					PackedByteArray id = p_f->get_buffer(cmdsize - 8);
					ids.push_back(String::hex_encode_buffer(id.ptr(), id.size()));
				}
				break;
			}
			p_f->seek(ps + cmdsize);
		}
	}

	return ids;
}

Vector<String> OS::get_debug_build_id(const String &p_path) const {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V(f.is_null(), Vector<String>());

	uint32_t magic = f->get_32();
	if (magic == 0x464c457f) {
		return _read_elf_debug_id(f);
	} else if ((magic & 0xffff) == 0x5a4d) {
		return _read_pe_debug_id(f);
	} else if (magic == 0xcefaedfe || magic == 0xfeedface || magic == 0xcffaedfe || magic == 0xfeedfacf || magic == 0xbebafeca || magic == 0xcafebabe || magic == 0xbfbafeca || magic == 0xcafebabf) {
		return _read_macho_debug_id(f);
	} else {
		ERR_FAIL_V_MSG(Vector<String>(), "Unsupported executable format.");
	}
}

OS::OS() {
	singleton = this;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));
}

OS::~OS() {
	memdelete(_logger);
	_logger = nullptr;
	OS::target_ticks = 0;
	singleton = nullptr;
}
