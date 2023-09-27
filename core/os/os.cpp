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

#include "core/io/json.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/input.h"
#include "core/os/midi_driver.h"
#include "core/project_settings.h"
#include "core/version_generated.gen.h"
#include "servers/audio_server.h"

#include <stdarg.h>
#include <thread>

OS *OS::singleton = nullptr;
uint64_t OS::target_ticks = 0;

OS *OS::get_singleton() {
	return singleton;
}

uint32_t OS::get_ticks_msec() const {
	return get_ticks_usec() / 1000;
}

String OS::get_iso_date_time(bool local) const {
	OS::Date date = get_date(local);
	OS::Time time = get_time(local);

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

	return itos(date.year).pad_zeros(2) + "-" +
			itos(date.month).pad_zeros(2) + "-" +
			itos(date.day).pad_zeros(2) + "T" +
			itos(time.hour).pad_zeros(2) + ":" +
			itos(time.min).pad_zeros(2) + ":" +
			itos(time.sec).pad_zeros(2) + timezone;
}

uint64_t OS::get_splash_tick_msec() const {
	return _msec_splash;
}
uint64_t OS::get_unix_time() const {
	return 0;
};
uint64_t OS::get_system_time_secs() const {
	return 0;
}
uint64_t OS::get_system_time_msecs() const {
	return 0;
}
double OS::get_subsecond_unix_time() const {
	return 0.0;
}

void OS::_set_logger(CompositeLogger *p_logger) {
	if (_logger) {
		memdelete(_logger);
	}
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

void OS::print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, Logger::ErrorType p_type) {
	_logger->log_error(p_function, p_file, p_line, p_code, p_rationale, p_type);
}

void OS::print(const char *p_format, ...) {
	va_list argp;
	va_start(argp, p_format);

	_logger->logv(p_format, argp, false);

	va_end(argp);
};

void OS::printerr(const char *p_format, ...) {
	va_list argp;
	va_start(argp, p_format);

	_logger->logv(p_format, argp, true);

	va_end(argp);
};

void OS::set_keep_screen_on(bool p_enabled) {
	_keep_screen_on = p_enabled;
}

bool OS::is_keep_screen_on() const {
	return _keep_screen_on;
}

void OS::set_low_processor_usage_mode(bool p_enabled) {
	low_processor_usage_mode = p_enabled;
}

bool OS::is_in_low_processor_usage_mode() const {
	return low_processor_usage_mode;
}

void OS::set_update_vital_only(bool p_enabled) {
	_update_vital_only = p_enabled;
}

void OS::set_low_processor_usage_mode_sleep_usec(int p_usec) {
	low_processor_usage_mode_sleep_usec = p_usec;
}

int OS::get_low_processor_usage_mode_sleep_usec() const {
	return low_processor_usage_mode_sleep_usec;
}

void OS::set_clipboard(const String &p_text) {
	_local_clipboard = p_text;
}

String OS::get_clipboard() const {
	return _local_clipboard;
}

bool OS::has_clipboard() const {
	return !get_clipboard().empty();
}

void OS::set_clipboard_primary(const String &p_text) {
	_primary_clipboard = p_text;
}

String OS::get_clipboard_primary() const {
	return _primary_clipboard;
}

String OS::get_executable_path() const {
	return _execpath;
}

int OS::get_process_id() const {
	return -1;
};

bool OS::is_stdout_verbose() const {
	return _verbose_stdout;
}

bool OS::is_stdout_debug_enabled() const {
	return _debug_stdout;
}

void OS::dump_memory_to_file(const char *p_file) {
	//Memory::dump_static_mem_to_file(p_file);
}

static FileAccess *_OSPRF = nullptr;

static void _OS_printres(Object *p_obj) {
	Resource *res = Object::cast_to<Resource>(p_obj);
	if (!res) {
		return;
	}

	String str = vformat("%s - %s - %s", res->to_string(), res->get_name(), res->get_path());
	if (_OSPRF) {
		_OSPRF->store_line(str);
	} else {
		print_line(str);
	}
}

bool OS::has_virtual_keyboard() const {
	return false;
}

void OS::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect, VirtualKeyboardType p_type, int p_max_input_length, int p_cursor_start, int p_cursor_end) {
}

void OS::hide_virtual_keyboard() {
}

int OS::get_virtual_keyboard_height() const {
	return 0;
}

uint32_t OS::keyboard_get_scancode_from_physical(uint32_t p_scancode) const {
	return p_scancode;
}

void OS::set_cursor_shape(CursorShape p_shape) {
}

OS::CursorShape OS::get_cursor_shape() const {
	return CURSOR_ARROW;
}

void OS::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
}

void OS::print_all_resources(String p_to_file) {
	ERR_FAIL_COND(p_to_file != "" && _OSPRF);
	if (p_to_file != "") {
		Error err;
		_OSPRF = FileAccess::open(p_to_file, FileAccess::WRITE, &err);
		if (err != OK) {
			_OSPRF = nullptr;
			ERR_FAIL_MSG("Can't print all resources to file: " + String(p_to_file) + ".");
		}
	}

	ObjectDB::debug_objects(_OS_printres);

	if (p_to_file != "") {
		if (_OSPRF) {
			memdelete(_OSPRF);
		}
		_OSPRF = nullptr;
	}
}

void OS::print_resources_in_use(bool p_short) {
	ResourceCache::dump(nullptr, p_short);
}

void OS::dump_resources_to_file(const char *p_file) {
	ResourceCache::dump(p_file);
}

void OS::set_no_window_mode(bool p_enable) {
	_no_window = p_enable;
}

bool OS::is_no_window_mode_enabled() const {
	return _no_window;
}

int OS::get_exit_code() const {
	return _exit_code;
}
void OS::set_exit_code(int p_code) {
	_exit_code = p_code;
	_is_custom_exit_code = true;
}

bool OS::is_custom_exit_code() {
	return _is_custom_exit_code;
}

String OS::get_locale() const {
	return "en";
}

// Non-virtual helper to extract the 2 or 3-letter language code from
// `get_locale()` in a way that's consistent for all platforms.
String OS::get_locale_language() const {
	return get_locale().left(3).replace("_", "");
}

// Embedded PCK offset.
uint64_t OS::get_embedded_pck_offset() const {
	return 0;
}

// Helper function to ensure that a dir name/path will be valid on the OS
String OS::get_safe_dir_name(const String &p_dir_name, bool p_allow_dir_separator) const {
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

// Path to data, config, cache, etc. OS-specific folders

// Get properly capitalized engine name for system paths
String OS::get_godot_dir_name() const {
	// Default to lowercase, so only override when different case is needed
	return String(VERSION_SHORT_NAME).to_lower();
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

// Path to macOS .app bundle resources
String OS::get_bundle_resource_dir() const {
	return ".";
};

// Path to macOS .app bundle embedded icon
String OS::get_bundle_icon_path() const {
	return String();
}

// OS specific path for user://
String OS::get_user_data_dir() const {
	return ".";
};

// Absolute path to res://
String OS::get_resource_dir() const {
	return ProjectSettings::get_singleton()->get_resource_path();
}

// Access system-specific dirs like Documents, Downloads, etc.
String OS::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
	return ".";
}

Error OS::shell_open(String p_uri) {
	return ERR_UNAVAILABLE;
};

// implement these with the canvas?
Error OS::dialog_show(String p_title, String p_description, Vector<String> p_buttons, Object *p_obj, String p_callback) {
	while (true) {
		print("%ls\n--------\n%ls\n", p_title.c_str(), p_description.c_str());
		for (int i = 0; i < p_buttons.size(); i++) {
			if (i > 0) {
				print(", ");
			}
			print("%i=%ls", i + 1, p_buttons[i].c_str());
		};
		print("\n");
		String res = get_stdin_string().strip_edges();
		if (!res.is_numeric()) {
			continue;
		}
		int n = res.to_int();
		if (n < 0 || n >= p_buttons.size()) {
			continue;
		}
		if (p_obj && p_callback != "") {
			p_obj->call_deferred(p_callback, n);
		}
		break;
	};
	return OK;
};

Error OS::dialog_input_text(String p_title, String p_description, String p_partial, Object *p_obj, String p_callback) {
	ERR_FAIL_COND_V(!p_obj, FAILED);
	ERR_FAIL_COND_V(p_callback == "", FAILED);
	print("%ls\n---------\n%ls\n[%ls]:\n", p_title.c_str(), p_description.c_str(), p_partial.c_str());

	String res = get_stdin_string().strip_edges();
	bool success = true;
	if (res == "") {
		res = p_partial;
	};

	p_obj->call_deferred(p_callback, success, res);

	return OK;
};

uint64_t OS::get_static_memory_usage() const {
	return Memory::get_mem_usage();
}
uint64_t OS::get_dynamic_memory_usage() const {
	return MemoryPool::total_memory;
}

uint64_t OS::get_static_memory_peak_usage() const {
	return Memory::get_mem_max_usage();
}

Error OS::set_cwd(const String &p_cwd) {
	return ERR_CANT_OPEN;
}

bool OS::has_touchscreen_ui_hint() const {
	//return false;
	return Input::get_singleton() && Input::get_singleton()->is_emulating_touch_from_mouse();
}

uint64_t OS::get_free_static_memory() const {
	return Memory::get_mem_available();
}

void OS::yield() {
}

void OS::set_screen_orientation(ScreenOrientation p_orientation) {
	_orientation = p_orientation;
}

OS::ScreenOrientation OS::get_screen_orientation() const {
	return (OS::ScreenOrientation)_orientation;
}

// Internal helper function that returns the screen orientation enum value from a string
// (generally coming from the Project Settings).
// This is required to keep compatibility with existing projects.
OS::ScreenOrientation OS::get_screen_orientation_from_string(const String &p_orientation) const {
	if (p_orientation == "portrait") {
		return OS::SCREEN_PORTRAIT;
	} else if (p_orientation == "reverse_landscape") {
		return OS::SCREEN_REVERSE_LANDSCAPE;
	} else if (p_orientation == "reverse_portrait") {
		return OS::SCREEN_REVERSE_PORTRAIT;
	} else if (p_orientation == "sensor_landscape") {
		return OS::SCREEN_SENSOR_LANDSCAPE;
	} else if (p_orientation == "sensor_portrait") {
		return OS::SCREEN_SENSOR_PORTRAIT;
	} else if (p_orientation == "sensor") {
		return OS::SCREEN_SENSOR;
	}

	return OS::SCREEN_LANDSCAPE;
}

void OS::ensure_user_data_dir() {
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

void OS::set_native_icon(const String &p_filename) {
}

void OS::set_icon(const Ref<Image> &p_icon) {
}

String OS::get_model_name() const {
	return "GenericDevice";
}

void OS::set_cmdline(const char *p_execpath, const List<String> &p_args) {
	_execpath = String::utf8(p_execpath);
	_cmdline = p_args;
};

void OS::release_rendering_thread() {
}

void OS::make_rendering_thread() {
}

void OS::swap_buffers() {
}

String OS::get_unique_id() const {
	return "";
}

int OS::get_processor_count() const {
	return std::thread::hardware_concurrency();
}

String OS::get_processor_name() const {
	return "";
}

Error OS::native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {
	return FAILED;
}

bool OS::native_video_is_playing() const {
	return false;
}

void OS::native_video_pause() {
}

void OS::native_video_unpause() {
}

void OS::native_video_stop() {
}

void OS::set_mouse_mode(MouseMode p_mode) {
}

bool OS::can_use_threads() const {
#ifdef NO_THREADS
	return false;
#else
	return true;
#endif
}

bool OS::tts_is_speaking() const {
	WARN_PRINT("TTS is not supported by this platform.");
	return false;
}

bool OS::tts_is_paused() const {
	WARN_PRINT("TTS is not supported by this platform.");
	return false;
}

void OS::tts_pause() {
	WARN_PRINT("TTS is not supported by this platformr.");
}

void OS::tts_resume() {
	WARN_PRINT("TTS is not supported by this platform.");
}

Array OS::tts_get_voices() const {
	WARN_PRINT("TTS is not supported by this platform.");
	return Array();
}

PoolStringArray OS::tts_get_voices_for_language(const String &p_language) const {
	PoolStringArray ret;
	Array voices = tts_get_voices();
	for (int i = 0; i < voices.size(); i++) {
		const Dictionary &voice = voices[i];
		if (voice.has("id") && voice.has("language") && voice["language"].operator String().begins_with(p_language)) {
			ret.push_back(voice["id"]);
		}
	}
	return ret;
}

void OS::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	WARN_PRINT("TTS is not supported by this platform.");
}

void OS::tts_stop() {
	WARN_PRINT("TTS is not supported by this platform.");
}

void OS::tts_set_utterance_callback(TTSUtteranceEvent p_event, Object *p_object, const StringName &p_callback) {
	ERR_FAIL_INDEX(p_event, OS::TTS_UTTERANCE_MAX);
	utterance_callback[p_event].object = p_object;
	utterance_callback[p_event].cb_name = p_callback;
}

void OS::tts_post_utterance_event(TTSUtteranceEvent p_event, int p_id, int p_pos) {
	ERR_FAIL_INDEX(p_event, OS::TTS_UTTERANCE_MAX);
	switch (p_event) {
		case OS::TTS_UTTERANCE_STARTED:
		case OS::TTS_UTTERANCE_ENDED:
		case OS::TTS_UTTERANCE_CANCELED: {
			if (utterance_callback[p_event].object != nullptr) {
				utterance_callback[p_event].object->call_deferred(utterance_callback[p_event].cb_name, p_id);
			}
		} break;
		case OS::TTS_UTTERANCE_BOUNDARY: {
			if (utterance_callback[p_event].object != nullptr) {
				utterance_callback[p_event].object->call_deferred(utterance_callback[p_event].cb_name, p_pos, p_id);
			}
		} break;
		default:
			break;
	}
}

OS::MouseMode OS::get_mouse_mode() const {
	return MOUSE_MODE_VISIBLE;
}

OS::LatinKeyboardVariant OS::get_latin_keyboard_variant() const {
	return LATIN_KEYBOARD_QWERTY;
}

int OS::keyboard_get_layout_count() const {
	return 0;
}

int OS::keyboard_get_current_layout() const {
	return -1;
}

void OS::keyboard_set_current_layout(int p_index) {}

String OS::keyboard_get_layout_language(int p_index) const {
	return "";
}

String OS::keyboard_get_layout_name(int p_index) const {
	return "";
}

bool OS::is_joy_known(int p_device) {
	return true;
}

String OS::get_joy_guid(int p_device) const {
	return "Default Joypad";
}

void OS::set_context(int p_context) {
}

OS::SwitchVSyncCallbackInThread OS::switch_vsync_function = nullptr;

void OS::set_use_vsync(bool p_enable) {
	_use_vsync = p_enable;
	if (switch_vsync_function) { //if a function was set, use function
		switch_vsync_function(p_enable);
	} else { //otherwise just call here
		_set_use_vsync(p_enable);
	}
}

bool OS::is_vsync_enabled() const {
	return _use_vsync;
}

void OS::set_vsync_via_compositor(bool p_enable) {
	_vsync_via_compositor = p_enable;
}

bool OS::is_vsync_via_compositor_enabled() const {
	return _vsync_via_compositor;
}

void OS::set_delta_smoothing(bool p_enabled) {
	_delta_smoothing_enabled = p_enabled;
}

bool OS::is_delta_smoothing_enabled() const {
	return _delta_smoothing_enabled;
}

OS::PowerState OS::get_power_state() {
	return POWERSTATE_UNKNOWN;
}
int OS::get_power_seconds_left() {
	return -1;
}
int OS::get_power_percent_left() {
	return -1;
}

void OS::set_has_server_feature_callback(HasServerFeatureCallback p_callback) {
	has_server_feature_callback = p_callback;
}

bool OS::has_feature(const String &p_feature) {
	if (p_feature == get_name()) {
		return true;
	}
#ifdef DEBUG_ENABLED
	if (p_feature == "debug") {
		return true;
	}
#else
	if (p_feature == "release") {
		return true;
	}
#endif
#ifdef TOOLS_ENABLED
	if (p_feature == "editor") {
		return true;
	}
#else
	if (p_feature == "standalone") {
		return true;
	}
#endif

	if (sizeof(void *) == 8 && p_feature == "64") {
		return true;
	}
	if (sizeof(void *) == 4 && p_feature == "32") {
		return true;
	}
#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
	if (p_feature == "x86_64") {
		return true;
	}
#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
	if (p_feature == "x86_32") {
		return true;
	}
	if (p_feature == "x86") {
		return true;
	}
#elif defined(__aarch64__) || defined(_M_ARM64)
	if (p_feature == "arm64") {
		return true;
	}
#elif defined(__arm__) || defined(_M_ARM)
	if (p_feature == "arm32") {
		return true;
	}
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

void OS::center_window() {
	if (is_window_fullscreen()) {
		return;
	}

	Point2 sp = get_screen_position(get_current_screen());
	Size2 scr = get_screen_size(get_current_screen());
	Size2 wnd = get_real_window_size();

	int x = sp.width + (scr.width - wnd.width) / 2;
	int y = sp.height + (scr.height - wnd.height) / 2;

	set_window_position(Vector2(x, y));
}

int OS::get_video_driver_count() const {
	return 2;
}

const char *OS::get_video_driver_name(int p_driver) const {
	switch (p_driver) {
		case VIDEO_DRIVER_GLES2:
			return "GLES2";
		case VIDEO_DRIVER_GLES3:
		default:
			return "GLES3";
	}
}

bool OS::is_offscreen_gl_available() const {
	return false;
}

void OS::set_offscreen_gl_current(bool p_current) {}

int OS::get_audio_driver_count() const {
	return AudioDriverManager::get_driver_count();
}

const char *OS::get_audio_driver_name(int p_driver) const {
	AudioDriver *driver = AudioDriverManager::get_driver(p_driver);
	ERR_FAIL_COND_V_MSG(!driver, "", "Cannot get audio driver at index '" + itos(p_driver) + "'.");
	return AudioDriverManager::get_driver(p_driver)->get_name();
}

void OS::set_restart_on_exit(bool p_restart, const List<String> &p_restart_arguments) {
	restart_on_exit = p_restart;
	restart_commandline = p_restart_arguments;
}

bool OS::is_restart_on_exit_set() const {
	return restart_on_exit;
}

List<String> OS::get_restart_on_exit_arguments() const {
	return restart_commandline;
}

PoolStringArray OS::get_connected_midi_inputs() {
	if (MIDIDriver::get_singleton()) {
		return MIDIDriver::get_singleton()->get_connected_inputs();
	}

	PoolStringArray list;
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

void OS::add_frame_delay(bool p_can_draw) {
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

void OS::benchmark_begin_measure(const String &p_what) {
#ifdef TOOLS_ENABLED
	start_benchmark_from[p_what] = OS::get_singleton()->get_ticks_usec();
#endif
}
void OS::benchmark_end_measure(const String &p_what) {
#ifdef TOOLS_ENABLED
	uint64_t total = OS::get_singleton()->get_ticks_usec() - start_benchmark_from[p_what];
	double total_f = double(total) / double(1000000);

	startup_benchmark_json[p_what] = total_f;
#endif
}

void OS::benchmark_dump() {
#ifdef TOOLS_ENABLED
	if (!use_benchmark) {
		return;
	}
	if (!benchmark_file.empty()) {
		FileAccess *f = FileAccess::open(benchmark_file, FileAccess::WRITE);
		if (f) {
			f->store_string(JSON::print(startup_benchmark_json, "\t", false));
		}
	} else {
		List<Variant> keys;
		startup_benchmark_json.get_key_list(&keys);
		print_line("BENCHMARK:");
		for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
			Variant &K = E->get();
			print_line("\t-" + K.operator String() + ": " + startup_benchmark_json[K] + " sec.");
		}
	}
#endif
}

OS::OS() {
	void *volatile stack_bottom;

	restart_on_exit = false;
	singleton = this;
	_keep_screen_on = true; // set default value to true, because this had been true before godot 2.0.
	low_processor_usage_mode = false;
	low_processor_usage_mode_sleep_usec = 10000;
	_update_vital_only = false;
	_update_pending = false;
	_verbose_stdout = false;
	_debug_stdout = false;
	_no_window = false;
	_exit_code = 0;
	_orientation = SCREEN_LANDSCAPE;
	_delta_smoothing_enabled = false;

	_render_thread_mode = RENDER_THREAD_SAFE;

	_allow_hidpi = false;
	_allow_layered = false;
	_stack_bottom = (void *)(&stack_bottom);

	_logger = nullptr;

	has_server_feature_callback = nullptr;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));
}

OS::~OS() {
	memdelete(_logger);
	singleton = nullptr;
}
