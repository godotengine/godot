/*************************************************************************/
/*  os.cpp                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "os.h"

#include "dir_access.h"
#include "global_config.h"
#include "input.h"
#include "os/file_access.h"

#include <stdarg.h>

OS *OS::singleton = NULL;

OS *OS::get_singleton() {

	return singleton;
}

uint32_t OS::get_ticks_msec() const {
	return get_ticks_usec() / 1000;
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
void OS::debug_break(){

	// something
};

void OS::print_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, ErrorType p_type) {

	const char *err_type;
	switch (p_type) {
		case ERR_ERROR: err_type = "**ERROR**"; break;
		case ERR_WARNING: err_type = "**WARNING**"; break;
		case ERR_SCRIPT: err_type = "**SCRIPT ERROR**"; break;
		case ERR_SHADER: err_type = "**SHADER ERROR**"; break;
	}

	if (p_rationale && *p_rationale)
		print("%s: %s\n ", err_type, p_rationale);
	print("%s: At: %s:%i:%s() - %s\n", err_type, p_file, p_line, p_function, p_code);
}

void OS::print(const char *p_format, ...) {

	va_list argp;
	va_start(argp, p_format);

	vprint(p_format, argp);

	va_end(argp);
};

void OS::printerr(const char *p_format, ...) {

	va_list argp;
	va_start(argp, p_format);

	vprint(p_format, argp, true);

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

void OS::set_clipboard(const String &p_text) {

	_local_clipboard = p_text;
}
String OS::get_clipboard() const {

	return _local_clipboard;
}

String OS::get_executable_path() const {

	return _execpath;
}

int OS::get_process_ID() const {

	return -1;
};

bool OS::is_stdout_verbose() const {

	return _verbose_stdout;
}

void OS::set_last_error(const char *p_error) {

	GLOBAL_LOCK_FUNCTION
	if (p_error == NULL)
		p_error = "Unknown Error";

	if (last_error)
		memfree(last_error);
	last_error = NULL;
	int len = 0;
	while (p_error[len++])
		;

	last_error = (char *)memalloc(len);
	for (int i = 0; i < len; i++)
		last_error[i] = p_error[i];
}

const char *OS::get_last_error() const {
	GLOBAL_LOCK_FUNCTION
	return last_error ? last_error : "";
}

void OS::dump_memory_to_file(const char *p_file) {

	//Memory::dump_static_mem_to_file(p_file);
}

static FileAccess *_OSPRF = NULL;

static void _OS_printres(Object *p_obj) {

	Resource *res = p_obj->cast_to<Resource>();
	if (!res)
		return;

	String str = itos(res->get_instance_ID()) + String(res->get_class()) + ":" + String(res->get_name()) + " - " + res->get_path();
	if (_OSPRF)
		_OSPRF->store_line(str);
	else
		print_line(str);
}

bool OS::has_virtual_keyboard() const {

	return false;
}

void OS::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect) {
}

void OS::hide_virtual_keyboard() {
}

void OS::print_all_resources(String p_to_file) {

	ERR_FAIL_COND(p_to_file != "" && _OSPRF);
	if (p_to_file != "") {

		Error err;
		_OSPRF = FileAccess::open(p_to_file, FileAccess::WRITE, &err);
		if (err != OK) {
			_OSPRF = NULL;
			ERR_FAIL_COND(err != OK);
		}
	}

	ObjectDB::debug_objects(_OS_printres);

	if (p_to_file != "") {

		if (_OSPRF)
			memdelete(_OSPRF);
		_OSPRF = NULL;
	}
}

void OS::print_resources_in_use(bool p_short) {

	ResourceCache::dump(NULL, p_short);
}

void OS::dump_resources_to_file(const char *p_file) {

	ResourceCache::dump(p_file);
}

void OS::clear_last_error() {

	GLOBAL_LOCK_FUNCTION
	if (last_error)
		memfree(last_error);
	last_error = NULL;
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
}

String OS::get_locale() const {

	return "en";
}

String OS::get_resource_dir() const {

	return GlobalConfig::get_singleton()->get_resource_path();
}

String OS::get_system_dir(SystemDir p_dir) const {

	return ".";
}

String OS::get_safe_application_name() const {
	String an = GlobalConfig::get_singleton()->get("application/name");
	Vector<String> invalid_char = String("\\ / : * ? \" < > |").split(" ");
	for (int i = 0; i < invalid_char.size(); i++) {
		an = an.replace(invalid_char[i], "-");
	}
	return an;
}

String OS::get_data_dir() const {

	return ".";
};

Error OS::shell_open(String p_uri) {
	return ERR_UNAVAILABLE;
};

// implement these with the canvas?
Error OS::dialog_show(String p_title, String p_description, Vector<String> p_buttons, Object *p_obj, String p_callback) {

	while (true) {

		print("%ls\n--------\n%ls\n", p_title.c_str(), p_description.c_str());
		for (int i = 0; i < p_buttons.size(); i++) {
			if (i > 0) print(", ");
			print("%i=%ls", i + 1, p_buttons[i].c_str());
		};
		print("\n");
		String res = get_stdin_string().strip_edges();
		if (!res.is_numeric())
			continue;
		int n = res.to_int();
		if (n < 0 || n >= p_buttons.size())
			continue;
		if (p_obj && p_callback != "")
			p_obj->call_deferred(p_callback, n);
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

int OS::get_static_memory_usage() const {

	return Memory::get_mem_usage();
}
int OS::get_dynamic_memory_usage() const {

	return MemoryPool::total_memory;
}

int OS::get_static_memory_peak_usage() const {

	return Memory::get_mem_max_usage();
}

Error OS::set_cwd(const String &p_cwd) {

	return ERR_CANT_OPEN;
}

bool OS::has_touchscreen_ui_hint() const {

	//return false;
	return Input::get_singleton() && Input::get_singleton()->is_emulating_touchscreen();
}

int OS::get_free_static_memory() const {

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

void OS::_ensure_data_dir() {

	String dd = get_data_dir();
	DirAccess *da = DirAccess::open(dd);
	if (da) {
		memdelete(da);
		return;
	}

	da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = da->make_dir_recursive(dd);
	if (err != OK) {
		ERR_EXPLAIN("Error attempting to create data dir: " + dd);
	}
	ERR_FAIL_COND(err != OK);

	memdelete(da);
}

void OS::set_icon(const Image &p_icon) {
}

String OS::get_model_name() const {

	return "GenericDevice";
}

void OS::set_cmdline(const char *p_execpath, const List<String> &p_args) {

	_execpath = p_execpath;
	_cmdline = p_args;
};

void OS::release_rendering_thread() {
}

void OS::make_rendering_thread() {
}

void OS::swap_buffers() {
}

String OS::get_unique_ID() const {

	ERR_FAIL_V("");
}

int OS::get_processor_count() const {

	return 1;
}

Error OS::native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {

	return FAILED;
};

bool OS::native_video_is_playing() const {

	return false;
};

void OS::native_video_pause(){

};

void OS::native_video_unpause(){

};

void OS::native_video_stop(){

};

void OS::set_mouse_mode(MouseMode p_mode) {
}

bool OS::can_use_threads() const {

#ifdef NO_THREADS
	return false;
#else
	return true;
#endif
}

OS::MouseMode OS::get_mouse_mode() const {

	return MOUSE_MODE_VISIBLE;
}

OS::LatinKeyboardVariant OS::get_latin_keyboard_variant() const {

	return LATIN_KEYBOARD_QWERTY;
}

bool OS::is_joy_known(int p_device) {
	return true;
}

String OS::get_joy_guid(int p_device) const {
	return "Default Joypad";
}

void OS::set_context(int p_context) {
}
void OS::set_use_vsync(bool p_enable) {
}

bool OS::is_vsync_enabled() const {

	return true;
}

PowerState OS::get_power_state() {
	return POWERSTATE_UNKNOWN;
}
int OS::get_power_seconds_left() {
	return -1;
}
int OS::get_power_percent_left() {
	return -1;
}

OS::OS() {
	last_error = NULL;
	singleton = this;
	_keep_screen_on = true; // set default value to true, because this had been true before godot 2.0.
	low_processor_usage_mode = false;
	_verbose_stdout = false;
	_no_window = false;
	_exit_code = 0;
	_orientation = SCREEN_LANDSCAPE;

	_render_thread_mode = RENDER_THREAD_SAFE;

	_allow_hidpi = true;
	Math::seed(1234567);
}

OS::~OS() {

	singleton = NULL;
}
