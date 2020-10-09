/*************************************************************************/
/*  core_bind.cpp                                                        */
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

#include "core_bind.h"

#include "core/crypto/crypto_core.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/file_access_compressed.h"
#include "core/io/file_access_encrypted.h"
#include "core/io/json.h"
#include "core/io/marshalls.h"
#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/project_settings.h"

/**
 *  Time constants borrowed from loc_time.h
 */
#define EPOCH_YR 1970 /* EPOCH = Jan 1 1970 00:00:00 */
#define SECS_DAY (24L * 60L * 60L)
#define LEAPYEAR(year) (!((year) % 4) && (((year) % 100) || !((year) % 400)))
#define YEARSIZE(year) (LEAPYEAR(year) ? 366 : 365)
#define SECOND_KEY "second"
#define MINUTE_KEY "minute"
#define HOUR_KEY "hour"
#define DAY_KEY "day"
#define MONTH_KEY "month"
#define YEAR_KEY "year"
#define WEEKDAY_KEY "weekday"
#define DST_KEY "dst"

/// Table of number of days in each month (for regular year and leap year)
static const unsigned int MONTH_DAYS_TABLE[2][12] = {
	{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
	{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

////// _ResourceLoader //////

_ResourceLoader *_ResourceLoader::singleton = nullptr;

Error _ResourceLoader::load_threaded_request(const String &p_path, const String &p_type_hint, bool p_use_sub_threads) {
	return ResourceLoader::load_threaded_request(p_path, p_type_hint, p_use_sub_threads);
}

_ResourceLoader::ThreadLoadStatus _ResourceLoader::load_threaded_get_status(const String &p_path, Array r_progress) {
	float progress = 0;
	ResourceLoader::ThreadLoadStatus tls = ResourceLoader::load_threaded_get_status(p_path, &progress);
	r_progress.resize(1);
	r_progress[0] = progress;
	return (ThreadLoadStatus)tls;
}

RES _ResourceLoader::load_threaded_get(const String &p_path) {
	Error error;
	RES res = ResourceLoader::load_threaded_get(p_path, &error);
	return res;
}

RES _ResourceLoader::load(const String &p_path, const String &p_type_hint, bool p_no_cache) {
	Error err = OK;
	RES ret = ResourceLoader::load(p_path, p_type_hint, p_no_cache, &err);

	ERR_FAIL_COND_V_MSG(err != OK, ret, "Error loading resource: '" + p_path + "'.");
	return ret;
}

Vector<String> _ResourceLoader::get_recognized_extensions_for_type(const String &p_type) {
	List<String> exts;
	ResourceLoader::get_recognized_extensions_for_type(p_type, &exts);
	Vector<String> ret;
	for (List<String>::Element *E = exts.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}

	return ret;
}

void _ResourceLoader::set_abort_on_missing_resources(bool p_abort) {
	ResourceLoader::set_abort_on_missing_resources(p_abort);
}

PackedStringArray _ResourceLoader::get_dependencies(const String &p_path) {
	List<String> deps;
	ResourceLoader::get_dependencies(p_path, &deps);

	PackedStringArray ret;
	for (List<String>::Element *E = deps.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}

	return ret;
}

bool _ResourceLoader::has_cached(const String &p_path) {
	String local_path = ProjectSettings::get_singleton()->localize_path(p_path);
	return ResourceCache::has(local_path);
}

bool _ResourceLoader::exists(const String &p_path, const String &p_type_hint) {
	return ResourceLoader::exists(p_path, p_type_hint);
}

void _ResourceLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_threaded_request", "path", "type_hint", "use_sub_threads"), &_ResourceLoader::load_threaded_request, DEFVAL(""), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("load_threaded_get_status", "path", "progress"), &_ResourceLoader::load_threaded_get_status, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("load_threaded_get", "path"), &_ResourceLoader::load_threaded_get);

	ClassDB::bind_method(D_METHOD("load", "path", "type_hint", "no_cache"), &_ResourceLoader::load, DEFVAL(""), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_recognized_extensions_for_type", "type"), &_ResourceLoader::get_recognized_extensions_for_type);
	ClassDB::bind_method(D_METHOD("set_abort_on_missing_resources", "abort"), &_ResourceLoader::set_abort_on_missing_resources);
	ClassDB::bind_method(D_METHOD("get_dependencies", "path"), &_ResourceLoader::get_dependencies);
	ClassDB::bind_method(D_METHOD("has_cached", "path"), &_ResourceLoader::has_cached);
	ClassDB::bind_method(D_METHOD("exists", "path", "type_hint"), &_ResourceLoader::exists, DEFVAL(""));

	BIND_ENUM_CONSTANT(THREAD_LOAD_INVALID_RESOURCE);
	BIND_ENUM_CONSTANT(THREAD_LOAD_IN_PROGRESS);
	BIND_ENUM_CONSTANT(THREAD_LOAD_FAILED);
	BIND_ENUM_CONSTANT(THREAD_LOAD_LOADED);
}

////// _ResourceSaver //////

Error _ResourceSaver::save(const String &p_path, const RES &p_resource, SaverFlags p_flags) {
	ERR_FAIL_COND_V_MSG(p_resource.is_null(), ERR_INVALID_PARAMETER, "Can't save empty resource to path '" + String(p_path) + "'.");
	return ResourceSaver::save(p_path, p_resource, p_flags);
}

Vector<String> _ResourceSaver::get_recognized_extensions(const RES &p_resource) {
	ERR_FAIL_COND_V_MSG(p_resource.is_null(), Vector<String>(), "It's not a reference to a valid Resource object.");
	List<String> exts;
	ResourceSaver::get_recognized_extensions(p_resource, &exts);
	Vector<String> ret;
	for (List<String>::Element *E = exts.front(); E; E = E->next()) {
		ret.push_back(E->get());
	}
	return ret;
}

_ResourceSaver *_ResourceSaver::singleton = nullptr;

void _ResourceSaver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("save", "path", "resource", "flags"), &_ResourceSaver::save, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_recognized_extensions", "type"), &_ResourceSaver::get_recognized_extensions);

	BIND_ENUM_CONSTANT(FLAG_RELATIVE_PATHS);
	BIND_ENUM_CONSTANT(FLAG_BUNDLE_RESOURCES);
	BIND_ENUM_CONSTANT(FLAG_CHANGE_PATH);
	BIND_ENUM_CONSTANT(FLAG_OMIT_EDITOR_PROPERTIES);
	BIND_ENUM_CONSTANT(FLAG_SAVE_BIG_ENDIAN);
	BIND_ENUM_CONSTANT(FLAG_COMPRESS);
	BIND_ENUM_CONSTANT(FLAG_REPLACE_SUBRESOURCE_PATHS);
}

////// _OS //////

PackedStringArray _OS::get_connected_midi_inputs() {
	return OS::get_singleton()->get_connected_midi_inputs();
}

void _OS::open_midi_inputs() {
	OS::get_singleton()->open_midi_inputs();
}

void _OS::close_midi_inputs() {
	OS::get_singleton()->close_midi_inputs();
}

void _OS::set_use_file_access_save_and_swap(bool p_enable) {
	FileAccess::set_backup_save(p_enable);
}

void _OS::set_low_processor_usage_mode(bool p_enabled) {
	OS::get_singleton()->set_low_processor_usage_mode(p_enabled);
}

bool _OS::is_in_low_processor_usage_mode() const {
	return OS::get_singleton()->is_in_low_processor_usage_mode();
}

void _OS::set_low_processor_usage_mode_sleep_usec(int p_usec) {
	OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(p_usec);
}

int _OS::get_low_processor_usage_mode_sleep_usec() const {
	return OS::get_singleton()->get_low_processor_usage_mode_sleep_usec();
}

String _OS::get_executable_path() const {
	return OS::get_singleton()->get_executable_path();
}

Error _OS::shell_open(String p_uri) {
	if (p_uri.begins_with("res://")) {
		WARN_PRINT("Attempting to open an URL with the \"res://\" protocol. Use `ProjectSettings.globalize_path()` to convert a Godot-specific path to a system path before opening it with `OS.shell_open()`.");
	} else if (p_uri.begins_with("user://")) {
		WARN_PRINT("Attempting to open an URL with the \"user://\" protocol. Use `ProjectSettings.globalize_path()` to convert a Godot-specific path to a system path before opening it with `OS.shell_open()`.");
	}
	return OS::get_singleton()->shell_open(p_uri);
}

int _OS::execute(const String &p_path, const Vector<String> &p_arguments, bool p_blocking, Array p_output, bool p_read_stderr) {
	OS::ProcessID pid = -2;
	int exitcode = 0;
	List<String> args;
	for (int i = 0; i < p_arguments.size(); i++) {
		args.push_back(p_arguments[i]);
	}
	String pipe;
	Error err = OS::get_singleton()->execute(p_path, args, p_blocking, &pid, &pipe, &exitcode, p_read_stderr);
	p_output.clear();
	p_output.push_back(pipe);
	if (err != OK) {
		return -1;
	} else if (p_blocking) {
		return exitcode;
	} else {
		return pid;
	}
}

Error _OS::kill(int p_pid) {
	return OS::get_singleton()->kill(p_pid);
}

int _OS::get_process_id() const {
	return OS::get_singleton()->get_process_id();
}

bool _OS::has_environment(const String &p_var) const {
	return OS::get_singleton()->has_environment(p_var);
}

String _OS::get_environment(const String &p_var) const {
	return OS::get_singleton()->get_environment(p_var);
}

String _OS::get_name() const {
	return OS::get_singleton()->get_name();
}

Vector<String> _OS::get_cmdline_args() {
	List<String> cmdline = OS::get_singleton()->get_cmdline_args();
	Vector<String> cmdlinev;
	for (List<String>::Element *E = cmdline.front(); E; E = E->next()) {
		cmdlinev.push_back(E->get());
	}

	return cmdlinev;
}

String _OS::get_locale() const {
	return OS::get_singleton()->get_locale();
}

String _OS::get_model_name() const {
	return OS::get_singleton()->get_model_name();
}

Error _OS::set_thread_name(const String &p_name) {
	return Thread::set_name(p_name);
}

bool _OS::has_feature(const String &p_feature) const {
	return OS::get_singleton()->has_feature(p_feature);
}

uint64_t _OS::get_static_memory_usage() const {
	return OS::get_singleton()->get_static_memory_usage();
}

uint64_t _OS::get_static_memory_peak_usage() const {
	return OS::get_singleton()->get_static_memory_peak_usage();
}

int _OS::get_exit_code() const {
	return OS::get_singleton()->get_exit_code();
}

void _OS::set_exit_code(int p_code) {
	if (p_code < 0 || p_code > 125) {
		WARN_PRINT("For portability reasons, the exit code should be set between 0 and 125 (inclusive).");
	}

	OS::get_singleton()->set_exit_code(p_code);
}

/**
 *  Get current datetime with consideration for utc and
 *     dst
 */
Dictionary _OS::get_datetime(bool utc) const {
	Dictionary dated = get_date(utc);
	Dictionary timed = get_time(utc);

	List<Variant> keys;
	timed.get_key_list(&keys);

	for (int i = 0; i < keys.size(); i++) {
		dated[keys[i]] = timed[keys[i]];
	}

	return dated;
}

Dictionary _OS::get_date(bool utc) const {
	OS::Date date = OS::get_singleton()->get_date(utc);
	Dictionary dated;
	dated[YEAR_KEY] = date.year;
	dated[MONTH_KEY] = date.month;
	dated[DAY_KEY] = date.day;
	dated[WEEKDAY_KEY] = date.weekday;
	dated[DST_KEY] = date.dst;
	return dated;
}

Dictionary _OS::get_time(bool utc) const {
	OS::Time time = OS::get_singleton()->get_time(utc);
	Dictionary timed;
	timed[HOUR_KEY] = time.hour;
	timed[MINUTE_KEY] = time.min;
	timed[SECOND_KEY] = time.sec;
	return timed;
}

/**
 *  Get an epoch time value from a dictionary of time values
 *  @p datetime must be populated with the following keys:
 *    day, hour, minute, month, second, year. (dst is ignored).
 *
 *    You can pass the output from
 *   get_datetime_from_unix_time directly into this function
 *
 * @param datetime dictionary of date and time values to convert
 *
 * @return epoch calculated
 */
int64_t _OS::get_unix_time_from_datetime(Dictionary datetime) const {
	// Bunch of conversion constants
	static const unsigned int SECONDS_PER_MINUTE = 60;
	static const unsigned int MINUTES_PER_HOUR = 60;
	static const unsigned int HOURS_PER_DAY = 24;
	static const unsigned int SECONDS_PER_HOUR = MINUTES_PER_HOUR * SECONDS_PER_MINUTE;
	static const unsigned int SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY;

	// Get all time values from the dictionary, set to zero if it doesn't exist.
	//   Risk incorrect calculation over throwing errors
	unsigned int second = ((datetime.has(SECOND_KEY)) ? static_cast<unsigned int>(datetime[SECOND_KEY]) : 0);
	unsigned int minute = ((datetime.has(MINUTE_KEY)) ? static_cast<unsigned int>(datetime[MINUTE_KEY]) : 0);
	unsigned int hour = ((datetime.has(HOUR_KEY)) ? static_cast<unsigned int>(datetime[HOUR_KEY]) : 0);
	unsigned int day = ((datetime.has(DAY_KEY)) ? static_cast<unsigned int>(datetime[DAY_KEY]) : 1);
	unsigned int month = ((datetime.has(MONTH_KEY)) ? static_cast<unsigned int>(datetime[MONTH_KEY]) : 1);
	unsigned int year = ((datetime.has(YEAR_KEY)) ? static_cast<unsigned int>(datetime[YEAR_KEY]) : 0);

	/// How many days come before each month (0-12)
	static const unsigned short int DAYS_PAST_THIS_YEAR_TABLE[2][13] = {
		/* Normal years.  */
		{ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 },
		/* Leap years.  */
		{ 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 }
	};

	ERR_FAIL_COND_V_MSG(second > 59, 0, "Invalid second value of: " + itos(second) + ".");

	ERR_FAIL_COND_V_MSG(minute > 59, 0, "Invalid minute value of: " + itos(minute) + ".");

	ERR_FAIL_COND_V_MSG(hour > 23, 0, "Invalid hour value of: " + itos(hour) + ".");

	ERR_FAIL_COND_V_MSG(month > 12 || month == 0, 0, "Invalid month value of: " + itos(month) + ".");

	// Do this check after month is tested as valid
	ERR_FAIL_COND_V_MSG(day > MONTH_DAYS_TABLE[LEAPYEAR(year)][month - 1] || day == 0, 0, "Invalid day value of '" + itos(day) + "' which is larger than '" + itos(MONTH_DAYS_TABLE[LEAPYEAR(year)][month - 1]) + "' or 0.");
	// Calculate all the seconds from months past in this year
	uint64_t SECONDS_FROM_MONTHS_PAST_THIS_YEAR = DAYS_PAST_THIS_YEAR_TABLE[LEAPYEAR(year)][month - 1] * SECONDS_PER_DAY;

	int64_t SECONDS_FROM_YEARS_PAST = 0;
	if (year >= EPOCH_YR) {
		for (unsigned int iyear = EPOCH_YR; iyear < year; iyear++) {
			SECONDS_FROM_YEARS_PAST += YEARSIZE(iyear) * SECONDS_PER_DAY;
		}
	} else {
		for (unsigned int iyear = EPOCH_YR - 1; iyear >= year; iyear--) {
			SECONDS_FROM_YEARS_PAST -= YEARSIZE(iyear) * SECONDS_PER_DAY;
		}
	}

	int64_t epoch =
			second +
			minute * SECONDS_PER_MINUTE +
			hour * SECONDS_PER_HOUR +
			// Subtract 1 from day, since the current day isn't over yet
			//   and we cannot count all 24 hours.
			(day - 1) * SECONDS_PER_DAY +
			SECONDS_FROM_MONTHS_PAST_THIS_YEAR +
			SECONDS_FROM_YEARS_PAST;
	return epoch;
}

/**
 *  Get a dictionary of time values when given epoch time
 *
 *  Dictionary Time values will be a union if values from #get_time
 *    and #get_date dictionaries (with the exception of dst =
 *    day light standard time, as it cannot be determined from epoch)
 *
 * @param unix_time_val epoch time to convert
 *
 * @return dictionary of date and time values
 */
Dictionary _OS::get_datetime_from_unix_time(int64_t unix_time_val) const {
	OS::Date date;
	OS::Time time;

	long dayclock, dayno;
	int year = EPOCH_YR;

	if (unix_time_val >= 0) {
		dayno = unix_time_val / SECS_DAY;
		dayclock = unix_time_val % SECS_DAY;
		/* day 0 was a thursday */
		date.weekday = static_cast<OS::Weekday>((dayno + 4) % 7);
		while (dayno >= YEARSIZE(year)) {
			dayno -= YEARSIZE(year);
			year++;
		}
	} else {
		dayno = (unix_time_val - SECS_DAY + 1) / SECS_DAY;
		dayclock = unix_time_val - dayno * SECS_DAY;
		date.weekday = static_cast<OS::Weekday>(((dayno % 7) + 11) % 7);
		do {
			year--;
			dayno += YEARSIZE(year);
		} while (dayno < 0);
	}

	time.sec = dayclock % 60;
	time.min = (dayclock % 3600) / 60;
	time.hour = dayclock / 3600;
	date.year = year;

	size_t imonth = 0;

	while ((unsigned long)dayno >= MONTH_DAYS_TABLE[LEAPYEAR(year)][imonth]) {
		dayno -= MONTH_DAYS_TABLE[LEAPYEAR(year)][imonth];
		imonth++;
	}

	/// Add 1 to month to make sure months are indexed starting at 1
	date.month = static_cast<OS::Month>(imonth + 1);

	date.day = dayno + 1;

	Dictionary timed;
	timed[HOUR_KEY] = time.hour;
	timed[MINUTE_KEY] = time.min;
	timed[SECOND_KEY] = time.sec;
	timed[YEAR_KEY] = date.year;
	timed[MONTH_KEY] = date.month;
	timed[DAY_KEY] = date.day;
	timed[WEEKDAY_KEY] = date.weekday;

	return timed;
}

Dictionary _OS::get_time_zone_info() const {
	OS::TimeZoneInfo info = OS::get_singleton()->get_time_zone_info();
	Dictionary infod;
	infod["bias"] = info.bias;
	infod["name"] = info.name;
	return infod;
}

double _OS::get_unix_time() const {
	return OS::get_singleton()->get_unix_time();
}

void _OS::delay_usec(uint32_t p_usec) const {
	OS::get_singleton()->delay_usec(p_usec);
}

void _OS::delay_msec(uint32_t p_msec) const {
	OS::get_singleton()->delay_usec(int64_t(p_msec) * 1000);
}

uint32_t _OS::get_ticks_msec() const {
	return OS::get_singleton()->get_ticks_msec();
}

uint64_t _OS::get_ticks_usec() const {
	return OS::get_singleton()->get_ticks_usec();
}

bool _OS::can_use_threads() const {
	return OS::get_singleton()->can_use_threads();
}

bool _OS::is_userfs_persistent() const {
	return OS::get_singleton()->is_userfs_persistent();
}

int _OS::get_processor_count() const {
	return OS::get_singleton()->get_processor_count();
}

bool _OS::is_stdout_verbose() const {
	return OS::get_singleton()->is_stdout_verbose();
}

void _OS::dump_memory_to_file(const String &p_file) {
	OS::get_singleton()->dump_memory_to_file(p_file.utf8().get_data());
}

struct _OSCoreBindImg {
	String path;
	Size2 size;
	int fmt;
	ObjectID id;
	int vram;
	bool operator<(const _OSCoreBindImg &p_img) const { return vram == p_img.vram ? id < p_img.id : vram > p_img.vram; }
};

void _OS::print_all_textures_by_size() {
	List<_OSCoreBindImg> imgs;
	int total = 0;
	{
		List<Ref<Resource>> rsrc;
		ResourceCache::get_cached_resources(&rsrc);

		for (List<Ref<Resource>>::Element *E = rsrc.front(); E; E = E->next()) {
			if (!E->get()->is_class("ImageTexture")) {
				continue;
			}

			Size2 size = E->get()->call("get_size");
			int fmt = E->get()->call("get_format");

			_OSCoreBindImg img;
			img.size = size;
			img.fmt = fmt;
			img.path = E->get()->get_path();
			img.vram = Image::get_image_data_size(img.size.width, img.size.height, Image::Format(img.fmt));
			img.id = E->get()->get_instance_id();
			total += img.vram;
			imgs.push_back(img);
		}
	}

	imgs.sort();

	for (List<_OSCoreBindImg>::Element *E = imgs.front(); E; E = E->next()) {
		total -= E->get().vram;
	}
}

void _OS::print_resources_by_type(const Vector<String> &p_types) {
	Map<String, int> type_count;

	List<Ref<Resource>> resources;
	ResourceCache::get_cached_resources(&resources);

	List<Ref<Resource>> rsrc;
	ResourceCache::get_cached_resources(&rsrc);

	for (List<Ref<Resource>>::Element *E = rsrc.front(); E; E = E->next()) {
		Ref<Resource> r = E->get();

		bool found = false;

		for (int i = 0; i < p_types.size(); i++) {
			if (r->is_class(p_types[i])) {
				found = true;
			}
		}
		if (!found) {
			continue;
		}

		if (!type_count.has(r->get_class())) {
			type_count[r->get_class()] = 0;
		}

		type_count[r->get_class()]++;
	}
}

void _OS::print_all_resources(const String &p_to_file) {
	OS::get_singleton()->print_all_resources(p_to_file);
}

void _OS::print_resources_in_use(bool p_short) {
	OS::get_singleton()->print_resources_in_use(p_short);
}

void _OS::dump_resources_to_file(const String &p_file) {
	OS::get_singleton()->dump_resources_to_file(p_file.utf8().get_data());
}

String _OS::get_user_data_dir() const {
	return OS::get_singleton()->get_user_data_dir();
}

bool _OS::is_debug_build() const {
#ifdef DEBUG_ENABLED
	return true;
#else
	return false;
#endif
}

String _OS::get_system_dir(SystemDir p_dir) const {
	return OS::get_singleton()->get_system_dir(OS::SystemDir(p_dir));
}

String _OS::get_keycode_string(uint32_t p_code) const {
	return keycode_get_string(p_code);
}

bool _OS::is_keycode_unicode(uint32_t p_unicode) const {
	return keycode_has_unicode(p_unicode);
}

int _OS::find_keycode_from_string(const String &p_code) const {
	return find_keycode(p_code);
}

bool _OS::request_permission(const String &p_name) {
	return OS::get_singleton()->request_permission(p_name);
}

bool _OS::request_permissions() {
	return OS::get_singleton()->request_permissions();
}

Vector<String> _OS::get_granted_permissions() const {
	return OS::get_singleton()->get_granted_permissions();
}

String _OS::get_unique_id() const {
	return OS::get_singleton()->get_unique_id();
}

int _OS::get_tablet_driver_count() const {
	return OS::get_singleton()->get_tablet_driver_count();
}

String _OS::get_tablet_driver_name(int p_driver) const {
	return OS::get_singleton()->get_tablet_driver_name(p_driver);
}

String _OS::get_current_tablet_driver() const {
	return OS::get_singleton()->get_current_tablet_driver();
}

void _OS::set_current_tablet_driver(const String &p_driver) {
	OS::get_singleton()->set_current_tablet_driver(p_driver);
}

_OS *_OS::singleton = nullptr;

void _OS::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_connected_midi_inputs"), &_OS::get_connected_midi_inputs);
	ClassDB::bind_method(D_METHOD("open_midi_inputs"), &_OS::open_midi_inputs);
	ClassDB::bind_method(D_METHOD("close_midi_inputs"), &_OS::close_midi_inputs);

	ClassDB::bind_method(D_METHOD("set_low_processor_usage_mode", "enable"), &_OS::set_low_processor_usage_mode);
	ClassDB::bind_method(D_METHOD("is_in_low_processor_usage_mode"), &_OS::is_in_low_processor_usage_mode);

	ClassDB::bind_method(D_METHOD("set_low_processor_usage_mode_sleep_usec", "usec"), &_OS::set_low_processor_usage_mode_sleep_usec);
	ClassDB::bind_method(D_METHOD("get_low_processor_usage_mode_sleep_usec"), &_OS::get_low_processor_usage_mode_sleep_usec);

	ClassDB::bind_method(D_METHOD("get_processor_count"), &_OS::get_processor_count);

	ClassDB::bind_method(D_METHOD("get_executable_path"), &_OS::get_executable_path);
	ClassDB::bind_method(D_METHOD("execute", "path", "arguments", "blocking", "output", "read_stderr"), &_OS::execute, DEFVAL(true), DEFVAL(Array()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("kill", "pid"), &_OS::kill);
	ClassDB::bind_method(D_METHOD("shell_open", "uri"), &_OS::shell_open);
	ClassDB::bind_method(D_METHOD("get_process_id"), &_OS::get_process_id);

	ClassDB::bind_method(D_METHOD("get_environment", "environment"), &_OS::get_environment);
	ClassDB::bind_method(D_METHOD("has_environment", "environment"), &_OS::has_environment);

	ClassDB::bind_method(D_METHOD("get_name"), &_OS::get_name);
	ClassDB::bind_method(D_METHOD("get_cmdline_args"), &_OS::get_cmdline_args);

	ClassDB::bind_method(D_METHOD("get_datetime", "utc"), &_OS::get_datetime, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_date", "utc"), &_OS::get_date, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_time", "utc"), &_OS::get_time, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_time_zone_info"), &_OS::get_time_zone_info);
	ClassDB::bind_method(D_METHOD("get_unix_time"), &_OS::get_unix_time);
	ClassDB::bind_method(D_METHOD("get_datetime_from_unix_time", "unix_time_val"), &_OS::get_datetime_from_unix_time);
	ClassDB::bind_method(D_METHOD("get_unix_time_from_datetime", "datetime"), &_OS::get_unix_time_from_datetime);

	ClassDB::bind_method(D_METHOD("get_exit_code"), &_OS::get_exit_code);
	ClassDB::bind_method(D_METHOD("set_exit_code", "code"), &_OS::set_exit_code);

	ClassDB::bind_method(D_METHOD("delay_usec", "usec"), &_OS::delay_usec);
	ClassDB::bind_method(D_METHOD("delay_msec", "msec"), &_OS::delay_msec);
	ClassDB::bind_method(D_METHOD("get_ticks_msec"), &_OS::get_ticks_msec);
	ClassDB::bind_method(D_METHOD("get_ticks_usec"), &_OS::get_ticks_usec);
	ClassDB::bind_method(D_METHOD("get_locale"), &_OS::get_locale);
	ClassDB::bind_method(D_METHOD("get_model_name"), &_OS::get_model_name);

	ClassDB::bind_method(D_METHOD("is_userfs_persistent"), &_OS::is_userfs_persistent);
	ClassDB::bind_method(D_METHOD("is_stdout_verbose"), &_OS::is_stdout_verbose);

	ClassDB::bind_method(D_METHOD("can_use_threads"), &_OS::can_use_threads);

	ClassDB::bind_method(D_METHOD("is_debug_build"), &_OS::is_debug_build);

	ClassDB::bind_method(D_METHOD("dump_memory_to_file", "file"), &_OS::dump_memory_to_file);
	ClassDB::bind_method(D_METHOD("dump_resources_to_file", "file"), &_OS::dump_resources_to_file);
	ClassDB::bind_method(D_METHOD("print_resources_in_use", "short"), &_OS::print_resources_in_use, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("print_all_resources", "tofile"), &_OS::print_all_resources, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("get_static_memory_usage"), &_OS::get_static_memory_usage);
	ClassDB::bind_method(D_METHOD("get_static_memory_peak_usage"), &_OS::get_static_memory_peak_usage);

	ClassDB::bind_method(D_METHOD("get_user_data_dir"), &_OS::get_user_data_dir);
	ClassDB::bind_method(D_METHOD("get_system_dir", "dir"), &_OS::get_system_dir);
	ClassDB::bind_method(D_METHOD("get_unique_id"), &_OS::get_unique_id);

	ClassDB::bind_method(D_METHOD("print_all_textures_by_size"), &_OS::print_all_textures_by_size);
	ClassDB::bind_method(D_METHOD("print_resources_by_type", "types"), &_OS::print_resources_by_type);

	ClassDB::bind_method(D_METHOD("get_keycode_string", "code"), &_OS::get_keycode_string);
	ClassDB::bind_method(D_METHOD("is_keycode_unicode", "code"), &_OS::is_keycode_unicode);
	ClassDB::bind_method(D_METHOD("find_keycode_from_string", "string"), &_OS::find_keycode_from_string);

	ClassDB::bind_method(D_METHOD("set_use_file_access_save_and_swap", "enabled"), &_OS::set_use_file_access_save_and_swap);

	ClassDB::bind_method(D_METHOD("set_thread_name", "name"), &_OS::set_thread_name);

	ClassDB::bind_method(D_METHOD("has_feature", "tag_name"), &_OS::has_feature);

	ClassDB::bind_method(D_METHOD("request_permission", "name"), &_OS::request_permission);
	ClassDB::bind_method(D_METHOD("request_permissions"), &_OS::request_permissions);
	ClassDB::bind_method(D_METHOD("get_granted_permissions"), &_OS::get_granted_permissions);

	ClassDB::bind_method(D_METHOD("get_tablet_driver_count"), &_OS::get_tablet_driver_count);
	ClassDB::bind_method(D_METHOD("get_tablet_driver_name", "idx"), &_OS::get_tablet_driver_name);
	ClassDB::bind_method(D_METHOD("get_current_tablet_driver"), &_OS::get_current_tablet_driver);
	ClassDB::bind_method(D_METHOD("set_current_tablet_driver", "name"), &_OS::set_current_tablet_driver);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "exit_code"), "set_exit_code", "get_exit_code");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "low_processor_usage_mode"), "set_low_processor_usage_mode", "is_in_low_processor_usage_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "low_processor_usage_mode_sleep_usec"), "set_low_processor_usage_mode_sleep_usec", "get_low_processor_usage_mode_sleep_usec");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tablet_driver"), "set_current_tablet_driver", "get_current_tablet_driver");

	// Those default values need to be specified for the docs generator,
	// to avoid using values from the documentation writer's own OS instance.
	ADD_PROPERTY_DEFAULT("tablet_driver", "");
	ADD_PROPERTY_DEFAULT("exit_code", 0);
	ADD_PROPERTY_DEFAULT("low_processor_usage_mode", false);
	ADD_PROPERTY_DEFAULT("low_processor_usage_mode_sleep_usec", 6900);

	BIND_ENUM_CONSTANT(VIDEO_DRIVER_GLES2);
	BIND_ENUM_CONSTANT(VIDEO_DRIVER_VULKAN);

	BIND_ENUM_CONSTANT(DAY_SUNDAY);
	BIND_ENUM_CONSTANT(DAY_MONDAY);
	BIND_ENUM_CONSTANT(DAY_TUESDAY);
	BIND_ENUM_CONSTANT(DAY_WEDNESDAY);
	BIND_ENUM_CONSTANT(DAY_THURSDAY);
	BIND_ENUM_CONSTANT(DAY_FRIDAY);
	BIND_ENUM_CONSTANT(DAY_SATURDAY);

	BIND_ENUM_CONSTANT(MONTH_JANUARY);
	BIND_ENUM_CONSTANT(MONTH_FEBRUARY);
	BIND_ENUM_CONSTANT(MONTH_MARCH);
	BIND_ENUM_CONSTANT(MONTH_APRIL);
	BIND_ENUM_CONSTANT(MONTH_MAY);
	BIND_ENUM_CONSTANT(MONTH_JUNE);
	BIND_ENUM_CONSTANT(MONTH_JULY);
	BIND_ENUM_CONSTANT(MONTH_AUGUST);
	BIND_ENUM_CONSTANT(MONTH_SEPTEMBER);
	BIND_ENUM_CONSTANT(MONTH_OCTOBER);
	BIND_ENUM_CONSTANT(MONTH_NOVEMBER);
	BIND_ENUM_CONSTANT(MONTH_DECEMBER);

	BIND_ENUM_CONSTANT(SYSTEM_DIR_DESKTOP);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_DCIM);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_DOCUMENTS);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_DOWNLOADS);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_MOVIES);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_MUSIC);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_PICTURES);
	BIND_ENUM_CONSTANT(SYSTEM_DIR_RINGTONES);
}

////// _Geometry2D //////

_Geometry2D *_Geometry2D::singleton = nullptr;

_Geometry2D *_Geometry2D::get_singleton() {
	return singleton;
}

bool _Geometry2D::is_point_in_circle(const Vector2 &p_point, const Vector2 &p_circle_pos, real_t p_circle_radius) {
	return Geometry2D::is_point_in_circle(p_point, p_circle_pos, p_circle_radius);
}

real_t _Geometry2D::segment_intersects_circle(const Vector2 &p_from, const Vector2 &p_to, const Vector2 &p_circle_pos, real_t p_circle_radius) {
	return Geometry2D::segment_intersects_circle(p_from, p_to, p_circle_pos, p_circle_radius);
}

Variant _Geometry2D::segment_intersects_segment(const Vector2 &p_from_a, const Vector2 &p_to_a, const Vector2 &p_from_b, const Vector2 &p_to_b) {
	Vector2 result;
	if (Geometry2D::segment_intersects_segment(p_from_a, p_to_a, p_from_b, p_to_b, &result)) {
		return result;
	} else {
		return Variant();
	}
}

Variant _Geometry2D::line_intersects_line(const Vector2 &p_from_a, const Vector2 &p_dir_a, const Vector2 &p_from_b, const Vector2 &p_dir_b) {
	Vector2 result;
	if (Geometry2D::line_intersects_line(p_from_a, p_dir_a, p_from_b, p_dir_b, result)) {
		return result;
	} else {
		return Variant();
	}
}

Vector<Vector2> _Geometry2D::get_closest_points_between_segments(const Vector2 &p1, const Vector2 &q1, const Vector2 &p2, const Vector2 &q2) {
	Vector2 r1, r2;
	Geometry2D::get_closest_points_between_segments(p1, q1, p2, q2, r1, r2);
	Vector<Vector2> r;
	r.resize(2);
	r.set(0, r1);
	r.set(1, r2);
	return r;
}

Vector2 _Geometry2D::get_closest_point_to_segment(const Vector2 &p_point, const Vector2 &p_a, const Vector2 &p_b) {
	Vector2 s[2] = { p_a, p_b };
	return Geometry2D::get_closest_point_to_segment(p_point, s);
}

Vector2 _Geometry2D::get_closest_point_to_segment_uncapped(const Vector2 &p_point, const Vector2 &p_a, const Vector2 &p_b) {
	Vector2 s[2] = { p_a, p_b };
	return Geometry2D::get_closest_point_to_segment_uncapped(p_point, s);
}

bool _Geometry2D::point_is_inside_triangle(const Vector2 &s, const Vector2 &a, const Vector2 &b, const Vector2 &c) const {
	return Geometry2D::is_point_in_triangle(s, a, b, c);
}

bool _Geometry2D::is_polygon_clockwise(const Vector<Vector2> &p_polygon) {
	return Geometry2D::is_polygon_clockwise(p_polygon);
}

bool _Geometry2D::is_point_in_polygon(const Point2 &p_point, const Vector<Vector2> &p_polygon) {
	return Geometry2D::is_point_in_polygon(p_point, p_polygon);
}

Vector<int> _Geometry2D::triangulate_polygon(const Vector<Vector2> &p_polygon) {
	return Geometry2D::triangulate_polygon(p_polygon);
}

Vector<int> _Geometry2D::triangulate_delaunay(const Vector<Vector2> &p_points) {
	return Geometry2D::triangulate_delaunay(p_points);
}

Vector<Point2> _Geometry2D::convex_hull(const Vector<Point2> &p_points) {
	return Geometry2D::convex_hull(p_points);
}

Array _Geometry2D::merge_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = Geometry2D::merge_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array _Geometry2D::clip_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = Geometry2D::clip_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array _Geometry2D::intersect_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = Geometry2D::intersect_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array _Geometry2D::exclude_polygons(const Vector<Vector2> &p_polygon_a, const Vector<Vector2> &p_polygon_b) {
	Vector<Vector<Point2>> polys = Geometry2D::exclude_polygons(p_polygon_a, p_polygon_b);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array _Geometry2D::clip_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon) {
	Vector<Vector<Point2>> polys = Geometry2D::clip_polyline_with_polygon(p_polyline, p_polygon);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array _Geometry2D::intersect_polyline_with_polygon(const Vector<Vector2> &p_polyline, const Vector<Vector2> &p_polygon) {
	Vector<Vector<Point2>> polys = Geometry2D::intersect_polyline_with_polygon(p_polyline, p_polygon);

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array _Geometry2D::offset_polygon(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type) {
	Vector<Vector<Point2>> polys = Geometry2D::offset_polygon(p_polygon, p_delta, Geometry2D::PolyJoinType(p_join_type));

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Array _Geometry2D::offset_polyline(const Vector<Vector2> &p_polygon, real_t p_delta, PolyJoinType p_join_type, PolyEndType p_end_type) {
	Vector<Vector<Point2>> polys = Geometry2D::offset_polyline(p_polygon, p_delta, Geometry2D::PolyJoinType(p_join_type), Geometry2D::PolyEndType(p_end_type));

	Array ret;

	for (int i = 0; i < polys.size(); ++i) {
		ret.push_back(polys[i]);
	}
	return ret;
}

Dictionary _Geometry2D::make_atlas(const Vector<Size2> &p_rects) {
	Dictionary ret;

	Vector<Size2i> rects;
	for (int i = 0; i < p_rects.size(); i++) {
		rects.push_back(p_rects[i]);
	}

	Vector<Point2i> result;
	Size2i size;

	Geometry2D::make_atlas(rects, result, size);

	Size2 r_size = size;
	Vector<Point2> r_result;
	for (int i = 0; i < result.size(); i++) {
		r_result.push_back(result[i]);
	}

	ret["points"] = r_result;
	ret["size"] = r_size;

	return ret;
}

void _Geometry2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_point_in_circle", "point", "circle_position", "circle_radius"), &_Geometry2D::is_point_in_circle);
	ClassDB::bind_method(D_METHOD("segment_intersects_segment", "from_a", "to_a", "from_b", "to_b"), &_Geometry2D::segment_intersects_segment);
	ClassDB::bind_method(D_METHOD("line_intersects_line", "from_a", "dir_a", "from_b", "dir_b"), &_Geometry2D::line_intersects_line);

	ClassDB::bind_method(D_METHOD("get_closest_points_between_segments", "p1", "q1", "p2", "q2"), &_Geometry2D::get_closest_points_between_segments);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment", "point", "s1", "s2"), &_Geometry2D::get_closest_point_to_segment);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment_uncapped", "point", "s1", "s2"), &_Geometry2D::get_closest_point_to_segment_uncapped);

	ClassDB::bind_method(D_METHOD("point_is_inside_triangle", "point", "a", "b", "c"), &_Geometry2D::point_is_inside_triangle);

	ClassDB::bind_method(D_METHOD("is_polygon_clockwise", "polygon"), &_Geometry2D::is_polygon_clockwise);
	ClassDB::bind_method(D_METHOD("is_point_in_polygon", "point", "polygon"), &_Geometry2D::is_point_in_polygon);
	ClassDB::bind_method(D_METHOD("triangulate_polygon", "polygon"), &_Geometry2D::triangulate_polygon);
	ClassDB::bind_method(D_METHOD("triangulate_delaunay", "points"), &_Geometry2D::triangulate_delaunay);
	ClassDB::bind_method(D_METHOD("convex_hull", "points"), &_Geometry2D::convex_hull);

	ClassDB::bind_method(D_METHOD("merge_polygons", "polygon_a", "polygon_b"), &_Geometry2D::merge_polygons);
	ClassDB::bind_method(D_METHOD("clip_polygons", "polygon_a", "polygon_b"), &_Geometry2D::clip_polygons);
	ClassDB::bind_method(D_METHOD("intersect_polygons", "polygon_a", "polygon_b"), &_Geometry2D::intersect_polygons);
	ClassDB::bind_method(D_METHOD("exclude_polygons", "polygon_a", "polygon_b"), &_Geometry2D::exclude_polygons);

	ClassDB::bind_method(D_METHOD("clip_polyline_with_polygon", "polyline", "polygon"), &_Geometry2D::clip_polyline_with_polygon);
	ClassDB::bind_method(D_METHOD("intersect_polyline_with_polygon", "polyline", "polygon"), &_Geometry2D::intersect_polyline_with_polygon);

	ClassDB::bind_method(D_METHOD("offset_polygon", "polygon", "delta", "join_type"), &_Geometry2D::offset_polygon, DEFVAL(JOIN_SQUARE));
	ClassDB::bind_method(D_METHOD("offset_polyline", "polyline", "delta", "join_type", "end_type"), &_Geometry2D::offset_polyline, DEFVAL(JOIN_SQUARE), DEFVAL(END_SQUARE));

	ClassDB::bind_method(D_METHOD("make_atlas", "sizes"), &_Geometry2D::make_atlas);

	BIND_ENUM_CONSTANT(OPERATION_UNION);
	BIND_ENUM_CONSTANT(OPERATION_DIFFERENCE);
	BIND_ENUM_CONSTANT(OPERATION_INTERSECTION);
	BIND_ENUM_CONSTANT(OPERATION_XOR);

	BIND_ENUM_CONSTANT(JOIN_SQUARE);
	BIND_ENUM_CONSTANT(JOIN_ROUND);
	BIND_ENUM_CONSTANT(JOIN_MITER);

	BIND_ENUM_CONSTANT(END_POLYGON);
	BIND_ENUM_CONSTANT(END_JOINED);
	BIND_ENUM_CONSTANT(END_BUTT);
	BIND_ENUM_CONSTANT(END_SQUARE);
	BIND_ENUM_CONSTANT(END_ROUND);
}

////// _Geometry3D //////

_Geometry3D *_Geometry3D::singleton = nullptr;

_Geometry3D *_Geometry3D::get_singleton() {
	return singleton;
}

Vector<Plane> _Geometry3D::build_box_planes(const Vector3 &p_extents) {
	return Geometry3D::build_box_planes(p_extents);
}

Vector<Plane> _Geometry3D::build_cylinder_planes(float p_radius, float p_height, int p_sides, Vector3::Axis p_axis) {
	return Geometry3D::build_cylinder_planes(p_radius, p_height, p_sides, p_axis);
}

Vector<Plane> _Geometry3D::build_capsule_planes(float p_radius, float p_height, int p_sides, int p_lats, Vector3::Axis p_axis) {
	return Geometry3D::build_capsule_planes(p_radius, p_height, p_sides, p_lats, p_axis);
}

Vector<Vector3> _Geometry3D::get_closest_points_between_segments(const Vector3 &p1, const Vector3 &p2, const Vector3 &q1, const Vector3 &q2) {
	Vector3 r1, r2;
	Geometry3D::get_closest_points_between_segments(p1, p2, q1, q2, r1, r2);
	Vector<Vector3> r;
	r.resize(2);
	r.set(0, r1);
	r.set(1, r2);
	return r;
}

Vector3 _Geometry3D::get_closest_point_to_segment(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b) {
	Vector3 s[2] = { p_a, p_b };
	return Geometry3D::get_closest_point_to_segment(p_point, s);
}

Vector3 _Geometry3D::get_closest_point_to_segment_uncapped(const Vector3 &p_point, const Vector3 &p_a, const Vector3 &p_b) {
	Vector3 s[2] = { p_a, p_b };
	return Geometry3D::get_closest_point_to_segment_uncapped(p_point, s);
}

Variant _Geometry3D::ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2) {
	Vector3 res;
	if (Geometry3D::ray_intersects_triangle(p_from, p_dir, p_v0, p_v1, p_v2, &res)) {
		return res;
	} else {
		return Variant();
	}
}

Variant _Geometry3D::segment_intersects_triangle(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_v0, const Vector3 &p_v1, const Vector3 &p_v2) {
	Vector3 res;
	if (Geometry3D::segment_intersects_triangle(p_from, p_to, p_v0, p_v1, p_v2, &res)) {
		return res;
	} else {
		return Variant();
	}
}

Vector<Vector3> _Geometry3D::segment_intersects_sphere(const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_sphere_pos, real_t p_sphere_radius) {
	Vector<Vector3> r;
	Vector3 res, norm;
	if (!Geometry3D::segment_intersects_sphere(p_from, p_to, p_sphere_pos, p_sphere_radius, &res, &norm)) {
		return r;
	}

	r.resize(2);
	r.set(0, res);
	r.set(1, norm);
	return r;
}

Vector<Vector3> _Geometry3D::segment_intersects_cylinder(const Vector3 &p_from, const Vector3 &p_to, float p_height, float p_radius) {
	Vector<Vector3> r;
	Vector3 res, norm;
	if (!Geometry3D::segment_intersects_cylinder(p_from, p_to, p_height, p_radius, &res, &norm)) {
		return r;
	}

	r.resize(2);
	r.set(0, res);
	r.set(1, norm);
	return r;
}

Vector<Vector3> _Geometry3D::segment_intersects_convex(const Vector3 &p_from, const Vector3 &p_to, const Vector<Plane> &p_planes) {
	Vector<Vector3> r;
	Vector3 res, norm;
	if (!Geometry3D::segment_intersects_convex(p_from, p_to, p_planes.ptr(), p_planes.size(), &res, &norm)) {
		return r;
	}

	r.resize(2);
	r.set(0, res);
	r.set(1, norm);
	return r;
}

Vector<Vector3> _Geometry3D::clip_polygon(const Vector<Vector3> &p_points, const Plane &p_plane) {
	return Geometry3D::clip_polygon(p_points, p_plane);
}

void _Geometry3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("build_box_planes", "extents"), &_Geometry3D::build_box_planes);
	ClassDB::bind_method(D_METHOD("build_cylinder_planes", "radius", "height", "sides", "axis"), &_Geometry3D::build_cylinder_planes, DEFVAL(Vector3::AXIS_Z));
	ClassDB::bind_method(D_METHOD("build_capsule_planes", "radius", "height", "sides", "lats", "axis"), &_Geometry3D::build_capsule_planes, DEFVAL(Vector3::AXIS_Z));

	ClassDB::bind_method(D_METHOD("get_closest_points_between_segments", "p1", "p2", "q1", "q2"), &_Geometry3D::get_closest_points_between_segments);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment", "point", "s1", "s2"), &_Geometry3D::get_closest_point_to_segment);

	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment_uncapped", "point", "s1", "s2"), &_Geometry3D::get_closest_point_to_segment_uncapped);

	ClassDB::bind_method(D_METHOD("ray_intersects_triangle", "from", "dir", "a", "b", "c"), &_Geometry3D::ray_intersects_triangle);
	ClassDB::bind_method(D_METHOD("segment_intersects_triangle", "from", "to", "a", "b", "c"), &_Geometry3D::segment_intersects_triangle);
	ClassDB::bind_method(D_METHOD("segment_intersects_sphere", "from", "to", "sphere_position", "sphere_radius"), &_Geometry3D::segment_intersects_sphere);
	ClassDB::bind_method(D_METHOD("segment_intersects_cylinder", "from", "to", "height", "radius"), &_Geometry3D::segment_intersects_cylinder);
	ClassDB::bind_method(D_METHOD("segment_intersects_convex", "from", "to", "planes"), &_Geometry3D::segment_intersects_convex);

	ClassDB::bind_method(D_METHOD("clip_polygon", "points", "plane"), &_Geometry3D::clip_polygon);
}

////// _File //////

Error _File::open_encrypted(const String &p_path, ModeFlags p_mode_flags, const Vector<uint8_t> &p_key) {
	Error err = open(p_path, p_mode_flags);
	if (err) {
		return err;
	}

	FileAccessEncrypted *fae = memnew(FileAccessEncrypted);
	err = fae->open_and_parse(f, p_key, (p_mode_flags == WRITE) ? FileAccessEncrypted::MODE_WRITE_AES256 : FileAccessEncrypted::MODE_READ);
	if (err) {
		memdelete(fae);
		close();
		return err;
	}
	f = fae;
	return OK;
}

Error _File::open_encrypted_pass(const String &p_path, ModeFlags p_mode_flags, const String &p_pass) {
	Error err = open(p_path, p_mode_flags);
	if (err) {
		return err;
	}

	FileAccessEncrypted *fae = memnew(FileAccessEncrypted);
	err = fae->open_and_parse_password(f, p_pass, (p_mode_flags == WRITE) ? FileAccessEncrypted::MODE_WRITE_AES256 : FileAccessEncrypted::MODE_READ);
	if (err) {
		memdelete(fae);
		close();
		return err;
	}

	f = fae;
	return OK;
}

Error _File::open_compressed(const String &p_path, ModeFlags p_mode_flags, CompressionMode p_compress_mode) {
	FileAccessCompressed *fac = memnew(FileAccessCompressed);

	fac->configure("GCPF", (Compression::Mode)p_compress_mode);

	Error err = fac->_open(p_path, p_mode_flags);

	if (err) {
		memdelete(fac);
		return err;
	}

	f = fac;
	return OK;
}

Error _File::open(const String &p_path, ModeFlags p_mode_flags) {
	close();
	Error err;
	f = FileAccess::open(p_path, p_mode_flags, &err);
	if (f) {
		f->set_endian_swap(eswap);
	}
	return err;
}

void _File::close() {
	if (f) {
		memdelete(f);
	}
	f = nullptr;
}

bool _File::is_open() const {
	return f != nullptr;
}

String _File::get_path() const {
	ERR_FAIL_COND_V_MSG(!f, "", "File must be opened before use.");
	return f->get_path();
}

String _File::get_path_absolute() const {
	ERR_FAIL_COND_V_MSG(!f, "", "File must be opened before use.");
	return f->get_path_absolute();
}

void _File::seek(int64_t p_position) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	f->seek(p_position);
}

void _File::seek_end(int64_t p_position) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	f->seek_end(p_position);
}

int64_t _File::get_position() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_position();
}

int64_t _File::get_len() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_len();
}

bool _File::eof_reached() const {
	ERR_FAIL_COND_V_MSG(!f, false, "File must be opened before use.");
	return f->eof_reached();
}

uint8_t _File::get_8() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_8();
}

uint16_t _File::get_16() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_16();
}

uint32_t _File::get_32() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_32();
}

uint64_t _File::get_64() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_64();
}

float _File::get_float() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_float();
}

double _File::get_double() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_double();
}

real_t _File::get_real() const {
	ERR_FAIL_COND_V_MSG(!f, 0, "File must be opened before use.");
	return f->get_real();
}

Vector<uint8_t> _File::get_buffer(int p_length) const {
	Vector<uint8_t> data;
	ERR_FAIL_COND_V_MSG(!f, data, "File must be opened before use.");

	ERR_FAIL_COND_V_MSG(p_length < 0, data, "Length of buffer cannot be smaller than 0.");
	if (p_length == 0) {
		return data;
	}

	Error err = data.resize(p_length);
	ERR_FAIL_COND_V_MSG(err != OK, data, "Can't resize data to " + itos(p_length) + " elements.");

	uint8_t *w = data.ptrw();
	int len = f->get_buffer(&w[0], p_length);
	ERR_FAIL_COND_V(len < 0, Vector<uint8_t>());

	if (len < p_length) {
		data.resize(p_length);
	}

	return data;
}

String _File::get_as_text() const {
	ERR_FAIL_COND_V_MSG(!f, String(), "File must be opened before use.");

	String text;
	size_t original_pos = f->get_position();
	f->seek(0);

	String l = get_line();
	while (!eof_reached()) {
		text += l + "\n";
		l = get_line();
	}
	text += l;

	f->seek(original_pos);

	return text;
}

String _File::get_md5(const String &p_path) const {
	return FileAccess::get_md5(p_path);
}

String _File::get_sha256(const String &p_path) const {
	return FileAccess::get_sha256(p_path);
}

String _File::get_line() const {
	ERR_FAIL_COND_V_MSG(!f, String(), "File must be opened before use.");
	return f->get_line();
}

Vector<String> _File::get_csv_line(const String &p_delim) const {
	ERR_FAIL_COND_V_MSG(!f, Vector<String>(), "File must be opened before use.");
	return f->get_csv_line(p_delim);
}

/**< use this for files WRITTEN in _big_ endian machines (ie, amiga/mac)
 * It's not about the current CPU type but file formats.
 * this flags get reset to false (little endian) on each open
 */

void _File::set_endian_swap(bool p_swap) {
	eswap = p_swap;
	if (f) {
		f->set_endian_swap(p_swap);
	}
}

bool _File::get_endian_swap() {
	return eswap;
}

Error _File::get_error() const {
	if (!f) {
		return ERR_UNCONFIGURED;
	}
	return f->get_error();
}

void _File::store_8(uint8_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_8(p_dest);
}

void _File::store_16(uint16_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_16(p_dest);
}

void _File::store_32(uint32_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_32(p_dest);
}

void _File::store_64(uint64_t p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_64(p_dest);
}

void _File::store_float(float p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_float(p_dest);
}

void _File::store_double(double p_dest) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_double(p_dest);
}

void _File::store_real(real_t p_real) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_real(p_real);
}

void _File::store_string(const String &p_string) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_string(p_string);
}

void _File::store_pascal_string(const String &p_string) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	f->store_pascal_string(p_string);
}

String _File::get_pascal_string() {
	ERR_FAIL_COND_V_MSG(!f, "", "File must be opened before use.");

	return f->get_pascal_string();
}

void _File::store_line(const String &p_string) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	f->store_line(p_string);
}

void _File::store_csv_line(const Vector<String> &p_values, const String &p_delim) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	f->store_csv_line(p_values, p_delim);
}

void _File::store_buffer(const Vector<uint8_t> &p_buffer) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");

	int len = p_buffer.size();
	if (len == 0) {
		return;
	}

	const uint8_t *r = p_buffer.ptr();

	f->store_buffer(&r[0], len);
}

bool _File::file_exists(const String &p_name) const {
	return FileAccess::exists(p_name);
}

void _File::store_var(const Variant &p_var, bool p_full_objects) {
	ERR_FAIL_COND_MSG(!f, "File must be opened before use.");
	int len;
	Error err = encode_variant(p_var, nullptr, len, p_full_objects);
	ERR_FAIL_COND_MSG(err != OK, "Error when trying to encode Variant.");

	Vector<uint8_t> buff;
	buff.resize(len);

	uint8_t *w = buff.ptrw();
	err = encode_variant(p_var, &w[0], len, p_full_objects);
	ERR_FAIL_COND_MSG(err != OK, "Error when trying to encode Variant.");

	store_32(len);
	store_buffer(buff);
}

Variant _File::get_var(bool p_allow_objects) const {
	ERR_FAIL_COND_V_MSG(!f, Variant(), "File must be opened before use.");
	uint32_t len = get_32();
	Vector<uint8_t> buff = get_buffer(len);
	ERR_FAIL_COND_V((uint32_t)buff.size() != len, Variant());

	const uint8_t *r = buff.ptr();

	Variant v;
	Error err = decode_variant(v, &r[0], len, nullptr, p_allow_objects);
	ERR_FAIL_COND_V_MSG(err != OK, Variant(), "Error when trying to encode Variant.");

	return v;
}

uint64_t _File::get_modified_time(const String &p_file) const {
	return FileAccess::get_modified_time(p_file);
}

void _File::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open_encrypted", "path", "mode_flags", "key"), &_File::open_encrypted);
	ClassDB::bind_method(D_METHOD("open_encrypted_with_pass", "path", "mode_flags", "pass"), &_File::open_encrypted_pass);
	ClassDB::bind_method(D_METHOD("open_compressed", "path", "mode_flags", "compression_mode"), &_File::open_compressed, DEFVAL(0));

	ClassDB::bind_method(D_METHOD("open", "path", "flags"), &_File::open);
	ClassDB::bind_method(D_METHOD("close"), &_File::close);
	ClassDB::bind_method(D_METHOD("get_path"), &_File::get_path);
	ClassDB::bind_method(D_METHOD("get_path_absolute"), &_File::get_path_absolute);
	ClassDB::bind_method(D_METHOD("is_open"), &_File::is_open);
	ClassDB::bind_method(D_METHOD("seek", "position"), &_File::seek);
	ClassDB::bind_method(D_METHOD("seek_end", "position"), &_File::seek_end, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_position"), &_File::get_position);
	ClassDB::bind_method(D_METHOD("get_len"), &_File::get_len);
	ClassDB::bind_method(D_METHOD("eof_reached"), &_File::eof_reached);
	ClassDB::bind_method(D_METHOD("get_8"), &_File::get_8);
	ClassDB::bind_method(D_METHOD("get_16"), &_File::get_16);
	ClassDB::bind_method(D_METHOD("get_32"), &_File::get_32);
	ClassDB::bind_method(D_METHOD("get_64"), &_File::get_64);
	ClassDB::bind_method(D_METHOD("get_float"), &_File::get_float);
	ClassDB::bind_method(D_METHOD("get_double"), &_File::get_double);
	ClassDB::bind_method(D_METHOD("get_real"), &_File::get_real);
	ClassDB::bind_method(D_METHOD("get_buffer", "len"), &_File::get_buffer);
	ClassDB::bind_method(D_METHOD("get_line"), &_File::get_line);
	ClassDB::bind_method(D_METHOD("get_csv_line", "delim"), &_File::get_csv_line, DEFVAL(","));
	ClassDB::bind_method(D_METHOD("get_as_text"), &_File::get_as_text);
	ClassDB::bind_method(D_METHOD("get_md5", "path"), &_File::get_md5);
	ClassDB::bind_method(D_METHOD("get_sha256", "path"), &_File::get_sha256);
	ClassDB::bind_method(D_METHOD("get_endian_swap"), &_File::get_endian_swap);
	ClassDB::bind_method(D_METHOD("set_endian_swap", "enable"), &_File::set_endian_swap);
	ClassDB::bind_method(D_METHOD("get_error"), &_File::get_error);
	ClassDB::bind_method(D_METHOD("get_var", "allow_objects"), &_File::get_var, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("store_8", "value"), &_File::store_8);
	ClassDB::bind_method(D_METHOD("store_16", "value"), &_File::store_16);
	ClassDB::bind_method(D_METHOD("store_32", "value"), &_File::store_32);
	ClassDB::bind_method(D_METHOD("store_64", "value"), &_File::store_64);
	ClassDB::bind_method(D_METHOD("store_float", "value"), &_File::store_float);
	ClassDB::bind_method(D_METHOD("store_double", "value"), &_File::store_double);
	ClassDB::bind_method(D_METHOD("store_real", "value"), &_File::store_real);
	ClassDB::bind_method(D_METHOD("store_buffer", "buffer"), &_File::store_buffer);
	ClassDB::bind_method(D_METHOD("store_line", "line"), &_File::store_line);
	ClassDB::bind_method(D_METHOD("store_csv_line", "values", "delim"), &_File::store_csv_line, DEFVAL(","));
	ClassDB::bind_method(D_METHOD("store_string", "string"), &_File::store_string);
	ClassDB::bind_method(D_METHOD("store_var", "value", "full_objects"), &_File::store_var, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("store_pascal_string", "string"), &_File::store_pascal_string);
	ClassDB::bind_method(D_METHOD("get_pascal_string"), &_File::get_pascal_string);

	ClassDB::bind_method(D_METHOD("file_exists", "path"), &_File::file_exists);
	ClassDB::bind_method(D_METHOD("get_modified_time", "file"), &_File::get_modified_time);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "endian_swap"), "set_endian_swap", "get_endian_swap");

	BIND_ENUM_CONSTANT(READ);
	BIND_ENUM_CONSTANT(WRITE);
	BIND_ENUM_CONSTANT(READ_WRITE);
	BIND_ENUM_CONSTANT(WRITE_READ);

	BIND_ENUM_CONSTANT(COMPRESSION_FASTLZ);
	BIND_ENUM_CONSTANT(COMPRESSION_DEFLATE);
	BIND_ENUM_CONSTANT(COMPRESSION_ZSTD);
	BIND_ENUM_CONSTANT(COMPRESSION_GZIP);
}

_File::~_File() {
	if (f) {
		memdelete(f);
	}
}

////// _Directory //////

Error _Directory::open(const String &p_path) {
	Error err;
	DirAccess *alt = DirAccess::open(p_path, &err);

	if (!alt) {
		return err;
	}
	if (d) {
		memdelete(d);
	}
	d = alt;
	dir_open = true;

	return OK;
}

bool _Directory::is_open() const {
	return d && dir_open;
}

Error _Directory::list_dir_begin(bool p_skip_navigational, bool p_skip_hidden) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");

	_list_skip_navigational = p_skip_navigational;
	_list_skip_hidden = p_skip_hidden;

	return d->list_dir_begin();
}

String _Directory::get_next() {
	ERR_FAIL_COND_V_MSG(!is_open(), "", "Directory must be opened before use.");

	String next = d->get_next();
	while (next != "" && ((_list_skip_navigational && (next == "." || next == "..")) || (_list_skip_hidden && d->current_is_hidden()))) {
		next = d->get_next();
	}
	return next;
}

bool _Directory::current_is_dir() const {
	ERR_FAIL_COND_V_MSG(!is_open(), false, "Directory must be opened before use.");
	return d->current_is_dir();
}

void _Directory::list_dir_end() {
	ERR_FAIL_COND_MSG(!is_open(), "Directory must be opened before use.");
	d->list_dir_end();
}

int _Directory::get_drive_count() {
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "Directory must be opened before use.");
	return d->get_drive_count();
}

String _Directory::get_drive(int p_drive) {
	ERR_FAIL_COND_V_MSG(!is_open(), "", "Directory must be opened before use.");
	return d->get_drive(p_drive);
}

int _Directory::get_current_drive() {
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "Directory must be opened before use.");
	return d->get_current_drive();
}

Error _Directory::change_dir(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, ERR_UNCONFIGURED, "Directory is not configured properly.");
	Error err = d->change_dir(p_dir);

	if (err != OK) {
		return err;
	}
	dir_open = true;

	return OK;
}

String _Directory::get_current_dir() {
	ERR_FAIL_COND_V_MSG(!is_open(), "", "Directory must be opened before use.");
	return d->get_current_dir();
}

Error _Directory::make_dir(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, ERR_UNCONFIGURED, "Directory is not configured properly.");
	if (!p_dir.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		Error err = d->make_dir(p_dir);
		memdelete(d);
		return err;
	}
	return d->make_dir(p_dir);
}

Error _Directory::make_dir_recursive(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, ERR_UNCONFIGURED, "Directory is not configured properly.");
	if (!p_dir.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		Error err = d->make_dir_recursive(p_dir);
		memdelete(d);
		return err;
	}
	return d->make_dir_recursive(p_dir);
}

bool _Directory::file_exists(String p_file) {
	ERR_FAIL_COND_V_MSG(!d, false, "Directory is not configured properly.");
	if (!p_file.is_rel_path()) {
		return FileAccess::exists(p_file);
	}

	return d->file_exists(p_file);
}

bool _Directory::dir_exists(String p_dir) {
	ERR_FAIL_COND_V_MSG(!d, false, "Directory is not configured properly.");
	if (!p_dir.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		bool exists = d->dir_exists(p_dir);
		memdelete(d);
		return exists;
	}

	return d->dir_exists(p_dir);
}

int _Directory::get_space_left() {
	ERR_FAIL_COND_V_MSG(!is_open(), 0, "Directory must be opened before use.");
	return d->get_space_left() / 1024 * 1024; //return value in megabytes, given binding is int
}

Error _Directory::copy(String p_from, String p_to) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");
	return d->copy(p_from, p_to);
}

Error _Directory::rename(String p_from, String p_to) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");
	if (!p_from.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_from);
		ERR_FAIL_COND_V_MSG(!d->file_exists(p_from), ERR_DOES_NOT_EXIST, "File does not exist.");
		Error err = d->rename(p_from, p_to);
		memdelete(d);
		return err;
	}

	ERR_FAIL_COND_V_MSG(!d->file_exists(p_from), ERR_DOES_NOT_EXIST, "File does not exist.");
	return d->rename(p_from, p_to);
}

Error _Directory::remove(String p_name) {
	ERR_FAIL_COND_V_MSG(!is_open(), ERR_UNCONFIGURED, "Directory must be opened before use.");
	if (!p_name.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_name);
		Error err = d->remove(p_name);
		memdelete(d);
		return err;
	}

	return d->remove(p_name);
}

void _Directory::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open", "path"), &_Directory::open);
	ClassDB::bind_method(D_METHOD("list_dir_begin", "skip_navigational", "skip_hidden"), &_Directory::list_dir_begin, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_next"), &_Directory::get_next);
	ClassDB::bind_method(D_METHOD("current_is_dir"), &_Directory::current_is_dir);
	ClassDB::bind_method(D_METHOD("list_dir_end"), &_Directory::list_dir_end);
	ClassDB::bind_method(D_METHOD("get_drive_count"), &_Directory::get_drive_count);
	ClassDB::bind_method(D_METHOD("get_drive", "idx"), &_Directory::get_drive);
	ClassDB::bind_method(D_METHOD("get_current_drive"), &_Directory::get_current_drive);
	ClassDB::bind_method(D_METHOD("change_dir", "todir"), &_Directory::change_dir);
	ClassDB::bind_method(D_METHOD("get_current_dir"), &_Directory::get_current_dir);
	ClassDB::bind_method(D_METHOD("make_dir", "path"), &_Directory::make_dir);
	ClassDB::bind_method(D_METHOD("make_dir_recursive", "path"), &_Directory::make_dir_recursive);
	ClassDB::bind_method(D_METHOD("file_exists", "path"), &_Directory::file_exists);
	ClassDB::bind_method(D_METHOD("dir_exists", "path"), &_Directory::dir_exists);
	//ClassDB::bind_method(D_METHOD("get_modified_time","file"),&_Directory::get_modified_time);
	ClassDB::bind_method(D_METHOD("get_space_left"), &_Directory::get_space_left);
	ClassDB::bind_method(D_METHOD("copy", "from", "to"), &_Directory::copy);
	ClassDB::bind_method(D_METHOD("rename", "from", "to"), &_Directory::rename);
	ClassDB::bind_method(D_METHOD("remove", "path"), &_Directory::remove);
}

_Directory::_Directory() {
	d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
}

_Directory::~_Directory() {
	if (d) {
		memdelete(d);
	}
}

////// _Marshalls //////

_Marshalls *_Marshalls::singleton = nullptr;

_Marshalls *_Marshalls::get_singleton() {
	return singleton;
}

String _Marshalls::variant_to_base64(const Variant &p_var, bool p_full_objects) {
	int len;
	Error err = encode_variant(p_var, nullptr, len, p_full_objects);
	ERR_FAIL_COND_V_MSG(err != OK, "", "Error when trying to encode Variant.");

	Vector<uint8_t> buff;
	buff.resize(len);
	uint8_t *w = buff.ptrw();

	err = encode_variant(p_var, &w[0], len, p_full_objects);
	ERR_FAIL_COND_V_MSG(err != OK, "", "Error when trying to encode Variant.");

	String ret = CryptoCore::b64_encode_str(&w[0], len);
	ERR_FAIL_COND_V(ret == "", ret);

	return ret;
}

Variant _Marshalls::base64_to_variant(const String &p_str, bool p_allow_objects) {
	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	Vector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1);
	uint8_t *w = buf.ptrw();

	size_t len = 0;
	ERR_FAIL_COND_V(CryptoCore::b64_decode(&w[0], buf.size(), &len, (unsigned char *)cstr.get_data(), strlen) != OK, Variant());

	Variant v;
	Error err = decode_variant(v, &w[0], len, nullptr, p_allow_objects);
	ERR_FAIL_COND_V_MSG(err != OK, Variant(), "Error when trying to decode Variant.");

	return v;
}

String _Marshalls::raw_to_base64(const Vector<uint8_t> &p_arr) {
	String ret = CryptoCore::b64_encode_str(p_arr.ptr(), p_arr.size());
	ERR_FAIL_COND_V(ret == "", ret);
	return ret;
}

Vector<uint8_t> _Marshalls::base64_to_raw(const String &p_str) {
	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	size_t arr_len = 0;
	Vector<uint8_t> buf;
	{
		buf.resize(strlen / 4 * 3 + 1);
		uint8_t *w = buf.ptrw();

		ERR_FAIL_COND_V(CryptoCore::b64_decode(&w[0], buf.size(), &arr_len, (unsigned char *)cstr.get_data(), strlen) != OK, Vector<uint8_t>());
	}
	buf.resize(arr_len);

	return buf;
}

String _Marshalls::utf8_to_base64(const String &p_str) {
	CharString cstr = p_str.utf8();
	String ret = CryptoCore::b64_encode_str((unsigned char *)cstr.get_data(), cstr.length());
	ERR_FAIL_COND_V(ret == "", ret);
	return ret;
}

String _Marshalls::base64_to_utf8(const String &p_str) {
	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	Vector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1 + 1);
	uint8_t *w = buf.ptrw();

	size_t len = 0;
	ERR_FAIL_COND_V(CryptoCore::b64_decode(&w[0], buf.size(), &len, (unsigned char *)cstr.get_data(), strlen) != OK, String());

	w[len] = 0;
	String ret = String::utf8((char *)&w[0]);

	return ret;
}

void _Marshalls::_bind_methods() {
	ClassDB::bind_method(D_METHOD("variant_to_base64", "variant", "full_objects"), &_Marshalls::variant_to_base64, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("base64_to_variant", "base64_str", "allow_objects"), &_Marshalls::base64_to_variant, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("raw_to_base64", "array"), &_Marshalls::raw_to_base64);
	ClassDB::bind_method(D_METHOD("base64_to_raw", "base64_str"), &_Marshalls::base64_to_raw);

	ClassDB::bind_method(D_METHOD("utf8_to_base64", "utf8_str"), &_Marshalls::utf8_to_base64);
	ClassDB::bind_method(D_METHOD("base64_to_utf8", "base64_str"), &_Marshalls::base64_to_utf8);
}

////// _Semaphore //////

void _Semaphore::wait() {
	semaphore.wait();
}

Error _Semaphore::try_wait() {
	return semaphore.try_wait() ? OK : ERR_BUSY;
}

void _Semaphore::post() {
	semaphore.post();
}

void _Semaphore::_bind_methods() {
	ClassDB::bind_method(D_METHOD("wait"), &_Semaphore::wait);
	ClassDB::bind_method(D_METHOD("try_wait"), &_Semaphore::try_wait);
	ClassDB::bind_method(D_METHOD("post"), &_Semaphore::post);
}

////// _Mutex //////

void _Mutex::lock() {
	mutex.lock();
}

Error _Mutex::try_lock() {
	return mutex.try_lock();
}

void _Mutex::unlock() {
	mutex.unlock();
}

void _Mutex::_bind_methods() {
	ClassDB::bind_method(D_METHOD("lock"), &_Mutex::lock);
	ClassDB::bind_method(D_METHOD("try_lock"), &_Mutex::try_lock);
	ClassDB::bind_method(D_METHOD("unlock"), &_Mutex::unlock);
}

////// _Thread //////

void _Thread::_start_func(void *ud) {
	Ref<_Thread> *tud = (Ref<_Thread> *)ud;
	Ref<_Thread> t = *tud;
	memdelete(tud);
	Callable::CallError ce;
	const Variant *arg[1] = { &t->userdata };

	Thread::set_name(t->target_method);

	t->ret = t->target_instance->call(t->target_method, arg, 1, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		String reason;
		switch (ce.error) {
			case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT: {
				reason = "Invalid Argument #" + itos(ce.argument);
			} break;
			case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {
				reason = "Too Many Arguments";
			} break;
			case Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS: {
				reason = "Too Few Arguments";
			} break;
			case Callable::CallError::CALL_ERROR_INVALID_METHOD: {
				reason = "Method Not Found";
			} break;
			default: {
			}
		}

		ERR_FAIL_MSG("Could not call function '" + t->target_method.operator String() + "' to start thread " + t->get_id() + ": " + reason + ".");
	}
}

Error _Thread::start(Object *p_instance, const StringName &p_method, const Variant &p_userdata, Priority p_priority) {
	ERR_FAIL_COND_V_MSG(active, ERR_ALREADY_IN_USE, "Thread already started.");
	ERR_FAIL_COND_V(!p_instance, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_method == StringName(), ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_priority, PRIORITY_MAX, ERR_INVALID_PARAMETER);

	ret = Variant();
	target_method = p_method;
	target_instance = p_instance;
	userdata = p_userdata;
	active = true;

	Ref<_Thread> *ud = memnew(Ref<_Thread>(this));

	Thread::Settings s;
	s.priority = (Thread::Priority)p_priority;
	thread = Thread::create(_start_func, ud, s);
	if (!thread) {
		active = false;
		target_method = StringName();
		target_instance = nullptr;
		userdata = Variant();
		return ERR_CANT_CREATE;
	}

	return OK;
}

String _Thread::get_id() const {
	if (!thread) {
		return String();
	}

	return itos(thread->get_id());
}

bool _Thread::is_active() const {
	return active;
}

Variant _Thread::wait_to_finish() {
	ERR_FAIL_COND_V_MSG(!thread, Variant(), "Thread must exist to wait for its completion.");
	ERR_FAIL_COND_V_MSG(!active, Variant(), "Thread must be active to wait for its completion.");
	Thread::wait_to_finish(thread);
	Variant r = ret;
	active = false;
	target_method = StringName();
	target_instance = nullptr;
	userdata = Variant();
	if (thread) {
		memdelete(thread);
	}
	thread = nullptr;

	return r;
}

void _Thread::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start", "instance", "method", "userdata", "priority"), &_Thread::start, DEFVAL(Variant()), DEFVAL(PRIORITY_NORMAL));
	ClassDB::bind_method(D_METHOD("get_id"), &_Thread::get_id);
	ClassDB::bind_method(D_METHOD("is_active"), &_Thread::is_active);
	ClassDB::bind_method(D_METHOD("wait_to_finish"), &_Thread::wait_to_finish);

	BIND_ENUM_CONSTANT(PRIORITY_LOW);
	BIND_ENUM_CONSTANT(PRIORITY_NORMAL);
	BIND_ENUM_CONSTANT(PRIORITY_HIGH);
}

_Thread::~_Thread() {
	ERR_FAIL_COND_MSG(active, "Reference to a Thread object was lost while the thread is still running...");
}

////// _ClassDB //////

PackedStringArray _ClassDB::get_class_list() const {
	List<StringName> classes;
	ClassDB::get_class_list(&classes);

	PackedStringArray ret;
	ret.resize(classes.size());
	int idx = 0;
	for (List<StringName>::Element *E = classes.front(); E; E = E->next()) {
		ret.set(idx++, E->get());
	}

	return ret;
}

PackedStringArray _ClassDB::get_inheriters_from_class(const StringName &p_class) const {
	List<StringName> classes;
	ClassDB::get_inheriters_from_class(p_class, &classes);

	PackedStringArray ret;
	ret.resize(classes.size());
	int idx = 0;
	for (List<StringName>::Element *E = classes.front(); E; E = E->next()) {
		ret.set(idx++, E->get());
	}

	return ret;
}

StringName _ClassDB::get_parent_class(const StringName &p_class) const {
	return ClassDB::get_parent_class(p_class);
}

bool _ClassDB::class_exists(const StringName &p_class) const {
	return ClassDB::class_exists(p_class);
}

bool _ClassDB::is_parent_class(const StringName &p_class, const StringName &p_inherits) const {
	return ClassDB::is_parent_class(p_class, p_inherits);
}

bool _ClassDB::can_instance(const StringName &p_class) const {
	return ClassDB::can_instance(p_class);
}

Variant _ClassDB::instance(const StringName &p_class) const {
	Object *obj = ClassDB::instance(p_class);
	if (!obj) {
		return Variant();
	}

	Reference *r = Object::cast_to<Reference>(obj);
	if (r) {
		return REF(r);
	} else {
		return obj;
	}
}

bool _ClassDB::has_signal(StringName p_class, StringName p_signal) const {
	return ClassDB::has_signal(p_class, p_signal);
}

Dictionary _ClassDB::get_signal(StringName p_class, StringName p_signal) const {
	MethodInfo signal;
	if (ClassDB::get_signal(p_class, p_signal, &signal)) {
		return signal.operator Dictionary();
	} else {
		return Dictionary();
	}
}

Array _ClassDB::get_signal_list(StringName p_class, bool p_no_inheritance) const {
	List<MethodInfo> signals;
	ClassDB::get_signal_list(p_class, &signals, p_no_inheritance);
	Array ret;

	for (List<MethodInfo>::Element *E = signals.front(); E; E = E->next()) {
		ret.push_back(E->get().operator Dictionary());
	}

	return ret;
}

Array _ClassDB::get_property_list(StringName p_class, bool p_no_inheritance) const {
	List<PropertyInfo> plist;
	ClassDB::get_property_list(p_class, &plist, p_no_inheritance);
	Array ret;
	for (List<PropertyInfo>::Element *E = plist.front(); E; E = E->next()) {
		ret.push_back(E->get().operator Dictionary());
	}

	return ret;
}

Variant _ClassDB::get_property(Object *p_object, const StringName &p_property) const {
	Variant ret;
	ClassDB::get_property(p_object, p_property, ret);
	return ret;
}

Error _ClassDB::set_property(Object *p_object, const StringName &p_property, const Variant &p_value) const {
	Variant ret;
	bool valid;
	if (!ClassDB::set_property(p_object, p_property, p_value, &valid)) {
		return ERR_UNAVAILABLE;
	} else if (!valid) {
		return ERR_INVALID_DATA;
	}
	return OK;
}

bool _ClassDB::has_method(StringName p_class, StringName p_method, bool p_no_inheritance) const {
	return ClassDB::has_method(p_class, p_method, p_no_inheritance);
}

Array _ClassDB::get_method_list(StringName p_class, bool p_no_inheritance) const {
	List<MethodInfo> methods;
	ClassDB::get_method_list(p_class, &methods, p_no_inheritance);
	Array ret;

	for (List<MethodInfo>::Element *E = methods.front(); E; E = E->next()) {
#ifdef DEBUG_METHODS_ENABLED
		ret.push_back(E->get().operator Dictionary());
#else
		Dictionary dict;
		dict["name"] = E->get().name;
		ret.push_back(dict);
#endif
	}

	return ret;
}

PackedStringArray _ClassDB::get_integer_constant_list(const StringName &p_class, bool p_no_inheritance) const {
	List<String> constants;
	ClassDB::get_integer_constant_list(p_class, &constants, p_no_inheritance);

	PackedStringArray ret;
	ret.resize(constants.size());
	int idx = 0;
	for (List<String>::Element *E = constants.front(); E; E = E->next()) {
		ret.set(idx++, E->get());
	}

	return ret;
}

bool _ClassDB::has_integer_constant(const StringName &p_class, const StringName &p_name) const {
	bool success;
	ClassDB::get_integer_constant(p_class, p_name, &success);
	return success;
}

int _ClassDB::get_integer_constant(const StringName &p_class, const StringName &p_name) const {
	bool found;
	int c = ClassDB::get_integer_constant(p_class, p_name, &found);
	ERR_FAIL_COND_V(!found, 0);
	return c;
}

StringName _ClassDB::get_category(const StringName &p_node) const {
	return ClassDB::get_category(p_node);
}

bool _ClassDB::is_class_enabled(StringName p_class) const {
	return ClassDB::is_class_enabled(p_class);
}

void _ClassDB::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_class_list"), &_ClassDB::get_class_list);
	ClassDB::bind_method(D_METHOD("get_inheriters_from_class", "class"), &_ClassDB::get_inheriters_from_class);
	ClassDB::bind_method(D_METHOD("get_parent_class", "class"), &_ClassDB::get_parent_class);
	ClassDB::bind_method(D_METHOD("class_exists", "class"), &_ClassDB::class_exists);
	ClassDB::bind_method(D_METHOD("is_parent_class", "class", "inherits"), &_ClassDB::is_parent_class);
	ClassDB::bind_method(D_METHOD("can_instance", "class"), &_ClassDB::can_instance);
	ClassDB::bind_method(D_METHOD("instance", "class"), &_ClassDB::instance);

	ClassDB::bind_method(D_METHOD("class_has_signal", "class", "signal"), &_ClassDB::has_signal);
	ClassDB::bind_method(D_METHOD("class_get_signal", "class", "signal"), &_ClassDB::get_signal);
	ClassDB::bind_method(D_METHOD("class_get_signal_list", "class", "no_inheritance"), &_ClassDB::get_signal_list, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("class_get_property_list", "class", "no_inheritance"), &_ClassDB::get_property_list, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("class_get_property", "object", "property"), &_ClassDB::get_property);
	ClassDB::bind_method(D_METHOD("class_set_property", "object", "property", "value"), &_ClassDB::set_property);

	ClassDB::bind_method(D_METHOD("class_has_method", "class", "method", "no_inheritance"), &_ClassDB::has_method, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("class_get_method_list", "class", "no_inheritance"), &_ClassDB::get_method_list, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("class_get_integer_constant_list", "class", "no_inheritance"), &_ClassDB::get_integer_constant_list, DEFVAL(false));

	ClassDB::bind_method(D_METHOD("class_has_integer_constant", "class", "name"), &_ClassDB::has_integer_constant);
	ClassDB::bind_method(D_METHOD("class_get_integer_constant", "class", "name"), &_ClassDB::get_integer_constant);

	ClassDB::bind_method(D_METHOD("class_get_category", "class"), &_ClassDB::get_category);
	ClassDB::bind_method(D_METHOD("is_class_enabled", "class"), &_ClassDB::is_class_enabled);
}

////// _Engine //////

void _Engine::set_iterations_per_second(int p_ips) {
	Engine::get_singleton()->set_iterations_per_second(p_ips);
}

int _Engine::get_iterations_per_second() const {
	return Engine::get_singleton()->get_iterations_per_second();
}

void _Engine::set_physics_jitter_fix(float p_threshold) {
	Engine::get_singleton()->set_physics_jitter_fix(p_threshold);
}

float _Engine::get_physics_jitter_fix() const {
	return Engine::get_singleton()->get_physics_jitter_fix();
}

float _Engine::get_physics_interpolation_fraction() const {
	return Engine::get_singleton()->get_physics_interpolation_fraction();
}

void _Engine::set_target_fps(int p_fps) {
	Engine::get_singleton()->set_target_fps(p_fps);
}

int _Engine::get_target_fps() const {
	return Engine::get_singleton()->get_target_fps();
}

float _Engine::get_frames_per_second() const {
	return Engine::get_singleton()->get_frames_per_second();
}

uint64_t _Engine::get_physics_frames() const {
	return Engine::get_singleton()->get_physics_frames();
}

uint64_t _Engine::get_idle_frames() const {
	return Engine::get_singleton()->get_idle_frames();
}

void _Engine::set_time_scale(float p_scale) {
	Engine::get_singleton()->set_time_scale(p_scale);
}

float _Engine::get_time_scale() {
	return Engine::get_singleton()->get_time_scale();
}

int _Engine::get_frames_drawn() {
	return Engine::get_singleton()->get_frames_drawn();
}

MainLoop *_Engine::get_main_loop() const {
	//needs to remain in OS, since it's actually OS that interacts with it, but it's better exposed here
	return OS::get_singleton()->get_main_loop();
}

Dictionary _Engine::get_version_info() const {
	return Engine::get_singleton()->get_version_info();
}

Dictionary _Engine::get_author_info() const {
	return Engine::get_singleton()->get_author_info();
}

Array _Engine::get_copyright_info() const {
	return Engine::get_singleton()->get_copyright_info();
}

Dictionary _Engine::get_donor_info() const {
	return Engine::get_singleton()->get_donor_info();
}

Dictionary _Engine::get_license_info() const {
	return Engine::get_singleton()->get_license_info();
}

String _Engine::get_license_text() const {
	return Engine::get_singleton()->get_license_text();
}

bool _Engine::is_in_physics_frame() const {
	return Engine::get_singleton()->is_in_physics_frame();
}

bool _Engine::has_singleton(const String &p_name) const {
	return Engine::get_singleton()->has_singleton(p_name);
}

Object *_Engine::get_singleton_object(const String &p_name) const {
	return Engine::get_singleton()->get_singleton_object(p_name);
}

void _Engine::set_editor_hint(bool p_enabled) {
	Engine::get_singleton()->set_editor_hint(p_enabled);
}

bool _Engine::is_editor_hint() const {
	return Engine::get_singleton()->is_editor_hint();
}

void _Engine::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_iterations_per_second", "iterations_per_second"), &_Engine::set_iterations_per_second);
	ClassDB::bind_method(D_METHOD("get_iterations_per_second"), &_Engine::get_iterations_per_second);
	ClassDB::bind_method(D_METHOD("set_physics_jitter_fix", "physics_jitter_fix"), &_Engine::set_physics_jitter_fix);
	ClassDB::bind_method(D_METHOD("get_physics_jitter_fix"), &_Engine::get_physics_jitter_fix);
	ClassDB::bind_method(D_METHOD("get_physics_interpolation_fraction"), &_Engine::get_physics_interpolation_fraction);
	ClassDB::bind_method(D_METHOD("set_target_fps", "target_fps"), &_Engine::set_target_fps);
	ClassDB::bind_method(D_METHOD("get_target_fps"), &_Engine::get_target_fps);

	ClassDB::bind_method(D_METHOD("set_time_scale", "time_scale"), &_Engine::set_time_scale);
	ClassDB::bind_method(D_METHOD("get_time_scale"), &_Engine::get_time_scale);

	ClassDB::bind_method(D_METHOD("get_frames_drawn"), &_Engine::get_frames_drawn);
	ClassDB::bind_method(D_METHOD("get_frames_per_second"), &_Engine::get_frames_per_second);
	ClassDB::bind_method(D_METHOD("get_physics_frames"), &_Engine::get_physics_frames);
	ClassDB::bind_method(D_METHOD("get_idle_frames"), &_Engine::get_idle_frames);

	ClassDB::bind_method(D_METHOD("get_main_loop"), &_Engine::get_main_loop);

	ClassDB::bind_method(D_METHOD("get_version_info"), &_Engine::get_version_info);
	ClassDB::bind_method(D_METHOD("get_author_info"), &_Engine::get_author_info);
	ClassDB::bind_method(D_METHOD("get_copyright_info"), &_Engine::get_copyright_info);
	ClassDB::bind_method(D_METHOD("get_donor_info"), &_Engine::get_donor_info);
	ClassDB::bind_method(D_METHOD("get_license_info"), &_Engine::get_license_info);
	ClassDB::bind_method(D_METHOD("get_license_text"), &_Engine::get_license_text);

	ClassDB::bind_method(D_METHOD("is_in_physics_frame"), &_Engine::is_in_physics_frame);

	ClassDB::bind_method(D_METHOD("has_singleton", "name"), &_Engine::has_singleton);
	ClassDB::bind_method(D_METHOD("get_singleton", "name"), &_Engine::get_singleton_object);

	ClassDB::bind_method(D_METHOD("set_editor_hint", "enabled"), &_Engine::set_editor_hint);
	ClassDB::bind_method(D_METHOD("is_editor_hint"), &_Engine::is_editor_hint);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_hint"), "set_editor_hint", "is_editor_hint");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations_per_second"), "set_iterations_per_second", "get_iterations_per_second");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "target_fps"), "set_target_fps", "get_target_fps");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_scale"), "set_time_scale", "get_time_scale");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "physics_jitter_fix"), "set_physics_jitter_fix", "get_physics_jitter_fix");
}

_Engine *_Engine::singleton = nullptr;

////// _JSON //////

void JSONParseResult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_error"), &JSONParseResult::get_error);
	ClassDB::bind_method(D_METHOD("get_error_string"), &JSONParseResult::get_error_string);
	ClassDB::bind_method(D_METHOD("get_error_line"), &JSONParseResult::get_error_line);
	ClassDB::bind_method(D_METHOD("get_result"), &JSONParseResult::get_result);

	ClassDB::bind_method(D_METHOD("set_error", "error"), &JSONParseResult::set_error);
	ClassDB::bind_method(D_METHOD("set_error_string", "error_string"), &JSONParseResult::set_error_string);
	ClassDB::bind_method(D_METHOD("set_error_line", "error_line"), &JSONParseResult::set_error_line);
	ClassDB::bind_method(D_METHOD("set_result", "result"), &JSONParseResult::set_result);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "error", PROPERTY_HINT_NONE, "Error", PROPERTY_USAGE_CLASS_IS_ENUM), "set_error", "get_error");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "error_string"), "set_error_string", "get_error_string");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "error_line"), "set_error_line", "get_error_line");
	ADD_PROPERTY(PropertyInfo(Variant::NIL, "result", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), "set_result", "get_result");
}

void JSONParseResult::set_error(Error p_error) {
	error = p_error;
}

Error JSONParseResult::get_error() const {
	return error;
}

void JSONParseResult::set_error_string(const String &p_error_string) {
	error_string = p_error_string;
}

String JSONParseResult::get_error_string() const {
	return error_string;
}

void JSONParseResult::set_error_line(int p_error_line) {
	error_line = p_error_line;
}

int JSONParseResult::get_error_line() const {
	return error_line;
}

void JSONParseResult::set_result(const Variant &p_result) {
	result = p_result;
}

Variant JSONParseResult::get_result() const {
	return result;
}

void _JSON::_bind_methods() {
	ClassDB::bind_method(D_METHOD("print", "value", "indent", "sort_keys"), &_JSON::print, DEFVAL(String()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("parse", "json"), &_JSON::parse);
}

String _JSON::print(const Variant &p_value, const String &p_indent, bool p_sort_keys) {
	return JSON::print(p_value, p_indent, p_sort_keys);
}

Ref<JSONParseResult> _JSON::parse(const String &p_json) {
	Ref<JSONParseResult> result;
	result.instance();

	result->error = JSON::parse(p_json, result->result, result->error_string, result->error_line);

	if (result->error != OK) {
		ERR_PRINT(vformat("Error parsing JSON at line %s: %s", result->error_line, result->error_string));
	}
	return result;
}

_JSON *_JSON::singleton = nullptr;

////// _EngineDebugger //////

void _EngineDebugger::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_active"), &_EngineDebugger::is_active);

	ClassDB::bind_method(D_METHOD("register_profiler", "name", "toggle", "add", "tick"), &_EngineDebugger::register_profiler);
	ClassDB::bind_method(D_METHOD("unregister_profiler", "name"), &_EngineDebugger::unregister_profiler);
	ClassDB::bind_method(D_METHOD("is_profiling", "name"), &_EngineDebugger::is_profiling);
	ClassDB::bind_method(D_METHOD("has_profiler", "name"), &_EngineDebugger::has_profiler);

	ClassDB::bind_method(D_METHOD("profiler_add_frame_data", "name", "data"), &_EngineDebugger::profiler_add_frame_data);
	ClassDB::bind_method(D_METHOD("profiler_enable", "name", "enable", "arguments"), &_EngineDebugger::profiler_enable, DEFVAL(Array()));

	ClassDB::bind_method(D_METHOD("register_message_capture", "name", "callable"), &_EngineDebugger::register_message_capture);
	ClassDB::bind_method(D_METHOD("unregister_message_capture", "name"), &_EngineDebugger::unregister_message_capture);
	ClassDB::bind_method(D_METHOD("has_capture", "name"), &_EngineDebugger::has_capture);

	ClassDB::bind_method(D_METHOD("send_message", "message", "data"), &_EngineDebugger::send_message);
}

bool _EngineDebugger::is_active() {
	return EngineDebugger::is_active();
}

void _EngineDebugger::register_profiler(const StringName &p_name, const Callable &p_toggle, const Callable &p_add, const Callable &p_tick) {
	ERR_FAIL_COND_MSG(profilers.has(p_name) || has_profiler(p_name), "Profiler already registered: " + p_name);
	profilers.insert(p_name, ProfilerCallable(p_toggle, p_add, p_tick));
	ProfilerCallable &p = profilers[p_name];
	EngineDebugger::Profiler profiler(
			&p,
			&_EngineDebugger::call_toggle,
			&_EngineDebugger::call_add,
			&_EngineDebugger::call_tick);
	EngineDebugger::register_profiler(p_name, profiler);
}

void _EngineDebugger::unregister_profiler(const StringName &p_name) {
	ERR_FAIL_COND_MSG(!profilers.has(p_name), "Profiler not registered: " + p_name);
	EngineDebugger::unregister_profiler(p_name);
	profilers.erase(p_name);
}

bool _EngineDebugger::_EngineDebugger::is_profiling(const StringName &p_name) {
	return EngineDebugger::is_profiling(p_name);
}

bool _EngineDebugger::has_profiler(const StringName &p_name) {
	return EngineDebugger::has_profiler(p_name);
}

void _EngineDebugger::profiler_add_frame_data(const StringName &p_name, const Array &p_data) {
	EngineDebugger::profiler_add_frame_data(p_name, p_data);
}

void _EngineDebugger::profiler_enable(const StringName &p_name, bool p_enabled, const Array &p_opts) {
	if (EngineDebugger::get_singleton()) {
		EngineDebugger::get_singleton()->profiler_enable(p_name, p_enabled, p_opts);
	}
}

void _EngineDebugger::register_message_capture(const StringName &p_name, const Callable &p_callable) {
	ERR_FAIL_COND_MSG(captures.has(p_name) || has_capture(p_name), "Capture already registered: " + p_name);
	captures.insert(p_name, p_callable);
	Callable &c = captures[p_name];
	EngineDebugger::Capture capture(&c, &_EngineDebugger::call_capture);
	EngineDebugger::register_message_capture(p_name, capture);
}

void _EngineDebugger::unregister_message_capture(const StringName &p_name) {
	ERR_FAIL_COND_MSG(!captures.has(p_name), "Capture not registered: " + p_name);
	EngineDebugger::unregister_message_capture(p_name);
	captures.erase(p_name);
}

bool _EngineDebugger::has_capture(const StringName &p_name) {
	return EngineDebugger::has_capture(p_name);
}

void _EngineDebugger::send_message(const String &p_msg, const Array &p_data) {
	ERR_FAIL_COND_MSG(!EngineDebugger::is_active(), "Can't send message. No active debugger");
	EngineDebugger::get_singleton()->send_message(p_msg, p_data);
}

void _EngineDebugger::call_toggle(void *p_user, bool p_enable, const Array &p_opts) {
	Callable &toggle = ((ProfilerCallable *)p_user)->callable_toggle;
	if (toggle.is_null()) {
		return;
	}
	Variant enable = p_enable, opts = p_opts;
	const Variant *args[2] = { &enable, &opts };
	Variant retval;
	Callable::CallError err;
	toggle.call(args, 2, retval, err);
	ERR_FAIL_COND_MSG(err.error != Callable::CallError::CALL_OK, "Error calling 'toggle' to callable: " + Variant::get_callable_error_text(toggle, args, 2, err));
}

void _EngineDebugger::call_add(void *p_user, const Array &p_data) {
	Callable &add = ((ProfilerCallable *)p_user)->callable_add;
	if (add.is_null()) {
		return;
	}
	Variant data = p_data;
	const Variant *args[1] = { &data };
	Variant retval;
	Callable::CallError err;
	add.call(args, 1, retval, err);
	ERR_FAIL_COND_MSG(err.error != Callable::CallError::CALL_OK, "Error calling 'add' to callable: " + Variant::get_callable_error_text(add, args, 1, err));
}

void _EngineDebugger::call_tick(void *p_user, float p_frame_time, float p_idle_time, float p_physics_time, float p_physics_frame_time) {
	Callable &tick = ((ProfilerCallable *)p_user)->callable_tick;
	if (tick.is_null()) {
		return;
	}
	Variant frame_time = p_frame_time, idle_time = p_idle_time, physics_time = p_physics_time, physics_frame_time = p_physics_frame_time;
	const Variant *args[4] = { &frame_time, &idle_time, &physics_time, &physics_frame_time };
	Variant retval;
	Callable::CallError err;
	tick.call(args, 4, retval, err);
	ERR_FAIL_COND_MSG(err.error != Callable::CallError::CALL_OK, "Error calling 'tick' to callable: " + Variant::get_callable_error_text(tick, args, 4, err));
}

Error _EngineDebugger::call_capture(void *p_user, const String &p_cmd, const Array &p_data, bool &r_captured) {
	Callable &capture = *(Callable *)p_user;
	if (capture.is_null()) {
		return FAILED;
	}
	Variant cmd = p_cmd, data = p_data;
	const Variant *args[2] = { &cmd, &data };
	Variant retval;
	Callable::CallError err;
	capture.call(args, 2, retval, err);
	ERR_FAIL_COND_V_MSG(err.error != Callable::CallError::CALL_OK, FAILED, "Error calling 'capture' to callable: " + Variant::get_callable_error_text(capture, args, 2, err));
	ERR_FAIL_COND_V_MSG(retval.get_type() != Variant::BOOL, FAILED, "Error calling 'capture' to callable: " + String(capture) + ". Return type is not bool.");
	r_captured = retval;
	return OK;
}

_EngineDebugger::~_EngineDebugger() {
	for (Map<StringName, Callable>::Element *E = captures.front(); E; E = E->next()) {
		EngineDebugger::unregister_message_capture(E->key());
	}
	captures.clear();
	for (Map<StringName, ProfilerCallable>::Element *E = profilers.front(); E; E = E->next()) {
		EngineDebugger::unregister_profiler(E->key());
	}
	profilers.clear();
}

_EngineDebugger *_EngineDebugger::singleton = nullptr;
