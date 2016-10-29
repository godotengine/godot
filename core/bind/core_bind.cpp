/*************************************************************************/
/*  core_bind.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "os/os.h"
#include "geometry.h"
#include "io/marshalls.h"
#include "io/base64.h"
#include "core/globals.h"
#include "io/file_access_encrypted.h"
#include "os/keyboard.h"

/**
 *  Time constants borrowed from loc_time.h
 */
#define EPOCH_YR  1970    /* EPOCH = Jan 1 1970 00:00:00 */
#define SECS_DAY  (24L * 60L * 60L)
#define LEAPYEAR(year)  (!((year) % 4) && (((year) % 100) || !((year) % 400)))
#define YEARSIZE(year)  (LEAPYEAR(year) ? 366 : 365)
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

_ResourceLoader *_ResourceLoader::singleton=NULL;

Ref<ResourceInteractiveLoader> _ResourceLoader::load_interactive(const String& p_path,const String& p_type_hint) {
	return ResourceLoader::load_interactive(p_path,p_type_hint);
}

RES _ResourceLoader::load(const String &p_path,const String& p_type_hint, bool p_no_cache) {

	RES ret =  ResourceLoader::load(p_path,p_type_hint, p_no_cache);
	return ret;
}

DVector<String> _ResourceLoader::get_recognized_extensions_for_type(const String& p_type) {

	List<String> exts;
	ResourceLoader::get_recognized_extensions_for_type(p_type,&exts);
	DVector<String> ret;
	for(List<String>::Element *E=exts.front();E;E=E->next()) {

		ret.push_back(E->get());
	}

	return ret;
}

void _ResourceLoader::set_abort_on_missing_resources(bool p_abort) {

	ResourceLoader::set_abort_on_missing_resources(p_abort);
}

StringArray _ResourceLoader::get_dependencies(const String& p_path) {

	List<String> deps;
	ResourceLoader::get_dependencies(p_path, &deps);

	StringArray ret;
	for(List<String>::Element *E=deps.front();E;E=E->next()) {
		ret.push_back(E->get());
	}

	return ret;
};

bool _ResourceLoader::has(const String &p_path) {

	String local_path = Globals::get_singleton()->localize_path(p_path);
	return ResourceCache::has(local_path);
};

Ref<ResourceImportMetadata> _ResourceLoader::load_import_metadata(const String& p_path) {

	return ResourceLoader::load_import_metadata(p_path);
}

void _ResourceLoader::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("load_interactive:ResourceInteractiveLoader","path","type_hint"),&_ResourceLoader::load_interactive,DEFVAL(""));
	ObjectTypeDB::bind_method(_MD("load:Resource","path","type_hint", "p_no_cache"),&_ResourceLoader::load,DEFVAL(""), DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("load_import_metadata:ResourceImportMetadata","path"),&_ResourceLoader::load_import_metadata);
	ObjectTypeDB::bind_method(_MD("get_recognized_extensions_for_type","type"),&_ResourceLoader::get_recognized_extensions_for_type);
	ObjectTypeDB::bind_method(_MD("set_abort_on_missing_resources","abort"),&_ResourceLoader::set_abort_on_missing_resources);
	ObjectTypeDB::bind_method(_MD("get_dependencies","path"),&_ResourceLoader::get_dependencies);
	ObjectTypeDB::bind_method(_MD("has","path"),&_ResourceLoader::has);
}

_ResourceLoader::_ResourceLoader() {

	singleton=this;
}


Error _ResourceSaver::save(const String &p_path,const RES& p_resource, uint32_t p_flags) {

	ERR_FAIL_COND_V(p_resource.is_null(),ERR_INVALID_PARAMETER);
	return ResourceSaver::save(p_path,p_resource, p_flags);
}

DVector<String> _ResourceSaver::get_recognized_extensions(const RES& p_resource) {

	ERR_FAIL_COND_V(p_resource.is_null(),DVector<String>());
	List<String> exts;
	ResourceSaver::get_recognized_extensions(p_resource,&exts);
	DVector<String> ret;
	for(List<String>::Element *E=exts.front();E;E=E->next()) {

		ret.push_back(E->get());
	}
	return ret;
}

_ResourceSaver *_ResourceSaver::singleton=NULL;


void _ResourceSaver::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("save","path","resource:Resource","flags"),&_ResourceSaver::save,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_recognized_extensions","type"),&_ResourceSaver::get_recognized_extensions);

	BIND_CONSTANT(FLAG_RELATIVE_PATHS);
	BIND_CONSTANT(FLAG_BUNDLE_RESOURCES);
	BIND_CONSTANT(FLAG_CHANGE_PATH);
	BIND_CONSTANT(FLAG_OMIT_EDITOR_PROPERTIES);
	BIND_CONSTANT(FLAG_SAVE_BIG_ENDIAN);
	BIND_CONSTANT(FLAG_COMPRESS);
}

_ResourceSaver::_ResourceSaver() {

	singleton=this;
}


/////////////////OS


Point2 _OS::get_mouse_pos() const {

	return OS::get_singleton()->get_mouse_pos();
}
void _OS::set_window_title(const String& p_title) {

	OS::get_singleton()->set_window_title(p_title);

}

int _OS::get_mouse_button_state() const {

	return OS::get_singleton()->get_mouse_button_state();
}

String _OS::get_unique_ID() const {
	return OS::get_singleton()->get_unique_ID();
}
bool _OS::has_touchscreen_ui_hint() const {

	return OS::get_singleton()->has_touchscreen_ui_hint();
}

void _OS::set_clipboard(const String& p_text) {


	OS::get_singleton()->set_clipboard(p_text);
}
String _OS::get_clipboard() const {

	return OS::get_singleton()->get_clipboard();

}

void _OS::set_video_mode(const Size2& p_size, bool p_fullscreen,bool p_resizeable,int p_screen) {

	OS::VideoMode vm;
	vm.width=p_size.width;
	vm.height=p_size.height;
	vm.fullscreen=p_fullscreen;
	vm.resizable=p_resizeable;
	OS::get_singleton()->set_video_mode( vm,p_screen);


}
Size2 _OS::get_video_mode(int p_screen) const {

	OS::VideoMode vm;
	vm = OS::get_singleton()->get_video_mode(p_screen);
	return Size2(vm.width,vm.height);

}
bool _OS::is_video_mode_fullscreen(int p_screen) const {

	OS::VideoMode vm;
	vm = OS::get_singleton()->get_video_mode(p_screen);
	return vm.fullscreen;

}


int _OS::get_screen_count() const {
	return OS::get_singleton()->get_screen_count();
}

int _OS::get_current_screen() const {
	return OS::get_singleton()->get_current_screen();
}

void _OS::set_current_screen(int p_screen) {
	OS::get_singleton()->set_current_screen(p_screen);
}

Point2 _OS::get_screen_position(int p_screen) const {
	return OS::get_singleton()->get_screen_position(p_screen);
}

Size2 _OS::get_screen_size(int p_screen) const {
	return OS::get_singleton()->get_screen_size(p_screen);
}

int _OS::get_screen_dpi(int p_screen) const {

	return  OS::get_singleton()->get_screen_dpi(p_screen);
}

Point2 _OS::get_window_position() const {
	return OS::get_singleton()->get_window_position();
}

void _OS::set_window_position(const Point2& p_position) {
	OS::get_singleton()->set_window_position(p_position);
}

Size2 _OS::get_window_size() const {
	return OS::get_singleton()->get_window_size();
}

void _OS::set_window_size(const Size2& p_size) {
	OS::get_singleton()->set_window_size(p_size);
}

void _OS::set_window_fullscreen(bool p_enabled) {
	OS::get_singleton()->set_window_fullscreen(p_enabled);
}

bool _OS::is_window_fullscreen() const {
	return OS::get_singleton()->is_window_fullscreen();
}

void _OS::set_window_resizable(bool p_enabled) {
	OS::get_singleton()->set_window_resizable(p_enabled);
}

bool _OS::is_window_resizable() const {
	return OS::get_singleton()->is_window_resizable();
}

void _OS::set_window_minimized(bool p_enabled) {
	OS::get_singleton()->set_window_minimized(p_enabled);
}

bool _OS::is_window_minimized() const {
	return OS::get_singleton()->is_window_minimized();
}

void _OS::set_window_maximized(bool p_enabled) {
	OS::get_singleton()->set_window_maximized(p_enabled);
}

bool _OS::is_window_maximized() const {
	return OS::get_singleton()->is_window_maximized();
}

void _OS::set_borderless_window(bool p_borderless) {
	OS::get_singleton()->set_borderless_window(p_borderless);
}

bool _OS::get_borderless_window() const {
	return OS::get_singleton()->get_borderless_window();
}

void _OS::set_use_file_access_save_and_swap(bool p_enable) {

	FileAccess::set_backup_save(p_enable);
}

bool _OS::is_video_mode_resizable(int p_screen) const {

	OS::VideoMode vm;
	vm = OS::get_singleton()->get_video_mode(p_screen);
	return vm.resizable;
}

Array _OS::get_fullscreen_mode_list(int p_screen) const {

	List<OS::VideoMode> vmlist;
	OS::get_singleton()->get_fullscreen_mode_list(&vmlist,p_screen);
	Array vmarr;
	for(List<OS::VideoMode>::Element *E=vmlist.front();E;E=E->next() ){

		vmarr.push_back(Size2(E->get().width,E->get().height));
	}

	return vmarr;
}

void _OS::set_iterations_per_second(int p_ips) {

	OS::get_singleton()->set_iterations_per_second(p_ips);
}
int _OS::get_iterations_per_second() const {

	return OS::get_singleton()->get_iterations_per_second();

}

void _OS::set_target_fps(int p_fps) {
	OS::get_singleton()->set_target_fps(p_fps);
}

float _OS::get_target_fps() const {
	return OS::get_singleton()->get_target_fps();
}

void _OS::set_low_processor_usage_mode(bool p_enabled) {

	OS::get_singleton()->set_low_processor_usage_mode(p_enabled);
}
bool _OS::is_in_low_processor_usage_mode() const {

	return OS::get_singleton()->is_in_low_processor_usage_mode();
}

String _OS::get_executable_path() const {

	return OS::get_singleton()->get_executable_path();
}

Error _OS::shell_open(String p_uri) {

	return OS::get_singleton()->shell_open(p_uri);
};


int _OS::execute(const String& p_path, const Vector<String> & p_arguments, bool p_blocking, Array p_output) {

	OS::ProcessID pid;
	List<String> args;
	for(int i=0;i<p_arguments.size();i++)
		args.push_back(p_arguments[i]);
	String pipe;
	Error err = OS::get_singleton()->execute(p_path,args,p_blocking,&pid, &pipe);
	p_output.clear();
	p_output.push_back(pipe);
	if (err != OK)
		return -1;
	else
		return pid;

}
Error _OS::kill(int p_pid) {

	return OS::get_singleton()->kill(p_pid);
}

int _OS::get_process_ID() const {

	return OS::get_singleton()->get_process_ID();
};


bool _OS::has_environment(const String& p_var) const {

	return OS::get_singleton()->has_environment(p_var);
}
String _OS::get_environment(const String& p_var) const {

	return OS::get_singleton()->get_environment(p_var);
}

String _OS::get_name() const {

	return OS::get_singleton()->get_name();
}
Vector<String> _OS::get_cmdline_args() {

	List<String> cmdline = OS::get_singleton()->get_cmdline_args();
	Vector<String> cmdlinev;
	for(List<String>::Element *E=cmdline.front();E;E=E->next()) {

		cmdlinev.push_back(E->get());
	}

	return cmdlinev;
}

String _OS::get_locale() const {

	return OS::get_singleton()->get_locale();
}

String _OS::get_latin_keyboard_variant() const {
	switch( OS::get_singleton()->get_latin_keyboard_variant() ) {
		case OS::LATIN_KEYBOARD_QWERTY: return "QWERTY";
		case OS::LATIN_KEYBOARD_QWERTZ: return "QWERTZ";
		case OS::LATIN_KEYBOARD_AZERTY: return "AZERTY";
		case OS::LATIN_KEYBOARD_QZERTY: return "QZERTY";
		case OS::LATIN_KEYBOARD_DVORAK: return "DVORAK";
		case OS::LATIN_KEYBOARD_NEO   : return "NEO";
		default: return "ERROR";
	}
}

String _OS::get_model_name() const {

    return OS::get_singleton()->get_model_name();
}

MainLoop *_OS::get_main_loop() const {

	return OS::get_singleton()->get_main_loop();
}

void _OS::set_time_scale(float p_scale) {
	OS::get_singleton()->set_time_scale(p_scale);
}

float _OS::get_time_scale() {

	return OS::get_singleton()->get_time_scale();
}

bool _OS::is_ok_left_and_cancel_right() const {

	return OS::get_singleton()->get_swap_ok_cancel();
}

Error _OS::set_thread_name(const String& p_name) {

	return Thread::set_name(p_name);
};

void _OS::set_use_vsync(bool p_enable) {
	OS::get_singleton()->set_use_vsync(p_enable);
}

bool _OS::is_vsnc_enabled() const {

	return OS::get_singleton()->is_vsnc_enabled();
}


/*
enum Weekday {
	DAY_SUNDAY,
	DAY_MONDAY,
	DAY_TUESDAY,
	DAY_WEDNESDAY,
	DAY_THURSDAY,
	DAY_FRIDAY,
	DAY_SATURDAY
};

enum Month {
	MONTH_JANUARY,
	MONTH_FEBRUARY,
	MONTH_MARCH,
	MONTH_APRIL,
	MONTH_MAY,
	MONTH_JUNE,
	MONTH_JULY,
	MONTH_AUGUST,
	MONTH_SEPTEMBER,
	MONTH_OCTOBER,
	MONTH_NOVEMBER,
	MONTH_DECEMBER
};
*/
/*
struct Date {

	int year;
	Month month;
	int day;
	Weekday weekday;
	bool dst;
};

struct Time {

	int hour;
	int min;
	int sec;
};
*/

int _OS::get_static_memory_usage() const {

	return OS::get_singleton()->get_static_memory_usage();

}

int _OS::get_static_memory_peak_usage() const {

	return OS::get_singleton()->get_static_memory_peak_usage();

}

int _OS::get_dynamic_memory_usage() const{

	return OS::get_singleton()->get_dynamic_memory_usage();
}


void _OS::set_icon(const Image& p_icon) {

	OS::get_singleton()->set_icon(p_icon);
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

	for(int i = 0; i < keys.size(); i++) {
		dated[keys[i]] = timed[keys[i]];
	}

	return dated;
}

Dictionary _OS::get_date(bool utc) const {

	OS::Date date = OS::get_singleton()->get_date(utc);
	Dictionary dated;
	dated[YEAR_KEY]=date.year;
	dated[MONTH_KEY]=date.month;
	dated[DAY_KEY]=date.day;
	dated[WEEKDAY_KEY]=date.weekday;
	dated[DST_KEY]=date.dst;
	return dated;
}

Dictionary _OS::get_time(bool utc) const {

	OS::Time time = OS::get_singleton()->get_time(utc);
	Dictionary timed;
	timed[HOUR_KEY]=time.hour;
	timed[MINUTE_KEY]=time.min;
	timed[SECOND_KEY]=time.sec;
	return timed;
}

/**
 *  Get a epoch time value from a dictionary of time values
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
uint64_t _OS::get_unix_time_from_datetime(Dictionary datetime) const {

	// Bunch of conversion constants
	static const unsigned int SECONDS_PER_MINUTE = 60;
	static const unsigned int MINUTES_PER_HOUR = 60;
	static const unsigned int HOURS_PER_DAY = 24;
	static const unsigned int SECONDS_PER_HOUR = MINUTES_PER_HOUR *
		SECONDS_PER_MINUTE;
	static const unsigned int SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY;

	// Get all time values from the dictionary, set to zero if it doesn't exist.
	//   Risk incorrect calculation over throwing errors
	unsigned int second = ((datetime.has(SECOND_KEY))?
			static_cast<unsigned int>(datetime[SECOND_KEY]): 0);
	unsigned int minute = ((datetime.has(MINUTE_KEY))?
			static_cast<unsigned int>(datetime[MINUTE_KEY]): 0);
	unsigned int hour = ((datetime.has(HOUR_KEY))?
			static_cast<unsigned int>(datetime[HOUR_KEY]): 0);
	unsigned int day = ((datetime.has(DAY_KEY))?
			static_cast<unsigned int>(datetime[DAY_KEY]): 0);
	unsigned int month = ((datetime.has(MONTH_KEY))?
			static_cast<unsigned int>(datetime[MONTH_KEY]) -1: 0);
	unsigned int year = ((datetime.has(YEAR_KEY))?
			static_cast<unsigned int>(datetime[YEAR_KEY]):0);

	/// How many days come before each month (0-12)
	static const unsigned short int DAYS_PAST_THIS_YEAR_TABLE[2][13] =
	{
		/* Normal years.  */
		{ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 },
		/* Leap years.  */
		{ 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 }
	};

	ERR_EXPLAIN("Invalid second value of: " + itos(second));
	ERR_FAIL_COND_V( second > 59, 0);

	ERR_EXPLAIN("Invalid minute value of: " + itos(minute));
	ERR_FAIL_COND_V( minute > 59, 0);

	ERR_EXPLAIN("Invalid hour value of: " + itos(hour));
	ERR_FAIL_COND_V( hour > 23, 0);

	ERR_EXPLAIN("Invalid month value of: " + itos(month+1));
	ERR_FAIL_COND_V( month+1 > 12, 0);

	// Do this check after month is tested as valid
	ERR_EXPLAIN("Invalid day value of: " + itos(day) + " which is larger "
			"than "+ itos(MONTH_DAYS_TABLE[LEAPYEAR(year)][month]));
	ERR_FAIL_COND_V( day > MONTH_DAYS_TABLE[LEAPYEAR(year)][month], 0);

	// Calculate all the seconds from months past in this year
	uint64_t SECONDS_FROM_MONTHS_PAST_THIS_YEAR =
		DAYS_PAST_THIS_YEAR_TABLE[LEAPYEAR(year)][month] * SECONDS_PER_DAY;

	uint64_t SECONDS_FROM_YEARS_PAST = 0;
	for(unsigned int iyear = EPOCH_YR; iyear < year; iyear++) {

		SECONDS_FROM_YEARS_PAST += YEARSIZE(iyear) *
			SECONDS_PER_DAY;
	}

	uint64_t epoch =
		second +
		minute * SECONDS_PER_MINUTE +
		hour * SECONDS_PER_HOUR +
		// Subtract 1 from day, since the current day isn't over yet
		//   and we cannot count all 24 hours.
		(day-1) * SECONDS_PER_DAY +
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
Dictionary _OS::get_datetime_from_unix_time( uint64_t unix_time_val) const {

	// Just fail if unix time is negative (when interpreted as an int).
	//  This means the user passed in a negative value by accident
	ERR_EXPLAIN("unix_time_val was really huge!"+ itos(unix_time_val) +
			" You probably passed in a negative value!");
	ERR_FAIL_COND_V( (int64_t)unix_time_val < 0, Dictionary());

	OS::Date date;
	OS::Time time;

	unsigned long dayclock, dayno;
	int year = EPOCH_YR;

	dayclock = (unsigned long)unix_time_val % SECS_DAY;
	dayno = (unsigned long)unix_time_val / SECS_DAY;

	time.sec = dayclock % 60;
	time.min = (dayclock % 3600) / 60;
	time.hour = dayclock / 3600;

	/* day 0 was a thursday */
	date.weekday = static_cast<OS::Weekday>((dayno + 4) % 7);

	while (dayno >= YEARSIZE(year)) {
		dayno -= YEARSIZE(year);
		year++;
	}

	date.year = year;

	size_t imonth = 0;

	while (dayno >= MONTH_DAYS_TABLE[LEAPYEAR(year)][imonth]) {
		dayno -= MONTH_DAYS_TABLE[LEAPYEAR(year)][imonth];
		imonth++;
	}

	/// Add 1 to month to make sure months are indexed starting at 1
	date.month = static_cast<OS::Month>(imonth+1);

	date.day = dayno + 1;

	Dictionary timed;
	timed[HOUR_KEY]=time.hour;
	timed[MINUTE_KEY]=time.min;
	timed[SECOND_KEY]=time.sec;
	timed[YEAR_KEY]=date.year;
	timed[MONTH_KEY]=date.month;
	timed[DAY_KEY]=date.day;
	timed[WEEKDAY_KEY]=date.weekday;

	return timed;
}

Dictionary _OS::get_time_zone_info() const {
	OS::TimeZoneInfo info = OS::get_singleton()->get_time_zone_info();
	Dictionary infod;
	infod["bias"] = info.bias;
	infod["name"] = info.name;
	return infod;
}

uint64_t _OS::get_unix_time() const {

	return OS::get_singleton()->get_unix_time();
}

uint64_t _OS::get_system_time_secs() const {
	return OS::get_singleton()->get_system_time_secs();
}

void _OS::delay_usec(uint32_t p_usec) const {

	OS::get_singleton()->delay_usec(p_usec);
}

void _OS::delay_msec(uint32_t p_msec) const {

	OS::get_singleton()->delay_usec(int64_t(p_msec)*1000);
}

uint32_t _OS::get_ticks_msec() const {

	return OS::get_singleton()->get_ticks_msec();
}

uint32_t _OS::get_splash_tick_msec() const {

	return OS::get_singleton()->get_splash_tick_msec();
}

bool _OS::can_use_threads() const {

	return OS::get_singleton()->can_use_threads();
}

bool _OS::can_draw() const {

	return OS::get_singleton()->can_draw();
}

int  _OS::get_frames_drawn() {

	return OS::get_singleton()->get_frames_drawn();
}

int _OS::get_processor_count() const {

	return OS::get_singleton()->get_processor_count();
}

bool _OS::is_stdout_verbose() const {

	return OS::get_singleton()->is_stdout_verbose();

}

void _OS::dump_memory_to_file(const String& p_file) {

	OS::get_singleton()->dump_memory_to_file(p_file.utf8().get_data());
}

struct _OSCoreBindImg {

	String path;
	Size2 size;
	int fmt;
	ObjectID id;
	int vram;
	bool operator<(const _OSCoreBindImg& p_img) const { return vram==p_img.vram ? id<p_img.id : vram > p_img.vram; }
};

void _OS::print_all_textures_by_size() {


	List<_OSCoreBindImg> imgs;
	int total=0;
	{
		List<Ref<Resource> > rsrc;
		ResourceCache::get_cached_resources(&rsrc);

		for (List<Ref<Resource> >::Element *E=rsrc.front();E;E=E->next()) {

			if (!E->get()->is_type("ImageTexture"))
				continue;

			Size2 size = E->get()->call("get_size");
			int fmt = E->get()->call("get_format");

			_OSCoreBindImg img;
			img.size=size;
			img.fmt=fmt;
			img.path=E->get()->get_path();
			img.vram=Image::get_image_data_size(img.size.width,img.size.height,Image::Format(img.fmt));
			img.id=E->get()->get_instance_ID();
			total+=img.vram;
			imgs.push_back(img);
		}
	}

	imgs.sort();

	for(List<_OSCoreBindImg>::Element *E=imgs.front();E;E=E->next()) {

		total-=E->get().vram;
	}
}

void _OS::print_resources_by_type(const Vector<String>& p_types) {

	Map<String,int> type_count;

	List<Ref<Resource> > resources;
	ResourceCache::get_cached_resources(&resources);

	List<Ref<Resource> > rsrc;
	ResourceCache::get_cached_resources(&rsrc);

	for (List<Ref<Resource> >::Element *E=rsrc.front();E;E=E->next()) {

		Ref<Resource> r = E->get();

		bool found = false;

		for (int i=0; i<p_types.size(); i++) {
			if (r->is_type(p_types[i]))
				found = true;
		}
		if (!found)
			continue;

		if (!type_count.has(r->get_type())) {
			type_count[r->get_type()]=0;
		}


		type_count[r->get_type()]++;
	}

};

bool _OS::has_virtual_keyboard() const {
	return OS::get_singleton()->has_virtual_keyboard();
}

void _OS::show_virtual_keyboard(const String& p_existing_text) {
	OS::get_singleton()->show_virtual_keyboard(p_existing_text, Rect2());
}

void _OS::hide_virtual_keyboard() {
	OS::get_singleton()->hide_virtual_keyboard();
}

void _OS::print_all_resources(const String& p_to_file ) {

	OS::get_singleton()->print_all_resources(p_to_file);
}

void _OS::print_resources_in_use(bool p_short) {

	OS::get_singleton()->print_resources_in_use(p_short);
}

void _OS::dump_resources_to_file(const String& p_file) {

	OS::get_singleton()->dump_resources_to_file(p_file.utf8().get_data());
}

String _OS::get_data_dir() const {

	return OS::get_singleton()->get_data_dir();
};

float _OS::get_frames_per_second() const {

	return OS::get_singleton()->get_frames_per_second();
}

Error _OS::native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {

	return OS::get_singleton()->native_video_play(p_path, p_volume, p_audio_track, p_subtitle_track);
};

bool _OS::native_video_is_playing() {

	return OS::get_singleton()->native_video_is_playing();
};

void _OS::native_video_pause() {

	OS::get_singleton()->native_video_pause();
};

void _OS::native_video_unpause() {
	OS::get_singleton()->native_video_unpause();
};

void _OS::native_video_stop() {

	OS::get_singleton()->native_video_stop();
};

void _OS::request_attention() {

	OS::get_singleton()->request_attention();
}

bool _OS::is_debug_build() const {

#ifdef DEBUG_ENABLED
	return true;
#else
	return false;
#endif

}

void _OS::set_screen_orientation(ScreenOrientation p_orientation) {

	OS::get_singleton()->set_screen_orientation(OS::ScreenOrientation(p_orientation));
}

_OS::ScreenOrientation _OS::get_screen_orientation() const {

	return ScreenOrientation(OS::get_singleton()->get_screen_orientation());
}

void _OS::set_keep_screen_on(bool p_enabled) {

	OS::get_singleton()->set_keep_screen_on(p_enabled);
}

bool _OS::is_keep_screen_on() const {

	return OS::get_singleton()->is_keep_screen_on();
}

String _OS::get_system_dir(SystemDir p_dir) const {

	return OS::get_singleton()->get_system_dir(OS::SystemDir(p_dir));
}

String _OS::get_custom_level() const {

	return OS::get_singleton()->get_custom_level();
}

String _OS::get_scancode_string(uint32_t p_code) const {

	return keycode_get_string(p_code);
}
bool _OS::is_scancode_unicode(uint32_t p_unicode) const {

	return keycode_has_unicode(p_unicode);
}
int _OS::find_scancode_from_string(const String& p_code) const {

	return find_keycode(p_code);
}

void _OS::alert(const String& p_alert,const String& p_title) {

	OS::get_singleton()->alert(p_alert,p_title);
}

Dictionary _OS::get_engine_version() const {

	return OS::get_singleton()->get_engine_version();
}

_OS *_OS::singleton=NULL;

void _OS::_bind_methods() {

	//ObjectTypeDB::bind_method(_MD("get_mouse_pos"),&_OS::get_mouse_pos);
	//ObjectTypeDB::bind_method(_MD("is_mouse_grab_enabled"),&_OS::is_mouse_grab_enabled);

	ObjectTypeDB::bind_method(_MD("set_clipboard","clipboard"),&_OS::set_clipboard);
	ObjectTypeDB::bind_method(_MD("get_clipboard"),&_OS::get_clipboard);

	ObjectTypeDB::bind_method(_MD("set_video_mode","size","fullscreen","resizable","screen"),&_OS::set_video_mode,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_video_mode_size","screen"),&_OS::get_video_mode,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("is_video_mode_fullscreen","screen"),&_OS::is_video_mode_fullscreen,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("is_video_mode_resizable","screen"),&_OS::is_video_mode_resizable,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_fullscreen_mode_list","screen"),&_OS::get_fullscreen_mode_list,DEFVAL(0));


	ObjectTypeDB::bind_method(_MD("get_screen_count"),&_OS::get_screen_count);
	ObjectTypeDB::bind_method(_MD("get_current_screen"),&_OS::get_current_screen);
	ObjectTypeDB::bind_method(_MD("set_current_screen","screen"),&_OS::set_current_screen);
	ObjectTypeDB::bind_method(_MD("get_screen_position","screen"),&_OS::get_screen_position,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_screen_size","screen"),&_OS::get_screen_size,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_screen_dpi","screen"),&_OS::get_screen_dpi,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_window_position"),&_OS::get_window_position);
	ObjectTypeDB::bind_method(_MD("set_window_position","position"),&_OS::set_window_position);
	ObjectTypeDB::bind_method(_MD("get_window_size"),&_OS::get_window_size);
	ObjectTypeDB::bind_method(_MD("set_window_size","size"),&_OS::set_window_size);
	ObjectTypeDB::bind_method(_MD("set_window_fullscreen","enabled"),&_OS::set_window_fullscreen);
	ObjectTypeDB::bind_method(_MD("is_window_fullscreen"),&_OS::is_window_fullscreen);
	ObjectTypeDB::bind_method(_MD("set_window_resizable","enabled"),&_OS::set_window_resizable);
	ObjectTypeDB::bind_method(_MD("is_window_resizable"),&_OS::is_window_resizable);
	ObjectTypeDB::bind_method(_MD("set_window_minimized", "enabled"),&_OS::set_window_minimized);
	ObjectTypeDB::bind_method(_MD("is_window_minimized"),&_OS::is_window_minimized);
	ObjectTypeDB::bind_method(_MD("set_window_maximized", "enabled"),&_OS::set_window_maximized);
	ObjectTypeDB::bind_method(_MD("is_window_maximized"),&_OS::is_window_maximized);
	ObjectTypeDB::bind_method(_MD("request_attention"), &_OS::request_attention);

	ObjectTypeDB::bind_method(_MD("set_borderless_window", "borderless"), &_OS::set_borderless_window);
	ObjectTypeDB::bind_method(_MD("get_borderless_window"), &_OS::get_borderless_window);

	ObjectTypeDB::bind_method(_MD("set_screen_orientation","orientation"),&_OS::set_screen_orientation);
	ObjectTypeDB::bind_method(_MD("get_screen_orientation"),&_OS::get_screen_orientation);

	ObjectTypeDB::bind_method(_MD("set_keep_screen_on","enabled"),&_OS::set_keep_screen_on);
	ObjectTypeDB::bind_method(_MD("is_keep_screen_on"),&_OS::is_keep_screen_on);

	ObjectTypeDB::bind_method(_MD("set_iterations_per_second","iterations_per_second"),&_OS::set_iterations_per_second);
	ObjectTypeDB::bind_method(_MD("get_iterations_per_second"),&_OS::get_iterations_per_second);
	ObjectTypeDB::bind_method(_MD("set_target_fps","target_fps"),&_OS::set_target_fps);
	ObjectTypeDB::bind_method(_MD("get_target_fps"),&_OS::get_target_fps);

	ObjectTypeDB::bind_method(_MD("set_time_scale","time_scale"),&_OS::set_time_scale);
	ObjectTypeDB::bind_method(_MD("get_time_scale"),&_OS::get_time_scale);

	ObjectTypeDB::bind_method(_MD("has_touchscreen_ui_hint"),&_OS::has_touchscreen_ui_hint);

	ObjectTypeDB::bind_method(_MD("set_window_title","title"),&_OS::set_window_title);

	ObjectTypeDB::bind_method(_MD("set_low_processor_usage_mode","enable"),&_OS::set_low_processor_usage_mode);
	ObjectTypeDB::bind_method(_MD("is_in_low_processor_usage_mode"),&_OS::is_in_low_processor_usage_mode);

	ObjectTypeDB::bind_method(_MD("get_processor_count"),&_OS::get_processor_count);

	ObjectTypeDB::bind_method(_MD("get_executable_path"),&_OS::get_executable_path);
	ObjectTypeDB::bind_method(_MD("execute","path","arguments","blocking","output"),&_OS::execute,DEFVAL(Array()));
	ObjectTypeDB::bind_method(_MD("kill","pid"),&_OS::kill);
	ObjectTypeDB::bind_method(_MD("shell_open","uri"),&_OS::shell_open);
	ObjectTypeDB::bind_method(_MD("get_process_ID"),&_OS::get_process_ID);

	ObjectTypeDB::bind_method(_MD("get_environment","environment"),&_OS::get_environment);
	ObjectTypeDB::bind_method(_MD("has_environment","environment"),&_OS::has_environment);

	ObjectTypeDB::bind_method(_MD("get_name"),&_OS::get_name);
	ObjectTypeDB::bind_method(_MD("get_cmdline_args"),&_OS::get_cmdline_args);
	ObjectTypeDB::bind_method(_MD("get_main_loop"),&_OS::get_main_loop);

	ObjectTypeDB::bind_method(_MD("get_datetime","utc"),&_OS::get_datetime,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("get_date","utc"),&_OS::get_date,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("get_time","utc"),&_OS::get_time,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("get_time_zone_info"),&_OS::get_time_zone_info);
	ObjectTypeDB::bind_method(_MD("get_unix_time"),&_OS::get_unix_time);
	ObjectTypeDB::bind_method(_MD("get_datetime_from_unix_time", "unix_time_val"),
			&_OS::get_datetime_from_unix_time);
	ObjectTypeDB::bind_method(_MD("get_unix_time_from_datetime", "datetime"),
			&_OS::get_unix_time_from_datetime);
	ObjectTypeDB::bind_method(_MD("get_system_time_secs"), &_OS::get_system_time_secs);

	ObjectTypeDB::bind_method(_MD("set_icon","icon"),&_OS::set_icon);

	ObjectTypeDB::bind_method(_MD("delay_usec","usec"),&_OS::delay_usec);
	ObjectTypeDB::bind_method(_MD("delay_msec","msec"),&_OS::delay_msec);
	ObjectTypeDB::bind_method(_MD("get_ticks_msec"),&_OS::get_ticks_msec);
	ObjectTypeDB::bind_method(_MD("get_splash_tick_msec"),&_OS::get_splash_tick_msec);
	ObjectTypeDB::bind_method(_MD("get_locale"),&_OS::get_locale);
	ObjectTypeDB::bind_method(_MD("get_latin_keyboard_variant"),&_OS::get_latin_keyboard_variant);
	ObjectTypeDB::bind_method(_MD("get_model_name"),&_OS::get_model_name);

	ObjectTypeDB::bind_method(_MD("get_custom_level"),&_OS::get_custom_level);

	ObjectTypeDB::bind_method(_MD("can_draw"),&_OS::can_draw);
	ObjectTypeDB::bind_method(_MD("get_frames_drawn"),&_OS::get_frames_drawn);
	ObjectTypeDB::bind_method(_MD("is_stdout_verbose"),&_OS::is_stdout_verbose);

	ObjectTypeDB::bind_method(_MD("can_use_threads"),&_OS::can_use_threads);

	ObjectTypeDB::bind_method(_MD("is_debug_build"),&_OS::is_debug_build);

	//ObjectTypeDB::bind_method(_MD("get_mouse_button_state"),&_OS::get_mouse_button_state);

	ObjectTypeDB::bind_method(_MD("dump_memory_to_file","file"),&_OS::dump_memory_to_file);
	ObjectTypeDB::bind_method(_MD("dump_resources_to_file","file"),&_OS::dump_resources_to_file);
	ObjectTypeDB::bind_method(_MD("has_virtual_keyboard"),&_OS::has_virtual_keyboard);
	ObjectTypeDB::bind_method(_MD("show_virtual_keyboard", "existing_text"),&_OS::show_virtual_keyboard,DEFVAL(""));
	ObjectTypeDB::bind_method(_MD("hide_virtual_keyboard"),&_OS::hide_virtual_keyboard);
	ObjectTypeDB::bind_method(_MD("print_resources_in_use","short"),&_OS::print_resources_in_use,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("print_all_resources","tofile"),&_OS::print_all_resources,DEFVAL(""));

	ObjectTypeDB::bind_method(_MD("get_static_memory_usage"),&_OS::get_static_memory_usage);
	ObjectTypeDB::bind_method(_MD("get_static_memory_peak_usage"),&_OS::get_static_memory_peak_usage);
	ObjectTypeDB::bind_method(_MD("get_dynamic_memory_usage"),&_OS::get_dynamic_memory_usage);

	ObjectTypeDB::bind_method(_MD("get_data_dir"),&_OS::get_data_dir);
	ObjectTypeDB::bind_method(_MD("get_system_dir","dir"),&_OS::get_system_dir);
	ObjectTypeDB::bind_method(_MD("get_unique_ID"),&_OS::get_unique_ID);

	ObjectTypeDB::bind_method(_MD("is_ok_left_and_cancel_right"),&_OS::is_ok_left_and_cancel_right);

	ObjectTypeDB::bind_method(_MD("get_frames_per_second"),&_OS::get_frames_per_second);

	ObjectTypeDB::bind_method(_MD("print_all_textures_by_size"),&_OS::print_all_textures_by_size);
	ObjectTypeDB::bind_method(_MD("print_resources_by_type","types"),&_OS::print_resources_by_type);

	ObjectTypeDB::bind_method(_MD("native_video_play","path","volume","audio_track","subtitle_track"),&_OS::native_video_play);
	ObjectTypeDB::bind_method(_MD("native_video_is_playing"),&_OS::native_video_is_playing);
	ObjectTypeDB::bind_method(_MD("native_video_stop"),&_OS::native_video_stop);
	ObjectTypeDB::bind_method(_MD("native_video_pause"),&_OS::native_video_pause);
	ObjectTypeDB::bind_method(_MD("native_video_unpause"),&_OS::native_video_unpause);

	ObjectTypeDB::bind_method(_MD("get_scancode_string","code"),&_OS::get_scancode_string);
	ObjectTypeDB::bind_method(_MD("is_scancode_unicode","code"),&_OS::is_scancode_unicode);
	ObjectTypeDB::bind_method(_MD("find_scancode_from_string","string"),&_OS::find_scancode_from_string);

	ObjectTypeDB::bind_method(_MD("set_use_file_access_save_and_swap","enabled"),&_OS::set_use_file_access_save_and_swap);

	ObjectTypeDB::bind_method(_MD("alert","text","title"),&_OS::alert,DEFVAL("Alert!"));

	ObjectTypeDB::bind_method(_MD("set_thread_name","name"),&_OS::set_thread_name);

	ObjectTypeDB::bind_method(_MD("set_use_vsync","enable"),&_OS::set_use_vsync);
	ObjectTypeDB::bind_method(_MD("is_vsnc_enabled"),&_OS::is_vsnc_enabled);

	ObjectTypeDB::bind_method(_MD("get_engine_version"),&_OS::get_engine_version);

	BIND_CONSTANT( DAY_SUNDAY );
	BIND_CONSTANT( DAY_MONDAY );
	BIND_CONSTANT( DAY_TUESDAY );
	BIND_CONSTANT( DAY_WEDNESDAY );
	BIND_CONSTANT( DAY_THURSDAY );
	BIND_CONSTANT( DAY_FRIDAY );
	BIND_CONSTANT( DAY_SATURDAY );

	BIND_CONSTANT( MONTH_JANUARY );
	BIND_CONSTANT( MONTH_FEBRUARY );
	BIND_CONSTANT( MONTH_MARCH );
	BIND_CONSTANT( MONTH_APRIL );
	BIND_CONSTANT( MONTH_MAY );
	BIND_CONSTANT( MONTH_JUNE );
	BIND_CONSTANT( MONTH_JULY );
	BIND_CONSTANT( MONTH_AUGUST );
	BIND_CONSTANT( MONTH_SEPTEMBER );
	BIND_CONSTANT( MONTH_OCTOBER );
	BIND_CONSTANT( MONTH_NOVEMBER );
	BIND_CONSTANT( MONTH_DECEMBER );

	BIND_CONSTANT( SCREEN_ORIENTATION_LANDSCAPE );
	BIND_CONSTANT( SCREEN_ORIENTATION_PORTRAIT );
	BIND_CONSTANT( SCREEN_ORIENTATION_REVERSE_LANDSCAPE );
	BIND_CONSTANT( SCREEN_ORIENTATION_REVERSE_PORTRAIT );
	BIND_CONSTANT( SCREEN_ORIENTATION_SENSOR_LANDSCAPE );
	BIND_CONSTANT( SCREEN_ORIENTATION_SENSOR_PORTRAIT );
	BIND_CONSTANT( SCREEN_ORIENTATION_SENSOR );

	BIND_CONSTANT( SYSTEM_DIR_DESKTOP);
	BIND_CONSTANT( SYSTEM_DIR_DCIM );
	BIND_CONSTANT( SYSTEM_DIR_DOCUMENTS );
	BIND_CONSTANT( SYSTEM_DIR_DOWNLOADS );
	BIND_CONSTANT( SYSTEM_DIR_MOVIES );
	BIND_CONSTANT( SYSTEM_DIR_MUSIC );
	BIND_CONSTANT( SYSTEM_DIR_PICTURES );
	BIND_CONSTANT( SYSTEM_DIR_RINGTONES );

}

_OS::_OS() {

	singleton=this;
}


///////////////////// GEOMETRY


_Geometry *_Geometry::singleton=NULL;

_Geometry *_Geometry::get_singleton() {

	return singleton;
}

DVector<Plane> _Geometry::build_box_planes(const Vector3& p_extents) {

	return Geometry::build_box_planes(p_extents);
}

DVector<Plane> _Geometry::build_cylinder_planes(float p_radius, float p_height, int p_sides, Vector3::Axis p_axis) {

	return Geometry::build_cylinder_planes(p_radius,p_height,p_sides,p_axis);
}
DVector<Plane> _Geometry::build_capsule_planes(float p_radius, float p_height, int p_sides, int p_lats, Vector3::Axis p_axis) {

	return Geometry::build_capsule_planes(p_radius,p_height,p_sides,p_lats,p_axis);
}

real_t _Geometry::segment_intersects_circle(const Vector2& p_from, const Vector2& p_to, const Vector2& p_circle_pos, real_t p_circle_radius) {

	return Geometry::segment_intersects_circle(p_from,p_to,p_circle_pos,p_circle_radius);
}

Variant _Geometry::segment_intersects_segment_2d(const Vector2& p_from_a,const Vector2& p_to_a,const Vector2& p_from_b,const Vector2& p_to_b) {

	Vector2 result;
	if (Geometry::segment_intersects_segment_2d(p_from_a, p_to_a, p_from_b, p_to_b, &result)) {

		return result;
	} else {
		return Variant();
	};
};

DVector<Vector2> _Geometry::get_closest_points_between_segments_2d( const Vector2& p1,const Vector2& q1, const Vector2& p2,const Vector2& q2) {

	Vector2 r1, r2;
	Geometry::get_closest_points_between_segments(p1,q1,p2,q2,r1,r2);
	DVector<Vector2> r;
	r.resize(2);
	r.set(0,r1);
	r.set(1,r2);
	return r;
}

DVector<Vector3> _Geometry::get_closest_points_between_segments(const Vector3& p1,const Vector3& p2,const Vector3& q1,const Vector3& q2) {

	Vector3 r1, r2;
	Geometry::get_closest_points_between_segments(p1,p2,q1,q2,r1,r2);
	DVector<Vector3> r;
	r.resize(2);
	r.set(0,r1);
	r.set(1,r2);
	return r;

}
Vector3 _Geometry::get_closest_point_to_segment(const Vector3& p_point, const Vector3& p_a,const Vector3& p_b) {

	Vector3 s[2]={p_a,p_b};
	return Geometry::get_closest_point_to_segment(p_point,s);
}
Variant _Geometry::ray_intersects_triangle( const Vector3& p_from, const Vector3& p_dir, const Vector3& p_v0,const Vector3& p_v1,const Vector3& p_v2) {

	Vector3 res;
	if (Geometry::ray_intersects_triangle(p_from,p_dir,p_v0,p_v1,p_v2,&res))
		return res;
	else
		return Variant();


}
Variant _Geometry::segment_intersects_triangle( const Vector3& p_from, const Vector3& p_to, const Vector3& p_v0,const Vector3& p_v1,const Vector3& p_v2) {

	Vector3 res;
	if (Geometry::segment_intersects_triangle(p_from,p_to,p_v0,p_v1,p_v2,&res))
		return res;
	else
		return Variant();

}

bool _Geometry::point_is_inside_triangle(const Vector2& s, const Vector2& a, const Vector2& b, const Vector2& c) const {

	return Geometry::is_point_in_triangle(s,a,b,c);
}

DVector<Vector3> _Geometry::segment_intersects_sphere( const Vector3& p_from, const Vector3& p_to, const Vector3& p_sphere_pos,real_t p_sphere_radius) {

	DVector<Vector3> r;
	Vector3 res,norm;
	if (!Geometry::segment_intersects_sphere(p_from,p_to,p_sphere_pos,p_sphere_radius,&res,&norm))
		return r;

	r.resize(2);
	r.set(0,res);
	r.set(1,norm);
	return r;
}
DVector<Vector3> _Geometry::segment_intersects_cylinder( const Vector3& p_from, const Vector3& p_to, float p_height,float p_radius) {

	DVector<Vector3> r;
	Vector3 res,norm;
	if (!Geometry::segment_intersects_cylinder(p_from,p_to,p_height,p_radius,&res,&norm))
		return r;

	r.resize(2);
	r.set(0,res);
	r.set(1,norm);
	return r;

}
DVector<Vector3> _Geometry::segment_intersects_convex(const Vector3& p_from, const Vector3& p_to,const Vector<Plane>& p_planes) {

	DVector<Vector3> r;
	Vector3 res,norm;
	if (!Geometry::segment_intersects_convex(p_from,p_to,p_planes.ptr(),p_planes.size(),&res,&norm))
		return r;

	r.resize(2);
	r.set(0,res);
	r.set(1,norm);
	return r;
}

Vector<int> _Geometry::triangulate_polygon(const Vector<Vector2>& p_polygon) {

	return Geometry::triangulate_polygon(p_polygon);
}

Dictionary _Geometry::make_atlas(const Vector<Size2>& p_rects) {

	Dictionary ret;

	Vector<Size2i> rects;
	for (int i=0; i<p_rects.size(); i++) {

		rects.push_back(p_rects[i]);
	};

	Vector<Point2i> result;
	Size2i size;

	Geometry::make_atlas(rects, result, size);

	Size2 r_size = size;
	Vector<Point2> r_result;
	for (int i=0; i<result.size(); i++) {

		r_result.push_back(result[i]);
	};


	ret["points"] = r_result;
	ret["size"] = r_size;

	return ret;
};


int _Geometry::get_uv84_normal_bit(const Vector3& p_vector) {

	return Geometry::get_uv84_normal_bit(p_vector);
}


void _Geometry::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("build_box_planes","extents"),&_Geometry::build_box_planes);
	ObjectTypeDB::bind_method(_MD("build_cylinder_planes","radius","height","sides","axis"),&_Geometry::build_cylinder_planes,DEFVAL(Vector3::AXIS_Z));
	ObjectTypeDB::bind_method(_MD("build_capsule_planes","radius","height","sides","lats","axis"),&_Geometry::build_capsule_planes,DEFVAL(Vector3::AXIS_Z));
	ObjectTypeDB::bind_method(_MD("segment_intersects_circle","segment_from","segment_to","circle_pos","circle_radius"),&_Geometry::segment_intersects_circle);
	ObjectTypeDB::bind_method(_MD("segment_intersects_segment_2d","from_a","to_a","from_b","to_b"),&_Geometry::segment_intersects_segment_2d);

	ObjectTypeDB::bind_method(_MD("get_closest_points_between_segments_2d","p1","q1","p2","q2"),&_Geometry::get_closest_points_between_segments_2d);
	ObjectTypeDB::bind_method(_MD("get_closest_points_between_segments","p1","p2","q1","q2"),&_Geometry::get_closest_points_between_segments);

	ObjectTypeDB::bind_method(_MD("get_closest_point_to_segment","point","s1","s2"),&_Geometry::get_closest_point_to_segment);

	ObjectTypeDB::bind_method(_MD("get_uv84_normal_bit","normal"),&_Geometry::get_uv84_normal_bit);

	ObjectTypeDB::bind_method(_MD("ray_intersects_triangle","from","dir","a","b","c"),&_Geometry::ray_intersects_triangle);
	ObjectTypeDB::bind_method(_MD("segment_intersects_triangle","from","to","a","b","c"),&_Geometry::segment_intersects_triangle);
	ObjectTypeDB::bind_method(_MD("segment_intersects_sphere","from","to","spos","sradius"),&_Geometry::segment_intersects_sphere);
	ObjectTypeDB::bind_method(_MD("segment_intersects_cylinder","from","to","height","radius"),&_Geometry::segment_intersects_cylinder);
	ObjectTypeDB::bind_method(_MD("segment_intersects_convex","from","to","planes"),&_Geometry::segment_intersects_convex);
	ObjectTypeDB::bind_method(_MD("point_is_inside_triangle","point","a","b","c"),&_Geometry::point_is_inside_triangle);

	ObjectTypeDB::bind_method(_MD("triangulate_polygon","polygon"),&_Geometry::triangulate_polygon);

	ObjectTypeDB::bind_method(_MD("make_atlas","sizes"),&_Geometry::make_atlas);
}


_Geometry::_Geometry() {
	singleton=this;
}


///////////////////////// FILE



Error _File::open_encrypted(const String& p_path, int p_mode_flags,const Vector<uint8_t>& p_key) {

	Error err = open(p_path,p_mode_flags);
	if (err)
		return err;

	FileAccessEncrypted *fae = memnew( FileAccessEncrypted );
	err = fae->open_and_parse(f,p_key,(p_mode_flags==WRITE)?FileAccessEncrypted::MODE_WRITE_AES256:FileAccessEncrypted::MODE_READ);
	if (err) {
		memdelete(fae);
		close();
		return err;
	}
	f=fae;
	return OK;
}

Error _File::open_encrypted_pass(const String& p_path, int p_mode_flags,const String& p_pass) {

	Error err = open(p_path,p_mode_flags);
	if (err)
		return err;

	FileAccessEncrypted *fae = memnew( FileAccessEncrypted );
	err = fae->open_and_parse_password(f,p_pass,(p_mode_flags==WRITE)?FileAccessEncrypted::MODE_WRITE_AES256:FileAccessEncrypted::MODE_READ);
	if (err) {
		memdelete(fae);
		close();
		return err;
	}

	f=fae;
	return OK;

}


Error _File::open(const String& p_path, int p_mode_flags) {

	close();
	Error err;
	f = FileAccess::open(p_path,p_mode_flags,&err);
	if (f)
		f->set_endian_swap(eswap);
	return err;

}

void _File::close(){

	if (f)
		memdelete(f);
	f=NULL;
}
bool _File::is_open() const{


	return f!=NULL;
}

void _File::seek(int64_t p_position){

	ERR_FAIL_COND(!f);
	f->seek(p_position);

}
void _File::seek_end(int64_t p_position){


	ERR_FAIL_COND(!f);
	f->seek_end(p_position);
}
int64_t _File::get_pos() const{


	ERR_FAIL_COND_V(!f,0);
	return f->get_pos();
}

int64_t _File::get_len() const{

	ERR_FAIL_COND_V(!f,0);
	return f->get_len();
}

bool _File::eof_reached() const{

	ERR_FAIL_COND_V(!f,false);
	return f->eof_reached();
}

uint8_t _File::get_8() const{

	ERR_FAIL_COND_V(!f,0);
	return f->get_8();

}
uint16_t _File::get_16() const{

	ERR_FAIL_COND_V(!f,0);
	return f->get_16();

}
uint32_t _File::get_32() const{

	ERR_FAIL_COND_V(!f,0);
	return f->get_32();

}
uint64_t _File::get_64() const{

	ERR_FAIL_COND_V(!f,0);
	return f->get_64();

}

float _File::get_float() const{

	ERR_FAIL_COND_V(!f,0);
	return f->get_float();

}
double _File::get_double() const{


	ERR_FAIL_COND_V(!f,0);
	return f->get_double();
}
real_t _File::get_real() const{


	ERR_FAIL_COND_V(!f,0);
	return f->get_real();
}

DVector<uint8_t> _File::get_buffer(int p_length) const{

	DVector<uint8_t> data;
	ERR_FAIL_COND_V(!f,data);

	ERR_FAIL_COND_V(p_length<0,data);
	if (p_length==0)
		return data;
	Error err = data.resize(p_length);
	ERR_FAIL_COND_V(err!=OK,data);
	DVector<uint8_t>::Write w = data.write();
	int len = f->get_buffer(&w[0],p_length);
	ERR_FAIL_COND_V( len < 0 , DVector<uint8_t>());

	w = DVector<uint8_t>::Write();

	if (len < p_length)
		data.resize(p_length);

	return data;

}


String _File::get_as_text() const {

	ERR_FAIL_COND_V(!f, String());

	String text;
	size_t original_pos = f->get_pos();
	f->seek(0);

	String l = get_line();
	while(!eof_reached()) {
		text+=l+"\n";
		l = get_line();
	}
	text+=l;

	f->seek(original_pos);

	return text;


}


String _File::get_md5(const String& p_path) const {

	return FileAccess::get_md5(p_path);

}

String _File::get_sha256(const String& p_path) const {

	return FileAccess::get_sha256(p_path);

}


String _File::get_line() const{

	ERR_FAIL_COND_V(!f,String());
	return f->get_line();

}

Vector<String> _File::get_csv_line(String delim) const {
	ERR_FAIL_COND_V(!f,Vector<String>());
	return f->get_csv_line(delim);
}

/**< use this for files WRITTEN in _big_ endian machines (ie, amiga/mac)
 * It's not about the current CPU type but file formats.
 * this flags get reset to false (little endian) on each open
 */

void _File::set_endian_swap(bool p_swap){


	eswap=p_swap;
	if (f)
		f->set_endian_swap(p_swap);

}
bool _File::get_endian_swap(){


	return eswap;
}

Error _File::get_error() const{

	if (!f)
		return ERR_UNCONFIGURED;
	return f->get_error();
}

void _File::store_8(uint8_t p_dest){

	ERR_FAIL_COND(!f);

	f->store_8(p_dest);
}
void _File::store_16(uint16_t p_dest){

	ERR_FAIL_COND(!f);

	f->store_16(p_dest);
}
void _File::store_32(uint32_t p_dest){

	ERR_FAIL_COND(!f);

	f->store_32(p_dest);
}
void _File::store_64(uint64_t p_dest){

	ERR_FAIL_COND(!f);

	f->store_64(p_dest);
}

void _File::store_float(float p_dest){

	ERR_FAIL_COND(!f);

	f->store_float(p_dest);
}
void _File::store_double(double p_dest){

	ERR_FAIL_COND(!f);

	f->store_double(p_dest);
}
void _File::store_real(real_t p_real){

	ERR_FAIL_COND(!f);

	f->store_real(p_real);
}

void _File::store_string(const String& p_string){

	ERR_FAIL_COND(!f);

	f->store_string(p_string);
}

void _File::store_pascal_string(const String& p_string) {

	ERR_FAIL_COND(!f);

	f->store_pascal_string(p_string);
};

String _File::get_pascal_string() {

	ERR_FAIL_COND_V(!f, "");

	return f->get_pascal_string();
};

void _File::store_line(const String& p_string){

	ERR_FAIL_COND(!f);
	f->store_line(p_string);
}

void _File::store_buffer(const DVector<uint8_t>& p_buffer){

	ERR_FAIL_COND(!f);

	int len = p_buffer.size();
	if (len==0)
		return;

	DVector<uint8_t>::Read r = p_buffer.read();

	f->store_buffer(&r[0],len);
}

bool _File::file_exists(const String& p_name) const{

	return FileAccess::exists(p_name);


}

void _File::store_var(const Variant& p_var) {

	ERR_FAIL_COND(!f);
	int len;
	Error err = encode_variant(p_var,NULL,len);
	ERR_FAIL_COND( err != OK );

	DVector<uint8_t> buff;
	buff.resize(len);
	DVector<uint8_t>::Write w = buff.write();

	err = encode_variant(p_var,&w[0],len);
	ERR_FAIL_COND( err != OK );
	w=DVector<uint8_t>::Write();

	store_32(len);
	store_buffer(buff);
}

Variant _File::get_var() const {

	ERR_FAIL_COND_V(!f,Variant());
	uint32_t len = get_32();
	DVector<uint8_t> buff = get_buffer(len);
	ERR_FAIL_COND_V(buff.size() != len, Variant());

	DVector<uint8_t>::Read r = buff.read();

	Variant v;
	Error err = decode_variant(v,&r[0],len);
	ERR_FAIL_COND_V( err!=OK, Variant() );

	return v;
}

void _File::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("open_encrypted","path","mode_flags","key"),&_File::open_encrypted);
	ObjectTypeDB::bind_method(_MD("open_encrypted_with_pass","path","mode_flags","pass"),&_File::open_encrypted_pass);

	ObjectTypeDB::bind_method(_MD("open","path","flags"),&_File::open);
	ObjectTypeDB::bind_method(_MD("close"),&_File::close);
	ObjectTypeDB::bind_method(_MD("is_open"),&_File::is_open);
	ObjectTypeDB::bind_method(_MD("seek","pos"),&_File::seek);
	ObjectTypeDB::bind_method(_MD("seek_end","pos"),&_File::seek_end,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_pos"),&_File::get_pos);
	ObjectTypeDB::bind_method(_MD("get_len"),&_File::get_len);
	ObjectTypeDB::bind_method(_MD("eof_reached"),&_File::eof_reached);
	ObjectTypeDB::bind_method(_MD("get_8"),&_File::get_8);
	ObjectTypeDB::bind_method(_MD("get_16"),&_File::get_16);
	ObjectTypeDB::bind_method(_MD("get_32"),&_File::get_32);
	ObjectTypeDB::bind_method(_MD("get_64"),&_File::get_64);
	ObjectTypeDB::bind_method(_MD("get_float"),&_File::get_float);
	ObjectTypeDB::bind_method(_MD("get_double"),&_File::get_double);
	ObjectTypeDB::bind_method(_MD("get_real"),&_File::get_real);
	ObjectTypeDB::bind_method(_MD("get_buffer","len"),&_File::get_buffer);
	ObjectTypeDB::bind_method(_MD("get_line"),&_File::get_line);
	ObjectTypeDB::bind_method(_MD("get_as_text"),&_File::get_as_text);
	ObjectTypeDB::bind_method(_MD("get_md5","path"),&_File::get_md5);
	ObjectTypeDB::bind_method(_MD("get_sha256","path"),&_File::get_sha256);
	ObjectTypeDB::bind_method(_MD("get_endian_swap"),&_File::get_endian_swap);
	ObjectTypeDB::bind_method(_MD("set_endian_swap","enable"),&_File::set_endian_swap);
	ObjectTypeDB::bind_method(_MD("get_error:Error"),&_File::get_error);
	ObjectTypeDB::bind_method(_MD("get_var"),&_File::get_var);
	ObjectTypeDB::bind_method(_MD("get_csv_line","delim"),&_File::get_csv_line,DEFVAL(","));

	ObjectTypeDB::bind_method(_MD("store_8","value"),&_File::store_8);
	ObjectTypeDB::bind_method(_MD("store_16","value"),&_File::store_16);
	ObjectTypeDB::bind_method(_MD("store_32","value"),&_File::store_32);
	ObjectTypeDB::bind_method(_MD("store_64","value"),&_File::store_64);
	ObjectTypeDB::bind_method(_MD("store_float","value"),&_File::store_float);
	ObjectTypeDB::bind_method(_MD("store_double","value"),&_File::store_double);
	ObjectTypeDB::bind_method(_MD("store_real","value"),&_File::store_real);
	ObjectTypeDB::bind_method(_MD("store_buffer","buffer"),&_File::store_buffer);
	ObjectTypeDB::bind_method(_MD("store_line","line"),&_File::store_line);
	ObjectTypeDB::bind_method(_MD("store_string","string"),&_File::store_string);
	ObjectTypeDB::bind_method(_MD("store_var","value"),&_File::store_var);

	ObjectTypeDB::bind_method(_MD("store_pascal_string","string"),&_File::store_pascal_string);
	ObjectTypeDB::bind_method(_MD("get_pascal_string"),&_File::get_pascal_string);

	ObjectTypeDB::bind_method(_MD("file_exists","path"),&_File::file_exists);

	BIND_CONSTANT( READ );
	BIND_CONSTANT( WRITE );
	BIND_CONSTANT( READ_WRITE );
	BIND_CONSTANT( WRITE_READ );
}

_File::_File(){

	f = NULL;
	eswap=false;

}

_File::~_File(){

	if (f)
		memdelete(f);

}

///////////////////////////////////////////////////////




Error _Directory::open(const String& p_path) {
	Error err;
	DirAccess *alt=DirAccess::open(p_path,&err);

	if (!alt)
		return err;
	if (d)
		memdelete(d);
	d=alt;

	return OK;
}

bool _Directory::list_dir_begin() {

	ERR_FAIL_COND_V(!d,false);
	return d->list_dir_begin();
}

String _Directory::get_next(){

	ERR_FAIL_COND_V(!d,"");
	return d->get_next();
}
bool _Directory::current_is_dir() const{

	ERR_FAIL_COND_V(!d,false);
	return d->current_is_dir();
}

void _Directory::list_dir_end(){

	ERR_FAIL_COND(!d);
	return d->list_dir_end();
}

int _Directory::get_drive_count(){

	ERR_FAIL_COND_V(!d,0);
	return d->get_drive_count();
}
String _Directory::get_drive(int p_drive){

	ERR_FAIL_COND_V(!d,"");
	return d->get_drive(p_drive);
}

Error _Directory::change_dir(String p_dir){

	ERR_FAIL_COND_V(!d,ERR_UNCONFIGURED);
	return d->change_dir(p_dir);
}
String _Directory::get_current_dir() {

	ERR_FAIL_COND_V(!d,"");
	return d->get_current_dir();
}
Error _Directory::make_dir(String p_dir){

	ERR_FAIL_COND_V(!d,ERR_UNCONFIGURED);
	if (!p_dir.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		Error err = d->make_dir(p_dir);
		memdelete(d);
		return err;

	}
	return d->make_dir(p_dir);
}
Error _Directory::make_dir_recursive(String p_dir){

	ERR_FAIL_COND_V(!d,ERR_UNCONFIGURED);
	if (!p_dir.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_dir);
		Error err = d->make_dir_recursive(p_dir);
		memdelete(d);
		return err;

	}
	return d->make_dir_recursive(p_dir);
}

bool _Directory::file_exists(String p_file){

	ERR_FAIL_COND_V(!d,false);

	if (!p_file.is_rel_path()) {
		return FileAccess::exists(p_file);
	}

	return d->file_exists(p_file);
}

bool _Directory::dir_exists(String p_dir) {
	ERR_FAIL_COND_V(!d,false);
	if (!p_dir.is_rel_path()) {

		DirAccess *d = DirAccess::create_for_path(p_dir);
		bool exists = d->dir_exists(p_dir);
		memdelete(d);
		return exists;

	} else {
		return d->dir_exists(p_dir);
	}
}

int _Directory::get_space_left(){

	ERR_FAIL_COND_V(!d,0);
	return d->get_space_left()/1024*1024; //return value in megabytes, given binding is int
}

Error _Directory::copy(String p_from,String p_to){

	ERR_FAIL_COND_V(!d,ERR_UNCONFIGURED);
	return d->copy(p_from,p_to);
}
Error _Directory::rename(String p_from, String p_to){

	ERR_FAIL_COND_V(!d,ERR_UNCONFIGURED);
	if (!p_from.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_from);
		Error err = d->rename(p_from,p_to);
		memdelete(d);
		return err;
	}

	return d->rename(p_from,p_to);

}
Error _Directory::remove(String p_name){

	ERR_FAIL_COND_V(!d,ERR_UNCONFIGURED);
	if (!p_name.is_rel_path()) {
		DirAccess *d = DirAccess::create_for_path(p_name);
		Error err = d->remove(p_name);
		memdelete(d);
		return err;
	}

	return d->remove(p_name);
}

void _Directory::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("open:Error","path"),&_Directory::open);
	ObjectTypeDB::bind_method(_MD("list_dir_begin"),&_Directory::list_dir_begin);
	ObjectTypeDB::bind_method(_MD("get_next"),&_Directory::get_next);
	ObjectTypeDB::bind_method(_MD("current_is_dir"),&_Directory::current_is_dir);
	ObjectTypeDB::bind_method(_MD("list_dir_end"),&_Directory::list_dir_end);
	ObjectTypeDB::bind_method(_MD("get_drive_count"),&_Directory::get_drive_count);
	ObjectTypeDB::bind_method(_MD("get_drive","idx"),&_Directory::get_drive);
	ObjectTypeDB::bind_method(_MD("change_dir:Error","todir"),&_Directory::change_dir);
	ObjectTypeDB::bind_method(_MD("get_current_dir"),&_Directory::get_current_dir);
	ObjectTypeDB::bind_method(_MD("make_dir:Error","path"),&_Directory::make_dir);
	ObjectTypeDB::bind_method(_MD("make_dir_recursive:Error","path"),&_Directory::make_dir_recursive);
	ObjectTypeDB::bind_method(_MD("file_exists","path"),&_Directory::file_exists);
	ObjectTypeDB::bind_method(_MD("dir_exists","path"),&_Directory::dir_exists);
//	ObjectTypeDB::bind_method(_MD("get_modified_time","file"),&_Directory::get_modified_time);
	ObjectTypeDB::bind_method(_MD("get_space_left"),&_Directory::get_space_left);
	ObjectTypeDB::bind_method(_MD("copy:Error","from","to"),&_Directory::copy);
	ObjectTypeDB::bind_method(_MD("rename:Error","from","to"),&_Directory::rename);
	ObjectTypeDB::bind_method(_MD("remove:Error","path"),&_Directory::remove);

}

_Directory::_Directory() {

	d = DirAccess::create(DirAccess::ACCESS_RESOURCES);
}

_Directory::~_Directory() {

	if (d)
		memdelete(d);
}

String _Marshalls::variant_to_base64(const Variant& p_var) {

	int len;
	Error err = encode_variant(p_var,NULL,len);
	ERR_FAIL_COND_V( err != OK, "" );

	DVector<uint8_t> buff;
	buff.resize(len);
	DVector<uint8_t>::Write w = buff.write();

	err = encode_variant(p_var,&w[0],len);
	ERR_FAIL_COND_V( err != OK, "" );

	int b64len = len / 3 * 4 + 4 + 1;
	DVector<uint8_t> b64buff;
	b64buff.resize(b64len);
	DVector<uint8_t>::Write w64 = b64buff.write();

	int strlen = base64_encode((char*)(&w64[0]), (char*)(&w[0]), len);
	//OS::get_singleton()->print("len is %i, vector size is %i\n", b64len, strlen);
	w64[strlen] = 0;
	String ret = (char*)&w64[0];

	return ret;
};

Variant _Marshalls::base64_to_variant(const String& p_str) {

	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	DVector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1);
	DVector<uint8_t>::Write w = buf.write();

	int len = base64_decode((char*)(&w[0]), (char*)cstr.get_data(), strlen);

	Variant v;
	Error err = decode_variant(v, &w[0], len);
	ERR_FAIL_COND_V( err!=OK, Variant() );

	return v;
};

String _Marshalls::raw_to_base64(const DVector<uint8_t> &p_arr) {

	int len = p_arr.size();
	DVector<uint8_t>::Read r = p_arr.read();

	int b64len = len / 3 * 4 + 4 + 1;
	DVector<uint8_t> b64buff;
	b64buff.resize(b64len);
	DVector<uint8_t>::Write w64 = b64buff.write();

	int strlen = base64_encode((char*)(&w64[0]), (char*)(&r[0]), len);
	w64[strlen] = 0;
	String ret = (char*)&w64[0];

	return ret;
};

DVector<uint8_t> _Marshalls::base64_to_raw(const String &p_str) {

	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	int arr_len;
	DVector<uint8_t> buf;
	{
		buf.resize(strlen / 4 * 3 + 1);
		DVector<uint8_t>::Write w = buf.write();

		arr_len = base64_decode((char*)(&w[0]), (char*)cstr.get_data(), strlen);
	};
	buf.resize(arr_len);

	// conversion from DVector<uint8_t> to raw array?
	return buf;
};

String _Marshalls::utf8_to_base64(const String& p_str) {

	CharString cstr = p_str.utf8();
	int len = cstr.length();

	int b64len = len / 3 * 4 + 4 + 1;
	DVector<uint8_t> b64buff;
	b64buff.resize(b64len);
	DVector<uint8_t>::Write w64 = b64buff.write();

	int strlen = base64_encode((char*)(&w64[0]), (char*)cstr.get_data(), len);

	w64[strlen] = 0;
	String ret = (char*)&w64[0];

	return ret;
};

String _Marshalls::base64_to_utf8(const String& p_str) {

	int strlen = p_str.length();
	CharString cstr = p_str.ascii();

	DVector<uint8_t> buf;
	buf.resize(strlen / 4 * 3 + 1 + 1);
	DVector<uint8_t>::Write w = buf.write();

	int len = base64_decode((char*)(&w[0]), (char*)cstr.get_data(), strlen);

	w[len] = 0;
	String ret = String::utf8((char*)&w[0]);

	return ret;
};


void _Marshalls::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("variant_to_base64:String","variant"),&_Marshalls::variant_to_base64);
	ObjectTypeDB::bind_method(_MD("base64_to_variant:Variant","base64_str"),&_Marshalls::base64_to_variant);

	ObjectTypeDB::bind_method(_MD("raw_to_base64:String","array"),&_Marshalls::raw_to_base64);
	ObjectTypeDB::bind_method(_MD("base64_to_raw:RawArray","base64_str"),&_Marshalls::base64_to_raw);

	ObjectTypeDB::bind_method(_MD("utf8_to_base64:String","utf8_str"),&_Marshalls::utf8_to_base64);
	ObjectTypeDB::bind_method(_MD("base64_to_utf8:String","base64_str"),&_Marshalls::base64_to_utf8);

};



////////////////




Error _Semaphore::wait() {

	return semaphore->wait();
}

Error _Semaphore::post() {

	return semaphore->post();
}


void _Semaphore::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("wait:Error"),&_Semaphore::wait);
	ObjectTypeDB::bind_method(_MD("post:Error"),&_Semaphore::post);

}


_Semaphore::_Semaphore() {

	semaphore =Semaphore::create();
}

_Semaphore::~_Semaphore(){

	memdelete(semaphore);
}


///////////////


void _Mutex::lock() {

	mutex->lock();
}

Error _Mutex::try_lock(){

	return mutex->try_lock();
}

void _Mutex::unlock(){

	mutex->unlock();
}

void _Mutex::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("lock"),&_Mutex::lock);
	ObjectTypeDB::bind_method(_MD("try_lock:Error"),&_Mutex::try_lock);
	ObjectTypeDB::bind_method(_MD("unlock"),&_Mutex::unlock);

}


_Mutex::_Mutex() {

	mutex =Mutex::create();
}

_Mutex::~_Mutex(){

	memdelete(mutex);
}


///////////////



void _Thread::_start_func(void *ud) {

	Ref<_Thread>* tud=(Ref<_Thread>*)ud;
	Ref<_Thread> t=*tud;
	memdelete(tud);
	Variant::CallError ce;
	const Variant* arg[1]={&t->userdata};

	Thread::set_name(t->target_method);

	t->ret=t->target_instance->call(t->target_method,arg,1,ce);
	if (ce.error!=Variant::CallError::CALL_OK) {

		String reason;
		switch(ce.error) {
			case Variant::CallError::CALL_ERROR_INVALID_ARGUMENT: {

				reason="Invalid Argument #"+itos(ce.argument);
			} break;
			case Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {

				reason="Too Many Arguments";
			} break;
			case Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS: {

				reason="Too Many Arguments";
			} break;
			case Variant::CallError::CALL_ERROR_INVALID_METHOD: {

				reason="Method Not Found";
			} break;
			default: {}
		}


		ERR_EXPLAIN("Could not call function '"+t->target_method.operator String()+"'' starting thread ID: "+t->get_id()+" Reason: "+reason);
		ERR_FAIL();
	}

}

Error _Thread::start(Object *p_instance,const StringName& p_method,const Variant& p_userdata,int p_priority) {

	ERR_FAIL_COND_V(active,ERR_ALREADY_IN_USE);
	ERR_FAIL_COND_V(!p_instance,ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_method==StringName(),ERR_INVALID_PARAMETER);
	ERR_FAIL_INDEX_V(p_priority,3,ERR_INVALID_PARAMETER);


	ret=Variant();
	target_method=p_method;
	target_instance=p_instance;
	userdata=p_userdata;
	active=true;

	Ref<_Thread> *ud = memnew( Ref<_Thread>(this) );

	Thread::Settings s;
	s.priority=(Thread::Priority)p_priority;
	thread = Thread::create(_start_func,ud,s);
	if (!thread) {
		active=false;
		target_method=StringName();
		target_instance=NULL;
		userdata=Variant();
		return ERR_CANT_CREATE;
	}

	return OK;
}

String _Thread::get_id() const {

	if (!thread)
		return String();

	return itos(thread->get_ID());
}

bool _Thread::is_active() const {

	return active;
}
Variant _Thread::wait_to_finish() {

	ERR_FAIL_COND_V(!thread,Variant());
	ERR_FAIL_COND_V(!active,Variant());
	Thread::wait_to_finish(thread);
	Variant r = ret;
	active=false;
	target_method=StringName();
	target_instance=NULL;
	userdata=Variant();
	thread=NULL;

	return r;
}

void _Thread::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("start:Error","instance","method","userdata","priority"),&_Thread::start,DEFVAL(Variant()),DEFVAL(PRIORITY_NORMAL));
	ObjectTypeDB::bind_method(_MD("get_id"),&_Thread::get_id);
	ObjectTypeDB::bind_method(_MD("is_active"),&_Thread::is_active);
	ObjectTypeDB::bind_method(_MD("wait_to_finish:Variant"),&_Thread::wait_to_finish);

	BIND_CONSTANT( PRIORITY_LOW );
	BIND_CONSTANT( PRIORITY_NORMAL );
	BIND_CONSTANT( PRIORITY_HIGH );

}
_Thread::_Thread() {

	active=false;
	thread=NULL;
	target_instance=NULL;
}

_Thread::~_Thread() {

	if (active) {
		ERR_EXPLAIN("Reference to a Thread object object was lost while the thread is still running..");
	}
	ERR_FAIL_COND(active==true);
}
