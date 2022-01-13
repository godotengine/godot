/*************************************************************************/
/*  time.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "time.h"

#include "core/os/os.h"

#define UNIX_EPOCH_YEAR_AD 1970 // 1970
#define SECONDS_PER_DAY (24 * 60 * 60) // 86400
#define IS_LEAP_YEAR(year) (!((year) % 4) && (((year) % 100) || !((year) % 400)))
#define YEAR_SIZE(year) (IS_LEAP_YEAR(year) ? 366 : 365)

#define YEAR_KEY "year"
#define MONTH_KEY "month"
#define DAY_KEY "day"
#define WEEKDAY_KEY "weekday"
#define HOUR_KEY "hour"
#define MINUTE_KEY "minute"
#define SECOND_KEY "second"
#define DST_KEY "dst"

// Table of number of days in each month (for regular year and leap year).
static const uint8_t MONTH_DAYS_TABLE[2][12] = {
	{ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 },
	{ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 }
};

VARIANT_ENUM_CAST(Time::Month);
VARIANT_ENUM_CAST(Time::Weekday);

#define UNIX_TIME_TO_HMS                                                     \
	uint8_t hour, minute, second;                                            \
	{                                                                        \
		/* The time of the day (in seconds since start of day). */           \
		uint32_t day_clock = Math::posmod(p_unix_time_val, SECONDS_PER_DAY); \
		/* On x86 these 4 lines can be optimized to only 2 divisions. */     \
		second = day_clock % 60;                                             \
		day_clock /= 60;                                                     \
		minute = day_clock % 60;                                             \
		hour = day_clock / 60;                                               \
	}

#define UNIX_TIME_TO_YMD                                                                    \
	int64_t year;                                                                           \
	Month month;                                                                            \
	uint8_t day;                                                                            \
	/* The day number since Unix epoch (0-index). Days before 1970 are negative. */         \
	int64_t day_number = Math::floor(p_unix_time_val / (double)SECONDS_PER_DAY);            \
	{                                                                                       \
		int64_t day_number_copy = day_number;                                               \
		year = UNIX_EPOCH_YEAR_AD;                                                          \
		uint8_t month_zero_index = 0;                                                       \
		while (day_number_copy >= YEAR_SIZE(year)) {                                        \
			day_number_copy -= YEAR_SIZE(year);                                             \
			year++;                                                                         \
		}                                                                                   \
		while (day_number_copy < 0) {                                                       \
			year--;                                                                         \
			day_number_copy += YEAR_SIZE(year);                                             \
		}                                                                                   \
		/* After the above, day_number now represents the day of the year (0-index). */     \
		while (day_number_copy >= MONTH_DAYS_TABLE[IS_LEAP_YEAR(year)][month_zero_index]) { \
			day_number_copy -= MONTH_DAYS_TABLE[IS_LEAP_YEAR(year)][month_zero_index];      \
			month_zero_index++;                                                             \
		}                                                                                   \
		/* After the above, day_number now represents the day of the month (0-index). */    \
		month = (Month)(month_zero_index + 1);                                              \
		day = day_number_copy + 1;                                                          \
	}

#define VALIDATE_YMDHMS                                                                                                                                                 \
	ERR_FAIL_COND_V_MSG(month == 0, 0, "Invalid month value of: " + itos(month) + ", months are 1-indexed and cannot be 0. See the Time.Month enum for valid values."); \
	ERR_FAIL_COND_V_MSG(month > 12, 0, "Invalid month value of: " + itos(month) + ". See the Time.Month enum for valid values.");                                       \
	ERR_FAIL_COND_V_MSG(hour > 23, 0, "Invalid hour value of: " + itos(hour) + ".");                                                                                    \
	ERR_FAIL_COND_V_MSG(minute > 59, 0, "Invalid minute value of: " + itos(minute) + ".");                                                                              \
	ERR_FAIL_COND_V_MSG(second > 59, 0, "Invalid second value of: " + itos(second) + " (leap seconds are not supported).");                                             \
	/* Do this check after month is tested as valid. */                                                                                                                 \
	ERR_FAIL_COND_V_MSG(day == 0, 0, "Invalid day value of: " + itos(month) + ", days are 1-indexed and cannot be 0.");                                                 \
	uint8_t days_in_this_month = MONTH_DAYS_TABLE[IS_LEAP_YEAR(year)][month - 1];                                                                                       \
	ERR_FAIL_COND_V_MSG(day > days_in_this_month, 0, "Invalid day value of: " + itos(day) + " which is larger than the maximum for this month, " + itos(days_in_this_month) + ".");

#define YMD_TO_DAY_NUMBER                                                           \
	/* The day number since Unix epoch (0-index). Days before 1970 are negative. */ \
	int64_t day_number = day - 1;                                                   \
	/* Add the days in the months to day_number. */                                 \
	for (int i = 0; i < month - 1; i++) {                                           \
		day_number += MONTH_DAYS_TABLE[IS_LEAP_YEAR(year)][i];                      \
	}                                                                               \
	/* Add the days in the years to day_number. */                                  \
	if (year >= UNIX_EPOCH_YEAR_AD) {                                               \
		for (int64_t iyear = UNIX_EPOCH_YEAR_AD; iyear < year; iyear++) {           \
			day_number += YEAR_SIZE(iyear);                                         \
		}                                                                           \
	} else {                                                                        \
		for (int64_t iyear = UNIX_EPOCH_YEAR_AD - 1; iyear >= year; iyear--) {      \
			day_number -= YEAR_SIZE(iyear);                                         \
		}                                                                           \
	}

#define PARSE_ISO8601_STRING                                     \
	int64_t year = UNIX_EPOCH_YEAR_AD;                           \
	Month month = MONTH_JANUARY;                                 \
	uint8_t day = 1;                                             \
	uint8_t hour = 0;                                            \
	uint8_t minute = 0;                                          \
	uint8_t second = 0;                                          \
	{                                                            \
		bool has_date = false, has_time = false;                 \
		String date, time;                                       \
		if (p_datetime.find_char('T') > 0) {                     \
			has_date = has_time = true;                          \
			Vector<String> array = p_datetime.split("T");        \
			date = array[0];                                     \
			time = array[1];                                     \
		} else if (p_datetime.find_char(' ') > 0) {              \
			has_date = has_time = true;                          \
			Vector<String> array = p_datetime.split(" ");        \
			date = array[0];                                     \
			time = array[1];                                     \
		} else if (p_datetime.find_char('-', 1) > 0) {           \
			has_date = true;                                     \
			date = p_datetime;                                   \
		} else if (p_datetime.find_char(':') > 0) {              \
			has_time = true;                                     \
			time = p_datetime;                                   \
		}                                                        \
		/* Set the variables from the contents of the string. */ \
		if (has_date) {                                          \
			Vector<int> array = date.split_ints("-", false);     \
			year = array[0];                                     \
			month = (Month)array[1];                             \
			day = array[2];                                      \
			/* Handle negative years. */                         \
			if (p_datetime.find_char('-') == 0) {                \
				year *= -1;                                      \
			}                                                    \
		}                                                        \
		if (has_time) {                                          \
			Vector<int> array = time.split_ints(":", false);     \
			hour = array[0];                                     \
			minute = array[1];                                   \
			second = array[2];                                   \
		}                                                        \
	}

#define EXTRACT_FROM_DICTIONARY                                                                   \
	/* Get all time values from the dictionary. If it doesn't exist, set the */                   \
	/* values to the default values for Unix epoch (1970-01-01 00:00:00). */                      \
	int64_t year = p_datetime.has(YEAR_KEY) ? int64_t(p_datetime[YEAR_KEY]) : UNIX_EPOCH_YEAR_AD; \
	Month month = Month((p_datetime.has(MONTH_KEY)) ? uint8_t(p_datetime[MONTH_KEY]) : 1);        \
	uint8_t day = p_datetime.has(DAY_KEY) ? uint8_t(p_datetime[DAY_KEY]) : 1;                     \
	uint8_t hour = p_datetime.has(HOUR_KEY) ? uint8_t(p_datetime[HOUR_KEY]) : 0;                  \
	uint8_t minute = p_datetime.has(MINUTE_KEY) ? uint8_t(p_datetime[MINUTE_KEY]) : 0;            \
	uint8_t second = p_datetime.has(SECOND_KEY) ? uint8_t(p_datetime[SECOND_KEY]) : 0;

Time *Time::singleton = nullptr;

Time *Time::get_singleton() {
	if (!singleton) {
		memnew(Time);
	}
	return singleton;
}

Dictionary Time::get_datetime_dict_from_unix_time(int64_t p_unix_time_val) const {
	UNIX_TIME_TO_HMS
	UNIX_TIME_TO_YMD
	Dictionary datetime;
	datetime[YEAR_KEY] = year;
	datetime[MONTH_KEY] = (uint8_t)month;
	datetime[DAY_KEY] = day;
	// Unix epoch was a Thursday (day 0 aka 1970-01-01).
	datetime[WEEKDAY_KEY] = Math::posmod(day_number + WEEKDAY_THURSDAY, 7);
	datetime[HOUR_KEY] = hour;
	datetime[MINUTE_KEY] = minute;
	datetime[SECOND_KEY] = second;

	return datetime;
}

Dictionary Time::get_date_dict_from_unix_time(int64_t p_unix_time_val) const {
	UNIX_TIME_TO_YMD
	Dictionary datetime;
	datetime[YEAR_KEY] = year;
	datetime[MONTH_KEY] = (uint8_t)month;
	datetime[DAY_KEY] = day;
	// Unix epoch was a Thursday (day 0 aka 1970-01-01).
	datetime[WEEKDAY_KEY] = Math::posmod(day_number + WEEKDAY_THURSDAY, 7);

	return datetime;
}

Dictionary Time::get_time_dict_from_unix_time(int64_t p_unix_time_val) const {
	UNIX_TIME_TO_HMS
	Dictionary datetime;
	datetime[HOUR_KEY] = hour;
	datetime[MINUTE_KEY] = minute;
	datetime[SECOND_KEY] = second;

	return datetime;
}

String Time::get_datetime_string_from_unix_time(int64_t p_unix_time_val, bool p_use_space) const {
	UNIX_TIME_TO_HMS
	UNIX_TIME_TO_YMD
	// vformat only supports up to 6 arguments, so we need to split this up into 2 parts.
	String timestamp = vformat("%04d-%02d-%02d", year, (uint8_t)month, day);
	if (p_use_space) {
		timestamp = vformat("%s %02d:%02d:%02d", timestamp, hour, minute, second);
	} else {
		timestamp = vformat("%sT%02d:%02d:%02d", timestamp, hour, minute, second);
	}

	return timestamp;
}

String Time::get_date_string_from_unix_time(int64_t p_unix_time_val) const {
	UNIX_TIME_TO_YMD
	// Android is picky about the types passed to make Variant, so we need a cast.
	return vformat("%04d-%02d-%02d", year, (uint8_t)month, day);
}

String Time::get_time_string_from_unix_time(int64_t p_unix_time_val) const {
	UNIX_TIME_TO_HMS
	return vformat("%02d:%02d:%02d", hour, minute, second);
}

Dictionary Time::get_datetime_dict_from_string(String p_datetime, bool p_weekday) const {
	PARSE_ISO8601_STRING
	Dictionary dict;
	dict[YEAR_KEY] = year;
	dict[MONTH_KEY] = (uint8_t)month;
	dict[DAY_KEY] = day;
	if (p_weekday) {
		YMD_TO_DAY_NUMBER
		// Unix epoch was a Thursday (day 0 aka 1970-01-01).
		dict[WEEKDAY_KEY] = Math::posmod(day_number + WEEKDAY_THURSDAY, 7);
	}
	dict[HOUR_KEY] = hour;
	dict[MINUTE_KEY] = minute;
	dict[SECOND_KEY] = second;

	return dict;
}

String Time::get_datetime_string_from_dict(Dictionary p_datetime, bool p_use_space) const {
	ERR_FAIL_COND_V_MSG(p_datetime.empty(), "", "Invalid datetime Dictionary: Dictionary is empty.");
	EXTRACT_FROM_DICTIONARY
	// vformat only supports up to 6 arguments, so we need to split this up into 2 parts.
	String timestamp = vformat("%04d-%02d-%02d", year, (uint8_t)month, day);
	if (p_use_space) {
		timestamp = vformat("%s %02d:%02d:%02d", timestamp, hour, minute, second);
	} else {
		timestamp = vformat("%sT%02d:%02d:%02d", timestamp, hour, minute, second);
	}
	return timestamp;
}

int64_t Time::get_unix_time_from_datetime_dict(Dictionary p_datetime) const {
	ERR_FAIL_COND_V_MSG(p_datetime.empty(), 0, "Invalid datetime Dictionary: Dictionary is empty");
	EXTRACT_FROM_DICTIONARY
	VALIDATE_YMDHMS
	YMD_TO_DAY_NUMBER
	return day_number * SECONDS_PER_DAY + hour * 3600 + minute * 60 + second;
}

int64_t Time::get_unix_time_from_datetime_string(String p_datetime) const {
	PARSE_ISO8601_STRING
	VALIDATE_YMDHMS
	YMD_TO_DAY_NUMBER
	return day_number * SECONDS_PER_DAY + hour * 3600 + minute * 60 + second;
}

Dictionary Time::get_datetime_dict_from_system(bool p_utc) const {
	OS::Date date = OS::get_singleton()->get_date(p_utc);
	OS::Time time = OS::get_singleton()->get_time(p_utc);
	Dictionary datetime;
	datetime[YEAR_KEY] = date.year;
	datetime[MONTH_KEY] = (uint8_t)date.month;
	datetime[DAY_KEY] = date.day;
	datetime[WEEKDAY_KEY] = (uint8_t)date.weekday;
	datetime[DST_KEY] = date.dst;
	datetime[HOUR_KEY] = time.hour;
	datetime[MINUTE_KEY] = time.min;
	datetime[SECOND_KEY] = time.sec;
	return datetime;
}

Dictionary Time::get_date_dict_from_system(bool p_utc) const {
	OS::Date date = OS::get_singleton()->get_date(p_utc);
	Dictionary date_dictionary;
	date_dictionary[YEAR_KEY] = date.year;
	date_dictionary[MONTH_KEY] = (uint8_t)date.month;
	date_dictionary[DAY_KEY] = date.day;
	date_dictionary[WEEKDAY_KEY] = (uint8_t)date.weekday;
	date_dictionary[DST_KEY] = date.dst;
	return date_dictionary;
}

Dictionary Time::get_time_dict_from_system(bool p_utc) const {
	OS::Time time = OS::get_singleton()->get_time(p_utc);
	Dictionary time_dictionary;
	time_dictionary[HOUR_KEY] = time.hour;
	time_dictionary[MINUTE_KEY] = time.min;
	time_dictionary[SECOND_KEY] = time.sec;
	return time_dictionary;
}

String Time::get_datetime_string_from_system(bool p_utc, bool p_use_space) const {
	OS::Date date = OS::get_singleton()->get_date(p_utc);
	OS::Time time = OS::get_singleton()->get_time(p_utc);
	// vformat only supports up to 6 arguments, so we need to split this up into 2 parts.
	String timestamp = vformat("%04d-%02d-%02d", date.year, (uint8_t)date.month, date.day);
	if (p_use_space) {
		timestamp = vformat("%s %02d:%02d:%02d", timestamp, time.hour, time.min, time.sec);
	} else {
		timestamp = vformat("%sT%02d:%02d:%02d", timestamp, time.hour, time.min, time.sec);
	}

	return timestamp;
}

String Time::get_date_string_from_system(bool p_utc) const {
	OS::Date date = OS::get_singleton()->get_date(p_utc);
	// Android is picky about the types passed to make Variant, so we need a cast.
	return vformat("%04d-%02d-%02d", date.year, (uint8_t)date.month, date.day);
}

String Time::get_time_string_from_system(bool p_utc) const {
	OS::Time time = OS::get_singleton()->get_time(p_utc);
	return vformat("%02d:%02d:%02d", time.hour, time.min, time.sec);
}

Dictionary Time::get_time_zone_from_system() const {
	OS::TimeZoneInfo info = OS::get_singleton()->get_time_zone_info();
	Dictionary timezone;
	timezone["bias"] = info.bias;
	timezone["name"] = info.name;
	return timezone;
}

double Time::get_unix_time_from_system() const {
	return OS::get_singleton()->get_unix_time();
}

uint64_t Time::get_ticks_msec() const {
	return OS::get_singleton()->get_ticks_msec();
}

uint64_t Time::get_ticks_usec() const {
	return OS::get_singleton()->get_ticks_usec();
}

void Time::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_datetime_dict_from_unix_time", "unix_time_val"), &Time::get_datetime_dict_from_unix_time);
	ClassDB::bind_method(D_METHOD("get_date_dict_from_unix_time", "unix_time_val"), &Time::get_date_dict_from_unix_time);
	ClassDB::bind_method(D_METHOD("get_time_dict_from_unix_time", "unix_time_val"), &Time::get_time_dict_from_unix_time);
	ClassDB::bind_method(D_METHOD("get_datetime_string_from_unix_time", "unix_time_val", "use_space"), &Time::get_datetime_string_from_unix_time, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_date_string_from_unix_time", "unix_time_val"), &Time::get_date_string_from_unix_time);
	ClassDB::bind_method(D_METHOD("get_time_string_from_unix_time", "unix_time_val"), &Time::get_time_string_from_unix_time);
	ClassDB::bind_method(D_METHOD("get_datetime_dict_from_string", "datetime", "weekday"), &Time::get_datetime_dict_from_string);
	ClassDB::bind_method(D_METHOD("get_datetime_string_from_dict", "datetime", "use_space"), &Time::get_datetime_string_from_dict);
	ClassDB::bind_method(D_METHOD("get_unix_time_from_datetime_dict", "datetime"), &Time::get_unix_time_from_datetime_dict);
	ClassDB::bind_method(D_METHOD("get_unix_time_from_datetime_string", "datetime"), &Time::get_unix_time_from_datetime_string);

	ClassDB::bind_method(D_METHOD("get_datetime_dict_from_system", "utc"), &Time::get_datetime_dict_from_system, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_date_dict_from_system", "utc"), &Time::get_date_dict_from_system, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_time_dict_from_system", "utc"), &Time::get_time_dict_from_system, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_datetime_string_from_system", "utc", "use_space"), &Time::get_datetime_string_from_system, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_date_string_from_system", "utc"), &Time::get_date_string_from_system, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_time_string_from_system", "utc"), &Time::get_time_string_from_system, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_time_zone_from_system"), &Time::get_time_zone_from_system);
	ClassDB::bind_method(D_METHOD("get_unix_time_from_system"), &Time::get_unix_time_from_system);
	ClassDB::bind_method(D_METHOD("get_ticks_msec"), &Time::get_ticks_msec);
	ClassDB::bind_method(D_METHOD("get_ticks_usec"), &Time::get_ticks_usec);

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

	BIND_ENUM_CONSTANT(WEEKDAY_SUNDAY);
	BIND_ENUM_CONSTANT(WEEKDAY_MONDAY);
	BIND_ENUM_CONSTANT(WEEKDAY_TUESDAY);
	BIND_ENUM_CONSTANT(WEEKDAY_WEDNESDAY);
	BIND_ENUM_CONSTANT(WEEKDAY_THURSDAY);
	BIND_ENUM_CONSTANT(WEEKDAY_FRIDAY);
	BIND_ENUM_CONSTANT(WEEKDAY_SATURDAY);
}

Time::Time() {
	ERR_FAIL_COND_MSG(singleton, "Singleton for Time already exists.");
	singleton = this;
}

Time::~Time() {
	singleton = nullptr;
}
