/**************************************************************************/
/*  test_time.h                                                           */
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

#ifndef TEST_TIME_H
#define TEST_TIME_H

#include "core/os/time.h"

#include "thirdparty/doctest/doctest.h"

#define YEAR_KEY "year"
#define MONTH_KEY "month"
#define DAY_KEY "day"
#define WEEKDAY_KEY "weekday"
#define HOUR_KEY "hour"
#define MINUTE_KEY "minute"
#define SECOND_KEY "second"
#define DST_KEY "dst"

namespace TestTime {

TEST_CASE("[Time] Datetime dictionary conversion to datetime string") {
	const Time *time = Time::get_singleton();

	Dictionary datetime;
	datetime[YEAR_KEY] = 1234;
	datetime[MONTH_KEY] = 5;
	datetime[DAY_KEY] = 6;
	datetime[HOUR_KEY] = 7;
	datetime[MINUTE_KEY] = 8;
	datetime[SECOND_KEY] = 9;

	Dictionary unix_epoch;
	unix_epoch[YEAR_KEY] = 1970;
	unix_epoch[MONTH_KEY] = 1;
	unix_epoch[DAY_KEY] = 1;
	unix_epoch[HOUR_KEY] = 0;
	unix_epoch[MINUTE_KEY] = 0;
	unix_epoch[SECOND_KEY] = 0;

	Dictionary dict;
	Dictionary dict_empty = Dictionary();

	dict = datetime.duplicate();
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict) == "1234-05-06T07:08:09", "Time get_datetime_string_from_datetime_dict: full dictionary.");
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict, true) == "1234-05-06 07:08:09", "Time get_datetime_string_from_datetime_dict: full dictionary, space delimiter requested.");
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict_empty) == "", "Time get_datetime_string_from_datetime_dict: empty dictionary.");

	dict.erase(YEAR_KEY);
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict) == vformat("%04d-05-06T07:08:09", unix_epoch[YEAR_KEY]), "Time get_datetime_string_from_datetime_dict: default YEAR_KEY.");
	dict[YEAR_KEY] = datetime[YEAR_KEY];

	dict.erase(MONTH_KEY);
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict) == vformat("1234-%02d-06T07:08:09", unix_epoch[MONTH_KEY]), "Time get_datetime_string_from_datetime_dict: default MONTH_KEY.");
	dict[MONTH_KEY] = datetime[MONTH_KEY];

	dict.erase(DAY_KEY);
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict) == vformat("1234-05-%02dT07:08:09", unix_epoch[DAY_KEY]), "Time get_datetime_string_from_datetime_dict: default DAY_KEY.");
	dict[DAY_KEY] = datetime[DAY_KEY];

	dict.erase(HOUR_KEY);
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict) == vformat("1234-05-06T%02d:08:09", unix_epoch[HOUR_KEY]), "Time get_datetime_string_from_datetime_dict: default HOUR_KEY.");
	dict[HOUR_KEY] = datetime[HOUR_KEY];

	dict.erase(MINUTE_KEY);
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict) == vformat("1234-05-06T07:%02d:09", unix_epoch[MINUTE_KEY]), "Time get_datetime_string_from_datetime_dict: default MINUTE_KEY.");
	dict[MINUTE_KEY] = datetime[MINUTE_KEY];

	dict.erase(SECOND_KEY);
	CHECK_MESSAGE(time->get_datetime_string_from_datetime_dict(dict) == vformat("1234-05-06T07:08:%02d", unix_epoch[SECOND_KEY]), "Time get_datetime_string_from_datetime_dict: default SECOND_KEY.");
	dict[SECOND_KEY] = datetime[SECOND_KEY];
}

TEST_CASE("[Time] Datetime string conversion to datetime dictionary") {
	const Time *time = Time::get_singleton();

	Dictionary datetime;
	datetime[YEAR_KEY] = 1234;
	datetime[MONTH_KEY] = 5;
	datetime[DAY_KEY] = 6;
	datetime[HOUR_KEY] = 7;
	datetime[MINUTE_KEY] = 8;
	datetime[SECOND_KEY] = 9;

	Dictionary unix_epoch;
	unix_epoch[YEAR_KEY] = 1970;
	unix_epoch[MONTH_KEY] = 1;
	unix_epoch[DAY_KEY] = 1;
	unix_epoch[HOUR_KEY] = 0;
	unix_epoch[MINUTE_KEY] = 0;
	unix_epoch[SECOND_KEY] = 0;

	Dictionary dict;
	Dictionary dict_empty = Dictionary();

	// date
	dict = unix_epoch.duplicate();
	dict[YEAR_KEY] = datetime[YEAR_KEY];
	dict[MONTH_KEY] = datetime[MONTH_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM.");

	dict[DAY_KEY] = datetime[DAY_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DD.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("12340506", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYYMMDD.");

	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("12-3405-06", false) == dict_empty, "Time get_datetime_dict_from_datetime_string: XX-XXXX-XX.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-05-06", false) == dict_empty, "Time get_datetime_dict_from_datetime_string: XXXX-XX-XX-XX.");

	// time
	dict = unix_epoch.duplicate();
	dict[HOUR_KEY] = datetime[HOUR_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T07", false) == dict, "Time get_datetime_dict_from_datetime_string: Thh.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("07", false) == dict, "Time get_datetime_dict_from_datetime_string: hh.");

	dict[MINUTE_KEY] = datetime[MINUTE_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T07:08", false) == dict, "Time get_datetime_dict_from_datetime_string: Thh:mm.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("07:08", false) == dict, "Time get_datetime_dict_from_datetime_string: hh:mm.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T0708", false) == dict, "Time get_datetime_dict_from_datetime_string: Thhmm.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("0708", false) == dict, "Time get_datetime_dict_from_datetime_string: hhmm.");

	dict[SECOND_KEY] = datetime[SECOND_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T07:08:09", false) == dict, "Time get_datetime_dict_from_datetime_string: Thh:mm:ss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("07:08:09", false) == dict, "Time get_datetime_dict_from_datetime_string: hh:mm:ss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T070809", false) == dict, "Time get_datetime_dict_from_datetime_string: Thhmmss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("070809", false) == dict, "Time get_datetime_dict_from_datetime_string: hhmmss.");

	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T07:0808:09", false) == dict_empty, "Time get_datetime_dict_from_datetime_string: Txx:xxxx:xx.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T07:08:08:09", false) == dict_empty, "Time get_datetime_dict_from_datetime_string: Txx:xx:xx:xx.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("T789", false) == dict_empty, "Time get_datetime_dict_from_datetime_string: Txxx.");

	// date & time
	dict = unix_epoch.duplicate();
	dict[YEAR_KEY] = datetime[YEAR_KEY];
	dict[MONTH_KEY] = datetime[MONTH_KEY];
	dict[HOUR_KEY] = datetime[HOUR_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05T07", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MMThh.");

	dict[MINUTE_KEY] = datetime[MINUTE_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05T07:08", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MMThh:mm.");

	dict[DAY_KEY] = datetime[DAY_KEY];
	dict[MINUTE_KEY] = unix_epoch[MINUTE_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06T07", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DDThh.");

	dict[MINUTE_KEY] = datetime[MINUTE_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06T07:08", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DDThh:mm.");

	dict[SECOND_KEY] = datetime[SECOND_KEY];
	dict[DAY_KEY] = unix_epoch[DAY_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05T07:08:09", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MMThh:mm:ss.");

	dict[DAY_KEY] = datetime[DAY_KEY];
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06T07:08:09", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DDThh:mm:ss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06 07:08:09", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DD hh:mm:ss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06T070809", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DDThhmmss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06 070809", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DD hhmmss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("12340506T07:08:09", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYYMMDDThh:mm:ss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("12340506 07:08:09", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYYMMDD hh:mm:ss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("12340506T070809", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYYMMDDThhmmss.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("12340506 070809", false) == dict, "Time get_datetime_dict_from_datetime_string: YYYYMMDD hhmmss.");

	CHECK_MESSAGE(!time->get_datetime_dict_from_datetime_string("1234-05-06T07:08:09", false).has(WEEKDAY_KEY), "Time get_datetime_dict_from_datetime_string: YYYY-MM-DDThh:mm:ss, WEEKDAY_KEY is not requested.");
	dict[WEEKDAY_KEY] = WEEKDAY_SATURDAY;
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05-06T07:08:09") == dict, "Time get_datetime_dict_from_datetime_string: YYYY-MM-DDThh:mm:ss, WEEKDAY_KEY is requested.");

	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05 1234-05-06 07:08:09", false) == dict_empty, "Time get_datetime_dict_from_datetime_string: <?> <?> <?>.");
	CHECK_MESSAGE(time->get_datetime_dict_from_datetime_string("1234-05T1234-05-06T07:08:09", false) == dict_empty, "Time get_datetime_dict_from_datetime_string: <?>T<?>T<?>.");
}

TEST_CASE("[Time] Unix time conversion to/from datetime string") {
	const Time *time = Time::get_singleton();

	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1970-01-01T00:00:00") == 0, "Time get_unix_time_from_datetime_string: The timestamp for Unix epoch is zero.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1970-01-01 00:00:00") == 0, "Time get_unix_time_from_datetime_string: The timestamp for Unix epoch with space is zero.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1970-01-01") == 0, "Time get_unix_time_from_datetime_string: The timestamp for Unix epoch without time is zero.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("00:00:00") == 0, "Time get_unix_time_from_datetime_string: The timestamp for zero time without date is zero.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1969-12-31T23:59:59") == -1, "Time get_unix_time_from_datetime_string: The timestamp for just before Unix epoch is negative one.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1234-05-06T07:08:09") == -23215049511, "Time get_unix_time_from_datetime_string: The timestamp for an arbitrary datetime is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1234-05-06 07:08:09") == -23215049511, "Time get_unix_time_from_datetime_string: The timestamp for an arbitrary datetime with space is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1234-05-06") == -23215075200, "Time get_unix_time_from_datetime_string: The timestamp for an arbitrary date without time is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("07:08:09") == 25689, "Time get_unix_time_from_datetime_string: The timestamp for an arbitrary time without date is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("2014-02-09T22:10:30") == 1391983830, "Time get_unix_time_from_datetime_string: The timestamp for GODOT IS OPEN SOURCE is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("2014-02-09 22:10:30") == 1391983830, "Time get_unix_time_from_datetime_string: The timestamp for GODOT IS OPEN SOURCE with space is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("2014-02-09") == 1391904000, "Time get_unix_time_from_datetime_string: The date for GODOT IS OPEN SOURCE without time is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("22:10:30") == 79830, "Time get_unix_time_from_datetime_string: The time for GODOT IS OPEN SOURCE without date is as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("-1000000000-01-01T00:00:00") == -31557014167219200, "Time get_unix_time_from_datetime_string: In the year negative a billion, Japan might not have been here.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string("1000000-01-01T00:00:00") == 31494784780800, "Time get_unix_time_from_datetime_string: The timestamp for the year a million is as expected.");

	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(0) == "1970-01-01T00:00:00", "Time get_datetime_string_from_unix_time: The timestamp string for Unix epoch is zero.");
	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(0, true) == "1970-01-01 00:00:00", "Time get_datetime_string_from_unix_time: The timestamp string for Unix epoch with space is zero.");
	CHECK_MESSAGE(time->get_date_string_from_unix_time(0) == "1970-01-01", "Time get_date_string_from_unix_time: The date string for zero is Unix epoch date.");
	CHECK_MESSAGE(time->get_time_string_from_unix_time(0) == "00:00:00", "Time get_time_string_from_unix_time: The date for zero zero is Unix epoch date.");
	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(-1) == "1969-12-31T23:59:59", "Time get_time_string_from_unix_time: The timestamp string for just before Unix epoch is as expected.");
	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(-23215049511) == "1234-05-06T07:08:09", "Time get_datetime_string_from_unix_time: The timestamp for an arbitrary datetime is as expected.");
	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(-23215049511, true) == "1234-05-06 07:08:09", "Time get_datetime_string_from_unix_time: The timestamp for an arbitrary datetime with space is as expected.");
	CHECK_MESSAGE(time->get_date_string_from_unix_time(-23215075200) == "1234-05-06", "Time get_date_string_from_unix_time: The timestamp for an arbitrary date without time is as expected.");
	CHECK_MESSAGE(time->get_time_string_from_unix_time(25689) == "07:08:09", "Time get_time_string_from_unix_time: The timestamp for an arbitrary time without date is as expected.");
	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(1391983830) == "2014-02-09T22:10:30", "Time get_datetime_string_from_unix_time: The timestamp for GODOT IS OPEN SOURCE is as expected.");
	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(1391983830, true) == "2014-02-09 22:10:30", "Time get_datetime_string_from_unix_time: The timestamp for GODOT IS OPEN SOURCE with space is as expected.");
	CHECK_MESSAGE(time->get_date_string_from_unix_time(1391904000) == "2014-02-09", "Time get_date_string_from_unix_time: The date for GODOT IS OPEN SOURCE without time is as expected.");
	CHECK_MESSAGE(time->get_time_string_from_unix_time(79830) == "22:10:30", "Time get_time_string_from_unix_time: The time for GODOT IS OPEN SOURCE without date is as expected.");
	CHECK_MESSAGE(time->get_datetime_string_from_unix_time(31494784780800) == "1000000-01-01T00:00:00", "Time get_datetime_string_from_unix_time: The timestamp for the year a million is as expected.");
	CHECK_MESSAGE(time->get_offset_string_from_offset_minutes(0) == "+00:00", "Time get_offset_string_from_offset_minutes: The offset string is as expected.");
	CHECK_MESSAGE(time->get_offset_string_from_offset_minutes(-600) == "-10:00", "Time get_offset_string_from_offset_minutes: The offset string is as expected.");
	CHECK_MESSAGE(time->get_offset_string_from_offset_minutes(345) == "+05:45", "Time get_offset_string_from_offset_minutes: The offset string is as expected.");
}

TEST_CASE("[Time] Unix time conversion to/from datetime dictionary") {
	const Time *time = Time::get_singleton();

	Dictionary datetime;
	datetime[YEAR_KEY] = 2014;
	datetime[MONTH_KEY] = 2;
	datetime[DAY_KEY] = 9;
	datetime[WEEKDAY_KEY] = (int64_t)Weekday::WEEKDAY_SUNDAY;
	datetime[HOUR_KEY] = 22;
	datetime[MINUTE_KEY] = 10;
	datetime[SECOND_KEY] = 30;

	Dictionary date_only;
	date_only[YEAR_KEY] = 2014;
	date_only[MONTH_KEY] = 2;
	date_only[DAY_KEY] = 9;
	date_only[WEEKDAY_KEY] = (int64_t)Weekday::WEEKDAY_SUNDAY;

	Dictionary time_only;
	time_only[HOUR_KEY] = 22;
	time_only[MINUTE_KEY] = 10;
	time_only[SECOND_KEY] = 30;

	CHECK_MESSAGE(time->get_unix_time_from_datetime_dict(datetime) == 1391983830, "Time get_unix_time_from_datetime_dict: The datetime dictionary for GODOT IS OPEN SOURCE is converted to a timestamp as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_dict(date_only) == 1391904000, "Time get_unix_time_from_datetime_dict: The date dictionary for GODOT IS OPEN SOURCE is converted to a timestamp as expected.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_dict(time_only) == 79830, "Time get_unix_time_from_datetime_dict: The time dictionary for GODOT IS OPEN SOURCE is converted to a timestamp as expected.");

	CHECK_MESSAGE(time->get_datetime_dict_from_unix_time(1391983830).hash() == datetime.hash(), "Time get_datetime_dict_from_unix_time: The datetime timestamp for GODOT IS OPEN SOURCE is converted to a dictionary as expected.");
	CHECK_MESSAGE(time->get_date_dict_from_unix_time(1391904000).hash() == date_only.hash(), "Time get_date_dict_from_unix_time: The date timestamp for GODOT IS OPEN SOURCE is converted to a dictionary as expected.");
	CHECK_MESSAGE(time->get_time_dict_from_unix_time(79830).hash() == time_only.hash(), "Time get_time_dict_from_unix_time: The time timestamp for GODOT IS OPEN SOURCE is converted to a dictionary as expected.");

	CHECK_MESSAGE((Weekday)(int)time->get_datetime_dict_from_unix_time(0)[WEEKDAY_KEY] == Weekday::WEEKDAY_THURSDAY, "Time get_datetime_dict_from_unix_time: The weekday for the Unix epoch is a Thursday as expected.");
	CHECK_MESSAGE((Weekday)(int)time->get_datetime_dict_from_unix_time(1391983830)[WEEKDAY_KEY] == Weekday::WEEKDAY_SUNDAY, "Time get_datetime_dict_from_unix_time: The weekday for GODOT IS OPEN SOURCE is a Sunday as expected.");
}

TEST_CASE("[Time] System time methods") {
	const Time *time = Time::get_singleton();

	const uint64_t ticks_msec = time->get_ticks_msec();
	const uint64_t ticks_usec = time->get_ticks_usec();

	CHECK_MESSAGE(time->get_unix_time_from_system() > 1000000000, "Time get_unix_time_from_system: The timestamp from system time doesn't fail and is very positive.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_dict(time->get_datetime_dict_from_system()) > 1000000000, "Time get_datetime_string_from_system: The timestamp from system time doesn't fail and is very positive.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_dict(time->get_date_dict_from_system()) > 1000000000, "Time get_datetime_string_from_system: The date from system time doesn't fail and is very positive.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_dict(time->get_time_dict_from_system()) < 86400, "Time get_datetime_string_from_system: The time from system time doesn't fail and is within the acceptable range.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string(time->get_datetime_string_from_system()) > 1000000000, "Time get_datetime_string_from_system: The timestamp from system time doesn't fail and is very positive.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string(time->get_date_string_from_system()) > 1000000000, "Time get_datetime_string_from_system: The date from system time doesn't fail and is very positive.");
	CHECK_MESSAGE(time->get_unix_time_from_datetime_string(time->get_time_string_from_system()) < 86400, "Time get_datetime_string_from_system: The time from system time doesn't fail and is within the acceptable range.");

	CHECK_MESSAGE(time->get_ticks_msec() >= ticks_msec, "Time get_ticks_msec: The value has not decreased.");
	CHECK_MESSAGE(time->get_ticks_usec() > ticks_usec, "Time get_ticks_usec: The value has increased.");
}

} // namespace TestTime

#endif // TEST_TIME_H
