/**************************************************************************/
/*  time.hpp                                                              */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Time : public Object {
	GDEXTENSION_CLASS(Time, Object)

	static Time *singleton;

public:
	enum Month {
		MONTH_JANUARY = 1,
		MONTH_FEBRUARY = 2,
		MONTH_MARCH = 3,
		MONTH_APRIL = 4,
		MONTH_MAY = 5,
		MONTH_JUNE = 6,
		MONTH_JULY = 7,
		MONTH_AUGUST = 8,
		MONTH_SEPTEMBER = 9,
		MONTH_OCTOBER = 10,
		MONTH_NOVEMBER = 11,
		MONTH_DECEMBER = 12,
	};

	enum Weekday {
		WEEKDAY_SUNDAY = 0,
		WEEKDAY_MONDAY = 1,
		WEEKDAY_TUESDAY = 2,
		WEEKDAY_WEDNESDAY = 3,
		WEEKDAY_THURSDAY = 4,
		WEEKDAY_FRIDAY = 5,
		WEEKDAY_SATURDAY = 6,
	};

	static Time *get_singleton();

	Dictionary get_datetime_dict_from_unix_time(int64_t p_unix_time_val) const;
	Dictionary get_date_dict_from_unix_time(int64_t p_unix_time_val) const;
	Dictionary get_time_dict_from_unix_time(int64_t p_unix_time_val) const;
	String get_datetime_string_from_unix_time(int64_t p_unix_time_val, bool p_use_space = false) const;
	String get_date_string_from_unix_time(int64_t p_unix_time_val) const;
	String get_time_string_from_unix_time(int64_t p_unix_time_val) const;
	Dictionary get_datetime_dict_from_datetime_string(const String &p_datetime, bool p_weekday) const;
	String get_datetime_string_from_datetime_dict(const Dictionary &p_datetime, bool p_use_space) const;
	int64_t get_unix_time_from_datetime_dict(const Dictionary &p_datetime) const;
	int64_t get_unix_time_from_datetime_string(const String &p_datetime) const;
	String get_offset_string_from_offset_minutes(int64_t p_offset_minutes) const;
	Dictionary get_datetime_dict_from_system(bool p_utc = false) const;
	Dictionary get_date_dict_from_system(bool p_utc = false) const;
	Dictionary get_time_dict_from_system(bool p_utc = false) const;
	String get_datetime_string_from_system(bool p_utc = false, bool p_use_space = false) const;
	String get_date_string_from_system(bool p_utc = false) const;
	String get_time_string_from_system(bool p_utc = false) const;
	Dictionary get_time_zone_from_system() const;
	double get_unix_time_from_system() const;
	uint64_t get_ticks_msec() const;
	uint64_t get_ticks_usec() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~Time();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Time::Month);
VARIANT_ENUM_CAST(Time::Weekday);

