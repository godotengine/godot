/**************************************************************************/
/*  usd_animation.cpp                                                     */
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

#include "usd_animation.h"

void USDAnimation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_name"), &USDAnimation::get_name);
	ClassDB::bind_method(D_METHOD("set_name", "name"), &USDAnimation::set_name);
	ClassDB::bind_method(D_METHOD("get_start_time"), &USDAnimation::get_start_time);
	ClassDB::bind_method(D_METHOD("set_start_time", "start_time"), &USDAnimation::set_start_time);
	ClassDB::bind_method(D_METHOD("get_end_time"), &USDAnimation::get_end_time);
	ClassDB::bind_method(D_METHOD("set_end_time", "end_time"), &USDAnimation::set_end_time);
	ClassDB::bind_method(D_METHOD("get_time_codes_per_second"), &USDAnimation::get_time_codes_per_second);
	ClassDB::bind_method(D_METHOD("set_time_codes_per_second", "time_codes_per_second"), &USDAnimation::set_time_codes_per_second);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "name"), "set_name", "get_name");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "start_time"), "set_start_time", "get_start_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "end_time"), "set_end_time", "get_end_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "time_codes_per_second"), "set_time_codes_per_second", "get_time_codes_per_second");
}

String USDAnimation::get_name() const {
	return name;
}

void USDAnimation::set_name(const String &p_name) {
	name = p_name;
}

double USDAnimation::get_start_time() const {
	return start_time;
}

void USDAnimation::set_start_time(double p_start_time) {
	start_time = p_start_time;
}

double USDAnimation::get_end_time() const {
	return end_time;
}

void USDAnimation::set_end_time(double p_end_time) {
	end_time = p_end_time;
}

double USDAnimation::get_time_codes_per_second() const {
	return time_codes_per_second;
}

void USDAnimation::set_time_codes_per_second(double p_time_codes_per_second) {
	time_codes_per_second = p_time_codes_per_second;
}
