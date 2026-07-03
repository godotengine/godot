/**************************************************************************/
/*  festival_weather.cpp                                                  */
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

#include "festival_weather.h"

#include "core/object/class_db.h"

#include "core/math/math_funcs.h"

FestivalWeather *FestivalWeather::singleton = nullptr;

FestivalWeather *FestivalWeather::get_singleton() { return singleton; }

FestivalWeather::Weather FestivalWeather::get_weather() const { return weather; }

String FestivalWeather::get_weather_name() const {
	return weather == WEATHER_RAIN ? "Rain" : "Sun";
}

bool FestivalWeather::is_rain() const { return weather == WEATHER_RAIN; }
bool FestivalWeather::is_sun() const { return weather == WEATHER_SUN; }

void FestivalWeather::set_weather(Weather p_weather) {
	if (weather == p_weather) {
		return;
	}
	weather = p_weather;
	emit_signal(SNAME("weather_changed"), (int)weather);
}

void FestivalWeather::roll(int64_t p_seed) {
	if (p_seed < 0) {
		Math::randomize();
	} else {
		Math::seed((uint64_t)p_seed);
	}
	set_weather((Math::rand() & 1) ? WEATHER_RAIN : WEATHER_SUN);
}

FestivalWeather::FestivalWeather() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

FestivalWeather::~FestivalWeather() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void FestivalWeather::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_weather"), &FestivalWeather::get_weather);
	ClassDB::bind_method(D_METHOD("get_weather_name"), &FestivalWeather::get_weather_name);
	ClassDB::bind_method(D_METHOD("is_rain"), &FestivalWeather::is_rain);
	ClassDB::bind_method(D_METHOD("is_sun"), &FestivalWeather::is_sun);
	ClassDB::bind_method(D_METHOD("set_weather", "weather"), &FestivalWeather::set_weather);
	ClassDB::bind_method(D_METHOD("roll", "seed"), &FestivalWeather::roll, DEFVAL(-1));

	ADD_SIGNAL(MethodInfo("weather_changed", PropertyInfo(Variant::INT, "weather")));

	BIND_ENUM_CONSTANT(WEATHER_SUN);
	BIND_ENUM_CONSTANT(WEATHER_RAIN);
}
