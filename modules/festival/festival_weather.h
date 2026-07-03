/**************************************************************************/
/*  festival_weather.h                                                    */
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

#pragma once

#include "core/object/object.h"
#include "core/variant/binder_common.h"

// Morning weather, rolled at the start of every run so repeated runs are never
// perfectly identical. Weather feeds NPC schedule variants and interaction gates.
class FestivalWeather : public Object {
	GDCLASS(FestivalWeather, Object);

	static FestivalWeather *singleton;

public:
	enum Weather {
		WEATHER_SUN,
		WEATHER_RAIN,
	};

private:
	Weather weather = WEATHER_SUN;

protected:
	static void _bind_methods();

public:
	static FestivalWeather *get_singleton();

	Weather get_weather() const;
	String get_weather_name() const;
	bool is_rain() const;
	bool is_sun() const;

	void set_weather(Weather p_weather);
	// Randomize weather for a new run. A negative seed uses a fresh random seed;
	// a non-negative seed is deterministic (useful for tests and shared puzzles).
	void roll(int64_t p_seed = -1);

	FestivalWeather();
	~FestivalWeather();
};

VARIANT_ENUM_CAST(FestivalWeather::Weather);
