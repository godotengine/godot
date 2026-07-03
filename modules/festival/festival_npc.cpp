/**************************************************************************/
/*  festival_npc.cpp                                                      */
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

#include "festival_npc.h"

#include "core/object/class_db.h"

#include "festival_clock.h"
#include "festival_director.h"
#include "festival_weather.h"

#include "core/config/engine.h"
#include "core/object/callable_mp.h"

void FestivalNPC::set_profile(const Ref<FestivalNPCProfile> &p_profile) {
	profile = p_profile;
	refresh_state();
}

Ref<FestivalNPCProfile> FestivalNPC::get_profile() const { return profile; }

void FestivalNPC::refresh_state() {
	if (profile.is_null()) {
		current_state = Dictionary();
		return;
	}
	FestivalClock *clock = FestivalClock::get_singleton();
	FestivalWeather *weather = FestivalWeather::get_singleton();
	const int phase = clock ? (int)clock->get_phase() : 0;
	const int wx = weather ? (int)weather->get_weather() : 0;
	current_state = profile->get_state_for(phase, wx);
	emit_signal(SNAME("state_changed"), current_state);
}

Dictionary FestivalNPC::get_current_state() const { return current_state; }

String FestivalNPC::get_current_location() const { return current_state.get("location", String()); }
String FestivalNPC::get_current_activity() const { return current_state.get("activity", String()); }

Dictionary FestivalNPC::interact() {
	Dictionary reaction;
	FestivalDirector *director = FestivalDirector::get_singleton();
	if (director && profile.is_valid()) {
		reaction = director->resolve_reaction(profile);
	}
	emit_signal(SNAME("interacted"), reaction);
	return reaction;
}

void FestivalNPC::_on_phase_changed(int p_from, int p_to) { refresh_state(); }
void FestivalNPC::_on_weather_changed(int p_weather) { refresh_state(); }
void FestivalNPC::_on_run_started(int p_weather) { refresh_state(); }

void FestivalNPC::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint()) {
				return;
			}
			FestivalClock *clock = FestivalClock::get_singleton();
			if (clock) {
				clock->connect(SNAME("phase_changed"), callable_mp(this, &FestivalNPC::_on_phase_changed));
			}
			FestivalWeather *weather = FestivalWeather::get_singleton();
			if (weather) {
				weather->connect(SNAME("weather_changed"), callable_mp(this, &FestivalNPC::_on_weather_changed));
			}
			FestivalDirector *director = FestivalDirector::get_singleton();
			if (director) {
				director->connect(SNAME("run_started"), callable_mp(this, &FestivalNPC::_on_run_started));
			}
			refresh_state();
		} break;
	}
}

FestivalNPC::FestivalNPC() {}

void FestivalNPC::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_profile", "profile"), &FestivalNPC::set_profile);
	ClassDB::bind_method(D_METHOD("get_profile"), &FestivalNPC::get_profile);
	ClassDB::bind_method(D_METHOD("refresh_state"), &FestivalNPC::refresh_state);
	ClassDB::bind_method(D_METHOD("get_current_state"), &FestivalNPC::get_current_state);
	ClassDB::bind_method(D_METHOD("get_current_location"), &FestivalNPC::get_current_location);
	ClassDB::bind_method(D_METHOD("get_current_activity"), &FestivalNPC::get_current_activity);
	ClassDB::bind_method(D_METHOD("interact"), &FestivalNPC::interact);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "profile", PROPERTY_HINT_RESOURCE_TYPE, "FestivalNPCProfile"), "set_profile", "get_profile");

	ADD_SIGNAL(MethodInfo("state_changed", PropertyInfo(Variant::DICTIONARY, "state")));
	ADD_SIGNAL(MethodInfo("interacted", PropertyInfo(Variant::DICTIONARY, "reaction")));
}
