/**************************************************************************/
/*  festival_clock.cpp                                                    */
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

#include "festival_clock.h"

#include "core/object/class_db.h"

FestivalClock *FestivalClock::singleton = nullptr;

FestivalClock *FestivalClock::get_singleton() { return singleton; }

FestivalClock::Phase FestivalClock::get_phase() const { return phase; }

String FestivalClock::get_phase_name() const {
	switch (phase) {
		case PHASE_MORNING:
			return "Morning";
		case PHASE_AFTERNOON:
			return "Afternoon";
		case PHASE_NIGHT:
			return "Night";
		case PHASE_ENDED:
			return "Ended";
	}
	return "Unknown";
}

bool FestivalClock::is_ended() const { return phase == PHASE_ENDED; }

bool FestivalClock::advance_phase() {
	if (phase == PHASE_ENDED) {
		return false;
	}
	const Phase old_phase = phase;
	phase = (Phase)((int)phase + 1);
	emit_signal(SNAME("phase_changed"), (int)old_phase, (int)phase);
	if (phase == PHASE_ENDED) {
		emit_signal(SNAME("run_ended"));
	}
	return true;
}

void FestivalClock::reset() {
	phase = PHASE_MORNING;
	milestones.clear();
}

void FestivalClock::trigger_milestone(const StringName &p_id) {
	if (milestones.has(p_id)) {
		return;
	}
	milestones.insert(p_id);
	emit_signal(SNAME("milestone_reached"), p_id);
}

bool FestivalClock::has_milestone(const StringName &p_id) const {
	return milestones.has(p_id);
}

PackedStringArray FestivalClock::get_milestones() const {
	PackedStringArray out;
	for (const StringName &E : milestones) {
		out.push_back(E);
	}
	return out;
}

FestivalClock::FestivalClock() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

FestivalClock::~FestivalClock() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void FestivalClock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_phase"), &FestivalClock::get_phase);
	ClassDB::bind_method(D_METHOD("get_phase_name"), &FestivalClock::get_phase_name);
	ClassDB::bind_method(D_METHOD("is_ended"), &FestivalClock::is_ended);
	ClassDB::bind_method(D_METHOD("advance_phase"), &FestivalClock::advance_phase);
	ClassDB::bind_method(D_METHOD("reset"), &FestivalClock::reset);
	ClassDB::bind_method(D_METHOD("trigger_milestone", "id"), &FestivalClock::trigger_milestone);
	ClassDB::bind_method(D_METHOD("has_milestone", "id"), &FestivalClock::has_milestone);
	ClassDB::bind_method(D_METHOD("get_milestones"), &FestivalClock::get_milestones);

	ADD_SIGNAL(MethodInfo("phase_changed", PropertyInfo(Variant::INT, "from_phase"), PropertyInfo(Variant::INT, "to_phase")));
	ADD_SIGNAL(MethodInfo("run_ended"));
	ADD_SIGNAL(MethodInfo("milestone_reached", PropertyInfo(Variant::STRING_NAME, "id")));

	BIND_ENUM_CONSTANT(PHASE_MORNING);
	BIND_ENUM_CONSTANT(PHASE_AFTERNOON);
	BIND_ENUM_CONSTANT(PHASE_NIGHT);
	BIND_ENUM_CONSTANT(PHASE_ENDED);
}
