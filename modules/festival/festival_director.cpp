/**************************************************************************/
/*  festival_director.cpp                                                 */
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

#include "festival_director.h"

#include "festival_clock.h"
#include "festival_notebook.h"
#include "festival_outfit.h"
#include "festival_registry.h"
#include "festival_weather.h"
#include "festival_world.h"

FestivalDirector *FestivalDirector::singleton = nullptr;

FestivalDirector *FestivalDirector::get_singleton() { return singleton; }

bool FestivalDirector::_matches_any(const StringName &p_current, const Variant &p_accepted) const {
	if (p_accepted.get_type() == Variant::ARRAY) {
		const Array a = p_accepted;
		for (int i = 0; i < a.size(); i++) {
			if (String(p_current) == String(a[i])) {
				return true;
			}
		}
		return false;
	}
	return String(p_current) == String(p_accepted);
}

bool FestivalDirector::_check_requirements(const Dictionary &d) const {
	FestivalWorld *world = FestivalWorld::get_singleton();
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	FestivalClock *clock = FestivalClock::get_singleton();
	FestivalWeather *weather = FestivalWeather::get_singleton();

	if (d.has("requires_outfit") && world) {
		if (!_matches_any(world->get_outfit(), d["requires_outfit"])) {
			return false;
		}
	}
	if (d.has("requires_role")) {
		if (!_matches_any(get_perceived_role(), d["requires_role"])) {
			return false;
		}
	}
	if (world) {
		const Array req_items = d.get("requires_items", Array());
		for (int i = 0; i < req_items.size(); i++) {
			const StringName id = req_items[i];
			if (!world->has_item(id, 1)) {
				return false;
			}
		}
		const Array req_presented = d.get("requires_presented", Array());
		for (int i = 0; i < req_presented.size(); i++) {
			const StringName id = req_presented[i];
			if (!world->is_presented(id)) {
				return false;
			}
		}
	}
	if (notebook) {
		const Array req_knowledge = d.get("requires_knowledge", Array());
		for (int i = 0; i < req_knowledge.size(); i++) {
			const StringName id = req_knowledge[i];
			if (!notebook->knows(id)) {
				return false;
			}
		}
	}
	if (world) {
		const Dictionary req_flags = d.get("requires_flags", Dictionary());
		const Array keys = req_flags.keys();
		for (int i = 0; i < keys.size(); i++) {
			const StringName flag = keys[i];
			if (world->get_flag(flag) != req_flags[keys[i]]) {
				return false;
			}
		}
	}
	if (clock) {
		const Array req_phase = d.get("requires_phase", Array());
		if (req_phase.size() > 0 && !req_phase.has((int)clock->get_phase())) {
			return false;
		}
	}
	if (weather) {
		const int req_weather = d.get("requires_weather", -1);
		if (req_weather >= 0 && req_weather != (int)weather->get_weather()) {
			return false;
		}
	}
	return true;
}

void FestivalDirector::_apply_outcomes(const Dictionary &d) {
	FestivalWorld *world = FestivalWorld::get_singleton();
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	FestivalClock *clock = FestivalClock::get_singleton();

	if (notebook) {
		const Array grants = d.get("grants_knowledge", Array());
		for (int i = 0; i < grants.size(); i++) {
			const StringName id = grants[i];
			notebook->learn(id);
		}
	}
	if (world) {
		const Dictionary set_flags = d.get("sets_flags", Dictionary());
		const Array keys = set_flags.keys();
		for (int i = 0; i < keys.size(); i++) {
			const StringName flag = keys[i];
			world->set_flag(flag, set_flags[keys[i]]);
		}
		const Array gives = d.get("gives_items", Array());
		for (int i = 0; i < gives.size(); i++) {
			const StringName id = gives[i];
			world->add_item(id, 1);
		}
		const Array takes = d.get("takes_items", Array());
		for (int i = 0; i < takes.size(); i++) {
			const StringName id = takes[i];
			world->remove_item(id, 1);
		}
	}
	if (clock) {
		const StringName milestone = d.get("trigger_milestone", StringName());
		if (milestone != StringName()) {
			clock->trigger_milestone(milestone);
		}
		if ((bool)d.get("advance_phase", false)) {
			clock->advance_phase();
		}
	}
}

Dictionary FestivalDirector::_find_interaction(const Ref<FestivalNPCProfile> &p_npc, const StringName &p_id) const {
	if (p_npc.is_null()) {
		return Dictionary();
	}
	const Array interactions = p_npc->get_interactions();
	for (int i = 0; i < interactions.size(); i++) {
		const Dictionary d = interactions[i];
		if (StringName(d.get("id", StringName())) == p_id) {
			return d;
		}
	}
	return Dictionary();
}

void FestivalDirector::begin_run(int64_t p_seed) {
	FestivalWorld *world = FestivalWorld::get_singleton();
	FestivalClock *clock = FestivalClock::get_singleton();
	FestivalWeather *weather = FestivalWeather::get_singleton();
	if (world) {
		world->reset();
	}
	if (clock) {
		clock->reset();
	}
	if (weather) {
		weather->roll(p_seed);
	}
	const int wx = weather ? (int)weather->get_weather() : 0;
	emit_signal(SNAME("run_started"), wx);
}

void FestivalDirector::end_run() {
	FestivalClock *clock = FestivalClock::get_singleton();
	if (clock) {
		while (!clock->is_ended()) {
			clock->advance_phase();
		}
	}
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	if (notebook) {
		notebook->save();
	}
	emit_signal(SNAME("run_concluded"));
}

StringName FestivalDirector::get_perceived_role() const {
	FestivalWorld *world = FestivalWorld::get_singleton();
	FestivalRegistry *registry = FestivalRegistry::get_singleton();
	if (!world || !registry) {
		return StringName();
	}
	Ref<FestivalOutfit> outfit = registry->get_outfit(world->get_outfit());
	return outfit.is_valid() ? outfit->get_role() : StringName();
}

int FestivalDirector::get_authority() const {
	FestivalWorld *world = FestivalWorld::get_singleton();
	FestivalRegistry *registry = FestivalRegistry::get_singleton();
	if (!world || !registry) {
		return 0;
	}
	Ref<FestivalOutfit> outfit = registry->get_outfit(world->get_outfit());
	return outfit.is_valid() ? outfit->get_authority() : 0;
}

Dictionary FestivalDirector::resolve_reaction(const Ref<FestivalNPCProfile> &p_npc) {
	Dictionary out;
	ERR_FAIL_COND_V(p_npc.is_null(), out);
	FestivalClock *clock = FestivalClock::get_singleton();
	FestivalWeather *weather = FestivalWeather::get_singleton();
	const int phase = clock ? (int)clock->get_phase() : 0;
	const int wx = weather ? (int)weather->get_weather() : 0;

	out["npc"] = p_npc->get_id();
	out["perceived_role"] = get_perceived_role();
	out["authority"] = get_authority();
	out["surface_personality"] = p_npc->get_surface_personality();
	out["state"] = p_npc->get_state_for(phase, wx);

	Array available;
	const Array interactions = p_npc->get_interactions();
	for (int i = 0; i < interactions.size(); i++) {
		const Dictionary d = interactions[i];
		if (_check_requirements(d)) {
			Dictionary summary;
			summary["id"] = d.get("id", StringName());
			summary["dialogue"] = d.get("dialogue", String());
			available.push_back(summary);
		}
	}
	out["available_interactions"] = available;
	return out;
}

Array FestivalDirector::get_available_interactions(const Ref<FestivalNPCProfile> &p_npc) {
	Array available;
	ERR_FAIL_COND_V(p_npc.is_null(), available);
	const Array interactions = p_npc->get_interactions();
	for (int i = 0; i < interactions.size(); i++) {
		const Dictionary d = interactions[i];
		if (_check_requirements(d)) {
			available.push_back(d.get("id", StringName()));
		}
	}
	return available;
}

bool FestivalDirector::can_interact(const Ref<FestivalNPCProfile> &p_npc, const StringName &p_interaction_id) const {
	const Dictionary d = _find_interaction(p_npc, p_interaction_id);
	if (d.is_empty()) {
		return false;
	}
	return _check_requirements(d);
}

bool FestivalDirector::apply_interaction(const Ref<FestivalNPCProfile> &p_npc, const StringName &p_interaction_id) {
	const Dictionary d = _find_interaction(p_npc, p_interaction_id);
	if (d.is_empty() || !_check_requirements(d)) {
		return false;
	}
	_apply_outcomes(d);
	emit_signal(SNAME("interaction_applied"), p_npc->get_id(), p_interaction_id, d.get("dialogue", String()));
	return true;
}

void FestivalDirector::learn(const StringName &p_id) {
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	if (notebook) {
		notebook->learn(p_id);
	}
}

bool FestivalDirector::knows(const StringName &p_id) const {
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	return notebook ? notebook->knows(p_id) : false;
}

bool FestivalDirector::load_notebook(const String &p_path) {
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	return notebook ? notebook->load(p_path) : false;
}

bool FestivalDirector::save_notebook(const String &p_path) const {
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	return notebook ? notebook->save(p_path) : false;
}

Dictionary FestivalDirector::get_context() const {
	Dictionary c;
	FestivalWorld *world = FestivalWorld::get_singleton();
	FestivalNotebook *notebook = FestivalNotebook::get_singleton();
	FestivalClock *clock = FestivalClock::get_singleton();
	FestivalWeather *weather = FestivalWeather::get_singleton();
	if (world) {
		c["outfit"] = world->get_outfit();
		c["items"] = world->get_items();
		c["presented"] = world->get_presented();
		c["flags"] = world->get_flags();
	}
	c["role"] = get_perceived_role();
	c["authority"] = get_authority();
	if (notebook) {
		c["knowledge"] = notebook->get_known();
	}
	if (clock) {
		c["phase"] = (int)clock->get_phase();
		c["phase_name"] = clock->get_phase_name();
	}
	if (weather) {
		c["weather"] = (int)weather->get_weather();
		c["weather_name"] = weather->get_weather_name();
	}
	return c;
}

FestivalDirector::FestivalDirector() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;
}

FestivalDirector::~FestivalDirector() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

void FestivalDirector::_bind_methods() {
	ClassDB::bind_method(D_METHOD("begin_run", "seed"), &FestivalDirector::begin_run, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("end_run"), &FestivalDirector::end_run);

	ClassDB::bind_method(D_METHOD("get_perceived_role"), &FestivalDirector::get_perceived_role);
	ClassDB::bind_method(D_METHOD("get_authority"), &FestivalDirector::get_authority);
	ClassDB::bind_method(D_METHOD("resolve_reaction", "npc"), &FestivalDirector::resolve_reaction);
	ClassDB::bind_method(D_METHOD("get_available_interactions", "npc"), &FestivalDirector::get_available_interactions);
	ClassDB::bind_method(D_METHOD("can_interact", "npc", "interaction_id"), &FestivalDirector::can_interact);
	ClassDB::bind_method(D_METHOD("apply_interaction", "npc", "interaction_id"), &FestivalDirector::apply_interaction);

	ClassDB::bind_method(D_METHOD("learn", "id"), &FestivalDirector::learn);
	ClassDB::bind_method(D_METHOD("knows", "id"), &FestivalDirector::knows);
	ClassDB::bind_method(D_METHOD("load_notebook", "path"), &FestivalDirector::load_notebook, DEFVAL("user://festival_notebook.cfg"));
	ClassDB::bind_method(D_METHOD("save_notebook", "path"), &FestivalDirector::save_notebook, DEFVAL("user://festival_notebook.cfg"));

	ClassDB::bind_method(D_METHOD("get_context"), &FestivalDirector::get_context);

	ADD_SIGNAL(MethodInfo("run_started", PropertyInfo(Variant::INT, "weather")));
	ADD_SIGNAL(MethodInfo("run_concluded"));
	ADD_SIGNAL(MethodInfo("interaction_applied", PropertyInfo(Variant::STRING_NAME, "npc"), PropertyInfo(Variant::STRING_NAME, "interaction_id"), PropertyInfo(Variant::STRING, "dialogue")));
}
