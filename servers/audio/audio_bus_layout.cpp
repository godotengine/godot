/**************************************************************************/
/*  audio_bus_layout.cpp                                                  */
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

#include "audio_bus_layout.h"

#include "core/object/class_db.h"
#include "scene/scene_string_names.h"
#include "servers/audio/audio_effect.h" // IWYU pragma: keep. For Bus::Effect.
#include "servers/audio/audio_server.h"

void AudioBusLayout::set_server_bus_layout() {
	// AudioServer::_set_server_bus_layout() exists for compatibility.
	// AudioServer::set_bus_layout() is deprecated and no longer part of all builds.
	// More internal untangling in the AudioServer is required
	// to apply bus layouts without involving Resources
	// and this function is the stopgap until that is done.
	AudioServer::get_singleton()->_set_server_bus_layout(this);
}

void AudioBusLayout::generate_from_server_bus_layout() {
	AudioServer *audioserver = AudioServer::get_singleton();
	ERR_FAIL_NULL(audioserver);

	const int server_bus_size = audioserver->get_bus_count();

	buses.resize(server_bus_size);

	Bus *buses_ptrw = buses.ptrw();

	for (int i = 0; i < server_bus_size; i++) {
		buses_ptrw[i].name = audioserver->get_bus_name(i);
		buses_ptrw[i].send = audioserver->get_bus_send(i);
		buses_ptrw[i].mute = audioserver->is_bus_mute(i);
		buses_ptrw[i].solo = audioserver->is_bus_solo(i);
		buses_ptrw[i].bypass = audioserver->is_bus_bypassing_effects(i);
		buses_ptrw[i].volume_db = audioserver->get_bus_volume_db(i);

		const int server_bus_effect_size = audioserver->get_bus_effect_count(i);

		buses_ptrw[i].effects.clear();

		for (int j = 0; j < server_bus_effect_size; j++) {
			AudioBusLayout::Bus::Effect fx;
			fx.effect = audioserver->get_bus_effect(i, j);
			fx.enabled = audioserver->is_bus_effect_enabled(i, j);
			buses_ptrw[i].effects.push_back(fx);
		}
	}
}

void AudioBusLayout::add_bus(int p_at_pos) {
	if (p_at_pos >= buses.size()) {
		p_at_pos = -1;
	} else if (p_at_pos == 0) {
		if (buses.size() > 1) {
			p_at_pos = 1;
		} else {
			p_at_pos = -1;
		}
	}

	String attempt = "New Bus";
	int attempts = 1;
	while (true) {
		bool name_free = true;
		for (const Bus &bus : buses) {
			if (bus.name == attempt) {
				name_free = false;
				break;
			}
		}

		if (!name_free) {
			attempts++;
			attempt = "New Bus " + itos(attempts);
		} else {
			break;
		}
	}

	if (p_at_pos == -1) {
		p_at_pos = buses.size();
		buses.resize(buses.size() + 1);
	} else {
		buses.insert(p_at_pos, Bus());
	}
	buses.write[p_at_pos].name = attempt;
	emit_changed();
}

void AudioBusLayout::remove_bus(int p_bus) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	ERR_FAIL_COND(p_bus == 0);

	buses.remove_at(p_bus);
	emit_changed();
}

void AudioBusLayout::move_bus(int p_bus, int p_to_index) {
	if (p_bus == p_to_index) {
		return;
	}

	ERR_FAIL_COND(p_bus < 1 || p_bus >= buses.size());
	ERR_FAIL_COND(p_to_index != -1 && (p_to_index < 1 || p_to_index > buses.size()));

	const Bus bus = buses[p_bus];
	buses.remove_at(p_bus);

	if (p_to_index == -1) {
		buses.push_back(bus);
	} else if (p_to_index < p_bus) {
		buses.insert(p_to_index, bus);
	} else {
		buses.insert(p_to_index - 1, bus);
	}
	emit_changed();
}

int AudioBusLayout::get_bus_count() const {
	return buses.size();
}

void AudioBusLayout::add_bus_effect(int p_bus, const Ref<AudioEffect> &p_effect, int p_at_pos) {
	ERR_FAIL_COND(p_effect.is_null());
	ERR_FAIL_INDEX(p_bus, buses.size());

	Bus::Effect fx;
	fx.effect = p_effect;
	fx.enabled = true;

	if (p_at_pos >= buses[p_bus].effects.size() || p_at_pos < 0) {
		buses.write[p_bus].effects.push_back(fx);
	} else {
		buses.write[p_bus].effects.insert(p_at_pos, fx);
	}
	emit_changed();
}

void AudioBusLayout::remove_bus_effect(int p_bus, int p_effect) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	buses.write[p_bus].effects.remove_at(p_effect);
	emit_changed();
}

int AudioBusLayout::get_bus_effect_count(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), 0);
	return buses[p_bus].effects.size();
}

void AudioBusLayout::set_bus_name(int p_bus, const String &p_name) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	buses.write[p_bus].name = p_name;
	emit_changed();
}

void AudioBusLayout::set_bus_send(int p_bus, const StringName &p_send) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	buses.write[p_bus].send = p_send;
	emit_changed();
}

void AudioBusLayout::set_bus_mute(int p_bus, bool p_mute) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	buses.write[p_bus].mute = p_mute;
	emit_changed();
}

void AudioBusLayout::set_bus_solo(int p_bus, bool p_solo) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	buses.write[p_bus].solo = p_solo;
	emit_changed();
}

void AudioBusLayout::set_bus_bypass_effects(int p_bus, bool p_enabled) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	buses.write[p_bus].bypass = p_enabled;
	emit_changed();
}

void AudioBusLayout::set_bus_volume_db(int p_bus, float p_volume_db) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	buses.write[p_bus].volume_db = p_volume_db;
	emit_changed();
}

String AudioBusLayout::get_bus_name(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), String());
	return buses[p_bus].name;
}

StringName AudioBusLayout::get_bus_send(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), StringName());
	return buses[p_bus].send;
}

bool AudioBusLayout::is_bus_mute(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);
	return buses[p_bus].mute;
}

bool AudioBusLayout::is_bus_solo(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);
	return buses[p_bus].solo;
}

bool AudioBusLayout::is_bus_bypassing_effects(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);
	return buses[p_bus].bypass;
}

float AudioBusLayout::get_bus_volume_db(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), 0);
	return buses[p_bus].volume_db;
}

void AudioBusLayout::set_bus_effect_enabled(int p_bus, int p_effect, bool p_enabled) {
	ERR_FAIL_INDEX(p_bus, buses.size());
	ERR_FAIL_INDEX(p_effect, buses[p_bus].effects.size());

	buses.write[p_bus].effects.write[p_effect].enabled = p_enabled;
	emit_changed();
}

Ref<AudioEffect> AudioBusLayout::get_bus_effect(int p_bus, int p_effect) {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), Ref<AudioEffect>());
	ERR_FAIL_INDEX_V(p_effect, buses[p_bus].effects.size(), Ref<AudioEffect>());
	return buses[p_bus].effects[p_effect].effect;
}

bool AudioBusLayout::is_bus_effect_enabled(int p_bus, int p_effect) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);
	ERR_FAIL_INDEX_V(p_effect, buses[p_bus].effects.size(), false);
	return buses[p_bus].effects[p_effect].enabled;
}

bool AudioBusLayout::_set(const StringName &p_name, const Variant &p_value) {
	String s = p_name;
	if (s.begins_with("bus/")) {
		int index = s.get_slicec('/', 1).to_int();
		if (buses.size() <= index) {
			buses.resize(index + 1);
		}

		Bus &bus = buses.write[index];

		String what = s.get_slicec('/', 2);

		if (what == "name") {
			bus.name = p_value;
		} else if (what == "solo") {
			bus.solo = p_value;
		} else if (what == "mute") {
			bus.mute = p_value;
		} else if (what == "bypass_fx") {
			bus.bypass = p_value;
		} else if (what == "volume_db") {
			bus.volume_db = p_value;
		} else if (what == "send") {
			bus.send = p_value;
		} else if (what == "effect") {
			int which = s.get_slicec('/', 3).to_int();
			if (bus.effects.size() <= which) {
				bus.effects.resize(which + 1);
			}

			Bus::Effect &fx = bus.effects.write[which];

			String fxwhat = s.get_slicec('/', 4);
			if (fxwhat == "effect") {
				fx.effect = p_value;
			} else if (fxwhat == "enabled") {
				fx.enabled = p_value;
			} else {
				return false;
			}

			return true;
		} else {
			return false;
		}

		return true;
	}

	return false;
}

bool AudioBusLayout::_get(const StringName &p_name, Variant &r_ret) const {
	String s = p_name;
	if (s.begins_with("bus/")) {
		int index = s.get_slicec('/', 1).to_int();
		if (index < 0 || index >= buses.size()) {
			return false;
		}

		const Bus &bus = buses[index];

		String what = s.get_slicec('/', 2);

		if (what == "name") {
			r_ret = bus.name;
		} else if (what == "solo") {
			r_ret = bus.solo;
		} else if (what == "mute") {
			r_ret = bus.mute;
		} else if (what == "bypass_fx") {
			r_ret = bus.bypass;
		} else if (what == "volume_db") {
			r_ret = bus.volume_db;
		} else if (what == "send") {
			r_ret = bus.send;
		} else if (what == "effect") {
			int which = s.get_slicec('/', 3).to_int();
			if (which < 0 || which >= bus.effects.size()) {
				return false;
			}

			const Bus::Effect &fx = bus.effects[which];

			String fxwhat = s.get_slicec('/', 4);
			if (fxwhat == "effect") {
				r_ret = fx.effect;
			} else if (fxwhat == "enabled") {
				r_ret = fx.enabled;
			} else {
				return false;
			}

			return true;
		} else {
			return false;
		}

		return true;
	}

	return false;
}

void AudioBusLayout::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < buses.size(); i++) {
		p_list->push_back(PropertyInfo(Variant::STRING, "bus/" + itos(i) + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/solo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/mute", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/bypass_fx", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "bus/" + itos(i) + "/volume_db", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "bus/" + itos(i) + "/send", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));

		for (int j = 0; j < buses[i].effects.size(); j++) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "bus/" + itos(i) + "/effect/" + itos(j) + "/effect", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
			p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/effect/" + itos(j) + "/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		}
	}
}

void AudioBusLayout::_bind_methods() {
	ClassDB::bind_method(D_METHOD("generate_from_server_bus_layout"), &AudioBusLayout::generate_from_server_bus_layout);
	ClassDB::bind_method(D_METHOD("set_server_bus_layout"), &AudioBusLayout::set_server_bus_layout);

	ClassDB::bind_method(D_METHOD("_add_bus", "at_pos"), &AudioBusLayout::add_bus, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("_remove_bus", "bus"), &AudioBusLayout::remove_bus);
	ClassDB::bind_method(D_METHOD("_move_bus", "to_index"), &AudioBusLayout::move_bus);

	ClassDB::bind_method(D_METHOD("_set_bus_name", "bus", "name"), &AudioBusLayout::set_bus_name);
	ClassDB::bind_method(D_METHOD("_set_bus_send", "bus", "send"), &AudioBusLayout::set_bus_send);
	ClassDB::bind_method(D_METHOD("_set_bus_mute", "bus", "mute"), &AudioBusLayout::set_bus_mute);
	ClassDB::bind_method(D_METHOD("_set_bus_solo", "bus", "solo"), &AudioBusLayout::set_bus_solo);
	ClassDB::bind_method(D_METHOD("_set_bus_bypass_effects", "bus", "enabled"), &AudioBusLayout::set_bus_bypass_effects);
	ClassDB::bind_method(D_METHOD("_set_bus_volume_db", "bus", "volume_db"), &AudioBusLayout::set_bus_volume_db);

	ClassDB::bind_method(D_METHOD("_add_bus_effect", "bus", "effect", "index"), &AudioBusLayout::add_bus_effect, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("_remove_bus_effect", "bus", "index"), &AudioBusLayout::remove_bus_effect);
	ClassDB::bind_method(D_METHOD("_set_bus_effect_enabled", "bus", "effect", "enabled"), &AudioBusLayout::set_bus_effect_enabled);
}

AudioBusLayout::AudioBusLayout() {
	buses.resize(1);
	buses.write[0].name = SceneStringName(Master);
}
