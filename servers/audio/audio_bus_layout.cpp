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

int AudioBusLayout::get_bus_count() const {
	return buses.size();
}

int AudioBusLayout::get_bus_effect_count(int p_bus) const {
	ERR_FAIL_INDEX_V(p_bus, buses.size(), 0);
	return buses[p_bus].effects.size();
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
}

AudioBusLayout::AudioBusLayout() {
	buses.resize(1);
	buses.write[0].name = SceneStringName(Master);
}
