/**************************************************************************/
/*  spx_audio_bus_pool.cpp                                                     */
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

#include "servers/audio/effects/audio_effect_panner.h"
#include "spx_audio_mgr.h"
#include "spx_engine.h"
#include "spx_res_mgr.h"
#include "spx_audio_bus_pool.h"

#define audioMgr SpxEngine::get_singleton()->get_audio()
#define audioPool SpxAudioBusPool::get_singleton()

// Initialize static members
SpxAudioBusPool *SpxAudioBusPool::singleton = nullptr;
StringName SpxAudioBusPool::STR_BUS_MASTER;
StringName SpxAudioBusPool::STR_BUS_SFX;
StringName SpxAudioBusPool::STR_BUS_MUSIC;


SpxAudioBusPool *SpxAudioBusPool::get_singleton() {
	return singleton;
}

StringName SpxAudioBusPool::get_bus_name(int id) {
	if (id == BUS_MASTER) {
		return STR_BUS_MASTER;
	} else if (id == BUS_SFX) {
		return STR_BUS_SFX;
	} else if (id == BUS_MUSIC) {
		return STR_BUS_MUSIC;
	}
	return String::num_real(float(id));
}
void SpxAudioBusPool::init() {

	SpxAudioBusPool::STR_BUS_MASTER = "Master";
	SpxAudioBusPool::STR_BUS_SFX = "Sfx";
	SpxAudioBusPool::STR_BUS_MUSIC = "Music";

	singleton = memnew(SpxAudioBusPool);
	// Start with DEFAULT_BUS_COUNT buses (includes master)
	AudioServer::get_singleton()->set_bus_count(DEFAULT_BUS_COUNT);
	// Master bus is at index 0, so we start from 1
	singleton->current_bus_count = DEFAULT_BUS_COUNT;
	// Set up buses
	AudioServer::get_singleton()->set_bus_name(BUS_SFX, String::num_real(float(BUS_SFX)));
	AudioServer::get_singleton()->set_bus_name(BUS_MUSIC, String::num_real(float(BUS_MUSIC)));

	AudioServer::get_singleton()->set_bus_send(BUS_SFX, STR_BUS_MASTER);
	AudioServer::get_singleton()->set_bus_send(BUS_MUSIC, STR_BUS_MASTER);

	// Add initial buses to pool (skipping the first 3 buses: Master, SFX, MUSIC)
	for (int i = 3; i < DEFAULT_BUS_COUNT; i++) {
		AudioServer::get_singleton()->set_bus_name(i, String::num_real(float(i)));
		AudioServer::get_singleton()->set_bus_send(i, STR_BUS_MASTER);
		singleton->free_buses.push_back(i);
	}
}

int SpxAudioBusPool::alloc() {
	if (free_buses.size() == 0) {
		// Expand bus pool if no free buses
		expand_buses();
	}

	// Get a bus from the pool
	int bus_id = free_buses[free_buses.size() - 1];
	free_buses.remove_at(free_buses.size() - 1);

	// Mark as active
	active_buses[bus_id] = true;

	return bus_id;
}

void SpxAudioBusPool::free(int id) {
	// Validate the bus ID
	if (id <= BUS_MUSIC || !active_buses.has(id) || !active_buses[id]) {
		print_error("Trying to free invalid bus ID: " + itos(id));
		return;
	}

	// Mark as inactive
	active_buses[id] = false;

	// Return to the pool
	free_buses.push_back(id);
	// reset the bus
	set_pan(id, 0.0);
	set_volume(id, 1.0);
}

void SpxAudioBusPool::set_volume(int id, GdFloat volume) {
	if (!is_valid_bus(id))
		return;

	// Convert to decibels (Godot uses decibel scale for volume)
	auto db = Math::linear_to_db(volume);
	AudioServer::get_singleton()->set_bus_volume_db(id, db);
}

GdFloat SpxAudioBusPool::get_volume(int id) {
	if (!is_valid_bus(id))
		return 0.0f;

	// Get volume in decibels
	float db = AudioServer::get_singleton()->get_bus_volume_db(id);
	return Math::db_to_linear(db);
}

void SpxAudioBusPool::set_pan(int id, GdFloat pan) {
	if (!is_valid_bus(id))
		return;

	// Clamp pan between -1 and 1
	pan = CLAMP(pan, -1.0f, 1.0f);

	// Check if there's already a panner effect
	int effect_count = AudioServer::get_singleton()->get_bus_effect_count(id);
	int panner_idx = -1;

	for (int i = 0; i < effect_count; i++) {
		Ref<AudioEffect> effect = AudioServer::get_singleton()->get_bus_effect(id, i);
		if (effect->is_class("AudioEffectPanner")) {
			panner_idx = i;
			break;
		}
	}

	// Create a panner effect if not exists
	Ref<AudioEffectPanner> panner;
	if (panner_idx == -1) {
		panner.instantiate();
		AudioServer::get_singleton()->add_bus_effect(id, panner);
	} else {
		panner = AudioServer::get_singleton()->get_bus_effect(id, panner_idx);
	}

	// Set the pan value
	panner->set_pan(pan);
}

GdFloat SpxAudioBusPool::get_pan(int id) {
	if (!is_valid_bus(id))
		return 0.0f;

	// Find panner effect
	int effect_count = AudioServer::get_singleton()->get_bus_effect_count(id);

	for (int i = 0; i < effect_count; i++) {
		Ref<AudioEffect> effect = AudioServer::get_singleton()->get_bus_effect(id, i);
		if (effect->is_class("AudioEffectPanner")) {
			Ref<AudioEffectPanner> panner = effect;
			return panner->get_pan();
		}
	}

	return 0.0f; // Default pan value if no panner found
}

void SpxAudioBusPool::expand_buses() {
	int new_count = current_bus_count + BUS_EXPANSION_SIZE;
	AudioServer::get_singleton()->set_bus_count(new_count);

	// Initialize new buses and add them to the pool
	for (int i = current_bus_count; i < new_count; i++) {
		AudioServer::get_singleton()->set_bus_name(i, String::num_real(float(i)));
		AudioServer::get_singleton()->set_bus_send(i, STR_BUS_MASTER);
		free_buses.push_back(i);
	}

	current_bus_count = new_count;
}

bool SpxAudioBusPool::is_valid_bus(int id) {
	if (id < 0 || id >= current_bus_count) {
		print_error("Invalid bus ID: " + itos(id));
		return false;
	}

	if (id > BUS_MUSIC && (!active_buses.has(id) || !active_buses[id])) {
		print_error("Bus ID not active: " + itos(id));
		return false;
	}

	return true;
}
