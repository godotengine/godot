/*************************************************************************/
/*  audio_server.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "audio_server.h"
#include "global_config.h"
#include "io/resource_loader.h"
#include "os/file_access.h"
#include "os/os.h"
#include "servers/audio/effects/audio_effect_compressor.h"
#ifdef TOOLS_ENABLED

#define MARK_EDITED set_edited(true);

#else

#define MARK_EDITED

#endif

AudioDriver *AudioDriver::singleton = NULL;
AudioDriver *AudioDriver::get_singleton() {

	return singleton;
}

void AudioDriver::set_singleton() {

	singleton = this;
}

void AudioDriver::audio_server_process(int p_frames, int32_t *p_buffer, bool p_update_mix_time) {

	if (p_update_mix_time)
		update_mix_time(p_frames);

	if (AudioServer::get_singleton())
		AudioServer::get_singleton()->_driver_process(p_frames, p_buffer);
}

void AudioDriver::update_mix_time(int p_frames) {

	_mix_amount += p_frames;
	_last_mix_time = OS::get_singleton()->get_ticks_usec();
}

double AudioDriver::get_mix_time() const {

	double total = (OS::get_singleton()->get_ticks_usec() - _last_mix_time) / 1000000.0;
	total += _mix_amount / (double)get_mix_rate();
	return total;
}

AudioDriver::AudioDriver() {

	_last_mix_time = 0;
	_mix_amount = 0;
}

AudioDriver *AudioDriverManager::drivers[MAX_DRIVERS];
int AudioDriverManager::driver_count = 0;

void AudioDriverManager::add_driver(AudioDriver *p_driver) {

	ERR_FAIL_COND(driver_count >= MAX_DRIVERS);
	drivers[driver_count++] = p_driver;
}

int AudioDriverManager::get_driver_count() {

	return driver_count;
}
AudioDriver *AudioDriverManager::get_driver(int p_driver) {

	ERR_FAIL_INDEX_V(p_driver, driver_count, NULL);
	return drivers[p_driver];
}

//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

void AudioServer::_driver_process(int p_frames, int32_t *p_buffer) {

	int todo = p_frames;

	while (todo) {

		if (to_mix == 0) {
			_mix_step();
		}

		int to_copy = MIN(to_mix, todo);

		Bus *master = buses[0];

		int from = buffer_size - to_mix;
		int from_buf = p_frames - todo;

		//master master, send to output
		int cs = master->channels.size();
		for (int k = 0; k < cs; k++) {

			if (master->channels[k].active) {

				AudioFrame *buf = master->channels[k].buffer.ptr();

				for (int j = 0; j < to_copy; j++) {

					float l = CLAMP(buf[from + j].l, -1.0, 1.0);
					int32_t vl = l * ((1 << 20) - 1);
					p_buffer[(from_buf + j) * (cs * 2) + k * 2 + 0] = vl << 11;

					float r = CLAMP(buf[from + j].r, -1.0, 1.0);
					int32_t vr = r * ((1 << 20) - 1);
					p_buffer[(from_buf + j) * (cs * 2) + k * 2 + 1] = vr << 11;
				}

			} else {
				for (int j = 0; j < to_copy; j++) {

					p_buffer[(from_buf + j) * (cs * 2) + k * 2 + 0] = 0;
					p_buffer[(from_buf + j) * (cs * 2) + k * 2 + 1] = 0;
				}
			}
		}

		todo -= to_copy;
		to_mix -= to_copy;
	}
}

void AudioServer::_mix_step() {

	for (int i = 0; i < buses.size(); i++) {
		Bus *bus = buses[i];
		bus->index_cache = i; //might be moved around by editor, so..
		for (int k = 0; k < bus->channels.size(); k++) {

			bus->channels[k].used = false;
		}
	}

	//make callbacks for mixing the audio
	for (Set<CallbackItem>::Element *E = callbacks.front(); E; E = E->next()) {

		E->get().callback(E->get().userdata);
	}

	for (int i = buses.size() - 1; i >= 0; i--) {
		//go bus by bus
		Bus *bus = buses[i];

		for (int k = 0; k < bus->channels.size(); k++) {

			if (bus->channels[k].active && !bus->channels[k].used) {
				//buffer was not used, but it's still active, so it must be cleaned
				AudioFrame *buf = bus->channels[k].buffer.ptr();

				for (uint32_t j = 0; j < buffer_size; j++) {

					buf[j] = AudioFrame(0, 0);
				}
			}
		}

		//process effects
		for (int j = 0; j < bus->effects.size(); j++) {

			if (!bus->effects[j].enabled)
				continue;

			for (int k = 0; k < bus->channels.size(); k++) {

				if (!bus->channels[k].active)
					continue;
				bus->channels[k].effect_instances[j]->process(bus->channels[k].buffer.ptr(), temp_buffer[k].ptr(), buffer_size);
			}

			//swap buffers, so internal buffer always has the right data
			for (int k = 0; k < bus->channels.size(); k++) {

				if (!buses[i]->channels[k].active)
					continue;
				SWAP(bus->channels[k].buffer, temp_buffer[k]);
			}
		}

		//process send

		Bus *send = NULL;

		if (i > 0) {
			//everything has a send save for master bus
			if (!bus_map.has(bus->send)) {
				send = buses[0];
			} else {
				send = bus_map[bus->send];
				if (send->index_cache >= bus->index_cache) { //invalid, send to master
					send = buses[0];
				}
			}
		}

		for (int k = 0; k < bus->channels.size(); k++) {

			if (!bus->channels[k].active)
				continue;

			AudioFrame *buf = bus->channels[k].buffer.ptr();

			AudioFrame peak = AudioFrame(0, 0);
			for (uint32_t j = 0; j < buffer_size; j++) {
				float l = ABS(buf[j].l);
				if (l > peak.l) {
					peak.l = l;
				}
				float r = ABS(buf[j].r);
				if (r > peak.r) {
					peak.r = r;
				}
			}

			bus->channels[k].peak_volume = AudioFrame(Math::linear2db(peak.l + 0.0000000001), Math::linear2db(peak.r + 0.0000000001));

			if (!bus->channels[k].used) {
				//see if any audio is contained, because channel was not used

				if (MAX(peak.r, peak.l) > Math::db2linear(channel_disable_treshold_db)) {
					bus->channels[k].last_mix_with_audio = mix_frames;
				} else if (mix_frames - bus->channels[k].last_mix_with_audio > channel_disable_frames) {
					bus->channels[k].active = false;
					continue; //went inactive, don't mix.
				}
			}

			if (send) {
				//if not master bus, send
				AudioFrame *target_buf = thread_get_channel_mix_buffer(send->index_cache, k);

				for (uint32_t j = 0; j < buffer_size; j++) {
					target_buf[j] += buf[j];
				}
			}
		}
	}

	mix_frames += buffer_size;
	to_mix = buffer_size;
}

AudioFrame *AudioServer::thread_get_channel_mix_buffer(int p_bus, int p_buffer) {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), NULL);
	ERR_FAIL_INDEX_V(p_buffer, buses[p_bus]->channels.size(), NULL);

	AudioFrame *data = buses[p_bus]->channels[p_buffer].buffer.ptr();

	if (!buses[p_bus]->channels[p_buffer].used) {
		buses[p_bus]->channels[p_buffer].used = true;
		buses[p_bus]->channels[p_buffer].active = true;
		buses[p_bus]->channels[p_buffer].last_mix_with_audio = mix_frames;
		for (uint32_t i = 0; i < buffer_size; i++) {
			data[i] = AudioFrame(0, 0);
		}
	}

	return data;
}

int AudioServer::thread_get_mix_buffer_size() const {

	return buffer_size;
}

int AudioServer::thread_find_bus_index(const StringName &p_name) {

	if (bus_map.has(p_name)) {
		return bus_map[p_name]->index_cache;
	} else {
		return 0;
	}
}

void AudioServer::set_bus_count(int p_count) {

	ERR_FAIL_COND(p_count < 1);
	ERR_FAIL_INDEX(p_count, 256);

	MARK_EDITED

	lock();
	int cb = buses.size();

	if (p_count < buses.size()) {
		for (int i = p_count; i < buses.size(); i++) {
			bus_map.erase(buses[i]->name);
			memdelete(buses[i]);
		}
	}

	buses.resize(p_count);

	for (int i = cb; i < buses.size(); i++) {

		String attempt = "New Bus";
		int attempts = 1;
		while (true) {

			bool name_free = true;
			for (int j = 0; j < i; j++) {

				if (buses[j]->name == attempt) {
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

		buses[i] = memnew(Bus);
		buses[i]->channels.resize(_get_channel_count());
		for (int j = 0; j < _get_channel_count(); j++) {
			buses[i]->channels[j].buffer.resize(buffer_size);
		}
		buses[i]->name = attempt;
		buses[i]->solo = false;
		buses[i]->mute = false;
		buses[i]->bypass = false;
		buses[i]->volume_db = 0;
		if (i > 0) {
			buses[i]->send = "Master";
		}

		bus_map[attempt] = buses[i];
	}

	unlock();

	emit_signal("bus_layout_changed");
}

void AudioServer::remove_bus(int p_index) {

	ERR_FAIL_INDEX(p_index, buses.size());
	ERR_FAIL_COND(p_index == 0);

	MARK_EDITED

	lock();
	bus_map.erase(buses[p_index]->name);
	memdelete(buses[p_index]);
	buses.remove(p_index);
	unlock();
}

void AudioServer::add_bus(int p_at_pos) {

	MARK_EDITED

	if (p_at_pos >= buses.size()) {
		p_at_pos = -1;
	} else if (p_at_pos == 0) {
		if (buses.size() > 1)
			p_at_pos = 1;
		else
			p_at_pos = -1;
	}

	String attempt = "New Bus";
	int attempts = 1;
	while (true) {

		bool name_free = true;
		for (int j = 0; j < buses.size(); j++) {

			if (buses[j]->name == attempt) {
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

	Bus *bus = memnew(Bus);
	bus->channels.resize(_get_channel_count());
	for (int j = 0; j < _get_channel_count(); j++) {
		bus->channels[j].buffer.resize(buffer_size);
	}
	bus->name = attempt;
	bus->solo = false;
	bus->mute = false;
	bus->bypass = false;
	bus->volume_db = 0;

	bus_map[attempt] = bus;

	if (p_at_pos == -1)
		buses.push_back(bus);
	else
		buses.insert(p_at_pos, bus);
}

void AudioServer::move_bus(int p_bus, int p_to_pos) {

	ERR_FAIL_COND(p_bus < 1 || p_bus >= buses.size());
	ERR_FAIL_COND(p_to_pos != -1 && (p_to_pos < 1 || p_to_pos > buses.size()));

	MARK_EDITED

	if (p_bus == p_to_pos)
		return;

	Bus *bus = buses[p_bus];
	buses.remove(p_bus);

	if (p_to_pos == -1) {
		buses.push_back(bus);
	} else if (p_to_pos < p_bus) {
		buses.insert(p_to_pos, bus);
	} else {
		buses.insert(p_to_pos - 1, bus);
	}
}

int AudioServer::get_bus_count() const {

	return buses.size();
}

void AudioServer::set_bus_name(int p_bus, const String &p_name) {

	ERR_FAIL_INDEX(p_bus, buses.size());
	if (p_bus == 0 && p_name != "Master")
		return; //bus 0 is always master

	MARK_EDITED

	lock();

	if (buses[p_bus]->name == p_name) {
		unlock();
		return;
	}

	String attempt = p_name;
	int attempts = 1;

	while (true) {

		bool name_free = true;
		for (int i = 0; i < buses.size(); i++) {

			if (buses[i]->name == attempt) {
				name_free = false;
				break;
			}
		}

		if (name_free) {
			break;
		}

		attempts++;
		attempt = p_name + " " + itos(attempts);
	}
	bus_map.erase(buses[p_bus]->name);
	buses[p_bus]->name = attempt;
	bus_map[attempt] = buses[p_bus];
	unlock();

	emit_signal("bus_layout_changed");
}
String AudioServer::get_bus_name(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), String());
	return buses[p_bus]->name;
}

void AudioServer::set_bus_volume_db(int p_bus, float p_volume_db) {

	ERR_FAIL_INDEX(p_bus, buses.size());

	MARK_EDITED

	buses[p_bus]->volume_db = p_volume_db;
}
float AudioServer::get_bus_volume_db(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), 0);
	return buses[p_bus]->volume_db;
}

void AudioServer::set_bus_send(int p_bus, const StringName &p_send) {

	ERR_FAIL_INDEX(p_bus, buses.size());

	MARK_EDITED

	buses[p_bus]->send = p_send;
}

StringName AudioServer::get_bus_send(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), StringName());
	return buses[p_bus]->send;
}

void AudioServer::set_bus_solo(int p_bus, bool p_enable) {

	ERR_FAIL_INDEX(p_bus, buses.size());

	MARK_EDITED

	buses[p_bus]->solo = p_enable;
}

bool AudioServer::is_bus_solo(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);

	return buses[p_bus]->solo;
}

void AudioServer::set_bus_mute(int p_bus, bool p_enable) {

	ERR_FAIL_INDEX(p_bus, buses.size());

	MARK_EDITED

	buses[p_bus]->mute = p_enable;
}
bool AudioServer::is_bus_mute(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);

	return buses[p_bus]->mute;
}

void AudioServer::set_bus_bypass_effects(int p_bus, bool p_enable) {

	ERR_FAIL_INDEX(p_bus, buses.size());

	MARK_EDITED

	buses[p_bus]->bypass = p_enable;
}
bool AudioServer::is_bus_bypassing_effects(int p_bus) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);

	return buses[p_bus]->bypass;
}

void AudioServer::_update_bus_effects(int p_bus) {

	for (int i = 0; i < buses[p_bus]->channels.size(); i++) {
		buses[p_bus]->channels[i].effect_instances.resize(buses[p_bus]->effects.size());
		for (int j = 0; j < buses[p_bus]->effects.size(); j++) {
			Ref<AudioEffectInstance> fx = buses[p_bus]->effects[j].effect->instance();
			if (fx->cast_to<AudioEffectCompressorInstance>()) {
				fx->cast_to<AudioEffectCompressorInstance>()->set_current_channel(i);
			}
			buses[p_bus]->channels[i].effect_instances[j] = fx;
		}
	}
}

void AudioServer::add_bus_effect(int p_bus, const Ref<AudioEffect> &p_effect, int p_at_pos) {

	ERR_FAIL_COND(p_effect.is_null());
	ERR_FAIL_INDEX(p_bus, buses.size());

	MARK_EDITED

	lock();

	Bus::Effect fx;
	fx.effect = p_effect;
	//fx.instance=p_effect->instance();
	fx.enabled = true;

	if (p_at_pos >= buses[p_bus]->effects.size() || p_at_pos < 0) {
		buses[p_bus]->effects.push_back(fx);
	} else {
		buses[p_bus]->effects.insert(p_at_pos, fx);
	}

	_update_bus_effects(p_bus);

	unlock();
}

void AudioServer::remove_bus_effect(int p_bus, int p_effect) {

	ERR_FAIL_INDEX(p_bus, buses.size());

	MARK_EDITED

	lock();

	buses[p_bus]->effects.remove(p_effect);
	_update_bus_effects(p_bus);

	unlock();
}

int AudioServer::get_bus_effect_count(int p_bus) {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), 0);

	return buses[p_bus]->effects.size();
}

Ref<AudioEffect> AudioServer::get_bus_effect(int p_bus, int p_effect) {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), Ref<AudioEffect>());
	ERR_FAIL_INDEX_V(p_effect, buses[p_bus]->effects.size(), Ref<AudioEffect>());

	return buses[p_bus]->effects[p_effect].effect;
}

void AudioServer::swap_bus_effects(int p_bus, int p_effect, int p_by_effect) {

	ERR_FAIL_INDEX(p_bus, buses.size());
	ERR_FAIL_INDEX(p_effect, buses[p_bus]->effects.size());
	ERR_FAIL_INDEX(p_by_effect, buses[p_bus]->effects.size());

	MARK_EDITED

	lock();
	SWAP(buses[p_bus]->effects[p_effect], buses[p_bus]->effects[p_by_effect]);
	_update_bus_effects(p_bus);
	unlock();
}

void AudioServer::set_bus_effect_enabled(int p_bus, int p_effect, bool p_enabled) {

	ERR_FAIL_INDEX(p_bus, buses.size());
	ERR_FAIL_INDEX(p_effect, buses[p_bus]->effects.size());

	MARK_EDITED

	buses[p_bus]->effects[p_effect].enabled = p_enabled;
}
bool AudioServer::is_bus_effect_enabled(int p_bus, int p_effect) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);
	ERR_FAIL_INDEX_V(p_effect, buses[p_bus]->effects.size(), false);
	return buses[p_bus]->effects[p_effect].enabled;
}

float AudioServer::get_bus_peak_volume_left_db(int p_bus, int p_channel) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), 0);
	ERR_FAIL_INDEX_V(p_channel, buses[p_bus]->channels.size(), 0);

	return buses[p_bus]->channels[p_channel].peak_volume.l;
}
float AudioServer::get_bus_peak_volume_right_db(int p_bus, int p_channel) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), 0);
	ERR_FAIL_INDEX_V(p_channel, buses[p_bus]->channels.size(), 0);

	return buses[p_bus]->channels[p_channel].peak_volume.r;
}

bool AudioServer::is_bus_channel_active(int p_bus, int p_channel) const {

	ERR_FAIL_INDEX_V(p_bus, buses.size(), false);
	ERR_FAIL_INDEX_V(p_channel, buses[p_bus]->channels.size(), false);

	return buses[p_bus]->channels[p_channel].active;
}

void AudioServer::init() {

	channel_disable_treshold_db = GLOBAL_DEF("audio/channel_disable_treshold_db", -60.0);
	channel_disable_frames = float(GLOBAL_DEF("audio/channel_disable_time", 2.0)) * get_mix_rate();
	buffer_size = 1024; //harcoded for now
	switch (get_speaker_mode()) {
		case SPEAKER_MODE_STEREO: {
			temp_buffer.resize(1);
		} break;
		case SPEAKER_SURROUND_51: {
			temp_buffer.resize(3);
		} break;
		case SPEAKER_SURROUND_71: {
			temp_buffer.resize(4);
		} break;
	}

	for (int i = 0; i < temp_buffer.size(); i++) {
		temp_buffer[i].resize(buffer_size);
	}

	mix_count = 0;
	set_bus_count(1);
	;
	set_bus_name(0, "Master");

	if (AudioDriver::get_singleton())
		AudioDriver::get_singleton()->start();
#ifdef TOOLS_ENABLED
	set_edited(false); //avoid editors from thinking this was edited
#endif
}

void AudioServer::load_default_bus_layout() {

	if (FileAccess::exists("res://default_bus_layout.tres")) {
		Ref<AudioBusLayout> default_layout = ResourceLoader::load("res://default_bus_layout.tres");
		if (default_layout.is_valid()) {
			set_bus_layout(default_layout);
		}
	}
}

void AudioServer::finish() {

	for (int i = 0; i < buses.size(); i++) {
		memdelete(buses[i]);
	}

	buses.clear();
}
void AudioServer::update() {
}

/* MISC config */

void AudioServer::lock() {

	AudioDriver::get_singleton()->lock();
}
void AudioServer::unlock() {

	AudioDriver::get_singleton()->unlock();
}

AudioServer::SpeakerMode AudioServer::get_speaker_mode() const {

	return (AudioServer::SpeakerMode)AudioDriver::get_singleton()->get_speaker_mode();
}
float AudioServer::get_mix_rate() const {

	return AudioDriver::get_singleton()->get_mix_rate();
}

float AudioServer::read_output_peak_db() const {

	return 0;
}

AudioServer *AudioServer::get_singleton() {

	return singleton;
}

double AudioServer::get_mix_time() const {

	return 0;
}
double AudioServer::get_output_delay() const {

	return 0;
}

AudioServer *AudioServer::singleton = NULL;

void *AudioServer::audio_data_alloc(uint32_t p_data_len, const uint8_t *p_from_data) {

	void *ad = memalloc(p_data_len);
	ERR_FAIL_COND_V(!ad, NULL);
	if (p_from_data) {
		copymem(ad, p_from_data, p_data_len);
	}

	audio_data_lock->lock();
	audio_data[ad] = p_data_len;
	audio_data_total_mem += p_data_len;
	audio_data_max_mem = MAX(audio_data_total_mem, audio_data_max_mem);
	audio_data_lock->unlock();

	return ad;
}

void AudioServer::audio_data_free(void *p_data) {

	audio_data_lock->lock();
	if (!audio_data.has(p_data)) {
		audio_data_lock->unlock();
		ERR_FAIL();
	}

	audio_data_total_mem -= audio_data[p_data];
	audio_data.erase(p_data);
	memfree(p_data);
	audio_data_lock->unlock();
}

size_t AudioServer::audio_data_get_total_memory_usage() const {

	return audio_data_total_mem;
}
size_t AudioServer::audio_data_get_max_memory_usage() const {

	return audio_data_max_mem;
}

void AudioServer::add_callback(AudioCallback p_callback, void *p_userdata) {
	lock();
	CallbackItem ci;
	ci.callback = p_callback;
	ci.userdata = p_userdata;
	callbacks.insert(ci);
	unlock();
}

void AudioServer::remove_callback(AudioCallback p_callback, void *p_userdata) {

	lock();
	CallbackItem ci;
	ci.callback = p_callback;
	ci.userdata = p_userdata;
	callbacks.erase(ci);
	unlock();
}

void AudioServer::set_bus_layout(const Ref<AudioBusLayout> &p_bus_layout) {

	ERR_FAIL_COND(p_bus_layout.is_null() || p_bus_layout->buses.size() == 0);

	lock();
	for (int i = 0; i < buses.size(); i++) {
		memdelete(buses[i]);
	}
	buses.resize(p_bus_layout->buses.size());
	bus_map.clear();
	for (int i = 0; i < p_bus_layout->buses.size(); i++) {
		Bus *bus = memnew(Bus);
		if (i == 0) {
			bus->name = "Master";
		} else {
			bus->name = p_bus_layout->buses[i].name;
			bus->send = p_bus_layout->buses[i].send;
		}

		bus->solo = p_bus_layout->buses[i].solo;
		bus->mute = p_bus_layout->buses[i].mute;
		bus->bypass = p_bus_layout->buses[i].bypass;
		bus->volume_db = p_bus_layout->buses[i].volume_db;

		for (int j = 0; j < p_bus_layout->buses[i].effects.size(); j++) {

			Ref<AudioEffect> fx = p_bus_layout->buses[i].effects[j].effect;

			if (fx.is_valid()) {

				Bus::Effect bfx;
				bfx.effect = fx;
				bfx.enabled = p_bus_layout->buses[i].effects[j].enabled;
				bus->effects.push_back(bfx);
			}
		}

		bus_map[bus->name] = bus;
		buses[i] = bus;

		buses[i]->channels.resize(_get_channel_count());
		for (int j = 0; j < _get_channel_count(); j++) {
			buses[i]->channels[j].buffer.resize(buffer_size);
		}
		_update_bus_effects(i);
	}
#ifdef TOOLS_ENABLED
	set_edited(false);
#endif
	unlock();
}

Ref<AudioBusLayout> AudioServer::generate_bus_layout() const {

	Ref<AudioBusLayout> state;
	state.instance();

	state->buses.resize(buses.size());

	for (int i = 0; i < buses.size(); i++) {

		state->buses[i].name = buses[i]->name;
		state->buses[i].send = buses[i]->send;
		state->buses[i].mute = buses[i]->mute;
		state->buses[i].solo = buses[i]->solo;
		state->buses[i].bypass = buses[i]->bypass;
		state->buses[i].volume_db = buses[i]->volume_db;
		for (int j = 0; j < buses[i]->effects.size(); j++) {
			AudioBusLayout::Bus::Effect fx;
			fx.effect = buses[i]->effects[j].effect;
			fx.enabled = buses[i]->effects[j].enabled;
			state->buses[i].effects.push_back(fx);
		}
	}

	return state;
}

void AudioServer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_bus_count", "amount"), &AudioServer::set_bus_count);
	ClassDB::bind_method(D_METHOD("get_bus_count"), &AudioServer::get_bus_count);

	ClassDB::bind_method(D_METHOD("remove_bus", "index"), &AudioServer::remove_bus);
	ClassDB::bind_method(D_METHOD("add_bus", "at_pos"), &AudioServer::add_bus, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("move_bus", "index", "to_index"), &AudioServer::move_bus);

	ClassDB::bind_method(D_METHOD("set_bus_name", "bus_idx", "name"), &AudioServer::set_bus_name);
	ClassDB::bind_method(D_METHOD("get_bus_name", "bus_idx"), &AudioServer::get_bus_name);

	ClassDB::bind_method(D_METHOD("set_bus_volume_db", "bus_idx", "volume_db"), &AudioServer::set_bus_volume_db);
	ClassDB::bind_method(D_METHOD("get_bus_volume_db", "bus_idx"), &AudioServer::get_bus_volume_db);

	ClassDB::bind_method(D_METHOD("set_bus_send", "bus_idx", "send"), &AudioServer::set_bus_send);
	ClassDB::bind_method(D_METHOD("get_bus_send", "bus_idx"), &AudioServer::get_bus_send);

	ClassDB::bind_method(D_METHOD("set_bus_solo", "bus_idx", "enable"), &AudioServer::set_bus_solo);
	ClassDB::bind_method(D_METHOD("is_bus_solo", "bus_idx"), &AudioServer::is_bus_solo);

	ClassDB::bind_method(D_METHOD("set_bus_mute", "bus_idx", "enable"), &AudioServer::set_bus_mute);
	ClassDB::bind_method(D_METHOD("is_bus_mute", "bus_idx"), &AudioServer::is_bus_mute);

	ClassDB::bind_method(D_METHOD("set_bus_bypass_effects", "bus_idx", "enable"), &AudioServer::set_bus_bypass_effects);
	ClassDB::bind_method(D_METHOD("is_bus_bypassing_effects", "bus_idx"), &AudioServer::is_bus_bypassing_effects);

	ClassDB::bind_method(D_METHOD("add_bus_effect", "bus_idx", "effect:AudioEffect"), &AudioServer::add_bus_effect, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("remove_bus_effect", "bus_idx", "effect_idx"), &AudioServer::remove_bus_effect);

	ClassDB::bind_method(D_METHOD("get_bus_effect_count", "bus_idx"), &AudioServer::add_bus_effect);
	ClassDB::bind_method(D_METHOD("get_bus_effect:AudioEffect", "bus_idx", "effect_idx"), &AudioServer::get_bus_effect);
	ClassDB::bind_method(D_METHOD("swap_bus_effects", "bus_idx", "effect_idx", "by_effect_idx"), &AudioServer::swap_bus_effects);

	ClassDB::bind_method(D_METHOD("set_bus_effect_enabled", "bus_idx", "effect_idx", "enabled"), &AudioServer::set_bus_effect_enabled);
	ClassDB::bind_method(D_METHOD("is_bus_effect_enabled", "bus_idx", "effect_idx"), &AudioServer::is_bus_effect_enabled);

	ClassDB::bind_method(D_METHOD("get_bus_peak_volume_left_db", "bus_idx", "channel"), &AudioServer::get_bus_peak_volume_left_db);
	ClassDB::bind_method(D_METHOD("get_bus_peak_volume_right_db", "bus_idx", "channel"), &AudioServer::get_bus_peak_volume_right_db);

	ClassDB::bind_method(D_METHOD("lock"), &AudioServer::lock);
	ClassDB::bind_method(D_METHOD("unlock"), &AudioServer::unlock);

	ClassDB::bind_method(D_METHOD("get_speaker_mode"), &AudioServer::get_speaker_mode);
	ClassDB::bind_method(D_METHOD("get_mix_rate"), &AudioServer::get_mix_rate);

	ClassDB::bind_method(D_METHOD("set_bus_layout", "bus_layout:AudioBusLayout"), &AudioServer::set_bus_layout);
	ClassDB::bind_method(D_METHOD("generate_bus_layout:AudioBusLayout"), &AudioServer::generate_bus_layout);

	ADD_SIGNAL(MethodInfo("bus_layout_changed"));
}

AudioServer::AudioServer() {

	singleton = this;
	audio_data_total_mem = 0;
	audio_data_max_mem = 0;
	audio_data_lock = Mutex::create();
	mix_frames = 0;
	to_mix = 0;
}

AudioServer::~AudioServer() {

	memdelete(audio_data_lock);
}

/////////////////////////////////

bool AudioBusLayout::_set(const StringName &p_name, const Variant &p_value) {

	String s = p_name;
	if (s.begins_with("bus/")) {
		int index = s.get_slice("/", 1).to_int();
		if (buses.size() <= index) {
			buses.resize(index + 1);
		}

		Bus &bus = buses[index];

		String what = s.get_slice("/", 2);

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
			int which = s.get_slice("/", 3).to_int();
			if (bus.effects.size() <= which) {
				bus.effects.resize(which + 1);
			}

			Bus::Effect &fx = bus.effects[which];

			String fxwhat = s.get_slice("/", 4);
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

		int index = s.get_slice("/", 1).to_int();
		if (index < 0 || index >= buses.size())
			return false;

		const Bus &bus = buses[index];

		String what = s.get_slice("/", 2);

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
			int which = s.get_slice("/", 3).to_int();
			if (which < 0 || which >= bus.effects.size()) {
				return false;
			}

			const Bus::Effect &fx = bus.effects[which];

			String fxwhat = s.get_slice("/", 4);
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
		p_list->push_back(PropertyInfo(Variant::STRING, "bus/" + itos(i) + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/solo", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/mute", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/bypass_fx", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::REAL, "bus/" + itos(i) + "/volume_db", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		p_list->push_back(PropertyInfo(Variant::REAL, "bus/" + itos(i) + "/send", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));

		for (int j = 0; j < buses[i].effects.size(); j++) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, "bus/" + itos(i) + "/effect/" + itos(j) + "/effect", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
			p_list->push_back(PropertyInfo(Variant::BOOL, "bus/" + itos(i) + "/effect/" + itos(j) + "/enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR));
		}
	}
}

AudioBusLayout::AudioBusLayout() {

	buses.resize(1);
	buses[0].name = "Master";
}
