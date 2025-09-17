/**************************************************************************/
/*  spx_audio.cpp                                                     */
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

#include "spx_audio.h"

#include "gdextension_spx_ext.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/main/node.h"
#include "spx_audio_mgr.h"
#include "spx_engine.h"
#include "spx_res_mgr.h"
#include "spx_audio_bus_pool.h"

#define audioMgr SpxEngine::get_singleton()->get_audio()
#define audioPool SpxAudioBusPool::get_singleton()

#define check_and_get_aid_audio_v()                                           \
	auto audio = _get_aid_audio(aid);                                         \
	if (audio == nullptr) {                                                   \
		return;                                                               \
	}

#define check_and_get_aid_audio_r(VALUE)                                      \
	auto audio = _get_aid_audio(aid);                                         \
	if (audio == nullptr) {                                                   \
		return VALUE;                                                         \
	}

AudioStreamPlayer2D *SpxAudio::_get_aid_audio(GdInt aid) {
	if (aid_audios.has(aid)) {
		return aid_audios[aid];
	}
	return nullptr;
}


void SpxAudio::on_create(GdInt id, Node *root) {
	this->root = root;
	this->id = id;
	bus_id = SpxAudioBusPool::BUS_SFX;
	bus_name = SpxAudioBusPool::STR_BUS_SFX;
}

void SpxAudio::stop_all() {
	for (List<AudioStreamPlayer2D *>::Element *item = audios.front(); item;) {
		item->get()->queue_free();
		item = item->next();
	}
	audios.clear();

	for (List<AudioStreamPlayer2D *>::Element *item = loop_audios.front(); item;) {
		item->get()->queue_free();
		item = item->next();
	}
	loop_audios.clear();

	aid_audios.clear();
}

void SpxAudio::on_destroy() {
	stop_all();
	// free the bus
	if (bus_id != SpxAudioBusPool::BUS_SFX) {
		audioPool->free(bus_id);
	}
}

void SpxAudio::on_update(float delta) {
	// check the audio is done
	for (auto item = audios.front(); item;) {
		const auto audio = item->get();
		auto *next = item->next();
		if (!audio->is_playing()) {
			audio->queue_free();
			audios.erase(item);
			for (const KeyValue<GdInt, AudioStreamPlayer2D *> &E : aid_audios) {
				if (E.value == audio) {
					aid_audios.erase(E.key);
					break;
				}
			}
		}
		item = next;
	}

	for (auto item = loop_audios.front(); item;) {
		const auto audio = item->get();
		auto *next = item->next();
		if (audio->get_stream().is_valid()&& !audio->is_playing()  && !audio->get_stream_paused()) {
			audio->play();
		}
		item = next;
	}
}

void SpxAudio::play(GdInt aid, GdString path, Node* owner, GdFloat attenuation, GdFloat max_distance) {
	auto path_str = SpxStr(path);
	Ref<AudioStream> stream = resMgr->load_audio(path_str);
	auto audio = memnew(AudioStreamPlayer2D);
	if(owner != nullptr){
		owner->add_child(audio);
	} else {
		root->add_child(audio);
	}
	audio->set_bus(bus_name);
	audio->set_stream(stream);
	audio->set_max_distance(max_distance);
	audio->set_attenuation(attenuation);
	audio->play();
	audio->set_name(path_str);
	audio->set_pitch_scale(get_pitch());
	audios.push_back(audio);
	aid_audios[aid] = audio;
}

GdBool SpxAudio::is_playing(GdInt aid) {
	check_and_get_aid_audio_r(false)
	return audio->is_playing();
}

void SpxAudio::pause(GdInt aid) {
	check_and_get_aid_audio_v()
	audio->set_stream_paused(true);
}

void SpxAudio::resume(GdInt aid) {
	check_and_get_aid_audio_v()
	audio->set_stream_paused(false);
}

void SpxAudio::stop(GdInt aid) {
	check_and_get_aid_audio_v()
	audios.erase(audio);
	loop_audios.erase(audio);
	aid_audios.erase(aid);
	audio->stop();
	audio->queue_free();
}

void SpxAudio::set_loop(GdInt aid, GdBool loop) {
	check_and_get_aid_audio_v()
	if (loop) {
		auto succ = audios.erase(audio);
		if (succ) {
			loop_audios.push_back(audio);
		}
	} else {
		auto succ = loop_audios.erase(audio);
		if (succ) {
			audios.push_back(audio);
		}
	}
}

GdBool SpxAudio::get_loop(GdInt aid) {
	check_and_get_aid_audio_r(false)
	return loop_audios.find(audio) != nullptr;
}

GdFloat SpxAudio::get_timer(GdInt aid) {
	check_and_get_aid_audio_r(0)
	return audio->get_playback_position();
}

void SpxAudio::set_timer(GdInt aid, GdFloat time) {
	check_and_get_aid_audio_v()
	audio->seek(time);
}

void SpxAudio::set_pitch(GdFloat pitch) {
	cur_pitch = pitch;
	// is need to update the pitch of the all audios ?
}

GdFloat SpxAudio::get_pitch() {
	return cur_pitch;
}

void SpxAudio::set_pan(GdFloat pan) {
	on_bus_dirty();
	audioPool->set_pan(bus_id, pan);
}

GdFloat SpxAudio::get_pan() {
	return audioPool->get_pan(bus_id);
}

void SpxAudio::set_volume(GdFloat volume) {
	on_bus_dirty();
	audioPool->set_volume(bus_id, volume);
}

GdFloat SpxAudio::get_volume() {
	return audioPool->get_volume(bus_id);
}

void SpxAudio::on_bus_dirty() {
	if (bus_id == SpxAudioBusPool::BUS_SFX) {
		bus_id = audioPool->alloc();
		bus_name = audioPool->get_bus_name(bus_id);

		for (auto item = audios.front(); item;) {
			item->get()->set_bus(bus_name);
			item = item->next();
		}

		for (auto item = loop_audios.front(); item;) {
			item->get()->set_bus(bus_name);
			item = item->next();
		}
	}
}
