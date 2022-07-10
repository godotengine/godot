/*************************************************************************/
/*  audio_stream_playlist.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_stream_playlist.h"

#include "core/math/math_funcs.h"
#include "core/string/print_string.h"

#include <iostream>
#include <utility>

AudioStreamPlaylist::AudioStreamPlaylist() {
}

Ref<AudioStreamPlayback> AudioStreamPlaylist::instantiate_playback() {
	Ref<AudioStreamPlaybackPlaylist> playback_playlist;
	playback_playlist.instantiate();
	playback_playlist->playlist = Ref<AudioStreamPlaylist>(this);
	playback_playlist->_update_playback_instances();
	playbacks.insert(playback_playlist.operator->());
	return playback_playlist;
}

String AudioStreamPlaylist::get_stream_name() const {
	return "Playlist";
}

float AudioStreamPlaylist::get_length() const {
	return 0;
}

void AudioStreamPlaylist::reset() {
	set_position(0);
}

void AudioStreamPlaylist::set_position(uint64_t p) {
	pos = p;
}

void AudioStreamPlaylist::set_stream_beats(int beats) {
	beat_count = beats;
}

int AudioStreamPlaylist::get_stream_beats() {
	return beat_count;
}

void AudioStreamPlaylist::set_list_stream(int stream_number, Ref<AudioStream> p_stream) {
	ERR_FAIL_COND(p_stream == this);
	ERR_FAIL_INDEX(stream_number, MAX_STREAMS);

	AudioServer::get_singleton()->lock();
	audio_streams[stream_number] = p_stream;
	for (AudioStreamPlaybackPlaylist *E : playbacks) {
		E->_update_playback_instances();
	}
	AudioServer::get_singleton()->unlock();
}

Ref<AudioStream> AudioStreamPlaylist::get_list_stream(int stream_number) {
	ERR_FAIL_INDEX_V(stream_number, MAX_STREAMS, Ref<AudioStream>());

	return audio_streams[stream_number];
}

void AudioStreamPlaylist::set_stream_count(int count) {
	stream_count = count;
}

int AudioStreamPlaylist::get_stream_count() {
	return stream_count;
}

void AudioStreamPlaylist::set_bpm(double beats_per_minute) {
	bpm = beats_per_minute;
}

double AudioStreamPlaylist::get_bpm() const {
	return bpm;
}

void AudioStreamPlaylist::set_shuffle(bool p_shuffle) {
	shuffle = p_shuffle;
}

bool AudioStreamPlaylist::get_shuffle() {
	return shuffle;
}

void AudioStreamPlaylist::set_loop(bool p_loop) {
	loop = p_loop;
}

bool AudioStreamPlaylist::get_loop() {
	return loop;
}

void AudioStreamPlaylist::_validate_property(PropertyInfo &property) const {
	String prop = property.name;
	if (prop.begins_with("stream_")) {
		int stream = prop.get_slicec('/', 0).get_slicec('_', 1).to_int();
		if (stream >= stream_count) {
			property.usage = 0;
		}
	}
}

void AudioStreamPlaylist::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_stream_count", "stream_count"), &AudioStreamPlaylist::set_stream_count);
	ClassDB::bind_method(D_METHOD("get_stream_count"), &AudioStreamPlaylist::get_stream_count);

	ClassDB::bind_method(D_METHOD("set_bpm", "bpm"), &AudioStreamPlaylist::set_bpm);
	ClassDB::bind_method(D_METHOD("get_bpm"), &AudioStreamPlaylist::get_bpm);

	ClassDB::bind_method(D_METHOD("set_list_stream", "stream_number", "audio_stream"), &AudioStreamPlaylist::set_list_stream);
	ClassDB::bind_method(D_METHOD("get_list_stream", "stream_number"), &AudioStreamPlaylist::get_list_stream);

	ClassDB::bind_method(D_METHOD("set_stream_beats", "beat_count"), &AudioStreamPlaylist::set_stream_beats);
	ClassDB::bind_method(D_METHOD("get_stream_beats"), &AudioStreamPlaylist::get_stream_beats);

	ClassDB::bind_method(D_METHOD("set_shuffle", "shuffle"), &AudioStreamPlaylist::set_shuffle);
	ClassDB::bind_method(D_METHOD("get_shuffle"), &AudioStreamPlaylist::get_shuffle);

	ClassDB::bind_method(D_METHOD("set_loop", "loop"), &AudioStreamPlaylist::set_loop);
	ClassDB::bind_method(D_METHOD("get_loop"), &AudioStreamPlaylist::get_loop);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "stream_count", PROPERTY_HINT_RANGE, "1," + itos(MAX_STREAMS), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_stream_count", "get_stream_count");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bpm", PROPERTY_HINT_RANGE, "0,400,0.01,or_greater"), "set_bpm", "get_bpm");
	//ADD_PROPERTY(PropertyInfo(Variant::INT, "beat_count", PROPERTY_HINT_RANGE, "0,400"), "set_stream_beats", "get_stream_beats");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "shuffle"), "set_shuffle", "get_shuffle");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "get_loop");

	for (int i = 0; i < MAX_STREAMS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "stream_" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "AudioStream", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_list_stream", "get_list_stream", i);
	}

	BIND_CONSTANT(MAX_STREAMS);
}

//////////////////////
//////////////////////

AudioStreamPlaybackPlaylist::AudioStreamPlaybackPlaylist() {
}

AudioStreamPlaybackPlaylist::~AudioStreamPlaybackPlaylist() {
	playlist->playbacks.erase(this);
}

void AudioStreamPlaybackPlaylist::stop() {
	active = false;
	playlist->reset();
}

void AudioStreamPlaybackPlaylist::start(float p_from_pos) {
	if (playlist->shuffle) {
		for (int i = 0; i < playlist->stream_count; i++) {
			std::swap(playback[i], playback[std::rand() % playlist->stream_count]);
		}
	} else {
		for (int i = 0; i < playlist->stream_count; i++) {
			if (playlist->audio_streams[i].is_valid()) {
				playback[i] = playlist->audio_streams[i]->instantiate_playback();
			} else {
				playback[i].unref();
			}
		}
	}

	current = 0;

	if (playlist->audio_streams[current].is_valid()) {
		fading_samples_total = (fading_time * playlist->sample_rate) / 1000;
		if (playlist->audio_streams[current]->get_bpm() == 0) {
			beat_size = playlist->sample_rate * 60 / playlist->bpm;
		} else {
			beat_size = playlist->sample_rate * 60 / playlist->audio_streams[current]->get_bpm();
		}
		if (playlist->audio_streams[current]->get_beat_count() == 0) {
			beat_amount_remaining = playlist->audio_streams[current]->get_length() * playlist->sample_rate;
		} else {
			beat_amount_remaining = playlist->audio_streams[current]->get_beat_count() * beat_size;
		}

		seek(p_from_pos);
		active = true;
		playback[current]->start();
	} else {
		active = false;
	}
}

void AudioStreamPlaybackPlaylist::seek(float p_time) {
	if (p_time < 0) {
		p_time = 0;
	}
	playlist->set_position(uint64_t(p_time * playlist->sample_rate) << MIX_FRAC_BITS);
}

void AudioStreamPlaybackPlaylist::add_stream_to_buffer(Ref<AudioStreamPlayback> playback, int samples, float p_rate_scale, float initial_volume, float final_volume) {
	playback->mix(aux_buffer, p_rate_scale, samples);
	for (int i = 0; i < samples; i++) {
		float c = float(i) / samples;
		float volume = initial_volume * (1.0 - c) + final_volume * c;
		pcm_buffer[i] += aux_buffer[i] * volume;
	}
}

void AudioStreamPlaybackPlaylist::clear_buffer(int samples) {
	for (int i = 0; i < samples; i++) {
		pcm_buffer[i] = AudioFrame(0.0, 0.0);
	}
}

int AudioStreamPlaybackPlaylist::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (active != true) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0.0, 0.0);
		}
		stop();
		return p_frames;

	} else {
		int mixed_frames = p_frames;
		int dst_offset = 0;

		while (mixed_frames > 0) {
			if (beat_amount_remaining == 0) {
				fading = true;
				if ((current + 1) < playlist->stream_count) {
					current = (current + 1) % playlist->stream_count;
				} else {
					if (playlist->loop) {
						current = 0;
						if (playlist->shuffle) {
							for (int i = 0; i < playlist->stream_count; i++) {
								std::swap(playback[i], playback[std::rand() % playlist->stream_count]);
							}
						}
					} else {
						stop();
					}
				}
				playback[current]->start();
				fading_samples = fading_samples_total;
				if (playlist->audio_streams[current]->get_bpm() == 0) {
					beat_size = playlist->sample_rate * 60 / playlist->bpm;
				} else {
					beat_size = playlist->sample_rate * 60 / playlist->audio_streams[current]->get_bpm();
				}
				if (playlist->audio_streams[current]->get_beat_count() == 0) {
					beat_amount_remaining = playlist->audio_streams[current]->get_length() * beat_size;
				} else {
					beat_amount_remaining = playlist->audio_streams[current]->get_beat_count() * beat_size;
				}
			}

			int to_mix = MIN(MIX_BUFFER_SIZE, MIN(mixed_frames, beat_amount_remaining));
			if (to_mix < 0) {
				to_mix = MIX_BUFFER_SIZE;
			}

			clear_buffer(to_mix);

			if (fading) {
				int to_fade = MIN(fading_samples, to_mix);
				float from_volume = 1.0 - float(fading_samples_total - fading_samples) / fading_samples_total;
				float to_volume = 1.0 - float(fading_samples_total - (fading_samples - to_fade)) / fading_samples_total;
				if (to_volume < 0.0)
					to_volume = 0.0;
				if (playlist->loop && current == 0) {
					add_stream_to_buffer(playback[playlist->stream_count - 1], to_fade, p_rate_scale, from_volume, to_volume);
				} else {
					add_stream_to_buffer(playback[current - 1], to_fade, p_rate_scale, from_volume, to_volume);
				}
				add_stream_to_buffer(playback[current], to_fade, p_rate_scale, (1.0 - from_volume), (1.0 - to_volume));
				fading_samples -= to_fade;
				if (fading_samples == 0) {
					fading = false;
					if (playlist->loop && current == 0) {
						playback[playlist->stream_count - 1]->stop();
					} else {
						playback[current - 1]->stop();
					}
				}
			} else {
				add_stream_to_buffer(playback[current], to_mix, p_rate_scale, 1.0, 1.0);
			}
			for (int i = 0; i < to_mix; i++) {
				p_buffer[i + dst_offset] = pcm_buffer[i];
			}
			dst_offset += to_mix;
			mixed_frames -= to_mix;
			beat_amount_remaining -= to_mix;
		}

		return p_frames;
	}
}

void AudioStreamPlaybackPlaylist::tag_used_streams() {
	if (current >= 0 && current < playlist->stream_count && playback[current].is_valid()) {
		Ref<AudioStream> p = playlist->audio_streams[current];
		if (p.is_valid()) {
			p->tag_used(playback[current]->get_playback_position());
		}
	}

	playlist->tag_used(0);
}

int AudioStreamPlaybackPlaylist::get_loop_count() const {
	return 0;
}

float AudioStreamPlaybackPlaylist::get_playback_position() const {
	return 0.0;
}

bool AudioStreamPlaybackPlaylist::is_playing() const {
	return active;
}

void AudioStreamPlaybackPlaylist::_update_playback_instances() {
	stop();

	for (int i = 0; i < playlist->stream_count; i++) {
		if (playlist->audio_streams[i].is_valid()) {
			playback[i] = playlist->audio_streams[i]->instantiate_playback();
		} else {
			playback[i].unref();
		}
	}
}

void AudioStreamPlaybackPlaylist::_update_bpm_info() {
	for (int i = 0; i < AudioStreamPlaylist::MAX_STREAMS; i++) {
		if (playlist->audio_streams[i]->get_bpm() == 0) {
			bpm_list[i] = playlist->bpm;
		} else {
			bpm_list[i] = playlist->audio_streams[i]->get_bpm();
		}

		if (playlist->audio_streams[i]->get_beat_count() == 0) {
			beats_list[i] = playlist->beat_count;
		} else {
			beats_list[i] = playlist->audio_streams[i]->get_beat_count();
		}
	}
}
