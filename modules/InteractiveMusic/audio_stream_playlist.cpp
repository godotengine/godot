#include "audio_stream_playlist.h"
#include "core/math/math_funcs.h"
#include "core/print_string.h"
#include <iostream>

AudioStreamPlaylist::AudioStreamPlaylist() {
	bpm = 120;
	stream_count = 1;
	sample_rate = 44100;
	stereo = true;

	beat_count = 20;
}

Ref<AudioStreamPlayback> AudioStreamPlaylist::instance_playback() {
	Ref<AudioStreamPlaybackPlaylist> playback_playlist;
	playback_playlist.instance();
	playback_playlist->playlist = Ref<AudioStreamPlaylist>(this);
	playback_playlist->_update_playback_instances();
	playbacks.insert(playback_playlist.operator->());
	return playback_playlist;
}

String AudioStreamPlaylist::get_stream_name() const {
	return "Playlist";
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
	for (Set<AudioStreamPlaybackPlaylist *>::Element *E = playbacks.front(); E; E = E->next()) {

		E->get()->_update_playback_instances();
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

void AudioStreamPlaylist::set_bpm(int beats_per_minute) {
	bpm = beats_per_minute;
}

int AudioStreamPlaylist::get_bpm() {
	return bpm;
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

	ADD_PROPERTY(PropertyInfo(Variant::INT, "stream_count", PROPERTY_HINT_RANGE, "1," + itos(MAX_STREAMS), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_stream_count", "get_stream_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bpm", PROPERTY_HINT_RANGE, "0,400"), "set_bpm", "get_bpm");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "beat_count", PROPERTY_HINT_RANGE, "0,400"), "set_stream_beats", "get_stream_beats");

	for (int i = 0; i < MAX_STREAMS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "stream_" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "AudioStream", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_list_stream", "get_list_stream", i);
	}

	BIND_CONSTANT(MAX_STREAMS);
}

//////////////////////
//////////////////////

AudioStreamPlaybackPlaylist::AudioStreamPlaybackPlaylist() :
		active(false) {
	current = 0;
	fading = false;
	fading_time = 1;
}

AudioStreamPlaybackPlaylist::~AudioStreamPlaybackPlaylist() {
	playlist->playbacks.erase(this);
}

void AudioStreamPlaybackPlaylist::stop() {
	active = false;
	playlist->reset();
}

void AudioStreamPlaybackPlaylist::start(float p_from_pos) {
	current = 0;
	fading_samples_total = fading_time * playlist->sample_rate;
	if (playlist->audio_streams[current]->get_bpm() == 0) {
		beat_size = playlist->sample_rate * 60 / playlist->bpm;
	} else {
		int bpm_test = playlist->audio_streams[current]->get_bpm();
		std::clog << "BPM= " << bpm_test << std::endl;
		beat_size = playlist->sample_rate * 60 / playlist->audio_streams[current]->get_bpm();
	}
	if (playlist->audio_streams[current]->get_beat_count() == 0) {
		beat_amount_remaining = playlist->audio_streams[current]->get_length() * beat_size;
		std::clog << "beats_amount_remaining= " << beat_amount_remaining << std::endl;

	} else {
		int beats_test = playlist->audio_streams[current]->get_beat_count();

		std::clog << "beats= " << beats_test << std::endl;

		beat_amount_remaining = playlist->audio_streams[current]->get_beat_count() * beat_size;
	}
	if (playlist->audio_streams[current].is_valid()) {
		seek(p_from_pos);
		active = true;
		playback[current]->start();
	} else {
		active = false;
	}
}

void AudioStreamPlaybackPlaylist::seek(float p_time) {
	float max = get_length();
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

void AudioStreamPlaybackPlaylist::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (active != true) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0.0, 0.0);
		}
		stop();
		return;
		
	} else {
		int dst_offset = 0;
		int fading_samples = 0;

		while (p_frames > 0) {

			if (beat_amount_remaining == 0) {
				fading = true;
				current = (current + 1) % playlist->stream_count;

				playback[current]->start();
				fading_samples = fading_samples_total;
				beat_size = playlist->sample_rate * 60 / bpm_list[current];
				if (playlist->audio_streams[current]->get_beat_count() == 0) {
					beat_amount_remaining = playlist->audio_streams[current]->get_length() * beat_size;
				} else {
					beat_amount_remaining = playlist->audio_streams[current]->get_beat_count() * beat_size;
				} //std::clog << "beat_amount_remaining = " << beats_list[current] << " * " << beat_size << std::endl;
				//std::clog << "beat_amount_remaining = " << beat_amount_remaining << std::endl;
			}

			int to_mix = MIN(MIX_BUFFER_SIZE, MIN(p_frames, beat_amount_remaining));
			if (to_mix < 0) {
				to_mix = MIX_BUFFER_SIZE;
			}

			clear_buffer(to_mix);

			if (fading) {
				int to_fade = MIN(fading_samples, to_mix);
				float from_volume = 1.0 - float(fading_samples) / fading_samples_total;
				float to_volume = 1.0 - float(fading_samples + to_fade) / fading_samples_total;
				add_stream_to_buffer(playback[current - 1], to_fade, p_rate_scale, from_volume, to_volume);
				fading_samples -= to_fade;
				if (fading_samples == 0) {
					fading = false;
					playback[current - 1]->stop();
				}
			}

			add_stream_to_buffer(playback[current], to_mix, p_rate_scale, 1.0, 1.0);

			for (int i = 0; i < to_mix; i++) {
				p_buffer[i + dst_offset] = pcm_buffer[i];
			}
			dst_offset += to_mix;
			p_frames -= to_mix;
			beat_amount_remaining -= to_mix;
			}
	}
}

int AudioStreamPlaybackPlaylist::get_loop_count() const {
	return 0;
}

float AudioStreamPlaybackPlaylist::get_playback_position() const {
	return 0.0;
}

float AudioStreamPlaybackPlaylist::get_length() const {
	return 0.0;
}

bool AudioStreamPlaybackPlaylist::is_playing() const {
	return active;
}

void AudioStreamPlaybackPlaylist::_update_playback_instances() {
	stop();

	for (int i = 0; i < playlist->stream_count; i++) {

		if (playlist->audio_streams[i].is_valid()) {
			playback[i] = playlist->audio_streams[i]->instance_playback();
		} else {
			playback[i].unref();
		}
	}
}

void AudioStreamPlaybackPlaylist::_update_bpm_info() {
	for (int i = 0; i < AudioStreamPlaylist::MAX_STREAMS; i++) {
		if (playlist->audio_streams[i]->get_bpm()==0) {
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
