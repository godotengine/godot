#include "audio_stream_transitioner.h"
#include "core/math/math_funcs.h"
#include "core/print_string.h"
#include <iostream>

AudioStreamTransitioner::AudioStreamTransitioner() {
	bpm = 120;
	transition_count = 1;
	clip_count = 1;
	sample_rate = 44100;
	stereo = true;
	
	active_clip_number = 0;
	for (int i = 0; i < transition_count; i++) {
		transitions[i].t_active = false;
	}
}

Ref<AudioStreamPlayback> AudioStreamTransitioner::instance_playback() {
	Ref<AudioStreamPlaybackTransitioner> playback_transitioner;
	playback_transitioner.instance();
	playback_transitioner->transitioner = Ref<AudioStreamTransitioner>(this);
	playback_transitioner->_update_playback_instances();
	playbacks.insert(playback_transitioner.operator->());
	return playback_transitioner;
}

String AudioStreamTransitioner::get_stream_name() const {
	return "Transitioner";
}

void AudioStreamTransitioner::reset() {
	set_position(0);
}

void AudioStreamTransitioner::set_position(uint64_t p) {
	pos = p;
}

void AudioStreamTransitioner::set_bpm(int beats_per_minute) {
	bpm = beats_per_minute;
}

int AudioStreamTransitioner::get_bpm() {
	return bpm;
}

void AudioStreamTransitioner::set_transition_count(int t_count) {
	transition_count = t_count;
}

int AudioStreamTransitioner::get_transition_count() {
	return transition_count;
}

void AudioStreamTransitioner::set_clip_count(int p_clip_count) {
	clip_count = p_clip_count;
}

int AudioStreamTransitioner::get_clip_count() {
	return clip_count;
}

void AudioStreamTransitioner::set_transition_fade_in(int t_number, int f_in) {
	transitions[t_number].fade_in_beats = f_in;
}

int AudioStreamTransitioner::get_transition_fade_in(int t_number) {
	return transitions[t_number].fade_in_beats;
}

void AudioStreamTransitioner::set_transition_fade_out(int t_number, int f_out) {
	transitions[t_number].fade_out_beats = f_out;
}

int AudioStreamTransitioner::get_transition_fade_out(int t_number) {
	return transitions[t_number].fade_out_beats;
}

void AudioStreamTransitioner::set_list_clip(int clip_number, Ref<AudioStream> p_clip) {
	ERR_FAIL_COND(p_clip == this);
	ERR_FAIL_INDEX(clip_number, MAX_STREAMS);

	AudioServer::get_singleton()->lock();
	clips[clip_number] = p_clip;
	for (Set<AudioStreamPlaybackTransitioner *>::Element *E = playbacks.front(); E; E = E->next()) {

		E->get()->_update_playback_instances();
	}
	AudioServer::get_singleton()->unlock();
}

Ref<AudioStream> AudioStreamTransitioner::get_list_clip(int clip_number) {
	ERR_FAIL_INDEX_V(clip_number, MAX_STREAMS, Ref<AudioStream>());

	return clips[clip_number];
}

void AudioStreamTransitioner::set_active_transition(int transition_number, bool trigger) {
	for (int i = 0; i < transition_count; i++) {
		transitions[i].t_active = false;
		_change_notify();
	}
	transitions[transition_number].t_active = trigger;
	active_transition = transitions[transition_number];
	
}

bool AudioStreamTransitioner::get_transition_state(int transition_number) {
	return transitions[transition_number].t_active;
}

void AudioStreamTransitioner::set_active_clip_number(int clip_number) {
	fading_clip_number = active_clip_number;
	active_clip_number = clip_number;
}

int AudioStreamTransitioner::get_active_clip_number() {
	return active_clip_number;
}

void AudioStreamTransitioner::go_to_clip(int clip_number, int transition_number) {
	active_clip_number = clip_number;
	set_active_transition(transition_number, true);
	active_transition.t_active = true;
}

void AudioStreamTransitioner::_validate_property(PropertyInfo &property) const {
	String prop = property.name;
	if (prop.begins_with("clip_")) {
		int clip = prop.get_slicec('_', 1).to_int();
		if (clip >= clip_count) {
			property.usage = 0;
		}
	}
	if (prop.begins_with("transition_")) {
		int transition = prop.get_slicec('/', 0).get_slicec('_', 1).to_int();
		if (transition >= transition_count) {
			property.usage = 0;
		}
	}
}

void AudioStreamTransitioner::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_bpm", "bpm"), &AudioStreamTransitioner::set_bpm);
	ClassDB::bind_method(D_METHOD("get_bpm"), &AudioStreamTransitioner::get_bpm);

	ClassDB::bind_method(D_METHOD("set_transition_count", "transition_count"), &AudioStreamTransitioner::set_transition_count);
	ClassDB::bind_method(D_METHOD("get_transition_count"), &AudioStreamTransitioner::get_transition_count);

	ClassDB::bind_method(D_METHOD("set_clip_count", "clip_count"), &AudioStreamTransitioner::set_clip_count);
	ClassDB::bind_method(D_METHOD("get_clip_count"), &AudioStreamTransitioner::get_clip_count);

	ClassDB::bind_method(D_METHOD("set_transition_fade_in", "transition_number", "fade_in_beats"), &AudioStreamTransitioner::set_transition_fade_in);
	ClassDB::bind_method(D_METHOD("get_transition_fade_in", "transition_number"), &AudioStreamTransitioner::get_transition_fade_in);

	ClassDB::bind_method(D_METHOD("set_transition_fade_out", "transition_number", "fade_out_beats"), &AudioStreamTransitioner::set_transition_fade_out);
	ClassDB::bind_method(D_METHOD("get_transition_fade_out", "transition_number"), &AudioStreamTransitioner::get_transition_fade_out);

	ClassDB::bind_method(D_METHOD("set_active_transition", "transition_number", "trigger"), &AudioStreamTransitioner::set_active_transition);
	ClassDB::bind_method(D_METHOD("get_transition_state", "transition_number"), &AudioStreamTransitioner::get_transition_state);

	ClassDB::bind_method(D_METHOD("set_active_clip_number", "clip_number"), &AudioStreamTransitioner::set_active_clip_number);
	ClassDB::bind_method(D_METHOD("get_active_clip_number"), &AudioStreamTransitioner::get_active_clip_number);

	ClassDB::bind_method(D_METHOD("set_list_clip", "clip_number", "clip"), &AudioStreamTransitioner::set_list_clip);
	ClassDB::bind_method(D_METHOD("get_list_clip", "clip_number"), &AudioStreamTransitioner::get_list_clip);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "clip_count", PROPERTY_HINT_RANGE, "1," + itos(MAX_STREAMS), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_clip_count", "get_clip_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "transition_count", PROPERTY_HINT_RANGE, "1," + itos(MAX_TRANSITIONS), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_transition_count", "get_transition_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bpm", PROPERTY_HINT_RANGE, "0,400"), "set_bpm", "get_bpm");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "active_clip", PROPERTY_HINT_RANGE, "0,64"), "set_active_clip_number", "get_active_clip_number");

	for (int i = 0; i < MAX_STREAMS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "clip_" + itos(i), PROPERTY_HINT_RESOURCE_TYPE, "AudioStream", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_list_clip", "get_list_clip", i);
	}

	for (int i = 0; i < MAX_TRANSITIONS; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/fade_in_beats", PROPERTY_HINT_RANGE, "0,40", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_transition_fade_in", "get_transition_fade_in", i);
		ADD_PROPERTYI(PropertyInfo(Variant::INT, "transition_" + itos(i) + "/fade_out_beats", PROPERTY_HINT_RANGE, "0,40", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_transition_fade_out", "get_transition_fade_out", i);
		ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "transition_" + itos(i) + "/active"), "set_active_transition", "get_transition_state", i);
	}

	BIND_CONSTANT(MAX_STREAMS);
	BIND_CONSTANT(MAX_TRANSITIONS);
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
AudioStreamPlaybackTransitioner::AudioStreamPlaybackTransitioner() :
		active(false) {
	current = 0;
	fading = false;
}

AudioStreamPlaybackTransitioner::~AudioStreamPlaybackTransitioner() {
	transitioner->playbacks.erase(this);
}

void AudioStreamPlaybackTransitioner::stop() {
	active = false;
	transitioner->reset();
}

void AudioStreamPlaybackTransitioner::start(float p_from_pos) {
	current = transitioner->active_clip_number;
	beat_size = transitioner->sample_rate * 60 / transitioner->bpm;
	if (transitioner->clips[current].is_valid()) {
		seek(p_from_pos);
		active = true;
		playbacks[current]->start();
	} else {
		active = false;
	}
}

void AudioStreamPlaybackTransitioner::seek(float p_time) {
	float max = get_length();
	if (p_time < 0) {
		p_time = 0;
	}
	transitioner->set_position(uint64_t(p_time * transitioner->sample_rate) << MIX_FRAC_BITS);
}

void AudioStreamPlaybackTransitioner::add_stream_to_buffer(Ref<AudioStreamPlayback> playback, int samples, float p_rate_scale, float initial_volume, float final_volume) {
	playback->mix(aux_buffer, p_rate_scale, samples);
	for (int i = 0; i < samples; i++) {
		float c = float(i) / samples;
		float volume = initial_volume * (1.0 - c) + final_volume * c;
		pcm_buffer[i] += aux_buffer[i] * volume;
	}
}

void AudioStreamPlaybackTransitioner::clear_buffer(int samples) {
	for (int i = 0; i < samples; i++) {
		pcm_buffer[i] = AudioFrame(0.0, 0.0);
	}
}

void AudioStreamPlaybackTransitioner::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (active != true) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0.0, 0.0);
		}
		stop();
		return;

	} else {
		int dst_offset = 0;
		
		while (p_frames > 0) {
			if (transitioner->active_transition.t_active) {
				current = transitioner->active_clip_number;
				previous = transitioner->fading_clip_number;
				fading = true;
				fade_out_samples_total = transitioner->active_transition.fade_out_beats * beat_size;//beat size is not set anywhere
				fade_in_samples_total = transitioner->active_transition.fade_in_beats * beat_size;
				transition_samples_total = MAX(fade_in_samples_total, fade_out_samples_total);
				playbacks[current]->start();
				transition_samples = transition_samples_total;
				fade_in_samples = fade_in_samples_total;
				fade_out_samples = fade_out_samples_total;
				transitioner->active_transition.t_active = false;
			}

			int to_mix = MIN(MIX_BUFFER_SIZE, p_frames);
			clear_buffer(to_mix);

			if (fading) {
				if (transition_samples < 0) {
					fading = false;
				} else {
					int to_fade_out = MIN(to_mix, fade_out_samples);
					int to_fade_in = MIN(to_mix, fade_in_samples);
					float fade_out_start_volume = 1.0 - float(fade_out_samples_total - fade_out_samples) / fade_out_samples_total;
					float fade_out_end_volume = 1.0 - float(fade_out_samples_total - (fade_out_samples - to_fade_out)) / fade_out_samples_total;
					float fade_in_start_volume = 1.0 - float(fade_in_samples) / fade_in_samples_total;
					float fade_in_end_volume = 1.0 - float(fade_in_samples - to_fade_in) / fade_in_samples_total;
					if (fade_out_samples > 0) {
						add_stream_to_buffer(playbacks[previous], to_fade_out, p_rate_scale, fade_out_start_volume, fade_out_end_volume);
					} else {
						playbacks[previous]->stop();
					}
					if (fade_in_samples > 0) {
						add_stream_to_buffer(playbacks[current], to_fade_in, p_rate_scale, fade_in_start_volume, fade_in_end_volume);
					} else {
						add_stream_to_buffer(playbacks[current], to_mix, p_rate_scale, 1.0, 1.0);
					}
					transition_samples -= to_mix;
					fade_out_samples -= to_fade_out;
					fade_in_samples -= to_fade_in;
				}
			} else {
				add_stream_to_buffer(playbacks[current], to_mix, p_rate_scale, 1.0, 1.0);
			}

			for (int i = 0; i < to_mix; i++) {
				p_buffer[i + dst_offset] = pcm_buffer[i];
			}
			dst_offset += to_mix;
			p_frames -= to_mix;
		}
	}
}

int AudioStreamPlaybackTransitioner::get_loop_count() const {
	return 0;
}

float AudioStreamPlaybackTransitioner::get_playback_position() const {
	return 0.0;
}

float AudioStreamPlaybackTransitioner::get_length() const {
	return 0.0;
}

bool AudioStreamPlaybackTransitioner::is_playing() const {
	return active;
}

void AudioStreamPlaybackTransitioner::_update_playback_instances() {
	stop();

	for (int i = 0; i < transitioner->clip_count; i++) {

		if (transitioner->clips[i].is_valid()) {
			playbacks[i] = transitioner->clips[i]->instance_playback(); //unhandled exception here
		} else {
			playbacks[i].unref();
		}
	}
}
