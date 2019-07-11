#include "audio_stream_transitioner.h"
#include "core/math/math_funcs.h"
#include "core/print_string.h"
#include <iostream>

AudioStreamTransitioner::AudioStreamTransitioner() {
	bpm = 120;
	transition_count = 1;
	sample_rate = 44100;
	stereo = true;
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
	return "Playlist";
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

void AudioStreamTransitioner::set_start_clip(Ref<AudioStream> s_clip) {
	ERR_FAIL_COND(s_clip == this);
	

	AudioServer::get_singleton()->lock();
	starting_stream = s_clip;
	for (Set<AudioStreamPlaybackTransitioner *>::Element *E = playbacks.front(); E; E = E->next()) {

		E->get()->_update_playback_instances();
	}
	AudioServer::get_singleton()->unlock();
}

Ref<AudioStream> AudioStreamTransitioner::get_start_clip() {
	return starting_stream;
}

void AudioStreamTransitioner::set_next_clip(int t_number, Ref<AudioStream> next_clip) {
	ERR_FAIL_COND(next_clip == this);

	AudioServer::get_singleton()->lock();
	transitions[t_number].to_stream = next_clip;
	for (Set<AudioStreamPlaybackTransitioner *>::Element *E = playbacks.front(); E; E = E->next()) {

		E->get()->_update_playback_instances();
	}
	AudioServer::get_singleton()->unlock();
}

Ref<AudioStream> AudioStreamTransitioner::get_next_clip(int t_number) {
	return transitions[t_number].to_stream;
}

void AudioStreamTransitioner::set_active_transition(int t_number) {
	for (int i; i < MAX_TRANSITIONS; i++) {
		transitions[i].transition_active = false;
	}

	transitions[t_number].transition_active = true;
}

