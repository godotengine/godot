#include "audio_stream_playback_playlist.h"

#include "core/math/math_funcs.h"
#include "core/print_string.h"

AudioStreamPlaybackPlaylist::AudioStreamPlaybackPlaylist() :
	active(false) {
	AudioServer::get_singleton()->lock();
	pcm_buffer = AudioServer::get_singleton()->audio_data_alloc(PCM_BUFFER_SIZE);
	zeromem(pcm_buffer, PCM_BUFFER_SIZE);
	AudioServer::get_singleton()->unlock();
}

AudioStreamPlaybackPlaylist::~AudioStreamPlaybackPlaylist() {
	if (pcm_buffer) {
		AudioServer::get_singleton()->audio_data_free(pcm_buffer);
		pcm_buffer = NULL;
	}
}

void AudioStreamPlaybackPlaylist::stop() {
	active = false;
	instance->reset;
}

void AudioStreamPlaybackPlaylist::start(float p_from_pos) {
	seek(p_from_pos);
	active = true;
}

void AudioStreamPlaybackPlaylist::seek(float p_time) {
	float max = get_length();
	if (p_time < 0) {
		p_time = 0;
	}
	instance->set_position(uint64_t(p_time * instance->mix_rate) << MIX_FRAC_BITS);
}

void AudioStreamPlaybackPlaylist::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	ERR_FAIL_COND(!active);
	if (!active) {
		return;
	}
	zeromem(pcm_buffer, PCM_BUFFER_SIZE);
	int16_t *buf = (int16_t *)pcm_buffer;
	instance->; //function from audiostreamplaylist which fills the buffer 

	for (int i = 0; i < p_frames; i++) {
		float sample = float(buf[i]) / 32767.0;
		p_buffer[i] = AudioFrame(sample, sample);
	}
}

