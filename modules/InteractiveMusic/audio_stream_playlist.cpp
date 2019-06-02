#include "audio_stream_playlist.h"
#include "core/math/math_funcs.h"
#include "core/print_string.h"


AudioStreamPlaylist::AudioStreamPlaylist() {

}

Ref<AudioStreamPlayback> AudioStreamPlaylist::instance_playback() {
	Ref<AudioStreamPlaybackPlaylist> playlist;
	playlist.instance();
	playlist->instance = Ref<AudioStreamPlaylist>(this);
	return playlist;
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

void AudioStreamPlaylist::set_stream_count(int count) {
	stream_count = count;
}

void AudioStreamPlaylist::set_bpm(int beats) {
	bpm = beats;
}

void AudioStreamPlaylist::set_order(OrderMode p_order) {
	p_order = order_mode;
}

void AudioStreamPlaylist::play(Vector<Ref<AudioStream> > audio_streams, int stream_count) {
	for (int i = 0; i < stream_count; i++) {
		//do something with audio_streams, looping through them and writing to a buffer?
		//


	}
}

void AudioStreamPlaylist::_validate_property(PropertyInfo &property) const {
	String prop = property.name;
	if (prop.begins_with("audio_")) {
		int stream = prop.get_slicec('/', 0).get_slicec('_', 1).to_int();
		if (stream >= stream_count) {
			property.usage = 0;
		}
	}
}


void AudioStreamPlaylist::_bind_methods() {
	
}






//////////////////////
//////////////////////


AudioStreamPlaybackPlaylist::AudioStreamPlaybackPlaylist() :
		active(false) {
	AudioServer::get_singleton()->lock();
	
	zeromem(pcm_buffer, buffer_size);
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
	instance->set_position(uint64_t(p_time * instance->sample_rate) << MIX_FRAC_BITS);
}

void AudioStreamPlaybackPlaylist::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	ERR_FAIL_COND(!active);
	if (!active) {
		return;
	}
	zeromem(pcm_buffer,buffer_size);
	int16_t *buf = (int16_t *)pcm_buffer;
	instance->play; //function from audiostreamplaylist which fills the buffer

	for (int i = 0; i < p_frames; i++) {
		float sample = float(buf[i]) / 32767.0;
		p_buffer[i] = AudioFrame(sample, sample);
	}
}
