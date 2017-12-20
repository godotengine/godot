/*************************************************************************/
/*  audio_stream.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "audio_stream.h"

//////////////////////////////

void AudioStreamPlaybackResampled::_begin_resample() {

	//clear cubic interpolation history
	internal_buffer[0] = AudioFrame(0.0, 0.0);
	internal_buffer[1] = AudioFrame(0.0, 0.0);
	internal_buffer[2] = AudioFrame(0.0, 0.0);
	internal_buffer[3] = AudioFrame(0.0, 0.0);
	//mix buffer
	_mix_internal(internal_buffer + 4, INTERNAL_BUFFER_LEN);
	mix_offset = 0;
}

void AudioStreamPlaybackResampled::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {

	float target_rate = AudioServer::get_singleton()->get_mix_rate() * p_rate_scale;

	uint64_t mix_increment = uint64_t((get_stream_sampling_rate() / double(target_rate)) * double(FP_LEN));

	for (int i = 0; i < p_frames; i++) {

		uint32_t idx = CUBIC_INTERP_HISTORY + uint32_t(mix_offset >> FP_BITS);
		//standard cubic interpolation (great quality/performance ratio)
		//this used to be moved to a LUT for greater performance, but nowadays CPU speed is generally faster than memory.
		float mu = (mix_offset & FP_MASK) / float(FP_LEN);
		AudioFrame y0 = internal_buffer[idx - 3];
		AudioFrame y1 = internal_buffer[idx - 2];
		AudioFrame y2 = internal_buffer[idx - 1];
		AudioFrame y3 = internal_buffer[idx - 0];

		float mu2 = mu * mu;
		AudioFrame a0 = y3 - y2 - y0 + y1;
		AudioFrame a1 = y0 - y1 - a0;
		AudioFrame a2 = y2 - y0;
		AudioFrame a3 = y1;

		p_buffer[i] = (a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3);

		mix_offset += mix_increment;

		while ((mix_offset >> FP_BITS) >= INTERNAL_BUFFER_LEN) {

			internal_buffer[0] = internal_buffer[INTERNAL_BUFFER_LEN + 0];
			internal_buffer[1] = internal_buffer[INTERNAL_BUFFER_LEN + 1];
			internal_buffer[2] = internal_buffer[INTERNAL_BUFFER_LEN + 2];
			internal_buffer[3] = internal_buffer[INTERNAL_BUFFER_LEN + 3];
			if (is_playing()) {
				_mix_internal(internal_buffer + 4, INTERNAL_BUFFER_LEN);
			} else {
				//fill with silence, not playing
				for (int i = 0; i < INTERNAL_BUFFER_LEN; ++i) {
					internal_buffer[i + 4] = AudioFrame(0, 0);
				}
			}
			mix_offset -= (INTERNAL_BUFFER_LEN << FP_BITS);
		}
	}
}
////////////////////////////////

void AudioStreamRandomPitch::set_audio_stream(const Ref<AudioStream> &p_audio_stream) {

	audio_stream = p_audio_stream;
	if (audio_stream.is_valid()) {
		for (Set<AudioStreamPlaybackRandomPitch *>::Element *E = playbacks.front(); E; E = E->next()) {
			E->get()->playback = audio_stream->instance_playback();
		}
	}
}

Ref<AudioStream> AudioStreamRandomPitch::get_audio_stream() const {

	return audio_stream;
}

void AudioStreamRandomPitch::set_random_pitch(float p_pitch) {

	if (p_pitch < 1)
		p_pitch = 1;
	random_pitch = p_pitch;
}

float AudioStreamRandomPitch::get_random_pitch() const {
	return random_pitch;
}

Ref<AudioStreamPlayback> AudioStreamRandomPitch::instance_playback() {
	Ref<AudioStreamPlaybackRandomPitch> playback;
	playback.instance();
	if (audio_stream.is_valid())
		playback->playback = audio_stream->instance_playback();

	playbacks.insert(playback.ptr());
	playback->random_pitch = Ref<AudioStreamRandomPitch>((AudioStreamRandomPitch *)this);
	return playback;
}

String AudioStreamRandomPitch::get_stream_name() const {

	if (audio_stream.is_valid()) {
		return "Random: " + audio_stream->get_name();
	}
	return "RandomPitch";
}

void AudioStreamRandomPitch::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_audio_stream", "stream"), &AudioStreamRandomPitch::set_audio_stream);
	ClassDB::bind_method(D_METHOD("get_audio_stream"), &AudioStreamRandomPitch::get_audio_stream);

	ClassDB::bind_method(D_METHOD("set_random_pitch", "scale"), &AudioStreamRandomPitch::set_random_pitch);
	ClassDB::bind_method(D_METHOD("get_random_pitch"), &AudioStreamRandomPitch::get_random_pitch);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "audio_stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream"), "set_audio_stream", "get_audio_stream");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "random_pitch", PROPERTY_HINT_RANGE, "1,16,0.01"), "set_random_pitch", "get_random_pitch");
}

AudioStreamRandomPitch::AudioStreamRandomPitch() {
	random_pitch = 1.1;
}

void AudioStreamPlaybackRandomPitch::start(float p_from_pos) {
	playing = playback;
	float range_from = 1.0 / random_pitch->random_pitch;
	float range_to = random_pitch->random_pitch;

	pitch_scale = range_from + Math::randf() * (range_to - range_from);

	if (playing.is_valid()) {
		playing->start(p_from_pos);
	}
}

void AudioStreamPlaybackRandomPitch::stop() {
	if (playing.is_valid()) {
		playing->stop();
		;
	}
}
bool AudioStreamPlaybackRandomPitch::is_playing() const {
	if (playing.is_valid()) {
		return playing->is_playing();
	}

	return false;
}

int AudioStreamPlaybackRandomPitch::get_loop_count() const {
	if (playing.is_valid()) {
		return playing->get_loop_count();
	}

	return 0;
}

float AudioStreamPlaybackRandomPitch::get_playback_position() const {
	if (playing.is_valid()) {
		return playing->get_playback_position();
	}

	return 0;
}
void AudioStreamPlaybackRandomPitch::seek(float p_time) {
	if (playing.is_valid()) {
		playing->seek(p_time);
	}
}

void AudioStreamPlaybackRandomPitch::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (playing.is_valid()) {
		playing->mix(p_buffer, p_rate_scale * pitch_scale, p_frames);
	} else {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0, 0);
		}
	}
}

float AudioStreamPlaybackRandomPitch::get_length() const {
	if (playing.is_valid()) {
		return playing->get_length();
	}

	return 0;
}

AudioStreamPlaybackRandomPitch::~AudioStreamPlaybackRandomPitch() {
	random_pitch->playbacks.erase(this);
}
