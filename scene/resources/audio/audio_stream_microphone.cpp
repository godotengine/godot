/**************************************************************************/
/*  audio_stream_microphone.cpp                                           */
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

#include "audio_stream_microphone.h"

#include "servers/audio/audio_driver.h"
#include "servers/audio/audio_server.h"

Ref<AudioStreamPlayback> AudioStreamMicrophone::instantiate_playback() {
	Ref<AudioStreamPlaybackMicrophone> playback;
	playback.instantiate();

	playbacks.insert(playback.ptr());

	playback->microphone = Ref<AudioStreamMicrophone>((AudioStreamMicrophone *)this);
	playback->active = false;

	return playback;
}

double AudioStreamMicrophone::get_length() const {
	return 0;
}

bool AudioStreamMicrophone::is_monophonic() const {
	return true;
}

int AudioStreamPlaybackMicrophone::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	AudioDriver::get_singleton()->lock();

	Vector<int32_t> buf = AudioDriver::get_singleton()->get_input_buffer();
	unsigned int input_size = AudioDriver::get_singleton()->get_input_size();
	int mix_rate = AudioDriver::get_singleton()->get_input_mix_rate();
	unsigned int playback_delay = MIN(((50 * mix_rate) / 1000) * 2, buf.size() >> 1);
#ifdef DEBUG_ENABLED
	unsigned int input_position = AudioDriver::get_singleton()->get_input_position();
#endif

	int mixed_frames = p_frames;

	if (playback_delay > input_size) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0.0f, 0.0f);
		}
		input_ofs = 0;
	} else {
		for (int i = 0; i < p_frames; i++) {
			if (input_size > input_ofs && (int)input_ofs < buf.size()) {
				float l = (buf[input_ofs++] >> 16) / 32768.f;
				if ((int)input_ofs >= buf.size()) {
					input_ofs = 0;
				}
				float r = (buf[input_ofs++] >> 16) / 32768.f;
				if ((int)input_ofs >= buf.size()) {
					input_ofs = 0;
				}

				p_buffer[i] = AudioFrame(l, r);
			} else {
				p_buffer[i] = AudioFrame(0.0f, 0.0f);
			}
		}
	}

#ifdef DEBUG_ENABLED
	if (input_ofs > input_position && (int)(input_ofs - input_position) < (p_frames * 2)) {
		print_verbose(String(get_class_name()) + " buffer underrun: input_position=" + itos(input_position) + " input_ofs=" + itos(input_ofs) + " input_size=" + itos(input_size));
	}
#endif

	AudioDriver::get_singleton()->unlock();

	return mixed_frames;
}

int AudioStreamPlaybackMicrophone::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	return AudioStreamPlaybackResampled::mix(p_buffer, p_rate_scale, p_frames);
}

float AudioStreamPlaybackMicrophone::get_stream_sampling_rate() {
	return AudioDriver::get_singleton()->get_input_mix_rate();
}

void AudioStreamPlaybackMicrophone::start(double p_from_pos) {
	if (active) {
		return;
	}

	input_ofs = 0;

	if (AudioServer::get_singleton()->set_input_device_active(true) == OK) {
		active = true;
		begin_resample();
	}
}

void AudioStreamPlaybackMicrophone::stop() {
	if (active) {
		AudioServer::get_singleton()->set_input_device_active(false);
		active = false;
	}
}

bool AudioStreamPlaybackMicrophone::is_playing() const {
	return active;
}

int AudioStreamPlaybackMicrophone::get_loop_count() const {
	return 0;
}

double AudioStreamPlaybackMicrophone::get_playback_position() const {
	return 0;
}

void AudioStreamPlaybackMicrophone::seek(double p_time) {
	// Can't seek a microphone input
}

void AudioStreamPlaybackMicrophone::tag_used_streams() {
	microphone->tag_used(0);
}

AudioStreamPlaybackMicrophone::~AudioStreamPlaybackMicrophone() {
	microphone->playbacks.erase(this);
	stop();
}

AudioStreamPlaybackMicrophone::AudioStreamPlaybackMicrophone() {
}
