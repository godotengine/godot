/*************************************************************************/
/*  audio_driver_pulseaudio.cpp                                          */
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
#include "audio_driver_pulseaudio.h"

#ifdef PULSEAUDIO_ENABLED

#include <pulse/error.h>

#include "global_config.h"

Error AudioDriverPulseAudio::init() {

	active = false;
	thread_exited = false;
	exit_thread = false;
	pcm_open = false;
	samples_in = NULL;
	samples_out = NULL;

	mix_rate = GLOBAL_DEF("audio/mix_rate", 44100);
	speaker_mode = SPEAKER_MODE_STEREO;
	channels = 2;

	pa_sample_spec spec;
	spec.format = PA_SAMPLE_S16LE;
	spec.channels = channels;
	spec.rate = mix_rate;

	int latency = GLOBAL_DEF("audio/output_latency", 25);
	buffer_size = nearest_power_of_2(latency * mix_rate / 1000);

	pa_buffer_attr attr;
	// set to appropriate buffer size from global settings
	attr.tlength = buffer_size;
	// set them to be automatically chosen
	attr.prebuf = (uint32_t)-1;
	attr.maxlength = (uint32_t)-1;
	attr.minreq = (uint32_t)-1;

	int error_code;
	pulse = pa_simple_new(NULL, // default server
			"Godot", // application name
			PA_STREAM_PLAYBACK,
			NULL, // default device
			"Sound", // stream description
			&spec,
			NULL, // use default channel map
			&attr, // use buffering attributes from above
			&error_code);

	if (pulse == NULL) {
		fprintf(stderr, "PulseAudio ERR: %s\n", pa_strerror(error_code));
		ERR_FAIL_COND_V(pulse == NULL, ERR_CANT_OPEN);
	}

	samples_in = memnew_arr(int32_t, buffer_size * channels);
	samples_out = memnew_arr(int16_t, buffer_size * channels);

	mutex = Mutex::create();
	thread = Thread::create(AudioDriverPulseAudio::thread_func, this);

	return OK;
}

float AudioDriverPulseAudio::get_latency() {

	if (latency == 0) { //only do this once since it's approximate anyway
		int error_code;
		pa_usec_t palat = pa_simple_get_latency(pulse, &error_code);
		latency = double(palat) / 1000000.0;
	}

	return latency;
}

void AudioDriverPulseAudio::thread_func(void *p_udata) {

	print_line("thread");
	AudioDriverPulseAudio *ad = (AudioDriverPulseAudio *)p_udata;

	while (!ad->exit_thread) {
		if (!ad->active) {
			for (unsigned int i = 0; i < ad->buffer_size * ad->channels; i++) {
				ad->samples_out[i] = 0;
			}

		} else {
			ad->lock();

			ad->audio_server_process(ad->buffer_size, ad->samples_in);

			ad->unlock();

			for (unsigned int i = 0; i < ad->buffer_size * ad->channels; i++) {
				ad->samples_out[i] = ad->samples_in[i] >> 16;
			}
		}

		// pa_simple_write always consumes the entire buffer

		int error_code;
		int byte_size = ad->buffer_size * sizeof(int16_t) * ad->channels;
		if (pa_simple_write(ad->pulse, ad->samples_out, byte_size, &error_code) < 0) {
			// can't recover here
			fprintf(stderr, "PulseAudio failed and can't recover: %s\n", pa_strerror(error_code));
			ad->active = false;
			ad->exit_thread = true;
			break;
		}
	}

	ad->thread_exited = true;
}

void AudioDriverPulseAudio::start() {

	active = true;
}

int AudioDriverPulseAudio::get_mix_rate() const {

	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverPulseAudio::get_speaker_mode() const {

	return speaker_mode;
}

void AudioDriverPulseAudio::lock() {

	if (!thread || !mutex)
		return;
	mutex->lock();
}

void AudioDriverPulseAudio::unlock() {

	if (!thread || !mutex)
		return;
	mutex->unlock();
}

void AudioDriverPulseAudio::finish() {

	if (!thread)
		return;

	exit_thread = true;
	Thread::wait_to_finish(thread);

	if (pulse)
		pa_simple_free(pulse);

	if (samples_in) {
		memdelete_arr(samples_in);
		memdelete_arr(samples_out);
	};

	memdelete(thread);
	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}

	thread = NULL;
}

AudioDriverPulseAudio::AudioDriverPulseAudio() {

	mutex = NULL;
	thread = NULL;
	pulse = NULL;
	latency = 0;
}

AudioDriverPulseAudio::~AudioDriverPulseAudio() {
}

#endif
