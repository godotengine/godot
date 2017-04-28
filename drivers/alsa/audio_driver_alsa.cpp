/*************************************************************************/
/*  audio_driver_alsa.cpp                                                */
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
#include "audio_driver_alsa.h"

#ifdef ALSA_ENABLED

#include "global_config.h"

#include <errno.h>

Error AudioDriverALSA::init() {

	active = false;
	thread_exited = false;
	exit_thread = false;
	pcm_open = false;
	samples_in = NULL;
	samples_out = NULL;

	mix_rate = GLOBAL_DEF("audio/mix_rate", 44100);
	speaker_mode = SPEAKER_MODE_STEREO;
	channels = 2;

	int status;
	snd_pcm_hw_params_t *hwparams;
	snd_pcm_sw_params_t *swparams;

#define CHECK_FAIL(m_cond)                                       \
	if (m_cond) {                                                \
		fprintf(stderr, "ALSA ERR: %s\n", snd_strerror(status)); \
		snd_pcm_close(pcm_handle);                               \
		ERR_FAIL_COND_V(m_cond, ERR_CANT_OPEN);                  \
	}

	//todo, add
	//6 chans - "plug:surround51"
	//4 chans - "plug:surround40";

	status = snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);

	ERR_FAIL_COND_V(status < 0, ERR_CANT_OPEN);

	snd_pcm_hw_params_alloca(&hwparams);

	status = snd_pcm_hw_params_any(pcm_handle, hwparams);
	CHECK_FAIL(status < 0);

	status = snd_pcm_hw_params_set_access(pcm_handle, hwparams, SND_PCM_ACCESS_RW_INTERLEAVED);
	CHECK_FAIL(status < 0);

	//not interested in anything else
	status = snd_pcm_hw_params_set_format(pcm_handle, hwparams, SND_PCM_FORMAT_S16_LE);
	CHECK_FAIL(status < 0);

	//todo: support 4 and 6
	status = snd_pcm_hw_params_set_channels(pcm_handle, hwparams, 2);
	CHECK_FAIL(status < 0);

	status = snd_pcm_hw_params_set_rate_near(pcm_handle, hwparams, &mix_rate, NULL);
	CHECK_FAIL(status < 0);

	int latency = GLOBAL_DEF("audio/output_latency", 25);
	buffer_size = nearest_power_of_2(latency * mix_rate / 1000);

	// set buffer size from project settings
	status = snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hwparams, &buffer_size);
	CHECK_FAIL(status < 0);

	// make period size 1/8
	period_size = buffer_size >> 3;
	status = snd_pcm_hw_params_set_period_size_near(pcm_handle, hwparams, &period_size, NULL);
	CHECK_FAIL(status < 0);

	unsigned int periods = 2;
	status = snd_pcm_hw_params_set_periods_near(pcm_handle, hwparams, &periods, NULL);
	CHECK_FAIL(status < 0);

	status = snd_pcm_hw_params(pcm_handle, hwparams);
	CHECK_FAIL(status < 0);

	//snd_pcm_hw_params_free(&hwparams);

	snd_pcm_sw_params_alloca(&swparams);

	status = snd_pcm_sw_params_current(pcm_handle, swparams);
	CHECK_FAIL(status < 0);

	status = snd_pcm_sw_params_set_avail_min(pcm_handle, swparams, period_size);
	CHECK_FAIL(status < 0);

	status = snd_pcm_sw_params_set_start_threshold(pcm_handle, swparams, 1);
	CHECK_FAIL(status < 0);

	status = snd_pcm_sw_params(pcm_handle, swparams);
	CHECK_FAIL(status < 0);

	samples_in = memnew_arr(int32_t, period_size * channels);
	samples_out = memnew_arr(int16_t, period_size * channels);

	snd_pcm_nonblock(pcm_handle, 0);

	mutex = Mutex::create();
	thread = Thread::create(AudioDriverALSA::thread_func, this);

	return OK;
};

void AudioDriverALSA::thread_func(void *p_udata) {

	AudioDriverALSA *ad = (AudioDriverALSA *)p_udata;

	while (!ad->exit_thread) {
		if (!ad->active) {
			for (unsigned int i = 0; i < ad->period_size * ad->channels; i++) {
				ad->samples_out[i] = 0;
			};
		} else {
			ad->lock();

			ad->audio_server_process(ad->period_size, ad->samples_in);

			ad->unlock();

			for (unsigned int i = 0; i < ad->period_size * ad->channels; i++) {
				ad->samples_out[i] = ad->samples_in[i] >> 16;
			}
		};

		int todo = ad->period_size;
		int total = 0;

		while (todo) {
			if (ad->exit_thread)
				break;
			uint8_t *src = (uint8_t *)ad->samples_out;
			int wrote = snd_pcm_writei(ad->pcm_handle, (void *)(src + (total * ad->channels)), todo);

			if (wrote < 0) {
				if (ad->exit_thread)
					break;

				if (wrote == -EAGAIN) {
					//can't write yet (though this is blocking..)
					usleep(1000);
					continue;
				}
				wrote = snd_pcm_recover(ad->pcm_handle, wrote, 0);
				if (wrote < 0) {
					//absolute fail
					fprintf(stderr, "ALSA failed and can't recover: %s\n", snd_strerror(wrote));
					ad->active = false;
					ad->exit_thread = true;
					break;
				}
				continue;
			};

			total += wrote;
			todo -= wrote;
		};
	};

	ad->thread_exited = true;
};

void AudioDriverALSA::start() {

	active = true;
};

int AudioDriverALSA::get_mix_rate() const {

	return mix_rate;
};

AudioDriver::SpeakerMode AudioDriverALSA::get_speaker_mode() const {

	return speaker_mode;
};

void AudioDriverALSA::lock() {

	if (!thread || !mutex)
		return;
	mutex->lock();
};

void AudioDriverALSA::unlock() {

	if (!thread || !mutex)
		return;
	mutex->unlock();
};

void AudioDriverALSA::finish() {

	if (!thread)
		return;

	exit_thread = true;
	Thread::wait_to_finish(thread);

	if (pcm_open)
		snd_pcm_close(pcm_handle);

	if (samples_in) {
		memdelete_arr(samples_in);
		memdelete_arr(samples_out);
	};

	memdelete(thread);
	if (mutex)
		memdelete(mutex);
	thread = NULL;
};

AudioDriverALSA::AudioDriverALSA() {

	mutex = NULL;
	thread = NULL;
	pcm_handle = NULL;
};

AudioDriverALSA::~AudioDriverALSA(){

};

#endif
