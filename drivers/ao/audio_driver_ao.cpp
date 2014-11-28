/*************************************************************************/
/*  audio_driver_ao.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014 Anton Yabchinskiy.                                 */
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
#include "audio_driver_ao.h"

#ifdef AO_ENABLED

#include "globals.h"
#include "os/os.h"

#include <cstring>

Error AudioDriverAO::init() {

	active = false;
	thread_exited = false;
	exit_thread = false;
	pcm_open = false;
	samples_in = NULL;

	mix_rate = 44100;
	output_format = OUTPUT_STEREO;
	channels = 2;

	ao_sample_format format;

	format.bits = 16;
	format.rate = mix_rate;
	format.channels = channels;
	format.byte_format = AO_FMT_LITTLE;
	format.matrix = "L,R";

	device = ao_open_live(ao_default_driver_id(), &format, NULL);
	ERR_FAIL_COND_V(device == NULL, ERR_CANT_OPEN);

	int latency = GLOBAL_DEF("audio/output_latency", 25);
	buffer_size = nearest_power_of_2( latency * mix_rate / 1000 );

	samples_in = memnew_arr(int32_t, buffer_size * channels);

	mutex = Mutex::create();
	thread = Thread::create(AudioDriverAO::thread_func, this);

	return OK;
};

void AudioDriverAO::thread_func(void* p_udata) {
	AudioDriverAO* ad = (AudioDriverAO*)p_udata;

	// Overwrite samples on conversion
	int16_t* samples_out = reinterpret_cast<int16_t*>(ad->samples_in);
	unsigned int n_samples = ad->buffer_size * ad->channels;
	unsigned int n_bytes = n_samples * sizeof(int16_t);

	while (!ad->exit_thread) {
		if (ad->active) {
			ad->lock();
			ad->audio_server_process(ad->buffer_size, ad->samples_in);
			ad->unlock();

			for (unsigned int i = 0; i < n_samples; i++) {
				samples_out[i] = ad->samples_in[i] >> 16;
			}
		} else {
			memset(samples_out, 0, n_bytes);
		}

		if (ad->exit_thread)
			break;

		if (!ao_play(ad->device, reinterpret_cast<char*>(samples_out), n_bytes)) {
			ERR_PRINT("ao_play() failed");
		}
	};

	ad->thread_exited = true;
};

void AudioDriverAO::start() {
	active = true;
};

int AudioDriverAO::get_mix_rate() const {
	return mix_rate;
};

AudioDriverSW::OutputFormat AudioDriverAO::get_output_format() const {
	return output_format;
};

void AudioDriverAO::lock() {
	if (!thread || !mutex)
		return;
	mutex->lock();
};

void AudioDriverAO::unlock() {
	if (!thread || !mutex)
		return;
	mutex->unlock();
};

void AudioDriverAO::finish() {
	if (!thread)
		return;

	exit_thread = true;
	Thread::wait_to_finish(thread);

	if (samples_in) {
		memdelete_arr(samples_in);
	};

	memdelete(thread);
	if (mutex)
		memdelete(mutex);

	if (device)
		ao_close(device);

	thread = NULL;
};

AudioDriverAO::AudioDriverAO() {
	mutex = NULL;
	thread = NULL;

	ao_initialize();
};

AudioDriverAO::~AudioDriverAO() {
	ao_shutdown();
};

#endif
