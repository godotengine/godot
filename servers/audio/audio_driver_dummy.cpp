/*************************************************************************/
/*  audio_driver_dummy.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_driver_dummy.h"

#include "core/os/os.h"
#include "core/project_settings.h"

Error AudioDriverDummy::init() {
	active = false;
	thread_exited = false;
	exit_thread = false;
	samples_in = nullptr;

	mix_rate = GLOBAL_GET("audio/mix_rate");
	speaker_mode = SPEAKER_MODE_STEREO;
	channels = 2;

	int latency = GLOBAL_GET("audio/output_latency");
	buffer_frames = closest_power_of_2(latency * mix_rate / 1000);

	samples_in = memnew_arr(int32_t, buffer_frames * channels);

	thread = Thread::create(AudioDriverDummy::thread_func, this);

	return OK;
};

void AudioDriverDummy::thread_func(void *p_udata) {
	AudioDriverDummy *ad = (AudioDriverDummy *)p_udata;

	uint64_t usdelay = (ad->buffer_frames / float(ad->mix_rate)) * 1000000;

	while (!ad->exit_thread) {
		if (ad->active) {
			ad->lock();

			ad->audio_server_process(ad->buffer_frames, ad->samples_in);

			ad->unlock();
		};

		OS::get_singleton()->delay_usec(usdelay);
	};

	ad->thread_exited = true;
};

void AudioDriverDummy::start() {
	active = true;
};

int AudioDriverDummy::get_mix_rate() const {
	return mix_rate;
};

AudioDriver::SpeakerMode AudioDriverDummy::get_speaker_mode() const {
	return speaker_mode;
};

void AudioDriverDummy::lock() {
	if (!thread) {
		return;
	}
	mutex.lock();
};

void AudioDriverDummy::unlock() {
	if (!thread) {
		return;
	}
	mutex.unlock();
};

void AudioDriverDummy::finish() {
	if (!thread) {
		return;
	}

	exit_thread = true;
	Thread::wait_to_finish(thread);

	if (samples_in) {
		memdelete_arr(samples_in);
	};

	memdelete(thread);
	thread = nullptr;
};
