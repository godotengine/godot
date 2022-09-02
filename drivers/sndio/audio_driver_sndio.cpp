/*************************************************************************/
/*  audio_driver_sndio.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_driver_sndio.h"

#ifdef SNDIO_ENABLED

#include "core/config/project_settings.h"
#include "core/os/os.h"

Error AudioDriverSndio::init() {
	active = false;
	thread_exited = false;
	exit_thread = false;
	speaker_mode = SPEAKER_MODE_STEREO;

	handle = sio_open(SIO_DEVANY, SIO_PLAY, 0);
	ERR_FAIL_COND_V(handle == NULL, ERR_CANT_OPEN);

	struct sio_par par;
	sio_initpar(&par);

	par.bits = 32;
	par.bps = 4;
	par.rate = GLOBAL_GET("audio/driver/mix_rate");
	par.appbufsz = 50 * par.rate / 1000;

	if (!sio_setpar(handle, &par)) {
		return ERR_CANT_OPEN;
	}

	if (!sio_getpar(handle, &par)) {
		return ERR_CANT_OPEN;
	}

	if (par.bits != 32 || par.bps != 4 || par.le != SIO_LE_NATIVE) {
		return ERR_CANT_OPEN;
	}

	if (!sio_start(handle)) {
		return ERR_CANT_OPEN;
	}

	mix_rate = par.rate;
	channels = par.pchan;
	period_size = par.appbufsz;

	samples.resize(period_size * channels);

	thread.start(AudioDriverSndio::thread_func, this);

	return OK;
}

void AudioDriverSndio::thread_func(void *p_udata) {
	AudioDriverSndio *ad = (AudioDriverSndio *)p_udata;

	for (size_t i = 0; i < ad->period_size * ad->channels; ++i) {
		ad->samples.write[i] = 0;
	}

	while (!ad->exit_thread) {
		ad->lock();
		ad->start_counting_ticks();

		if (ad->active) {
			ad->audio_server_process(ad->period_size, ad->samples.ptrw());
		}

		ad->stop_counting_ticks();
		ad->unlock();

		size_t bytes = ad->period_size * ad->channels * sizeof(int32_t);
		if (sio_write(ad->handle, ad->samples.ptr(), bytes) != bytes) {
			ERR_PRINT("sndio: fatal error");
			ad->exit_thread = true;
		}
	}

	ad->thread_exited = true;
}

void AudioDriverSndio::start() {
	active = true;
}

int AudioDriverSndio::get_mix_rate() const {
	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverSndio::get_speaker_mode() const {
	return speaker_mode;
}

void AudioDriverSndio::lock() {
	mutex.lock();
}

void AudioDriverSndio::unlock() {
	mutex.unlock();
}

void AudioDriverSndio::finish() {
	exit_thread = true;
	thread.wait_to_finish();

	if (handle) {
		sio_close(handle);
		handle = NULL;
	}
}

#endif
