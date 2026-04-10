/**************************************************************************/
/*  audio_driver_sndio.cpp                                                */
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

#include "audio_driver_sndio.h"

#ifdef SNDIO_ENABLED

#include "core/config/engine.h"

Error AudioDriverSndio::AudioDeviceSndio::start(unsigned int p_mode) {
	ERR_FAIL_COND_V_MSG(handle, ERR_ALREADY_IN_USE, "sndio: device already started.");
	handle = sio_open(SIO_DEVANY, p_mode, 0);
	ERR_FAIL_NULL_V_MSG(handle, ERR_CANT_OPEN, "sndio: failed to open device.");

	sio_initpar(&parameters);

	parameters.bits = 32;
	parameters.bps = 4;
	parameters.sig = 1;
	parameters.le = SIO_LE_NATIVE;
	parameters.msb = 1;
	parameters.rchan = 2;
	parameters.rate = AudioDriver::get_configured_mix_rate();
	parameters.appbufsz = Engine::get_singleton()->get_audio_output_latency() * parameters.rate / 1000.0f;

	if (!sio_setpar(handle, &parameters) || !sio_getpar(handle, &parameters)) {
		close();
		ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "sndio: unable to set device parameters.");
	}

	if (parameters.sig != 1 || parameters.le != SIO_LE_NATIVE) {
		close();
		ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "sndio: received unsupported parameters.");
	}

	if (p_mode == SIO_REC) {
		if (parameters.rchan > 2 || parameters.rchan < 1) {
			close();
			ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "sndio: received unsupported input channel count: " + itos(parameters.rchan));
		}
		channels = 2;
	} else {
		channels = parameters.pchan;
		if (parameters.pchan == 1 || parameters.pchan == 3 || parameters.pchan == 5 || parameters.pchan == 7) {
			// The last channel will be mixed manually.
			channels++;
		} else if (parameters.pchan < 1 || parameters.pchan > 8) {
			// Fallback to stereo
			WARN_PRINT("sndio: received unsupported output channel count: " + itos(parameters.pchan));
			parameters.pchan = channels = 2;
			if (!sio_setpar(handle, &parameters) || parameters.pchan != 2) {
				close();
				ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "sndio: unable to fallback to stereo output.");
			}
		}
	}

	if (!sio_start(handle)) {
		close();
		ERR_FAIL_V_MSG(ERR_CANT_OPEN, "sndio: failed to start device.");
	}

	return OK;
}

void AudioDriverSndio::AudioDeviceSndio::close() {
	if (handle) {
		sio_close(handle);
		handle = nullptr;
	}
}

void AudioDriverSndio::thread_func(void *p_udata) {
	AudioDriverSndio *ad = static_cast<AudioDriverSndio *>(p_udata);
	AudioDeviceSndio *dev = &ad->device;
	LocalVector<int32_t> frames;
	LocalVector<int8_t> remixed_frames;
	bool remixing = dev->channels != dev->parameters.pchan || dev->parameters.bps != sizeof(int32_t) || (!dev->parameters.msb && dev->parameters.bits < dev->parameters.bps * 8);

	frames.resize(dev->parameters.appbufsz * dev->channels);
	if (remixing) {
		remixed_frames.resize(dev->parameters.appbufsz * dev->parameters.pchan * dev->parameters.bps);
	}
	size_t bytes = remixing ? remixed_frames.size() * sizeof(int8_t) : frames.size() * sizeof(int32_t);

	while (!dev->exit_thread.is_set()) {
		ad->lock();
		ad->start_counting_ticks();

		if (ad->active.is_set()) {
			ad->audio_server_process(dev->parameters.appbufsz, frames.ptr());
		}

		ad->stop_counting_ticks();
		ad->unlock();

		if (remixing) {
			if (dev->parameters.bps < sizeof(int32_t)) {
				for (size_t sample_index = 0; sample_index < remixed_frames.size(); sample_index++) {
					remixed_frames[sample_index] = 0;
				}
			}

			for (size_t sample_index = 0; sample_index < frames.size(); sample_index++) {
				if (!dev->parameters.msb) {
					frames[sample_index] >>= (dev->parameters.bps * 8) - dev->parameters.bits;
				}
			}

			for (size_t in_index = 0, out_index = 0; in_index < frames.size();) {
				for (unsigned int channel_index = 0; channel_index < dev->parameters.pchan - (dev->channels == dev->parameters.pchan ? 0 : 1); channel_index++) {
					memcpy(remixed_frames.ptr() + (out_index++ * dev->parameters.bps),
							(int8_t *)frames.ptr() + (in_index++ * sizeof(int32_t)) + (sizeof(int32_t) - dev->parameters.bps),
							dev->parameters.bps);
				}

				if (dev->channels != dev->parameters.pchan) {
					// Mix last left + right channels.
					int32_t combined_sample = frames[in_index++] / 2;
					combined_sample += frames[in_index++] / 2;
					memcpy(remixed_frames.ptr() + (out_index++ * dev->parameters.bps),
							(int8_t *)&combined_sample + ((sizeof(int32_t)) - dev->parameters.bps),
							dev->parameters.bps);
				}
			}
		}

		ERR_BREAK_MSG(sio_write(dev->handle, remixing ? (void *)remixed_frames.ptr() : (void *)frames.ptr(), bytes) != bytes, "sndio: did not receive requested number of output bytes.");
	}
}

void AudioDriverSndio::input_thread_func(void *p_udata) {
	AudioDriverSndio *ad = static_cast<AudioDriverSndio *>(p_udata);
	AudioDeviceSndio *dev = &ad->input_device;
	LocalVector<int8_t> frames;
	LocalVector<int32_t> remixed_frames;
	bool remixing = dev->parameters.bps != sizeof(int32_t) || (!dev->parameters.msb && dev->parameters.bits < dev->parameters.bps * 8);

	frames.resize(dev->parameters.appbufsz * dev->parameters.rchan * dev->parameters.bps);
	if (remixing) {
		remixed_frames.resize(dev->parameters.appbufsz * dev->channels);
	}

	while (!dev->exit_thread.is_set()) {
		size_t bytes = sio_read(dev->handle, frames.ptr(), frames.size());
		ERR_BREAK_MSG(bytes == 0, "sndio: received 0 bytes.");

		if (!ad->active.is_set()) {
			continue;
		}

		if (remixing) {
			for (size_t sample_index = 0; sample_index < bytes / dev->parameters.bps; sample_index++) {
				int32_t sample = 0;
				memcpy((int8_t *)&sample + (sizeof(int32_t) - dev->parameters.bps), frames.ptr() + (sample_index * dev->parameters.bps), dev->parameters.bps);

				if (!dev->parameters.msb) {
					sample <<= (dev->parameters.bps * 8) - dev->parameters.bits;
				}

				remixed_frames[sample_index] = sample;
			}
		}

		ad->lock();
		ad->start_counting_ticks();

		for (size_t sample_index = 0; sample_index < (remixing ? remixed_frames.size() : frames.size() / dev->parameters.bps); sample_index++) {
			int32_t *sample = (remixing ? remixed_frames.ptr() : (int32_t *)frames.ptr()) + sample_index;
			ad->input_buffer_write(*sample);
			if (dev->parameters.rchan == 1) {
				// If input device is mono, convert it to stereo
				ad->input_buffer_write(*sample);
			}
		}

		ad->stop_counting_ticks();
		ad->unlock();
	}
}

Error AudioDriverSndio::init() {
#ifdef SOWRAP_ENABLED
#ifdef DEBUG_ENABLED
	int dylibloader_verbose = 1;
#else
	int dylibloader_verbose = 0;
#endif
	if (initialize_sndio(dylibloader_verbose)) {
		return ERR_CANT_OPEN;
	}
#endif
	Error err = device.start(SIO_PLAY);
	if (err == OK) {
		thread.start(AudioDriverSndio::thread_func, this);
	}

	return err;
}

int AudioDriverSndio::get_mix_rate() const {
	mutex.lock();
	int mix_rate = device.parameters.rate;
	mutex.unlock();

	return mix_rate;
}

int AudioDriverSndio::get_input_mix_rate() const {
	mutex.lock();
	int input_mix_rate = input_device.parameters.rate;
	mutex.unlock();

	return input_mix_rate;
}

AudioDriver::SpeakerMode AudioDriverSndio::get_speaker_mode() const {
	mutex.lock();
	int channels = device.channels;
	mutex.unlock();

	return get_speaker_mode_by_total_channels(channels);
}

float AudioDriverSndio::get_latency() {
	lock();
	float latency = ((float)device.parameters.appbufsz / (float)device.parameters.rate) * 1000.0f;
	unlock();

	return latency;
}

void AudioDriverSndio::finish() {
	device.exit_thread.set();
	if (thread.is_started()) {
		thread.wait_to_finish();
	}

	device.close();
	input_stop();
}

Error AudioDriverSndio::input_start() {
	lock();

	Error err = input_device.start(SIO_REC);
	input_buffer_init(input_device.parameters.appbufsz);
	if (err == OK) {
		input_device.exit_thread.clear();
		input_thread_mutex.lock();
		input_thread.start(AudioDriverSndio::input_thread_func, this);
		input_thread_mutex.unlock();
	}

	unlock();

	return err;
}

Error AudioDriverSndio::input_stop() {
	input_device.exit_thread.set();
	input_thread_mutex.lock();
	if (input_thread.is_started()) {
		input_thread.wait_to_finish();
	}
	input_thread_mutex.unlock();

	lock();
	input_device.close();
	unlock();

	return OK;
}

#endif
