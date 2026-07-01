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

Error AudioDriverSndio::AudioDeviceSndio::start(bool p_record) {
	ERR_FAIL_COND_V_MSG(handle, ERR_ALREADY_IN_USE, "sndio: device already started.");
	handle = sio_open(SIO_DEVANY, p_record ? SIO_REC : SIO_PLAY, 0);
	ERR_FAIL_NULL_V_MSG(handle, ERR_CANT_OPEN, "sndio: failed to open device.");

	sio_initpar(&parameters);

	parameters.bits = 32;
	parameters.bps = 4;
	parameters.sig = true;
	parameters.le = SIO_LE_NATIVE;
	parameters.msb = true;
	parameters.rchan = 2;
	parameters.rate = AudioDriver::get_configured_mix_rate();
	parameters.appbufsz = Engine::get_singleton()->get_audio_output_latency() * parameters.rate / 1000.0f;

	if (!sio_setpar(handle, &parameters) || !sio_getpar(handle, &parameters)) {
		close();
		ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "sndio: unable to set device parameters.");
	}

	if (p_record) {
		channels = 2;
		if (parameters.rchan < 1) {
			close();
			ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "sndio: input device has zero channels.");
		}
	} else {
		if (parameters.pchan < 1 || parameters.pchan > 8) {
			// Fall back to stereo
			WARN_PRINT("sndio: received unsupported output channel count: " + itos(parameters.pchan));
			parameters.pchan = 2;
			if (!sio_setpar(handle, &parameters) || !sio_getpar(handle, &parameters) || parameters.pchan < 1 || parameters.pchan > 8) {
				close();
				ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "sndio: unable to fall back to stereo output.");
			}
		}

		channels = parameters.pchan;

		if (parameters.pchan == 1 || parameters.pchan == 3 || parameters.pchan == 5 || parameters.pchan == 7) {
			// The last channel will be mixed manually.
			channels++;
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
	bool remixing = dev->parameters.pchan != dev->channels || dev->parameters.bps != sizeof(int32_t) ||
			(!dev->parameters.msb && dev->parameters.bits < dev->parameters.bps * 8) ||
			!dev->parameters.sig || dev->parameters.le != SIO_LE_NATIVE;

	frames.resize(dev->parameters.appbufsz * dev->channels);
	if (remixing) {
		remixed_frames.resize(dev->parameters.appbufsz * dev->parameters.pchan * dev->parameters.bps);
	}
	size_t bytes = remixing ? remixed_frames.size() : frames.size() * sizeof(int32_t);

	uint8_t shift = dev->parameters.msb ? (sizeof(int32_t) * 8) - (dev->parameters.bps * 8) : (sizeof(int32_t) * 8) - dev->parameters.bits;
	uint32_t bias = dev->parameters.sig ? 0 : (1U << ((sizeof(int32_t) * 8) - 1)) >> shift;

	while (!dev->exit_thread.is_set()) {
		ad->lock();
		ad->start_counting_ticks();

		if (ad->active.is_set()) {
			ad->audio_server_process(dev->parameters.appbufsz, frames.ptr());
		}

		ad->stop_counting_ticks();
		ad->unlock();

		if (remixing) {
			for (size_t sample_index = 0; sample_index < frames.size(); sample_index++) {
				frames[sample_index] >>= shift;
				frames[sample_index] += bias;
			}

			for (size_t in_index = 0, out_index = 0; in_index < frames.size();) {
				for (unsigned int channel_index = 0; channel_index < dev->channels; channel_index++) {
					memcpy(remixed_frames.ptr() + (out_index++ * dev->parameters.bps), frames.ptr() + in_index++, dev->parameters.bps);
				}

				if (dev->parameters.pchan == 1 || dev->parameters.pchan == 3 || dev->parameters.pchan == 5 || dev->parameters.pchan == 7) {
					// Mix last left + right channels.
					int32_t combined_sample = frames[in_index++] / 2;
					combined_sample += frames[in_index++] / 2;
					memcpy(remixed_frames.ptr() + (out_index++ * dev->parameters.bps), &combined_sample, dev->parameters.bps);
				}
			}

			if (dev->parameters.le != SIO_LE_NATIVE) {
				// Reverse byte order.
				for (size_t sample_index = 0; sample_index < remixed_frames.size() / dev->parameters.bps; sample_index++) {
					for (uint8_t byte_index = 0; byte_index < dev->parameters.bps / 2; byte_index++) {
						int8_t temp_byte = remixed_frames[(sample_index * dev->parameters.bps) + byte_index];
						remixed_frames[(sample_index * dev->parameters.bps) + byte_index] = remixed_frames[(sample_index * dev->parameters.bps) + (dev->parameters.bps - byte_index - 1)];
						remixed_frames[(sample_index * dev->parameters.bps) + (dev->parameters.bps - byte_index - 1)] = temp_byte;
					}
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
	bool remixing = dev->parameters.rchan > dev->channels || dev->parameters.bps != sizeof(int32_t) ||
			(!dev->parameters.msb && dev->parameters.bits < dev->parameters.bps * 8) ||
			!dev->parameters.sig || dev->parameters.le != SIO_LE_NATIVE;

	frames.resize(dev->parameters.appbufsz * dev->parameters.rchan * dev->parameters.bps);
	if (remixing) {
		remixed_frames.resize(dev->parameters.appbufsz * MIN(dev->channels, dev->parameters.rchan));
	}

	uint8_t shift = dev->parameters.msb ? (sizeof(int32_t) * 8) - (dev->parameters.bps * 8) : (sizeof(int32_t) * 8) - dev->parameters.bits;
	uint32_t bias = dev->parameters.sig ? 0 : (1U << ((sizeof(int32_t) * 8) - 1)) >> shift;

	while (!dev->exit_thread.is_set()) {
		size_t bytes = sio_read(dev->handle, frames.ptr(), frames.size());
		ERR_BREAK_MSG(bytes == 0, "sndio: received 0 bytes.");

		if (!ad->active.is_set()) {
			continue;
		}

		if (remixing) {
			unsigned int channel_index = 0;
			for (size_t in_index = 0, out_index = 0; in_index < bytes / dev->parameters.bps; in_index++) {
				if (channel_index == dev->parameters.rchan) {
					channel_index = 0;
				}

				// Skip channels after 2.
				if (channel_index++ >= dev->channels) {
					continue;
				}

				size_t in_offset = in_index * dev->parameters.bps;

				if (dev->parameters.le != SIO_LE_NATIVE) {
					// Reverse byte order.
					for (uint8_t byte_index = 0; byte_index < dev->parameters.bps / 2; byte_index++) {
						int8_t temp_byte = frames[in_offset + byte_index];
						frames[in_offset] = frames[in_offset + (dev->parameters.bps - byte_index - 1)];
						frames[in_offset + (dev->parameters.bps - byte_index - 1)] = temp_byte;
					}
				}

				memcpy(remixed_frames.ptr() + out_index, frames.ptr() + in_offset, dev->parameters.bps);
				remixed_frames[out_index] -= bias;
				remixed_frames[out_index] <<= shift;
				out_index++;
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
	Error err = device.start(false);
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

	Error err = input_device.start(true);
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
