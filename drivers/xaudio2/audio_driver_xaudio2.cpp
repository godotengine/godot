/**************************************************************************/
/*  audio_driver_xaudio2.cpp                                              */
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

#include "audio_driver_xaudio2.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"

Error AudioDriverXAudio2::init() {
	active.clear();
	exit_thread.clear();
	pcm_open = false;

	// TODO: `wave_format.nChannels` and `buffer_format` are hardcoded.
	buffer_format = BUFFER_FORMAT_INTEGER_16;

	wave_format.nChannels = 2;
	wave_format.cbSize = 0;
	wave_format.nSamplesPerSec = _get_configured_mix_rate();
	wave_format.wFormatTag = WAVE_FORMAT_PCM;
	wave_format.wBitsPerSample = get_size_of_sample(buffer_format) * 8;
	wave_format.nBlockAlign = wave_format.nChannels * wave_format.wBitsPerSample / 8;
	wave_format.nAvgBytesPerSec = wave_format.nSamplesPerSec * wave_format.nBlockAlign;

	int latency = Engine::get_singleton()->get_audio_output_latency();
	buffer_frames = MIN(latency * wave_format.nSamplesPerSec / 1000, XAUDIO2_MAX_BUFFER_BYTES / (wave_format.nChannels * get_size_of_sample(buffer_format)));

	for (int i = 0; i < AUDIO_BUFFERS; i++) {
		memset(&xaudio_buffer[i], 0, sizeof(xaudio_buffer[i]));
		xaudio_buffer[i].AudioBytes = (uint32_t)buffer_frames * wave_format.nChannels * get_size_of_sample(buffer_format);

		samples_out[i].resize(xaudio_buffer[i].AudioBytes);
		xaudio_buffer[i].pAudioData = (const BYTE *)samples_out[i].ptr();
	}

	HRESULT hr;
	hr = XAudio2Create(&xaudio, 0, XAUDIO2_DEFAULT_PROCESSOR);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_UNAVAILABLE, "Error creating XAudio2 engine.");

	hr = xaudio->CreateMasteringVoice(&mastering_voice);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_UNAVAILABLE, "Error creating XAudio2 mastering voice.");

	hr = xaudio->CreateSourceVoice(&source_voice, &wave_format, 0, XAUDIO2_MAX_FREQ_RATIO, &voice_callback);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_UNAVAILABLE, "Error creating XAudio2 source voice. Error code: " + itos(hr) + ".");

	thread.start(AudioDriverXAudio2::thread_func, this);

	return OK;
}

void AudioDriverXAudio2::thread_func(void *p_udata) {
	AudioDriverXAudio2 *ad = static_cast<AudioDriverXAudio2 *>(p_udata);

	while (!ad->exit_thread.is_set() && !ad->active.is_set()) {
		OS::get_singleton()->delay_usec(1000);
	}

	while (!ad->exit_thread.is_set()) {
		ad->lock();
		ad->start_counting_ticks();

		ad->audio_server_process(ad->buffer_frames, ad->samples_out[ad->current_buffer].ptr());

		ad->stop_counting_ticks();
		ad->unlock();

		ad->source_voice->SubmitSourceBuffer(&ad->xaudio_buffer[ad->current_buffer]);
		ad->current_buffer = (ad->current_buffer + 1) % AUDIO_BUFFERS;

		XAUDIO2_VOICE_STATE state;
		while (ad->source_voice->GetState(&state), state.BuffersQueued > AUDIO_BUFFERS - 1) {
			WaitForSingleObject(ad->voice_callback.buffer_end_event, INFINITE);
		}
	}
}

void AudioDriverXAudio2::start() {
	active.set();
	HRESULT hr = source_voice->Start(0);
	ERR_FAIL_COND_MSG(hr != S_OK, "Error starting XAudio2 driver. Error code: " + itos(hr) + ".");
}

int AudioDriverXAudio2::get_mix_rate() const {
	return wave_format.nSamplesPerSec;
}

float AudioDriverXAudio2::get_latency() {
	XAUDIO2_PERFORMANCE_DATA perf_data;
	xaudio->GetPerformanceData(&perf_data);
	if (perf_data.CurrentLatencyInSamples) {
		return (float)perf_data.CurrentLatencyInSamples / wave_format.nSamplesPerSec;
	} else {
		return 0;
	}
}

int AudioDriverXAudio2::get_output_channels() const {
	return wave_format.nChannels;
}

AudioDriver::BufferFormat AudioDriverXAudio2::get_output_buffer_format() const {
	return buffer_format;
}

void AudioDriverXAudio2::lock() {
	mutex.lock();
}

void AudioDriverXAudio2::unlock() {
	mutex.unlock();
}

void AudioDriverXAudio2::finish() {
	exit_thread.set();
	if (thread.is_started()) {
		thread.wait_to_finish();
	}

	if (source_voice) {
		source_voice->Stop(0);
		source_voice->DestroyVoice();
	}

	mastering_voice->DestroyVoice();
}
