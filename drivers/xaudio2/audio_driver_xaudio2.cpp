/*************************************************************************/
/*  audio_driver_xaudio2.cpp                                             */
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

#include "audio_driver_xaudio2.h"

#include "core/os/os.h"
#include "core/project_settings.h"

const char *AudioDriverXAudio2::get_name() const {
	return "XAudio2";
}

Error AudioDriverXAudio2::init() {
	active.clear();
	exit_thread.clear();
	pcm_open = false;
	samples_in = NULL;

	mix_rate = GLOBAL_GET("audio/mix_rate");
	// FIXME: speaker_mode seems unused in the Xaudio2 driver so far
	speaker_mode = SPEAKER_MODE_STEREO;
	channels = 2;

	int latency = GLOBAL_GET("audio/output_latency");
	buffer_size = closest_power_of_2(latency * mix_rate / 1000);

	samples_in = memnew_arr(int32_t, buffer_size * channels);
	for (int i = 0; i < AUDIO_BUFFERS; i++) {
		samples_out[i] = memnew_arr(int16_t, buffer_size * channels);
		xaudio_buffer[i].AudioBytes = buffer_size * channels * sizeof(int16_t);
		xaudio_buffer[i].pAudioData = (const BYTE *)(samples_out[i]);
		xaudio_buffer[i].Flags = 0;
	}

	HRESULT hr;
	hr = XAudio2Create(&xaudio, 0, XAUDIO2_DEFAULT_PROCESSOR);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_UNAVAILABLE, "Error creating XAudio2 engine.");

	hr = xaudio->CreateMasteringVoice(&mastering_voice);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_UNAVAILABLE, "Error creating XAudio2 mastering voice.");

	wave_format.nChannels = channels;
	wave_format.cbSize = 0;
	wave_format.nSamplesPerSec = mix_rate;
	wave_format.wFormatTag = WAVE_FORMAT_PCM;
	wave_format.wBitsPerSample = 16;
	wave_format.nBlockAlign = channels * wave_format.wBitsPerSample >> 3;
	wave_format.nAvgBytesPerSec = mix_rate * wave_format.nBlockAlign;

	hr = xaudio->CreateSourceVoice(&source_voice, &wave_format, 0, XAUDIO2_MAX_FREQ_RATIO, &voice_callback);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_UNAVAILABLE, "Error creating XAudio2 source voice. Error code: " + itos(hr) + ".");

	thread.start(AudioDriverXAudio2::thread_func, this);

	return OK;
}

void AudioDriverXAudio2::thread_func(void *p_udata) {
	AudioDriverXAudio2 *ad = (AudioDriverXAudio2 *)p_udata;

	while (!ad->exit_thread.is_set()) {
		if (!ad->active.is_set()) {
			for (int i = 0; i < AUDIO_BUFFERS; i++) {
				ad->xaudio_buffer[i].Flags = XAUDIO2_END_OF_STREAM;
			}

		} else {
			ad->lock();
			ad->start_counting_ticks();

			ad->audio_server_process(ad->buffer_size, ad->samples_in);

			ad->stop_counting_ticks();
			ad->unlock();

			for (unsigned int i = 0; i < ad->buffer_size * ad->channels; i++) {
				ad->samples_out[ad->current_buffer][i] = ad->samples_in[i] >> 16;
			}

			ad->xaudio_buffer[ad->current_buffer].Flags = 0;
			ad->xaudio_buffer[ad->current_buffer].AudioBytes = ad->buffer_size * ad->channels * sizeof(int16_t);
			ad->xaudio_buffer[ad->current_buffer].pAudioData = (const BYTE *)(ad->samples_out[ad->current_buffer]);
			ad->xaudio_buffer[ad->current_buffer].PlayBegin = 0;
			ad->source_voice->SubmitSourceBuffer(&(ad->xaudio_buffer[ad->current_buffer]));

			ad->current_buffer = (ad->current_buffer + 1) % AUDIO_BUFFERS;

			XAUDIO2_VOICE_STATE state;
			while (ad->source_voice->GetState(&state), state.BuffersQueued > AUDIO_BUFFERS - 1) {
				WaitForSingleObject(ad->voice_callback.buffer_end_event, INFINITE);
			}
		}
	}
}

void AudioDriverXAudio2::start() {
	active.set();
	HRESULT hr = source_voice->Start(0);
	ERR_FAIL_COND_MSG(hr != S_OK, "Error starting XAudio2 driver. Error code: " + itos(hr) + ".");
}

int AudioDriverXAudio2::get_mix_rate() const {
	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverXAudio2::get_speaker_mode() const {
	return speaker_mode;
}

float AudioDriverXAudio2::get_latency() {
	XAUDIO2_PERFORMANCE_DATA perf_data;
	xaudio->GetPerformanceData(&perf_data);
	if (perf_data.CurrentLatencyInSamples) {
		return (float)(perf_data.CurrentLatencyInSamples / ((float)mix_rate));
	} else {
		return 0;
	}
}

void AudioDriverXAudio2::lock() {
	mutex.lock();
}
void AudioDriverXAudio2::unlock() {
	mutex.unlock();
}

void AudioDriverXAudio2::finish() {
	if (!thread.is_started())
		return;

	exit_thread.set();
	thread.wait_to_finish();

	if (source_voice) {
		source_voice->Stop(0);
		source_voice->DestroyVoice();
	}

	if (samples_in) {
		memdelete_arr(samples_in);
	}
	if (samples_out[0]) {
		for (int i = 0; i < AUDIO_BUFFERS; i++) {
			memdelete_arr(samples_out[i]);
		}
	}

	mastering_voice->DestroyVoice();
}

AudioDriverXAudio2::AudioDriverXAudio2() :
		current_buffer(0) {
	wave_format = { 0 };
	for (int i = 0; i < AUDIO_BUFFERS; i++) {
		xaudio_buffer[i] = { 0 };
		samples_out[i] = 0;
	}
}

AudioDriverXAudio2::~AudioDriverXAudio2() {
}
