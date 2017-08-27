/*************************************************************************/
/*  audio_driver_winrt.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "audio_driver_winrt.h"

#include "globals.h"
#include "os/os.h"

using namespace Windows::Media;
using namespace Windows::Media::Core;
using namespace Windows::Media::MediaProperties;
using namespace Windows::Media::Editing;
using namespace Windows::Foundation;

const char *AudioDriverWinRT::get_name() const {
	return "WinRT";
}

Error AudioDriverWinRT::init() {

	active = false;
	thread_exited = false;
	exit_thread = false;
	pcm_open = false;
	samples_in = NULL;

	mix_rate = 48000;
	output_format = OUTPUT_STEREO;
	channels = 2;

	int latency = GLOBAL_DEF("audio/output_latency", 25);
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
	if (hr != S_OK) {
		ERR_EXPLAIN("Error creating XAudio2 engine.");
		ERR_FAIL_V(ERR_UNAVAILABLE);
	}
	hr = xaudio->CreateMasteringVoice(&mastering_voice);
	if (hr != S_OK) {
		ERR_EXPLAIN("Error creating XAudio2 mastering voice.");
		ERR_FAIL_V(ERR_UNAVAILABLE);
	}

	wave_format.nChannels = channels;
	wave_format.cbSize = 0;
	wave_format.nSamplesPerSec = mix_rate;
	wave_format.wFormatTag = WAVE_FORMAT_PCM;
	wave_format.wBitsPerSample = 16;
	wave_format.nBlockAlign = channels * wave_format.wBitsPerSample >> 3;
	wave_format.nAvgBytesPerSec = mix_rate * wave_format.nBlockAlign;

	voice_callback = memnew(XAudio2DriverVoiceCallback);

	hr = xaudio->CreateSourceVoice(&source_voice, &wave_format, 0, XAUDIO2_MAX_FREQ_RATIO, voice_callback);
	if (hr != S_OK) {
		ERR_EXPLAIN("Error creating XAudio2 source voice. " + itos(hr));
		ERR_FAIL_V(ERR_UNAVAILABLE);
	}

	mutex = Mutex::create();
	thread = Thread::create(AudioDriverWinRT::thread_func, this);

	return OK;
};

void AudioDriverWinRT::thread_func(void *p_udata) {

	AudioDriverWinRT *ad = (AudioDriverWinRT *)p_udata;

	uint64_t usdelay = (ad->buffer_size / float(ad->mix_rate)) * 1000000;

	while (!ad->exit_thread) {

		if (!ad->active) {

			for (int i = 0; i < AUDIO_BUFFERS; i++) {
				ad->xaudio_buffer[i].Flags = XAUDIO2_END_OF_STREAM;
			}

		} else {

			ad->lock();

			ad->audio_server_process(ad->buffer_size, ad->samples_in);

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
				WaitForSingleObject(ad->voice_callback->buffer_end_event, INFINITE);
			}
		}
	};

	ad->thread_exited = true;
};

void AudioDriverWinRT::start() {

	active = true;
	HRESULT hr = source_voice->Start(0);
	if (hr != S_OK) {
		ERR_EXPLAIN("XAudio2 start error " + itos(hr));
		ERR_FAIL();
	}
};

int AudioDriverWinRT::get_mix_rate() const {

	return mix_rate;
};

AudioDriverSW::OutputFormat AudioDriverWinRT::get_output_format() const {

	return output_format;
};

float AudioDriverWinRT::get_latency() {

	XAUDIO2_PERFORMANCE_DATA perf_data;
	xaudio->GetPerformanceData(&perf_data);
	if (perf_data.CurrentLatencyInSamples) {
		return (float)(perf_data.CurrentLatencyInSamples / ((float)mix_rate));
	} else {
		return 0;
	}
}

void AudioDriverWinRT::lock() {

	if (!thread || !mutex)
		return;
	mutex->lock();
};
void AudioDriverWinRT::unlock() {

	if (!thread || !mutex)
		return;
	mutex->unlock();
};

void AudioDriverWinRT::finish() {

	if (!thread)
		return;

	exit_thread = true;
	Thread::wait_to_finish(thread);

	if (source_voice) {
		source_voice->Stop(0);
		memdelete(source_voice);
	}

	if (samples_in) {
		memdelete_arr(samples_in);
	};
	if (samples_out[0]) {
		for (int i = 0; i < AUDIO_BUFFERS; i++) {
			memdelete_arr(samples_out[i]);
		}
	};

	memdelete(voice_callback);
	memdelete(mastering_voice);

	memdelete(thread);
	if (mutex)
		memdelete(mutex);
	thread = NULL;
};

AudioDriverWinRT::AudioDriverWinRT() {

	mutex = NULL;
	thread = NULL;
	wave_format = { 0 };
	for (int i = 0; i < AUDIO_BUFFERS; i++) {
		xaudio_buffer[i] = { 0 };
		samples_out[i] = 0;
	}
	current_buffer = 0;
};

AudioDriverWinRT::~AudioDriverWinRT(){

};
