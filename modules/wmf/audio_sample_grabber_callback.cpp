/**************************************************************************/
/*  audio_sample_grabber_callback.cpp                                     */
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

#include "audio_sample_grabber_callback.h"
#include "core/string/print_string.h"
#include "video_stream_wmf.h"
#include <mfapi.h>
#include <minwindef.h>
#include <shlwapi.h>
#include <cassert>
#include <cstdio>
#include <new>

AudioSampleGrabberCallback::AudioSampleGrabberCallback(VideoStreamPlaybackWMF *p_playback, Mutex &p_mtx) :
		m_cRef(1), playback(p_playback) {
	AudioServer *audio_server = AudioServer::get_singleton();
	if (audio_server) {
		output_sample_rate = audio_server->get_mix_rate();
		print_line("AudioSampleGrabberCallback: Using Godot mix rate: " + itos(output_sample_rate));
	} else {
		output_sample_rate = 44100; // Fallback
		print_line("AudioSampleGrabberCallback: AudioServer not available, using fallback: " + itos(output_sample_rate));
	}
}

HRESULT AudioSampleGrabberCallback::CreateInstance(AudioSampleGrabberCallback **ppCB, VideoStreamPlaybackWMF *playback, Mutex &mtx) {
	*ppCB = new (std::nothrow) AudioSampleGrabberCallback(playback, mtx);
	if (ppCB == nullptr) {
		return E_OUTOFMEMORY;
	}
	return S_OK;
}

AudioSampleGrabberCallback::~AudioSampleGrabberCallback() {
}

STDMETHODIMP AudioSampleGrabberCallback::QueryInterface(REFIID riid, void **ppv) {
	static const QITAB qit[] = {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4838)
#endif
		QITABENT(AudioSampleGrabberCallback, IMFSampleGrabberSinkCallback),
#ifdef _MSC_VER
#pragma warning(pop)
#endif
		{ 0, 0 }
	};
	return QISearch(this, qit, riid, ppv);
}

STDMETHODIMP_(ULONG)
AudioSampleGrabberCallback::AddRef() {
	return InterlockedIncrement(&m_cRef);
}

STDMETHODIMP_(ULONG)
AudioSampleGrabberCallback::Release() {
	ULONG cRef = InterlockedDecrement(&m_cRef);
	if (cRef == 0) {
		delete this;
	}
	return cRef;
}

// IMFClockStateSink methods
STDMETHODIMP AudioSampleGrabberCallback::OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset) {
	return S_OK;
}

STDMETHODIMP AudioSampleGrabberCallback::OnClockStop(MFTIME hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP AudioSampleGrabberCallback::OnClockPause(MFTIME hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP AudioSampleGrabberCallback::OnClockRestart(MFTIME hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP AudioSampleGrabberCallback::OnClockSetRate(MFTIME hnsSystemTime, float flRate) {
	return S_OK;
}

STDMETHODIMP AudioSampleGrabberCallback::OnSetPresentationClock(IMFPresentationClock *pClock) {
	return S_OK;
}

STDMETHODIMP AudioSampleGrabberCallback::OnProcessSample(REFGUID guidMajorMediaType,
		DWORD dwSampleFlags,
		LONGLONG llSampleTime,
		LONGLONG llSampleDuration,
		const BYTE *pSampleBuffer,
		DWORD dwSampleSize) {
	if (input_sample_rate == 0 || input_channels == 0) {
		return S_OK;
	}

	Vector<float> input_float;
	convert_to_float(pSampleBuffer, dwSampleSize, input_float);

	Vector<float> output_float;
	if (input_sample_rate != output_sample_rate) {
		resample_audio(input_float.ptr(), input_float.size() / input_channels, output_float);
	} else {
		output_float = input_float;
	}

	if (playback) {
		playback->add_audio_data(llSampleTime, output_float);
	}

	return S_OK;
}

STDMETHODIMP AudioSampleGrabberCallback::OnShutdown() {
	print_line("AudioSampleGrabberCallback::OnShutdown");
	return S_OK;
}

void AudioSampleGrabberCallback::set_audio_format(int sample_rate, int channels) {
	input_sample_rate = sample_rate;
	input_channels = channels;

	if (output_sample_rate > 0) {
		resample_ratio = (double)output_sample_rate / (double)input_sample_rate;
	}

	print_line("Audio format set: " + itos(input_sample_rate) + "Hz, " + itos(input_channels) + " channels");
	print_line("Resample ratio: " + rtos(resample_ratio));
}

void AudioSampleGrabberCallback::set_output_format(int sample_rate, int channels) {
	output_sample_rate = sample_rate;
	output_channels = channels;

	if (input_sample_rate > 0) {
		resample_ratio = (double)output_sample_rate / (double)input_sample_rate;
	}
}

void AudioSampleGrabberCallback::convert_to_float(const BYTE *input, DWORD input_size, Vector<float> &output) {
	// Assume 16-bit PCM input (most common)
	int sample_count = input_size / 2; // 2 bytes per 16-bit sample
	output.resize(sample_count);

	const int16_t *input_samples = reinterpret_cast<const int16_t *>(input);
	float *output_samples = output.ptrw();

	for (int i = 0; i < sample_count; i++) {
		// Convert 16-bit PCM to float (-1.0 to 1.0)
		output_samples[i] = (float)input_samples[i] / 32768.0f;
	}
}

void AudioSampleGrabberCallback::resample_audio(const float *input, int input_samples, Vector<float> &output) {
	if (resample_ratio == 1.0) {
		int total_samples = input_samples * input_channels;
		output.resize(total_samples);
		memcpy(output.ptrw(), input, total_samples * sizeof(float));
		return;
	}

	// Simple linear interpolation resampling
	int output_samples = (int)(input_samples * resample_ratio);
	output.resize(output_samples * output_channels);

	float *output_ptr = output.ptrw();

	for (int i = 0; i < output_samples; i++) {
		double src_index = (double)i / resample_ratio;
		int src_index_int = (int)src_index;
		double frac = src_index - src_index_int;

		for (int ch = 0; ch < input_channels && ch < output_channels; ch++) {
			float sample1 = 0.0f;
			float sample2 = 0.0f;

			if (src_index_int < input_samples) {
				sample1 = input[src_index_int * input_channels + ch];
			}
			if (src_index_int + 1 < input_samples) {
				sample2 = input[(src_index_int + 1) * input_channels + ch];
			}

			// Linear interpolation
			float interpolated = sample1 + (sample2 - sample1) * frac;
			output_ptr[i * output_channels + ch] = interpolated;
		}

		// Fill remaining output channels with zeros if input has fewer channels
		for (int ch = input_channels; ch < output_channels; ch++) {
			output_ptr[i * output_channels + ch] = 0.0f;
		}
	}
}
