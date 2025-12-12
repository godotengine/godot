/**************************************************************************/
/*  audio_sample_grabber_callback.h                                       */
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

#pragma once

#include "core/os/mutex.h"
#include "core/templates/vector.h"
#include "servers/audio_server.h"
#include <mfapi.h>
#include <mfidl.h>

class VideoStreamPlaybackWMF;

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

class AudioSampleGrabberCallback : public IMFSampleGrabberSinkCallback {
private:
	long m_cRef;
	VideoStreamPlaybackWMF *playback;

	int input_sample_rate = 0;
	int input_channels = 0;
	int output_sample_rate = 0; // Will be set from AudioServer
	int output_channels = 2; // Stereo

	Vector<float> resample_buffer;
	double resample_ratio = 1.0;

public:
	AudioSampleGrabberCallback(VideoStreamPlaybackWMF *playback, Mutex &mtx);
	virtual ~AudioSampleGrabberCallback();

	static HRESULT CreateInstance(AudioSampleGrabberCallback **ppCB, VideoStreamPlaybackWMF *playback, Mutex &mtx);

	// IUnknown methods
	STDMETHODIMP QueryInterface(REFIID riid, void **ppv) override;
	STDMETHODIMP_(ULONG)
	AddRef() override;
	STDMETHODIMP_(ULONG)
	Release() override;

	// IMFClockStateSink methods
	STDMETHODIMP OnClockStart(MFTIME hnsSystemTime, LONGLONG llClockStartOffset) override;
	STDMETHODIMP OnClockStop(MFTIME hnsSystemTime) override;
	STDMETHODIMP OnClockPause(MFTIME hnsSystemTime) override;
	STDMETHODIMP OnClockRestart(MFTIME hnsSystemTime) override;
	STDMETHODIMP OnClockSetRate(MFTIME hnsSystemTime, float flRate) override;

	// IMFSampleGrabberSinkCallback methods
	STDMETHODIMP OnSetPresentationClock(IMFPresentationClock *pClock) override;
	STDMETHODIMP OnProcessSample(REFGUID guidMajorMediaType, DWORD dwSampleFlags,
			LONGLONG llSampleTime, LONGLONG llSampleDuration, const BYTE *pSampleBuffer,
			DWORD dwSampleSize) override;
	STDMETHODIMP OnShutdown() override;

	void set_audio_format(int sample_rate, int channels);
	void set_output_format(int sample_rate, int channels);

private:
	void resample_audio(const float *input, int input_samples, Vector<float> &output);
	void convert_to_float(const BYTE *input, DWORD input_size, Vector<float> &output);
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
