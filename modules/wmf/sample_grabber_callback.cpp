/**************************************************************************/
/*  sample_grabber_callback.cpp                                           */
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

#include "sample_grabber_callback.h"
#include "core/string/print_string.h"
#include "video_stream_wmf.h"
#include <mfapi.h>
#include <minwindef.h>
#include <shlwapi.h>
#include <cassert>
#include <cstdio>
#include <new>

#define CHECK_HR(func)                                                         \
	if (SUCCEEDED(hr)) {                                                       \
		hr = (func);                                                           \
		if (FAILED(hr)) {                                                      \
			print_line(vformat("%s failed, return:", __FUNCTION__, itos(hr))); \
		}                                                                      \
	}

SampleGrabberCallback::SampleGrabberCallback(VideoStreamPlaybackWMF *p_playback, Mutex &p_mtx) :
		m_cRef(1), playback(p_playback) {
}

HRESULT SampleGrabberCallback::CreateInstance(SampleGrabberCallback **p_ppCB, VideoStreamPlaybackWMF *p_playback, Mutex &p_mtx) {
	*p_ppCB = new (std::nothrow) SampleGrabberCallback(p_playback, p_mtx);

	if (p_ppCB == nullptr) {
		return E_OUTOFMEMORY;
	}
	return S_OK;
}

SampleGrabberCallback::~SampleGrabberCallback() {
}

STDMETHODIMP SampleGrabberCallback::QueryInterface(REFIID p_riid, void **p_ppv) {
	static const QITAB qit[] = {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4838)
#endif
		QITABENT(SampleGrabberCallback, IMFSampleGrabberSinkCallback),
		QITABENT(SampleGrabberCallback, IMFClockStateSink),
#ifdef _MSC_VER
#pragma warning(pop)
#endif
		{ 0, 0 }
	};
	return QISearch(this, qit, p_riid, p_ppv);
}

STDMETHODIMP_(ULONG)
SampleGrabberCallback::AddRef() {
	return InterlockedIncrement(&m_cRef);
}

STDMETHODIMP_(ULONG)
SampleGrabberCallback::Release() {
	ULONG cRef = InterlockedDecrement(&m_cRef);
	if (cRef == 0) {
		delete this;
	}
	return cRef;
}

STDMETHODIMP SampleGrabberCallback::OnClockStart(MFTIME p_hnsSystemTime, LONGLONG p_llClockStartOffset) {
	return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockStop(MFTIME p_hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockPause(MFTIME p_hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockRestart(MFTIME p_hnsSystemTime) {
	return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnClockSetRate(MFTIME p_hnsSystemTime, float p_flRate) {
	return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnSetPresentationClock(IMFPresentationClock *p_clock) {
	return S_OK;
}

HRESULT SampleGrabberCallback::CreateMediaSample(DWORD p_data, IMFSample **p_sample) {
	assert(p_sample);

	HRESULT hr = S_OK;

	IMFSample *pSample = nullptr;
	IMFMediaBuffer *pBuffer = nullptr;

	CHECK_HR(MFCreateSample(&pSample));
	CHECK_HR(MFCreateMemoryBuffer(p_data, &pBuffer));
	CHECK_HR(pSample->AddBuffer(pBuffer));

	*p_sample = pSample;
	(*p_sample)->AddRef();

	return hr;
}

STDMETHODIMP SampleGrabberCallback::OnProcessSample(REFGUID p_guid_major_media_type,
		DWORD dwSampleFlags,
		LONGLONG llSampleTime,
		LONGLONG llSampleDuration,
		const BYTE *pSampleBuffer,
		DWORD dwSampleSize) {
	HRESULT hr = S_OK;

	if (m_pColorTransform == nullptr) {
		print_line("Color transform is null!");
		return E_FAIL;
	}

	const int p_rgba32_frame_size = width * height * 4; // RGB32 = 4 bytes per pixel (RGBA)

	// Create fresh samples for each frame to avoid reuse issues
	IMFSample *pInputSample = nullptr;
	IMFSample *p_output_sample = nullptr;

	CHECK_HR(CreateMediaSample(dwSampleSize, &pInputSample));
	if (FAILED(hr)) {
		print_line("Failed to create input sample");
		return hr;
	}

	CHECK_HR(CreateMediaSample(p_rgba32_frame_size, &p_output_sample));
	if (FAILED(hr)) {
		print_line("Failed to create output sample");
		if (pInputSample) {
			pInputSample->Release();
		}
		return hr;
	}

	// Set up input sample
	IMFMediaBuffer *pInputBuffer = nullptr;
	CHECK_HR(pInputSample->SetSampleTime(llSampleTime));
	CHECK_HR(pInputSample->SetSampleDuration(llSampleDuration));
	CHECK_HR(pInputSample->GetBufferByIndex(0, &pInputBuffer));

	BYTE *pInputData = nullptr;
	CHECK_HR(pInputBuffer->Lock(&pInputData, NULL, NULL));
	memcpy(pInputData, pSampleBuffer, dwSampleSize);
	CHECK_HR(pInputBuffer->SetCurrentLength(dwSampleSize));
	CHECK_HR(pInputBuffer->Unlock());

	// Process the color conversion
	DWORD ProcessStatus = 0;
	CHECK_HR(m_pColorTransform->ProcessInput(0, pInputSample, 0));
	if (FAILED(hr)) {
		print_line("Failed to process input: " + itos(hr));
		pInputSample->Release();
		p_output_sample->Release();
		return hr;
	}

	MFT_OUTPUT_DATA_BUFFER outputDataBuffer;
	outputDataBuffer.dwStreamID = 0;
	outputDataBuffer.dwStatus = 0;
	outputDataBuffer.pEvents = NULL;
	outputDataBuffer.pSample = p_output_sample;

	CHECK_HR(m_pColorTransform->ProcessOutput(0, 1, &outputDataBuffer, &ProcessStatus));
	if (FAILED(hr)) {
		print_line("Failed to process output: " + itos(hr));
		pInputSample->Release();
		p_output_sample->Release();
		return hr;
	}

	// Get the converted RGB data
	IMFMediaBuffer *pOutputBuffer = nullptr;
	CHECK_HR(outputDataBuffer.pSample->GetBufferByIndex(0, &pOutputBuffer));

	BYTE *outData = nullptr;
	DWORD outDataLen = 0;
	CHECK_HR(pOutputBuffer->Lock(&outData, NULL, &outDataLen));

	if (outDataLen == (DWORD)p_rgba32_frame_size) {
		FrameData *frame = playback->get_next_writable_frame();
		frame->sample_time = llSampleTime;

		uint8_t *dst = frame->data.ptrw();

		// WMF Video Processor outputs ARGB32, but Godot expects RGBA8
		// We need to swap R and B channels to fix the color issue
		const uint8_t *src = outData;
		const int pixel_count = width * height;

		for (int i = 0; i < pixel_count; i++) {
			// ARGB32 format: [A][R][G][B] (little-endian: B,G,R,A in memory)
			// RGBA8 format:  [R][G][B][A]
			// So we need to convert B,G,R,A -> R,G,B,A

			uint8_t b = src[i * 4 + 0]; // Blue
			uint8_t g = src[i * 4 + 1]; // Green
			uint8_t r = src[i * 4 + 2]; // Red
			uint8_t a = src[i * 4 + 3]; // Alpha

			// Write in RGBA order
			dst[i * 4 + 0] = r; // Red
			dst[i * 4 + 1] = g; // Green
			dst[i * 4 + 2] = b; // Blue
			dst[i * 4 + 3] = a; // Alpha
		}

		playback->write_frame_done();
	} else {
		print_line("Unexpected output data size: " + itos(outDataLen) + " expected: " + itos(p_rgba32_frame_size));
	}

	CHECK_HR(pOutputBuffer->Unlock());

	pInputSample->Release();
	p_output_sample->Release();

	return S_OK;
}

STDMETHODIMP SampleGrabberCallback::OnShutdown() {
	return S_OK;
}

void SampleGrabberCallback::set_frame_size(int p_width, int p_height) {
	width = p_width;
	height = p_height;
}
