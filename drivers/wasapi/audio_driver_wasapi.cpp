/**************************************************************************/
/*  audio_driver_wasapi.cpp                                               */
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

#ifdef WASAPI_ENABLED

#include "audio_driver_wasapi.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"

#include <stdint.h> // INT32_MAX

#include <functiondiscoverykeys.h>

#ifndef PKEY_Device_FriendlyName

#undef DEFINE_PROPERTYKEY
/* clang-format off */
#define DEFINE_PROPERTYKEY(id, a, b, c, d, e, f, g, h, i, j, k, l) \
	const PROPERTYKEY id = { { a, b, c, { d, e, f, g, h, i, j, k, } }, l };
/* clang-format on */

DEFINE_PROPERTYKEY(PKEY_Device_FriendlyName, 0xa45c254e, 0xdf1c, 0x4efd, 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0, 14);
#endif

const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
const IID IID_IAudioClient = __uuidof(IAudioClient);
const IID IID_IAudioRenderClient = __uuidof(IAudioRenderClient);
const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);

#define SAFE_RELEASE(memory)   \
	if ((memory) != nullptr) { \
		(memory)->Release();   \
		(memory) = nullptr;    \
	}

#define REFTIMES_PER_SEC 10000000
#define REFTIMES_PER_MILLISEC 10000

static bool default_output_device_changed = false;
static bool default_input_device_changed = false;

// Silence warning due to a COM API weirdness (GH-35194).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

class CMMNotificationClient : public IMMNotificationClient {
	LONG _cRef = 1;
	AudioDriverWASAPI *driver;

public:
	CMMNotificationClient() {}
	virtual ~CMMNotificationClient() {}

	void set_driver(AudioDriverWASAPI *p_driver) {
		driver = p_driver;
	}

	ULONG STDMETHODCALLTYPE AddRef() {
		return InterlockedIncrement(&_cRef);
	}

	ULONG STDMETHODCALLTYPE Release() {
		ULONG ulRef = InterlockedDecrement(&_cRef);
		if (0 == ulRef) {
			delete this;
		}
		return ulRef;
	}

	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, VOID **ppvInterface) {
		if (IID_IUnknown == riid) {
			AddRef();
			*ppvInterface = (IUnknown *)this;
		} else if (__uuidof(IMMNotificationClient) == riid) {
			AddRef();
			*ppvInterface = (IMMNotificationClient *)this;
		} else {
			*ppvInterface = nullptr;
			return E_NOINTERFACE;
		}
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceAdded(LPCWSTR pwstrDeviceId) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceRemoved(LPCWSTR pwstrDeviceId) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceStateChanged(LPCWSTR pwstrDeviceId, DWORD dwNewState) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDefaultDeviceChanged(EDataFlow flow, ERole role, LPCWSTR pwstrDeviceId) {
		if (role == eConsole) {
			if (driver) {
				driver->default_device_changed(flow);
			}
		}

		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnPropertyValueChanged(LPCWSTR pwstrDeviceId, const PROPERTYKEY key) {
		return S_OK;
	}
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

static CMMNotificationClient notif_client;

Error AudioDriverWASAPI::audio_device_init(AudioDeviceWASAPI *p_device, bool p_input, bool p_reinit) {
	WAVEFORMATEX *pwfex;
	IMMDeviceEnumerator *enumerator = nullptr;
	IMMDevice *output_device = nullptr;

	// TODO: Not sure if this has to be created over and over. Docs are also not clear if a new one has to created to receive fresh values.
	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	if (p_device->device_name == "Default") {
		hr = enumerator->GetDefaultAudioEndpoint(p_input ? eCapture : eRender, eConsole, &output_device);
	} else {
		IMMDeviceCollection *devices = nullptr;

		hr = enumerator->EnumAudioEndpoints(p_input ? eCapture : eRender, DEVICE_STATE_ACTIVE, &devices);
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

		LPWSTR strId = nullptr;
		bool found = false;

		UINT count = 0;
		hr = devices->GetCount(&count);
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

		for (ULONG i = 0; i < count && !found; i++) {
			IMMDevice *tmp_device = nullptr;

			hr = devices->Item(i, &tmp_device);
			ERR_BREAK(hr != S_OK);

			IPropertyStore *props = nullptr;
			hr = tmp_device->OpenPropertyStore(STGM_READ, &props);
			ERR_BREAK(hr != S_OK);

			PROPVARIANT propvar;
			PropVariantInit(&propvar);

			hr = props->GetValue(PKEY_Device_FriendlyName, &propvar);
			ERR_BREAK(hr != S_OK);

			if (p_device->device_name == String(propvar.pwszVal)) {
				hr = tmp_device->GetId(&strId);
				ERR_BREAK(hr != S_OK);

				found = true;
			}

			PropVariantClear(&propvar);
			props->Release();
			tmp_device->Release();
		}

		if (found) {
			hr = enumerator->GetDevice(strId, &output_device);
		}

		if (strId) {
			CoTaskMemFree(strId);
		}

		if (output_device == nullptr) {
			hr = enumerator->GetDefaultAudioEndpoint(p_input ? eCapture : eRender, eConsole, &output_device);
		}
	}

	if (p_reinit) {
		// In case we're trying to re-initialize the device, prevent throwing this error on the console,
		// otherwise if there is currently no device available this will spam the console.
		if (hr != S_OK) {
			return ERR_CANT_OPEN;
		}
	} else {
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);
	}

	// Receive notifications when the user changes the output device.
	// This is a bit smarter in XAudio2 since it uses a virtual output that automatically redirects the output to the active output device, which
	// means it also automatically handles device loss.
	hr = enumerator->RegisterEndpointNotificationCallback(&notif_client);
	SAFE_RELEASE(enumerator)

	if (hr != S_OK) {
		ERR_PRINT("WASAPI: RegisterEndpointNotificationCallback error");
	}

	hr = output_device->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, (void **)&p_device->audio_client);

	SAFE_RELEASE(output_device)

	if (p_reinit) {
		if (hr != S_OK) {
			return ERR_CANT_OPEN;
		}
	} else {
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);
	}

	// Since we open in shared mode, this will return the wave format that all shared audio clients must use.
	hr = p_device->audio_client->GetMixFormat(&pwfex);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	print_verbose("WASAPI: wFormatTag = " + itos(pwfex->wFormatTag));
	print_verbose("WASAPI: nChannels = " + itos(pwfex->nChannels));
	print_verbose("WASAPI: nSamplesPerSec = " + itos(pwfex->nSamplesPerSec));
	print_verbose("WASAPI: nAvgBytesPerSec = " + itos(pwfex->nAvgBytesPerSec));
	print_verbose("WASAPI: nBlockAlign = " + itos(pwfex->nBlockAlign));
	print_verbose("WASAPI: wBitsPerSample = " + itos(pwfex->wBitsPerSample));
	print_verbose("WASAPI: cbSize = " + itos(pwfex->cbSize));

	// This can be adjusted if needed. But doesn't look like Godot has support for setting the
	// wanted channels, and instead just use what we say in get_speaker_mode.
	p_device->channels = pwfex->nChannels;

	// Since we're using WASAPI Shared Mode we can't control any of these, we just tag along
	p_device->format_tag = pwfex->wFormatTag;
	p_device->bits_per_sample = pwfex->wBitsPerSample;

	// Store the format tag so we can read it easier.
	if (p_device->format_tag == WAVE_FORMAT_EXTENSIBLE) {
		WAVEFORMATEXTENSIBLE *wfex = (WAVEFORMATEXTENSIBLE *)pwfex;

		if (wfex->SubFormat == KSDATAFORMAT_SUBTYPE_PCM) {
			p_device->format_tag = WAVE_FORMAT_PCM;
		} else if (wfex->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT) {
			p_device->format_tag = WAVE_FORMAT_IEEE_FLOAT;
		} else {
			ERR_PRINT("WASAPI: Format not supported");
			ERR_FAIL_V(ERR_CANT_OPEN);
		}
	} else {
		if (p_device->format_tag != WAVE_FORMAT_PCM && p_device->format_tag != WAVE_FORMAT_IEEE_FLOAT) {
			ERR_PRINT("WASAPI: Format not supported");
			ERR_FAIL_V(ERR_CANT_OPEN);
		}
	}

	// Let WASAPI do sample rate conversion and up/downsample channels internally if needed.
	// This will hit if the project configuration uses a different sample rate or channels than the WASAPI shared engine.
	// Writing a good resampler is very non-trivial so we should absolutely favor this as long as possible.
	DWORD streamflags = 0;
	if ((DWORD)mix_rate != pwfex->nSamplesPerSec) {
		// Default quality means that it is intended to be audible, and not using shortcuts for meter calculations.
		streamflags |= AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
		pwfex->nSamplesPerSec = mix_rate;
		pwfex->nAvgBytesPerSec = pwfex->nSamplesPerSec * pwfex->nChannels * (pwfex->wBitsPerSample / 8);
	}

	if (!p_input) {
		// Let WASAPI notify us when it wants more samples (push method).
		streamflags |= AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
	}

	// We open in shared mode. Exclusive mode is more intricate and can result in lower latency (can even have DMA writes to external hardware)
	// however as it is exclusive, other programs will not be able to play to the speakers anymore which is not suitable for most applications.
	// This function will fail if the chosen output device is currently opened in another application in exclusive mode.
	//
	// Passing in 0 here for the buffer duration means that we will get the smallest possible buffer (also lowest latency).
	hr = p_device->audio_client->Initialize(AUDCLNT_SHAREMODE_SHARED, streamflags, p_input ? REFTIMES_PER_SEC : 0, 0, pwfex, nullptr);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_CANT_OPEN, "WASAPI: Initialize failed with error: 0x" + String::num_uint64(hr, 16) + ".");

	if (!p_input) {
		// This is the event that WASAPI will signal when it wants more data from us (push method).
		// The initial state for this event must be unsignaled!
		p_device->feed_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
		ERR_FAIL_COND_V_MSG(p_device->feed_event == nullptr, ERR_CANT_OPEN, "WASAPI: Could not create event handle with error: " + String::num(GetLastError()) + ".");

		hr = p_device->audio_client->SetEventHandle(p_device->feed_event);
		ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_CANT_OPEN, "WASAPI: Could not set event handle with error: 0x" + String::num_uint64(hr, 16) + ".");
	}

	// We send in 0 as the buffer size for Initialize, which means the WASAPI engine will automatically assign the smallest buffer size it can use.
	// This value is the maximum size that can be written to a buffer received from GetBuffer, though it is allowed to write less.
	// The WASAPI engine will deny values that are lower than its internal minimal value.
	// Therefore it is important to call this so we can exactly know what we get.
	UINT32 max_frames;
	hr = p_device->audio_client->GetBufferSize(&max_frames);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	if (!p_input) {
		buffer_frames = max_frames;

		// Note that this will always return 0 because we are using the default buffer size
		// with the push method, so we are always topping off without waiting.
		int64_t latency = 0;
		audio_output.audio_client->GetStreamLatency(&latency);

		// WASAPI REFERENCE_TIME units are 100 nanoseconds per unit
		// https://docs.microsoft.com/en-us/windows/win32/directshow/reference-time
		// Convert REFTIME to seconds as godot uses for latency
		real_latency = (float)latency / (float)REFTIMES_PER_SEC;
	}

	if (p_input) {
		hr = p_device->audio_client->GetService(IID_IAudioCaptureClient, (void **)&p_device->capture_client);
	} else {
		hr = p_device->audio_client->GetService(IID_IAudioRenderClient, (void **)&p_device->render_client);
	}
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	// Free memory
	CoTaskMemFree(pwfex);
	SAFE_RELEASE(output_device)

	return OK;
}

Error AudioDriverWASAPI::init_output_device(bool p_reinit) {
	Error err = audio_device_init(&audio_output, false, p_reinit);
	if (err != OK) {
		return err;
	}

	// Sample rate is independent of channels (ref: https://stackoverflow.com/questions/11048825/audio-sample-frequency-rely-on-channels)
	samples_in.resize(buffer_frames * audio_output.channels);

	// FIXME: Why is this capture state reset in the output function?
	input_position = 0;
	input_size = 0;

	print_verbose("WASAPI: detected " + itos(audio_output.channels) + " channels");
	print_verbose("WASAPI: audio buffer frames: " + itos(buffer_frames) + " calculated latency: " + itos(buffer_frames * 1000 / mix_rate) + "ms");

	return OK;
}

Error AudioDriverWASAPI::init_input_device(bool p_reinit) {
	Error err = audio_device_init(&audio_input, true, p_reinit);
	if (err != OK) {
		return err;
	}

	// Get the max frames
	UINT32 max_frames;
	HRESULT hr = audio_input.audio_client->GetBufferSize(&max_frames);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	input_buffer_init(max_frames);

	return OK;
}

void AudioDriverWASAPI::audio_device_finish(AudioDeviceWASAPI *p_device) {
	if (p_device->audio_client) {
		p_device->audio_client->Stop();
	}

	// Only used for capture.
	p_device->active.clear();

	if (p_device->feed_event) {
		CloseHandle(p_device->feed_event);
		p_device->feed_event = nullptr;
	}

	SAFE_RELEASE(p_device->audio_client)
	SAFE_RELEASE(p_device->render_client)
	SAFE_RELEASE(p_device->capture_client)
}

void AudioDriverWASAPI::finish_output_device() {
	audio_device_finish(&audio_output);
}

void AudioDriverWASAPI::finish_input_device() {
	audio_device_finish(&audio_input);
}

// Only to be called in the render thread after init has already happened!
Error AudioDriverWASAPI::remake_output_device() {
	finish_output_device();

	// At this point the wake event will be signaled. If the remake works, then the event will become signaled again and everything works.
	// If the remake fails, then it will remain unsignaled and the render thread goes back to waiting for a working output device.

	Error err = init_output_device(true); // This will fail if the device isn't in the system anymore.
	if (err == OK) {
		start(); // Will signal the wake event.
	} else {
		ResetEvent(render_wake); // Not active anymore. Put to waiting.
	}

	return err;
}

Error AudioDriverWASAPI::init() {
	mix_rate = _get_configured_mix_rate();

	// This below is cryptic but it is assumed that if we fail to initialize now, then maybe a new working device will be inserted to the system later.
	Error err = init_output_device();
	if (err != OK) {
		ERR_PRINT("WASAPI: init_output_device error");
	}

	// The initial state for this event must be unsignaled!
	// Literally impossible for this call to fail. You be the judge if you need to check the return value of this.
	// It is however fundamental to the logic for this to exist.
	render_wake = CreateEventW(nullptr, TRUE, FALSE, nullptr);
	ERR_FAIL_COND_V_MSG(render_wake == nullptr, ERR_CANT_OPEN, "WASAPI: Could not create render wake handle with error: " + String::num(GetLastError()) + ".");

	exit_thread.clear();

	render_thread.start(render_thread_func, this);
	capture_thread.start(capture_thread_func, this);

	notif_client.set_driver(this);

	return OK;
}

int AudioDriverWASAPI::get_mix_rate() const {
	return mix_rate;
}

float AudioDriverWASAPI::get_latency() {
	return real_latency;
}

AudioDriver::SpeakerMode AudioDriverWASAPI::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(audio_output.channels);
}

PackedStringArray AudioDriverWASAPI::audio_device_get_list(bool p_input) {
	PackedStringArray list;
	IMMDeviceCollection *devices = nullptr;
	IMMDeviceEnumerator *enumerator = nullptr;

	list.push_back(String("Default"));

	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, PackedStringArray());

	hr = enumerator->EnumAudioEndpoints(p_input ? eCapture : eRender, DEVICE_STATE_ACTIVE, &devices);
	ERR_FAIL_COND_V(hr != S_OK, PackedStringArray());

	UINT count = 0;
	hr = devices->GetCount(&count);
	ERR_FAIL_COND_V(hr != S_OK, PackedStringArray());

	for (ULONG i = 0; i < count; i++) {
		IMMDevice *output_device = nullptr;

		hr = devices->Item(i, &output_device);
		ERR_BREAK(hr != S_OK);

		IPropertyStore *props = nullptr;
		hr = output_device->OpenPropertyStore(STGM_READ, &props);
		ERR_BREAK(hr != S_OK);

		PROPVARIANT propvar;
		PropVariantInit(&propvar);

		hr = props->GetValue(PKEY_Device_FriendlyName, &propvar);
		ERR_BREAK(hr != S_OK);

		list.push_back(String(propvar.pwszVal));

		PropVariantClear(&propvar);
		props->Release();
		output_device->Release();
	}

	devices->Release();
	enumerator->Release();
	return list;
}

void AudioDriverWASAPI::lock_render() {
	audio_output.mutex.lock();
}

void AudioDriverWASAPI::unlock_render() {
	audio_output.mutex.unlock();
}

void AudioDriverWASAPI::lock_capture() {
	audio_input.mutex.lock();
}

void AudioDriverWASAPI::unlock_capture() {
	audio_input.mutex.unlock();
}

// Called in the main thread from notif_client.
// This is messy because Godot does not have any cross thread communication. Anyway.
// The logic for remaking the devices is done on the respective thread to keep it a little bit simpler.

void AudioDriverWASAPI::default_device_changed(EDataFlow p_flow) {
	if (p_flow == eRender) {
		default_output_device_changed = true;
		SetEvent(render_wake); // Wake if render thread is currently waiting for a working output device.
	} else if (p_flow == eCapture) {
		default_input_device_changed = true;
	}
}

PackedStringArray AudioDriverWASAPI::get_output_device_list() {
	return audio_device_get_list(false);
}

String AudioDriverWASAPI::get_output_device() {
	lock_render();
	String name = audio_output.device_name;
	unlock_render();

	return name;
}

void AudioDriverWASAPI::set_output_device(const String &p_name) {
	lock_render();
	audio_output.new_device = p_name;
	unlock_render();
}

int32_t AudioDriverWASAPI::read_sample(WORD format_tag, int bits_per_sample, BYTE *buffer, int i) {
	if (format_tag == WAVE_FORMAT_PCM) {
		int32_t sample = 0;
		switch (bits_per_sample) {
			case 8:
				sample = int32_t(((int8_t *)buffer)[i]) << 24;
				break;

			case 16:
				sample = int32_t(((int16_t *)buffer)[i]) << 16;
				break;

			case 24:
				sample |= int32_t(((int8_t *)buffer)[i * 3 + 2]) << 24;
				sample |= int32_t(((int8_t *)buffer)[i * 3 + 1]) << 16;
				sample |= int32_t(((int8_t *)buffer)[i * 3 + 0]) << 8;
				break;

			case 32:
				sample = ((int32_t *)buffer)[i];
				break;
		}

		return sample;
	} else if (format_tag == WAVE_FORMAT_IEEE_FLOAT) {
		return int32_t(((float *)buffer)[i] * 32768.0) << 16;
	} else {
		ERR_PRINT("WASAPI: Unknown format tag");
	}

	return 0;
}

void AudioDriverWASAPI::write_sample(WORD format_tag, int bits_per_sample, BYTE *buffer, unsigned int i, int32_t sample) {
	if (format_tag == WAVE_FORMAT_PCM) {
		switch (bits_per_sample) {
			case 8:
				((int8_t *)buffer)[i] = sample >> 24;
				break;

			case 16:
				((int16_t *)buffer)[i] = sample >> 16;
				break;

			case 24:
				((int8_t *)buffer)[i * 3 + 2] = sample >> 24;
				((int8_t *)buffer)[i * 3 + 1] = sample >> 16;
				((int8_t *)buffer)[i * 3 + 0] = sample >> 8;
				break;

			case 32:
				((int32_t *)buffer)[i] = sample;
				break;
		}
	} else if (format_tag == WAVE_FORMAT_IEEE_FLOAT) {
		((float *)buffer)[i] = (sample >> 16) / 32768.f;
	} else {
		ERR_PRINT("WASAPI: Unknown format tag");
	}
}

// Render and capture are using different threads to not disturb each other, especially since we use the push method for rendering.
void AudioDriverWASAPI::render_thread_func(void *p_udata) {
	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

	AudioDriverWASAPI *ad = static_cast<AudioDriverWASAPI *>(p_udata);
	AudioDeviceWASAPI *dev = &ad->audio_output;

	while (!ad->exit_thread.is_set()) {
		// First wait to become active at all, but also wait if we don't have any device.
		WaitForSingleObjectEx(ad->render_wake, INFINITE, FALSE);

		// Let's specially handle the case when woken up before close.
		if (ad->exit_thread.is_set()) {
			break;
		}

		// So we are not closing. This hits if we have been woken up to remake the output device.
		// Really messy because Godot doesn't have any thread communication.
		if (dev->audio_client == nullptr) {
			if (default_output_device_changed) {
				default_output_device_changed = false;
				Error err = ad->remake_output_device();
				if (err != OK) {
					continue; // Still not working. Back to waiting.
				}
			}

			// This will not hit, but probably better for safety.
			if (dev->audio_client == nullptr) {
				ResetEvent(ad->render_wake);
				continue;
			}

			// If we get here, we have transitioned to a new device and things are back to normal.
		}

		// Play out the engine samples to the speakers.

		HRESULT hr = E_FAIL; // Annoyingly, the device may be removed at any point so we have to constantly check the return values.
		UINT32 padding = 0; // How many samples are in the WASAPI buffer already.
		bool need_to_remake_device = false; // Will be set in case device disappears, or if the user changes device.

		while (true) {
			// Wait for WASAPI to notify us that it wants more samples (push method).
			DWORD waited = WaitForSingleObjectEx(dev->feed_event, 200, FALSE);

			// Buffer needs topping up. Note that in exclusive mode you don't need to do this, as you have control yourself
			// over how much or how little you want to write.
			if (waited == WAIT_OBJECT_0) {
				hr = dev->audio_client->GetCurrentPadding(&padding);

				if (FAILED(hr)) {
					break;
				}

				// TODO: Should probably have a threshold here so you dont potentially wake up to process 1 sample.
				if (padding < ad->buffer_frames) {
					break;
				}
			} else {
				// This is an error.
				if (waited != WAIT_TIMEOUT) {
					hr = HRESULT_FROM_WIN32(GetLastError());
					break;
				}
			}
		}

		if (SUCCEEDED(hr)) {
			// This is how many frames we need to receive from the engine.
			DWORD avail_frames = ad->buffer_frames - padding;

			// Receive needed samples from engine.
			// TODO: Does this always need to be called?
			// TODO: What format are we even getting here? Does Godot just reconfigure itself to what our wave format?
			ad->audio_server_process(avail_frames, ad->samples_in.ptrw());

			// Receive a buffer from WASAPI that we can write to.
			// Note that this pointer will not point to the beginning of the buffer if the current padding
			// is greater than zero (as the existing will be data placed before).
			BYTE *buf;
			hr = dev->render_client->GetBuffer(avail_frames, &buf);

			if (SUCCEEDED(hr)) {
				// Put the engine samples into the WASAPI buffer, converting the sample format as needed.

				for (unsigned int i = 0; i < avail_frames * dev->channels; i++) {
					write_sample(dev->format_tag, dev->bits_per_sample, buf, i, ad->samples_in.write[i]);
				}

				// Now we are done, blast to the speakers.
				hr = dev->render_client->ReleaseBuffer(avail_frames, 0);
			}
		}

		// Below is either transition to another device or transition to no device.
		// Either case we have to remake our state.

		// So the device got removed or something. Unfortunately WASAPI does not have a virtual output device (like XAudio2)
		// which would automatically handle annoying cases like this. In this case we have to remake everything
		if (FAILED(hr)) {
			ERR_PRINT("WASAPI: Device error: 0x" + String::num_uint64(hr, 16) + ".");
			need_to_remake_device = true;
		}

		// Have to lock access to the device names.
		ad->lock_render();

		// This hits if the user changed the Windows default device. This new device may have new caps so we have to remake everything.
		if (dev->device_name == "Default" && default_output_device_changed) {
			need_to_remake_device = true;
			default_output_device_changed = false;
		}

		// This hits if the engine is using a custom audio output that is not the default. Very rare case.

		if (dev->device_name != dev->new_device) {
			dev->device_name = dev->new_device;
			need_to_remake_device = true;
		}

		ad->unlock_render();

		// This may not succeed. If the output device is completely removed from the system then remaking will fail, but we cannot know that for sure.
		// In case of failure, we will hit the outside path to wait until a new working output device is added to the system.
		if (need_to_remake_device) {
			need_to_remake_device = false;
			ad->remake_output_device();
		}
	}

	CoUninitialize();
}

void AudioDriverWASAPI::capture_thread_func(void *p_udata) {
	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

	AudioDriverWASAPI *ad = static_cast<AudioDriverWASAPI *>(p_udata);

	while (!ad->exit_thread.is_set()) {
		uint32_t read_frames = 0;

		ad->lock_capture();
		ad->start_counting_ticks();

		if (ad->audio_input.active.is_set()) {
			UINT32 packet_length = 0;
			BYTE *data;
			UINT32 num_frames_available;
			DWORD flags;

			HRESULT hr = ad->audio_input.capture_client->GetNextPacketSize(&packet_length);
			if (hr == S_OK) {
				while (packet_length != 0) {
					hr = ad->audio_input.capture_client->GetBuffer(&data, &num_frames_available, &flags, nullptr, nullptr);
					ERR_BREAK(hr != S_OK);

					// fixme: Only works for floating point atm
					for (UINT32 j = 0; j < num_frames_available; j++) {
						int32_t l, r;

						if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
							l = r = 0;
						} else {
							if (ad->audio_input.channels == 2) {
								l = read_sample(ad->audio_input.format_tag, ad->audio_input.bits_per_sample, data, j * 2);
								r = read_sample(ad->audio_input.format_tag, ad->audio_input.bits_per_sample, data, j * 2 + 1);
							} else if (ad->audio_input.channels == 1) {
								l = r = read_sample(ad->audio_input.format_tag, ad->audio_input.bits_per_sample, data, j);
							} else {
								l = r = 0;
								ERR_PRINT("WASAPI: unsupported channel count in microphone!");
							}
						}

						ad->input_buffer_write(l);
						ad->input_buffer_write(r);
					}

					read_frames += num_frames_available;

					hr = ad->audio_input.capture_client->ReleaseBuffer(num_frames_available);
					ERR_BREAK(hr != S_OK);

					hr = ad->audio_input.capture_client->GetNextPacketSize(&packet_length);
					ERR_BREAK(hr != S_OK);
				}
			}

			// If we're using the Default output device and it changed finish it so we'll re-init the output device
			if (ad->audio_input.device_name == "Default" && default_input_device_changed) {
				ad->finish_input_device();
				default_input_device_changed = false;
			}

			// User selected a new input device, finish the current one so we'll init the new input device
			if (ad->audio_input.device_name != ad->audio_input.new_device) {
				ad->audio_input.device_name = ad->audio_input.new_device;
				ad->finish_input_device();
			}

			if (!ad->audio_input.audio_client) {
				Error err = ad->init_input_device(true);
				if (err == OK) {
					ad->input_start();
				}
			}
		}

		ad->stop_counting_ticks();
		ad->unlock_capture();

		// Let the thread rest a while if we haven't read or write anything
		if (read_frames == 0) {
			// FIXME: This should be removed in favor of being event driven.
			OS::get_singleton()->delay_usec(1000);
		}
	}
	CoUninitialize();
}

void AudioDriverWASAPI::start() {
	if (audio_output.audio_client) {
		HRESULT hr = audio_output.audio_client->Start();
		if (SUCCEEDED(hr)) {
			SetEvent(render_wake); // Now active.
		} else {
			ERR_PRINT("WASAPI: Start failed.");
		}
	}
}

// FIXME: Lock what?
void AudioDriverWASAPI::lock() {
	lock_render();
	lock_capture();
}

// FIXME: Unlock what?
void AudioDriverWASAPI::unlock() {
	unlock_render();
	unlock_capture();
}

void AudioDriverWASAPI::finish() {
	exit_thread.set();

	SetEvent(render_wake); // Wake up to exit if we are not active already.

	if (render_thread.is_started()) {
		render_thread.wait_to_finish();
	}

	if (capture_thread.is_started()) {
		capture_thread.wait_to_finish();
	}

	CloseHandle(render_wake);

	finish_input_device();
	finish_output_device();

	notif_client.set_driver(nullptr);
}

Error AudioDriverWASAPI::input_start() {
	Error err = init_input_device();
	if (err != OK) {
		ERR_PRINT("WASAPI: init_input_device error");
		return err;
	}

	if (audio_input.active.is_set()) {
		return FAILED;
	}

	audio_input.audio_client->Start();
	audio_input.active.set();
	return OK;
}

Error AudioDriverWASAPI::input_stop() {
	if (audio_input.active.is_set()) {
		audio_input.audio_client->Stop();
		audio_input.active.clear();

		return OK;
	}

	return FAILED;
}

PackedStringArray AudioDriverWASAPI::get_input_device_list() {
	return audio_device_get_list(true);
}

String AudioDriverWASAPI::get_input_device() {
	lock_capture();
	String name = audio_input.device_name;
	unlock_capture();

	return name;
}

void AudioDriverWASAPI::set_input_device(const String &p_name) {
	lock_capture();
	audio_input.new_device = p_name;
	unlock_capture();
}

AudioDriverWASAPI::AudioDriverWASAPI() {
	samples_in.clear();
}

#endif // WASAPI_ENABLED
