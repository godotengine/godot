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

#include <functiondiscoverykeys.h>

// Define IAudioClient3 if not already defined by MinGW headers.
#if defined __MINGW32__ || defined __MINGW64__

#ifndef __IAudioClient3_FWD_DEFINED__
#define __IAudioClient3_FWD_DEFINED__

typedef interface IAudioClient3 IAudioClient3;

#endif // __IAudioClient3_FWD_DEFINED__

#ifndef __IAudioClient3_INTERFACE_DEFINED__
#define __IAudioClient3_INTERFACE_DEFINED__

// `IAudioClient2` has a non-virtual destructor.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
MIDL_INTERFACE("7ED4EE07-8E67-4CD4-8C1A-2B7A5987AD42")
IAudioClient3 : public IAudioClient2 {
#pragma GCC diagnostic pop
public:
	virtual HRESULT STDMETHODCALLTYPE GetSharedModeEnginePeriod(
			/* [annotation][in] */
			_In_ const WAVEFORMATEX *pFormat,
			/* [annotation][out] */
			_Out_ UINT32 *pDefaultPeriodInFrames,
			/* [annotation][out] */
			_Out_ UINT32 *pFundamentalPeriodInFrames,
			/* [annotation][out] */
			_Out_ UINT32 *pMinPeriodInFrames,
			/* [annotation][out] */
			_Out_ UINT32 *pMaxPeriodInFrames) = 0;

	virtual HRESULT STDMETHODCALLTYPE GetCurrentSharedModeEnginePeriod(
			/* [unique][annotation][out] */
			_Out_ WAVEFORMATEX * *ppFormat,
			/* [annotation][out] */
			_Out_ UINT32 * pCurrentPeriodInFrames) = 0;

	virtual HRESULT STDMETHODCALLTYPE InitializeSharedAudioStream(
			/* [annotation][in] */
			_In_ DWORD StreamFlags,
			/* [annotation][in] */
			_In_ UINT32 PeriodInFrames,
			/* [annotation][in] */
			_In_ const WAVEFORMATEX *pFormat,
			/* [annotation][in] */
			_In_opt_ LPCGUID AudioSessionGuid) = 0;
};
__CRT_UUID_DECL(IAudioClient3, 0x7ED4EE07, 0x8E67, 0x4CD4, 0x8C, 0x1A, 0x2B, 0x7A, 0x59, 0x87, 0xAD, 0x42)

#endif // __IAudioClient3_INTERFACE_DEFINED__

#endif // __MINGW32__ || __MINGW64__

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
const IID IID_IAudioClient3 = __uuidof(IAudioClient3);
const IID IID_IAudioRenderClient = __uuidof(IAudioRenderClient);
const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);

#define REFTIMES_PER_SEC 10000000
#define REFTIMES_PER_MILLISEC 10000

#define CAPTURE_BUFFER_CHANNELS 2

static bool default_output_device_changed = false;
static bool default_input_device_changed = false;

// Silence warning due to a COM API weirdness (GH-35194).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif

class CMMNotificationClient : public IMMNotificationClient {
	LONG _cRef = 1;

public:
	CMMNotificationClient() {}
	virtual ~CMMNotificationClient() {}

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
			if (flow == eRender) {
				default_output_device_changed = true;
			} else if (flow == eCapture) {
				default_input_device_changed = true;
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

Error AudioDriverWASAPI::audio_device_init(AudioDeviceWASAPI *p_device, bool p_input, bool p_reinit, bool p_no_audio_client_3) {
	WAVEFORMATEX *pwfex;
	ComPtr<IMMDeviceEnumerator> enumerator = nullptr;
	ComPtr<IMMDevice> output_device = nullptr;

	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	if (p_device->device_name == "Default") {
		hr = enumerator->GetDefaultAudioEndpoint(p_input ? eCapture : eRender, eConsole, &output_device);
	} else {
		ComPtr<IMMDeviceCollection> devices = nullptr;

		hr = enumerator->EnumAudioEndpoints(p_input ? eCapture : eRender, DEVICE_STATE_ACTIVE, &devices);
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

		LPWSTR strId = nullptr;
		bool found = false;

		UINT count = 0;
		hr = devices->GetCount(&count);
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

		for (ULONG i = 0; i < count && !found; i++) {
			ComPtr<IMMDevice> tmp_device = nullptr;

			hr = devices->Item(i, &tmp_device);
			ERR_BREAK(hr != S_OK);

			ComPtr<IPropertyStore> props = nullptr;
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

	hr = enumerator->RegisterEndpointNotificationCallback(&notif_client);

	if (hr != S_OK) {
		ERR_PRINT("WASAPI: RegisterEndpointNotificationCallback error");
	}

	using_audio_client_3 = !p_input; // IID_IAudioClient3 is only used for adjustable output latency (not input).

	if (p_no_audio_client_3) {
		using_audio_client_3 = false;
	}

	if (using_audio_client_3) {
		hr = output_device->Activate(IID_IAudioClient3, CLSCTX_ALL, nullptr, (void **)&p_device->audio_client);
		if (hr != S_OK) {
			// IID_IAudioClient3 will never activate on OS versions before Windows 10.
			// Older Windows versions should fall back gracefully.
			using_audio_client_3 = false;
			print_verbose("WASAPI: Couldn't activate output_device with IAudioClient3 interface, falling back to IAudioClient interface");
		} else {
			print_verbose("WASAPI: Activated output_device using IAudioClient3 interface");
		}
	}
	if (!using_audio_client_3) {
		hr = output_device->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, (void **)&p_device->audio_client);
	}

	if (p_reinit) {
		if (hr != S_OK) {
			return ERR_CANT_OPEN;
		}
	} else {
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);
	}

	if (using_audio_client_3) {
		AudioClientProperties audioProps{};
		audioProps.cbSize = sizeof(AudioClientProperties);
		audioProps.bIsOffload = FALSE;
		audioProps.eCategory = AudioCategory_GameEffects;

		hr = ((IAudioClient3 *)p_device->audio_client.Get())->SetClientProperties(&audioProps);
		ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_CANT_OPEN, "WASAPI: SetClientProperties failed with error 0x" + String::num_uint64(hr, 16) + ".");
	}

	hr = p_device->audio_client->GetMixFormat(&pwfex);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	print_verbose("WASAPI: wFormatTag = " + itos(pwfex->wFormatTag));
	print_verbose("WASAPI: nChannels = " + itos(pwfex->nChannels));
	print_verbose("WASAPI: nSamplesPerSec = " + itos(pwfex->nSamplesPerSec));
	print_verbose("WASAPI: nAvgBytesPerSec = " + itos(pwfex->nAvgBytesPerSec));
	print_verbose("WASAPI: nBlockAlign = " + itos(pwfex->nBlockAlign));
	print_verbose("WASAPI: wBitsPerSample = " + itos(pwfex->wBitsPerSample));
	print_verbose("WASAPI: cbSize = " + itos(pwfex->cbSize));

	WAVEFORMATEX *closest = nullptr;
	hr = p_device->audio_client->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, pwfex, &closest);
	if (hr == S_FALSE) {
		WARN_PRINT("WASAPI: Mix format is not supported by the output_device");
		if (closest) {
			print_verbose("WASAPI: closest->wFormatTag = " + itos(closest->wFormatTag));
			print_verbose("WASAPI: closest->nChannels = " + itos(closest->nChannels));
			print_verbose("WASAPI: closest->nSamplesPerSec = " + itos(closest->nSamplesPerSec));
			print_verbose("WASAPI: closest->nAvgBytesPerSec = " + itos(closest->nAvgBytesPerSec));
			print_verbose("WASAPI: closest->nBlockAlign = " + itos(closest->nBlockAlign));
			print_verbose("WASAPI: closest->wBitsPerSample = " + itos(closest->wBitsPerSample));
			print_verbose("WASAPI: closest->cbSize = " + itos(closest->cbSize));

			WARN_PRINT("WASAPI: Using closest match instead");
			pwfex = closest;
		}
	}

	// Since we're using WASAPI Shared Mode we can't control any of these, we just tag along.
	p_device->channels = pwfex->nChannels;
	p_device->frame_size = (pwfex->wBitsPerSample / 8) * p_device->channels;

	WORD format_tag = pwfex->wFormatTag;
	if (format_tag == WAVE_FORMAT_EXTENSIBLE) {
		WAVEFORMATEXTENSIBLE *wfex = (WAVEFORMATEXTENSIBLE *)pwfex;

		if (wfex->SubFormat == KSDATAFORMAT_SUBTYPE_PCM) {
			format_tag = WAVE_FORMAT_PCM;
		} else if (wfex->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT) {
			format_tag = WAVE_FORMAT_IEEE_FLOAT;
		} else {
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "WASAPI: Format not supported.");
		}
	} else {
		if (format_tag != WAVE_FORMAT_PCM && format_tag != WAVE_FORMAT_IEEE_FLOAT) {
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "WASAPI: Format not supported.");
		}
	}

	if (format_tag == WAVE_FORMAT_PCM) {
		switch (pwfex->wBitsPerSample) {
			case 8:
				p_device->buffer_format = BUFFER_FORMAT_INTEGER_8;
				break;

			case 16:
				p_device->buffer_format = BUFFER_FORMAT_INTEGER_16;
				break;

			case 24:
				p_device->buffer_format = BUFFER_FORMAT_INTEGER_24;
				break;

			case 32:
				p_device->buffer_format = BUFFER_FORMAT_INTEGER_32;
				break;

			default:
				ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("WASAPI: Unsupported bits per sample: %d.", pwfex->wBitsPerSample));
				break;
		}
	} else {
		p_device->buffer_format = BUFFER_FORMAT_FLOAT;
	}

	if (!using_audio_client_3) {
		DWORD streamflags = 0;
		if ((DWORD)mix_rate != pwfex->nSamplesPerSec) {
			streamflags |= AUDCLNT_STREAMFLAGS_RATEADJUST;
			pwfex->nSamplesPerSec = mix_rate;
			pwfex->nAvgBytesPerSec = pwfex->nSamplesPerSec * pwfex->nChannels * (pwfex->wBitsPerSample / 8);
		}
		hr = p_device->audio_client->Initialize(AUDCLNT_SHAREMODE_SHARED, streamflags, p_input ? REFTIMES_PER_SEC : 0, 0, pwfex, nullptr);
		ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_CANT_OPEN, "WASAPI: Initialize failed with error 0x" + String::num_uint64(hr, 16) + ".");
		UINT32 max_frames;
		hr = p_device->audio_client->GetBufferSize(&max_frames);
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

		// Due to WASAPI Shared Mode we have no control of the buffer size.
		if (!p_input) {
			buffer_frames = max_frames;

			int64_t latency = 0;
			audio_output.audio_client->GetStreamLatency(&latency);
			// WASAPI REFERENCE_TIME units are 100 nanoseconds per unit.
			// Ref: https://docs.microsoft.com/en-us/windows/win32/directshow/reference-time.
			// Convert REFTIME to seconds as godot uses for latency.
			real_latency = (float)latency / (float)REFTIMES_PER_SEC;
		}

	} else {
		IAudioClient3 *device_audio_client_3 = (IAudioClient3 *)p_device->audio_client.Get();

		// AUDCLNT_STREAMFLAGS_RATEADJUST is an invalid flag with IAudioClient3, therefore we have to use
		// the closest supported mix rate supported by the audio driver.
		mix_rate = pwfex->nSamplesPerSec;
		print_verbose("WASAPI: mix_rate = " + itos(mix_rate));

		UINT32 default_period_frames, fundamental_period_frames, min_period_frames, max_period_frames;
		hr = device_audio_client_3->GetSharedModeEnginePeriod(
				pwfex,
				&default_period_frames,
				&fundamental_period_frames,
				&min_period_frames,
				&max_period_frames);
		if (hr != S_OK) {
			print_verbose("WASAPI: GetSharedModeEnginePeriod failed with error 0x" + String::num_uint64(hr, 16) + ", falling back to IAudioClient.");
			CoTaskMemFree(pwfex);
			return audio_device_init(p_device, p_input, p_reinit, true);
		}

		// Period frames must be an integral multiple of fundamental_period_frames or IAudioClient3 initialization will fail,
		// so we need to select the closest multiple to the user-specified latency.
		UINT32 desired_period_frames = target_latency_ms * mix_rate / 1000;
		UINT32 period_frames = (desired_period_frames / fundamental_period_frames) * fundamental_period_frames;
		if (ABS((int64_t)period_frames - (int64_t)desired_period_frames) > ABS((int64_t)(period_frames + fundamental_period_frames) - (int64_t)desired_period_frames)) {
			period_frames = period_frames + fundamental_period_frames;
		}
		period_frames = CLAMP(period_frames, min_period_frames, max_period_frames);
		print_verbose("WASAPI: fundamental_period_frames = " + itos(fundamental_period_frames));
		print_verbose("WASAPI: min_period_frames = " + itos(min_period_frames));
		print_verbose("WASAPI: max_period_frames = " + itos(max_period_frames));
		print_verbose("WASAPI: selected a period frame size of " + itos(period_frames));
		buffer_frames = period_frames;

		hr = device_audio_client_3->InitializeSharedAudioStream(0, period_frames, pwfex, nullptr);
		if (hr != S_OK) {
			print_verbose("WASAPI: InitializeSharedAudioStream failed with error 0x" + String::num_uint64(hr, 16) + ", falling back to IAudioClient.");
			CoTaskMemFree(pwfex);
			return audio_device_init(p_device, p_input, p_reinit, true);
		} else {
			uint32_t output_latency_in_frames;
			WAVEFORMATEX *current_pwfex;
			hr = device_audio_client_3->GetCurrentSharedModeEnginePeriod(&current_pwfex, &output_latency_in_frames);
			if (hr == OK) {
				real_latency = (float)output_latency_in_frames / (float)current_pwfex->nSamplesPerSec;
				CoTaskMemFree(current_pwfex);
			} else {
				print_verbose("WASAPI: GetCurrentSharedModeEnginePeriod failed with error 0x" + String::num_uint64(hr, 16) + ", falling back to IAudioClient.");
				CoTaskMemFree(pwfex);
				return audio_device_init(p_device, p_input, p_reinit, true);
			}
		}
	}

	if (p_input) {
		hr = p_device->audio_client->GetService(IID_IAudioCaptureClient, (void **)&p_device->capture_client);
	} else {
		hr = p_device->audio_client->GetService(IID_IAudioRenderClient, (void **)&p_device->render_client);
	}
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	// Free memory.
	CoTaskMemFree(pwfex);

	return OK;
}

Error AudioDriverWASAPI::init_output_device(bool p_reinit) {
	Error err = audio_device_init(&audio_output, false, p_reinit);
	if (err != OK) {
		return err;
	}

	if (!are_output_channels_recommended(audio_output.channels))
		WARN_PRINT("WASAPI: Unsupported number of channels: " + itos(audio_output.channels));

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

	// Get the max frames.
	UINT32 max_frames;
	HRESULT hr = audio_input.audio_client->GetBufferSize(&max_frames);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	input_buffer_init(max_frames);

	return OK;
}

Error AudioDriverWASAPI::audio_device_finish(AudioDeviceWASAPI *p_device) {
	if (p_device->active.is_set()) {
		if (p_device->audio_client) {
			p_device->audio_client->Stop();
		}
		p_device->active.clear();
	}

	p_device->audio_client.Reset();
	p_device->render_client.Reset();
	p_device->capture_client.Reset();

	return OK;
}

Error AudioDriverWASAPI::finish_output_device() {
	return audio_device_finish(&audio_output);
}

Error AudioDriverWASAPI::finish_input_device() {
	return audio_device_finish(&audio_input);
}

Error AudioDriverWASAPI::init() {
	mix_rate = _get_configured_mix_rate();

	target_latency_ms = Engine::get_singleton()->get_audio_output_latency();

	exit_thread.clear();

	Error err = init_output_device();
	ERR_FAIL_COND_V_MSG(err != OK, err, "WASAPI: init_output_device error.");

	thread.start(thread_func, this);

	return OK;
}

int AudioDriverWASAPI::get_mix_rate() const {
	return mix_rate;
}

float AudioDriverWASAPI::get_latency() {
	return real_latency;
}

int AudioDriverWASAPI::get_output_channels() const {
	return audio_output.channels;
}

AudioDriver::BufferFormat AudioDriverWASAPI::get_output_buffer_format() const {
	return audio_output.buffer_format;
}

PackedStringArray AudioDriverWASAPI::audio_device_get_list(bool p_input) {
	PackedStringArray list;
	ComPtr<IMMDeviceCollection> devices = nullptr;
	ComPtr<IMMDeviceEnumerator> enumerator = nullptr;

	list.push_back(String("Default"));

	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, PackedStringArray());

	hr = enumerator->EnumAudioEndpoints(p_input ? eCapture : eRender, DEVICE_STATE_ACTIVE, &devices);
	ERR_FAIL_COND_V(hr != S_OK, PackedStringArray());

	UINT count = 0;
	hr = devices->GetCount(&count);
	ERR_FAIL_COND_V(hr != S_OK, PackedStringArray());

	for (ULONG i = 0; i < count; i++) {
		ComPtr<IMMDevice> output_device = nullptr;

		hr = devices->Item(i, &output_device);
		ERR_BREAK(hr != S_OK);

		ComPtr<IPropertyStore> props = nullptr;
		hr = output_device->OpenPropertyStore(STGM_READ, &props);
		ERR_BREAK(hr != S_OK);

		PROPVARIANT propvar;
		PropVariantInit(&propvar);

		hr = props->GetValue(PKEY_Device_FriendlyName, &propvar);
		ERR_BREAK(hr != S_OK);

		list.push_back(String(propvar.pwszVal));

		PropVariantClear(&propvar);
	}

	return list;
}

PackedStringArray AudioDriverWASAPI::get_output_device_list() {
	return audio_device_get_list(false);
}

String AudioDriverWASAPI::get_output_device() {
	lock();
	String name = audio_output.device_name;
	unlock();

	return name;
}

void AudioDriverWASAPI::set_output_device(const String &p_name) {
	lock();
	audio_output.new_device = p_name;
	unlock();
}

void AudioDriverWASAPI::thread_func(void *p_udata) {
	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

	AudioDriverWASAPI *ad = static_cast<AudioDriverWASAPI *>(p_udata);

	while (!ad->exit_thread.is_set()) {
		uint32_t read_frames = 0;
		uint32_t written_frames = 0;

		ad->lock();
		ad->start_counting_ticks();

		if (ad->audio_output.audio_client) {
			UINT32 buffer_size;
			UINT32 cur_frames;
			bool invalidated = false;
			HRESULT hr = ad->audio_output.audio_client->GetBufferSize(&buffer_size);
			if (hr != S_OK) {
				ERR_PRINT("WASAPI: GetBufferSize error");
			}
			hr = ad->audio_output.audio_client->GetCurrentPadding(&cur_frames);
			if (hr == S_OK) {
				// Check how much frames are available on the WASAPI buffer.
				UINT32 write_frames = buffer_size - cur_frames;
				if (write_frames > 0) {
					BYTE *buffer = nullptr;
					hr = ad->audio_output.render_client->GetBuffer(write_frames, &buffer);
					if (hr == S_OK) {
						ad->audio_server_process(write_frames, buffer, ad->audio_output.active.is_set());

						hr = ad->audio_output.render_client->ReleaseBuffer(write_frames, 0);
						if (hr != S_OK) {
							ERR_PRINT("WASAPI: Release buffer error");
						}

						written_frames += write_frames;
					} else if (hr == AUDCLNT_E_DEVICE_INVALIDATED) {
						// `output_device` is not valid anymore, reopen it.

						Error err = ad->finish_output_device();
						if (err != OK) {
							ERR_PRINT("WASAPI: finish_output_device error");
						}
					} else {
						ERR_PRINT("WASAPI: Get buffer error");
						ad->exit_thread.set();
					}
				}
			} else if (hr == AUDCLNT_E_DEVICE_INVALIDATED) {
				invalidated = true;
			} else {
				ERR_PRINT("WASAPI: GetCurrentPadding error");
			}

			if (invalidated) {
				// `output_device` is not valid anymore.
				WARN_PRINT("WASAPI: Current output_device invalidated, closing output_device");

				Error err = ad->finish_output_device();
				if (err != OK) {
					ERR_PRINT("WASAPI: finish_output_device error");
				}
			}
		}

		// If we're using the Default output device and it changed finish it so we'll re-init the output device.
		if (ad->audio_output.device_name == "Default" && default_output_device_changed) {
			Error err = ad->finish_output_device();
			if (err != OK) {
				ERR_PRINT("WASAPI: finish_output_device error");
			}

			default_output_device_changed = false;
		}

		// User selected a new output device, finish the current one so we'll init the new output device.
		if (ad->audio_output.device_name != ad->audio_output.new_device) {
			ad->audio_output.device_name = ad->audio_output.new_device;
			Error err = ad->finish_output_device();
			if (err != OK) {
				ERR_PRINT("WASAPI: finish_output_device error");
			}
		}

		if (!ad->audio_output.audio_client) {
			Error err = ad->init_output_device(true);
			if (err == OK) {
				ad->start();
			}
		}

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

					if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
						ad->input_process(num_frames_available, nullptr);
					} else {
						ad->input_process(num_frames_available, data);
					}

					read_frames += num_frames_available;

					hr = ad->audio_input.capture_client->ReleaseBuffer(num_frames_available);
					ERR_BREAK(hr != S_OK);

					hr = ad->audio_input.capture_client->GetNextPacketSize(&packet_length);
					ERR_BREAK(hr != S_OK);
				}
			}

			// If we're using the Default output device and it changed finish it so we'll re-init the output device.
			if (ad->audio_input.device_name == "Default" && default_input_device_changed) {
				Error err = ad->finish_input_device();
				if (err != OK) {
					ERR_PRINT("WASAPI: finish_input_device error");
				}

				default_input_device_changed = false;
			}

			// User selected a new input device, finish the current one so we'll init the new input device.
			if (ad->audio_input.device_name != ad->audio_input.new_device) {
				ad->audio_input.device_name = ad->audio_input.new_device;
				Error err = ad->finish_input_device();
				if (err != OK) {
					ERR_PRINT("WASAPI: finish_input_device error");
				}
			}

			if (!ad->audio_input.audio_client) {
				Error err = ad->init_input_device(true);
				if (err == OK) {
					ad->input_start();
				}
			}
		}

		ad->stop_counting_ticks();
		ad->unlock();

		// Let the thread rest a while if we haven't read or written anything.
		if (written_frames == 0 && read_frames == 0) {
			OS::get_singleton()->delay_usec(1000);
		}
	}
	CoUninitialize();
}

void AudioDriverWASAPI::start() {
	if (audio_output.audio_client) {
		HRESULT hr = audio_output.audio_client->Start();
		if (hr != S_OK) {
			ERR_PRINT("WASAPI: Start failed");
		} else {
			audio_output.active.set();
		}
	}
}

void AudioDriverWASAPI::lock() {
	mutex.lock();
}

void AudioDriverWASAPI::unlock() {
	mutex.unlock();
}

void AudioDriverWASAPI::finish() {
	exit_thread.set();
	if (thread.is_started()) {
		thread.wait_to_finish();
	}

	finish_input_device();
	finish_output_device();
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

int AudioDriverWASAPI::get_input_channels() const {
	return audio_input.channels;
}

AudioDriver::BufferFormat AudioDriverWASAPI::get_input_buffer_format() const {
	return audio_input.buffer_format;
}

PackedStringArray AudioDriverWASAPI::get_input_device_list() {
	return audio_device_get_list(true);
}

String AudioDriverWASAPI::get_input_device() {
	lock();
	String name = audio_input.device_name;
	unlock();

	return name;
}

void AudioDriverWASAPI::set_input_device(const String &p_name) {
	lock();
	audio_input.new_device = p_name;
	unlock();
}

#endif // WASAPI_ENABLED
