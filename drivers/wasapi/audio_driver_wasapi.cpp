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

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

// Define IAudioClient3 if not already defined by MinGW headers
#if defined __MINGW32__ || defined __MINGW64__

#ifndef __IAudioClient3_FWD_DEFINED__
#define __IAudioClient3_FWD_DEFINED__

typedef interface IAudioClient3 IAudioClient3;

#endif // __IAudioClient3_FWD_DEFINED__

#ifndef __IAudioClient3_INTERFACE_DEFINED__
#define __IAudioClient3_INTERFACE_DEFINED__

// clang-format off
MIDL_INTERFACE("7ED4EE07-8E67-4CD4-8C1A-2B7A5987AD42")
IAudioClient3 : public IAudioClient2 {
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
// clang-format on
__CRT_UUID_DECL(IAudioClient3, 0x7ED4EE07, 0x8E67, 0x4CD4, 0x8C, 0x1A, 0x2B, 0x7A, 0x59, 0x87, 0xAD, 0x42)

#endif // __IAudioClient3_INTERFACE_DEFINED__

#endif // __MINGW32__ || __MINGW64__

#ifndef PKEY_Device_FriendlyNameGodot

#undef DEFINE_PROPERTYKEY
/* clang-format off */
#define DEFINE_PROPERTYKEY(id, a, b, c, d, e, f, g, h, i, j, k, l) \
	const PROPERTYKEY id = { { a, b, c, { d, e, f, g, h, i, j, k, } }, l };
/* clang-format on */

DEFINE_PROPERTYKEY(PKEY_Device_FriendlyNameGodot, 0xa45c254e, 0xdf1c, 0x4efd, 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0, 14);
#endif

const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
const IID IID_IAudioClient = __uuidof(IAudioClient);
const IID IID_IAudioClient3 = __uuidof(IAudioClient3);
const IID IID_IAudioRenderClient = __uuidof(IAudioRenderClient);
const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);

#define SAFE_RELEASE(memory)   \
	if ((memory) != nullptr) { \
		(memory)->Release();   \
		(memory) = nullptr;    \
	}

#define REFTIMES_PER_SEC 10000000
#define REFTIMES_PER_MILLISEC 10000

#define CAPTURE_BUFFER_CHANNELS 2

static bool default_output_device_changed = false;
static bool default_input_device_changed = false;
static int output_reinit_countdown = 0;
static int input_reinit_countdown = 0;

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wnon-virtual-dtor") // Silence warning due to a COM API weirdness (GH-35194).

class CMMNotificationClient : public IMMNotificationClient {
	LONG _cRef = 1;

public:
	ComPtr<IMMDeviceEnumerator> enumerator = nullptr;

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

GODOT_GCC_WARNING_POP

static CMMNotificationClient notif_client;

// Helper function to find the correct audio device (default or by name)
ComPtr<IMMDevice> AudioDriverWASAPI::find_output_device(AudioDeviceWASAPI *p_device, bool p_input) {
	ComPtr<IMMDeviceEnumerator> enumerator = nullptr;
	ComPtr<IMMDevice> output_device = nullptr;

	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, nullptr);

	if (p_device->device_name == "Default") {
		hr = enumerator->GetDefaultAudioEndpoint(p_input ? eCapture : eRender, eConsole, &output_device);
	} else {
		ComPtr<IMMDeviceCollection> devices = nullptr;

		hr = enumerator->EnumAudioEndpoints(p_input ? eCapture : eRender, DEVICE_STATE_ACTIVE, &devices);
		ERR_FAIL_COND_V(hr != S_OK, nullptr);

		LPWSTR strId = nullptr;
		bool found = false;

		UINT count = 0;
		hr = devices->GetCount(&count);
		ERR_FAIL_COND_V(hr != S_OK, nullptr);

		for (ULONG i = 0; i < count && !found; i++) {
			ComPtr<IMMDevice> tmp_device = nullptr;

			hr = devices->Item(i, &tmp_device);
			ERR_BREAK_MSG(hr != S_OK, "Cannot get devices item.");

			ComPtr<IPropertyStore> props = nullptr;
			hr = tmp_device->OpenPropertyStore(STGM_READ, &props);
			ERR_BREAK_MSG(hr != S_OK, "Cannot open property store.");

			PROPVARIANT propvar;
			PropVariantInit(&propvar);

			hr = props->GetValue(PKEY_Device_FriendlyNameGodot, &propvar);
			ERR_BREAK_MSG(hr != S_OK, "Cannot get value.");

			if (p_device->device_name == String(propvar.pwszVal)) {
				hr = tmp_device->GetId(&strId);
				if (unlikely(hr != S_OK)) {
					PropVariantClear(&propvar);
					ERR_PRINT("Cannot get device ID string.");
					break;
				}

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

	return output_device;
}

// Helper function to register/unregister notification callbacks
void AudioDriverWASAPI::register_notification_callback(ComPtr<IMMDeviceEnumerator> &enumerator) {
	if (notif_client.enumerator != nullptr) {
		notif_client.enumerator->UnregisterEndpointNotificationCallback(&notif_client);
		notif_client.enumerator = nullptr;
	}
	HRESULT hr = enumerator->RegisterEndpointNotificationCallback(&notif_client);
	if (hr == S_OK) {
		notif_client.enumerator = enumerator;
	} else {
		ERR_PRINT("WASAPI: RegisterEndpointNotificationCallback error");
	}
}

// Helper function to activate audio client (IAudioClient3 or IAudioClient)
HRESULT AudioDriverWASAPI::activate_audio_client(ComPtr<IMMDevice> &device, bool use_client3, IAudioClient **out_client) {
	HRESULT hr = S_OK;
	
	if (use_client3) {
		hr = device->Activate(IID_IAudioClient3, CLSCTX_ALL, nullptr, (void **)out_client);
		if (hr != S_OK) {
			// IID_IAudioClient3 will never activate on OS versions before Windows 10.
			// Older Windows versions should fall back gracefully.
			print_verbose("WASAPI: Couldn't activate device with IAudioClient3 interface, falling back to IAudioClient interface");
			return S_FALSE; // Signal to fall back to IAudioClient
		} else {
			print_verbose("WASAPI: Activated device using IAudioClient3 interface");
		}
	}
	
	if (!use_client3 || hr == S_FALSE) {
		hr = device->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, (void **)out_client);
	}
	
	return hr;
}

// Helper function to get and validate mix format
WAVEFORMATEX* AudioDriverWASAPI::get_and_validate_mix_format(IAudioClient *audio_client, bool &used_closest) {
	WAVEFORMATEX *pwfex = nullptr;
	HRESULT hr = audio_client->GetMixFormat(&pwfex);
	ERR_FAIL_COND_V(hr != S_OK, nullptr);
	
	// From this point onward, CoTaskMemFree(pwfex) must be called before returning or pwfex will leak!
	
	print_verbose("WASAPI: wFormatTag = " + itos(pwfex->wFormatTag));
	print_verbose("WASAPI: nChannels = " + itos(pwfex->nChannels));
	print_verbose("WASAPI: nSamplesPerSec = " + itos(pwfex->nSamplesPerSec));
	print_verbose("WASAPI: nAvgBytesPerSec = " + itos(pwfex->nAvgBytesPerSec));
	print_verbose("WASAPI: nBlockAlign = " + itos(pwfex->nBlockAlign));
	print_verbose("WASAPI: wBitsPerSample = " + itos(pwfex->wBitsPerSample));
	print_verbose("WASAPI: cbSize = " + itos(pwfex->cbSize));

	WAVEFORMATEX *closest = nullptr;
	hr = audio_client->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, pwfex, &closest);
	if (hr == S_FALSE) {
		WARN_PRINT("WASAPI: Mix format is not supported by the device");
		if (closest) {
			print_verbose("WASAPI: closest->wFormatTag = " + itos(closest->wFormatTag));
			print_verbose("WASAPI: closest->nChannels = " + itos(closest->nChannels));
			print_verbose("WASAPI: closest->nSamplesPerSec = " + itos(closest->nSamplesPerSec));
			print_verbose("WASAPI: closest->nAvgBytesPerSec = " + itos(closest->nAvgBytesPerSec));
			print_verbose("WASAPI: closest->nBlockAlign = " + itos(closest->nBlockAlign));
			print_verbose("WASAPI: closest->wBitsPerSample = " + itos(closest->wBitsPerSample));
			print_verbose("WASAPI: closest->cbSize = " + itos(closest->cbSize));

			WARN_PRINT("WASAPI: Using closest match instead");
			CoTaskMemFree(pwfex);
			pwfex = closest;
			used_closest = true;
		}
	} else {
		used_closest = false;
	}
	
	return pwfex;
}

// Helper function to validate and set format
bool AudioDriverWASAPI::validate_and_set_format(AudioDeviceWASAPI *p_device, WAVEFORMATEX *pwfex) {
	// Since we're using WASAPI Shared Mode we can't control any of these, we just tag along
	p_device->channels = pwfex->nChannels;
	p_device->format_tag = pwfex->wFormatTag;
	p_device->bits_per_sample = pwfex->wBitsPerSample;
	p_device->frame_size = (p_device->bits_per_sample / 8) * p_device->channels;

	if (p_device->format_tag == WAVE_FORMAT_EXTENSIBLE) {
		WAVEFORMATEXTENSIBLE *wfex = (WAVEFORMATEXTENSIBLE *)pwfex;

		if (wfex->SubFormat == KSDATAFORMAT_SUBTYPE_PCM) {
			p_device->format_tag = WAVE_FORMAT_PCM;
		} else if (wfex->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT) {
			p_device->format_tag = WAVE_FORMAT_IEEE_FLOAT;
		} else {
			ERR_PRINT("WASAPI: Format not supported");
			return false;
		}
	} else {
		if (p_device->format_tag != WAVE_FORMAT_PCM && p_device->format_tag != WAVE_FORMAT_IEEE_FLOAT) {
			ERR_PRINT("WASAPI: Format not supported");
			return false;
		}
	}
	
	return true;
}

// Helper function to initialize audio client
HRESULT AudioDriverWASAPI::initialize_audio_client(AudioDeviceWASAPI *p_device, WAVEFORMATEX *pwfex, bool p_input, bool use_client3) {
	HRESULT hr = S_OK;
	
	if (!use_client3) {
		DWORD streamflags = 0;
		if ((DWORD)mix_rate != pwfex->nSamplesPerSec) {
			streamflags |= AUDCLNT_STREAMFLAGS_RATEADJUST;
			pwfex->nSamplesPerSec = mix_rate;
			pwfex->nAvgBytesPerSec = pwfex->nSamplesPerSec * pwfex->nChannels * (pwfex->wBitsPerSample / 8);
		}
		hr = p_device->audio_client->Initialize(AUDCLNT_SHAREMODE_SHARED, streamflags, p_input ? REFTIMES_PER_SEC : 0, 0, pwfex, nullptr);
	} else {
		IAudioClient3 *device_audio_client_3 = (IAudioClient3 *)p_device->audio_client;

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
			return S_FALSE; // Signal to fall back to IAudioClient
		}

		// Period frames must be an integral multiple of fundamental_period_frames or IAudioClient3 initialization will fail,
		// so we need to select the closest multiple to the user-specified latency.
		UINT32 desired_period_frames = target_latency_ms * mix_rate / 1000;
		UINT32 period_frames = (desired_period_frames / fundamental_period_frames) * fundamental_period_frames;
		if (Math::abs((int64_t)period_frames - (int64_t)desired_period_frames) > Math::abs((int64_t)(period_frames + fundamental_period_frames) - (int64_t)desired_period_frames)) {
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
			return S_FALSE; // Signal to fall back to IAudioClient
		}
	}
	
	return hr;
}

// Helper function to setup buffer and latency
void AudioDriverWASAPI::setup_buffer_and_latency(AudioDeviceWASAPI *p_device, bool p_input, bool use_client3, WAVEFORMATEX *pwfex) {
	if (!use_client3) {
		UINT32 max_frames;
		HRESULT hr = p_device->audio_client->GetBufferSize(&max_frames);
		if (unlikely(hr != S_OK)) {
			ERR_PRINT("WASAPI: GetBufferSize failed");
			return;
		}

		// Due to WASAPI Shared Mode we have no control of the buffer size
		if (!p_input) {
			buffer_frames = max_frames;

			int64_t latency = 0;
			audio_output.audio_client->GetStreamLatency(&latency);
			// WASAPI REFERENCE_TIME units are 100 nanoseconds per unit
			// https://docs.microsoft.com/en-us/windows/win32/directshow/reference-time
			// Convert REFTIME to seconds as godot uses for latency
			real_latency = (float)latency / (float)REFTIMES_PER_SEC;
		}
	} else {
		IAudioClient3 *device_audio_client_3 = (IAudioClient3 *)p_device->audio_client;
		uint32_t output_latency_in_frames;
		WAVEFORMATEX *current_pwfex;
		HRESULT hr = device_audio_client_3->GetCurrentSharedModeEnginePeriod(&current_pwfex, &output_latency_in_frames);
		if (hr == OK) {
			real_latency = (float)output_latency_in_frames / (float)current_pwfex->nSamplesPerSec;
			CoTaskMemFree(current_pwfex);
		} else {
			print_verbose("WASAPI: GetCurrentSharedModeEnginePeriod failed with error 0x" + String::num_uint64(hr, 16) + ", falling back to IAudioClient.");
		}
	}
}

// Helper function to acquire service clients
HRESULT AudioDriverWASAPI::acquire_service_clients(AudioDeviceWASAPI *p_device, bool p_input) {
	HRESULT hr = S_OK;
	
	if (p_input) {
		hr = p_device->audio_client->GetService(IID_IAudioCaptureClient, (void **)&p_device->capture_client);
	} else {
		hr = p_device->audio_client->GetService(IID_IAudioRenderClient, (void **)&p_device->render_client);
	}
	
	return hr;
}

Error AudioDriverWASAPI::audio_device_init(AudioDeviceWASAPI *p_device, bool p_input, bool p_reinit, bool p_no_audio_client_3) {
	// This function can be called recursively, so clean up before starting:
	audio_device_finish(p_device);

	// 1. Find the correct audio device
	ComPtr<IMMDevice> output_device = find_output_device(p_device, p_input);
	if (!output_device) {
		if (p_reinit) {
			return ERR_CANT_OPEN;
		} else {
			ERR_FAIL_COND_V_MSG(false, ERR_CANT_OPEN, "WASAPI: Failed to find audio device");
		}
	}

	// 2. Register notification callback
	ComPtr<IMMDeviceEnumerator> enumerator = nullptr;
	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);
	
	register_notification_callback(enumerator);

	// 3. Determine if we should use IAudioClient3
	using_audio_client_3 = !p_input; // IID_IAudioClient3 is only used for adjustable output latency (not input)
	if (p_no_audio_client_3) {
		using_audio_client_3 = false;
	}

	// 4. Activate audio client
	hr = activate_audio_client(output_device, using_audio_client_3, &p_device->audio_client);
	if (hr == S_FALSE) {
		// Fall back to IAudioClient
		using_audio_client_3 = false;
		hr = activate_audio_client(output_device, false, &p_device->audio_client);
	}

	if (p_reinit) {
		if (hr != S_OK) {
			return ERR_CANT_OPEN;
		}
	} else {
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);
	}

	// 5. Set client properties for IAudioClient3
	if (using_audio_client_3) {
		AudioClientProperties audioProps{};
		audioProps.cbSize = sizeof(AudioClientProperties);
		audioProps.bIsOffload = FALSE;
		audioProps.eCategory = AudioCategory_GameEffects;

		hr = ((IAudioClient3 *)p_device->audio_client)->SetClientProperties(&audioProps);
		ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_CANT_OPEN, "WASAPI: SetClientProperties failed with error 0x" + String::num_uint64(hr, 16) + ".");
	}

	// 6. Get and validate mix format
	bool used_closest = false;
	WAVEFORMATEX *pwfex = get_and_validate_mix_format(p_device->audio_client, used_closest);
	if (!pwfex) {
		return ERR_CANT_OPEN;
	}

	// 7. Validate and set format
	if (!validate_and_set_format(p_device, pwfex)) {
		CoTaskMemFree(pwfex);
		return ERR_CANT_OPEN;
	}

	// 8. Initialize audio client
	hr = initialize_audio_client(p_device, pwfex, p_input, using_audio_client_3);
	if (hr == S_FALSE) {
		// Fall back to IAudioClient
		using_audio_client_3 = false;
		hr = initialize_audio_client(p_device, pwfex, p_input, false);
	}
	
	if (p_reinit) {
		if (hr != S_OK) {
			CoTaskMemFree(pwfex);
			return ERR_CANT_OPEN;
		}
	} else {
		if (unlikely(hr != S_OK)) {
			CoTaskMemFree(pwfex);
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, "WASAPI: Initialize failed with error 0x" + String::num_uint64(hr, 16) + ".");
		}
	}

	// 9. Setup buffer and latency
	setup_buffer_and_latency(p_device, p_input, using_audio_client_3, pwfex);

	// 10. Acquire service clients
	hr = acquire_service_clients(p_device, p_input);
	if (unlikely(hr != S_OK)) {
		CoTaskMemFree(pwfex);
		ERR_FAIL_V(ERR_CANT_OPEN);
	}

	// 11. Cleanup
	CoTaskMemFree(pwfex);
	return OK;
}

Error AudioDriverWASAPI::init_output_device(bool p_reinit) {
	Error err = audio_device_init(&audio_output, false, p_reinit);
	if (err != OK) {
		// We've tried to init the device, but have failed. Time to clean up.
		Error finish_err = finish_output_device();
		if (finish_err != OK) {
			ERR_PRINT("WASAPI: finish_output_device error after failed output audio_device_init");
		}
		return err;
	}

	switch (audio_output.channels) {
		case 1: // Mono
		case 3: // Surround 2.1
		case 5: // Surround 5.0
		case 7: // Surround 7.0
			// We will downmix as required.
			channels = audio_output.channels + 1;
			break;

		case 2: // Stereo
		case 4: // Surround 3.1
		case 6: // Surround 5.1
		case 8: // Surround 7.1
			channels = audio_output.channels;
			break;

		default:
			WARN_PRINT("WASAPI: Unsupported number of channels: " + itos(audio_output.channels));
			channels = 2;
			break;
	}

	// Sample rate is independent of channels (ref: https://stackoverflow.com/questions/11048825/audio-sample-frequency-rely-on-channels)
	samples_in.resize(buffer_frames * channels);

	input_position = 0;
	input_size = 0;

	print_verbose("WASAPI: detected " + itos(audio_output.channels) + " channels");
	print_verbose("WASAPI: audio buffer frames: " + itos(buffer_frames) + " calculated latency: " + itos(buffer_frames * 1000 / mix_rate) + "ms");

	return OK;
}

Error AudioDriverWASAPI::init_input_device(bool p_reinit) {
	Error err = audio_device_init(&audio_input, true, p_reinit);
	if (err != OK) {
		// We've tried to init the device, but have failed. Time to clean up.
		Error finish_err = finish_input_device();
		if (finish_err != OK) {
			ERR_PRINT("WASAPI: finish_input_device error after failed input audio_device_init");
		}
		return err;
	}

	// Get the max frames
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

	SAFE_RELEASE(p_device->audio_client)
	SAFE_RELEASE(p_device->render_client)
	SAFE_RELEASE(p_device->capture_client)

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

AudioDriver::SpeakerMode AudioDriverWASAPI::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(channels);
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

		hr = props->GetValue(PKEY_Device_FriendlyNameGodot, &propvar);
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

void AudioDriverWASAPI::write_sample(WORD format_tag, int bits_per_sample, BYTE *buffer, int i, int32_t sample) {
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

// Helper function to write audio buffer with channel conversion
void AudioDriverWASAPI::write_audio_buffer(BYTE *buffer, UINT32 write_frames, uint32_t &write_ofs) {
	// We're using WASAPI Shared Mode so we must convert the buffer
	if (channels == audio_output.channels) {
		for (unsigned int i = 0; i < write_frames * channels; i++) {
			write_sample(audio_output.format_tag, audio_output.bits_per_sample, buffer, i, samples_in.write[write_ofs++]);
		}
	} else if (channels == audio_output.channels + 1) {
		// Pass all channels except the last two as-is, and then mix the last two
		// together as one channel. E.g. stereo -> mono, or 3.1 -> 2.1.
		unsigned int last_chan = audio_output.channels - 1;
		for (unsigned int i = 0; i < write_frames; i++) {
			for (unsigned int j = 0; j < last_chan; j++) {
				write_sample(audio_output.format_tag, audio_output.bits_per_sample, buffer, i * audio_output.channels + j, samples_in.write[write_ofs++]);
			}
			int32_t l = samples_in.write[write_ofs++];
			int32_t r = samples_in.write[write_ofs++];
			int32_t c = (int32_t)(((int64_t)l + (int64_t)r) / 2);
			write_sample(audio_output.format_tag, audio_output.bits_per_sample, buffer, i * audio_output.channels + last_chan, c);
		}
	} else {
		for (unsigned int i = 0; i < write_frames; i++) {
			for (unsigned int j = 0; j < MIN(channels, audio_output.channels); j++) {
				write_sample(audio_output.format_tag, audio_output.bits_per_sample, buffer, i * audio_output.channels + j, samples_in.write[write_ofs++]);
			}
			if (audio_output.channels > channels) {
				for (unsigned int j = channels; j < audio_output.channels; j++) {
					write_sample(audio_output.format_tag, audio_output.bits_per_sample, buffer, i * audio_output.channels + j, 0);
				}
			}
		}
	}
}

// Helper function to read audio buffer from input
void AudioDriverWASAPI::read_audio_buffer(BYTE *data, UINT32 num_frames_available, DWORD flags) {
	// fixme: Only works for floating point atm
	for (UINT32 j = 0; j < num_frames_available; j++) {
		int32_t l, r;

		if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
			l = r = 0;
		} else {
			if (audio_input.channels == 2) {
				l = read_sample(audio_input.format_tag, audio_input.bits_per_sample, data, j * 2);
				r = read_sample(audio_input.format_tag, audio_input.bits_per_sample, data, j * 2 + 1);
			} else if (audio_input.channels == 1) {
				l = r = read_sample(audio_input.format_tag, audio_input.bits_per_sample, data, j);
			} else {
				l = r = 0;
				ERR_PRINT("WASAPI: unsupported channel count in microphone!");
			}
		}

		input_buffer_write(l);
		input_buffer_write(r);
	}
}

// Helper function to process audio output
void AudioDriverWASAPI::process_audio_output(uint32_t &avail_frames, uint32_t &write_ofs, uint32_t &written_frames) {
	if (avail_frames > 0 && audio_output.audio_client) {
		UINT32 buffer_size;
		UINT32 cur_frames;
		bool invalidated = false;
		HRESULT hr = audio_output.audio_client->GetBufferSize(&buffer_size);
		if (hr != S_OK) {
			ERR_PRINT("WASAPI: GetBufferSize error");
		}
		hr = audio_output.audio_client->GetCurrentPadding(&cur_frames);
		if (hr == S_OK) {
			// Check how much frames are available on the WASAPI buffer
			UINT32 write_frames = MIN(buffer_size - cur_frames, avail_frames);
			if (write_frames > 0) {
				BYTE *buffer = nullptr;
				hr = audio_output.render_client->GetBuffer(write_frames, &buffer);
				if (hr == S_OK) {
					write_audio_buffer(buffer, write_frames, write_ofs);

					hr = audio_output.render_client->ReleaseBuffer(write_frames, 0);
					if (hr != S_OK) {
						ERR_PRINT("WASAPI: Release buffer error");
					}

					avail_frames -= write_frames;
					written_frames += write_frames;
				} else if (hr == AUDCLNT_E_DEVICE_INVALIDATED) {
					// output_device is not valid anymore, reopen it
					Error err = finish_output_device();
					if (err != OK) {
						ERR_PRINT("WASAPI: finish_output_device error");
					} else {
						// We reopened the output device and samples_in may have resized, so invalidate the current avail_frames
						avail_frames = 0;
					}
				} else {
					ERR_PRINT("WASAPI: Get buffer error");
					exit_thread.set();
				}
			}
		} else if (hr == AUDCLNT_E_DEVICE_INVALIDATED) {
			invalidated = true;
		} else {
			ERR_PRINT("WASAPI: GetCurrentPadding error");
		}

		if (invalidated) {
			// output_device is not valid anymore
			WARN_PRINT("WASAPI: Current output_device invalidated, closing output_device");

			Error err = finish_output_device();
			if (err != OK) {
				ERR_PRINT("WASAPI: finish_output_device error");
			}
		}
	}
}

// Helper function to process audio input
void AudioDriverWASAPI::process_audio_input(uint32_t &read_frames) {
	if (audio_input.active.is_set()) {
		UINT32 packet_length = 0;
		BYTE *data;
		UINT32 num_frames_available;
		DWORD flags;

		HRESULT hr = audio_input.capture_client->GetNextPacketSize(&packet_length);
		if (hr == S_OK) {
			while (packet_length != 0) {
				hr = audio_input.capture_client->GetBuffer(&data, &num_frames_available, &flags, nullptr, nullptr);
				ERR_BREAK(hr != S_OK);

				read_audio_buffer(data, num_frames_available, flags);

				read_frames += num_frames_available;

				hr = audio_input.capture_client->ReleaseBuffer(num_frames_available);
				ERR_BREAK(hr != S_OK);

				hr = audio_input.capture_client->GetNextPacketSize(&packet_length);
				ERR_BREAK(hr != S_OK);
			}
		}
	}
}

// Helper function to handle output device changes
void AudioDriverWASAPI::handle_output_device_changes() {
	// If we're using the Default output device and it changed finish it so we'll re-init the output device
	if (audio_output.device_name == "Default" && default_output_device_changed) {
		Error err = finish_output_device();
		if (err != OK) {
			ERR_PRINT("WASAPI: finish_output_device error");
		}

		default_output_device_changed = false;
	}

	// User selected a new output device, finish the current one so we'll init the new output device
	if (audio_output.device_name != audio_output.new_device) {
		audio_output.device_name = audio_output.new_device;
		Error err = finish_output_device();
		if (err != OK) {
			ERR_PRINT("WASAPI: finish_output_device error");
		}
	}

	if (!audio_output.audio_client) {
		if (output_reinit_countdown < 1) {
			Error err = init_output_device(true);
			if (err == OK) {
				start();
			} else {
				output_reinit_countdown = 1000;
			}
		} else {
			output_reinit_countdown--;
		}
	}
}

// Helper function to handle input device changes
void AudioDriverWASAPI::handle_input_device_changes() {
	if (audio_input.active.is_set()) {
		// If we're using the Default input device and it changed finish it so we'll re-init the input device
		if (audio_input.device_name == "Default" && default_input_device_changed) {
			Error err = finish_input_device();
			if (err != OK) {
				ERR_PRINT("WASAPI: finish_input_device error");
			}

			default_input_device_changed = false;
		}

		// User selected a new input device, finish the current one so we'll init the new input device
		if (audio_input.device_name != audio_input.new_device) {
			audio_input.device_name = audio_input.new_device;
			Error err = finish_input_device();
			if (err != OK) {
				ERR_PRINT("WASAPI: finish_input_device error");
			}
		}

		if (!audio_input.audio_client) {
			if (input_reinit_countdown < 1) {
				Error err = init_input_device(true);
				if (err == OK) {
					input_start();
				} else {
					input_reinit_countdown = 1000;
				}
			} else {
				input_reinit_countdown--;
			}
		}
	}
}

void AudioDriverWASAPI::thread_func(void *p_udata) {
	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

	AudioDriverWASAPI *ad = static_cast<AudioDriverWASAPI *>(p_udata);
	uint32_t avail_frames = 0;
	uint32_t write_ofs = 0;

	while (!ad->exit_thread.is_set()) {
		uint32_t read_frames = 0;
		uint32_t written_frames = 0;

		// 1. Generate audio data if needed
		if (avail_frames == 0) {
			ad->lock();
			ad->start_counting_ticks();

			if (ad->audio_output.active.is_set()) {
				ad->audio_server_process(ad->buffer_frames, ad->samples_in.ptrw());
			} else {
				for (int i = 0; i < ad->samples_in.size(); i++) {
					ad->samples_in.write[i] = 0;
				}
			}

			avail_frames = ad->buffer_frames;
			write_ofs = 0;

			ad->stop_counting_ticks();
			ad->unlock();
		}

		ad->lock();
		ad->start_counting_ticks();

		// 2. Process audio output
		ad->process_audio_output(avail_frames, write_ofs, written_frames);

		// 3. Handle device changes
		ad->handle_output_device_changes();
		ad->handle_input_device_changes();

		// 4. Process audio input
		ad->process_audio_input(read_frames);

		ad->stop_counting_ticks();
		ad->unlock();

		// 5. Thread control - let the thread rest if no work was done
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

AudioDriverWASAPI::AudioDriverWASAPI() {
	samples_in.clear();
}

#endif // WASAPI_ENABLED
