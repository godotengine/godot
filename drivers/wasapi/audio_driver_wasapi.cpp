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

#include "core/os/os.h"
#include "core/project_settings.h"

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

#define SAFE_RELEASE(memory) \
	if ((memory) != NULL) {  \
		(memory)->Release(); \
		(memory) = NULL;     \
	}

#define REFTIMES_PER_SEC 10000000
#define REFTIMES_PER_MILLISEC 10000

#define CAPTURE_BUFFER_CHANNELS 2

static bool default_render_device_changed = false;
static bool default_capture_device_changed = false;

class CMMNotificationClient : public IMMNotificationClient {
	LONG _cRef;
	IMMDeviceEnumerator *_pEnumerator;

public:
	CMMNotificationClient() :
			_cRef(1),
			_pEnumerator(NULL) {}
	virtual ~CMMNotificationClient() {
		if ((_pEnumerator) != NULL) {
			(_pEnumerator)->Release();
			(_pEnumerator) = NULL;
		}
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
			*ppvInterface = NULL;
			return E_NOINTERFACE;
		}
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceAdded(LPCWSTR pwstrDeviceId) {
		return S_OK;
	};

	HRESULT STDMETHODCALLTYPE OnDeviceRemoved(LPCWSTR pwstrDeviceId) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceStateChanged(LPCWSTR pwstrDeviceId, DWORD dwNewState) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDefaultDeviceChanged(EDataFlow flow, ERole role, LPCWSTR pwstrDeviceId) {
		if (role == eConsole) {
			if (flow == eRender) {
				default_render_device_changed = true;
			} else if (flow == eCapture) {
				default_capture_device_changed = true;
			}
		}

		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnPropertyValueChanged(LPCWSTR pwstrDeviceId, const PROPERTYKEY key) {
		return S_OK;
	}
};

static CMMNotificationClient notif_client;

Error AudioDriverWASAPI::audio_device_init(AudioDeviceWASAPI *p_device, bool p_capture, bool reinit) {
	WAVEFORMATEX *pwfex;
	IMMDeviceEnumerator *enumerator = NULL;
	IMMDevice *device = NULL;

	CoInitialize(NULL);

	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	if (p_device->device_name == "Default") {
		hr = enumerator->GetDefaultAudioEndpoint(p_capture ? eCapture : eRender, eConsole, &device);
	} else {
		IMMDeviceCollection *devices = NULL;

		hr = enumerator->EnumAudioEndpoints(p_capture ? eCapture : eRender, DEVICE_STATE_ACTIVE, &devices);
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

		LPWSTR strId = NULL;
		bool found = false;

		UINT count = 0;
		hr = devices->GetCount(&count);
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

		for (ULONG i = 0; i < count && !found; i++) {
			IMMDevice *tmp_device = NULL;

			hr = devices->Item(i, &tmp_device);
			ERR_BREAK(hr != S_OK);

			IPropertyStore *props = NULL;
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
			hr = enumerator->GetDevice(strId, &device);
		}

		if (strId) {
			CoTaskMemFree(strId);
		}

		if (device == NULL) {
			hr = enumerator->GetDefaultAudioEndpoint(p_capture ? eCapture : eRender, eConsole, &device);
		}
	}

	if (reinit) {
		// In case we're trying to re-initialize the device prevent throwing this error on the console,
		// otherwise if there is currently no device available this will spam the console.
		if (hr != S_OK) {
			return ERR_CANT_OPEN;
		}
	} else {
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);
	}

	hr = enumerator->RegisterEndpointNotificationCallback(&notif_client);
	SAFE_RELEASE(enumerator)

	if (hr != S_OK) {
		ERR_PRINT("WASAPI: RegisterEndpointNotificationCallback error");
	}

	hr = device->Activate(IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&p_device->audio_client);
	SAFE_RELEASE(device)

	if (reinit) {
		if (hr != S_OK) {
			return ERR_CANT_OPEN;
		}
	} else {
		ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);
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

	WAVEFORMATEX *closest = NULL;
	hr = p_device->audio_client->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, pwfex, &closest);
	if (hr == S_FALSE) {
		WARN_PRINT("WASAPI: Mix format is not supported by the Device");
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
			ERR_FAIL_V(ERR_CANT_OPEN);
		}
	} else {
		if (p_device->format_tag != WAVE_FORMAT_PCM && p_device->format_tag != WAVE_FORMAT_IEEE_FLOAT) {
			ERR_PRINT("WASAPI: Format not supported");
			ERR_FAIL_V(ERR_CANT_OPEN);
		}
	}

	DWORD streamflags = 0;
	if ((DWORD)mix_rate != pwfex->nSamplesPerSec) {
		streamflags |= AUDCLNT_STREAMFLAGS_RATEADJUST;
		pwfex->nSamplesPerSec = mix_rate;
		pwfex->nAvgBytesPerSec = pwfex->nSamplesPerSec * pwfex->nChannels * (pwfex->wBitsPerSample / 8);
	}

	hr = p_device->audio_client->Initialize(AUDCLNT_SHAREMODE_SHARED, streamflags, p_capture ? REFTIMES_PER_SEC : 0, 0, pwfex, NULL);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_CANT_OPEN, "WASAPI: Initialize failed with error 0x" + String::num_uint64(hr, 16) + ".");

	if (p_capture) {
		hr = p_device->audio_client->GetService(IID_IAudioCaptureClient, (void **)&p_device->capture_client);
	} else {
		hr = p_device->audio_client->GetService(IID_IAudioRenderClient, (void **)&p_device->render_client);
	}
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	// Free memory
	CoTaskMemFree(pwfex);
	SAFE_RELEASE(device)

	return OK;
}

Error AudioDriverWASAPI::init_render_device(bool reinit) {
	Error err = audio_device_init(&audio_output, false, reinit);
	if (err != OK)
		return err;

	switch (audio_output.channels) {
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

	UINT32 max_frames;
	HRESULT hr = audio_output.audio_client->GetBufferSize(&max_frames);
	ERR_FAIL_COND_V(hr != S_OK, ERR_CANT_OPEN);

	// Due to WASAPI Shared Mode we have no control of the buffer size
	buffer_frames = max_frames;

	// Sample rate is independent of channels (ref: https://stackoverflow.com/questions/11048825/audio-sample-frequency-rely-on-channels)
	samples_in.resize(buffer_frames * channels);

	input_position = 0;
	input_size = 0;

	print_verbose("WASAPI: detected " + itos(channels) + " channels");
	print_verbose("WASAPI: audio buffer frames: " + itos(buffer_frames) + " calculated latency: " + itos(buffer_frames * 1000 / mix_rate) + "ms");

	return OK;
}

Error AudioDriverWASAPI::init_capture_device(bool reinit) {
	Error err = audio_device_init(&audio_input, true, reinit);
	if (err != OK)
		return err;

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

Error AudioDriverWASAPI::finish_render_device() {
	return audio_device_finish(&audio_output);
}

Error AudioDriverWASAPI::finish_capture_device() {
	return audio_device_finish(&audio_input);
}

Error AudioDriverWASAPI::init() {
	mix_rate = GLOBAL_GET("audio/mix_rate");

	Error err = init_render_device();
	if (err != OK) {
		ERR_PRINT("WASAPI: init_render_device error");
	}

	exit_thread.clear();

	thread.start(thread_func, this);

	return OK;
}

int AudioDriverWASAPI::get_mix_rate() const {
	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverWASAPI::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(channels);
}

Array AudioDriverWASAPI::audio_device_get_list(bool p_capture) {
	Array list;
	IMMDeviceCollection *devices = NULL;
	IMMDeviceEnumerator *enumerator = NULL;

	list.push_back(String("Default"));

	CoInitialize(NULL);

	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, NULL, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V(hr != S_OK, Array());

	hr = enumerator->EnumAudioEndpoints(p_capture ? eCapture : eRender, DEVICE_STATE_ACTIVE, &devices);
	ERR_FAIL_COND_V(hr != S_OK, Array());

	UINT count = 0;
	hr = devices->GetCount(&count);
	ERR_FAIL_COND_V(hr != S_OK, Array());

	for (ULONG i = 0; i < count; i++) {
		IMMDevice *device = NULL;

		hr = devices->Item(i, &device);
		ERR_BREAK(hr != S_OK);

		IPropertyStore *props = NULL;
		hr = device->OpenPropertyStore(STGM_READ, &props);
		ERR_BREAK(hr != S_OK);

		PROPVARIANT propvar;
		PropVariantInit(&propvar);

		hr = props->GetValue(PKEY_Device_FriendlyName, &propvar);
		ERR_BREAK(hr != S_OK);

		list.push_back(String(propvar.pwszVal));

		PropVariantClear(&propvar);
		props->Release();
		device->Release();
	}

	devices->Release();
	enumerator->Release();
	return list;
}

Array AudioDriverWASAPI::get_device_list() {
	return audio_device_get_list(false);
}

String AudioDriverWASAPI::get_device() {
	lock();
	String name = audio_output.device_name;
	unlock();

	return name;
}

void AudioDriverWASAPI::set_device(String device) {
	lock();
	audio_output.new_device = device;
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

void AudioDriverWASAPI::thread_func(void *p_udata) {
	AudioDriverWASAPI *ad = (AudioDriverWASAPI *)p_udata;
	uint32_t avail_frames = 0;
	uint32_t write_ofs = 0;

	while (!ad->exit_thread.is_set()) {
		uint32_t read_frames = 0;
		uint32_t written_frames = 0;

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

		if (avail_frames > 0 && ad->audio_output.audio_client) {
			UINT32 cur_frames;
			bool invalidated = false;
			HRESULT hr = ad->audio_output.audio_client->GetCurrentPadding(&cur_frames);
			if (hr == S_OK) {
				// Check how much frames are available on the WASAPI buffer
				UINT32 write_frames = MIN(ad->buffer_frames - cur_frames, avail_frames);
				if (write_frames > 0) {
					BYTE *buffer = NULL;
					hr = ad->audio_output.render_client->GetBuffer(write_frames, &buffer);
					if (hr == S_OK) {
						// We're using WASAPI Shared Mode so we must convert the buffer
						if (ad->channels == ad->audio_output.channels) {
							for (unsigned int i = 0; i < write_frames * ad->channels; i++) {
								ad->write_sample(ad->audio_output.format_tag, ad->audio_output.bits_per_sample, buffer, i, ad->samples_in.write[write_ofs++]);
							}
						} else {
							for (unsigned int i = 0; i < write_frames; i++) {
								for (unsigned int j = 0; j < MIN(ad->channels, ad->audio_output.channels); j++) {
									ad->write_sample(ad->audio_output.format_tag, ad->audio_output.bits_per_sample, buffer, i * ad->audio_output.channels + j, ad->samples_in.write[write_ofs++]);
								}
								if (ad->audio_output.channels > ad->channels) {
									for (unsigned int j = ad->channels; j < ad->audio_output.channels; j++) {
										ad->write_sample(ad->audio_output.format_tag, ad->audio_output.bits_per_sample, buffer, i * ad->audio_output.channels + j, 0);
									}
								}
							}
						}

						hr = ad->audio_output.render_client->ReleaseBuffer(write_frames, 0);
						if (hr != S_OK) {
							ERR_PRINT("WASAPI: Release buffer error");
						}

						avail_frames -= write_frames;
						written_frames += write_frames;
					} else if (hr == AUDCLNT_E_DEVICE_INVALIDATED) {
						// Device is not valid anymore, reopen it

						Error err = ad->finish_render_device();
						if (err != OK) {
							ERR_PRINT("WASAPI: finish_render_device error");
						} else {
							// We reopened the device and samples_in may have resized, so invalidate the current avail_frames
							avail_frames = 0;
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
				// Device is not valid anymore
				WARN_PRINT("WASAPI: Current device invalidated, closing device");

				Error err = ad->finish_render_device();
				if (err != OK) {
					ERR_PRINT("WASAPI: finish_render_device error");
				}
			}
		}

		// If we're using the Default device and it changed finish it so we'll re-init the device
		if (ad->audio_output.device_name == "Default" && default_render_device_changed) {
			Error err = ad->finish_render_device();
			if (err != OK) {
				ERR_PRINT("WASAPI: finish_render_device error");
			}

			default_render_device_changed = false;
		}

		// User selected a new device, finish the current one so we'll init the new device
		if (ad->audio_output.device_name != ad->audio_output.new_device) {
			ad->audio_output.device_name = ad->audio_output.new_device;
			Error err = ad->finish_render_device();
			if (err != OK) {
				ERR_PRINT("WASAPI: finish_render_device error");
			}
		}

		if (!ad->audio_output.audio_client) {
			Error err = ad->init_render_device(true);
			if (err == OK) {
				ad->start();
			}

			avail_frames = 0;
			write_ofs = 0;
		}

		if (ad->audio_input.active.is_set()) {
			UINT32 packet_length = 0;
			BYTE *data;
			UINT32 num_frames_available;
			DWORD flags;

			HRESULT hr = ad->audio_input.capture_client->GetNextPacketSize(&packet_length);
			if (hr == S_OK) {
				while (packet_length != 0) {
					hr = ad->audio_input.capture_client->GetBuffer(&data, &num_frames_available, &flags, NULL, NULL);
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

			// If we're using the Default device and it changed finish it so we'll re-init the device
			if (ad->audio_input.device_name == "Default" && default_capture_device_changed) {
				Error err = ad->finish_capture_device();
				if (err != OK) {
					ERR_PRINT("WASAPI: finish_capture_device error");
				}

				default_capture_device_changed = false;
			}

			// User selected a new device, finish the current one so we'll init the new device
			if (ad->audio_input.device_name != ad->audio_input.new_device) {
				ad->audio_input.device_name = ad->audio_input.new_device;
				Error err = ad->finish_capture_device();
				if (err != OK) {
					ERR_PRINT("WASAPI: finish_capture_device error");
				}
			}

			if (!ad->audio_input.audio_client) {
				Error err = ad->init_capture_device(true);
				if (err == OK) {
					ad->capture_start();
				}
			}
		}

		ad->stop_counting_ticks();
		ad->unlock();

		// Let the thread rest a while if we haven't read or write anything
		if (written_frames == 0 && read_frames == 0) {
			OS::get_singleton()->delay_usec(1000);
		}
	}
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
	thread.wait_to_finish();

	finish_capture_device();
	finish_render_device();
}

Error AudioDriverWASAPI::capture_start() {
	Error err = init_capture_device();
	if (err != OK) {
		ERR_PRINT("WASAPI: init_capture_device error");
		return err;
	}

	if (audio_input.active.is_set()) {
		return FAILED;
	}

	audio_input.audio_client->Start();
	audio_input.active.set();
	return OK;
}

Error AudioDriverWASAPI::capture_stop() {
	if (audio_input.active.is_set()) {
		audio_input.audio_client->Stop();
		audio_input.active.clear();

		return OK;
	}

	return FAILED;
}

void AudioDriverWASAPI::capture_set_device(const String &p_name) {
	lock();
	audio_input.new_device = p_name;
	unlock();
}

Array AudioDriverWASAPI::capture_get_device_list() {
	return audio_device_get_list(true);
}

String AudioDriverWASAPI::capture_get_device() {
	lock();
	String name = audio_input.device_name;
	unlock();

	return name;
}

AudioDriverWASAPI::AudioDriverWASAPI() {
	samples_in.clear();

	channels = 0;
	mix_rate = 0;
	buffer_frames = 0;
}

#endif
