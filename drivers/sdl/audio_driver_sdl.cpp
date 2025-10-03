/**************************************************************************/
/*  audio_driver_sdl.cpp                                                  */
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

#include "audio_driver_sdl.h"

#include <SDL3/SDL.h>

template <>
class MutexLock<AudioDriverSDL::AudioStreamManager> {
	SDL_AudioStream *stream;

public:
	_ALWAYS_INLINE_ explicit MutexLock(const AudioDriverSDL::AudioStreamManager &p_manager) :
			stream(p_manager.stream) {
		if (likely(stream != nullptr)) {
			ERR_FAIL_COND_MSG(!SDL_LockAudioStream(stream), SDL_GetError());
		}
	}

	_ALWAYS_INLINE_ ~MutexLock() {
		if (likely(stream != nullptr)) {
			ERR_FAIL_COND_MSG(!SDL_UnlockAudioStream(stream), SDL_GetError());
		}
	}
};

AudioDriverSDL *AudioDriverSDL::singleton = nullptr;

bool SDLCALL AudioDriverSDL::AudioStreamManager::event_watch(void *userdata, SDL_Event *event) {
	AudioDriverSDL *ad = AudioDriverSDL::get_singleton();
	AudioStreamManager *manager = static_cast<AudioStreamManager *>(userdata);

	switch (event->type) {
		case SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED: {
			MutexLock output_lock(ad->output_manager);
			MutexLock input_lock(ad->input_manager);
			MutexLock lock(ad->mutex);
			if (event->adevice.which == manager->device_id) {
				manager->update_spec();
			}
			break;
		}

		case SDL_EVENT_AUDIO_DEVICE_REMOVED: {
			MutexLock output_lock(ad->output_manager);
			MutexLock input_lock(ad->input_manager);
			MutexLock lock(ad->mutex);
			if (event->adevice.which == manager->device_id) {
				manager->set_device_name("Default");
			}
			break;
		}

		default:
			break;
	}
	return true;
}

void SDLCALL AudioDriverSDL::AudioStreamManager::output_stream_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount) {
	AudioDriverSDL *ad = AudioDriverSDL::get_singleton();
	MutexLock lock(ad->mutex);

	while (additional_amount != 0) {
		int len = MIN(additional_amount, ad->samples_in.size());

		ad->audio_server_process(len / SDL_AUDIO_FRAMESIZE(ad->output_manager.spec), (int32_t *)ad->samples_in.ptr());
		ERR_FAIL_COND_MSG(!SDL_PutAudioStreamData(stream, ad->samples_in.ptr(), len), SDL_GetError());

		additional_amount -= len;
	}
}

void SDLCALL AudioDriverSDL::AudioStreamManager::input_stream_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount) {
	AudioDriverSDL *ad = AudioDriverSDL::get_singleton();
	MutexLock lock(ad->mutex);

	int8_t *input_buffer_ptr = (int8_t *)ad->input_buffer.ptrw();

	int frames = SDL_GetAudioStreamData(stream, input_buffer_ptr + ad->input_buffer_position, ad->input_buffer_size - ad->input_buffer_position);
	ERR_FAIL_COND_MSG(frames == -1, SDL_GetError());

	if (ad->input_buffer_position + frames < ad->input_buffer_size) {
		ad->input_buffer_position += frames;
	} else {
		// SDL fully filled the buffer, try to write.extra data.
		int leftover_frames = SDL_GetAudioStreamData(stream, input_buffer_ptr, ad->input_buffer_position);
		ERR_FAIL_COND_MSG(leftover_frames == -1, SDL_GetError());

		ad->input_buffer_position = leftover_frames;
		frames += leftover_frames;
		if (unlikely(frames < additional_amount)) {
			WARN_PRINT("Input buffer isn't large enough to get all the data.");
		}
	}
	ad->input_buffer_wrote(frames / sizeof(int32_t));
}

bool AudioDriverSDL::AudioStreamManager::update_spec() {
	AudioDriverSDL *ad = AudioDriverSDL::get_singleton();

	SDL_AudioSpec device_spec;
	int sample_frames = 0;
	ERR_FAIL_COND_V_MSG(!SDL_GetAudioDeviceFormat(device_id, &device_spec, &sample_frames), false, SDL_GetError());

	spec = device_spec;

	if (is_input) {
		// Godot always uses 2 channels for input.
		spec.channels = 2;

		spec.freq = ad->output_manager.spec.freq;
	} else {
		// Godot always mixes an even amount of channels.
		switch (device_spec.channels) {
			case 1:
			case 3:
			case 5:
			case 7:
				spec.channels = device_spec.channels + 1;
				break;

			case 2:
			case 4:
			case 6:
			case 8:
				break;

			default:
				// Shouldn't ever happen, but just in case.
				WARN_PRINT(vformat("SDL: Unsupported number of channels: %d. Defaulting to 2 channels.", device_spec.channels));
				spec.channels = 2;
				break;
		}
	}

	// Godot exposes the signed 32-bit formatted buffers.
	// TODO: Expose the internal float buffer.
	spec.format = SDL_AUDIO_S32;

	if (is_input) {
		ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamFormat(stream, nullptr, &spec), false, SDL_GetError());
	} else {
		ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamFormat(stream, &spec, nullptr), false, SDL_GetError());

		if (ad->input_manager.device_id != 0 && ad->input_manager.spec.freq != spec.freq) {
			ad->input_manager.update_spec();
		}
	}

	if (is_input) {
		ad->input_buffer_position = 0;

		ad->input_buffer_init(sample_frames * spec.freq / device_spec.freq);
		ad->input_buffer_size = ad->input_buffer.size() * sizeof(int32_t);
	} else {
		ad->samples_in.resize(sample_frames * SDL_AUDIO_FRAMESIZE(spec));
	}

	return true;
}

bool AudioDriverSDL::AudioStreamManager::init_stream() {
	if (unlikely(stream == nullptr)) {
		stream = SDL_CreateAudioStream(nullptr, nullptr);
		ERR_FAIL_COND_V_MSG(stream == nullptr, false, SDL_GetError());

		// These only fail if `stream` is nullptr, so the `stream` shouldn't leak on error.
		if (is_input) {
			ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamPutCallback(stream, input_stream_callback, nullptr), false, SDL_GetError());
		} else {
			ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamGetCallback(stream, output_stream_callback, nullptr), false, SDL_GetError());
		}
	}

	return true;
}

bool AudioDriverSDL::AudioStreamManager::init() {
	if (!has_event_watch) {
		ERR_FAIL_COND_V_MSG(!SDL_AddEventWatch(event_watch, this), false, SDL_GetError());
		has_event_watch = true;
	}

	device_id = SDL_OpenAudioDevice(get_device_id_by_name(), nullptr);
	ERR_FAIL_COND_V_MSG(device_id == 0, false, SDL_GetError());

	return update_spec();
}

bool AudioDriverSDL::AudioStreamManager::start() {
	ERR_FAIL_COND_V(!SDL_BindAudioStream(device_id, stream), false);
	return true;
}

void AudioDriverSDL::AudioStreamManager::stop() {
	if (likely(device_id != 0)) {
		SDL_CloseAudioDevice(device_id);
		device_id = 0;
	}
}

void AudioDriverSDL::AudioStreamManager::finish() {
	stop();

	if (likely(has_event_watch)) {
		SDL_RemoveEventWatch(event_watch, this);
		has_event_watch = false;
	}
	if (likely(stream != nullptr)) {
		SDL_DestroyAudioStream(stream);
		stream = nullptr;
	}
}

SDL_AudioSpec AudioDriverSDL::AudioStreamManager::get_spec() const {
	return spec;
}

SDL_AudioDeviceID AudioDriverSDL::AudioStreamManager::get_device_id_by_name() {
	if (device_name == "Default") {
		return is_input ? SDL_AUDIO_DEVICE_DEFAULT_RECORDING : SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
	}

	int devices_count = 0;
	SDL_AudioDeviceID *devices = nullptr;

	if (is_input) {
		devices = SDL_GetAudioRecordingDevices(&devices_count);
	} else {
		devices = SDL_GetAudioPlaybackDevices(&devices_count);
	}
	ERR_FAIL_COND_V_MSG(devices == nullptr, 0, SDL_GetError());

	for (int i = 0; i < devices_count; i++) {
		const char *name = SDL_GetAudioDeviceName(devices[i]);
		ERR_CONTINUE_MSG(name == nullptr, SDL_GetError());

		if (device_name == name) {
			SDL_AudioDeviceID new_device_id = devices[i];
			SDL_free(devices);
			return new_device_id;
		}
	}

	SDL_free(devices);
	device_name = "Default";
	return get_device_id_by_name();
}

PackedStringArray AudioDriverSDL::AudioStreamManager::get_device_list(bool p_input) {
	Vector<String> device_list = { "Default" };
	int devices_count = 0;
	SDL_AudioDeviceID *devices = nullptr;

	if (p_input) {
		devices = SDL_GetAudioRecordingDevices(&devices_count);
	} else {
		devices = SDL_GetAudioPlaybackDevices(&devices_count);
	}
	ERR_FAIL_COND_V_MSG(devices == nullptr, device_list, SDL_GetError());

	device_list.resize(devices_count + 1);
	String *device_names_ptr = device_list.ptrw();

	int device_names_pos = 1;
	for (int i = 0; i < devices_count; i++) {
		const char *name = SDL_GetAudioDeviceName(devices[i]);
		ERR_CONTINUE_MSG(name == nullptr, SDL_GetError());

		device_names_ptr[device_names_pos] = name;
		device_names_pos++;
	}

	device_list.resize(device_names_pos);
	SDL_free(devices);
	return device_list;
}

String AudioDriverSDL::AudioStreamManager::get_device_name() const {
	return device_name;
}

bool AudioDriverSDL::AudioStreamManager::set_device_name(const String &p_name) {
	if (device_name == p_name) {
		return true;
	}

	stop();
	device_name = p_name;
	if (unlikely(!init() || !start())) {
		stop();
		return p_name != "Default" ? set_device_name("Default") : false;
	}

	return true;
}

Error AudioDriverSDL::init() {
	ERR_FAIL_COND_V_MSG(!SDL_Init(SDL_INIT_AUDIO), FAILED, SDL_GetError());

	MutexLock lock(mutex);
	if (unlikely(!output_manager.init_stream() || !input_manager.init_stream())) {
		output_manager.finish();
		input_manager.finish();
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return FAILED;
	}

	lock.temp_unlock();
	MutexLock output_lock(output_manager);
	MutexLock input_lock(input_manager);
	lock.temp_relock();
	if (unlikely(!output_manager.init())) {
		output_manager.finish();
		input_manager.finish();
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return FAILED;
	}
	return OK;
}

void AudioDriverSDL::start() {
	MutexLock output_lock(output_manager);
	MutexLock lock(mutex);
	if (unlikely(!output_manager.start())) {
		output_manager.stop();
	}
}

int AudioDriverSDL::get_mix_rate() const {
	return output_manager.get_spec().freq;
}

AudioDriver::SpeakerMode AudioDriverSDL::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(output_manager.get_spec().channels);
}

float AudioDriverSDL::get_latency() {
	// TODO: Implement `get_latency` later as it's not supported by SDL.
	return 0;
}

void AudioDriverSDL::lock() {
	mutex.lock();
}

void AudioDriverSDL::unlock() {
	mutex.unlock();
}

void AudioDriverSDL::finish() {
	MutexLock output_lock(output_manager);
	MutexLock input_lock(input_manager);
	MutexLock lock(mutex);

	output_manager.finish();
	input_manager.finish();

	SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

PackedStringArray AudioDriverSDL::get_output_device_list() {
	return AudioStreamManager::get_device_list(false);
}

String AudioDriverSDL::get_output_device() {
	MutexLock lock(mutex);
	return output_manager.get_device_name();
}

void AudioDriverSDL::set_output_device(const String &p_name) {
	MutexLock output_lock(output_manager);
	MutexLock input_lock(input_manager);
	MutexLock lock(mutex);
	output_manager.set_device_name(p_name);
}

Error AudioDriverSDL::input_start() {
	MutexLock output_lock(output_manager);
	MutexLock input_lock(input_manager);
	MutexLock lock(mutex);
	if (unlikely(!input_manager.init() || !input_manager.start())) {
		input_manager.stop();
		return FAILED;
	}
	return OK;
}

Error AudioDriverSDL::input_stop() {
	MutexLock input_lock(input_manager);
	MutexLock lock(mutex);
	input_manager.stop();
	return OK;
}

PackedStringArray AudioDriverSDL::get_input_device_list() {
	return AudioStreamManager::get_device_list(true);
}

String AudioDriverSDL::get_input_device() {
	MutexLock lock(mutex);
	return input_manager.get_device_name();
}

void AudioDriverSDL::set_input_device(const String &p_name) {
	MutexLock output_lock(output_manager);
	MutexLock input_lock(input_manager);
	MutexLock lock(mutex);
	input_manager.set_device_name(p_name);
}
