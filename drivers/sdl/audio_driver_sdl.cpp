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

AudioDriverSDL *AudioDriverSDL::singleton = nullptr;

void AudioDriverSDL::AudioDevice::close() {
	if (unlikely(id == 0)) {
		return;
	}

	SDL_CloseAudioDevice(id);
	id = 0;
}

AudioDriverSDL::AudioDevice &AudioDriverSDL::AudioDevice::operator=(AudioDevice &&p_other) {
	close();

	id = p_other.id;
	p_other.id = 0;

	return *this;
}

AudioDriverSDL::AudioStream::AudioStream(const SDL_AudioSpec *p_src_spec, const SDL_AudioSpec *p_dst_spec) {
	stream = SDL_CreateAudioStream(p_src_spec, p_dst_spec);
}

void AudioDriverSDL::AudioStream::destroy() {
	if (unlikely(stream == nullptr)) {
		return;
	}

	SDL_DestroyAudioStream(stream);
	stream = nullptr;
}

AudioDriverSDL::AudioStream &AudioDriverSDL::AudioStream::operator=(AudioStream &&p_other) {
	destroy();

	stream = p_other.stream;
	p_other.stream = nullptr;

	return *this;
}

AudioDriverSDL::AudioDevice::AudioDevice(SDL_AudioDeviceID p_id) {
	id = SDL_OpenAudioDevice(p_id, nullptr);
}

bool SDLCALL AudioDriverSDL::AudioStreamManager::event_watch(void *userdata, SDL_Event *event) {
	AudioDriverSDL *ad = AudioDriverSDL::get_singleton();
	AudioStreamManager *manager = static_cast<AudioStreamManager *>(userdata);

	switch (event->type) {
		case SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED: {
			MutexLock lock(ad->mutex);
			if (event->adevice.which == manager->device.id) {
				print_line("SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED");
				manager->update_spec(manager->device, event->adevice.recording);
			}
			break;
		}

		case SDL_EVENT_AUDIO_DEVICE_REMOVED: {
			MutexLock lock(ad->mutex);
			if (event->adevice.which == manager->device.id) {
				print_line("SDL_EVENT_AUDIO_DEVICE_REMOVED");
				// SDL doesn't emit this event on default devices as they change automatically.
				manager->set_device_name("Default", event->adevice.recording);
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

	// print_line(vformat("ascked: %d, has: %d, total: %d", additional_amount, ad->samples_in.size(), total_amount));
	while (additional_amount != 0) {
		int len = MIN(additional_amount, ad->samples_in.size());

		ad->audio_server_process(len / SDL_AUDIO_FRAMESIZE(ad->output_info.spec), (int32_t *)ad->samples_in.ptr());
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
	print_line(vformat("ascked: %d, has: %d, total: %d, pos: %d", additional_amount, ad->input_buffer_size, total_amount, ad->input_position));
}

bool AudioDriverSDL::AudioStreamManager::update_spec(const AudioDevice &p_device, bool p_input) {
	AudioDriverSDL *ad = AudioDriverSDL::get_singleton();

	SDL_AudioSpec device_spec;
	int sample_frmaes = 0;
	ERR_FAIL_COND_V_MSG(!SDL_GetAudioDeviceFormat(p_device.id, &device_spec, &sample_frmaes), false, SDL_GetError());

	spec = device_spec;

	if (p_input) {
		// Make
		spec.freq = ad->output_info.spec.freq;

		// Godot always uses 2 channels for input.
		spec.channels = 2;
	} else {
		// Godot always mixes an even amount of channels.
		if (device_spec.channels % 2 != 0) {
			spec.channels = device_spec.channels + 1;
		}
	}

	// Godot always uses signed 32-bit format.
	// TODO: Expose the internal float buffer.
	spec.format = SDL_AUDIO_S32;

	if (stream.stream == nullptr) {
		if (p_input) {
			// Will be destroyed automatically if the function fails.
			AudioStream tmp_stream(nullptr, &spec);
			ERR_FAIL_COND_V_MSG(!tmp_stream.is_created(), false, SDL_GetError());
			ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamPutCallback(tmp_stream.stream, input_stream_callback, nullptr), false, SDL_GetError());

			stream = std::move(tmp_stream);
		} else {
			// Will be destroyed automatically if the function fails.
			AudioStream tmp_stream(&spec, nullptr);
			ERR_FAIL_COND_V_MSG(!tmp_stream.is_created(), false, SDL_GetError());
			ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamGetCallback(tmp_stream.stream, output_stream_callback, nullptr), false, SDL_GetError());

			stream = std::move(tmp_stream);
		}
	} else {
		if (p_input) {
			ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamFormat(stream.stream, nullptr, &spec), false, SDL_GetError());
		} else {
			ERR_FAIL_COND_V_MSG(!SDL_SetAudioStreamFormat(stream.stream, &spec, nullptr), false, SDL_GetError());
		}
	}

	int size = sample_frmaes * SDL_AUDIO_FRAMESIZE(spec);
	if (p_input) {
		ad->input_buffer_position = 0;

		// `input_buffer_init` accounts for the format and the channel count;
		ad->input_buffer_init(size / (sizeof(int32_t) * 2));
		ad->input_buffer_size = ad->input_buffer.size() * sizeof(int32_t);
	} else {
		ad->samples_in.resize(size);
	}

	return true;
}

bool AudioDriverSDL::AudioStreamManager::init(bool p_input) {
	if (!has_event_watch) {
		ERR_FAIL_COND_V_MSG(!SDL_AddEventWatch(event_watch, this), false, SDL_GetError());
		has_event_watch = true;
	}

	// Will be closed automatically if the function fails.
	AudioDevice tmp_device(get_device_id(p_input));
	ERR_FAIL_COND_V_MSG(!tmp_device.is_opened(), false, SDL_GetError());

	SDL_AudioSpec device_spec;
	ERR_FAIL_COND_V_MSG(!SDL_GetAudioDeviceFormat(tmp_device.id, &device_spec, nullptr), false, SDL_GetError());

	if (unlikely(!update_spec(tmp_device, p_input))) {
		return false;
	}

	device = std::move(tmp_device);
	return true;
}

bool AudioDriverSDL::AudioStreamManager::start(bool p_input) {
	if (unlikely(!SDL_BindAudioStream(device.id, stream.stream))) {
		stop();
		ERR_PRINT(SDL_GetError());
		return false;
	}
	return true;
}

void AudioDriverSDL::AudioStreamManager::stop() {
	device.close();
}

void AudioDriverSDL::AudioStreamManager::finish() {
	stop();
	stream.destroy();

	if (has_event_watch) {
		SDL_RemoveEventWatch(event_watch, this);
	}
}

SDL_AudioSpec AudioDriverSDL::AudioStreamManager::get_spec() const {
	return spec;
}

SDL_AudioDeviceID AudioDriverSDL::AudioStreamManager::get_device_id(bool p_input) {
	if (device_name == "Default") {
		return p_input ? SDL_AUDIO_DEVICE_DEFAULT_RECORDING : SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
	}

	int devices_count = 0;
	SDL_AudioDeviceID *devices = nullptr;

	if (p_input) {
		devices = SDL_GetAudioRecordingDevices(&devices_count);
	} else {
		devices = SDL_GetAudioPlaybackDevices(&devices_count);
	}
	ERR_FAIL_COND_V_MSG(devices == nullptr, 0, SDL_GetError());

	for (int i = 0; i < devices_count; i++) {
		const char *name = SDL_GetAudioDeviceName(devices[i]);
		ERR_CONTINUE_MSG(name == nullptr, SDL_GetError());

		if (device_name == name) {
			SDL_AudioDeviceID device_id = devices[i];
			SDL_free(devices);
			return device_id;
		}
	}

	SDL_free(devices);
	device_name = "Default";
	return get_device_id(p_input);
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

bool AudioDriverSDL::AudioStreamManager::set_device_name(const String &p_name, bool p_input) {
	if (device_name == p_name) {
		return true;
	}

	stop();
	device_name = p_name;
	return init(p_input) && start(p_input);
}

Error AudioDriverSDL::init() {
	ERR_FAIL_COND_V_MSG(!SDL_InitSubSystem(SDL_INIT_AUDIO), FAILED, SDL_GetError());

	if (!unlikely(output_info.init(false))) {
		SDL_QuitSubSystem(SDL_INIT_AUDIO);
		return FAILED;
	}
	return OK;
}

void AudioDriverSDL::start() {
	MutexLock lock(mutex);
	output_info.start(false);
}

int AudioDriverSDL::get_mix_rate() const {
	return output_info.get_spec().freq;
}

AudioDriver::SpeakerMode AudioDriverSDL::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(output_info.get_spec().channels);
}

float AudioDriverSDL::get_latency() {
	// TODO: Implement get_latency() as it's not supported by SDL.
	return 0;
}

void AudioDriverSDL::lock() {
	mutex.lock();
}

void AudioDriverSDL::unlock() {
	mutex.unlock();
}

void AudioDriverSDL::finish() {
	MutexLock lock(mutex);

	output_info.finish();
	input_info.finish();

	SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

PackedStringArray AudioDriverSDL::get_output_device_list() {
	return AudioStreamManager::get_device_list(false);
}

String AudioDriverSDL::get_output_device() {
	MutexLock lock(mutex);
	return output_info.get_device_name();
}

void AudioDriverSDL::set_output_device(const String &p_name) {
	MutexLock lock(mutex);
	output_info.set_device_name(p_name, false);
}

Error AudioDriverSDL::input_start() {
	MutexLock lock(mutex);
	return input_info.init(true) && input_info.start(true) ? OK : FAILED;
}

Error AudioDriverSDL::input_stop() {
	MutexLock lock(mutex);
	input_info.stop();
	return OK;
}

PackedStringArray AudioDriverSDL::get_input_device_list() {
	return AudioStreamManager::get_device_list(true);
}

String AudioDriverSDL::get_input_device() {
	MutexLock lock(mutex);
	return input_info.get_device_name();
}

void AudioDriverSDL::set_input_device(const String &p_name) {
	MutexLock lock(mutex);
	input_info.set_device_name(p_name, true);
}
