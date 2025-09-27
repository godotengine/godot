/**************************************************************************/
/*  audio_driver_sdl.h                                                    */
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

#include "servers/audio_server.h"

#include <SDL3/SDL_audio.h>

union SDL_Event;

class AudioDriverSDL : public AudioDriver {
	// For easier memory management.
	struct AudioDevice {
		SDL_AudioDeviceID id = 0;

		bool is_opened() { return id != 0; }
		void close();

		AudioDevice &operator=(AudioDevice &&p_other);

		AudioDevice() = default;
		AudioDevice(SDL_AudioDeviceID p_id);

		~AudioDevice() { close(); }
	};

	// For easier memory management.
	struct AudioStream {
		SDL_AudioStream *stream = nullptr;

		bool is_created() { return stream != nullptr; }
		void destroy();

		AudioStream &operator=(AudioStream &&p_other);

		AudioStream() = default;
		AudioStream(const SDL_AudioSpec *p_src_spec, const SDL_AudioSpec *p_dst_spec);

		~AudioStream() { destroy(); }
	};

	class AudioStreamManager {
		String device_name = "Default";
		SDL_AudioSpec spec = {};
		bool has_event_watch = false;

		AudioDevice device;
		AudioStream stream;

		static bool SDLCALL event_watch(void *userdata, SDL_Event *event);

		static void SDLCALL output_stream_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount);
		static void SDLCALL input_stream_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount);

		bool update_spec(const AudioDevice &p_device, bool p_input);

		SDL_AudioDeviceID get_device_id(bool p_input);

	public:
		bool init(bool p_input);
		bool start(bool p_input);

		void stop();
		void finish();

		SDL_AudioSpec get_spec() const;

		String get_device_name() const;
		bool set_device_name(const String &p_name, bool p_input);

		static PackedStringArray get_device_list(bool p_input);
	};

	Mutex mutex;

	AudioStreamManager output_info;
	AudioStreamManager input_info;

	// Use `int` for size as this is what SDL uses and use `tight` as it will be rarely resized.
	LocalVector<int8_t, int, false, true> samples_in;
	int input_buffer_size = 0;
	int input_buffer_position = 0;

	static AudioDriverSDL *singleton;

public:
	virtual const char *get_name() const override {
		return "SDL";
	}

	virtual Error init() override;
	virtual void start() override;
	virtual int get_mix_rate() const override;
	virtual SpeakerMode get_speaker_mode() const override;
	virtual float get_latency() override;

	virtual void lock() override;
	virtual void unlock() override;
	virtual void finish() override;

	virtual PackedStringArray get_output_device_list() override;
	virtual String get_output_device() override;
	virtual void set_output_device(const String &p_name) override;

	virtual Error input_start() override;
	virtual Error input_stop() override;

	virtual PackedStringArray get_input_device_list() override;
	virtual String get_input_device() override;
	virtual void set_input_device(const String &p_name) override;

	static AudioDriverSDL *get_singleton() { return singleton; }

	AudioDriverSDL() { singleton = this; }
	~AudioDriverSDL() { singleton = nullptr; }
};
