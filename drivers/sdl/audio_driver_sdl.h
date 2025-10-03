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
	class AudioStreamManager {
		friend class MutexLock<AudioStreamManager>;

		String device_name = "Default";
		SDL_AudioSpec spec = {};
		bool has_event_watch = false;

		SDL_AudioDeviceID device_id = 0;
		SDL_AudioStream *stream = nullptr;

		bool is_input = false;

		static bool SDLCALL event_watch(void *userdata, SDL_Event *event);

		static void SDLCALL output_stream_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount);
		static void SDLCALL input_stream_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount);

		// Requires `output_manager`, `input_manager` and `AudioDriverSDL` to be locked.
		bool update_spec();

		// Requires `AudioDriverSDL` to be locked.
		SDL_AudioDeviceID get_device_id_by_name();

	public:
		// Requires `AudioDriverSDL` to be locked.
		bool init_stream();
		// Requires `output_manager`, `input_manager` and `AudioDriverSDL` to be locked.
		bool init();
		// Requires `this` and `AudioDriverSDL` to be locked.
		bool start();

		// Requires `this` and `AudioDriverSDL` to be locked.
		void stop();
		// Requires `this` and `AudioDriverSDL` to be locked.
		void finish();

		// Requires `AudioDriverSDL` to be locked.
		SDL_AudioSpec get_spec() const;

		// Requires `AudioDriverSDL` to be locked.
		String get_device_name() const;
		// Requires `output_manager`, `input_manager` and `AudioDriverSDL` to be locked.
		bool set_device_name(const String &p_name);

		static PackedStringArray get_device_list(bool p_input);

		AudioStreamManager(bool p_input) :
				is_input(p_input) {}
		~AudioStreamManager() { finish(); }
	};

	friend class MutexLock<AudioStreamManager>;

	// It's important for `mutex` to be locked after locking `output_manager` and `input_manager`
	// because the SDL callbacks lock `output_manager` or `input_manager` first and only then lock `mutex`.
	Mutex mutex;

	AudioStreamManager output_manager = AudioStreamManager(false);
	AudioStreamManager input_manager = AudioStreamManager(true);

	// Use `int` for size as this is what SDL uses.
	LocalVector<int8_t, int> samples_in;
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
