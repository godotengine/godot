/**************************************************************************/
/*  audio_driver.h                                                        */
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

#include "core/templates/vector.h"
#include "core/variant/variant.h"
#include "servers/audio/audio_server_enums.h"

struct AudioFrame;
class AudioDriverDummy;
class AudioSample;
class AudioStream;
class AudioSamplePlayback;

class AudioDriver {
	static AudioDriver *singleton;
	uint64_t _last_mix_time = 0;
	uint64_t _last_mix_frames = 0;

#ifdef DEBUG_ENABLED
	SafeNumeric<uint64_t> prof_ticks;
	SafeNumeric<uint64_t> prof_time;
#endif

protected:
	Vector<int32_t> input_buffer;
	unsigned int input_position = 0;
	unsigned int input_size = 0;

	void audio_server_process(int p_frames, int32_t *p_buffer, bool p_update_mix_time = true);
	void update_mix_time(int p_frames);
	void input_buffer_init(int driver_buffer_frames);
	void input_buffer_write(int32_t sample);

	int _get_configured_mix_rate();

#ifdef DEBUG_ENABLED
	void start_counting_ticks();
	void stop_counting_ticks();
#else
	_FORCE_INLINE_ void start_counting_ticks() {}
	_FORCE_INLINE_ void stop_counting_ticks() {}
#endif

public:
	double get_time_since_last_mix(); //useful for video -> audio sync
	double get_time_to_next_mix();

	enum SpeakerMode {
		SPEAKER_MODE_STEREO = AuSE::SPEAKER_MODE_STEREO,
		SPEAKER_SURROUND_31 = AuSE::SPEAKER_SURROUND_31,
		SPEAKER_SURROUND_51 = AuSE::SPEAKER_SURROUND_51,
		SPEAKER_SURROUND_71 = AuSE::SPEAKER_SURROUND_71,
	};

	static AudioDriver *get_singleton();
	void set_singleton();

	// Virtual API to implement.

	virtual const char *get_name() const = 0;

	virtual Error init() = 0;
	virtual void start() = 0;
	virtual int get_mix_rate() const = 0;
	virtual int get_input_mix_rate() const { return get_mix_rate(); }
	virtual SpeakerMode get_speaker_mode() const = 0;
	virtual float get_latency() { return 0; }

	virtual void lock() = 0;
	virtual void unlock() = 0;
	virtual void finish() = 0;

	virtual PackedStringArray get_output_device_list();
	virtual String get_output_device();
	virtual void set_output_device(const String &p_name) {}

	virtual Error input_start() { return FAILED; }
	virtual Error input_stop() { return FAILED; }

	virtual PackedStringArray get_input_device_list();
	virtual String get_input_device() { return "Default"; }
	virtual void set_input_device(const String &p_name) {}

	//

	SpeakerMode get_speaker_mode_by_total_channels(int p_channels) const;
	int get_total_channels_by_speaker_mode(SpeakerMode) const;

	Vector<int32_t> get_input_buffer() { return input_buffer; }
	unsigned int get_input_position() { return input_position; }
	unsigned int get_input_size() { return input_size; }

#ifdef DEBUG_ENABLED
	uint64_t get_profiling_time() const { return prof_time.get(); }
	void reset_profiling_time() { prof_time.set(0); }
#endif

	// Samples handling.
	virtual bool is_stream_registered_as_sample(const Ref<AudioStream> &p_stream) const {
		return false;
	}
	virtual void register_sample(const Ref<AudioSample> &p_sample) {}
	virtual void unregister_sample(const Ref<AudioSample> &p_sample) {}
	virtual void start_sample_playback(const Ref<AudioSamplePlayback> &p_playback);
	virtual void stop_sample_playback(const Ref<AudioSamplePlayback> &p_playback) {}
	virtual void set_sample_playback_pause(const Ref<AudioSamplePlayback> &p_playback, bool p_paused) {}
	virtual bool is_sample_playback_active(const Ref<AudioSamplePlayback> &p_playback) { return false; }
	virtual double get_sample_playback_position(const Ref<AudioSamplePlayback> &p_playback) { return false; }
	virtual void update_sample_playback_pitch_scale(const Ref<AudioSamplePlayback> &p_playback, float p_pitch_scale = 0.0f) {}
	virtual void set_sample_playback_bus_volumes_linear(const Ref<AudioSamplePlayback> &p_playback, const HashMap<StringName, Vector<AudioFrame>> &p_bus_volumes) {}

	virtual void set_sample_bus_count(int p_count) {}
	virtual void remove_sample_bus(int p_bus) {}
	virtual void add_sample_bus(int p_at_pos = -1) {}
	virtual void move_sample_bus(int p_bus, int p_to_pos) {}
	virtual void set_sample_bus_send(int p_bus, const StringName &p_send) {}
	virtual void set_sample_bus_volume_db(int p_bus, float p_volume_db) {}
	virtual void set_sample_bus_solo(int p_bus, bool p_enable) {}
	virtual void set_sample_bus_mute(int p_bus, bool p_enable) {}

	AudioDriver() {}
	virtual ~AudioDriver() {}
};

class AudioDriverManager {
	enum {
		MAX_DRIVERS = 10
	};

	static AudioDriver *drivers[MAX_DRIVERS];
	static int driver_count;

	static AudioDriverDummy dummy_driver;

public:
	static const int DEFAULT_MIX_RATE = 44100;

	static void add_driver(AudioDriver *p_driver);
	static void initialize(int p_driver);
	static int get_driver_count();
	static AudioDriver *get_driver(int p_driver);
};
