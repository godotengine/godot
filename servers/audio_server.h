/*************************************************************************/
/*  audio_server.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef AUDIO_SERVER_H
#define AUDIO_SERVER_H

#include "core/class_db.h"
#include "core/math/audio_frame.h"
#include "core/os/os.h"
#include "core/variant.h"
#include "servers/audio/audio_effect.h"

class AudioDriverDummy;
class AudioStream;
class AudioStreamSample;

class AudioDriver {
	static AudioDriver *singleton;
	uint64_t _last_mix_time;
	uint64_t _last_mix_frames;

#ifdef DEBUG_ENABLED
	uint64_t prof_ticks;
	uint64_t prof_time;
#endif

protected:
	Vector<int32_t> input_buffer;
	unsigned int input_position;
	unsigned int input_size;

	void audio_server_process(int p_frames, int32_t *p_buffer, bool p_update_mix_time = true);
	void update_mix_time(int p_frames);
	void input_buffer_init(int driver_buffer_frames);
	void input_buffer_write(int32_t sample);

#ifdef DEBUG_ENABLED
	_FORCE_INLINE_ void start_counting_ticks() { prof_ticks = OS::get_singleton()->get_ticks_usec(); }
	_FORCE_INLINE_ void stop_counting_ticks() { prof_time += OS::get_singleton()->get_ticks_usec() - prof_ticks; }
#else
	_FORCE_INLINE_ void start_counting_ticks() {}
	_FORCE_INLINE_ void stop_counting_ticks() {}
#endif

public:
	double get_time_since_last_mix() const; //useful for video -> audio sync
	double get_time_to_next_mix() const;

	enum SpeakerMode {
		SPEAKER_MODE_STEREO,
		SPEAKER_SURROUND_31,
		SPEAKER_SURROUND_51,
		SPEAKER_SURROUND_71,
	};

	static AudioDriver *get_singleton();
	void set_singleton();

	virtual const char *get_name() const = 0;

	virtual Error init() = 0;
	virtual void start() = 0;
	virtual int get_mix_rate() const = 0;
	virtual SpeakerMode get_speaker_mode() const = 0;
	virtual Array get_device_list();
	virtual String get_device();
	virtual void set_device(String device) {}
	virtual void lock() = 0;
	virtual void unlock() = 0;
	virtual void finish() = 0;

	virtual Error capture_start() { return FAILED; }
	virtual Error capture_stop() { return FAILED; }
	virtual void capture_set_device(const String &p_name) {}
	virtual String capture_get_device() { return "Default"; }
	virtual Array capture_get_device_list(); // TODO: convert this and get_device_list to PackedStringArray

	virtual float get_latency() { return 0; }

	SpeakerMode get_speaker_mode_by_total_channels(int p_channels) const;
	int get_total_channels_by_speaker_mode(SpeakerMode) const;

	Vector<int32_t> get_input_buffer() { return input_buffer; }
	unsigned int get_input_position() { return input_position; }
	unsigned int get_input_size() { return input_size; }

#ifdef DEBUG_ENABLED
	uint64_t get_profiling_time() const { return prof_time; }
	void reset_profiling_time() { prof_time = 0; }
#endif

	AudioDriver();
	virtual ~AudioDriver() {}
};

class AudioDriverManager {
	enum {

		MAX_DRIVERS = 10
	};

	static const int DEFAULT_MIX_RATE = 44100;
	static const int DEFAULT_OUTPUT_LATENCY = 15;

	static AudioDriver *drivers[MAX_DRIVERS];
	static int driver_count;

	static AudioDriverDummy dummy_driver;

public:
	static void add_driver(AudioDriver *p_driver);
	static void initialize(int p_driver);
	static int get_driver_count();
	static AudioDriver *get_driver(int p_driver);
};

class AudioBusLayout;

class AudioServer : public Object {
	GDCLASS(AudioServer, Object);

public:
	//re-expose this here, as AudioDriver is not exposed to script
	enum SpeakerMode {
		SPEAKER_MODE_STEREO,
		SPEAKER_SURROUND_31,
		SPEAKER_SURROUND_51,
		SPEAKER_SURROUND_71,
	};

	enum {
		AUDIO_DATA_INVALID_ID = -1
	};

	typedef void (*AudioCallback)(void *p_userdata);

private:
	uint64_t mix_time;
	int mix_size;

	uint32_t buffer_size;
	uint64_t mix_count;
	uint64_t mix_frames;
#ifdef DEBUG_ENABLED
	uint64_t prof_time;
#endif

	float channel_disable_threshold_db;
	uint32_t channel_disable_frames;

	int channel_count;
	int to_mix;

	float global_rate_scale;

	struct Bus {
		StringName name;
		bool solo;
		bool mute;
		bool bypass;

		bool soloed;

		//Each channel is a stereo pair.
		struct Channel {
			bool used;
			bool active;
			AudioFrame peak_volume;
			Vector<AudioFrame> buffer;
			Vector<Ref<AudioEffectInstance>> effect_instances;
			uint64_t last_mix_with_audio;
			Channel() {
				last_mix_with_audio = 0;
				used = false;
				active = false;
				peak_volume = AudioFrame(0, 0);
			}
		};

		Vector<Channel> channels;

		struct Effect {
			Ref<AudioEffect> effect;
			bool enabled;
#ifdef DEBUG_ENABLED
			uint64_t prof_time;
#endif
		};

		Vector<Effect> effects;
		float volume_db;
		StringName send;
		int index_cache;
	};

	Vector<Vector<AudioFrame>> temp_buffer; //temp_buffer for each level
	Vector<Bus *> buses;
	Map<StringName, Bus *> bus_map;

	void _update_bus_effects(int p_bus);

	static AudioServer *singleton;

	void init_channels_and_buffers();

	void _mix_step();

	struct CallbackItem {
		AudioCallback callback;
		void *userdata;

		bool operator<(const CallbackItem &p_item) const {
			return (callback == p_item.callback ? userdata < p_item.userdata : callback < p_item.callback);
		}
	};

	Set<CallbackItem> callbacks;
	Set<CallbackItem> update_callbacks;

	friend class AudioDriver;
	void _driver_process(int p_frames, int32_t *p_buffer);

protected:
	static void _bind_methods();

public:
	_FORCE_INLINE_ int get_channel_count() const {
		switch (get_speaker_mode()) {
			case SPEAKER_MODE_STEREO:
				return 1;
			case SPEAKER_SURROUND_31:
				return 2;
			case SPEAKER_SURROUND_51:
				return 3;
			case SPEAKER_SURROUND_71:
				return 4;
		}
		ERR_FAIL_V(1);
	}

	//do not use from outside audio thread
	bool thread_has_channel_mix_buffer(int p_bus, int p_buffer) const;
	AudioFrame *thread_get_channel_mix_buffer(int p_bus, int p_buffer);
	int thread_get_mix_buffer_size() const;
	int thread_find_bus_index(const StringName &p_name);

	void set_bus_count(int p_count);
	int get_bus_count() const;

	void remove_bus(int p_index);
	void add_bus(int p_at_pos = -1);

	void move_bus(int p_bus, int p_to_pos);

	void set_bus_name(int p_bus, const String &p_name);
	String get_bus_name(int p_bus) const;
	int get_bus_index(const StringName &p_bus_name) const;

	int get_bus_channels(int p_bus) const;

	void set_bus_volume_db(int p_bus, float p_volume_db);
	float get_bus_volume_db(int p_bus) const;

	void set_bus_send(int p_bus, const StringName &p_send);
	StringName get_bus_send(int p_bus) const;

	void set_bus_solo(int p_bus, bool p_enable);
	bool is_bus_solo(int p_bus) const;

	void set_bus_mute(int p_bus, bool p_enable);
	bool is_bus_mute(int p_bus) const;

	void set_bus_bypass_effects(int p_bus, bool p_enable);
	bool is_bus_bypassing_effects(int p_bus) const;

	void add_bus_effect(int p_bus, const Ref<AudioEffect> &p_effect, int p_at_pos = -1);
	void remove_bus_effect(int p_bus, int p_effect);

	int get_bus_effect_count(int p_bus);
	Ref<AudioEffect> get_bus_effect(int p_bus, int p_effect);
	Ref<AudioEffectInstance> get_bus_effect_instance(int p_bus, int p_effect, int p_channel = 0);

	void swap_bus_effects(int p_bus, int p_effect, int p_by_effect);

	void set_bus_effect_enabled(int p_bus, int p_effect, bool p_enabled);
	bool is_bus_effect_enabled(int p_bus, int p_effect) const;

	float get_bus_peak_volume_left_db(int p_bus, int p_channel) const;
	float get_bus_peak_volume_right_db(int p_bus, int p_channel) const;

	bool is_bus_channel_active(int p_bus, int p_channel) const;

	void set_global_rate_scale(float p_scale);
	float get_global_rate_scale() const;

	virtual void init();
	virtual void finish();
	virtual void update();
	virtual void load_default_bus_layout();

	/* MISC config */

	virtual void lock();
	virtual void unlock();

	virtual SpeakerMode get_speaker_mode() const;
	virtual float get_mix_rate() const;

	virtual float read_output_peak_db() const;

	static AudioServer *get_singleton();

	virtual double get_output_latency() const;
	virtual double get_time_to_next_mix() const;
	virtual double get_time_since_last_mix() const;

	void add_callback(AudioCallback p_callback, void *p_userdata);
	void remove_callback(AudioCallback p_callback, void *p_userdata);

	void add_update_callback(AudioCallback p_callback, void *p_userdata);
	void remove_update_callback(AudioCallback p_callback, void *p_userdata);

	void set_bus_layout(const Ref<AudioBusLayout> &p_bus_layout);
	Ref<AudioBusLayout> generate_bus_layout() const;

	Array get_device_list();
	String get_device();
	void set_device(String device);

	Array capture_get_device_list();
	String capture_get_device();
	void capture_set_device(const String &p_name);

	AudioServer();
	virtual ~AudioServer();
};

VARIANT_ENUM_CAST(AudioServer::SpeakerMode)

class AudioBusLayout : public Resource {
	GDCLASS(AudioBusLayout, Resource);

	friend class AudioServer;

	struct Bus {
		StringName name;
		bool solo;
		bool mute;
		bool bypass;

		struct Effect {
			Ref<AudioEffect> effect;
			bool enabled;
		};

		Vector<Effect> effects;

		float volume_db;
		StringName send;

		Bus() {
			solo = false;
			mute = false;
			bypass = false;
			volume_db = 0;
		}
	};

	Vector<Bus> buses;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	AudioBusLayout();
};

typedef AudioServer AS;

#endif // AUDIO_SERVER_H
