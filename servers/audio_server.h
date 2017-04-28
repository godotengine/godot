/*************************************************************************/
/*  audio_server.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "audio_frame.h"
#include "object.h"
#include "servers/audio/audio_effect.h"
#include "variant.h"

class AudioDriver {

	static AudioDriver *singleton;
	uint64_t _last_mix_time;
	uint64_t _mix_amount;

protected:
	void audio_server_process(int p_frames, int32_t *p_buffer, bool p_update_mix_time = true);
	void update_mix_time(int p_frames);

public:
	double get_mix_time() const; //useful for video -> audio sync

	enum SpeakerMode {
		SPEAKER_MODE_STEREO,
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
	virtual void lock() = 0;
	virtual void unlock() = 0;
	virtual void finish() = 0;

	virtual float get_latency() { return 0; }

	AudioDriver();
	virtual ~AudioDriver() {}
};

class AudioDriverManager {

	enum {

		MAX_DRIVERS = 10
	};

	static AudioDriver *drivers[MAX_DRIVERS];
	static int driver_count;

public:
	static void add_driver(AudioDriver *p_driver);
	static int get_driver_count();
	static AudioDriver *get_driver(int p_driver);
};

class AudioBusLayout;

class AudioServer : public Object {

	GDCLASS(AudioServer, Object)
public:
	//re-expose this her, as AudioDriver is not exposed to script
	enum SpeakerMode {
		SPEAKER_MODE_STEREO,
		SPEAKER_SURROUND_51,
		SPEAKER_SURROUND_71,
	};

	enum {
		AUDIO_DATA_INVALID_ID = -1
	};

	typedef void (*AudioCallback)(void *p_userdata);

private:
	uint32_t buffer_size;
	uint64_t mix_count;
	uint64_t mix_frames;

	float channel_disable_treshold_db;
	uint32_t channel_disable_frames;

	int to_mix;

	struct Bus {

		StringName name;
		bool solo;
		bool mute;
		bool bypass;

		//Each channel is a stereo pair.
		struct Channel {
			bool used;
			bool active;
			AudioFrame peak_volume;
			Vector<AudioFrame> buffer;
			Vector<Ref<AudioEffectInstance> > effect_instances;
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
		};

		Vector<Effect> effects;
		float volume_db;
		StringName send;
		int index_cache;
	};

	Vector<Vector<AudioFrame> > temp_buffer; //temp_buffer for each level
	Vector<Bus *> buses;
	Map<StringName, Bus *> bus_map;

	_FORCE_INLINE_ int _get_channel_count() const {
		switch (AudioDriver::get_singleton()->get_speaker_mode()) {
			case AudioDriver::SPEAKER_MODE_STEREO: return 1;
			case AudioDriver::SPEAKER_SURROUND_51: return 3;
			case AudioDriver::SPEAKER_SURROUND_71: return 4;
		}
		ERR_FAIL_V(1);
	}

	void _update_bus_effects(int p_bus);

	static AudioServer *singleton;

	// TODO create an audiodata pool to optimize memory

	Map<void *, uint32_t> audio_data;
	size_t audio_data_total_mem;
	size_t audio_data_max_mem;

	Mutex *audio_data_lock;

	void _mix_step();

	struct CallbackItem {

		AudioCallback callback;
		void *userdata;

		bool operator<(const CallbackItem &p_item) const {
			return (callback == p_item.callback ? userdata < p_item.userdata : callback < p_item.callback);
		}
	};

	Set<CallbackItem> callbacks;

	friend class AudioDriver;
	void _driver_process(int p_frames, int32_t *p_buffer);

protected:
	static void _bind_methods();

public:
	//do not use from outside audio thread
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

	void swap_bus_effects(int p_bus, int p_effect, int p_by_effect);

	void set_bus_effect_enabled(int p_bus, int p_effect, bool p_enabled);
	bool is_bus_effect_enabled(int p_bus, int p_effect) const;

	float get_bus_peak_volume_left_db(int p_bus, int p_channel) const;
	float get_bus_peak_volume_right_db(int p_bus, int p_channel) const;

	bool is_bus_channel_active(int p_bus, int p_channel) const;

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

	virtual double get_mix_time() const; //useful for video -> audio sync
	virtual double get_output_delay() const;

	void *audio_data_alloc(uint32_t p_data_len, const uint8_t *p_from_data = NULL);
	void audio_data_free(void *p_data);

	size_t audio_data_get_total_memory_usage() const;
	size_t audio_data_get_max_memory_usage() const;

	void add_callback(AudioCallback p_callback, void *p_userdata);
	void remove_callback(AudioCallback p_callback, void *p_userdata);

	void set_bus_layout(const Ref<AudioBusLayout> &p_bus_layout);
	Ref<AudioBusLayout> generate_bus_layout() const;

	AudioServer();
	virtual ~AudioServer();
};

VARIANT_ENUM_CAST(AudioServer::SpeakerMode)

class AudioBusLayout : public Resource {

	GDCLASS(AudioBusLayout, Resource)

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
