/*************************************************************************/
/*  audio_server.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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

#include "variant.h"
#include "object.h"
#include "audio_frame.h"
#include "servers/audio/audio_effect.h"


class AudioDriver {


	static AudioDriver *singleton;
	uint64_t _last_mix_time;
	uint64_t _mix_amount;


protected:

	void audio_server_process(int p_frames,int32_t *p_buffer,bool p_update_mix_time=true);
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

	virtual const char* get_name() const=0;

	virtual Error init()=0;
	virtual void start()=0;
	virtual int get_mix_rate() const =0;
	virtual SpeakerMode get_speaker_mode() const=0;
	virtual void lock()=0;
	virtual void unlock()=0;
	virtual void finish()=0;

	virtual float get_latency() { return 0; }




	AudioDriver();
	virtual ~AudioDriver() {}
};



class AudioDriverManager {

	enum {

		MAX_DRIVERS=10
	};

	static AudioDriver *drivers[MAX_DRIVERS];
	static int driver_count;
public:

	static void add_driver(AudioDriver *p_driver);
	static int get_driver_count();
	static AudioDriver *get_driver(int p_driver);
};


class AudioServer : public Object {

	GDCLASS( AudioServer, Object )
public:
	enum BusMode {
		BUS_MODE_STEREO,
		BUS_MODE_SURROUND
	};

	//re-expose this her, as AudioDriver is not exposed to script
	enum SpeakerMode {
		SPEAKER_MODE_STEREO,
		SPEAKER_SURROUND_51,
		SPEAKER_SURROUND_71,
	};
private:
	uint32_t buffer_size;

	struct Bus {

		String name;
		BusMode mode;
		Vector<AudioFrame> buffer;

		struct Effect {
			Ref<AudioEffect> effect;
			Ref<AudioEffectInstance> instance;
			bool enabled;
		};

		Vector<Effect> effects;

		float volume_db;
	};


	Vector<Bus> buses;


	static void _bind_methods();

	static AudioServer* singleton;
public:


	void set_bus_count(int p_count);
	int get_bus_count() const;

	void set_bus_mode(int p_bus,BusMode p_mode);
	BusMode get_bus_mode(int p_bus) const;

	void set_bus_name(int p_bus,const String& p_name);
	String get_bus_name(int p_bus) const;

	void set_bus_volume_db(int p_bus,float p_volume_db);
	float get_bus_volume_db(int p_bus) const;

	void add_bus_effect(int p_bus,const Ref<AudioEffect>& p_effect,int p_at_pos=-1);
	void remove_bus_effect(int p_bus,int p_effect);

	int get_bus_effect_count(int p_bus);
	Ref<AudioEffect> get_bus_effect(int p_bus,int p_effect);

	void swap_bus_effects(int p_bus,int p_effect,int p_by_effect);

	void set_bus_effect_enabled(int p_bus,int p_effect,bool p_enabled);
	bool is_bus_effect_enabled(int p_bus,int p_effect) const;

	virtual void init();
	virtual void finish();
	virtual void update();

	/* MISC config */

	virtual void lock();
	virtual void unlock();


	virtual SpeakerMode get_speaker_mode() const;
	virtual float get_mix_rate() const;

	virtual float read_output_peak_db() const;

	static AudioServer *get_singleton();

	virtual double get_mix_time() const; //useful for video -> audio sync
	virtual double get_output_delay() const;

	AudioServer();
	virtual ~AudioServer();
};

VARIANT_ENUM_CAST( AudioServer::BusMode )
VARIANT_ENUM_CAST( AudioServer::SpeakerMode )

typedef AudioServer AS;


#endif // AUDIO_SERVER_H
