/*************************************************************************/
/*  audio_server_sw.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef AUDIO_SERVER_SW_H
#define AUDIO_SERVER_SW_H

#include "servers/audio_server.h"
#include "servers/audio/audio_mixer_sw.h"
#include "servers/audio/voice_rb_sw.h"
#include "self_list.h"
#include "os/thread_safe.h"
#include "os/thread.h"
class AudioServerSW : public AudioServer {

	OBJ_TYPE( AudioServerSW, AudioServer );

	_THREAD_SAFE_CLASS_

	enum {
		INTERNAL_BUFFER_SIZE=4096,
		STREAM_SCALE_BITS=12

	};

	SampleManagerSW *sample_manager;
	AudioMixerSW *mixer;

	virtual AudioMixer *get_mixer();
	virtual void audio_mixer_chunk_callback(int p_frames);

	struct Voice {

		float volume;
		volatile bool active;
		SelfList<Voice> active_item;
		AudioMixer::ChannelID channel;


		Voice () : active_item(this) { channel=AudioMixer::INVALID_CHANNEL; active=false;}
	};

	mutable RID_Owner<Voice> voice_owner;
	SelfList<Voice>::List active_list;

	struct Stream {
		bool active;		
		List<Stream*>::Element *E;
		AudioStream *audio_stream;
		EventStream *event_stream;
		float volume_scale;
	};

	List<Stream*> active_audio_streams;

	//List<Stream*> event_streams;

	int32_t * internal_buffer;
	int internal_buffer_channels;
	int32_t * stream_buffer;

	mutable RID_Owner<Stream> stream_owner;

	float stream_volume;
	float stream_volume_scale;
	float fx_volume_scale;
	float event_voice_volume_scale;
	float peak_left,peak_right;
	uint32_t max_peak;

	double _output_delay;

	VoiceRBSW voice_rb;

	bool exit_update_thread;
	Thread *thread;
	static void _thread_func(void *self);

	void _update_streams(bool p_thread);
	void driver_process_chunk(int p_frames,int32_t *p_buffer);

	AudioMixerSW::InterpolationType mixer_interp;
	bool mixer_use_fx;
	uint64_t mixer_step_usecs;

	static void _mixer_callback(void *p_udata);
friend class AudioDriverSW;
	void driver_process(int p_frames,int32_t *p_buffer);
public:


	/* SAMPLE API */

	virtual RID sample_create(SampleFormat p_format, bool p_stereo, int p_length);

	virtual void sample_set_description(RID p_sample, const String& p_description);
	virtual String sample_get_description(RID p_sample, const String& p_description) const;

	virtual SampleFormat sample_get_format(RID p_sample) const;
	virtual bool sample_is_stereo(RID p_sample) const;
	virtual int sample_get_length(RID p_sample) const;
	const void* sample_get_data_ptr(RID p_sample) const;

	virtual void sample_set_data(RID p_sample, const DVector<uint8_t>& p_buffer);
	virtual const DVector<uint8_t> sample_get_data(RID p_sample) const;

	virtual void sample_set_mix_rate(RID p_sample,int p_rate);
	virtual int sample_get_mix_rate(RID p_sample) const;

	virtual void sample_set_loop_format(RID p_sample,SampleLoopFormat p_format);
	virtual SampleLoopFormat sample_get_loop_format(RID p_sample) const;

	virtual void sample_set_loop_begin(RID p_sample,int p_pos);
	virtual int sample_get_loop_begin(RID p_sample) const;

	virtual void sample_set_loop_end(RID p_sample,int p_pos);
	virtual int sample_get_loop_end(RID p_sample) const;

	/* VOICE API */

	virtual RID voice_create();

	virtual void voice_play(RID p_voice, RID p_sample);

	virtual void voice_set_volume(RID p_voice, float p_db);
	virtual void voice_set_pan(RID p_voice, float p_pan, float p_depth=0,float height=0); //pan and depth go from -1 to 1
	virtual void voice_set_filter(RID p_voice, FilterType p_type, float p_cutoff, float p_resonance,float p_gain=0);
	virtual void voice_set_chorus(RID p_voice, float p_chorus );
	virtual void voice_set_reverb(RID p_voice, ReverbRoomType p_room_type, float p_reverb);
	virtual void voice_set_mix_rate(RID p_voice, int p_mix_rate);
	virtual void voice_set_positional(RID p_voice, bool p_positional);

	virtual float voice_get_volume(RID p_voice) const;
	virtual float voice_get_pan(RID p_voice) const; //pan and depth go from -1 to 1
	virtual float voice_get_pan_depth(RID p_voice) const; //pan and depth go from -1 to 1
	virtual float voice_get_pan_height(RID p_voice) const; //pan and depth go from -1 to 1
	virtual FilterType voice_get_filter_type(RID p_voice) const;
	virtual float voice_get_filter_cutoff(RID p_voice) const;
	virtual float voice_get_filter_resonance(RID p_voice) const;
	virtual float voice_get_chorus(RID p_voice) const;
	virtual ReverbRoomType voice_get_reverb_type(RID p_voice) const;
	virtual float voice_get_reverb(RID p_voice) const;

	virtual int voice_get_mix_rate(RID p_voice) const;
	virtual bool voice_is_positional(RID p_voice) const;

	virtual void voice_stop(RID p_voice);
	virtual bool voice_is_active(RID p_voice) const;

	/* STREAM API */

	virtual RID audio_stream_create(AudioStream *p_stream);
	virtual RID event_stream_create(EventStream *p_stream);

	virtual void stream_set_active(RID p_stream, bool p_active);
	virtual bool stream_is_active(RID p_stream) const;

	virtual void stream_set_volume_scale(RID p_stream, float p_scale);
	virtual float stream_set_volume_scale(RID p_stream) const;

	virtual void free(RID p_id);

	virtual void init();
	virtual void finish();
	virtual void update();

	virtual void lock();
	virtual void unlock();
	virtual int get_default_channel_count() const;
	virtual int get_default_mix_rate() const;

	void set_mixer_params(AudioMixerSW::InterpolationType p_interp, bool p_use_fx);

	virtual void set_stream_global_volume_scale(float p_volume);
	virtual void set_fx_global_volume_scale(float p_volume);
	virtual void set_event_voice_global_volume_scale(float p_volume);


	virtual float get_stream_global_volume_scale() const;
	virtual float get_fx_global_volume_scale() const;
	virtual float get_event_voice_global_volume_scale() const;

	virtual uint32_t read_output_peak() const;

	virtual double get_mix_time() const; //useful for video -> audio sync

	virtual double get_output_delay() const;


	AudioServerSW(SampleManagerSW *p_sample_manager);
	~AudioServerSW();

};


class AudioDriverSW {


	static AudioDriverSW *singleton;
	uint64_t _last_mix_time;
	uint64_t _mix_amount;


protected:

	void audio_server_process(int p_frames,int32_t *p_buffer,bool p_update_mix_time=true);
	void update_mix_time(int p_frames);

public:


	double get_mix_time() const; //useful for video -> audio sync

	enum OutputFormat {

		OUTPUT_MONO,
		OUTPUT_STEREO,
		OUTPUT_QUAD,
		OUTPUT_5_1
	};

	static AudioDriverSW *get_singleton();
	void set_singleton();

	virtual const char* get_name() const=0;

	virtual Error init()=0;
	virtual void start()=0;
	virtual int get_mix_rate() const =0;
	virtual OutputFormat get_output_format() const=0;
	virtual void lock()=0;
	virtual void unlock()=0;
	virtual void finish()=0;




	AudioDriverSW();
	virtual ~AudioDriverSW() {};
};



class AudioDriverManagerSW {

	enum {

		MAX_DRIVERS=10
	};

	static AudioDriverSW *drivers[MAX_DRIVERS];
	static int driver_count;
public:

	static void add_driver(AudioDriverSW *p_driver);
	static int get_driver_count();
	static AudioDriverSW *get_driver(int p_driver);
};

#endif // AUDIO_SERVER_SW_H
