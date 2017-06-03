/*************************************************************************/
/*  audio_server_javascript.h                                            */
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
#ifndef AUDIO_SERVER_JAVASCRIPT_H
#define AUDIO_SERVER_JAVASCRIPT_H
#if 0
#include "servers/audio_server.h"

class AudioServerJavascript  : public AudioServer  {

	GDCLASS(AudioServerJavascript,AudioServer);

	enum {
		INTERNAL_BUFFER_SIZE=4096,
		STREAM_SCALE_BITS=12

	};

	AudioMixer *get_mixer();
	void audio_mixer_chunk_callback(int p_frames);

	struct Sample {
		SampleFormat format;
		SampleLoopFormat loop_format;
		int loop_begin;
		int loop_end;
		int length;
		int index;
		int mix_rate;
		bool stereo;

		Vector<float> tmp_data;
	};

	mutable RID_Owner<Sample> sample_owner;
	int sample_base;

	struct Voice {
		int index;
		float volume;
		float pan;
		float pan_depth;
		float pan_height;

		float chorus;
		ReverbRoomType reverb_type;
		float reverb;

		int mix_rate;
		int sample_mix_rate;
		bool positional;

		bool active;

	};

	mutable RID_Owner<Voice> voice_owner;

	int voice_base;

	struct Stream {
		bool active;
		List<Stream*>::Element *E;
		AudioStream *audio_stream;
		EventStream *event_stream;
		float volume_scale;
	};

	List<Stream*> active_audio_streams;

	//List<Stream*> event_streams;

	float * internal_buffer;
	int internal_buffer_channels;
	int32_t * stream_buffer;

	mutable RID_Owner<Stream> stream_owner;

	float stream_volume;
	float stream_volume_scale;

	float event_voice_scale;
	float fx_volume_scale;


	void driver_process_chunk(int p_frames);

	int webaudio_mix_rate;


	static AudioServerJavascript *singleton;
public:

	void mix_to_js(int p_frames);
	/* SAMPLE API */

	virtual RID sample_create(SampleFormat p_format, bool p_stereo, int p_length);

	virtual void sample_set_description(RID p_sample, const String& p_description);
	virtual String sample_get_description(RID p_sample) const;

	virtual SampleFormat sample_get_format(RID p_sample) const;
	virtual bool sample_is_stereo(RID p_sample) const;
	virtual int sample_get_length(RID p_sample) const;
	virtual const void* sample_get_data_ptr(RID p_sample) const;


	virtual void sample_set_data(RID p_sample, const PoolVector<uint8_t>& p_buffer);
	virtual PoolVector<uint8_t> sample_get_data(RID p_sample) const;

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

	virtual void voice_set_volume(RID p_voice, float p_volume);
	virtual void voice_set_pan(RID p_voice, float p_pan, float p_depth=0,float height=0); //pan and depth go from -1 to 1
	virtual void voice_set_filter(RID p_voice, FilterType p_type, float p_cutoff, float p_resonance, float p_gain=0);
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

	/* Audio Physics API */

	virtual void free(RID p_id);

	virtual void init();
	virtual void finish();
	virtual void update();

	/* MISC config */

	virtual void lock();
	virtual void unlock();
	virtual int get_default_channel_count() const;
	virtual int get_default_mix_rate() const;

	virtual void set_stream_global_volume_scale(float p_volume);
	virtual void set_fx_global_volume_scale(float p_volume);
	virtual void set_event_voice_global_volume_scale(float p_volume);

	virtual float get_stream_global_volume_scale() const;
	virtual float get_fx_global_volume_scale() const;
	virtual float get_event_voice_global_volume_scale() const;

	virtual uint32_t read_output_peak() const;

	static AudioServer *get_singleton();

	virtual double get_mix_time() const; //useful for video -> audio sync
	virtual double get_output_delay() const;


	AudioServerJavascript();
};

#endif // AUDIO_SERVER_JAVASCRIPT_H
#endif
