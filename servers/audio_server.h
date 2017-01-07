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

class AudioMixer {
protected:

	void audio_mixer_chunk_call(int p_frames);
public:

	enum {

		INVALID_CHANNEL=0xFFFFFFFF
	};

	typedef uint32_t ChannelID;

	/* CHANNEL API */

	enum FilterType {
		FILTER_NONE,
		FILTER_LOWPASS,
		FILTER_BANDPASS,
		FILTER_HIPASS,
		FILTER_NOTCH,
		FILTER_PEAK,
		FILTER_BANDLIMIT, ///< cutoff is LP resonace is HP
		FILTER_LOW_SHELF,
		FILTER_HIGH_SHELF

	};

	enum ReverbRoomType {

		REVERB_SMALL,
		REVERB_MEDIUM,
		REVERB_LARGE,
		REVERB_HALL,
		MAX_REVERBS
	};

	virtual ChannelID channel_alloc(RID p_sample)=0;

	virtual void channel_set_volume(ChannelID p_channel, float p_gain)=0;
	virtual void channel_set_pan(ChannelID p_channel, float p_pan, float p_depth=0,float height=0)=0; //pan and depth go from -1 to 1
	virtual void channel_set_filter(ChannelID p_channel, FilterType p_type, float p_cutoff, float p_resonance, float p_gain=1.0)=0;
	virtual void channel_set_chorus(ChannelID p_channel, float p_chorus )=0;
	virtual void channel_set_reverb(ChannelID p_channel, ReverbRoomType p_room_type, float p_reverb)=0;
	virtual void channel_set_mix_rate(ChannelID p_channel, int p_mix_rate)=0;
	virtual void channel_set_positional(ChannelID p_channel, bool p_positional)=0;

	virtual float channel_get_volume(ChannelID p_channel) const=0;
	virtual float channel_get_pan(ChannelID p_channel) const=0; //pan and depth go from -1 to 1
	virtual float channel_get_pan_depth(ChannelID p_channel) const=0; //pan and depth go from -1 to 1
	virtual float channel_get_pan_height(ChannelID p_channel) const=0; //pan and depth go from -1 to 1
	virtual FilterType channel_get_filter_type(ChannelID p_channel) const=0;
	virtual float channel_get_filter_cutoff(ChannelID p_channel) const=0;
	virtual float channel_get_filter_resonance(ChannelID p_channel) const=0;
	virtual float channel_get_filter_gain(ChannelID p_channel) const=0;
	virtual float channel_get_chorus(ChannelID p_channel) const=0;
	virtual ReverbRoomType channel_get_reverb_type(ChannelID p_channel) const=0;
	virtual float channel_get_reverb(ChannelID p_channel) const=0;

	virtual int channel_get_mix_rate(ChannelID p_channel) const=0;
	virtual bool channel_is_positional(ChannelID p_channel) const=0;
	virtual bool channel_is_valid(ChannelID p_channel) const=0;


	virtual void channel_free(ChannelID p_channel)=0;

	virtual void set_mixer_volume(float p_volume)=0;


	virtual ~AudioMixer() {}
};


class AudioServer : public Object {

	GDCLASS( AudioServer, Object );

	static AudioServer *singleton;
protected:
friend class AudioStream;
friend class EventStream;
friend class AudioMixer;

	virtual AudioMixer *get_mixer()=0;
	virtual void audio_mixer_chunk_callback(int p_frames)=0;

	static void _bind_methods();
public:


	class EventStream {
	protected:
		AudioMixer *get_mixer() const;
	public:
		virtual void update(uint64_t p_usec)=0;

		virtual ~EventStream() {}
	};

	class AudioStream {
	public:
		virtual int get_channel_count() const=0;
		virtual void set_mix_rate(int p_rate)=0; //notify the stream of the mix rate
		virtual bool mix(int32_t *p_buffer,int p_frames)=0;
		virtual void update()=0;
		virtual bool can_update_mt() const { return true; }
		virtual ~AudioStream() {}
	};


	enum SampleFormat {

		SAMPLE_FORMAT_PCM8,
		SAMPLE_FORMAT_PCM16,
		SAMPLE_FORMAT_IMA_ADPCM
	};

	enum SampleLoopFormat {
		SAMPLE_LOOP_NONE,
		SAMPLE_LOOP_FORWARD,
		SAMPLE_LOOP_PING_PONG // not supported in every platform

	};

	/* SAMPLE API */

	virtual RID sample_create(SampleFormat p_format, bool p_stereo, int p_length)=0;

	virtual void sample_set_description(RID p_sample, const String& p_description)=0;
	virtual String sample_get_description(RID p_sample) const=0;

	virtual SampleFormat sample_get_format(RID p_sample) const=0;
	virtual bool sample_is_stereo(RID p_sample) const=0;
	virtual int sample_get_length(RID p_sample) const=0;
	virtual const void* sample_get_data_ptr(RID p_sample) const=0;

	virtual void sample_set_signed_data(RID p_sample, const PoolVector<float>& p_buffer);
	virtual void sample_set_data(RID p_sample, const PoolVector<uint8_t>& p_buffer)=0;
	virtual PoolVector<uint8_t> sample_get_data(RID p_sample) const=0;

	virtual void sample_set_mix_rate(RID p_sample,int p_rate)=0;
	virtual int sample_get_mix_rate(RID p_sample) const=0;

	virtual void sample_set_loop_format(RID p_sample,SampleLoopFormat p_format)=0;
	virtual SampleLoopFormat sample_get_loop_format(RID p_sample) const=0;

	virtual void sample_set_loop_begin(RID p_sample,int p_pos)=0;
	virtual int sample_get_loop_begin(RID p_sample) const=0;

	virtual void sample_set_loop_end(RID p_sample,int p_pos)=0;
	virtual int sample_get_loop_end(RID p_sample) const=0;


	/* VOICE API */

	enum FilterType {
		FILTER_NONE,
		FILTER_LOWPASS,
		FILTER_BANDPASS,
		FILTER_HIPASS,
		FILTER_NOTCH,
		FILTER_PEAK,
		FILTER_BANDLIMIT, ///< cutoff is LP resonace is HP
		FILTER_LOW_SHELF,
		FILTER_HIGH_SHELF
	};

	enum ReverbRoomType {

		REVERB_SMALL,
		REVERB_MEDIUM,
		REVERB_LARGE,
		REVERB_HALL
	};

	virtual RID voice_create()=0;

	virtual void voice_play(RID p_voice, RID p_sample)=0;

	virtual void voice_set_volume(RID p_voice, float p_volume)=0;
	virtual void voice_set_pan(RID p_voice, float p_pan, float p_depth=0,float height=0)=0; //pan and depth go from -1 to 1
	virtual void voice_set_filter(RID p_voice, FilterType p_type, float p_cutoff, float p_resonance, float p_gain=0)=0;
	virtual void voice_set_chorus(RID p_voice, float p_chorus )=0;
	virtual void voice_set_reverb(RID p_voice, ReverbRoomType p_room_type, float p_reverb)=0;
	virtual void voice_set_mix_rate(RID p_voice, int p_mix_rate)=0;
	virtual void voice_set_positional(RID p_voice, bool p_positional)=0;

	virtual float voice_get_volume(RID p_voice) const=0;
	virtual float voice_get_pan(RID p_voice) const=0; //pan and depth go from -1 to 1
	virtual float voice_get_pan_depth(RID p_voice) const=0; //pan and depth go from -1 to 1
	virtual float voice_get_pan_height(RID p_voice) const=0; //pan and depth go from -1 to 1
	virtual FilterType voice_get_filter_type(RID p_voice) const=0;
	virtual float voice_get_filter_cutoff(RID p_voice) const=0;
	virtual float voice_get_filter_resonance(RID p_voice) const=0;
	virtual float voice_get_chorus(RID p_voice) const=0;
	virtual ReverbRoomType voice_get_reverb_type(RID p_voice) const=0;
	virtual float voice_get_reverb(RID p_voice) const=0;

	virtual int voice_get_mix_rate(RID p_voice) const=0;
	virtual bool voice_is_positional(RID p_voice) const=0;

	virtual void voice_stop(RID p_voice)=0;
	virtual bool voice_is_active(RID p_voice) const=0;

	/* STREAM API */

	virtual RID audio_stream_create(AudioStream *p_stream)=0;
	virtual RID event_stream_create(EventStream *p_stream)=0;

	virtual void stream_set_active(RID p_stream, bool p_active)=0;
	virtual bool stream_is_active(RID p_stream) const=0;

	virtual void stream_set_volume_scale(RID p_stream, float p_scale)=0;
	virtual float stream_set_volume_scale(RID p_stream) const=0;

	/* Audio Physics API */

	virtual void free(RID p_id)=0;

	virtual void init()=0;
	virtual void finish()=0;
	virtual void update()=0;

	/* MISC config */

	virtual void lock()=0;
	virtual void unlock()=0;
	virtual int get_default_channel_count() const=0;
	virtual int get_default_mix_rate() const=0;

	virtual void set_stream_global_volume_scale(float p_volume)=0;
	virtual void set_fx_global_volume_scale(float p_volume)=0;
	virtual void set_event_voice_global_volume_scale(float p_volume)=0;

	virtual float get_stream_global_volume_scale() const=0;
	virtual float get_fx_global_volume_scale() const=0;
	virtual float get_event_voice_global_volume_scale() const=0;

	virtual uint32_t read_output_peak() const=0;

	static AudioServer *get_singleton();

	virtual double get_mix_time() const=0; //useful for video -> audio sync
	virtual double get_output_delay() const=0;

	AudioServer();
	virtual ~AudioServer();
};

VARIANT_ENUM_CAST( AudioServer::SampleFormat );
VARIANT_ENUM_CAST( AudioServer::SampleLoopFormat );
VARIANT_ENUM_CAST( AudioServer::FilterType );
VARIANT_ENUM_CAST( AudioServer::ReverbRoomType );

typedef AudioServer AS;

#endif // AUDIO_SERVER_H
