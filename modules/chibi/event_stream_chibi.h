/*************************************************************************/
/*  event_stream_chibi.h                                                 */
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
#ifndef EVENT_STREAM_CHIBI_H
#define EVENT_STREAM_CHIBI_H

#include "scene/resources/event_stream.h"
#include "cp_sample_manager.h"
#include "cp_mixer.h"
#include "cp_song.h"
#include "cp_file_access_wrapper.h"
#include "cp_player_data.h"
#include "resource.h"
#include "servers/audio_server.h"
#include "os/file_access.h"
#include "io/resource_loader.h"

/** SAMPLE MANAGER **/

class CPSampleManagerImpl : public CPSampleManager {

	struct SampleData {

		RID rid;
		bool stereo;
		bool is16;
		int len;
		int mixfreq;
		int loop_begin;
		int loop_end;
		int locks;
		DVector<uint8_t> lock;
		DVector<uint8_t>::Write w;
		CPSample_Loop_Type loop_type;
	};


	_FORCE_INLINE_ SampleData* _getsd(CPSample_ID p_id) {

		return ((SampleData*)p_id._private);
	}
	Set<SampleData*> valid;

public:

	_FORCE_INLINE_ RID get_rid(CPSample_ID p_id) { return _getsd(p_id)->rid; }
	virtual CPSample_ID create(bool p_16bits,bool p_stereo,int32_t p_len);
	virtual void recreate(CPSample_ID p_id,bool p_16bits,bool p_stereo,int32_t p_len);
	virtual void destroy(CPSample_ID p_id);
	virtual bool check(CPSample_ID p_id); // return false if invalid

	virtual void set_c5_freq(CPSample_ID p_id,int32_t p_freq);
	virtual void set_loop_begin(CPSample_ID p_id,int32_t p_begin);
	virtual void set_loop_end(CPSample_ID p_id,int32_t p_end);
	virtual void set_loop_type(CPSample_ID p_id,CPSample_Loop_Type p_type);
	virtual void set_chunk(CPSample_ID p_id,int32_t p_index,void *p_data,int p_data_len);


	virtual int32_t get_loop_begin(CPSample_ID p_id);
	virtual int32_t get_loop_end(CPSample_ID p_id);
	virtual CPSample_Loop_Type get_loop_type(CPSample_ID p_id);
	virtual int32_t get_c5_freq(CPSample_ID p_id);
	virtual int32_t get_size(CPSample_ID p_id);
	virtual bool is_16bits(CPSample_ID p_id);
	virtual bool is_stereo(CPSample_ID p_id);
	virtual bool lock_data(CPSample_ID p_id);
	virtual void *get_data(CPSample_ID p_id); /* WARNING: Not all sample managers
may be able to implement this, it depends on the mixer in use! */
	virtual int16_t get_data(CPSample_ID p_id, int p_sample, int p_channel=0); /// Does not need locking
	virtual void set_data(CPSample_ID p_id, int p_sample, int16_t p_data,int p_channel=0); /// Does not need locking
	virtual void unlock_data(CPSample_ID p_id);

	virtual void get_chunk(CPSample_ID p_id,int32_t p_index,void *p_data,int p_data_len);

};


/** MIXER **/

class CPMixerImpl : public CPMixer {

	enum {
		MAX_VOICES=64
	};

	struct Voice {

		AudioMixer::ChannelID channel;
		CPSample_ID sample;
		float freq_mult;
		float reverb;
		Voice() { reverb=0.0; }
	};

	Voice voices[MAX_VOICES];


	int callback_interval;
	int callback_timeout;
	void (*callback)(void*);
	void *userdata;
	float voice_scale;
	float tempo_scale;
	float pitch_scale;
	AudioMixer::ReverbRoomType reverb_type;
	AudioMixer *mixer;
public:

	void process_usecs(int p_usec,float p_volume,float p_pitch_scale,float p_tempo_scale);

	/* Callback */

	virtual void set_callback_interval(int p_interval_us); //in usecs, for tracker it's 2500000/tempo
	virtual void set_callback(void (*p_callback)(void*),void *p_userdata);

	/* Voice Control */

	virtual void setup_voice(int p_voice_index,CPSample_ID p_sample_id,int32_t p_start_index) ;
	virtual void stop_voice(int p_voice_index) ;
	virtual void set_voice_frequency(int p_voice_index,int32_t p_freq) ; //in freq*FREQUENCY_BITS
	virtual void set_voice_panning(int p_voice_index,int p_pan) ;
	virtual void set_voice_volume(int p_voice_index,int p_vol) ;
	virtual void set_voice_filter(int p_filter,bool p_enabled,uint8_t p_cutoff, uint8_t p_resonance );
	virtual void set_voice_reverb_send(int p_voice_index,int p_reverb);
	virtual void set_voice_chorus_send(int p_voice_index,int p_chorus); /* 0 - 255 */

	virtual void set_reverb_mode(ReverbMode p_mode);
	virtual void set_chorus_params(unsigned int p_delay_ms,unsigned int p_separation_ms,unsigned int p_depth_ms10,unsigned int p_speed_hz10);


	/* Info retrieving */

	virtual int32_t get_voice_sample_pos_index(int p_voice_index) ;
	virtual int get_voice_panning(int p_voice_index) ;
	virtual int get_voice_volume(int p_voice_index) ;
	virtual CPSample_ID get_voice_sample_id(int p_voice_index) ;
	virtual bool is_voice_active(int p_voice_index);
	virtual int get_active_voice_count() { return 0; }
	virtual int get_total_voice_count() { return MAX_VOICES; }


	virtual uint32_t get_mix_frequency() { return 0; }

	/* Methods below only work with software mixers, meant for software-based sound drivers, hardware mixers ignore them */
	virtual int32_t process(int32_t p_frames) { return 0; }
	virtual int32_t *get_mixdown_buffer_ptr() { return NULL; }
	virtual void set_mix_frequency(int32_t p_mix_frequency) {};

	CPMixerImpl(AudioMixer *p_mixer=NULL);
	virtual ~CPMixerImpl() {}
};

/** FILE ACCESS **/

class CPFileAccessWrapperImpl : public CPFileAccessWrapper {

	FileAccess *f;
public:


	virtual Error open(const char *p_filename, int p_mode_flags);
	virtual void close();

	virtual void seek(uint32_t p_position);
	virtual void seek_end();
	virtual uint32_t get_pos();

	virtual bool eof_reached();

	virtual uint8_t get_byte();
	virtual void get_byte_array(uint8_t *p_dest,int p_elements);
	virtual void get_word_array(uint16_t *p_dest,int p_elements);

	virtual uint16_t get_word();
	virtual uint32_t get_dword();

	virtual void set_endian_conversion(bool p_swap);
	virtual bool is_open();

	virtual Error get_error();

	virtual void store_byte(uint8_t p_dest);
	virtual void store_byte_array(const uint8_t *p_dest,int p_elements);

	virtual void store_word(uint16_t p_dest);
	virtual void store_dword(uint32_t p_dest);

	CPFileAccessWrapperImpl() { f=NULL; }
	virtual ~CPFileAccessWrapperImpl(){ if (f) memdelete(f); }

};



/////////////////////

class EventStreamChibi;

class EventStreamPlaybackChibi : public EventStreamPlayback {

	GDCLASS(EventStreamPlaybackChibi,EventStreamPlayback);

	CPMixerImpl mixer;
	uint64_t total_usec;
	Ref<EventStreamChibi> stream;
	mutable CPPlayer *player;
	bool loop;
	int last_order;
	int loops;
	virtual Error _play();
	virtual bool _update(AudioMixer* p_mixer, uint64_t p_usec);
	virtual void _stop();
	float volume;
	float tempo_scale;
	float pitch_scale;


public:


	virtual void set_paused(bool p_paused);
	virtual bool is_paused() const;

	virtual void set_loop(bool p_loop);
	virtual bool is_loop_enabled() const;

	virtual int get_loop_count() const;

	virtual float get_pos() const;
	virtual void seek_pos(float p_time);

	virtual void set_volume(float p_vol);
	virtual float get_volume() const;

	virtual void set_pitch_scale(float p_pitch_scale);
	virtual float get_pitch_scale() const;

	virtual void set_tempo_scale(float p_tempo_scale);
	virtual float get_tempo_scale() const;

	virtual void set_channel_volume(int p_channel,float p_volume);
	virtual float get_channel_volume(int p_channel) const;

	virtual float get_last_note_time(int p_channel) const;

	EventStreamPlaybackChibi(Ref<EventStreamChibi> p_stream=Ref<EventStreamChibi>());
	~EventStreamPlaybackChibi();
};


class EventStreamChibi : public EventStream {

	GDCLASS(EventStreamChibi,EventStream);

friend class ResourceFormatLoaderChibi;
friend class EventStreamPlaybackChibi;
	//I think i didn't know what const was when i wrote this more than a decade ago
	//so it goes mutable :(
	mutable CPSong song;


public:

	virtual Ref<EventStreamPlayback> instance_playback();

	virtual String get_stream_name() const;

	virtual float get_length() const;

	virtual int get_channel_count() const { return 64; } //tracker limit

	EventStreamChibi();
};


class ResourceFormatLoaderChibi : public ResourceFormatLoader {

public:
	virtual RES load(const String &p_path,const String& p_original_path="",Error *r_error=NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};

void initialize_chibi();
void finalize_chibi();

#endif // EVENT_STREAM_CHIBI_H
