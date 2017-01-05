/*************************************************************************/
/*  audio_server_sw.cpp                                                  */
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
#include "audio_server_sw.h"
#include "globals.h"
#include "os/os.h"

struct _AudioDriverLock {

	_AudioDriverLock() { if (AudioDriverSW::get_singleton()) AudioDriverSW::get_singleton()->lock(); }
	~_AudioDriverLock() { if (AudioDriverSW::get_singleton()) AudioDriverSW::get_singleton()->unlock(); }

};

#define AUDIO_LOCK _AudioDriverLock _adlock;

AudioMixer *AudioServerSW::get_mixer() {

	return mixer;
}

/* CALLBACKS */

void AudioServerSW::audio_mixer_chunk_callback(int p_frames) {
/*
	for(List<Stream*>::Element *E=event_streams.front();E;E=E->next()) {

		if (E->get()->active)
			E->get()->audio_stream->mix(NULL,p_frames);
	}
*/
}

void AudioServerSW::_mixer_callback(void *p_udata) {

	AudioServerSW *self = (AudioServerSW*)p_udata;
	for(List<Stream*>::Element *E=self->active_audio_streams.front();E;E=E->next()) {

		if (!E->get()->active)
			continue;

		EventStream *es=E->get()->event_stream;
		if (!es)
			continue;

		es->update(self->mixer_step_usecs);
	}

}

void AudioServerSW::driver_process_chunk(int p_frames,int32_t *p_buffer) {



	int samples=p_frames*internal_buffer_channels;

	for(int i=0;i<samples;i++) {
		internal_buffer[i]=0;
	}

	while(voice_rb.commands_left()) {

		VoiceRBSW::Command cmd = voice_rb.pop_command();

		if (cmd.type==VoiceRBSW::Command::CMD_CHANGE_ALL_FX_VOLUMES) {

			SelfList<Voice>*al =  active_list.first();
			while(al) {

				Voice *v=al->self();
				if (v->channel!=AudioMixer::INVALID_CHANNEL) {
					mixer->channel_set_volume(v->channel,v->volume*fx_volume_scale);
				}
				al=al->next();
			}

			continue;
		}
		if (!voice_owner.owns(cmd.voice))
			continue;


		Voice *v = voice_owner.get(cmd.voice);

		switch(cmd.type) {
			case VoiceRBSW::Command::CMD_NONE: {


			} break;
			case VoiceRBSW::Command::CMD_PLAY: {

				if (v->channel!=AudioMixer::INVALID_CHANNEL)
					mixer->channel_free(v->channel);

				RID sample = cmd.play.sample;
				if (!sample_manager->is_sample(sample))
					continue;

				v->channel=mixer->channel_alloc(sample);
				v->volume=1.0;
				mixer->channel_set_volume(v->channel,fx_volume_scale);
				if (v->channel==AudioMixer::INVALID_CHANNEL) {
#ifdef AUDIO_DEBUG
					WARN_PRINT("AUDIO: all channels used, failed to allocate voice");
#endif
					v->active=false;
					break; // no voices left?
				}

				v->active=true; // this kind of ensures it works
				if (!v->active_item.in_list())
					active_list.add(&v->active_item);

			} break;
			case VoiceRBSW::Command::CMD_STOP: {

				if (v->channel!=AudioMixer::INVALID_CHANNEL) {
					mixer->channel_free(v->channel);
					if (v->active_item.in_list()) {
						active_list.remove(&v->active_item);
					}
				}
				v->active=false;
			} break;
			case VoiceRBSW::Command::CMD_SET_VOLUME: {


				if (v->channel!=AudioMixer::INVALID_CHANNEL) {
					v->volume=cmd.volume.volume;
					mixer->channel_set_volume(v->channel,cmd.volume.volume*fx_volume_scale);
				}

			} break;
			case VoiceRBSW::Command::CMD_SET_PAN: {

				if (v->channel!=AudioMixer::INVALID_CHANNEL)
					mixer->channel_set_pan(v->channel,cmd.pan.pan,cmd.pan.depth,cmd.pan.height);

			} break;
			case VoiceRBSW::Command::CMD_SET_FILTER: {


				if (v->channel!=AudioMixer::INVALID_CHANNEL)
					mixer->channel_set_filter(v->channel,(AudioMixer::FilterType)cmd.filter.type,cmd.filter.cutoff,cmd.filter.resonance,cmd.filter.gain);
			} break;
			case VoiceRBSW::Command::CMD_SET_CHORUS: {

				if (v->channel!=AudioMixer::INVALID_CHANNEL)
					mixer->channel_set_chorus(v->channel,cmd.chorus.send);

			} break;
			case VoiceRBSW::Command::CMD_SET_REVERB: {

				if (v->channel!=AudioMixer::INVALID_CHANNEL)
					mixer->channel_set_reverb(v->channel,(AudioMixer::ReverbRoomType)cmd.reverb.room,cmd.reverb.send);

			} break;
			case VoiceRBSW::Command::CMD_SET_MIX_RATE: {

				if (v->channel!=AudioMixer::INVALID_CHANNEL)
					mixer->channel_set_mix_rate(v->channel,cmd.mix_rate.mix_rate);

			} break;
			case VoiceRBSW::Command::CMD_SET_POSITIONAL: {

				if (v->channel!=AudioMixer::INVALID_CHANNEL)
					mixer->channel_set_positional(v->channel,cmd.positional.positional);

			} break;
			default: {}

		}
	}

	mixer->mix(internal_buffer,p_frames);
	//uint64_t stepsize=mixer->get_step_usecs();


	for(List<Stream*>::Element *E=active_audio_streams.front();E;E=E->next()) {

		ERR_CONTINUE(!E->get()->active); // bug?


		AudioStream *as=E->get()->audio_stream;
		if (!as)
			continue;

		int channels=as->get_channel_count();
		if (channels==0)
			continue; // does not want mix
		if (!as->mix(stream_buffer,p_frames))
			continue; //nothing was mixed!!

		int32_t stream_vol_scale=(stream_volume*stream_volume_scale*E->get()->volume_scale)*(1<<STREAM_SCALE_BITS);

#define STRSCALE(m_val)	(((m_val>>STREAM_SCALE_BITS)*stream_vol_scale)>>8)
		switch(internal_buffer_channels) {

			case 2: {

				switch(channels) {
					case 1: {

						for(int i=0;i<p_frames;i++) {

							internal_buffer[(i<<1)+0]+=STRSCALE(stream_buffer[i]);
							internal_buffer[(i<<1)+1]+=STRSCALE(stream_buffer[i]);
						}
					} break;
					case 2: {

						for(int i=0;i<p_frames*2;i++) {

							internal_buffer[i]+=STRSCALE(stream_buffer[i]);
						}
					} break;
					case 4: {

						for(int i=0;i<p_frames;i++) {

							internal_buffer[(i<<2)+0]+=STRSCALE((stream_buffer[(i<<2)+0]+stream_buffer[(i<<2)+2])>>1);
							internal_buffer[(i<<2)+1]+=STRSCALE((stream_buffer[(i<<2)+1]+stream_buffer[(i<<2)+3])>>1);
						}
					} break;

				} break;

			} break;
			case 4: {

				switch(channels) {
					case 1: {

						for(int i=0;i<p_frames;i++) {

							internal_buffer[(i<<2)+0]+=STRSCALE(stream_buffer[i]);
							internal_buffer[(i<<2)+1]+=STRSCALE(stream_buffer[i]);
							internal_buffer[(i<<2)+2]+=STRSCALE(stream_buffer[i]);
							internal_buffer[(i<<2)+3]+=STRSCALE(stream_buffer[i]);
						}
					} break;
					case 2: {

						for(int i=0;i<p_frames*2;i++) {

							internal_buffer[(i<<2)+0]+=STRSCALE(stream_buffer[(i<<1)+0]);
							internal_buffer[(i<<2)+1]+=STRSCALE(stream_buffer[(i<<1)+1]);
							internal_buffer[(i<<2)+2]+=STRSCALE(stream_buffer[(i<<1)+0]);
							internal_buffer[(i<<2)+3]+=STRSCALE(stream_buffer[(i<<1)+1]);
						}
					} break;
					case 4: {

						for(int i=0;i<p_frames*4;i++) {
							internal_buffer[i]+=STRSCALE(stream_buffer[i]);
						}
					} break;

				} break;

			} break;
			case 6: {


			} break;
		}

#undef STRSCALE
	}

	SelfList<Voice> *activeE=active_list.first();
	while(activeE) {

		SelfList<Voice> *activeN=activeE->next();
		if (activeE->self()->channel==AudioMixer::INVALID_CHANNEL || !mixer->channel_is_valid(activeE->self()->channel)) {

			active_list.remove(activeE);
			activeE->self()->active=false;

		}
		activeE=activeN;
	}

	uint32_t peak=0;
	for(int i=0;i<samples;i++) {
		//clamp to (1<<24) using branchless code
		int32_t in = internal_buffer[i];
#ifdef DEBUG_ENABLED
		{
		  int mask = (in >> (32 - 1));
		  uint32_t p = (in + mask) ^ mask;
		  if (p>peak)
			peak=p;
		}
#endif
		int32_t lo = -0x800000, hi=0x7FFFFF;
		lo-=in;
		hi-=in;
		in += (lo & ((lo < 0) - 1)) + (hi & ((hi > 0) - 1));
		p_buffer[i]=in<<8;
	}

	if (peak>max_peak)
		max_peak=peak;
}

void AudioServerSW::driver_process(int p_frames,int32_t *p_buffer) {


	_output_delay=p_frames/double(AudioDriverSW::get_singleton()->get_mix_rate());
	//process in chunks to make sure to never process more than INTERNAL_BUFFER_SIZE
	int todo=p_frames;
	while(todo) {

		int tomix=MIN(todo,INTERNAL_BUFFER_SIZE);
		driver_process_chunk(tomix,p_buffer);
		p_buffer+=tomix;
		todo-=tomix;
	}


}

/* SAMPLE API */

RID AudioServerSW::sample_create(SampleFormat p_format, bool p_stereo, int p_length) {

	AUDIO_LOCK

	return sample_manager->sample_create(p_format,p_stereo,p_length);
}

void AudioServerSW::sample_set_description(RID p_sample, const String& p_description) {

	AUDIO_LOCK
	sample_manager->sample_set_description(p_sample,p_description);
}
String AudioServerSW::sample_get_description(RID p_sample) const {

	AUDIO_LOCK
	return sample_manager->sample_get_description(p_sample);
}

AS::SampleFormat AudioServerSW::sample_get_format(RID p_sample) const {
	//AUDIO_LOCK
	return sample_manager->sample_get_format(p_sample);
}
bool AudioServerSW::sample_is_stereo(RID p_sample) const {
	//AUDIO_LOCK
	return sample_manager->sample_is_stereo(p_sample);
}
int AudioServerSW::sample_get_length(RID p_sample) const  {
	///AUDIO_LOCK
	return sample_manager->sample_get_length(p_sample);
}

const void* AudioServerSW::sample_get_data_ptr(RID p_sample) const  {
	///AUDIO_LOCK
	return sample_manager->sample_get_data_ptr(p_sample);
}

void AudioServerSW::sample_set_data(RID p_sample, const DVector<uint8_t>& p_buffer) {
	AUDIO_LOCK
	sample_manager->sample_set_data(p_sample,p_buffer);
}
DVector<uint8_t> AudioServerSW::sample_get_data(RID p_sample) const {
	AUDIO_LOCK
	return sample_manager->sample_get_data(p_sample);
}

void AudioServerSW::sample_set_mix_rate(RID p_sample,int p_rate) {
	AUDIO_LOCK
	sample_manager->sample_set_mix_rate(p_sample,p_rate);
}
int AudioServerSW::sample_get_mix_rate(RID p_sample) const {
	AUDIO_LOCK
	return sample_manager->sample_get_mix_rate(p_sample);
}

void AudioServerSW::sample_set_loop_format(RID p_sample,SampleLoopFormat p_format) {
	AUDIO_LOCK
	sample_manager->sample_set_loop_format(p_sample,p_format);
}
AS::SampleLoopFormat AudioServerSW::sample_get_loop_format(RID p_sample) const {
	AUDIO_LOCK
	return sample_manager->sample_get_loop_format(p_sample);
}

void AudioServerSW::sample_set_loop_begin(RID p_sample,int p_pos) {
	AUDIO_LOCK
	sample_manager->sample_set_loop_begin(p_sample,p_pos);
}
int AudioServerSW::sample_get_loop_begin(RID p_sample) const {
	AUDIO_LOCK
	return sample_manager->sample_get_loop_begin(p_sample);
}

void AudioServerSW::sample_set_loop_end(RID p_sample,int p_pos) {
	AUDIO_LOCK
	sample_manager->sample_set_loop_end(p_sample,p_pos);
}
int AudioServerSW::sample_get_loop_end(RID p_sample) const {
	AUDIO_LOCK
	return sample_manager->sample_get_loop_end(p_sample);
}

/* VOICE API */

RID AudioServerSW::voice_create() {

	Voice * v = memnew( Voice );
	v->channel=AudioMixer::INVALID_CHANNEL;

	AUDIO_LOCK
	return voice_owner.make_rid(v);

}
void AudioServerSW::voice_play(RID p_voice, RID p_sample) {

	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND(!v);
	v->active=true; // force actvive (will be disabled later i gues..)

	//stop old, start new
	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_PLAY;
	cmd.voice=p_voice;
	cmd.play.sample=p_sample;
	voice_rb.push_command(cmd);

}

void AudioServerSW::voice_set_volume(RID p_voice, float p_volume) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_SET_VOLUME;
	cmd.voice=p_voice;
	cmd.volume.volume=p_volume;
	voice_rb.push_command(cmd);

}
void AudioServerSW::voice_set_pan(RID p_voice, float p_pan, float p_depth,float p_height) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_SET_PAN;
	cmd.voice=p_voice;
	cmd.pan.pan=p_pan;
	cmd.pan.depth=p_depth;
	cmd.pan.height=p_height;
	voice_rb.push_command(cmd);

}
void AudioServerSW::voice_set_filter(RID p_voice, FilterType p_type, float p_cutoff, float p_resonance,float p_gain) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_SET_FILTER;
	cmd.voice=p_voice;
	cmd.filter.type=p_type;
	cmd.filter.cutoff=p_cutoff;
	cmd.filter.resonance=p_resonance;
	cmd.filter.gain=p_gain;
	voice_rb.push_command(cmd);

}
void AudioServerSW::voice_set_chorus(RID p_voice, float p_chorus ) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_SET_CHORUS;
	cmd.voice=p_voice;
	cmd.chorus.send=p_chorus;
	voice_rb.push_command(cmd);

}
void AudioServerSW::voice_set_reverb(RID p_voice, ReverbRoomType p_room_type, float p_reverb) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_SET_REVERB;
	cmd.voice=p_voice;
	cmd.reverb.room=p_room_type;
	cmd.reverb.send=p_reverb;

	voice_rb.push_command(cmd);

}
void AudioServerSW::voice_set_mix_rate(RID p_voice, int p_mix_rate) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_SET_MIX_RATE;
	cmd.voice=p_voice;
	cmd.mix_rate.mix_rate=p_mix_rate;
	voice_rb.push_command(cmd);

}
void AudioServerSW::voice_set_positional(RID p_voice, bool p_positional) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_SET_POSITIONAL;
	cmd.voice=p_voice;
	cmd.positional.positional=p_positional;
	voice_rb.push_command(cmd);

}

float AudioServerSW::voice_get_volume(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_volume( v->channel );

}
float AudioServerSW::voice_get_pan(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_pan( v->channel );

}
float AudioServerSW::voice_get_pan_depth(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_pan_depth( v->channel );

}
float AudioServerSW::voice_get_pan_height(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_pan_height( v->channel );

}
AS::FilterType AudioServerSW::voice_get_filter_type(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, AS::FILTER_NONE);

	return (AS::FilterType)mixer->channel_get_filter_type(v->channel);

}
float AudioServerSW::voice_get_filter_cutoff(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_filter_cutoff( v->channel );

}
float AudioServerSW::voice_get_filter_resonance(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_filter_resonance( v->channel );

}
float AudioServerSW::voice_get_chorus(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_chorus( v->channel );

}
AS::ReverbRoomType AudioServerSW::voice_get_reverb_type(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, REVERB_SMALL);

	return (AS::ReverbRoomType)mixer->channel_get_reverb_type( v->channel );

}
float AudioServerSW::voice_get_reverb(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_reverb( v->channel );

}

int AudioServerSW::voice_get_mix_rate(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_get_mix_rate( v->channel );

}
bool AudioServerSW::voice_is_positional(RID p_voice) const {

	AUDIO_LOCK
	Voice *v = voice_owner.get( p_voice );
	ERR_FAIL_COND_V(!v, 0);

	return mixer->channel_is_positional( v->channel );

}

void AudioServerSW::voice_stop(RID p_voice) {

	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_STOP;
	cmd.voice=p_voice;
	voice_rb.push_command(cmd);

	//return mixer->channel_free( v->channel );

}

bool AudioServerSW::voice_is_active(RID p_voice) const {

	Voice *v = voice_owner.get(p_voice);
	ERR_FAIL_COND_V(!v,false);
	return v->active;

}

/* STREAM API */

RID AudioServerSW::audio_stream_create(AudioStream *p_stream) {

	AUDIO_LOCK
	Stream *s = memnew(Stream);
	s->audio_stream=p_stream;
	s->event_stream=NULL;
	s->active=false;
	s->E=NULL;
	s->volume_scale=1.0;
	p_stream->set_mix_rate(AudioDriverSW::get_singleton()->get_mix_rate());

	return stream_owner.make_rid(s);
}

RID AudioServerSW::event_stream_create(EventStream *p_stream) {

	AUDIO_LOCK
	Stream *s = memnew(Stream);
	s->audio_stream=NULL;
	s->event_stream=p_stream;
	s->active=false;
	s->E=NULL;
	s->volume_scale=1.0;
	//p_stream->set_mix_rate(AudioDriverSW::get_singleton()->get_mix_rate());

	return stream_owner.make_rid(s);


}


void AudioServerSW::stream_set_active(RID p_stream, bool p_active) {


	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND(!s);
	_THREAD_SAFE_METHOD_

	if (s->active==p_active)
		return;
	AUDIO_LOCK;
	s->active=p_active;
	if (p_active)
		s->E=active_audio_streams.push_back(s);
	else {
		active_audio_streams.erase(s->E);
		s->E=NULL;
	}


}

bool AudioServerSW::stream_is_active(RID p_stream) const {

	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND_V(!s,false);
	return s->active;
}

void AudioServerSW::stream_set_volume_scale(RID p_stream, float p_scale) {

	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND(!s);
	s->volume_scale=p_scale;

}

float AudioServerSW::stream_set_volume_scale(RID p_stream) const {

	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND_V(!s,0);
	return s->volume_scale;

}


void AudioServerSW::free(RID p_id) {

	if(voice_owner.owns(p_id)) {

		Voice *v = voice_owner.get(p_id);
		AUDIO_LOCK
		mixer->channel_free( v->channel );
		voice_owner.free(p_id);
		memdelete(v);

	} else if (stream_owner.owns(p_id)) {


		Stream *s=stream_owner.get(p_id);

		if (s->active) {
			stream_set_active(p_id,false);
		}

		memdelete(s);
		stream_owner.free(p_id);

	} else if (sample_manager->is_sample(p_id)) {

		AUDIO_LOCK
		sample_manager->free(p_id);
	}

}

void AudioServerSW::_thread_func(void *self) {

	Thread::set_name("AudioServerSW");

	AudioServerSW *as=(AudioServerSW *)self;

	while (!as->exit_update_thread) {
		as->_update_streams(true);
		OS::get_singleton()->delay_usec(5000);
	}

}

void AudioServerSW::init() {

	int latency = GLOBAL_DEF("audio/mixer_latency",10);
	internal_buffer_channels=2; // read from driver
	internal_buffer = memnew_arr(int32_t,INTERNAL_BUFFER_SIZE*internal_buffer_channels);
	stream_buffer = memnew_arr(int32_t,INTERNAL_BUFFER_SIZE*4); //max 4 channels
	AudioMixerSW::MixChannels mix_chans = AudioMixerSW::MIX_STEREO;

	switch(AudioDriverSW::get_singleton()->get_output_format()) {

		case AudioDriverSW::OUTPUT_MONO:
		case AudioDriverSW::OUTPUT_STEREO:
			mix_chans=AudioMixerSW::MIX_STEREO;
			break;
		case AudioDriverSW::OUTPUT_QUAD:
		case AudioDriverSW::OUTPUT_5_1:
			mix_chans=AudioMixerSW::MIX_QUAD;
			break;
	}

	mixer = memnew( AudioMixerSW( sample_manager, latency, AudioDriverSW::get_singleton()->get_mix_rate(),mix_chans,mixer_use_fx,mixer_interp,_mixer_callback,this ) );
	mixer_step_usecs=mixer->get_step_usecs();

	_output_delay=0;

	stream_volume=0.3;
	// start the audio driver
	if (AudioDriverSW::get_singleton())
		AudioDriverSW::get_singleton()->start();

#ifndef NO_THREADS
	exit_update_thread=false;
	thread = Thread::create(_thread_func,this);
#endif

}

void AudioServerSW::finish() {

#ifndef NO_THREADS
	exit_update_thread=true;
	Thread::wait_to_finish(thread);
	memdelete(thread);
#endif

	if (AudioDriverSW::get_singleton())
		AudioDriverSW::get_singleton()->finish();

	memdelete_arr(internal_buffer);
	memdelete_arr(stream_buffer);
	memdelete(mixer);

}

void AudioServerSW::_update_streams(bool p_thread) {

	_THREAD_SAFE_METHOD_
	for(List<Stream*>::Element *E=active_audio_streams.front();E;) { //stream might be removed durnig this callback

		List<Stream*>::Element *N=E->next();

		if (E->get()->audio_stream && p_thread == E->get()->audio_stream->can_update_mt())
			E->get()->audio_stream->update();

		E=N;
	}

}

void AudioServerSW::update() {

	_update_streams(false);
#ifdef NO_THREADS

	_update_streams(true);
#endif
}


void AudioServerSW::lock() {

	AudioDriverSW::get_singleton()->lock();
}

void AudioServerSW::unlock() {
	AudioDriverSW::get_singleton()->unlock();

}

int AudioServerSW::get_default_mix_rate() const {

	return AudioDriverSW::get_singleton()->get_mix_rate();
}
int AudioServerSW::get_default_channel_count() const {
	return internal_buffer_channels;
}

void AudioServerSW::set_mixer_params(AudioMixerSW::InterpolationType p_interp, bool p_use_fx) {

	mixer_interp=p_interp;
	mixer_use_fx=p_use_fx;
}

void AudioServerSW::set_stream_global_volume_scale(float p_volume) {

	stream_volume_scale=p_volume;
}

float AudioServerSW::get_stream_global_volume_scale() const {

	return stream_volume_scale;


}

void AudioServerSW::set_fx_global_volume_scale(float p_volume) {

	fx_volume_scale=p_volume;
	//mixer->set_mixer_volume(fx_volume_scale);
	VoiceRBSW::Command cmd;
	cmd.type=VoiceRBSW::Command::CMD_CHANGE_ALL_FX_VOLUMES;
	cmd.voice=RID();
	cmd.volume.volume=p_volume;
	voice_rb.push_command(cmd);

}


float AudioServerSW::get_fx_global_volume_scale() const {

	return fx_volume_scale;
}

void AudioServerSW::set_event_voice_global_volume_scale(float p_volume) {

	event_voice_volume_scale=p_volume;
	//mixer->set_mixer_volume(event_voice_volume_scale);
}


float AudioServerSW::get_event_voice_global_volume_scale() const {

	return event_voice_volume_scale;
}

double AudioServerSW::get_output_delay() const {

	return _output_delay+AudioDriverSW::get_singleton()->get_latency();
}

double AudioServerSW::get_mix_time() const {

	return AudioDriverSW::get_singleton()->get_mix_time();
}

uint32_t AudioServerSW::read_output_peak() const {

	uint32_t val = max_peak;
	uint32_t *p = (uint32_t*)&max_peak;
	*p=0;
	return val;
}

AudioServerSW::AudioServerSW(SampleManagerSW *p_sample_manager) {

	sample_manager=p_sample_manager;
	String interp = GLOBAL_DEF("audio/mixer_interp","linear");
	GlobalConfig::get_singleton()->set_custom_property_info("audio/mixer_interp",PropertyInfo(Variant::STRING,"audio/mixer_interp",PROPERTY_HINT_ENUM,"raw,linear,cubic"));
	if (interp=="raw")
		mixer_interp=AudioMixerSW::INTERPOLATION_RAW;
	else if (interp=="cubic")
		mixer_interp=AudioMixerSW::INTERPOLATION_CUBIC;
	else
		mixer_interp=AudioMixerSW::INTERPOLATION_LINEAR;
	mixer_use_fx = GLOBAL_DEF("audio/use_chorus_reverb",true);
	stream_volume_scale=GLOBAL_DEF("audio/stream_volume_scale",1.0);
	fx_volume_scale=GLOBAL_DEF("audio/fx_volume_scale",1.0);
	event_voice_volume_scale=GLOBAL_DEF("audio/event_voice_volume_scale",0.5);
	max_peak=0;


}

AudioServerSW::~AudioServerSW() {

}


AudioDriverSW *AudioDriverSW::singleton=NULL;
AudioDriverSW *AudioDriverSW::get_singleton() {

	return singleton;
}

void AudioDriverSW::set_singleton() {

	singleton=this;
}

void AudioDriverSW::audio_server_process(int p_frames,int32_t *p_buffer,bool p_update_mix_time) {

	AudioServerSW * audio_server = static_cast<AudioServerSW*>(AudioServer::get_singleton());
	if (p_update_mix_time)
		update_mix_time(p_frames);
	audio_server->driver_process(p_frames,p_buffer);
}

void AudioDriverSW::update_mix_time(int p_frames) {

	_mix_amount+=p_frames;
	_last_mix_time=OS::get_singleton()->get_ticks_usec();
}

double AudioDriverSW::get_mix_time() const {

	double total = (OS::get_singleton()->get_ticks_usec() - _last_mix_time) / 1000000.0;
	total+=_mix_amount/(double)get_mix_rate();
	return total;

}


AudioDriverSW::AudioDriverSW() {

	_last_mix_time=0;
	_mix_amount=0;
}


AudioDriverSW *AudioDriverManagerSW::drivers[MAX_DRIVERS];
int AudioDriverManagerSW::driver_count=0;



void AudioDriverManagerSW::add_driver(AudioDriverSW *p_driver) {

	ERR_FAIL_COND(driver_count>=MAX_DRIVERS);
	drivers[driver_count++]=p_driver;
}

int AudioDriverManagerSW::get_driver_count() {

	return driver_count;
}
AudioDriverSW *AudioDriverManagerSW::get_driver(int p_driver) {

	ERR_FAIL_INDEX_V(p_driver,driver_count,NULL);
	return drivers[p_driver];
}

