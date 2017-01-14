/*************************************************************************/
/*  event_stream_chibi.cpp                                               */
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
#include "event_stream_chibi.h"
#include "cp_loader_it.h"
#include "cp_loader_xm.h"
#include "cp_loader_s3m.h"
#include "cp_loader_mod.h"

static CPSampleManagerImpl *sample_manager;
static ResourceFormatLoaderChibi *resource_loader;

CPSample_ID CPSampleManagerImpl::create(bool p_16bits,bool p_stereo,int32_t p_len) {

	AudioServer::SampleFormat sf=p_16bits?AudioServer::SAMPLE_FORMAT_PCM16:AudioServer::SAMPLE_FORMAT_PCM8;

	SampleData *sd = memnew( SampleData );
	sd->rid = AudioServer::get_singleton()->sample_create(sf,p_stereo,p_len);
	sd->stereo=p_stereo;
	sd->len=p_len;
	sd->is16=p_16bits;
	sd->mixfreq=44100;
	sd->loop_begin=0;
	sd->loop_end=0;
	sd->loop_type=CP_LOOP_NONE;
	sd->locks=0;
#ifdef DEBUG_ENABLED
	valid.insert(sd);
#endif
	CPSample_ID sid;
	sid._private=sd;
	return sid;
}

void CPSampleManagerImpl::recreate(CPSample_ID p_id,bool p_16bits,bool p_stereo,int32_t p_len){

	AudioServer::SampleFormat sf=p_16bits?AudioServer::SAMPLE_FORMAT_PCM16:AudioServer::SAMPLE_FORMAT_PCM8;
	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif
	AudioServer::get_singleton()->free(sd->rid);
	sd->rid = AudioServer::get_singleton()->sample_create(sf,p_stereo,p_len);
	sd->stereo=p_stereo;
	sd->len=p_len;
	sd->is16=p_16bits;
	sd->mixfreq=44100;
	sd->loop_begin=0;
	sd->loop_end=0;
	sd->loop_type=CP_LOOP_NONE;
}
void CPSampleManagerImpl::destroy(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
	valid.erase(sd);
#endif
	AudioServer::get_singleton()->free(sd->rid);

	memdelete(sd);
}
bool CPSampleManagerImpl::check(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	return valid.has(sd);
#else
	return _getsd(p_id)!=NULL;
#endif
}

void CPSampleManagerImpl::set_c5_freq(CPSample_ID p_id,int32_t p_freq){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif
	sd->mixfreq=p_freq;
	AudioServer::get_singleton()->sample_set_mix_rate(sd->rid,p_freq);

}
void CPSampleManagerImpl::set_loop_begin(CPSample_ID p_id,int32_t p_begin){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif
	sd->loop_begin=p_begin;
	AudioServer::get_singleton()->sample_set_loop_begin(sd->rid,p_begin);

}
void CPSampleManagerImpl::set_loop_end(CPSample_ID p_id,int32_t p_end){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif
	sd->loop_end=p_end;
	AudioServer::get_singleton()->sample_set_loop_end(sd->rid,p_end);

}
void CPSampleManagerImpl::set_loop_type(CPSample_ID p_id,CPSample_Loop_Type p_type){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif

	sd->loop_type=p_type;
	AudioServer::get_singleton()->sample_set_loop_format(sd->rid,AudioServer::SampleLoopFormat(p_type));


}
void CPSampleManagerImpl::set_chunk(CPSample_ID p_id,int32_t p_index,void *p_data,int p_data_len){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif

	ERR_FAIL();
}


int32_t CPSampleManagerImpl::get_loop_begin(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),0);
#endif

	return sd->loop_begin;

}
int32_t CPSampleManagerImpl::get_loop_end(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),0);
#endif

	return sd->loop_end;
}
CPSample_Loop_Type CPSampleManagerImpl::get_loop_type(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),CP_LOOP_NONE);
#endif

	return sd->loop_type;
}
int32_t CPSampleManagerImpl::get_c5_freq(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),0);
#endif

	return sd->mixfreq;
}
int32_t CPSampleManagerImpl::get_size(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),0);
#endif

	return sd->len;

}
bool CPSampleManagerImpl::is_16bits(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),false);
#endif

	return sd->is16;

}
bool CPSampleManagerImpl::is_stereo(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),false);
#endif
	return sd->stereo;


}
bool CPSampleManagerImpl::lock_data(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),0);
#endif

	sd->locks++;
	if (sd->locks==1) {
		sd->lock=AudioServer::get_singleton()->sample_get_data(sd->rid);
		sd->w=sd->lock.write();
	}

	return true;
}
void *CPSampleManagerImpl::get_data(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),0);
#endif

	ERR_FAIL_COND_V(sd->locks==0,0);
	return sd->w.ptr();
}

int16_t CPSampleManagerImpl::get_data(CPSample_ID p_id, int p_sample, int p_channel){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!valid.has(sd),0);
#endif

	ERR_FAIL_V(0);
	lock_data(p_id);

	int sofs = sd->stereo ? 2:1;
	uint16_t v=0;
	if (sd->is16) {
		int16_t *p=(int16_t*)sd->w.ptr();
		v=p[p_sample*sofs+p_channel];
	} else {
		int8_t *p=(int8_t*)sd->w.ptr();
		v=p[p_sample*sofs+p_channel];
	}

	unlock_data(p_id);

	return v;
}
void CPSampleManagerImpl::set_data(CPSample_ID p_id, int p_sample, int16_t p_data,int p_channel){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif

	ERR_FAIL();
	lock_data(p_id);

	int sofs = sd->stereo ? 2:1;
	if (sd->is16) {
		int16_t *p=(int16_t*)sd->w.ptr();
		p[p_sample*sofs+p_channel]=p_data;
	} else {
		int8_t *p=(int8_t*)sd->w.ptr();
		p[p_sample*sofs+p_channel]=p_data;
	}

	unlock_data(p_id);

}
void CPSampleManagerImpl::unlock_data(CPSample_ID p_id){

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif

	ERR_FAIL_COND(sd->locks==0);

	sd->locks--;
	if (sd->locks==0) {
		sd->w=PoolVector<uint8_t>::Write();
		AudioServer::get_singleton()->sample_set_data(sd->rid,sd->lock);
		sd->lock=PoolVector<uint8_t>();
	}
}

void CPSampleManagerImpl::get_chunk(CPSample_ID p_id,int32_t p_index,void *p_data,int p_data_len) {

	SampleData *sd=_getsd(p_id);
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(!valid.has(sd));
#endif

	ERR_FAIL();
}


/** MIXER **/

void CPMixerImpl::set_callback_interval(int p_interval_us) {

	callback_interval=p_interval_us;
}

void CPMixerImpl::set_callback(void (*p_callback)(void*),void *p_userdata) {

	callback=p_callback;
	userdata=p_userdata;
}

void CPMixerImpl::setup_voice(int p_voice_index,CPSample_ID p_sample_id,int32_t p_start_index) {

	Voice &v=voices[p_voice_index];
	if (v.channel!=AudioMixer::INVALID_CHANNEL) {
		mixer->channel_free(v.channel);
	}
	v.channel=mixer->channel_alloc(sample_manager->get_rid(p_sample_id));
	v.freq_mult = sample_manager->get_c5_freq(p_sample_id)/261.6255653006;
	v.sample = p_sample_id;
}

void CPMixerImpl::stop_voice(int p_voice_index)  {

	Voice &v=voices[p_voice_index];
	if (v.channel==AudioMixer::INVALID_CHANNEL)
		return;

	mixer->channel_free(v.channel);
	v.channel=AudioMixer::INVALID_CHANNEL;

}

void CPMixerImpl::set_voice_frequency(int p_voice_index,int32_t p_freq) {

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);
	float f = p_freq / 256.0;
	f*=pitch_scale;
	mixer->channel_set_mix_rate(v.channel,f * v.freq_mult );
}

void CPMixerImpl::set_voice_panning(int p_voice_index,int p_pan) {

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);
	if (p_pan==CP_PAN_SURROUND)
		p_pan=CP_PAN_CENTER;
	float p = p_pan / 256.0;
	mixer->channel_set_pan(v.channel,p);

}

void CPMixerImpl::set_voice_volume(int p_voice_index,int p_vol) {


	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);
	float vol = p_vol/512.0;
	vol*=voice_scale;
	mixer->channel_set_volume(v.channel,vol);
	mixer->channel_set_reverb(v.channel,reverb_type,vol*v.reverb);
}

void CPMixerImpl::set_voice_filter(int p_voice_index,bool p_enabled,uint8_t p_cutoff, uint8_t p_resonance ){

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);

}

void CPMixerImpl::set_voice_reverb_send(int p_voice_index,int p_reverb){

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);
	v.reverb=p_reverb/255.0;
	//mixer->channel_set_reverb(v.channel,reverb_type,p_reverb/255.0);

}

void CPMixerImpl::set_voice_chorus_send(int p_voice_index,int p_chorus){

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);
	mixer->channel_set_chorus(v.channel,p_chorus/255.0);

}


void CPMixerImpl::set_reverb_mode(ReverbMode p_mode){

//	Voice &v=voices[p_voice_index];
//	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);
	switch(p_mode) {
		case CPMixer::REVERB_MODE_STUDIO_SMALL: reverb_type=AudioMixer::REVERB_SMALL; break;
		case CPMixer::REVERB_MODE_STUDIO_MEDIUM: reverb_type=AudioMixer::REVERB_MEDIUM; break;
		case CPMixer::REVERB_MODE_STUDIO_LARGE: reverb_type=AudioMixer::REVERB_LARGE; break;
		case CPMixer::REVERB_MODE_HALL: reverb_type=AudioMixer::REVERB_HALL; break;
		default: reverb_type=AudioMixer::REVERB_SMALL; break;
	}

}

void CPMixerImpl::set_chorus_params(unsigned int p_delay_ms,unsigned int p_separation_ms,unsigned int p_depth_ms10,unsigned int p_speed_hz10){

//	Voice &v=voices[p_voice_index];
//	ERR_FAIL_COND(v.channel==AudioMixer::INVALID_CHANNEL);

}



/* Info retrieving */

int32_t CPMixerImpl::get_voice_sample_pos_index(int p_voice_index) {

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND_V(v.channel==AudioMixer::INVALID_CHANNEL,0);
	return 0;

}

int CPMixerImpl::get_voice_panning(int p_voice_index) {

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND_V(!is_voice_active(p_voice_index),0);
	return mixer->channel_get_pan(v.channel)*CP_PAN_RIGHT;

}

int CPMixerImpl::get_voice_volume(int p_voice_index) {

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND_V(!is_voice_active(p_voice_index),0);
	return mixer->channel_get_volume(v.channel);


}

CPSample_ID CPMixerImpl::get_voice_sample_id(int p_voice_index) {

	Voice &v=voices[p_voice_index];
	ERR_FAIL_COND_V(v.channel==AudioMixer::INVALID_CHANNEL,CPSample_ID());
	return v.sample;


}

bool CPMixerImpl::is_voice_active(int p_voice_index){

	Voice &v=voices[p_voice_index];
	if (v.channel==AudioMixer::INVALID_CHANNEL)
		return false;
	if (!mixer->channel_is_valid(v.channel))
		v.channel=AudioMixer::INVALID_CHANNEL;

	return v.channel!=AudioMixer::INVALID_CHANNEL;
}

void CPMixerImpl::process_usecs(int p_usec,float p_volume,float p_pitch_scale,float p_tempo_scale) {

	ERR_FAIL_COND(callback_interval==0);
	//update this somewhere
	pitch_scale=p_pitch_scale;
	tempo_scale=p_tempo_scale;
	voice_scale = AudioServer::get_singleton()->get_event_voice_global_volume_scale()*p_volume;
	while(p_usec) {

		if (p_usec>=callback_timeout) {

			p_usec-=callback_timeout;
			callback_timeout=0;
			if (callback) {
				callback(userdata);
			}
			callback_timeout=callback_interval*(1.0/p_tempo_scale);

		} else {

			callback_timeout-=p_usec;
			p_usec=0;
		}
	}
}


CPMixerImpl::CPMixerImpl(AudioMixer *p_mixer) {

	callback_interval=1;
	callback_timeout=0;
	userdata=0;
	callback=0;
	tempo_scale=1.0;
	pitch_scale=1.0;
	mixer=p_mixer;
	voice_scale = AudioServer::get_singleton()->get_event_voice_global_volume_scale();
	reverb_type = AudioMixer::REVERB_SMALL;

}

/** FILE ACCESS WRAPPER **/


CPFileAccessWrapperImpl::Error CPFileAccessWrapperImpl::open(const char *p_filename, int p_mode_flags) {

	ERR_FAIL_COND_V(p_mode_flags&WRITE,ERROR_WRITING_FILE);
	close();
	f = FileAccess::open(String::utf8(p_filename),p_mode_flags);
	if (!f)
		return ERROR_FILE_NOT_FOUND;
	return OK;
}

void CPFileAccessWrapperImpl::close(){

	if (f)
		memdelete(f);
	f=NULL;


}

void CPFileAccessWrapperImpl::seek(uint32_t p_position){

	f->seek(p_position);
}
void CPFileAccessWrapperImpl::seek_end(){

	f->seek_end();
}
uint32_t CPFileAccessWrapperImpl::get_pos(){

	return f->get_pos();
}

bool CPFileAccessWrapperImpl::eof_reached(){

	return f->eof_reached();
}

uint8_t CPFileAccessWrapperImpl::get_byte(){

	return f->get_8();
}
void CPFileAccessWrapperImpl::get_byte_array(uint8_t *p_dest,int p_elements){

	f->get_buffer(p_dest,p_elements);
}
void CPFileAccessWrapperImpl::get_word_array(uint16_t *p_dest,int p_elements){

	for(int i=0;i<p_elements;i++) {
		p_dest[i]=f->get_16();
	}

}

uint16_t CPFileAccessWrapperImpl::get_word(){

	return f->get_16();
}
uint32_t CPFileAccessWrapperImpl::get_dword(){

	return f->get_32();
}

void CPFileAccessWrapperImpl::set_endian_conversion(bool p_swap){

	f->set_endian_swap(p_swap);
}
bool CPFileAccessWrapperImpl::is_open(){

	return f!=NULL;
}

CPFileAccessWrapperImpl::Error CPFileAccessWrapperImpl::get_error(){

	return (f->get_error()!=::OK)?ERROR_READING_FILE:OK;
}

void CPFileAccessWrapperImpl::store_byte(uint8_t p_dest){

}
void CPFileAccessWrapperImpl::store_byte_array(const uint8_t *p_dest,int p_elements){

}

void CPFileAccessWrapperImpl::store_word(uint16_t p_dest){

}
void CPFileAccessWrapperImpl::store_dword(uint32_t p_dest){

}

////////////////////////////////////////////////


Error EventStreamPlaybackChibi::_play() {

	last_order=0;
	loops=0;
	player->play_start_song();
	total_usec=0;

	return OK;
}

bool EventStreamPlaybackChibi::_update(AudioMixer* p_mixer, uint64_t p_usec){

	total_usec+=p_usec;
	mixer.process_usecs(p_usec,volume,pitch_scale,tempo_scale);
	int order=player->get_current_order();
	if (order<last_order) {
		if (!loop) {
			stop();
		} else {
			loops++;
		}
	}
	last_order=order;
	return false;
}
void EventStreamPlaybackChibi::_stop(){

	player->play_stop();
}

void EventStreamPlaybackChibi::set_paused(bool p_paused){

}
bool EventStreamPlaybackChibi::is_paused() const{

	return false;
}
void EventStreamPlaybackChibi::set_loop(bool p_loop){

	loop=p_loop;

}
bool EventStreamPlaybackChibi::is_loop_enabled() const{

	return loop;
}

int EventStreamPlaybackChibi::get_loop_count() const{

	//return player->is
	return loops;
}

float EventStreamPlaybackChibi::get_pos() const{

	return double(total_usec)/1000000.0;
}
void EventStreamPlaybackChibi::seek_pos(float p_time){

	WARN_PRINT("seek_pos unimplemented.");
}

void EventStreamPlaybackChibi::set_volume(float p_volume) {

	volume=p_volume;
}

float EventStreamPlaybackChibi::get_volume() const{

	return volume;
}

void EventStreamPlaybackChibi::set_pitch_scale(float p_pitch_scale) {

	pitch_scale=p_pitch_scale;
}

float EventStreamPlaybackChibi::get_pitch_scale() const{

	return pitch_scale;
}

void EventStreamPlaybackChibi::set_tempo_scale(float p_tempo_scale) {

	tempo_scale=p_tempo_scale;
}

float EventStreamPlaybackChibi::get_tempo_scale() const{

	return tempo_scale;
}


void EventStreamPlaybackChibi::set_channel_volume(int p_channel,float p_volume) {


	if (p_channel>=64)
		return;
	player->set_channel_global_volume(p_channel,p_volume*256);
}



float EventStreamPlaybackChibi::get_channel_volume(int p_channel) const{

	return player->get_channel_global_volume(p_channel)/256.0;

}

float EventStreamPlaybackChibi::get_last_note_time(int p_channel) const {


	double v = (player->get_channel_last_note_time_usec(p_channel))/1000000.0;
	if (v<0)
		v=-1;
	return v;
}

EventStreamPlaybackChibi::EventStreamPlaybackChibi(Ref<EventStreamChibi> p_stream) : mixer(_get_mixer()) {

	stream=p_stream;
	player = memnew( CPPlayer(&mixer,&p_stream->song) );
	loop=false;
	last_order=0;
	loops=0;
	volume=1.0;
	pitch_scale=1.0;
	tempo_scale=1.0;
}
EventStreamPlaybackChibi::~EventStreamPlaybackChibi(){

	player->play_stop();
	memdelete(player);
}

////////////////////////////////////////////////////

Ref<EventStreamPlayback> EventStreamChibi::instance_playback() {

	return Ref<EventStreamPlayback>( memnew(EventStreamPlaybackChibi(Ref<EventStreamChibi>(this))) );
}

String EventStreamChibi::get_stream_name() const{

	return song.get_name();

}



float EventStreamChibi::get_length() const{

	return 1;
}


EventStreamChibi::EventStreamChibi() {


}



//////////////////////////////////////////////////////////////////




RES ResourceFormatLoaderChibi::load(const String &p_path, const String& p_original_path, Error *r_error) {

	if (r_error)
		*r_error=ERR_FILE_CANT_OPEN;
	String el = p_path.get_extension().to_lower();

	CPFileAccessWrapperImpl f;

	if (el=="it") {

		Ref<EventStreamChibi> esc( memnew( EventStreamChibi ) );
		CPLoader_IT loader(&f);
		CPLoader::Error err = loader.load_song(p_path.utf8().get_data(),&esc->song,false);
		ERR_FAIL_COND_V(err!=CPLoader::FILE_OK,RES());
		if (r_error)
			*r_error=OK;

		return esc;

	} else if (el=="xm") {

		Ref<EventStreamChibi> esc( memnew( EventStreamChibi ) );
		CPLoader_XM loader(&f);
		CPLoader::Error err=loader.load_song(p_path.utf8().get_data(),&esc->song,false);
		ERR_FAIL_COND_V(err!=CPLoader::FILE_OK,RES());
		if (r_error)
			*r_error=OK;
		return esc;

	} else if (el=="s3m") {

		Ref<EventStreamChibi> esc( memnew( EventStreamChibi ) );
		CPLoader_S3M loader(&f);
		CPLoader::Error err=loader.load_song(p_path.utf8().get_data(),&esc->song,false);
		ERR_FAIL_COND_V(err!=CPLoader::FILE_OK,RES());
		if (r_error)
			*r_error=OK;

		return esc;

	} else if (el=="mod") {

		Ref<EventStreamChibi> esc( memnew( EventStreamChibi ) );
		CPLoader_MOD loader(&f);
		CPLoader::Error err=loader.load_song(p_path.utf8().get_data(),&esc->song,false);
		ERR_FAIL_COND_V(err!=CPLoader::FILE_OK,RES());
		if (r_error)
			*r_error=OK;
		return esc;
	}

	return RES();

}

void ResourceFormatLoaderChibi::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("it");
	p_extensions->push_back("xm");
	p_extensions->push_back("s3m");
	p_extensions->push_back("mod");
}
bool ResourceFormatLoaderChibi::handles_type(const String& p_type) const {

	return (p_type=="EventStreamChibi" || p_type=="EventStream");
}

String ResourceFormatLoaderChibi::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el=="it" || el=="s3m" || el=="xm" || el=="mod")
		return "EventStreamChibi";
	return "";
}

/////////////////////////////////////////////////////////////////
void initialize_chibi() {

	sample_manager = memnew( CPSampleManagerImpl );
	resource_loader = memnew( ResourceFormatLoaderChibi );
	ClassDB::register_class<EventStreamChibi>();
	ResourceLoader::add_resource_format_loader( resource_loader );
}

void finalize_chibi() {

	memdelete( sample_manager );
	memdelete( resource_loader );
}

