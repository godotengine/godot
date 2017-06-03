/*************************************************************************/
/*  gibberish_stream.cpp                                                 */
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
#include "gibberish_stream.h"
#include "servers/audio_server.h"

//TODO: This class needs to be adapted to the new AudioStream API,
// or dropped if nobody cares about fixing it :) (GH-3307)

#if 0

int AudioStreamGibberish::get_channel_count() const {

	return 1;
}


static float _get_vol_at_pos(int p_pos, int p_len, int p_x_fade) {

	if (p_pos < p_x_fade)
		return float(p_pos)/p_x_fade;
	else if (p_pos>(p_len-p_x_fade))
		return float(p_len-p_pos)/p_x_fade;
	else
		return 1.0;

}
 int AudioStreamGibberish::randomize() {

	if (rand_idx==_rand_pool.size()) {

		for(int i=0;i<_rand_pool.size();i++) {

			SWAP(_rand_pool[i],_rand_pool[Math::rand()%_rand_pool.size()]);
		}
		rand_idx=0;
	}

	return _rand_pool[rand_idx++];
}

bool AudioStreamGibberish::mix(int32_t *p_buffer, int p_frames) {

	if (!active)
		return false;

	zeromem(p_buffer,p_frames*sizeof(int32_t));

	if (!paused && active_voices==0) {

		active_voices=1;
		playback[0].idx=randomize();
		playback[0].fp_pos=0;
		playback[0].scale=Math::random(1,1+pitch_random_scale);
	}

	for(int i=0;i<active_voices;i++) {

		RID s = _samples[playback[i].idx]->get_rid();

		uint64_t fp_pos=playback[i].fp_pos;
		const void *data = AudioServer::get_singleton()->sample_get_data_ptr(s);
		bool is16 = AudioServer::get_singleton()->sample_get_format(s)==AudioServer::SAMPLE_FORMAT_PCM16;
		int skip = AudioServer::get_singleton()->sample_is_stereo(s) ? 1: 0;
		uint64_t max = AudioServer::get_singleton()->sample_get_length(s) * uint64_t(FP_LEN);
		int mrate = AudioServer::get_singleton()->sample_get_mix_rate(s) * pitch_scale * playback[i].scale;
		uint64_t increment = uint64_t(mrate) * uint64_t(FP_LEN) / get_mix_rate();


		float vol_begin = _get_vol_at_pos(fp_pos>>FP_BITS,max>>FP_BITS,xfade_time*mrate);
		float vol_end = _get_vol_at_pos((fp_pos+p_frames*increment)>>FP_BITS,max>>FP_BITS,xfade_time*mrate);

		int32_t vol = CLAMP(int32_t(vol_begin * 65535),0,65535);
		int32_t vol_to = CLAMP(int32_t(vol_end * 65535),0,65535);
		int32_t vol_inc = (vol_to-vol)/p_frames;

		bool done=false;

		if (is16) {

			const int16_t *smp = (int16_t*)data;
			for(int i=0;i<p_frames;i++) {

				if (fp_pos >= max) {
					done=true;
					break;
				}

				int idx = (fp_pos>>FP_BITS)<<skip;
				p_buffer[i]+=int32_t(smp[idx])*vol;
				vol+=vol_inc;

				fp_pos+=increment;
			}
		} else {

			const int8_t *smp = (int8_t*)data;
			for(int i=0;i<p_frames;i++) {

				if (fp_pos >= max) {
					done=true;
					break;
				}

				int idx = (fp_pos>>FP_BITS)<<skip;
				p_buffer[i]+=(int32_t(smp[idx])<<8)*vol;
				vol+=vol_inc;
				fp_pos+=increment;
			}

		}

		playback[i].fp_pos=fp_pos;
		if (!paused && active_voices==1 && (vol_end < vol_begin || done)) {
			//xfade to something else i gues
			active_voices=2;
			playback[1].idx=randomize();
			playback[1].fp_pos=0;
			playback[1].scale=Math::random(1,1+pitch_random_scale);
		}

		if (done) {

			if (i==0 && active_voices==2) {
				playback[0]=playback[1];
				i--;
			}
			active_voices--;

		}
	}

	return true;
}


void AudioStreamGibberish::play() {
	if (active)
		stop();


	if (!phonemes.is_valid())
		return;


	List<StringName> slist;
	phonemes->get_sample_list(&slist);
	if (slist.size()==0)
		return;

	_samples.resize(slist.size());
	_rand_pool.resize(slist.size());

	int i=0;
	for(List<StringName>::Element *E=slist.front();E;E=E->next()) {

		_rand_pool[i]=i;
		_samples[i++]=phonemes->get_sample(E->get());
	}

	rand_idx=0;
	active_voices=0;
	active=true;
}

void AudioStreamGibberish::stop(){

	active=false;


}

bool AudioStreamGibberish::is_playing() const {

	return active;
}


void AudioStreamGibberish::set_paused(bool p_paused){

	paused=p_paused;
}

bool AudioStreamGibberish::is_paused(bool p_paused) const{

	return paused;
}

void AudioStreamGibberish::set_loop(bool p_enable){


}

bool AudioStreamGibberish::has_loop() const{

	return false;
}


float AudioStreamGibberish::get_length() const{

	return 0;
}


String AudioStreamGibberish::get_stream_name() const{

	return "Gibberish";
}


int AudioStreamGibberish::get_loop_count() const{

	return 0;
}


float AudioStreamGibberish::get_pos() const{

	return 0;
}

void AudioStreamGibberish::seek_pos(float p_time){


}


AudioStream::UpdateMode AudioStreamGibberish::get_update_mode() const{

	return AudioStream::UPDATE_NONE;
}

void AudioStreamGibberish::update(){


}


void AudioStreamGibberish::set_phonemes(const Ref<SampleLibrary>& p_phonemes) {

	phonemes=p_phonemes;

}

Ref<SampleLibrary> AudioStreamGibberish::get_phonemes() const {

	return phonemes;
}

void AudioStreamGibberish::set_xfade_time(float p_xfade) {

	xfade_time=p_xfade;
}

float AudioStreamGibberish::get_xfade_time() const {

	return xfade_time;
}

void AudioStreamGibberish::set_pitch_scale(float p_scale) {

	pitch_scale=p_scale;
}

float AudioStreamGibberish::get_pitch_scale() const {

	return pitch_scale;
}

void AudioStreamGibberish::set_pitch_random_scale(float p_random_scale) {

	pitch_random_scale=p_random_scale;
}

float AudioStreamGibberish::get_pitch_random_scale() const {

	return pitch_random_scale;
}

void AudioStreamGibberish::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_phonemes","phonemes"),&AudioStreamGibberish::set_phonemes);
	ClassDB::bind_method(D_METHOD("get_phonemes"),&AudioStreamGibberish::get_phonemes);

	ClassDB::bind_method(D_METHOD("set_pitch_scale","pitch_scale"),&AudioStreamGibberish::set_pitch_scale);
	ClassDB::bind_method(D_METHOD("get_pitch_scale"),&AudioStreamGibberish::get_pitch_scale);

	ClassDB::bind_method(D_METHOD("set_pitch_random_scale","pitch_random_scale"),&AudioStreamGibberish::set_pitch_random_scale);
	ClassDB::bind_method(D_METHOD("get_pitch_random_scale"),&AudioStreamGibberish::get_pitch_random_scale);

	ClassDB::bind_method(D_METHOD("set_xfade_time","sec"),&AudioStreamGibberish::set_xfade_time);
	ClassDB::bind_method(D_METHOD("get_xfade_time"),&AudioStreamGibberish::get_xfade_time);

	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"phonemes",PROPERTY_HINT_RESOURCE_TYPE,"SampleLibrary"),"set_phonemes","get_phonemes");
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"pitch_scale",PROPERTY_HINT_RANGE,"0.01,64,0.01"),"set_pitch_scale","get_pitch_scale");
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"pitch_random_scale",PROPERTY_HINT_RANGE,"0,64,0.01"),"set_pitch_random_scale","get_pitch_random_scale");
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"xfade_sec",PROPERTY_HINT_RANGE,"0.001,0.5,0.001"),"set_xfade_time","get_xfade_time");

}

AudioStreamGibberish::AudioStreamGibberish() {

	xfade_time=0.1;
	pitch_scale=1;
	pitch_random_scale=0;
	active=false;
	paused=false;
	active_voices=0;
}
#endif
