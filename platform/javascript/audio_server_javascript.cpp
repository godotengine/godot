/*************************************************************************/
/*  audio_server_javascript.cpp                                          */
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
#include "audio_server_javascript.h"
#if 0
#include "emscripten.h"

AudioMixer *AudioServerJavascript::get_mixer() {

	return NULL;
}

void AudioServerJavascript::audio_mixer_chunk_callback(int p_frames){


}


RID AudioServerJavascript::sample_create(SampleFormat p_format, bool p_stereo, int p_length) {

	Sample *sample = memnew( Sample );
	sample->format=p_format;
	sample->stereo=p_stereo;
	sample->length=p_length;
	sample->loop_begin=0;
	sample->loop_end=p_length;
	sample->loop_format=SAMPLE_LOOP_NONE;
	sample->mix_rate=44100;
	sample->index=-1;

	return sample_owner.make_rid(sample);

}

void AudioServerJavascript::sample_set_description(RID p_sample, const String& p_description){


}
String AudioServerJavascript::sample_get_description(RID p_sample) const{

	return String();
}

AudioServerJavascript::SampleFormat AudioServerJavascript::sample_get_format(RID p_sample) const{

	return SAMPLE_FORMAT_PCM8;
}
bool AudioServerJavascript::sample_is_stereo(RID p_sample) const{

	const Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!sample,false);
	return sample->stereo;

}
int AudioServerJavascript::sample_get_length(RID p_sample) const{
	const Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!sample,0);
	return sample->length;
}
const void* AudioServerJavascript::sample_get_data_ptr(RID p_sample) const{

	return NULL;
}

void AudioServerJavascript::sample_set_data(RID p_sample, const PoolVector<uint8_t>& p_buffer){

	Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND(!sample);
	int chans = sample->stereo?2:1;

	Vector<float> buffer;
	buffer.resize(sample->length*chans);
	PoolVector<uint8_t>::Read r=p_buffer.read();
	if (sample->format==SAMPLE_FORMAT_PCM8) {
		const int8_t*ptr = (const int8_t*)r.ptr();
		for(int i=0;i<sample->length*chans;i++) {
			buffer[i]=ptr[i]/128.0;
		}
	} else if (sample->format==SAMPLE_FORMAT_PCM16){
		const int16_t*ptr = (const int16_t*)r.ptr();
		for(int i=0;i<sample->length*chans;i++) {
			buffer[i]=ptr[i]/32768.0;
		}
	} else {
		ERR_EXPLAIN("Unsupported for now");
		ERR_FAIL();
	}

	sample->tmp_data=buffer;



}
PoolVector<uint8_t> AudioServerJavascript::sample_get_data(RID p_sample) const{


	return PoolVector<uint8_t>();
}

void AudioServerJavascript::sample_set_mix_rate(RID p_sample,int p_rate){
	Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND(!sample);
	sample->mix_rate=p_rate;

}

int AudioServerJavascript::sample_get_mix_rate(RID p_sample) const{
	const Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!sample,0);
	return sample->mix_rate;
}


void AudioServerJavascript::sample_set_loop_format(RID p_sample,SampleLoopFormat p_format){

	Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND(!sample);
	sample->loop_format=p_format;

}

AudioServerJavascript::SampleLoopFormat AudioServerJavascript::sample_get_loop_format(RID p_sample) const {

	const Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!sample,SAMPLE_LOOP_NONE);
	return sample->loop_format;
}

void AudioServerJavascript::sample_set_loop_begin(RID p_sample,int p_pos){

	Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND(!sample);
	sample->loop_begin=p_pos;

}
int AudioServerJavascript::sample_get_loop_begin(RID p_sample) const{

	const Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!sample,0);
	return sample->loop_begin;
}

void AudioServerJavascript::sample_set_loop_end(RID p_sample,int p_pos){

	Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND(!sample);
	sample->loop_end=p_pos;

}
int AudioServerJavascript::sample_get_loop_end(RID p_sample) const{

	const Sample *sample = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!sample,0);
	return sample->loop_end;
}


/* VOICE API */

RID AudioServerJavascript::voice_create(){

	Voice *voice = memnew( Voice );

	voice->index=voice_base;
	voice->volume=1.0;
	voice->pan=0.0;
	voice->pan_depth=.0;
	voice->pan_height=0.0;
	voice->chorus=0;
	voice->reverb_type=REVERB_SMALL;
	voice->reverb=0;
	voice->mix_rate=-1;
	voice->positional=false;
	voice->active=false;

	/* clang-format off */
	EM_ASM_( {
		_as_voices[$0] = null;
		_as_voice_gain[$0] = _as_audioctx.createGain();
		_as_voice_pan[$0] = _as_audioctx.createStereoPanner();
		_as_voice_gain[$0].connect(_as_voice_pan[$0]);
		_as_voice_pan[$0].connect(_as_audioctx.destination);
	}, voice_base);
	/* clang-format on */

	voice_base++;

	return voice_owner.make_rid( voice );
}

void AudioServerJavascript::voice_play(RID p_voice, RID p_sample){

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND(!voice);
	Sample *sample=sample_owner.get(p_sample);
	ERR_FAIL_COND(!sample);

	// due to how webaudio works, sample cration is deferred until used
	// sorry! WebAudio absolutely sucks


	if (sample->index==-1) {
		//create sample if not created
		ERR_FAIL_COND(sample->tmp_data.size()==0);
		sample->index=sample_base;
		/* clang-format off */
		EM_ASM_({
			_as_samples[$0] = _as_audioctx.createBuffer($1, $2, $3);
		}, sample_base, sample->stereo ? 2 : 1, sample->length, sample->mix_rate);
		/* clang-format on */

		sample_base++;
		int chans = sample->stereo?2:1;


		for(int i=0;i<chans;i++) {
			/* clang-format off */
			EM_ASM_({
				_as_edited_buffer = _as_samples[$0].getChannelData($1);
			}, sample->index, i);
			/* clang-format on */

			for(int j=0;j<sample->length;j++) {
				/* clang-format off */
				EM_ASM_({
					_as_edited_buffer[$0] = $1;
				}, j, sample->tmp_data[j * chans + i]);
				/* clang-format on */
			}
		}

		sample->tmp_data.clear();
	}


	voice->sample_mix_rate=sample->mix_rate;
	if (voice->mix_rate==-1) {
		voice->mix_rate=voice->sample_mix_rate;
	}

	float freq_diff = Math::log(float(voice->mix_rate)/float(voice->sample_mix_rate))/Math::log(2.0);
	int detune = int(freq_diff*1200.0);

	/* clang-format off */
	EM_ASM_({
		if (_as_voices[$0] !== null) {
			_as_voices[$0].stop(); //stop and byebye
		}
		_as_voices[$0] = _as_audioctx.createBufferSource();
		_as_voices[$0].connect(_as_voice_gain[$0]);
		_as_voices[$0].buffer = _as_samples[$1];
		_as_voices[$0].loopStart.value = $1;
		_as_voices[$0].loopEnd.value = $2;
		_as_voices[$0].loop.value = $3;
		_as_voices[$0].detune.value = $6;
		_as_voice_pan[$0].pan.value = $4;
		_as_voice_gain[$0].gain.value = $5;
		_as_voices[$0].start();
		_as_voices[$0].onended = function() {
			_as_voices[$0].disconnect(_as_voice_gain[$0]);
			_as_voices[$0] = null;
		}
	}, voice->index, sample->index, sample->mix_rate * sample->loop_begin, sample->mix_rate * sample->loop_end, sample->loop_format != SAMPLE_LOOP_NONE, voice->pan, voice->volume * fx_volume_scale, detune);
	/* clang-format on */

	voice->active=true;
}

void AudioServerJavascript::voice_set_volume(RID p_voice, float p_volume){

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND(!voice);

	voice->volume=p_volume;

	if (voice->active) {
		/* clang-format off */
		EM_ASM_({
			_as_voice_gain[$0].gain.value = $1;
		}, voice->index, voice->volume * fx_volume_scale);
		/* clang-format on */
	}

}
void AudioServerJavascript::voice_set_pan(RID p_voice, float p_pan, float p_depth,float height){

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND(!voice);

	voice->pan=p_pan;
	voice->pan_depth=p_depth;
	voice->pan_height=height;

	if (voice->active) {
		/* clang-format off */
		EM_ASM_({
			_as_voice_pan[$0].pan.value = $1;
		}, voice->index, voice->pan);
		/* clang-format on */
	}
}
void AudioServerJavascript::voice_set_filter(RID p_voice, FilterType p_type, float p_cutoff, float p_resonance, float p_gain){

}
void AudioServerJavascript::voice_set_chorus(RID p_voice, float p_chorus ){

}
void AudioServerJavascript::voice_set_reverb(RID p_voice, ReverbRoomType p_room_type, float p_reverb){

}
void AudioServerJavascript::voice_set_mix_rate(RID p_voice, int p_mix_rate){

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND(!voice);

	voice->mix_rate=p_mix_rate;

	if (voice->active) {

		float freq_diff = Math::log(float(voice->mix_rate)/float(voice->sample_mix_rate))/Math::log(2.0);
		int detune = int(freq_diff*1200.0);
		/* clang-format off */
		EM_ASM_({
			_as_voices[$0].detune.value = $1;
		}, voice->index, detune);
		/* clang-format on */
	}
}
void AudioServerJavascript::voice_set_positional(RID p_voice, bool p_positional){

}

float AudioServerJavascript::voice_get_volume(RID p_voice) const{

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND_V(!voice,0);

	return voice->volume;
}
float AudioServerJavascript::voice_get_pan(RID p_voice) const{

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND_V(!voice,0);

	return voice->pan;
}
float AudioServerJavascript::voice_get_pan_depth(RID p_voice) const{
	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND_V(!voice,0);

	return voice->pan_depth;
}
float AudioServerJavascript::voice_get_pan_height(RID p_voice) const{

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND_V(!voice,0);

	return voice->pan_height;
}
AudioServerJavascript::FilterType AudioServerJavascript::voice_get_filter_type(RID p_voice) const{

	return FILTER_NONE;
}
float AudioServerJavascript::voice_get_filter_cutoff(RID p_voice) const{

	return 0;
}
float AudioServerJavascript::voice_get_filter_resonance(RID p_voice) const{

	return 0;
}
float AudioServerJavascript::voice_get_chorus(RID p_voice) const{

	return 0;
}
AudioServerJavascript::ReverbRoomType AudioServerJavascript::voice_get_reverb_type(RID p_voice) const{

	return REVERB_SMALL;
}
float AudioServerJavascript::voice_get_reverb(RID p_voice) const{

	return 0;
}

int AudioServerJavascript::voice_get_mix_rate(RID p_voice) const{

	return 44100;
}

bool AudioServerJavascript::voice_is_positional(RID p_voice) const{

	return false;
}

void AudioServerJavascript::voice_stop(RID p_voice){

	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND(!voice);

	if (voice->active) {
		/* clang-format off */
		EM_ASM_({
			if (_as_voices[$0] !== null) {
				_as_voices[$0].stop();
				_as_voices[$0].disconnect(_as_voice_gain[$0]);
				_as_voices[$0] = null;
			}
		}, voice->index);
		/* clang-format on */

		voice->active=false;
	}


}
bool AudioServerJavascript::voice_is_active(RID p_voice) const{
	Voice* voice=voice_owner.get(p_voice);
	ERR_FAIL_COND_V(!voice,false);

	return voice->active;
}

/* STREAM API */

RID AudioServerJavascript::audio_stream_create(AudioStream *p_stream) {


	Stream *s = memnew(Stream);
	s->audio_stream=p_stream;
	s->event_stream=NULL;
	s->active=false;
	s->E=NULL;
	s->volume_scale=1.0;
	p_stream->set_mix_rate(webaudio_mix_rate);

	return stream_owner.make_rid(s);
}

RID AudioServerJavascript::event_stream_create(EventStream *p_stream) {


	Stream *s = memnew(Stream);
	s->audio_stream=NULL;
	s->event_stream=p_stream;
	s->active=false;
	s->E=NULL;
	s->volume_scale=1.0;
	//p_stream->set_mix_rate(AudioDriverJavascript::get_singleton()->get_mix_rate());

	return stream_owner.make_rid(s);


}


void AudioServerJavascript::stream_set_active(RID p_stream, bool p_active) {


	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND(!s);

	if (s->active==p_active)
		return;

	s->active=p_active;
	if (p_active)
		s->E=active_audio_streams.push_back(s);
	else {
		active_audio_streams.erase(s->E);
		s->E=NULL;
	}
}

bool AudioServerJavascript::stream_is_active(RID p_stream) const {

	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND_V(!s,false);
	return s->active;
}

void AudioServerJavascript::stream_set_volume_scale(RID p_stream, float p_scale) {

	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND(!s);
	s->volume_scale=p_scale;

}

float AudioServerJavascript::stream_set_volume_scale(RID p_stream) const {

	Stream *s = stream_owner.get(p_stream);
	ERR_FAIL_COND_V(!s,0);
	return s->volume_scale;

}


/* Audio Physics API */

void AudioServerJavascript::free(RID p_id){

	if (voice_owner.owns(p_id)) {
		Voice* voice=voice_owner.get(p_id);
		ERR_FAIL_COND(!voice);

		if (voice->active) {
			/* clang-format off */
			EM_ASM_({
				 if (_as_voices[$0] !== null) {
					_as_voices[$0].stop();
					_as_voices[$0].disconnect(_as_voice_gain[$0]);
				 }
			}, voice->index);
			/* clang-format on */
		}

		/* clang-format off */
		EM_ASM_({
			delete _as_voices[$0];
			_as_voice_gain[$0].disconnect(_as_voice_pan[$0]);
			delete _as_voice_gain[$0];
			_as_voice_pan[$0].disconnect(_as_audioctx.destination);
			delete _as_voice_pan[$0];
		}, voice->index);
		/* clang-format on */

		voice_owner.free(p_id);
		memdelete(voice);

	} else if (sample_owner.owns(p_id)) {

		Sample *sample = sample_owner.get(p_id);
		ERR_FAIL_COND(!sample);

		/* clang-format off */
		EM_ASM_({
			delete _as_samples[$0];
		}, sample->index);
		/* clang-format on */

		sample_owner.free(p_id);
		memdelete(sample);

	} else if (stream_owner.owns(p_id)) {


		Stream *s=stream_owner.get(p_id);

		if (s->active) {
			stream_set_active(p_id,false);
		}

		memdelete(s);
		stream_owner.free(p_id);
	}
}

extern "C" {


void audio_server_mix_function(int p_frames) {

	//print_line("MIXI! "+itos(p_frames));
	static_cast<AudioServerJavascript*>(AudioServerJavascript::get_singleton())->mix_to_js(p_frames);
}

}

void AudioServerJavascript::mix_to_js(int p_frames) {


	//process in chunks to make sure to never process more than INTERNAL_BUFFER_SIZE
	int todo=p_frames;
	int offset=0;

	while(todo) {

		int tomix=MIN(todo,INTERNAL_BUFFER_SIZE);
		driver_process_chunk(tomix);

		/* clang-format off */
		EM_ASM_({
			var data = HEAPF32.subarray($0 / 4, $0 / 4 + $2 * 2);

			for (var channel = 0; channel < _as_output_buffer.numberOfChannels; channel++) {
				var outputData = _as_output_buffer.getChannelData(channel);
				// Loop through samples
				for (var sample = 0; sample < $2; sample++) {
					// make output equal to the same as the input
					outputData[sample + $1] = data[sample * 2 + channel];
				}
			}
		}, internal_buffer, offset, tomix);
		/* clang-format on */

		todo-=tomix;
		offset+=tomix;
	}
}

void AudioServerJavascript::init(){

	/*
	// clang-format off
	EM_ASM(
		console.log('server is ' + audio_server);
	);
	// clang-format on
	*/


	//int latency = GLOBAL_DEF("javascript/audio_latency",16384);

	internal_buffer_channels=2;
	internal_buffer = memnew_arr(float,INTERNAL_BUFFER_SIZE*internal_buffer_channels);
	stream_buffer = memnew_arr(int32_t,INTERNAL_BUFFER_SIZE*4); //max 4 channels

	stream_volume=0.3;

	int buffer_latency=16384;

	/* clang-format off */
	EM_ASM_( {
		_as_script_node = _as_audioctx.createScriptProcessor($0, 0, 2);
		_as_script_node.connect(_as_audioctx.destination);
		console.log(_as_script_node.bufferSize);

		_as_script_node.onaudioprocess = function(audioProcessingEvent) {
		// The output buffer contains the samples that will be modified and played
			_as_output_buffer = audioProcessingEvent.outputBuffer;
			audio_server_mix_function(_as_output_buffer.getChannelData(0).length);
		}
	}, buffer_latency);
	/* clang-format on */


}

void AudioServerJavascript::finish(){

}
void AudioServerJavascript::update(){

	for(List<Stream*>::Element *E=active_audio_streams.front();E;) { //stream might be removed durnig this callback

		List<Stream*>::Element *N=E->next();

		if (E->get()->audio_stream)
			E->get()->audio_stream->update();

		E=N;
	}
}

/* MISC config */

void AudioServerJavascript::lock(){

}
void AudioServerJavascript::unlock(){

}
int AudioServerJavascript::get_default_channel_count() const{

	return 1;
}
int AudioServerJavascript::get_default_mix_rate() const{

	return 44100;
}

void AudioServerJavascript::set_stream_global_volume_scale(float p_volume){

	stream_volume_scale=p_volume;
}
void AudioServerJavascript::set_fx_global_volume_scale(float p_volume){

	fx_volume_scale=p_volume;
}
void AudioServerJavascript::set_event_voice_global_volume_scale(float p_volume){

}

float AudioServerJavascript::get_stream_global_volume_scale() const{
	return 1;
}
float AudioServerJavascript::get_fx_global_volume_scale() const{

	return 1;
}
float AudioServerJavascript::get_event_voice_global_volume_scale() const{

	return 1;
}

uint32_t AudioServerJavascript::read_output_peak() const{

	return 0;
}

AudioServerJavascript *AudioServerJavascript::singleton=NULL;

AudioServer *AudioServerJavascript::get_singleton() {
	return singleton;
}

double AudioServerJavascript::get_mix_time() const{

	return 0;
}
double AudioServerJavascript::get_output_delay() const {

	return 0;
}


void AudioServerJavascript::driver_process_chunk(int p_frames) {



	int samples=p_frames*internal_buffer_channels;

	for(int i=0;i<samples;i++) {
		internal_buffer[i]=0;
	}


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

#define STRSCALE(m_val) ((((m_val >> STREAM_SCALE_BITS) * stream_vol_scale) >> 8) / 8388608.0)
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


		}

#undef STRSCALE
	}
}


/*void AudioServerSW::driver_process(int p_frames,int32_t *p_buffer) {


	_output_delay=p_frames/double(AudioDriverSW::get_singleton()->get_mix_rate());
	//process in chunks to make sure to never process more than INTERNAL_BUFFER_SIZE
	int todo=p_frames;
	while(todo) {

		int tomix=MIN(todo,INTERNAL_BUFFER_SIZE);
		driver_process_chunk(tomix,p_buffer);
		p_buffer+=tomix;
		todo-=tomix;
	}


}*/

AudioServerJavascript::AudioServerJavascript() {

	singleton=this;
	sample_base=1;
	voice_base=1;
	/* clang-format off */
	EM_ASM(
		_as_samples = {};
		_as_voices = {};
		_as_voice_pan = {};
		_as_voice_gain = {};

		_as_audioctx = new (window.AudioContext || window.webkitAudioContext)();

		audio_server_mix_function = Module.cwrap('audio_server_mix_function', 'void', ['number']);
	);
	/* clang-format on */

	/* clang-format off */
	webaudio_mix_rate = EM_ASM_INT_V(
		return _as_audioctx.sampleRate;
	);
	/* clang-format on */
	print_line("WEBAUDIO MIX RATE: "+itos(webaudio_mix_rate));
	event_voice_scale=1.0;
	fx_volume_scale=1.0;
	stream_volume_scale=1.0;

}
#endif
