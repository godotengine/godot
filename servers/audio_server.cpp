/*************************************************************************/
/*  audio_server.cpp                                                     */
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
#include "audio_server.h"
#include "globals.h"

void AudioMixer::audio_mixer_chunk_call(int p_frames) {

	AudioServer::get_singleton()->audio_mixer_chunk_callback(p_frames);
}

AudioMixer *AudioServer::EventStream::get_mixer() const {

	return AudioServer::get_singleton()->get_mixer();
}

AudioServer *AudioServer::singleton=NULL;

AudioServer *AudioServer::get_singleton() {

	return singleton;
}

void AudioServer::sample_set_signed_data(RID p_sample, const PoolVector<float>& p_buffer) {

	SampleFormat format = sample_get_format(p_sample);

	ERR_EXPLAIN("IMA ADPCM is not supported.");
	ERR_FAIL_COND(format==SAMPLE_FORMAT_IMA_ADPCM);

	int len = p_buffer.size();
	ERR_FAIL_COND( len == 0 );

	PoolVector<uint8_t> data;
	PoolVector<uint8_t>::Write w;
	PoolVector<float>::Read r = p_buffer.read();

	switch(format) {
		case SAMPLE_FORMAT_PCM8: {
			data.resize(len);
			w=data.write();

			int8_t *samples8 = (int8_t*)w.ptr();

			for(int i=0;i<len;i++) {

				float sample = Math::floor( r[i] * (1<<8) );
				if (sample<-128)
					sample=-128;
				else if (sample>127)
					sample=127;
				samples8[i]=sample;
			}
		} break;
		case SAMPLE_FORMAT_PCM16: {
			data.resize(len*2);
			w=data.write();

			int16_t *samples16 = (int16_t*)w.ptr();

			for(int i=0;i<len;i++) {

				float sample = Math::floor( r[i] * (1<<16) );
				if (sample<-32768)
					sample=-32768;
				else if (sample>32767)
					sample=32767;
				samples16[i]=sample;
			}
		} break;
	}

	w = PoolVector<uint8_t>::Write();

	sample_set_data(p_sample,data);


}

void AudioServer::_bind_methods() {

	ClassDB::bind_method(_MD("sample_create","format","stereo","length"), &AudioServer::sample_create );
	ClassDB::bind_method(_MD("sample_set_description","sample","description"), &AudioServer::sample_set_description );
	ClassDB::bind_method(_MD("sample_get_description","sample"), &AudioServer::sample_get_description );

	ClassDB::bind_method(_MD("sample_get_format","sample"), &AudioServer::sample_get_format );
	ClassDB::bind_method(_MD("sample_is_stereo","sample"), &AudioServer::sample_is_stereo );
	ClassDB::bind_method(_MD("sample_get_length","sample"), &AudioServer::sample_get_length );

	ClassDB::bind_method(_MD("sample_set_signed_data","sample","data"), &AudioServer::sample_set_signed_data );
	ClassDB::bind_method(_MD("sample_set_data","sample","data"), &AudioServer::sample_set_data );
	ClassDB::bind_method(_MD("sample_get_data","sample"), &AudioServer::sample_get_data );

	ClassDB::bind_method(_MD("sample_set_mix_rate","sample","mix_rate"), &AudioServer::sample_set_mix_rate );
	ClassDB::bind_method(_MD("sample_get_mix_rate","sample"), &AudioServer::sample_get_mix_rate );

	ClassDB::bind_method(_MD("sample_set_loop_format","sample","loop_format"), &AudioServer::sample_set_loop_format );
	ClassDB::bind_method(_MD("sample_get_loop_format","sample"), &AudioServer::sample_get_loop_format );


	ClassDB::bind_method(_MD("sample_set_loop_begin","sample","pos"), &AudioServer::sample_set_loop_begin );
	ClassDB::bind_method(_MD("sample_get_loop_begin","sample"), &AudioServer::sample_get_loop_begin );

	ClassDB::bind_method(_MD("sample_set_loop_end","sample","pos"), &AudioServer::sample_set_loop_end );
	ClassDB::bind_method(_MD("sample_get_loop_end","sample"), &AudioServer::sample_get_loop_end );



	ClassDB::bind_method(_MD("voice_create"), &AudioServer::voice_create );
	ClassDB::bind_method(_MD("voice_play","voice","sample"), &AudioServer::voice_play );
	ClassDB::bind_method(_MD("voice_set_volume","voice","volume"), &AudioServer::voice_set_volume );
	ClassDB::bind_method(_MD("voice_set_pan","voice","pan","depth","height"), &AudioServer::voice_set_pan,DEFVAL(0),DEFVAL(0) );
	ClassDB::bind_method(_MD("voice_set_filter","voice","type","cutoff","resonance","gain"), &AudioServer::voice_set_filter,DEFVAL(0) );
	ClassDB::bind_method(_MD("voice_set_chorus","voice","chorus"), &AudioServer::voice_set_chorus );
	ClassDB::bind_method(_MD("voice_set_reverb","voice","room","reverb"), &AudioServer::voice_set_reverb );
	ClassDB::bind_method(_MD("voice_set_mix_rate","voice","rate"), &AudioServer::voice_set_mix_rate );
	ClassDB::bind_method(_MD("voice_set_positional","voice","enabled"), &AudioServer::voice_set_positional );


	ClassDB::bind_method(_MD("voice_get_volume","voice"), &AudioServer::voice_get_volume );
	ClassDB::bind_method(_MD("voice_get_pan","voice"), &AudioServer::voice_get_pan );
	ClassDB::bind_method(_MD("voice_get_pan_height","voice"), &AudioServer::voice_get_pan_height );
	ClassDB::bind_method(_MD("voice_get_pan_depth","voice"), &AudioServer::voice_get_pan_depth );
	ClassDB::bind_method(_MD("voice_get_filter_type","voice"), &AudioServer::voice_get_filter_type );
	ClassDB::bind_method(_MD("voice_get_filter_cutoff","voice"), &AudioServer::voice_get_filter_cutoff );
	ClassDB::bind_method(_MD("voice_get_filter_resonance","voice"), &AudioServer::voice_get_filter_resonance );
	ClassDB::bind_method(_MD("voice_get_chorus","voice"), &AudioServer::voice_get_chorus );
	ClassDB::bind_method(_MD("voice_get_reverb_type","voice"), &AudioServer::voice_get_reverb_type );
	ClassDB::bind_method(_MD("voice_get_reverb","voice"), &AudioServer::voice_get_reverb );
	ClassDB::bind_method(_MD("voice_get_mix_rate","voice"), &AudioServer::voice_get_mix_rate );
	ClassDB::bind_method(_MD("voice_is_positional","voice"), &AudioServer::voice_is_positional );

	ClassDB::bind_method(_MD("voice_stop","voice"), &AudioServer::voice_stop );

	ClassDB::bind_method(_MD("free_rid","rid"), &AudioServer::free );

	ClassDB::bind_method(_MD("set_stream_global_volume_scale","scale"), &AudioServer::set_stream_global_volume_scale );
	ClassDB::bind_method(_MD("get_stream_global_volume_scale"), &AudioServer::get_stream_global_volume_scale );

	ClassDB::bind_method(_MD("set_fx_global_volume_scale","scale"), &AudioServer::set_fx_global_volume_scale );
	ClassDB::bind_method(_MD("get_fx_global_volume_scale"), &AudioServer::get_fx_global_volume_scale );

	ClassDB::bind_method(_MD("set_event_voice_global_volume_scale","scale"), &AudioServer::set_event_voice_global_volume_scale );
	ClassDB::bind_method(_MD("get_event_voice_global_volume_scale"), &AudioServer::get_event_voice_global_volume_scale );

	BIND_CONSTANT( SAMPLE_FORMAT_PCM8 );
	BIND_CONSTANT( SAMPLE_FORMAT_PCM16 );
	BIND_CONSTANT( SAMPLE_FORMAT_IMA_ADPCM );

	BIND_CONSTANT( SAMPLE_LOOP_NONE );
	BIND_CONSTANT( SAMPLE_LOOP_FORWARD );
	BIND_CONSTANT( SAMPLE_LOOP_PING_PONG );

	BIND_CONSTANT( FILTER_NONE );
	BIND_CONSTANT( FILTER_LOWPASS );
	BIND_CONSTANT( FILTER_BANDPASS );
	BIND_CONSTANT( FILTER_HIPASS );
	BIND_CONSTANT( FILTER_NOTCH );
	BIND_CONSTANT( FILTER_BANDLIMIT ); ///< cutoff is LP resonace is HP

	BIND_CONSTANT( REVERB_SMALL );
	BIND_CONSTANT( REVERB_MEDIUM );
	BIND_CONSTANT( REVERB_LARGE );
	BIND_CONSTANT( REVERB_HALL );

	GLOBAL_DEF("audio/stream_buffering_ms",500);
	GLOBAL_DEF("audio/video_delay_compensation_ms",300);

}

AudioServer::AudioServer() {

	singleton=this;
}

AudioServer::~AudioServer() {


}
