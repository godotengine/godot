/*************************************************************************/
/*  sample_player.cpp                                                    */
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
#include "sample_player.h"

#include "servers/audio_server.h"


bool SamplePlayer::_set(const StringName& p_name, const Variant& p_value) {

	String name=p_name;

	if (name=="play/play") {
		if (library.is_valid()) {

			String what=p_value;
			if (what=="")
				stop_all();
			else
				play(what);

			played_back=what;
		}
	} else if (name=="config/samples")
		set_sample_library(p_value);
	else if (name=="config/polyphony")
		set_polyphony(p_value);
	else if (name.begins_with("default/")) {

		String what=name.right(8);

		if (what=="volume_db")
			set_default_volume_db(p_value);
		else if (what=="pitch_scale")
			set_default_pitch_scale(p_value);
		else if (what=="pan")
			_default.pan=p_value;
		else if (what=="depth")
			_default.depth=p_value;
		else if (what=="height")
			_default.height=p_value;
		else if (what=="filter/type")
			_default.filter_type=FilterType(p_value.operator int());
		else if (what=="filter/cutoff")
			_default.filter_cutoff=p_value;
		else if (what=="filter/resonance")
			_default.filter_resonance=p_value;
		else if (what=="filter/gain")
			_default.filter_gain=p_value;
		else if (what=="reverb_room")
			_default.reverb_room=ReverbRoomType(p_value.operator int());
		else if (what=="reverb_send")
			_default.reverb_send=p_value;
		else if (what=="chorus_send")
			_default.chorus_send=p_value;
		else
			return false;


	} else
		return false;

	return true;
}

bool SamplePlayer::_get(const StringName& p_name,Variant &r_ret) const {


	String name=p_name;

	if (name=="play/play") {
		r_ret=played_back;
	} else if (name=="config/polyphony") {
		r_ret= get_polyphony();
	} else if (name=="config/samples") {

		r_ret= get_sample_library();
	} else if (name.begins_with("default/")) {

			String what=name.right(8);

			if (what=="volume_db")
				r_ret= get_default_volume_db();
			else if (what=="pitch_scale")
				r_ret= get_default_pitch_scale();
			else if (what=="pan")
				r_ret= _default.pan;
			else if (what=="depth")
				r_ret= _default.depth;
			else if (what=="height")
				r_ret= _default.height;
			else if (what=="filter/type")
				r_ret= _default.filter_type;
			else if (what=="filter/cutoff")
				r_ret= _default.filter_cutoff;
			else if (what=="filter/resonance")
				r_ret= _default.filter_resonance;
			else if (what=="filter/gain")
				r_ret= _default.filter_gain;
			else if (what=="reverb_room")
				r_ret= _default.reverb_room;
			else if (what=="reverb_send")
				r_ret= _default.reverb_send;
			else if (what=="chorus_send")
				r_ret= _default.chorus_send;
			else
				return false;


	} else
		return false;

	return true;
}

void SamplePlayer::_get_property_list(List<PropertyInfo> *p_list) const {

	String en="";
	if (library.is_valid()) {
		List<StringName> samples;
		Ref<SampleLibrary> ncl=library;
		ncl->get_sample_list(&samples);
		for (List<StringName>::Element *E=samples.front();E;E=E->next()) {

			en+=",";
			en+=E->get();
		}
	}

	p_list->push_back( PropertyInfo( Variant::STRING, "play/play", PROPERTY_HINT_ENUM, en,PROPERTY_USAGE_EDITOR|PROPERTY_USAGE_ANIMATE_AS_TRIGGER));
	p_list->push_back( PropertyInfo( Variant::INT, "config/polyphony", PROPERTY_HINT_RANGE, "1,256,1"));
	p_list->push_back( PropertyInfo( Variant::OBJECT, "config/samples", PROPERTY_HINT_RESOURCE_TYPE, "SampleLibrary"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/volume_db", PROPERTY_HINT_RANGE, "-80,24,0.01"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/pitch_scale", PROPERTY_HINT_RANGE, "0.01,48,0.01"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/pan", PROPERTY_HINT_RANGE, "-1,1,0.01"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/depth", PROPERTY_HINT_RANGE, "-1,1,0.01"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/height", PROPERTY_HINT_RANGE, "-1,1,0.01"));
	p_list->push_back( PropertyInfo( Variant::INT, "default/filter/type", PROPERTY_HINT_ENUM, "Disabled,Lowpass,Bandpass,Highpass,Notch,Peak,BandLimit,LowShelf,HighShelf"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/filter/cutoff", PROPERTY_HINT_RANGE, "20,16384.0,0.01"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/filter/resonance", PROPERTY_HINT_RANGE, "0,4,0.01"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/filter/gain", PROPERTY_HINT_RANGE, "0,2,0.01"));
	p_list->push_back( PropertyInfo( Variant::INT, "default/reverb_room", PROPERTY_HINT_ENUM, "Small,Medium,Large,Hall"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/reverb_send", PROPERTY_HINT_RANGE, "0,1,0.01"));
	p_list->push_back( PropertyInfo( Variant::REAL, "default/chorus_send", PROPERTY_HINT_RANGE, "0,1,0.01"));


}


SamplePlayer::Voice::Voice() {

	voice=AudioServer::get_singleton()->voice_create();
	clear();
}


void SamplePlayer::Voice::clear() {

	check=0;

	mix_rate=44100;
	volume=1;
	pan=0;
	pan_depth=0;
	pan_height=0;
	filter_type=FILTER_NONE;
	filter_cutoff=0;
	filter_resonance=0;
	chorus_send=0;
	reverb_room=REVERB_HALL;
	reverb_send=0;
	active=false;

}
SamplePlayer::Voice::~Voice() {

	AudioServer::get_singleton()->free(voice);
}


void SamplePlayer::set_polyphony(int p_voice_count) {

	ERR_FAIL_COND( p_voice_count <1 || p_voice_count >0xFFFE );

	voices.resize(p_voice_count);
}

int SamplePlayer::get_polyphony() const {

	return voices.size();
}

SamplePlayer::VoiceID SamplePlayer::play(const String& p_name,bool unique) {

	if (library.is_null())
		return INVALID_VOICE_ID;
	ERR_FAIL_COND_V( !library->has_sample(p_name), INVALID_VOICE_ID );

	Ref<Sample> sample = library->get_sample(p_name);
	float vol_change = library->sample_get_volume_db(p_name);
	float pitch_change = library->sample_get_pitch_scale(p_name);

	last_check++;
	last_id = (last_id + 1) % voices.size();

	Voice&v = voices[last_id];
	v.clear();


	v.mix_rate=sample->get_mix_rate()*(_default.pitch_scale*pitch_change);
	v.sample_mix_rate=sample->get_mix_rate();
	v.check=last_check;
	v.volume=Math::db2linear(_default.volume_db+vol_change);
	v.pan=_default.pan;
	v.pan_depth=_default.depth;
	v.pan_height=_default.height;
	v.filter_type=_default.filter_type;
	v.filter_cutoff=_default.filter_cutoff;
	v.filter_resonance=_default.filter_resonance;
	v.filter_gain=_default.filter_gain;
	v.chorus_send=_default.chorus_send;
	v.reverb_room=_default.reverb_room;
	v.reverb_send=_default.reverb_send;

	AudioServer::get_singleton()->voice_play(v.voice,sample->get_rid());
	AudioServer::get_singleton()->voice_set_mix_rate(v.voice,v.mix_rate);
	AudioServer::get_singleton()->voice_set_volume(v.voice,v.volume);
	AudioServer::get_singleton()->voice_set_pan(v.voice,v.pan,v.pan_depth,v.pan_height);
	AudioServer::get_singleton()->voice_set_filter(v.voice,(AudioServer::FilterType)v.filter_type,v.filter_cutoff,v.filter_resonance,v.filter_gain);
	AudioServer::get_singleton()->voice_set_chorus(v.voice,v.chorus_send);
	AudioServer::get_singleton()->voice_set_reverb(v.voice,(AudioServer::ReverbRoomType)v.reverb_room,v.reverb_send);

	v.active=true;

	if (unique) {

		for(int i=0;i<voices.size();i++) {

			if (!voices[i].active || uint32_t(i)==last_id)
				continue;

			AudioServer::get_singleton()->voice_stop(voices[i].voice);

			voices[i].clear();
		}

	}

	return last_id | (last_check<<16);
}

void SamplePlayer::stop_all() {


	for(int i=0;i<voices.size();i++) {

		if (!voices[i].active)
			continue;

		AudioServer::get_singleton()->voice_stop(voices[i].voice);
		voices[i].clear();
	}

}

#define _GET_VOICE\
	uint32_t voice=p_voice&0xFFFF;\
	ERR_FAIL_COND(voice >= (uint32_t)voices.size());\
	Voice &v=voices[voice];\
	if (v.check!=uint32_t(p_voice>>16))\
		return;\
	ERR_FAIL_COND(!v.active);

void SamplePlayer::stop(VoiceID p_voice) {

	_GET_VOICE

	AudioServer::get_singleton()->voice_stop(v.voice);
	v.active=false;

}

void SamplePlayer::set_mix_rate(VoiceID p_voice, int p_mix_rate) {

	_GET_VOICE

	v.mix_rate=p_mix_rate;
	AudioServer::get_singleton()->voice_set_mix_rate(v.voice,v.mix_rate);

}
void SamplePlayer::set_pitch_scale(VoiceID p_voice, float p_pitch_scale) {

	_GET_VOICE

	v.mix_rate=v.sample_mix_rate*p_pitch_scale;
	AudioServer::get_singleton()->voice_set_mix_rate(v.voice,v.mix_rate);

}
void SamplePlayer::set_volume(VoiceID p_voice, float p_volume) {


	_GET_VOICE
	v.volume=p_volume;
	AudioServer::get_singleton()->voice_set_volume(v.voice,v.volume);

}

void SamplePlayer::set_volume_db(VoiceID p_voice, float p_db) {

	//@TODO handle 0 volume as -80db or something
	_GET_VOICE
	v.volume=Math::db2linear(p_db);
	AudioServer::get_singleton()->voice_set_volume(v.voice,v.volume);

}

void SamplePlayer::set_pan(VoiceID p_voice, float p_pan,float p_pan_depth,float p_pan_height) {

	_GET_VOICE
	v.pan=p_pan;
	v.pan_depth=p_pan_depth;
	v.pan_height=p_pan_height;

	AudioServer::get_singleton()->voice_set_pan(v.voice,v.pan,v.pan_depth,v.pan_height);

}

void SamplePlayer::set_filter(VoiceID p_voice,FilterType p_filter,float p_cutoff,float p_resonance,float p_gain) {

	_GET_VOICE
	v.filter_type=p_filter;
	v.filter_cutoff=p_cutoff;
	v.filter_resonance=p_resonance;
	v.filter_gain=p_gain;

	AudioServer::get_singleton()->voice_set_filter(v.voice,(AudioServer::FilterType)p_filter,p_cutoff,p_resonance);

}
void SamplePlayer::set_chorus(VoiceID p_voice,float p_send) {

	_GET_VOICE
	v.chorus_send=p_send;

	AudioServer::get_singleton()->voice_set_chorus(v.voice,p_send);

}
void SamplePlayer::set_reverb(VoiceID p_voice,ReverbRoomType p_room,float p_send) {

	_GET_VOICE
	v.reverb_room=p_room;
	v.reverb_send=p_send;

	AudioServer::get_singleton()->voice_set_reverb(v.voice,(AudioServer::ReverbRoomType)p_room,p_send);

}

#define _GET_VOICE_V(m_ret)\
	uint32_t voice=p_voice&0xFFFF;\
	ERR_FAIL_COND_V(voice >= (uint32_t)voices.size(),m_ret);\
	const Voice &v=voices[voice];\
	if (v.check!=(p_voice>>16))\
		return m_ret;\
	ERR_FAIL_COND_V(!v.active,m_ret);


int SamplePlayer::get_mix_rate(VoiceID p_voice) const {

	_GET_VOICE_V(0);

	return v.mix_rate;
}
float SamplePlayer::get_pitch_scale(VoiceID p_voice) const {

	_GET_VOICE_V(0);
	return v.sample_mix_rate/(float)v.mix_rate;
}
float SamplePlayer::get_volume(VoiceID p_voice) const {

	_GET_VOICE_V(0);
	return v.volume;
}


float SamplePlayer::get_volume_db(VoiceID p_voice) const {

	_GET_VOICE_V(0);
	return Math::linear2db(v.volume);
}

float SamplePlayer::get_pan(VoiceID p_voice) const {

	_GET_VOICE_V(0);
	return v.pan;
}
float SamplePlayer::get_pan_depth(VoiceID p_voice) const {


	_GET_VOICE_V(0);
	return v.pan_depth;
}
float SamplePlayer::get_pan_height(VoiceID p_voice) const {

	_GET_VOICE_V(0);

	return v.pan_height;
}
SamplePlayer::FilterType SamplePlayer::get_filter_type(VoiceID p_voice) const {

	_GET_VOICE_V(FILTER_NONE);

	return v.filter_type;
}
float SamplePlayer::get_filter_cutoff(VoiceID p_voice) const {

	_GET_VOICE_V(0);

	return v.filter_cutoff;
}
float SamplePlayer::get_filter_resonance(VoiceID p_voice) const {

	_GET_VOICE_V(0);

	return v.filter_resonance;
}

float SamplePlayer::get_filter_gain(VoiceID p_voice) const {

	_GET_VOICE_V(0);

	return v.filter_gain;
}
float SamplePlayer::get_chorus(VoiceID p_voice) const {

	_GET_VOICE_V(0);

	return v.chorus_send;
}
SamplePlayer::ReverbRoomType SamplePlayer::get_reverb_room(VoiceID p_voice) const {

	_GET_VOICE_V(REVERB_SMALL);

	return v.reverb_room;
}

float SamplePlayer::get_reverb(VoiceID p_voice) const {

	_GET_VOICE_V(0);

	return v.reverb_send;
}

bool SamplePlayer::is_voice_active(VoiceID p_voice) const {

	_GET_VOICE_V(false);
	return v.active && AudioServer::get_singleton()->voice_is_active(v.voice);

}
bool SamplePlayer::is_active() const {

	for(int i=0;i<voices.size();i++) {

		if (voices[i].active && AudioServer::get_singleton()->voice_is_active(voices[i].voice))
			return true;


	}

	return false;
}



void SamplePlayer::set_sample_library(const Ref<SampleLibrary>& p_library) {

	library=p_library;
	_change_notify();
}

Ref<SampleLibrary> SamplePlayer::get_sample_library() const {

	return library;
}



void SamplePlayer::set_default_pitch_scale(float p_pitch_scale) {

	_default.pitch_scale=p_pitch_scale;
}
void SamplePlayer::set_default_volume(float p_volume) {

	_default.volume_db=Math::linear2db(p_volume);
}
void SamplePlayer::set_default_volume_db(float p_db) {

	_default.volume_db=p_db;
}
void SamplePlayer::set_default_pan(float p_pan,float p_pan_depth,float p_pan_height) {

	_default.pan=p_pan;
	_default.depth=p_pan_depth;
	_default.height=p_pan_height;

}
void SamplePlayer::set_default_filter(FilterType p_filter,float p_cutoff,float p_resonance,float p_gain) {

	_default.filter_type=p_filter;
	_default.filter_cutoff=p_cutoff;
	_default.filter_resonance=p_resonance;
	_default.filter_gain=p_gain;
}
void SamplePlayer::set_default_chorus(float p_send) {

	_default.chorus_send=p_send;

}
void SamplePlayer::set_default_reverb(ReverbRoomType p_room,float p_send) {

	_default.reverb_room=p_room;
	_default.reverb_send=p_send;
}

float SamplePlayer::get_default_volume() const {

	return Math::db2linear(_default.volume_db);
}
float SamplePlayer::get_default_volume_db() const {

	return _default.volume_db;
}
float SamplePlayer::get_default_pitch_scale() const {

	return _default.pitch_scale;
}


float SamplePlayer::get_default_pan() const {

	return _default.pan;
}
float SamplePlayer::get_default_pan_depth() const {

	return _default.depth;
}
float SamplePlayer::get_default_pan_height() const {

	return _default.height;
}
SamplePlayer::FilterType SamplePlayer::get_default_filter_type() const {

	return _default.filter_type;
}
float SamplePlayer::get_default_filter_cutoff() const {

	return _default.filter_cutoff;
}
float SamplePlayer::get_default_filter_resonance() const {

	return _default.filter_resonance;
}
float SamplePlayer::get_default_filter_gain() const {

	return _default.filter_gain;
}
float SamplePlayer::get_default_chorus() const {

	return _default.chorus_send;
}
SamplePlayer::ReverbRoomType SamplePlayer::get_default_reverb_room() const {

	return _default.reverb_room;
}
float SamplePlayer::get_default_reverb() const {

	return _default.reverb_send;
}

String SamplePlayer::get_configuration_warning() const {

	if (library.is_null()) {
		return TTR("A SampleLibrary resource must be created or set in the 'samples' property in order for SamplePlayer to play sound.");
	}

	return String();
}

void SamplePlayer::_bind_methods() {

	ClassDB::bind_method(_MD("set_sample_library","library:SampleLibrary"),&SamplePlayer::set_sample_library );
	ClassDB::bind_method(_MD("get_sample_library:SampleLibrary"),&SamplePlayer::get_sample_library );

	ClassDB::bind_method(_MD("set_polyphony","max_voices"),&SamplePlayer::set_polyphony );
	ClassDB::bind_method(_MD("get_polyphony"),&SamplePlayer::get_polyphony );

	ClassDB::bind_method(_MD("play","name","unique"),&SamplePlayer::play, DEFVAL(false) );
	ClassDB::bind_method(_MD("stop","voice"),&SamplePlayer::stop );
	ClassDB::bind_method(_MD("stop_all"),&SamplePlayer::stop_all );

	ClassDB::bind_method(_MD("set_mix_rate","voice","hz"),&SamplePlayer::set_mix_rate );
	ClassDB::bind_method(_MD("set_pitch_scale","voice","ratio"),&SamplePlayer::set_pitch_scale );
	ClassDB::bind_method(_MD("set_volume","voice","volume"),&SamplePlayer::set_volume );
	ClassDB::bind_method(_MD("set_volume_db","voice","db"),&SamplePlayer::set_volume_db );
	ClassDB::bind_method(_MD("set_pan","voice","pan","depth","height"),&SamplePlayer::set_pan,DEFVAL(0),DEFVAL(0) );
	ClassDB::bind_method(_MD("set_filter","voice","type","cutoff_hz","resonance","gain"),&SamplePlayer::set_filter,DEFVAL(0) );
	ClassDB::bind_method(_MD("set_chorus","voice","send"),&SamplePlayer::set_chorus );
	ClassDB::bind_method(_MD("set_reverb","voice","room_type","send"),&SamplePlayer::set_reverb );

	ClassDB::bind_method(_MD("get_mix_rate","voice"),&SamplePlayer::get_mix_rate );
	ClassDB::bind_method(_MD("get_pitch_scale","voice"),&SamplePlayer::get_pitch_scale );
	ClassDB::bind_method(_MD("get_volume","voice"),&SamplePlayer::get_volume );
	ClassDB::bind_method(_MD("get_volume_db","voice"),&SamplePlayer::get_volume_db );
	ClassDB::bind_method(_MD("get_pan","voice"),&SamplePlayer::get_pan );
	ClassDB::bind_method(_MD("get_pan_depth","voice"),&SamplePlayer::get_pan_depth );
	ClassDB::bind_method(_MD("get_pan_height","voice"),&SamplePlayer::get_pan_height );
	ClassDB::bind_method(_MD("get_filter_type","voice"),&SamplePlayer::get_filter_type );
	ClassDB::bind_method(_MD("get_filter_cutoff","voice"),&SamplePlayer::get_filter_cutoff );
	ClassDB::bind_method(_MD("get_filter_resonance","voice"),&SamplePlayer::get_filter_resonance );
	ClassDB::bind_method(_MD("get_filter_gain","voice"),&SamplePlayer::get_filter_gain );
	ClassDB::bind_method(_MD("get_chorus","voice"),&SamplePlayer::get_chorus );
	ClassDB::bind_method(_MD("get_reverb_room","voice"),&SamplePlayer::get_reverb_room );
	ClassDB::bind_method(_MD("get_reverb","voice"),&SamplePlayer::get_reverb );

	ClassDB::bind_method(_MD("set_default_pitch_scale","ratio"),&SamplePlayer::set_default_pitch_scale );
	ClassDB::bind_method(_MD("set_default_volume","volume"),&SamplePlayer::set_default_volume );
	ClassDB::bind_method(_MD("set_default_volume_db","db"),&SamplePlayer::set_default_volume_db );
	ClassDB::bind_method(_MD("set_default_pan","pan","depth","height"),&SamplePlayer::set_default_pan,DEFVAL(0),DEFVAL(0) );
	ClassDB::bind_method(_MD("set_default_filter","type","cutoff_hz","resonance","gain"),&SamplePlayer::set_default_filter,DEFVAL(0) );
	ClassDB::bind_method(_MD("set_default_chorus","send"),&SamplePlayer::set_default_chorus );
	ClassDB::bind_method(_MD("set_default_reverb","room_type","send"),&SamplePlayer::set_default_reverb );

	ClassDB::bind_method(_MD("get_default_pitch_scale"),&SamplePlayer::get_default_pitch_scale );
	ClassDB::bind_method(_MD("get_default_volume"),&SamplePlayer::get_default_volume );
	ClassDB::bind_method(_MD("get_default_volume_db"),&SamplePlayer::get_default_volume_db );
	ClassDB::bind_method(_MD("get_default_pan"),&SamplePlayer::get_default_pan );
	ClassDB::bind_method(_MD("get_default_pan_depth"),&SamplePlayer::get_default_pan_depth );
	ClassDB::bind_method(_MD("get_default_pan_height"),&SamplePlayer::get_default_pan_height );
	ClassDB::bind_method(_MD("get_default_filter_type"),&SamplePlayer::get_default_filter_type );
	ClassDB::bind_method(_MD("get_default_filter_cutoff"),&SamplePlayer::get_default_filter_cutoff );
	ClassDB::bind_method(_MD("get_default_filter_resonance"),&SamplePlayer::get_default_filter_resonance );
	ClassDB::bind_method(_MD("get_default_filter_gain"),&SamplePlayer::get_default_filter_gain );
	ClassDB::bind_method(_MD("get_default_chorus"),&SamplePlayer::get_default_chorus );
	ClassDB::bind_method(_MD("get_default_reverb_room"),&SamplePlayer::get_default_reverb_room );
	ClassDB::bind_method(_MD("get_default_reverb"),&SamplePlayer::get_default_reverb );

	ClassDB::bind_method(_MD("is_active"),&SamplePlayer::is_active );
	ClassDB::bind_method(_MD("is_voice_active","voice"),&SamplePlayer::is_voice_active );

	BIND_CONSTANT( FILTER_NONE);
	BIND_CONSTANT( FILTER_LOWPASS);
	BIND_CONSTANT( FILTER_BANDPASS);
	BIND_CONSTANT( FILTER_HIPASS);
	BIND_CONSTANT( FILTER_NOTCH);
	BIND_CONSTANT( FILTER_PEAK);
	BIND_CONSTANT( FILTER_BANDLIMIT); ///< cutoff is LP resonace is HP
	BIND_CONSTANT( FILTER_LOW_SHELF);
	BIND_CONSTANT( FILTER_HIGH_SHELF);

	BIND_CONSTANT( REVERB_SMALL );
	BIND_CONSTANT( REVERB_MEDIUM  );
	BIND_CONSTANT( REVERB_LARGE  );
	BIND_CONSTANT( REVERB_HALL );

	BIND_CONSTANT( INVALID_VOICE_ID );

}


SamplePlayer::SamplePlayer() {

	voices.resize(1);

	_default.pitch_scale=1;
	_default.volume_db=0;
	_default.pan=0;
	_default.depth=0;
	_default.height=0;
	_default.filter_type=FILTER_NONE;
	_default.filter_cutoff=5000;
	_default.filter_resonance=1;
	_default.filter_gain=1;
	_default.chorus_send=0;
	_default.reverb_room=REVERB_LARGE;
	_default.reverb_send=0;
	last_id=0;
	last_check=0;


}

SamplePlayer::~SamplePlayer() {


}
