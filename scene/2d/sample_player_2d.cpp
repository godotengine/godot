/*************************************************************************/
/*  sample_player_2d.cpp                                                 */
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
#include "sample_player_2d.h"

#include "servers/audio_server.h"
#include "servers/audio_server.h"
#include "servers/spatial_sound_server.h"


bool SamplePlayer2D::_set(const StringName& p_name, const Variant& p_value) {

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
	} else
		return false;

	return true;
}

bool SamplePlayer2D::_get(const StringName& p_name,Variant &r_ret) const {


	String name=p_name;

	if (name=="play/play") {
		r_ret=played_back;
	} else
		return false;

	return true;
}

void SamplePlayer2D::_get_property_list(List<PropertyInfo> *p_list) const {

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
}

void SamplePlayer2D::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {

			SpatialSound2DServer::get_singleton()->source_set_polyphony(get_source_rid(),polyphony);


		} break;
	}

}

void SamplePlayer2D::set_sample_library(const Ref<SampleLibrary>& p_library) {

	library=p_library;
	_change_notify();
	update_configuration_warning();
}

Ref<SampleLibrary> SamplePlayer2D::get_sample_library() const {

	return library;
}

void SamplePlayer2D::set_polyphony(int p_voice_count) {

	ERR_FAIL_COND(p_voice_count<0 || p_voice_count>64);
	polyphony=p_voice_count;
	if (get_source_rid().is_valid())
		SpatialSound2DServer::get_singleton()->source_set_polyphony(get_source_rid(),polyphony);

}

int SamplePlayer2D::get_polyphony() const {

	return polyphony;
}

SamplePlayer2D::VoiceID SamplePlayer2D::play(const String& p_sample,int p_voice) {

	if (!get_source_rid().is_valid())
		return INVALID_VOICE;
	if (library.is_null())
		return INVALID_VOICE;
	if (!library->has_sample(p_sample))
		return INVALID_VOICE;
	Ref<Sample> sample = library->get_sample(p_sample);
	float vol_change = library->sample_get_volume_db(p_sample);
	float pitch_change = library->sample_get_pitch_scale(p_sample);

	VoiceID vid = SpatialSound2DServer::get_singleton()->source_play_sample(get_source_rid(),sample->get_rid(),sample->get_mix_rate()*pitch_change,p_voice);
	if (vol_change)
		SpatialSound2DServer::get_singleton()->source_voice_set_volume_scale_db(get_source_rid(),vid,vol_change);


	if (random_pitch_scale) {
		float ps = Math::random(-random_pitch_scale,random_pitch_scale);
		if (ps>0)
			ps=1.0+ps;
		else
			ps=1.0/(1.0-ps);
		SpatialSound2DServer::get_singleton()->source_voice_set_pitch_scale(get_source_rid(),vid,ps*pitch_change);

	}

	return vid;
}
//voices
void SamplePlayer2D::voice_set_pitch_scale(VoiceID p_voice, float p_pitch_scale) {

	if (!get_source_rid().is_valid())
		return;

	SpatialSound2DServer::get_singleton()->source_voice_set_pitch_scale(get_source_rid(),p_voice,p_pitch_scale);

}

void SamplePlayer2D::voice_set_volume_scale_db(VoiceID p_voice, float p_volume_db) {

	if (!get_source_rid().is_valid())
		return;
	SpatialSound2DServer::get_singleton()->source_voice_set_volume_scale_db(get_source_rid(),p_voice,p_volume_db);

}

bool SamplePlayer2D::is_voice_active(VoiceID p_voice) const {

	if (!get_source_rid().is_valid())
		return false;
	return SpatialSound2DServer::get_singleton()->source_is_voice_active(get_source_rid(),p_voice);

}

void SamplePlayer2D::stop_voice(VoiceID p_voice) {

	if (!get_source_rid().is_valid())
		return;
	SpatialSound2DServer::get_singleton()->source_stop_voice(get_source_rid(),p_voice);

}

void SamplePlayer2D::stop_all() {

	if (!get_source_rid().is_valid())
		return;

	for(int i=0;i<polyphony;i++) {

		SpatialSound2DServer::get_singleton()->source_stop_voice(get_source_rid(),i);
	}
}

void SamplePlayer2D::set_random_pitch_scale(float p_scale) {
	random_pitch_scale=p_scale;
}

float SamplePlayer2D::get_random_pitch_scale() const {

	return random_pitch_scale;
}

String SamplePlayer2D::get_configuration_warning() const {

	if (library.is_null()) {
		return TTR("A SampleLibrary resource must be created or set in the 'samples' property in order for SamplePlayer to play sound.");
	}

	return String();
}

void SamplePlayer2D::_bind_methods() {


	ClassDB::bind_method(_MD("set_sample_library","library:SampleLibrary"),&SamplePlayer2D::set_sample_library);
	ClassDB::bind_method(_MD("get_sample_library:SampleLibrary"),&SamplePlayer2D::get_sample_library);

	ClassDB::bind_method(_MD("set_polyphony","max_voices"),&SamplePlayer2D::set_polyphony);
	ClassDB::bind_method(_MD("get_polyphony"),&SamplePlayer2D::get_polyphony);

	ClassDB::bind_method(_MD("play","sample","voice"),&SamplePlayer2D::play,DEFVAL(NEXT_VOICE));
	//voices,DEV
	ClassDB::bind_method(_MD("voice_set_pitch_scale","voice","ratio"),&SamplePlayer2D::voice_set_pitch_scale);
	ClassDB::bind_method(_MD("voice_set_volume_scale_db","voice","db"),&SamplePlayer2D::voice_set_volume_scale_db);

	ClassDB::bind_method(_MD("is_voice_active","voice"),&SamplePlayer2D::is_voice_active);
	ClassDB::bind_method(_MD("stop_voice","voice"),&SamplePlayer2D::stop_voice);
	ClassDB::bind_method(_MD("stop_all"),&SamplePlayer2D::stop_all);

	ClassDB::bind_method(_MD("set_random_pitch_scale","val"),&SamplePlayer2D::set_random_pitch_scale);
	ClassDB::bind_method(_MD("get_random_pitch_scale"),&SamplePlayer2D::get_random_pitch_scale);

	BIND_CONSTANT( INVALID_VOICE );
	BIND_CONSTANT( NEXT_VOICE );

	ADD_GROUP("Config","");
	ADD_PROPERTY( PropertyInfo( Variant::INT, "polyphony", PROPERTY_HINT_RANGE, "1,64,1"),_SCS("set_polyphony"),_SCS("get_polyphony"));
	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "samples", PROPERTY_HINT_RESOURCE_TYPE,"SampleLibrary"),_SCS("set_sample_library"),_SCS("get_sample_library"));
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "pitch_random", PROPERTY_HINT_RESOURCE_TYPE,"SampleLibrary"),_SCS("set_random_pitch_scale"),_SCS("get_random_pitch_scale"));


}


SamplePlayer2D::SamplePlayer2D() {

	polyphony=1;
	random_pitch_scale=0;

}

SamplePlayer2D::~SamplePlayer2D() {


}
