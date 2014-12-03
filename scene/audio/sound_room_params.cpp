/*************************************************************************/
/*  sound_room_params.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "sound_room_params.h"

#include "scene/main/viewport.h"

#ifndef _3D_DISABLED
void SoundRoomParams::_update_sound_room() {

	if (!room.is_valid())
		return;

	for(int i=0;i<PARAM_MAX;i++) {

		SpatialSoundServer::get_singleton()->room_set_param(room,SpatialSoundServer::RoomParam(i),params[i]);

	}

	SpatialSoundServer::get_singleton()->room_set_reverb(room,SpatialSoundServer::RoomReverb(reverb));
	SpatialSoundServer::get_singleton()->room_set_force_params_to_all_sources(room,force_params_for_all_sources);
}


void SoundRoomParams::_notification(int p_what) {


	switch(p_what) {


		case NOTIFICATION_ENTER_TREE: {
//#if 0
			Node *n=this;
			Room *room_instance=NULL;
			while(n) {

				room_instance=n->cast_to<Room>();
				if (room_instance) {

					break;
				}
				if (n->cast_to<Viewport>())
					break;

				n=n->get_parent();
			}


			if (room_instance) {
				room=room_instance->get_sound_room();
			} else {
				room=get_viewport()->find_world()->get_sound_space();
			}

			_update_sound_room();
//#endif

		} break;
		case NOTIFICATION_EXIT_TREE: {

			room=RID();

		} break;
	}
}


void SoundRoomParams::set_param(Params p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,PARAM_MAX);
	params[p_param]=p_value;
	if (room.is_valid())
		SpatialSoundServer::get_singleton()->room_set_param(room,SpatialSoundServer::RoomParam(p_param),p_value);
}

float SoundRoomParams::get_param(Params p_param) const {

	ERR_FAIL_INDEX_V(p_param,PARAM_MAX,0);
	return params[p_param];
}


void SoundRoomParams::set_reverb_mode(Reverb p_mode) {

	ERR_FAIL_INDEX(p_mode,4);
	reverb=p_mode;
	if (room.is_valid())
		SpatialSoundServer::get_singleton()->room_set_reverb(room,SpatialSoundServer::RoomReverb(p_mode));
}

SoundRoomParams::Reverb SoundRoomParams::get_reverb_mode() const {

	return reverb;
}


void SoundRoomParams::set_force_params_to_all_sources(bool p_force) {

	force_params_for_all_sources=p_force;
	if (room.is_valid())
		SpatialSoundServer::get_singleton()->room_set_force_params_to_all_sources(room,p_force);
}

bool SoundRoomParams::is_forcing_params_to_all_sources() {

	return force_params_for_all_sources;
}


void SoundRoomParams::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_param","param","value"),&SoundRoomParams::set_param );
	ObjectTypeDB::bind_method(_MD("get_param","param"),&SoundRoomParams::get_param );

	ObjectTypeDB::bind_method(_MD("set_reverb_mode","reverb_mode","value"),&SoundRoomParams::set_reverb_mode );
	ObjectTypeDB::bind_method(_MD("get_reverb_mode","reverb_mode"),&SoundRoomParams::get_reverb_mode );

	ObjectTypeDB::bind_method(_MD("set_force_params_to_all_sources","enabled"),&SoundRoomParams::set_force_params_to_all_sources );
	ObjectTypeDB::bind_method(_MD("is_forcing_params_to_all_sources"),&SoundRoomParams::is_forcing_params_to_all_sources );


	ADD_PROPERTY( PropertyInfo( Variant::INT, "reverb/mode", PROPERTY_HINT_ENUM, "Small,Medium,Large,Hall"), _SCS("set_reverb_mode"), _SCS("get_reverb_mode") );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/speed_of_scale", PROPERTY_HINT_RANGE, "0.01,16,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_SPEED_OF_SOUND_SCALE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/doppler_factor",PROPERTY_HINT_RANGE, "0.01,16,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_DOPPLER_FACTOR );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/pitch_scale",PROPERTY_HINT_RANGE, "0.01,32,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_PITCH_SCALE );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/volume_scale_db",PROPERTY_HINT_RANGE, "-80,24,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_VOLUME_SCALE_DB );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/reverb_send",PROPERTY_HINT_RANGE, "0,1,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_REVERB_SEND );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/chorus_send",PROPERTY_HINT_RANGE, "0,1,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_CHORUS_SEND );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/attenuation_scale",PROPERTY_HINT_RANGE, "0.01,32,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_SCALE );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/attenuation_hf_cutoff",PROPERTY_HINT_RANGE, "30,16384,1"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_HF_CUTOFF );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/attenuation_hf_floor_db",PROPERTY_HINT_RANGE, "-80,24,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_HF_FLOOR_DB );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/attenuation_hf_ratio_exp",PROPERTY_HINT_RANGE, "0.01,32,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_HF_RATIO_EXP );
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "params/attenuation_reverb_scale",PROPERTY_HINT_RANGE, "0.01,32,0.01"), _SCS("set_param"), _SCS("get_param"), PARAM_ATTENUATION_REVERB_SCALE );
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "force_to_all_sources"),_SCS("set_force_params_to_all_sources"),_SCS("is_forcing_params_to_all_sources") );

}


SoundRoomParams::SoundRoomParams() {

	reverb=REVERB_HALL;
	params[PARAM_SPEED_OF_SOUND_SCALE]=1;
	params[PARAM_DOPPLER_FACTOR]=1.0;
	params[PARAM_PITCH_SCALE]=1.0;
	params[PARAM_VOLUME_SCALE_DB]=0;
	params[PARAM_REVERB_SEND]=0;
	params[PARAM_CHORUS_SEND]=0;
	params[PARAM_ATTENUATION_SCALE]=1.0;
	params[PARAM_ATTENUATION_HF_CUTOFF]=5000;
	params[PARAM_ATTENUATION_HF_FLOOR_DB]=-24.0;
	params[PARAM_ATTENUATION_HF_RATIO_EXP]=1.0;
	params[PARAM_ATTENUATION_REVERB_SCALE]=0.0;
	force_params_for_all_sources=false;
}
#endif
