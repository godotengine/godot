/*************************************************************************/
/*  spatial_player.cpp                                                   */
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
#include "spatial_player.h"

#include "servers/audio_server.h"
#include "camera.h"
#include "servers/spatial_sound_server.h"
#include "scene/resources/surface_tool.h"


void SpatialPlayer::_notification(int p_what) {


	switch(p_what) {

		case NOTIFICATION_ENTER_WORLD: {
			//find the sound space

			source_rid = SpatialSoundServer::get_singleton()->source_create(get_world()->get_sound_space());
			for(int i=0;i<PARAM_MAX;i++)
				set_param(Param(i),params[i]);


		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			SpatialSoundServer::get_singleton()->source_set_transform(source_rid,get_global_transform());

		} break;
		case NOTIFICATION_EXIT_WORLD: {

			if (source_rid.is_valid())
				SpatialSoundServer::get_singleton()->free(source_rid);

		} break;
	}

}


void SpatialPlayer::set_param( Param p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,PARAM_MAX);
	params[p_param]=p_value;
	if (p_param==PARAM_EMISSION_CONE_DEGREES) {
		update_gizmo();
	}
	if (source_rid.is_valid())
		SpatialSoundServer::get_singleton()->source_set_param(source_rid,(SpatialSoundServer::SourceParam)p_param,p_value);

}

float SpatialPlayer::get_param( Param p_param) const {

	ERR_FAIL_INDEX_V(p_param,PARAM_MAX,0);
	return params[p_param];

}

bool SpatialPlayer::_can_gizmo_scale() const {

	return false;
}

void SpatialPlayer::_bind_methods() {


	ClassDB::bind_method(_MD("set_param","param","value"),&SpatialPlayer::set_param);
	ClassDB::bind_method(_MD("get_param","param"),&SpatialPlayer::get_param);

	BIND_CONSTANT( PARAM_VOLUME_DB );
	BIND_CONSTANT( PARAM_PITCH_SCALE );
	BIND_CONSTANT( PARAM_ATTENUATION_MIN_DISTANCE );
	BIND_CONSTANT( PARAM_ATTENUATION_MAX_DISTANCE );
	BIND_CONSTANT( PARAM_ATTENUATION_DISTANCE_EXP );
	BIND_CONSTANT( PARAM_EMISSION_CONE_DEGREES );
	BIND_CONSTANT( PARAM_EMISSION_CONE_ATTENUATION_DB );
	BIND_CONSTANT( PARAM_MAX );

	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "volume_db",PROPERTY_HINT_RANGE, "-80,24,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_VOLUME_DB);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "pitch_scale",PROPERTY_HINT_RANGE, "0.001,32,0.001"),_SCS("set_param"),_SCS("get_param"),PARAM_PITCH_SCALE);
	ADD_GROUP("Attenuation","attenuation_");
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "attenuation_min_distance",PROPERTY_HINT_RANGE, "0.01,4096,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_ATTENUATION_MIN_DISTANCE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "attenuation_max_distance",PROPERTY_HINT_RANGE, "0.01,4096,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_ATTENUATION_MAX_DISTANCE);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "attenuation_distance_exp",PROPERTY_HINT_EXP_EASING, "attenuation"),_SCS("set_param"),_SCS("get_param"),PARAM_ATTENUATION_DISTANCE_EXP);
	ADD_GROUP("Emission Cone","emission_cone_");
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "emission_cone_degrees",PROPERTY_HINT_RANGE, "0,180,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_EMISSION_CONE_DEGREES);
	ADD_PROPERTYI( PropertyInfo( Variant::REAL, "emission_cone_attenuation_db",PROPERTY_HINT_RANGE, "-80,24,0.01"),_SCS("set_param"),_SCS("get_param"),PARAM_EMISSION_CONE_ATTENUATION_DB);

}


SpatialPlayer::SpatialPlayer() {

	params[PARAM_VOLUME_DB]=0.0;
	params[PARAM_PITCH_SCALE]=1.0;
	params[PARAM_ATTENUATION_MIN_DISTANCE]=1;
	params[PARAM_ATTENUATION_MAX_DISTANCE]=100;
	params[PARAM_ATTENUATION_DISTANCE_EXP]=1.0; //linear (and not really good)
	params[PARAM_EMISSION_CONE_DEGREES]=180.0; //cone disabled
	params[PARAM_EMISSION_CONE_ATTENUATION_DB]=-6.0; //minus 6 db attenuation

}

SpatialPlayer::~SpatialPlayer() {


}
