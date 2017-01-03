/*************************************************************************/
/*  spatial_player.h                                                     */
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
#ifndef SPATIAL_PLAYER_H
#define SPATIAL_PLAYER_H


#include "scene/3d/spatial.h"
#include "scene/main/node.h"
#include "scene/resources/sample_library.h"
#include "servers/spatial_sound_server.h"
#include "scene/main/viewport.h"

class SpatialPlayer : public Spatial {

	GDCLASS(SpatialPlayer,Spatial);
public:


	enum Param {

		PARAM_VOLUME_DB=SpatialSoundServer::SOURCE_PARAM_VOLUME_DB,
		PARAM_PITCH_SCALE=SpatialSoundServer::SOURCE_PARAM_PITCH_SCALE,
		PARAM_ATTENUATION_MIN_DISTANCE=SpatialSoundServer::SOURCE_PARAM_ATTENUATION_MIN_DISTANCE,
		PARAM_ATTENUATION_MAX_DISTANCE=SpatialSoundServer::SOURCE_PARAM_ATTENUATION_MAX_DISTANCE,
		PARAM_ATTENUATION_DISTANCE_EXP=SpatialSoundServer::SOURCE_PARAM_ATTENUATION_DISTANCE_EXP,
		PARAM_EMISSION_CONE_DEGREES=SpatialSoundServer::SOURCE_PARAM_EMISSION_CONE_DEGREES,
		PARAM_EMISSION_CONE_ATTENUATION_DB=SpatialSoundServer::SOURCE_PARAM_EMISSION_CONE_ATTENUATION_DB,
		PARAM_MAX=SpatialSoundServer::SOURCE_PARAM_MAX
	};

private:

	float params[PARAM_MAX];
	RID source_rid;

	virtual bool _can_gizmo_scale() const;



protected:

	_FORCE_INLINE_ RID get_source_rid() const { return source_rid; }

	void _notification(int p_what);

	static void _bind_methods();

public:

	void set_param( Param p_param, float p_value);
	float get_param( Param p_param) const;


	SpatialPlayer();
	~SpatialPlayer();


};

VARIANT_ENUM_CAST( SpatialPlayer::Param );
#endif // SPATIAL_PLAYER_H
