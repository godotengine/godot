/*************************************************************************/
/*  sound_room_params.h                                                  */
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
#ifndef SOUND_ROOM_PARAMS_H
#define SOUND_ROOM_PARAMS_H

#include "scene/main/node.h"
#include "servers/spatial_sound_server.h"


#ifndef _3D_DISABLED

#include "scene/3d/room_instance.h"
class SoundRoomParams : public Node {

	GDCLASS( SoundRoomParams, Node );
public:

	enum Params {
		PARAM_SPEED_OF_SOUND_SCALE=SpatialSoundServer::ROOM_PARAM_SPEED_OF_SOUND_SCALE,
		PARAM_DOPPLER_FACTOR=SpatialSoundServer::ROOM_PARAM_DOPPLER_FACTOR,
		PARAM_PITCH_SCALE=SpatialSoundServer::ROOM_PARAM_PITCH_SCALE,
		PARAM_VOLUME_SCALE_DB=SpatialSoundServer::ROOM_PARAM_VOLUME_SCALE_DB,
		PARAM_REVERB_SEND=SpatialSoundServer::ROOM_PARAM_REVERB_SEND,
		PARAM_CHORUS_SEND=SpatialSoundServer::ROOM_PARAM_CHORUS_SEND,
		PARAM_ATTENUATION_SCALE=SpatialSoundServer::ROOM_PARAM_ATTENUATION_SCALE,
		PARAM_ATTENUATION_HF_CUTOFF=SpatialSoundServer::ROOM_PARAM_ATTENUATION_HF_CUTOFF,
		PARAM_ATTENUATION_HF_FLOOR_DB=SpatialSoundServer::ROOM_PARAM_ATTENUATION_HF_FLOOR_DB,
		PARAM_ATTENUATION_HF_RATIO_EXP=SpatialSoundServer::ROOM_PARAM_ATTENUATION_HF_RATIO_EXP,
		PARAM_ATTENUATION_REVERB_SCALE=SpatialSoundServer::ROOM_PARAM_ATTENUATION_REVERB_SCALE,
		PARAM_MAX=SpatialSoundServer::ROOM_PARAM_MAX
	};

	enum Reverb {
		REVERB_SMALL,
		REVERB_MEDIUM,
		REVERB_LARGE,
		REVERB_HALL
	};
private:

	RID room;

	float params[PARAM_MAX];
	Reverb reverb;
	bool force_params_for_all_sources;
	void _update_sound_room();


protected:

	void _notification(int p_what);
	static void _bind_methods();

public:


	void set_param(Params p_param, float p_value);
	float get_param(Params p_param) const;

	void set_reverb_mode(Reverb p_mode);
	Reverb get_reverb_mode() const;

	void set_force_params_to_all_sources(bool p_force);
	bool is_forcing_params_to_all_sources();

	SoundRoomParams();
};

VARIANT_ENUM_CAST(SoundRoomParams::Params);
VARIANT_ENUM_CAST(SoundRoomParams::Reverb);

#endif

#endif // SOUND_ROOM_PARAMS_H
