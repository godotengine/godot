/*************************************************************************/
/*  spatial_sample_player.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SPATIAL_SAMPLE_PLAYER_H
#define SPATIAL_SAMPLE_PLAYER_H

#include "scene/3d/spatial_player.h"
#include "scene/resources/sample_library.h"
#include "servers/spatial_sound_server.h"

class SpatialSamplePlayer : public SpatialPlayer {

	OBJ_TYPE(SpatialSamplePlayer, SpatialPlayer);

public:
	enum {

		INVALID_VOICE = SpatialSoundServer::SOURCE_INVALID_VOICE,
		NEXT_VOICE = SpatialSoundServer::SOURCE_NEXT_VOICE
	};

	typedef int VoiceID;

private:
	Ref<SampleLibrary> library;
	int polyphony;
	String played_back;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_sample_library(const Ref<SampleLibrary> &p_library);
	Ref<SampleLibrary> get_sample_library() const;

	void set_polyphony(int p_voice_count);
	int get_polyphony() const;

	VoiceID play(const String &p_sample, int p_voice = NEXT_VOICE);
	//voices
	void voice_set_pitch_scale(VoiceID p_voice, float p_pitch_scale);
	void voice_set_volume_scale_db(VoiceID p_voice, float p_volume_db);

	bool is_voice_active(VoiceID p_voice) const;
	void stop_voice(VoiceID p_voice);
	void stop_all();

	String get_configuration_warning() const;

	SpatialSamplePlayer();
	~SpatialSamplePlayer();
};

#endif // SPATIAL_SAMPLE_PLAYER_H
