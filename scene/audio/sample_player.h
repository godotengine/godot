/*************************************************************************/
/*  sample_player.h                                                      */
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
#ifndef SAMPLE_PLAYER_H
#define SAMPLE_PLAYER_H

#include "scene/main/node.h"
#include "scene/resources/sample_library.h"

class SamplePlayer : public Node {

	GDCLASS( SamplePlayer, Node );
	OBJ_CATEGORY("Audio Nodes");
public:


	enum FilterType {
		FILTER_NONE,
		FILTER_LOWPASS,
		FILTER_BANDPASS,
		FILTER_HIPASS,
		FILTER_NOTCH,
		FILTER_PEAK,
		FILTER_BANDLIMIT, ///< cutoff is LP resonace is HP
		FILTER_LOW_SHELF,
		FILTER_HIGH_SHELF,
	};

	enum ReverbRoomType {

		REVERB_SMALL,
		REVERB_MEDIUM,
		REVERB_LARGE,
		REVERB_HALL
	};

	enum {

		INVALID_VOICE_ID=0xFFFFFFFF
	};

	typedef uint32_t VoiceID;

private:

	Ref<SampleLibrary> library;

	struct Voice {

		RID voice;
		uint32_t check;
		bool active;

		int sample_mix_rate;
		int mix_rate;
		float volume;
		float pan;
		float pan_depth;
		float pan_height;
		FilterType filter_type;
		float filter_cutoff;
		float filter_resonance;
		float filter_gain;
		float chorus_send;
		ReverbRoomType reverb_room;
		float reverb_send;

		void clear();
		Voice();
		~Voice();
	};

	Vector<Voice> voices;

	struct Default {

		float reverb_send;
		float pitch_scale;
		float volume_db;
		float pan;
		float depth;
		float height;
		FilterType filter_type;
		float filter_cutoff;
		float filter_resonance;
		float filter_gain;
		float chorus_send;
		ReverbRoomType reverb_room;

	} _default;

	uint32_t last_id;
	uint16_t last_check;
	String played_back;
protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:

	void set_sample_library(const Ref<SampleLibrary>& p_library);
	Ref<SampleLibrary> get_sample_library() const;

	void set_polyphony(int p_voice_count);
	int get_polyphony() const;

	VoiceID play(const String& p_name,bool unique=false);
	void stop(VoiceID p_voice);
	void stop_all();
	bool is_voice_active(VoiceID) const;
	bool is_active() const;

	void set_mix_rate(VoiceID p_voice, int p_mix_rate);
	void set_pitch_scale(VoiceID p_voice, float p_pitch_scale);
	void set_volume(VoiceID p_voice, float p_volume);
	void set_volume_db(VoiceID p_voice, float p_db);
	void set_pan(VoiceID p_voice, float p_pan,float p_pan_depth=0,float p_pan_height=0);
	void set_filter(VoiceID p_voice,FilterType p_filter,float p_cutoff,float p_resonance,float p_gain);
	void set_chorus(VoiceID p_voice,float p_send);
	void set_reverb(VoiceID p_voice,ReverbRoomType p_room,float p_send);

	int get_mix_rate(VoiceID p_voice) const;
	float get_pitch_scale(VoiceID p_voice) const;
	float get_volume(VoiceID p_voice) const;
	float get_volume_db(VoiceID p_voice) const;

	float get_pan(VoiceID p_voice) const;
	float get_pan_depth(VoiceID p_voice) const;
	float get_pan_height(VoiceID p_voice) const;
	FilterType get_filter_type(VoiceID p_voice) const;
	float get_filter_cutoff(VoiceID p_voice) const;
	float get_filter_resonance(VoiceID p_voice) const;
	float get_filter_gain(VoiceID p_voice) const;
	float get_chorus(VoiceID p_voice) const;
	ReverbRoomType get_reverb_room(VoiceID p_voice) const;
	float get_reverb(VoiceID p_voice) const;



	void set_default_pitch_scale(float p_pitch_scale);
	void set_default_volume(float p_volume);
	void set_default_volume_db(float p_db);
	void set_default_pan(float p_pan,float p_pan_depth=0,float p_pan_height=0);
	void set_default_filter(FilterType p_filter,float p_cutoff,float p_resonance,float p_gain);
	void set_default_chorus(float p_send);
	void set_default_reverb(ReverbRoomType p_room,float p_send);

	float get_default_volume() const;
	float get_default_volume_db() const;
	float get_default_pitch_scale() const;
	float get_default_pan() const;
	float get_default_pan_depth() const;
	float get_default_pan_height() const;
	FilterType get_default_filter_type() const;
	float get_default_filter_cutoff() const;
	float get_default_filter_resonance() const;
	float get_default_filter_gain() const;
	float get_default_chorus() const;
	ReverbRoomType get_default_reverb_room() const;
	float get_default_reverb() const;

	String get_configuration_warning() const;

	SamplePlayer();
	~SamplePlayer();
};

VARIANT_ENUM_CAST( SamplePlayer::FilterType );
VARIANT_ENUM_CAST( SamplePlayer::ReverbRoomType );

#endif // SAMPLE_PLAYER_H
