/*************************************************************************/
/*  audio_effect_eq.h                                                    */
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

#ifndef AUDIOEFFECTEQ_H
#define AUDIOEFFECTEQ_H

#include "servers/audio/audio_effect.h"
#include "servers/audio/effects/eq.h"

class AudioEffectEQ;

class AudioEffectEQInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectEQInstance, AudioEffectInstance);
	friend class AudioEffectEQ;
	Ref<AudioEffectEQ> base;
	int cur_freqs_rev;

	Vector<EQ::BandProcess> bands[2];
	Vector<float> gains;

public:
	virtual void process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count);
};

class AudioEffectEQ : public AudioEffect {
	GDCLASS(AudioEffectEQ, AudioEffect);

	friend class AudioEffectEQInstance;

protected:
	EQ eq;
	Vector<float> gain;
	Map<StringName, int> prop_band_map;
	Vector<String> band_names;
	int freqs_rev;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	Ref<AudioEffectInstance> instance();
	int get_band_count() const;
	float get_band_frequency(int p_band) const;
	void set_band_gain_db(int p_band, float p_volume);
	float get_band_gain_db(int p_band) const;

	AudioEffectEQ();
};

class AudioEffectEQPreset : public AudioEffectEQ {
	GDCLASS(AudioEffectEQPreset, AudioEffectEQ);

protected:
	AudioEffectEQPreset(EQ::Preset p_preset = EQ::PRESET_6_BANDS);
};

class AudioEffectEQ6 : public AudioEffectEQPreset {
	GDCLASS(AudioEffectEQ6, AudioEffectEQPreset);

public:
	AudioEffectEQ6() :
			AudioEffectEQPreset(EQ::PRESET_6_BANDS) {}
};

class AudioEffectEQ10 : public AudioEffectEQPreset {
	GDCLASS(AudioEffectEQ10, AudioEffectEQPreset);

public:
	AudioEffectEQ10() :
			AudioEffectEQPreset(EQ::PRESET_10_BANDS) {}
};

class AudioEffectEQ21 : public AudioEffectEQPreset {
	GDCLASS(AudioEffectEQ21, AudioEffectEQPreset);

public:
	AudioEffectEQ21() :
			AudioEffectEQPreset(EQ::PRESET_21_BANDS) {}
};

class AudioEffectEQCustom : public AudioEffectEQ {
	GDCLASS(AudioEffectEQCustom, AudioEffectEQ);

protected:
	Vector<float> freqs;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	AudioEffectEQCustom();

	void set_band_count(int p_count);
	void set_band_frequency(int p_band, float p_freq);
};

#endif // AUDIOEFFECTEQ_H
