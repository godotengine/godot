/*************************************************************************/
/*  audio_effect_eq.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "audio_effect_eq.h"
#include "servers/audio_server.h"

void AudioEffectEQInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {

	int band_count = bands[0].size();
	EQ::BandProcess *proc_l = bands[0].ptrw();
	EQ::BandProcess *proc_r = bands[1].ptrw();
	float *bgain = gains.ptrw();
	for (int i = 0; i < band_count; i++) {
		bgain[i] = Math::db2linear(base->gain[i]);
	}

	for (int i = 0; i < p_frame_count; i++) {

		AudioFrame src = p_src_frames[i];
		AudioFrame dst = AudioFrame(0, 0);

		for (int j = 0; j < band_count; j++) {

			float l = src.l;
			float r = src.r;

			proc_l[j].process_one(l);
			proc_r[j].process_one(r);

			dst.l += l * bgain[j];
			dst.r += r * bgain[j];
		}

		p_dst_frames[i] = dst;
	}
}

Ref<AudioEffectInstance> AudioEffectEQ::instance() {
	Ref<AudioEffectEQInstance> ins;
	ins.instance();
	ins->base = Ref<AudioEffectEQ>(this);
	ins->gains.resize(eq.get_band_count());
	for (int i = 0; i < 2; i++) {
		ins->bands[i].resize(eq.get_band_count());
		for (int j = 0; j < ins->bands[i].size(); j++) {
			ins->bands[i][j] = eq.get_band_processor(j);
		}
	}

	return ins;
}

void AudioEffectEQ::set_band_gain_db(int p_band, float p_volume) {
	ERR_FAIL_INDEX(p_band, gain.size());
	gain[p_band] = p_volume;
}

float AudioEffectEQ::get_band_gain_db(int p_band) const {
	ERR_FAIL_INDEX_V(p_band, gain.size(), 0);

	return gain[p_band];
}
int AudioEffectEQ::get_band_count() const {
	return gain.size();
}

bool AudioEffectEQ::_set(const StringName &p_name, const Variant &p_value) {

	const Map<StringName, int>::Element *E = prop_band_map.find(p_name);
	if (E) {
		set_band_gain_db(E->get(), p_value);
		return true;
	}

	return false;
}

bool AudioEffectEQ::_get(const StringName &p_name, Variant &r_ret) const {

	const Map<StringName, int>::Element *E = prop_band_map.find(p_name);
	if (E) {
		r_ret = get_band_gain_db(E->get());
		return true;
	}

	return false;
}

void AudioEffectEQ::_get_property_list(List<PropertyInfo> *p_list) const {

	for (int i = 0; i < band_names.size(); i++) {

		p_list->push_back(PropertyInfo(Variant::REAL, band_names[i], PROPERTY_HINT_RANGE, "-60,24,0.1"));
	}
}

void AudioEffectEQ::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_band_gain_db", "band_idx", "volume_db"), &AudioEffectEQ::set_band_gain_db);
	ClassDB::bind_method(D_METHOD("get_band_gain_db", "band_idx"), &AudioEffectEQ::get_band_gain_db);
	ClassDB::bind_method(D_METHOD("get_band_count"), &AudioEffectEQ::get_band_count);
}

AudioEffectEQ::AudioEffectEQ(EQ::Preset p_preset) {

	eq.set_mix_rate(AudioServer::get_singleton()->get_mix_rate());
	eq.set_preset_band_mode(p_preset);
	gain.resize(eq.get_band_count());
	for (int i = 0; i < gain.size(); i++) {
		gain[i] = 0.0;
		String name = "band_db/" + itos(eq.get_band_frequency(i)) + "_hz";
		prop_band_map[name] = i;
		band_names.push_back(name);
	}
}
