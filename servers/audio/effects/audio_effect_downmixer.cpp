/*************************************************************************/
/*  audio_effect_downmixer.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_effect_downmixer.h"
#include "core/math/math_funcs.h"
#include "servers/audio_server.h"

void AudioEffectDownmixerInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	float this_mix_strength_delta = base->strength - last_strength;
	for (int frame = 0; frame < p_frame_count; frame++) {
		float mix_fraction = (float)frame / p_frame_count;
		float this_frame_strength = last_strength + mix_fraction * this_mix_strength_delta;
		float mono_val = p_src_frames[frame].l + p_src_frames[frame].r;
		p_dst_frames[frame] = p_src_frames[frame] * (1.0f - this_frame_strength) + AudioFrame(mono_val, mono_val) * this_frame_strength;
	}
	last_strength = base->strength;
}

Ref<AudioEffectInstance> AudioEffectDownmixer::instantiate() {
	Ref<AudioEffectDownmixerInstance> ins;
	ins.instantiate();
	ins->last_strength = strength;
	ins->base = Ref<AudioEffectDownmixer>(this);

	return ins;
}

void AudioEffectDownmixer::set_strength(float p_strength) {
	strength = p_strength;
}

float AudioEffectDownmixer::get_strength() const {
	return strength;
}

void AudioEffectDownmixer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_strength", "strength"), &AudioEffectDownmixer::set_strength);
	ClassDB::bind_method(D_METHOD("get_strength"), &AudioEffectDownmixer::get_strength);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_strength", "get_strength");
}

AudioEffectDownmixer::AudioEffectDownmixer() {
	strength = 0;
}
