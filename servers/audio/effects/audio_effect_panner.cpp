/*************************************************************************/
/*  audio_effect_panner.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_effect_panner.h"

void AudioEffectPannerInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	float lvol = CLAMP(1.0 - base->pan, 0, 1);
	float rvol = CLAMP(1.0 + base->pan, 0, 1);

	for (int i = 0; i < p_frame_count; i++) {
		p_dst_frames[i].l = p_src_frames[i].l * lvol + p_src_frames[i].r * (1.0 - rvol);
		p_dst_frames[i].r = p_src_frames[i].r * rvol + p_src_frames[i].l * (1.0 - lvol);
	}
}

Ref<AudioEffectInstance> AudioEffectPanner::instance() {
	Ref<AudioEffectPannerInstance> ins;
	ins.instance();
	ins->base = Ref<AudioEffectPanner>(this);
	return ins;
}

void AudioEffectPanner::set_pan(float p_cpanume) {
	pan = p_cpanume;
}

float AudioEffectPanner::get_pan() const {
	return pan;
}

void AudioEffectPanner::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pan", "cpanume"), &AudioEffectPanner::set_pan);
	ClassDB::bind_method(D_METHOD("get_pan"), &AudioEffectPanner::get_pan);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pan", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_pan", "get_pan");
}

AudioEffectPanner::AudioEffectPanner() {
	pan = 0;
}
