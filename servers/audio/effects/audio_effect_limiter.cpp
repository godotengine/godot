/*************************************************************************/
/*  audio_effect_limiter.cpp                                             */
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

#include "audio_effect_limiter.h"
#include "core/math/math_funcs.h"

// Soft clipping starts to reduce the peaks a little below the threshold level
// and progressively increases its effect as the input level increases such that
// the threshold is never exceeded.
//
// This is implemented as a logistics function because it's designed to increase near
// linearly and taper up to a certain max value. This is exactly what we want for
// this soft clipping function.
void AudioEffectLimiterInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	// The threshold from which the limiter begins to be active, in decibels. Value can range from -30 to 0.
	float threshold = Math::db2linear(base->threshold);

	// The waveform's maximum allowed value, in decibels. Value can range from -20 to -0.1.
	float ceiling = Math::db2linear(base->ceiling);

	// Applies a gain to the limited waves, in decibels. Value can range from 0 to 6.
	// This will be the slope (k) of the logistics curve. 0 will end up creating a hard clip.
	float soft_clip = Math::db2linear(base->soft_clip);

	// The difference between the threshold and the ceiling
	// This is used as the maximum value (L) in the logistics function.
	float limit = ceiling - threshold;

	// The halfway point of the threshold and the ceiling.
	// This will serve as our x0 in the logistics transform.
	float mid_point = limit / 2.0;

	for (int i = 0; i < p_frame_count; i++) {
		float spl0 = p_src_frames[i].l;
		float spl1 = p_src_frames[i].r;

		// The signs of the frame.
		float sign0 = (spl0 < 0.0 ? -1.0 : 1.0);
		float sign1 = (spl1 < 0.0 ? -1.0 : 1.0);

		// The values of the frame.
		float abs0 = Math::abs(spl0);
		float abs1 = Math::abs(spl1);

		// We only activate when we hit the threshold.
		if (abs0 > threshold) {
			// Pull the value down to the threshold.
			abs0 -= threshold;
			// Apply the logistics transformation.
			abs0 = limit / (1 + exp(-soft_clip * (abs0 - mid_point)));
			// Bring it back up.
			abs0 += threshold;
			// Apply this value to spl0.
			spl0 = sign0 * abs0;
		}

		// Ditto with spl1.
		if (abs1 > threshold) {
			abs1 -= threshold;
			abs1 = ceiling / (1 + exp(-soft_clip * (abs1 - mid_point)));
			abs1 += threshold;
			spl1 = sign1 * abs1;
		}

		p_dst_frames[i].l = spl0;
		p_dst_frames[i].r = spl1;
	}
}

Ref<AudioEffectInstance> AudioEffectLimiter::instantiate() {
	Ref<AudioEffectLimiterInstance> ins;
	ins.instantiate();
	ins->base = Ref<AudioEffectLimiter>(this);

	return ins;
}

void AudioEffectLimiter::set_threshold_db(float p_threshold) {
	threshold = p_threshold;
}

float AudioEffectLimiter::get_threshold_db() const {
	return threshold;
}

void AudioEffectLimiter::set_ceiling_db(float p_ceiling) {
	ceiling = p_ceiling;
}

float AudioEffectLimiter::get_ceiling_db() const {
	return ceiling;
}

void AudioEffectLimiter::set_soft_clip_db(float p_soft_clip) {
	soft_clip = p_soft_clip;
}

float AudioEffectLimiter::get_soft_clip_db() const {
	return soft_clip;
}

void AudioEffectLimiter::set_soft_clip_ratio(float p_soft_clip) {
	soft_clip_ratio = p_soft_clip;
}

float AudioEffectLimiter::get_soft_clip_ratio() const {
	return soft_clip_ratio;
}

void AudioEffectLimiter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_ceiling_db", "ceiling"), &AudioEffectLimiter::set_ceiling_db);
	ClassDB::bind_method(D_METHOD("get_ceiling_db"), &AudioEffectLimiter::get_ceiling_db);

	ClassDB::bind_method(D_METHOD("set_threshold_db", "threshold"), &AudioEffectLimiter::set_threshold_db);
	ClassDB::bind_method(D_METHOD("get_threshold_db"), &AudioEffectLimiter::get_threshold_db);

	ClassDB::bind_method(D_METHOD("set_soft_clip_db", "soft_clip"), &AudioEffectLimiter::set_soft_clip_db);
	ClassDB::bind_method(D_METHOD("get_soft_clip_db"), &AudioEffectLimiter::get_soft_clip_db);

	ClassDB::bind_method(D_METHOD("set_soft_clip_ratio", "soft_clip"), &AudioEffectLimiter::set_soft_clip_ratio);
	ClassDB::bind_method(D_METHOD("get_soft_clip_ratio"), &AudioEffectLimiter::get_soft_clip_ratio);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ceiling_db", PROPERTY_HINT_RANGE, "-20,0,0.1"), "set_ceiling_db", "get_ceiling_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "threshold_db", PROPERTY_HINT_RANGE, "-30,-0.1,0.1"), "set_threshold_db", "get_threshold_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "soft_clip_db", PROPERTY_HINT_RANGE, "0,6,0.1"), "set_soft_clip_db", "get_soft_clip_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "soft_clip_ratio", PROPERTY_HINT_RANGE, "3,20,0.1"), "set_soft_clip_ratio", "get_soft_clip_ratio");
}

AudioEffectLimiter::AudioEffectLimiter() {
	threshold = -0.1;
	ceiling = 0;
	soft_clip = 2;
	soft_clip_ratio = 10;
}
