/**************************************************************************/
/*  audio_effect_hard_limiter.cpp                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "audio_effect_hard_limiter.h"

#include "servers/audio_server.h"

void AudioEffectHardLimiterInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	float sample_rate = AudioServer::get_singleton()->get_mix_rate();

	float ceiling = Math::db_to_linear(base->ceiling);
	float release = base->release;
	float attack = base->attack;
	float pre_gain = Math::db_to_linear(base->pre_gain);

	for (int i = 0; i < p_frame_count; i++) {
		float sample_left = p_src_frames[i].left;
		float sample_right = p_src_frames[i].right;

		sample_left *= pre_gain;
		sample_right *= pre_gain;

		float largest_sample = MAX(Math::abs(sample_left), Math::abs(sample_right));

		release_factor = MAX(0.0, release_factor - 1.0 / sample_rate);
		release_factor = MIN(release_factor, release);

		if (release_factor > 0.0) {
			gain = Math::lerp(gain_target, 1.0f, 1.0f - release_factor / release);
		}

		if (largest_sample * gain > ceiling) {
			gain_target = ceiling / largest_sample;
			release_factor = release;
			attack_factor = attack;
		}

		// Lerp gain over attack time to avoid distortion.
		attack_factor = MAX(0.0f, attack_factor - 1.0f / sample_rate);
		if (attack_factor > 0.0) {
			gain = Math::lerp(gain_target, gain, 1.0f - attack_factor / attack);
		}

		int bucket_id = gain_bucket_cursor / gain_bucket_size;

		// If first item within the current bucket, reset the bucket.
		if (gain_bucket_cursor % gain_bucket_size == 0) {
			gain_buckets[bucket_id] = 1.0f;
		}

		gain_buckets[bucket_id] = MIN(gain_buckets[bucket_id], gain);

		gain_bucket_cursor = (gain_bucket_cursor + 1) % gain_samples_to_store;

		for (int j = 0; j < (int)gain_buckets.size(); j++) {
			gain = MIN(gain, gain_buckets[j]);
		}

		// Introduce latency by grabbing the AudioFrame stored previously,
		// then overwrite it with current audioframe, then update circular
		// buffer cursor.
		float dst_buffer_left = sample_buffer_left[sample_cursor];
		float dst_buffer_right = sample_buffer_right[sample_cursor];

		sample_buffer_left[sample_cursor] = sample_left;
		sample_buffer_right[sample_cursor] = sample_right;

		sample_cursor = (sample_cursor + 1) % sample_buffer_left.size();

		p_dst_frames[i].left = dst_buffer_left * gain;
		p_dst_frames[i].right = dst_buffer_right * gain;
	}
}

Ref<AudioEffectInstance> AudioEffectHardLimiter::instantiate() {
	Ref<AudioEffectHardLimiterInstance> ins;
	ins.instantiate();
	ins->base = Ref<AudioEffectHardLimiter>(this);

	float mix_rate = AudioServer::get_singleton()->get_mix_rate();

	for (int i = 0; i < (int)Math::ceil(mix_rate * attack) + 1; i++) {
		ins->sample_buffer_left.push_back(0.0f);
		ins->sample_buffer_right.push_back(0.0f);
	}

	ins->gain_samples_to_store = (int)Math::ceil(mix_rate * (attack + sustain) + 1);
	ins->gain_bucket_size = (int)(mix_rate * attack);

	for (int i = 0; i < ins->gain_samples_to_store; i += ins->gain_bucket_size) {
		ins->gain_buckets.push_back(1.0f);
	}

	return ins;
}

void AudioEffectHardLimiter::set_ceiling_db(float p_ceiling) {
	ceiling = p_ceiling;
}

float AudioEffectHardLimiter::get_ceiling_db() const {
	return ceiling;
}

float AudioEffectHardLimiter::get_pre_gain_db() const {
	return pre_gain;
}

void AudioEffectHardLimiter::set_pre_gain_db(const float p_pre_gain) {
	pre_gain = p_pre_gain;
}

float AudioEffectHardLimiter::get_release() const {
	return release;
}

void AudioEffectHardLimiter::set_release(const float p_release) {
	release = p_release;
}

void AudioEffectHardLimiter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_ceiling_db", "ceiling"), &AudioEffectHardLimiter::set_ceiling_db);
	ClassDB::bind_method(D_METHOD("get_ceiling_db"), &AudioEffectHardLimiter::get_ceiling_db);

	ClassDB::bind_method(D_METHOD("set_pre_gain_db", "p_pre_gain"), &AudioEffectHardLimiter::set_pre_gain_db);
	ClassDB::bind_method(D_METHOD("get_pre_gain_db"), &AudioEffectHardLimiter::get_pre_gain_db);

	ClassDB::bind_method(D_METHOD("set_release", "p_release"), &AudioEffectHardLimiter::set_release);
	ClassDB::bind_method(D_METHOD("get_release"), &AudioEffectHardLimiter::get_release);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "pre_gain_db", PROPERTY_HINT_RANGE, "-24,24,0.01,suffix:dB"), "set_pre_gain_db", "get_pre_gain_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "ceiling_db", PROPERTY_HINT_RANGE, "-24,0.0,0.01,suffix:dB"), "set_ceiling_db", "get_ceiling_db");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "release", PROPERTY_HINT_RANGE, "0.01,3,0.01"), "set_release", "get_release");
}
