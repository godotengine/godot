/**************************************************************************/
/*  audio_effect_bitcrusher.cpp                                           */
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

#include "audio_effect_bitcrusher.h"
#include "servers/audio_server.h"

void AudioEffectBitCrusherInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	//multiply volume interpolating to avoid clicks if this changes
	for (int i = 0; i < p_frame_count; i++) {
		AudioFrame current_frame = p_src_frames[i];
		
		// sample rate
		step_acc += 1.f;
		if (step_acc >= base->_sample_steps) {
			step_acc -= base->_sample_steps;
			last_frame = current_frame;
		}

		//dry & wet
		p_dst_frames[i] = current_frame * base->dry + last_frame * base->wet;
	}
}

Ref<AudioEffectInstance> AudioEffectBitCrusher::instantiate() {
	Ref<AudioEffectBitCrusherInstance> ins;
	ins.instantiate();
	ins->base = Ref<AudioEffectBitCrusher>(this);
	return ins;
}

void AudioEffectBitCrusher::set_dry(float d) {
	dry = d;
}

float AudioEffectBitCrusher::get_dry() const {
	return dry;
}

void AudioEffectBitCrusher::set_wet(float w) {
	wet = w;
}

float AudioEffectBitCrusher::get_wet() const {
	return wet;
}

void AudioEffectBitCrusher::set_samplerate(float s) {
	samplerate = s;
	_sample_steps = AudioServer::get_singleton()->get_mix_rate() / s;
}

float AudioEffectBitCrusher::get_samplerate() const {
	return samplerate;
}

void AudioEffectBitCrusher::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_dry", "dry"), &AudioEffectBitCrusher::set_dry);
	ClassDB::bind_method(D_METHOD("get_dry"), &AudioEffectBitCrusher::get_dry);

	ClassDB::bind_method(D_METHOD("set_wet", "wet"), &AudioEffectBitCrusher::set_wet);
	ClassDB::bind_method(D_METHOD("get_wet"), &AudioEffectBitCrusher::get_wet);

	ClassDB::bind_method(D_METHOD("set_samplerate", "samplerate"), &AudioEffectBitCrusher::set_samplerate);
	ClassDB::bind_method(D_METHOD("get_samplerate"), &AudioEffectBitCrusher::get_samplerate);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dry", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_dry", "get_dry");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "wet", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_wet", "get_wet");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "samplerate", PROPERTY_HINT_RANGE, "0,44100,0.1,suffix:Hz"), "set_samplerate", "get_samplerate");
}

AudioEffectBitCrusher::AudioEffectBitCrusher() {
	dry = 0.f;
	wet = 1.f;
	samplerate = AudioServer::get_singleton()->get_mix_rate();
}
