/**************************************************************************/
/*  audio_stream_with_effects.cpp                                         */
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

#include "audio_stream_with_effects.h"

#include "servers/audio_server.h"

Ref<AudioStreamPlayback> AudioStreamWithEffects::instantiate_playback() {
	Ref<AudioStreamPlaybackWithEffects> effect_playback;
	effect_playback.instantiate();
	effect_playback->stream = Ref<AudioStreamWithEffects>(this);

	// Start effect instances. Don't use _update_playback_effect() since that would call effect_instances.resize() a bunch.
	effect_playback->effect_instances.resize(effects.size());
	for (int i = 0; i < effects.size(); i++) {
		if (effects[i].effect.is_valid()) {
			effect_playback->effect_instances.write[i] = effects[i].effect->instantiate();
		}
	}

	playbacks.insert(effect_playback.operator->());
	return effect_playback;
}

double AudioStreamWithEffects::get_length() const {
	if (audio_stream.is_valid()) {
		return audio_stream->get_length();
	}
	return 0;
}

String AudioStreamWithEffects::get_stream_name() const {
	return "";
}

void AudioStreamWithEffects::set_stream(Ref<AudioStream> p_stream) {
	audio_stream = p_stream;
	for (AudioStreamPlaybackWithEffects *E : playbacks) {
		E->stop();
	}
}

Ref<AudioStream> AudioStreamWithEffects::get_stream() const {
	return audio_stream;
}

void AudioStreamWithEffects::set_effect(int p_index, Ref<AudioEffect> p_effect) {
	ERR_FAIL_INDEX(p_index, effects.size());

	if (effects[p_index].effect == p_effect) {
		return;
	}

	effects.write[p_index].effect = p_effect;

	for (AudioStreamPlaybackWithEffects *E : playbacks) {
		E->_update_playback_effect(p_index);
	}
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

Ref<AudioEffect> AudioStreamWithEffects::get_effect(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, effects.size(), Ref<AudioEffect>());
	return effects[p_index].effect;
}

// p_index_to is relative to the array prior to the removal of p_index_from.
// Example: [0, 1, 2, 3], move(1, 3) => [0, 2, 1, 3]
// Example: [0, 1, 2, 3], move(1, 4) => [0, 2, 3, 1]
void AudioStreamWithEffects::move_effect(int p_index_from, int p_index_to) {
	ERR_FAIL_INDEX(p_index_from, effects.size());
	// p_index_to == effects.size() is valid (move to end).
	ERR_FAIL_COND(p_index_to > effects.size());

	if (p_index_to < 0) {
		p_index_to = effects.size();
	}

	effects.insert(p_index_to, effects[p_index_from]);

	if (p_index_from > p_index_to) {
		// Moving the element backwards.
		effects.remove_at(p_index_from + 1);
		for (int i = p_index_to; i < p_index_from + 1; i++) {
			for (AudioStreamPlaybackWithEffects *E : playbacks) {
				E->_update_playback_effect(i);
			}
		}
	} else {
		// Moving the element forwards.
		effects.remove_at(p_index_from);
		for (int i = p_index_from; i < p_index_to; i++) {
			for (AudioStreamPlaybackWithEffects *E : playbacks) {
				E->_update_playback_effect(i);
			}
		}
	}

	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

void AudioStreamWithEffects::remove_effect(int p_index) {
	ERR_FAIL_INDEX(p_index, effects.size());
	effects.remove_at(p_index);

	for (int i = p_index; i < effects.size() + 1; i++) {
		for (AudioStreamPlaybackWithEffects *E : playbacks) {
			E->_update_playback_effect(i);
		}
	}
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

void AudioStreamWithEffects::add_effect(int p_index, Ref<AudioEffect> p_effect, bool p_bypass, bool p_process_when_bypassed) {
	ERR_FAIL_COND(p_index > effects.size());

	if (p_index < 0) {
		p_index = effects.size();
	}

	EffectEntry entry{ p_effect, p_bypass, p_process_when_bypassed };
	effects.insert(p_index, entry);

	for (int i = p_index; i < effects.size(); i++) {
		for (AudioStreamPlaybackWithEffects *E : playbacks) {
			E->_update_playback_effect(i);
		}
	}
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

void AudioStreamWithEffects::set_effect_bypass_enabled(int p_index, bool p_bypass) {
	ERR_FAIL_INDEX(p_index, effects.size());
	effects.write[p_index].bypass = p_bypass;
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

bool AudioStreamWithEffects::get_effect_bypass_enabled(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, effects.size(), false);
	return effects[p_index].bypass;
}

void AudioStreamWithEffects::set_effect_process_when_bypassed_enabled(int p_index, bool p_process_when_bypassed) {
	ERR_FAIL_INDEX(p_index, effects.size());
	effects.write[p_index].process_when_bypassed = p_process_when_bypassed;
	emit_signal(CoreStringName(changed));
	notify_property_list_changed();
}

bool AudioStreamWithEffects::get_effect_process_when_bypassed_enabled(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, effects.size(), false);
	return effects[p_index].process_when_bypassed;
}

void AudioStreamWithEffects::set_effect_count(int p_count) {
	int size_original = effects.size();

	effects.resize(p_count);

	// Remove effects when lowering effect count.
	for (int i = size_original - 1; i >= p_count; i--) {
		for (AudioStreamPlaybackWithEffects *E : playbacks) {
			E->_update_playback_effect(i);
		}
	}
}

int AudioStreamWithEffects::get_effect_count() const {
	return effects.size();
}

void AudioStreamWithEffects::set_tail_time(float p_time) {
	tail_time = MAX(0.0, p_time);
}

float AudioStreamWithEffects::get_tail_time() const {
	return tail_time;
}

void AudioStreamWithEffects::set_tail_fade_curve(float p_exponent) {
	tail_fade_curve = MAX(p_exponent, 0.0001f);
}

float AudioStreamWithEffects::get_tail_fade_curve() const {
	return tail_fade_curve;
}

void AudioStreamWithEffects::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_stream", "stream"), &AudioStreamWithEffects::set_stream);
	ClassDB::bind_method(D_METHOD("get_stream"), &AudioStreamWithEffects::get_stream);

	ClassDB::bind_method(D_METHOD("set_effect_count", "count"), &AudioStreamWithEffects::set_effect_count);
	ClassDB::bind_method(D_METHOD("get_effect_count"), &AudioStreamWithEffects::get_effect_count);

	ClassDB::bind_method(D_METHOD("set_effect", "index", "effect"), &AudioStreamWithEffects::set_effect);
	ClassDB::bind_method(D_METHOD("get_effect", "index"), &AudioStreamWithEffects::get_effect);

	ClassDB::bind_method(D_METHOD("add_effect", "index", "effect", "bypass", "process_when_bypassed"), &AudioStreamWithEffects::add_effect, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("move_effect", "index_from", "index_to"), &AudioStreamWithEffects::move_effect);
	ClassDB::bind_method(D_METHOD("remove_effect", "index"), &AudioStreamWithEffects::remove_effect);

	ClassDB::bind_method(D_METHOD("set_effect_bypass_enabled", "index", "enabled"), &AudioStreamWithEffects::set_effect_bypass_enabled);
	ClassDB::bind_method(D_METHOD("get_effect_bypass_enabled", "index"), &AudioStreamWithEffects::get_effect_bypass_enabled);
	ClassDB::bind_method(D_METHOD("set_effect_process_when_bypassed_enabled", "index", "enabled"), &AudioStreamWithEffects::set_effect_process_when_bypassed_enabled);
	ClassDB::bind_method(D_METHOD("get_effect_process_when_bypassed_enabled", "index"), &AudioStreamWithEffects::get_effect_process_when_bypassed_enabled);

	ClassDB::bind_method(D_METHOD("set_tail_time", "time"), &AudioStreamWithEffects::set_tail_time);
	ClassDB::bind_method(D_METHOD("get_tail_time"), &AudioStreamWithEffects::get_tail_time);
	ClassDB::bind_method(D_METHOD("set_tail_fade_curve", "time"), &AudioStreamWithEffects::set_tail_fade_curve);
	ClassDB::bind_method(D_METHOD("get_tail_fade_curve"), &AudioStreamWithEffects::get_tail_fade_curve);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stream", PROPERTY_HINT_RESOURCE_TYPE, "AudioStream", PROPERTY_USAGE_DEFAULT), "set_stream", "get_stream");

	ADD_GROUP("Tail", "tail");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tail_time", PROPERTY_HINT_RANGE, "0,10,or_greater,suffix:Seconds"), "set_tail_time", "get_tail_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "tail_fade_curve", PROPERTY_HINT_EXP_EASING, "attenuation"), "set_tail_fade_curve", "get_tail_fade_curve");

	ADD_ARRAY("effects", "effect_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "effect_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_effect_count", "get_effect_count");

	EffectEntry default_effect;

	base_property_helper.set_prefix("effect_");
	base_property_helper.set_array_length_getter(&AudioStreamWithEffects::get_effect_count);
	base_property_helper.register_property(PropertyInfo(Variant::OBJECT, "effect", PROPERTY_HINT_RESOURCE_TYPE, "AudioEffect"), default_effect.effect, &AudioStreamWithEffects::set_effect, &AudioStreamWithEffects::get_effect);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "bypass", PROPERTY_HINT_NONE), default_effect.bypass, &AudioStreamWithEffects::set_effect_bypass_enabled, &AudioStreamWithEffects::get_effect_bypass_enabled);
	base_property_helper.register_property(PropertyInfo(Variant::BOOL, "process_when_bypassed", PROPERTY_HINT_NONE), default_effect.process_when_bypassed, &AudioStreamWithEffects::set_effect_process_when_bypassed_enabled, &AudioStreamWithEffects::get_effect_process_when_bypassed_enabled);
	PropertyListHelper::register_base_helper(&base_property_helper);
}

AudioStreamWithEffects::AudioStreamWithEffects() {
	property_helper.setup_for_instance(base_property_helper, this);
}

//////////////////////
//////////////////////

void AudioStreamPlaybackWithEffects::start(double p_from_pos) {
	if (active) {
		return;
	}

	active = true;

	if (stream->audio_stream.is_valid()) {
		playback = stream->audio_stream->instantiate_playback();
	} else {
		stop();
	}

	if (playback.is_valid()) {
		playback->start(p_from_pos);
	} else {
		stop();
	}
}

void AudioStreamPlaybackWithEffects::stop() {
	effect_instances.clear();

	if (!active) {
		return;
	}

	active = false;

	if (playback.is_valid()) {
		playback->stop();
	}
}

void AudioStreamPlaybackWithEffects::seek(double p_time) {
	if (playback.is_valid()) {
		playback->seek(p_time);
	}
}

int AudioStreamPlaybackWithEffects::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	if (!active) {
		return 0;
	}

	if (playback.is_valid() && playback->is_playing()) {
		tail_mult_acc = 1.0;
	}

	bool stream_active = false;
	float tail_inc = 1.0 / (stream->tail_time * float(AudioServer::get_singleton()->get_mix_rate()));

	int mixed_samples = p_frames;
	int todo = p_frames;

	while (todo) {
		int to_mix = MIN(todo, MIX_BUFFER_SIZE);

		// Populate the mix buffer.
		if (playback.is_valid() && playback->is_playing()) {
			stream_active = true;
			playback->mix(mix_buffer, p_rate_scale, to_mix);
		} else {
			// Add silence to the mix buffer in case effects such as reverb are ringing out.
			for (int i = 0; i < to_mix; i++) {
				mix_buffer[i] = AudioFrame(0, 0);
			}
		}

		// Apply the effects.
		for (int i = 0; i < stream->effects.size(); i++) {
			if (stream->effects[i].bypass && !stream->effects[i].process_when_bypassed) {
				continue;
			}

			// Can evaluate to true when changing the size of stream->effects while playing.
			// Continuing if true prevents crashing.
			// TODO: Should we just break instead of continue? It only counts up.
			if (i >= effect_instances.size()) {
				continue;
			}

			if (effect_instances[i].is_valid()) {
				effect_instances[i]->process(mix_buffer, temp_buffer, to_mix);
				if (!stream->effects[i].bypass) {
					SWAP(mix_buffer, temp_buffer);
				}
			}
		}

		// Put the mixed samples into the buffer and apply tail fade if applicable.
		if (stream_active) {
			for (int i = 0; i < to_mix; i++) {
				p_buffer[i] = mix_buffer[i];
			}
		} else {
			for (int i = 0; i < to_mix; i++) {
				p_buffer[i] = mix_buffer[i];
				p_buffer[i] *= pow(CLAMP(tail_mult_acc, 0.0, 1.0), stream->tail_fade_curve);
				tail_mult_acc -= tail_inc;
			}
		}

		p_buffer += to_mix;
		todo -= to_mix;
	}

	if (!stream_active) {
		if (tail_mult_acc <= 0.0) {
			active = false;
		}
	}

	return mixed_samples;
}

void AudioStreamPlaybackWithEffects::tag_used_streams() {
	if (playback.is_valid()) {
		playback->tag_used_streams();
	}
	stream->tag_used(0);
}

int AudioStreamPlaybackWithEffects::get_loop_count() const {
	if (playback.is_valid()) {
		return playback->get_loop_count();
	}
	return 0;
}

double AudioStreamPlaybackWithEffects::get_playback_position() const {
	if (playback.is_valid()) {
		return playback->get_playback_position();
	}
	return 0;
}

bool AudioStreamPlaybackWithEffects::is_playing() const {
	return active;
}

AudioStreamPlaybackWithEffects::~AudioStreamPlaybackWithEffects() {
	if (stream.is_valid()) {
		stream->playbacks.erase(this);
	}
}

// Called when the effects list on the stream changes.
// This is needed to keep the stream playback playing when changing effects on the stream.
// This should only be called for effects that are moved/changed on the stream effects list,
// otherwise effects that don't need to be reset will be reset.
void AudioStreamPlaybackWithEffects::_update_playback_effect(int p_index) {
	if (!active || p_index >= stream->effects.size()) {
		return;
	}
	if (p_index >= effect_instances.size()) {
		effect_instances.resize(p_index + 1);
	}
	if (effect_instances[p_index].is_valid()) {
		effect_instances.set(p_index, nullptr);
	}
	if (stream->effects[p_index].effect.is_valid()) {
		effect_instances.write[p_index] = stream->effects[p_index].effect->instantiate();
	}
}
