/**************************************************************************/
/*  audio_stream_with_effects.h                                           */
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

#pragma once

#include "servers/audio/audio_stream.h"

class AudioStreamPlaybackWithEffects;

class AudioStreamWithEffects : public AudioStream {
	GDCLASS(AudioStreamWithEffects, AudioStream)

	AudioServer::PlaybackType playback_type;

private:
	friend class AudioStreamPlaybackWithEffects;

	struct EffectEntry {
		Ref<AudioEffect> effect;
		bool bypass = false;
	};

	static inline PropertyListHelper base_property_helper;
	PropertyListHelper property_helper;

	Ref<AudioStream> audio_stream; // Child Stream
	Vector<EffectEntry> effects;

	HashSet<AudioStreamPlaybackWithEffects *> playbacks;

	float tail_time = 0.0;
	float tail_fade_curve = 1.0;

public:
	virtual Ref<AudioStreamPlayback> instantiate_playback() override;
	virtual String get_stream_name() const override;
	virtual double get_length() const override;
	virtual bool is_meta_stream() const override { return false; }

	void set_stream(Ref<AudioStream>);
	Ref<AudioStream> get_stream() const;

	void set_effect(int p_index, Ref<AudioEffect> p_effect);
	Ref<AudioEffect> get_effect(int p_index) const;

	void add_effect(int p_index, Ref<AudioEffect> p_effect, bool p_bypass = false);
	void move_effect(int p_index_from, int p_index_to);
	void remove_effect(int p_index);

	void set_effect_bypass_enabled(int p_index, bool p_bypass);
	bool get_effect_bypass_enabled(int p_index) const;

	void set_effect_count(int p_count);
	int get_effect_count() const;

	void set_tail_time(float p_time);
	float get_tail_time() const;

	void set_tail_fade_curve(float p_exponent);
	float get_tail_fade_curve() const;

	AudioStreamWithEffects();

protected:
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value) { return property_helper.property_set_value(p_name, p_value); }
	bool _get(const StringName &p_name, Variant &r_ret) const { return property_helper.property_get_value(p_name, r_ret); }
	void _get_property_list(List<PropertyInfo> *p_list) const { property_helper.get_property_list(p_list); }
	bool _property_can_revert(const StringName &p_name) const { return property_helper.property_can_revert(p_name); }
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const { return property_helper.property_get_revert(p_name, r_property); }
};

class AudioStreamPlaybackWithEffects : public AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackWithEffects, AudioStreamPlayback)
	friend class AudioStreamWithEffects;

private:
	enum {
		MIX_BUFFER_SIZE = 128
	};

	Ref<AudioStreamWithEffects> stream; // The effect stream this playback is instantiated from
	Ref<AudioStreamPlayback> playback; // The playback of the child stream
	LocalVector<Ref<AudioEffectInstance>> effect_instances;

	void _update_playback_effect(int p_index);

	bool active = false;
	AudioFrame mix_buffer[MIX_BUFFER_SIZE];
	AudioFrame temp_buffer[MIX_BUFFER_SIZE];

	float tail_mult_acc = 0.0;

public:
	virtual void start(double p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;
	virtual int get_loop_count() const override; // times it looped
	virtual double get_playback_position() const override;
	virtual void seek(double p_time) override;
	virtual int mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override;

	virtual void tag_used_streams() override;

	Ref<AudioEffectInstance> get_effect_instance(int p_index);
	int get_effect_instance_count() const;

	~AudioStreamPlaybackWithEffects();

protected:
	static void _bind_methods();
};
