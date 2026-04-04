/**************************************************************************/
/*  audio_server.hpp                                                      */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioBusLayout;
class AudioEffect;
class AudioEffectInstance;
class AudioStream;

class AudioServer : public Object {
	GDEXTENSION_CLASS(AudioServer, Object)

	static AudioServer *singleton;

public:
	enum SpeakerMode {
		SPEAKER_MODE_STEREO = 0,
		SPEAKER_SURROUND_31 = 1,
		SPEAKER_SURROUND_51 = 2,
		SPEAKER_SURROUND_71 = 3,
	};

	enum PlaybackType {
		PLAYBACK_TYPE_DEFAULT = 0,
		PLAYBACK_TYPE_STREAM = 1,
		PLAYBACK_TYPE_SAMPLE = 2,
		PLAYBACK_TYPE_MAX = 3,
	};

	static AudioServer *get_singleton();

	void set_bus_count(int32_t p_amount);
	int32_t get_bus_count() const;
	void remove_bus(int32_t p_index);
	void add_bus(int32_t p_at_position = -1);
	void move_bus(int32_t p_index, int32_t p_to_index);
	void set_bus_name(int32_t p_bus_idx, const String &p_name);
	String get_bus_name(int32_t p_bus_idx) const;
	int32_t get_bus_index(const StringName &p_bus_name) const;
	int32_t get_bus_channels(int32_t p_bus_idx) const;
	void set_bus_volume_db(int32_t p_bus_idx, float p_volume_db);
	float get_bus_volume_db(int32_t p_bus_idx) const;
	void set_bus_volume_linear(int32_t p_bus_idx, float p_volume_linear);
	float get_bus_volume_linear(int32_t p_bus_idx) const;
	void set_bus_send(int32_t p_bus_idx, const StringName &p_send);
	StringName get_bus_send(int32_t p_bus_idx) const;
	void set_bus_solo(int32_t p_bus_idx, bool p_enable);
	bool is_bus_solo(int32_t p_bus_idx) const;
	void set_bus_mute(int32_t p_bus_idx, bool p_enable);
	bool is_bus_mute(int32_t p_bus_idx) const;
	void set_bus_bypass_effects(int32_t p_bus_idx, bool p_enable);
	bool is_bus_bypassing_effects(int32_t p_bus_idx) const;
	void add_bus_effect(int32_t p_bus_idx, const Ref<AudioEffect> &p_effect, int32_t p_at_position = -1);
	void remove_bus_effect(int32_t p_bus_idx, int32_t p_effect_idx);
	int32_t get_bus_effect_count(int32_t p_bus_idx);
	Ref<AudioEffect> get_bus_effect(int32_t p_bus_idx, int32_t p_effect_idx);
	Ref<AudioEffectInstance> get_bus_effect_instance(int32_t p_bus_idx, int32_t p_effect_idx, int32_t p_channel = 0);
	void swap_bus_effects(int32_t p_bus_idx, int32_t p_effect_idx, int32_t p_by_effect_idx);
	void set_bus_effect_enabled(int32_t p_bus_idx, int32_t p_effect_idx, bool p_enabled);
	bool is_bus_effect_enabled(int32_t p_bus_idx, int32_t p_effect_idx) const;
	float get_bus_peak_volume_left_db(int32_t p_bus_idx, int32_t p_channel) const;
	float get_bus_peak_volume_right_db(int32_t p_bus_idx, int32_t p_channel) const;
	void set_playback_speed_scale(float p_scale);
	float get_playback_speed_scale() const;
	void lock();
	void unlock();
	AudioServer::SpeakerMode get_speaker_mode() const;
	float get_mix_rate() const;
	float get_input_mix_rate() const;
	String get_driver_name() const;
	PackedStringArray get_output_device_list();
	String get_output_device();
	void set_output_device(const String &p_name);
	double get_time_to_next_mix() const;
	double get_time_since_last_mix() const;
	double get_output_latency() const;
	PackedStringArray get_input_device_list();
	String get_input_device();
	void set_input_device(const String &p_name);
	Error set_input_device_active(bool p_active);
	int32_t get_input_frames_available();
	int32_t get_input_buffer_length_frames();
	PackedVector2Array get_input_frames(int32_t p_frames);
	void set_bus_layout(const Ref<AudioBusLayout> &p_bus_layout);
	Ref<AudioBusLayout> generate_bus_layout() const;
	void set_enable_tagging_used_audio_streams(bool p_enable);
	bool is_stream_registered_as_sample(const Ref<AudioStream> &p_stream);
	void register_stream_as_sample(const Ref<AudioStream> &p_stream);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~AudioServer();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AudioServer::SpeakerMode);
VARIANT_ENUM_CAST(AudioServer::PlaybackType);

