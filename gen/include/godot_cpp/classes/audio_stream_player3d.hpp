/**************************************************************************/
/*  audio_stream_player3d.hpp                                             */
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

#include <godot_cpp/classes/audio_server.hpp>
#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioStream;
class AudioStreamPlayback;

class AudioStreamPlayer3D : public Node3D {
	GDEXTENSION_CLASS(AudioStreamPlayer3D, Node3D)

public:
	enum AttenuationModel {
		ATTENUATION_INVERSE_DISTANCE = 0,
		ATTENUATION_INVERSE_SQUARE_DISTANCE = 1,
		ATTENUATION_LOGARITHMIC = 2,
		ATTENUATION_DISABLED = 3,
	};

	enum DopplerTracking {
		DOPPLER_TRACKING_DISABLED = 0,
		DOPPLER_TRACKING_IDLE_STEP = 1,
		DOPPLER_TRACKING_PHYSICS_STEP = 2,
	};

	void set_stream(const Ref<AudioStream> &p_stream);
	Ref<AudioStream> get_stream() const;
	void set_volume_db(float p_volume_db);
	float get_volume_db() const;
	void set_volume_linear(float p_volume_linear);
	float get_volume_linear() const;
	void set_unit_size(float p_unit_size);
	float get_unit_size() const;
	void set_max_db(float p_max_db);
	float get_max_db() const;
	void set_pitch_scale(float p_pitch_scale);
	float get_pitch_scale() const;
	void play(float p_from_position = 0.0);
	void seek(float p_to_position);
	void stop();
	bool is_playing() const;
	float get_playback_position();
	void set_bus(const StringName &p_bus);
	StringName get_bus() const;
	void set_autoplay(bool p_enable);
	bool is_autoplay_enabled() const;
	void set_playing(bool p_enable);
	void set_max_distance(float p_meters);
	float get_max_distance() const;
	void set_area_mask(uint32_t p_mask);
	uint32_t get_area_mask() const;
	void set_emission_angle(float p_degrees);
	float get_emission_angle() const;
	void set_emission_angle_enabled(bool p_enabled);
	bool is_emission_angle_enabled() const;
	void set_emission_angle_filter_attenuation_db(float p_db);
	float get_emission_angle_filter_attenuation_db() const;
	void set_attenuation_filter_cutoff_hz(float p_degrees);
	float get_attenuation_filter_cutoff_hz() const;
	void set_attenuation_filter_db(float p_db);
	float get_attenuation_filter_db() const;
	void set_attenuation_model(AudioStreamPlayer3D::AttenuationModel p_model);
	AudioStreamPlayer3D::AttenuationModel get_attenuation_model() const;
	void set_doppler_tracking(AudioStreamPlayer3D::DopplerTracking p_mode);
	AudioStreamPlayer3D::DopplerTracking get_doppler_tracking() const;
	void set_stream_paused(bool p_pause);
	bool get_stream_paused() const;
	void set_max_polyphony(int32_t p_max_polyphony);
	int32_t get_max_polyphony() const;
	void set_panning_strength(float p_panning_strength);
	float get_panning_strength() const;
	bool has_stream_playback();
	Ref<AudioStreamPlayback> get_stream_playback();
	void set_playback_type(AudioServer::PlaybackType p_playback_type);
	AudioServer::PlaybackType get_playback_type() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AudioStreamPlayer3D::AttenuationModel);
VARIANT_ENUM_CAST(AudioStreamPlayer3D::DopplerTracking);

