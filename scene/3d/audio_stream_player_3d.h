/**************************************************************************/
/*  audio_stream_player_3d.h                                              */
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

#ifndef AUDIO_STREAM_PLAYER_3D_H
#define AUDIO_STREAM_PLAYER_3D_H

#include "core/os/mutex.h"
#include "scene/3d/area_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/velocity_tracker_3d.h"
#include "servers/audio/audio_filter_sw.h"
#include "servers/audio/audio_stream.h"
#include "servers/audio_server.h"

class Camera3D;
class AudioStreamPlayer3D : public Node3D {
	GDCLASS(AudioStreamPlayer3D, Node3D);

public:
	enum AttenuationModel {
		ATTENUATION_INVERSE_DISTANCE,
		ATTENUATION_INVERSE_SQUARE_DISTANCE,
		ATTENUATION_LOGARITHMIC,
		ATTENUATION_DISABLED,
	};

	enum DopplerTracking {
		DOPPLER_TRACKING_DISABLED,
		DOPPLER_TRACKING_IDLE_STEP,
		DOPPLER_TRACKING_PHYSICS_STEP
	};

private:
	enum {
		MAX_OUTPUTS = 8,
		MAX_INTERSECT_AREAS = 32

	};

	Vector<Ref<AudioStreamPlayback>> stream_playbacks;
	Ref<AudioStream> stream;

	SafeFlag active{ false };
	SafeNumeric<float> setplay{ -1.0 };
	Ref<AudioStreamPlayback> setplayback;

	AttenuationModel attenuation_model = ATTENUATION_INVERSE_DISTANCE;
	float volume_db = 0.0;
	float unit_size = 10.0;
	float max_db = 3.0;
	float pitch_scale = 1.0;
	// Internally used to take doppler tracking into account.
	float actual_pitch_scale = 1.0;
	bool autoplay = false;
	StringName bus = SNAME("Master");
	int max_polyphony = 1;

	uint64_t last_mix_count = -1;
	bool force_update_panning = false;

	static void _calc_output_vol(const Vector3 &source_dir, real_t tightness, Vector<AudioFrame> &output);

	void _calc_reverb_vol(Area3D *area, Vector3 listener_area_pos, Vector<AudioFrame> direct_path_vol, Vector<AudioFrame> &reverb_vol);

	static void _listener_changed_cb(void *self) { reinterpret_cast<AudioStreamPlayer3D *>(self)->force_update_panning = true; }

	void _set_playing(bool p_enable);
	bool _is_active() const;
	StringName _get_actual_bus();
	Area3D *_get_overriding_area();
	Vector<AudioFrame> _update_panning();

	void _on_bus_layout_changed();
	void _on_bus_renamed(int p_bus_index, const StringName &p_old_name, const StringName &p_new_name);

	uint32_t area_mask = 1;

	bool emission_angle_enabled = false;
	float emission_angle = 45.0;
	float emission_angle_filter_attenuation_db = -12.0;
	float attenuation_filter_cutoff_hz = 5000.0;
	float attenuation_filter_db = -24.0;

	float linear_attenuation = 0;

	float max_distance = 0.0;

	Ref<VelocityTracker3D> velocity_tracker;

	DopplerTracking doppler_tracking = DOPPLER_TRACKING_DISABLED;

	float _get_attenuation_db(float p_distance) const;

	float panning_strength = 1.0f;
	float cached_global_panning_strength = 0.5f;

protected:
	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_stream(Ref<AudioStream> p_stream);
	Ref<AudioStream> get_stream() const;

	void set_volume_db(float p_volume);
	float get_volume_db() const;

	void set_unit_size(float p_volume);
	float get_unit_size() const;

	void set_max_db(float p_boost);
	float get_max_db() const;

	void set_pitch_scale(float p_pitch_scale);
	float get_pitch_scale() const;

	void play(float p_from_pos = 0.0);
	void seek(float p_seconds);
	void stop();
	bool is_playing() const;
	float get_playback_position();

	void set_bus(const StringName &p_bus);
	StringName get_bus() const;

	void set_max_polyphony(int p_max_polyphony);
	int get_max_polyphony() const;

	void set_autoplay(bool p_enable);
	bool is_autoplay_enabled();

	void set_max_distance(float p_metres);
	float get_max_distance() const;

	void set_area_mask(uint32_t p_mask);
	uint32_t get_area_mask() const;

	void set_emission_angle_enabled(bool p_enable);
	bool is_emission_angle_enabled() const;

	void set_emission_angle(float p_angle);
	float get_emission_angle() const;

	void set_emission_angle_filter_attenuation_db(float p_angle_attenuation_db);
	float get_emission_angle_filter_attenuation_db() const;

	void set_attenuation_filter_cutoff_hz(float p_hz);
	float get_attenuation_filter_cutoff_hz() const;

	void set_attenuation_filter_db(float p_db);
	float get_attenuation_filter_db() const;

	void set_attenuation_model(AttenuationModel p_model);
	AttenuationModel get_attenuation_model() const;

	void set_doppler_tracking(DopplerTracking p_tracking);
	DopplerTracking get_doppler_tracking() const;

	void set_stream_paused(bool p_pause);
	bool get_stream_paused() const;

	void set_panning_strength(float p_panning_strength);
	float get_panning_strength() const;

	bool has_stream_playback();
	Ref<AudioStreamPlayback> get_stream_playback();

	AudioStreamPlayer3D();
	~AudioStreamPlayer3D();
};

VARIANT_ENUM_CAST(AudioStreamPlayer3D::AttenuationModel)
VARIANT_ENUM_CAST(AudioStreamPlayer3D::DopplerTracking)

#endif // AUDIO_STREAM_PLAYER_3D_H
