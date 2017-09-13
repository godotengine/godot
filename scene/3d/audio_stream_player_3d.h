#ifndef AUDIO_STREAM_PLAYER_3D_H
#define AUDIO_STREAM_PLAYER_3D_H

#include "scene/3d/spatial.h"
#include "scene/3d/spatial_velocity_tracker.h"
#include "servers/audio/audio_filter_sw.h"
#include "servers/audio/audio_stream.h"
#include "servers/audio_server.h"

class Camera;
class AudioStreamPlayer3D : public Spatial {

	GDCLASS(AudioStreamPlayer3D, Spatial)
public:
	enum AttenuationModel {
		ATTENUATION_INVERSE_DISTANCE,
		ATTENUATION_INVERSE_SQUARE_DISTANCE,
		ATTENUATION_LOGARITHMIC,
	};

	enum OutOfRangeMode {
		OUT_OF_RANGE_MIX,
		OUT_OF_RANGE_PAUSE,
	};

	enum DopplerTracking {
		DOPPLER_TRACKING_DISABLED,
		DOPPLER_TRACKING_IDLE_STEP,
		DOPPLER_TRACKING_FIXED_STEP
	};

private:
	enum {
		MAX_OUTPUTS = 8,
		MAX_INTERSECT_AREAS = 32

	};

	struct Output {

		AudioFilterSW filter;
		AudioFilterSW::Processor filter_process[6];
		AudioFrame vol[4];
		float filter_gain;
		float pitch_scale;
		int bus_index;
		int reverb_bus_index;
		AudioFrame reverb_vol[4];
		Viewport *viewport; //pointer only used for reference to previous mix

		Output() {
			filter_gain = 0;
			viewport = NULL;
			reverb_bus_index = -1;
			bus_index = -1;
		}
	};

	Output outputs[MAX_OUTPUTS];
	volatile int output_count;
	volatile bool output_ready;

	//these are used by audio thread to have a reference of previous volumes (for ramping volume and avoiding clicks)
	Output prev_outputs[MAX_OUTPUTS];
	int prev_output_count;

	Ref<AudioStreamPlayback> stream_playback;
	Ref<AudioStream> stream;
	Vector<AudioFrame> mix_buffer;

	volatile float setseek;
	volatile bool active;
	volatile float setplay;

	AttenuationModel attenuation_model;
	float unit_db;
	float unit_size;
	float max_db;
	bool autoplay;
	StringName bus;

	void _mix_audio();
	static void _mix_audios(void *self) { reinterpret_cast<AudioStreamPlayer3D *>(self)->_mix_audio(); }

	void _set_playing(bool p_enable);
	bool _is_active() const;

	void _bus_layout_changed();

	uint32_t area_mask;

	bool emission_angle_enabled;
	float emission_angle;
	float emission_angle_filter_attenuation_db;
	float attenuation_filter_cutoff_hz;
	float attenuation_filter_db;

	float max_distance;

	Ref<SpatialVelocityTracker> velocity_tracker;

	DopplerTracking doppler_tracking;

	OutOfRangeMode out_of_range_mode;

	float _get_attenuation_db(float p_distance) const;

protected:
	void _validate_property(PropertyInfo &property) const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_stream(Ref<AudioStream> p_stream);
	Ref<AudioStream> get_stream() const;

	void set_unit_db(float p_volume);
	float get_unit_db() const;

	void set_unit_size(float p_volume);
	float get_unit_size() const;

	void set_max_db(float p_boost);
	float get_max_db() const;

	void play(float p_from_pos = 0.0);
	void seek(float p_seconds);
	void stop();
	bool is_playing() const;
	float get_pos();

	void set_bus(const StringName &p_bus);
	StringName get_bus() const;

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

	void set_out_of_range_mode(OutOfRangeMode p_mode);
	OutOfRangeMode get_out_of_range_mode() const;

	void set_doppler_tracking(DopplerTracking p_tracking);
	DopplerTracking get_doppler_tracking() const;

	AudioStreamPlayer3D();
	~AudioStreamPlayer3D();
};

VARIANT_ENUM_CAST(AudioStreamPlayer3D::AttenuationModel)
VARIANT_ENUM_CAST(AudioStreamPlayer3D::OutOfRangeMode)
VARIANT_ENUM_CAST(AudioStreamPlayer3D::DopplerTracking)
#endif // AUDIO_STREAM_PLAYER_3D_H
