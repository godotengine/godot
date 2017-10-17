#ifndef AUDIO_STREAM_PLAYER_2D_H
#define AUDIO_STREAM_PLAYER_2D_H

#include "scene/2d/node_2d.h"
#include "servers/audio/audio_stream.h"
#include "servers/audio_server.h"

class AudioStreamPlayer2D : public Node2D {

	GDCLASS(AudioStreamPlayer2D, Node2D)

private:
	enum {
		MAX_OUTPUTS = 8,
		MAX_INTERSECT_AREAS = 32

	};

	struct Output {

		AudioFrame vol;
		int bus_index;
		Viewport *viewport; //pointer only used for reference to previous mix
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

	float volume_db;
	bool autoplay;
	StringName bus;

	void _mix_audio();
	static void _mix_audios(void *self) { reinterpret_cast<AudioStreamPlayer2D *>(self)->_mix_audio(); }

	void _set_playing(bool p_enable);
	bool _is_active() const;

	void _bus_layout_changed();

	uint32_t area_mask;

	float max_distance;
	float attenuation;

protected:
	void _validate_property(PropertyInfo &property) const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_stream(Ref<AudioStream> p_stream);
	Ref<AudioStream> get_stream() const;

	void set_volume_db(float p_volume);
	float get_volume_db() const;

	void play(float p_from_pos = 0.0);
	void seek(float p_seconds);
	void stop();
	bool is_playing() const;
	float get_playback_position();

	void set_bus(const StringName &p_bus);
	StringName get_bus() const;

	void set_autoplay(bool p_enable);
	bool is_autoplay_enabled();

	void set_max_distance(float p_pixels);
	float get_max_distance() const;

	void set_attenuation(float p_curve);
	float get_attenuation() const;

	void set_area_mask(uint32_t p_mask);
	uint32_t get_area_mask() const;

	AudioStreamPlayer2D();
	~AudioStreamPlayer2D();
};

#endif
