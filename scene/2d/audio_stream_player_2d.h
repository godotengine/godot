/*************************************************************************/
/*  audio_stream_player_2d.h                                             */
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

#ifndef AUDIO_STREAM_PLAYER_2D_H
#define AUDIO_STREAM_PLAYER_2D_H

#include "scene/2d/node_2d.h"
#include "servers/audio/audio_stream.h"
#include "servers/audio_server.h"

class AudioStreamPlayer2D : public Node2D {
	GDCLASS(AudioStreamPlayer2D, Node2D);

private:
	enum {
		MAX_OUTPUTS = 8,
		MAX_INTERSECT_AREAS = 32

	};

	struct Output {
		AudioFrame vol;
		int bus_index = 0;
		Viewport *viewport = nullptr; //pointer only used for reference to previous mix
	};

	Vector<Ref<AudioStreamPlayback>> stream_playbacks;
	Ref<AudioStream> stream;

	SafeFlag active{ false };
	SafeNumeric<float> setplay{ -1.0 };

	Vector<AudioFrame> volume_vector;

	uint64_t last_mix_count = -1;

	float volume_db = 0.0;
	float pitch_scale = 1.0;
	bool autoplay = false;
	StringName default_bus = SNAME("Master");
	int max_polyphony = 1;

	void _set_playing(bool p_enable);
	bool _is_active() const;

	StringName _get_actual_bus();
	void _update_panning();
	void _bus_layout_changed();

	static void _listener_changed_cb(void *self) { reinterpret_cast<AudioStreamPlayer2D *>(self)->_update_panning(); }

	uint32_t area_mask = 1;

	float max_distance = 2000.0;
	float attenuation = 1.0;

protected:
	void _validate_property(PropertyInfo &property) const override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_stream(Ref<AudioStream> p_stream);
	Ref<AudioStream> get_stream() const;

	void set_volume_db(float p_volume);
	float get_volume_db() const;

	void set_pitch_scale(float p_pitch_scale);
	float get_pitch_scale() const;

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

	void set_stream_paused(bool p_pause);
	bool get_stream_paused() const;

	void set_max_polyphony(int p_max_polyphony);
	int get_max_polyphony() const;

	Ref<AudioStreamPlayback> get_stream_playback();

	AudioStreamPlayer2D();
	~AudioStreamPlayer2D();
};

#endif
