/*************************************************************************/
/*  spatial_stream_player.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SPATIAL_STREAM_PLAYER_H
#define SPATIAL_STREAM_PLAYER_H

#include "scene/resources/audio_stream.h"
#include "scene/3d/spatial_player.h"
#include "servers/audio/audio_rb_resampler.h"

class SpatialStreamPlayer : public SpatialPlayer {

	GDCLASS(SpatialStreamPlayer,SpatialPlayer);

	_THREAD_SAFE_CLASS_

	struct InternalStream : public AudioServer::AudioStream {
		SpatialStreamPlayer *player;
		virtual int get_channel_count() const;
		virtual void set_mix_rate(int p_rate); //notify the stream of the mix rate
		virtual bool mix(int32_t *p_buffer,int p_frames);
		virtual void update();
	};


	InternalStream internal_stream;
	Ref<AudioStreamPlayback> playback;
	Ref<AudioStream> stream;

	int sp_get_channel_count() const;
	void sp_set_mix_rate(int p_rate); //notify the stream of the mix rate
	bool sp_mix(int32_t *p_buffer,int p_frames);
	void sp_update();

	int server_mix_rate;

	RID stream_rid;
	bool paused;
	bool autoplay;
	bool loops;
	float volume;
	float loop_point;
	int buffering_ms;

	AudioRBResampler resampler;

	bool _play;
	void _set_play(bool p_play);
	bool _get_play() const;
protected:
	void _notification(int p_what);

	static void _bind_methods();
public:

	void set_stream(const Ref<AudioStream> &p_stream);
	Ref<AudioStream> get_stream() const;

	void play(float p_from_offset=0);
	void stop();
	bool is_playing() const;

	void set_paused(bool p_paused);
	bool is_paused() const;

	void set_loop(bool p_enable);
	bool has_loop() const;

	void set_volume(float p_vol);
	float get_volume() const;

	void set_loop_restart_time(float p_secs);
	float get_loop_restart_time() const;

	void set_volume_db(float p_db);
	float get_volume_db() const;

	String get_stream_name() const;

	int get_loop_count() const;

	float get_pos() const;
	void seek_pos(float p_time);
	float get_length() const;
	void set_autoplay(bool p_vol);
	bool has_autoplay() const;

	void set_buffering_msec(int p_msec);
	int get_buffering_msec() const;

	SpatialStreamPlayer();
	~SpatialStreamPlayer();
};

#endif // SPATIAL_STREAM_PLAYER_H
