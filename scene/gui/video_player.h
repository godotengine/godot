/*************************************************************************/
/*  video_player.h                                                       */
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

#ifndef VIDEO_PLAYER_H
#define VIDEO_PLAYER_H

#include "scene/gui/control.h"
#include "scene/resources/video_stream.h"
#include "servers/audio/audio_rb_resampler.h"
#include "servers/audio_server.h"

class VideoPlayer : public Control {
	GDCLASS(VideoPlayer, Control);

	struct Output {
		AudioFrame vol;
		int bus_index;
		Viewport *viewport; //pointer only used for reference to previous mix
	};
	Ref<VideoStreamPlayback> playback;
	Ref<VideoStream> stream;

	int sp_get_channel_count() const;
	bool mix(AudioFrame *p_buffer, int p_frames);

	RID stream_rid;

	Ref<ImageTexture> texture;

	AudioRBResampler resampler;
	Vector<AudioFrame> mix_buffer;
	int wait_resampler, wait_resampler_limit;

	bool paused;
	bool autoplay;
	float volume;
	double last_audio_time;
	bool expand;
	bool loops;
	int buffering_ms;
	int audio_track;
	int bus_index;

	StringName bus;

	void _mix_audio();
	static int _audio_mix_callback(void *p_udata, const float *p_data, int p_frames);
	static void _mix_audios(void *p_self);

protected:
	static void _bind_methods();
	void _notification(int p_notification);
	void _validate_property(PropertyInfo &p_property) const;

public:
	Size2 get_minimum_size() const;
	void set_expand(bool p_expand);
	bool has_expand() const;

	Ref<Texture> get_video_texture() const;

	void set_stream(const Ref<VideoStream> &p_stream);
	Ref<VideoStream> get_stream() const;

	void play();
	void stop();
	bool is_playing() const;

	void set_paused(bool p_paused);
	bool is_paused() const;

	void set_volume(float p_vol);
	float get_volume() const;

	void set_volume_db(float p_db);
	float get_volume_db() const;

	String get_stream_name() const;
	float get_stream_position() const;
	void set_stream_position(float p_position);

	void set_autoplay(bool p_enable);
	bool has_autoplay() const;

	void set_audio_track(int p_track);
	int get_audio_track() const;

	void set_buffering_msec(int p_msec);
	int get_buffering_msec() const;

	void set_bus(const StringName &p_bus);
	StringName get_bus() const;

	VideoPlayer();
	~VideoPlayer();
};

#endif // VIDEO_PLAYER_H
