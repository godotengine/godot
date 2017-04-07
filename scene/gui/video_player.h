/*************************************************************************/
/*  video_player.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

class VideoPlayer : public Control {

	GDCLASS(VideoPlayer, Control);

	/*	struct InternalStream : public AudioServer::AudioStream {
		VideoPlayer *player;
		virtual int get_channel_count() const;
		virtual void set_mix_rate(int p_rate); //notify the stream of the mix rate
		virtual bool mix(int32_t *p_buffer,int p_frames);
		virtual void update();
	};
*/

	//	InternalStream internal_stream;
	Ref<VideoStreamPlayback> playback;
	Ref<VideoStream> stream;

	int sp_get_channel_count() const;
	void sp_set_mix_rate(int p_rate); //notify the stream of the mix rate
	bool sp_mix(int32_t *p_buffer, int p_frames);
	void sp_update();

	RID stream_rid;

	Ref<ImageTexture> texture;
	Image last_frame;

	AudioRBResampler resampler;

	bool paused;
	bool autoplay;
	float volume;
	double last_audio_time;
	bool expand;
	bool loops;
	int buffering_ms;
	int server_mix_rate;
	int audio_track;

	static int _audio_mix_callback(void *p_udata, const int16_t *p_data, int p_frames);

protected:
	static void _bind_methods();
	void _notification(int p_notification);

public:
	Size2 get_minimum_size() const;
	void set_expand(bool p_expand);
	bool has_expand() const;

	Ref<Texture> get_video_texture();

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
	float get_stream_pos() const;

	void set_autoplay(bool p_vol);
	bool has_autoplay() const;

	void set_audio_track(int p_track);
	int get_audio_track() const;

	void set_buffering_msec(int p_msec);
	int get_buffering_msec() const;

	VideoPlayer();
	~VideoPlayer();
};

#endif
