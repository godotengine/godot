/*************************************************************************/
/*  audio_stream_ogg_vorbis.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef AUDIO_STREAM_STB_VORBIS_H
#define AUDIO_STREAM_STB_VORBIS_H

#include "io/resource_loader.h"
#include "servers/audio/audio_stream.h"

#define STB_VORBIS_HEADER_ONLY
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "thirdparty/misc/stb_vorbis.c"
#pragma GCC diagnostic pop
#undef STB_VORBIS_HEADER_ONLY

class AudioStreamOGGVorbis;

class AudioStreamPlaybackOGGVorbis : public AudioStreamPlaybackResampled {

	GDCLASS(AudioStreamPlaybackOGGVorbis, AudioStreamPlaybackResampled)

	stb_vorbis *ogg_stream;
	stb_vorbis_alloc ogg_alloc;
	uint32_t frames_mixed;
	bool active;
	int loops;

	friend class AudioStreamOGGVorbis;

	Ref<AudioStreamOGGVorbis> vorbis_stream;

protected:
	virtual void _mix_internal(AudioFrame *p_buffer, int p_frames);
	virtual float get_stream_sampling_rate();

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;

	virtual int get_loop_count() const; //times it looped

	virtual float get_playback_position() const;
	virtual void seek(float p_time);

	virtual float get_length() const; //if supported, otherwise return 0

	AudioStreamPlaybackOGGVorbis() {}
	~AudioStreamPlaybackOGGVorbis();
};

class AudioStreamOGGVorbis : public AudioStream {

	GDCLASS(AudioStreamOGGVorbis, AudioStream)
	OBJ_SAVE_TYPE(AudioStream) //children are all saved as AudioStream, so they can be exchanged
	RES_BASE_EXTENSION("oggstr");

	friend class AudioStreamPlaybackOGGVorbis;

	void *data;
	uint32_t data_len;

	int decode_mem_size;
	float sample_rate;
	int channels;
	float length;
	bool loop;
	float loop_offset;

protected:
	static void _bind_methods();

public:
	void set_loop(bool p_enable);
	bool has_loop() const;

	void set_loop_offset(float p_seconds);
	float get_loop_offset() const;

	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;

	void set_data(const PoolVector<uint8_t> &p_data);
	PoolVector<uint8_t> get_data() const;

	AudioStreamOGGVorbis();
};

#endif
