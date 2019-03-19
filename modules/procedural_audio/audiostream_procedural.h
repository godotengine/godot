/*************************************************************************/
/*  audiostream_procedural.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef AUDIOSTREAM_PROCEDURAL_H
#define AUDIOSTREAM_PROCEDURAL_H

#include "core/io/stream_peer.h"
#include "core/reference.h"
#include "core/resource.h"
#include "servers/audio/audio_stream.h"

class AudioStreamPlaybackProcedural;

class AudioStreamProcedural : public AudioStream {
	GDCLASS(AudioStreamProcedural, AudioStream)
private:
	friend class AudioStreamPlaybackProcedural;
	uint64_t pos;
	int mix_rate;
	bool stereo;
	int buffer_frame_count;
	Set<AudioStreamPlaybackProcedural *> playbacks;
	void resize_buffer();

public:
	void reset();
	void set_position(uint64_t pos);
	uint64_t get_position();
	void set_stereo(bool stereo);
	bool get_stereo();
	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;
	virtual void generate_frames(Ref<StreamPeerBuffer> byte_buffer);
	virtual float get_length() const { return 0; }
	void set_mix_rate(int mix_rate);
	int get_mix_rate();
	void set_buffer_frame_count(int byte_buffer_size);
	int get_buffer_frame_count();
	AudioStreamProcedural();
	~AudioStreamProcedural();

protected:
	static void _bind_methods();
};

class AudioStreamPlaybackProcedural : public AudioStreamPlaybackResampled {
	GDCLASS(AudioStreamPlaybackProcedural, AudioStreamPlaybackResampled)
	friend class AudioStreamProcedural;

private:
	enum {
		MIX_FRAC_BITS = 13,
	};
	bool active;
	int internal_frame;
	int external_frame;
	Ref<AudioStreamProcedural> stream;
	Ref<StreamPeerBuffer> byte_buffer;
	void resize_buffer();

protected:
	virtual void _mix_internal(AudioFrame *p_buffer, int p_frames);

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;
	virtual int get_loop_count() const; // times it looped
	virtual float get_playback_position() const;
	virtual void seek(float p_time);
	virtual float get_length() const; // if supported, otherwise return 0
	virtual float get_stream_sampling_rate();
	AudioStreamPlaybackProcedural();
	~AudioStreamPlaybackProcedural();
};

#endif
