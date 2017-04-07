/*************************************************************************/
/*  audio_stream.h                                                       */
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
#ifndef AUDIO_STREAM_H
#define AUDIO_STREAM_H

#include "resource.h"
#include "servers/audio_server.h"

class AudioStreamPlayback : public Reference {

	GDCLASS(AudioStreamPlayback, Reference)

public:
	virtual void start(float p_from_pos = 0.0) = 0;
	virtual void stop() = 0;
	virtual bool is_playing() const = 0;

	virtual int get_loop_count() const = 0; //times it looped

	virtual float get_pos() const = 0;
	virtual void seek_pos(float p_time) = 0;

	virtual void mix(AudioFrame *p_bufer, float p_rate_scale, int p_frames) = 0;

	virtual float get_length() const = 0; //if supported, otherwise return 0
};

class AudioStreamPlaybackResampled : public AudioStreamPlayback {

	GDCLASS(AudioStreamPlaybackResampled, AudioStreamPlayback)

	enum {
		FP_BITS = 16, //fixed point used for resampling
		FP_LEN = (1 << FP_BITS),
		FP_MASK = FP_LEN - 1,
		INTERNAL_BUFFER_LEN = 256,
		CUBIC_INTERP_HISTORY = 4
	};

	AudioFrame internal_buffer[INTERNAL_BUFFER_LEN + CUBIC_INTERP_HISTORY];
	uint64_t mix_offset;

protected:
	void _begin_resample();
	virtual void _mix_internal(AudioFrame *p_bufer, int p_frames) = 0;
	virtual float get_stream_sampling_rate() = 0;

public:
	virtual void mix(AudioFrame *p_bufer, float p_rate_scale, int p_frames);

	AudioStreamPlaybackResampled() { mix_offset = 0; }
};

class AudioStream : public Resource {

	GDCLASS(AudioStream, Resource)
	OBJ_SAVE_TYPE(AudioStream) //children are all saved as AudioStream, so they can be exchanged

public:
	virtual Ref<AudioStreamPlayback> instance_playback() = 0;
	virtual String get_stream_name() const = 0;
};

#endif // AUDIO_STREAM_H
