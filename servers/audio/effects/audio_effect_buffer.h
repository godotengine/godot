/*************************************************************************/
/*  audio_effect_buffer.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef AUDIOEFFECTBUFFER_H
#define AUDIOEFFECTBUFFER_H

#include "servers/audio/audio_effect.h"

class AudioEffectBuffer;

class AudioEffectBufferInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectBufferInstance, AudioEffectInstance)
	friend class AudioEffectBuffer;
	Ref<AudioEffectBuffer> base;

	Vector<Vector2> buffer_read;
	Vector<Vector2> buffer_write;

	Mutex *mutex;

	int read_pos;
	int read_size;
	int write_pos;
	int write_size;

	int avail_frames_to_read();
	int avail_frames_to_write();

	Vector2 read_frame();
	void write_frame(const Vector2 &p_frame);

	PoolVector2Array read_frames();
	void write_frames(const PoolVector2Array &p_frames);

public:
	virtual void process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count);

	AudioEffectBufferInstance();
	~AudioEffectBufferInstance();
};

class AudioEffectBuffer : public AudioEffect {
	GDCLASS(AudioEffectBuffer, AudioEffect)

	friend class AudioEffectBufferInstance;
	Ref<AudioEffectBufferInstance> current_instance;

protected:
	static void _bind_methods();

	int avail_frames_to_read();
	int avail_frames_to_write();

	Vector2 read_frame();
	void write_frame(const Vector2 &p_frame);

	PoolVector2Array read_frames();
	void write_frames(const PoolVector2Array &p_frames);

public:
	Ref<AudioEffectInstance> instance();
};

#endif // AUDIOEFFECTBUFFER_H
