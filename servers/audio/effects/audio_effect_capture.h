/**************************************************************************/
/*  audio_effect_capture.h                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/math/audio_frame.h"
#include "core/object/ref_counted.h"
#include "core/templates/ring_buffer.h"
#include "servers/audio/audio_effect.h"
#include "servers/audio/audio_server.h"

class AudioEffectCapture;

class AudioEffectCaptureInstance : public AudioEffectInstance {
	GDCLASS(AudioEffectCaptureInstance, AudioEffectInstance);
	friend class AudioEffectCapture;

	RingBuffer<AudioFrame> buffer;
	SafeNumeric<uint64_t> discarded_frames;
	SafeNumeric<uint64_t> pushed_frames;
	float buffer_length_seconds = 0.1f;
	bool buffer_initialized = false;

	bool initialize_buffer();

protected:
	static void _bind_methods();

public:
	virtual void process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) override;
	virtual bool process_silence() const override;

	void set_buffer_length(float p_buffer_length_seconds);
	float get_buffer_length();

	bool can_get_buffer(int p_frames) const;
	PackedVector2Array get_buffer(int p_len);
	void clear_buffer();

	int get_frames_available() const;
	int64_t get_discarded_frames() const;
	int get_buffer_length_frames() const;
	int64_t get_pushed_frames() const;
};

class AudioEffectCapture : public AudioEffect {
	GDCLASS(AudioEffectCapture, AudioEffect)
	friend class AudioEffectCaptureInstance;

	float buffer_length_seconds = 0.1f;
	Vector<Ref<AudioEffectCaptureInstance>> instances;

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	bool _can_get_buffer_bind_compat_92532(int p_frames) const;
	PackedVector2Array _get_buffer_bind_compat_92532(int p_len);
	void _clear_buffer_bind_compat_92532();

	int _get_frames_available_bind_compat_92532() const;
	int64_t _get_discarded_frames_bind_compat_92532() const;
	int _get_buffer_length_frames_bind_compat_92532() const;
	int64_t _get_pushed_frames_bind_compat_92532() const;

	static void _bind_compatibility_methods();
#endif

public:
	virtual Ref<AudioEffectInstance> instantiate() override;

	virtual void set_channel_count(int p_channel_count) override;

	void set_buffer_length(float p_buffer_length_seconds);
	float get_buffer_length();

	bool can_get_buffer(int p_frames, int p_channel = 0) const;
	PackedVector2Array get_buffer(int p_len, int p_channel = 0);
	void clear_buffer(int p_channel = 0);

	int get_frames_available(int p_channel = 0) const;
	int64_t get_discarded_frames(int p_channel = 0) const;
	int get_buffer_length_frames(int p_channel = 0) const;
	int64_t get_pushed_frames(int p_channel = 0) const;
};
