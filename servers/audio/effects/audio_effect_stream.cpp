/*************************************************************************/
/*  audio_effect_stream.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_effect_stream.h"

void AudioEffectStream::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_audio_frames"), &AudioEffectStream::get_audio_frames);
	ClassDB::bind_method(D_METHOD("set_streaming_active", "active"), &AudioEffectStream::set_streaming_active);
	ClassDB::bind_method(D_METHOD("is_streaming_active"), &AudioEffectStream::is_streaming_active);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "streaming"), "set_streaming_active", "is_streaming_active");
}

Ref<AudioEffectInstance> AudioEffectStream::instance() {
	Ref<AudioEffectStreamInstance> ins;
	ins.instance();
	ins->base = Ref<AudioEffectStream>(this);
	ins->is_recording = false;

	//Re-using the buffer size calculations from audio_effect_delay.cpp
	float ring_buffer_max_size = IO_BUFFER_SIZE_MS;
	ring_buffer_max_size /= 1000.0; //convert to seconds
	ring_buffer_max_size *= AudioServer::get_singleton()->get_mix_rate();

	int ringbuff_size = ring_buffer_max_size;

	int bits = 0;

	while (ringbuff_size > 0) {
		bits++;
		ringbuff_size /= 2;
	}

	ringbuff_size = 1 << bits;
	ins->ring_buffer_mask = ringbuff_size - 1;
	ins->ring_buffer_pos = 0;

	ins->ring_buffer.resize(ringbuff_size);

	ins->ring_buffer_read_pos = 0;
	current_instance = ins;
	if (buffering_active) {
		ins->init();
	}

	return ins;
}

Vector<float> AudioEffectStream::get_audio_frames(int32_t p_frames) {
	return current_instance->get_audio_frames(p_frames);
}

void AudioEffectStream::set_streaming_active(bool p_active) {
	if (p_active) {
		if (current_instance == 0) {
			WARN_PRINTS("Streaming should not be set as active before Godot has initialized.");
			buffering_active = false;
			return;
		}

		buffering_active = true;
		current_instance->init();
	} else {
		buffering_active = false;
	}
}

bool AudioEffectStream::is_streaming_active() const {
	return buffering_active;
}

AudioEffectStream::AudioEffectStream() {
}

AudioEffectStream::~AudioEffectStream() {
}

Vector<float> AudioEffectStreamInstance::get_audio_frames(int32_t p_frames) {
	ERR_FAIL_COND_V(p_frames >= ring_buffer.size(), Vector<float>());
	int to_read = ring_buffer_pos - ring_buffer_read_pos;
	if (p_frames < to_read) {
		return Vector<float>();
	}
	to_read = p_frames;

	AudioFrame *rb_buf = ring_buffer.ptrw();
	Vector<float> streaming_data;

	while (to_read) {
		AudioFrame buffered_frame = rb_buf[ring_buffer_read_pos & ring_buffer_mask];
		streaming_data.push_back(buffered_frame.l);
		streaming_data.push_back(buffered_frame.r);

		ring_buffer_read_pos++;
		to_read--;
	}
	return streaming_data;
}

void AudioEffectStreamInstance::init() {
	//Reset recorder status
	ring_buffer_pos = 0;
	ring_buffer_read_pos = 0;

	is_recording = true;
}

void AudioEffectStreamInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	if (!is_recording) {
		for (int i = 0; i < p_frame_count; i++) {
			p_dst_frames[i] = p_src_frames[i];
		}
		return;
	}

	//Add incoming audio frames to the IO ring buffer
	const AudioFrame *src = p_src_frames;
	AudioFrame *rb_buf = ring_buffer.ptrw();
	for (int i = 0; i < p_frame_count; i++) {
		p_dst_frames[i] = p_src_frames[i];
		rb_buf[ring_buffer_pos & ring_buffer_mask] = src[i];
		ring_buffer_pos++;
	}
}

bool AudioEffectStreamInstance::process_silence() const {
	return true;
}

AudioEffectStreamInstance::AudioEffectStreamInstance() {
}

AudioEffectStreamInstance::~AudioEffectStreamInstance() {
}
