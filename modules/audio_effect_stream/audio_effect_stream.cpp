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
	ClassDB::bind_method(D_METHOD("init"), &AudioEffectStream::init);
	ClassDB::bind_method(D_METHOD("is_streaming_active"), &AudioEffectStream::is_streaming_active);
}

Ref<AudioEffectInstance> AudioEffectStream::instance() {
	Ref<AudioEffectStreamInstance> ins;
	ins.instance();
	ins->base = Ref<AudioEffectStream>(this);
	current_instance = ins;

	return ins;
}

Ref<RingBufferAudioFrame> AudioEffectStream::init(int32_t p_ring_buffer_max_size) {
	if (current_instance == 0) {
		WARN_PRINT("Streaming should not be set as active before Godot has initialized.");
		buffering_active = false;
		return nullptr;
	}

	buffering_active = true;
	current_instance->output_ring_buffer.instance();
	int32_t higher_power = Math::ceil(Math::log(double(p_ring_buffer_max_size)) / Math::log(2.0));
	current_instance->output_ring_buffer->get().resize(higher_power);
	current_instance->set_streaming(true);
	return current_instance->output_ring_buffer;
}

bool AudioEffectStream::is_streaming_active() const {
	return buffering_active;
}

AudioEffectStream::AudioEffectStream() {
}

AudioEffectStream::~AudioEffectStream() {
	current_instance.unref();
}

void AudioEffectStreamInstance::init() {
	set_streaming(true);
}

void AudioEffectStreamInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	Ref<RingBufferAudioFrame> ring_buffer = output_ring_buffer;

	for (int i = 0; i < p_frame_count; i++) {
		p_dst_frames[i] = p_src_frames[i];
	}

	if (!get_streaming() || ring_buffer.is_null() || !ring_buffer->get().size()) {
		return;
	}

	//Add incoming audio frames to the IO ring buffer
	ring_buffer->get().write(p_src_frames, p_frame_count);
}

bool AudioEffectStreamInstance::process_silence() const {
	return true;
}

AudioEffectStreamInstance::AudioEffectStreamInstance() {
}

AudioEffectStreamInstance::~AudioEffectStreamInstance() {
	output_ring_buffer.unref();
}

void AudioEffectStreamInstance::set_streaming(bool val) {
	is_streaming = val;
}

bool AudioEffectStreamInstance::get_streaming() const {
	return is_streaming;
}
