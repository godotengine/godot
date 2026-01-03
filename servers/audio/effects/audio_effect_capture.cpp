/**************************************************************************/
/*  audio_effect_capture.cpp                                              */
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

#include "audio_effect_capture.h"
#include "audio_effect_capture.compat.inc"

#include "servers/audio/audio_server.h"

void AudioEffectCaptureInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {
	for (int i = 0; i < p_frame_count; i++) {
		p_dst_frames[i] = p_src_frames[i];
	}

	if (buffer.space_left() >= p_frame_count) {
		// Add incoming audio frames to the IO ring buffer
		int32_t ret = buffer.write(p_src_frames, p_frame_count);
		ERR_FAIL_COND_MSG(ret != p_frame_count, "Failed to add data to effect capture ring buffer despite sufficient space.");
		pushed_frames.add(p_frame_count);
	} else {
		discarded_frames.add(p_frame_count);
	}
}

bool AudioEffectCaptureInstance::process_silence() const {
	return true;
}

bool AudioEffectCaptureInstance::initialize_buffer() {
	if (!buffer_initialized) {
		float target_buffer_size = AudioServer::get_singleton()->get_mix_rate() * buffer_length_seconds;
		ERR_FAIL_COND_V(target_buffer_size <= 0 || target_buffer_size >= (1 << 27), false);
		buffer.resize(nearest_shift((uint32_t)target_buffer_size));
		buffer_initialized = true;
	}
	return true;
}

bool AudioEffectCaptureInstance::can_get_buffer(int p_frames) const {
	ERR_FAIL_COND_V(!buffer_initialized, false);
	return buffer.data_left() >= p_frames;
}

PackedVector2Array AudioEffectCaptureInstance::get_buffer(int p_frames) {
	ERR_FAIL_COND_V(!buffer_initialized, PackedVector2Array());
	ERR_FAIL_INDEX_V(p_frames, buffer.size(), PackedVector2Array());
	int data_left = buffer.data_left();
	if (data_left < p_frames || p_frames == 0) {
		return PackedVector2Array();
	}

	PackedVector2Array ret;
	ret.resize(p_frames);

	Vector<AudioFrame> streaming_data;
	streaming_data.resize(p_frames);
	buffer.read(streaming_data.ptrw(), p_frames);
	for (int32_t i = 0; i < p_frames; i++) {
		ret.write[i] = Vector2(streaming_data[i].left, streaming_data[i].right);
	}
	return ret;
}

void AudioEffectCaptureInstance::clear_buffer() {
	const int32_t data_left = buffer.data_left();
	buffer.advance_read(data_left);
}

void AudioEffectCaptureInstance::set_buffer_length(float p_buffer_length_seconds) {
	buffer_length_seconds = p_buffer_length_seconds;
}

float AudioEffectCaptureInstance::get_buffer_length() {
	return buffer_length_seconds;
}

int AudioEffectCaptureInstance::get_frames_available() const {
	ERR_FAIL_COND_V(!buffer_initialized, 0);
	return buffer.data_left();
}

int64_t AudioEffectCaptureInstance::get_discarded_frames() const {
	ERR_FAIL_COND_V(!buffer_initialized, 0);
	return discarded_frames.get();
}

int AudioEffectCaptureInstance::get_buffer_length_frames() const {
	ERR_FAIL_COND_V(!buffer_initialized, 0);
	return buffer.size();
}

int64_t AudioEffectCaptureInstance::get_pushed_frames() const {
	ERR_FAIL_COND_V(!buffer_initialized, 0);
	return pushed_frames.get();
}

void AudioEffectCaptureInstance::_bind_methods() {
	ClassDB::bind_method(D_METHOD("can_get_buffer", "frames"), &AudioEffectCaptureInstance::can_get_buffer);
	ClassDB::bind_method(D_METHOD("get_buffer", "frames"), &AudioEffectCaptureInstance::get_buffer);
	ClassDB::bind_method(D_METHOD("clear_buffer"), &AudioEffectCaptureInstance::clear_buffer);
	ClassDB::bind_method(D_METHOD("set_buffer_length", "buffer_length_seconds"), &AudioEffectCaptureInstance::set_buffer_length);
	ClassDB::bind_method(D_METHOD("get_buffer_length"), &AudioEffectCaptureInstance::get_buffer_length);
	ClassDB::bind_method(D_METHOD("get_frames_available"), &AudioEffectCaptureInstance::get_frames_available);
	ClassDB::bind_method(D_METHOD("get_discarded_frames"), &AudioEffectCaptureInstance::get_discarded_frames);
	ClassDB::bind_method(D_METHOD("get_buffer_length_frames"), &AudioEffectCaptureInstance::get_buffer_length_frames);
	ClassDB::bind_method(D_METHOD("get_pushed_frames"), &AudioEffectCaptureInstance::get_pushed_frames);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "buffer_length", PROPERTY_HINT_RANGE, "0.01,10,0.01,suffix:s"), "set_buffer_length", "get_buffer_length");
}

Ref<AudioEffectInstance> AudioEffectCapture::instantiate() {
	ERR_FAIL_INDEX_V(get_channel(), instances.size(), Ref<AudioEffectInstance>());
	Ref<AudioEffectCaptureInstance> ins;
	ins.instantiate();
	ins->buffer_length_seconds = buffer_length_seconds;
	ERR_FAIL_COND_V(!ins->initialize_buffer(), Ref<AudioEffectInstance>());
	instances.write[get_channel()] = ins;

	return ins;
}

void AudioEffectCapture::set_channel_count(int p_channel_count) {
	instances.resize(p_channel_count);
}

void AudioEffectCapture::set_buffer_length(float p_buffer_length_seconds) {
	buffer_length_seconds = p_buffer_length_seconds;
	for (int i = 0; i < instances.size(); i++) {
		if (instances[i].is_valid()) {
			instances.write[i]->set_buffer_length(p_buffer_length_seconds);
		}
	}
}

float AudioEffectCapture::get_buffer_length() {
	return buffer_length_seconds;
}

int AudioEffectCapture::get_frames_available(int p_channel) const {
	ERR_FAIL_INDEX_V(p_channel, instances.size(), 0);
	ERR_FAIL_COND_V(instances[p_channel].is_null(), 0);
	return instances[p_channel]->get_frames_available();
}

int64_t AudioEffectCapture::get_discarded_frames(int p_channel) const {
	ERR_FAIL_INDEX_V(p_channel, instances.size(), 0);
	ERR_FAIL_COND_V(instances[p_channel].is_null(), 0);
	return instances[p_channel]->get_discarded_frames();
}

int AudioEffectCapture::get_buffer_length_frames(int p_channel) const {
	ERR_FAIL_INDEX_V(p_channel, instances.size(), 0);
	ERR_FAIL_COND_V(instances[p_channel].is_null(), 0);
	return instances[p_channel]->get_buffer_length_frames();
}

int64_t AudioEffectCapture::get_pushed_frames(int p_channel) const {
	ERR_FAIL_INDEX_V(p_channel, instances.size(), 0);
	ERR_FAIL_COND_V(instances[p_channel].is_null(), 0);
	return instances[p_channel]->get_pushed_frames();
}

bool AudioEffectCapture::can_get_buffer(int p_frames, int p_channel) const {
	ERR_FAIL_INDEX_V(p_channel, instances.size(), false);
	ERR_FAIL_COND_V(instances[p_channel].is_null(), false);
	return instances[p_channel]->can_get_buffer(p_frames);
}

PackedVector2Array AudioEffectCapture::get_buffer(int p_frames, int p_channel) {
	ERR_FAIL_INDEX_V(p_channel, instances.size(), PackedVector2Array());
	ERR_FAIL_COND_V(instances[p_channel].is_null(), PackedVector2Array());
	return instances.write[p_channel]->get_buffer(p_frames);
}

void AudioEffectCapture::clear_buffer(int p_channel) {
	ERR_FAIL_INDEX(p_channel, instances.size());
	ERR_FAIL_COND(instances[p_channel].is_null());
	instances.write[p_channel]->clear_buffer();
}

void AudioEffectCapture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("can_get_buffer", "frames", "channel"), &AudioEffectCapture::can_get_buffer, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_buffer", "frames", "channel"), &AudioEffectCapture::get_buffer, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("clear_buffer", "channel"), &AudioEffectCapture::clear_buffer, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("set_buffer_length", "buffer_length_seconds"), &AudioEffectCapture::set_buffer_length);
	ClassDB::bind_method(D_METHOD("get_buffer_length"), &AudioEffectCapture::get_buffer_length);
	ClassDB::bind_method(D_METHOD("get_frames_available", "channel"), &AudioEffectCapture::get_frames_available, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_discarded_frames", "channel"), &AudioEffectCapture::get_discarded_frames, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_buffer_length_frames", "channel"), &AudioEffectCapture::get_buffer_length_frames, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("get_pushed_frames", "channel"), &AudioEffectCapture::get_pushed_frames, DEFVAL(0));

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "buffer_length", PROPERTY_HINT_RANGE, "0.01,10,0.01,suffix:s"), "set_buffer_length", "get_buffer_length");
}
