/*************************************************************************/
/*  audio_effect_auffer.cpp                                              */
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

#include "audio_effect_buffer.h"

#define BUFFER_SIZE 1024

void AudioEffectBufferInstance::process(const AudioFrame *p_src_frames, AudioFrame *p_dst_frames, int p_frame_count) {

	mutex->lock();
	read_pos = 0;
	read_size = 0;

	for (int i = 0; i < p_frame_count; i++) {
		AudioFrame frame = p_src_frames[i];

		if (read_size < buffer_read.size()) {
			buffer_read.write[read_size++] = Vector2(frame.l, frame.r);
		}

		if (i < buffer_write.size()) {
			frame += AudioFrame(buffer_write[i].x, buffer_write[i].y);
		}

		p_dst_frames[i] = frame;
	}

	write_pos = 0;
	mutex->unlock();
}

AudioEffectBufferInstance::AudioEffectBufferInstance() {

	mutex = Mutex::create();

	buffer_read.resize(BUFFER_SIZE);
	buffer_write.resize(BUFFER_SIZE);

	read_pos = 0;
	read_size = 0;
	write_pos = 0;
	write_size = buffer_write.size();
}

int AudioEffectBufferInstance::avail_frames_to_read() {

	return read_size - read_pos;
}

int AudioEffectBufferInstance::avail_frames_to_write() {

	return write_size - write_pos;
}

Vector2 AudioEffectBufferInstance::read_frame() {

	Vector2 frame;

	mutex->lock();

	if (read_pos < read_size) {
		frame = buffer_read[read_pos++];
	}

	mutex->unlock();
	return frame;
}

void AudioEffectBufferInstance::write_frame(const Vector2 &p_frame) {

	mutex->lock();

	if (write_pos < write_size) {
		buffer_write.write[write_pos++] = p_frame;
	}

	mutex->unlock();
}

PoolVector2Array AudioEffectBufferInstance::read_frames() {

	PoolVector2Array frames;

	mutex->lock();

	frames.resize(avail_frames_to_read());
	PoolVector2Array::Write w = frames.write();
	for (int i = 0; read_pos < read_size; i++) {
		w[i] = buffer_read[read_pos++];
	}

	mutex->unlock();
	return frames;
}

void AudioEffectBufferInstance::write_frames(const PoolVector2Array &p_frames) {

	mutex->lock();

	for (int i = 0; write_pos < write_size; i++) {
		buffer_write.write[write_pos++] = p_frames[i];
	}

	mutex->unlock();
}

AudioEffectBufferInstance::~AudioEffectBufferInstance() {

	if (mutex) {
		memdelete(mutex);
		mutex = NULL;
	}
}

Ref<AudioEffectInstance> AudioEffectBuffer::instance() {
	Ref<AudioEffectBufferInstance> ins;
	ins.instance();
	ins->base = Ref<AudioEffectBuffer>(this);
	current_instance = ins;
	return ins;
}

Vector2 AudioEffectBuffer::read_frame() {

	if (current_instance.is_valid()) {
		return current_instance->read_frame();
	}
	return Vector2();
}

void AudioEffectBuffer::write_frame(const Vector2 &p_frame) {

	if (current_instance.is_valid()) {
		current_instance->write_frame(p_frame);
	}
}

PoolVector2Array AudioEffectBuffer::read_frames() {

	if (current_instance.is_valid()) {
		return current_instance->read_frames();
	}
	return PoolVector2Array();
}

void AudioEffectBuffer::write_frames(const PoolVector2Array &p_frames) {

	if (current_instance.is_valid()) {
		current_instance->write_frames(p_frames);
	}
}

int AudioEffectBuffer::avail_frames_to_read() {

	if (current_instance.is_valid()) {
		return current_instance->avail_frames_to_read();
	}
	return 0;
}

int AudioEffectBuffer::avail_frames_to_write() {

	if (current_instance.is_valid()) {
		return current_instance->avail_frames_to_write();
	}
	return 0;
}

void AudioEffectBuffer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("avail_frames_to_read"), &AudioEffectBuffer::avail_frames_to_read);
	ClassDB::bind_method(D_METHOD("avail_frames_to_write"), &AudioEffectBuffer::avail_frames_to_write);

	ClassDB::bind_method(D_METHOD("write_frame", "frame"), &AudioEffectBuffer::write_frame);
	ClassDB::bind_method(D_METHOD("read_frame"), &AudioEffectBuffer::read_frame);

	ClassDB::bind_method(D_METHOD("write_frames", "frames"), &AudioEffectBuffer::write_frames);
	ClassDB::bind_method(D_METHOD("read_frames"), &AudioEffectBuffer::read_frames);
}
