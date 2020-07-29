/*************************************************************************/
/*  stream_audio.cpp                                                     */
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

#include "stream_audio.h"

void StreamAudio::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_audio_frames", "frames"), &StreamAudio::get_audio_frames);
	ClassDB::bind_method(D_METHOD("set_audio_effect_stream", "bus", "effect"), &StreamAudio::set_audio_effect_stream);
	ClassDB::bind_method(D_METHOD("clear"), &StreamAudio::clear);
}

StreamAudio::StreamAudio() {
}

StreamAudio::~StreamAudio() {
	clear();
}
Vector<float> StreamAudio::get_audio_frames(int32_t p_frames) {
	ERR_FAIL_COND_V(ring_buffer.is_null(), Vector<float>());
	ERR_FAIL_COND_V(p_frames >= ring_buffer->get().size(), Vector<float>());
	int data_left = ring_buffer->get().data_left();
	if (data_left < p_frames || !data_left) {
		return Vector<float>();
	}
	data_left = p_frames;

	Vector<AudioFrame> streaming_data;
	streaming_data.resize(data_left);
	ring_buffer->get().read(streaming_data.ptrw(), data_left);
	Vector<float> output;
	output.resize(data_left * 2);
	for (int32_t i = 0; i < streaming_data.size(); i++) {
		output.write[(i * 2) + 0] = streaming_data[i].l;
		output.write[(i * 2) + 1] = streaming_data[i].r;
	}
	return output;
}

void StreamAudio::set_audio_effect_stream(int32_t p_bus, int32_t p_effect) {
	Ref<AudioEffectStream> current_effect = AudioServer::get_singleton()->get_bus_effect(p_bus, p_effect);
	if (current_effect.is_null()) {
		return;
	}
	float ring_buffer_max_size = IO_BUFFER_SIZE_MS;
	ring_buffer_max_size /= 1000.0; //convert to seconds
	ring_buffer_max_size *= AudioServer::get_singleton()->get_mix_rate();
	ring_buffer = current_effect->init(ring_buffer_max_size);
}

void StreamAudio::clear() {
	if (ring_buffer.is_null()) {
		return;
	}
	const int32_t data_left = ring_buffer->get().data_left();
	ring_buffer->get().advance_read(data_left);
}
