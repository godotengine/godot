/**************************************************************************/
/*  audio_stream_qoa.cpp                                                  */
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

#define QOA_IMPLEMENTATION
#define QOA_NO_STDIO

#include "audio_stream_qoa.h"

#include "core/io/file_access.h"

int AudioStreamPlaybackQOA::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	if (!active) {
		return 0;
	}

	int todo = p_frames;
	int frames_mixed_this_step = p_frames;

	uint32_t begin_limit = (qoa_stream->loop_mode != AudioStreamQOA::LOOP_DISABLED) ? qoa_stream->loop_begin : 0;
	uint32_t end_limit = (qoa_stream->loop_mode != AudioStreamQOA::LOOP_DISABLED) ? qoa_stream->loop_end : qoad->samples;

	while (todo && active) {
		if (decoded_len <= decoded_offset) {
			// Decode the next or previous QOA frame
			data_offset += int(frame_data_len) * increment;
			qoa_decode_frame(qoa_stream->data.ptr() + data_offset, frame_data_len, qoad, decoded, &decoded_len);
			decoded_offset = increment > 0 ? 0 : decoded_len - 1;
		}

		uint32_t dec_index = decoded_offset * qoad->channels;
		p_buffer[p_frames - todo][0] = decoded[qoa_stream->stereo ? dec_index++ : dec_index];
		p_buffer[p_frames - todo][1] = decoded[dec_index];
		p_buffer[p_frames - todo] /= 32767.0f;

		--todo;

		if (frames_mixed <= begin_limit + 1) {
			// Begin of file or loop
			if (qoa_stream->loop_mode == AudioStreamQOA::LOOP_PINGPONG) {
				increment = 1;
			} else if (qoa_stream->loop_mode == AudioStreamQOA::LOOP_BACKWARD) {
				seek(double(end_limit - 1) / qoa_stream->mix_rate);
			}
		}

		if (frames_mixed >= end_limit - 1) {
			// End of file or loop
			if (qoa_stream->loop_mode == AudioStreamQOA::LOOP_FORWARD) {
				seek(double(begin_limit) / qoa_stream->mix_rate);
			} else if (qoa_stream->loop_mode == AudioStreamQOA::LOOP_PINGPONG) {
				increment = -1;
			} else if (qoa_stream->loop_mode == AudioStreamQOA::LOOP_DISABLED) {
				frames_mixed_this_step = p_frames - todo;
				//fill remainder with silence
				for (int i = p_frames - todo; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
				todo = 0;
			}
		}

		frames_mixed += increment;
		decoded_offset += increment;
	}
	return frames_mixed_this_step;
}

float AudioStreamPlaybackQOA::get_stream_sampling_rate() {
	return qoa_stream->mix_rate;
}

void AudioStreamPlaybackQOA::start(double p_from_pos) {
	active = true;
	seek(p_from_pos);
	if (qoa_stream->loop_mode == AudioStreamQOA::LOOP_BACKWARD) {
		increment = -1;
	}
	begin_resample();
}

void AudioStreamPlaybackQOA::stop() {
	active = false;
}

bool AudioStreamPlaybackQOA::is_playing() const {
	return active;
}

int AudioStreamPlaybackQOA::get_loop_count() const {
	return 0;
}

double AudioStreamPlaybackQOA::get_playback_position() const {
	return double(frames_mixed) / qoa_stream->mix_rate;
}

void AudioStreamPlaybackQOA::seek(double p_time) {
	if (!active) {
		return;
	}

	if (p_time >= qoa_stream->get_length()) {
		p_time = 0;
	}

	frames_mixed = uint32_t(qoa_stream->mix_rate * p_time);
	uint32_t new_data_offset = 8 + frames_mixed / QOA_FRAME_LEN * frame_data_len;

	if (new_data_offset != data_offset) {
		qoa_decode_frame(qoa_stream->data.ptr() + new_data_offset, frame_data_len, qoad, decoded, &decoded_len);
	}
	decoded_offset = frames_mixed % QOA_FRAME_LEN;
	data_offset = new_data_offset;
}

void AudioStreamPlaybackQOA::tag_used_streams() {
	qoa_stream->tag_used(get_playback_position());
}

AudioStreamPlaybackQOA::~AudioStreamPlaybackQOA() {
	if (qoad) {
		memfree(qoad);
	}
	if (decoded) {
		memfree(decoded);
	}
}

Ref<AudioStreamPlayback> AudioStreamQOA::instantiate_playback() {
	Ref<AudioStreamPlaybackQOA> qoas;

	ERR_FAIL_COND_V_MSG(data.is_empty(), qoas,
			"This AudioStreamQOA does not have an audio file assigned "
			"to it. AudioStreamQOA should not be created from the "
			"inspector or with `.new()`. Instead, load an audio file.");

	qoas.instantiate();
	qoas->qoa_stream = Ref<AudioStreamQOA>(this);

	qoas->qoad = (qoa_desc *)memalloc(sizeof(qoa_desc));
	qoa_decode_header(data.ptr(), QOA_MIN_FILESIZE, qoas->qoad);

	qoas->frame_data_len = qoa_max_frame_size(qoas->qoad);
	qoas->decoded = (short *)memalloc(qoas->qoad->channels * QOA_FRAME_LEN * sizeof(short) * 2);

	qoas->data_offset = 0;
	qoas->frames_mixed = 0;
	qoas->active = false;

	ERR_FAIL_NULL_V(qoas->qoad, Ref<AudioStreamPlaybackQOA>());

	return qoas;
}

String AudioStreamQOA::get_stream_name() const {
	return "";
}

void AudioStreamQOA::clear_data() {
	data.clear();
}

void AudioStreamQOA::set_data(const Vector<uint8_t> &p_data) {
	int src_data_len = p_data.size();
	const uint8_t *src_datar = p_data.ptr();

	qoa_desc qoad;
	uint32_t ffp = qoa_decode_header(src_datar, src_data_len, &qoad);
	ERR_FAIL_COND_MSG(ffp != 8, "Failed to decode QOA header. Make sure it is a valid QOA audio file.");

	stereo = qoad.channels > 1;
	mix_rate = qoad.samplerate;
	length = float(qoad.samples) / (mix_rate);
	clear_data();

	data.resize(src_data_len);
	memcpy(data.ptrw(), src_datar, src_data_len);
	data_len = src_data_len;
}

Vector<uint8_t> AudioStreamQOA::get_data() const {
	return data;
}

void AudioStreamQOA::set_loop_mode(LoopMode p_loop_mode) {
	loop_mode = p_loop_mode;
}

AudioStreamQOA::LoopMode AudioStreamQOA::get_loop_mode() const {
	return loop_mode;
}

void AudioStreamQOA::set_loop_begin(int p_frame) {
	loop_begin = p_frame;
}

int AudioStreamQOA::get_loop_begin() const {
	return loop_begin;
}

void AudioStreamQOA::set_loop_end(int p_frame) {
	loop_end = p_frame;
}

int AudioStreamQOA::get_loop_end() const {
	return loop_end;
}

void AudioStreamQOA::set_mix_rate(int p_hz) {
	mix_rate = p_hz;
}

int AudioStreamQOA::get_mix_rate() const {
	return mix_rate;
}

double AudioStreamQOA::get_length() const {
	return length;
}

void AudioStreamQOA::set_stereo(bool p_stereo) {
	stereo = p_stereo;
}

bool AudioStreamQOA::is_stereo() const {
	return stereo;
}

bool AudioStreamQOA::is_monophonic() const {
	return false;
}

void AudioStreamQOA::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamQOA::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamQOA::get_data);

	ClassDB::bind_method(D_METHOD("set_loop_mode", "loop_mode"), &AudioStreamQOA::set_loop_mode);
	ClassDB::bind_method(D_METHOD("get_loop_mode"), &AudioStreamQOA::get_loop_mode);

	ClassDB::bind_method(D_METHOD("set_loop_begin", "seconds"), &AudioStreamQOA::set_loop_begin);
	ClassDB::bind_method(D_METHOD("get_loop_begin"), &AudioStreamQOA::get_loop_begin);

	ClassDB::bind_method(D_METHOD("set_loop_end", "seconds"), &AudioStreamQOA::set_loop_end);
	ClassDB::bind_method(D_METHOD("get_loop_end"), &AudioStreamQOA::get_loop_end);

	ClassDB::bind_method(D_METHOD("set_mix_rate", "hz"), &AudioStreamQOA::set_mix_rate);
	ClassDB::bind_method(D_METHOD("get_mix_rate"), &AudioStreamQOA::get_mix_rate);

	ClassDB::bind_method(D_METHOD("set_stereo", "stereo"), &AudioStreamQOA::set_stereo);
	ClassDB::bind_method(D_METHOD("is_stereo"), &AudioStreamQOA::is_stereo);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_mode", PROPERTY_HINT_ENUM, "Disabled,Forward,Ping-Pong,Backward"), "set_loop_mode", "get_loop_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_begin"), "set_loop_begin", "get_loop_begin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_end"), "set_loop_end", "get_loop_end");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_rate"), "set_mix_rate", "get_mix_rate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stereo"), "set_stereo", "is_stereo");

	BIND_ENUM_CONSTANT(LOOP_DISABLED);
	BIND_ENUM_CONSTANT(LOOP_FORWARD);
	BIND_ENUM_CONSTANT(LOOP_PINGPONG);
	BIND_ENUM_CONSTANT(LOOP_BACKWARD);
}

AudioStreamQOA::AudioStreamQOA() {
}

AudioStreamQOA::~AudioStreamQOA() {
	clear_data();
}
