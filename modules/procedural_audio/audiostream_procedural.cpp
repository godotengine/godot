/*************************************************************************/
/*  audiostream_procedural.cpp                                           */
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

#include "audiostream_procedural.h"
#include "core/math/math_funcs.h"
#include "core/print_string.h"

////////////////////////////// AudioStreamProcedural

AudioStreamProcedural::AudioStreamProcedural() :
		mix_rate(44100),
		stereo(false),
		buffer_frame_count(128) {}

AudioStreamProcedural::~AudioStreamProcedural() {}

Ref<AudioStreamPlayback> AudioStreamProcedural::instance_playback() {
	Ref<AudioStreamPlaybackProcedural> playback;
	playback.instance();

	playback->stream = Ref<AudioStreamProcedural>((AudioStreamProcedural *)this);
	playback->active = false;
	playback->resize_buffer();

	playbacks.insert(playback.ptr());

	return playback;
}

String AudioStreamProcedural::get_stream_name() const {
	return "Procedural";
}

void AudioStreamProcedural::resize_buffer() {
	for (Set<AudioStreamPlaybackProcedural *>::Element *e = playbacks.front(); e;
			e = e->next()) {
		e->get()->resize_buffer();
	}
}

void AudioStreamProcedural::reset() {
	set_position(0);
}

void AudioStreamProcedural::set_position(uint64_t p) {
	if (get_script_instance() &&
			get_script_instance()->has_method("set_position")) {
		get_script_instance()->call("set_position", p);
		return;
	}

	pos = p;
}

uint64_t AudioStreamProcedural::get_position() {
	if (get_script_instance() &&
			get_script_instance()->has_method("get_position")) {
		return get_script_instance()->call("get_position");
	}

	return pos;
}

void AudioStreamProcedural::generate_frames(Ref<StreamPeerBuffer> byte_buffer) {
	if (get_script_instance() &&
			get_script_instance()->has_method("generate_frames")) {
		get_script_instance()->call("generate_frames", byte_buffer);
		return;
	}

	double mix_rate_by_tone_1 = double(mix_rate) / double(350);
	double mix_rate_by_tone_2 = double(mix_rate) / double(440);

	byte_buffer->seek(0);
	if (stereo) {
		for (int i = 0; i < buffer_frame_count; i++) {
			double two_pi_pos = 2.0 * Math_PI * double(pos + i);
			byte_buffer->put_float(sin(two_pi_pos / mix_rate_by_tone_1));
			byte_buffer->put_float(sin(two_pi_pos / mix_rate_by_tone_2));
		}
	} else {
		for (int i = 0; i < buffer_frame_count; i++) {
			double two_pi_pos = 2.0 * Math_PI * double(pos + i);
			byte_buffer->put_float(
					(sin(two_pi_pos / mix_rate_by_tone_1) + sin(two_pi_pos / mix_rate_by_tone_2)) *
					0.5);
		}
	}
	pos += buffer_frame_count;
}

void AudioStreamProcedural::set_mix_rate(int mix_rate) {
	if (this->mix_rate != mix_rate) {
		this->mix_rate = mix_rate;
		resize_buffer();
	}
}

int AudioStreamProcedural::get_mix_rate() {
	return mix_rate;
}

bool AudioStreamProcedural::get_stereo() {
	return stereo;
}

void AudioStreamProcedural::set_stereo(bool stereo) {
	if (this->stereo != stereo) {
		this->stereo = stereo;
		resize_buffer();
	}
}

void AudioStreamProcedural::set_buffer_frame_count(int buffer_frame_count) {
	if (this->buffer_frame_count != buffer_frame_count) {
		this->buffer_frame_count = buffer_frame_count;
		resize_buffer();
	}
}

int AudioStreamProcedural::get_buffer_frame_count() {
	return buffer_frame_count;
}

void AudioStreamProcedural::_bind_methods() {
	ClassDB::bind_method(D_METHOD("reset"), &AudioStreamProcedural::reset);
	ClassDB::bind_method(D_METHOD("get_stream_name"),
			&AudioStreamProcedural::get_stream_name);
	ClassDB::bind_method(D_METHOD("generate_frames", "byte_buffer"),
			&AudioStreamProcedural::generate_frames);
	ClassDB::bind_method(D_METHOD("set_mix_rate", "mix_rate"),
			&AudioStreamProcedural::set_mix_rate);
	ClassDB::bind_method(D_METHOD("get_mix_rate"),
			&AudioStreamProcedural::get_mix_rate);
	ClassDB::bind_method(D_METHOD("set_buffer_frame_count", "buffer_frame_count"),
			&AudioStreamProcedural::set_buffer_frame_count);
	ClassDB::bind_method(D_METHOD("get_buffer_frame_count"),
			&AudioStreamProcedural::get_buffer_frame_count);
	ClassDB::bind_method(D_METHOD("set_position", "position"),
			&AudioStreamProcedural::set_position);
	ClassDB::bind_method(D_METHOD("get_position"),
			&AudioStreamProcedural::get_position);
	ClassDB::bind_method(D_METHOD("set_stereo", "stereo"),
			&AudioStreamProcedural::set_stereo);
	ClassDB::bind_method(D_METHOD("get_stereo"),
			&AudioStreamProcedural::get_stereo);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_rate"), "set_mix_rate",
			"get_mix_rate");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "buffer_frame_count"),
			"set_buffer_frame_count", "get_buffer_frame_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "position"), "set_position",
			"get_position");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stereo"), "set_stereo",
			"get_stereo");
}

////////////////////////////// AudioStreamPlaybackProcedural

AudioStreamPlaybackProcedural::AudioStreamPlaybackProcedural() :
		active(false),
		internal_frame(0),
		external_frame(0) {
	byte_buffer.instance();
}

AudioStreamPlaybackProcedural::~AudioStreamPlaybackProcedural() {
	stream->playbacks.erase(this);
	byte_buffer->clear();
	byte_buffer.unref();
}

void AudioStreamPlaybackProcedural::resize_buffer() {
	// Although this buffer is not allocated by the AudioServer, we're still
	// taking out a lock to be sure there are no attempts to access it during
	// a resize...
	AudioServer::get_singleton()->lock();
	byte_buffer->resize(stream->buffer_frame_count * 4 *
						(stream->stereo ? 2 : 1));
	AudioServer::get_singleton()->unlock();
}

void AudioStreamPlaybackProcedural::stop() {
	active = false;
	stream->reset();
}

void AudioStreamPlaybackProcedural::start(float p_from_pos) {
	seek(p_from_pos);
	active = true;
}

void AudioStreamPlaybackProcedural::seek(float p_time) {
	if (p_time < 0) {
		p_time = 0;
	}
	stream->set_position(uint64_t(p_time * stream->mix_rate) << MIX_FRAC_BITS);
}

void AudioStreamPlaybackProcedural::_mix_internal(AudioFrame *p_buffer,
		int p_frames) {
	ERR_FAIL_COND(!active);

	if (!active) {
		return;
	}

	// Ensure that p_buffer is filled with p_frames-worth of data while also
	// allowing arbitrary sizes of byte_buffer...
	while (internal_frame < p_frames) {
		if (external_frame == 0)
			stream->generate_frames(byte_buffer);

		int byte_buffer_pos;

		if (stream->stereo) {
			byte_buffer_pos = external_frame * 8;
			for (; internal_frame < p_frames &&
					external_frame < stream->buffer_frame_count;
					internal_frame++, external_frame++) {
				byte_buffer->seek(byte_buffer_pos);
				float sample_l = byte_buffer->get_float();
				byte_buffer->seek(byte_buffer_pos + 4);
				float sample_r = byte_buffer->get_float();
				p_buffer[internal_frame] = AudioFrame(sample_l, sample_r);
				byte_buffer_pos += 8;
			}

		} else {
			byte_buffer_pos = external_frame * 4;
			for (; internal_frame < p_frames &&
					external_frame < stream->buffer_frame_count;
					internal_frame++, external_frame++) {
				byte_buffer->seek(byte_buffer_pos);
				float sample = byte_buffer->get_float();
				p_buffer[internal_frame] = AudioFrame(sample, sample);
				byte_buffer_pos += 4;
			}
		}

		if (external_frame >= stream->buffer_frame_count)
			external_frame = 0;
	}

	internal_frame = 0;
}

float AudioStreamPlaybackProcedural::get_stream_sampling_rate() {
	return float(stream->mix_rate);
}

int AudioStreamPlaybackProcedural::get_loop_count() const {
	return 0;
}

float AudioStreamPlaybackProcedural::get_playback_position() const {
	return 0.0;
}

float AudioStreamPlaybackProcedural::get_length() const {
	return 0.0;
}

bool AudioStreamPlaybackProcedural::is_playing() const {
	return active;
}