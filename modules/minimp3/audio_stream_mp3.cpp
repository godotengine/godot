/*************************************************************************/
/*  audio_stream_mp3.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#define MINIMP3_ONLY_MP3
#define MINIMP3_FLOAT_OUTPUT
#define MINIMP3_IMPLEMENTATION
#define MINIMP3_NO_STDIO

#include "audio_stream_mp3.h"

#include "core/io/file_access.h"

int AudioStreamPlaybackMP3::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	ERR_FAIL_COND_V(!active, 0);

	int todo = p_frames;

	int frames_mixed_this_step = p_frames;

	while (todo && active) {
		mp3dec_frame_info_t frame_info;
		mp3d_sample_t *buf_frame = nullptr;

		int samples_mixed = mp3dec_ex_read_frame(mp3d, &buf_frame, &frame_info, mp3_stream->channels);

		if (samples_mixed) {
			p_buffer[p_frames - todo] = AudioFrame(buf_frame[0], buf_frame[samples_mixed - 1]);
			--todo;
			++frames_mixed;
		}

		else {
			//EOF
			if (mp3_stream->loop) {
				seek(mp3_stream->loop_offset);
				loops++;
			} else {
				frames_mixed_this_step = p_frames - todo;
				//fill remainder with silence
				for (int i = p_frames - todo; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
				todo = 0;
			}
		}
	}
	return frames_mixed_this_step;
}

float AudioStreamPlaybackMP3::get_stream_sampling_rate() {
	return mp3_stream->sample_rate;
}

void AudioStreamPlaybackMP3::start(float p_from_pos) {
	active = true;
	seek(p_from_pos);
	loops = 0;
	_begin_resample();
}

void AudioStreamPlaybackMP3::stop() {
	active = false;
}

bool AudioStreamPlaybackMP3::is_playing() const {
	return active;
}

int AudioStreamPlaybackMP3::get_loop_count() const {
	return loops;
}

float AudioStreamPlaybackMP3::get_playback_position() const {
	return float(frames_mixed) / mp3_stream->sample_rate;
}

void AudioStreamPlaybackMP3::seek(float p_time) {
	if (!active) {
		return;
	}

	if (p_time >= mp3_stream->get_length()) {
		p_time = 0;
	}

	frames_mixed = uint32_t(mp3_stream->sample_rate * p_time);
	mp3dec_ex_seek(mp3d, (uint64_t)frames_mixed * mp3_stream->channels);
}

AudioStreamPlaybackMP3::~AudioStreamPlaybackMP3() {
	if (mp3d) {
		mp3dec_ex_close(mp3d);
		memfree(mp3d);
	}
}

Ref<AudioStreamPlayback> AudioStreamMP3::instance_playback() {
	Ref<AudioStreamPlaybackMP3> mp3s;

	ERR_FAIL_COND_V_MSG(data.is_empty(), mp3s,
			"This AudioStreamMP3 does not have an audio file assigned "
			"to it. AudioStreamMP3 should not be created from the "
			"inspector or with `.new()`. Instead, load an audio file.");

	mp3s.instantiate();
	mp3s->mp3_stream = Ref<AudioStreamMP3>(this);
	mp3s->mp3d = (mp3dec_ex_t *)memalloc(sizeof(mp3dec_ex_t));

	int errorcode = mp3dec_ex_open_buf(mp3s->mp3d, data.ptr(), data_len, MP3D_SEEK_TO_SAMPLE);

	mp3s->frames_mixed = 0;
	mp3s->active = false;
	mp3s->loops = 0;

	if (errorcode) {
		ERR_FAIL_COND_V(errorcode, Ref<AudioStreamPlaybackMP3>());
	}

	return mp3s;
}

String AudioStreamMP3::get_stream_name() const {
	return ""; //return stream_name;
}

void AudioStreamMP3::clear_data() {
	data.clear();
}

void AudioStreamMP3::set_data(const Vector<uint8_t> &p_data) {
	int src_data_len = p_data.size();
	const uint8_t *src_datar = p_data.ptr();

	mp3dec_ex_t mp3d;
	int err = mp3dec_ex_open_buf(&mp3d, src_datar, src_data_len, MP3D_SEEK_TO_SAMPLE);
	ERR_FAIL_COND_MSG(err || mp3d.info.hz == 0, "Failed to decode mp3 file. Make sure it is a valid mp3 audio file.");

	channels = mp3d.info.channels;
	sample_rate = mp3d.info.hz;
	length = float(mp3d.samples) / (sample_rate * float(channels));

	mp3dec_ex_close(&mp3d);

	clear_data();

	data.resize(src_data_len);
	memcpy(data.ptrw(), src_datar, src_data_len);
	data_len = src_data_len;
}

Vector<uint8_t> AudioStreamMP3::get_data() const {
	return data;
}

void AudioStreamMP3::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamMP3::has_loop() const {
	return loop;
}

void AudioStreamMP3::set_loop_offset(float p_seconds) {
	loop_offset = p_seconds;
}

float AudioStreamMP3::get_loop_offset() const {
	return loop_offset;
}

float AudioStreamMP3::get_length() const {
	return length;
}

bool AudioStreamMP3::is_monophonic() const {
	return false;
}

void AudioStreamMP3::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamMP3::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamMP3::get_data);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamMP3::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamMP3::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamMP3::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamMP3::get_loop_offset);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "loop_offset"), "set_loop_offset", "get_loop_offset");
}

AudioStreamMP3::AudioStreamMP3() {
}

AudioStreamMP3::~AudioStreamMP3() {
	clear_data();
}
