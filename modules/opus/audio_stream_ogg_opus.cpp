/*************************************************************************/
/*  audio_stream_ogg_opus.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_stream_ogg_opus.h"

#include "core/os/file_access.h"

static const int OPUS_SAMPLERATE = 48000;

void AudioStreamPlaybackOGGOpus::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	ERR_FAIL_COND(!active);

	int todo = p_frames;
	bool mixed_was_zero = false; // for detecting infinite loop

	while (todo && active) {
		float *buffer = (float *)(p_buffer + p_frames - todo);
		int mixed = op_read_float_stereo(opus_file, buffer, todo * 2);
		if (mixed > 0) {
			mixed_was_zero = false;
		}
		if (mixed < 0) {
			// error
			for (int i = p_frames - todo; i < p_frames; i++) {
				p_buffer[i] = AudioFrame(0, 0);
			}
			return;
		}

		todo -= mixed;
		frames_mixed += mixed;

		if (mixed == 0) {
			//end of file!
			if (opus_stream->loop && !mixed_was_zero) {
				//loop
				seek(opus_stream->loop_offset);
				loops++;
			} else {
				for (int i = p_frames - todo; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
				todo = 0;
			}
			mixed_was_zero = true;
		}
	}
}

float AudioStreamPlaybackOGGOpus::get_stream_sampling_rate() {
	return OPUS_SAMPLERATE;
}

void AudioStreamPlaybackOGGOpus::start(float p_from_pos) {
	active = true;
	seek(p_from_pos);
	loops = 0;
	_begin_resample();
}

void AudioStreamPlaybackOGGOpus::stop() {
	active = false;
}

bool AudioStreamPlaybackOGGOpus::is_playing() const {
	return active;
}

int AudioStreamPlaybackOGGOpus::get_loop_count() const {
	return loops;
}

float AudioStreamPlaybackOGGOpus::get_playback_position() const {
	return float(frames_mixed) / OPUS_SAMPLERATE;
}

void AudioStreamPlaybackOGGOpus::seek(float p_time) {
	if (!active)
		return;

	if (p_time >= opus_stream->get_length()) {
		p_time = 0;
	}
	frames_mixed = uint32_t(OPUS_SAMPLERATE * p_time);

	int error = op_pcm_seek(opus_file, frames_mixed);
	ERR_FAIL_COND_MSG(error != 0, "Opus seek failed.");
}

AudioStreamPlaybackOGGOpus::~AudioStreamPlaybackOGGOpus() {
}

Ref<AudioStreamPlayback> AudioStreamOGGOpus::instance_playback() {
	Ref<AudioStreamPlaybackOGGOpus> opus;

	ERR_FAIL_COND_V_MSG(data == NULL, opus,
			"This AudioStreamOGGOpus does not have an audio file assigned "
			"to it. AudioStreamOGGOpus should not be created from the "
			"inspector or with `.new()`. Instead, load an audio file.");

	opus.instance();
	opus->opus_stream = Ref<AudioStreamOGGOpus>(this);
	opus->frames_mixed = 0;
	opus->active = false;

	opus->opus_file = op_open_memory((const unsigned char *)data, data_len, NULL);
	ERR_FAIL_COND_V(!opus->opus_file, Ref<AudioStreamPlaybackOGGOpus>());

	return opus;
}

String AudioStreamOGGOpus::get_stream_name() const {
	return ""; //return stream_name;
}

void AudioStreamOGGOpus::clear_data() {
	if (data) {
		memfree(data);
		data = NULL;
		data_len = 0;
	}
}

void AudioStreamOGGOpus::set_data(const Vector<uint8_t> &p_data) {
	int src_data_len = p_data.size();
	const uint8_t *src_datar = p_data.ptr();

	// open to fetch metadata
	OggOpusFile *opus_file = op_open_memory(src_datar, src_data_len, NULL);
	ERR_FAIL_COND_MSG(opus_file == NULL, "Could not open opus stream.");

	int64_t length_i = op_pcm_total(opus_file, -1);
	length = (length_i > 0) ? (float(length_i) / OPUS_SAMPLERATE) : 0;

	op_free(opus_file);

	clear_data();

	// copy
	data = memalloc(src_data_len);
	copymem(data, src_datar, src_data_len);
	data_len = src_data_len;
}

Vector<uint8_t> AudioStreamOGGOpus::get_data() const {
	Vector<uint8_t> vdata;

	if (data_len && data) {
		vdata.resize(data_len);
		{
			uint8_t *w = vdata.ptrw();
			copymem(w, data, data_len);
		}
	}

	return vdata;
}

void AudioStreamOGGOpus::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamOGGOpus::has_loop() const {
	return loop;
}

void AudioStreamOGGOpus::set_loop_offset(float p_seconds) {
	loop_offset = p_seconds;
}

float AudioStreamOGGOpus::get_loop_offset() const {
	return loop_offset;
}

float AudioStreamOGGOpus::get_length() const {
	return length;
}

void AudioStreamOGGOpus::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamOGGOpus::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamOGGOpus::get_data);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamOGGOpus::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamOGGOpus::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamOGGOpus::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamOGGOpus::get_loop_offset);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "loop_offset"), "set_loop_offset", "get_loop_offset");
}

AudioStreamOGGOpus::AudioStreamOGGOpus() {
	data = NULL;
	data_len = 0;
	length = 0;
	loop_offset = 0;
	loop = false;
}

AudioStreamOGGOpus::~AudioStreamOGGOpus() {
	clear_data();
}
