/**************************************************************************/
/*  audio_stream_opus.cpp                                                 */
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

#include "audio_stream_opus.h"

#include "core/io/file_access.h"
#include "core/object/class_db.h"

static const int OPUS_SAMPLERATE = 48000;

int AudioStreamPlaybackOpus::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	ERR_FAIL_COND_V(!active, 0);

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
			return p_frames - todo;
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
	return p_frames - todo;
}

float AudioStreamPlaybackOpus::get_stream_sampling_rate() {
	return OPUS_SAMPLERATE;
}

void AudioStreamPlaybackOpus::start(double p_from_pos) {
	active = true;
	seek(p_from_pos);
	loops = 0;
	begin_resample();
}

void AudioStreamPlaybackOpus::stop() {
	active = false;
}

bool AudioStreamPlaybackOpus::is_playing() const {
	return active;
}

int AudioStreamPlaybackOpus::get_loop_count() const {
	return loops;
}

double AudioStreamPlaybackOpus::get_playback_position() const {
	return double(frames_mixed) / OPUS_SAMPLERATE;
}

void AudioStreamPlaybackOpus::seek(double p_time) {
	if (!active) {
		return;
	}

	if (p_time >= opus_stream->get_length()) {
		p_time = 0;
	}
	frames_mixed = uint32_t(OPUS_SAMPLERATE * p_time);

	int error = op_pcm_seek(opus_file, frames_mixed);
	ERR_FAIL_COND_MSG(error != 0, "Opus seek failed.");
}

AudioStreamPlaybackOpus::~AudioStreamPlaybackOpus() {
}

Ref<AudioStreamPlayback> AudioStreamOpus::instantiate_playback() {
	Ref<AudioStreamPlaybackOpus> opus;

	ERR_FAIL_COND_V_MSG(data == nullptr, nullptr,
			"This AudioStreamOpus does not have an audio file assigned "
			"to it. AudioStreamOpus should not be created from the "
			"inspector or with `.new()`. Instead, load an audio file.");

	opus.instantiate();
	opus->opus_stream = Ref<AudioStreamOpus>(this);
	opus->frames_mixed = 0;
	opus->active = false;
	opus->loops = 0;

	opus->opus_file = op_open_memory((const unsigned char *)data, data_len, nullptr);
	ERR_FAIL_COND_V(!opus->opus_file, Ref<AudioStreamPlaybackOpus>());

	return opus;
}

String AudioStreamOpus::get_stream_name() const {
	return ""; //return stream_name;
}

void AudioStreamOpus::clear_data() {
	if (data) {
		memfree(data);
		data = nullptr;
		data_len = 0;
	}
}

void AudioStreamOpus::set_data(const Vector<uint8_t> &p_data) {
	int src_data_len = p_data.size();
	const uint8_t *src_datar = p_data.ptr();

	// Open file to fetch metadata
	OggOpusFile *opus_file = op_open_memory(src_datar, src_data_len, nullptr);
	ERR_FAIL_COND_MSG(opus_file == nullptr, "Could not open opus stream.");

	int64_t length_i = op_pcm_total(opus_file, -1);
	length = (length_i > 0) ? (float(length_i) / OPUS_SAMPLERATE) : 0;

	// Tags (parsed same as OGG Vorbis)
	const OpusTags *opus_tags = op_tags(opus_file, -1);
	Dictionary dictionary;
	if (opus_tags != nullptr) {
		for (int i = 0; i < opus_tags->comments; i++) {
			String c = String::utf8(opus_tags->user_comments[i]);
			int equals = c.find_char('=');

#ifdef TOOLS_ENABLED
			if (equals == -1) {
				WARN_PRINT(vformat(R"(Invalid comment in Ogg Opus file "%s", should contain '=': "%s".)", get_path(), c));
				continue;
			}
#endif

			String tag = c.substr(0, equals);
			String tag_value = c.substr(equals + 1);

			dictionary[tag.to_lower()] = tag_value;
		}
	}
	tags = dictionary;

	// Close file
	op_free(opus_file);

	// Clear existing data
	clear_data();

	// Copy data
	data = memalloc(src_data_len);
	memcpy(data, src_datar, src_data_len);
	data_len = src_data_len;
}

Vector<uint8_t> AudioStreamOpus::get_data() const {
	Vector<uint8_t> vdata;

	if (data_len && data) {
		vdata.resize(data_len);
		{
			uint8_t *w = vdata.ptrw();
			memcpy(w, data, data_len);
		}
	}

	return vdata;
}

void AudioStreamOpus::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamOpus::has_loop() const {
	return loop;
}

void AudioStreamOpus::set_loop_offset(float p_seconds) {
	loop_offset = p_seconds;
}

float AudioStreamOpus::get_loop_offset() const {
	return loop_offset;
}

double AudioStreamOpus::get_length() const {
	return length;
}

Ref<AudioSample> AudioStreamOpus::generate_sample() const {
	Ref<AudioSample> sample;
	sample.instantiate();
	sample->stream = this;
	sample->loop_mode = loop
			? AudioSample::LoopMode::LOOP_FORWARD
			: AudioSample::LoopMode::LOOP_DISABLED;
	sample->loop_begin = loop_offset;
	sample->loop_end = 0;
	return sample;
}

Ref<AudioStreamOpus> AudioStreamOpus::load_from_buffer(const Vector<uint8_t> &p_stream_data) {
	Ref<AudioStreamOpus> opus_stream;
	opus_stream.instantiate();

	opus_stream->set_data(p_stream_data);

	return opus_stream;
}

Ref<AudioStreamOpus> AudioStreamOpus::load_from_file(const String &p_path) {
	const Vector<uint8_t> stream_data = FileAccess::get_file_as_bytes(p_path);
	ERR_FAIL_COND_V_MSG(stream_data.is_empty(), Ref<AudioStreamOpus>(), vformat("Cannot open file '%s'.", p_path));
	return load_from_buffer(stream_data);
}

void AudioStreamOpus::set_bpm(double p_bpm) {
	ERR_FAIL_COND(p_bpm < 0);
	bpm = p_bpm;
	emit_changed();
}

double AudioStreamOpus::get_bpm() const {
	return bpm;
}

void AudioStreamOpus::set_beat_count(int p_beat_count) {
	ERR_FAIL_COND(p_beat_count < 0);
	beat_count = p_beat_count;
	emit_changed();
}

int AudioStreamOpus::get_beat_count() const {
	return beat_count;
}

void AudioStreamOpus::set_bar_beats(int p_bar_beats) {
	ERR_FAIL_COND(p_bar_beats < 2);
	bar_beats = p_bar_beats;
	emit_changed();
}

int AudioStreamOpus::get_bar_beats() const {
	return bar_beats;
}

void AudioStreamOpus::set_tags(const Dictionary &p_tags) {
	tags = p_tags;
}

Dictionary AudioStreamOpus::get_tags() const {
	return tags;
}

bool AudioStreamOpus::is_monophonic() const {
	return false;
}

void AudioStreamOpus::_bind_methods() {
	ClassDB::bind_static_method("AudioStreamOpus", D_METHOD("load_from_buffer", "stream_data"), &AudioStreamOpus::load_from_buffer);
	ClassDB::bind_static_method("AudioStreamOpus", D_METHOD("load_from_file", "path"), &AudioStreamOpus::load_from_file);

	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamOpus::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamOpus::get_data);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamOpus::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamOpus::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamOpus::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamOpus::get_loop_offset);

	ClassDB::bind_method(D_METHOD("set_bpm", "bpm"), &AudioStreamOpus::set_bpm);
	ClassDB::bind_method(D_METHOD("get_bpm"), &AudioStreamOpus::get_bpm);

	ClassDB::bind_method(D_METHOD("set_beat_count", "count"), &AudioStreamOpus::set_beat_count);
	ClassDB::bind_method(D_METHOD("get_beat_count"), &AudioStreamOpus::get_beat_count);

	ClassDB::bind_method(D_METHOD("set_bar_beats", "count"), &AudioStreamOpus::set_bar_beats);
	ClassDB::bind_method(D_METHOD("get_bar_beats"), &AudioStreamOpus::get_bar_beats);

	ClassDB::bind_method(D_METHOD("set_tags", "tags"), &AudioStreamOpus::set_tags);
	ClassDB::bind_method(D_METHOD("get_tags"), &AudioStreamOpus::get_tags);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bpm", PROPERTY_HINT_RANGE, "0,400,0.01,or_greater"), "set_bpm", "get_bpm");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "beat_count", PROPERTY_HINT_RANGE, "0,512,1,or_greater"), "set_beat_count", "get_beat_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bar_beats", PROPERTY_HINT_RANGE, "2,32,1,or_greater"), "set_bar_beats", "get_bar_beats");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "tags", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_tags", "get_tags");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "loop_offset"), "set_loop_offset", "get_loop_offset");
}

AudioStreamOpus::AudioStreamOpus() {
	data = nullptr;
	data_len = 0;
	length = 0;
	loop_offset = 0;
	loop = false;
}

AudioStreamOpus::~AudioStreamOpus() {
	clear_data();
}
