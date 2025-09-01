/**************************************************************************/
/*  audio_stream_ogg_opus.cpp                                             */
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

#include "audio_stream_ogg_opus.h"

#include "core/io/file_access.h"

int AudioStreamPlaybackOggOpus::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	ERR_FAIL_COND_V(!ready, 0);
	if (!active) {
		return 0;
	}

	int todo = p_frames;

	int beat_length_frames = -1;
	bool use_loop = looping_override ? looping : opus_stream->has_loop();
	if (use_loop && opus_stream->get_bpm() > 0 && opus_stream->get_beat_count() > 0) {
		beat_length_frames = opus_stream->get_beat_count() * int(get_stream_sampling_rate()) * 60 / int(opus_stream->get_bpm());
	}

	while (todo > 0 && active) {
		AudioFrame *buffer = p_buffer + (p_frames - todo);
		int to_mix = todo;
		if (beat_length_frames >= 0 && (beat_length_frames - int(frames_mixed)) < to_mix) {
			to_mix = MAX(0, beat_length_frames - int(frames_mixed));
		}

		int mixed = _mix_frames(buffer, to_mix);
		ERR_FAIL_COND_V(mixed < 0, 0);
		todo -= mixed;
		frames_mixed += mixed;

		if (loop_fade_remaining < FADE_SIZE) {
			int to_fade = loop_fade_remaining + MIN(FADE_SIZE - loop_fade_remaining, mixed);
			for (int i = loop_fade_remaining; i < to_fade; i++) {
				buffer[i - loop_fade_remaining] += loop_fade[i] * (float(FADE_SIZE - i) / float(FADE_SIZE));
			}
			loop_fade_remaining = to_fade;
		}

		if (beat_length_frames >= 0) {
			if (use_loop && beat_length_frames <= int(frames_mixed)) {
				// Pre-fill fade buffer.
				float tmp[FADE_SIZE * 2];
				int r = op_read_float_stereo(opus_file, tmp, FADE_SIZE * 2);
				int faded_mix = MAX(r, 0);
				for (int i = 0; i < FADE_SIZE; i++) {
					if (i < faded_mix) {
						loop_fade[i].left = tmp[i * 2 + 0];
						loop_fade[i].right = tmp[i * 2 + 1];
					} else {
						loop_fade[i] = AudioFrame(0, 0);
					}
				}
				loop_fade_remaining = 0;

				seek(opus_stream->get_loop_offset());
				loops++;
				continue;
			}
		}

		// Check EOF: op_read_* returns 0 at end of stream.
		if (mixed == 0) {
			if (use_loop) {
				seek(opus_stream->get_loop_offset());
				loops++;
			} else {
				for (int i = p_frames - todo; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
			}
		}
	}

	return p_frames - todo;
}

int AudioStreamPlaybackOggOpus::_mix_frames(AudioFrame *p_buffer, int p_frames) {
	if (p_frames <= 0) {
		return 0;
	}
	// libopusfile always decodes at 48kHz. We'll read float stereo regardless of original mapping.
	const int buf_vals = p_frames * 2; // stereo interleaved
	float tmp[FADE_SIZE * 2];
	int total = 0;
	while (total < p_frames) {
		int chunk = MIN(FADE_SIZE, p_frames - total);
		int r = op_read_float_stereo(opus_file, tmp, chunk * 2);
		if (r < 0) {
			// Error or hole, treat as silence for this call.
			WARN_PRINT("Error reading Opus stream");
			break;
		}
		if (r == 0) {
			// EOF
			break;
		}
		// r is samples per channel written (interleaved stereo => 2*r values used)
		for (int i = 0; i < r; i++) {
			p_buffer[total + i].left = tmp[i * 2 + 0];
			p_buffer[total + i].right = tmp[i * 2 + 1];
		}
		total += r;
	}
	// Zero-pad remainder if any
	for (int i = total; i < p_frames; i++) {
		p_buffer[i] = AudioFrame(0, 0);
	}
	return total;
}

float AudioStreamPlaybackOggOpus::get_stream_sampling_rate() {
	// some site told me that libopusfile always outputs 48000 Hz.
	return 48000.0f;
}

bool AudioStreamPlaybackOggOpus::_alloc_opus() {
	// Expect the stream resource to carry raw buffer in metadata.
	ERR_FAIL_COND_V(opus_stream.is_null(), false);
	ERR_FAIL_COND_V(!opus_stream->has_meta("_opus_buffer"), false);
	Vector<uint8_t> data = opus_stream->get_meta("_opus_buffer");
	int err = 0;
	opus_file = op_open_memory((const unsigned char *)data.ptr(), data.size(), &err);
	ERR_FAIL_COND_V_MSG(!opus_file || err != 0, false, "Failed to open Opus stream from memory");
	ready = true;
	return true;
}

void AudioStreamPlaybackOggOpus::start(double p_from_pos) {
	ERR_FAIL_COND(!ready);
	loop_fade_remaining = FADE_SIZE;
	active = true;
	seek(p_from_pos);
	loops = 0;
	begin_resample();
}

void AudioStreamPlaybackOggOpus::stop() { active = false; }

bool AudioStreamPlaybackOggOpus::is_playing() const { return active; }

int AudioStreamPlaybackOggOpus::get_loop_count() const { return loops; }

double AudioStreamPlaybackOggOpus::get_playback_position() const {
	// op_pcm_tell returns position in samples at 48 kHz per channel.
	if (!opus_file) return 0.0;
	ogg_int64_t pos = op_pcm_tell(opus_file);
	if (pos < 0) pos = 0;
	return double(pos) / 48000.0;
}

void AudioStreamPlaybackOggOpus::tag_used_streams() {
	opus_stream->tag_used(get_playback_position());
}

void AudioStreamPlaybackOggOpus::set_parameter(const StringName &p_name, const Variant &p_value) {
	if (p_name == SNAME("looping")) {
		if (p_value == Variant()) {
			looping_override = false;
			looping = false;
		} else {
			looping_override = true;
			looping = p_value;
		}
	}
}

Variant AudioStreamPlaybackOggOpus::get_parameter(const StringName &p_name) const {
	if (looping_override && p_name == SNAME("looping")) {
		return looping;
	}
	return Variant();
}

void AudioStreamPlaybackOggOpus::seek(double p_time) {
	ERR_FAIL_COND(!ready);
	if (!active) return;

	if (p_time < 0) p_time = 0;
	ogg_int64_t sample = ogg_int64_t(p_time * 48000.0);
	int err = op_pcm_seek(opus_file, sample);
	if (err != 0) {
		WARN_PRINT("Opus seek failed");
	}
	frames_mixed = uint32_t(sample);
}

void AudioStreamPlaybackOggOpus::set_is_sample(bool p_is_sample) { _is_sample = p_is_sample; }
bool AudioStreamPlaybackOggOpus::get_is_sample() const { return _is_sample; }
Ref<AudioSamplePlayback> AudioStreamPlaybackOggOpus::get_sample_playback() const { return sample_playback; }
void AudioStreamPlaybackOggOpus::set_sample_playback(const Ref<AudioSamplePlayback> &p_playback) {
	sample_playback = p_playback;
	if (sample_playback.is_valid()) {
		sample_playback->stream_playback = Ref<AudioStreamPlayback>(this);
	}
}

AudioStreamPlaybackOggOpus::~AudioStreamPlaybackOggOpus() {
	if (opus_file) {
		op_free(opus_file);
		opus_file = nullptr;
	}
}

Ref<AudioStreamPlayback> AudioStreamOggOpus::instantiate_playback() {
	Ref<AudioStreamPlaybackOggOpus> p;
	p.instantiate();
	p->opus_stream = Ref<AudioStreamOggOpus>(this);
	if (p->_alloc_opus()) {
		return p;
	}
	return Ref<AudioStreamPlayback>();
}

String AudioStreamOggOpus::get_stream_name() const { return ""; }

static bool _opus_parse_tags(OggOpusFile *of, Dictionary &r_tags) {
	const OpusTags *t = op_tags(of, -1);
	if (!t) return false;
	Dictionary d;
	for (int i = 0; i < t->comments; i++) {
		String c = String::utf8(t->user_comments[i]);
		int eq = c.find_char('=');
		if (eq <= 0) continue;
		String key = c.substr(0, eq).to_lower();
		String val = c.substr(eq + 1);
		d[key] = val;
	}
	r_tags = d;
	return true;
}

Ref<AudioStreamOggOpus> AudioStreamOggOpus::load_from_buffer(const Vector<uint8_t> &p_stream_data) {
	Ref<AudioStreamOggOpus> s;
	s.instantiate();

	int of_err = 0;
	OggOpusFile *of = op_open_memory((const unsigned char *)p_stream_data.ptr(), p_stream_data.size(), &of_err);
	ERR_FAIL_COND_V_MSG(!of || of_err != 0, Ref<AudioStreamOggOpus>(), "Failed to open Opus from memory");

	ogg_int64_t total_pcm = op_pcm_total(of, -1);
	double length = total_pcm > 0 ? double(total_pcm) / 48000.0 : 0.0;
	s->set_tags(Dictionary());
	Dictionary t;
	_opus_parse_tags(of, t);
	s->set_tags(t);

	// Store the raw data inside the resource as metadata to allow playback instances to open their own decoders.
	s->set_meta("_opus_buffer", p_stream_data);
	s->set_meta("_opus_length", length);

	op_free(of);
	return s;
}

Ref<AudioStreamOggOpus> AudioStreamOggOpus::load_from_file(const String &p_path) {
	const Vector<uint8_t> stream_data = FileAccess::get_file_as_bytes(p_path);
	ERR_FAIL_COND_V_MSG(stream_data.is_empty(), Ref<AudioStreamOggOpus>(), vformat("Cannot open file '%s'.", p_path));
	return load_from_buffer(stream_data);
}

void AudioStreamOggOpus::set_loop(bool p_enable) { loop = p_enable; }
bool AudioStreamOggOpus::has_loop() const { return loop; }
void AudioStreamOggOpus::set_loop_offset(double p_seconds) { loop_offset = p_seconds; }
double AudioStreamOggOpus::get_loop_offset() const { return loop_offset; }

double AudioStreamOggOpus::get_length() const {
	if (has_meta("_opus_length")) {
		return (double)get_meta("_opus_length");
	}
	return 0.0;
}

void AudioStreamOggOpus::set_bpm(double p_bpm) {
	ERR_FAIL_COND(p_bpm < 0);
	bpm = p_bpm;
	emit_changed();
}
double AudioStreamOggOpus::get_bpm() const { return bpm; }
void AudioStreamOggOpus::set_beat_count(int p_beat_count) {
	ERR_FAIL_COND(p_beat_count < 0);
	beat_count = p_beat_count;
	emit_changed();
}
int AudioStreamOggOpus::get_beat_count() const { return beat_count; }
void AudioStreamOggOpus::set_bar_beats(int p_bar_beats) {
	ERR_FAIL_COND(p_bar_beats < 2);
	bar_beats = p_bar_beats;
	emit_changed();
}
int AudioStreamOggOpus::get_bar_beats() const { return bar_beats; }
void AudioStreamOggOpus::set_tags(const Dictionary &p_tags) { tags = p_tags; }
Dictionary AudioStreamOggOpus::get_tags() const { return tags; }

void AudioStreamOggOpus::get_parameter_list(List<Parameter> *r_parameters) {
	r_parameters->push_back(Parameter(PropertyInfo(Variant::BOOL, "looping", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CHECKABLE), Variant()));
}

Ref<AudioSample> AudioStreamOggOpus::generate_sample() const {
	Ref<AudioSample> sample;
	sample.instantiate();
	sample->stream = this;
	sample->loop_mode = loop ? AudioSample::LoopMode::LOOP_FORWARD : AudioSample::LoopMode::LOOP_DISABLED;
	sample->loop_begin = loop_offset;
	sample->loop_end = 0;
	return sample;
}

void AudioStreamOggOpus::_bind_methods() {
	ClassDB::bind_static_method("AudioStreamOggOpus", D_METHOD("load_from_buffer", "stream_data"), &AudioStreamOggOpus::load_from_buffer);
	ClassDB::bind_static_method("AudioStreamOggOpus", D_METHOD("load_from_file", "path"), &AudioStreamOggOpus::load_from_file);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamOggOpus::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamOggOpus::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamOggOpus::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamOggOpus::get_loop_offset);

	ClassDB::bind_method(D_METHOD("set_bpm", "bpm"), &AudioStreamOggOpus::set_bpm);
	ClassDB::bind_method(D_METHOD("get_bpm"), &AudioStreamOggOpus::get_bpm);

	ClassDB::bind_method(D_METHOD("set_beat_count", "count"), &AudioStreamOggOpus::set_beat_count);
	ClassDB::bind_method(D_METHOD("get_beat_count"), &AudioStreamOggOpus::get_beat_count);

	ClassDB::bind_method(D_METHOD("set_bar_beats", "count"), &AudioStreamOggOpus::set_bar_beats);
	ClassDB::bind_method(D_METHOD("get_bar_beats"), &AudioStreamOggOpus::get_bar_beats);

	ClassDB::bind_method(D_METHOD("set_tags", "tags"), &AudioStreamOggOpus::set_tags);
	ClassDB::bind_method(D_METHOD("get_tags"), &AudioStreamOggOpus::get_tags);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "tags", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_tags", "get_tags");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "loop_offset"), "set_loop_offset", "get_loop_offset");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bpm", PROPERTY_HINT_RANGE, "0,400,0.01,or_greater"), "set_bpm", "get_bpm");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "beat_count", PROPERTY_HINT_RANGE, "0,512,1,or_greater"), "set_beat_count", "get_beat_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bar_beats", PROPERTY_HINT_RANGE, "2,32,1,or_greater"), "set_bar_beats", "get_bar_beats");
}

AudioStreamOggOpus::AudioStreamOggOpus() {}
AudioStreamOggOpus::~AudioStreamOggOpus() {}
