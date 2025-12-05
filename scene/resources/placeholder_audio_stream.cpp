/**************************************************************************/
/*  placeholder_audio_stream.cpp                                          */
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

#include "placeholder_audio_stream.h"

void PlaceholderAudioStreamPlayback::start(double p_from_pos) {
	seek(p_from_pos);
	sign = 1;
	active = true;
}

void PlaceholderAudioStreamPlayback::stop() {
	active = false;
}

bool PlaceholderAudioStreamPlayback::is_playing() const {
	return active;
}

int PlaceholderAudioStreamPlayback::get_loop_count() const {
	return 0;
}

double PlaceholderAudioStreamPlayback::get_playback_position() const {
	return double(offset);
}

void PlaceholderAudioStreamPlayback::seek(double p_time) {
	double max = base->get_length();
	if (p_time < 0) {
		p_time = 0;
	} else if (p_time >= max) {
		p_time = max - 0.001;
	}

	offset = int64_t(p_time);
}

void PlaceholderAudioStreamPlayback::tag_used_streams() {
	base->tag_used(get_playback_position());
}

void PlaceholderAudioStreamPlayback::set_is_sample(bool p_is_sample) {
	_is_sample = p_is_sample;
}

bool PlaceholderAudioStreamPlayback::get_is_sample() const {
	return _is_sample;
}

Ref<AudioSamplePlayback> PlaceholderAudioStreamPlayback::get_sample_playback() const {
	return sample_playback;
}

void PlaceholderAudioStreamPlayback::set_sample_playback(const Ref<AudioSamplePlayback> &p_playback) {
	sample_playback = p_playback;
	if (sample_playback.is_valid()) {
		sample_playback->stream_playback = Ref<AudioStreamPlayback>(this);
	}
}

int PlaceholderAudioStreamPlayback::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	if (!active) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0, 0);
		}
		return 0;
	}

	uint64_t len = base->get_length() * 1000; // TODO: Calculate length correctly.

	int64_t loop_begin = base->loop_begin;
	int64_t loop_end = base->loop_end;
	int64_t begin_limit = (base->loop_mode != PlaceholderAudioStream::LOOP_DISABLED) ? loop_begin : 0;
	int64_t end_limit = (base->loop_mode != PlaceholderAudioStream::LOOP_DISABLED) ? loop_end : len - 1;

	int32_t todo = p_frames;

	if (base->loop_mode == PlaceholderAudioStream::LOOP_BACKWARD) {
		sign = -1;
	}

	int8_t increment = sign;

	//looping

	PlaceholderAudioStream::LoopMode loop_format = base->loop_mode;

	/* audio data */

	AudioFrame *dst_buff = p_buffer;

	while (todo > 0) {
		int64_t limit = 0;
		int32_t target = 0, aux = 0;

		/** LOOP CHECKING **/

		if (increment < 0) {
			/* going backwards */

			if (loop_format != PlaceholderAudioStream::LOOP_DISABLED && offset < loop_begin) {
				/* loopstart reached */
				if (loop_format == PlaceholderAudioStream::LOOP_PINGPONG) {
					/* bounce ping pong */
					offset = loop_begin + (loop_begin - offset);
					increment = -increment;
					sign *= -1;
				} else {
					/* go to loop-end */
					offset = loop_end - (loop_begin - offset);
				}
			} else {
				/* check for sample not reaching beginning */
				if (offset < 0) {
					active = false;
					break;
				}
			}
		} else {
			/* going forward */
			if (loop_format != PlaceholderAudioStream::LOOP_DISABLED && offset >= loop_end) {
				/* loopend reached */

				if (loop_format == PlaceholderAudioStream::LOOP_PINGPONG) {
					/* bounce ping pong */
					offset = loop_end - (offset - loop_end);
					increment = -increment;
					sign *= -1;
				} else {
					/* go to loop-begin */

					offset = loop_begin + (offset - loop_end);
				}
			} else {
				/* no loop, check for end of sample */
				if ((uint64_t)offset >= len) {
					active = false;
					break;
				}
			}
		}

		/** MIXCOUNT COMPUTING **/

		/* next possible limit (looppoints or sample begin/end */
		limit = (increment < 0) ? begin_limit : end_limit;

		/* compute what is shorter, the todo or the limit? */
		aux = (limit - offset) / increment + 1;
		target = (aux < todo) ? aux : todo; /* mix target is the shorter buffer */

		/* check just in case */
		if (target <= 0) {
			active = false;
			break;
		}

		todo -= target;

		dst_buff += target;
	}

	if (todo) {
		int mixed_frames = p_frames - todo;
		//bit was missing from mix
		int todo_ofs = p_frames - todo;
		for (int i = todo_ofs; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0, 0);
		}
		return mixed_frames;
	}
	return p_frames;
}

int PlaceholderAudioStreamPlayback::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	constexpr int INTERNAL_BUFFER_LEN = 128;
	int mixed_frames_total = -1;

	int i;
	for (i = 0; i < p_frames; i++) {
		AudioFrame af;
		_mix_internal(&af, INTERNAL_BUFFER_LEN);
	}
	if (mixed_frames_total == -1 && i == p_frames) {
		mixed_frames_total = p_frames;
	}
	return mixed_frames_total;
}

///////////

void PlaceholderAudioStream::set_length(double p_length) {
	length = p_length;
}

double PlaceholderAudioStream::get_length() const {
	return length;
}

void PlaceholderAudioStream::set_loop_mode(const LoopMode p_loop_mode) {
	loop_mode = p_loop_mode;
}

PlaceholderAudioStream::LoopMode PlaceholderAudioStream::get_loop_mode() const {
	return loop_mode;
}

void PlaceholderAudioStream::set_loop_begin(int p_frame) {
	loop_begin = p_frame;
}

int PlaceholderAudioStream::get_loop_begin() const {
	return loop_begin;
}

void PlaceholderAudioStream::set_loop_end(int p_frame) {
	loop_end = p_frame;
}

int PlaceholderAudioStream::get_loop_end() const {
	return loop_end;
}

void PlaceholderAudioStream::set_tags(const Dictionary &p_tags) {
	tags = p_tags;
}

Dictionary PlaceholderAudioStream::get_tags() const {
	return tags;
}

Ref<AudioStreamPlayback> PlaceholderAudioStream::instantiate_playback() {
	Ref<PlaceholderAudioStreamPlayback> sample;
	sample.instantiate();
	sample->base = Ref<PlaceholderAudioStream>(this);
	return sample;
}

void PlaceholderAudioStream::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_length"), &PlaceholderAudioStream::set_length);

	ClassDB::bind_method(D_METHOD("set_loop_mode"), &PlaceholderAudioStream::set_loop_mode);
	ClassDB::bind_method(D_METHOD("get_loop_mode"), &PlaceholderAudioStream::get_loop_mode);

	ClassDB::bind_method(D_METHOD("set_loop_begin", "loop_begin"), &PlaceholderAudioStream::set_loop_begin);
	ClassDB::bind_method(D_METHOD("get_loop_begin"), &PlaceholderAudioStream::get_loop_begin);

	ClassDB::bind_method(D_METHOD("set_loop_end", "loop_end"), &PlaceholderAudioStream::set_loop_end);
	ClassDB::bind_method(D_METHOD("get_loop_end"), &PlaceholderAudioStream::get_loop_end);

	ClassDB::bind_method(D_METHOD("set_tags"), &PlaceholderAudioStream::set_tags);
	ClassDB::bind_method(D_METHOD("get_tags"), &PlaceholderAudioStream::get_tags);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "0.0,100.0,0.001,or_greater,suffix:s"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_mode", PROPERTY_HINT_ENUM, "Disabled,Forward,Ping-Pong,Backward"), "set_loop_mode", "get_loop_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_begin"), "set_loop_begin", "get_loop_begin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_end"), "set_loop_end", "get_loop_end");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "tags", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_tags", "get_tags");

	BIND_ENUM_CONSTANT(LOOP_DISABLED);
	BIND_ENUM_CONSTANT(LOOP_FORWARD);
	BIND_ENUM_CONSTANT(LOOP_PINGPONG);
	BIND_ENUM_CONSTANT(LOOP_BACKWARD);
}

PlaceholderAudioStream::PlaceholderAudioStream() {
}

PlaceholderAudioStream::~PlaceholderAudioStream() {
}
