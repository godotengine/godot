/**************************************************************************/
/*  audio_stream_playback_scheduled.cpp                                   */
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

#include "audio_stream_playback_scheduled.h"

#include "servers/audio/audio_server.h"

void AudioStreamPlaybackScheduled::start(double p_from_pos) {
	ERR_FAIL_COND_MSG(base_playback.is_null(), "base_playback must be set.");

	active = true;
	base_playback->start(p_from_pos);

	uint64_t mix_start = AudioServer::get_singleton()->get_mixed_frames();
	if (mix_start > scheduled_start_frame) {
		WARN_PRINT_ED(vformat("Sound (%s) was scheduled for absolute time %.4f, which has already passed. Playing immediately.",
				get_instance_id(),
				get_scheduled_start_time()));
	}
}

void AudioStreamPlaybackScheduled::stop() {
	active = false;
	if (base_playback.is_null()) {
		return;
	}
	base_playback->stop();
}

bool AudioStreamPlaybackScheduled::is_playing() const {
	if (base_playback.is_null()) {
		return false;
	}
	uint64_t mix_start = AudioServer::get_singleton()->get_mixed_frames();
	if (mix_start < scheduled_start_frame) {
		return false;
	}
	return base_playback->is_playing();
}

int AudioStreamPlaybackScheduled::get_loop_count() const {
	if (base_playback.is_null()) {
		return 0;
	}
	return base_playback->get_loop_count();
}

double AudioStreamPlaybackScheduled::get_playback_position() const {
	if (base_playback.is_null()) {
		return 0.0;
	}
	return base_playback->get_playback_position();
}

void AudioStreamPlaybackScheduled::seek(double p_time) {
	if (base_playback.is_null()) {
		return;
	}
	base_playback->seek(p_time);
}

void AudioStreamPlaybackScheduled::tag_used_streams() {
	if (base_playback.is_null()) {
		return;
	}
	base_playback->tag_used_streams();
}

void AudioStreamPlaybackScheduled::set_parameter(const StringName &p_name, const Variant &p_value) {
	if (base_playback.is_null()) {
		return;
	}
	base_playback->set_parameter(p_name, p_value);
}

Variant AudioStreamPlaybackScheduled::get_parameter(const StringName &p_name) const {
	if (base_playback.is_null()) {
		return Variant();
	}
	return base_playback->get_parameter(p_name);
}

int AudioStreamPlaybackScheduled::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	// Pre-clear buffer.
	for (int i = 0; i < p_frames; i++) {
		p_buffer[i] = AudioFrame(0, 0);
	}

	if (!active) {
		return 0;
	}
	if (base_playback.is_null()) {
		return 0;
	}
	ERR_FAIL_COND_V_MSG(scheduled_end_frame > 0 && scheduled_end_frame < scheduled_start_frame, 0, "Scheduled end time is before scheduled start time.");

	AudioFrame *buf = p_buffer;
	unsigned int todo = p_frames;
	uint64_t mix_start = AudioServer::get_singleton()->get_mixed_frames();
	uint64_t mix_end = mix_start + p_frames;

	// Fill buffer with silence frames if sound isn't scheduled to start yet.
	if (mix_start < scheduled_start_frame) {
		unsigned int silence_frames = (unsigned int)MIN(scheduled_start_frame - mix_start, todo);
		for (unsigned int i = 0; i < silence_frames; i++) {
			buf[i] = AudioFrame(0, 0);
		}
		todo -= silence_frames;
		buf += silence_frames;

		if (todo == 0) {
			return p_frames;
		}
	}

	// Limit the amount of mixing if the sound is scheduled to end within this
	// chunk.
	if (scheduled_end_frame > 0 && scheduled_end_frame < mix_end) {
		unsigned int frames_to_skip = (unsigned int)MIN(mix_end - scheduled_end_frame, todo);
		todo -= base_playback->mix(buf, p_rate_scale, todo - frames_to_skip);
		stop();
		return p_frames - todo;
	}

	// Normal mixing.
	todo -= base_playback->mix(buf, p_rate_scale, todo);

	return p_frames - todo;
}

void AudioStreamPlaybackScheduled::set_is_sample(bool p_is_sample) {
	if (base_playback.is_null()) {
		return;
	}
	base_playback->set_is_sample(p_is_sample);
}

bool AudioStreamPlaybackScheduled::get_is_sample() const {
	if (base_playback.is_null()) {
		return false;
	}
	return base_playback->get_is_sample();
}

void AudioStreamPlaybackScheduled::set_sample_playback(const Ref<AudioSamplePlayback> &p_playback) {
	if (base_playback.is_null()) {
		return;
	}
	base_playback->set_sample_playback(p_playback);
}

Ref<AudioSamplePlayback> AudioStreamPlaybackScheduled::get_sample_playback() const {
	if (base_playback.is_null()) {
		return Ref<AudioSamplePlayback>();
	}
	return base_playback->get_sample_playback();
}

void AudioStreamPlaybackScheduled::cancel() {
	uint64_t mix_start = AudioServer::get_singleton()->get_mixed_frames();
	if (mix_start < scheduled_start_frame) {
		stop();
	}
}

bool AudioStreamPlaybackScheduled::is_scheduled() const {
	uint64_t mix_start = AudioServer::get_singleton()->get_mixed_frames();
	return active && mix_start < scheduled_start_frame;
}

void AudioStreamPlaybackScheduled::set_scheduled_start_time(double p_start_time) {
	scheduled_start_frame = int64_t(p_start_time * AudioServer::get_singleton()->get_mix_rate());

	if (active) {
		uint64_t mix_start = AudioServer::get_singleton()->get_mixed_frames();
		if (mix_start > scheduled_start_frame) {
			WARN_PRINT_ED(vformat("Sound (%s) was scheduled for absolute time %.4f, which has already passed. Playing immediately.",
					get_instance_id(),
					get_scheduled_start_time()));
		}
	}
}

double AudioStreamPlaybackScheduled::get_scheduled_start_time() const {
	return scheduled_start_frame / AudioServer::get_singleton()->get_mix_rate();
}

void AudioStreamPlaybackScheduled::set_scheduled_end_time(double p_end_time) {
	scheduled_end_frame = int64_t(p_end_time * AudioServer::get_singleton()->get_mix_rate());
}

double AudioStreamPlaybackScheduled::get_scheduled_end_time() const {
	return scheduled_end_frame / AudioServer::get_singleton()->get_mix_rate();
}

void AudioStreamPlaybackScheduled::set_base_playback(const Ref<AudioStreamPlayback> &p_playback) {
	ERR_FAIL_COND_MSG(p_playback == this, "Cannot assign base_playback to self.");
	stop();
	base_playback = p_playback;
}

Ref<AudioStreamPlayback> AudioStreamPlaybackScheduled::get_base_playback() const {
	return base_playback;
}

void AudioStreamPlaybackScheduled::_bind_methods() {
	ClassDB::bind_method(D_METHOD("cancel"), &AudioStreamPlaybackScheduled::cancel);
	ClassDB::bind_method(D_METHOD("is_scheduled"), &AudioStreamPlaybackScheduled::is_scheduled);

	ClassDB::bind_method(D_METHOD("set_base_playback", "base_playback"), &AudioStreamPlaybackScheduled::set_base_playback);
	ClassDB::bind_method(D_METHOD("get_base_playback"), &AudioStreamPlaybackScheduled::get_base_playback);

	ClassDB::bind_method(D_METHOD("set_scheduled_start_time", "start_time"), &AudioStreamPlaybackScheduled::set_scheduled_start_time);
	ClassDB::bind_method(D_METHOD("get_scheduled_start_time"), &AudioStreamPlaybackScheduled::get_scheduled_start_time);

	ClassDB::bind_method(D_METHOD("set_scheduled_end_time", "end_time"), &AudioStreamPlaybackScheduled::set_scheduled_end_time);
	ClassDB::bind_method(D_METHOD("get_scheduled_end_time"), &AudioStreamPlaybackScheduled::get_scheduled_end_time);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "base_playback", PROPERTY_HINT_RESOURCE_TYPE, "AudioStreamPlayback"), "set_base_playback", "get_base_playback");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scheduled_start_time"), "set_scheduled_start_time", "get_scheduled_start_time");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "scheduled_end_time"), "set_scheduled_end_time", "get_scheduled_end_time");
}
