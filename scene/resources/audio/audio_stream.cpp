/**************************************************************************/
/*  audio_stream.cpp                                                      */
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

#include "audio_stream.h"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "servers/audio/audio_driver.h"
#include "servers/audio/audio_server.h"

void AudioStreamPlayback::start(double p_from_pos) {
	GDVIRTUAL_CALL(_start, p_from_pos);
}

void AudioStreamPlayback::stop() {
	GDVIRTUAL_CALL(_stop);
}

bool AudioStreamPlayback::is_playing() const {
	bool ret = false;
	GDVIRTUAL_CALL(_is_playing, ret);
	return ret;
}

int AudioStreamPlayback::get_loop_count() const {
	int ret = 0;
	GDVIRTUAL_CALL(_get_loop_count, ret);
	return ret;
}

double AudioStreamPlayback::get_playback_position() const {
	double ret = 0.0;
	GDVIRTUAL_CALL(_get_playback_position, ret);
	return ret;
}

void AudioStreamPlayback::seek(double p_time) {
	GDVIRTUAL_CALL(_seek, p_time);
}

int AudioStreamPlayback::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	int ret = 0;
	GDVIRTUAL_CALL(_mix, p_buffer, p_rate_scale, p_frames, ret);
	return ret;
}

PackedVector2Array AudioStreamPlayback::_mix_audio_bind(float p_rate_scale, int p_frames) {
	Vector<AudioFrame> frames = mix_audio(p_rate_scale, p_frames);

	PackedVector2Array res;
	res.resize(frames.size());

	Vector2 *res_ptrw = res.ptrw();
	for (int i = 0; i < frames.size(); i++) {
		res_ptrw[i] = Vector2(frames[i].left, frames[i].right);
	}

	return res;
}

Vector<AudioFrame> AudioStreamPlayback::mix_audio(float p_rate_scale, int p_frames) {
	Vector<AudioFrame> res;
	res.resize(p_frames);

	int frames = mix(res.ptrw(), p_rate_scale, p_frames);
	res.resize(frames);

	return res;
}

void AudioStreamPlayback::start_playback(double p_from_pos) {
	start(p_from_pos);
}

void AudioStreamPlayback::stop_playback() {
	stop();
}

void AudioStreamPlayback::seek_playback(double p_time) {
	seek(p_time);
}

void AudioStreamPlayback::tag_used_streams() {
	GDVIRTUAL_CALL(_tag_used_streams);
}

void AudioStreamPlayback::set_parameter(const StringName &p_name, const Variant &p_value) {
	GDVIRTUAL_CALL(_set_parameter, p_name, p_value);
}

Variant AudioStreamPlayback::get_parameter(const StringName &p_name) const {
	Variant ret;
	GDVIRTUAL_CALL(_get_parameter, p_name, ret);
	return ret;
}

Ref<AudioSamplePlayback> AudioStreamPlayback::get_sample_playback() const {
	return nullptr;
}

void AudioStreamPlayback::_bind_methods() {
	GDVIRTUAL_BIND(_start, "from_pos")
	GDVIRTUAL_BIND(_stop)
	GDVIRTUAL_BIND(_is_playing)
	GDVIRTUAL_BIND(_get_loop_count)
	GDVIRTUAL_BIND(_get_playback_position)
	GDVIRTUAL_BIND(_seek, "position")
	GDVIRTUAL_BIND(_mix, "buffer", "rate_scale", "frames");
	GDVIRTUAL_BIND(_tag_used_streams);
	GDVIRTUAL_BIND(_set_parameter, "name", "value");
	GDVIRTUAL_BIND(_get_parameter, "name");

	ClassDB::bind_method(D_METHOD("set_sample_playback", "playback_sample"), &AudioStreamPlayback::set_sample_playback);
	ClassDB::bind_method(D_METHOD("get_sample_playback"), &AudioStreamPlayback::get_sample_playback);
	ClassDB::bind_method(D_METHOD("mix_audio", "rate_scale", "frames"), &AudioStreamPlayback::_mix_audio_bind);
	ClassDB::bind_method(D_METHOD("start", "from_pos"), &AudioStreamPlayback::start_playback, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("seek", "time"), &AudioStreamPlayback::seek_playback, DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("stop"), &AudioStreamPlayback::stop_playback);
	ClassDB::bind_method(D_METHOD("get_loop_count"), &AudioStreamPlayback::get_loop_count);
	ClassDB::bind_method(D_METHOD("get_playback_position"), &AudioStreamPlayback::get_playback_position);
	ClassDB::bind_method(D_METHOD("is_playing"), &AudioStreamPlayback::is_playing);
}

AudioStreamPlayback::AudioStreamPlayback() {}

AudioStreamPlayback::~AudioStreamPlayback() {
	if (get_sample_playback().is_valid() && likely(AudioServer::get_singleton() != nullptr)) {
		AudioServer::get_singleton()->stop_sample_playback(get_sample_playback());
	}
}

//////////////////////////////

void AudioStreamPlaybackResampled::begin_resample() {
	//clear cubic interpolation history
	internal_buffer[0] = AudioFrame(0.0, 0.0);
	internal_buffer[1] = AudioFrame(0.0, 0.0);
	internal_buffer[2] = AudioFrame(0.0, 0.0);
	internal_buffer[3] = AudioFrame(0.0, 0.0);
	//mix buffer
	_mix_internal(internal_buffer + 4, INTERNAL_BUFFER_LEN);
	mix_offset = 0;
}

int AudioStreamPlaybackResampled::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	int ret = 0;
	GDVIRTUAL_CALL(_mix_resampled, p_buffer, p_frames, ret);
	return ret;
}
float AudioStreamPlaybackResampled::get_stream_sampling_rate() {
	float ret = 0;
	GDVIRTUAL_CALL(_get_stream_sampling_rate, ret);
	return ret;
}

void AudioStreamPlaybackResampled::_bind_methods() {
	ClassDB::bind_method(D_METHOD("begin_resample"), &AudioStreamPlaybackResampled::begin_resample);

	GDVIRTUAL_BIND(_mix_resampled, "dst_buffer", "frame_count");
	GDVIRTUAL_BIND(_get_stream_sampling_rate);
}

int AudioStreamPlaybackResampled::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {
	float target_rate = AudioServer::get_singleton()->get_mix_rate();
	float playback_speed_scale = AudioServer::get_singleton()->get_playback_speed_scale();

	uint64_t mix_increment = uint64_t(((get_stream_sampling_rate() * p_rate_scale * playback_speed_scale) / double(target_rate)) * double(FP_LEN));

	int mixed_frames_total = -1;

	int i;
	for (i = 0; i < p_frames; i++) {
		uint32_t idx = CUBIC_INTERP_HISTORY + uint32_t(mix_offset >> FP_BITS);
		//standard cubic interpolation (great quality/performance ratio)
		//this used to be moved to a LUT for greater performance, but nowadays CPU speed is generally faster than memory.
		float mu = (mix_offset & FP_MASK) / float(FP_LEN);
		AudioFrame y0 = internal_buffer[idx - 3];
		AudioFrame y1 = internal_buffer[idx - 2];
		AudioFrame y2 = internal_buffer[idx - 1];
		AudioFrame y3 = internal_buffer[idx - 0];

		if (idx >= internal_buffer_end && mixed_frames_total == -1) {
			// The internal buffer ends somewhere in this range, and we haven't yet recorded the number of good frames we have.
			mixed_frames_total = i;
		}

		float mu2 = mu * mu;
		float h11 = mu2 * (mu - 1);
		float z = mu2 - h11;
		float h01 = z - h11;
		float h10 = mu - z;

		p_buffer[i] = y1 + (y2 - y1) * h01 + ((y2 - y0) * h10 + (y3 - y1) * h11) * 0.5;

		mix_offset += mix_increment;

		while ((mix_offset >> FP_BITS) >= INTERNAL_BUFFER_LEN) {
			internal_buffer[0] = internal_buffer[INTERNAL_BUFFER_LEN + 0];
			internal_buffer[1] = internal_buffer[INTERNAL_BUFFER_LEN + 1];
			internal_buffer[2] = internal_buffer[INTERNAL_BUFFER_LEN + 2];
			internal_buffer[3] = internal_buffer[INTERNAL_BUFFER_LEN + 3];
			int mixed_frames = _mix_internal(internal_buffer + 4, INTERNAL_BUFFER_LEN);
			if (mixed_frames != INTERNAL_BUFFER_LEN) {
				// internal_buffer[mixed_frames] is the first frame of silence.
				internal_buffer_end = mixed_frames;
			} else {
				// The internal buffer does not contain the first frame of silence.
				internal_buffer_end = -1;
			}
			mix_offset -= (INTERNAL_BUFFER_LEN << FP_BITS);
		}
	}
	if (mixed_frames_total == -1 && i == p_frames) {
		mixed_frames_total = p_frames;
	}
	return mixed_frames_total;
}

////////////////////////////////

Ref<AudioStreamPlayback> AudioStream::instantiate_playback() {
	Ref<AudioStreamPlayback> ret;
	GDVIRTUAL_CALL(_instantiate_playback, ret);
	return ret;
}

double AudioStream::get_length() const {
	double ret = 0;
	GDVIRTUAL_CALL(_get_length, ret);
	return ret;
}

bool AudioStream::is_monophonic() const {
	bool ret = true;
	GDVIRTUAL_CALL(_is_monophonic, ret);
	return ret;
}

double AudioStream::get_bpm() const {
	double ret = 0;
	GDVIRTUAL_CALL(_get_bpm, ret);
	return ret;
}

bool AudioStream::has_loop() const {
	bool ret = false;
	GDVIRTUAL_CALL(_has_loop, ret);
	return ret;
}

int AudioStream::get_bar_beats() const {
	int ret = 0;
	GDVIRTUAL_CALL(_get_bar_beats, ret);
	return ret;
}

int AudioStream::get_beat_count() const {
	int ret = 0;
	GDVIRTUAL_CALL(_get_beat_count, ret);
	return ret;
}

Dictionary AudioStream::get_tags() const {
	Dictionary ret;
	GDVIRTUAL_CALL(_get_tags, ret);
	return ret;
}

void AudioStream::tag_used(float p_offset) {
	if (tagged_frame != AudioServer::get_singleton()->get_mixed_frames()) {
		offset_count = 0;
		tagged_frame = AudioServer::get_singleton()->get_mixed_frames();
	}
	if (offset_count < MAX_TAGGED_OFFSETS) {
		tagged_offsets[offset_count++] = p_offset;
	}
}

uint64_t AudioStream::get_tagged_frame() const {
	return tagged_frame;
}

uint32_t AudioStream::get_tagged_frame_count() const {
	return offset_count;
}

float AudioStream::get_tagged_frame_offset(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, MAX_TAGGED_OFFSETS, 0);
	return tagged_offsets[p_index];
}

void AudioStream::get_parameter_list(List<Parameter> *r_parameters) {
	TypedArray<Dictionary> ret;
	GDVIRTUAL_CALL(_get_parameter_list, ret);
	for (int i = 0; i < ret.size(); i++) {
		Dictionary d = ret[i];
		ERR_CONTINUE(!d.has("default_value"));
		r_parameters->push_back(Parameter(PropertyInfo::from_dict(d), d["default_value"]));
	}
}

Ref<AudioSample> AudioStream::generate_sample() const {
	ERR_FAIL_COND_V_MSG(!can_be_sampled(), nullptr, "Cannot generate a sample for a stream that cannot be sampled.");
	Ref<AudioSample> sample;
	sample.instantiate();
	sample->stream = this;
	return sample;
}

void AudioStream::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_length"), &AudioStream::get_length);
	ClassDB::bind_method(D_METHOD("is_monophonic"), &AudioStream::is_monophonic);
	ClassDB::bind_method(D_METHOD("instantiate_playback"), &AudioStream::instantiate_playback);
	ClassDB::bind_method(D_METHOD("can_be_sampled"), &AudioStream::can_be_sampled);
	ClassDB::bind_method(D_METHOD("generate_sample"), &AudioStream::generate_sample);
	ClassDB::bind_method(D_METHOD("is_meta_stream"), &AudioStream::is_meta_stream);

	GDVIRTUAL_BIND(_instantiate_playback);
#ifndef DISABLE_DEPRECATED
	GDVIRTUAL_BIND(_get_stream_name);
#endif
	GDVIRTUAL_BIND(_get_length);
	GDVIRTUAL_BIND(_is_monophonic);
	GDVIRTUAL_BIND(_get_bpm)
	GDVIRTUAL_BIND(_get_beat_count)
	GDVIRTUAL_BIND(_get_tags);
	GDVIRTUAL_BIND(_get_parameter_list)
	GDVIRTUAL_BIND(_has_loop);
	GDVIRTUAL_BIND(_get_bar_beats);

	ADD_SIGNAL(MethodInfo("parameter_list_changed"));
}
