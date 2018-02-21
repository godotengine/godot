/*************************************************************************/
/*  audio_stream_libopenmpt.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "audio_stream_libopenmpt.h"

#include "os/file_access.h"

void AudioStreamPlaybackLibopenmpt::_mix_internal(AudioFrame *p_buffer, int p_frames) {

	ERR_FAIL_COND(!active);

	int todo = p_frames;

	int start_buffer = 0;

	while (todo && active) {
		float *buffer = (float *)p_buffer;
		if (start_buffer > 0) {
			buffer = (buffer + start_buffer * 2);
		}

		int mixed = openmpt_module->read_interleaved_stereo(libopenmpt_stream->sample_rate, todo, buffer);

		todo -= mixed;
		frames_mixed += mixed;

		if (todo) {
			//end of file!
			if (libopenmpt_stream->loop) {
				//loop
				seek(libopenmpt_stream->loop_offset);
				loops++;
				// we still have buffer to fill, start from this element in the next iteration.
				start_buffer = p_frames - todo;
			}
			else {
				for (int i = p_frames - todo; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
				todo = 0;
			}
		}
	}
}

float AudioStreamPlaybackLibopenmpt::get_stream_sampling_rate() {

	return libopenmpt_stream->sample_rate;
}

void AudioStreamPlaybackLibopenmpt::start(float p_from_pos) {

	active = true;
	seek(p_from_pos);
	loops = 0;
	_begin_resample();
}

void AudioStreamPlaybackLibopenmpt::stop() {

	active = false;
}
bool AudioStreamPlaybackLibopenmpt::is_playing() const {

	return active;
}

int AudioStreamPlaybackLibopenmpt::get_loop_count() const {

	return loops;
}

float AudioStreamPlaybackLibopenmpt::get_playback_position() const {

	return (float)openmpt_module->get_position_seconds();
}

void AudioStreamPlaybackLibopenmpt::seek(float p_time) {

	if (!active)
		return;

	if (p_time >= (float)openmpt_module->get_duration_seconds() || p_time < 0)
		p_time = 0;
	frames_mixed = uint32_t(get_stream_sampling_rate() * p_time);
	openmpt_module->set_position_seconds((float)p_time);
}

AudioStreamPlaybackLibopenmpt::~AudioStreamPlaybackLibopenmpt() {
	if (openmpt_module) {
		delete openmpt_module;
		openmpt_module = NULL;
	}
}

Ref<AudioStreamPlayback> AudioStreamLibopenmpt::instance_playback() {

	Ref<AudioStreamPlaybackLibopenmpt> lom;

	ERR_FAIL_COND_V(data == NULL, lom);

	lom.instance();
	lom->libopenmpt_stream = Ref<AudioStreamLibopenmpt>(this);

	lom->openmpt_module = new openmpt::module(data, data_len);
	ERR_FAIL_COND_V(!lom->openmpt_module, lom);

	lom->frames_mixed = 0;
	lom->active = false;
	lom->loops = 0;

	return lom;
}

String AudioStreamLibopenmpt::get_stream_name() const {

	return ""; //return stream_name;
}

void AudioStreamLibopenmpt::clear_data() {
	if (data) {
		AudioServer::get_singleton()->audio_data_free(data);
		data = NULL;
		data_len = 0;
	}
}

void AudioStreamLibopenmpt::set_data(const PoolVector<uint8_t> &p_data) {

	uint32_t src_data_len = p_data.size();
	PoolVector<uint8_t>::Read src_datar = p_data.read();

	openmpt::module lom((const uint8_t *)src_datar.ptr(), src_data_len);

	// 48000 @ stereo is the desired output format by libopenmpt
	// see https://lib.openmpt.org/doc/libopenmpt_cpp_overview.html
	sample_rate = 48000;
	channels = 2;
	length = (float)lom.get_duration_seconds();

	// free any existing data
	clear_data();

	data = AudioServer::get_singleton()->audio_data_alloc(src_data_len, src_datar.ptr());
	data_len = src_data_len;
}

PoolVector<uint8_t> AudioStreamLibopenmpt::get_data() const {

	PoolVector<uint8_t> vdata;

	if (data_len && data) {
		vdata.resize(data_len);
		{
			PoolVector<uint8_t>::Write w = vdata.write();
			copymem(w.ptr(), data, data_len);
		}
	}

	return vdata;
}

void AudioStreamLibopenmpt::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamLibopenmpt::has_loop() const {

	return loop;
}

void AudioStreamLibopenmpt::set_loop_offset(float p_seconds) {
	loop_offset = p_seconds;
}

float AudioStreamLibopenmpt::get_loop_offset() const {
	return loop_offset;
}

float AudioStreamLibopenmpt::get_length() const {

	return length;
}

void AudioStreamLibopenmpt::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &AudioStreamLibopenmpt::set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &AudioStreamLibopenmpt::get_data);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamLibopenmpt::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamLibopenmpt::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamLibopenmpt::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamLibopenmpt::get_loop_offset);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "loop_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_loop_offset", "get_loop_offset");
}

AudioStreamLibopenmpt::AudioStreamLibopenmpt() {

	data = NULL;
	length = 0;
	sample_rate = 1;
	channels = 1;
	loop_offset = 0;
	loop = false;
}

AudioStreamLibopenmpt::~AudioStreamLibopenmpt() {
	clear_data();
}
