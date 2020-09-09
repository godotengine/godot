/*************************************************************************/
/*  audio_stream_ogg_vorbis.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "audio_stream_ogg_vorbis.h"

#include "core/os/file_access.h"

void AudioStreamPlaybackOGGVorbis::_mix_internal(AudioFrame *p_buffer, int p_frames) {
	ERR_FAIL_COND(!active);

	int todo = p_frames;

	int start_buffer = 0;

	while (todo && active) {
		float *buffer = (float *)p_buffer;
		if (start_buffer > 0) {
			buffer = (buffer + start_buffer * 2);
		}
		int mixed = stb_vorbis_get_samples_float_interleaved(ogg_stream, 2, buffer, todo * 2);
		if (vorbis_stream->channels == 1 && mixed > 0) {
			//mix mono to stereo
			for (int i = start_buffer; i < mixed; i++) {
				p_buffer[i].r = p_buffer[i].l;
			}
		}
		todo -= mixed;
		frames_mixed += mixed;

		if (todo) {
			//end of file!
			bool is_not_empty = mixed > 0 || stb_vorbis_stream_length_in_samples(ogg_stream) > 0;
			if (vorbis_stream->loop && is_not_empty) {
				//loop
				seek(vorbis_stream->loop_offset);
				loops++;
				// we still have buffer to fill, start from this element in the next iteration.
				start_buffer = p_frames - todo;
			} else {
				for (int i = p_frames - todo; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
				todo = 0;
			}
		}
	}
}

float AudioStreamPlaybackOGGVorbis::get_stream_sampling_rate() {
	return vorbis_stream->sample_rate;
}

void AudioStreamPlaybackOGGVorbis::start(float p_from_pos) {
	active = true;
	seek(p_from_pos);
	loops = 0;
	_begin_resample();
}

void AudioStreamPlaybackOGGVorbis::stop() {
	active = false;
}

bool AudioStreamPlaybackOGGVorbis::is_playing() const {
	return active;
}

int AudioStreamPlaybackOGGVorbis::get_loop_count() const {
	return loops;
}

float AudioStreamPlaybackOGGVorbis::get_playback_position() const {
	return float(frames_mixed) / vorbis_stream->sample_rate;
}

void AudioStreamPlaybackOGGVorbis::seek(float p_time) {
	if (!active) {
		return;
	}

	if (p_time >= vorbis_stream->get_length()) {
		p_time = 0;
	}
	frames_mixed = uint32_t(vorbis_stream->sample_rate * p_time);

	stb_vorbis_seek(ogg_stream, frames_mixed);
}

AudioStreamPlaybackOGGVorbis::~AudioStreamPlaybackOGGVorbis() {
	if (ogg_alloc.alloc_buffer) {
		stb_vorbis_close(ogg_stream);
		memfree(ogg_alloc.alloc_buffer);
	}
}

Ref<AudioStreamPlayback> AudioStreamOGGVorbis::instance_playback() {
	Ref<AudioStreamPlaybackOGGVorbis> ovs;

	ERR_FAIL_COND_V(data == nullptr, ovs);

	ovs.instance();
	ovs->vorbis_stream = Ref<AudioStreamOGGVorbis>(this);
	ovs->ogg_alloc.alloc_buffer = (char *)memalloc(decode_mem_size);
	ovs->ogg_alloc.alloc_buffer_length_in_bytes = decode_mem_size;
	ovs->frames_mixed = 0;
	ovs->active = false;
	ovs->loops = 0;
	int error;
	ovs->ogg_stream = stb_vorbis_open_memory((const unsigned char *)data, data_len, &error, &ovs->ogg_alloc);
	if (!ovs->ogg_stream) {
		memfree(ovs->ogg_alloc.alloc_buffer);
		ovs->ogg_alloc.alloc_buffer = nullptr;
		ERR_FAIL_COND_V(!ovs->ogg_stream, Ref<AudioStreamPlaybackOGGVorbis>());
	}

	return ovs;
}

String AudioStreamOGGVorbis::get_stream_name() const {
	return ""; //return stream_name;
}

void AudioStreamOGGVorbis::clear_data() {
	if (data) {
		memfree(data);
		data = nullptr;
		data_len = 0;
	}
}

void AudioStreamOGGVorbis::set_data(const Vector<uint8_t> &p_data) {
	int src_data_len = p_data.size();
	uint32_t alloc_try = 1024;
	Vector<char> alloc_mem;
	char *w;
	stb_vorbis *ogg_stream = nullptr;
	stb_vorbis_alloc ogg_alloc;

	// Vorbis comments may be up to UINT32_MAX, but that's arguably pretty rare.
	// Let's go with 2^30 so we don't risk going out of bounds.
	const uint32_t MAX_TEST_MEM = 1 << 30;

	while (alloc_try < MAX_TEST_MEM) {
		alloc_mem.resize(alloc_try);
		w = alloc_mem.ptrw();

		ogg_alloc.alloc_buffer = w;
		ogg_alloc.alloc_buffer_length_in_bytes = alloc_try;

		const uint8_t *src_datar = p_data.ptr();

		int error;
		ogg_stream = stb_vorbis_open_memory((const unsigned char *)src_datar, src_data_len, &error, &ogg_alloc);

		if (!ogg_stream && error == VORBIS_outofmem) {
			alloc_try *= 2;
		} else {
			ERR_FAIL_COND(alloc_try == MAX_TEST_MEM);
			ERR_FAIL_COND(ogg_stream == nullptr);

			stb_vorbis_info info = stb_vorbis_get_info(ogg_stream);

			channels = info.channels;
			sample_rate = info.sample_rate;
			decode_mem_size = alloc_try;
			//does this work? (it's less mem..)
			//decode_mem_size = ogg_alloc.alloc_buffer_length_in_bytes + info.setup_memory_required + info.temp_memory_required + info.max_frame_size;

			length = stb_vorbis_stream_length_in_seconds(ogg_stream);
			stb_vorbis_close(ogg_stream);

			// free any existing data
			clear_data();

			data = memalloc(src_data_len);
			copymem(data, src_datar, src_data_len);
			data_len = src_data_len;

			break;
		}
	}

	ERR_FAIL_COND_MSG(alloc_try == MAX_TEST_MEM, vformat("Couldn't set vorbis data even with an alloc buffer of %d bytes, report bug.", MAX_TEST_MEM));
}

Vector<uint8_t> AudioStreamOGGVorbis::get_data() const {
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

void AudioStreamOGGVorbis::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamOGGVorbis::has_loop() const {
	return loop;
}

void AudioStreamOGGVorbis::set_loop_offset(float p_seconds) {
	loop_offset = p_seconds;
}

float AudioStreamOGGVorbis::get_loop_offset() const {
	return loop_offset;
}

float AudioStreamOGGVorbis::get_length() const {
	return length;
}

void AudioStreamOGGVorbis::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamOGGVorbis::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamOGGVorbis::get_data);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamOGGVorbis::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamOGGVorbis::has_loop);

	ClassDB::bind_method(D_METHOD("set_loop_offset", "seconds"), &AudioStreamOGGVorbis::set_loop_offset);
	ClassDB::bind_method(D_METHOD("get_loop_offset"), &AudioStreamOGGVorbis::get_loop_offset);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop"), "set_loop", "has_loop");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "loop_offset"), "set_loop_offset", "get_loop_offset");
}

AudioStreamOGGVorbis::AudioStreamOGGVorbis() {
	data = nullptr;
	data_len = 0;
	length = 0;
	sample_rate = 1;
	channels = 1;
	loop_offset = 0;
	decode_mem_size = 0;
	loop = false;
}

AudioStreamOGGVorbis::~AudioStreamOGGVorbis() {
	clear_data();
}
