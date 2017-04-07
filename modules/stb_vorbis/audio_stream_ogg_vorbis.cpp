/*************************************************************************/
/*  audio_stream_ogg_vorbis.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "os/file_access.h"
#include "thirdparty/stb_vorbis/stb_vorbis.c"

void AudioStreamPlaybackOGGVorbis::_mix_internal(AudioFrame *p_buffer, int p_frames) {

	ERR_FAIL_COND(!active);

	int todo = p_frames;

	while (todo) {

		int mixed = stb_vorbis_get_samples_float_interleaved(ogg_stream, 2, (float *)p_buffer, todo * 2);
		todo -= mixed;

		if (todo) {
			//end of file!
			if (vorbis_stream->loop) {
				//loop
				seek_pos(0);
				loops++;
			} else {
				for (int i = mixed; i < p_frames; i++) {
					p_buffer[i] = AudioFrame(0, 0);
				}
				active = false;
			}
		}
	}
}

float AudioStreamPlaybackOGGVorbis::get_stream_sampling_rate() {

	return vorbis_stream->sample_rate;
}

void AudioStreamPlaybackOGGVorbis::start(float p_from_pos) {

	seek_pos(p_from_pos);
	active = true;
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

float AudioStreamPlaybackOGGVorbis::get_pos() const {

	return float(frames_mixed) / vorbis_stream->sample_rate;
}
void AudioStreamPlaybackOGGVorbis::seek_pos(float p_time) {

	if (!active)
		return;

	stb_vorbis_seek(ogg_stream, uint32_t(p_time * vorbis_stream->sample_rate));
}

float AudioStreamPlaybackOGGVorbis::get_length() const {

	return vorbis_stream->length;
}

AudioStreamPlaybackOGGVorbis::~AudioStreamPlaybackOGGVorbis() {
	if (ogg_alloc.alloc_buffer) {
		AudioServer::get_singleton()->audio_data_free(ogg_alloc.alloc_buffer);
		stb_vorbis_close(ogg_stream);
	}
}

Ref<AudioStreamPlayback> AudioStreamOGGVorbis::instance_playback() {

	Ref<AudioStreamPlaybackOGGVorbis> ovs;
	printf("instance at %p, data %p\n", this, data);

	ERR_FAIL_COND_V(data == NULL, ovs);

	ovs.instance();
	ovs->vorbis_stream = Ref<AudioStreamOGGVorbis>(this);
	ovs->ogg_alloc.alloc_buffer = (char *)AudioServer::get_singleton()->audio_data_alloc(decode_mem_size);
	ovs->ogg_alloc.alloc_buffer_length_in_bytes = decode_mem_size;
	ovs->frames_mixed = 0;
	ovs->active = false;
	ovs->loops = 0;
	int error;
	ovs->ogg_stream = stb_vorbis_open_memory((const unsigned char *)data, data_len, &error, &ovs->ogg_alloc);
	if (!ovs->ogg_stream) {

		AudioServer::get_singleton()->audio_data_free(ovs->ogg_alloc.alloc_buffer);
		ovs->ogg_alloc.alloc_buffer = NULL;
		ERR_FAIL_COND_V(!ovs->ogg_stream, Ref<AudioStreamPlaybackOGGVorbis>());
	}

	return ovs;
}

String AudioStreamOGGVorbis::get_stream_name() const {

	return ""; //return stream_name;
}

void AudioStreamOGGVorbis::set_data(const PoolVector<uint8_t> &p_data) {

	int src_data_len = p_data.size();
#define MAX_TEST_MEM (1 << 20)

	uint32_t alloc_try = 1024;
	PoolVector<char> alloc_mem;
	PoolVector<char>::Write w;
	stb_vorbis *ogg_stream = NULL;
	stb_vorbis_alloc ogg_alloc;

	while (alloc_try < MAX_TEST_MEM) {

		alloc_mem.resize(alloc_try);
		w = alloc_mem.write();

		ogg_alloc.alloc_buffer = w.ptr();
		ogg_alloc.alloc_buffer_length_in_bytes = alloc_try;

		PoolVector<uint8_t>::Read src_datar = p_data.read();

		int error;
		ogg_stream = stb_vorbis_open_memory((const unsigned char *)src_datar.ptr(), src_data_len, &error, &ogg_alloc);

		if (!ogg_stream && error == VORBIS_outofmem) {
			w = PoolVector<char>::Write();
			alloc_try *= 2;
		} else {

			ERR_FAIL_COND(alloc_try == MAX_TEST_MEM);
			ERR_FAIL_COND(ogg_stream == NULL);

			stb_vorbis_info info = stb_vorbis_get_info(ogg_stream);

			channels = info.channels;
			sample_rate = info.sample_rate;
			decode_mem_size = alloc_try;
			//does this work? (it's less mem..)
			//decode_mem_size = ogg_alloc.alloc_buffer_length_in_bytes + info.setup_memory_required + info.temp_memory_required + info.max_frame_size;

			//print_line("succeeded "+itos(ogg_alloc.alloc_buffer_length_in_bytes)+" setup "+itos(info.setup_memory_required)+" setup temp "+itos(info.setup_temp_memory_required)+" temp "+itos(info.temp_memory_required)+" maxframe"+itos(info.max_frame_size));

			length = stb_vorbis_stream_length_in_seconds(ogg_stream);
			stb_vorbis_close(ogg_stream);

			data = AudioServer::get_singleton()->audio_data_alloc(src_data_len, src_datar.ptr());
			data_len = src_data_len;

			break;
		}
	}

	printf("create at %p, data %p\n", this, data);
}

PoolVector<uint8_t> AudioStreamOGGVorbis::get_data() const {

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

void AudioStreamOGGVorbis::set_loop(bool p_enable) {
	loop = p_enable;
}

bool AudioStreamOGGVorbis::has_loop() const {

	return loop;
}

void AudioStreamOGGVorbis::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamOGGVorbis::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamOGGVorbis::get_data);

	ClassDB::bind_method(D_METHOD("set_loop", "enable"), &AudioStreamOGGVorbis::set_loop);
	ClassDB::bind_method(D_METHOD("has_loop"), &AudioStreamOGGVorbis::has_loop);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_data", "get_data");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "loop", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_loop", "has_loop");
}

AudioStreamOGGVorbis::AudioStreamOGGVorbis() {

	data = NULL;
	length = 0;
	sample_rate = 1;
	channels = 1;
	decode_mem_size = 0;
	loop = false;
}
