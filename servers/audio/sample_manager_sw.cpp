/*************************************************************************/
/*  sample_manager_sw.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "sample_manager_sw.h"

#include "print_string.h"

SampleManagerSW::~SampleManagerSW() {
}

RID SampleManagerMallocSW::sample_create(AS::SampleFormat p_format, bool p_stereo, int p_length) {

	Sample *s = memnew(Sample);
	int datalen = p_length;
	if (p_format == AS::SAMPLE_FORMAT_PCM16)
		datalen *= 2;
	else if (p_format == AS::SAMPLE_FORMAT_IMA_ADPCM) {
		if (datalen & 1) {
			datalen++;
		}
		datalen /= 2;
		datalen += 4;
	}

	if (p_stereo)
		datalen *= 2;

#define SAMPLE_EXTRA 16

	s->data = memalloc(datalen + SAMPLE_EXTRA); //help the interpolator by allocating a little more..
	for (int i = 0; i < SAMPLE_EXTRA; i++) {

		uint8_t *data = (uint8_t *)s->data;
		data[datalen + i] = 0;
	}
	if (!s->data) {

		memdelete(s);
		ERR_EXPLAIN("Cannot allocate sample of requested size.");
		ERR_FAIL_V(RID());
	}

	s->format = p_format;
	s->length = p_length;
	s->length_bytes = datalen;
	s->stereo = p_stereo;
	s->loop_begin = 0;
	s->loop_end = 0;
	s->loop_format = AS::SAMPLE_LOOP_NONE;
	s->mix_rate = 44100;

	AudioServer::get_singleton()->lock();
	RID rid = sample_owner.make_rid(s);
	AudioServer::get_singleton()->unlock();

	return rid;
}

void SampleManagerMallocSW::sample_set_description(RID p_sample, const String &p_description) {

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND(!s);

	s->description = p_description;
}

String SampleManagerMallocSW::sample_get_description(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, String());

	return s->description;
}

AS::SampleFormat SampleManagerMallocSW::sample_get_format(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, AS::SAMPLE_FORMAT_PCM8);

	return s->format;
}

bool SampleManagerMallocSW::sample_is_stereo(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, false);

	return s->stereo;
}
int SampleManagerMallocSW::sample_get_length(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, -1);

	return s->length;
}

void SampleManagerMallocSW::sample_set_data(RID p_sample, const DVector<uint8_t> &p_buffer) {

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND(!s);

	int buff_size = p_buffer.size();
	ERR_FAIL_COND(buff_size == 0);

	ERR_EXPLAIN("Sample buffer size does not match sample size.");
	//print_line("len bytes: "+itos(s->length_bytes)+" bufsize: "+itos(buff_size));
	ERR_FAIL_COND(s->length_bytes != buff_size);
	DVector<uint8_t>::Read buffer_r = p_buffer.read();
	const uint8_t *src = buffer_r.ptr();
	uint8_t *dst = (uint8_t *)s->data;
	//print_line("set data: "+itos(s->length_bytes));

	for (int i = 0; i < s->length_bytes; i++) {

		dst[i] = src[i];
	}

	switch (s->format) {

		case AS::SAMPLE_FORMAT_PCM8: {

			if (s->stereo) {
				dst[s->length] = dst[s->length - 2];
				dst[s->length + 1] = dst[s->length - 1];
			} else {

				dst[s->length] = dst[s->length - 1];
			}

		} break;
		case AS::SAMPLE_FORMAT_PCM16: {

			if (s->stereo) {
				dst[s->length] = dst[s->length - 4];
				dst[s->length + 1] = dst[s->length - 3];
				dst[s->length + 2] = dst[s->length - 2];
				dst[s->length + 3] = dst[s->length - 1];
			} else {

				dst[s->length] = dst[s->length - 2];
				dst[s->length + 1] = dst[s->length - 1];
			}

		} break;
	}
}

const DVector<uint8_t> SampleManagerMallocSW::sample_get_data(RID p_sample) const {

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, DVector<uint8_t>());

	DVector<uint8_t> ret_buffer;
	ret_buffer.resize(s->length_bytes);
	DVector<uint8_t>::Write buffer_w = ret_buffer.write();
	uint8_t *dst = buffer_w.ptr();
	const uint8_t *src = (const uint8_t *)s->data;

	for (int i = 0; i < s->length_bytes; i++) {

		dst[i] = src[i];
	}

	buffer_w = DVector<uint8_t>::Write(); //unlock

	return ret_buffer;
}

void *SampleManagerMallocSW::sample_get_data_ptr(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, NULL);

	return s->data;
}

void SampleManagerMallocSW::sample_set_mix_rate(RID p_sample, int p_rate) {

	ERR_FAIL_COND(p_rate < 1);

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND(!s);

	s->mix_rate = p_rate;
}
int SampleManagerMallocSW::sample_get_mix_rate(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, -1);

	return s->mix_rate;
}
void SampleManagerMallocSW::sample_set_loop_format(RID p_sample, AS::SampleLoopFormat p_format) {

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND(!s);

	s->loop_format = p_format;
}
AS::SampleLoopFormat SampleManagerMallocSW::sample_get_loop_format(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, AS::SAMPLE_LOOP_NONE);

	return s->loop_format;
}

void SampleManagerMallocSW::sample_set_loop_begin(RID p_sample, int p_pos) {

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND(!s);
	ERR_FAIL_INDEX(p_pos, s->length);

	s->loop_begin = p_pos;
}
int SampleManagerMallocSW::sample_get_loop_begin(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, -1);

	return s->loop_begin;
}

void SampleManagerMallocSW::sample_set_loop_end(RID p_sample, int p_pos) {

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND(!s);
	if (p_pos > s->length)
		p_pos = s->length;
	s->loop_end = p_pos;
}
int SampleManagerMallocSW::sample_get_loop_end(RID p_sample) const {

	const Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND_V(!s, -1);

	return s->loop_end;
}

bool SampleManagerMallocSW::is_sample(RID p_sample) const {

	return sample_owner.owns(p_sample);
}
void SampleManagerMallocSW::free(RID p_sample) {

	Sample *s = sample_owner.get(p_sample);
	ERR_FAIL_COND(!s);
	AudioServer::get_singleton()->lock();
	sample_owner.free(p_sample);
	AudioServer::get_singleton()->unlock();

	memfree(s->data);
	memdelete(s);
}

SampleManagerMallocSW::SampleManagerMallocSW() {
}

SampleManagerMallocSW::~SampleManagerMallocSW() {

	// check for sample leakage
	List<RID> owned_list;
	sample_owner.get_owned_list(&owned_list);

	while (owned_list.size()) {

		Sample *s = sample_owner.get(owned_list.front()->get());
		String err = "Leaked sample of size: " + itos(s->length_bytes) + " description: " + s->description;
		ERR_PRINT(err.utf8().get_data());
		free(owned_list.front()->get());
		owned_list.pop_front();
	}
}
