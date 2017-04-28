/*************************************************************************/
/*  audio_stream_sample.cpp                                              */
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
#include "audio_stream_sample.h"

void AudioStreamPlaybackSample::start(float p_from_pos) {

	for (int i = 0; i < 2; i++) {
		ima_adpcm[i].step_index = 0;
		ima_adpcm[i].predictor = 0;
		ima_adpcm[i].loop_step_index = 0;
		ima_adpcm[i].loop_predictor = 0;
		ima_adpcm[i].last_nibble = -1;
		ima_adpcm[i].loop_pos = 0x7FFFFFFF;
		ima_adpcm[i].window_ofs = 0;
		ima_adpcm[i].ptr = (const uint8_t *)base->data;
		ima_adpcm[i].ptr += AudioStreamSample::DATA_PAD;
	}

	seek_pos(p_from_pos);
	sign = 1;
	active = true;
}

void AudioStreamPlaybackSample::stop() {

	active = false;
}

bool AudioStreamPlaybackSample::is_playing() const {

	return active;
}

int AudioStreamPlaybackSample::get_loop_count() const {

	return 0;
}

float AudioStreamPlaybackSample::get_pos() const {

	return float(offset >> MIX_FRAC_BITS) / base->mix_rate;
}
void AudioStreamPlaybackSample::seek_pos(float p_time) {

	if (base->format == AudioStreamSample::FORMAT_IMA_ADPCM)
		return; //no seeking in ima-adpcm

	float max = get_length();
	if (p_time < 0) {
		p_time = 0;
	} else if (p_time >= max) {
		p_time = max - 0.001;
	}

	offset = uint64_t(p_time * base->mix_rate) << MIX_FRAC_BITS;
}

template <class Depth, bool is_stereo, bool is_ima_adpcm>
void AudioStreamPlaybackSample::do_resample(const Depth *p_src, AudioFrame *p_dst, int64_t &offset, int32_t &increment, uint32_t amount, IMA_ADPCM_State *ima_adpcm) {

	// this function will be compiled branchless by any decent compiler

	int32_t final, final_r, next, next_r;
	while (amount--) {

		int64_t pos = offset >> MIX_FRAC_BITS;
		if (is_stereo && !is_ima_adpcm)
			pos <<= 1;

		if (is_ima_adpcm) {

			int64_t sample_pos = pos + ima_adpcm[0].window_ofs;

			while (sample_pos > ima_adpcm[0].last_nibble) {

				static const int16_t _ima_adpcm_step_table[89] = {
					7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
					19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
					50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
					130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
					337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
					876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
					2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
					5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
					15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
				};

				static const int8_t _ima_adpcm_index_table[16] = {
					-1, -1, -1, -1, 2, 4, 6, 8,
					-1, -1, -1, -1, 2, 4, 6, 8
				};

				for (int i = 0; i < (is_stereo ? 2 : 1); i++) {

					int16_t nibble, diff, step;

					ima_adpcm[i].last_nibble++;
					const uint8_t *src_ptr = ima_adpcm[i].ptr;

					uint8_t nbb = src_ptr[(ima_adpcm[i].last_nibble >> 1) * (is_stereo ? 2 : 1) + i];
					nibble = (ima_adpcm[i].last_nibble & 1) ? (nbb >> 4) : (nbb & 0xF);
					step = _ima_adpcm_step_table[ima_adpcm[i].step_index];

					ima_adpcm[i].step_index += _ima_adpcm_index_table[nibble];
					if (ima_adpcm[i].step_index < 0)
						ima_adpcm[i].step_index = 0;
					if (ima_adpcm[i].step_index > 88)
						ima_adpcm[i].step_index = 88;

					diff = step >> 3;
					if (nibble & 1)
						diff += step >> 2;
					if (nibble & 2)
						diff += step >> 1;
					if (nibble & 4)
						diff += step;
					if (nibble & 8)
						diff = -diff;

					ima_adpcm[i].predictor += diff;
					if (ima_adpcm[i].predictor < -0x8000)
						ima_adpcm[i].predictor = -0x8000;
					else if (ima_adpcm[i].predictor > 0x7FFF)
						ima_adpcm[i].predictor = 0x7FFF;

					/* store loop if there */
					if (ima_adpcm[i].last_nibble == ima_adpcm[i].loop_pos) {

						ima_adpcm[i].loop_step_index = ima_adpcm[i].step_index;
						ima_adpcm[i].loop_predictor = ima_adpcm[i].predictor;
					}

					//printf("%i - %i - pred %i\n",int(ima_adpcm[i].last_nibble),int(nibble),int(ima_adpcm[i].predictor));
				}
			}

			final = ima_adpcm[0].predictor;
			if (is_stereo) {
				final_r = ima_adpcm[1].predictor;
			}

		} else {
			final = p_src[pos];
			if (is_stereo)
				final_r = p_src[pos + 1];

			if (sizeof(Depth) == 1) { /* conditions will not exist anymore when compiled! */
				final <<= 8;
				if (is_stereo)
					final_r <<= 8;
			}

			if (is_stereo) {

				next = p_src[pos + 2];
				next_r = p_src[pos + 3];
			} else {
				next = p_src[pos + 1];
			}

			if (sizeof(Depth) == 1) {
				next <<= 8;
				if (is_stereo)
					next_r <<= 8;
			}

			int32_t frac = int64_t(offset & MIX_FRAC_MASK);

			final = final + ((next - final) * frac >> MIX_FRAC_BITS);
			if (is_stereo)
				final_r = final_r + ((next_r - final_r) * frac >> MIX_FRAC_BITS);
		}

		if (!is_stereo) {
			final_r = final; //copy to right channel if stereo
		}

		p_dst->l = final / 32767.0;
		p_dst->r = final_r / 32767.0;
		p_dst++;

		offset += increment;
	}
}

void AudioStreamPlaybackSample::mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) {

	if (!base->data || !active) {
		for (int i = 0; i < p_frames; i++) {
			p_buffer[i] = AudioFrame(0, 0);
		}
		return;
	}

	int len = base->data_bytes;
	switch (base->format) {
		case AudioStreamSample::FORMAT_8_BITS: len /= 1; break;
		case AudioStreamSample::FORMAT_16_BITS: len /= 2; break;
		case AudioStreamSample::FORMAT_IMA_ADPCM: len *= 2; break;
	}

	if (base->stereo) {
		len /= 2;
	}

	/* some 64-bit fixed point precaches */

	int64_t loop_begin_fp = ((int64_t)len << MIX_FRAC_BITS);
	int64_t loop_end_fp = ((int64_t)base->loop_end << MIX_FRAC_BITS);
	int64_t length_fp = ((int64_t)len << MIX_FRAC_BITS);
	int64_t begin_limit = (base->loop_mode != AudioStreamSample::LOOP_DISABLED) ? loop_begin_fp : 0;
	int64_t end_limit = (base->loop_mode != AudioStreamSample::LOOP_DISABLED) ? loop_end_fp : length_fp;
	bool is_stereo = base->stereo;

	int32_t todo = p_frames;

	float base_rate = AudioServer::get_singleton()->get_mix_rate();
	float srate = base->mix_rate;
	srate *= p_rate_scale;
	float fincrement = srate / base_rate;
	int32_t increment = int32_t(fincrement * MIX_FRAC_LEN);
	increment *= sign;

	//looping

	AudioStreamSample::LoopMode loop_format = base->loop_mode;
	AudioStreamSample::Format format = base->format;

	/* audio data */

	uint8_t *dataptr = (uint8_t *)base->data;
	const void *data = dataptr + AudioStreamSample::DATA_PAD;
	AudioFrame *dst_buff = p_buffer;

	if (format == AudioStreamSample::FORMAT_IMA_ADPCM) {

		if (loop_format != AudioStreamSample::LOOP_DISABLED) {
			ima_adpcm[0].loop_pos = loop_begin_fp >> MIX_FRAC_BITS;
			ima_adpcm[1].loop_pos = loop_begin_fp >> MIX_FRAC_BITS;
			loop_format = AudioStreamSample::LOOP_FORWARD;
		}
	}

	while (todo > 0) {

		int64_t limit = 0;
		int32_t target = 0, aux = 0;

		/** LOOP CHECKING **/

		if (increment < 0) {
			/* going backwards */

			if (loop_format != AudioStreamSample::LOOP_DISABLED && offset < loop_begin_fp) {
				/* loopstart reached */
				if (loop_format == AudioStreamSample::LOOP_PING_PONG) {
					/* bounce ping pong */
					offset = loop_begin_fp + (loop_begin_fp - offset);
					increment = -increment;
					sign *= -1;
				} else {
					/* go to loop-end */
					offset = loop_end_fp - (loop_begin_fp - offset);
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
			if (loop_format != AudioStreamSample::LOOP_DISABLED && offset >= loop_end_fp) {
				/* loopend reached */

				if (loop_format == AudioStreamSample::LOOP_PING_PONG) {
					/* bounce ping pong */
					offset = loop_end_fp - (offset - loop_end_fp);
					increment = -increment;
					sign *= -1;
				} else {
					/* go to loop-begin */

					if (format == AudioStreamSample::FORMAT_IMA_ADPCM) {
						for (int i = 0; i < 2; i++) {
							ima_adpcm[i].step_index = ima_adpcm[i].loop_step_index;
							ima_adpcm[i].predictor = ima_adpcm[i].loop_predictor;
							ima_adpcm[i].last_nibble = loop_begin_fp >> MIX_FRAC_BITS;
						}
						offset = loop_begin_fp;
					} else {
						offset = loop_begin_fp + (offset - loop_end_fp);
					}
				}
			} else {
				/* no loop, check for end of sample */
				if (offset >= length_fp) {

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

		switch (base->format) {
			case AudioStreamSample::FORMAT_8_BITS: {

				if (is_stereo)
					do_resample<int8_t, true, false>((int8_t *)data, dst_buff, offset, increment, target, ima_adpcm);
				else
					do_resample<int8_t, false, false>((int8_t *)data, dst_buff, offset, increment, target, ima_adpcm);
			} break;
			case AudioStreamSample::FORMAT_16_BITS: {
				if (is_stereo)
					do_resample<int16_t, true, false>((int16_t *)data, dst_buff, offset, increment, target, ima_adpcm);
				else
					do_resample<int16_t, false, false>((int16_t *)data, dst_buff, offset, increment, target, ima_adpcm);

			} break;
			case AudioStreamSample::FORMAT_IMA_ADPCM: {
				if (is_stereo)
					do_resample<int8_t, true, true>((int8_t *)data, dst_buff, offset, increment, target, ima_adpcm);
				else
					do_resample<int8_t, false, true>((int8_t *)data, dst_buff, offset, increment, target, ima_adpcm);

			} break;
		}

		dst_buff += target;
	}
}

float AudioStreamPlaybackSample::get_length() const {

	int len = base->data_bytes;
	switch (base->format) {
		case AudioStreamSample::FORMAT_8_BITS: len /= 1; break;
		case AudioStreamSample::FORMAT_16_BITS: len /= 2; break;
		case AudioStreamSample::FORMAT_IMA_ADPCM: len *= 2; break;
	}

	if (base->stereo) {
		len /= 2;
	}

	return float(len) / base->mix_rate;
}

AudioStreamPlaybackSample::AudioStreamPlaybackSample() {

	active = false;
	offset = 0;
	sign = 1;
}

/////////////////////

void AudioStreamSample::set_format(Format p_format) {

	format = p_format;
}

AudioStreamSample::Format AudioStreamSample::get_format() const {

	return format;
}

void AudioStreamSample::set_loop_mode(LoopMode p_loop_mode) {

	loop_mode = p_loop_mode;
}
AudioStreamSample::LoopMode AudioStreamSample::get_loop_mode() const {

	return loop_mode;
}

void AudioStreamSample::set_loop_begin(int p_frame) {

	loop_begin = p_frame;
}
int AudioStreamSample::get_loop_begin() const {

	return loop_begin;
}

void AudioStreamSample::set_loop_end(int p_frame) {

	loop_end = p_frame;
}
int AudioStreamSample::get_loop_end() const {

	return loop_end;
}

void AudioStreamSample::set_mix_rate(int p_hz) {

	mix_rate = p_hz;
}
int AudioStreamSample::get_mix_rate() const {

	return mix_rate;
}
void AudioStreamSample::set_stereo(bool p_enable) {

	stereo = p_enable;
}
bool AudioStreamSample::is_stereo() const {

	return stereo;
}

void AudioStreamSample::set_data(const PoolVector<uint8_t> &p_data) {

	AudioServer::get_singleton()->lock();
	if (data) {
		AudioServer::get_singleton()->audio_data_free(data);
		data = NULL;
		data_bytes = 0;
	}

	int datalen = p_data.size();
	if (datalen) {

		PoolVector<uint8_t>::Read r = p_data.read();
		int alloc_len = datalen + DATA_PAD * 2;
		data = AudioServer::get_singleton()->audio_data_alloc(alloc_len); //alloc with some padding for interpolation
		zeromem(data, alloc_len);
		uint8_t *dataptr = (uint8_t *)data;
		copymem(dataptr + DATA_PAD, r.ptr(), datalen);
		data_bytes = datalen;
	}

	AudioServer::get_singleton()->unlock();
}
PoolVector<uint8_t> AudioStreamSample::get_data() const {

	PoolVector<uint8_t> pv;

	if (data) {
		pv.resize(data_bytes);
		{

			PoolVector<uint8_t>::Write w = pv.write();
			copymem(w.ptr(), data, data_bytes);
		}
	}

	return pv;
}

Ref<AudioStreamPlayback> AudioStreamSample::instance_playback() {

	Ref<AudioStreamPlaybackSample> sample;
	sample.instance();
	sample->base = Ref<AudioStreamSample>(this);
	return sample;
}

String AudioStreamSample::get_stream_name() const {

	return "";
}

void AudioStreamSample::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_format", "format"), &AudioStreamSample::set_format);
	ClassDB::bind_method(D_METHOD("get_format"), &AudioStreamSample::get_format);

	ClassDB::bind_method(D_METHOD("set_loop_mode", "loop_mode"), &AudioStreamSample::set_loop_mode);
	ClassDB::bind_method(D_METHOD("get_loop_mode"), &AudioStreamSample::get_loop_mode);

	ClassDB::bind_method(D_METHOD("set_loop_begin", "loop_begin"), &AudioStreamSample::set_loop_begin);
	ClassDB::bind_method(D_METHOD("get_loop_begin"), &AudioStreamSample::get_loop_begin);

	ClassDB::bind_method(D_METHOD("set_loop_end", "loop_end"), &AudioStreamSample::set_loop_end);
	ClassDB::bind_method(D_METHOD("get_loop_end"), &AudioStreamSample::get_loop_end);

	ClassDB::bind_method(D_METHOD("set_mix_rate", "mix_rate"), &AudioStreamSample::set_mix_rate);
	ClassDB::bind_method(D_METHOD("get_mix_rate"), &AudioStreamSample::get_mix_rate);

	ClassDB::bind_method(D_METHOD("set_stereo", "stereo"), &AudioStreamSample::set_stereo);
	ClassDB::bind_method(D_METHOD("is_stereo"), &AudioStreamSample::is_stereo);

	ClassDB::bind_method(D_METHOD("set_data", "data"), &AudioStreamSample::set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &AudioStreamSample::get_data);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "format", PROPERTY_HINT_ENUM, "8-Bit,16-Bit,IMA-ADPCM"), "set_format", "get_format");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_mode", PROPERTY_HINT_ENUM, "Disabled,Forward,Ping-Pong"), "set_loop_mode", "get_loop_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_begin"), "set_loop_begin", "get_loop_begin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_end"), "set_loop_end", "get_loop_end");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mix_rate"), "set_mix_rate", "get_mix_rate");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stereo"), "set_stereo", "is_stereo");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_data", "get_data");
}

AudioStreamSample::AudioStreamSample() {
	format = FORMAT_8_BITS;
	loop_mode = LOOP_DISABLED;
	stereo = false;
	loop_begin = 0;
	loop_end = 0;
	mix_rate = 44100;
	data = NULL;
	data_bytes = 0;
}
AudioStreamSample::~AudioStreamSample() {

	if (data) {
		AudioServer::get_singleton()->audio_data_free(data);
		data = NULL;
		data_bytes = 0;
	}
}
