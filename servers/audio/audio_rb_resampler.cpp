/*************************************************************************/
/*  audio_rb_resampler.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "audio_rb_resampler.h"
#include "servers/audio_server.h"
#include "core/math/math_funcs.h"
#include "os/os.h"

int AudioRBResampler::get_channel_count() const {

	if (!rb)
		return 0;

	return channels;
}


AudioFrame InterpolateHermite4pt3oX(AudioFrame x0, AudioFrame x1, AudioFrame x2, AudioFrame x3, float t)
{
	AudioFrame c0 = x1;
	AudioFrame c1 = (x2 - x0) * .5F;
	AudioFrame c2 = x0 - ( x1* 2.5F) + (x2 * 2) - (x3 * .5F);
	AudioFrame c3 = ((x3 - x0) * .5F) + ((x1 - x2) * 1.5F);
	return (((((c3 * t) + c2) * t) + c1) * t) + c0;
}

float InterpolateHermite4pt3oX(float x0, float x1, float x2, float x3, float t)
{
	float c0 = x1;
	float c1 = .5F * (x2 - x0);
	float c2 = x0 - (2.5F * x1) + (2 * x2) - (.5F * x3);
	float c3 = (.5F * (x3 - x0)) + (1.5F * (x1 - x2));
	return (((((c3 * t) + c2) * t) + c1) * t) + c0;
}

uint32_t AudioRBResampler::copy_stereo(AudioFrame *dest, int count)
{
	int32_t read = 0;
	int32_t pos = rb_read_pos;
	while(read < count)
	{
		dest[read] = AudioFrame(rb[pos],rb[(pos+1) & rb_mask]);
		pos = (pos + 2) & rb_mask;
		++read;

	}
}


template <int C>
uint32_t AudioRBResampler::_resample(AudioFrame *p_dest, int p_todo) {

	uint32_t read = offset & MIX_FRAC_MASK;

	float ratio = float(target_mix_rate) / float(src_mix_rate);
	int frames_to_skip = int(ratio);
	int steps = int(1.0f/ratio);
	int step = 0;

	float mu = ratio;
	float mu_increment = ratio;

	uint32_t pos = rb_read_pos;

	OS::get_singleton()->print("Resample p_todo: %d, ratio: %f, target_rate: %d, src_rate: %d\n", p_todo, ratio, target_mix_rate, src_mix_rate);

	uint32_t a_index, b_index, c_index, d_index;

	for (int i = 0; i < p_todo; i++) {

		mu = ratio + mu_increment*step;
		
		if (mu>=1)
		{
			pos += frames_to_skip + 1;
			mu = ratio;
			step = 0;
		}

		++step;

		// since this is a template with a known compile time value (C), conditionals go away when compiling.
		if (C == 1) {

			if (pos == 0)
				a_index = 0;
			else
				a_index = (pos - C ) & rb_mask;
			
			if ( Math::abs(rb_read_pos-rb_write_pos) > (2*C))
			{
				c_index = (pos + C) & rb_mask;
				d_index = (pos + (2*C)) & rb_mask;
			}
			else
			{
				c_index = pos;
				d_index = pos;
			}

			float a = rb[a_index];
			float b = rb[b_index];
			float c = rb[c_index];
			float d = rb[d_index];

			p_dest[i] =  AudioFrame(0,0)+InterpolateHermite4pt3oX(a,b,c,d, mu);

		}
		if (C == 2) {

			b_index = pos;

			if (pos == 0)
				a_index = 0;
			else
				a_index = (pos - C ) & rb_mask;
			
			if ( Math::abs(rb_read_pos-rb_write_pos) > (2*C))
			{
				c_index = (pos + C) & rb_mask;
				d_index = (pos + (2*C)) & rb_mask;
			}
			else
			{
				c_index = pos;
				d_index = pos;
			}
			// OS::get_singleton()->print("Resample a_index: %d b_index %d c_index %d d_index %d\n", a_index, b_index, c_index, d_index);
			AudioFrame a = AudioFrame(rb[a_index], rb[a_index+1]);
			AudioFrame b = AudioFrame(rb[b_index], rb[b_index+1]);
			AudioFrame c = AudioFrame(rb[c_index], rb[c_index+1]);
			AudioFrame d = AudioFrame(rb[d_index], rb[d_index+1]);

			p_dest[i*C] = InterpolateHermite4pt3oX(a,b,c,d, mu);

		}

		if (C == 4) {

			if (pos == 0)
				a_index = 0;
			else
				a_index = (pos - C ) & rb_mask;
			
			if ( Math::abs(rb_read_pos-rb_write_pos) > (2*C))
			{
				c_index = (pos + C) & rb_mask;
				d_index = (pos + (2*C)) & rb_mask;
			}
			else
			{
				c_index = pos;
				d_index = pos;
			}

			AudioFrame a = AudioFrame(rb[a_index+0], rb[a_index+1]);
			AudioFrame b = AudioFrame(rb[b_index+0], rb[b_index+1]);
			AudioFrame c = AudioFrame(rb[c_index+0], rb[c_index+1]);
			AudioFrame d = AudioFrame(rb[d_index+0], rb[d_index+1]);

			p_dest[(i*C)+0] = InterpolateHermite4pt3oX(a,b,c,d, mu);

			a = AudioFrame(rb[a_index+2], rb[a_index+3]);
			b = AudioFrame(rb[b_index+2], rb[b_index+3]);
			c = AudioFrame(rb[c_index+2], rb[c_index+3]);
			d = AudioFrame(rb[d_index+2], rb[d_index+3]);

			p_dest[(i*C)+1] = InterpolateHermite4pt3oX(a,b,c,d, mu);

		}

		if (C == 6) {

			if (pos == 0)
				a_index = 0;
			else
				a_index = (pos - C ) & rb_mask;
			
			if ( Math::abs(rb_read_pos-rb_write_pos) > (2*C))
			{
				c_index = (pos + C) & rb_mask;
				d_index = (pos + (2*C)) & rb_mask;
			}
			else
			{
				c_index = pos;
				d_index = pos;
			}

			AudioFrame a = AudioFrame(rb[a_index+0], rb[a_index+1]);
			AudioFrame b = AudioFrame(rb[b_index+0], rb[b_index+1]);
			AudioFrame c = AudioFrame(rb[c_index+0], rb[c_index+1]);
			AudioFrame d = AudioFrame(rb[d_index+0], rb[d_index+1]);

			p_dest[(i*C)+0] = InterpolateHermite4pt3oX(a,b,c,d, mu);

			a = AudioFrame(rb[a_index+2], rb[a_index+3]);
			b = AudioFrame(rb[b_index+2], rb[b_index+3]);
			c = AudioFrame(rb[c_index+2], rb[c_index+3]);
			d = AudioFrame(rb[d_index+2], rb[d_index+3]);

			p_dest[(i*C)+1] = InterpolateHermite4pt3oX(a,b,c,d, mu);

			a = AudioFrame(rb[a_index+4], rb[a_index+5]);
			b = AudioFrame(rb[b_index+4], rb[b_index+5]);
			c = AudioFrame(rb[c_index+4], rb[c_index+5]);
			d = AudioFrame(rb[d_index+4], rb[d_index+5]);

			p_dest[(i*C)+2] = InterpolateHermite4pt3oX(a,b,c,d, mu);

			
		}
		read+=C;
	}

	OS::get_singleton()->print("Resample fillled the requsted frames");

	return read ; //rb_read_pos=offset>>MIX_FRAC_BITS;
}

bool AudioRBResampler::mix(AudioFrame *p_dest, int p_frames) {

	if (!rb)
		return false;

	int write_pos_cache = rb_write_pos;

	int32_t increment = (src_mix_rate * MIX_FRAC_LEN) / target_mix_rate;

	int rb_todo;

	if (write_pos_cache == rb_read_pos) {
		return false; //out of buffer

	} else if (rb_read_pos < write_pos_cache) {

		rb_todo = write_pos_cache - rb_read_pos; //-1?
	} else {

		rb_todo = (rb_len - rb_read_pos) + write_pos_cache; //-1?
	}

	int todo = MIN(((int64_t(rb_todo) << MIX_FRAC_BITS) / increment) + 1, p_frames);
	{
		int read = 0;
		// switch (channels) {
		// 	case 1: read = _resample<1>(p_dest, todo); break;
		// 	case 2: read = _resample<2>(p_dest, todo); break;
		// 	case 4: read = _resample<4>(p_dest, todo); break;
		// 	case 6: read = _resample<6>(p_dest, todo); break;
		// }
		read = copy_stereo(p_dest, todo);

		OS::get_singleton()->print("todo: %d rb_todo: %d increment: %d p_frames: %d read: %d channels: %d\n", todo, rb_todo, increment, p_frames, read, channels);

		//end of stream, fadeout
		int remaining = p_frames - todo;
		if (remaining && todo > 0) {

			//print_line("fadeout");
			for (uint32_t c = 0; c < channels; c++) {

				for (int i = 0; i < todo; i++) {

					AudioFrame samp = p_dest[i * channels + c];
					float mul = float(i)/float(todo);
					
					p_dest[i * channels + c] = samp * mul;
				}
			}
		}

		//zero out what remains there to avoid glitches
		for (uint32_t i = todo * channels; i < int(p_frames) * channels; i++) {

			p_dest[i] = AudioFrame(0,0);
		}

		if (read > rb_todo)
			read = rb_todo;

		rb_read_pos = (rb_read_pos + read) & rb_mask;
	}

	return true;
}

Error AudioRBResampler::setup(int p_channels, int p_src_mix_rate, int p_target_mix_rate, int p_buffer_msec, int p_minbuff_needed) {

	ERR_FAIL_COND_V(p_channels != 1 && p_channels != 2 && p_channels != 4 && p_channels != 6, ERR_INVALID_PARAMETER);

	//float buffering_sec = int(GLOBAL_DEF("audio/stream_buffering_ms",500))/1000.0;
	int desired_rb_bits = nearest_shift(MAX((p_buffer_msec / 1000.0) * p_src_mix_rate, p_minbuff_needed));

	bool recreate = !rb;

	if (rb && (uint32_t(desired_rb_bits) != rb_bits || channels != uint32_t(p_channels))) {
		//recreate

		memdelete_arr(rb);
		memdelete_arr(read_buf);
		recreate = true;
	}

	if (recreate) {

		channels = p_channels;
		rb_bits = desired_rb_bits;
		rb_len = (1 << rb_bits);
		rb_mask = rb_len - 1;
		rb = memnew_arr(float, rb_len * p_channels);
		read_buf = memnew_arr(float, rb_len * p_channels);
	}

	src_mix_rate = p_src_mix_rate;
	target_mix_rate = p_target_mix_rate;
	offset = 0;
	rb_read_pos = 0;
	rb_write_pos = 0;

	//avoid maybe strange noises upon load
	for (unsigned int i = 0; i < (rb_len * channels); i++) {

		rb[i] = 0;
		read_buf[i] = 0;
	}


		OS::get_singleton()->print("RB SETUP: msec: %d, array-length-frames: %d, array-length-floats: %d\n", p_buffer_msec, rb_len, rb_len * p_channels);


	return OK;
}

void AudioRBResampler::clear() {

	if (!rb)
		return;

	//should be stopped at this point but just in case
	if (rb) {
		memdelete_arr(rb);
		memdelete_arr(read_buf);
	}
	rb = NULL;
	offset = 0;
	rb_read_pos = 0;
	rb_write_pos = 0;
	read_buf = NULL;
}

AudioRBResampler::AudioRBResampler() {

	rb = NULL;
	offset = 0;
	read_buf = NULL;
	rb_read_pos = 0;
	rb_write_pos = 0;

	rb_bits = 0;
	rb_len = 0;
	rb_mask = 0;
	read_buff_len = 0;
	channels = 0;
	src_mix_rate = 0;
	target_mix_rate = 0;
}

AudioRBResampler::~AudioRBResampler() {

	if (rb) {
		memdelete_arr(rb);
		memdelete_arr(read_buf);
	}
}
