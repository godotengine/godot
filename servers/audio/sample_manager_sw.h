/*************************************************************************/
/*  sample_manager_sw.h                                                  */
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
#ifndef SAMPLE_MANAGER_SW_H
#define SAMPLE_MANAGER_SW_H

#include "servers/audio_server.h"

class SampleManagerSW {
public:
	/* SAMPLE API */

	virtual RID sample_create(AS::SampleFormat p_format, bool p_stereo, int p_length) = 0;

	virtual void sample_set_description(RID p_sample, const String &p_description) = 0;
	virtual String sample_get_description(RID p_sample) const = 0;

	virtual AS::SampleFormat sample_get_format(RID p_sample) const = 0;
	virtual bool sample_is_stereo(RID p_sample) const = 0;
	virtual int sample_get_length(RID p_sample) const = 0;

	virtual void sample_set_data(RID p_sample, const DVector<uint8_t> &p_buffer) = 0;
	virtual const DVector<uint8_t> sample_get_data(RID p_sample) const = 0;

	virtual void *sample_get_data_ptr(RID p_sample) const = 0;

	virtual void sample_set_mix_rate(RID p_sample, int p_rate) = 0;
	virtual int sample_get_mix_rate(RID p_sample) const = 0;

	virtual void sample_set_loop_format(RID p_sample, AS::SampleLoopFormat p_format) = 0;
	virtual AS::SampleLoopFormat sample_get_loop_format(RID p_sample) const = 0;

	virtual void sample_set_loop_begin(RID p_sample, int p_pos) = 0;
	virtual int sample_get_loop_begin(RID p_sample) const = 0;

	virtual void sample_set_loop_end(RID p_sample, int p_pos) = 0;
	virtual int sample_get_loop_end(RID p_sample) const = 0;

	virtual bool is_sample(RID) const = 0;
	virtual void free(RID p_sample) = 0;

	virtual ~SampleManagerSW();
};

class SampleManagerMallocSW : public SampleManagerSW {

	struct Sample {

		void *data;
		int length;
		int length_bytes;
		AS::SampleFormat format;
		bool stereo;
		AS::SampleLoopFormat loop_format;
		int loop_begin;
		int loop_end;
		int mix_rate;
		String description;
	};

	mutable RID_Owner<Sample> sample_owner;

public:
	/* SAMPLE API */

	virtual RID sample_create(AS::SampleFormat p_format, bool p_stereo, int p_length);

	virtual void sample_set_description(RID p_sample, const String &p_description);
	virtual String sample_get_description(RID p_sample) const;

	virtual AS::SampleFormat sample_get_format(RID p_sample) const;
	virtual bool sample_is_stereo(RID p_sample) const;
	virtual int sample_get_length(RID p_sample) const;

	virtual void sample_set_data(RID p_sample, const DVector<uint8_t> &p_buffer);
	virtual const DVector<uint8_t> sample_get_data(RID p_sample) const;

	virtual void *sample_get_data_ptr(RID p_sample) const;

	virtual void sample_set_mix_rate(RID p_sample, int p_rate);
	virtual int sample_get_mix_rate(RID p_sample) const;

	virtual void sample_set_loop_format(RID p_sample, AS::SampleLoopFormat p_format);
	virtual AS::SampleLoopFormat sample_get_loop_format(RID p_sample) const;

	virtual void sample_set_loop_begin(RID p_sample, int p_pos);
	virtual int sample_get_loop_begin(RID p_sample) const;

	virtual void sample_set_loop_end(RID p_sample, int p_pos);
	virtual int sample_get_loop_end(RID p_sample) const;

	virtual bool is_sample(RID) const;
	virtual void free(RID p_sample);

	SampleManagerMallocSW();
	virtual ~SampleManagerMallocSW();
};

#endif // SAMPLE_MANAGER_SW_H
