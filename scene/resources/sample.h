/*************************************************************************/
/*  sample.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef SAMPLE_H
#define SAMPLE_H

#include "servers/audio_server.h"
#include "resource.h"

class Sample : public Resource {

	GDCLASS(Sample, Resource );
	RES_BASE_EXTENSION("smp");
public:

	enum Format {

		FORMAT_PCM8,
		FORMAT_PCM16,
		FORMAT_IMA_ADPCM
	};

	enum LoopFormat {
		LOOP_NONE,
		LOOP_FORWARD,
		LOOP_PING_PONG // not supported in every platform

	};

private:

	Format format;
	int length;
	bool stereo;

	LoopFormat loop_format;
	int loop_begin;
	int loop_end;
	int mix_rate;

	RID sample;


	void _set_data(const Dictionary& p_data);
	Dictionary _get_data() const;

protected:

	static void _bind_methods();

public:


	void create(Format p_format, bool p_stereo, int p_length);

	Format get_format() const;
	bool is_stereo() const;
	int get_length() const;

	void set_data(const PoolVector<uint8_t>& p_buffer);
	PoolVector<uint8_t> get_data() const;

	void set_mix_rate(int p_rate);
	int get_mix_rate() const;

	void set_loop_format(LoopFormat p_format);
	LoopFormat get_loop_format() const;

	void set_loop_begin(int p_pos);
	int get_loop_begin() const;

	void set_loop_end(int p_pos);
	int get_loop_end() const;

	virtual RID get_rid() const;
	Sample();
	~Sample();
};

VARIANT_ENUM_CAST( Sample::Format );
VARIANT_ENUM_CAST( Sample::LoopFormat );

#endif // SAMPLE_H
