/*************************************************************************/
/*  cp_sample.h                                                          */
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
#ifndef CPSAMPLE_H
#define CPSAMPLE_H


#include "cp_config.h"
#include "cp_sample_manager.h"
class CPSample {

public:
	enum VibratoType {
		VIBRATO_SINE,
		VIBRATO_SAW,
		VIBRATO_SQUARE,
		VIBRATO_RANDOM

	};

private:
	
	enum { NAME_MAX_LEN=26 };
	
	char name[NAME_MAX_LEN];

	uint8_t default_volume; /* 0.. 64 */
	uint8_t global_volume; /* 0.. 64 */

	bool pan_enabled;
	uint8_t pan;  /* 0.. 64 */

	VibratoType vibrato_type;
	uint8_t vibrato_speed; /* 0.. 64 */
	uint8_t vibrato_depth; /* 0.. 64 */
	uint8_t vibrato_rate; /* 0.. 64 */

	CPSample_ID id;
	
	void copy_from(const CPSample &p_sample);
public:

	
	void operator=(const CPSample &p_sample);
	
	const char * get_name() const;
	void set_name(const char *p_name);

	void set_default_volume(uint8_t p_vol);
	uint8_t get_default_volume() const;
	
	void set_global_volume(uint8_t p_vol);
	uint8_t get_global_volume() const;
	
	void set_pan_enabled(bool p_vol);
	bool is_pan_enabled() const;
	
	void set_pan(uint8_t p_pan);
	uint8_t get_pan() const;

	void set_vibrato_type(VibratoType p_vibrato_type);
	VibratoType get_vibrato_type() const;

	void set_vibrato_speed(uint8_t p_vibrato_speed) ;
	uint8_t get_vibrato_speed() const;

	void set_vibrato_depth(uint8_t p_vibrato_depth);
	uint8_t get_vibrato_depth() const;

	void set_vibrato_rate(uint8_t p_vibrato_rate);
	uint8_t get_vibrato_rate() const;

	void set_sample_data(CPSample_ID);
	CPSample_ID get_sample_data() const;
	
	void reset();
	
	CPSample(const CPSample&p_from);
	CPSample();
	~CPSample();
			
};




#endif
