/*************************************************************************/
/*  cp_sample_manager.h                                                  */
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
#ifndef CP_SAMPLE_MANAGER_H
#define CP_SAMPLE_MANAGER_H

#include "cp_config.h"
#include "cp_sample_defs.h"

/**
@author Juan Linietsky
*/

/* abstract base CPSample_ID class */

struct CPSample_ID {

	void *_private;

	bool operator==(const CPSample_ID &p_other) const { return _private == p_other._private; }
	bool operator!=(const CPSample_ID &p_other) const { return _private != p_other._private; }
	bool is_null() const { return _private == 0; }
	CPSample_ID(void *p_private = 0) { _private = p_private; };
};

class CPSampleManager {

	static CPSampleManager *singleton;

public:
	/* get the singleton instance */
	static CPSampleManager *get_singleton();

	virtual void copy_to(CPSample_ID p_from, CPSample_ID &p_to); ///< if p_to is null, it gets created

	virtual CPSample_ID create(bool p_16bits, bool p_stereo, int32_t p_len) = 0;
	virtual void recreate(CPSample_ID p_id, bool p_16bits, bool p_stereo, int32_t p_len) = 0;
	virtual void destroy(CPSample_ID p_id) = 0;
	virtual bool check(CPSample_ID p_id) = 0; // return false if invalid

	virtual void set_c5_freq(CPSample_ID p_id, int32_t p_freq) = 0;
	virtual void set_loop_begin(CPSample_ID p_id, int32_t p_begin) = 0;
	virtual void set_loop_end(CPSample_ID p_id, int32_t p_end) = 0;
	virtual void set_loop_type(CPSample_ID p_id, CPSample_Loop_Type p_type) = 0;
	virtual void set_chunk(CPSample_ID p_id, int32_t p_index, void *p_data, int p_data_len) = 0;

	virtual int32_t get_loop_begin(CPSample_ID p_id) = 0;
	virtual int32_t get_loop_end(CPSample_ID p_id) = 0;
	virtual CPSample_Loop_Type get_loop_type(CPSample_ID p_id) = 0;
	virtual int32_t get_c5_freq(CPSample_ID p_id) = 0;
	virtual int32_t get_size(CPSample_ID p_id) = 0;
	virtual bool is_16bits(CPSample_ID p_id) = 0;
	virtual bool is_stereo(CPSample_ID p_id) = 0;
	virtual bool lock_data(CPSample_ID p_id) = 0;
	virtual void *get_data(CPSample_ID p_id) = 0; /* WARNING: Not all sample managers
may be able to implement this, it depends on the mixer in use! */
	virtual int16_t get_data(CPSample_ID p_id, int p_sample, int p_channel = 0) = 0; /// Does not need locking
	virtual void set_data(CPSample_ID p_id, int p_sample, int16_t p_data, int p_channel = 0) = 0; /// Does not need locking
	virtual void unlock_data(CPSample_ID p_id) = 0;

	virtual void get_chunk(CPSample_ID p_id, int32_t p_index, void *p_data, int p_data_len) = 0;

	CPSampleManager();
	virtual ~CPSampleManager() {}
};

#endif
