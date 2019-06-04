/*************************************************************************/
/*  cp_loader.h                                                          */
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
#ifndef CP_LOADER_H
#define CP_LOADER_H

#include "cp_file_access_wrapper.h"
#include "cp_song.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class CPLoader {

public:
	enum Error {
		FILE_OK,
		FILE_UNRECOGNIZED,
		FILE_CANNOT_OPEN,
		FILE_CORRUPTED,
		FILE_OUT_OF_MEMORY,
	};

	virtual bool can_load_song() = 0;
	virtual bool can_load_sample() = 0;
	virtual bool can_load_instrument() = 0;

	virtual Error load_song(const char *p_file, CPSong *p_song, bool p_sampleset) = 0;
	virtual Error load_sample(const char *p_file, CPSample *p_sample) = 0;
	virtual Error load_instrument(const char *p_file, CPSong *p_song, int p_instr_idx) = 0;

	virtual ~CPLoader() {}
};

#endif
