/*************************************************************************/
/*  cp_loader_xm.h                                                       */
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

#ifndef CP_LOADER_XM_H
#define CP_LOADER_XM_H

#include "cp_loader.h"

/**
  *@author red
  */

class CPLoader_XM : public CPLoader {

	struct XM_Header {

		uint8_t idtext[18];
		uint8_t songname[21];
		uint8_t hex1a; // ?
		uint8_t trackername[21];
		uint16_t version;
		uint32_t headersize; //from here

		uint16_t songlength; //pattern ordertable
		uint16_t restart_pos;
		uint16_t channels_used;
		uint16_t patterns_used;
		uint16_t instruments_used;
		uint16_t use_linear_freq;
		uint16_t tempo;
		uint16_t speed;
		uint8_t orderlist[256];

	} header;

	CPFileAccessWrapper *file;

	Error load_instrument_internal(CPInstrument *pint, bool p_xi, int p_cpos, int p_hsize, int p_sampnumb = -1);
	CPSong *song;

public:
	bool can_load_song() { return true; }
	bool can_load_sample() { return false; }
	bool can_load_instrument() { return true; }

	Error load_song(const char *p_file, CPSong *p_song, bool p_sampleset);
	Error load_sample(const char *p_file, CPSample *p_sample);
	Error load_instrument(const char *p_file, CPSong *p_song, int p_instr_idx);

	CPLoader_XM(CPFileAccessWrapper *p_file);
	~CPLoader_XM();
};

#endif
