/*************************************************************************/
/*  cp_loader_it.h                                                       */
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

#ifndef CP_LOADER_IT_H
#define CP_LOADER_IT_H

#include "cp_loader.h"
/**
  *@author Juan Linietsky
  */

/******************************
 loader_it.h
 ----------
Impulse Tracker Module CPLoader!
It lacks support for old
instrument files methinks...
and some other things like
midi.
********************************/

class AuxSampleData; //used for internal crap

class CPLoader_IT : public CPLoader {

	CPFileAccessWrapper *file;
	CPSong *song;

	struct IT_Header {
		uint8_t blank01[2];
		uint16_t ordnum;
		uint16_t insnum;
		uint16_t smpnum;
		uint16_t patnum;
		uint16_t cwt; /* Created with tracker (y.xx = 0x0yxx) */
		uint16_t cmwt; /* Compatible with tracker ver > than val. */
		uint16_t flags;
		uint16_t special; /* bit 0 set = song message attached */
		uint16_t msglength;
		uint32_t msgoffset;
		bool is_chibi;
	};

	/* Variables to store temp data */
	IT_Header header;

	/* CPSong Info Methods */
	Error load_header(bool p_dont_set);
	Error load_orders();
	Error load_message();

	/* CPPattern Methods */
	Error load_patterns();

	/* CPSample Methods */

	Error load_samples();
	Error load_sample(CPSample *p_sample);
	CPSample_ID load_sample_data(AuxSampleData &p_sample_data);

	// CPSample decompression

	uint32_t read_n_bits_from_IT_compressed_block(uint8_t p_bits_to_read);
	bool read_IT_compressed_block(bool p_16bits);
	void free_IT_compressed_block();
	bool load_sample_8bits_IT_compressed(void *p_dest_buffer, int p_buffsize);
	bool load_sample_16bits_IT_compressed(void *p_dest_buffer, int p_buffsize);
	uint32_t *source_buffer; /* source buffer */
	uint32_t *source_position; /* actual reading position */
	uint8_t source_remaining_bits; /* bits remaining in read dword */
	uint8_t *pat_data;

	/* CPInstruments Methods */
	Error load_effects();
	Error load_instruments();
	Error load_instrument(CPInstrument *p_instrument, int *p_samples = 0);
	void load_envelope(CPEnvelope *p_envelope, bool *p_has_filter_flag = 0);

public:
	bool can_load_song();
	bool can_load_sample();
	bool can_load_instrument();

	Error load_song(const char *p_file, CPSong *p_song, bool p_sampleset = false);
	Error load_sample(const char *p_file, CPSample *p_sample);
	Error load_instrument(const char *p_file, CPSong *p_song, int p_instr_idx);

	CPLoader_IT(CPFileAccessWrapper *p_file);
};

#endif
