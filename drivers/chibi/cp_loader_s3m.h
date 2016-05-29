/*************************************************************************/
/*  cp_loader_s3m.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#ifndef CP_LOADER_S3M_H
#define CP_LOADER_S3M_H

#include "cp_loader.h"

/**
  *@author Juan Linietsky
  */
/******************************
 loader_s3m.h
 ----------
Scream Tracker Module CPLoader!
It lacks support for
individual sample loading
and reorganizing the columns.
********************************/




class CPLoader_S3M : public CPLoader  {

	struct S3M_Header {
	        char  songname[28];
	        uint8_t t1a;
	        uint8_t type;
	        uint8_t unused1[2];
	        uint16_t ordnum;
	        uint16_t insnum;
	        uint16_t patnum;
	        uint16_t flags;
	        uint16_t tracker;
	        uint16_t fileformat;
	        char  scrm[5];
	        uint8_t mastervol;
	        uint8_t initspeed;
	        uint8_t inittempo;
	        uint8_t mastermult;
	        uint8_t ultraclick;
	        uint8_t pantable;
	        uint8_t unused2[8];
	        uint16_t special;
	        uint8_t channels[32];
		uint8_t pannings[32];
		uint8_t orderlist[300];
	};

	
	int sample_parapointers[CPSong::MAX_SAMPLES];
	int pattern_parapointers[CPSong::MAX_PATTERNS];
	
	Error load_header();
	void set_header();
	Error load_sample(CPSample *p_sample);
	Error load_pattern(CPPattern *p_pattern);
	Error load_patterns();

	Error load_samples();
	
	S3M_Header header;
        int sample_count;
	int pattern_count;
	
	CPFileAccessWrapper *file;
	CPSong *song;
public:

	bool can_load_song() { return true; }
	bool can_load_sample() { return false; }
	bool can_load_instrument() { return false; }
	
	Error load_song(const char *p_file,CPSong *p_song,bool p_sampleset);
	Error load_sample(const char *p_file,CPSample *p_sample);
	Error load_instrument(const char *p_file,CPSong *p_song,int p_instr_idx);
	
	CPLoader_S3M(CPFileAccessWrapper *p_file);
	~CPLoader_S3M();
};



#endif
