/*************************************************************************/
/*  cp_loader_mod.h                                                      */
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
#ifndef CP_LOADER_MOD_H
#define CP_LOADER_MOD_H
#include "cp_loader.h"
/**
	@author Juan Linietsky <reduz@gmail.com>
*/
class CPLoader_MOD : public CPLoader {

	CPFileAccessWrapper *file;
public:

	bool can_load_song() { return true; }
	bool can_load_sample() { return false; }
	bool can_load_instrument() { return false; }
	
	Error load_song(const char *p_file,CPSong *p_song,bool p_sampleset);
	Error load_sample(const char *p_file,CPSample *p_sample) { return FILE_UNRECOGNIZED; }
	Error load_instrument(const char *p_file,CPSong *p_song,int p_instr_idx) { return FILE_UNRECOGNIZED; }
	
	CPLoader_MOD(CPFileAccessWrapper *p_file);
	~CPLoader_MOD();
};

#endif
