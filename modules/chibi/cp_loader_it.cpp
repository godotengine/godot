/*************************************************************************/
/*  cp_loader_it.cpp                                                     */
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
#include "cp_loader_it.h"

bool CPLoader_IT::can_load_song() { return true; }
bool CPLoader_IT::can_load_sample() { return true; }
bool CPLoader_IT::can_load_instrument() { return true; }

CPLoader::Error CPLoader_IT::load_song(const char *p_file,CPSong *p_song, bool p_sampleset) {
	
	
	song=p_song;

	if (file->open( p_file, CPFileAccessWrapper::READ )!=CPFileAccessWrapper::OK)
		return CPLoader::FILE_CANNOT_OPEN;
	
	
	Error err;
	
	char aux_identifier[4];
	file->get_byte_array((uint8_t*)aux_identifier,4);

	if (	aux_identifier[0]!='I' ||
		aux_identifier[1]!='M' ||
		aux_identifier[2]!='P' ||
		aux_identifier[3]!='M') {


			CP_PRINTERR("IT CPLoader CPSong: Failed Identifier");
			return FILE_UNRECOGNIZED;
	}


	if (p_sampleset) {

		song->reset(false,true,true,false);

		if ((err=load_header(true))) {
			file->close();
			return err;
		}

		if ((err=load_samples())) {
			file->close();
			return err;
		}

		if ((err=load_instruments())) {
			file->close();
			return err;
		}

		return FILE_OK;
	}

	song->reset();

	if ((err=load_header(false))) {
		file->close();
		return err;
	}
	
	if ((err=load_orders())) {
		file->close();
		return err;
	}
	
	if ((err=load_patterns())) {
		file->close();
		return err;
	}
	
	if ((err=load_samples())) {
		file->close();
		return err;
	}
	
	if ((err=load_effects())) {
		file->close();
		return err;
	}

	if ((err=load_instruments())) {
		file->close();
		return err;
	}
	
	if ((err=load_message())) {
		file->close();
		return err;
	}

	file->close();
	return FILE_OK;
	
}




CPLoader::Error CPLoader_IT::load_sample(const char *p_file,CPSample *p_sample) {
	
	if (file->open( p_file, CPFileAccessWrapper::READ )!=CPFileAccessWrapper::OK)
		return CPLoader::FILE_CANNOT_OPEN;
	
	p_sample->reset();
	CPLoader::Error res=load_sample(p_sample);
		
	file->close();
	
	return res;
}
CPLoader::Error CPLoader_IT::load_instrument(const char *p_file,CPSong *p_song,int p_instr_idx) {
	
	CP_FAIL_INDEX_V(p_instr_idx,CPSong::MAX_INSTRUMENTS,CPLoader::FILE_CANNOT_OPEN);
	
	if (file->open( p_file, CPFileAccessWrapper::READ )!=CPFileAccessWrapper::OK)
		return CPLoader::FILE_CANNOT_OPEN;
		
	
	p_song->get_instrument( p_instr_idx )->reset();
	
	
	int samples=0;
	CPLoader::Error res=load_instrument( p_song->get_instrument( p_instr_idx ), &samples );
		
	if (res) {
		file->close();
		return res;
	}
	
	
	char exchange[CPSong::MAX_SAMPLES];
	for (int i=0;i<CPSong::MAX_SAMPLES;i++)
		exchange[i]=0;
			
	for (int i=0;i<samples;i++) {
		
		file->seek( 554+i*80 ); //i think this should work?! seems to.. but i'm not sure	
		
		/* find free sample */
		
		int free_idx=-1;
		for (int s=0;s<CPSong::MAX_SAMPLES;s++) {
			
			if (p_song->get_sample( s )->get_sample_data().is_null()) {
				free_idx=s;
				break;
			}
		}
		if (free_idx==-1)
			break; //can't seem to be able to load more samples
		
		exchange[i]=free_idx;
		res=load_sample( p_song->get_sample( free_idx ) );
		
		if (res) {
			
			file->close();
			return res;
		}
	}
	
	for (int i=0;i<CPNote::NOTES;i++) {
		
		int smp=song->get_instrument(p_instr_idx)->get_sample_number(i);
		
		if (smp>=CPSong::MAX_SAMPLES)
			continue;
		
		if (smp<0)
			continue;
		
		smp=exchange[smp];
		
		song->get_instrument(p_instr_idx)->set_sample_number(i,smp);
		
	}
	
	file->close();

	return res;
	
}

CPLoader_IT::CPLoader_IT(CPFileAccessWrapper *p_file) {

	file=p_file;

}
