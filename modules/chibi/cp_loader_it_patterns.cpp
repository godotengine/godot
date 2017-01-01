/*************************************************************************/
/*  cp_loader_it_patterns.cpp                                            */
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


CPLoader::Error CPLoader_IT::load_patterns() {


	for (int i=0;i<header.patnum;i++) {

		if (i>=CPSong::MAX_PATTERNS)
			break;
			
		/* Position where pattern offsets are stored */
		file->seek(0xC0+header.ordnum+header.insnum*4+header.smpnum*4+i*4);
		uint32_t pattern_offset=file->get_dword();
		
		if (pattern_offset==0) {

			continue;
		}
			
		uint16_t pat_size;
		uint16_t pat_length;

		int row=0,flag,channel,j;
		uint8_t aux_byte;
		uint32_t reserved;
		uint8_t chan_mask[64]; //mask cache for each
		CPNote last_value[64]; //last value of each

		for (j=0;j<64;j++) {

			chan_mask[j]=0;
			last_value[j].clear();
		}

		file->seek(pattern_offset);

		pat_size=file->get_word();
		pat_length=file->get_word();
		reserved=file->get_dword();

		song->get_pattern(i)->set_length( pat_length );
		
		do {

			aux_byte=file->get_byte();
			flag=aux_byte;

			if ( flag==0 ) {

				row++;
			} else {

				channel=(flag-1) & 63;

				if ( flag & 128 ) {

					aux_byte=file->get_byte();
					chan_mask[channel]=aux_byte;
				}

				CPNote note; //note used for reading

				if ( chan_mask[channel]&1 ) { // read note
			
					aux_byte=file->get_byte();
					
					if ( aux_byte<120 )
						note.note=aux_byte;
					else if ( aux_byte==255 ) 	
						note.note=CPNote::OFF;
					else if ( aux_byte==254 )
						note.note=CPNote::CUT;

					last_value[channel].note=note.note;
				}
					

				if ( chan_mask[channel]&2 ) {

					aux_byte=file->get_byte();
					if ( aux_byte<100 )
						note.instrument=aux_byte-1;

					last_value[channel].instrument=note.instrument;
				}
				if ( chan_mask[channel]&4 ) {

					aux_byte=file->get_byte();
					if ( aux_byte<213 )
						note.volume=aux_byte;

					last_value[channel].volume=note.volume;
				}
				if ( chan_mask[channel]&8 ) {

					aux_byte=file->get_byte();
					if ( aux_byte>0 ) 
						note.command=aux_byte-1;
					
					
					last_value[channel].command=note.command;

					note.parameter=file->get_byte();
					
					last_value[channel].parameter=note.parameter;
				}

				if ( chan_mask[channel]&16 ) {

					note.note=last_value[channel].note;
				}

				if ( chan_mask[channel]&32 ) {

					note.instrument=last_value[channel].instrument;
				}
				if ( chan_mask[channel]&64 ) {

					note.volume=last_value[channel].volume;
				}
				if ( chan_mask[channel]&128 ) {

					note.command=last_value[channel].command;
					note.parameter=last_value[channel].parameter;
				}
				
				song->get_pattern(i)->set_note(channel,row,note);
			}
			
			
		} while(row<pat_length);

	}

	return FILE_OK;
}

