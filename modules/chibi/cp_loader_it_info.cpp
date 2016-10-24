/*************************************************************************/
/*  cp_loader_it_info.cpp                                                */
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
#include "cp_loader_it.h"



CPLoader::Error CPLoader_IT::load_header(bool p_dont_set) {

	
	char aux_songname[26];		
		
	file->get_byte_array((uint8_t*)aux_songname,26);
	if (!p_dont_set)
		song->set_name( aux_songname );
	
	uint8_t aux_hlmin=file->get_byte();
	uint8_t aux_hlmaj=file->get_byte();

	if (aux_hlmin==0) aux_hlmin=4;
	if (aux_hlmaj==0) aux_hlmaj=16;

	if (!p_dont_set) {
		song->set_row_highlight_minor( aux_hlmin );
		song->set_row_highlight_major( aux_hlmaj );
	}

	header.ordnum=file->get_word();
	header.insnum=file->get_word();
	header.smpnum=file->get_word();
	header.patnum=file->get_word();
	
	header.cwt=file->get_word();		/* Created with tracker (y.xx = 0x0yxx) */
	header.cmwt=file->get_word();		/* Compatible with tracker ver > than val. */
	header.flags=file->get_word();
	
	if (!p_dont_set) {
		song->set_stereo( header.flags & 1 );
		song->set_linear_slides( header.flags & 8 );
		song->set_old_effects( header.flags & 16 );
		song->set_compatible_gxx( header.flags & 32 );
		song->set_instruments( header.flags & 4 );
	}
	
	
	header.special=file->get_word();
	if (!p_dont_set) {

		song->set_global_volume( file->get_byte() );
		song->set_mixing_volume( file->get_byte() );
		song->set_speed( file->get_byte() );
		song->set_tempo( file->get_byte() );
		song->set_stereo_separation( file->get_byte() );

	} else {

		file->get_byte(); // skip
		file->get_byte(); // skip
		file->get_byte(); // skip
		file->get_byte(); // skip
		file->get_byte(); // skip
	}
	file->get_byte(); // ZERO Byte
	header.msglength=file->get_word();
	header.msgoffset=file->get_dword();
	char chibi[4];
	file->get_byte_array((uint8_t*)chibi,4);
	header.is_chibi=(chibi[0]=='C' && chibi[1]=='H' && chibi[2]=='B' && chibi[3]=='I');
	
	for (int i=0;i<64;i++) {
		
		uint8_t panbyte=file->get_byte();
		
		uint8_t pan_dst=(panbyte<65) ? panbyte : 32;
		bool surround_dst=(panbyte==100);
		bool mute_dst=(panbyte>=128);

		if (!p_dont_set) {
			song->set_channel_pan( i, pan_dst );
			song->set_channel_surround( i, surround_dst );
			song->set_channel_mute( i, mute_dst );
		}
	}
	for (int i=0;i<64;i++) {
		unsigned char cv = file->get_byte();
		if (!p_dont_set)
			song->set_channel_volume( i, cv );
	}

	CP_ERR_COND_V( file->eof_reached(),FILE_CORRUPTED );
	CP_ERR_COND_V( file->get_error(),FILE_CORRUPTED );

	return FILE_OK;
}

CPLoader::Error CPLoader_IT::load_effects() {
	
	if (!header.is_chibi)
		return FILE_OK; //no effects, regular IT file
	
	/* GOTO End of IT header */
	file->seek(0xC0+header.ordnum+header.insnum*4+header.smpnum*4+header.patnum*4);
	
	
	if (file->get_byte()>0) //not made with this version, ignore extended info
		return FILE_OK;
	
	/* Chibitracker Extended info */

	switch(file->get_byte()) {
		
		case CPSong::REVERB_MODE_ROOM: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_ROOM );
		} break;
		case CPSong::REVERB_MODE_STUDIO_SMALL: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_STUDIO_SMALL );
			
		} break;
		case CPSong::REVERB_MODE_STUDIO_MEDIUM: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_STUDIO_MEDIUM );
			
		} break;
		case CPSong::REVERB_MODE_STUDIO_LARGE: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_STUDIO_LARGE );
			
		} break;
		case CPSong::REVERB_MODE_HALL: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_HALL );
			
		} break;
		case CPSong::REVERB_MODE_SPACE_ECHO: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_SPACE_ECHO );
			
		} break;

		case CPSong::REVERB_MODE_ECHO: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_ECHO );
			
		} break;
		case CPSong::REVERB_MODE_DELAY: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_DELAY );
			
		} break;
		case CPSong::REVERB_MODE_HALF_ECHO: {
			
			song->set_reverb_mode( CPSong::REVERB_MODE_HALF_ECHO );
			
		} break;
	
	}
		
	//chorus
	song->set_chorus_speed_hz10( file->get_byte() );
	song->set_chorus_delay_ms( file->get_byte() );
	song->set_chorus_depth_ms10( file->get_byte() );
	song->set_chorus_separation_ms( file->get_byte() );
	
	for (int i=0;i<CPPattern::WIDTH;i++) {
		song->set_channel_reverb(i,file->get_byte());
	}
	for (int i=0;i<CPPattern::WIDTH;i++) {
		song->set_channel_chorus(i,file->get_byte());
	}
	
	return FILE_OK;
	
}

CPLoader::Error CPLoader_IT::load_message() {

	
	if (!(header.special & 1)) {

		return FILE_OK;
	}		


	file->seek(header.msgoffset);

	//(void*)tmpmsg=malloc(header.msglength+1);

	char message[8000];

	
	char *tmpmsg = message;

	file->get_byte_array((uint8_t*)tmpmsg,header.msglength);
	tmpmsg[header.msglength]=0;
	
	for (int i=0;i<header.msglength;i++) if (tmpmsg[i]=='\r') tmpmsg[i]='\n';

	song->set_message(tmpmsg);
	
	return FILE_OK;
}

CPLoader::Error CPLoader_IT::load_orders() {

	file->seek(0xC0);
	
	
	for (int i=0;i<header.ordnum;i++) {
		
		uint8_t aux_order=file->get_byte();
		CPOrder order=CP_ORDER_NONE;
		
		
		if (i>=CPSong::MAX_ORDERS)
			continue;
		if (aux_order==254)  {

			order=CP_ORDER_BREAK;

		} else if (aux_order<200) {

			order=aux_order;
			//nothing!

		} 
		song->set_order(i,order);
		
	}
	
	if (file->eof_reached() || file->get_error()) {


		return FILE_CORRUPTED;

	}
	
	return FILE_OK;
}



