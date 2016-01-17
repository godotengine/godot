/*************************************************************************/
/*  cp_loader_it_instruments.cpp                                         */
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

enum EnvFlags {
	ENV_ON=1,
	ENV_LOOP=2,
	ENV_SUSLOOP=4,
	ENV_CARRY=8,
	ENV_FILTER=128
};

void CPLoader_IT::load_envelope(CPEnvelope *p_envelope,bool*p_has_filter_flag) { 

	uint8_t flags=file->get_byte();
	uint8_t points=file->get_byte();
	uint8_t begin=file->get_byte();
	uint8_t end=file->get_byte();
	uint8_t susbegin=file->get_byte();
	uint8_t susend=file->get_byte();
	
	p_envelope->reset();
	
	for (int i=0;i<25;i++) {

		uint8_t height=file->get_byte();
		int8_t &signed_height=(int8_t&)height;
		uint16_t tick=file->get_word();
		
		if (i>=points)
			continue;
		p_envelope->add_position( tick, signed_height );
	
	}

	p_envelope->set_enabled( flags & ENV_ON );
	p_envelope->set_carry_enabled( flags & ENV_CARRY);
	
	p_envelope->set_loop_enabled( flags & ENV_LOOP );
	p_envelope->set_loop_begin( begin );
	p_envelope->set_loop_end( end );
	
	p_envelope->set_sustain_loop_enabled( flags & ENV_SUSLOOP );
	p_envelope->set_sustain_loop_begin( susbegin );
	p_envelope->set_sustain_loop_end( susend );
	
	if (p_has_filter_flag)
		*p_has_filter_flag=flags&ENV_FILTER;
	
	file->get_byte(); //zerobyte
	
	//fill with stuff if the envelope hass less than 2 points
	while(p_envelope->get_node_count()<2) {
		
		p_envelope->add_position( 30*p_envelope->get_node_count(), p_envelope->get_min()==0 ? 64 : 0, false );
	}
}


CPLoader::Error CPLoader_IT::load_instrument(CPInstrument *p_instrument,int *p_samples) {



	char aux_header[4];
	
	file->get_byte_array((uint8_t*)aux_header,4);	

	
	if (	aux_header[0]!='I' ||
		       aux_header[1]!='M' ||
		       aux_header[2]!='P' ||
		       aux_header[3]!='I') {
		CP_PRINTERR("IT CPLoader CPInstrument: Failed Identifier");
	
		return FILE_UNRECOGNIZED;
	}
		
	

	// Ignore deprecated 8.3 filename field
	for (int i=0;i<12;i++) file->get_byte();
	
	//Ignore zerobyte
	file->get_byte();		/* (byte) CPInstrument type (always 0) */
	
	switch( file->get_byte() ) { /* New CPNote Action [0,1,2,3] */
		case 0: p_instrument->set_NNA_type( CPInstrument::NNA_NOTE_CUT ) ; break;
		case 1: p_instrument->set_NNA_type( CPInstrument::NNA_NOTE_CONTINUE ) ; break;
		case 2: p_instrument->set_NNA_type( CPInstrument::NNA_NOTE_OFF ) ; break;
		case 3: p_instrument->set_NNA_type( CPInstrument::NNA_NOTE_FADE ) ; break;
	};
	switch( file->get_byte() ) { // Duplicate Check Type
		case 0: p_instrument->set_DC_type( CPInstrument::DCT_DISABLED ); break ;		
		case 1: p_instrument->set_DC_type( CPInstrument::DCT_NOTE ); break ;		
		case 2: p_instrument->set_DC_type( CPInstrument::DCT_SAMPLE ); break ;		
		case 3: p_instrument->set_DC_type( CPInstrument::DCT_INSTRUMENT ); break ;		
	}
	switch( file->get_byte() ) { //Duplicate Check Action
		case 0: p_instrument->set_DC_action( CPInstrument::DCA_NOTE_CUT ); break ;
		case 1: p_instrument->set_DC_action( CPInstrument::DCA_NOTE_OFF ); break ;
		case 2: p_instrument->set_DC_action( CPInstrument::DCA_NOTE_FADE ); break ;
	}
	
	int fade = file->get_word();
	//intf("AFADE: %i\n",fade);
	if (fade>CPInstrument::MAX_FADEOUT) //needs to be clipped because of horrible modplug doings
		fade=CPInstrument::MAX_FADEOUT;
	
	p_instrument->set_volume_fadeout( fade );
	p_instrument->set_pan_pitch_separation( file->get_byte() );
	p_instrument->set_pan_pitch_center( file->get_byte() );
	p_instrument->set_volume_global_amount( file->get_byte() );
	uint8_t pan=file->get_byte();
	p_instrument->set_pan_default_amount(pan&0x7F);
	p_instrument->set_pan_default_enabled( !(pan&0x80) );
	p_instrument->set_volume_random_variation( file->get_byte() );
	p_instrument->set_pan_random_variation( file->get_byte() );
	
	
	
	file->get_word(); //empty (version)
	uint8_t samples=file->get_byte();
	if (p_samples)
		*p_samples=samples;
	file->get_byte(); //empty
	char aux_name[26];	
	file->get_byte_array((uint8_t*)aux_name,26);
	p_instrument->set_name(aux_name);
	
	uint8_t cutoff=file->get_byte();
		
	p_instrument->set_filter_default_cutoff(cutoff&0x7F);
	p_instrument->set_filter_use_default_cutoff(cutoff&0x80); 
	
	uint8_t resonance=file->get_byte();
		
	p_instrument->set_filter_default_resonance(resonance&0x7F);
	p_instrument->set_filter_use_default_resonance(resonance&0x80); 
	
	file->get_dword(); //MIDI, IGNORED!
	
	/* CPNote -> CPSample table */
	for (uint8_t i=0;i<CPNote::NOTES;i++) {
		
		
		uint8_t note=file->get_byte();
		if (note>=CPNote::NOTES)
			note=0;
		p_instrument->set_note_number(i,note);		
		
		uint8_t samp=file->get_byte();
		if (samp==0 || samp>99)
			samp=CPNote::EMPTY;
		else 
			samp--;
		
		
		p_instrument->set_sample_number(i,samp);
	

	}

	
	load_envelope( p_instrument->get_volume_envelope() );
	load_envelope( p_instrument->get_pan_envelope() );
	bool use_as_filter;
	load_envelope( p_instrument->get_pitch_filter_envelope(), &use_as_filter );
	p_instrument->set_pitch_use_as_filter( use_as_filter );

	return FILE_OK;

}


CPLoader::Error CPLoader_IT::load_instruments() {


	for (int i=0;i<header.insnum;i++) {

		
		file->seek(0xC0+header.ordnum+i*4);
		uint32_t final_location=file->get_dword();
		file->seek( final_location );
		
		Error err=load_instrument( song->get_instrument( i ) );
		if (err)
			return err;

	}

	return FILE_OK;

	if (file->eof_reached() || file->get_error())
		return FILE_CORRUPTED;
}


