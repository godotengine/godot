/*************************************************************************/
/*  cp_loader_it_samples.cpp                                             */
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
#include "cp_sample.h"

struct AuxSampleData {


	uint32_t fileofs;
	uint32_t c5spd;
	uint32_t length;
	uint32_t loop_begin;
	uint32_t loop_end;
	bool loop_enabled;
	bool pingpong_enabled;
	bool is16bit;
	bool stereo;
	bool exists;
	bool compressed;

};


enum IT_Sample_Flags {

	IT_SAMPLE_EXISTS=1,
	IT_SAMPLE_16BITS=2,
	IT_SAMPLE_STEREO=4,
	IT_SAMPLE_COMPRESSED=8,
	IT_SAMPLE_LOOPED=16,
	IT_SAMPLE_SUSTAIN_LOOPED=32,
	IT_SAMPLE_LOOP_IS_PINGPONG=64,
	IT_SAMPLE_SUSTAIN_LOOP_IS_PINGPONG=128
};


CPLoader::Error CPLoader_IT::load_sample(CPSample *p_sample) {


	AuxSampleData aux_sample_data;

	char aux_header[4];

	file->get_byte_array((uint8_t*)aux_header,4);
	
	if (	aux_header[0]!='I' ||
		aux_header[1]!='M' ||
		aux_header[2]!='P' ||
		       aux_header[3]!='S') {
		
		//CP_PRINTERR("IT CPLoader CPSample: Failed Identifier");
		return FILE_UNRECOGNIZED;
	}
		

	// Ignore deprecated 8.3 filename
	for (int i=0;i<12;i++) file->get_byte();
	
	file->get_byte(); //ignore zerobyte
	
	p_sample->set_global_volume( file->get_byte() );
	
	/* SAMPLE FLAGS */
       	uint8_t flags=file->get_byte();
	aux_sample_data.loop_enabled=flags&IT_SAMPLE_LOOPED;
	aux_sample_data.pingpong_enabled=flags&IT_SAMPLE_LOOP_IS_PINGPONG;
	aux_sample_data.is16bit=flags&IT_SAMPLE_16BITS;
	aux_sample_data.exists=flags&IT_SAMPLE_EXISTS;
	aux_sample_data.stereo=flags&IT_SAMPLE_STEREO;
	aux_sample_data.compressed=flags&IT_SAMPLE_COMPRESSED;
	
	p_sample->set_default_volume(file->get_byte());
	/* SAMPLE NAME */
	char aux_name[26];
	file->get_byte_array((uint8_t*)aux_name,26);
	p_sample->set_name(aux_name);
	
	// ??
	uint8_t convert_flag=file->get_byte();
	// PAN
	uint8_t pan=file->get_byte();
	p_sample->set_pan( pan&0x7F ); 
	p_sample->set_pan_enabled( pan & 0x80 );

	aux_sample_data.length=file->get_dword();

	
	aux_sample_data.loop_begin= file->get_dword();
	aux_sample_data.loop_end= file->get_dword();
	aux_sample_data.c5spd=file->get_dword();
	/*p_sample->data.set_sustain_loop_begin=*/file->get_dword();
	/*p_sample->data.sustain_loop_end=*/file->get_dword();
	aux_sample_data.fileofs=file->get_dword();
	p_sample->set_vibrato_speed( file->get_byte() );
	p_sample->set_vibrato_depth( file->get_byte() );
	p_sample->set_vibrato_rate( file->get_byte() );
	switch( file->get_byte() ) {
		/* Vibrato Wave: 0=sine, 1=rampdown, 2=square, 3=random */
		case 0: p_sample->set_vibrato_type( CPSample::VIBRATO_SINE ); break;
		case 1: p_sample->set_vibrato_type( CPSample::VIBRATO_SAW ); break;
		case 2: p_sample->set_vibrato_type( CPSample::VIBRATO_SQUARE ); break;
		case 3: p_sample->set_vibrato_type( CPSample::VIBRATO_RANDOM ); break;
		default: p_sample->set_vibrato_type( CPSample::VIBRATO_SINE ); break;
	}
	
	//printf("Name %s - Flags: fileofs :%i - c5spd %i - len %i 16b %i - data?: %i\n",p_sample->get_name(),aux_sample_data.fileofs,aux_sample_data.c5spd, aux_sample_data.length, aux_sample_data.is16bit,aux_sample_data.exists);
	CPSample_ID samp_id;
	
	if (aux_sample_data.exists) {
		samp_id=load_sample_data(aux_sample_data);
		CPSampleManager::get_singleton()->set_c5_freq(samp_id,aux_sample_data.c5spd);
		CPSampleManager::get_singleton()->set_loop_begin( samp_id,aux_sample_data.loop_begin );
		CPSampleManager::get_singleton()->set_loop_end( samp_id,aux_sample_data.loop_end );
		CPSample_Loop_Type loop_type=aux_sample_data.loop_enabled?( aux_sample_data.pingpong_enabled? CP_LOOP_BIDI: CP_LOOP_FORWARD):CP_LOOP_NONE;
		CPSampleManager::get_singleton()->set_loop_end( samp_id,aux_sample_data.loop_end );
		CPSampleManager::get_singleton()->set_loop_type( samp_id, loop_type);
		
	}
	
	//printf("Loaded id is null?: %i\n",samp_id.is_null());
	p_sample->set_sample_data(samp_id);
	if (!samp_id.is_null()) {
		
		//printf("Loaded ID: stereo: %i len %i 16bit %i\n",CPSampleManager::get_singleton()->is_stereo(samp_id), CPSampleManager::get_singleton()->get_size( samp_id), CPSampleManager::get_singleton()->is_16bits( samp_id) );
	}
	
	CP_ERR_COND_V( file->eof_reached(),FILE_CORRUPTED );
	CP_ERR_COND_V( file->get_error(),FILE_CORRUPTED );
	
	return FILE_OK;

}

CPSample_ID CPLoader_IT::load_sample_data(AuxSampleData& p_sample_data) {


	int aux_sample_properties = (p_sample_data.is16bit?IT_SAMPLE_16BITS:0)|(p_sample_data.compressed?IT_SAMPLE_COMPRESSED:0)|(p_sample_data.stereo?IT_SAMPLE_STEREO:0);

	file->seek(p_sample_data.fileofs);
	
	CPSampleManager *sm=CPSampleManager::get_singleton();

	CPSample_ID id;
	
	switch (aux_sample_properties) {

		case (0):  // 8 bits, mono
		case (IT_SAMPLE_16BITS):  // 16 bits mono
		case (IT_SAMPLE_STEREO):  // 8 bits stereo
		case (IT_SAMPLE_16BITS|IT_SAMPLE_STEREO): { // 16 bits mono

			id=sm->create(p_sample_data.is16bit,p_sample_data.stereo,p_sample_data.length); 
			if (id.is_null())
				break;

			sm->lock_data(id);

			int16_t *ptr16 = (int16_t*)sm->get_data(id);
			int8_t *ptr8=(int8_t*)ptr16;

			int chans=p_sample_data.stereo?2:1;

			if (p_sample_data.is16bit) {

				for (int c=0;c<chans;c++) {

					for (int i=0;i<p_sample_data.length;i++) {

						ptr16[i*chans+c]=file->get_word();
					}
				}
			} else {

				for (int c=0;c<chans;c++) {

					for (int i=0;i<p_sample_data.length;i++) {

						ptr8[i*chans+c]=file->get_byte();
					}
				}

			}

			sm->unlock_data(id);

		} break;
		case (IT_SAMPLE_COMPRESSED): { // 8 bits compressed


			id=sm->create(false,false,p_sample_data.length); 
			if (id.is_null())
				break;
			sm->lock_data(id);
			
			if ( load_sample_8bits_IT_compressed((void*)sm->get_data( id),p_sample_data.length) ) {

				sm->unlock_data(id);
				sm->destroy(id);
				
				break;
			}

			sm->unlock_data(id);


		} break;
		case (IT_SAMPLE_16BITS|IT_SAMPLE_COMPRESSED): { // 16 bits compressed


			id=sm->create(true,false,p_sample_data.length); 
			if (id.is_null())
				break;
			sm->lock_data(id);
			
			if ( load_sample_16bits_IT_compressed((void*)sm->get_data(id),p_sample_data.length) ) {

				sm->unlock_data(id);
				sm->destroy(id);
				break;
			}

			sm->unlock_data(id);

		} break;
		default: {
			
			// I dont know how to handle stereo compressed, does that exist?
		} break;

	}


	return id;
}


CPLoader::Error CPLoader_IT::load_samples() {

	for (int i=0;i<header.smpnum;i++) {

		//seek to sample 
		file->seek(0xC0+header.ordnum+header.insnum*4+i*4);
		
		uint32_t final_location=file->get_dword();
		file->seek( final_location );
		

		Error err=load_sample(song->get_sample(i));
		CP_ERR_COND_V(err,err);

	}

	if (file->eof_reached() || file->get_error())
		return FILE_CORRUPTED;

	return FILE_OK;
}
/* * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE

 -The following sample decompression code is based on xmp's code.(http://xmp.helllabs.org) which is based in openCP code.

* NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE * NOTICE */

uint32_t CPLoader_IT::read_n_bits_from_IT_compressed_block (uint8_t p_bits_to_read) {

    uint32_t aux_return_value;
    uint32_t val;

    uint8_t *buffer=(uint8_t*)source_position;
    if ( p_bits_to_read <= source_remaining_bits ) {

    	val=buffer[3];
	val<<=8;
    	val|=buffer[2];
	val<<=8;
    	val|=buffer[1];
	val<<=8;
    	val|=buffer[0];

	aux_return_value = val & ((1 << p_bits_to_read) - 1);
	val >>= p_bits_to_read;
	source_remaining_bits -= p_bits_to_read;

	buffer[3]=val>>24;
    	buffer[2]=(val>>16)&0xFF;
    	buffer[1]=(val>>8)&0xFF;
    	buffer[0]=(val)&0xFF;

    } else {
    	aux_return_value=buffer[3];
	aux_return_value<<=8;
    	aux_return_value|=buffer[2];
	aux_return_value<<=8;
    	aux_return_value|=buffer[1];
	aux_return_value<<=8;
    	aux_return_value|=buffer[0];

	uint32_t nbits = p_bits_to_read - source_remaining_bits;
	source_position++;

        buffer+=4;
    	val=buffer[3];
	val<<=8;
    	val|=buffer[2];
	val<<=8;
    	val|=buffer[1];
	val<<=8;
    	val|=buffer[0];
	aux_return_value |= ((val & ((1 << nbits) - 1)) << source_remaining_bits);
	val >>= nbits;
	source_remaining_bits = 32 - nbits;
	buffer[3]=val>>24;
    	buffer[2]=(val>>16)&0xFF;
    	buffer[1]=(val>>8)&0xFF;
    	buffer[0]=(val)&0xFF;

    }

    return aux_return_value;
}

bool CPLoader_IT::read_IT_compressed_block (bool p_16bits) {

	uint16_t size;

	size=file->get_word();

	if (file->eof_reached() || file->get_error()) return true;

	pat_data = (uint8_t*)CP_ALLOC( 4* ((size >> 2) + 2) );
	if (!pat_data)
		return true;
	

	source_buffer=(uint32_t*)pat_data;
	file->get_byte_array((uint8_t*)source_buffer,size);
	
	if (file->eof_reached() || file->get_error()) {
		
		free_IT_compressed_block();
		return true;
	}
	
	source_position = source_buffer;
	source_remaining_bits = 32;

	return false;
}

void CPLoader_IT::free_IT_compressed_block () {


	if (pat_data) {
		CP_FREE(pat_data);
		pat_data=NULL;
	}

}

bool CPLoader_IT::load_sample_8bits_IT_compressed(void *p_dest_buffer,int p_buffsize) {

	int8_t *dest_buffer;		/* destination buffer which will be returned */
   	uint16_t block_length;		/* length of compressed data block in samples */
	uint16_t block_position;		/* position in block */
	uint8_t bit_width;			/* actual "bit width" */
	uint16_t aux_value;			/* value read from file to be processed */
	int8_t d1, d2;		/* integrator buffers (d2 for it2.15) */
	int8_t *dest_position;		/* position in output buffer */
	int8_t v;			/* sample value */
	bool it215; // is this an it215 module?

	dest_buffer = (int8_t *) p_dest_buffer;

	if (dest_buffer==NULL) 
		return true;

	for (int i=0;i<p_buffsize;i++)
		dest_buffer[i]=0;


	dest_position = dest_buffer;

	it215=(header.cmwt==0x215);

	/* now unpack data till the dest buffer is full */

	while (p_buffsize) {
	/* read a new block of compressed data and reset variables */
		if ( read_IT_compressed_block(false) ) {
			CP_PRINTERR("Out of memory decompressing IT CPSample");
			return true;
		}


		block_length = (p_buffsize < 0x8000) ? p_buffsize : 0x8000;

		block_position = 0;

		bit_width = 9;		/* start with width of 9 bits */

		d1 = d2 = 0;		/* reset integrator buffers */

	/* now uncompress the data block */
		while ( block_position < block_length ) {

			aux_value = read_n_bits_from_IT_compressed_block(bit_width);			/* read bits */

			if ( bit_width < 7 ) { /* method 1 (1-6 bits) */

				if ( aux_value == (1 << (bit_width - 1)) ) { /* check for "100..." */

					aux_value = read_n_bits_from_IT_compressed_block(3) + 1; /* yes -> read new width; */
		    			bit_width = (aux_value < bit_width) ? aux_value : aux_value + 1;
							/* and expand it */
		    			continue; /* ... next value */
				}

			} else if ( bit_width < 9 ) { /* method 2 (7-8 bits) */

				uint8_t border = (0xFF >> (9 - bit_width)) - 4;
							/* lower border for width chg */

				if ( aux_value > border && aux_value <= (border + 8) ) {

					aux_value -= border; /* convert width to 1-8 */
					bit_width = (aux_value < bit_width) ? aux_value : aux_value + 1;
							/* and expand it */
		    			continue; /* ... next value */
				}


			} else if ( bit_width == 9 ) { /* method 3 (9 bits) */

				if ( aux_value & 0x100 ) {			/* bit 8 set? */

					bit_width = (aux_value + 1) & 0xff;		/* new width... */
		    			continue;				/* ... and next value */
				}

			} else { /* illegal width, abort */

				
				free_IT_compressed_block();
				CP_PRINTERR("CPSample has illegal BitWidth ");
				return true;
			}

			/* now expand value to signed byte */
			if ( bit_width < 8 ) {

				uint8_t tmp_shift = 8 - bit_width;

				v=(aux_value << tmp_shift);
				v>>=tmp_shift;

			} else v = (int8_t) aux_value;

			/* integrate upon the sample values */
			d1 += v;
	    		d2 += d1;

			/* ... and store it into the buffer */
			*(dest_position++) = it215 ? d2 : d1;
			block_position++;

		}

		/* now subtract block lenght from total length and go on */
		free_IT_compressed_block();
		p_buffsize -= block_length;
	}


	return false;
}

bool CPLoader_IT::load_sample_16bits_IT_compressed(void *p_dest_buffer,int p_buffsize) {

	int16_t *dest_buffer;		/* destination buffer which will be returned */
   	uint16_t block_length;		/* length of compressed data block in samples */
	uint16_t block_position;		/* position in block */
	uint8_t bit_width;			/* actual "bit width" */
	uint32_t aux_value;			/* value read from file to be processed */
	int16_t d1, d2;		/* integrator buffers (d2 for it2.15) */
	int16_t *dest_position;		/* position in output buffer */
	int16_t v;			/* sample value */

	bool it215; // is this an it215 module?

	dest_buffer = (int16_t *) p_dest_buffer;

	if (dest_buffer==NULL) 
		return true;

	for (int i=0;i<p_buffsize;i++)
		dest_buffer[i]=0;

	dest_position = dest_buffer;

	it215=(header.cmwt==0x215);


	while (p_buffsize) {
	/* read a new block of compressed data and reset variables */
		if ( read_IT_compressed_block(true) ) {

			return true;
		}


		block_length = (p_buffsize < 0x4000) ? p_buffsize : 0x4000;

		block_position = 0;

		bit_width = 17;		/* start with width of 9 bits */

		d1 = d2 = 0;		/* reset integrator buffers */

		while ( block_position < block_length ) {

			aux_value = read_n_bits_from_IT_compressed_block(bit_width);			/* read bits */

			if ( bit_width < 7 ) { /* method 1 (1-6 bits) */

				if ( (signed)aux_value == (1 << (bit_width - 1)) ) { /* check for "100..." */

					aux_value = read_n_bits_from_IT_compressed_block(4) + 1; /* yes -> read new width; */
		    			bit_width = (aux_value < bit_width) ? aux_value : aux_value + 1;
							/* and expand it */
		    			continue; /* ... next value */
				}

			} else if ( bit_width < 17 ) {

				uint16_t border = (0xFFFF >> (17 - bit_width)) - 8;

				if ( (int)aux_value > (int)border && (int)aux_value <= ((int)border + 16) ) {

					aux_value -= border; /* convert width to 1-8 */
					bit_width = (aux_value < bit_width) ? aux_value : aux_value + 1;
							/* and expand it */
		    			continue; /* ... next value */
				}


			} else if ( bit_width == 17 ) {

				if ( aux_value & 0x10000 ) {			/* bit 8 set? */

					bit_width = (aux_value + 1) & 0xff;		/* new width... */
		    			continue;				/* ... and next value */
				}

			} else { /* illegal width, abort */

				CP_PRINTERR("CPSample has illegal BitWidth ");

				free_IT_compressed_block();
			
				return true;
			}

			/* now expand value to signed byte */
			if ( bit_width < 16 ) {

				uint8_t tmp_shift = 16 - bit_width;

				v=(aux_value << tmp_shift);
				v>>=tmp_shift;

			} else v = (int16_t) aux_value;

			/* integrate upon the sample values */
			d1 += v;
	    		d2 += d1;

			/* ... and store it into the buffer */
			*(dest_position++) = it215 ? d2 : d1;
			block_position++;

		}

		/* now subtract block lenght from total length and go on */
		free_IT_compressed_block();
		p_buffsize -= block_length;
	}


	return false;

}



