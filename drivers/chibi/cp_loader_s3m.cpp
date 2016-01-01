/*************************************************************************/
/*  cp_loader_s3m.cpp                                                    */
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

#include "cp_loader_s3m.h"

#define BITBOOL(m_exp) ((m_exp)?1:0)


CPLoader::Error CPLoader_S3M::load_header() {

	int i;

	
        file->get_byte_array((uint8_t*)header.songname,28);
        header.t1a=file->get_byte();
        header.type=file->get_byte();
        file->get_byte_array((uint8_t*)header.unused1,2);
        header.ordnum=file->get_word();
        header.insnum=file->get_word();
        header.patnum=file->get_word();
        header.flags=file->get_word();
        header.tracker=file->get_word();
        header.fileformat=file->get_word();
        file->get_byte_array((uint8_t*)header.scrm,4);
        header.scrm[4]=0;
	
	if (header.scrm[0]!='S' || header.scrm[1]!='C' || header.scrm[2]!='R' || header.scrm[3]!='M')
		return FILE_UNRECOGNIZED;

        header.mastervol=file->get_byte();
        header.initspeed=file->get_byte();
        header.inittempo=file->get_byte();
        header.mastermult=file->get_byte();
        header.ultraclick=file->get_byte();
        header.pantable=file->get_byte();
        file->get_byte_array((uint8_t*)header.unused2,8);
        header.special=file->get_word();
        file->get_byte_array((uint8_t*)header.channels,32);

        file->get_byte_array((uint8_t*)header.orderlist,header.ordnum);

        header.scrm[4]=0;
	if (header.scrm[0]!='S' || header.scrm[1]!='C' || header.scrm[2]!='R' || header.scrm[3]!='M') //again?
		return FILE_UNRECOGNIZED;
	//sample parapointers
	for (i=0;i<header.insnum;i++) {

		int parapointer;
		parapointer=file->get_word();
		parapointer=(parapointer*16);
		sample_parapointers[i]=parapointer;
	}
	//pattern
	for (i=0;i<header.patnum;i++) {

		int parapointer;
		parapointer=file->get_word();
		parapointer=(parapointer*16);
		pattern_parapointers[i]=parapointer;
	}

        if (header.pantable==252) {

	        file->get_byte_array((uint8_t*)header.pannings,32);
	}

	return FILE_OK;


}


void CPLoader_S3M::set_header() {
	
	


	song->set_name( header.songname );
//	song->variables.filename=
	
	song->set_row_highlight_minor( 4 );
	song->set_row_highlight_major( 16 );
	song->set_mixing_volume( header.mastervol );
	song->set_linear_slides( false );
	song->set_old_effects( !(header.flags&64) );
	song->set_compatible_gxx( true );

	song->set_global_volume( header.mastermult );
	song->set_speed( header.initspeed );
	song->set_tempo( header.inittempo );

	//[TODO] Set Panning Positions.. ?

	for (int i=0;i<header.ordnum;i++) song->set_order(i,header.orderlist[i]);
	
}

CPLoader::Error  CPLoader_S3M::load_sample(CPSample *p_sample) {



	        int type=file->get_byte();
		
		char filename[13];
                file->get_byte_array((uint8_t*)filename,12);
		filename[12]=0;
		
		
		uint32_t samplepos=(uint32_t)file->get_byte() << 16;
                samplepos|=file->get_word();
		samplepos*=16;	
//		printf("sample at %i\n",samplepos);
		/**/
                int sample_size=file->get_dword();
		
	
		int loop_begin=file->get_dword();
		int loop_end=file->get_dword();

		int def_volume=file->get_byte();;
		int dsk=file->get_byte();
		int pack=file->get_byte();
		
		int flags=file->get_byte();
		int c2speed=file->get_dword();
		
		file->get_dword(); //useless crap
		file->get_dword();
		file->get_dword();
		
		
		char name[29];
                file->get_byte_array((uint8_t*)name,28);
		name[28]=0;
		
		p_sample->set_default_volume(def_volume);
		p_sample->set_name(name);
		
		char scrs[5];
		file->get_byte_array((uint8_t*)scrs,4);
		scrs[4]=0;

		

                bool data_is_16bits=flags&4;
		bool data_is_stereo=flags&2;

		if (type==0) {
			//empty sample
			return FILE_OK;
		}
			

		if ((type!=1) || scrs[0]!='S' || scrs[1]!='C' || scrs[2]!='R' || scrs[3]!='S' ) {
//			printf("type: %i, %c%c%c%c\n",type,scrs[0],scrs[1],scrs[2],scrs[3]);
			CP_PRINTERR("Not an S3M CPSample!");
			return FILE_CORRUPTED;
		}

		//p_sample->data.set_c5_freq(p_sample->c2spd<<1);

		file->seek(samplepos);

		int real_sample_size=sample_size<<BITBOOL(data_is_16bits);
		real_sample_size<<=BITBOOL(data_is_stereo);

		CPSampleManager *sm=CPSampleManager::get_singleton();
		
		CPSample_ID id =sm->create( data_is_16bits, data_is_stereo, sample_size );
		
		if (id.is_null())
			return FILE_OUT_OF_MEMORY;

		sm->lock_data(id);
		void *dataptr = sm->get_data(id);
		
		int chans = (data_is_stereo?2:1);
		for (int c=0;c<chans;c++) {
			for (int i=0;i<sample_size;i++) {
				
				if (data_is_16bits) {
						
					uint16_t s=file->get_word();
					s-=32768; //toggle sign
					
					int16_t *v=(int16_t*)&s;
					((int16_t*)dataptr)[i*chans+c]=*v;
				} else {
						
						
					int8_t *v;
					uint8_t s=file->get_byte();
					s-=128; //toggle sign
					v=(int8_t*)&s;
					((int8_t*)dataptr)[i*chans+c]=*v;

				}

			}
			
		}

		sm->unlock_data(id);

						  
		sm->set_loop_begin( id, loop_begin );
		sm->set_loop_end( id, loop_end );
		sm->set_loop_type( id, (flags&1) ? CP_LOOP_FORWARD : CP_LOOP_NONE );
		sm->set_c5_freq( id, c2speed << 1 );
		p_sample->set_sample_data(id);
		
		/* Scream tracker previous to 3.10 seems to be buggy, as in, wont save what is after the sample loop, including the loop end point. Because of this I must fix it by habd */
		if (flags&1) {
			
			for (int c=0;c<(data_is_stereo?2:1);c++) {
				sm->set_data( id, loop_end, sm->get_data( id, loop_begin,c ),c );
				
			}
		}
				  

		return FILE_OK;

}


CPLoader::Error CPLoader_S3M::load_pattern(CPPattern *p_pattern) {

        int row=0,flag,ch;
	CPNote n;
	int length,accum=0;

	length=file->get_word();
        p_pattern->set_length(64);

        /* clear pattern data */
        while((row<64) && (accum<=length) ) {
                flag=file->get_byte();
		accum++;

		n.clear();
                if(flag) {
                       // ch=remap[flag&31];
//                        ch=remap[flag&31];
//                        if(ch!=-1)
//                                n=s3mbuf[(64U*ch)+row];
//                        else
//                                n=&dummy;

			ch=flag&31;

                        if(flag&32) {
                                n.note=file->get_byte();
				if (n.note==255) {

					n.note=CPNote::EMPTY;
				} else if (n.note==254) {

					n.note=CPNote::CUT;
				} else {
				
                                	n.note=((n.note>>4)*12)+(n.note&0xF);
				}

                                n.instrument=file->get_byte()-1;
				accum+=2;

                        }
                        if(flag&64) {
                                n.volume=file->get_byte();
                                if (n.volume>64) n.volume=64;
				accum++;

                        }
                        if(flag&128) {
                                n.command=file->get_byte()-1;
                                n.parameter=file->get_byte();
				accum+=2;
                        }
		
			p_pattern->set_note(ch,row,n);
                } else row++;
        }
        return FILE_OK;


}

CPLoader::Error CPLoader_S3M::load_sample(const char *p_file,CPSample *p_sample) {
	
	return FILE_UNRECOGNIZED;
}
CPLoader::Error CPLoader_S3M::load_instrument(const char *p_file,CPSong *p_song,int p_instr_idx) {
	
	return FILE_UNRECOGNIZED;
	
}


CPLoader::Error CPLoader_S3M::load_samples() {

	int i;

	for(i=0;i<header.insnum;i++) {

		file->seek(sample_parapointers[i]);
		load_sample(song->get_sample(i));
		sample_count++;
	}

	return FILE_OK;
}

CPLoader::Error CPLoader_S3M::load_patterns() {

	int i;

	Error err;
	for(i=0;i<header.patnum;i++) {

		file->seek(pattern_parapointers[i]);
		
		err=load_pattern(song->get_pattern(i) );
		CP_ERR_COND_V(err,err);

		
		pattern_count++;
	}
	return FILE_OK;

}

CPLoader::Error CPLoader_S3M::load_song(const char *p_file,CPSong *p_song,bool p_sampleset) {

	song=p_song;

	if (file->open(p_file,CPFileAccessWrapper::READ)) {
		//printf("Can't open file! %s\n",p_file);
		return FILE_CANNOT_OPEN;
	};
	
        sample_count=0;
	pattern_count=0;

	//printf("LOADING HEADER\n");
	CPLoader::Error err;
	if ((err=load_header())) {
		file->close();
		CP_ERR_COND_V(err,err);
		
	}		

	song->reset(); //file type recognized, reset song!
	
	set_header();
	
	//printf("LOADING SAMPLES\n");
	
	if ((err=load_samples())) {
		file->close();

		CP_ERR_COND_V(err,err);
	}		

	//printf("LOADING PATTERNS\n");
	
	if ((err=load_patterns())) {

		file->close();
		return err;
	}		

	file->close();

	return FILE_OK;
}



CPLoader_S3M::CPLoader_S3M(CPFileAccessWrapper *p_file){

	file=p_file;

}
CPLoader_S3M::~CPLoader_S3M(){
}

