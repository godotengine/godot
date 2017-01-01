/*************************************************************************/
/*  cp_loader_xm.cpp                                                     */
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

#include "cp_loader_xm.h"
#include "cp_tables.h"

#define ABORT_LOAD  { file->close(); return FILE_CORRUPTED; }




CPLoader::Error CPLoader_XM::load_song(const char *p_file,CPSong *p_song,bool p_sampleset) {

	song=p_song;

	if (file->open(p_file,CPFileAccessWrapper::READ)) {

		return FILE_CANNOT_OPEN;
        };


        /**************************************
        LOAD HEADER
        ***************************************/

  	file->get_byte_array(header.idtext,17);
   	header.idtext[17]=0;

  	file->get_byte_array(header.songname,20);
	


   	header.songname[20]=0;
  	header.hex1a=file->get_byte();
	if (header.hex1a!=0x1A) { //XM "magic" byte..  this sucks :)

		file->close(); 
		return FILE_UNRECOGNIZED;
      		
        }

	
	//magic byte sucks, but can't do much about it.. 
	
	song->reset(); //must reset the song

	song->set_name( (const char*)header.songname );
	
  	file->get_byte_array(header.trackername,20);
  	header.trackername[20]=0;


  	header.version=file->get_word();

  	header.headersize=file->get_dword();

	header.songlength=file->get_word();

  	header.restart_pos=file->get_word();

  	header.channels_used=file->get_word();

  	header.patterns_used=file->get_word();

  	header.instruments_used=file->get_word();

	song->set_linear_slides( file->get_word() );

	song->set_speed( file->get_word() );

	song->set_tempo( file->get_word() );
	song->set_instruments( true );

  	file->get_byte_array(header.orderlist,256);
	
	for (int i=0;i<header.songlength;i++) {

		if (i>199)
			break;
		song->set_order(i,header.orderlist[i]);
	}

        /**************************************
        LOAD PATTERNS
        ***************************************/

        for (int i=0;i<header.patterns_used;i++) {

          	uint32_t aux,rows;

           	aux=file->get_dword(); //length
           	aux=file->get_byte(); //packing type
           	rows=aux=file->get_word(); //rows!

		song->get_pattern(i)->set_length( aux );

           	aux=file->get_word(); //packed size
            	if (aux==0)
             		continue;
                //unpaaack!
		for(int j=0;j<(int)rows;j++)
			for(int k=0;k<header.channels_used;k++) {

				CPNote aux_note;
         			uint8_t aux_byte;
				//uint8_t field;
	                    	aux_byte=file->get_byte(); //packing type
                      		if (!(aux_byte&0x80)) {

                          		aux_note.note=aux_byte;
                        		aux_byte=0xFE; //if bit 7 not set, read all of them except the note
                          	}

                       		if (aux_byte&1) aux_note.note=file->get_byte();
                         	if (aux_byte&2) aux_note.instrument=file->get_byte();
                         	if (aux_byte&4) aux_note.volume=file->get_byte();
                         	if (aux_byte&8) aux_note.command=file->get_byte();
                         	if (aux_byte&16) aux_note.parameter=file->get_byte();

				if (aux_note.note!=CPNote::EMPTY) {

					if (aux_note.note==97) aux_note.note=CPNote::OFF;
                                  	else {
                                  		aux_note.note+=11; //octave minus one (XM C-0 is 1, not zero )
                                   	}
                             	}
				if (aux_note.instrument!=CPNote::EMPTY) {

                                 	if ((aux_note.instrument>0) && (aux_note.instrument<100))
                                  		 aux_note.instrument--;
                                     	else
						 aux_note.instrument=CPNote::EMPTY;
                             	}
				if (aux_note.volume!=CPNote::EMPTY) {

                                 	if (aux_note.volume<0x10) {}
                                  	else if (aux_note.volume<0x50) {

						aux_note.volume-=0x10;

                                     	} else if (aux_note.volume<0x60) {
                                                //
						aux_note.volume=CPNote::EMPTY;

                                     	} else if (aux_note.volume<0x70) {
                                                //60 -- volume slide down
						aux_note.volume-=0x60;
      						if (aux_note.volume>9) aux_note.volume=9;
						aux_note.volume+=95;
						
                                     	} else if (aux_note.volume<0x80) {
                                                //70 -- volume slide up
						aux_note.volume-=0x70;
      						if (aux_note.volume>9) aux_note.volume=9;
						aux_note.volume+=85;


                                     	} else if (aux_note.volume<0x90) {
                                                //80 -- fine volume slide down
						aux_note.volume-=0x80;
      						if (aux_note.volume>9) aux_note.volume=9;
						aux_note.volume+=75;


                                     	} else if (aux_note.volume<0xA0) {
                                                //9 -- fine volume slide up

						aux_note.volume-=0x90;
      						if (aux_note.volume>9) aux_note.volume=9;

						aux_note.volume+=65;



                                     	} else if (aux_note.volume<0xB0) {
                                                //A -- set vibrato speed
						aux_note.volume=CPNote::EMPTY;

                                     	} else if (aux_note.volume<0xC0) {
                                                //B -- vibrato
						aux_note.volume-=0xB0;
      						if (aux_note.volume>9) aux_note.volume=9;
						aux_note.volume+=203;


                                     	} else if (aux_note.volume<0xD0) {
                                                //C -- set panning
                                                int aux=aux_note.volume-=0xC0;
						aux=aux*65/0xF;
      						aux_note.volume=128+aux;

                                     	} else if (aux_note.volume<0xE0) {
						aux_note.volume=CPNote::EMPTY;


                                     	} else if (aux_note.volume<0xF0) {
						aux_note.volume=CPNote::EMPTY;


                                     	} else {
                                                //F -- tone porta
						aux_note.volume-=0xF0;
      						aux_note.volume*=9;
      						aux_note.volume/=0xF;
            					aux_note.volume+=193;
      					}
                             	}
				if (aux_note.command!=CPNote::EMPTY) {

                                   	switch(aux_note.command) {

                                                case 0x0:
        						aux_note.command='J'-'A';
              						break;		
                                                case 0x1:
        						aux_note.command='F'-'A';
              						break;
                                                case 0x2:
        						aux_note.command='E'-'A';
              						break;
                                                case 0x3:
        						aux_note.command='G'-'A';
              						break;
                                                case 0x4:
        						aux_note.command='H'-'A';
                                                        break;
                                                case 0x5:
        						aux_note.command='L'-'A';
                                                        break;
                                                case 0x6:
        						aux_note.command='K'-'A';
                                                        break;
                                                case 0x7:
        						aux_note.command='R'-'A';
                                                        break;
                                                case 0x8:
        						aux_note.command='X'-'A';
                                                        break;
                                                case 0x9:
        						aux_note.command='O'-'A';
                                                        break;
                                                case 0xa:
        						aux_note.command='D'-'A';
                                                        break;
                                                case 0xb:
        						aux_note.command='B'-'A';
                                                        break;
                                                case 0xc:
        						//printf("XM Import: Warning! effect C (set volume) not implemented!\n");
                                                        break;
                                                case 0xd:
        						aux_note.command='C'-'A';
                                                        break;

                                                case 0xe: /* Extended effects */

                                                	aux_note.command='S'-'A';
                                                        switch(aux_note.parameter>>4) {
                                                                case 0x1: /* XM fine porta up */
									if (!(aux_note.parameter&0xF)) { aux_note.command=CPNote::EMPTY; aux_note.parameter=0; break; }
                                                                	aux_note.command='F'-'A';
                                                            	        aux_note.parameter=0xF0|(aux_note.parameter&0xF);
                                                                        break;
                                                                case 0x2: /* XM fine porta down */
									if (!(aux_note.parameter&0xF)) { aux_note.command=CPNote::EMPTY; aux_note.parameter=0; break; }
                                                                	aux_note.command='E'-'A';
                                                            	        aux_note.parameter=0xF0|(aux_note.parameter&0xF);
                                                                        break;
                                                                case 0xa: /* XM fine volume up */
									if (!(aux_note.parameter&0xF)) { aux_note.command=CPNote::EMPTY; aux_note.parameter=0; break; }
                                                                	aux_note.command='D'-'A';
                                                            	        aux_note.parameter=0x0F|((aux_note.parameter&0xF)<<4);

                                                                        break;
                                                                case 0xb: /* XM fine volume down */
									if (!(aux_note.parameter&0xF)) { aux_note.command=CPNote::EMPTY; aux_note.parameter=0; break; }
                                                                	aux_note.command='D'-'A';
                                                            	        aux_note.parameter=0xF0|(aux_note.parameter&0xF);

                                                                        break;
                                                                case 0x9: /* XM fine volume down */
									if (!(aux_note.parameter&0xF)) { aux_note.command=CPNote::EMPTY; aux_note.parameter=0; break; }
                                                                	aux_note.command='Q'-'A';
                                                            	        aux_note.parameter=0x00|(aux_note.parameter&0xF);
                                                                        break;

                                                                case 0xc: //notecut
                                                                	
                                                            	        aux_note.parameter=0xC0|(aux_note.parameter&0xF);
                                                                        break;

                                                                case 0xd: //notedelay
                                                                	
                                                            	        aux_note.parameter=0xD0|(aux_note.parameter&0xF);
                                                                        break;

                                                                case 0xe: //patterndelay
                                                                	
                                                            	        aux_note.parameter=0xE0|(aux_note.parameter&0xF);
                                                                        break;
                                                        }

                                                        break;
                                                case 0xf:
                                                	if (aux_note.parameter<32) {
	        						aux_note.command='A'-'A';
               						} else {
	        						aux_note.command='T'-'A';
                       					}
                                                        break;
                                                case 'G'-55:
        						aux_note.command='V'-'A';
                                                        break;
                                                case 'H'-55:
        						aux_note.command='W'-'A';
                                                        break;
                                                case 'K'-55:
							if (aux_note.note!=CPNote::EMPTY) break;
							aux_note.note=CPNote::OFF;
                                                        break;
                                                case 'P'-55:
        						aux_note.command='P'-'A';
                                                        break;
                                                case 'R'-55:
        						aux_note.command='Q'-'A';
                                                        break;
              					case 'T'-55:
        						aux_note.command='I'-'A';
                                                        break;
              					default: {

							aux_note.command=CPNote::EMPTY;
                     				}
              				}


				}

				song->get_pattern( i)->set_note( k,j,aux_note );
		    }
	}

        /**************************************
        LOAD INSTRUMENTS!
        ***************************************/

        for (int i=0;i<header.instruments_used;i++) {


          	uint32_t aux;
           	int sampnum;
		
		CPInstrument &instrument=*song->get_instrument(i);
                uint32_t cpos=file->get_pos();
		//printf("pos is %i\n",cpos);



/* +4 */	uint32_t hsize=file->get_dword(); //header length

		char instrname[23];
		instrname[22]=0;
		
  		file->get_byte_array((uint8_t*)instrname,22);
//XM_LOAD_DEBUG printf("name is %s\n",instrname);

/* +27 */	aux=file->get_byte(); //byte that must be ignored
//XM_LOAD_DEBUG printf("header size is %i\n",hsize);

/* +29 */	sampnum=file->get_word(); 

//XM_LOAD_DEBUG printf("samples %i\n",sampnum);


		instrument.set_name( instrname );
//		printf("Header Len: %i, CPInstrument %i, %i samples , name:  s,\n",hsize,i,sampnum,instrname);
		
		if (sampnum==0) {
			//aux=file->get_dword(); //Why is this for? -- for nothing, skipped
			if (hsize) {
				
				file->seek( cpos+hsize ); //skip header if size has been specified
			}
			continue;
		}

/* +33 */      	file->get_dword(); 

		if (Error result=load_instrument_internal(&instrument,false,cpos,hsize,sampnum)) {

			CP_PRINTERR("Error loading instrument");
			file->close();
			return result;
		}

	}
// 
	file->close();
	return FILE_OK;
}

CPLoader::Error CPLoader_XM::load_instrument_internal(CPInstrument *p_instr,bool p_xi,int p_cpos, int p_hsize, int p_sampnum) {

		int sampnum;
		uint32_t aux;
		uint8_t notenumb[96];
		uint16_t panenv[24],volenv[24];
		int volpoints,panpoints;
		int vol_loop_begin,vol_loop_end,vol_sustain_loop;
		int pan_loop_begin,pan_loop_end,pan_sustain_loop;
		char instrname[23];
		int sample_index[16]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}; //-1 means no index!

		instrname[22]=0;


/* +129 */	file->get_byte_array((uint8_t*)notenumb,96);
		for (int j=0;j<24;j++) {
			volenv[j]=file->get_word();
		}
		for (int j=0;j<24;j++) {
			panenv[j]=file->get_word();
		}

/* +177 */
/* +225 */
/* +226 */	volpoints=file->get_byte();
/* +227 */	panpoints=file->get_byte();
/* +230 */	vol_sustain_loop=file->get_byte();
/* +228 */	vol_loop_begin=file->get_byte();
/* +229 */	vol_loop_end=file->get_byte();

//XM_LOAD_DEBUG 	printf("1- volpoints: %i, panpoints: %i, susloop: %i, loop begin: %i, loop end %i\n",volpoints,panpoints,vol_sustain_loop,vol_loop_begin,vol_loop_end);
		pan_sustain_loop=file->get_byte();
/* +231 */	pan_loop_begin=file->get_byte();
/* +232 */	pan_loop_end=file->get_byte();



/* +234 */	aux=file->get_byte();
		p_instr->get_volume_envelope()->reset();
		p_instr->get_volume_envelope()->set_enabled(aux&1);
		p_instr->get_volume_envelope()->set_sustain_loop_enabled((aux&2)?true:false);
		p_instr->get_volume_envelope()->set_loop_enabled((aux&4)?true:false);
/* +235 */	aux=file->get_byte();
		p_instr->get_pan_envelope()->reset();
		p_instr->get_pan_envelope()->set_enabled(aux&1);
		p_instr->get_pan_envelope()->set_sustain_loop_enabled((aux&2)?true:false);
		p_instr->get_pan_envelope()->set_loop_enabled((aux&4)?true:false);

/* +239 */	aux=file->get_dword(); // sadly, cant use those
/* +241 */     	p_instr->set_volume_fadeout( file->get_word() >> 4 );
/* +243 */	aux=file->get_word(); // reserved!



		for (int j=0;j<volpoints;j++) {
			int ofs=volenv[j*2];
			int val=volenv[j*2+1];
			p_instr->get_volume_envelope()->add_position(ofs,val);

		}

		//make sure minimum is 2
		while (p_instr->get_volume_envelope()->get_node_count()<2) {
			
			p_instr->get_volume_envelope()->add_position( p_instr->get_volume_envelope()->get_node_count()*20,64 );
		}
		
		for (int j=0;j<panpoints;j++) {
			int ofs=panenv[j*2];
			int val=panenv[j*2+1];
			p_instr->get_pan_envelope()->add_position(ofs,val-32);
		}
		
		//make sure minimum is 2
		while (p_instr->get_pan_envelope()->get_node_count()<2) {
			
			p_instr->get_pan_envelope()->add_position( p_instr->get_pan_envelope()->get_node_count()*20,0 );
		}
		
		
		p_instr->get_volume_envelope()->set_loop_begin(vol_loop_begin);
		p_instr->get_volume_envelope()->set_loop_end(vol_loop_end);
		p_instr->get_volume_envelope()->set_sustain_loop_end(vol_sustain_loop);
		p_instr->get_volume_envelope()->set_sustain_loop_begin(vol_sustain_loop);
		p_instr->get_pan_envelope()->set_loop_begin(pan_loop_begin);
		p_instr->get_pan_envelope()->set_loop_end(pan_loop_end);
		p_instr->get_pan_envelope()->set_sustain_loop_end(pan_sustain_loop);
		p_instr->get_pan_envelope()->set_sustain_loop_begin(pan_sustain_loop);


		if (!p_xi) {

			if ((file->get_pos()-p_cpos)<p_hsize) {

				uint8_t junkbuster[500];

				//printf("extra junk XM instrument in header! hsize is %i, extra junk: %i\n",p_hsize,(file->get_pos()-p_cpos));
				//printf("extra: %i\n",p_hsize-(file->get_pos()-p_cpos));
				file->get_byte_array((uint8_t*)junkbuster,p_hsize-(file->get_pos()-p_cpos));
			}
			
			sampnum=p_sampnum;
		} else {

			uint8_t junkbuster[500];
			file->get_byte_array((uint8_t*)junkbuster,20);	//14 bytes?

			sampnum=file->get_word();

		}


		CPSampleManager *sm=CPSampleManager::get_singleton();

		/*SAMPLE!!*/

		for (int j=0;j<sampnum;j++) {

			if (j>16) ABORT_LOAD;

			
			int s_idx=-1;
			for (int s=0;s<CPSong::MAX_SAMPLES;s++) {
				
				if (song->get_sample(s)->get_sample_data().is_null()) {
					//empty sample!
					s_idx=s;
					break;
				}
			}
				
			if (s_idx==-1) ABORT_LOAD;
			//printf("free sample: %i\n",s_idx);
			

			CPSample& sample=*song->get_sample(s_idx);
			
			int sample_size=file->get_dword();
			int tmp_loop_begin=file->get_dword();

			int tmp_loop_end=file->get_dword();
			
			sample.set_default_volume(file->get_byte());

			uint8_t ftb=file->get_byte();
			int8_t *fts=(int8_t*)&ftb;
			int finetune=*fts;
			uint32_t flags=file->get_byte();
			
			if (flags&16) { // is 16 bits.. at flag 16.. fun :)
				
				tmp_loop_end/=2;
				tmp_loop_begin/=2;
				sample_size/=2;
			}
			
			
			CPSample_ID sample_data=sm->create( flags&16, false, sample_size );
			
			sample.set_sample_data(sample_data);
			sm->set_loop_begin(sample_data,tmp_loop_begin);
			sm->set_loop_end(sample_data,tmp_loop_end+tmp_loop_begin);
			
			sm->set_loop_type( sample_data, (flags&3)?( (flags&2) ? CP_LOOP_BIDI : CP_LOOP_FORWARD ):CP_LOOP_NONE );
			
			

			sample.set_pan_enabled(true);
			sample.set_pan(file->get_byte()*64/255);
			uint8_t noteb=file->get_byte();
			int8_t *notes=(int8_t*)&noteb;
			int note_offset=*notes;
			note_offset+=48;
			//note_offset+=60;
			
			
			
			//int linear_period=10*12*16*4 - (note_offset)*16*4 - finetune/2;
			//int freq=(int)(8363*pow(2.0,(double)(6*12*16*4 - linear_period) / (double)(12*16*4)));
			
			//sm->set_c5_freq( sample_data, freq);
			sm->set_c5_freq( sample_data, CPTables::get_linear_frequency(CPTables::get_linear_period(note_offset<<1,finetune)) );
			//printf("NOTE %i,fine %i\n",note_offset,finetune);

			char auxb;
			auxb=file->get_byte(); //reserved?
			file->get_byte_array((uint8_t*)instrname,22);
			sample.set_name(instrname);

			sample_index[j]=s_idx;
		}

		/*SAMPLE __DATA__!!*/

		for (int j=0;j<sampnum;j++) {

			if (sample_index[j]==-1) continue;

			CPSample *sample=song->get_sample(sample_index[j]);
			CPSample_ID sid=sample->get_sample_data();

			sm->lock_data(sid);

			void*dataptr=sm->get_data(sid);

			if (sm->is_16bits( sid)) {

				int16_t old=0;


				for (int k=0;k<sm->get_size(sid);k++) {

					int16_t newsample;
					int16_t sampleval=file->get_word();
					newsample=sampleval+old;
					old=newsample;

					((int16_t*)dataptr)[k]=newsample;
					//sm->set_data( sid, k, newsample );
				}
			} else {

				int8_t old=0;


				for (int k=0;k<sm->get_size(sid);k++) {

					int8_t newsample;
					int8_t sampleval=file->get_byte();
					newsample=sampleval+old;
					old=newsample;
					
					((int8_t*)dataptr)[k]=newsample;

					//sm->set_data( sid, k, (int16_t)newsample << 8 );
					
				}
			}

			sm->unlock_data(sid);

		}

		for (int j=0;j<96;j++) {

			int val=notenumb[j];
			if ((val<0) || (val>15)) continue;
			else val=sample_index[val];
			if (val==-1) continue;
			p_instr->set_sample_number( 12+j,val );
		}


	return FILE_OK;
}



CPLoader::Error CPLoader_XM::load_sample(const char *p_file,CPSample *p_sample) {
	
	return FILE_UNRECOGNIZED;
}


/* Compute CPInstrument Info */
CPLoader::Error CPLoader_XM::load_instrument(const char *p_file,CPSong *p_song,int p_instr_idx) {

	if ( file->open(p_file,CPFileAccessWrapper::READ) ) return FILE_CANNOT_OPEN;	
	//int i;
	song=p_song;
	CPInstrument& instr=*p_song->get_instrument( p_instr_idx );
  	int aux;


        char buffer[500];
        file->get_byte_array((uint8_t*)buffer,0x15);
        buffer[8]=0;
	if (	buffer[0]!='E' ||
		buffer[1]!='x' ||
		buffer[2]!='t' ||
		buffer[3]!='e' ||
		buffer[4]!='n' ||
		buffer[5]!='d' ||
		buffer[6]!='e' ||
		       buffer[7]!='d') {
		file->close();
		return FILE_UNRECOGNIZED;
        }

        file->get_byte_array((uint8_t*)buffer,0x16);
        buffer[0x16]=0;
	instr.set_name(buffer);
	aux=file->get_byte(); //says ignore ti
	/*if(aux!=0x1a) { I'm not sure. this is supposed to be ignored...

		file->close();
		return FILE_UNRECOGNIZED;
	} */

        file->get_byte_array((uint8_t*)buffer,0x14); //somethingaboutthename
        aux=file->get_word(); //version or blahblah
 	
  	if (load_instrument_internal(&instr,true,0,0)) {

		file->close();
		return FILE_CORRUPTED;
	}

	file->close(); //ook, we got it..


	return FILE_OK;

}



CPLoader_XM::CPLoader_XM(CPFileAccessWrapper *p_file){
	
	file=p_file;
}
CPLoader_XM::~CPLoader_XM(){
}

