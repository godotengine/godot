/*************************************************************************/
/*  cp_player_data_effects.cpp                                           */
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

#include "cp_player_data.h"



/**********************
   complex effects
***********************/
#define RANDOM_MAX 2147483647

static inline int32_t cp_random_generate(int32_t *seed) {
	int32_t k;
	int32_t s = (int32_t)(*seed);
	if (s == 0)
		s = 0x12345987;
	k = s / 127773;
	s = 16807 * (s - k * 127773) - 2836 * k;
	if (s < 0)
		s += 2147483647;
	(*seed) = (int32_t)s;
	return (int32_t)(s & RANDOM_MAX);
}



void CPPlayer::do_vibrato(int p_track,bool fine) {

        uint8_t q;
        uint16_t temp=0;

	if ((control.ticks_counter==0) && control.channel[p_track].row_has_note) control.channel[p_track].vibrato_position=0;

        q=(control.channel[p_track].vibrato_position>>2)&0x1f;

        switch (control.channel[p_track].vibrato_type) {
                case 0: /* sine */
                        temp=vibrato_table[q];
                        break;
                case 1: /* square wave */
                	temp=255;
                        break;
                case 2: /* ramp down */
                        q<<=3;
                        if (control.channel[p_track].vibrato_position<0) q=255-q;
                        temp=q;
                        break;
                case 3: /* random */
			temp=cp_random_generate(&control.random_seed) %256;//getrandom(256);
                        break;
        }

        temp*=control.channel[p_track].vibrato_depth;

	if (song->has_old_effects()) {

	        temp>>=7;
	} else {

	        temp>>=8;
	}

	if (!fine) temp<<=2;

        if (control.channel[p_track].vibrato_position>=0) {

                control.channel[p_track].period=control.channel[p_track].aux_period+temp;
        } else {
	
                control.channel[p_track].period=control.channel[p_track].aux_period-temp;
	}

        if (!song->has_old_effects() || control.ticks_counter) control.channel[p_track].vibrato_position+=control.channel[p_track].vibrato_speed;
}


void CPPlayer::do_pitch_slide_down(int p_track,uint8_t inf) {

	uint8_t hi,lo;

        if (inf) control.channel[p_track].pitch_slide_info=inf;
        else inf=control.channel[p_track].pitch_slide_info;

        hi=inf>>4;
        lo=inf&0xf;
		
        if (hi==0xf) {

                if (!control.ticks_counter) control.channel[p_track].aux_period+=(uint16_t)lo<<2;
        } else if (hi==0xe) {

                if (!control.ticks_counter) control.channel[p_track].aux_period+=lo;
        } else {

                if (control.ticks_counter) control.channel[p_track].aux_period+=(uint16_t)inf<<2;
        }
}

void CPPlayer::do_pitch_slide_up(int p_track,uint8_t inf) {

	uint8_t hi,lo;

        if (inf) control.channel[p_track].pitch_slide_info=inf;
        else inf=control.channel[p_track].pitch_slide_info;

        hi=inf>>4;
        lo=inf&0xf;
		
        if (hi==0xf) {

                if (!control.ticks_counter) control.channel[p_track].aux_period-=(uint16_t)lo<<2;
        } else if (hi==0xe) {

                if (!control.ticks_counter) control.channel[p_track].aux_period-=lo;
        } else {

                if (control.ticks_counter) control.channel[p_track].aux_period-=(uint16_t)inf<<2;
        }
}

void CPPlayer::do_pitch_slide_to_note(int p_track) {

        if (control.ticks_counter) {
                int dist;

                /* We have to slide a->period towards a->wantedperiod, compute the
                   difference between those two values */
                dist=control.channel[p_track].period-control.channel[p_track].slide_to_period;

            /* if they are equal or if portamentospeed is too big... */
		if ((!dist)||((control.channel[p_track].portamento_speed<<2)>cp_intabs(dist))) {
                        /* ... make tmpperiod equal tperiod */
                        control.channel[p_track].aux_period=control.channel[p_track].period=control.channel[p_track].slide_to_period;
                } else {

			if (dist>0) {

                        	control.channel[p_track].aux_period-=control.channel[p_track].portamento_speed<<2;
                        	control.channel[p_track].period-=control.channel[p_track].portamento_speed<<2; /* dist>0 slide up */
	                } else {
        	                control.channel[p_track].aux_period+=control.channel[p_track].portamento_speed<<2;
                	        control.channel[p_track].period+=control.channel[p_track].portamento_speed<<2; /* dist<0 slide down */
			}
		}

        } else {

                control.channel[p_track].aux_period=control.channel[p_track].period;
	}
}

void CPPlayer::do_tremor(int p_track) {

        uint8_t on,off,inf;

	inf=control.channel[p_track].current_parameter;

        if (inf) {
                control.channel[p_track].tremor_info=inf;
        } else {
                inf= control.channel[p_track].tremor_info;
                if (!inf) return;
        }

        //if (!control.ticks_counter) return;

        on=(inf>>4);
        off=(inf&0xf);

        control.channel[p_track].tremor_position%=(on+off);
        control.channel[p_track].volume=(control.channel[p_track].tremor_position<on)?control.channel[p_track].aux_volume:0;
        control.channel[p_track].tremor_position++;
}

void CPPlayer::do_pan_slide(int p_track) {

        uint8_t lo,hi,inf;
        int16_t pan;

	inf=control.channel[p_track].current_parameter;

        if (inf) control.channel[p_track].channel_pan_slide_info=inf;
        else inf=control.channel[p_track].channel_pan_slide_info;

        lo=inf&0xf;
        hi=inf>>4;

        pan=(control.channel[p_track].panning==PAN_SURROUND)?PAN_CENTER:control.channel[p_track].panning;

        if (!hi)
                pan+=lo<<2;
        else
          if (!lo) {
                pan-=hi<<2;
        } else
          if (hi==0xf) {
                if (!control.ticks_counter) pan+=lo<<2;
        } else
          if (lo==0xf) {
                if (!control.ticks_counter) pan-=hi<<2;
        }
	//this sets both chan & voice paning
        control.channel[p_track].panning=(pan<PAN_LEFT)?PAN_LEFT:(pan>PAN_RIGHT?PAN_RIGHT:pan);
	control.channel[p_track].channel_panning=control.channel[p_track].panning;
}

void CPPlayer::do_volume_slide(int p_track,int inf) {

	uint8_t hi,lo;

	lo=inf&0xf;
	hi=inf>>4;
			
	if (!lo) {

		if ((control.ticks_counter)) control.channel[p_track].aux_volume+=hi;

	} else if (!hi) {

		if ((control.ticks_counter)) control.channel[p_track].aux_volume-=lo;

	} else if (lo==0xf) {

		if (!control.ticks_counter) control.channel[p_track].aux_volume+=(hi?hi:0xf);
	} else if (hi==0xf) {

		if (!control.ticks_counter) control.channel[p_track].aux_volume-=(lo?lo:0xf);
	} else return;

	if (control.channel[p_track].aux_volume<0) {

		control.channel[p_track].aux_volume=0;
	} else if (control.channel[p_track].aux_volume>64) {

		control.channel[p_track].aux_volume=64;
	}
}

void CPPlayer::do_channel_volume_slide(int p_track) {

        uint8_t lo, hi,inf;

	inf=control.channel[p_track].current_parameter;

        if (inf) control.channel[p_track].channel_volume_slide_info=inf;
        inf=control.channel[p_track].channel_volume_slide_info;

        lo=inf&0xf;
        hi=inf>>4;

        if (!hi)
                control.channel[p_track].channel_volume-=lo;
        else
          if (!lo) {
                control.channel[p_track].channel_volume+=hi;
        } else
          if (hi==0xf) {
                if (!control.ticks_counter) control.channel[p_track].channel_volume-=lo;
        } else
          if (lo==0xf) {
                if (!control.ticks_counter) control.channel[p_track].channel_volume+=hi;
        }

        if (control.channel[p_track].channel_volume<0) control.channel[p_track].channel_volume=0;
        if (control.channel[p_track].channel_volume>64) control.channel[p_track].channel_volume=64;
}

void CPPlayer::do_tremolo(int p_track) {

        uint8_t q;
        int16_t temp=0;

	if ((control.ticks_counter==0) && control.channel[p_track].row_has_note) control.channel[p_track].tremolo_position=0;

        q=(control.channel[p_track].tremolo_position>>2)&0x1f;

        switch (control.channel[p_track].tremolo_type) {
                case 0: /* sine */
                        temp=vibrato_table[q];
                        break;
                case 1: /* ramp down */
                        q<<=3;
                        if (control.channel[p_track].tremolo_position<0) q=255-q;
                        temp=q;
                        break;
                case 2: /* square wave */
                        temp=255;
                        break;
                case 3: /* random */
			temp=cp_random_generate(&control.random_seed) % 256;//getrandom(256);
                        break;
        }

        temp*=control.channel[p_track].tremolo_depth;
        temp>>=7;



        if (control.channel[p_track].tremolo_position>=0) {


                control.channel[p_track].volume=control.channel[p_track].aux_volume+temp;
                if (control.channel[p_track].volume>64) control.channel[p_track].volume=64;
        } else {

		control.channel[p_track].volume=control.channel[p_track].aux_volume-temp;
                if (control.channel[p_track].volume<0) control.channel[p_track].volume=0;
        }

        /*if (control.ticks_counter)*/ control.channel[p_track].tremolo_position+=control.channel[p_track].tremolo_speed;

}

void CPPlayer::do_arpegio(int p_track) {

        uint8_t note,dat;
	//note=control.channel[p_track].note;
	note=0;

	if (control.channel[p_track].current_parameter) {

		control.channel[p_track].arpegio_info=control.channel[p_track].current_parameter;
	}

	dat=control.channel[p_track].arpegio_info;
				
	if (dat) {

		switch (control.ticks_counter%3) {
			
			case 1: {

	                        note+=(dat>>4);

			} break;
			case 2: {
	
                                note+=(dat&0xf);
			} break;
		}

		if (song->has_linear_slides()) {

			control.channel[p_track].period=control.channel[p_track].aux_period-cp_intabs(get_period((uint16_t)46,0)-get_period((uint16_t)44,0))*note;
		} else if (control.channel[p_track].sample_ptr) {
			
			control.channel[p_track].period=get_period( (((uint16_t)control.channel[p_track].note)+note)<<1,CPSampleManager::get_singleton()->get_c5_freq( (control.channel[p_track].sample_ptr->get_sample_data())));
                }
		
        	control.channel[p_track].has_own_period=true;
        }


}


void CPPlayer::do_retrig(int p_track) {

	uint8_t inf;

	inf=control.channel[p_track].current_parameter;

        if (inf) {

       		control.channel[p_track].retrig_volslide=inf>>4;
                control.channel[p_track].retrig_speed=inf&0xf;
        }

        /* only retrigger if low nibble > 0 */
        if ( control.channel[p_track].retrig_speed>0) {

                if ( !control.channel[p_track].retrig_counter ) {
                        /* when retrig counter reaches 0, reset counter and restart the
                           sample */
                        if (control.channel[p_track].kick!=KICK_NOTE) control.channel[p_track].kick=KICK_NOTEOFF;
			control.channel[p_track].retrig_counter=control.channel[p_track].retrig_speed;


                        if ((control.ticks_counter)/*||(pf->flags&UF_S3MSLIDES)*/) {
                                switch (control.channel[p_track].retrig_volslide) {
                                        case 1:
                                        case 2:
                                        case 3:
                                        case 4:
                                        case 5:
                                                control.channel[p_track].aux_volume-=(1<<(control.channel[p_track].retrig_volslide-1));
                                                break;
                                        case 6:
                                                control.channel[p_track].aux_volume=(2*control.channel[p_track].aux_volume)/3;
                                                break;
                                        case 7:
                                                control.channel[p_track].aux_volume>>=1;
                                                break;
                                        case 9:
                                        case 0xa:
                                        case 0xb:
                                        case 0xc:
                                        case 0xd:
                                                control.channel[p_track].aux_volume+=(1<<(control.channel[p_track].retrig_volslide-9));
                                                break;
                                        case 0xe:
                                                control.channel[p_track].aux_volume=(3*control.channel[p_track].aux_volume)>>1;
                                                break;
                                        case 0xf:
                                                control.channel[p_track].aux_volume=control.channel[p_track].aux_volume<<1;
                                                break;
                                }
                                if (control.channel[p_track].aux_volume<0) control.channel[p_track].aux_volume=0;
                                else if (control.channel[p_track].aux_volume>64) control.channel[p_track].aux_volume=64;
                        }
                }
                control.channel[p_track].retrig_counter--; /* countdown  */
        }
}

void CPPlayer::do_global_volume_slide(int p_track) {

        uint8_t lo,hi,inf;

	inf=control.channel[p_track].current_parameter;

        if (inf) control.channel[p_track].global_volume_slide_info=inf;
        inf=control.channel[p_track].global_volume_slide_info;

        lo=inf&0xf;
        hi=inf>>4;

        if (!lo) {
                if (control.ticks_counter) control.global_volume+=hi;
        } else
          if (!hi) {
                if (control.ticks_counter) control.global_volume-=lo;
        } else
          if (lo==0xf) {
                if (!control.ticks_counter) control.global_volume+=hi;
        } else
          if (hi==0xf) {
                if (!control.ticks_counter) control.global_volume-=lo;
        }

        if (control.global_volume<0) control.global_volume=0;
        if (control.global_volume>128) control.global_volume=128;
}

void CPPlayer::do_panbrello(int p_track) {

        uint8_t q;
        int32_t temp=0;

        q=control.channel[p_track].panbrello_position;

        switch (control.channel[p_track].panbrello_type) {
                case 0: {/* sine */
                        temp=panbrello_table[q];
                } break;
                case 1: {/* square wave */
                        temp=(q<0x80)?64:0;
                } break;
                case 2: {/* ramp down */
                        q<<=3;
                        temp=q;
                } break;
                case 3: {/* random */
                        if (control.channel[p_track].panbrello_position>=control.channel[p_track].panbrello_speed) {
                                control.channel[p_track].panbrello_position=0;
				temp=cp_random_generate(&control.random_seed)%256;//getrandom(256);
			}
                } break;
        }


	
        temp=temp*(int)control.channel[p_track].panbrello_depth/0xF;
        temp<<=1;
	if (control.channel[p_track].channel_panning!=PAN_SURROUND)
		temp+=control.channel[p_track].channel_panning;

        control.channel[p_track].panning=(temp<PAN_LEFT)?PAN_LEFT:(temp>PAN_RIGHT?PAN_RIGHT:temp);
        control.channel[p_track].panbrello_position+=control.channel[p_track].panbrello_speed;
}

/******************
      S effect
*******************/


void CPPlayer::do_effect_S(int p_track) {

        uint8_t inf,c,dat;

	dat=control.channel[p_track].current_parameter;
	
        inf=dat&0xf;
        c=dat>>4;

        if (!dat) {
                c=control.channel[p_track].current_S_effect;
                inf=control.channel[p_track].current_S_data;
        } else {
                control.channel[p_track].current_S_effect=c;
                control.channel[p_track].current_S_data=inf;
        }

        switch (c) {
                case 1: {/* S1x set glissando voice */
		// this is unsupported in IT!

  			control.channel[p_track].chorus_send=inf*0xFF/0xF;
  			
                }break;
                case 2: /* S2x set finetune */
		//Also not supported!	
                        break;
                case 3: /* S3x set vibrato waveform */
			if (inf<4) control.channel[p_track].vibrato_type=inf;
                        break;
                case 4: /* S4x set tremolo waveform */
			if (inf<4) control.channel[p_track].tremolo_type=inf;
                        break;
                case 5: /* S5x panbrello */
			if (inf<4) control.channel[p_track].panbrello_type=inf;
                        break;
                case 6: {/* S6x delay x number of frames (patdly) */

			if (control.ticks_counter) break;
                        if (!control.pattern_delay_2) control.pattern_delay_1=inf+1; /* only once, when vbtick=0 */

                } break;
                case 7: /* S7x instrument / NNA commands */
			
			if (!song->has_instruments())
				break;
			switch(inf) {
				
				case 0x3: {
					
					control.channel[p_track].NNA_type=CPInstrument::NNA_NOTE_CUT;
				} break;
				case 0x4:  {
					
					control.channel[p_track].NNA_type=CPInstrument::NNA_NOTE_CONTINUE;
				} break;
				case 0x5:  {
					
					control.channel[p_track].NNA_type=CPInstrument::NNA_NOTE_OFF;
				} break;
				case 0x6:  {
					
					control.channel[p_track].NNA_type=CPInstrument::NNA_NOTE_FADE;
				} break;
				case 0x7:  {
					
					if (control.channel[p_track].slave_voice)
						control.channel[p_track].slave_voice->volume_envelope_ctrl.active=false;
				} break;
				case 0x8:  {
					
					if (control.channel[p_track].slave_voice)
						control.channel[p_track].slave_voice->volume_envelope_ctrl.active=true;
					
				} break;
				case 0x9:  {
					
					if (control.channel[p_track].slave_voice)
						control.channel[p_track].slave_voice->panning_envelope_ctrl.active=false;
					
				} break;
				case 0xA: {
					
					if (control.channel[p_track].slave_voice)
						control.channel[p_track].slave_voice->panning_envelope_ctrl.active=true;
					
				} break;
				case 0xB: {
					if (control.channel[p_track].slave_voice)
						control.channel[p_track].slave_voice->pitch_envelope_ctrl.active=false;
					
				} break;
				case 0xC: {
					
					if (control.channel[p_track].slave_voice)
						control.channel[p_track].slave_voice->pitch_envelope_ctrl.active=true;
					
				} break;
				
			} break;
				
                        break;
                case 8: {/* S8x set panning position */

//			if (pf->panflag) {
                                if (inf<=8) inf<<=4;
                                else inf*=17;
                                control.channel[p_track].panning=control.channel[p_track].channel_panning=inf;
//                        }
                } break;

                case 9: { /* S9x set surround sound */
                        //if (pf->panflag)
                                control.channel[p_track].panning=control.channel[p_track].channel_panning=PAN_SURROUND;
                } break;
                case 0xA:{ /* SAy set high order sample offset yxx00h */

				if (control.channel[p_track].current_parameter) control.channel[p_track].hi_offset=(int32_t)inf<<16;
				control.channel[p_track].sample_start_index=control.channel[p_track].hi_offset|control.channel[p_track].lo_offset;
                } break;
                case 0xB: { /* SBx pattern loop */
                        if (control.ticks_counter) break;

                        if (inf) { /* set reppos or repcnt ? */
                                /* set repcnt, so check if repcnt already is set, which means we
                                   are already looping */
                                if (control.channel[p_track].pattern_loop_count>0)
                                        control.channel[p_track].pattern_loop_count--; /* already looping, decrease counter */
                                else {
                                        control.channel[p_track].pattern_loop_count=inf; /* not yet looping, so set repcnt */
                                }

                                if (control.channel[p_track].pattern_loop_count>0) { /* jump to reppos if repcnt>0 */

					control.position=control.previous_position; // This will also anulate any Cxx or break..

					control.position.current_row=control.channel[p_track].pattern_loop_position;
					control.position.forbid_jump=true;
				}

                        } else  {


				control.channel[p_track].pattern_loop_position=control.position.current_row-1;
			}

                } break;
                case 0xC: { /* SCx notecut */

			if (control.ticks_counter>=inf) {

				control.channel[p_track].aux_volume=0;
				control.channel[p_track].note_end_flags|=END_NOTE_OFF;
				control.channel[p_track].note_end_flags|=END_NOTE_KILL;
			}
                } break;
                case 0xD: {/* SDx notedelay */

			if (!control.ticks_counter) {

                                control.channel[p_track].note_delay=inf;

                        } else if (control.channel[p_track].note_delay) {

                                control.channel[p_track].note_delay--;
                        }
			
                } break;
                case 0xF: {/* SEx patterndelay */

			if (control.ticks_counter) break;
                        if (!control.pattern_delay_2) control.pattern_delay_1=inf+1; /* only once, when vbtick=0 */

                } break;
        }
}








/*********************
    volume effects
**********************/

void CPPlayer::run_volume_column_effects(int p_track) {

	uint8_t param=control.channel[p_track].current_volume_parameter;


	switch ('A'+control.channel[p_track].current_volume_command) {

		case 'A': {

			if (param>0) control.channel[p_track].volcol_volume_slide=param;
			else param=control.channel[p_track].volcol_volume_slide;
			
   			do_volume_slide(p_track,param*0x10+0xF);

		} break;
		case 'B': {

			if (param>0) control.channel[p_track].volcol_volume_slide=param;
			else param=control.channel[p_track].volcol_volume_slide;

			do_volume_slide(p_track,0xF0+param);

		} break;
		case 'C': {

			if (param>0) control.channel[p_track].volcol_volume_slide=param;
			else param=control.channel[p_track].volcol_volume_slide;

			do_volume_slide(p_track,param*0x10);
		} break;
		case 'D': {

			if (param>0) control.channel[p_track].volcol_volume_slide=param;
			else param=control.channel[p_track].volcol_volume_slide;
			do_volume_slide(p_track,param);

		} break;
		case 'E': {

			do_pitch_slide_down(p_track,param<<2);
		} break;
		case 'F': {

			do_pitch_slide_up(p_track,param<<2);
		} break;
		case 'G': {
		
             	        const uint8_t slide_table[]={0,1,4,8,16,32,64,96,128,255};
			if (param) {

				control.channel[p_track].portamento_speed=slide_table[param];
			}

			if (control.channel[p_track].period && (control.channel[p_track].old_note<=120)) {

				if ( (!control.ticks_counter) && (control.channel[p_track].new_instrument) ){
				
					//control.channel[p_track].kick=KICK_NOTE;
					//control.channel[p_track].sample_start_index=0; // < am i stupid?
				} else {

					control.channel[p_track].kick=(control.channel[p_track].kick==KICK_NOTE)?KICK_ENVELOPE:KICK_NOTHING;
					do_pitch_slide_to_note(p_track);
					control.channel[p_track].has_own_period=true;
				}

			}
		} break;
		case 'H': {


			if (!control.ticks_counter) {
				if (param&0x0f) control.channel[p_track].vibrato_depth=param;
			}
			control.channel[p_track].doing_vibrato=true;			
                        if (control.external_vibrato) break;			
			if (control.channel[p_track].period) {

				do_vibrato(p_track,false);
				control.channel[p_track].has_own_period=true;
			}                                		

		} break;
	}
}
/*********************
        table
**********************/


void CPPlayer::run_effects(int p_track) {

	switch ('A'+control.channel[p_track].current_command) {

		case 'A': {

			if ((control.ticks_counter>0) || (control.pattern_delay_2>0)) break;

                        int new_speed;

			new_speed=control.channel[p_track].current_parameter % 128;

			if (new_speed>0) {
				control.speed=new_speed;
				control.ticks_counter=0;
	        	}
		} break;
		case 'B': {

       			int next_order;

                        if (control.ticks_counter || control.position.forbid_jump) break;

			control.position.current_row=0;

			if (control.play_mode==PLAY_PATTERN) break;

			next_order=get_song_next_order_idx(song, (int)control.channel[p_track].current_parameter-1);

       			if (next_order!=-1) {
       				// Do we have a "next order?"
       				control.position.current_pattern=song->get_order(next_order);
       				control.position.force_next_order=next_order;
					
       			} else {
       				// no, probably the user deleted the orderlist.
       				control.play_mode=PLAY_NOTHING;
       				reset();
       			}
		} break;
		case 'C': {

			int next_order;

                        if (control.ticks_counter || control.position.forbid_jump) break;

			control.position.current_row=control.channel[p_track].current_parameter;

			if (control.play_mode==PLAY_PATTERN) {

				if (control.position.current_row>=song->get_pattern(control.position.current_pattern)->get_length()) {

					control.position.current_row=0;
				}

				break;
			}

			next_order=get_song_next_order_idx(song, (int)control.position.current_order);

       			if (next_order!=-1) {
       				// Do we have a "next order?"
       				control.position.current_pattern=song->get_order(next_order);

				if (control.position.current_row>=song->get_pattern(song->get_order(next_order))->get_length()) {

					control.position.current_row=0;
				}

       				control.position.force_next_order=next_order;
					
       			} else {
       				// no, probably the user deleted the orderlist.
       				control.play_mode=PLAY_NOTHING;
       				reset();
       			}

		} break;
		case 'D': {
		
			uint8_t inf ;
			//explicitslides=1;
			inf=control.channel[p_track].current_parameter;

			if (inf) control.channel[p_track].volume_slide_info=inf;
			else inf=control.channel[p_track].volume_slide_info;
	
			do_volume_slide(p_track,inf);

		} break;
		case 'E': {
		
		        uint8_t inf;
			
			inf=control.channel[p_track].current_parameter;
			do_pitch_slide_down(p_track,inf);

		} break;
		case 'F': {
		
		        uint8_t inf;
			
			inf=control.channel[p_track].current_parameter;
			do_pitch_slide_up(p_track,inf);

		} break;
		case 'G': {

			if (control.channel[p_track].current_parameter) {

				control.channel[p_track].portamento_speed=control.channel[p_track].current_parameter;
			}

			if (control.channel[p_track].period && (control.channel[p_track].old_note<=120)) {

				if ( (!control.ticks_counter) && (control.channel[p_track].new_instrument) ){
				

					control.channel[p_track].kick=KICK_NOTE;
					control.channel[p_track].sample_start_index=0;

				} else {

					control.channel[p_track].kick=(control.channel[p_track].kick==KICK_NOTE)?KICK_ENVELOPE:KICK_NOTHING;
				}

				do_pitch_slide_to_note(p_track);
				control.channel[p_track].has_own_period=true;
			}

		} break;
      		case 'H': {

			uint8_t dat;
			
			control.channel[p_track].doing_vibrato=true;

			dat=control.channel[p_track].current_parameter;

			if (!control.ticks_counter) {
				if (dat&0x0f) control.channel[p_track].vibrato_depth=dat&0xf;
				if (dat&0xf0) control.channel[p_track].vibrato_speed=(dat&0xf0)>>2;
			}

                        if (control.external_vibrato) break;			
			
			if (control.channel[p_track].period) {

				do_vibrato(p_track,false);
				control.channel[p_track].has_own_period=true;
			}                                		

		} break;
		case 'I': {

                        do_tremor(p_track);
			control.channel[p_track].has_own_volume=true;
		} break;
		case 'J': {

			do_arpegio(p_track);
		} break;
		case 'K': {

			uint8_t inf ;
			//explicitslides=1;
			inf=control.channel[p_track].current_parameter;
			
			control.channel[p_track].doing_vibrato=true;


			if (inf) control.channel[p_track].volume_slide_info=inf;
			else inf=control.channel[p_track].volume_slide_info;
	
			do_volume_slide(p_track,inf);

                        if (control.external_vibrato) break;						
			
			if (control.channel[p_track].period) {

				do_vibrato(p_track,false);
				control.channel[p_track].has_own_period=true;
			}                                		

		} break;
		case 'L': {
			uint8_t inf ;
			//explicitslides=1;
			inf=control.channel[p_track].current_parameter;

			if (inf) control.channel[p_track].volume_slide_info=inf;
			else inf=control.channel[p_track].volume_slide_info;
	
			do_volume_slide(p_track,inf);

			if (control.channel[p_track].period && (control.channel[p_track].old_note<=120)) {
				if ( (!control.ticks_counter) && (control.channel[p_track].new_instrument) ){
				
					control.channel[p_track].kick=KICK_NOTE;
					control.channel[p_track].sample_start_index=0;

				} else {

					control.channel[p_track].kick=(control.channel[p_track].kick==KICK_NOTE)?KICK_ENVELOPE:KICK_NOTHING;
				}

				do_pitch_slide_to_note(p_track);
				control.channel[p_track].has_own_period=true;
			}
		} break;
		case 'M': {
		                control.channel[p_track].channel_volume=control.channel[p_track].current_parameter;
                                if (control.channel[p_track].channel_volume>64) control.channel[p_track].channel_volume=64;
                                else if (control.channel[p_track].channel_volume<0) control.channel[p_track].channel_volume=0;
		} break;
		case 'N': {

			do_channel_volume_slide(p_track);
		}
		case 'O': {

			if (!control.ticks_counter) {
			
				if (control.channel[p_track].current_parameter) control.channel[p_track].lo_offset=(uint16_t)control.channel[p_track].current_parameter<<8;
				control.channel[p_track].sample_start_index=control.channel[p_track].hi_offset|control.channel[p_track].lo_offset;

				//if ((control.channel[p_track].sample_ptr!=NULL)&&(control.channel[p_track].sample_start_index>control.channel[p_track].sample_ptr->data.size)) {
					//TODO, O effect				
					//a->start=a->s->flags&(SF_LOOP|SF_BIDI)?a->s->loopstart:a->s->length;
                                //}
			}
		} break;
		case 'P': {

			do_pan_slide(p_track);
		} break;
		case 'Q': {
			do_retrig(p_track);

		} break;
		case 'R': {


			uint8_t dat;

			if (control.channel[p_track].current_parameter) {

				control.channel[p_track].tremolo_info=control.channel[p_track].current_parameter;
			}

			dat=control.channel[p_track].tremolo_info;
		
			if (!control.ticks_counter && dat) {

				if (dat&0x0f) control.channel[p_track].tremolo_depth=dat&0xf;
				if (dat&0xf0) control.channel[p_track].tremolo_speed=(dat&0xf0)>>2;
			}

			do_tremolo(p_track);
			control.channel[p_track].has_own_volume=true;
	
		} break;
		case 'S': {

			do_effect_S(p_track);
		} break;
		case 'T': {
                        uint8_t dat;
		        int16_t temp=control.tempo;

			if (control.pattern_delay_2) return;

			if (control.channel[p_track].current_parameter) {

				control.channel[p_track].tempo_slide_info=control.channel[p_track].current_parameter;
			}

			dat=control.channel[p_track].tempo_slide_info;

			if (dat>=0x20) {

				if (control.ticks_counter) break;
				control.tempo=dat;	
			} else {

				if (!control.ticks_counter) break;

				if (dat&0x10) {
			
					temp+=(dat&0x0f);
				} else {

			                temp-=dat;
				}
				control.tempo=(temp>255)?255:(temp<0x20?0x20:temp);
		        }
		
		} break;
		case 'U': {
			
			uint8_t dat;

			dat=control.channel[p_track].current_parameter;
			control.channel[p_track].doing_vibrato=true;
			if (!control.ticks_counter) {
				if (dat&0x0f) control.channel[p_track].vibrato_depth=dat&0xf;
				if (dat&0xf0) control.channel[p_track].vibrato_speed=(dat&0xf0)>>2;
			}

                        if (control.external_vibrato) break;						

			if (control.channel[p_track].period) {

				do_vibrato(p_track,true);
				control.channel[p_track].has_own_period=true;
			}                                		
		} break;
		case 'V': {

			control.global_volume=control.channel[p_track].current_parameter;
			if (control.global_volume>128) control.global_volume=128;
		} break;
		case 'W': {
                       do_global_volume_slide(p_track);
		} break;
		case 'X': {
			//sets both channel and current
			control.channel[p_track].channel_panning=control.channel[p_track].current_parameter;
			control.channel[p_track].panning=control.channel[p_track].current_parameter;
		} break;
		case 'Y': {

			uint8_t dat;

			if (control.channel[p_track].current_parameter) {

				control.channel[p_track].panbrello_info=control.channel[p_track].current_parameter;
			}

			dat=control.channel[p_track].panbrello_info;

			if (!control.ticks_counter) {

				if (dat&0x0f) control.channel[p_track].panbrello_depth=(dat&0xf);
				if (dat&0xf0) control.channel[p_track].panbrello_speed=(dat&0xf0)>>4;
			}

			//if (pf->panflag)
			if (control.channel[p_track].panning!=PAN_SURROUND)do_panbrello(p_track);
		
		} break;
		case 'Z': {
			//I DO! cuttoff!
			uint16_t dat=control.channel[p_track].current_parameter;
			
			if (dat<0x80) {
			
				control.channel[p_track].filter.it_cutoff=dat*2;
				if (control.channel[p_track].filter.it_cutoff>0x80)
					control.channel[p_track].filter.it_cutoff++;
			} else if (dat<0x90) {
			
				control.channel[p_track].filter.it_reso=(dat-0x80)*0x10;
			} else {
			
				control.channel[p_track].reverb_send=(dat-0x90)*255/0x6F;
			}
			
		} break;

	}

}

void CPPlayer::pre_process_effects() {

//        MP_VOICE *aout;
	int i;

        for (i=0;i<CPPattern::WIDTH;i++) {

                //a=&pf->control[mp_channel];

               // if ((aout=a->slave)) {
                //        a->fadevol=aout->fadevol;
                 //       a->period=aout->period;
                 //       if (a->kick==KICK_KEYOFF) a->keyoff=aout->keyoff;
                //}

                //if (!a->row) continue;
                //UniSetRow(a->row);
		control.channel[i].has_own_period=false;
		control.channel[i].has_own_volume=false;
		control.channel[i].doing_vibrato=false;
                //explicitslides=0;
                //pt_playeffects();
		if (control.ticks_counter<control.speed) {
	
			run_effects(i);
			run_volume_column_effects(i);
		}

                /* continue volume slide if necessary for XM and IT */
                //if (pf->flags&UF_BGSLIDES) {
                 //       if (!explicitslides && a->sliding)
                 //               DoS3MVolSlide(0);
                 //       else if (a->tmpvolume) a->sliding=explicitslides;
                //}

                if (!control.channel[i].has_own_period) control.channel[i].period=control.channel[i].aux_period;
                if (!control.channel[i].has_own_volume) control.channel[i].volume=control.channel[i].aux_volume;

                if ((control.channel[i].sample_ptr!=NULL) && !(song->has_instruments() && (control.channel[i].instrument_ptr==NULL))) {

			if (song->has_instruments()) {

                                control.channel[i].output_volume=	
						(control.channel[i].volume*control.channel[i].sample_ptr->get_global_volume()*control.channel[i].instrument_ptr->get_volume_global_amount())/2048;
				control.channel[i].output_volume=control.channel[i].output_volume*control.channel[i].random_volume_variation/100;
				
                        } else {

				control.channel[i].output_volume=
						(control.channel[i].volume*control.channel[i].sample_ptr->get_global_volume())>>4;
				
			}

			if (control.channel[i].output_volume>256) {
				
				control.channel[i].output_volume=256;

			} else if (control.channel[i].output_volume<0) {

				control.channel[i].output_volume=0;
			}

			
               }
        }

}
