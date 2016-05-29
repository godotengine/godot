/*************************************************************************/
/*  cp_player_data_control.cpp                                           */
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

void CPPlayer::play_start_pattern(int p_pattern) {

	play_start(p_pattern,-1,-1);
}

void CPPlayer::play_start_song() {

	play_start(-1,-1,-1);
}

void CPPlayer::play_start_song_from_order(int p_order) {

	play_start(-1,p_order,-1);
}

void CPPlayer::play_start_song_from_order_and_row(int p_order,int p_row) {

	play_start(-1,p_order,p_row);
}

void CPPlayer::play_start(int p_pattern, int p_order, int p_row,bool p_lock) {


	if (control.play_mode!=PLAY_NOTHING) play_stop();


	reset();

        if (p_pattern!=-1) {

		control.play_mode=PLAY_PATTERN;
		control.position.current_pattern=p_pattern;
		control.position.current_row=(p_row!=-1)?p_row:0;

	} else {
	
		control.position.current_order=get_song_next_order_idx(song,(p_order==-1)?p_order:p_order-1);
		if (control.position.current_order!=-1) {

			control.play_mode=PLAY_SONG;
			control.position.current_pattern=song->get_order(control.position.current_order);
			control.position.current_row=(p_row!=-1)?p_row:0;
		} 
	}


	control.reached_end=(control.play_mode==PLAY_NOTHING);
	
	
}

void CPPlayer::play_stop() {

	int i;


	control.play_mode=PLAY_NOTHING;

	for (i=0;i<control.max_voices;i++) {

		voice[i].reset();
		mixer->stop_voice(i);
	}

	for (i=0;i<CPPattern::WIDTH;i++) {

		control.channel[i].reset();
	}

	reset();

}

void CPPlayer::play_note(int p_channel,CPNote note,bool p_reserve) {



        if (control.play_mode==PLAY_NOTHING) {

		control.ticks_counter=0;
	}

	/*control.channel[p_channel].reset();
	control.channel[p_channel].channel_volume=song->get_channel_volume(p_channel);
	control.channel[p_channel].channel_panning=((int)song->get_channel_pan( p_channel)*255/64);*/
	if (p_reserve) {
		control.channel[p_channel].mute=false;
		control.channel[p_channel].reserved=true;
	} else {
		
		control.channel[p_channel].reserved=false;
		
	}
	process_note(p_channel,note);



}


int CPPlayer::get_voice_volume(int p_voice) {
	
	return voice[p_voice].display_volume;
}


int CPPlayer::get_voice_envelope_pos(int p_voice,CPEnvelope *p_envelope) {

	int i,tmp_index=-1;

	i=p_voice;




	if ((song->has_instruments()) && (voice[i].instrument_ptr!=NULL) && (voice[i].fadeout_volume>0)) {

		if ((p_envelope==voice[i].instrument_ptr->get_volume_envelope()) && (voice[i].instrument_ptr->get_volume_envelope()->is_enabled())) {

			tmp_index=voice[i].volume_envelope_ctrl.pos_index;
		}

		if ((p_envelope==voice[i].instrument_ptr->get_pan_envelope()) && (voice[i].instrument_ptr->get_pan_envelope()->is_enabled())) {

			tmp_index=voice[i].panning_envelope_ctrl.pos_index;
		}

		if ((p_envelope==voice[i].instrument_ptr->get_pitch_filter_envelope()) && (voice[i].instrument_ptr->get_pitch_filter_envelope()->is_enabled())) {
		

			tmp_index=voice[i].pitch_envelope_ctrl.pos_index;
		}

	}



	return tmp_index;
}

void CPPlayer::goto_next_order() {


	if (control.play_mode!=PLAY_SONG) return;



	control.position.current_row=0;


	control.position.current_order=get_song_next_order_idx(song, control.position.current_order);



	if (control.position.current_order==-1) {

         	reset();	
	}

	control.position.current_pattern=song->get_order(control.position.current_order); 	


}
void CPPlayer::goto_previous_order() {

	if (control.play_mode!=PLAY_SONG) return;


	int next_order,current_order;

	control.position.current_row=0;

	current_order=control.position.current_order;

	next_order=get_song_next_order_idx(song, current_order);

	while ((next_order!=control.position.current_order) && (next_order!=-1)) {

		current_order=next_order;
		next_order=get_song_next_order_idx(song, current_order);
	}

	if (next_order==-1) {

         	reset();	
	} else {

		control.position.current_order=current_order;
	        control.position.current_pattern=song->get_order(control.position.current_order);

	}



}

int CPPlayer::get_channel_voice(int p_channel) {

	if (control.channel[p_channel].slave_voice==NULL) return -1;
	else return control.channel[p_channel].slave_voice_index;
}

const char* CPPlayer::get_voice_sample_name(int p_voice) {

	const char *name = NULL;



	if (!voice[p_voice].sample_ptr) name=voice[p_voice].sample_ptr->get_name();



	return name;

}


bool CPPlayer::is_voice_active(int p_voice) {

	return !( ((voice[p_voice].kick==KICK_NOTHING)||(voice[p_voice].kick==KICK_ENVELOPE))&&!mixer->is_voice_active(p_voice) ); 
		
}		
		


int CPPlayer::get_voice_envelope_pos(int p_voice,CPInstrument::EnvelopeType p_env_type) {
	
	if (!is_voice_active(p_voice))
		return -1;
	
	Voice_Control::Envelope_Control *env=0;
	
	switch (p_env_type) {
		
		case CPInstrument::VOLUME_ENVELOPE: env=&voice[p_voice].volume_envelope_ctrl; break;
		case CPInstrument::PAN_ENVELOPE: env=&voice[p_voice].panning_envelope_ctrl; break;
		case CPInstrument::PITCH_ENVELOPE: env=&voice[p_voice].pitch_envelope_ctrl; break;
	
	}
	
	if (!env)
		return -1;
	
	if (!env->active || env->terminated)
		return -1;
	
	return env->pos_index;
}


CPEnvelope* CPPlayer::get_voice_envelope(int p_voice,CPInstrument::EnvelopeType p_env_type) {
	
	CPInstrument *ins=voice[p_voice].instrument_ptr;
	
	if (!ins)
		return 0;
	
	switch( p_env_type ) {
		
		
		case CPInstrument::VOLUME_ENVELOPE: return ins->get_volume_envelope(); 
		case CPInstrument::PAN_ENVELOPE: return ins->get_pan_envelope(); 
		case CPInstrument::PITCH_ENVELOPE: return ins->get_pitch_filter_envelope();
	};
	
	return 0;
		
}

const char * CPPlayer::get_voice_instrument_name(int p_voice) {



	const char *name = NULL;



	if (voice[p_voice].instrument_ptr!=NULL) name=voice[p_voice].instrument_ptr->get_name();



	return name;

}
void CPPlayer::set_filters_enabled(bool p_enable){

	control.filters=p_enable;
}

int CPPlayer::get_voice_sample_index(int p_voice) {

	return voice[p_voice].sample_index;
}
