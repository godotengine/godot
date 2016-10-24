/*************************************************************************/
/*  cp_player_data.cpp                                                   */
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
#include <stdio.h>


CPPlayer::CPPlayer(CPMixer *p_mixer,CPSong *p_song){

	song=p_song;
	mixer=p_mixer;
	control.max_voices=p_mixer->get_total_voice_count()-1; //leave one for the sample
	control.force_no_nna=false;
	control.external_vibrato=false;
	control.filters=true;
	control.random_seed=128364; //anything
	control.play_mode=0;
	set_virtual_channels(p_mixer->get_total_voice_count());
	mixer->set_callback( &CPPlayer::callback_function, this );

	reset();
}
CPPlayer::~CPPlayer(){
}

void CPPlayer::set_virtual_channels(int p_amount) {

	if (p_amount<1) return;
	if (p_amount>mixer->get_total_voice_count())
		return;
	
	control.max_voices=p_amount;

}


void CPPlayer::callback_function(void *p_userdata) {
	
	CPPlayer*pd=(CPPlayer*)p_userdata;
	pd->process_tick();

}

void CPPlayer::process_tick() {

	handle_tick();
	mixer->set_callback_interval( 2500000/control.tempo );
	song_usecs+=2500000/control.tempo;
}

void CPPlayer::reset() {

	if ( mixer==NULL ) return ;
	if ( song==NULL ) return ;

	int i;

	for (i=0;i<control.max_voices;i++) {

         	voice[i].reset();
		mixer->stop_voice(i);
	}
	
	for (i=0;i<CPPattern::WIDTH;i++) {

         	control.channel[i].reset();
		control.channel[i].channel_volume=song->get_channel_volume(i);
		control.channel[i].channel_panning=((int)song->get_channel_pan( i)*PAN_RIGHT/64);
		if (song->is_channel_surround(i))
			control.channel[i].channel_panning=PAN_SURROUND;
		control.channel[i].mute=song->is_channel_mute( i );
		control.channel[i].chorus_send=song->get_channel_chorus(i)*0xFF/64;
		control.channel[i].reverb_send=song->get_channel_reverb(i)*0xFF/64;
	}


	control.speed=song->get_speed();
	control.tempo=song->get_tempo();
	control.global_volume=song->get_global_volume();

	control.position.current_pattern=0;
	control.position.current_row=0;
	control.position.current_order=0;
        control.position.force_next_order=-1;
	control.ticks_counter=control.speed;
        control.position.forbid_jump=false;

	song_usecs=0;
	
}

int64_t CPPlayer::get_channel_last_note_time_usec(int p_channel) const {

	CP_FAIL_INDEX_V(p_channel,64,-1);
	return control.channel[p_channel].last_event_usecs;

}

void CPPlayer::set_channel_global_volume(int p_channel,int p_volume) {

	CP_FAIL_INDEX(p_channel,64);
	control.channel[p_channel].channel_global_volume=CLAMP(p_volume,0,255);

}

int CPPlayer::get_channel_global_volume(int p_channel) const{

	CP_FAIL_INDEX_V(p_channel,64,-1);
	return control.channel[p_channel].channel_global_volume;

}

bool CPPlayer::reached_end_of_song() {

	return control.reached_end;

}
void CPPlayer::set_force_external_vibratos(bool p_force) {

	control.external_vibrato=p_force;
}
void CPPlayer::set_force_no_nna(bool p_force) {

	control.force_no_nna=p_force;
}
