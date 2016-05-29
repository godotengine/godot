/*************************************************************************/
/*  cp_instrument.cpp                                                    */
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
#include "cp_instrument.h"
#include "cp_song.h"
#include "cp_note.h"



const char *CPInstrument::get_name() {

	return name;

}
void CPInstrument::set_name(const char *p_name) {
	
	
	if (p_name==NULL) {
		name[0]=0;
		return;
	}
	
	
	bool done=false;
	for (int i=0;i<MAX_NAME_LEN;i++) {
		
		
		name[i]=done?0:p_name[i];
		if (!done && p_name[i]==0)
			done=true;
	}
	
	name[MAX_NAME_LEN-1]=0; /* just in case */
	
	
}

void CPInstrument::set_sample_number(uint8_t p_note,uint8_t p_sample_id) {
	
	CP_ERR_COND(p_note>=CPNote::NOTES);
	CP_ERR_COND(p_sample_id>CPSong::MAX_SAMPLES && p_sample_id!=CPNote::EMPTY);
	data.sample_number[p_note]=p_sample_id;
	
	
}
uint8_t CPInstrument::get_sample_number(uint8_t p_note) {
	
	CP_ERR_COND_V(p_note>=CPNote::NOTES,0);
	return data.sample_number[p_note];
}

void CPInstrument::set_note_number(uint8_t p_note,uint8_t p_note_id) {
	
	CP_ERR_COND(p_note>=CPNote::NOTES);
	CP_ERR_COND(p_note_id>=CPNote::NOTES && p_note_id!=CPNote::EMPTY);
	data.note_number[p_note]=p_note_id;
	
}
uint8_t CPInstrument::get_note_number(uint8_t p_note) {
	
	CP_ERR_COND_V(p_note>=CPNote::NOTES,0);
	return data.note_number[p_note];
	
}

void CPInstrument::set_NNA_type(NNA_Type p_NNA_type) {
	
	data.NNA_type=p_NNA_type;
}
CPInstrument::NNA_Type CPInstrument::get_NNA_type() {
	
	return data.NNA_type;
}

void CPInstrument::set_DC_type(DC_Type p_DC_type) {
	
	data.DC_type=p_DC_type;
}
CPInstrument::DC_Type CPInstrument::get_DC_type() {
	
	return data.DC_type;
	
}
	
void CPInstrument::set_DC_action(DC_Action p_DC_action) {
	
	data.DC_action=p_DC_action;
}
CPInstrument::DC_Action CPInstrument::get_DC_action() {
	
	return data.DC_action;
}

/* Volume */	

void CPInstrument::set_volume_global_amount(uint8_t p_amount) {
	
	CP_ERR_COND(p_amount>MAX_VOLUME);
	data.volume.global_amount=p_amount;
	
}
uint8_t CPInstrument::get_volume_global_amount() {
	
	return data.volume.global_amount;
}

void CPInstrument::set_volume_fadeout(uint16_t p_amount) {
	CP_ERR_COND(p_amount>MAX_FADEOUT);
	data.volume.fadeout=p_amount; 
}
uint16_t CPInstrument::get_volume_fadeout() {
	
	return data.volume.fadeout;
}
void CPInstrument::set_volume_random_variation(uint8_t p_amount) {
	
	CP_ERR_COND(p_amount>MAX_VOLUME_RANDOM);
	data.volume.random_variation=p_amount;
}
uint8_t CPInstrument::get_volume_random_variation() {
	
	return data.volume.random_variation;
}
	
/* Panning */

void CPInstrument::set_pan_default_amount(uint8_t p_amount) {
	
	CP_ERR_COND(p_amount>MAX_PAN);
	data.pan.default_amount=p_amount;
}
uint8_t CPInstrument::get_pan_default_amount() {
	
	return data.pan.default_amount;
}

void CPInstrument::set_pan_default_enabled(bool p_enabled) {
	
	data.pan.use_default=p_enabled;
}
bool CPInstrument::is_pan_default_enabled() {
	
	return data.pan.use_default;
	
}

void CPInstrument::set_pan_pitch_separation(int8_t p_amount) {
	
	CP_ERR_COND(p_amount<-32);
	CP_ERR_COND(p_amount>32);
	data.pan.pitch_separation=p_amount;
}
int8_t CPInstrument::get_pan_pitch_separation() {
	
	return data.pan.pitch_separation;
}

void CPInstrument::set_pan_pitch_center(uint8_t p_amount) {
	
	CP_ERR_COND(p_amount>=CPNote::NOTES);
	data.pan.pitch_center=p_amount;
}
uint8_t CPInstrument::get_pan_pitch_center() {
	
	return data.pan.pitch_center;
}

void CPInstrument::set_pan_random_variation(uint8_t p_amount) {
	
	CP_ERR_COND(p_amount>MAX_PAN_RANDOM);
	data.pan.random_variation=p_amount;
}
uint8_t CPInstrument::get_pan_random_variation() {
	
	return data.pan.random_variation;
}

/* Pitch / Filter */

void CPInstrument::set_pitch_use_as_filter(bool p_enabled) {
	
	data.pitch.use_as_filter=p_enabled;
}
bool CPInstrument::is_pitch_use_as_filter() {
	
	return data.pitch.use_as_filter;
}

void CPInstrument::set_filter_use_default_cutoff(bool p_enabled) {
	
	data.pitch.use_default_cutoff=p_enabled;

}
bool CPInstrument::filter_use_default_cutoff() {
	
	return data.pitch.use_default_cutoff;
}

void CPInstrument::set_filter_default_cutoff(uint8_t p_amount) {
	
	CP_ERR_COND(p_amount>MAX_FILTER_CUTOFF);
	data.pitch.default_cutoff=p_amount;
}
uint8_t CPInstrument::get_filter_default_cutoff() {
	
	return data.pitch.default_cutoff;
}

void CPInstrument::set_filter_use_default_resonance(bool p_enabled) {
	
	data.pitch.use_default_resonance=p_enabled;
}
bool CPInstrument::filter_use_default_resonance() {
	
	return data.pitch.use_default_resonance;
}

void CPInstrument::set_filter_default_resonance(uint8_t p_amount) {
	
	CP_ERR_COND(p_amount>MAX_FILTER_RESONANCE);
	data.pitch.default_resonance=p_amount;
	
}
uint8_t CPInstrument::get_filter_default_resonance() {
	
	return data.pitch.default_resonance;
}

/* Envelopes */


CPEnvelope* CPInstrument::get_volume_envelope() {
	
	return &data.volume.envelope;
}
CPEnvelope* CPInstrument::get_pan_envelope() {
	
	return &data.pan.envelope;	
}
CPEnvelope* CPInstrument::get_pitch_filter_envelope() {
	
	return &data.pitch.envelope;	
	
	
}


void CPInstrument::reset() {
	
	name[0]=0;
	
	data.NNA_type=NNA_NOTE_CUT;
	data.DC_action=DCA_NOTE_CUT;
	data.DC_type=DCT_DISABLED;
	
	for (int i=0;i<CPNote::NOTES;i++) {
		
		data.sample_number[i]=CPNote::EMPTY;
		data.note_number[i]=i;
	}
	
	data.volume.envelope.reset();
	data.volume.envelope.set_max(64);
	data.volume.envelope.set_min(0);
	data.volume.envelope.add_position(0,64,false);
	data.volume.envelope.add_position(30,64,false);
	
	data.volume.global_amount=MAX_VOLUME;
	data.volume.fadeout=0;
	data.volume.random_variation=0;
	
	data.pan.envelope.reset();
	data.pan.envelope.set_max(32);
	data.pan.envelope.set_min(-32);
	data.pan.envelope.add_position(0,0,false);
	data.pan.envelope.add_position(30,0,false);
	
	data.pan.default_amount=32;
	data.pan.pitch_center=48;
	data.pan.pitch_separation=0;
	data.pan.use_default=false;
	data.pan.random_variation=0;
	
	
	data.pitch.envelope.reset();
	data.pitch.envelope.set_max(32);
	data.pitch.envelope.set_min(-32);
	data.pitch.envelope.add_position(0,0,false);
	data.pitch.envelope.add_position(30,0,false);
	data.pitch.use_as_filter=false;
	data.pitch.use_default_cutoff=false;
	data.pitch.use_default_resonance=false;
	data.pitch.default_cutoff=0;
	data.pitch.default_resonance=0;
	
}

bool CPInstrument::is_empty() {
	
	bool has_sample=false;
	
	for (int i=0;i<CPNote::NOTES;i++) {
		
		if (data.sample_number[i]!=CPNote::EMPTY) {
			
			has_sample=true;
			break;
		}
	}

	return !has_sample;
}

CPInstrument::CPInstrument() {
	
	reset();
	
}

