/*************************************************************************/
/*  cp_instrument.h                                                      */
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
#ifndef CP_INSTRUMENT_H
#define CP_INSTRUMENT_H


#include "cp_config.h"
#include "cp_note.h"
#include "cp_envelope.h"

class CPInstrument {
public:


	enum NNA_Type {

		NNA_NOTE_CUT,
		NNA_NOTE_CONTINUE,
		NNA_NOTE_OFF,
		NNA_NOTE_FADE
	};

	enum DC_Type {

		DCT_DISABLED,
		DCT_NOTE,
		DCT_SAMPLE,
		DCT_INSTRUMENT
	};

	enum DC_Action
	{

		DCA_NOTE_CUT,
		DCA_NOTE_OFF,
		DCA_NOTE_FADE,
	};

	enum EnvelopeType {
          	VOLUME_ENVELOPE,
		PAN_ENVELOPE,
		PITCH_ENVELOPE
	};

	
	enum {
		MAX_NAME_LEN=26,
		MAX_ENVELOPE_NODES=25,
		ENVELOPE_FRAC_BITS=8,
		MAX_VOLUME=128,
		MAX_FADEOUT=256,
		MAX_PAN=128,
		MAX_VOLUME_RANDOM=100,			
		MAX_PAN_RANDOM=64, //what did this guy have inside his head?
		
		MAX_FILTER_CUTOFF=127,
		MAX_FILTER_RESONANCE=127
				
	};


	struct Data {


		uint8_t sample_number[CPNote::NOTES];
		uint8_t note_number[CPNote::NOTES];

		NNA_Type NNA_type;
		DC_Type DC_type;
		DC_Action DC_action;

		struct Volume {

			CPEnvelope envelope;
			uint8_t global_amount;
			uint16_t fadeout;
			uint8_t random_variation;

		} volume;

		struct Pan {

			CPEnvelope envelope;
			bool use_default;
			uint8_t default_amount;
			int8_t pitch_separation;
			uint8_t pitch_center;
			uint8_t random_variation;

		} pan;

		struct Pitch {

			CPEnvelope envelope;
			bool use_as_filter;
			bool use_default_cutoff;
			uint8_t default_cutoff;
			bool use_default_resonance;
			uint8_t default_resonance;
		} pitch;

	};
	
private:



	Data data;
	char name[MAX_NAME_LEN];
	
public:

	/* CPInstrument General */
	
	const char *get_name();
	void set_name(const char *p_name);
	
	void set_sample_number(uint8_t p_note,uint8_t p_sample_id);
	uint8_t get_sample_number(uint8_t p_note);
	
	void set_note_number(uint8_t p_note,uint8_t p_note_id);
	uint8_t get_note_number(uint8_t p_note);

	void set_NNA_type(NNA_Type p_NNA_type);
	NNA_Type get_NNA_type();

	void set_DC_type(DC_Type p_DC_type);
	DC_Type get_DC_type();
		
	void set_DC_action(DC_Action p_DC_action);
	DC_Action get_DC_action();

	/* Volume */	

	void set_volume_global_amount(uint8_t p_amount);
	uint8_t get_volume_global_amount();

	void set_volume_fadeout(uint16_t p_amount);
	uint16_t get_volume_fadeout();

	void set_volume_random_variation(uint8_t p_amount);
	uint8_t get_volume_random_variation();
		
	/* Panning */

	void set_pan_default_amount(uint8_t p_amount);
	uint8_t get_pan_default_amount();

	void set_pan_default_enabled(bool p_enabled);
	bool is_pan_default_enabled();
	
	void set_pan_pitch_separation(int8_t p_amount);
	int8_t get_pan_pitch_separation();
	
	void set_pan_pitch_center(uint8_t p_amount);
	uint8_t get_pan_pitch_center();

	void set_pan_random_variation(uint8_t p_amount);
	uint8_t get_pan_random_variation();

	/* Pitch / Filter */

	void set_pitch_use_as_filter(bool p_enabled);
	bool is_pitch_use_as_filter();
	
	void set_filter_use_default_cutoff(bool p_enabled);
	bool filter_use_default_cutoff();

	void set_filter_default_cutoff(uint8_t p_amount);
	uint8_t get_filter_default_cutoff();
	
	void set_filter_use_default_resonance(bool p_enabled);
	bool filter_use_default_resonance();

	void set_filter_default_resonance(uint8_t p_amount);
	uint8_t get_filter_default_resonance();

	CPEnvelope* get_volume_envelope();
	CPEnvelope* get_pan_envelope();
	CPEnvelope* get_pitch_filter_envelope();
		
	bool is_empty();
	
	void reset();
	CPInstrument();

};



#endif


