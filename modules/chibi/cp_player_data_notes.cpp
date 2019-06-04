/*************************************************************************/
/*  cp_player_data_notes.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "cp_sample_manager.h"

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

void CPPlayer::process_new_note(int p_track, uint8_t p_note) { // if there's really a new note....

	if (control.channel[p_track].real_note != 255) {
		control.channel[p_track].old_note = control.channel[p_track].real_note;
	}

	control.channel[p_track].real_note = p_note;

	control.channel[p_track].kick = KICK_NOTE;

	control.channel[p_track].sample_start_index = -1;
	control.channel[p_track].sliding = 0;
	control.channel[p_track].row_has_note = true;
	control.channel[p_track].last_event_usecs = song_usecs;

	if (control.channel[p_track].panbrello_type) control.channel[p_track].panbrello_position = 0;
}

bool CPPlayer::process_new_instrument(int p_track, uint8_t p_instrument) {

	//	bool different_instrument=false;
	ERR_FAIL_INDEX_V(p_instrument, CPSong::MAX_INSTRUMENTS, false);

	if (song->has_instruments()) {

		control.channel[p_track].instrument_ptr = song->get_instrument(p_instrument);
	} else {

		control.channel[p_track].instrument_ptr = NULL;
	}

	control.channel[p_track].retrig_counter = 0;
	control.channel[p_track].tremor_position = 0;
	control.channel[p_track].sample_offset_fine = 0;
	int old_instr_index = control.channel[p_track].instrument_index;
	control.channel[p_track].instrument_index = p_instrument;

	return (old_instr_index != p_instrument);
}

// returns if it was able to process
bool CPPlayer::process_note_and_instrument(int p_track, int p_note, int p_instrument) {

	bool aux_result;
	aux_result = false;
	CPSample *aux_sample = 0; // current sample
	int dest_sample_index;
	bool new_instrument = false;

	control.channel[p_track].row_has_note = false; // wise man says.. "we dont have a note... until we really know we have a note".
	control.channel[p_track].new_instrument = false;

	if ((p_note < 0) && (p_instrument < 0)) return aux_result; // nothing to do here
	if ((p_note == 255) && (p_instrument == 255)) return aux_result;

	if ((p_note >= 0) && (p_note < 120)) {

		process_new_note(p_track, p_note);

	} else if (p_note == CPNote::CUT) {

		control.channel[p_track].aux_volume = 0;
		control.channel[p_track].note_end_flags |= END_NOTE_OFF;
		control.channel[p_track].note_end_flags |= END_NOTE_KILL;
		return aux_result;

	} else if ((p_note == CPNote::OFF) && (song->has_instruments())) {

		if (control.channel[p_track].instrument_ptr != NULL) {

			control.channel[p_track].note_end_flags |= END_NOTE_OFF;

			if (!control.channel[p_track].instrument_ptr->get_volume_envelope()->is_enabled() || control.channel[p_track].instrument_ptr->get_volume_envelope()->is_loop_enabled()) {

				control.channel[p_track].note_end_flags |= END_NOTE_FADE;
			}
		}

		return aux_result;
	} else
		return aux_result; // invalid note!

	if ((p_instrument >= 0) && (p_instrument < CPSong::MAX_INSTRUMENTS)) {
		new_instrument = process_new_instrument(p_track, p_instrument);

		if (song->has_instruments()) {
			// If we're in instrument mode...
			if (control.channel[p_track].instrument_ptr->get_sample_number(control.channel[p_track].real_note) >= CPSong::MAX_SAMPLES) {

				control.channel[p_track].kick = KICK_NOTHING;
				return aux_result;

			} else {
				dest_sample_index = control.channel[p_track].instrument_ptr->get_sample_number(control.channel[p_track].real_note);
				control.channel[p_track].note = control.channel[p_track].instrument_ptr->get_note_number(control.channel[p_track].real_note);
			}

		} else {
			// If we're in sample mode...
			dest_sample_index = control.channel[p_track].instrument_index;
			control.channel[p_track].note = control.channel[p_track].real_note;
		}

		control.channel[p_track].sample_index = dest_sample_index;
		aux_sample = song->get_sample(dest_sample_index);

		if (!CPSampleManager::get_singleton()->check(aux_sample->get_sample_data())) {
			/* INVALID SAMPLE */
			control.channel[p_track].kick = KICK_NOTHING;
			return aux_result;
		}

		aux_sample = song->get_sample(dest_sample_index);
	} else {

		if (!control.channel[p_track].sample_ptr)
			return aux_result;

		if (song->has_instruments()) {

			if (!control.channel[p_track].instrument_ptr)
				return aux_result;

			control.channel[p_track].note = control.channel[p_track].instrument_ptr->get_note_number(control.channel[p_track].real_note);

		} else {

			control.channel[p_track].note = control.channel[p_track].real_note;
		}

		aux_sample = control.channel[p_track].sample_ptr;
	}

	if (p_instrument >= CPSong::MAX_INSTRUMENTS && control.channel[p_track].sample_ptr != aux_sample) {

		control.channel[p_track].new_instrument = (control.channel[p_track].period > 0);
	}

	control.channel[p_track].sample_ptr = aux_sample;

	/* channel or instrument determined panning ? */

	control.channel[p_track].panning = control.channel[p_track].channel_panning;

	/* set filter,if any ? */

	if (aux_sample->is_pan_enabled()) {

		control.channel[p_track].panning = (int)aux_sample->get_pan() * 255 / 64;

	} else if (song->has_instruments() && (control.channel[p_track].instrument_ptr->is_pan_default_enabled())) {

		control.channel[p_track].panning = (int)control.channel[p_track].instrument_ptr->get_pan_default_amount() * 255 / 64;
	}

	if (song->has_instruments()) {

		/* Pitch-Pan Separation */
		if ((control.channel[p_track].instrument_ptr->get_pan_pitch_separation() != 0) && (control.channel[p_track].channel_panning != PAN_SURROUND)) {

			control.channel[p_track].panning += ((control.channel[p_track].real_note - control.channel[p_track].instrument_ptr->get_pan_pitch_center()) * control.channel[p_track].instrument_ptr->get_pan_pitch_separation()) / 8;

			if (control.channel[p_track].panning < PAN_LEFT) control.channel[p_track].panning = PAN_LEFT;
			if (control.channel[p_track].panning > PAN_RIGHT) control.channel[p_track].panning = PAN_RIGHT;
		}

		/* Random Volume Variation */
		if (control.channel[p_track].instrument_ptr->get_volume_random_variation() > 0) {

			control.channel[p_track].random_volume_variation = 100 - (cp_random_generate(&control.random_seed) % control.channel[p_track].instrument_ptr->get_volume_random_variation());

		} else {

			control.channel[p_track].random_volume_variation = 100;
		}

		/* Random Pan Variation */
		if ((control.channel[p_track].instrument_ptr->get_pan_random_variation() > 0) && (control.channel[p_track].panning != PAN_SURROUND)) {

			int aux_pan_modifier;

			aux_pan_modifier = (cp_random_generate(&control.random_seed) % (control.channel[p_track].instrument_ptr->get_pan_random_variation() << 2));
			if ((cp_random_generate(&control.random_seed) % 2) == 1) aux_pan_modifier = 0 - aux_pan_modifier; /* it's 5am, let me sleep :) */

			control.channel[p_track].panning += aux_pan_modifier;

			if (control.channel[p_track].panning < PAN_LEFT) control.channel[p_track].panning = PAN_LEFT;
			if (control.channel[p_track].panning > PAN_RIGHT) control.channel[p_track].panning = PAN_RIGHT;
		}

		/*filter*/

		if (control.channel[p_track].instrument_ptr->filter_use_default_cutoff()) {

			control.channel[p_track].filter.it_cutoff = control.channel[p_track].instrument_ptr->get_filter_default_cutoff() * 2;
		}

		if (control.channel[p_track].instrument_ptr->filter_use_default_resonance()) {

			control.channel[p_track].filter.it_reso = control.channel[p_track].instrument_ptr->get_filter_default_resonance() * 2;
		}

		/*envelopes*/

		control.channel[p_track].volume_envelope_on = control.channel[p_track].instrument_ptr->get_volume_envelope()->is_enabled();
		control.channel[p_track].panning_envelope_on = control.channel[p_track].instrument_ptr->get_pan_envelope()->is_enabled();
		control.channel[p_track].pitch_envelope_on = control.channel[p_track].instrument_ptr->get_pitch_filter_envelope()->is_enabled();
		control.channel[p_track].NNA_type = control.channel[p_track].instrument_ptr->get_NNA_type();
		control.channel[p_track].duplicate_check_type = control.channel[p_track].instrument_ptr->get_DC_type();
		control.channel[p_track].duplicate_check_action = control.channel[p_track].instrument_ptr->get_DC_action();

	} else {

		control.channel[p_track].NNA_type = CPInstrument::NNA_NOTE_CUT;
		control.channel[p_track].duplicate_check_type = CPInstrument::DCT_DISABLED;
		control.channel[p_track].duplicate_check_action = CPInstrument::DCA_NOTE_CUT;
	}

	if (p_instrument < CPSong::MAX_INSTRUMENTS) { // instrument change

		control.channel[p_track].volume = control.channel[p_track].aux_volume = aux_sample->get_default_volume();
	}

	control.channel[p_track].slide_to_period = control.channel[p_track].aux_period = get_period((uint16_t)(control.channel[p_track].note) << 1, CPSampleManager::get_singleton()->get_c5_freq((aux_sample->get_sample_data())));

	control.channel[p_track].note_end_flags = END_NOTE_NOTHING; /* clears flags */

	return true;
}

void CPPlayer::process_volume_column(int p_track, uint8_t p_volume) {

	control.channel[p_track].current_volume_command = CPNote::EMPTY;
	control.channel[p_track].current_volume_parameter = CPNote::EMPTY;

	if (p_volume < 65) { // VOLUME

		control.channel[p_track].aux_volume = p_volume;
	} else if (p_volume < 125) { // Volume Command

		control.channel[p_track].current_volume_command = (p_volume - 65) / 10;
		control.channel[p_track].current_volume_parameter = (p_volume - 65) % 10;
	} else if (p_volume < 193) { // PAN

		control.channel[p_track].channel_panning = (p_volume - 128) * PAN_RIGHT / 64;
		control.channel[p_track].panning = control.channel[p_track].channel_panning;

	} else if (p_volume < 213) { //More volume Commands

		control.channel[p_track].current_volume_command = ((p_volume - 193) / 10) + 6;
		control.channel[p_track].current_volume_parameter = (p_volume - 193) % 10;
	}
}

void CPPlayer::process_note(int p_track, CPNote p_note) {

	if (p_note.note != CPNote::SCRIPT) {

		process_note_and_instrument(p_track, p_note.note, p_note.instrument);
		process_volume_column(p_track, p_note.volume);
		control.channel[p_track].current_command = p_note.command;
		control.channel[p_track].current_parameter = p_note.parameter;

	} else {

		CPNote n = song->get_pattern(control.position.current_pattern)->get_transformed_script_note(p_track, control.position.current_row);
		process_note(p_track, n);

		song->get_pattern(control.position.current_pattern)->scripted_clone(p_track, control.position.current_row);
	}
}
