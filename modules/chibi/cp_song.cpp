/*************************************************************************/
/*  cp_song.cpp                                                          */
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
#include "cp_song.h"

void CPSong::set_name(const char *p_name) {

	if (p_name == NULL) {
		variables.name[0] = 0;
		return;
	}

	bool done = false;
	for (int i = 0; i < MAX_SONG_NAME; i++) {

		variables.name[i] = done ? 0 : p_name[i];
		if (!done && p_name[i] == 0)
			done = true;
	}

	variables.name[MAX_SONG_NAME - 1] = 0; /* just in case */
}

const char *CPSong::get_name() {

	return variables.name;
}

void CPSong::set_message(const char *p_message) {

	if (p_message == NULL) {
		variables.message[0] = 0;
		return;
	}

	bool done = false;
	for (int i = 0; i < MAX_MESSAGE_LEN; i++) {

		variables.message[i] = done ? 0 : p_message[i];
		if (!done && p_message[i] == 0)
			done = true;
	}

	variables.message[MAX_MESSAGE_LEN - 1] = 0; /* just in case */
}

const char *CPSong::get_message() {

	return variables.message;
}

void CPSong::set_row_highlight_minor(int p_hl_minor) {

	variables.row_highlight_minor = p_hl_minor;
}
int CPSong::get_row_highlight_minor() {

	return variables.row_highlight_minor;
}

void CPSong::set_row_highlight_major(int p_hl_major) {

	variables.row_highlight_major = p_hl_major;

} /* 0 .. 256 */
int CPSong::get_row_highlight_major() {

	return variables.row_highlight_major;

} /* 0 .. 256 */

void CPSong::set_mixing_volume(int p_mix_volume) {

	variables.mixing_volume = p_mix_volume;
} /* 0 .. 128 */
int CPSong::get_mixing_volume() {

	return variables.mixing_volume;

} /* 0 .. 128 */

void CPSong::set_global_volume(int p_global_volume) {

	initial_variables.global_volume = p_global_volume;

} /* 0 .. 128 */
int CPSong::get_global_volume() {

	return initial_variables.global_volume;

} /* 0 .. 128 */

void CPSong::set_stereo_separation(int p_separation) {

	variables.stereo_separation = p_separation;

} /* 0 .. 128 */
int CPSong::get_stereo_separation() {

	return variables.stereo_separation;
} /* 0 .. 128 */

void CPSong::set_stereo(bool p_stereo) {

	variables.use_stereo = p_stereo;
}
bool CPSong::is_stereo() {

	return variables.use_stereo;
}

void CPSong::set_instruments(bool p_instruments) {

	variables.use_instruments = p_instruments;
}
bool CPSong::has_instruments() {

	return variables.use_instruments;
}

void CPSong::set_linear_slides(bool p_linear_slides) {

	variables.use_linear_slides = p_linear_slides;
}
bool CPSong::has_linear_slides() {

	return variables.use_linear_slides;
}

void CPSong::set_old_effects(bool p_old_effects) {

	variables.old_effects = p_old_effects;
}
bool CPSong::has_old_effects() {

	return variables.old_effects;
}

void CPSong::set_compatible_gxx(bool p_compatible_gxx) {

	variables.compatible_gxx = p_compatible_gxx;
}
bool CPSong::has_compatible_gxx() {

	return variables.compatible_gxx;
}

void CPSong::set_speed(int p_speed) {

	CP_ERR_COND(p_speed < MIN_SPEED);
	CP_ERR_COND(p_speed > MAX_SPEED);

	initial_variables.speed = p_speed;

} /* 1 .. 255 */
int CPSong::get_speed() {

	return initial_variables.speed;

} /* 1 .. 255 */

void CPSong::set_tempo(int p_tempo) {

	CP_ERR_COND(p_tempo < MIN_TEMPO);
	CP_ERR_COND(p_tempo > MAX_TEMPO);

	initial_variables.tempo = p_tempo;

} /* MIN_TEMPO .. MAX_TEMPO */
int CPSong::get_tempo() {

	return initial_variables.tempo;

} /* MIN_TEMPO .. MAX_TEMPO */

void CPSong::set_channel_pan(int p_channel, int p_pan) {

	CP_FAIL_INDEX(p_channel, CPPattern::WIDTH);
	CP_FAIL_INDEX(p_pan, CHANNEL_MAX_PAN + 1);

	initial_variables.channel[p_channel].pan = p_pan;

} /* 0 .. CHANNEL_MAX_PAN */
int CPSong::get_channel_pan(int p_channel) {

	CP_FAIL_INDEX_V(p_channel, CPPattern::WIDTH, -1);

	return initial_variables.channel[p_channel].pan;
}

void CPSong::set_channel_volume(int p_channel, int p_volume) {

	CP_FAIL_INDEX(p_channel, CPPattern::WIDTH);
	CP_FAIL_INDEX(p_volume, CHANNEL_MAX_VOLUME + 1);

	initial_variables.channel[p_channel].volume = p_volume;

} /* 0 .. CHANNEL_MAX_VOLUME */

int CPSong::get_channel_volume(int p_channel) {

	CP_FAIL_INDEX_V(p_channel, CPPattern::WIDTH, -1);

	return initial_variables.channel[p_channel].volume;
}

void CPSong::set_channel_chorus(int p_channel, int p_chorus) {

	CP_FAIL_INDEX(p_channel, CPPattern::WIDTH);
	CP_FAIL_INDEX(p_chorus, CHANNEL_MAX_CHORUS + 1);

	initial_variables.channel[p_channel].chorus = p_chorus;

} /* 0 .. CHANNEL_MAX_CHORUS */

int CPSong::get_channel_chorus(int p_channel) {

	CP_FAIL_INDEX_V(p_channel, CPPattern::WIDTH, -1);

	return initial_variables.channel[p_channel].chorus;
}

void CPSong::set_channel_reverb(int p_channel, int p_reverb) {

	CP_FAIL_INDEX(p_channel, CPPattern::WIDTH);
	CP_FAIL_INDEX(p_reverb, CHANNEL_MAX_REVERB + 1);

	initial_variables.channel[p_channel].reverb = p_reverb;

} /* 0 .. CHANNEL_MAX_CHORUS */

int CPSong::get_channel_reverb(int p_channel) {

	CP_FAIL_INDEX_V(p_channel, CPPattern::WIDTH, -1);

	return initial_variables.channel[p_channel].reverb;
}

void CPSong::set_channel_surround(int p_channel, bool p_surround) {

	CP_FAIL_INDEX(p_channel, CPPattern::WIDTH);
	initial_variables.channel[p_channel].surround = p_surround;
}
bool CPSong::is_channel_surround(int p_channel) {

	CP_FAIL_INDEX_V(p_channel, CPPattern::WIDTH, false);

	return initial_variables.channel[p_channel].surround;
}

void CPSong::set_channel_mute(int p_channel, bool p_mute) {

	CP_FAIL_INDEX(p_channel, CPPattern::WIDTH);

	initial_variables.channel[p_channel].mute = p_mute;
}
bool CPSong::is_channel_mute(int p_channel) {

	CP_FAIL_INDEX_V(p_channel, CPPattern::WIDTH, false);

	return initial_variables.channel[p_channel].mute;
}

/* arrays of stuff */

CPPattern *CPSong::get_pattern(int p_pattern) {

	CP_FAIL_INDEX_V(p_pattern, MAX_PATTERNS, NULL);

	return &pattern[p_pattern];
}
CPSample *CPSong::get_sample(int p_sample) {

	CP_FAIL_INDEX_V(p_sample, MAX_SAMPLES, NULL);

	return &sample[p_sample];
}
CPInstrument *CPSong::get_instrument(int p_instrument) {

	CP_FAIL_INDEX_V(p_instrument, MAX_INSTRUMENTS, NULL);

	return &instrument[p_instrument];
}

int CPSong::get_order(int p_order) {

	CP_FAIL_INDEX_V(p_order, MAX_ORDERS, CP_ORDER_NONE);

	return order[p_order];
}
void CPSong::set_order(int p_order, int p_pattern) {

	CP_FAIL_INDEX(p_order, MAX_ORDERS);

	order[p_order] = p_pattern;
}

void CPSong::clear_instrument_with_samples(int p_instrument) {

	CPInstrument *ins = get_instrument(p_instrument);
	if (!ins)
		return;

	for (int i = 0; i < CPNote::NOTES; i++) {

		CPSample *s = get_sample(ins->get_sample_number(i));

		if (!s)
			continue;

		if (s->get_sample_data().is_null())
			continue;

		s->reset();
	}
	ins->reset();
}

void CPSong::make_instrument_from_sample(int p_sample) {

	if (!has_instruments())
		return;
	CP_ERR_COND(!get_sample(p_sample));

	for (int i = 0; i < MAX_INSTRUMENTS; i++) {

		CPInstrument *ins = get_instrument(i);

		bool empty_slot = true;
		for (int n = 0; n < CPNote::NOTES; n++) {

			if (ins->get_sample_number(n) < MAX_SAMPLES) {

				empty_slot = false;
				break;
			}
		}

		if (!empty_slot)
			continue;

		for (int n = 0; n < CPNote::NOTES; n++) {

			ins->set_sample_number(n, p_sample);
			ins->set_note_number(n, n);
		}

		ins->set_name(get_sample(p_sample)->get_name());
		break;
	}
}

void CPSong::make_instruments_from_samples() {

	for (int i = 0; i < MAX_SAMPLES; i++) {

		CPInstrument *ins = get_instrument(i);

		if (!ins)
			continue;

		ins->reset();

		CPSample *s = get_sample(i);

		if (!s)
			continue;

		ins->set_name(s->get_name());

		if (s->get_sample_data().is_null())
			continue;

		for (int j = 0; j < CPNote::NOTES; j++)
			ins->set_sample_number(j, i);
	}
}

void CPSong::reset(bool p_clear_patterns, bool p_clear_samples, bool p_clear_instruments, bool p_clear_variables) {

	if (p_clear_variables) {
		variables.name[0] = 0;
		variables.message[0] = 0;
		variables.row_highlight_major = 16;
		variables.row_highlight_minor = 4;
		variables.mixing_volume = 48;
		variables.old_effects = false;
		if (p_clear_instruments) //should not be cleared, if not clearing instruments!!
			variables.use_instruments = false;
		variables.stereo_separation = 128;
		variables.use_linear_slides = true;
		variables.use_stereo = true;

		initial_variables.global_volume = 128;
		initial_variables.speed = 6;
		initial_variables.tempo = 125;

		for (int i = 0; i < CPPattern::WIDTH; i++) {

			initial_variables.channel[i].pan = 32;
			initial_variables.channel[i].volume = CHANNEL_MAX_VOLUME;
			initial_variables.channel[i].mute = false;
			initial_variables.channel[i].surround = false;
			initial_variables.channel[i].chorus = 0;
			initial_variables.channel[i].reverb = 0;
		}

		effects.chorus.delay_ms = 6;
		effects.chorus.separation_ms = 3;
		effects.chorus.depth_ms10 = 6,
		effects.chorus.speed_hz10 = 5;
		effects.reverb_mode = REVERB_MODE_ROOM;
	}

	if (p_clear_samples) {
		for (int i = 0; i < MAX_SAMPLES; i++)
			get_sample(i)->reset();
	}

	if (p_clear_instruments) {
		for (int i = 0; i < MAX_INSTRUMENTS; i++)
			get_instrument(i)->reset();
	}

	if (p_clear_patterns) {
		for (int i = 0; i < MAX_PATTERNS; i++)
			get_pattern(i)->clear();

		for (int i = 0; i < MAX_ORDERS; i++)
			set_order(i, CP_ORDER_NONE);
	}
}

CPSong::ReverbMode CPSong::get_reverb_mode() {

	return effects.reverb_mode;
}
void CPSong::set_reverb_mode(ReverbMode p_mode) {

	effects.reverb_mode = p_mode;
}

void CPSong::set_chorus_delay_ms(int p_amount) {

	effects.chorus.delay_ms = p_amount;
}
void CPSong::set_chorus_separation_ms(int p_amount) {

	effects.chorus.separation_ms = p_amount;
}
void CPSong::set_chorus_depth_ms10(int p_amount) {

	effects.chorus.depth_ms10 = p_amount;
}
void CPSong::set_chorus_speed_hz10(int p_amount) {

	effects.chorus.speed_hz10 = p_amount;
}

int CPSong::get_chorus_delay_ms() {

	return effects.chorus.delay_ms;
}
int CPSong::get_chorus_separation_ms() {

	return effects.chorus.separation_ms;
}
int CPSong::get_chorus_depth_ms10() {

	return effects.chorus.depth_ms10;
}
int CPSong::get_chorus_speed_hz10() {

	return effects.chorus.speed_hz10;
}

void CPSong::cleanup_unused_patterns() {

	for (int i = 0; i < MAX_PATTERNS; i++) {

		bool used = false;
		if (get_pattern(i)->is_empty())
			continue;

		for (int j = 0; j < MAX_ORDERS; j++) {

			if (get_order(j) == i) {
				used = true;
			}
		}

		if (!used)
			get_pattern(i)->clear();
	}
}
void CPSong::cleanup_unused_instruments() {

	if (!has_instruments())
		return;

	bool instr_found[MAX_INSTRUMENTS];
	for (int i = 0; i < MAX_INSTRUMENTS; i++)
		instr_found[i] = false;

	for (int i = 0; i < MAX_PATTERNS; i++) {

		if (get_pattern(i)->is_empty())
			continue;

		for (int row = 0; row < get_pattern(i)->get_length(); row++) {

			for (int col = 0; col < CPPattern::WIDTH; col++) {

				CPNote n;
				n = get_pattern(i)->get_note(col, row);

				if (n.instrument < MAX_INSTRUMENTS)
					instr_found[n.instrument] = true;
			}
		}
	}

	for (int i = 0; i < MAX_INSTRUMENTS; i++)
		if (!instr_found[i])
			get_instrument(i)->reset();
}
void CPSong::cleanup_unused_samples() {

	if (!has_instruments())
		return;

	bool sample_found[MAX_SAMPLES];
	for (int i = 0; i < MAX_INSTRUMENTS; i++)
		sample_found[i] = false;

	for (int i = 0; i < MAX_PATTERNS; i++) {

		if (get_pattern(i)->is_empty())
			continue;

		for (int row = 0; row < get_pattern(i)->get_length(); row++) {

			for (int col = 0; col < CPPattern::WIDTH; col++) {

				CPNote n;
				n = get_pattern(i)->get_note(col, row);

				if (n.instrument >= MAX_SAMPLES)
					continue;

				if (has_instruments()) {

					for (int nt = 0; nt < CPNote::NOTES; nt++) {

						int smp = get_instrument(n.instrument)->get_sample_number(nt);
						if (smp < MAX_SAMPLES)
							sample_found[smp] = true;
					}

				} else {
					if (n.instrument < MAX_SAMPLES)
						sample_found[n.instrument] = true;
				}
			}
		}
	}

	for (int i = 0; i < MAX_SAMPLES; i++)
		if (!sample_found[i])
			get_sample(i)->reset();
}
void CPSong::cleanup_unused_orders() {

	bool finito = false;
	for (int j = 0; j < MAX_ORDERS; j++) {

		if (get_order(j) == CP_ORDER_NONE)
			finito = true;
		if (finito)
			set_order(j, CP_ORDER_NONE);
	}
}

void CPSong::clear_all_default_pan() {

	for (int i = 0; i < MAX_INSTRUMENTS; i++)
		get_instrument(i)->set_pan_default_enabled(false); //die!

	for (int i = 0; i < MAX_SAMPLES; i++)
		get_sample(i)->set_pan_enabled(false); //die!
}

void CPSong::clear_all_default_vol() {

	for (int i = 0; i < MAX_SAMPLES; i++)
		get_sample(i)->set_default_volume(64); //die!
	for (int i = 0; i < MAX_INSTRUMENTS; i++)
		get_instrument(i)->set_volume_global_amount(CPInstrument::MAX_VOLUME);
}

int CPSong::get_order_in_use_count() {

	int order_count = 0;

	for (int i = (MAX_ORDERS - 1); i >= 0; i--) {

		if (get_order(i) != CP_ORDER_NONE) {
			order_count = i + 1;
			break;
		}
	}

	return order_count;
}
int CPSong::get_pattern_in_use_count() {

	int pattern_count = 0;

	for (int i = (CPSong::MAX_PATTERNS - 1); i >= 0; i--) {

		if (!get_pattern(i)->is_empty()) {
			pattern_count = i + 1;
			break;
		}
	}

	return pattern_count;
}

int CPSong::get_instrument_in_use_count() {

	int instrument_count = 0;

	for (int i = (CPSong::MAX_INSTRUMENTS - 1); i >= 0; i--) {

		CPInstrument *ins = get_instrument(i);
		bool in_use = false;

		for (int s = 0; s < CPNote::NOTES; s++) {

			int smp_idx = ins->get_sample_number(s);
			if (smp_idx < 0 || smp_idx >= CPSong::MAX_SAMPLES)
				continue;

			if (!get_sample(smp_idx)->get_sample_data().is_null()) {
				in_use = true;
				break;
			}
		}

		if (in_use) {
			instrument_count = i + 1;
			break;
		}
	}

	return instrument_count;
}
#include <stdio.h>
int CPSong::get_channels_in_use() {

	int max = 0;

	for (int p = 0; p < CPSong::MAX_PATTERNS; p++) {

		CPPattern *pat = get_pattern(p);
		if (pat->is_empty())
			continue;

		for (int c = (CPPattern::WIDTH - 1); c >= 0; c--) {

			if (c < max)
				break;

			bool has_note = false;
			for (int r = 0; r < pat->get_length(); r++) {

				CPNote n = pat->get_note(c, r);
				if (!n.is_empty()) {
					has_note = true;
					break;
				}
			}

			if (has_note) {

				max = c + 1;
			}
		}
	}

	return max;
}

void CPSong::separate_in_one_sample_instruments(int p_instrument) {

	CP_ERR_COND(!variables.use_instruments);
	CP_FAIL_INDEX(p_instrument, MAX_INSTRUMENTS);

	int remapped_count = 0;

	signed char remap[MAX_SAMPLES];

	for (int i = 0; i < MAX_SAMPLES; i++) {

		remap[i] = -1;
	}

	/* Find remaps */
	CPInstrument *ins = get_instrument(p_instrument);
	for (int i = 0; i < CPNote::NOTES; i++) {

		int sn = ins->get_sample_number(i);

		// check for unusable sample
		if (sn < 0 || sn >= MAX_SAMPLES || get_sample(sn)->get_sample_data().is_null())
			continue;
		printf("sample %i\n", sn);
		if (remap[sn] != -1) {
			printf("already mapped to %i\n", remap[sn]);
			continue;
		}

		printf("isn't remapped\n");

		// find remap

		for (int j = 0; j < MAX_INSTRUMENTS; j++) {

			if (!get_instrument(j)->is_empty())
				continue;

			printf("map to %i\n", j);

			//copy
			*get_instrument(j) = *ins;

			// assign samples
			for (int k = 0; k < CPNote::NOTES; k++) {

				get_instrument(j)->set_note_number(k, k);
				get_instrument(j)->set_sample_number(k, sn);
			}
			remap[sn] = j;
			remapped_count++;
			break;
		}

		CP_ERR_COND(remap[sn] == -1); // no more free instruments
	}

	printf("remapped %i\n", remapped_count);

	if (remapped_count < 2) {
		//undo if only one is remapped
		for (int i = 0; i < MAX_SAMPLES; i++) {

			if (remap[i] != -1) {

				get_instrument(remap[i])->reset();
			}
		}
		return;
	}

	/* remap all song */

	for (int p = 0; p < CPSong::MAX_PATTERNS; p++) {

		CPPattern *pat = get_pattern(p);
		if (pat->is_empty())
			continue;

		for (int c = 0; c < CPPattern::WIDTH; c++) {

			for (int r = 0; r < pat->get_length(); r++) {

				CPNote n = pat->get_note(c, r);
				if (n.note < CPNote::NOTES && n.instrument == p_instrument) {

					int sn = ins->get_sample_number(n.note);
					if (remap[sn] == -1)
						pat->set_note(c, r, CPNote());
					else {

						n.instrument = remap[sn];
						pat->set_note(c, r, n);
					}
				}
			}
		}
	}

	ins->reset();
}

CPSong::CPSong() {

	reset();
}
CPSong::~CPSong() {
}

int get_song_next_order_idx(CPSong *p_song, int p_order_idx) {

	int baseorder, order_counter;

	order_counter = -1;

	baseorder = p_order_idx;

	do {

		baseorder++;
		if (baseorder > (CPSong::MAX_ORDERS - 1)) baseorder = 0;
		order_counter++;

	} while ((p_song->get_order(baseorder) >= (CPSong::MAX_PATTERNS)) && (order_counter < CPSong::MAX_ORDERS));

	if (order_counter == CPSong::MAX_ORDERS) {

		return -1;

	} else {

		return baseorder;
	}
}
