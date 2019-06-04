/*************************************************************************/
/*  cp_player_data_utils.cpp                                             */
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
uint8_t CPPlayer::vibrato_table[32] = {
	0, 24, 49, 74, 97, 120, 141, 161, 180, 197, 212, 224, 235, 244, 250, 253,
	255, 253, 250, 244, 235, 224, 212, 197, 180, 161, 141, 120, 97, 74, 49, 24
};

uint8_t CPPlayer::auto_vibrato_table[128] = {
	0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 18, 20, 21, 23,
	24, 25, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 44,
	45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 54, 55, 56, 57, 57, 58,
	59, 59, 60, 60, 61, 61, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63,
	64, 63, 63, 63, 63, 63, 63, 63, 62, 62, 62, 61, 61, 60, 60, 59,
	59, 58, 57, 57, 56, 55, 54, 54, 53, 52, 51, 50, 49, 48, 47, 46,
	45, 44, 42, 41, 40, 39, 38, 36, 35, 34, 32, 31, 30, 28, 27, 25,
	24, 23, 21, 20, 18, 17, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1
};

int8_t CPPlayer::panbrello_table[256] = {
	0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 16, 17, 19, 20, 22, 23,
	24, 26, 27, 29, 30, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44,
	45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 56, 57, 58, 59,
	59, 60, 60, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 64, 64,
	64, 64, 64, 64, 64, 64, 63, 63, 63, 62, 62, 62, 61, 61, 60, 60,
	59, 59, 58, 57, 56, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,
	45, 44, 43, 42, 41, 39, 38, 37, 36, 34, 33, 32, 30, 29, 27, 26,
	24, 23, 22, 20, 19, 17, 16, 14, 12, 11, 9, 8, 6, 5, 3, 2,
	0, -2, -3, -5, -6, -8, -9, -11, -12, -14, -16, -17, -19, -20, -22, -23,
	-24, -26, -27, -29, -30, -32, -33, -34, -36, -37, -38, -39, -41, -42, -43, -44,
	-45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -56, -57, -58, -59,
	-59, -60, -60, -61, -61, -62, -62, -62, -63, -63, -63, -64, -64, -64, -64, -64,
	-64, -64, -64, -64, -64, -64, -63, -63, -63, -62, -62, -62, -61, -61, -60, -60,
	-59, -59, -58, -57, -56, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46,
	-45, -44, -43, -42, -41, -39, -38, -37, -36, -34, -33, -32, -30, -29, -27, -26,
	-24, -23, -22, -20, -19, -17, -16, -14, -12, -11, -9, -8, -6, -5, -3, -2
};

int32_t CPPlayer::get_period(uint16_t p_note, int32_t p_c5speed) {

	if (song->has_linear_slides()) {

		return CPTables::get_linear_period(p_note, 0);
	} else {

		return CPTables::get_log_period(p_note >> 1, p_c5speed >> 1);
	}
}

int32_t CPPlayer::get_frequency(int32_t period) {

	if (song->has_linear_slides()) {

		return CPTables::get_linear_frequency(period);
	} else {

		return CPTables::get_old_frequency(period);
	}
}

int CPPlayer::find_empty_voice() {

	int i;
	int min_priority, min_priority_chan = 0, priority;

	for (i = 0; i < control.max_voices; i++) {

		if (((voice[i].kick == KICK_NOTHING) || (voice[i].kick == KICK_ENVELOPE)) && !mixer->is_voice_active(i)) {

			return i;
		}
	}

	// todo more

	for (i = 0; i < control.max_voices; i++) {
		/* allow us to take over a nonexisting sample */
		//		if ((voice[i].s==NULL)
		//			return k;

		if ((voice[i].kick == KICK_NOTHING) || (voice[i].kick == KICK_ENVELOPE)) {

			priority = voice[i].total_volume << ((CPSampleManager::get_singleton()->get_loop_type(voice[i].sample_ptr->get_sample_data()) != CP_LOOP_NONE) ? 1 : 0);

			if ((voice[i].has_master_channel) && (&voice[i] == voice[i].master_channel->slave_voice)) {

				priority <<= 2;
			}

			if ((i == 0) || (priority < min_priority)) {
				min_priority = priority;
				min_priority_chan = i;
			}
		}
	}

	if (min_priority > 8000 * 7) return -1; /* what the fuck is this? */

	return min_priority_chan;
}
