/*************************************************************************/
/*  cp_player_data_nna.cpp                                               */
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

#include "cp_player_data.h"

void CPPlayer::process_NNAs() {

	int i;

	if (!song->has_instruments()) return;

	for (i=0;i<CPPattern::WIDTH;i++) {

		Channel_Control *aux_chn_ctrl = &control.channel[i];

		if (aux_chn_ctrl->kick==KICK_NOTE) {

			bool k=false;

			if (aux_chn_ctrl->slave_voice!=NULL) {

				Voice_Control *aux_voc_ctrl;

				aux_voc_ctrl=aux_chn_ctrl->slave_voice;
				
				if (aux_chn_ctrl->instrument_index==aux_chn_ctrl->slave_voice->instrument_index) { //maybe carry
					
					aux_chn_ctrl->carry.pan=aux_chn_ctrl->slave_voice->panning_envelope_ctrl;
					aux_chn_ctrl->carry.vol=aux_chn_ctrl->slave_voice->volume_envelope_ctrl;
					aux_chn_ctrl->carry.pitch=aux_chn_ctrl->slave_voice->pitch_envelope_ctrl;
					aux_chn_ctrl->carry.maybe=true;
				} else 
					aux_chn_ctrl->carry.maybe=false;
				
				if (aux_voc_ctrl->NNA_type != CPInstrument::NNA_NOTE_CUT) {
					/* Make sure the old MP_VOICE channel knows it has no
					   master now ! */
					

					
					aux_chn_ctrl->slave_voice=NULL;
					/* assume the channel is taken by NNA */
					aux_voc_ctrl->has_master_channel=false;

					switch (aux_voc_ctrl->NNA_type) {
						case CPInstrument::NNA_NOTE_CONTINUE: {

						} break;
						case CPInstrument::NNA_NOTE_OFF: {


							aux_voc_ctrl->note_end_flags|=END_NOTE_OFF;

							if (!aux_voc_ctrl->volume_envelope_ctrl.active || aux_voc_ctrl->instrument_ptr->get_volume_envelope()->is_loop_enabled()) {
								aux_voc_ctrl->note_end_flags|=END_NOTE_FADE;
							}
						} break;
						case CPInstrument::NNA_NOTE_FADE: {

							aux_voc_ctrl->note_end_flags|=END_NOTE_FADE;
						} break;
					}
				} 
			}

			if (aux_chn_ctrl->duplicate_check_type!=CPInstrument::DCT_DISABLED) {
				int i;

				for (i=0;i<control.max_voices;i++) {
					if (!mixer->is_voice_active(i)||
					   (voice[i].master_channel!=aux_chn_ctrl) ||
					   (aux_chn_ctrl->instrument_index!=voice[i].instrument_index))
						continue;

					Voice_Control *aux_voc_ctrl;

					aux_voc_ctrl=&voice[i];

					k=false;
					switch (aux_chn_ctrl->duplicate_check_type) {
						case CPInstrument::DCT_NOTE:
							if (aux_chn_ctrl->note==aux_voc_ctrl->note)
								k=true;
							break;
						case CPInstrument::DCT_SAMPLE:
							if (aux_chn_ctrl->sample_ptr==aux_voc_ctrl->sample_ptr)
								k=true;
							break;
						case CPInstrument::DCT_INSTRUMENT:
							k=true;
							break;
					}
					if (k) {
						switch (aux_chn_ctrl->duplicate_check_action) {
							case CPInstrument::DCA_NOTE_CUT: {
								aux_voc_ctrl->fadeout_volume=0;
							} break;
							case CPInstrument::DCA_NOTE_OFF: {

								aux_voc_ctrl->note_end_flags|=END_NOTE_OFF;

								if (!aux_voc_ctrl->volume_envelope_ctrl.active || aux_chn_ctrl->instrument_ptr->get_volume_envelope()->is_loop_enabled()) {

									aux_voc_ctrl->note_end_flags|=END_NOTE_FADE;
								}

							} break;
							case CPInstrument::DCA_NOTE_FADE: {
								aux_voc_ctrl->note_end_flags|=END_NOTE_FADE;
							} break;
						}
					}	
				}

			}	
		} /* if (aux_chn_ctrl->kick==KICK_NOTE) */
	}
}
