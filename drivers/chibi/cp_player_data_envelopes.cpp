/*************************************************************************/
/*  cp_player_data_envelopes.cpp                                         */
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


void CPPlayer::Voice_Control::start_envelope(CPEnvelope *p_envelope,Envelope_Control *p_envelope_ctrl,Envelope_Control *p_from_env) {


	if (p_from_env && p_envelope->is_carry_enabled() && !p_from_env->terminated) {

		
		*p_envelope_ctrl=*p_from_env;
	} else {
		p_envelope_ctrl->pos_index=0;
		p_envelope_ctrl->status=1;
		p_envelope_ctrl->sustain_looping=p_envelope->is_sustain_loop_enabled();
		p_envelope_ctrl->looping=p_envelope->is_loop_enabled();
		p_envelope_ctrl->terminated=false;
		p_envelope_ctrl->kill=false;
		p_envelope_ctrl->value=p_envelope->get_height_at_pos(p_envelope_ctrl->pos_index);
	}
}

bool CPPlayer::Voice_Control::process_envelope(CPEnvelope *p_envelope,Envelope_Control *p_envelope_ctrl) {

	if (!p_envelope_ctrl->active) 
		return false;

	if (note_end_flags&END_NOTE_OFF) p_envelope_ctrl->sustain_looping=false;

	p_envelope_ctrl->value=p_envelope->get_height_at_pos(p_envelope_ctrl->pos_index);
	if (p_envelope_ctrl->value==CPEnvelope::NO_POINT)
		return false;
	

	p_envelope_ctrl->pos_index++;

	if (p_envelope_ctrl->sustain_looping) {

		if (p_envelope_ctrl->pos_index>p_envelope->get_node(p_envelope->get_sustain_loop_end()).tick_offset) {

			p_envelope_ctrl->pos_index=p_envelope->get_node(p_envelope->get_sustain_loop_begin()).tick_offset;
		}

	} else if (p_envelope_ctrl->looping) {

		if (p_envelope_ctrl->pos_index>p_envelope->get_node(p_envelope->get_loop_end()).tick_offset) {

			p_envelope_ctrl->pos_index=p_envelope->get_node(p_envelope->get_loop_begin()).tick_offset;
		}

	}

	if (p_envelope_ctrl->pos_index>p_envelope->get_node(p_envelope->get_node_count()-1).tick_offset) {

		p_envelope_ctrl->terminated=true;
		p_envelope_ctrl->pos_index=p_envelope->get_node(p_envelope->get_node_count()-1).tick_offset;
		if (p_envelope->get_node(p_envelope->get_node_count()-1).value==0) p_envelope_ctrl->kill=true;
	}

	return true;
}
