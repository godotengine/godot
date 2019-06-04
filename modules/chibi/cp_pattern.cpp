/*************************************************************************/
/*  cp_pattern.cpp                                                       */
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
#include "cp_pattern.h"

void CPPattern::clear() {

	if (event_count > 0) {

		CP_FREE(events);
		events = NULL;
		event_count = 0;
	}

	length = DEFAULT_LEN;
}

bool CPPattern::resize_event_list_to(uint32_t p_events) {

	//Module is slow in some cpus, so this should be fast enough
	uint32_t new_size = ((p_events - 1) & (~((1 << RESIZE_EVERY_BITS) - 1))) + (1 << RESIZE_EVERY_BITS);

	CP_ERR_COND_V(new_size < p_events, true); //bugARM_INFO

	if (event_count == 0 && new_size == 0)
		return false; //nothing to do

	if (event_count == 0) {

		events = (Event *)CP_ALLOC(new_size * sizeof(Event));

	} else if (new_size == 0) {

		CP_FREE(events);
		events = NULL;
	} else {

		CP_ERR_COND_V(events == NULL, true);
		events = (Event *)CP_REALLOC(events, new_size * sizeof(Event));
	}

	event_count = p_events;

	return false;
}

int32_t CPPattern::get_event_pos(uint16_t p_target_pos) {

	if (event_count == 0)
		return -1;

	int low = 0;
	int high = event_count - 1;
	int middle;

	while (low <= high) {
		middle = (low + high) / 2;

		if (p_target_pos == events[middle].pos) { //match
			break;
		} else if (p_target_pos < events[middle].pos)
			high = middle - 1; //search low end of array
		else
			low = middle + 1; //search high end of array
	}

	/* adapt so we are behind 2 */

	if (events[middle].pos < p_target_pos)
		middle++;
	return middle;

	/* Linear search for now */

	/*
	int32_t pos_idx=0;
		
	for (;pos_idx<event_count;pos_idx++) {
		
		if (event_list[pos_idx].pos>=p_target_pos)
			break;
		
	} */

	//return pos_idx;
}

bool CPPattern::erase_event_at_pos(uint16_t p_pos) {

	if (event_count == 0)
		return false;

	Event *event_list = events;

	int32_t pos_idx = get_event_pos(p_pos);
	if (pos_idx == -1) {
		CP_ERR_COND_V(pos_idx == -1, true);
	}

	if (pos_idx == event_count || event_list[pos_idx].pos != p_pos) {
		/* Nothing to Erase */
		return false;
	}

	for (int32_t i = pos_idx; i < (event_count - 1); i++) {

		event_list[i] = event_list[i + 1];
	}

	resize_event_list_to(event_count - 1);

	return false;
}

bool CPPattern::set_note(uint8_t p_column, uint16_t p_row, const CPNote &p_note) {

	CP_ERR_COND_V(p_column >= WIDTH, true);
	CP_ERR_COND_V(p_row >= length, true);

	int32_t new_pos;
	uint16_t target_pos = p_row * WIDTH + p_column;

	if (p_note.is_empty()) {
		bool res = erase_event_at_pos(target_pos);

		return res;
		;
	}

	Event *event_list = 0;

	if (event_count == 0) {
		/* If no events, create the first */

		if (resize_event_list_to(1)) {

			CP_PRINTERR("Can't resize event list to 1");
			return true;
		}

		event_list = events;
		if (event_list == 0) {

			CP_PRINTERR("Can't get event list");
			return true;
		}

		new_pos = 0;

	} else {
		/* Prepare to add */

		event_list = events;
		if (event_list == 0) {

			CP_PRINTERR("Can't get event list");
			return true;
		}

		int32_t pos_idx = get_event_pos(target_pos);

		if (pos_idx == -1) {

			CP_PRINTERR("Can't find add position");
			return true;
		}

		if (pos_idx == event_count || event_list[pos_idx].pos != target_pos) {
			/* If the note being modified didnt exist, then we add it */

			//resize, and return if out of mem
			if (resize_event_list_to(event_count + 1)) {

				CP_PRINTERR("Can't resize event list");
				return true;
			}
			event_list = events;
			if (event_list == 0) {

				CP_PRINTERR("Can't get event list");
				return true;
			}

			//make room for new pos, this wont do a thing if pos_idx was ==event_count
			for (int32_t i = (event_count - 1); i > pos_idx; i--) {
				event_list[i] = event_list[i - 1];
			}

		} /* Else it means that position is taken, so we just modify it! */

		new_pos = pos_idx;
	}

	event_list[new_pos].pos = target_pos;
	event_list[new_pos].note = p_note.note;
	event_list[new_pos].instrument = p_note.instrument;
	event_list[new_pos].volume = p_note.volume;
	event_list[new_pos].command = p_note.command;
	event_list[new_pos].parameter = p_note.parameter;
	event_list[new_pos].script_source_sign = p_note.script_source_sign;
	event_list[new_pos].cloned = p_note.cloned;

	return false;
}
CPNote CPPattern::get_note(uint8_t p_column, uint16_t p_row) {

	if (p_column == CPNote::EMPTY) return CPNote();

	CP_ERR_COND_V(p_column >= WIDTH, CPNote());
	CP_ERR_COND_V(p_row >= length, CPNote());

	if (event_count == 0)
		return CPNote();

	Event *event_list = events;

	CP_ERR_COND_V(event_list == 0, CPNote());

	uint16_t target_pos = p_row * WIDTH + p_column;
	int32_t pos_idx = get_event_pos(target_pos);
	if (pos_idx == -1) {

		CP_PRINTERR("Can't find event pos");
		return CPNote();
	}

	if (pos_idx >= event_count || event_list[pos_idx].pos != target_pos) {
		/* no note found */

		return CPNote();
	}

	CPNote n;
	n.note = event_list[pos_idx].note;
	n.instrument = event_list[pos_idx].instrument;
	n.volume = event_list[pos_idx].volume;
	n.command = event_list[pos_idx].command;
	n.parameter = event_list[pos_idx].parameter;
	n.script_source_sign = event_list[pos_idx].script_source_sign;
	n.cloned = event_list[pos_idx].cloned;

	return n;
}

CPNote CPPattern::get_transformed_script_note(uint8_t p_column, uint16_t p_row) {

	CPNote n = get_note(p_column, p_row);

	// get source channel and note

	int channel = get_scripted_note_target_channel(p_column, p_row);
	CPNote src_n = get_note(channel, 0);

	if (src_n.note == CPNote::SCRIPT) return CPNote();

	script_transform_note(src_n, n);

	return src_n;
}

int CPPattern::get_scripted_note_target_channel(uint8_t p_column, uint16_t p_row) {

	CPNote n = get_note(p_column, p_row);

	if (n.note != CPNote::SCRIPT) return CPNote::EMPTY;

	int channel = n.instrument;

	if (n.script_source_sign == '\0') {

		if (channel < 0 || channel >= CPPattern::WIDTH) return CPNote::EMPTY;

	} else {

		channel = p_column + ((n.script_source_sign == '+') ? 1 : -1) * (channel + 1);
		if (channel < 0 || channel >= CPPattern::WIDTH) return CPNote::EMPTY;
	}

	return channel;
}

void CPPattern::scripted_clone(uint8_t p_column, uint16_t p_row) {

	int channel = get_scripted_note_target_channel(p_column, p_row);
	int src_row = 1;
	CPNote script_n = get_note(p_column, p_row);

	for (int row = p_row + 1; row < length; ++row) {

		CPNote src_n = get_note(channel, src_row);
		CPNote target_n = get_note(p_column, row);

		if (target_n.note != CPNote::SCRIPT) {
			if (src_n.note == CPNote::SCRIPT) {
				src_n = CPNote();
				channel = CPNote::EMPTY;
			}

			script_transform_note(src_n, script_n);

			src_n.cloned = true;
			set_note(p_column, row, src_n);

		} else {

			return;
		}

		src_row++;
	}
}

void CPPattern::scripted_clone_remove(uint8_t p_column, uint16_t p_row) {

	if (get_note(p_column, p_row).cloned)
		set_note(p_column, p_row, CPNote());

	for (int row = p_row + 1; row < length; ++row) {

		CPNote target_n = get_note(p_column, row);

		if (target_n.note != CPNote::SCRIPT) {

			set_note(p_column, row, CPNote());

		} else {

			return;
		}
	}
}

void CPPattern::script_transform_note(CPNote &n, const CPNote &p_note) {

	// set instrument

	if (n.note < CPNote::NOTES && p_note.volume != CPNote::EMPTY) {

		n.instrument = p_note.volume;
	}

	// transpose

	if (n.note < CPNote::NOTES && p_note.command != CPNote::EMPTY) {

		int transpose = (p_note.parameter & 0xF) + (p_note.parameter / 0x10) * 12;

		if (p_note.command == '^') {

			if (n.note >= CPNote::NOTES - transpose)
				n.note = CPNote::NOTES - 1;
			else
				n.note += transpose;

		} else if (p_note.command == 'v') {

			if (n.note <= transpose)
				n.note = 0;
			else
				n.note -= transpose;
		}
	}
}

bool CPPattern::update_scripted_clones_sourcing_channel(int channel) {

	bool updated = false;

	for (int x = 0; x < WIDTH; ++x) {

		for (int y = 0; y < length; ++y) {

			if (channel == get_scripted_note_target_channel(x, y)) {

				scripted_clone(x, y);
				updated = true;
			}
		}
	}

	return updated;
}

void CPPattern::set_length(uint16_t p_rows) {

	if (event_count == 0) {

		if (p_rows >= MIN_ROWS)
			length = p_rows;

		return;
	}

	if (p_rows < MIN_ROWS) {

		return;
	}

	if (p_rows < length) {

		Event *event_list = events;
		if (event_list == 0) {

			CP_PRINTERR("get_event_list() Failed");
			return;
		}

		uint16_t target_pos = p_rows * WIDTH;
		int32_t pos_idx = get_event_pos(target_pos);

		if (pos_idx == -1) {

			CP_ERR_COND(pos_idx == -1);
		}

		if (resize_event_list_to(pos_idx)) {

			CP_PRINTERR("resize_event_list_to(pos_idx) Failed");
			return;
		}
	}

	length = p_rows;
}
#if 0
void CPPattern::copy_to(CPPattern *p_pattern) const {
	
	
	
	
	p_pattern->clear();
	p_pattern->length=length;
	
	
	if (!event_count)
		return;
	

	
	int bufsiz=MemPool_Wrapper::get_singleton()->get_mem_size( mem_handle );
	MemPool_Handle aux_mem_handle=MemPool_Wrapper::get_singleton()->alloc_mem( bufsiz );
	
	if (aux_mem_handle.is_null()) {
		
		CP_PRINTERR("own handle is null");

		return;		
	}
			
	
	if (MemPool_Wrapper::get_singleton()->lock_mem(aux_mem_handle)) {
		CP_PRINTERR("Unable to lock aux new handle");		

		return;
		
	}
	
	if (MemPool_Wrapper::get_singleton()->lock_mem(mem_handle)) {
		
		CP_PRINTERR("Unable to lock own handle");

		return;
	}
	
	uint8_t* srcuint8_tt8_t*)MemPool_Wrapper::get_singleton()->get_mem(mem_handle);
	uint8_t* dstuint8_tt8_t*)MemPool_Wrapper::get_singleton()->get_mem(aux_mem_handle);
	
	for (int i=0;i<bufsiz;i++) 
		dst[i]=src[i];
	
	MemPool_Wrapper::get_singleton()->unlock_mem(mem_handle);
	MemPool_Wrapper::get_singleton()->unlock_mem(aux_mem_handle);
	
	p_pattern->mem_handle=aux_mem_handle;
	p_pattern->event_count=event_count;
	

}
#endif
uint16_t CPPattern::get_length() {

	return length;
}
CPPattern::CPPattern() {

	length = DEFAULT_LEN;
	event_count = 0;
	clear();
}
bool CPPattern::is_empty() {

	return events == NULL;
}

CPPattern::~CPPattern() {

	clear();
}
