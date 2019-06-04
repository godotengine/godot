/*************************************************************************/
/*  cp_pattern.h                                                         */
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
#ifndef CP_PATTERN_H
#define CP_PATTERN_H

#include "cp_note.h"

class CPPattern {
public:
	enum {
		WIDTH = 64,
		DEFAULT_LEN = 64,
		RESIZE_EVERY_BITS = 4,
		MIN_ROWS = 1, //otherwise clipboard wont work
		MAX_LEN = 256

	};

private:
	struct Event {

		uint16_t pos; //column*WIDTH+row
		uint8_t note;
		uint8_t instrument;
		uint8_t volume;
		uint8_t command;
		uint8_t parameter;
		unsigned int script_source_sign;
		bool cloned;
	};

	uint16_t length;
	uint32_t event_count;
	Event *events;

	int32_t get_event_pos(uint16_t p_target_pos);
	bool erase_event_at_pos(uint16_t p_pos);

	bool resize_event_list_to(uint32_t p_events);

	void operator=(const CPPattern &p_pattern); //no operator=
public:
	bool is_empty();
	void clear();

	bool set_note(uint8_t p_column, uint16_t p_row, const CPNote &p_note); //true if no more memory
	CPNote get_note(uint8_t p_column, uint16_t p_row);

	CPNote get_transformed_script_note(uint8_t p_column, uint16_t p_row);
	int get_scripted_note_target_channel(uint8_t p_column, uint16_t p_row);
	void scripted_clone(uint8_t p_column, uint16_t p_row);
	void scripted_clone_remove(uint8_t p_column, uint16_t p_row);
	void script_transform_note(CPNote &n, const CPNote &p_note);
	bool update_scripted_clones_sourcing_channel(int channel);

	//void copy_to(CPPattern *p_pattern) const;
	void set_length(uint16_t p_rows);
	uint16_t get_length();
	CPPattern();
	~CPPattern();
};

#endif
