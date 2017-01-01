/*************************************************************************/
/*  cp_note.h                                                            */
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
#ifndef CP_NOTE_H
#define CP_NOTE_H

#include "cp_config.h"

struct CPNote {

	enum {

		NOTES=120,
		OFF=254,
		CUT=253,
		EMPTY=255,
		SCRIPT=252,
	};


	uint8_t note;
	uint8_t instrument;
	uint8_t volume;
	uint8_t command;
	uint8_t parameter;
	unsigned int script_source_sign;
	bool cloned;

	void clear() {

		note=EMPTY;
		instrument=EMPTY;
		volume=EMPTY;
		command=EMPTY;
		parameter=0;
		script_source_sign='\0';
		cloned=false;
	}
	
	void raise() {

		if (note<(NOTES-1))
		    note++;
		else if (note==SCRIPT && parameter<0xFF)
		    parameter++;
	}

	void lower() {

		if ((note>0) && (note<NOTES))
		    note--;
		else if (note==SCRIPT && parameter>0)
		    parameter--;

	}

	bool operator== (const CPNote &rvalue) {

		return (
			 (note==rvalue.note) &&
			 (instrument==rvalue.instrument) &&
			 (volume==rvalue.volume) &&
			 (command==rvalue.command) &&
			 (parameter==rvalue.parameter)
			);
	}

	bool is_empty() const { return (note==EMPTY && instrument==EMPTY && volume==EMPTY && command==EMPTY && parameter==0 && !cloned); }
	CPNote() {

		clear();
	}
};


#endif

