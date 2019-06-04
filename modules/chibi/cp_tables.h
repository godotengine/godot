/*************************************************************************/
/*  cp_tables.h                                                          */
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

#ifndef CPTABLES_H
#define CPTABLES_H

#include "cp_config.h"

/**conversion CPTables/functions
  *@author Juan Linietsky
  */

/******************************
 CPTables.h
 --------

CPTables methods for miscelaneous
conversion utilities
********************************/

class CPTables {
public:
	enum { OCTAVE = 12 };

	static uint16_t old_period_table[OCTAVE * 2];
	static uint16_t log_table[104];
	static int32_t linear_period_to_freq_tab[768];

	static int32_t get_old_period(uint16_t note, int32_t speed);
	static int32_t get_amiga_period(uint16_t note, int32_t fine);
	static int32_t get_linear_period(uint16_t note, int32_t fine);
	static int32_t get_linear_frequency(int32_t period);
	static int32_t get_old_frequency(int32_t period);
	static int32_t get_log_period(uint16_t note, int32_t p_c5freq);

	CPTables();
	~CPTables();
};

#endif
