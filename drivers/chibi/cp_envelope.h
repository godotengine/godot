/*************************************************************************/
/*  cp_envelope.h                                                        */
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

#ifndef CP_ENVELOPE_H
#define CP_ENVELOPE_H

#include "cp_config.h"

/**envelope?
  *@author Juan Linietsky
  */

/******************************
 envelope.h
 ----------

Proovides an envelope, and basic functions
for it that can be used for both player
and interface
********************************/


class CPEnvelope {
	enum {

		MAX_POINTS=25
	};

	struct Point {
	
		uint16_t tick_offset;
		int16_t value;
	};

	Point node[MAX_POINTS];
	
	int8_t node_count;

	bool on;
	bool carry;

	bool loop_on;

	uint8_t loop_begin_node;
	uint8_t loop_end_node;

	bool sustain_loop_on;
	uint8_t sustain_loop_begin_node;
	uint8_t sustain_loop_end_node;

	
	int8_t max_value;
	int8_t min_value;
	

public:
	enum {
		
		NO_POINT=-5000,
	};
	
	void set_max(int8_t p_max) { max_value=p_max; }
	int8_t get_max() { return max_value; }
	void set_min(int8_t p_min) { min_value=p_min; }
	int8_t get_min() { return min_value; }

	uint8_t get_node_count();
	const Point& get_node(int p_idx);

	void set_position(int p_node,int p_x,int p_y);
	int add_position(int p_x,int p_y,bool p_move_loops=true);
	void del_position(int p_node);

	void set_loop_enabled(bool p_enabled);
	bool is_loop_enabled();
	void set_loop_begin(int pos);
	void set_loop_end(int pos);
	uint8_t get_loop_begin();
	uint8_t get_loop_end();

	void set_sustain_loop_enabled(bool p_enabled);
	bool is_sustain_loop_enabled();
	void set_sustain_loop_begin(int pos);
	void set_sustain_loop_end(int pos);
	uint8_t get_sustain_loop_begin();
	uint8_t get_sustain_loop_end();
	
	void set_enabled(bool p_enabled);
	bool is_enabled();
	
	void set_carry_enabled(bool p_enabled);
	bool is_carry_enabled();
	
	void reset();
	int get_height_at_pos(int pos);
	float get_interp_height_at_pos(float pos);
	
	
	CPEnvelope();
		
};

#endif
