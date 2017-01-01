/*************************************************************************/
/*  cp_envelope.cpp                                                      */
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
#include "cp_envelope.h"


CPEnvelope::CPEnvelope() {


	reset();
}

void CPEnvelope::reset() {



	on=false;
	carry=false;
	loop_on=false;
	loop_begin_node=0;
	loop_end_node=0;
	sustain_loop_on=false;
	sustain_loop_begin_node=0;
	sustain_loop_end_node=0;
	node_count=0;

}

int CPEnvelope::get_height_at_pos(int pos) {

	if (node_count && pos>node[node_count-1].tick_offset)
		return node[node_count-1].value;
	
	int begin_x,begin_y;
	int end_x,end_y,xdif;
	int count=0;
	int limit=-1;

	if (node_count<2) return NO_POINT;

	while ((count<node_count) && (limit==-1)) {

		if (node[count].tick_offset>=pos) limit=count;
		count++;
	}

	if (pos==0) return node[0].value;

	if (limit==-1) return NO_POINT;

	begin_x=node[limit-1].tick_offset;
	end_x=node[limit].tick_offset;
	begin_y=node[limit-1].value;
	end_y=node[limit].value;

	xdif=end_x-begin_x;
	return begin_y+((pos-begin_x)*(end_y-begin_y))/(xdif?xdif:1);
}

/*
int CPEnvelope::get_fx_height_at_pos(int pos) {

	if (node_count && pos>node[node_count-1].tick_offset)
		return node[node_count-1].value<<FX_HEIGHT_BITS;
	
	int begin_x,begin_y;
	int end_x,end_y,xdif;
	int count=0;
	int limit=-1;

	if (node_count<2) return NO_POINT;

	while ((count<node_count) && (limit==-1)) {

		if (node[count].tick_offset>=pos) limit=count;
		count++;
	}

	if (pos==0) return node[0].value<<FX_HEIGHT_BITS;

	if (limit==-1) return NO_POINT;

	begin_x=node[limit-1].tick_offset;
	end_x=node[limit].tick_offset;
	begin_y=node[limit-1].value;
	end_y=node[limit].value;

	xdif=end_x-begin_x;
	return (begin_y<<FX_HEIGHT_BITS)+((pos-begin_x)*(end_y-begin_y)*(int)(1<<FX_HEIGHT_BITS))/(xdif?xdif:1);
}
*/

float CPEnvelope::get_interp_height_at_pos(float pos) {

	if (node_count && pos>node[node_count-1].tick_offset)
		return node[node_count-1].value;

	int begin_x,begin_y;
	int end_x,end_y,xdif;
	int count=0;
	int limit=-1;

	if (node_count<2) return NO_POINT;

	while ((count<node_count) && (limit==-1)) {

		if (node[count].tick_offset>=pos) limit=count;
		count++;
	}

	if (pos==0) return node[0].value;

	if (limit==-1) return NO_POINT;

	begin_x=node[limit-1].tick_offset;
	end_x=node[limit].tick_offset;
	begin_y=node[limit-1].value;
	end_y=node[limit].value;

	xdif=end_x-begin_x;
	return begin_y+((pos-begin_x)*(end_y-begin_y))/(xdif?xdif:1);
}

void CPEnvelope::set_position(int p_node,int p_x,int p_y) {

	if (p_node>=node_count) return;
	


	if (p_node==0) {

		p_x=0;

	} else if (p_x<=node[p_node-1].tick_offset) {

		p_x=node[p_node-1].tick_offset+1;

	} else if ((p_node<(node_count-1)) && (p_x>=node[p_node+1].tick_offset)) {

		p_x=node[p_node+1].tick_offset-1;
	}

	if (p_x>=9999) p_x=9999;

	if (p_y>max_value) p_y=max_value;
	if (p_y<min_value) p_y=min_value;

	
	node[p_node].tick_offset=p_x;
        node[p_node].value=p_y;
	

	
}

int CPEnvelope::add_position(int p_x,int p_y,bool p_move_loops) {

	if (node_count==MAX_POINTS) return -1;

	
	int i,new_node;

	// if this is assigning an existing node, let's quit.
	for (i=0;i<node_count;i++) if (p_x==node[i].tick_offset) return -1;


	i=0;
	while ((i<node_count) && (p_x>=node[i].tick_offset)) i++;
	
	new_node=i;
	node_count++;

	if (p_move_loops) {
		if (loop_begin_node>=new_node) loop_begin_node++;
		if (loop_end_node>=new_node) loop_end_node++;
		if (sustain_loop_begin_node>=new_node) sustain_loop_begin_node++;
		if (sustain_loop_end_node>=new_node) sustain_loop_end_node++;
	}
	for (i=node_count-1;i>new_node;i--) node[i]=node[i-1];


		
        set_position(new_node,p_x,p_y);


	
	return new_node;
	
}

void CPEnvelope::set_loop_begin(int pos) {

	if ((pos<0) || (pos>=node_count)) return;


	
	loop_begin_node=pos;

	if (loop_end_node<loop_begin_node) loop_end_node=loop_begin_node;



}

void CPEnvelope::set_loop_end(int pos) {

	if ((pos<0) || (pos>=node_count)) return;


	
        loop_end_node=pos;
	
	if (loop_end_node<loop_begin_node) loop_begin_node=loop_end_node;


	

}


void CPEnvelope::set_sustain_loop_begin(int pos) {

	if ((pos<0) || (pos>=node_count)) return;


	
	sustain_loop_begin_node=pos;

	if (sustain_loop_end_node<sustain_loop_begin_node) sustain_loop_end_node=sustain_loop_begin_node;



}

void CPEnvelope::set_sustain_loop_end(int pos) {

	if ((pos<0) || (pos>=node_count)) return;


	
        sustain_loop_end_node=pos;
	
	if (sustain_loop_end_node<sustain_loop_begin_node) sustain_loop_begin_node=sustain_loop_end_node;



}

void CPEnvelope::set_loop_enabled(bool p_enabled) {
	
	loop_on=p_enabled;
}
bool CPEnvelope::is_loop_enabled() {
	
	return loop_on;
}


void CPEnvelope::set_sustain_loop_enabled(bool p_enabled) {
	
	sustain_loop_on=p_enabled;
}
bool CPEnvelope::is_sustain_loop_enabled() {
	
	return sustain_loop_on;
}

void CPEnvelope::del_position(int p_node) {

	if ((node_count<3) || (p_node<=0) || (p_node>=node_count)) return;


	
	int i;

	if (loop_begin_node>=p_node) loop_begin_node--;
	if (loop_end_node>=p_node) loop_end_node--;
	if (sustain_loop_begin_node>=p_node) sustain_loop_begin_node--;
	if (sustain_loop_end_node>=p_node) sustain_loop_end_node--;

	for (i=p_node;i<node_count-1;i++) node[i]=node[i+1];

	node_count--;
	

	
}

uint8_t CPEnvelope::get_loop_begin() {
	
	
	return loop_begin_node;
}
uint8_t CPEnvelope::get_loop_end() {
	
	return loop_end_node;	
}

uint8_t CPEnvelope::get_sustain_loop_begin() {
	
	
	return sustain_loop_begin_node;
}
uint8_t CPEnvelope::get_sustain_loop_end() {
	
	return sustain_loop_end_node;	
}



void CPEnvelope::set_enabled(bool p_enabled) {
	
	on=p_enabled;
}

bool CPEnvelope::is_enabled() {
	
	return on;	
}

void CPEnvelope::set_carry_enabled(bool p_enabled) {
	
	carry=p_enabled;
}
bool CPEnvelope::is_carry_enabled() {

	return carry;
}

uint8_t CPEnvelope::get_node_count() {
	
	return node_count;	
}

const CPEnvelope::Point& CPEnvelope::get_node(int p_idx) {
	
	if (p_idx<0 || p_idx>=node_count)
		return node[node_count-1];
	
	return node[p_idx];
	
}


