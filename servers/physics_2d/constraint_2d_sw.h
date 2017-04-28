/*************************************************************************/
/*  constraint_2d_sw.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef CONSTRAINT_2D_SW_H
#define CONSTRAINT_2D_SW_H

#include "body_2d_sw.h"

class Constraint2DSW : public RID_Data {

	Body2DSW **_body_ptr;
	int _body_count;
	uint64_t island_step;
	Constraint2DSW *island_next;
	Constraint2DSW *island_list_next;

	RID self;

protected:
	Constraint2DSW(Body2DSW **p_body_ptr = NULL, int p_body_count = 0) {
		_body_ptr = p_body_ptr;
		_body_count = p_body_count;
		island_step = 0;
	}

public:
	_FORCE_INLINE_ void set_self(const RID &p_self) { self = p_self; }
	_FORCE_INLINE_ RID get_self() const { return self; }

	_FORCE_INLINE_ uint64_t get_island_step() const { return island_step; }
	_FORCE_INLINE_ void set_island_step(uint64_t p_step) { island_step = p_step; }

	_FORCE_INLINE_ Constraint2DSW *get_island_next() const { return island_next; }
	_FORCE_INLINE_ void set_island_next(Constraint2DSW *p_next) { island_next = p_next; }

	_FORCE_INLINE_ Constraint2DSW *get_island_list_next() const { return island_list_next; }
	_FORCE_INLINE_ void set_island_list_next(Constraint2DSW *p_next) { island_list_next = p_next; }

	_FORCE_INLINE_ Body2DSW **get_body_ptr() const { return _body_ptr; }
	_FORCE_INLINE_ int get_body_count() const { return _body_count; }

	virtual bool setup(real_t p_step) = 0;
	virtual void solve(real_t p_step) = 0;

	virtual ~Constraint2DSW() {}
};

#endif // CONSTRAINT_2D_SW_H
