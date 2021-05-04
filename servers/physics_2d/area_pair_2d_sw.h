/*************************************************************************/
/*  area_pair_2d_sw.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef AREA_PAIR_2D_SW_H
#define AREA_PAIR_2D_SW_H

#include "area_2d_sw.h"
#include "body_2d_sw.h"
#include "constraint_2d_sw.h"

class AreaPair2DSW : public Constraint2DSW {
	Body2DSW *body;
	Area2DSW *area;
	int body_shape;
	int area_shape;
	bool colliding;

public:
	bool setup(real_t p_step);
	void solve(real_t p_step);

	AreaPair2DSW(Body2DSW *p_body, int p_body_shape, Area2DSW *p_area, int p_area_shape);
	~AreaPair2DSW();
};

class Area2Pair2DSW : public Constraint2DSW {
	Area2DSW *area_a;
	Area2DSW *area_b;
	int shape_a;
	int shape_b;
	bool colliding;
	bool area_a_monitorable;
	bool area_b_monitorable;

public:
	bool setup(real_t p_step);
	void solve(real_t p_step);

	Area2Pair2DSW(Area2DSW *p_area_a, int p_shape_a, Area2DSW *p_area_b, int p_shape_b);
	~Area2Pair2DSW();
};

#endif // AREA_PAIR_2D_SW_H
