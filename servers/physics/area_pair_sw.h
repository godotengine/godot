/*************************************************************************/
/*  area_pair_sw.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef AREA_PAIR_SW_H
#define AREA_PAIR_SW_H

#include "area_sw.h"
#include "body_sw.h"
#include "constraint_sw.h"

class AreaPairSW : public ConstraintSW {
	BodySW *body;
	AreaSW *area;
	int body_shape;
	int area_shape;
	bool colliding;

public:
	bool setup(real_t p_step);
	void solve(real_t p_step);

	AreaPairSW(BodySW *p_body, int p_body_shape, AreaSW *p_area, int p_area_shape);
	~AreaPairSW();
};

class Area2PairSW : public ConstraintSW {
	AreaSW *area_a;
	AreaSW *area_b;
	int shape_a;
	int shape_b;
	bool colliding;

public:
	bool setup(real_t p_step);
	void solve(real_t p_step);

	Area2PairSW(AreaSW *p_area_a, int p_shape_a, AreaSW *p_area_b, int p_shape_b);
	~Area2PairSW();
};

#endif // AREA_PAIR__SW_H
