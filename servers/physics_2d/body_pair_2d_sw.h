/**************************************************************************/
/*  body_pair_2d_sw.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef BODY_PAIR_2D_SW_H
#define BODY_PAIR_2D_SW_H

#include "body_2d_sw.h"
#include "constraint_2d_sw.h"

class BodyPair2DSW : public Constraint2DSW {
	enum {
		MAX_CONTACTS = 2
	};
	union {
		struct {
			Body2DSW *A;
			Body2DSW *B;
		};

		Body2DSW *_arr[2];
	};

	int shape_A;
	int shape_B;

	Space2DSW *space;

	struct Contact {
		Vector2 position;
		Vector2 normal;
		Vector2 local_A, local_B;
		real_t acc_normal_impulse; // accumulated normal impulse (Pn)
		real_t acc_tangent_impulse; // accumulated tangent impulse (Pt)
		real_t acc_bias_impulse; // accumulated normal impulse for position bias (Pnb)
		real_t mass_normal, mass_tangent;
		real_t bias;

		real_t depth;
		bool active;
		Vector2 rA, rB;
		bool reused;
		real_t bounce;
	};

	Vector2 offset_B; //use local A coordinates to avoid numerical issues on collision detection

	Vector2 sep_axis;
	Contact contacts[MAX_CONTACTS];
	int contact_count;
	bool collided;
	bool oneway_disabled;
	int cc;

	bool _test_ccd(real_t p_step, Body2DSW *p_A, int p_shape_A, const Transform2D &p_xform_A, Body2DSW *p_B, int p_shape_B, const Transform2D &p_xform_B, bool p_swap_result = false);
	void _validate_contacts();
	static void _add_contact(const Vector2 &p_point_A, const Vector2 &p_point_B, void *p_self);
	_FORCE_INLINE_ void _contact_added_callback(const Vector2 &p_point_A, const Vector2 &p_point_B);

public:
	bool setup(real_t p_step);
	void solve(real_t p_step);

	BodyPair2DSW(Body2DSW *p_A, int p_shape_A, Body2DSW *p_B, int p_shape_B);
	~BodyPair2DSW();
};

#endif // BODY_PAIR_2D_SW_H
