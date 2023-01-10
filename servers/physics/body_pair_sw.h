/**************************************************************************/
/*  body_pair_sw.h                                                        */
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

#ifndef BODY_PAIR_SW_H
#define BODY_PAIR_SW_H

#include "body_sw.h"
#include "constraint_sw.h"

class BodyPairSW : public ConstraintSW {
	enum {

		MAX_CONTACTS = 4
	};

	union {
		struct {
			BodySW *A;
			BodySW *B;
		};

		BodySW *_arr[2];
	};

	int shape_A;
	int shape_B;

	struct Contact {
		Vector3 position;
		Vector3 normal;
		Vector3 local_A, local_B;
		real_t acc_normal_impulse; // accumulated normal impulse (Pn)
		Vector3 acc_tangent_impulse; // accumulated tangent impulse (Pt)
		real_t acc_bias_impulse; // accumulated normal impulse for position bias (Pnb)
		real_t acc_bias_impulse_center_of_mass; // accumulated normal impulse for position bias applied to com
		real_t mass_normal;
		real_t bias;
		real_t bounce;

		real_t depth;
		bool active;
		Vector3 rA, rB; // Offset in world orientation with respect to center of mass
	};

	Vector3 offset_B; //use local A coordinates to avoid numerical issues on collision detection

	Vector3 sep_axis;
	Contact contacts[MAX_CONTACTS];
	int contact_count;
	bool collided;

	static void _contact_added_callback(const Vector3 &p_point_A, const Vector3 &p_point_B, void *p_userdata);

	void contact_added_callback(const Vector3 &p_point_A, const Vector3 &p_point_B);

	void validate_contacts();
	bool _test_ccd(real_t p_step, BodySW *p_A, int p_shape_A, const Transform &p_xform_A, BodySW *p_B, int p_shape_B, const Transform &p_xform_B);

	SpaceSW *space;

public:
	bool setup(real_t p_step);
	void solve(real_t p_step);

	BodyPairSW(BodySW *p_A, int p_shape_A, BodySW *p_B, int p_shape_B);
	~BodyPairSW();
};

#endif // BODY_PAIR_SW_H
