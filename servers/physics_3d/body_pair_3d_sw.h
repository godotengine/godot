/*************************************************************************/
/*  body_pair_3d_sw.h                                                    */
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

#ifndef BODY_PAIR_3D_SW_H
#define BODY_PAIR_3D_SW_H

#include "body_3d_sw.h"
#include "constraint_3d_sw.h"
#include "core/templates/local_vector.h"
#include "soft_body_3d_sw.h"

class BodyContact3DSW : public Constraint3DSW {
protected:
	struct Contact {
		Vector3 position;
		Vector3 normal;
		int index_A, index_B;
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

	Vector3 sep_axis;
	bool collided = false;

	Space3DSW *space = nullptr;

	BodyContact3DSW(Body3DSW **p_body_ptr = nullptr, int p_body_count = 0) :
			Constraint3DSW(p_body_ptr, p_body_count) {
	}
};

class BodyPair3DSW : public BodyContact3DSW {
	enum {
		MAX_CONTACTS = 4
	};

	union {
		struct {
			Body3DSW *A;
			Body3DSW *B;
		};

		Body3DSW *_arr[2] = { nullptr, nullptr };
	};

	int shape_A = 0;
	int shape_B = 0;

	bool dynamic_A = false;
	bool dynamic_B = false;

	bool report_contacts_only = false;

	Vector3 offset_B; //use local A coordinates to avoid numerical issues on collision detection

	Contact contacts[MAX_CONTACTS];
	int contact_count = 0;

	static void _contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B, void *p_userdata);

	void contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B);

	void validate_contacts();
	bool _test_ccd(real_t p_step, Body3DSW *p_A, int p_shape_A, const Transform &p_xform_A, Body3DSW *p_B, int p_shape_B, const Transform &p_xform_B);

public:
	virtual bool setup(real_t p_step) override;
	virtual bool pre_solve(real_t p_step) override;
	virtual void solve(real_t p_step) override;

	BodyPair3DSW(Body3DSW *p_A, int p_shape_A, Body3DSW *p_B, int p_shape_B);
	~BodyPair3DSW();
};

class BodySoftBodyPair3DSW : public BodyContact3DSW {
	Body3DSW *body = nullptr;
	SoftBody3DSW *soft_body = nullptr;

	int body_shape = 0;

	bool body_dynamic = false;

	LocalVector<Contact> contacts;

	static void _contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B, void *p_userdata);

	void contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B);

	void validate_contacts();

public:
	virtual bool setup(real_t p_step) override;
	virtual bool pre_solve(real_t p_step) override;
	virtual void solve(real_t p_step) override;

	virtual SoftBody3DSW *get_soft_body_ptr(int p_index) const override { return soft_body; }
	virtual int get_soft_body_count() const override { return 1; }

	BodySoftBodyPair3DSW(Body3DSW *p_A, int p_shape_A, SoftBody3DSW *p_B);
	~BodySoftBodyPair3DSW();
};

#endif // BODY_PAIR_3D_SW_H
