/**************************************************************************/
/*  godot_body_pair_3d.h                                                  */
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

#ifndef GODOT_BODY_PAIR_3D_H
#define GODOT_BODY_PAIR_3D_H

#include "godot_body_3d.h"
#include "godot_constraint_3d.h"
#include "godot_soft_body_3d.h"

#include "core/templates/local_vector.h"

// warning: the large volume of logging from these flags can reduce FPS. Turn them off for performance checks.
// #define PHYS_SOLVER_LOG		// per-frame stats
// #define PHYS_SOLVER_VERBOSE	// per-iteration stats - if this is enabled, you must also enable PHYS_SOLVER_LOG.

class GodotBodyContact3D : public GodotConstraint3D {
protected:
	struct FrictionTangent {
		Vector3 tangent;
		Vector3 prior_tangent;
		real_t acc_impulse = 0.0;
	};

	struct Contact {
		Vector3 position;
		Vector3 normal;
		int index_A = 0, index_B = 0;
		Vector3 local_A, local_B;
		Vector3 acc_impulse; // accumulated impulse - only one of the object's impulse is needed as impulse_a == -impulse_b
		real_t acc_normal_impulse = 0.0; // accumulated normal impulse (Pn)

		FrictionTangent friction_tangents[2]; // 2 tangent directions of friction by lambda.

		real_t acc_bias_impulse = 0.0; // accumulated normal impulse for position bias (Pnb)
		real_t acc_bias_impulse_center_of_mass = 0.0; // accumulated normal impulse for position bias applied to com
		real_t mass_normal = 0.0;
		real_t bias = 0.0;
		real_t bounce = 0.0;

		real_t depth = 0.0;
		bool active = false;
		bool used = false;
		Vector3 rA, rB; // Offset in world orientation with respect to center of mass

#ifdef PHYS_SOLVER_LOG
		int impulse_iterations = 0; // only for logging
#endif
	};

	Vector3 sep_axis;
	bool collided = false;
	bool check_ccd = false;

	GodotSpace3D *space = nullptr;

	GodotBodyContact3D(GodotBody3D **p_body_ptr = nullptr, int p_body_count = 0) :
			GodotConstraint3D(p_body_ptr, p_body_count) {
	}
};

class GodotBodyPair3D : public GodotBodyContact3D {
	enum {
		MAX_CONTACTS = 4
	};

	union {
		struct {
			GodotBody3D *A;
			GodotBody3D *B;
		};

		GodotBody3D *_arr[2] = { nullptr, nullptr };
	};

	int shape_A = 0;
	int shape_B = 0;

	bool collide_A = false;
	bool collide_B = false;

	bool report_contacts_only = false;

	Vector3 offset_B; //use local A coordinates to avoid numerical issues on collision detection
	real_t inv_mass_combined = 0.0;

	Contact contacts[MAX_CONTACTS];
	int contact_count = 0;
	// contacts from last frame, used during add-contact to persist accumulated impulse for warm start. (aka caching)
	Contact prior_contacts[MAX_CONTACTS];
	int prior_contact_count = 0;

// frame & iteration are just for verbose logging
#ifdef PHYS_SOLVER_LOG
	int frame_count = 0; // this is the frame count for this body pair, not overall game frames
	int iteration_count = 0; // solver iteration (resets each frame)
#endif

	static void _contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B, void *p_userdata);

	void contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B);

	bool _test_ccd(real_t p_step, GodotBody3D *p_A, int p_shape_A, const Transform3D &p_xform_A, GodotBody3D *p_B, int p_shape_B, const Transform3D &p_xform_B);
	void _solve_tangent(real_t p_step, Contact &c, int tangent_index, real_t jtMax);

public:
	virtual bool setup(real_t p_step) override;
	virtual bool pre_solve(real_t p_step) override;
	virtual void solve(real_t p_step) override;
	virtual void post_solve() override;

	GodotBodyPair3D(GodotBody3D *p_A, int p_shape_A, GodotBody3D *p_B, int p_shape_B);
	~GodotBodyPair3D();
};

class GodotBodySoftBodyPair3D : public GodotBodyContact3D {
	GodotBody3D *body = nullptr;
	GodotSoftBody3D *soft_body = nullptr;

	int body_shape = 0;

	bool body_collides = false;
	bool soft_body_collides = false;

	bool report_contacts_only = false;

	LocalVector<Contact> contacts;
	LocalVector<Contact> prior_contacts;

	static void _contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B, void *p_userdata);

	void contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B);
	void _solve_tangent(real_t p_step, Contact &c, int tangent_index, real_t jtMax, real_t inv_mass_combined);

public:
	virtual bool setup(real_t p_step) override;
	virtual bool pre_solve(real_t p_step) override;
	virtual void solve(real_t p_step) override;

	virtual GodotSoftBody3D *get_soft_body_ptr(int p_index) const override { return soft_body; }
	virtual int get_soft_body_count() const override { return 1; }

	GodotBodySoftBodyPair3D(GodotBody3D *p_A, int p_shape_A, GodotSoftBody3D *p_B);
	~GodotBodySoftBodyPair3D();
};

#endif // GODOT_BODY_PAIR_3D_H
