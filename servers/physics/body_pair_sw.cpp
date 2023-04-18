/**************************************************************************/
/*  body_pair_sw.cpp                                                      */
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

#include "body_pair_sw.h"

#include "collision_solver_sw.h"
#include "core/os/os.h"
#include "space_sw.h"

#define MIN_VELOCITY 0.0001
#define MAX_BIAS_ROTATION (Math_PI / 8)

void BodyPairSW::_contact_added_callback(const Vector3 &p_point_A, const Vector3 &p_point_B, void *p_userdata) {
	BodyPairSW *pair = (BodyPairSW *)p_userdata;
	pair->contact_added_callback(p_point_A, p_point_B);
}

void BodyPairSW::contact_added_callback(const Vector3 &p_point_A, const Vector3 &p_point_B) {
	// check if we already have the contact

	//Vector3 local_A = A->get_inv_transform().xform(p_point_A);
	//Vector3 local_B = B->get_inv_transform().xform(p_point_B);

	Vector3 local_A = A->get_inv_transform().basis.xform(p_point_A);
	Vector3 local_B = B->get_inv_transform().basis.xform(p_point_B - offset_B);

	int new_index = contact_count;

	ERR_FAIL_COND(new_index >= (MAX_CONTACTS + 1));

	Contact contact;

	contact.acc_normal_impulse = 0;
	contact.acc_bias_impulse = 0;
	contact.acc_bias_impulse_center_of_mass = 0;
	contact.acc_tangent_impulse = Vector3();
	contact.local_A = local_A;
	contact.local_B = local_B;
	contact.normal = (p_point_A - p_point_B).normalized();
	contact.mass_normal = 0; // will be computed in setup()

	// attempt to determine if the contact will be reused
	real_t contact_recycle_radius = space->get_contact_recycle_radius();

	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		if (c.local_A.distance_squared_to(local_A) < (contact_recycle_radius * contact_recycle_radius) &&
				c.local_B.distance_squared_to(local_B) < (contact_recycle_radius * contact_recycle_radius)) {
			contact.acc_normal_impulse = c.acc_normal_impulse;
			contact.acc_bias_impulse = c.acc_bias_impulse;
			contact.acc_bias_impulse_center_of_mass = c.acc_bias_impulse_center_of_mass;
			contact.acc_tangent_impulse = c.acc_tangent_impulse;
			new_index = i;
			break;
		}
	}

	// figure out if the contact amount must be reduced to fit the new contact

	if (new_index == MAX_CONTACTS) {
		// remove the contact with the minimum depth

		int least_deep = -1;
		real_t min_depth = 1e10;

		for (int i = 0; i <= contact_count; i++) {
			Contact &c = (i == contact_count) ? contact : contacts[i];
			Vector3 global_A = A->get_transform().basis.xform(c.local_A);
			Vector3 global_B = B->get_transform().basis.xform(c.local_B) + offset_B;

			Vector3 axis = global_A - global_B;
			real_t depth = axis.dot(c.normal);

			if (depth < min_depth) {
				min_depth = depth;
				least_deep = i;
			}
		}

		ERR_FAIL_COND(least_deep == -1);

		if (least_deep < contact_count) { //replace the last deep contact by the new one

			contacts[least_deep] = contact;
		}

		return;
	}

	contacts[new_index] = contact;

	if (new_index == contact_count) {
		contact_count++;
	}
}

void BodyPairSW::validate_contacts() {
	//make sure to erase contacts that are no longer valid

	real_t contact_max_separation = space->get_contact_max_separation();
	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];

		Vector3 global_A = A->get_transform().basis.xform(c.local_A);
		Vector3 global_B = B->get_transform().basis.xform(c.local_B) + offset_B;
		Vector3 axis = global_A - global_B;
		real_t depth = axis.dot(c.normal);

		if (depth < -contact_max_separation || (global_B + c.normal * depth - global_A).length() > contact_max_separation) {
			// contact no longer needed, remove

			if ((i + 1) < contact_count) {
				// swap with the last one
				SWAP(contacts[i], contacts[contact_count - 1]);
			}

			i--;
			contact_count--;
		}
	}
}

bool BodyPairSW::_test_ccd(real_t p_step, BodySW *p_A, int p_shape_A, const Transform &p_xform_A, BodySW *p_B, int p_shape_B, const Transform &p_xform_B) {
	Vector3 motion = p_A->get_linear_velocity() * p_step;
	real_t mlen = motion.length();
	if (mlen < CMP_EPSILON) {
		return false;
	}

	Vector3 mnormal = motion / mlen;

	real_t min, max;
	p_A->get_shape(p_shape_A)->project_range(mnormal, p_xform_A, min, max);
	bool fast_object = mlen > (max - min) * 0.3; //going too fast in that direction

	if (!fast_object) { //did it move enough in this direction to even attempt raycast? let's say it should move more than 1/3 the size of the object in that axis
		return false;
	}

	//cast a segment from support in motion normal, in the same direction of motion by motion length
	//support is the worst case collision point, so real collision happened before
	Vector3 s = p_A->get_shape(p_shape_A)->get_support(p_xform_A.basis.xform(mnormal).normalized());
	Vector3 from = p_xform_A.xform(s);
	Vector3 to = from + motion;

	Transform from_inv = p_xform_B.affine_inverse();

	Vector3 local_from = from_inv.xform(from - mnormal * mlen * 0.1); //start from a little inside the bounding box
	Vector3 local_to = from_inv.xform(to);

	Vector3 rpos, rnorm;
	if (!p_B->get_shape(p_shape_B)->intersect_segment(local_from, local_to, rpos, rnorm)) {
		return false;
	}

	//shorten the linear velocity so it does not hit, but gets close enough, next frame will hit softly or soft enough
	Vector3 hitpos = p_xform_B.xform(rpos);

	real_t newlen = hitpos.distance_to(from) - (max - min) * 0.01;
	p_A->set_linear_velocity((mnormal * newlen) / p_step);

	return true;
}

real_t combine_bounce(BodySW *A, BodySW *B) {
	return CLAMP(A->get_bounce() + B->get_bounce(), 0, 1);
}

real_t combine_friction(BodySW *A, BodySW *B) {
	return ABS(MIN(A->get_friction(), B->get_friction()));
}

bool BodyPairSW::setup(real_t p_step) {
	//cannot collide
	if (!A->test_collision_mask(B) || A->has_exception(B->get_self()) || B->has_exception(A->get_self())) {
		collided = false;
		return false;
	}

	bool report_contacts_only = false;
	if ((A->get_mode() <= PhysicsServer::BODY_MODE_KINEMATIC) && (B->get_mode() <= PhysicsServer::BODY_MODE_KINEMATIC)) {
		if ((A->get_max_contacts_reported() > 0) || (B->get_max_contacts_reported() > 0)) {
			report_contacts_only = true;
		} else {
			collided = false;
			return false;
		}
	}

	offset_B = B->get_transform().get_origin() - A->get_transform().get_origin();

	validate_contacts();

	Vector3 offset_A = A->get_transform().get_origin();
	Transform xform_Au = Transform(A->get_transform().basis, Vector3());
	Transform xform_A = xform_Au * A->get_shape_transform(shape_A);

	Transform xform_Bu = B->get_transform();
	xform_Bu.origin -= offset_A;
	Transform xform_B = xform_Bu * B->get_shape_transform(shape_B);

	ShapeSW *shape_A_ptr = A->get_shape(shape_A);
	ShapeSW *shape_B_ptr = B->get_shape(shape_B);

	bool collided = CollisionSolverSW::solve_static(shape_A_ptr, xform_A, shape_B_ptr, xform_B, _contact_added_callback, this, &sep_axis);
	this->collided = collided;

	if (!collided) {
		//test ccd (currently just a raycast)

		if (A->is_continuous_collision_detection_enabled() && A->get_mode() > PhysicsServer::BODY_MODE_KINEMATIC && B->get_mode() <= PhysicsServer::BODY_MODE_KINEMATIC) {
			_test_ccd(p_step, A, shape_A, xform_A, B, shape_B, xform_B);
		}

		if (B->is_continuous_collision_detection_enabled() && B->get_mode() > PhysicsServer::BODY_MODE_KINEMATIC && A->get_mode() <= PhysicsServer::BODY_MODE_KINEMATIC) {
			_test_ccd(p_step, B, shape_B, xform_B, A, shape_A, xform_A);
		}

		return false;
	}

	real_t max_penetration = space->get_contact_max_allowed_penetration();

	real_t bias = (real_t)0.3;

	if (shape_A_ptr->get_custom_bias() || shape_B_ptr->get_custom_bias()) {
		if (shape_A_ptr->get_custom_bias() == 0) {
			bias = shape_B_ptr->get_custom_bias();
		} else if (shape_B_ptr->get_custom_bias() == 0) {
			bias = shape_A_ptr->get_custom_bias();
		} else {
			bias = (shape_B_ptr->get_custom_bias() + shape_A_ptr->get_custom_bias()) * 0.5;
		}
	}

	real_t inv_dt = 1.0 / p_step;

	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		c.active = false;

		Vector3 global_A = xform_Au.xform(c.local_A);
		Vector3 global_B = xform_Bu.xform(c.local_B);

		real_t depth = c.normal.dot(global_A - global_B);

		if (depth <= 0) {
			continue;
		}

#ifdef DEBUG_ENABLED

		if (space->is_debugging_contacts()) {
			space->add_debug_contact(global_A + offset_A);
			space->add_debug_contact(global_B + offset_A);
		}
#endif

		c.rA = global_A - A->get_center_of_mass();
		c.rB = global_B - B->get_center_of_mass() - offset_B;

		// contact query reporting...

		if (A->can_report_contacts()) {
			Vector3 crA = A->get_angular_velocity().cross(c.rA) + A->get_linear_velocity();
			A->add_contact(global_A, -c.normal, depth, shape_A, global_B, shape_B, B->get_instance_id(), B->get_self(), crA);
		}

		if (B->can_report_contacts()) {
			Vector3 crB = B->get_angular_velocity().cross(c.rB) + B->get_linear_velocity();
			B->add_contact(global_B, c.normal, depth, shape_B, global_A, shape_A, A->get_instance_id(), A->get_self(), crB);
		}

		if (report_contacts_only) {
			collided = false;
			continue;
		}

		c.active = true;

		// Precompute normal mass, tangent mass, and bias.
		Vector3 inertia_A = A->get_inv_inertia_tensor().xform(c.rA.cross(c.normal));
		Vector3 inertia_B = B->get_inv_inertia_tensor().xform(c.rB.cross(c.normal));
		real_t kNormal = A->get_inv_mass() + B->get_inv_mass();
		kNormal += c.normal.dot(inertia_A.cross(c.rA)) + c.normal.dot(inertia_B.cross(c.rB));
		c.mass_normal = 1.0f / kNormal;

		c.bias = -bias * inv_dt * MIN(0.0f, -depth + max_penetration);
		c.depth = depth;

		Vector3 j_vec = c.normal * c.acc_normal_impulse + c.acc_tangent_impulse;
		A->apply_impulse(c.rA + A->get_center_of_mass(), -j_vec);
		B->apply_impulse(c.rB + B->get_center_of_mass(), j_vec);
		c.acc_bias_impulse = 0;
		c.acc_bias_impulse_center_of_mass = 0;

		c.bounce = combine_bounce(A, B);
		if (c.bounce) {
			Vector3 crA = A->get_prev_angular_velocity().cross(c.rA);
			Vector3 crB = B->get_prev_angular_velocity().cross(c.rB);
			Vector3 dv = B->get_prev_linear_velocity() + crB - A->get_prev_linear_velocity() - crA;
			//normal impule
			c.bounce = c.bounce * dv.dot(c.normal);
		}
	}

	return true;
}

void BodyPairSW::solve(real_t p_step) {
	if (!collided) {
		return;
	}

	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		if (!c.active) {
			continue;
		}

		c.active = false; //try to deactivate, will activate itself if still needed

		//bias impulse

		Vector3 crbA = A->get_biased_angular_velocity().cross(c.rA);
		Vector3 crbB = B->get_biased_angular_velocity().cross(c.rB);
		Vector3 dbv = B->get_biased_linear_velocity() + crbB - A->get_biased_linear_velocity() - crbA;

		real_t vbn = dbv.dot(c.normal);

		if (Math::abs(-vbn + c.bias) > MIN_VELOCITY) {
			real_t jbn = (-vbn + c.bias) * c.mass_normal;
			real_t jbnOld = c.acc_bias_impulse;
			c.acc_bias_impulse = MAX(jbnOld + jbn, 0.0f);

			Vector3 jb = c.normal * (c.acc_bias_impulse - jbnOld);

			A->apply_bias_impulse(c.rA + A->get_center_of_mass(), -jb, MAX_BIAS_ROTATION / p_step);
			B->apply_bias_impulse(c.rB + B->get_center_of_mass(), jb, MAX_BIAS_ROTATION / p_step);

			crbA = A->get_biased_angular_velocity().cross(c.rA);
			crbB = B->get_biased_angular_velocity().cross(c.rB);
			dbv = B->get_biased_linear_velocity() + crbB - A->get_biased_linear_velocity() - crbA;

			vbn = dbv.dot(c.normal);

			if (Math::abs(-vbn + c.bias) > MIN_VELOCITY) {
				real_t jbn_com = (-vbn + c.bias) / (A->get_inv_mass() + B->get_inv_mass());
				real_t jbnOld_com = c.acc_bias_impulse_center_of_mass;
				c.acc_bias_impulse_center_of_mass = MAX(jbnOld_com + jbn_com, 0.0f);

				Vector3 jb_com = c.normal * (c.acc_bias_impulse_center_of_mass - jbnOld_com);

				A->apply_bias_impulse(A->get_center_of_mass(), -jb_com, 0.0f);
				B->apply_bias_impulse(B->get_center_of_mass(), jb_com, 0.0f);
			}

			c.active = true;
		}

		Vector3 crA = A->get_angular_velocity().cross(c.rA);
		Vector3 crB = B->get_angular_velocity().cross(c.rB);
		Vector3 dv = B->get_linear_velocity() + crB - A->get_linear_velocity() - crA;

		//normal impulse
		real_t vn = dv.dot(c.normal);

		if (Math::abs(vn) > MIN_VELOCITY) {
			real_t jn = -(c.bounce + vn) * c.mass_normal;
			real_t jnOld = c.acc_normal_impulse;
			c.acc_normal_impulse = MAX(jnOld + jn, 0.0f);

			Vector3 j = c.normal * (c.acc_normal_impulse - jnOld);

			A->apply_impulse(c.rA + A->get_center_of_mass(), -j);
			B->apply_impulse(c.rB + B->get_center_of_mass(), j);

			c.active = true;
		}

		//friction impulse

		real_t friction = combine_friction(A, B);

		Vector3 lvA = A->get_linear_velocity() + A->get_angular_velocity().cross(c.rA);
		Vector3 lvB = B->get_linear_velocity() + B->get_angular_velocity().cross(c.rB);

		Vector3 dtv = lvB - lvA;
		real_t tn = c.normal.dot(dtv);

		// tangential velocity
		Vector3 tv = dtv - c.normal * tn;
		real_t tvl = tv.length();

		if (tvl > MIN_VELOCITY) {
			tv /= tvl;

			Vector3 temp1 = A->get_inv_inertia_tensor().xform(c.rA.cross(tv));
			Vector3 temp2 = B->get_inv_inertia_tensor().xform(c.rB.cross(tv));

			real_t t = -tvl / (A->get_inv_mass() + B->get_inv_mass() + tv.dot(temp1.cross(c.rA) + temp2.cross(c.rB)));

			Vector3 jt = t * tv;

			Vector3 jtOld = c.acc_tangent_impulse;
			c.acc_tangent_impulse += jt;

			real_t fi_len = c.acc_tangent_impulse.length();
			real_t jtMax = c.acc_normal_impulse * friction;

			if (fi_len > CMP_EPSILON && fi_len > jtMax) {
				c.acc_tangent_impulse *= jtMax / fi_len;
			}

			jt = c.acc_tangent_impulse - jtOld;

			A->apply_impulse(c.rA + A->get_center_of_mass(), -jt);
			B->apply_impulse(c.rB + B->get_center_of_mass(), jt);

			c.active = true;
		}
	}
}

BodyPairSW::BodyPairSW(BodySW *p_A, int p_shape_A, BodySW *p_B, int p_shape_B) :
		ConstraintSW(_arr, 2) {
	A = p_A;
	B = p_B;
	shape_A = p_shape_A;
	shape_B = p_shape_B;
	space = A->get_space();
	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
	contact_count = 0;
	collided = false;
}

BodyPairSW::~BodyPairSW() {
	A->remove_constraint(this);
	B->remove_constraint(this);
}
