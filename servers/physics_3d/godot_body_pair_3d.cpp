/**************************************************************************/
/*  godot_body_pair_3d.cpp                                                */
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

#include "godot_body_pair_3d.h"

#include "godot_collision_solver_3d.h"
#include "godot_space_3d.h"

#include "core/os/os.h"

#define MIN_VELOCITY 0.0001
#define MAX_BIAS_ROTATION (Math_PI / 8)

void GodotBodyPair3D::_contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B, void *p_userdata) {
	GodotBodyPair3D *pair = static_cast<GodotBodyPair3D *>(p_userdata);
	pair->contact_added_callback(p_point_A, p_index_A, p_point_B, p_index_B);
}

void GodotBodyPair3D::contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B) {
	Vector3 local_A = A->get_inv_transform().basis.xform(p_point_A);
	Vector3 local_B = B->get_inv_transform().basis.xform(p_point_B - offset_B);

	// Collision solver may attempt to add the same contact twice when testing collisions with faces, we ignore the duplicate collisions.
	real_t contact_recycle_radius_sq = space->get_contact_recycle_radius() * space->get_contact_recycle_radius();
	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		if (c.local_A.distance_squared_to(local_A) < contact_recycle_radius_sq &&
				c.local_B.distance_squared_to(local_B) < contact_recycle_radius_sq) {
			return;
		}
	}

	int new_index = contact_count;

	ERR_FAIL_COND(new_index >= (MAX_CONTACTS + 1));

	Contact contact;
	contact.index_A = p_index_A;
	contact.index_B = p_index_B;
	contact.local_A = local_A;
	contact.local_B = local_B;
	contact.normal = (p_point_A - p_point_B).normalized();

	// see if this contact matches one from prior iteration. If so, copy the accumulated impulses to allow warm start.

	real_t max_separation_sq = space->get_contact_max_separation() * space->get_contact_max_separation();

	for (int i = 0; i < prior_contact_count; i++) {
		Contact &c = prior_contacts[i];
		if (!c.used && // c.used prevents us from matching the same cached contact to > 1 new contact.
				c.local_A.distance_squared_to(local_A) < max_separation_sq &&
				c.local_B.distance_squared_to(local_B) < max_separation_sq) {
			contact.acc_normal_impulse = c.acc_normal_impulse;
			contact.acc_bias_impulse = c.acc_bias_impulse;
			contact.acc_bias_impulse_center_of_mass = c.acc_bias_impulse_center_of_mass;
			contact.friction_tangents[0].acc_impulse = c.friction_tangents[0].acc_impulse;
			contact.friction_tangents[1].acc_impulse = c.friction_tangents[1].acc_impulse;
			contact.friction_tangents[0].prior_tangent = c.friction_tangents[0].tangent;
			contact.friction_tangents[1].prior_tangent = c.friction_tangents[1].tangent;
			c.used = true; // prevents the cached contact from being used twice.
			break;
		}
	}

	// Figure out if the contact amount must be reduced to fit the new contact.
	if (new_index == MAX_CONTACTS) {
		// Remove the contact with the minimum depth.

		const Basis &basis_A = A->get_transform().basis;
		const Basis &basis_B = B->get_transform().basis;

		int least_deep = -1;
		real_t min_depth;

		// Start with depth for new contact.
		{
			Vector3 global_A = basis_A.xform(contact.local_A);
			Vector3 global_B = basis_B.xform(contact.local_B) + offset_B;

			Vector3 axis = global_A - global_B;
			min_depth = axis.dot(contact.normal);
		}

		for (int i = 0; i < contact_count; i++) {
			const Contact &c = contacts[i];
			Vector3 global_A = basis_A.xform(c.local_A);
			Vector3 global_B = basis_B.xform(c.local_B) + offset_B;

			Vector3 axis = global_A - global_B;
			real_t depth = axis.dot(c.normal);

			if (depth < min_depth) {
				min_depth = depth;
				least_deep = i;
			}
		}

		if (least_deep > -1) {
			// Replace the least deep contact by the new one.
			contacts[least_deep] = contact;
		}

		return;
	}

	contacts[new_index] = contact;
	contact_count++;
}

// _test_ccd prevents tunneling by slowing down a high velocity body that is about to collide so that next frame it will be at an appropriate location to collide (i.e. slight overlap)
// Warning: the way velocity is adjusted down to cause a collision means the momentum will be weaker than it should for a bounce!
// Process: only proceed if body A's motion is high relative to its size.
// cast forward along motion vector to see if A is going to enter/pass B's collider next frame, only proceed if it does.
// adjust the velocity of A down so that it will just slightly intersect the collider instead of blowing right past it.
bool GodotBodyPair3D::_test_ccd(real_t p_step, GodotBody3D *p_A, int p_shape_A, const Transform3D &p_xform_A, GodotBody3D *p_B, int p_shape_B, const Transform3D &p_xform_B) {
	Vector3 motion = p_A->get_linear_velocity() * p_step;
	real_t mlen = motion.length();
	if (mlen < CMP_EPSILON) {
		return false;
	}

	Vector3 mnormal = motion / mlen;

	real_t min = 0.0, max = 0.0;
	p_A->get_shape(p_shape_A)->project_range(mnormal, p_xform_A, min, max);

	// Did it move enough in this direction to even attempt raycast?
	// Let's say it should move more than 1/3 the size of the object in that axis.
	bool fast_object = mlen > (max - min) * 0.3;
	if (!fast_object) {
		return false; // moving slow enough that there's no chance of tunneling.
	}

	// A is moving fast enough that tunneling might occur. See if it's really about to collide.

	// Cast a segment from support in motion normal, in the same direction of motion by motion length.
	// Support point will the farthest forward collision point along the movement vector.
	// i.e. the point that should hit B first if any collision does occur.

	// convert mnormal into body A's local xform because get_support requires (and returns) local coordinates.
	Vector3 s = p_A->get_shape(p_shape_A)->get_support(p_xform_A.basis.xform_inv(mnormal).normalized());
	Vector3 from = p_xform_A.xform(s);
	Vector3 to = from + motion;

	Transform3D from_inv = p_xform_B.affine_inverse();

	// Back up 10% of the per-frame motion behind the support point and use that as the beginning of our cast.
	// At high speeds, this may mean we're actually casting from well behind the body instead of inside it, which is odd. But it still works out.
	Vector3 local_from = from_inv.xform(from - motion * 0.1);
	Vector3 local_to = from_inv.xform(to);

	Vector3 rpos, rnorm;
	if (!p_B->get_shape(p_shape_B)->intersect_segment(local_from, local_to, rpos, rnorm, true)) {
		// there was no hit. Since the segment is the length of per-frame motion, this means the bodies will not
		// actually collide yet on next frame. We'll probably check again next frame once they're closer.
		return false;
	}

	// Shorten the linear velocity so it will collide next frame.
	Vector3 hitpos = p_xform_B.xform(rpos);

	real_t newlen = hitpos.distance_to(from) + (max - min) * 0.01; // adding 1% of body length to the distance between collision and support point should cause body A's support point to arrive just within B's collider next frame.

	p_A->set_linear_velocity((mnormal * newlen) / p_step);

	return true;
}

real_t combine_bounce(GodotBody3D *A, GodotBody3D *B) {
	return CLAMP(A->get_bounce() + B->get_bounce(), 0, 1);
}

real_t combine_friction(GodotBody3D *A, GodotBody3D *B) {
	return ABS(MIN(A->get_friction(), B->get_friction()));
}

void get_tangents(Vector3 &normal, Vector3 &outT1, Vector3 &outT2) {
	// tangent 1 is perpendicular
	if (abs(normal.x) > abs(normal.y)) {
		outT1 = Vector3(normal.z, 0.0f, -normal.x);
	} else {
		outT1 = Vector3(0.0f, normal.z, -normal.y);
	}
	outT1.normalize();
	outT2 = normal.cross(outT1);
}

bool GodotBodyPair3D::setup(real_t p_step) {
#ifdef PHYS_SOLVER_LOG
	frame_count++;
	iteration_count = 0;
#endif

	check_ccd = false;

	if (!A->interacts_with(B) || A->has_exception(B->get_self()) || B->has_exception(A->get_self())) {
		collided = false;
		return false;
	}

	collide_A = (A->get_mode() > PhysicsServer3D::BODY_MODE_KINEMATIC) && A->collides_with(B);
	collide_B = (B->get_mode() > PhysicsServer3D::BODY_MODE_KINEMATIC) && B->collides_with(A);

	report_contacts_only = false;
	if (!collide_A && !collide_B) {
		if ((A->get_max_contacts_reported() > 0) || (B->get_max_contacts_reported() > 0)) {
			report_contacts_only = true;
		} else {
			collided = false;
			return false;
		}
	}

	offset_B = B->get_transform().get_origin() - A->get_transform().get_origin();

	// reset contacts, keep prior frame's contacts as cache for warm start.
	std::swap(prior_contacts, contacts);
	prior_contact_count = contact_count;
	contact_count = 0;

	const Vector3 &offset_A = A->get_transform().get_origin();
	Transform3D xform_Au = Transform3D(A->get_transform().basis, Vector3());
	Transform3D xform_A = xform_Au * A->get_shape_transform(shape_A);

	Transform3D xform_Bu = B->get_transform();
	xform_Bu.origin -= offset_A;
	Transform3D xform_B = xform_Bu * B->get_shape_transform(shape_B);

	GodotShape3D *shape_A_ptr = A->get_shape(shape_A);
	GodotShape3D *shape_B_ptr = B->get_shape(shape_B);

	collided = GodotCollisionSolver3D::solve_static(shape_A_ptr, xform_A, shape_B_ptr, xform_B, _contact_added_callback, this, &sep_axis);

	if (!collided) {
		if (A->is_continuous_collision_detection_enabled() && collide_A) {
			check_ccd = true;
			return true;
		}

		if (B->is_continuous_collision_detection_enabled() && collide_B) {
			check_ccd = true;
			return true;
		}

		return false;
	}

	return true;
}

bool GodotBodyPair3D::pre_solve(real_t p_step) {
	if (!collided) {
		if (check_ccd) {
			const Vector3 &offset_A = A->get_transform().get_origin();
			Transform3D xform_Au = Transform3D(A->get_transform().basis, Vector3());
			Transform3D xform_A = xform_Au * A->get_shape_transform(shape_A);

			Transform3D xform_Bu = B->get_transform();
			xform_Bu.origin -= offset_A;
			Transform3D xform_B = xform_Bu * B->get_shape_transform(shape_B);

			if (A->is_continuous_collision_detection_enabled() && collide_A) {
				_test_ccd(p_step, A, shape_A, xform_A, B, shape_B, xform_B);
			}

			if (B->is_continuous_collision_detection_enabled() && collide_B) {
				_test_ccd(p_step, B, shape_B, xform_B, A, shape_A, xform_A);
			}
		}

		return false;
	}

	real_t max_penetration = space->get_contact_max_allowed_penetration(); // AKA slop. bias impulse will allow this much penetration without correction.

	real_t bias = space->get_contact_bias(); // to avoid overshoot, bias impulse will only attempt to correct for 80% of the penetration error per frame (unless overridden below)

	GodotShape3D *shape_A_ptr = A->get_shape(shape_A);
	GodotShape3D *shape_B_ptr = B->get_shape(shape_B);

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

	bool do_process = false;

	const Basis &basis_A = A->get_transform().basis;
	const Basis &basis_B = B->get_transform().basis;

	Basis zero_basis;
	zero_basis.set_zero();

	const Basis &inv_inertia_tensor_A = collide_A ? A->get_inv_inertia_tensor() : zero_basis;
	const Basis &inv_inertia_tensor_B = collide_B ? B->get_inv_inertia_tensor() : zero_basis;

	// usually inv_mass will not change, but recalculate every frame just in case kinematic mode or mass of the bodies has changed:
	inv_mass_combined = collide_A ? A->get_inv_mass() : 0.0;
	inv_mass_combined += collide_B ? B->get_inv_mass() : 0.0;

#ifdef PHYS_SOLVER_LOG
	print_line(vformat("Frame: %d | pre_solve | contact_count | %d | lin vel | %s | ang vel | %s", frame_count, contact_count, String(B->get_linear_velocity()), String(B->get_angular_velocity())));
#endif

	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		c.active = false;

		Vector3 global_A = basis_A.xform(c.local_A);
		Vector3 global_B = basis_B.xform(c.local_B) + offset_B;

		Vector3 axis = global_A - global_B;
		real_t depth = axis.dot(c.normal);

#ifdef PHYS_SOLVER_LOG
		print_line(vformat("Frame: %d | pre_solve | contact | %d | local_b | %s | depth | %f", frame_count, i, String(c.local_B), depth));
#endif

		if (depth <= 0.0) {
			continue;
		}

#ifdef DEBUG_ENABLED
		if (space->is_debugging_contacts()) {
			const Vector3 &offset_A = A->get_transform().get_origin();
			space->add_debug_contact(global_A + offset_A);
			space->add_debug_contact(global_B + offset_A);
		}
#endif

		c.rA = global_A - A->get_center_of_mass();
		c.rB = global_B - B->get_center_of_mass() - offset_B;

		get_tangents(c.normal, c.friction_tangents[0].tangent, c.friction_tangents[1].tangent);
#ifdef PHYS_SOLVER_VERBOSE
		print_line(vformat("Frame: %d | get_tangents | normal | %s | tan 1 | %s | tan 2 | %s", frame_count, String(c.normal), String(c.friction_tangents[0].tangent), String(c.friction_tangents[1].tangent)));
#endif

		// Precompute normal mass, tangent mass, and bias.
		Vector3 inertia_A = inv_inertia_tensor_A.xform(c.rA.cross(c.normal));
		Vector3 inertia_B = inv_inertia_tensor_B.xform(c.rB.cross(c.normal));
		real_t kNormal = inv_mass_combined + c.normal.dot(inertia_A.cross(c.rA)) + c.normal.dot(inertia_B.cross(c.rB));
		c.mass_normal = 1.0f / kNormal;

		c.bias = -bias * inv_dt * MIN(0.0f, -depth + max_penetration);
		c.depth = depth;

		// Create and apply the "warm start" impulse from prior frame's accumulated impulses. Benefits of warm start:
		// 1) if situation is very similar to last frame, the first solver iteration will find no adjustment necessary because this warm start impulse already did exactly what the solver would eventually have found.
		// 2) if the situation is so complex we couldn't reach a good solution in the iteration limit last frame, this allows us to continue where we left off instead of starting over and never getting a good solution.

		Vector3 j_vec = c.normal * c.acc_normal_impulse;

		// reproject prior friction lambdas onto the new friction tangents:
		Vector3 old_friction = c.friction_tangents[0].acc_impulse * c.friction_tangents[0].prior_tangent;
		c.friction_tangents[0].acc_impulse = old_friction.dot(c.friction_tangents[0].tangent);
		j_vec += c.friction_tangents[0].tangent * c.friction_tangents[0].acc_impulse;
		old_friction = c.friction_tangents[1].acc_impulse * c.friction_tangents[1].prior_tangent;
		c.friction_tangents[1].acc_impulse = old_friction.dot(c.friction_tangents[1].tangent);
		j_vec += c.friction_tangents[1].tangent * c.friction_tangents[1].acc_impulse;

		c.acc_impulse -= j_vec;

		// contact query reporting...

		if (A->can_report_contacts()) {
			Vector3 crA = A->get_angular_velocity().cross(c.rA) + A->get_linear_velocity();
			A->add_contact(global_A, -c.normal, depth, shape_A, global_B, shape_B, B->get_instance_id(), B->get_self(), crA, c.acc_impulse);
		}

		if (B->can_report_contacts()) {
			Vector3 crB = B->get_angular_velocity().cross(c.rB) + B->get_linear_velocity();
			B->add_contact(global_B, c.normal, depth, shape_B, global_A, shape_A, A->get_instance_id(), A->get_self(), crB, -c.acc_impulse);
		}

		if (report_contacts_only) {
			collided = false;
			continue;
		}

		// Each contact starts active so it gets at least 1 solver iteration. On each iteration contacts may go inactive and will not be evaluated on any more iterations that frame.
		c.active = true;
		do_process = true;

		if (collide_A) {
			A->apply_impulse(-j_vec, c.rA + A->get_center_of_mass());
		}
		if (collide_B) {
#ifdef PHYS_SOLVER_VERBOSE
			print_line(vformat("Frame: %d.%d | warm | %s | %s", frame_count, iteration_count, String(j_vec), String(c.local_B)));
#endif
			B->apply_impulse(j_vec, c.rB + B->get_center_of_mass());
		}
		// end of warm start.

		c.bounce = combine_bounce(A, B);
		if (c.bounce) {
			Vector3 crA = A->get_prev_angular_velocity().cross(c.rA);
			Vector3 crB = B->get_prev_angular_velocity().cross(c.rB);
			Vector3 dv = B->get_prev_linear_velocity() + crB - A->get_prev_linear_velocity() - crA;
			c.bounce = c.bounce * dv.dot(c.normal);
		}
	}

	return do_process;
}

/*	The solver uses the sequential impulses approach. Key concepts:
 *	 - will perform several iterations each frame trying to find a balance of impulses between all contact points.
 *	 - in a very simple collision only 1 iteration should be needed, but when rigidbodies are stacked we might get so many interacting contacts that we can't fully solve it within 1 frame's iterations.
 *	 - there are 3 types of impulses: bias (to correct for penetration), normal (to eliminate velocity), & tangent (to apply friction)
 *	 - since very often each frame's solution will be similar to the prior one, the accumulated impulses are cached and used to "warm start" the next frame
 *
 *	For further information, see GDC talks by Erin Catto, and many useful posts in various forums by Dirk Gregorius.
 */

void GodotBodyPair3D::solve(real_t p_step) {
	if (!collided) {
		return;
	}

#ifdef PHYS_SOLVER_LOG
	iteration_count++;
#endif

	const real_t max_bias_av = MAX_BIAS_ROTATION / p_step;

	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		if (!c.active) {
			continue;
		}

		c.active = false; //try to deactivate, will reactivate if any impulse gets applied this solver iteration

		//bias impulse (fixes penetration)

		Vector3 crbA = A->get_biased_angular_velocity().cross(c.rA);
		Vector3 crbB = B->get_biased_angular_velocity().cross(c.rB);
		Vector3 dbv = B->get_biased_linear_velocity() + crbB - A->get_biased_linear_velocity() - crbA;

		real_t vbn = dbv.dot(c.normal);

		if (Math::abs(-vbn + c.bias) > MIN_VELOCITY) {
			real_t jbn = (-vbn + c.bias) * c.mass_normal;
			real_t jbnOld = c.acc_bias_impulse;
			c.acc_bias_impulse = MAX(jbnOld + jbn, 0.0f);

			Vector3 jb = c.normal * (c.acc_bias_impulse - jbnOld);

			if (!jb.is_zero_approx()) {
				if (collide_A) {
					A->apply_bias_impulse(-jb, c.rA + A->get_center_of_mass(), max_bias_av);
				}
				if (collide_B) {
#ifdef PHYS_SOLVER_VERBOSE
					print_line(vformat("Frame: %d.%d | bias | %s | %s | %f", frame_count, iteration_count, String(jb), String(c.local_B), max_bias_av));
#endif
					B->apply_bias_impulse(jb, c.rB + B->get_center_of_mass(), max_bias_av);
				}
				c.active = true;
			}

			crbA = A->get_biased_angular_velocity().cross(c.rA);
			crbB = B->get_biased_angular_velocity().cross(c.rB);
			dbv = B->get_biased_linear_velocity() + crbB - A->get_biased_linear_velocity() - crbA;

			vbn = dbv.dot(c.normal);

			if (Math::abs(-vbn + c.bias) > MIN_VELOCITY) {
				real_t jbn_com = (-vbn + c.bias) / inv_mass_combined;
				real_t jbnOld_com = c.acc_bias_impulse_center_of_mass;
				c.acc_bias_impulse_center_of_mass = MAX(jbnOld_com + jbn_com, 0.0f);

				Vector3 jb_com = c.normal * (c.acc_bias_impulse_center_of_mass - jbnOld_com);

				if (!jb_com.is_zero_approx()) {
					if (collide_A) {
						A->apply_bias_impulse(-jb_com, A->get_center_of_mass(), 0.0f);
					}
					if (collide_B) {
#ifdef PHYS_SOLVER_VERBOSE
						print_line(vformat("Frame: %d.%d | bias_com | %s | centerOfMass", frame_count, iteration_count, String(jb_com)));
#endif
						B->apply_bias_impulse(jb_com, B->get_center_of_mass(), 0.0f);
					}
					c.active = true;
				}
			}
		}

		Vector3 crA = A->get_angular_velocity().cross(c.rA);
		Vector3 crB = B->get_angular_velocity().cross(c.rB);
		Vector3 dv = B->get_linear_velocity() + crB - A->get_linear_velocity() - crA;

		//normal impulse (eliminates velocity)
		real_t vn = dv.dot(c.normal);

		// warning: comparing to MIN_VELOCITY can cause slight difference between accumulated normal impulse and gravity to persist frame-over frame,
		// causing linear velocity to grow, causing more and more bias corrections over time. But if we go lower than MIN_VELOCITY, it takes a lot more iterations to settle.
		if (Math::abs(vn) > MIN_VELOCITY) {
			real_t jn = -(c.bounce + vn) * c.mass_normal;
			real_t jnOld = c.acc_normal_impulse;
			c.acc_normal_impulse = MAX(jnOld + jn, 0.0f);

			Vector3 j = c.normal * (c.acc_normal_impulse - jnOld);

			if (collide_A) {
				A->apply_impulse(-j, c.rA + A->get_center_of_mass());
			}
			if (collide_B) {
#ifdef PHYS_SOLVER_VERBOSE
				print_line(vformat("Frame: %d.%d | normal_impulse | %s | %s", frame_count, iteration_count, String(j), String(c.local_B)));
#endif
				B->apply_impulse(j, c.rB + B->get_center_of_mass());
			}
			c.acc_impulse -= j;

			c.active = true;
		}
	}

	// tangent impulses (friction)
	// By doing friction after all contacts have calculated normal impulses we prevent friction impulses from causing an imbalance between the normal impulses.

	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		if (!c.active) {
			continue;
		}

		real_t friction = combine_friction(A, B);
		real_t jtMax = c.acc_normal_impulse * friction; // friction is greater the higher the normal impulse. e.g. the bottom box in a stack is harder to shift than the top.

		_solve_tangent(p_step, c, 0, jtMax);
		_solve_tangent(p_step, c, 1, jtMax);
	}

#ifdef PHYS_SOLVER_LOG
	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		if (c.active) {
			c.impulse_iterations++;
		}
	}
#endif
}

void GodotBodyPair3D::post_solve() {
#ifdef PHYS_SOLVER_LOG
	int max_iter = 0;
	for (int i = 0; i < contact_count; i++) {
		Contact &c = contacts[i];
		print_line(vformat("Frame: %d | post_solve | contact | %d | iterations | %d | accum norm | %f | accum tan 1 | %f | accum tan 2 | %f", frame_count, i, c.impulse_iterations, c.acc_normal_impulse, c.friction_tangents[0].acc_impulse, c.friction_tangents[1].acc_impulse));
		max_iter = MAX(max_iter, c.impulse_iterations);
	}
	print_line(vformat("Frame: %d | post_solve | iterations | %d", frame_count, max_iter));
#endif
}

void GodotBodyPair3D::_solve_tangent(real_t p_step, Contact &c, int tangent_index, real_t jtMax) {
	// these tensors could be calculated in pre_solve instead of each solve_tangent. Worth it?
	Basis zero_basis;
	zero_basis.set_zero();
	const Basis &inv_inertia_tensor_A = collide_A ? A->get_inv_inertia_tensor() : zero_basis;
	const Basis &inv_inertia_tensor_B = collide_B ? B->get_inv_inertia_tensor() : zero_basis;

	FrictionTangent &tan = c.friction_tangents[tangent_index];
	Vector3 tv = tan.tangent;
	Vector3 ra_cross_tan = c.rA.cross(tv);
	Vector3 rb_cross_tan = c.rB.cross(tv);
	real_t jv = tv.dot(A->get_linear_velocity() - B->get_linear_velocity());
	// angular velocity:
	if (collide_A) {
		jv += ra_cross_tan.dot(A->get_angular_velocity());
	}
	if (collide_B) {
		jv -= rb_cross_tan.dot(B->get_angular_velocity());
	}

	Vector3 temp1 = inv_inertia_tensor_A.xform(ra_cross_tan);
	Vector3 temp2 = inv_inertia_tensor_B.xform(rb_cross_tan);
	real_t inv_eff_mass = (inv_mass_combined + tv.dot(temp1.cross(c.rA) + temp2.cross(c.rB)));

	real_t lambda = jv / inv_eff_mass;

	if (abs(lambda) > MIN_VELOCITY) {
		real_t jtOld = tan.acc_impulse;
		tan.acc_impulse = CLAMP(jtOld + lambda, -jtMax, jtMax);

		lambda = tan.acc_impulse - jtOld;
		Vector3 jt = lambda * tv;

		if (collide_A) {
			A->apply_impulse(-jt, c.rA + A->get_center_of_mass());
		}
		if (collide_B) {
#ifdef PHYS_SOLVER_VERBOSE
			print_line(vformat("Frame: %d.%d | tangent_impulse_%d | %s | %s", frame_count, iteration_count, tangent_index, String(jt), String(c.local_B)));
#endif
			B->apply_impulse(jt, c.rB + B->get_center_of_mass());
		}
		c.acc_impulse -= jt;

		c.active = true;
	}
}

GodotBodyPair3D::GodotBodyPair3D(GodotBody3D *p_A, int p_shape_A, GodotBody3D *p_B, int p_shape_B) :
		GodotBodyContact3D(_arr, 2) {
	A = p_A;
	B = p_B;
	shape_A = p_shape_A;
	shape_B = p_shape_B;
	space = A->get_space();
	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

GodotBodyPair3D::~GodotBodyPair3D() {
	A->remove_constraint(this);
	B->remove_constraint(this);
}

void GodotBodySoftBodyPair3D::_contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B, void *p_userdata) {
	GodotBodySoftBodyPair3D *pair = static_cast<GodotBodySoftBodyPair3D *>(p_userdata);
	pair->contact_added_callback(p_point_A, p_index_A, p_point_B, p_index_B);
}

void GodotBodySoftBodyPair3D::contact_added_callback(const Vector3 &p_point_A, int p_index_A, const Vector3 &p_point_B, int p_index_B) {
	Vector3 local_A = body->get_inv_transform().xform(p_point_A);
	Vector3 local_B = p_point_B - soft_body->get_node_position(p_index_B);

	// Collision solver may attempt to add the same contact twice when testing collisions with faces, we ignore the duplicate collisions.
	real_t contact_recycle_radius_sq = space->get_contact_recycle_radius() * space->get_contact_recycle_radius();
	uint32_t contact_count = contacts.size();
	for (uint32_t contact_index = 0; contact_index < contact_count; ++contact_index) {
		Contact &c = contacts[contact_index];
		if (c.local_A.distance_squared_to(local_A) < contact_recycle_radius_sq &&
				c.local_B.distance_squared_to(local_B) < contact_recycle_radius_sq) {
			return;
		}
	}

	Contact contact;
	contact.index_A = p_index_A;
	contact.index_B = p_index_B;
	contact.local_A = local_A;
	contact.local_B = local_B;
	contact.normal = (p_point_A - p_point_B).normalized();

	// see if this contact matches one from prior iteration. If so, copy the accumulated impulses to allow warm start.
	real_t max_separation_sq = space->get_contact_max_separation() * space->get_contact_max_separation();

	for (uint32_t contact_index = 0; contact_index < contact_count; ++contact_index) {
		Contact &c = contacts[contact_index];
		if (c.index_B == p_index_B) {
			if (!c.used && // c.used prevents us from matching the same cached contact to > 1 new contact.
					c.local_A.distance_squared_to(local_A) < max_separation_sq &&
					c.local_B.distance_squared_to(local_B) < max_separation_sq) {
				contact.acc_normal_impulse = c.acc_normal_impulse;
				contact.acc_bias_impulse = c.acc_bias_impulse;
				contact.acc_bias_impulse_center_of_mass = c.acc_bias_impulse_center_of_mass;
				contact.friction_tangents[0].acc_impulse = c.friction_tangents[0].acc_impulse;
				contact.friction_tangents[1].acc_impulse = c.friction_tangents[1].acc_impulse;
				contact.friction_tangents[0].prior_tangent = c.friction_tangents[0].tangent;
				contact.friction_tangents[1].prior_tangent = c.friction_tangents[1].tangent;
				c.used = true; // prevents the cached contact from being used twice.
				break;
			}
		}
	}

	contacts.push_back(contact);
}

bool GodotBodySoftBodyPair3D::setup(real_t p_step) {
	if (!body->interacts_with(soft_body) || body->has_exception(soft_body->get_self()) || soft_body->has_exception(body->get_self())) {
		collided = false;
		return false;
	}

	body_collides = (body->get_mode() > PhysicsServer3D::BODY_MODE_KINEMATIC) && body->collides_with(soft_body);
	soft_body_collides = soft_body->collides_with(body);

	if (!body_collides && !soft_body_collides) {
		if (body->get_max_contacts_reported() > 0) {
			report_contacts_only = true;
		} else {
			collided = false;
			return false;
		}
	}

	const Transform3D &xform_Au = body->get_transform();
	Transform3D xform_A = xform_Au * body->get_shape_transform(body_shape);

	Transform3D xform_Bu = soft_body->get_transform();
	Transform3D xform_B = xform_Bu * soft_body->get_shape_transform(0);

	// reset contacts, keep prior frame's contacts as cache for warm start.
	SWAP(prior_contacts, contacts);
	contacts.reset();

	GodotShape3D *shape_A_ptr = body->get_shape(body_shape);
	GodotShape3D *shape_B_ptr = soft_body->get_shape(0);

	collided = GodotCollisionSolver3D::solve_static(shape_A_ptr, xform_A, shape_B_ptr, xform_B, _contact_added_callback, this, &sep_axis);

	return collided;
}

bool GodotBodySoftBodyPair3D::pre_solve(real_t p_step) {
	if (!collided) {
		return false;
	}

	real_t max_penetration = space->get_contact_max_allowed_penetration(); // AKA slop. bias impulse will allow this much penetration without correction.

	real_t bias = space->get_contact_bias(); // to avoid overshoot, bias impulse will only attempt to correct for 80% of the penetration error per frame (unless overridden below)

	GodotShape3D *shape_A_ptr = body->get_shape(body_shape);

	if (shape_A_ptr->get_custom_bias()) {
		bias = shape_A_ptr->get_custom_bias();
	}

	real_t inv_dt = 1.0 / p_step;

	bool do_process = false;

	const Transform3D &transform_A = body->get_transform();

	Basis zero_basis;
	zero_basis.set_zero();

	const Basis &body_inv_inertia_tensor = body_collides ? body->get_inv_inertia_tensor() : zero_basis;

	real_t body_inv_mass = body_collides ? body->get_inv_mass() : 0.0;

	uint32_t contact_count = contacts.size();
	for (uint32_t contact_index = 0; contact_index < contact_count; ++contact_index) {
		Contact &c = contacts[contact_index];
		c.active = false;

		real_t node_inv_mass = soft_body_collides ? soft_body->get_node_inv_mass(c.index_B) : 0.0;
		if ((node_inv_mass == 0.0) && (body_inv_mass == 0.0)) {
			continue;
		}

		Vector3 global_A = transform_A.xform(c.local_A);
		Vector3 global_B = soft_body->get_node_position(c.index_B) + c.local_B;
		Vector3 axis = global_A - global_B;
		real_t depth = axis.dot(c.normal);

		if (depth <= 0.0) {
			continue;
		}

#ifdef DEBUG_ENABLED
		if (space->is_debugging_contacts()) {
			space->add_debug_contact(global_A);
			space->add_debug_contact(global_B);
		}
#endif

		c.rA = global_A - transform_A.origin - body->get_center_of_mass();
		c.rB = global_B;

		get_tangents(c.normal, c.friction_tangents[0].tangent, c.friction_tangents[1].tangent);

		// Precompute normal mass, tangent mass, and bias.
		Vector3 inertia_A = body_inv_inertia_tensor.xform(c.rA.cross(c.normal));
		real_t kNormal = body_inv_mass + node_inv_mass;
		kNormal += c.normal.dot(inertia_A.cross(c.rA));
		c.mass_normal = 1.0f / kNormal;

		c.bias = -bias * inv_dt * MIN(0.0f, -depth + max_penetration);
		c.depth = depth;

		// Create and apply the "warm start" impulse from prior frame's accumulated impulses. Benefits of warm start:
		// 1) if situation is very similar to last frame, the first solver iteration will find no adjustment necessary because this warm start impulse already did exactly what the solver would eventually have found.
		// 2) if the situation is so complex we couldn't reach a good solution in the iteration limit last frame, this allows us to continue where we left off instead of starting over and never getting a good solution.

		Vector3 j_vec = c.normal * c.acc_normal_impulse;

		// reproject prior friction lambdas onto the new friction tangents:
		Vector3 old_friction = c.friction_tangents[0].acc_impulse * c.friction_tangents[0].prior_tangent;
		c.friction_tangents[0].acc_impulse = old_friction.dot(c.friction_tangents[0].tangent);
		j_vec += c.friction_tangents[0].tangent * c.friction_tangents[0].acc_impulse;
		old_friction = c.friction_tangents[1].acc_impulse * c.friction_tangents[1].prior_tangent;
		c.friction_tangents[1].acc_impulse = old_friction.dot(c.friction_tangents[1].tangent);
		j_vec += c.friction_tangents[1].tangent * c.friction_tangents[1].acc_impulse;

		if (body_collides) {
			body->apply_impulse(-j_vec, c.rA + body->get_center_of_mass());
		}
		if (soft_body_collides) {
			soft_body->apply_node_impulse(c.index_B, j_vec);
		}
		c.acc_impulse -= j_vec;

		if (body->can_report_contacts()) {
			Vector3 crA = body->get_angular_velocity().cross(c.rA) + body->get_linear_velocity();
			body->add_contact(global_A, -c.normal, depth, body_shape, global_B, 0, soft_body->get_instance_id(), soft_body->get_self(), crA, c.acc_impulse);
		}

		if (report_contacts_only) {
			collided = false;
			continue;
		}

		// Each contact starts active so it gets at least 1 solver iteration. On each iteration contacts may go inactive and will not be evaluated on any more iterations that frame.
		c.active = true;
		do_process = true;

		if (body_collides) {
			body->set_active(true);
		}

		c.bounce = body->get_bounce();

		if (c.bounce) {
			Vector3 crA = body->get_angular_velocity().cross(c.rA);
			Vector3 dv = soft_body->get_node_velocity(c.index_B) - body->get_linear_velocity() - crA;

			// Normal impulse.
			c.bounce = c.bounce * dv.dot(c.normal);
		}
	}

	return do_process;
}

void GodotBodySoftBodyPair3D::solve(real_t p_step) {
	if (!collided) {
		return;
	}

	const real_t max_bias_av = MAX_BIAS_ROTATION / p_step;

	real_t body_inv_mass = body_collides ? body->get_inv_mass() : 0.0;

	uint32_t contact_count = contacts.size();
	for (uint32_t contact_index = 0; contact_index < contact_count; ++contact_index) {
		Contact &c = contacts[contact_index];
		if (!c.active) {
			continue;
		}

		c.active = false; //try to deactivate, will reactivate if any impulse gets applied this solver iteration

		real_t node_inv_mass = soft_body_collides ? soft_body->get_node_inv_mass(c.index_B) : 0.0;

		// Bias impulse (fixes penetration)
		Vector3 crbA = body->get_biased_angular_velocity().cross(c.rA);
		Vector3 dbv = soft_body->get_node_biased_velocity(c.index_B) - body->get_biased_linear_velocity() - crbA;

		real_t vbn = dbv.dot(c.normal);

		if (Math::abs(-vbn + c.bias) > MIN_VELOCITY) {
			real_t jbn = (-vbn + c.bias) * c.mass_normal;
			real_t jbnOld = c.acc_bias_impulse;
			c.acc_bias_impulse = MAX(jbnOld + jbn, 0.0f);

			Vector3 jb = c.normal * (c.acc_bias_impulse - jbnOld);

			if (!jb.is_zero_approx()) {
				if (body_collides) {
					body->apply_bias_impulse(-jb, c.rA + body->get_center_of_mass(), max_bias_av);
				}
				if (soft_body_collides) {
					soft_body->apply_node_bias_impulse(c.index_B, jb);
				}
				c.active = true;
			}

			crbA = body->get_biased_angular_velocity().cross(c.rA);
			dbv = soft_body->get_node_biased_velocity(c.index_B) - body->get_biased_linear_velocity() - crbA;

			vbn = dbv.dot(c.normal);

			if (Math::abs(-vbn + c.bias) > MIN_VELOCITY) {
				real_t jbn_com = (-vbn + c.bias) / (body_inv_mass + node_inv_mass);
				real_t jbnOld_com = c.acc_bias_impulse_center_of_mass;
				c.acc_bias_impulse_center_of_mass = MAX(jbnOld_com + jbn_com, 0.0f);

				Vector3 jb_com = c.normal * (c.acc_bias_impulse_center_of_mass - jbnOld_com);

				if (!jb_com.is_zero_approx()) {
					if (body_collides) {
						body->apply_bias_impulse(-jb_com, body->get_center_of_mass(), 0.0f);
					}
					if (soft_body_collides) {
						soft_body->apply_node_bias_impulse(c.index_B, jb_com);
					}
					c.active = true;
				}
			}
		}

		Vector3 crA = body->get_angular_velocity().cross(c.rA);
		Vector3 dv = soft_body->get_node_velocity(c.index_B) - body->get_linear_velocity() - crA;

		// Normal impulse (eliminates velocity)
		real_t vn = dv.dot(c.normal);

		if (Math::abs(vn) > MIN_VELOCITY) {
			real_t jn = -(c.bounce + vn) * c.mass_normal;
			real_t jnOld = c.acc_normal_impulse;
			c.acc_normal_impulse = MAX(jnOld + jn, 0.0f);

			Vector3 j = c.normal * (c.acc_normal_impulse - jnOld);

			if (body_collides) {
				body->apply_impulse(-j, c.rA + body->get_center_of_mass());
			}
			if (soft_body_collides) {
				soft_body->apply_node_impulse(c.index_B, j);
			}
			c.acc_impulse -= j;

			c.active = true;
		}
	}

	// tangent impulses (friction)
	for (uint32_t contact_index = 0; contact_index < contact_count; ++contact_index) {
		Contact &c = contacts[contact_index];
		if (!c.active) {
			continue;
		}

		real_t node_inv_mass = soft_body_collides ? soft_body->get_node_inv_mass(c.index_B) : 0.0;
		real_t friction = body->get_friction();
		real_t jtMax = c.acc_normal_impulse * friction; // friction is greater the higher the normal impulse. e.g. the bottom box in a stack is harder to shift than the top.

		_solve_tangent(p_step, c, 0, jtMax, node_inv_mass + body_inv_mass);
		_solve_tangent(p_step, c, 1, jtMax, node_inv_mass + body_inv_mass);
	}
}

void GodotBodySoftBodyPair3D::_solve_tangent(real_t p_step, Contact &c, int tangent_index, real_t jtMax, real_t inv_mass_combined) {
	// these tensors could be calculated in pre_solve instead of each solve_tangent. Worth it?
	Basis zero_basis;
	zero_basis.set_zero();
	const Basis &body_inv_inertia_tensor = body_collides ? body->get_inv_inertia_tensor() : zero_basis;

	FrictionTangent &tan = c.friction_tangents[tangent_index];
	Vector3 tv = tan.tangent;
	Vector3 ra_cross_tan = c.rA.cross(tv);
	real_t jv = tv.dot(body->get_linear_velocity() - soft_body->get_node_velocity(c.index_B));
	// angular velocity:
	if (body_collides) {
		jv += ra_cross_tan.dot(body->get_angular_velocity());
	}
	// soft body doesn't add any angular velocity

	Vector3 temp1 = body_inv_inertia_tensor.xform(ra_cross_tan);
	real_t inv_eff_mass = (inv_mass_combined + tv.dot(temp1.cross(c.rA)));

	real_t lambda = jv / inv_eff_mass;

	if (abs(lambda) > MIN_VELOCITY) {
		real_t jtOld = tan.acc_impulse;
		tan.acc_impulse = CLAMP(jtOld + lambda, -jtMax, jtMax);

		lambda = tan.acc_impulse - jtOld;
		Vector3 jt = lambda * tv;

		if (body_collides) {
			body->apply_impulse(-jt, c.rA + body->get_center_of_mass());
		}
		if (soft_body_collides) {
			soft_body->apply_node_impulse(c.index_B, jt);
		}
		c.acc_impulse -= jt;

		c.active = true;
	}
}

GodotBodySoftBodyPair3D::GodotBodySoftBodyPair3D(GodotBody3D *p_A, int p_shape_A, GodotSoftBody3D *p_B) :
		GodotBodyContact3D(&body, 1) {
	body = p_A;
	soft_body = p_B;
	body_shape = p_shape_A;
	space = p_A->get_space();
	body->add_constraint(this, 0);
	soft_body->add_constraint(this);
}

GodotBodySoftBodyPair3D::~GodotBodySoftBodyPair3D() {
	body->remove_constraint(this);
	soft_body->remove_constraint(this);
}
