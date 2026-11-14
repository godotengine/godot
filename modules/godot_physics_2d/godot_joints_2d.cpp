/**************************************************************************/
/*  godot_joints_2d.cpp                                                   */
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

#include "godot_joints_2d.h"

#include "godot_space_2d.h"

//based on chipmunk joint constraints

/* Copyright (c) 2007 Scott Lembcke
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Math::PI and Math::TAU are double, which may slow things down and mess with boundary conditions when compared with real_t
static constexpr real_t PI = Math::PI;
static constexpr real_t TAU = Math::TAU;

void GodotJoint2D::copy_settings_from(GodotJoint2D *p_joint) {
	set_self(p_joint->get_self());
	set_max_force(p_joint->get_max_force());
	set_bias(p_joint->get_bias());
	set_max_bias(p_joint->get_max_bias());
	disable_collisions_between_bodies(p_joint->is_disabled_collisions_between_bodies());
}

static inline real_t k_scalar(GodotBody2D *a, GodotBody2D *b, const Vector2 &rA, const Vector2 &rB, const Vector2 &n) {
	real_t value = 0.0;

	{
		value += a->get_inv_mass();
		real_t rcn = (rA - a->get_center_of_mass()).cross(n);
		value += a->get_inv_inertia() * rcn * rcn;
	}

	if (b) {
		value += b->get_inv_mass();
		real_t rcn = (rB - b->get_center_of_mass()).cross(n);
		value += b->get_inv_inertia() * rcn * rcn;
	}

	return value;
}

static inline Vector2
relative_velocity(GodotBody2D *a, GodotBody2D *b, Vector2 rA, Vector2 rB) {
	Vector2 sum = a->get_linear_velocity() - (rA - a->get_center_of_mass()).orthogonal() * a->get_angular_velocity();
	if (b) {
		return (b->get_linear_velocity() - (rB - b->get_center_of_mass()).orthogonal() * b->get_angular_velocity()) - sum;
	} else {
		return -sum;
	}
}

static inline real_t
normal_relative_velocity(GodotBody2D *a, GodotBody2D *b, Vector2 rA, Vector2 rB, Vector2 n) {
	return relative_velocity(a, b, rA, rB).dot(n);
}

// Returns normalized [-Pi, Pi).
static real_t _angle_normalize_slow(real_t angle) {
	if (unlikely(std::fabs(angle) >= 3 * PI)) {
		angle = std::remainder(angle, TAU);
	}
	return angle - std::copysign(static_cast<real_t>(angle >= PI || angle < -PI) * TAU, angle);
}

// Returns normalized [-Pi, Pi) with inlined early-out.
_FORCE_INLINE_ static real_t angle_normalize(real_t angle) {
	if (likely(std::fabs(angle) < PI)) {
		return angle;
	}
	// Let the compiler decide whether to inline the slow path or not
	return _angle_normalize_slow(angle);
}

_FORCE_INLINE_ static Vector2 cross(real_t p_other, const Vector2 &p_vec) {
	return Vector2(-p_other * p_vec.y, p_other * p_vec.x);
}

bool GodotPinJoint2D::setup(real_t p_step) {
	const GodotBody2D &body_A = *A;
	const GodotBody2D &body_B = *B;

	dynamic_A = body_A.get_mode() > PS2DE::BODY_MODE_KINEMATIC;
	dynamic_B = B && body_B.get_mode() > PS2DE::BODY_MODE_KINEMATIC;
	if (!dynamic_A && !dynamic_B) {
		return false;
	}
	GodotSpace2D *space = body_A.get_space();
	ERR_FAIL_NULL_V(space, false);
	real_t bias_coeff = (get_bias() == 0 ? space->get_constraint_bias() : get_bias());
	real_t inv_step = 1 / p_step;
	w_max = inv_step * MAX_SINGLE_FRAME_ROTATION;

	rA = body_A.get_transform().basis_xform(anchor_A);
	rB = B ? body_B.get_transform().basis_xform(anchor_B) : Vector2();

	// Bracketing critical for floating point precision.
	Vector2 delta = (rB - rA) + ((B ? body_B.get_transform().get_origin() : anchor_B) - A->get_transform().get_origin());
	bias_vel = -delta * bias_coeff * inv_step;

	// Calculate effective mass.
	Transform2D K;
	inv_inertia_A = 0;
	inv_inertia_B = 0;
	if (dynamic_A) {
		rA -= body_A.get_center_of_mass();
		K[0].x += body_A.get_inv_mass() + body_A.get_inv_inertia() * rA.y * rA.y;
		K[1].x -= body_A.get_inv_inertia() * rA.x * rA.y;
		K[0].y -= body_A.get_inv_inertia() * rA.x * rA.y;
		K[1].y += body_A.get_inv_mass() + body_A.get_inv_inertia() * rA.x * rA.x;
		if constexpr (ENABLE_PARALLEL_AXIS) {
			if (body_A.get_inv_inertia() > 0) {
				DEV_ASSERT(body_A.inertia > 0);
				inv_inertia_A = 1 / (body_A.inertia + body_A.mass * rA.dot(rA));
			}
		} else {
			inv_inertia_A = body_A.get_inv_inertia();
		}
	}
	if (dynamic_B) {
		rB -= body_B.get_center_of_mass();
		K[0].x += body_B.get_inv_mass() + body_B.get_inv_inertia() * rB.y * rB.y;
		K[1].x -= body_B.get_inv_inertia() * rB.x * rB.y;
		K[0].y -= body_B.get_inv_inertia() * rB.x * rB.y;
		K[1].y += body_B.get_inv_mass() + body_B.get_inv_inertia() * rB.x * rB.x;
		if constexpr (ENABLE_PARALLEL_AXIS) {
			if (body_B.get_inv_inertia() > 0) {
				DEV_ASSERT(body_B.inertia > 0);
				inv_inertia_B = 1 / (body_B.inertia + body_B.mass * rB.dot(rB));
			}
		} else {
			inv_inertia_B = body_B.get_inv_inertia();
		}
	}
	K[0].x += softness;
	K[1].y += softness;
	M = K.affine_inverse();
	i_sum = 0.0;
	if (inv_inertia_A != 0 || inv_inertia_B != 0) {
		i_sum = 1 / (inv_inertia_A + inv_inertia_B);
	}

	if constexpr (ENABLE_PARALLEL_AXIS) {
		if (dynamic_A && dynamic_B) {
			DEV_ASSERT((body_A.mass > 0) && (body_B.mass > 0));
			system_inv_mass = 1 / (body_A.mass + body_B.mass);
			system_rA = body_A.get_center_of_mass();
			system_rB = body_B.get_center_of_mass() + (body_B.get_transform().get_origin() - body_A.get_transform().get_origin());
			Vector2 system_com = system_inv_mass * (body_A.mass * system_rA + body_B.mass * system_rB);
			system_rA -= system_com;
			system_rB -= system_com;

			// It is possible that one of the bodies is PS2DE::BODY_MODE_RIGID_LINEAR
			// or has inertia still set to 0 from construction (in either case, inv_inertia_X will also be 0)
			DEV_ASSERT((body_A.inertia == 0) ? (inv_inertia_A == 0) : (inv_inertia_A > 0));
			DEV_ASSERT((body_B.inertia == 0) ? (inv_inertia_B == 0) : (inv_inertia_B > 0));
			DEV_ASSERT((body_A.get_mode() == PS2DE::BODY_MODE_RIGID_LINEAR) ? (inv_inertia_A == 0) : true);
			DEV_ASSERT((body_B.get_mode() == PS2DE::BODY_MODE_RIGID_LINEAR) ? (inv_inertia_B == 0) : true);

			if ((inv_inertia_A > 0) && (inv_inertia_B > 0)) {
				// Both of the bodies are freely rotating with finite inertia.
				system_inv_inertia = 1 / (body_A.inertia + body_B.inertia + body_A.mass * system_rA.dot(system_rA) + body_B.mass * system_rB.dot(system_rB));
			}
		}
	}

	// Setup motor for warm start or cold start
	motor_impulse = motor_enabled ? motor_impulse : 0;
	motor_w = CLAMP(motor_target_velocity, -w_max, w_max);
	max_motor_impulse = MOTOR_MAX_TORQUE * p_step;

	solve_limits = false;
	if (angular_limit_enabled) {
		bias_w = 0;
		limit_bias_impulse = 0; // cold start for bias
		real_t dist = angle_normalize(body_A.get_transform().columns[0].angle_to(
											  B ? body_B.get_transform().columns[0] : Vector2(1, 0)) -
				initial_angle - angular_limit_center);
		// `angular_limit_center` is the center of the restricted arc.
		// `angular_limit_extent` is the distance from this center to the equidistant limits (upper and lower)
		// `dist` is the distance from this center to our current angle.
		if (angular_limit_extent > PI - MIN_ANGULAR_LIMIT_SEPERATION * 0.5) {
			// The user has set the permitted arc to an angle too small to enforce without issues.
			// Implement this as a weld to be held at `dist == PI`.
			solve_limits = true;
			bias_w = angle_normalize(PI - dist) * inv_step * bias_coeff;
			bias_w = CLAMP(bias_w, -w_max, w_max);
			if constexpr (ENABLE_SPLIT_IMPULSE) {
				limit_upper_w = 0;
				limit_lower_w = 0;
			} else {
				limit_upper_w = bias_w;
				limit_lower_w = bias_w;
			}
		} else if (angular_limit_extent > MIN_ANGULAR_LIMIT_SEPERATION * 0.5) {
			solve_limits = true;
			limit_upper_w = inv_step * (-angular_limit_extent - dist + (dist >= 0) * TAU);
			limit_lower_w = inv_step * (angular_limit_extent - dist - (dist < 0) * TAU);
			if (limit_upper_w < 0) {
				// Upper limit breached.
				if constexpr (ENABLE_SPLIT_IMPULSE) {
					bias_w = limit_upper_w * bias_coeff;
					limit_upper_w = 0;
				} else {
					limit_upper_w *= bias_coeff;
				}
			}
			if (limit_lower_w > 0) {
				// Lower limit breached.
				if constexpr (ENABLE_SPLIT_IMPULSE) {
					bias_w = limit_lower_w * bias_coeff;
					limit_lower_w = 0;
				} else {
					limit_lower_w *= bias_coeff;
				}
			}
			bias_w = CLAMP(bias_w, -w_max, w_max);
		} // else the user has set the restricted arc too small to enforce.
	}
	if (!solve_limits) {
		// Setup for cold start for the next time `solve_limits` is true
		limit_upper_impulse = 0;
		limit_lower_impulse = 0;
	}
	return true;
}

bool GodotPinJoint2D::pre_solve(real_t p_step) {
	if (!dynamic_A && !dynamic_B) {
		return false;
	}
	// Warm start impulses
	_apply_torque_impulse(motor_impulse + limit_upper_impulse + limit_lower_impulse);
	_apply_impulse(linear_impulse);
	return true;
}

void GodotPinJoint2D::solve(real_t p_step) {
	if constexpr (ENABLE_PARALLEL_AXIS) {
		// The linear motion still fights angular, so it should go first, and then angular will correct for it.
		_solve_linear();
		_solve_angular();
	} else {
		// Traditional approach. Angular motion will fight the linear motion and vice versa.
		// Angular should go first as it produces the larger error.
		_solve_angular();
		_solve_linear();
	}
}

_FORCE_INLINE_ void GodotPinJoint2D::_solve_angular() {
	GodotBody2D &body_A = *A;
	GodotBody2D &body_B = *B;
	real_t clamp_partial = 0;
	real_t motor_partial = 0;
	real_t limit_upper_partial = 0;
	real_t limit_lower_partial = 0;

	real_t rel_w = (B ? body_B.get_angular_velocity() : 0) - body_A.get_angular_velocity();
	if (motor_enabled) {
		motor_partial = i_sum * (motor_w - rel_w);
		motor_partial = CLAMP(motor_partial + motor_impulse, -max_motor_impulse, max_motor_impulse) - motor_impulse;
		motor_impulse += motor_partial;
	}
	if (solve_limits) {
		if constexpr (ENABLE_SPLIT_IMPULSE) {
			// Apply bias as a positional pass.
			real_t bias_rel_w = (B ? body_B.get_biased_angular_velocity() : 0) - body_A.get_biased_angular_velocity();
			real_t bias_partial = i_sum * (bias_w - bias_rel_w);
			if (limit_upper_w != limit_lower_w) {
				if (bias_w > 0) {
					bias_partial = MAX(0, bias_partial + limit_bias_impulse) - limit_bias_impulse;
				} else if (bias_w < 0) {
					bias_partial = MIN(bias_partial + limit_bias_impulse, 0) - limit_bias_impulse;
				} else {
					bias_partial = 0; // bias_w == 0 means no bias
				}
			} // else user has made restricted zone small enough to make this a weld. Bias should be unclamped.

			limit_bias_impulse += bias_partial;
			_apply_torque_impulse<apply_to_bias>(bias_partial);
		} // else bias will already have been incorporated into limit_upper_w and limit_lower_w during setup

		// Compute and clamp limit impulses.
		limit_lower_partial = i_sum * (limit_lower_w - rel_w) - motor_partial;
		limit_lower_partial = MAX(0, limit_lower_partial + limit_lower_impulse) - limit_lower_impulse;
		limit_lower_impulse += limit_lower_partial;

		limit_upper_partial = i_sum * (limit_upper_w - rel_w) - motor_partial - limit_lower_partial;
		limit_upper_partial = MIN(limit_upper_partial + limit_upper_impulse, 0) - limit_upper_impulse;
		limit_upper_impulse += limit_upper_partial;
	}
	if constexpr (ENABLE_ANGULAR_VEL_CLAMP && ENABLE_PARALLEL_AXIS) {
		// Clamp relative angular velocity using equal and opposite impulse (to preserve momentum).
		// Note the primary purpose is to clamp angular velocity generated from the linear solve.
		if constexpr (ENABLE_SPLIT_IMPULSE) {
			real_t bias_rel_w = (B ? body_B.get_biased_angular_velocity() : 0) - body_A.get_biased_angular_velocity();
			if (std::fabs(bias_rel_w) > w_max) {
				real_t clamp_biased_rel_w = CLAMP(bias_rel_w, -w_max, w_max);
				real_t clamp_impulse = i_sum * (clamp_biased_rel_w - bias_rel_w);
				_apply_torque_impulse<apply_to_bias>(clamp_impulse);
			}
		}
		rel_w = rel_w + (motor_partial + limit_lower_partial + limit_upper_partial) * (inv_inertia_A + inv_inertia_B);
		if (std::fabs(rel_w) > w_max) {
			real_t clamp_rel_w = CLAMP(rel_w, -w_max, w_max);
			clamp_partial = i_sum * (clamp_rel_w - rel_w);
			if (!solve_limits && !motor_enabled) {
				_apply_torque_impulse(limit_upper_partial + limit_lower_partial + motor_partial + clamp_partial);
			}
		}
	}
	if (solve_limits || motor_enabled) {
		_apply_torque_impulse(limit_upper_partial + limit_lower_partial + motor_partial + clamp_partial);
	}
}

_FORCE_INLINE_ void GodotPinJoint2D::_solve_linear() {
	GodotBody2D &body_A = *A;
	GodotBody2D &body_B = *B;
	Vector2 linear_partial;

	Vector2 rel_vel = -body_A.get_linear_velocity() - cross(body_A.get_angular_velocity(), rA);
	if (B) {
		rel_vel += body_B.get_linear_velocity() + cross(body_B.get_angular_velocity(), rB);
	}

	if (ENABLE_SPLIT_IMPULSE && softness == 0) {
		// Apply bias as a positional pass. Don't do this when using soft constraints
		Vector2 bias_rel_vel = -body_A.get_biased_linear_velocity() - cross(body_A.get_biased_angular_velocity(), rA);
		if (B) {
			bias_rel_vel += body_B.get_biased_linear_velocity() + cross(body_B.get_biased_angular_velocity(), rB);
		}
		_apply_impulse<apply_to_bias>(M.basis_xform(bias_vel - bias_rel_vel));
		linear_partial = M.basis_xform(-rel_vel);
	} else {
		linear_partial = M.basis_xform(bias_vel - rel_vel - softness * linear_impulse);
	}

	linear_impulse += linear_partial;
	_apply_impulse(linear_partial);
}

template <bool APPLY_TO_BIAS>
void GodotPinJoint2D::_apply_torque_impulse(float p_torque_impulse) {
	GodotBody2D &body_A = *A;
	GodotBody2D &body_B = *B;
	Vector2 a_vel, b_vel;
	real_t a_angular_vel, b_angular_vel;

	DEV_ASSERT(dynamic_A == false ? inv_inertia_A == 0 : true);
	a_angular_vel = inv_inertia_A * -p_torque_impulse;
	b_angular_vel = inv_inertia_B * p_torque_impulse;

	if (ENABLE_PARALLEL_AXIS) {
		// Apply torque impulse around the pin
		a_vel = -cross(a_angular_vel, rA);
		b_vel = -cross(b_angular_vel, rB);

		if (dynamic_A && dynamic_B) {
			// Relative body velocity is now accurate, but system velocity may not be.
			// The two body system must preserve angular and linear momentum.
			// Use this law to correct motion.
			Vector2 system_vel = system_inv_mass * (a_vel * body_A.mass + b_vel * body_B.mass);
			a_vel -= system_vel;
			b_vel -= system_vel;

			if (inv_inertia_A > 0 && inv_inertia_B > 0) {
				// Both of the bodies are freely rotating with non-infinite inertia.
				real_t system_angular_vel = system_inv_inertia * (body_A.inertia * a_angular_vel + body_B.inertia * b_angular_vel + body_A.mass * system_rA.cross(a_vel) + body_B.mass * system_rB.cross(b_vel));
				a_angular_vel -= system_angular_vel;
				b_angular_vel -= system_angular_vel;
				a_vel -= cross(system_angular_vel, system_rA);
				b_vel -= cross(system_angular_vel, system_rB);
			}
		}
	}
	if constexpr (APPLY_TO_BIAS) {
		if (dynamic_A) {
			body_A.biased_angular_velocity += a_angular_vel;
			body_A.biased_linear_velocity += a_vel;
		}
		if (dynamic_B) {
			body_B.biased_angular_velocity += b_angular_vel;
			body_B.biased_linear_velocity += b_vel;
		}
	} else {
		if (dynamic_A) {
			body_A.angular_velocity += a_angular_vel;
			body_A.linear_velocity += a_vel;
		}
		if (dynamic_B) {
			body_B.angular_velocity += b_angular_vel;
			body_B.linear_velocity += b_vel;
		}
	}
}

template <bool APPLY_TO_BIAS>
_FORCE_INLINE_ void GodotPinJoint2D::_apply_impulse(Vector2 p_impulse) {
	GodotBody2D &body_A = *A;
	GodotBody2D &body_B = *B;

	if constexpr (APPLY_TO_BIAS) {
		if (dynamic_A) {
			body_A.biased_linear_velocity += -p_impulse * body_A._inv_mass;
			body_A.biased_angular_velocity += body_A._inv_inertia * rA.cross(-p_impulse);
		}
		if (dynamic_B) {
			body_B.biased_linear_velocity += p_impulse * body_B._inv_mass;
			body_B.biased_angular_velocity += body_B._inv_inertia * rB.cross(p_impulse);
		}
	} else {
		if (dynamic_A) {
			body_A.linear_velocity += -p_impulse * body_A._inv_mass;
			body_A.angular_velocity += body_A._inv_inertia * rA.cross(-p_impulse);
		}
		if (dynamic_B) {
			body_B.linear_velocity += p_impulse * body_B._inv_mass;
			body_B.angular_velocity += body_B._inv_inertia * rB.cross(p_impulse);
		}
	}
}

void GodotPinJoint2D::set_param(PS2DE::PinJointParam p_param, real_t p_value) {
	switch (p_param) {
		case PS2DE::PIN_JOINT_SOFTNESS: {
			softness = p_value;
		} break;
		case PS2DE::PIN_JOINT_LIMIT_UPPER: {
			real_t lower_limit_rel_upper = angle_normalize(angular_limit_center + angular_limit_extent - p_value);
			if (lower_limit_rel_upper <= 0) {
				lower_limit_rel_upper += TAU;
			}
			angular_limit_center = angle_normalize(p_value + lower_limit_rel_upper * real_t(0.5));
			angular_limit_extent = std::fabs(angle_normalize(angular_limit_center - p_value));
		} break;
		case PS2DE::PIN_JOINT_LIMIT_LOWER: {
			real_t upper_limit_rel_lower = angle_normalize(angular_limit_center - angular_limit_extent - p_value);
			if (upper_limit_rel_lower >= 0) {
				upper_limit_rel_lower -= TAU;
			}
			angular_limit_center = angle_normalize(p_value + upper_limit_rel_lower * real_t(0.5));
			angular_limit_extent = std::fabs(angle_normalize(p_value - angular_limit_center));
		} break;
		case PS2DE::PIN_JOINT_MOTOR_TARGET_VELOCITY: {
			motor_target_velocity = p_value;
		} break;
	}
}

real_t GodotPinJoint2D::get_param(PS2DE::PinJointParam p_param) const {
	switch (p_param) {
		case PS2DE::PIN_JOINT_SOFTNESS: {
			return softness;
		}
		case PS2DE::PIN_JOINT_LIMIT_UPPER: {
			return angle_normalize(angular_limit_center - angular_limit_extent);
		}
		case PS2DE::PIN_JOINT_LIMIT_LOWER: {
			return angle_normalize(angular_limit_center + angular_limit_extent);
		}
		case PS2DE::PIN_JOINT_MOTOR_TARGET_VELOCITY: {
			return motor_target_velocity;
		}
	}
	ERR_FAIL_V(0);
}

void GodotPinJoint2D::set_flag(PS2DE::PinJointFlag p_flag, bool p_enabled) {
	switch (p_flag) {
		case PS2DE::PIN_JOINT_FLAG_ANGULAR_LIMIT_ENABLED: {
			angular_limit_enabled = p_enabled;
		} break;
		case PS2DE::PIN_JOINT_FLAG_MOTOR_ENABLED: {
			motor_enabled = p_enabled;
		} break;
	}
}

bool GodotPinJoint2D::get_flag(PS2DE::PinJointFlag p_flag) const {
	switch (p_flag) {
		case PS2DE::PIN_JOINT_FLAG_ANGULAR_LIMIT_ENABLED: {
			return angular_limit_enabled;
		}
		case PS2DE::PIN_JOINT_FLAG_MOTOR_ENABLED: {
			return motor_enabled;
		}
	}
	ERR_FAIL_V(false);
}

GodotPinJoint2D::GodotPinJoint2D(const Vector2 &p_pos, GodotBody2D *p_body_a, GodotBody2D *p_body_b) :
		GodotJoint2D(_arr, p_body_b ? 2 : 1) {
	A = p_body_a;
	B = p_body_b;
	GodotBody2D &body_A = *A;
	GodotBody2D &body_B = *B;

	// This assumes there is no scale in the transform. Written this way to avoid floating point precision issues.
	anchor_A = body_A.get_inv_transform().basis_xform(p_pos - body_A.get_transform().get_origin());
	anchor_B = B ? body_B.get_inv_transform().basis_xform(p_pos - body_B.get_transform().get_origin()) : p_pos;
	initial_angle = angle_normalize(body_A.get_transform().columns[0].angle_to(B ? body_B.get_transform().columns[0] : Vector2(1, 0)));

	body_A.add_constraint(this, 0);
	if (B) {
		body_B.add_constraint(this, 1);
	}
}

//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

static inline void
k_tensor(GodotBody2D *a, GodotBody2D *b, Vector2 r1, Vector2 r2, Vector2 *k1, Vector2 *k2) {
	// calculate mass matrix
	// If I wasn't lazy and wrote a proper matrix class, this wouldn't be so gross...
	real_t k11, k12, k21, k22;
	real_t m_sum = a->get_inv_mass() + b->get_inv_mass();

	// start with I*m_sum
	k11 = m_sum;
	k12 = 0.0f;
	k21 = 0.0f;
	k22 = m_sum;

	r1 -= a->get_center_of_mass();
	r2 -= b->get_center_of_mass();

	// add the influence from r1
	real_t a_i_inv = a->get_inv_inertia();
	real_t r1xsq = r1.x * r1.x * a_i_inv;
	real_t r1ysq = r1.y * r1.y * a_i_inv;
	real_t r1nxy = -r1.x * r1.y * a_i_inv;
	k11 += r1ysq;
	k12 += r1nxy;
	k21 += r1nxy;
	k22 += r1xsq;

	// add the influence from r2
	real_t b_i_inv = b->get_inv_inertia();
	real_t r2xsq = r2.x * r2.x * b_i_inv;
	real_t r2ysq = r2.y * r2.y * b_i_inv;
	real_t r2nxy = -r2.x * r2.y * b_i_inv;
	k11 += r2ysq;
	k12 += r2nxy;
	k21 += r2nxy;
	k22 += r2xsq;

	// invert
	real_t determinant = k11 * k22 - k12 * k21;
	ERR_FAIL_COND(determinant == 0.0);

	real_t det_inv = 1.0f / determinant;
	*k1 = Vector2(k22 * det_inv, -k12 * det_inv);
	*k2 = Vector2(-k21 * det_inv, k11 * det_inv);
}

static _FORCE_INLINE_ Vector2
mult_k(const Vector2 &vr, const Vector2 &k1, const Vector2 &k2) {
	return Vector2(vr.dot(k1), vr.dot(k2));
}

bool GodotGrooveJoint2D::setup(real_t p_step) {
	dynamic_A = (A->get_mode() > PS2DE::BODY_MODE_KINEMATIC);
	dynamic_B = (B->get_mode() > PS2DE::BODY_MODE_KINEMATIC);

	if (!dynamic_A && !dynamic_B) {
		return false;
	}

	GodotSpace2D *space = A->get_space();
	ERR_FAIL_NULL_V(space, false);

	// calculate endpoints in worldspace
	Vector2 ta = A->get_transform().xform(A_groove_1);
	Vector2 tb = A->get_transform().xform(A_groove_2);

	// calculate axis
	Vector2 n = -(tb - ta).orthogonal().normalized();
	real_t d = ta.dot(n);

	xf_normal = n;
	rB = B->get_transform().basis_xform(B_anchor);

	// calculate tangential distance along the axis of rB
	real_t td = (B->get_transform().get_origin() + rB).cross(n);
	// calculate clamping factor and rB
	if (td <= ta.cross(n)) {
		clamp = 1.0f;
		rA = ta - A->get_transform().get_origin();
	} else if (td >= tb.cross(n)) {
		clamp = -1.0f;
		rA = tb - A->get_transform().get_origin();
	} else {
		clamp = 0.0f;
		//joint->r1 = cpvsub(cpvadd(cpvmult(cpvperp(n), -td), cpvmult(n, d)), a->p);
		rA = ((-n.orthogonal() * -td) + n * d) - A->get_transform().get_origin();
	}

	// Calculate mass tensor
	k_tensor(A, B, rA, rB, &k1, &k2);

	// compute max impulse
	jn_max = get_max_force() * p_step;

	// calculate bias velocity
	//cpVect delta = cpvsub(cpvadd(b->p, joint->r2), cpvadd(a->p, joint->r1));
	//joint->bias = cpvclamp(cpvmult(delta, -joint->constraint.biasCoef*dt_inv), joint->constraint.maxBias);

	Vector2 delta = (B->get_transform().get_origin() + rB) - (A->get_transform().get_origin() + rA);

	real_t _b = get_bias();
	gbias = (delta * -(_b == 0 ? space->get_constraint_bias() : _b) * (1.0 / p_step)).limit_length(get_max_bias());

	correct = true;
	return true;
}

bool GodotGrooveJoint2D::pre_solve(real_t p_step) {
	// Apply accumulated impulse.
	if (dynamic_A) {
		A->apply_impulse(-jn_acc, rA);
	}
	if (dynamic_B) {
		B->apply_impulse(jn_acc, rB);
	}

	return true;
}

void GodotGrooveJoint2D::solve(real_t p_step) {
	// compute impulse
	Vector2 vr = relative_velocity(A, B, rA, rB);

	Vector2 j = mult_k(gbias - vr, k1, k2);
	Vector2 jOld = jn_acc;
	j += jOld;

	jn_acc = (((clamp * j.cross(xf_normal)) > 0) ? j : j.project(xf_normal)).limit_length(jn_max);

	j = jn_acc - jOld;

	if (dynamic_A) {
		A->apply_impulse(-j, rA);
	}
	if (dynamic_B) {
		B->apply_impulse(j, rB);
	}
}

GodotGrooveJoint2D::GodotGrooveJoint2D(const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, GodotBody2D *p_body_a, GodotBody2D *p_body_b) :
		GodotJoint2D(_arr, 2) {
	A = p_body_a;
	B = p_body_b;

	A_groove_1 = A->get_inv_transform().xform(p_a_groove1);
	A_groove_2 = A->get_inv_transform().xform(p_a_groove2);
	B_anchor = B->get_inv_transform().xform(p_b_anchor);
	A_groove_normal = -(A_groove_2 - A_groove_1).normalized().orthogonal();

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}

//////////////////////////////////////////////
//////////////////////////////////////////////
//////////////////////////////////////////////

bool GodotDampedSpringJoint2D::setup(real_t p_step) {
	dynamic_A = (A->get_mode() > PS2DE::BODY_MODE_KINEMATIC);
	dynamic_B = (B->get_mode() > PS2DE::BODY_MODE_KINEMATIC);

	if (!dynamic_A && !dynamic_B) {
		return false;
	}

	rA = A->get_transform().basis_xform(anchor_A);
	rB = B->get_transform().basis_xform(anchor_B);

	Vector2 delta = (B->get_transform().get_origin() + rB) - (A->get_transform().get_origin() + rA);
	real_t dist = delta.length();

	if (dist) {
		n = delta / dist;
	} else {
		n = Vector2();
	}

	real_t k = k_scalar(A, B, rA, rB, n);
	n_mass = 1.0f / k;

	target_vrn = 0.0f;
	v_coef = 1.0f - Math::exp(-damping * (p_step)*k);

	// Calculate spring force.
	real_t f_spring = (rest_length - dist) * stiffness;
	j = n * f_spring * (p_step);

	return true;
}

bool GodotDampedSpringJoint2D::pre_solve(real_t p_step) {
	// Apply spring force.
	if (dynamic_A) {
		A->apply_impulse(-j, rA);
	}
	if (dynamic_B) {
		B->apply_impulse(j, rB);
	}

	return true;
}

void GodotDampedSpringJoint2D::solve(real_t p_step) {
	// compute relative velocity
	real_t vrn = normal_relative_velocity(A, B, rA, rB, n) - target_vrn;

	// compute velocity loss from drag
	// not 100% certain this is derived correctly, though it makes sense
	real_t v_damp = -vrn * v_coef;
	target_vrn = vrn + v_damp;
	Vector2 j_new = n * v_damp * n_mass;

	if (dynamic_A) {
		A->apply_impulse(-j_new, rA);
	}
	if (dynamic_B) {
		B->apply_impulse(j_new, rB);
	}
}

void GodotDampedSpringJoint2D::set_param(PS2DE::DampedSpringParam p_param, real_t p_value) {
	switch (p_param) {
		case PS2DE::DAMPED_SPRING_REST_LENGTH: {
			rest_length = p_value;
		} break;
		case PS2DE::DAMPED_SPRING_DAMPING: {
			damping = p_value;
		} break;
		case PS2DE::DAMPED_SPRING_STIFFNESS: {
			stiffness = p_value;
		} break;
	}
}

real_t GodotDampedSpringJoint2D::get_param(PS2DE::DampedSpringParam p_param) const {
	switch (p_param) {
		case PS2DE::DAMPED_SPRING_REST_LENGTH: {
			return rest_length;
		} break;
		case PS2DE::DAMPED_SPRING_DAMPING: {
			return damping;
		} break;
		case PS2DE::DAMPED_SPRING_STIFFNESS: {
			return stiffness;
		} break;
	}

	ERR_FAIL_V(0);
}

GodotDampedSpringJoint2D::GodotDampedSpringJoint2D(const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, GodotBody2D *p_body_a, GodotBody2D *p_body_b) :
		GodotJoint2D(_arr, 2) {
	A = p_body_a;
	B = p_body_b;
	anchor_A = A->get_inv_transform().xform(p_anchor_a);
	anchor_B = B->get_inv_transform().xform(p_anchor_b);

	rest_length = p_anchor_a.distance_to(p_anchor_b);

	A->add_constraint(this, 0);
	B->add_constraint(this, 1);
}
