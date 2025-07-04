/**************************************************************************/
/*  jolt_distance_joint_3d.cpp                                            */
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

#include "jolt_distance_joint_3d.h"

#include "../misc/jolt_type_conversions.h"
#include "../objects/jolt_body_3d.h"
#include "../spaces/jolt_space_3d.h"

#include "Jolt/Physics/Constraints/DistanceConstraint.h"

JoltDistanceJoint3D::JoltDistanceJoint3D(
		const JoltJoint3D &p_old_joint,
		JoltBody3D *p_body_a,
		JoltBody3D *p_body_b,
		const Vector3 &p_local_a,
		const Vector3 &p_local_b) :
		JoltJoint3D(
				p_old_joint,
				p_body_a,
				p_body_b,
				Transform3D({}, p_local_a),
				Transform3D({}, p_local_b)) {
	rebuild();
}

double JoltDistanceJoint3D::get_jolt_param(Param p_param) const {
	switch (p_param) {
		case JoltPhysicsServer3D::DISTANCE_JOINT_LIMITS_SPRING_STIFFNESS: {
			return limit_spring_stiffness;
		} break;
		case JoltPhysicsServer3D::DISTANCE_JOINT_LIMITS_SPRING_DAMPING: {
			return limit_spring_damping;
		} break;
		case JoltPhysicsServer3D::DISTANCE_JOINT_DISTANCE_MIN: {
			return distance_min;
		} break;
		case JoltPhysicsServer3D::DISTANCE_JOINT_DISTANCE_MAX: {
			return distance_max;
		} break;
		default: {
			ERR_FAIL_V_MSG(0.0, vformat("Unhandled parameter: '%d'. This should not happen. Please report this.", p_param));
		} break;
	}
}

void JoltDistanceJoint3D::set_jolt_param(Param p_param, double p_value) {
	switch (p_param) {
		case JoltPhysicsServer3D::DISTANCE_JOINT_LIMITS_SPRING_STIFFNESS: {
			limit_spring_stiffness = p_value;
			_limit_spring_changed();
		} break;
		case JoltPhysicsServer3D::DISTANCE_JOINT_LIMITS_SPRING_DAMPING: {
			limit_spring_damping = p_value;
			_limit_spring_changed();
		} break;
		case JoltPhysicsServer3D::DISTANCE_JOINT_DISTANCE_MIN: {
			distance_min = p_value;
			_distance_changed();
		} break;
		case JoltPhysicsServer3D::DISTANCE_JOINT_DISTANCE_MAX: {
			distance_max = p_value;
			_distance_changed();
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'. This should not happen. Please report this.", p_param));
		} break;
	}
}

void JoltDistanceJoint3D::rebuild() {
	destroy();

	JoltSpace3D *space = get_space();

	if (space == nullptr) {
		return;
	}

	JPH::Body *jolt_body_a = body_a != nullptr ? body_a->get_jolt_body() : nullptr;
	JPH::Body *jolt_body_b = body_a != nullptr ? body_b->get_jolt_body() : nullptr;

	ERR_FAIL_COND(jolt_body_a == nullptr && jolt_body_b == nullptr);

	Transform3D shifted_ref_a;
	Transform3D shifted_ref_b;

	_shift_reference_frames(Vector3(), Vector3(), shifted_ref_a, shifted_ref_b);

	jolt_ref = _build_constraint(jolt_body_a, jolt_body_b, shifted_ref_a, shifted_ref_b);

	space->add_joint(this);

	_update_enabled();
	_update_iterations();
}

JPH::Constraint *JoltDistanceJoint3D::_build_constraint(
		JPH::Body *p_jolt_body_a,
		JPH::Body *p_jolt_body_b,
		const Transform3D &p_shifted_ref_a,
		const Transform3D &p_shifted_ref_b) {
	JPH::DistanceConstraintSettings constraint_settings;
	constraint_settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;
	constraint_settings.mPoint1 = to_jolt_r(p_shifted_ref_a.origin);
	constraint_settings.mPoint2 = to_jolt_r(p_shifted_ref_b.origin);
	constraint_settings.mMinDistance = (float)distance_min;
	constraint_settings.mMaxDistance = (float)distance_max;
	constraint_settings.mLimitsSpringSettings.mStiffness = (float)limit_spring_stiffness;
	constraint_settings.mLimitsSpringSettings.mDamping = (float)limit_spring_damping;

	if (p_jolt_body_a == nullptr) {
		return constraint_settings.Create(JPH::Body::sFixedToWorld, *p_jolt_body_b);
	} else if (p_jolt_body_b == nullptr) {
		return constraint_settings.Create(*p_jolt_body_a, JPH::Body::sFixedToWorld);
	} else {
		return constraint_settings.Create(*p_jolt_body_a, *p_jolt_body_b);
	}
}

void JoltDistanceJoint3D::_limit_spring_changed() {
	rebuild();
	_wake_up_bodies();
}

void JoltDistanceJoint3D::_limit_distance_changed() {
	rebuild();
	_wake_up_bodies();
}

void JoltDistanceJoint3D::_distance_changed() {
	rebuild();
	_wake_up_bodies();
}
