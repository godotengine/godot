/*************************************************************************/
/*  joints_3d_sw.h                                                       */
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

/*
Adapted to Godot from the Bullet library.
*/

/*
Bullet Continuous Collision Detection and Physics Library
ConeTwistJointSW is Copyright (c) 2007 Starbreeze Studios

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef JOINTS_3D_SW_H
#define JOINTS_3D_SW_H

#include "body_3d_sw.h"
#include "constraint_3d_sw.h"
#include "core/math/transform.h"

class Joint3DSW : public Constraint3DSW {
public:
	virtual bool setup(real_t p_step) { return false; }
	virtual void solve(real_t p_step) {}

	void copy_settings_from(Joint3DSW *p_joint) {
		set_self(p_joint->get_self());
		set_priority(p_joint->get_priority());
		disable_collisions_between_bodies(p_joint->is_disabled_collisions_between_bodies());
	}

	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_TYPE_MAX; }
	_FORCE_INLINE_ Joint3DSW(Body3DSW **p_body_ptr = nullptr, int p_body_count = 0) :
			Constraint3DSW(p_body_ptr, p_body_count) {
	}

	virtual ~Joint3DSW() {
		for (int i = 0; i < get_body_count(); i++) {
			Body3DSW *body = get_body_ptr()[i];
			if (body) {
				body->remove_constraint(this);
			}
		}
	}
};

class JacobianEntry3DSW {
public:
	JacobianEntry3DSW() {}

	// Constraint between two different RigidBodies
	JacobianEntry3DSW(
			const Basis &world2A,
			const Basis &world2B,
			const Vector3 &rel_pos1, const Vector3 &rel_pos2,
			const Vector3 &jointAxis,
			const Vector3 &inertiaInvA,
			const real_t massInvA,
			const Vector3 &inertiaInvB,
			const real_t massInvB);

	// Angular constraint between two different RigidBodies
	JacobianEntry3DSW(const Vector3 &jointAxis,
			const Basis &world2A,
			const Basis &world2B,
			const Vector3 &inertiaInvA,
			const Vector3 &inertiaInvB);

	// Angular constraint between two different RigidBodies
	JacobianEntry3DSW(const Vector3 &axisInA,
			const Vector3 &axisInB,
			const Vector3 &inertiaInvA,
			const Vector3 &inertiaInvB);

	// Constraint on one RigidBody
	JacobianEntry3DSW(
			const Basis &world2A,
			const Vector3 &rel_pos1, const Vector3 &rel_pos2,
			const Vector3 &jointAxis,
			const Vector3 &inertiaInvA,
			const real_t massInvA);

	real_t getDiagonal() const;
	// For two constraints on the same RigidBody (for example vehicle friction)
	real_t getNonDiagonal(const JacobianEntry3DSW &jacB, const real_t massInvA) const;
	// For two constraints on sharing two same RigidBodies (for example two contact points between two RigidBodies)
	real_t getNonDiagonal(const JacobianEntry3DSW &jacB, const real_t massInvA, const real_t massInvB) const;
	real_t getRelativeVelocity(const Vector3 &linvelA, const Vector3 &angvelA, const Vector3 &linvelB, const Vector3 &angvelB);

	Vector3 m_linearJointAxis;
	Vector3 m_aJ;
	Vector3 m_bJ;
	Vector3 m_0MinvJt;
	Vector3 m_1MinvJt;
	//Optimization: Can be stored in the w/last component of one of the vectors
	real_t m_Adiag;
};

/// ConeTwistJoint3DSW

/*
ConeTwistJointSW is Copyright (c) 2007 Starbreeze Studios

Written by: Marcus Hennix
*/

class ConeTwistJoint3DSW : public Joint3DSW {
	union {
		struct {
			Body3DSW *A;
			Body3DSW *B;
		};
		Body3DSW *_arr[2];
	};

	JacobianEntry3DSW m_jac[3]; // 3 orthogonal linear constraints

	real_t m_appliedImpulse;
	Transform m_rbAFrame;
	Transform m_rbBFrame;

	real_t m_limitSoftness;
	real_t m_biasFactor;
	real_t m_relaxationFactor;

	real_t m_swingSpan1;
	real_t m_swingSpan2;
	real_t m_twistSpan;

	Vector3 m_swingAxis;
	Vector3 m_twistAxis;

	real_t m_kSwing;
	real_t m_kTwist;

	real_t m_twistLimitSign;
	real_t m_swingCorrection;
	real_t m_twistCorrection;

	real_t m_accSwingLimitImpulse;
	real_t m_accTwistLimitImpulse;

	bool m_angularOnly;
	bool m_solveTwistLimit;
	bool m_solveSwingLimit;

public:
	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_TYPE_CONE_TWIST; }
	virtual bool setup(real_t p_timestep);
	virtual void solve(real_t p_timestep);

	ConeTwistJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &rbAFrame, const Transform &rbBFrame);
	void setAngularOnly(bool angularOnly) { m_angularOnly = angularOnly; }
	void setLimit(real_t _swingSpan1, real_t _swingSpan2, real_t _twistSpan, real_t _softness = 0.8f, real_t _biasFactor = 0.3f, real_t _relaxationFactor = 1.0f) {
		m_swingSpan1 = _swingSpan1;
		m_swingSpan2 = _swingSpan2;
		m_twistSpan = _twistSpan;

		m_limitSoftness = _softness;
		m_biasFactor = _biasFactor;
		m_relaxationFactor = _relaxationFactor;
	}

	inline int getSolveTwistLimit() { return m_solveTwistLimit; }
	inline int getSolveSwingLimit() { return m_solveTwistLimit; }
	inline real_t getTwistLimitSign() { return m_twistLimitSign; }

	void set_param(PhysicsServer3D::ConeTwistJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer3D::ConeTwistJointParam p_param) const;
};

/// Generic6DOFJoint3DSW

/*
2007-09-09
Generic6DOFJointSW Refactored by Francisco Le?n
email: projectileman@yahoo.com
http://gimpact.sf.net
*/

// Rotation Limit structure for generic joints
class G6DOFRotationalLimitMotor3DSW {
public:
	// Limit_parameters
	real_t m_loLimit; // Joint limit
	real_t m_hiLimit; // joint limit
	real_t m_targetVelocity; // Target motor velocity
	real_t m_maxMotorForce; // Max force on motor
	real_t m_maxLimitForce; // Max force on limit
	real_t m_damping; // Damping
	real_t m_limitSoftness; // Relaxation factor
	real_t m_ERP; // Error tolerance factor when joint is at limit
	real_t m_bounce; // Restitution factor
	bool m_enableMotor;
	bool m_enableLimit;

	// Temp_variables
	real_t m_currentLimitError; // How much is violated this limit
	int m_currentLimit; // 0=free, 1=at lo limit, 2=at hi limit
	real_t m_accumulatedImpulse;

	G6DOFRotationalLimitMotor3DSW() {
		m_accumulatedImpulse = 0.f;
		m_targetVelocity = 0;
		m_maxMotorForce = 0.1f;
		m_maxLimitForce = 300.0f;
		m_loLimit = -1e30;
		m_hiLimit = 1e30;
		m_ERP = 0.5f;
		m_bounce = 0.0f;
		m_damping = 1.0f;
		m_limitSoftness = 0.5f;
		m_currentLimit = 0;
		m_currentLimitError = 0;
		m_enableMotor = false;
		m_enableLimit = false;
	}

	// Is limited
	bool isLimited() {
		return (m_loLimit < m_hiLimit);
	}

	// Need apply correction
	bool needApplyTorques() {
		return (m_enableMotor || m_currentLimit != 0);
	}

	// Calculates  error
	int testLimitValue(real_t test_value);

	// Apply the correction impulses for two bodies
	real_t solveAngularLimits(real_t timeStep, Vector3 &axis, real_t jacDiagABInv, Body3DSW *body0, Body3DSW *body1);
};

class G6DOFTranslationalLimitMotor3DSW {
public:
	Vector3 m_lowerLimit; // The constraint lower limits
	Vector3 m_upperLimit; // The constraint upper limits
	Vector3 m_accumulatedImpulse;
	// Linear_Limit_parameters
	Vector3 m_limitSoftness; // Softness for linear limit
	Vector3 m_damping; // Damping for linear limit
	Vector3 m_restitution; // Bounce parameter for linear limit

	bool enable_limit[3];

	G6DOFTranslationalLimitMotor3DSW() {
		m_lowerLimit = Vector3(0.f, 0.f, 0.f);
		m_upperLimit = Vector3(0.f, 0.f, 0.f);
		m_accumulatedImpulse = Vector3(0.f, 0.f, 0.f);
		m_limitSoftness = Vector3(1, 1, 1) * 0.7f;
		m_damping = Vector3(1, 1, 1) * real_t(1.0f);
		m_restitution = Vector3(1, 1, 1) * real_t(0.5f);
		enable_limit[0] = true;
		enable_limit[1] = true;
		enable_limit[2] = true;
	}

	// Test limit
	// upper < lower:  Free
	// upper == lower: Locked
	// upper > lower:  Limited
	// limitIndex: first 3 are linear, next 3 are angular
	inline bool isLimited(int limitIndex) {
		return (m_upperLimit[limitIndex] >= m_lowerLimit[limitIndex]);
	}

	real_t solveLinearAxis(
			real_t timeStep,
			real_t jacDiagABInv,
			Body3DSW *body1, const Vector3 &pointInA,
			Body3DSW *body2, const Vector3 &pointInB,
			int limit_index,
			const Vector3 &axis_normal_on_a,
			const Vector3 &anchorPos);
};

class Generic6DOFJoint3DSW : public Joint3DSW {
protected:
	union {
		struct {
			Body3DSW *A;
			Body3DSW *B;
		};
		Body3DSW *_arr[2];
	};

	// Relative_frames
	Transform m_frameInA; // The constraint space w.r.t body A
	Transform m_frameInB; // The constraint space w.r.t body B
	// Jacobians
	JacobianEntry3DSW m_jacLinear[3]; // 3 orthogonal linear constraints
	JacobianEntry3DSW m_jacAng[3]; // 3 orthogonal angular constraints
	// Linear_Limit_parameters
	G6DOFTranslationalLimitMotor3DSW m_linearLimits;
	// Hinge_parameters
	G6DOFRotationalLimitMotor3DSW m_angularLimits[3];

protected:
	// Temporary variables
	real_t m_timeStep;
	Transform m_calculatedTransformA;
	Transform m_calculatedTransformB;
	Vector3 m_calculatedAxisAngleDiff;
	Vector3 m_calculatedAxis[3];
	Vector3 m_AnchorPos; // Point between pivots of bodies A and B to solve linear axes
	bool m_useLinearReferenceFrameA;

	Generic6DOFJoint3DSW(Generic6DOFJoint3DSW const &) = delete;
	void operator=(Generic6DOFJoint3DSW const &) = delete;

	void buildLinearJacobian(
			JacobianEntry3DSW &jacLinear, const Vector3 &normalWorld,
			const Vector3 &pivotAInW, const Vector3 &pivotBInW);

	void buildAngularJacobian(JacobianEntry3DSW &jacAngular, const Vector3 &jointAxisW);

	// Calcs the euler angles between the two bodies.
	void calculateAngleInfo();

public:
	Generic6DOFJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &frameInA, const Transform &frameInB, bool useLinearReferenceFrameA);

	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_TYPE_6DOF; }
	virtual bool setup(real_t p_timestep);
	virtual void solve(real_t p_timestep);

	// Calcs global transform of the offsets
	void calculateTransforms();
	// Gets the global transform of the offset for body A
	const Transform &getCalculatedTransformA() const { return m_calculatedTransformA; }
	// Gets the global transform of the offset for body B
	const Transform &getCalculatedTransformB() const { return m_calculatedTransformB; }
	const Transform &getFrameOffsetA() const { return m_frameInA; }
	const Transform &getFrameOffsetB() const { return m_frameInB; }
	Transform &getFrameOffsetA() { return m_frameInA; }
	Transform &getFrameOffsetB() { return m_frameInB; }
	// Performs Jacobian calculation, and also calculates angle differences and axis
	void updateRHS(real_t timeStep);
	// Get the rotation axis in global coordinates
	Vector3 getAxis(int axis_index) const;
	// Get the relative Euler angle
	real_t getAngle(int axis_index) const;
	// Test angular limit.
	bool testAngularLimitMotor(int axis_index);
	void setLinearLowerLimit(const Vector3 &linearLower) { m_linearLimits.m_lowerLimit = linearLower; }
	void setLinearUpperLimit(const Vector3 &linearUpper) { m_linearLimits.m_upperLimit = linearUpper; }
	void setAngularLowerLimit(const Vector3 &angularLower) {
		m_angularLimits[0].m_loLimit = angularLower.x;
		m_angularLimits[1].m_loLimit = angularLower.y;
		m_angularLimits[2].m_loLimit = angularLower.z;
	}
	void setAngularUpperLimit(const Vector3 &angularUpper) {
		m_angularLimits[0].m_hiLimit = angularUpper.x;
		m_angularLimits[1].m_hiLimit = angularUpper.y;
		m_angularLimits[2].m_hiLimit = angularUpper.z;
	}
	// Retrieves the angular limit informacion
	G6DOFRotationalLimitMotor3DSW *getRotationalLimitMotor(int index) { return &m_angularLimits[index]; }
	// Retrieves the  limit informacion
	G6DOFTranslationalLimitMotor3DSW *getTranslationalLimitMotor() { return &m_linearLimits; }

	// First 3 are linear, next 3 are angular
	void setLimit(int axis, real_t lo, real_t hi) {
		if (axis < 3) {
			m_linearLimits.m_lowerLimit[axis] = lo;
			m_linearLimits.m_upperLimit[axis] = hi;
		} else {
			m_angularLimits[axis - 3].m_loLimit = lo;
			m_angularLimits[axis - 3].m_hiLimit = hi;
		}
	}

	// Test limit
	// upper < lower:  Free
	// upper == lower: Locked
	// upper > lower:  Limited
	// limitIndex: first 3 are linear, next 3 are angular
	bool isLimited(int limitIndex) {
		if (limitIndex < 3) {
			return m_linearLimits.isLimited(limitIndex);
		}
		return m_angularLimits[limitIndex - 3].isLimited();
	}

	const Body3DSW *getRigidBodyA() const { return A; }
	const Body3DSW *getRigidBodyB() const { return B; }

	void calcAnchorPos();
	void set_param(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param, real_t p_value);
	real_t get_param(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisParam p_param) const;
	void set_flag(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag, bool p_value);
	bool get_flag(Vector3::Axis p_axis, PhysicsServer3D::G6DOFJointAxisFlag p_flag) const;
};

/// HingeJoint3DSW

class HingeJoint3DSW : public Joint3DSW {
	union {
		struct {
			Body3DSW *A;
			Body3DSW *B;
		};
		Body3DSW *_arr[2];
	};

	JacobianEntry3DSW m_jac[3]; // 3 orthogonal linear constraints
	JacobianEntry3DSW m_jacAng[3]; // 2 orthogonal angular constraints + 1 for limit/motor

	Transform m_rbAFrame; // Constraint axis. Assumes z is hinge axis.
	Transform m_rbBFrame;

	real_t m_motorTargetVelocity;
	real_t m_maxMotorImpulse;
	real_t m_limitSoftness;
	real_t m_biasFactor;
	real_t m_relaxationFactor;
	real_t m_lowerLimit;
	real_t m_upperLimit;
	real_t m_kHinge;
	real_t m_limitSign;
	real_t m_correction;
	real_t m_accLimitImpulse;
	real_t tau;

	bool m_useLimit;
	bool m_angularOnly;
	bool m_enableAngularMotor;
	bool m_solveLimit;

	real_t m_appliedImpulse;

public:
	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_TYPE_HINGE; }
	virtual bool setup(real_t p_step);
	virtual void solve(real_t p_step);

	real_t get_hinge_angle();

	void set_param(PhysicsServer3D::HingeJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer3D::HingeJointParam p_param) const;

	void set_flag(PhysicsServer3D::HingeJointFlag p_flag, bool p_value);
	bool get_flag(PhysicsServer3D::HingeJointFlag p_flag) const;

	HingeJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &frameA, const Transform &frameB);
	HingeJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Vector3 &pivotInA, const Vector3 &pivotInB, const Vector3 &axisInA, const Vector3 &axisInB);
};

/// PinJoint3DSW

class PinJoint3DSW : public Joint3DSW {
	union {
		struct {
			Body3DSW *A;
			Body3DSW *B;
		};

		Body3DSW *_arr[2];
	};

	real_t m_tau; // Bias
	real_t m_damping;
	real_t m_impulseClamp;
	real_t m_appliedImpulse;

	JacobianEntry3DSW m_jac[3]; // 3 orthogonal linear constraints

	Vector3 m_pivotInA;
	Vector3 m_pivotInB;

public:
	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_TYPE_PIN; }
	virtual bool setup(real_t p_step);
	virtual void solve(real_t p_step);

	void set_param(PhysicsServer3D::PinJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer3D::PinJointParam p_param) const;
	void set_pos_a(const Vector3 &p_pos) { m_pivotInA = p_pos; }
	void set_pos_b(const Vector3 &p_pos) { m_pivotInB = p_pos; }

	Vector3 get_position_a() { return m_pivotInA; }
	Vector3 get_position_b() { return m_pivotInB; }

	PinJoint3DSW(Body3DSW *p_body_a, const Vector3 &p_pos_a, Body3DSW *p_body_b, const Vector3 &p_pos_b);
	~PinJoint3DSW();
};

/// SliderJoint3DSW

/*
Added by Roman Ponomarev (rponom@gmail.com)
April 04, 2008
*/

#define SLIDER_CONSTRAINT_DEF_SOFTNESS (real_t(1.0))
#define SLIDER_CONSTRAINT_DEF_DAMPING (real_t(1.0))
#define SLIDER_CONSTRAINT_DEF_RESTITUTION (real_t(0.7))

class SliderJoint3DSW : public Joint3DSW {
protected:
	union {
		struct {
			Body3DSW *A;
			Body3DSW *B;
		};

		Body3DSW *_arr[2];
	};

	Transform m_frameInA;
	Transform m_frameInB;

	// Linear limits
	real_t m_lowerLinLimit;
	real_t m_upperLinLimit;
	// Angular limits
	real_t m_lowerAngLimit;
	real_t m_upperAngLimit;

	// Softness, restitution and damping for different cases
	// Moving inside linear limits
	real_t m_softnessDirLin;
	real_t m_restitutionDirLin;
	real_t m_dampingDirLin;
	// Moving inside angular limits
	real_t m_softnessDirAng;
	real_t m_restitutionDirAng;
	real_t m_dampingDirAng;
	// Hitting linear limit
	real_t m_softnessLimLin;
	real_t m_restitutionLimLin;
	real_t m_dampingLimLin;
	// Hitting angular limit
	real_t m_softnessLimAng;
	real_t m_restitutionLimAng;
	real_t m_dampingLimAng;
	// Against linear constraint axis
	real_t m_softnessOrthoLin;
	real_t m_restitutionOrthoLin;
	real_t m_dampingOrthoLin;
	// Against angular constraint axis
	real_t m_softnessOrthoAng;
	real_t m_restitutionOrthoAng;
	real_t m_dampingOrthoAng;

	// For interlal use
	bool m_solveLinLim;
	bool m_solveAngLim;

	JacobianEntry3DSW m_jacLin[3];
	real_t m_jacLinDiagABInv[3];
	JacobianEntry3DSW m_jacAng[3];

	real_t m_timeStep;
	Transform m_calculatedTransformA;
	Transform m_calculatedTransformB;

	Vector3 m_sliderAxis;
	Vector3 m_realPivotAInW;
	Vector3 m_realPivotBInW;
	Vector3 m_projPivotInW;
	Vector3 m_delta;
	Vector3 m_depth;
	Vector3 m_relPosA;
	Vector3 m_relPosB;

	real_t m_linPos;
	real_t m_angDepth;
	real_t m_kAngle;

	bool m_poweredLinMotor;
	real_t m_targetLinMotorVelocity;
	real_t m_maxLinMotorForce;
	real_t m_accumulatedLinMotorImpulse;

	bool m_poweredAngMotor;
	real_t m_targetAngMotorVelocity;
	real_t m_maxAngMotorForce;
	real_t m_accumulatedAngMotorImpulse;

	void initParams();

public:
	SliderJoint3DSW(Body3DSW *rbA, Body3DSW *rbB, const Transform &frameInA, const Transform &frameInB);

	const Body3DSW *getRigidBodyA() const { return A; }
	const Body3DSW *getRigidBodyB() const { return B; }
	const Transform &getCalculatedTransformA() const { return m_calculatedTransformA; }
	const Transform &getCalculatedTransformB() const { return m_calculatedTransformB; }
	const Transform &getFrameOffsetA() const { return m_frameInA; }
	const Transform &getFrameOffsetB() const { return m_frameInB; }
	Transform &getFrameOffsetA() { return m_frameInA; }
	Transform &getFrameOffsetB() { return m_frameInB; }
	real_t getLowerLinLimit() { return m_lowerLinLimit; }
	void setLowerLinLimit(real_t lowerLimit) { m_lowerLinLimit = lowerLimit; }
	real_t getUpperLinLimit() { return m_upperLinLimit; }
	void setUpperLinLimit(real_t upperLimit) { m_upperLinLimit = upperLimit; }
	real_t getLowerAngLimit() { return m_lowerAngLimit; }
	void setLowerAngLimit(real_t lowerLimit) { m_lowerAngLimit = lowerLimit; }
	real_t getUpperAngLimit() { return m_upperAngLimit; }
	void setUpperAngLimit(real_t upperLimit) { m_upperAngLimit = upperLimit; }

	real_t getSoftnessDirLin() { return m_softnessDirLin; }
	real_t getRestitutionDirLin() { return m_restitutionDirLin; }
	real_t getDampingDirLin() { return m_dampingDirLin; }
	real_t getSoftnessDirAng() { return m_softnessDirAng; }
	real_t getRestitutionDirAng() { return m_restitutionDirAng; }
	real_t getDampingDirAng() { return m_dampingDirAng; }
	real_t getSoftnessLimLin() { return m_softnessLimLin; }
	real_t getRestitutionLimLin() { return m_restitutionLimLin; }
	real_t getDampingLimLin() { return m_dampingLimLin; }
	real_t getSoftnessLimAng() { return m_softnessLimAng; }
	real_t getRestitutionLimAng() { return m_restitutionLimAng; }
	real_t getDampingLimAng() { return m_dampingLimAng; }
	real_t getSoftnessOrthoLin() { return m_softnessOrthoLin; }
	real_t getRestitutionOrthoLin() { return m_restitutionOrthoLin; }
	real_t getDampingOrthoLin() { return m_dampingOrthoLin; }
	real_t getSoftnessOrthoAng() { return m_softnessOrthoAng; }
	real_t getRestitutionOrthoAng() { return m_restitutionOrthoAng; }
	real_t getDampingOrthoAng() { return m_dampingOrthoAng; }
	void setSoftnessDirLin(real_t softnessDirLin) { m_softnessDirLin = softnessDirLin; }
	void setRestitutionDirLin(real_t restitutionDirLin) { m_restitutionDirLin = restitutionDirLin; }
	void setDampingDirLin(real_t dampingDirLin) { m_dampingDirLin = dampingDirLin; }
	void setSoftnessDirAng(real_t softnessDirAng) { m_softnessDirAng = softnessDirAng; }
	void setRestitutionDirAng(real_t restitutionDirAng) { m_restitutionDirAng = restitutionDirAng; }
	void setDampingDirAng(real_t dampingDirAng) { m_dampingDirAng = dampingDirAng; }
	void setSoftnessLimLin(real_t softnessLimLin) { m_softnessLimLin = softnessLimLin; }
	void setRestitutionLimLin(real_t restitutionLimLin) { m_restitutionLimLin = restitutionLimLin; }
	void setDampingLimLin(real_t dampingLimLin) { m_dampingLimLin = dampingLimLin; }
	void setSoftnessLimAng(real_t softnessLimAng) { m_softnessLimAng = softnessLimAng; }
	void setRestitutionLimAng(real_t restitutionLimAng) { m_restitutionLimAng = restitutionLimAng; }
	void setDampingLimAng(real_t dampingLimAng) { m_dampingLimAng = dampingLimAng; }
	void setSoftnessOrthoLin(real_t softnessOrthoLin) { m_softnessOrthoLin = softnessOrthoLin; }
	void setRestitutionOrthoLin(real_t restitutionOrthoLin) { m_restitutionOrthoLin = restitutionOrthoLin; }
	void setDampingOrthoLin(real_t dampingOrthoLin) { m_dampingOrthoLin = dampingOrthoLin; }
	void setSoftnessOrthoAng(real_t softnessOrthoAng) { m_softnessOrthoAng = softnessOrthoAng; }
	void setRestitutionOrthoAng(real_t restitutionOrthoAng) { m_restitutionOrthoAng = restitutionOrthoAng; }
	void setDampingOrthoAng(real_t dampingOrthoAng) { m_dampingOrthoAng = dampingOrthoAng; }
	void setPoweredLinMotor(bool onOff) { m_poweredLinMotor = onOff; }
	bool getPoweredLinMotor() { return m_poweredLinMotor; }
	void setTargetLinMotorVelocity(real_t targetLinMotorVelocity) { m_targetLinMotorVelocity = targetLinMotorVelocity; }
	real_t getTargetLinMotorVelocity() { return m_targetLinMotorVelocity; }
	void setMaxLinMotorForce(real_t maxLinMotorForce) { m_maxLinMotorForce = maxLinMotorForce; }
	real_t getMaxLinMotorForce() { return m_maxLinMotorForce; }
	void setPoweredAngMotor(bool onOff) { m_poweredAngMotor = onOff; }
	bool getPoweredAngMotor() { return m_poweredAngMotor; }
	void setTargetAngMotorVelocity(real_t targetAngMotorVelocity) { m_targetAngMotorVelocity = targetAngMotorVelocity; }
	real_t getTargetAngMotorVelocity() { return m_targetAngMotorVelocity; }
	void setMaxAngMotorForce(real_t maxAngMotorForce) { m_maxAngMotorForce = maxAngMotorForce; }
	real_t getMaxAngMotorForce() { return m_maxAngMotorForce; }
	real_t getLinearPos() { return m_linPos; }

	// Access for ODE solver
	bool getSolveLinLimit() { return m_solveLinLim; }
	real_t getLinDepth() { return m_depth[0]; }
	bool getSolveAngLimit() { return m_solveAngLim; }
	real_t getAngDepth() { return m_angDepth; }
	// Shared code used by ODE solver
	void calculateTransforms();
	void testLinLimits();
	void testAngLimits();
	// Access for PE Solver
	Vector3 getAncorInA();
	Vector3 getAncorInB();

	void set_param(PhysicsServer3D::SliderJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer3D::SliderJointParam p_param) const;

	bool setup(real_t p_step);
	void solve(real_t p_step);

	virtual PhysicsServer3D::JointType get_type() const { return PhysicsServer3D::JOINT_TYPE_SLIDER; }
};

#endif // JOINTS_3D_SW_H
