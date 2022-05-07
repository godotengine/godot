/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

/// 2009 March: b3Generic6DofConstraint refactored by Roman Ponomarev
/// Added support for generic constraint solver through getInfo1/getInfo2 methods

/*
2007-09-09
b3Generic6DofConstraint Refactored by Francisco Le?n
email: projectileman@yahoo.com
http://gimpact.sf.net
*/

#ifndef B3_GENERIC_6DOF_CONSTRAINT_H
#define B3_GENERIC_6DOF_CONSTRAINT_H

#include "Bullet3Common/b3Vector3.h"
#include "b3JacobianEntry.h"
#include "b3TypedConstraint.h"

struct b3RigidBodyData;

//! Rotation Limit structure for generic joints
class b3RotationalLimitMotor
{
public:
	//! limit_parameters
	//!@{
	b3Scalar m_loLimit;         //!< joint limit
	b3Scalar m_hiLimit;         //!< joint limit
	b3Scalar m_targetVelocity;  //!< target motor velocity
	b3Scalar m_maxMotorForce;   //!< max force on motor
	b3Scalar m_maxLimitForce;   //!< max force on limit
	b3Scalar m_damping;         //!< Damping.
	b3Scalar m_limitSoftness;   //! Relaxation factor
	b3Scalar m_normalCFM;       //!< Constraint force mixing factor
	b3Scalar m_stopERP;         //!< Error tolerance factor when joint is at limit
	b3Scalar m_stopCFM;         //!< Constraint force mixing factor when joint is at limit
	b3Scalar m_bounce;          //!< restitution factor
	bool m_enableMotor;

	//!@}

	//! temp_variables
	//!@{
	b3Scalar m_currentLimitError;  //!  How much is violated this limit
	b3Scalar m_currentPosition;    //!  current value of angle
	int m_currentLimit;            //!< 0=free, 1=at lo limit, 2=at hi limit
	b3Scalar m_accumulatedImpulse;
	//!@}

	b3RotationalLimitMotor()
	{
		m_accumulatedImpulse = 0.f;
		m_targetVelocity = 0;
		m_maxMotorForce = 6.0f;
		m_maxLimitForce = 300.0f;
		m_loLimit = 1.0f;
		m_hiLimit = -1.0f;
		m_normalCFM = 0.f;
		m_stopERP = 0.2f;
		m_stopCFM = 0.f;
		m_bounce = 0.0f;
		m_damping = 1.0f;
		m_limitSoftness = 0.5f;
		m_currentLimit = 0;
		m_currentLimitError = 0;
		m_enableMotor = false;
	}

	b3RotationalLimitMotor(const b3RotationalLimitMotor& limot)
	{
		m_targetVelocity = limot.m_targetVelocity;
		m_maxMotorForce = limot.m_maxMotorForce;
		m_limitSoftness = limot.m_limitSoftness;
		m_loLimit = limot.m_loLimit;
		m_hiLimit = limot.m_hiLimit;
		m_normalCFM = limot.m_normalCFM;
		m_stopERP = limot.m_stopERP;
		m_stopCFM = limot.m_stopCFM;
		m_bounce = limot.m_bounce;
		m_currentLimit = limot.m_currentLimit;
		m_currentLimitError = limot.m_currentLimitError;
		m_enableMotor = limot.m_enableMotor;
	}

	//! Is limited
	bool isLimited()
	{
		if (m_loLimit > m_hiLimit) return false;
		return true;
	}

	//! Need apply correction
	bool needApplyTorques()
	{
		if (m_currentLimit == 0 && m_enableMotor == false) return false;
		return true;
	}

	//! calculates  error
	/*!
	calculates m_currentLimit and m_currentLimitError.
	*/
	int testLimitValue(b3Scalar test_value);

	//! apply the correction impulses for two bodies
	b3Scalar solveAngularLimits(b3Scalar timeStep, b3Vector3& axis, b3Scalar jacDiagABInv, b3RigidBodyData* body0, b3RigidBodyData* body1);
};

class b3TranslationalLimitMotor
{
public:
	b3Vector3 m_lowerLimit;  //!< the constraint lower limits
	b3Vector3 m_upperLimit;  //!< the constraint upper limits
	b3Vector3 m_accumulatedImpulse;
	//! Linear_Limit_parameters
	//!@{
	b3Vector3 m_normalCFM;          //!< Constraint force mixing factor
	b3Vector3 m_stopERP;            //!< Error tolerance factor when joint is at limit
	b3Vector3 m_stopCFM;            //!< Constraint force mixing factor when joint is at limit
	b3Vector3 m_targetVelocity;     //!< target motor velocity
	b3Vector3 m_maxMotorForce;      //!< max force on motor
	b3Vector3 m_currentLimitError;  //!  How much is violated this limit
	b3Vector3 m_currentLinearDiff;  //!  Current relative offset of constraint frames
	b3Scalar m_limitSoftness;       //!< Softness for linear limit
	b3Scalar m_damping;             //!< Damping for linear limit
	b3Scalar m_restitution;         //! Bounce parameter for linear limit
	//!@}
	bool m_enableMotor[3];
	int m_currentLimit[3];  //!< 0=free, 1=at lower limit, 2=at upper limit

	b3TranslationalLimitMotor()
	{
		m_lowerLimit.setValue(0.f, 0.f, 0.f);
		m_upperLimit.setValue(0.f, 0.f, 0.f);
		m_accumulatedImpulse.setValue(0.f, 0.f, 0.f);
		m_normalCFM.setValue(0.f, 0.f, 0.f);
		m_stopERP.setValue(0.2f, 0.2f, 0.2f);
		m_stopCFM.setValue(0.f, 0.f, 0.f);

		m_limitSoftness = 0.7f;
		m_damping = b3Scalar(1.0f);
		m_restitution = b3Scalar(0.5f);
		for (int i = 0; i < 3; i++)
		{
			m_enableMotor[i] = false;
			m_targetVelocity[i] = b3Scalar(0.f);
			m_maxMotorForce[i] = b3Scalar(0.f);
		}
	}

	b3TranslationalLimitMotor(const b3TranslationalLimitMotor& other)
	{
		m_lowerLimit = other.m_lowerLimit;
		m_upperLimit = other.m_upperLimit;
		m_accumulatedImpulse = other.m_accumulatedImpulse;

		m_limitSoftness = other.m_limitSoftness;
		m_damping = other.m_damping;
		m_restitution = other.m_restitution;
		m_normalCFM = other.m_normalCFM;
		m_stopERP = other.m_stopERP;
		m_stopCFM = other.m_stopCFM;

		for (int i = 0; i < 3; i++)
		{
			m_enableMotor[i] = other.m_enableMotor[i];
			m_targetVelocity[i] = other.m_targetVelocity[i];
			m_maxMotorForce[i] = other.m_maxMotorForce[i];
		}
	}

	//! Test limit
	/*!
    - free means upper < lower,
    - locked means upper == lower
    - limited means upper > lower
    - limitIndex: first 3 are linear, next 3 are angular
    */
	inline bool isLimited(int limitIndex)
	{
		return (m_upperLimit[limitIndex] >= m_lowerLimit[limitIndex]);
	}
	inline bool needApplyForce(int limitIndex)
	{
		if (m_currentLimit[limitIndex] == 0 && m_enableMotor[limitIndex] == false) return false;
		return true;
	}
	int testLimitValue(int limitIndex, b3Scalar test_value);

	b3Scalar solveLinearAxis(
		b3Scalar timeStep,
		b3Scalar jacDiagABInv,
		b3RigidBodyData& body1, const b3Vector3& pointInA,
		b3RigidBodyData& body2, const b3Vector3& pointInB,
		int limit_index,
		const b3Vector3& axis_normal_on_a,
		const b3Vector3& anchorPos);
};

enum b36DofFlags
{
	B3_6DOF_FLAGS_CFM_NORM = 1,
	B3_6DOF_FLAGS_CFM_STOP = 2,
	B3_6DOF_FLAGS_ERP_STOP = 4
};
#define B3_6DOF_FLAGS_AXIS_SHIFT 3  // bits per axis

/// b3Generic6DofConstraint between two rigidbodies each with a pivotpoint that descibes the axis location in local space
/*!
b3Generic6DofConstraint can leave any of the 6 degree of freedom 'free' or 'locked'.
currently this limit supports rotational motors<br>
<ul>
<li> For Linear limits, use b3Generic6DofConstraint.setLinearUpperLimit, b3Generic6DofConstraint.setLinearLowerLimit. You can set the parameters with the b3TranslationalLimitMotor structure accsesible through the b3Generic6DofConstraint.getTranslationalLimitMotor method.
At this moment translational motors are not supported. May be in the future. </li>

<li> For Angular limits, use the b3RotationalLimitMotor structure for configuring the limit.
This is accessible through b3Generic6DofConstraint.getLimitMotor method,
This brings support for limit parameters and motors. </li>

<li> Angulars limits have these possible ranges:
<table border=1 >
<tr>
	<td><b>AXIS</b></td>
	<td><b>MIN ANGLE</b></td>
	<td><b>MAX ANGLE</b></td>
</tr><tr>
	<td>X</td>
	<td>-PI</td>
	<td>PI</td>
</tr><tr>
	<td>Y</td>
	<td>-PI/2</td>
	<td>PI/2</td>
</tr><tr>
	<td>Z</td>
	<td>-PI</td>
	<td>PI</td>
</tr>
</table>
</li>
</ul>

*/
B3_ATTRIBUTE_ALIGNED16(class)
b3Generic6DofConstraint : public b3TypedConstraint
{
protected:
	//! relative_frames
	//!@{
	b3Transform m_frameInA;  //!< the constraint space w.r.t body A
	b3Transform m_frameInB;  //!< the constraint space w.r.t body B
	//!@}

	//! Jacobians
	//!@{
	//    b3JacobianEntry	m_jacLinear[3];//!< 3 orthogonal linear constraints
	//    b3JacobianEntry	m_jacAng[3];//!< 3 orthogonal angular constraints
	//!@}

	//! Linear_Limit_parameters
	//!@{
	b3TranslationalLimitMotor m_linearLimits;
	//!@}

	//! hinge_parameters
	//!@{
	b3RotationalLimitMotor m_angularLimits[3];
	//!@}

protected:
	//! temporal variables
	//!@{
	b3Transform m_calculatedTransformA;
	b3Transform m_calculatedTransformB;
	b3Vector3 m_calculatedAxisAngleDiff;
	b3Vector3 m_calculatedAxis[3];
	b3Vector3 m_calculatedLinearDiff;
	b3Scalar m_timeStep;
	b3Scalar m_factA;
	b3Scalar m_factB;
	bool m_hasStaticBody;

	b3Vector3 m_AnchorPos;  // point betwen pivots of bodies A and B to solve linear axes

	bool m_useLinearReferenceFrameA;
	bool m_useOffsetForConstraintFrame;

	int m_flags;

	//!@}

	b3Generic6DofConstraint& operator=(b3Generic6DofConstraint& other)
	{
		b3Assert(0);
		(void)other;
		return *this;
	}

	int setAngularLimits(b3ConstraintInfo2 * info, int row_offset, const b3Transform& transA, const b3Transform& transB, const b3Vector3& linVelA, const b3Vector3& linVelB, const b3Vector3& angVelA, const b3Vector3& angVelB);

	int setLinearLimits(b3ConstraintInfo2 * info, int row, const b3Transform& transA, const b3Transform& transB, const b3Vector3& linVelA, const b3Vector3& linVelB, const b3Vector3& angVelA, const b3Vector3& angVelB);

	// tests linear limits
	void calculateLinearInfo();

	//! calcs the euler angles between the two bodies.
	void calculateAngleInfo();

public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	b3Generic6DofConstraint(int rbA, int rbB, const b3Transform& frameInA, const b3Transform& frameInB, bool useLinearReferenceFrameA, const b3RigidBodyData* bodies);

	//! Calcs global transform of the offsets
	/*!
	Calcs the global transform for the joint offset for body A an B, and also calcs the agle differences between the bodies.
	\sa b3Generic6DofConstraint.getCalculatedTransformA , b3Generic6DofConstraint.getCalculatedTransformB, b3Generic6DofConstraint.calculateAngleInfo
	*/
	void calculateTransforms(const b3Transform& transA, const b3Transform& transB, const b3RigidBodyData* bodies);

	void calculateTransforms(const b3RigidBodyData* bodies);

	//! Gets the global transform of the offset for body A
	/*!
    \sa b3Generic6DofConstraint.getFrameOffsetA, b3Generic6DofConstraint.getFrameOffsetB, b3Generic6DofConstraint.calculateAngleInfo.
    */
	const b3Transform& getCalculatedTransformA() const
	{
		return m_calculatedTransformA;
	}

	//! Gets the global transform of the offset for body B
	/*!
    \sa b3Generic6DofConstraint.getFrameOffsetA, b3Generic6DofConstraint.getFrameOffsetB, b3Generic6DofConstraint.calculateAngleInfo.
    */
	const b3Transform& getCalculatedTransformB() const
	{
		return m_calculatedTransformB;
	}

	const b3Transform& getFrameOffsetA() const
	{
		return m_frameInA;
	}

	const b3Transform& getFrameOffsetB() const
	{
		return m_frameInB;
	}

	b3Transform& getFrameOffsetA()
	{
		return m_frameInA;
	}

	b3Transform& getFrameOffsetB()
	{
		return m_frameInB;
	}

	virtual void getInfo1(b3ConstraintInfo1 * info, const b3RigidBodyData* bodies);

	void getInfo1NonVirtual(b3ConstraintInfo1 * info, const b3RigidBodyData* bodies);

	virtual void getInfo2(b3ConstraintInfo2 * info, const b3RigidBodyData* bodies);

	void getInfo2NonVirtual(b3ConstraintInfo2 * info, const b3Transform& transA, const b3Transform& transB, const b3Vector3& linVelA, const b3Vector3& linVelB, const b3Vector3& angVelA, const b3Vector3& angVelB, const b3RigidBodyData* bodies);

	void updateRHS(b3Scalar timeStep);

	//! Get the rotation axis in global coordinates
	b3Vector3 getAxis(int axis_index) const;

	//! Get the relative Euler angle
	/*!
	\pre b3Generic6DofConstraint::calculateTransforms() must be called previously.
	*/
	b3Scalar getAngle(int axis_index) const;

	//! Get the relative position of the constraint pivot
	/*!
	\pre b3Generic6DofConstraint::calculateTransforms() must be called previously.
	*/
	b3Scalar getRelativePivotPosition(int axis_index) const;

	void setFrames(const b3Transform& frameA, const b3Transform& frameB, const b3RigidBodyData* bodies);

	//! Test angular limit.
	/*!
	Calculates angular correction and returns true if limit needs to be corrected.
	\pre b3Generic6DofConstraint::calculateTransforms() must be called previously.
	*/
	bool testAngularLimitMotor(int axis_index);

	void setLinearLowerLimit(const b3Vector3& linearLower)
	{
		m_linearLimits.m_lowerLimit = linearLower;
	}

	void getLinearLowerLimit(b3Vector3 & linearLower)
	{
		linearLower = m_linearLimits.m_lowerLimit;
	}

	void setLinearUpperLimit(const b3Vector3& linearUpper)
	{
		m_linearLimits.m_upperLimit = linearUpper;
	}

	void getLinearUpperLimit(b3Vector3 & linearUpper)
	{
		linearUpper = m_linearLimits.m_upperLimit;
	}

	void setAngularLowerLimit(const b3Vector3& angularLower)
	{
		for (int i = 0; i < 3; i++)
			m_angularLimits[i].m_loLimit = b3NormalizeAngle(angularLower[i]);
	}

	void getAngularLowerLimit(b3Vector3 & angularLower)
	{
		for (int i = 0; i < 3; i++)
			angularLower[i] = m_angularLimits[i].m_loLimit;
	}

	void setAngularUpperLimit(const b3Vector3& angularUpper)
	{
		for (int i = 0; i < 3; i++)
			m_angularLimits[i].m_hiLimit = b3NormalizeAngle(angularUpper[i]);
	}

	void getAngularUpperLimit(b3Vector3 & angularUpper)
	{
		for (int i = 0; i < 3; i++)
			angularUpper[i] = m_angularLimits[i].m_hiLimit;
	}

	//! Retrieves the angular limit informacion
	b3RotationalLimitMotor* getRotationalLimitMotor(int index)
	{
		return &m_angularLimits[index];
	}

	//! Retrieves the  limit informacion
	b3TranslationalLimitMotor* getTranslationalLimitMotor()
	{
		return &m_linearLimits;
	}

	//first 3 are linear, next 3 are angular
	void setLimit(int axis, b3Scalar lo, b3Scalar hi)
	{
		if (axis < 3)
		{
			m_linearLimits.m_lowerLimit[axis] = lo;
			m_linearLimits.m_upperLimit[axis] = hi;
		}
		else
		{
			lo = b3NormalizeAngle(lo);
			hi = b3NormalizeAngle(hi);
			m_angularLimits[axis - 3].m_loLimit = lo;
			m_angularLimits[axis - 3].m_hiLimit = hi;
		}
	}

	//! Test limit
	/*!
    - free means upper < lower,
    - locked means upper == lower
    - limited means upper > lower
    - limitIndex: first 3 are linear, next 3 are angular
    */
	bool isLimited(int limitIndex)
	{
		if (limitIndex < 3)
		{
			return m_linearLimits.isLimited(limitIndex);
		}
		return m_angularLimits[limitIndex - 3].isLimited();
	}

	virtual void calcAnchorPos(const b3RigidBodyData* bodies);  // overridable

	int get_limit_motor_info2(b3RotationalLimitMotor * limot,
							  const b3Transform& transA, const b3Transform& transB, const b3Vector3& linVelA, const b3Vector3& linVelB, const b3Vector3& angVelA, const b3Vector3& angVelB,
							  b3ConstraintInfo2* info, int row, b3Vector3& ax1, int rotational, int rotAllowed = false);

	// access for UseFrameOffset
	bool getUseFrameOffset() { return m_useOffsetForConstraintFrame; }
	void setUseFrameOffset(bool frameOffsetOnOff) { m_useOffsetForConstraintFrame = frameOffsetOnOff; }

	///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5).
	///If no axis is provided, it uses the default axis for this constraint.
	virtual void setParam(int num, b3Scalar value, int axis = -1);
	///return the local value of parameter
	virtual b3Scalar getParam(int num, int axis = -1) const;

	void setAxis(const b3Vector3& axis1, const b3Vector3& axis2, const b3RigidBodyData* bodies);
};

#endif  //B3_GENERIC_6DOF_CONSTRAINT_H
