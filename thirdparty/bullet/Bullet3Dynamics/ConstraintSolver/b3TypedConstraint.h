/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2010 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_TYPED_CONSTRAINT_H
#define B3_TYPED_CONSTRAINT_H

#include "Bullet3Common/b3Scalar.h"
#include "b3SolverConstraint.h"

class b3Serializer;

//Don't change any of the existing enum values, so add enum types at the end for serialization compatibility
enum b3TypedConstraintType
{
	B3_POINT2POINT_CONSTRAINT_TYPE = 3,
	B3_HINGE_CONSTRAINT_TYPE,
	B3_CONETWIST_CONSTRAINT_TYPE,
	B3_D6_CONSTRAINT_TYPE,
	B3_SLIDER_CONSTRAINT_TYPE,
	B3_CONTACT_CONSTRAINT_TYPE,
	B3_D6_SPRING_CONSTRAINT_TYPE,
	B3_GEAR_CONSTRAINT_TYPE,
	B3_FIXED_CONSTRAINT_TYPE,
	B3_MAX_CONSTRAINT_TYPE
};

enum b3ConstraintParams
{
	B3_CONSTRAINT_ERP = 1,
	B3_CONSTRAINT_STOP_ERP,
	B3_CONSTRAINT_CFM,
	B3_CONSTRAINT_STOP_CFM
};

#if 1
#define b3AssertConstrParams(_par) b3Assert(_par)
#else
#define b3AssertConstrParams(_par)
#endif

B3_ATTRIBUTE_ALIGNED16(struct)
b3JointFeedback
{
	b3Vector3 m_appliedForceBodyA;
	b3Vector3 m_appliedTorqueBodyA;
	b3Vector3 m_appliedForceBodyB;
	b3Vector3 m_appliedTorqueBodyB;
};

struct b3RigidBodyData;

///TypedConstraint is the baseclass for Bullet constraints and vehicles
B3_ATTRIBUTE_ALIGNED16(class)
b3TypedConstraint : public b3TypedObject
{
	int m_userConstraintType;

	union {
		int m_userConstraintId;
		void* m_userConstraintPtr;
	};

	b3Scalar m_breakingImpulseThreshold;
	bool m_isEnabled;
	bool m_needsFeedback;
	int m_overrideNumSolverIterations;

	b3TypedConstraint& operator=(b3TypedConstraint& other)
	{
		b3Assert(0);
		(void)other;
		return *this;
	}

protected:
	int m_rbA;
	int m_rbB;
	b3Scalar m_appliedImpulse;
	b3Scalar m_dbgDrawSize;
	b3JointFeedback* m_jointFeedback;

	///internal method used by the constraint solver, don't use them directly
	b3Scalar getMotorFactor(b3Scalar pos, b3Scalar lowLim, b3Scalar uppLim, b3Scalar vel, b3Scalar timeFact);

public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	virtual ~b3TypedConstraint(){};
	b3TypedConstraint(b3TypedConstraintType type, int bodyA, int bodyB);

	struct b3ConstraintInfo1
	{
		int m_numConstraintRows, nub;
	};

	struct b3ConstraintInfo2
	{
		// integrator parameters: frames per second (1/stepsize), default error
		// reduction parameter (0..1).
		b3Scalar fps, erp;

		// for the first and second body, pointers to two (linear and angular)
		// n*3 jacobian sub matrices, stored by rows. these matrices will have
		// been initialized to 0 on entry. if the second body is zero then the
		// J2xx pointers may be 0.
		b3Scalar *m_J1linearAxis, *m_J1angularAxis, *m_J2linearAxis, *m_J2angularAxis;

		// elements to jump from one row to the next in J's
		int rowskip;

		// right hand sides of the equation J*v = c + cfm * lambda. cfm is the
		// "constraint force mixing" vector. c is set to zero on entry, cfm is
		// set to a constant value (typically very small or zero) value on entry.
		b3Scalar *m_constraintError, *cfm;

		// lo and hi limits for variables (set to -/+ infinity on entry).
		b3Scalar *m_lowerLimit, *m_upperLimit;

		// findex vector for variables. see the LCP solver interface for a
		// description of what this does. this is set to -1 on entry.
		// note that the returned indexes are relative to the first index of
		// the constraint.
		int* findex;
		// number of solver iterations
		int m_numIterations;

		//damping of the velocity
		b3Scalar m_damping;
	};

	int getOverrideNumSolverIterations() const
	{
		return m_overrideNumSolverIterations;
	}

	///override the number of constraint solver iterations used to solve this constraint
	///-1 will use the default number of iterations, as specified in SolverInfo.m_numIterations
	void setOverrideNumSolverIterations(int overideNumIterations)
	{
		m_overrideNumSolverIterations = overideNumIterations;
	}

	///internal method used by the constraint solver, don't use them directly
	virtual void setupSolverConstraint(b3ConstraintArray & ca, int solverBodyA, int solverBodyB, b3Scalar timeStep)
	{
		(void)ca;
		(void)solverBodyA;
		(void)solverBodyB;
		(void)timeStep;
	}

	///internal method used by the constraint solver, don't use them directly
	virtual void getInfo1(b3ConstraintInfo1 * info, const b3RigidBodyData* bodies) = 0;

	///internal method used by the constraint solver, don't use them directly
	virtual void getInfo2(b3ConstraintInfo2 * info, const b3RigidBodyData* bodies) = 0;

	///internal method used by the constraint solver, don't use them directly
	void internalSetAppliedImpulse(b3Scalar appliedImpulse)
	{
		m_appliedImpulse = appliedImpulse;
	}
	///internal method used by the constraint solver, don't use them directly
	b3Scalar internalGetAppliedImpulse()
	{
		return m_appliedImpulse;
	}

	b3Scalar getBreakingImpulseThreshold() const
	{
		return m_breakingImpulseThreshold;
	}

	void setBreakingImpulseThreshold(b3Scalar threshold)
	{
		m_breakingImpulseThreshold = threshold;
	}

	bool isEnabled() const
	{
		return m_isEnabled;
	}

	void setEnabled(bool enabled)
	{
		m_isEnabled = enabled;
	}

	///internal method used by the constraint solver, don't use them directly
	virtual void solveConstraintObsolete(b3SolverBody& /*bodyA*/, b3SolverBody& /*bodyB*/, b3Scalar /*timeStep*/){};

	int getRigidBodyA() const
	{
		return m_rbA;
	}
	int getRigidBodyB() const
	{
		return m_rbB;
	}

	int getRigidBodyA()
	{
		return m_rbA;
	}
	int getRigidBodyB()
	{
		return m_rbB;
	}

	int getUserConstraintType() const
	{
		return m_userConstraintType;
	}

	void setUserConstraintType(int userConstraintType)
	{
		m_userConstraintType = userConstraintType;
	};

	void setUserConstraintId(int uid)
	{
		m_userConstraintId = uid;
	}

	int getUserConstraintId() const
	{
		return m_userConstraintId;
	}

	void setUserConstraintPtr(void* ptr)
	{
		m_userConstraintPtr = ptr;
	}

	void* getUserConstraintPtr()
	{
		return m_userConstraintPtr;
	}

	void setJointFeedback(b3JointFeedback * jointFeedback)
	{
		m_jointFeedback = jointFeedback;
	}

	const b3JointFeedback* getJointFeedback() const
	{
		return m_jointFeedback;
	}

	b3JointFeedback* getJointFeedback()
	{
		return m_jointFeedback;
	}

	int getUid() const
	{
		return m_userConstraintId;
	}

	bool needsFeedback() const
	{
		return m_needsFeedback;
	}

	///enableFeedback will allow to read the applied linear and angular impulse
	///use getAppliedImpulse, getAppliedLinearImpulse and getAppliedAngularImpulse to read feedback information
	void enableFeedback(bool needsFeedback)
	{
		m_needsFeedback = needsFeedback;
	}

	///getAppliedImpulse is an estimated total applied impulse.
	///This feedback could be used to determine breaking constraints or playing sounds.
	b3Scalar getAppliedImpulse() const
	{
		b3Assert(m_needsFeedback);
		return m_appliedImpulse;
	}

	b3TypedConstraintType getConstraintType() const
	{
		return b3TypedConstraintType(m_objectType);
	}

	void setDbgDrawSize(b3Scalar dbgDrawSize)
	{
		m_dbgDrawSize = dbgDrawSize;
	}
	b3Scalar getDbgDrawSize()
	{
		return m_dbgDrawSize;
	}

	///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5).
	///If no axis is provided, it uses the default axis for this constraint.
	virtual void setParam(int num, b3Scalar value, int axis = -1) = 0;

	///return the local value of parameter
	virtual b3Scalar getParam(int num, int axis = -1) const = 0;

	//	virtual	int	calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	//virtual	const char*	serialize(void* dataBuffer, b3Serializer* serializer) const;
};

// returns angle in range [-B3_2_PI, B3_2_PI], closest to one of the limits
// all arguments should be normalized angles (i.e. in range [-B3_PI, B3_PI])
B3_FORCE_INLINE b3Scalar b3AdjustAngleToLimits(b3Scalar angleInRadians, b3Scalar angleLowerLimitInRadians, b3Scalar angleUpperLimitInRadians)
{
	if (angleLowerLimitInRadians >= angleUpperLimitInRadians)
	{
		return angleInRadians;
	}
	else if (angleInRadians < angleLowerLimitInRadians)
	{
		b3Scalar diffLo = b3Fabs(b3NormalizeAngle(angleLowerLimitInRadians - angleInRadians));
		b3Scalar diffHi = b3Fabs(b3NormalizeAngle(angleUpperLimitInRadians - angleInRadians));
		return (diffLo < diffHi) ? angleInRadians : (angleInRadians + B3_2_PI);
	}
	else if (angleInRadians > angleUpperLimitInRadians)
	{
		b3Scalar diffHi = b3Fabs(b3NormalizeAngle(angleInRadians - angleUpperLimitInRadians));
		b3Scalar diffLo = b3Fabs(b3NormalizeAngle(angleInRadians - angleLowerLimitInRadians));
		return (diffLo < diffHi) ? (angleInRadians - B3_2_PI) : angleInRadians;
	}
	else
	{
		return angleInRadians;
	}
}

// clang-format off
///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct	b3TypedConstraintData
{
	int		m_bodyA;
	int		m_bodyB;
	char	*m_name;

	int	m_objectType;
	int	m_userConstraintType;
	int	m_userConstraintId;
	int	m_needsFeedback;

	float	m_appliedImpulse;
	float	m_dbgDrawSize;

	int	m_disableCollisionsBetweenLinkedBodies;
	int	m_overrideNumSolverIterations;

	float	m_breakingImpulseThreshold;
	int		m_isEnabled;
	
};

// clang-format on

/*B3_FORCE_INLINE	int	b3TypedConstraint::calculateSerializeBufferSize() const
{
	return sizeof(b3TypedConstraintData);
}
*/

class b3AngularLimit
{
private:
	b3Scalar
		m_center,
		m_halfRange,
		m_softness,
		m_biasFactor,
		m_relaxationFactor,
		m_correction,
		m_sign;

	bool
		m_solveLimit;

public:
	/// Default constructor initializes limit as inactive, allowing free constraint movement
	b3AngularLimit()
		: m_center(0.0f),
		  m_halfRange(-1.0f),
		  m_softness(0.9f),
		  m_biasFactor(0.3f),
		  m_relaxationFactor(1.0f),
		  m_correction(0.0f),
		  m_sign(0.0f),
		  m_solveLimit(false)
	{
	}

	/// Sets all limit's parameters.
	/// When low > high limit becomes inactive.
	/// When high - low > 2PI limit is ineffective too becouse no angle can exceed the limit
	void set(b3Scalar low, b3Scalar high, b3Scalar _softness = 0.9f, b3Scalar _biasFactor = 0.3f, b3Scalar _relaxationFactor = 1.0f);

	/// Checks conastaint angle against limit. If limit is active and the angle violates the limit
	/// correction is calculated.
	void test(const b3Scalar angle);

	/// Returns limit's softness
	inline b3Scalar getSoftness() const
	{
		return m_softness;
	}

	/// Returns limit's bias factor
	inline b3Scalar getBiasFactor() const
	{
		return m_biasFactor;
	}

	/// Returns limit's relaxation factor
	inline b3Scalar getRelaxationFactor() const
	{
		return m_relaxationFactor;
	}

	/// Returns correction value evaluated when test() was invoked
	inline b3Scalar getCorrection() const
	{
		return m_correction;
	}

	/// Returns sign value evaluated when test() was invoked
	inline b3Scalar getSign() const
	{
		return m_sign;
	}

	/// Gives half of the distance between min and max limit angle
	inline b3Scalar getHalfRange() const
	{
		return m_halfRange;
	}

	/// Returns true when the last test() invocation recognized limit violation
	inline bool isLimit() const
	{
		return m_solveLimit;
	}

	/// Checks given angle against limit. If limit is active and angle doesn't fit it, the angle
	/// returned is modified so it equals to the limit closest to given angle.
	void fit(b3Scalar& angle) const;

	/// Returns correction value multiplied by sign value
	b3Scalar getError() const;

	b3Scalar getLow() const;

	b3Scalar getHigh() const;
};

#endif  //B3_TYPED_CONSTRAINT_H
