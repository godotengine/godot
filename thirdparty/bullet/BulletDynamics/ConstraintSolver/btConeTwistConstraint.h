/*
Bullet Continuous Collision Detection and Physics Library
btConeTwistConstraint is Copyright (c) 2007 Starbreeze Studios

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Written by: Marcus Hennix
*/



/*
Overview:

btConeTwistConstraint can be used to simulate ragdoll joints (upper arm, leg etc).
It is a fixed translation, 3 degree-of-freedom (DOF) rotational "joint".
It divides the 3 rotational DOFs into swing (movement within a cone) and twist.
Swing is divided into swing1 and swing2 which can have different limits, giving an elliptical shape.
(Note: the cone's base isn't flat, so this ellipse is "embedded" on the surface of a sphere.)

In the contraint's frame of reference:
twist is along the x-axis,
and swing 1 and 2 are along the z and y axes respectively.
*/



#ifndef BT_CONETWISTCONSTRAINT_H
#define BT_CONETWISTCONSTRAINT_H

#include "LinearMath/btVector3.h"
#include "btJacobianEntry.h"
#include "btTypedConstraint.h"

#ifdef BT_USE_DOUBLE_PRECISION
#define btConeTwistConstraintData2	btConeTwistConstraintDoubleData
#define btConeTwistConstraintDataName	"btConeTwistConstraintDoubleData"
#else
#define btConeTwistConstraintData2	btConeTwistConstraintData 
#define btConeTwistConstraintDataName	"btConeTwistConstraintData" 
#endif //BT_USE_DOUBLE_PRECISION


class btRigidBody;

enum btConeTwistFlags
{
	BT_CONETWIST_FLAGS_LIN_CFM = 1,
	BT_CONETWIST_FLAGS_LIN_ERP = 2,
	BT_CONETWIST_FLAGS_ANG_CFM = 4
};

///btConeTwistConstraint can be used to simulate ragdoll joints (upper arm, leg etc)
ATTRIBUTE_ALIGNED16(class) btConeTwistConstraint : public btTypedConstraint
{
#ifdef IN_PARALLELL_SOLVER
public:
#endif
	btJacobianEntry	m_jac[3]; //3 orthogonal linear constraints

	btTransform m_rbAFrame; 
	btTransform m_rbBFrame;

	btScalar	m_limitSoftness;
	btScalar	m_biasFactor;
	btScalar	m_relaxationFactor;

	btScalar	m_damping;

	btScalar	m_swingSpan1;
	btScalar	m_swingSpan2;
	btScalar	m_twistSpan;

	btScalar	m_fixThresh;

	btVector3   m_swingAxis;
	btVector3	m_twistAxis;

	btScalar	m_kSwing;
	btScalar	m_kTwist;

	btScalar	m_twistLimitSign;
	btScalar	m_swingCorrection;
	btScalar	m_twistCorrection;

	btScalar	m_twistAngle;

	btScalar	m_accSwingLimitImpulse;
	btScalar	m_accTwistLimitImpulse;

	bool		m_angularOnly;
	bool		m_solveTwistLimit;
	bool		m_solveSwingLimit;

	bool	m_useSolveConstraintObsolete;

	// not yet used...
	btScalar	m_swingLimitRatio;
	btScalar	m_twistLimitRatio;
	btVector3   m_twistAxisA;

	// motor
	bool		 m_bMotorEnabled;
	bool		 m_bNormalizedMotorStrength;
	btQuaternion m_qTarget;
	btScalar	 m_maxMotorImpulse;
	btVector3	 m_accMotorImpulse;
	
	// parameters
	int			m_flags;
	btScalar	m_linCFM;
	btScalar	m_linERP;
	btScalar	m_angCFM;
	
protected:

	void init();

	void computeConeLimitInfo(const btQuaternion& qCone, // in
		btScalar& swingAngle, btVector3& vSwingAxis, btScalar& swingLimit); // all outs

	void computeTwistLimitInfo(const btQuaternion& qTwist, // in
		btScalar& twistAngle, btVector3& vTwistAxis); // all outs

	void adjustSwingAxisToUseEllipseNormal(btVector3& vSwingAxis) const;


public:

	BT_DECLARE_ALIGNED_ALLOCATOR();

	btConeTwistConstraint(btRigidBody& rbA,btRigidBody& rbB,const btTransform& rbAFrame, const btTransform& rbBFrame);
	
	btConeTwistConstraint(btRigidBody& rbA,const btTransform& rbAFrame);

	virtual void	buildJacobian();

	virtual void getInfo1 (btConstraintInfo1* info);

	void	getInfo1NonVirtual(btConstraintInfo1* info);
	
	virtual void getInfo2 (btConstraintInfo2* info);
	
	void	getInfo2NonVirtual(btConstraintInfo2* info,const btTransform& transA,const btTransform& transB,const btMatrix3x3& invInertiaWorldA,const btMatrix3x3& invInertiaWorldB);

	virtual	void	solveConstraintObsolete(btSolverBody& bodyA,btSolverBody& bodyB,btScalar	timeStep);

    
	void	updateRHS(btScalar	timeStep);


	const btRigidBody& getRigidBodyA() const
	{
		return m_rbA;
	}
	const btRigidBody& getRigidBodyB() const
	{
		return m_rbB;
	}

	void	setAngularOnly(bool angularOnly)
	{
		m_angularOnly = angularOnly;
	}
	
	bool    getAngularOnly() const
	{
	    return m_angularOnly;
	}

	void	setLimit(int limitIndex,btScalar limitValue)
	{
		switch (limitIndex)
		{
		case 3:
			{
				m_twistSpan = limitValue;
				break;
			}
		case 4:
			{
				m_swingSpan2 = limitValue;
				break;
			}
		case 5:
			{
				m_swingSpan1 = limitValue;
				break;
			}
		default:
			{
			}
		};
	}

    btScalar getLimit(int limitIndex) const
	{
		switch (limitIndex)
		{
		case 3:
			{
				return m_twistSpan;
				break;
			}
		case 4:
			{
				return m_swingSpan2;
				break;
			}
		case 5:
			{
				return m_swingSpan1;
				break;
			}
		default:
			{
			    btAssert(0 && "Invalid limitIndex specified for btConeTwistConstraint");
			    return 0.0;
			}
		};
	}

	// setLimit(), a few notes:
	// _softness:
	//		0->1, recommend ~0.8->1.
	//		describes % of limits where movement is free.
	//		beyond this softness %, the limit is gradually enforced until the "hard" (1.0) limit is reached.
	// _biasFactor:
	//		0->1?, recommend 0.3 +/-0.3 or so.
	//		strength with which constraint resists zeroth order (angular, not angular velocity) limit violation.
	// __relaxationFactor:
	//		0->1, recommend to stay near 1.
	//		the lower the value, the less the constraint will fight velocities which violate the angular limits.
	void	setLimit(btScalar _swingSpan1,btScalar _swingSpan2,btScalar _twistSpan, btScalar _softness = 1.f, btScalar _biasFactor = 0.3f, btScalar _relaxationFactor = 1.0f)
	{
		m_swingSpan1 = _swingSpan1;
		m_swingSpan2 = _swingSpan2;
		m_twistSpan  = _twistSpan;

		m_limitSoftness =  _softness;
		m_biasFactor = _biasFactor;
		m_relaxationFactor = _relaxationFactor;
	}

	const btTransform& getAFrame() const { return m_rbAFrame; };	
	const btTransform& getBFrame() const { return m_rbBFrame; };

	inline int getSolveTwistLimit()
	{
		return m_solveTwistLimit;
	}

	inline int getSolveSwingLimit()
	{
		return m_solveSwingLimit;
	}

	inline btScalar getTwistLimitSign()
	{
		return m_twistLimitSign;
	}

	void calcAngleInfo();
	void calcAngleInfo2(const btTransform& transA, const btTransform& transB,const btMatrix3x3& invInertiaWorldA,const btMatrix3x3& invInertiaWorldB);

	inline btScalar getSwingSpan1() const
	{
		return m_swingSpan1;
	}
	inline btScalar getSwingSpan2() const
	{
		return m_swingSpan2;
	}
	inline btScalar getTwistSpan() const
	{
		return m_twistSpan;
	}
	inline btScalar getLimitSoftness() const
	{
		return m_limitSoftness;
	}
	inline btScalar getBiasFactor() const
	{
		return m_biasFactor;
	}
	inline btScalar getRelaxationFactor() const
	{
		return m_relaxationFactor;
	}
	inline btScalar getTwistAngle() const
	{
		return m_twistAngle;
	}
	bool isPastSwingLimit() { return m_solveSwingLimit; }

	btScalar getDamping() const { return m_damping; }
	void setDamping(btScalar damping) { m_damping = damping; }

	void enableMotor(bool b) { m_bMotorEnabled = b; }
	bool isMotorEnabled() const { return m_bMotorEnabled; }
	btScalar getMaxMotorImpulse() const { return m_maxMotorImpulse; }
	bool isMaxMotorImpulseNormalized() const { return m_bNormalizedMotorStrength; }
	void setMaxMotorImpulse(btScalar maxMotorImpulse) { m_maxMotorImpulse = maxMotorImpulse; m_bNormalizedMotorStrength = false; }
	void setMaxMotorImpulseNormalized(btScalar maxMotorImpulse) { m_maxMotorImpulse = maxMotorImpulse; m_bNormalizedMotorStrength = true; }

	btScalar getFixThresh() { return m_fixThresh; }
	void setFixThresh(btScalar fixThresh) { m_fixThresh = fixThresh; }

	// setMotorTarget:
	// q: the desired rotation of bodyA wrt bodyB.
	// note: if q violates the joint limits, the internal target is clamped to avoid conflicting impulses (very bad for stability)
	// note: don't forget to enableMotor()
	void setMotorTarget(const btQuaternion &q);
	const btQuaternion& getMotorTarget() const { return m_qTarget; }

	// same as above, but q is the desired rotation of frameA wrt frameB in constraint space
	void setMotorTargetInConstraintSpace(const btQuaternion &q);

	btVector3 GetPointForAngle(btScalar fAngleInRadians, btScalar fLength) const;

	///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5). 
	///If no axis is provided, it uses the default axis for this constraint.
	virtual	void setParam(int num, btScalar value, int axis = -1);

	virtual void setFrames(const btTransform& frameA, const btTransform& frameB);

	const btTransform& getFrameOffsetA() const
	{
		return m_rbAFrame;
	}

	const btTransform& getFrameOffsetB() const
	{
		return m_rbBFrame;
	}


	///return the local value of parameter
	virtual	btScalar getParam(int num, int axis = -1) const;

	int getFlags() const
	{
		return m_flags;
	}

	virtual	int	calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual	const char*	serialize(void* dataBuffer, btSerializer* serializer) const;

};


	
struct	btConeTwistConstraintDoubleData
{
	btTypedConstraintDoubleData	m_typeConstraintData;
	btTransformDoubleData m_rbAFrame;
	btTransformDoubleData m_rbBFrame;

	//limits
	double	m_swingSpan1;
	double	m_swingSpan2;
	double	m_twistSpan;
	double	m_limitSoftness;
	double	m_biasFactor;
	double	m_relaxationFactor;

	double	m_damping;
		
	

};

#ifdef BT_BACKWARDS_COMPATIBLE_SERIALIZATION
///this structure is not used, except for loading pre-2.82 .bullet files
struct	btConeTwistConstraintData
{
	btTypedConstraintData	m_typeConstraintData;
	btTransformFloatData m_rbAFrame;
	btTransformFloatData m_rbBFrame;

	//limits
	float	m_swingSpan1;
	float	m_swingSpan2;
	float	m_twistSpan;
	float	m_limitSoftness;
	float	m_biasFactor;
	float	m_relaxationFactor;

	float	m_damping;
		
	char m_pad[4];

};
#endif //BT_BACKWARDS_COMPATIBLE_SERIALIZATION
//

SIMD_FORCE_INLINE int	btConeTwistConstraint::calculateSerializeBufferSize() const
{
	return sizeof(btConeTwistConstraintData2);

}


	///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE const char*	btConeTwistConstraint::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btConeTwistConstraintData2* cone = (btConeTwistConstraintData2*) dataBuffer;
	btTypedConstraint::serialize(&cone->m_typeConstraintData,serializer);

	m_rbAFrame.serialize(cone->m_rbAFrame);
	m_rbBFrame.serialize(cone->m_rbBFrame);
	
	cone->m_swingSpan1 = m_swingSpan1;
	cone->m_swingSpan2 = m_swingSpan2;
	cone->m_twistSpan = m_twistSpan;
	cone->m_limitSoftness = m_limitSoftness;
	cone->m_biasFactor = m_biasFactor;
	cone->m_relaxationFactor = m_relaxationFactor;
	cone->m_damping = m_damping;

	return btConeTwistConstraintDataName;
}


#endif //BT_CONETWISTCONSTRAINT_H
