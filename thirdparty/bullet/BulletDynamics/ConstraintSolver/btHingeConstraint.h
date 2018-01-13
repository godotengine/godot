/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

/* Hinge Constraint by Dirk Gregorius. Limits added by Marcus Hennix at Starbreeze Studios */

#ifndef BT_HINGECONSTRAINT_H
#define BT_HINGECONSTRAINT_H

#define _BT_USE_CENTER_LIMIT_ 1


#include "LinearMath/btVector3.h"
#include "btJacobianEntry.h"
#include "btTypedConstraint.h"

class btRigidBody;

#ifdef BT_USE_DOUBLE_PRECISION
#define btHingeConstraintData	btHingeConstraintDoubleData2 //rename to 2 for backwards compatibility, so we can still load the 'btHingeConstraintDoubleData' version
#define btHingeConstraintDataName	"btHingeConstraintDoubleData2" 
#else
#define btHingeConstraintData	btHingeConstraintFloatData
#define btHingeConstraintDataName	"btHingeConstraintFloatData"
#endif //BT_USE_DOUBLE_PRECISION



enum btHingeFlags
{
	BT_HINGE_FLAGS_CFM_STOP = 1,
	BT_HINGE_FLAGS_ERP_STOP = 2,
	BT_HINGE_FLAGS_CFM_NORM = 4,
	BT_HINGE_FLAGS_ERP_NORM = 8
};


/// hinge constraint between two rigidbodies each with a pivotpoint that descibes the axis location in local space
/// axis defines the orientation of the hinge axis
ATTRIBUTE_ALIGNED16(class) btHingeConstraint : public btTypedConstraint
{
#ifdef IN_PARALLELL_SOLVER
public:
#endif
	btJacobianEntry	m_jac[3]; //3 orthogonal linear constraints
	btJacobianEntry	m_jacAng[3]; //2 orthogonal angular constraints+ 1 for limit/motor

	btTransform m_rbAFrame; // constraint axii. Assumes z is hinge axis.
	btTransform m_rbBFrame;

	btScalar	m_motorTargetVelocity;
	btScalar	m_maxMotorImpulse;


#ifdef	_BT_USE_CENTER_LIMIT_
	btAngularLimit	m_limit;
#else
	btScalar	m_lowerLimit;	
	btScalar	m_upperLimit;	
	btScalar	m_limitSign;
	btScalar	m_correction;

	btScalar	m_limitSoftness; 
	btScalar	m_biasFactor; 
	btScalar	m_relaxationFactor; 

	bool		m_solveLimit;
#endif

	btScalar	m_kHinge;


	btScalar	m_accLimitImpulse;
	btScalar	m_hingeAngle;
	btScalar	m_referenceSign;

	bool		m_angularOnly;
	bool		m_enableAngularMotor;
	bool		m_useSolveConstraintObsolete;
	bool		m_useOffsetForConstraintFrame;
	bool		m_useReferenceFrameA;

	btScalar	m_accMotorImpulse;

	int			m_flags;
	btScalar	m_normalCFM;
	btScalar	m_normalERP;
	btScalar	m_stopCFM;
	btScalar	m_stopERP;

	
public:

	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btHingeConstraint(btRigidBody& rbA,btRigidBody& rbB, const btVector3& pivotInA,const btVector3& pivotInB, const btVector3& axisInA,const btVector3& axisInB, bool useReferenceFrameA = false);

	btHingeConstraint(btRigidBody& rbA,const btVector3& pivotInA,const btVector3& axisInA, bool useReferenceFrameA = false);
	
	btHingeConstraint(btRigidBody& rbA,btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool useReferenceFrameA = false);

	btHingeConstraint(btRigidBody& rbA,const btTransform& rbAFrame, bool useReferenceFrameA = false);


	virtual void	buildJacobian();

	virtual void getInfo1 (btConstraintInfo1* info);

	void getInfo1NonVirtual(btConstraintInfo1* info);

	virtual void getInfo2 (btConstraintInfo2* info);

	void	getInfo2NonVirtual(btConstraintInfo2* info,const btTransform& transA,const btTransform& transB,const btVector3& angVelA,const btVector3& angVelB);

	void	getInfo2Internal(btConstraintInfo2* info,const btTransform& transA,const btTransform& transB,const btVector3& angVelA,const btVector3& angVelB);
	void	getInfo2InternalUsingFrameOffset(btConstraintInfo2* info,const btTransform& transA,const btTransform& transB,const btVector3& angVelA,const btVector3& angVelB);
		

	void	updateRHS(btScalar	timeStep);

	const btRigidBody& getRigidBodyA() const
	{
		return m_rbA;
	}
	const btRigidBody& getRigidBodyB() const
	{
		return m_rbB;
	}

	btRigidBody& getRigidBodyA()	
	{		
		return m_rbA;	
	}	

	btRigidBody& getRigidBodyB()	
	{		
		return m_rbB;	
	}

	btTransform& getFrameOffsetA()
	{
	return m_rbAFrame;
	}

	btTransform& getFrameOffsetB()
	{
		return m_rbBFrame;
	}

	void setFrames(const btTransform& frameA, const btTransform& frameB);
	
	void	setAngularOnly(bool angularOnly)
	{
		m_angularOnly = angularOnly;
	}

	void	enableAngularMotor(bool enableMotor,btScalar targetVelocity,btScalar maxMotorImpulse)
	{
		m_enableAngularMotor  = enableMotor;
		m_motorTargetVelocity = targetVelocity;
		m_maxMotorImpulse = maxMotorImpulse;
	}

	// extra motor API, including ability to set a target rotation (as opposed to angular velocity)
	// note: setMotorTarget sets angular velocity under the hood, so you must call it every tick to
	//       maintain a given angular target.
	void enableMotor(bool enableMotor) 	{ m_enableAngularMotor = enableMotor; }
	void setMaxMotorImpulse(btScalar maxMotorImpulse) { m_maxMotorImpulse = maxMotorImpulse; }
	void setMotorTargetVelocity(btScalar motorTargetVelocity) { m_motorTargetVelocity = motorTargetVelocity; }
	void setMotorTarget(const btQuaternion& qAinB, btScalar dt); // qAinB is rotation of body A wrt body B.
	void setMotorTarget(btScalar targetAngle, btScalar dt);


	void	setLimit(btScalar low,btScalar high,btScalar _softness = 0.9f, btScalar _biasFactor = 0.3f, btScalar _relaxationFactor = 1.0f)
	{
#ifdef	_BT_USE_CENTER_LIMIT_
		m_limit.set(low, high, _softness, _biasFactor, _relaxationFactor);
#else
		m_lowerLimit = btNormalizeAngle(low);
		m_upperLimit = btNormalizeAngle(high);
		m_limitSoftness =  _softness;
		m_biasFactor = _biasFactor;
		m_relaxationFactor = _relaxationFactor;
#endif
	}
	
	btScalar getLimitSoftness() const
	{
#ifdef	_BT_USE_CENTER_LIMIT_
		return m_limit.getSoftness();
#else
		return m_limitSoftness;
#endif
	}

	btScalar getLimitBiasFactor() const
	{
#ifdef	_BT_USE_CENTER_LIMIT_
		return m_limit.getBiasFactor();
#else
		return m_biasFactor;
#endif
	}

	btScalar getLimitRelaxationFactor() const
	{
#ifdef	_BT_USE_CENTER_LIMIT_
		return m_limit.getRelaxationFactor();
#else
		return m_relaxationFactor;
#endif
	}

	void	setAxis(btVector3& axisInA)
	{
		btVector3 rbAxisA1, rbAxisA2;
		btPlaneSpace1(axisInA, rbAxisA1, rbAxisA2);
		btVector3 pivotInA = m_rbAFrame.getOrigin();
//		m_rbAFrame.getOrigin() = pivotInA;
		m_rbAFrame.getBasis().setValue( rbAxisA1.getX(),rbAxisA2.getX(),axisInA.getX(),
										rbAxisA1.getY(),rbAxisA2.getY(),axisInA.getY(),
										rbAxisA1.getZ(),rbAxisA2.getZ(),axisInA.getZ() );

		btVector3 axisInB = m_rbA.getCenterOfMassTransform().getBasis() * axisInA;

		btQuaternion rotationArc = shortestArcQuat(axisInA,axisInB);
		btVector3 rbAxisB1 =  quatRotate(rotationArc,rbAxisA1);
		btVector3 rbAxisB2 = axisInB.cross(rbAxisB1);

		m_rbBFrame.getOrigin() = m_rbB.getCenterOfMassTransform().inverse()(m_rbA.getCenterOfMassTransform()(pivotInA));

		m_rbBFrame.getBasis().setValue( rbAxisB1.getX(),rbAxisB2.getX(),axisInB.getX(),
										rbAxisB1.getY(),rbAxisB2.getY(),axisInB.getY(),
										rbAxisB1.getZ(),rbAxisB2.getZ(),axisInB.getZ() );
		m_rbBFrame.getBasis() = m_rbB.getCenterOfMassTransform().getBasis().inverse() * m_rbBFrame.getBasis();

	}

    bool hasLimit() const {
#ifdef  _BT_USE_CENTER_LIMIT_
        return m_limit.getHalfRange() > 0;
#else
        return m_lowerLimit <= m_upperLimit;
#endif
    }

	btScalar	getLowerLimit() const
	{
#ifdef	_BT_USE_CENTER_LIMIT_
	return m_limit.getLow();
#else
	return m_lowerLimit;
#endif
	}

	btScalar	getUpperLimit() const
	{
#ifdef	_BT_USE_CENTER_LIMIT_
	return m_limit.getHigh();
#else		
	return m_upperLimit;
#endif
	}


	///The getHingeAngle gives the hinge angle in range [-PI,PI]
	btScalar getHingeAngle();

	btScalar getHingeAngle(const btTransform& transA,const btTransform& transB);

	void testLimit(const btTransform& transA,const btTransform& transB);


	const btTransform& getAFrame() const { return m_rbAFrame; };	
	const btTransform& getBFrame() const { return m_rbBFrame; };

	btTransform& getAFrame() { return m_rbAFrame; };	
	btTransform& getBFrame() { return m_rbBFrame; };

	inline int getSolveLimit()
	{
#ifdef	_BT_USE_CENTER_LIMIT_
	return m_limit.isLimit();
#else
	return m_solveLimit;
#endif
	}

	inline btScalar getLimitSign()
	{
#ifdef	_BT_USE_CENTER_LIMIT_
	return m_limit.getSign();
#else
		return m_limitSign;
#endif
	}

	inline bool getAngularOnly() 
	{ 
		return m_angularOnly; 
	}
	inline bool getEnableAngularMotor() 
	{ 
		return m_enableAngularMotor; 
	}
	inline btScalar getMotorTargetVelocity() 
	{ 
		return m_motorTargetVelocity; 
	}
	inline btScalar getMaxMotorImpulse() 
	{ 
		return m_maxMotorImpulse; 
	}
	// access for UseFrameOffset
	bool getUseFrameOffset() { return m_useOffsetForConstraintFrame; }
	void setUseFrameOffset(bool frameOffsetOnOff) { m_useOffsetForConstraintFrame = frameOffsetOnOff; }
	// access for UseReferenceFrameA
	bool getUseReferenceFrameA() const { return m_useReferenceFrameA; }
	void setUseReferenceFrameA(bool useReferenceFrameA) { m_useReferenceFrameA = useReferenceFrameA; }

	///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5). 
	///If no axis is provided, it uses the default axis for this constraint.
	virtual	void	setParam(int num, btScalar value, int axis = -1);
	///return the local value of parameter
	virtual	btScalar getParam(int num, int axis = -1) const;
	
	virtual	int getFlags() const
	{
  	    return m_flags;
	}

	virtual	int	calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual	const char*	serialize(void* dataBuffer, btSerializer* serializer) const;


};


//only for backward compatibility
#ifdef BT_BACKWARDS_COMPATIBLE_SERIALIZATION
///this structure is not used, except for loading pre-2.82 .bullet files
struct	btHingeConstraintDoubleData
{
	btTypedConstraintData	m_typeConstraintData;
	btTransformDoubleData m_rbAFrame; // constraint axii. Assumes z is hinge axis.
	btTransformDoubleData m_rbBFrame;
	int			m_useReferenceFrameA;
	int			m_angularOnly;
	int			m_enableAngularMotor;
	float	m_motorTargetVelocity;
	float	m_maxMotorImpulse;

	float	m_lowerLimit;
	float	m_upperLimit;
	float	m_limitSoftness;
	float	m_biasFactor;
	float	m_relaxationFactor;

};
#endif //BT_BACKWARDS_COMPATIBLE_SERIALIZATION

///The getAccumulatedHingeAngle returns the accumulated hinge angle, taking rotation across the -PI/PI boundary into account
ATTRIBUTE_ALIGNED16(class) btHingeAccumulatedAngleConstraint : public btHingeConstraint
{
protected:
	btScalar	m_accumulatedAngle;
public:

	BT_DECLARE_ALIGNED_ALLOCATOR();
	
	btHingeAccumulatedAngleConstraint(btRigidBody& rbA,btRigidBody& rbB, const btVector3& pivotInA,const btVector3& pivotInB, const btVector3& axisInA,const btVector3& axisInB, bool useReferenceFrameA = false)
	:btHingeConstraint(rbA,rbB,pivotInA,pivotInB, axisInA,axisInB, useReferenceFrameA )
	{
		m_accumulatedAngle=getHingeAngle();
	}

	btHingeAccumulatedAngleConstraint(btRigidBody& rbA,const btVector3& pivotInA,const btVector3& axisInA, bool useReferenceFrameA = false)
	:btHingeConstraint(rbA,pivotInA,axisInA, useReferenceFrameA)
	{
		m_accumulatedAngle=getHingeAngle();
	}
	
	btHingeAccumulatedAngleConstraint(btRigidBody& rbA,btRigidBody& rbB, const btTransform& rbAFrame, const btTransform& rbBFrame, bool useReferenceFrameA = false)
	:btHingeConstraint(rbA,rbB, rbAFrame, rbBFrame, useReferenceFrameA )
	{
		m_accumulatedAngle=getHingeAngle();
	}

	btHingeAccumulatedAngleConstraint(btRigidBody& rbA,const btTransform& rbAFrame, bool useReferenceFrameA = false)
	:btHingeConstraint(rbA,rbAFrame, useReferenceFrameA )
	{
		m_accumulatedAngle=getHingeAngle();
	}
	btScalar getAccumulatedHingeAngle();
	void	setAccumulatedHingeAngle(btScalar accAngle);
	virtual void getInfo1 (btConstraintInfo1* info);

};

struct	btHingeConstraintFloatData
{
	btTypedConstraintData	m_typeConstraintData;
	btTransformFloatData m_rbAFrame; // constraint axii. Assumes z is hinge axis.
	btTransformFloatData m_rbBFrame;
	int			m_useReferenceFrameA;
	int			m_angularOnly;
	
	int			m_enableAngularMotor;
	float	m_motorTargetVelocity;
	float	m_maxMotorImpulse;

	float	m_lowerLimit;
	float	m_upperLimit;
	float	m_limitSoftness;
	float	m_biasFactor;
	float	m_relaxationFactor;

};



///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct	btHingeConstraintDoubleData2
{
	btTypedConstraintDoubleData	m_typeConstraintData;
	btTransformDoubleData m_rbAFrame; // constraint axii. Assumes z is hinge axis.
	btTransformDoubleData m_rbBFrame;
	int			m_useReferenceFrameA;
	int			m_angularOnly;
	int			m_enableAngularMotor;
	double		m_motorTargetVelocity;
	double		m_maxMotorImpulse;

	double		m_lowerLimit;
	double		m_upperLimit;
	double		m_limitSoftness;
	double		m_biasFactor;
	double		m_relaxationFactor;
	char	m_padding1[4];

};




SIMD_FORCE_INLINE	int	btHingeConstraint::calculateSerializeBufferSize() const
{
	return sizeof(btHingeConstraintData);
}

	///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE	const char*	btHingeConstraint::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btHingeConstraintData* hingeData = (btHingeConstraintData*)dataBuffer;
	btTypedConstraint::serialize(&hingeData->m_typeConstraintData,serializer);

	m_rbAFrame.serialize(hingeData->m_rbAFrame);
	m_rbBFrame.serialize(hingeData->m_rbBFrame);

	hingeData->m_angularOnly = m_angularOnly;
	hingeData->m_enableAngularMotor = m_enableAngularMotor;
	hingeData->m_maxMotorImpulse = float(m_maxMotorImpulse);
	hingeData->m_motorTargetVelocity = float(m_motorTargetVelocity);
	hingeData->m_useReferenceFrameA = m_useReferenceFrameA;
#ifdef	_BT_USE_CENTER_LIMIT_
	hingeData->m_lowerLimit = float(m_limit.getLow());
	hingeData->m_upperLimit = float(m_limit.getHigh());
	hingeData->m_limitSoftness = float(m_limit.getSoftness());
	hingeData->m_biasFactor = float(m_limit.getBiasFactor());
	hingeData->m_relaxationFactor = float(m_limit.getRelaxationFactor());
#else
	hingeData->m_lowerLimit = float(m_lowerLimit);
	hingeData->m_upperLimit = float(m_upperLimit);
	hingeData->m_limitSoftness = float(m_limitSoftness);
	hingeData->m_biasFactor = float(m_biasFactor);
	hingeData->m_relaxationFactor = float(m_relaxationFactor);
#endif

	// Fill padding with zeros to appease msan.
#ifdef BT_USE_DOUBLE_PRECISION
	hingeData->m_padding1[0] = 0;
	hingeData->m_padding1[1] = 0;
	hingeData->m_padding1[2] = 0;
	hingeData->m_padding1[3] = 0;
#endif

	return btHingeConstraintDataName;
}

#endif //BT_HINGECONSTRAINT_H
