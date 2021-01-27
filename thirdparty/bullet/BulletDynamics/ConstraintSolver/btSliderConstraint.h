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

/*
Added by Roman Ponomarev (rponom@gmail.com)
April 04, 2008

TODO:
 - add clamping od accumulated impulse to improve stability
 - add conversion for ODE constraint solver
*/

#ifndef BT_SLIDER_CONSTRAINT_H
#define BT_SLIDER_CONSTRAINT_H

#include "LinearMath/btScalar.h"  //for BT_USE_DOUBLE_PRECISION

#ifdef BT_USE_DOUBLE_PRECISION
#define btSliderConstraintData2 btSliderConstraintDoubleData
#define btSliderConstraintDataName "btSliderConstraintDoubleData"
#else
#define btSliderConstraintData2 btSliderConstraintData
#define btSliderConstraintDataName "btSliderConstraintData"
#endif  //BT_USE_DOUBLE_PRECISION

#include "LinearMath/btVector3.h"
#include "btJacobianEntry.h"
#include "btTypedConstraint.h"

class btRigidBody;

#define SLIDER_CONSTRAINT_DEF_SOFTNESS (btScalar(1.0))
#define SLIDER_CONSTRAINT_DEF_DAMPING (btScalar(1.0))
#define SLIDER_CONSTRAINT_DEF_RESTITUTION (btScalar(0.7))
#define SLIDER_CONSTRAINT_DEF_CFM (btScalar(0.f))

enum btSliderFlags
{
	BT_SLIDER_FLAGS_CFM_DIRLIN = (1 << 0),
	BT_SLIDER_FLAGS_ERP_DIRLIN = (1 << 1),
	BT_SLIDER_FLAGS_CFM_DIRANG = (1 << 2),
	BT_SLIDER_FLAGS_ERP_DIRANG = (1 << 3),
	BT_SLIDER_FLAGS_CFM_ORTLIN = (1 << 4),
	BT_SLIDER_FLAGS_ERP_ORTLIN = (1 << 5),
	BT_SLIDER_FLAGS_CFM_ORTANG = (1 << 6),
	BT_SLIDER_FLAGS_ERP_ORTANG = (1 << 7),
	BT_SLIDER_FLAGS_CFM_LIMLIN = (1 << 8),
	BT_SLIDER_FLAGS_ERP_LIMLIN = (1 << 9),
	BT_SLIDER_FLAGS_CFM_LIMANG = (1 << 10),
	BT_SLIDER_FLAGS_ERP_LIMANG = (1 << 11)
};

ATTRIBUTE_ALIGNED16(class)
btSliderConstraint : public btTypedConstraint
{
protected:
	///for backwards compatibility during the transition to 'getInfo/getInfo2'
	bool m_useSolveConstraintObsolete;
	bool m_useOffsetForConstraintFrame;
	btTransform m_frameInA;
	btTransform m_frameInB;
	// use frameA fo define limits, if true
	bool m_useLinearReferenceFrameA;
	// linear limits
	btScalar m_lowerLinLimit;
	btScalar m_upperLinLimit;
	// angular limits
	btScalar m_lowerAngLimit;
	btScalar m_upperAngLimit;
	// softness, restitution and damping for different cases
	// DirLin - moving inside linear limits
	// LimLin - hitting linear limit
	// DirAng - moving inside angular limits
	// LimAng - hitting angular limit
	// OrthoLin, OrthoAng - against constraint axis
	btScalar m_softnessDirLin;
	btScalar m_restitutionDirLin;
	btScalar m_dampingDirLin;
	btScalar m_cfmDirLin;

	btScalar m_softnessDirAng;
	btScalar m_restitutionDirAng;
	btScalar m_dampingDirAng;
	btScalar m_cfmDirAng;

	btScalar m_softnessLimLin;
	btScalar m_restitutionLimLin;
	btScalar m_dampingLimLin;
	btScalar m_cfmLimLin;

	btScalar m_softnessLimAng;
	btScalar m_restitutionLimAng;
	btScalar m_dampingLimAng;
	btScalar m_cfmLimAng;

	btScalar m_softnessOrthoLin;
	btScalar m_restitutionOrthoLin;
	btScalar m_dampingOrthoLin;
	btScalar m_cfmOrthoLin;

	btScalar m_softnessOrthoAng;
	btScalar m_restitutionOrthoAng;
	btScalar m_dampingOrthoAng;
	btScalar m_cfmOrthoAng;

	// for interlal use
	bool m_solveLinLim;
	bool m_solveAngLim;

	int m_flags;

	btJacobianEntry m_jacLin[3];
	btScalar m_jacLinDiagABInv[3];

	btJacobianEntry m_jacAng[3];

	btScalar m_timeStep;
	btTransform m_calculatedTransformA;
	btTransform m_calculatedTransformB;

	btVector3 m_sliderAxis;
	btVector3 m_realPivotAInW;
	btVector3 m_realPivotBInW;
	btVector3 m_projPivotInW;
	btVector3 m_delta;
	btVector3 m_depth;
	btVector3 m_relPosA;
	btVector3 m_relPosB;

	btScalar m_linPos;
	btScalar m_angPos;

	btScalar m_angDepth;
	btScalar m_kAngle;

	bool m_poweredLinMotor;
	btScalar m_targetLinMotorVelocity;
	btScalar m_maxLinMotorForce;
	btScalar m_accumulatedLinMotorImpulse;

	bool m_poweredAngMotor;
	btScalar m_targetAngMotorVelocity;
	btScalar m_maxAngMotorForce;
	btScalar m_accumulatedAngMotorImpulse;

	//------------------------
	void initParams();

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	// constructors
	btSliderConstraint(btRigidBody & rbA, btRigidBody & rbB, const btTransform& frameInA, const btTransform& frameInB, bool useLinearReferenceFrameA);
	btSliderConstraint(btRigidBody & rbB, const btTransform& frameInB, bool useLinearReferenceFrameA);

	// overrides

	virtual void getInfo1(btConstraintInfo1 * info);

	void getInfo1NonVirtual(btConstraintInfo1 * info);

	virtual void getInfo2(btConstraintInfo2 * info);

	void getInfo2NonVirtual(btConstraintInfo2 * info, const btTransform& transA, const btTransform& transB, const btVector3& linVelA, const btVector3& linVelB, btScalar rbAinvMass, btScalar rbBinvMass);

	// access
	const btRigidBody& getRigidBodyA() const { return m_rbA; }
	const btRigidBody& getRigidBodyB() const { return m_rbB; }
	const btTransform& getCalculatedTransformA() const { return m_calculatedTransformA; }
	const btTransform& getCalculatedTransformB() const { return m_calculatedTransformB; }
	const btTransform& getFrameOffsetA() const { return m_frameInA; }
	const btTransform& getFrameOffsetB() const { return m_frameInB; }
	btTransform& getFrameOffsetA() { return m_frameInA; }
	btTransform& getFrameOffsetB() { return m_frameInB; }
	btScalar getLowerLinLimit() { return m_lowerLinLimit; }
	void setLowerLinLimit(btScalar lowerLimit) { m_lowerLinLimit = lowerLimit; }
	btScalar getUpperLinLimit() { return m_upperLinLimit; }
	void setUpperLinLimit(btScalar upperLimit) { m_upperLinLimit = upperLimit; }
	btScalar getLowerAngLimit() { return m_lowerAngLimit; }
	void setLowerAngLimit(btScalar lowerLimit) { m_lowerAngLimit = btNormalizeAngle(lowerLimit); }
	btScalar getUpperAngLimit() { return m_upperAngLimit; }
	void setUpperAngLimit(btScalar upperLimit) { m_upperAngLimit = btNormalizeAngle(upperLimit); }
	bool getUseLinearReferenceFrameA() { return m_useLinearReferenceFrameA; }
	btScalar getSoftnessDirLin() { return m_softnessDirLin; }
	btScalar getRestitutionDirLin() { return m_restitutionDirLin; }
	btScalar getDampingDirLin() { return m_dampingDirLin; }
	btScalar getSoftnessDirAng() { return m_softnessDirAng; }
	btScalar getRestitutionDirAng() { return m_restitutionDirAng; }
	btScalar getDampingDirAng() { return m_dampingDirAng; }
	btScalar getSoftnessLimLin() { return m_softnessLimLin; }
	btScalar getRestitutionLimLin() { return m_restitutionLimLin; }
	btScalar getDampingLimLin() { return m_dampingLimLin; }
	btScalar getSoftnessLimAng() { return m_softnessLimAng; }
	btScalar getRestitutionLimAng() { return m_restitutionLimAng; }
	btScalar getDampingLimAng() { return m_dampingLimAng; }
	btScalar getSoftnessOrthoLin() { return m_softnessOrthoLin; }
	btScalar getRestitutionOrthoLin() { return m_restitutionOrthoLin; }
	btScalar getDampingOrthoLin() { return m_dampingOrthoLin; }
	btScalar getSoftnessOrthoAng() { return m_softnessOrthoAng; }
	btScalar getRestitutionOrthoAng() { return m_restitutionOrthoAng; }
	btScalar getDampingOrthoAng() { return m_dampingOrthoAng; }
	void setSoftnessDirLin(btScalar softnessDirLin) { m_softnessDirLin = softnessDirLin; }
	void setRestitutionDirLin(btScalar restitutionDirLin) { m_restitutionDirLin = restitutionDirLin; }
	void setDampingDirLin(btScalar dampingDirLin) { m_dampingDirLin = dampingDirLin; }
	void setSoftnessDirAng(btScalar softnessDirAng) { m_softnessDirAng = softnessDirAng; }
	void setRestitutionDirAng(btScalar restitutionDirAng) { m_restitutionDirAng = restitutionDirAng; }
	void setDampingDirAng(btScalar dampingDirAng) { m_dampingDirAng = dampingDirAng; }
	void setSoftnessLimLin(btScalar softnessLimLin) { m_softnessLimLin = softnessLimLin; }
	void setRestitutionLimLin(btScalar restitutionLimLin) { m_restitutionLimLin = restitutionLimLin; }
	void setDampingLimLin(btScalar dampingLimLin) { m_dampingLimLin = dampingLimLin; }
	void setSoftnessLimAng(btScalar softnessLimAng) { m_softnessLimAng = softnessLimAng; }
	void setRestitutionLimAng(btScalar restitutionLimAng) { m_restitutionLimAng = restitutionLimAng; }
	void setDampingLimAng(btScalar dampingLimAng) { m_dampingLimAng = dampingLimAng; }
	void setSoftnessOrthoLin(btScalar softnessOrthoLin) { m_softnessOrthoLin = softnessOrthoLin; }
	void setRestitutionOrthoLin(btScalar restitutionOrthoLin) { m_restitutionOrthoLin = restitutionOrthoLin; }
	void setDampingOrthoLin(btScalar dampingOrthoLin) { m_dampingOrthoLin = dampingOrthoLin; }
	void setSoftnessOrthoAng(btScalar softnessOrthoAng) { m_softnessOrthoAng = softnessOrthoAng; }
	void setRestitutionOrthoAng(btScalar restitutionOrthoAng) { m_restitutionOrthoAng = restitutionOrthoAng; }
	void setDampingOrthoAng(btScalar dampingOrthoAng) { m_dampingOrthoAng = dampingOrthoAng; }
	void setPoweredLinMotor(bool onOff) { m_poweredLinMotor = onOff; }
	bool getPoweredLinMotor() { return m_poweredLinMotor; }
	void setTargetLinMotorVelocity(btScalar targetLinMotorVelocity) { m_targetLinMotorVelocity = targetLinMotorVelocity; }
	btScalar getTargetLinMotorVelocity() { return m_targetLinMotorVelocity; }
	void setMaxLinMotorForce(btScalar maxLinMotorForce) { m_maxLinMotorForce = maxLinMotorForce; }
	btScalar getMaxLinMotorForce() { return m_maxLinMotorForce; }
	void setPoweredAngMotor(bool onOff) { m_poweredAngMotor = onOff; }
	bool getPoweredAngMotor() { return m_poweredAngMotor; }
	void setTargetAngMotorVelocity(btScalar targetAngMotorVelocity) { m_targetAngMotorVelocity = targetAngMotorVelocity; }
	btScalar getTargetAngMotorVelocity() { return m_targetAngMotorVelocity; }
	void setMaxAngMotorForce(btScalar maxAngMotorForce) { m_maxAngMotorForce = maxAngMotorForce; }
	btScalar getMaxAngMotorForce() { return m_maxAngMotorForce; }

	btScalar getLinearPos() const { return m_linPos; }
	btScalar getAngularPos() const { return m_angPos; }

	// access for ODE solver
	bool getSolveLinLimit() { return m_solveLinLim; }
	btScalar getLinDepth() { return m_depth[0]; }
	bool getSolveAngLimit() { return m_solveAngLim; }
	btScalar getAngDepth() { return m_angDepth; }
	// shared code used by ODE solver
	void calculateTransforms(const btTransform& transA, const btTransform& transB);
	void testLinLimits();
	void testAngLimits();
	// access for PE Solver
	btVector3 getAncorInA();
	btVector3 getAncorInB();
	// access for UseFrameOffset
	bool getUseFrameOffset() { return m_useOffsetForConstraintFrame; }
	void setUseFrameOffset(bool frameOffsetOnOff) { m_useOffsetForConstraintFrame = frameOffsetOnOff; }

	void setFrames(const btTransform& frameA, const btTransform& frameB)
	{
		m_frameInA = frameA;
		m_frameInB = frameB;
		calculateTransforms(m_rbA.getCenterOfMassTransform(), m_rbB.getCenterOfMassTransform());
		buildJacobian();
	}

	///override the default global value of a parameter (such as ERP or CFM), optionally provide the axis (0..5).
	///If no axis is provided, it uses the default axis for this constraint.
	virtual void setParam(int num, btScalar value, int axis = -1);
	///return the local value of parameter
	virtual btScalar getParam(int num, int axis = -1) const;

	virtual int getFlags() const
	{
		return m_flags;
	}

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, btSerializer* serializer) const;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64

struct btSliderConstraintData
{
	btTypedConstraintData m_typeConstraintData;
	btTransformFloatData m_rbAFrame;  // constraint axii. Assumes z is hinge axis.
	btTransformFloatData m_rbBFrame;

	float m_linearUpperLimit;
	float m_linearLowerLimit;

	float m_angularUpperLimit;
	float m_angularLowerLimit;

	int m_useLinearReferenceFrameA;
	int m_useOffsetForConstraintFrame;
};

struct btSliderConstraintDoubleData
{
	btTypedConstraintDoubleData m_typeConstraintData;
	btTransformDoubleData m_rbAFrame;  // constraint axii. Assumes z is hinge axis.
	btTransformDoubleData m_rbBFrame;

	double m_linearUpperLimit;
	double m_linearLowerLimit;

	double m_angularUpperLimit;
	double m_angularLowerLimit;

	int m_useLinearReferenceFrameA;
	int m_useOffsetForConstraintFrame;
};

SIMD_FORCE_INLINE int btSliderConstraint::calculateSerializeBufferSize() const
{
	return sizeof(btSliderConstraintData2);
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
SIMD_FORCE_INLINE const char* btSliderConstraint::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btSliderConstraintData2* sliderData = (btSliderConstraintData2*)dataBuffer;
	btTypedConstraint::serialize(&sliderData->m_typeConstraintData, serializer);

	m_frameInA.serialize(sliderData->m_rbAFrame);
	m_frameInB.serialize(sliderData->m_rbBFrame);

	sliderData->m_linearUpperLimit = m_upperLinLimit;
	sliderData->m_linearLowerLimit = m_lowerLinLimit;

	sliderData->m_angularUpperLimit = m_upperAngLimit;
	sliderData->m_angularLowerLimit = m_lowerAngLimit;

	sliderData->m_useLinearReferenceFrameA = m_useLinearReferenceFrameA;
	sliderData->m_useOffsetForConstraintFrame = m_useOffsetForConstraintFrame;

	return btSliderConstraintDataName;
}

#endif  //BT_SLIDER_CONSTRAINT_H
