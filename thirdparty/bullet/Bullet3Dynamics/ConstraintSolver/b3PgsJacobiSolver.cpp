/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2012 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

//enable B3_SOLVER_DEBUG if you experience solver crashes
//#define B3_SOLVER_DEBUG
//#define COMPUTE_IMPULSE_DENOM 1
//It is not necessary (redundant) to refresh contact manifolds, this refresh has been moved to the collision algorithms.

//#define DISABLE_JOINTS

#include "b3PgsJacobiSolver.h"
#include "Bullet3Common/b3MinMax.h"
#include "b3TypedConstraint.h"
#include <new>
#include "Bullet3Common/b3StackAlloc.h"

//#include "b3SolverBody.h"
//#include "b3SolverConstraint.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include <string.h>  //for memset
//#include "../../dynamics/basic_demo/Stubs/AdlContact4.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Contact4.h"

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

static b3Transform getWorldTransform(b3RigidBodyData* rb)
{
	b3Transform newTrans;
	newTrans.setOrigin(rb->m_pos);
	newTrans.setRotation(rb->m_quat);
	return newTrans;
}

static const b3Matrix3x3& getInvInertiaTensorWorld(b3InertiaData* inertia)
{
	return inertia->m_invInertiaWorld;
}

static const b3Vector3& getLinearVelocity(b3RigidBodyData* rb)
{
	return rb->m_linVel;
}

static const b3Vector3& getAngularVelocity(b3RigidBodyData* rb)
{
	return rb->m_angVel;
}

static b3Vector3 getVelocityInLocalPoint(b3RigidBodyData* rb, const b3Vector3& rel_pos)
{
	//we also calculate lin/ang velocity for kinematic objects
	return getLinearVelocity(rb) + getAngularVelocity(rb).cross(rel_pos);
}

struct b3ContactPoint
{
	b3Vector3 m_positionWorldOnA;
	b3Vector3 m_positionWorldOnB;
	b3Vector3 m_normalWorldOnB;
	b3Scalar m_appliedImpulse;
	b3Scalar m_distance;
	b3Scalar m_combinedRestitution;

	///information related to friction
	b3Scalar m_combinedFriction;
	b3Vector3 m_lateralFrictionDir1;
	b3Vector3 m_lateralFrictionDir2;
	b3Scalar m_appliedImpulseLateral1;
	b3Scalar m_appliedImpulseLateral2;
	b3Scalar m_combinedRollingFriction;
	b3Scalar m_contactMotion1;
	b3Scalar m_contactMotion2;
	b3Scalar m_contactCFM1;
	b3Scalar m_contactCFM2;

	bool m_lateralFrictionInitialized;

	b3Vector3 getPositionWorldOnA()
	{
		return m_positionWorldOnA;
	}
	b3Vector3 getPositionWorldOnB()
	{
		return m_positionWorldOnB;
	}
	b3Scalar getDistance()
	{
		return m_distance;
	}
};

void getContactPoint(b3Contact4* contact, int contactIndex, b3ContactPoint& pointOut)
{
	pointOut.m_appliedImpulse = 0.f;
	pointOut.m_appliedImpulseLateral1 = 0.f;
	pointOut.m_appliedImpulseLateral2 = 0.f;
	pointOut.m_combinedFriction = contact->getFrictionCoeff();
	pointOut.m_combinedRestitution = contact->getRestituitionCoeff();
	pointOut.m_combinedRollingFriction = 0.f;
	pointOut.m_contactCFM1 = 0.f;
	pointOut.m_contactCFM2 = 0.f;
	pointOut.m_contactMotion1 = 0.f;
	pointOut.m_contactMotion2 = 0.f;
	pointOut.m_distance = contact->getPenetration(contactIndex);  //??0.01f
	b3Vector3 normalOnB = contact->m_worldNormalOnB;
	normalOnB.normalize();  //is this needed?

	b3Vector3 l1, l2;
	b3PlaneSpace1(normalOnB, l1, l2);

	pointOut.m_normalWorldOnB = normalOnB;
	//printf("normalOnB = %f,%f,%f\n",normalOnB.getX(),normalOnB.getY(),normalOnB.getZ());
	pointOut.m_lateralFrictionDir1 = l1;
	pointOut.m_lateralFrictionDir2 = l2;
	pointOut.m_lateralFrictionInitialized = true;

	b3Vector3 worldPosB = contact->m_worldPosB[contactIndex];
	pointOut.m_positionWorldOnB = worldPosB;
	pointOut.m_positionWorldOnA = worldPosB + normalOnB * pointOut.m_distance;
}

int getNumContacts(b3Contact4* contact)
{
	return contact->getNPoints();
}

b3PgsJacobiSolver::b3PgsJacobiSolver(bool usePgs)
	: m_usePgs(usePgs),
	  m_numSplitImpulseRecoveries(0),
	  m_btSeed2(0)
{
}

b3PgsJacobiSolver::~b3PgsJacobiSolver()
{
}

void b3PgsJacobiSolver::solveContacts(int numBodies, b3RigidBodyData* bodies, b3InertiaData* inertias, int numContacts, b3Contact4* contacts, int numConstraints, b3TypedConstraint** constraints)
{
	b3ContactSolverInfo infoGlobal;
	infoGlobal.m_splitImpulse = false;
	infoGlobal.m_timeStep = 1.f / 60.f;
	infoGlobal.m_numIterations = 4;  //4;
									 //	infoGlobal.m_solverMode|=B3_SOLVER_USE_2_FRICTION_DIRECTIONS|B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS|B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION;
	//infoGlobal.m_solverMode|=B3_SOLVER_USE_2_FRICTION_DIRECTIONS|B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS;
	infoGlobal.m_solverMode |= B3_SOLVER_USE_2_FRICTION_DIRECTIONS;

	//if (infoGlobal.m_solverMode & B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS)
	//if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS) && (infoGlobal.m_solverMode & B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION))

	solveGroup(bodies, inertias, numBodies, contacts, numContacts, constraints, numConstraints, infoGlobal);

	if (!numContacts)
		return;
}

/// b3PgsJacobiSolver Sequentially applies impulses
b3Scalar b3PgsJacobiSolver::solveGroup(b3RigidBodyData* bodies,
									   b3InertiaData* inertias,
									   int numBodies,
									   b3Contact4* manifoldPtr,
									   int numManifolds,
									   b3TypedConstraint** constraints,
									   int numConstraints,
									   const b3ContactSolverInfo& infoGlobal)
{
	B3_PROFILE("solveGroup");
	//you need to provide at least some bodies

	solveGroupCacheFriendlySetup(bodies, inertias, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal);

	solveGroupCacheFriendlyIterations(constraints, numConstraints, infoGlobal);

	solveGroupCacheFriendlyFinish(bodies, inertias, numBodies, infoGlobal);

	return 0.f;
}

#ifdef USE_SIMD
#include <emmintrin.h>
#define b3VecSplat(x, e) _mm_shuffle_ps(x, x, _MM_SHUFFLE(e, e, e, e))
static inline __m128 b3SimdDot3(__m128 vec0, __m128 vec1)
{
	__m128 result = _mm_mul_ps(vec0, vec1);
	return _mm_add_ps(b3VecSplat(result, 0), _mm_add_ps(b3VecSplat(result, 1), b3VecSplat(result, 2)));
}
#endif  //USE_SIMD

// Project Gauss Seidel or the equivalent Sequential Impulse
void b3PgsJacobiSolver::resolveSingleConstraintRowGenericSIMD(b3SolverBody& body1, b3SolverBody& body2, const b3SolverConstraint& c)
{
#ifdef USE_SIMD
	__m128 cpAppliedImp = _mm_set1_ps(c.m_appliedImpulse);
	__m128 lowerLimit1 = _mm_set1_ps(c.m_lowerLimit);
	__m128 upperLimit1 = _mm_set1_ps(c.m_upperLimit);
	__m128 deltaImpulse = _mm_sub_ps(_mm_set1_ps(c.m_rhs), _mm_mul_ps(_mm_set1_ps(c.m_appliedImpulse), _mm_set1_ps(c.m_cfm)));
	__m128 deltaVel1Dotn = _mm_add_ps(b3SimdDot3(c.m_contactNormal.mVec128, body1.internalGetDeltaLinearVelocity().mVec128), b3SimdDot3(c.m_relpos1CrossNormal.mVec128, body1.internalGetDeltaAngularVelocity().mVec128));
	__m128 deltaVel2Dotn = _mm_sub_ps(b3SimdDot3(c.m_relpos2CrossNormal.mVec128, body2.internalGetDeltaAngularVelocity().mVec128), b3SimdDot3((c.m_contactNormal).mVec128, body2.internalGetDeltaLinearVelocity().mVec128));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel1Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel2Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	b3SimdScalar sum = _mm_add_ps(cpAppliedImp, deltaImpulse);
	b3SimdScalar resultLowerLess, resultUpperLess;
	resultLowerLess = _mm_cmplt_ps(sum, lowerLimit1);
	resultUpperLess = _mm_cmplt_ps(sum, upperLimit1);
	__m128 lowMinApplied = _mm_sub_ps(lowerLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowMinApplied), _mm_andnot_ps(resultLowerLess, deltaImpulse));
	c.m_appliedImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowerLimit1), _mm_andnot_ps(resultLowerLess, sum));
	__m128 upperMinApplied = _mm_sub_ps(upperLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultUpperLess, deltaImpulse), _mm_andnot_ps(resultUpperLess, upperMinApplied));
	c.m_appliedImpulse = _mm_or_ps(_mm_and_ps(resultUpperLess, c.m_appliedImpulse), _mm_andnot_ps(resultUpperLess, upperLimit1));
	__m128 linearComponentA = _mm_mul_ps(c.m_contactNormal.mVec128, body1.internalGetInvMass().mVec128);
	__m128 linearComponentB = _mm_mul_ps((c.m_contactNormal).mVec128, body2.internalGetInvMass().mVec128);
	__m128 impulseMagnitude = deltaImpulse;
	body1.internalGetDeltaLinearVelocity().mVec128 = _mm_add_ps(body1.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentA, impulseMagnitude));
	body1.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(body1.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentA.mVec128, impulseMagnitude));
	body2.internalGetDeltaLinearVelocity().mVec128 = _mm_sub_ps(body2.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentB, impulseMagnitude));
	body2.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(body2.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentB.mVec128, impulseMagnitude));
#else
	resolveSingleConstraintRowGeneric(body1, body2, c);
#endif
}

// Project Gauss Seidel or the equivalent Sequential Impulse
void b3PgsJacobiSolver::resolveSingleConstraintRowGeneric(b3SolverBody& body1, b3SolverBody& body2, const b3SolverConstraint& c)
{
	b3Scalar deltaImpulse = c.m_rhs - b3Scalar(c.m_appliedImpulse) * c.m_cfm;
	const b3Scalar deltaVel1Dotn = c.m_contactNormal.dot(body1.internalGetDeltaLinearVelocity()) + c.m_relpos1CrossNormal.dot(body1.internalGetDeltaAngularVelocity());
	const b3Scalar deltaVel2Dotn = -c.m_contactNormal.dot(body2.internalGetDeltaLinearVelocity()) + c.m_relpos2CrossNormal.dot(body2.internalGetDeltaAngularVelocity());

	//	const b3Scalar delta_rel_vel	=	deltaVel1Dotn-deltaVel2Dotn;
	deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
	deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;

	const b3Scalar sum = b3Scalar(c.m_appliedImpulse) + deltaImpulse;
	if (sum < c.m_lowerLimit)
	{
		deltaImpulse = c.m_lowerLimit - c.m_appliedImpulse;
		c.m_appliedImpulse = c.m_lowerLimit;
	}
	else if (sum > c.m_upperLimit)
	{
		deltaImpulse = c.m_upperLimit - c.m_appliedImpulse;
		c.m_appliedImpulse = c.m_upperLimit;
	}
	else
	{
		c.m_appliedImpulse = sum;
	}

	body1.internalApplyImpulse(c.m_contactNormal * body1.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
	body2.internalApplyImpulse(-c.m_contactNormal * body2.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
}

void b3PgsJacobiSolver::resolveSingleConstraintRowLowerLimitSIMD(b3SolverBody& body1, b3SolverBody& body2, const b3SolverConstraint& c)
{
#ifdef USE_SIMD
	__m128 cpAppliedImp = _mm_set1_ps(c.m_appliedImpulse);
	__m128 lowerLimit1 = _mm_set1_ps(c.m_lowerLimit);
	__m128 upperLimit1 = _mm_set1_ps(c.m_upperLimit);
	__m128 deltaImpulse = _mm_sub_ps(_mm_set1_ps(c.m_rhs), _mm_mul_ps(_mm_set1_ps(c.m_appliedImpulse), _mm_set1_ps(c.m_cfm)));
	__m128 deltaVel1Dotn = _mm_add_ps(b3SimdDot3(c.m_contactNormal.mVec128, body1.internalGetDeltaLinearVelocity().mVec128), b3SimdDot3(c.m_relpos1CrossNormal.mVec128, body1.internalGetDeltaAngularVelocity().mVec128));
	__m128 deltaVel2Dotn = _mm_sub_ps(b3SimdDot3(c.m_relpos2CrossNormal.mVec128, body2.internalGetDeltaAngularVelocity().mVec128), b3SimdDot3((c.m_contactNormal).mVec128, body2.internalGetDeltaLinearVelocity().mVec128));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel1Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel2Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	b3SimdScalar sum = _mm_add_ps(cpAppliedImp, deltaImpulse);
	b3SimdScalar resultLowerLess, resultUpperLess;
	resultLowerLess = _mm_cmplt_ps(sum, lowerLimit1);
	resultUpperLess = _mm_cmplt_ps(sum, upperLimit1);
	__m128 lowMinApplied = _mm_sub_ps(lowerLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowMinApplied), _mm_andnot_ps(resultLowerLess, deltaImpulse));
	c.m_appliedImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowerLimit1), _mm_andnot_ps(resultLowerLess, sum));
	__m128 linearComponentA = _mm_mul_ps(c.m_contactNormal.mVec128, body1.internalGetInvMass().mVec128);
	__m128 linearComponentB = _mm_mul_ps((c.m_contactNormal).mVec128, body2.internalGetInvMass().mVec128);
	__m128 impulseMagnitude = deltaImpulse;
	body1.internalGetDeltaLinearVelocity().mVec128 = _mm_add_ps(body1.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentA, impulseMagnitude));
	body1.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(body1.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentA.mVec128, impulseMagnitude));
	body2.internalGetDeltaLinearVelocity().mVec128 = _mm_sub_ps(body2.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentB, impulseMagnitude));
	body2.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(body2.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentB.mVec128, impulseMagnitude));
#else
	resolveSingleConstraintRowLowerLimit(body1, body2, c);
#endif
}

// Project Gauss Seidel or the equivalent Sequential Impulse
void b3PgsJacobiSolver::resolveSingleConstraintRowLowerLimit(b3SolverBody& body1, b3SolverBody& body2, const b3SolverConstraint& c)
{
	b3Scalar deltaImpulse = c.m_rhs - b3Scalar(c.m_appliedImpulse) * c.m_cfm;
	const b3Scalar deltaVel1Dotn = c.m_contactNormal.dot(body1.internalGetDeltaLinearVelocity()) + c.m_relpos1CrossNormal.dot(body1.internalGetDeltaAngularVelocity());
	const b3Scalar deltaVel2Dotn = -c.m_contactNormal.dot(body2.internalGetDeltaLinearVelocity()) + c.m_relpos2CrossNormal.dot(body2.internalGetDeltaAngularVelocity());

	deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
	deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;
	const b3Scalar sum = b3Scalar(c.m_appliedImpulse) + deltaImpulse;
	if (sum < c.m_lowerLimit)
	{
		deltaImpulse = c.m_lowerLimit - c.m_appliedImpulse;
		c.m_appliedImpulse = c.m_lowerLimit;
	}
	else
	{
		c.m_appliedImpulse = sum;
	}
	body1.internalApplyImpulse(c.m_contactNormal * body1.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
	body2.internalApplyImpulse(-c.m_contactNormal * body2.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
}

void b3PgsJacobiSolver::resolveSplitPenetrationImpulseCacheFriendly(
	b3SolverBody& body1,
	b3SolverBody& body2,
	const b3SolverConstraint& c)
{
	if (c.m_rhsPenetration)
	{
		m_numSplitImpulseRecoveries++;
		b3Scalar deltaImpulse = c.m_rhsPenetration - b3Scalar(c.m_appliedPushImpulse) * c.m_cfm;
		const b3Scalar deltaVel1Dotn = c.m_contactNormal.dot(body1.internalGetPushVelocity()) + c.m_relpos1CrossNormal.dot(body1.internalGetTurnVelocity());
		const b3Scalar deltaVel2Dotn = -c.m_contactNormal.dot(body2.internalGetPushVelocity()) + c.m_relpos2CrossNormal.dot(body2.internalGetTurnVelocity());

		deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
		deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;
		const b3Scalar sum = b3Scalar(c.m_appliedPushImpulse) + deltaImpulse;
		if (sum < c.m_lowerLimit)
		{
			deltaImpulse = c.m_lowerLimit - c.m_appliedPushImpulse;
			c.m_appliedPushImpulse = c.m_lowerLimit;
		}
		else
		{
			c.m_appliedPushImpulse = sum;
		}
		body1.internalApplyPushImpulse(c.m_contactNormal * body1.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
		body2.internalApplyPushImpulse(-c.m_contactNormal * body2.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
	}
}

void b3PgsJacobiSolver::resolveSplitPenetrationSIMD(b3SolverBody& body1, b3SolverBody& body2, const b3SolverConstraint& c)
{
#ifdef USE_SIMD
	if (!c.m_rhsPenetration)
		return;

	m_numSplitImpulseRecoveries++;

	__m128 cpAppliedImp = _mm_set1_ps(c.m_appliedPushImpulse);
	__m128 lowerLimit1 = _mm_set1_ps(c.m_lowerLimit);
	__m128 upperLimit1 = _mm_set1_ps(c.m_upperLimit);
	__m128 deltaImpulse = _mm_sub_ps(_mm_set1_ps(c.m_rhsPenetration), _mm_mul_ps(_mm_set1_ps(c.m_appliedPushImpulse), _mm_set1_ps(c.m_cfm)));
	__m128 deltaVel1Dotn = _mm_add_ps(b3SimdDot3(c.m_contactNormal.mVec128, body1.internalGetPushVelocity().mVec128), b3SimdDot3(c.m_relpos1CrossNormal.mVec128, body1.internalGetTurnVelocity().mVec128));
	__m128 deltaVel2Dotn = _mm_sub_ps(b3SimdDot3(c.m_relpos2CrossNormal.mVec128, body2.internalGetTurnVelocity().mVec128), b3SimdDot3((c.m_contactNormal).mVec128, body2.internalGetPushVelocity().mVec128));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel1Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel2Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	b3SimdScalar sum = _mm_add_ps(cpAppliedImp, deltaImpulse);
	b3SimdScalar resultLowerLess, resultUpperLess;
	resultLowerLess = _mm_cmplt_ps(sum, lowerLimit1);
	resultUpperLess = _mm_cmplt_ps(sum, upperLimit1);
	__m128 lowMinApplied = _mm_sub_ps(lowerLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowMinApplied), _mm_andnot_ps(resultLowerLess, deltaImpulse));
	c.m_appliedPushImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowerLimit1), _mm_andnot_ps(resultLowerLess, sum));
	__m128 linearComponentA = _mm_mul_ps(c.m_contactNormal.mVec128, body1.internalGetInvMass().mVec128);
	__m128 linearComponentB = _mm_mul_ps((c.m_contactNormal).mVec128, body2.internalGetInvMass().mVec128);
	__m128 impulseMagnitude = deltaImpulse;
	body1.internalGetPushVelocity().mVec128 = _mm_add_ps(body1.internalGetPushVelocity().mVec128, _mm_mul_ps(linearComponentA, impulseMagnitude));
	body1.internalGetTurnVelocity().mVec128 = _mm_add_ps(body1.internalGetTurnVelocity().mVec128, _mm_mul_ps(c.m_angularComponentA.mVec128, impulseMagnitude));
	body2.internalGetPushVelocity().mVec128 = _mm_sub_ps(body2.internalGetPushVelocity().mVec128, _mm_mul_ps(linearComponentB, impulseMagnitude));
	body2.internalGetTurnVelocity().mVec128 = _mm_add_ps(body2.internalGetTurnVelocity().mVec128, _mm_mul_ps(c.m_angularComponentB.mVec128, impulseMagnitude));
#else
	resolveSplitPenetrationImpulseCacheFriendly(body1, body2, c);
#endif
}

unsigned long b3PgsJacobiSolver::b3Rand2()
{
	m_btSeed2 = (1664525L * m_btSeed2 + 1013904223L) & 0xffffffff;
	return m_btSeed2;
}

//See ODE: adam's all-int straightforward(?) dRandInt (0..n-1)
int b3PgsJacobiSolver::b3RandInt2(int n)
{
	// seems good; xor-fold and modulus
	const unsigned long un = static_cast<unsigned long>(n);
	unsigned long r = b3Rand2();

	// note: probably more aggressive than it needs to be -- might be
	//       able to get away without one or two of the innermost branches.
	if (un <= 0x00010000UL)
	{
		r ^= (r >> 16);
		if (un <= 0x00000100UL)
		{
			r ^= (r >> 8);
			if (un <= 0x00000010UL)
			{
				r ^= (r >> 4);
				if (un <= 0x00000004UL)
				{
					r ^= (r >> 2);
					if (un <= 0x00000002UL)
					{
						r ^= (r >> 1);
					}
				}
			}
		}
	}

	return (int)(r % un);
}

void b3PgsJacobiSolver::initSolverBody(int bodyIndex, b3SolverBody* solverBody, b3RigidBodyData* rb)
{
	solverBody->m_deltaLinearVelocity.setValue(0.f, 0.f, 0.f);
	solverBody->m_deltaAngularVelocity.setValue(0.f, 0.f, 0.f);
	solverBody->internalGetPushVelocity().setValue(0.f, 0.f, 0.f);
	solverBody->internalGetTurnVelocity().setValue(0.f, 0.f, 0.f);

	if (rb)
	{
		solverBody->m_worldTransform = getWorldTransform(rb);
		solverBody->internalSetInvMass(b3MakeVector3(rb->m_invMass, rb->m_invMass, rb->m_invMass));
		solverBody->m_originalBodyIndex = bodyIndex;
		solverBody->m_angularFactor = b3MakeVector3(1, 1, 1);
		solverBody->m_linearFactor = b3MakeVector3(1, 1, 1);
		solverBody->m_linearVelocity = getLinearVelocity(rb);
		solverBody->m_angularVelocity = getAngularVelocity(rb);
	}
	else
	{
		solverBody->m_worldTransform.setIdentity();
		solverBody->internalSetInvMass(b3MakeVector3(0, 0, 0));
		solverBody->m_originalBodyIndex = bodyIndex;
		solverBody->m_angularFactor.setValue(1, 1, 1);
		solverBody->m_linearFactor.setValue(1, 1, 1);
		solverBody->m_linearVelocity.setValue(0, 0, 0);
		solverBody->m_angularVelocity.setValue(0, 0, 0);
	}
}

b3Scalar b3PgsJacobiSolver::restitutionCurve(b3Scalar rel_vel, b3Scalar restitution)
{
	b3Scalar rest = restitution * -rel_vel;
	return rest;
}

void b3PgsJacobiSolver::setupFrictionConstraint(b3RigidBodyData* bodies, b3InertiaData* inertias, b3SolverConstraint& solverConstraint, const b3Vector3& normalAxis, int solverBodyIdA, int solverBodyIdB, b3ContactPoint& cp, const b3Vector3& rel_pos1, const b3Vector3& rel_pos2, b3RigidBodyData* colObj0, b3RigidBodyData* colObj1, b3Scalar relaxation, b3Scalar desiredVelocity, b3Scalar cfmSlip)
{
	solverConstraint.m_contactNormal = normalAxis;
	b3SolverBody& solverBodyA = m_tmpSolverBodyPool[solverBodyIdA];
	b3SolverBody& solverBodyB = m_tmpSolverBodyPool[solverBodyIdB];

	b3RigidBodyData* body0 = &bodies[solverBodyA.m_originalBodyIndex];
	b3RigidBodyData* body1 = &bodies[solverBodyB.m_originalBodyIndex];

	solverConstraint.m_solverBodyIdA = solverBodyIdA;
	solverConstraint.m_solverBodyIdB = solverBodyIdB;

	solverConstraint.m_friction = cp.m_combinedFriction;
	solverConstraint.m_originalContactPoint = 0;

	solverConstraint.m_appliedImpulse = 0.f;
	solverConstraint.m_appliedPushImpulse = 0.f;

	{
		b3Vector3 ftorqueAxis1 = rel_pos1.cross(solverConstraint.m_contactNormal);
		solverConstraint.m_relpos1CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentA = body0 ? getInvInertiaTensorWorld(&inertias[solverBodyA.m_originalBodyIndex]) * ftorqueAxis1 : b3MakeVector3(0, 0, 0);
	}
	{
		b3Vector3 ftorqueAxis1 = rel_pos2.cross(-solverConstraint.m_contactNormal);
		solverConstraint.m_relpos2CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentB = body1 ? getInvInertiaTensorWorld(&inertias[solverBodyB.m_originalBodyIndex]) * ftorqueAxis1 : b3MakeVector3(0, 0, 0);
	}

	b3Scalar scaledDenom;

	{
		b3Vector3 vec;
		b3Scalar denom0 = 0.f;
		b3Scalar denom1 = 0.f;
		if (body0)
		{
			vec = (solverConstraint.m_angularComponentA).cross(rel_pos1);
			denom0 = body0->m_invMass + normalAxis.dot(vec);
		}
		if (body1)
		{
			vec = (-solverConstraint.m_angularComponentB).cross(rel_pos2);
			denom1 = body1->m_invMass + normalAxis.dot(vec);
		}

		b3Scalar denom;
		if (m_usePgs)
		{
			scaledDenom = denom = relaxation / (denom0 + denom1);
		}
		else
		{
			denom = relaxation / (denom0 + denom1);
			b3Scalar countA = body0->m_invMass ? b3Scalar(m_bodyCount[solverBodyA.m_originalBodyIndex]) : 1.f;
			b3Scalar countB = body1->m_invMass ? b3Scalar(m_bodyCount[solverBodyB.m_originalBodyIndex]) : 1.f;

			scaledDenom = relaxation / (denom0 * countA + denom1 * countB);
		}

		solverConstraint.m_jacDiagABInv = denom;
	}

	{
		b3Scalar rel_vel;
		b3Scalar vel1Dotn = solverConstraint.m_contactNormal.dot(body0 ? solverBodyA.m_linearVelocity : b3MakeVector3(0, 0, 0)) + solverConstraint.m_relpos1CrossNormal.dot(body0 ? solverBodyA.m_angularVelocity : b3MakeVector3(0, 0, 0));
		b3Scalar vel2Dotn = -solverConstraint.m_contactNormal.dot(body1 ? solverBodyB.m_linearVelocity : b3MakeVector3(0, 0, 0)) + solverConstraint.m_relpos2CrossNormal.dot(body1 ? solverBodyB.m_angularVelocity : b3MakeVector3(0, 0, 0));

		rel_vel = vel1Dotn + vel2Dotn;

		//		b3Scalar positionalError = 0.f;

		b3SimdScalar velocityError = desiredVelocity - rel_vel;
		b3SimdScalar velocityImpulse = velocityError * b3SimdScalar(scaledDenom);  //solverConstraint.m_jacDiagABInv);
		solverConstraint.m_rhs = velocityImpulse;
		solverConstraint.m_cfm = cfmSlip;
		solverConstraint.m_lowerLimit = 0;
		solverConstraint.m_upperLimit = 1e10f;
	}
}

b3SolverConstraint& b3PgsJacobiSolver::addFrictionConstraint(b3RigidBodyData* bodies, b3InertiaData* inertias, const b3Vector3& normalAxis, int solverBodyIdA, int solverBodyIdB, int frictionIndex, b3ContactPoint& cp, const b3Vector3& rel_pos1, const b3Vector3& rel_pos2, b3RigidBodyData* colObj0, b3RigidBodyData* colObj1, b3Scalar relaxation, b3Scalar desiredVelocity, b3Scalar cfmSlip)
{
	b3SolverConstraint& solverConstraint = m_tmpSolverContactFrictionConstraintPool.expandNonInitializing();
	solverConstraint.m_frictionIndex = frictionIndex;
	setupFrictionConstraint(bodies, inertias, solverConstraint, normalAxis, solverBodyIdA, solverBodyIdB, cp, rel_pos1, rel_pos2,
							colObj0, colObj1, relaxation, desiredVelocity, cfmSlip);
	return solverConstraint;
}

void b3PgsJacobiSolver::setupRollingFrictionConstraint(b3RigidBodyData* bodies, b3InertiaData* inertias, b3SolverConstraint& solverConstraint, const b3Vector3& normalAxis1, int solverBodyIdA, int solverBodyIdB,
													   b3ContactPoint& cp, const b3Vector3& rel_pos1, const b3Vector3& rel_pos2,
													   b3RigidBodyData* colObj0, b3RigidBodyData* colObj1, b3Scalar relaxation,
													   b3Scalar desiredVelocity, b3Scalar cfmSlip)

{
	b3Vector3 normalAxis = b3MakeVector3(0, 0, 0);

	solverConstraint.m_contactNormal = normalAxis;
	b3SolverBody& solverBodyA = m_tmpSolverBodyPool[solverBodyIdA];
	b3SolverBody& solverBodyB = m_tmpSolverBodyPool[solverBodyIdB];

	b3RigidBodyData* body0 = &bodies[m_tmpSolverBodyPool[solverBodyIdA].m_originalBodyIndex];
	b3RigidBodyData* body1 = &bodies[m_tmpSolverBodyPool[solverBodyIdB].m_originalBodyIndex];

	solverConstraint.m_solverBodyIdA = solverBodyIdA;
	solverConstraint.m_solverBodyIdB = solverBodyIdB;

	solverConstraint.m_friction = cp.m_combinedRollingFriction;
	solverConstraint.m_originalContactPoint = 0;

	solverConstraint.m_appliedImpulse = 0.f;
	solverConstraint.m_appliedPushImpulse = 0.f;

	{
		b3Vector3 ftorqueAxis1 = -normalAxis1;
		solverConstraint.m_relpos1CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentA = body0 ? getInvInertiaTensorWorld(&inertias[solverBodyA.m_originalBodyIndex]) * ftorqueAxis1 : b3MakeVector3(0, 0, 0);
	}
	{
		b3Vector3 ftorqueAxis1 = normalAxis1;
		solverConstraint.m_relpos2CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentB = body1 ? getInvInertiaTensorWorld(&inertias[solverBodyB.m_originalBodyIndex]) * ftorqueAxis1 : b3MakeVector3(0, 0, 0);
	}

	{
		b3Vector3 iMJaA = body0 ? getInvInertiaTensorWorld(&inertias[solverBodyA.m_originalBodyIndex]) * solverConstraint.m_relpos1CrossNormal : b3MakeVector3(0, 0, 0);
		b3Vector3 iMJaB = body1 ? getInvInertiaTensorWorld(&inertias[solverBodyB.m_originalBodyIndex]) * solverConstraint.m_relpos2CrossNormal : b3MakeVector3(0, 0, 0);
		b3Scalar sum = 0;
		sum += iMJaA.dot(solverConstraint.m_relpos1CrossNormal);
		sum += iMJaB.dot(solverConstraint.m_relpos2CrossNormal);
		solverConstraint.m_jacDiagABInv = b3Scalar(1.) / sum;
	}

	{
		b3Scalar rel_vel;
		b3Scalar vel1Dotn = solverConstraint.m_contactNormal.dot(body0 ? solverBodyA.m_linearVelocity : b3MakeVector3(0, 0, 0)) + solverConstraint.m_relpos1CrossNormal.dot(body0 ? solverBodyA.m_angularVelocity : b3MakeVector3(0, 0, 0));
		b3Scalar vel2Dotn = -solverConstraint.m_contactNormal.dot(body1 ? solverBodyB.m_linearVelocity : b3MakeVector3(0, 0, 0)) + solverConstraint.m_relpos2CrossNormal.dot(body1 ? solverBodyB.m_angularVelocity : b3MakeVector3(0, 0, 0));

		rel_vel = vel1Dotn + vel2Dotn;

		//		b3Scalar positionalError = 0.f;

		b3SimdScalar velocityError = desiredVelocity - rel_vel;
		b3SimdScalar velocityImpulse = velocityError * b3SimdScalar(solverConstraint.m_jacDiagABInv);
		solverConstraint.m_rhs = velocityImpulse;
		solverConstraint.m_cfm = cfmSlip;
		solverConstraint.m_lowerLimit = 0;
		solverConstraint.m_upperLimit = 1e10f;
	}
}

b3SolverConstraint& b3PgsJacobiSolver::addRollingFrictionConstraint(b3RigidBodyData* bodies, b3InertiaData* inertias, const b3Vector3& normalAxis, int solverBodyIdA, int solverBodyIdB, int frictionIndex, b3ContactPoint& cp, const b3Vector3& rel_pos1, const b3Vector3& rel_pos2, b3RigidBodyData* colObj0, b3RigidBodyData* colObj1, b3Scalar relaxation, b3Scalar desiredVelocity, b3Scalar cfmSlip)
{
	b3SolverConstraint& solverConstraint = m_tmpSolverContactRollingFrictionConstraintPool.expandNonInitializing();
	solverConstraint.m_frictionIndex = frictionIndex;
	setupRollingFrictionConstraint(bodies, inertias, solverConstraint, normalAxis, solverBodyIdA, solverBodyIdB, cp, rel_pos1, rel_pos2,
								   colObj0, colObj1, relaxation, desiredVelocity, cfmSlip);
	return solverConstraint;
}

int b3PgsJacobiSolver::getOrInitSolverBody(int bodyIndex, b3RigidBodyData* bodies, b3InertiaData* inertias)
{
	//b3Assert(bodyIndex< m_tmpSolverBodyPool.size());

	b3RigidBodyData& body = bodies[bodyIndex];
	int curIndex = -1;
	if (m_usePgs || body.m_invMass == 0.f)
	{
		if (m_bodyCount[bodyIndex] < 0)
		{
			curIndex = m_tmpSolverBodyPool.size();
			b3SolverBody& solverBody = m_tmpSolverBodyPool.expand();
			initSolverBody(bodyIndex, &solverBody, &body);
			solverBody.m_originalBodyIndex = bodyIndex;
			m_bodyCount[bodyIndex] = curIndex;
		}
		else
		{
			curIndex = m_bodyCount[bodyIndex];
		}
	}
	else
	{
		b3Assert(m_bodyCount[bodyIndex] > 0);
		m_bodyCountCheck[bodyIndex]++;
		curIndex = m_tmpSolverBodyPool.size();
		b3SolverBody& solverBody = m_tmpSolverBodyPool.expand();
		initSolverBody(bodyIndex, &solverBody, &body);
		solverBody.m_originalBodyIndex = bodyIndex;
	}

	b3Assert(curIndex >= 0);
	return curIndex;
}
#include <stdio.h>

void b3PgsJacobiSolver::setupContactConstraint(b3RigidBodyData* bodies, b3InertiaData* inertias, b3SolverConstraint& solverConstraint,
											   int solverBodyIdA, int solverBodyIdB,
											   b3ContactPoint& cp, const b3ContactSolverInfo& infoGlobal,
											   b3Vector3& vel, b3Scalar& rel_vel, b3Scalar& relaxation,
											   b3Vector3& rel_pos1, b3Vector3& rel_pos2)
{
	const b3Vector3& pos1 = cp.getPositionWorldOnA();
	const b3Vector3& pos2 = cp.getPositionWorldOnB();

	b3SolverBody* bodyA = &m_tmpSolverBodyPool[solverBodyIdA];
	b3SolverBody* bodyB = &m_tmpSolverBodyPool[solverBodyIdB];

	b3RigidBodyData* rb0 = &bodies[bodyA->m_originalBodyIndex];
	b3RigidBodyData* rb1 = &bodies[bodyB->m_originalBodyIndex];

	//			b3Vector3 rel_pos1 = pos1 - colObj0->getWorldTransform().getOrigin();
	//			b3Vector3 rel_pos2 = pos2 - colObj1->getWorldTransform().getOrigin();
	rel_pos1 = pos1 - bodyA->getWorldTransform().getOrigin();
	rel_pos2 = pos2 - bodyB->getWorldTransform().getOrigin();

	relaxation = 1.f;

	b3Vector3 torqueAxis0 = rel_pos1.cross(cp.m_normalWorldOnB);
	solverConstraint.m_angularComponentA = rb0 ? getInvInertiaTensorWorld(&inertias[bodyA->m_originalBodyIndex]) * torqueAxis0 : b3MakeVector3(0, 0, 0);
	b3Vector3 torqueAxis1 = rel_pos2.cross(cp.m_normalWorldOnB);
	solverConstraint.m_angularComponentB = rb1 ? getInvInertiaTensorWorld(&inertias[bodyB->m_originalBodyIndex]) * -torqueAxis1 : b3MakeVector3(0, 0, 0);

	b3Scalar scaledDenom;
	{
#ifdef COMPUTE_IMPULSE_DENOM
		b3Scalar denom0 = rb0->computeImpulseDenominator(pos1, cp.m_normalWorldOnB);
		b3Scalar denom1 = rb1->computeImpulseDenominator(pos2, cp.m_normalWorldOnB);
#else
		b3Vector3 vec;
		b3Scalar denom0 = 0.f;
		b3Scalar denom1 = 0.f;
		if (rb0)
		{
			vec = (solverConstraint.m_angularComponentA).cross(rel_pos1);
			denom0 = rb0->m_invMass + cp.m_normalWorldOnB.dot(vec);
		}
		if (rb1)
		{
			vec = (-solverConstraint.m_angularComponentB).cross(rel_pos2);
			denom1 = rb1->m_invMass + cp.m_normalWorldOnB.dot(vec);
		}
#endif  //COMPUTE_IMPULSE_DENOM

		b3Scalar denom;
		if (m_usePgs)
		{
			scaledDenom = denom = relaxation / (denom0 + denom1);
		}
		else
		{
			denom = relaxation / (denom0 + denom1);

			b3Scalar countA = rb0->m_invMass ? b3Scalar(m_bodyCount[bodyA->m_originalBodyIndex]) : 1.f;
			b3Scalar countB = rb1->m_invMass ? b3Scalar(m_bodyCount[bodyB->m_originalBodyIndex]) : 1.f;
			scaledDenom = relaxation / (denom0 * countA + denom1 * countB);
		}
		solverConstraint.m_jacDiagABInv = denom;
	}

	solverConstraint.m_contactNormal = cp.m_normalWorldOnB;
	solverConstraint.m_relpos1CrossNormal = torqueAxis0;
	solverConstraint.m_relpos2CrossNormal = -torqueAxis1;

	b3Scalar restitution = 0.f;
	b3Scalar penetration = cp.getDistance() + infoGlobal.m_linearSlop;

	{
		b3Vector3 vel1, vel2;

		vel1 = rb0 ? getVelocityInLocalPoint(rb0, rel_pos1) : b3MakeVector3(0, 0, 0);
		vel2 = rb1 ? getVelocityInLocalPoint(rb1, rel_pos2) : b3MakeVector3(0, 0, 0);

		//			b3Vector3 vel2 = rb1 ? rb1->getVelocityInLocalPoint(rel_pos2) : b3Vector3(0,0,0);
		vel = vel1 - vel2;
		rel_vel = cp.m_normalWorldOnB.dot(vel);

		solverConstraint.m_friction = cp.m_combinedFriction;

		restitution = restitutionCurve(rel_vel, cp.m_combinedRestitution);
		if (restitution <= b3Scalar(0.))
		{
			restitution = 0.f;
		};
	}

	///warm starting (or zero if disabled)
	if (infoGlobal.m_solverMode & B3_SOLVER_USE_WARMSTARTING)
	{
		solverConstraint.m_appliedImpulse = cp.m_appliedImpulse * infoGlobal.m_warmstartingFactor;
		if (rb0)
			bodyA->internalApplyImpulse(solverConstraint.m_contactNormal * bodyA->internalGetInvMass(), solverConstraint.m_angularComponentA, solverConstraint.m_appliedImpulse);
		if (rb1)
			bodyB->internalApplyImpulse(solverConstraint.m_contactNormal * bodyB->internalGetInvMass(), -solverConstraint.m_angularComponentB, -(b3Scalar)solverConstraint.m_appliedImpulse);
	}
	else
	{
		solverConstraint.m_appliedImpulse = 0.f;
	}

	solverConstraint.m_appliedPushImpulse = 0.f;

	{
		b3Scalar vel1Dotn = solverConstraint.m_contactNormal.dot(rb0 ? bodyA->m_linearVelocity : b3MakeVector3(0, 0, 0)) + solverConstraint.m_relpos1CrossNormal.dot(rb0 ? bodyA->m_angularVelocity : b3MakeVector3(0, 0, 0));
		b3Scalar vel2Dotn = -solverConstraint.m_contactNormal.dot(rb1 ? bodyB->m_linearVelocity : b3MakeVector3(0, 0, 0)) + solverConstraint.m_relpos2CrossNormal.dot(rb1 ? bodyB->m_angularVelocity : b3MakeVector3(0, 0, 0));
		b3Scalar rel_vel = vel1Dotn + vel2Dotn;

		b3Scalar positionalError = 0.f;
		b3Scalar velocityError = restitution - rel_vel;  // * damping;

		b3Scalar erp = infoGlobal.m_erp2;
		if (!infoGlobal.m_splitImpulse || (penetration > infoGlobal.m_splitImpulsePenetrationThreshold))
		{
			erp = infoGlobal.m_erp;
		}

		if (penetration > 0)
		{
			positionalError = 0;

			velocityError -= penetration / infoGlobal.m_timeStep;
		}
		else
		{
			positionalError = -penetration * erp / infoGlobal.m_timeStep;
		}

		b3Scalar penetrationImpulse = positionalError * scaledDenom;  //solverConstraint.m_jacDiagABInv;
		b3Scalar velocityImpulse = velocityError * scaledDenom;       //solverConstraint.m_jacDiagABInv;

		if (!infoGlobal.m_splitImpulse || (penetration > infoGlobal.m_splitImpulsePenetrationThreshold))
		{
			//combine position and velocity into rhs
			solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;
			solverConstraint.m_rhsPenetration = 0.f;
		}
		else
		{
			//split position and velocity into rhs and m_rhsPenetration
			solverConstraint.m_rhs = velocityImpulse;
			solverConstraint.m_rhsPenetration = penetrationImpulse;
		}
		solverConstraint.m_cfm = 0.f;
		solverConstraint.m_lowerLimit = 0;
		solverConstraint.m_upperLimit = 1e10f;
	}
}

void b3PgsJacobiSolver::setFrictionConstraintImpulse(b3RigidBodyData* bodies, b3InertiaData* inertias, b3SolverConstraint& solverConstraint,
													 int solverBodyIdA, int solverBodyIdB,
													 b3ContactPoint& cp, const b3ContactSolverInfo& infoGlobal)
{
	b3SolverBody* bodyA = &m_tmpSolverBodyPool[solverBodyIdA];
	b3SolverBody* bodyB = &m_tmpSolverBodyPool[solverBodyIdB];

	{
		b3SolverConstraint& frictionConstraint1 = m_tmpSolverContactFrictionConstraintPool[solverConstraint.m_frictionIndex];
		if (infoGlobal.m_solverMode & B3_SOLVER_USE_WARMSTARTING)
		{
			frictionConstraint1.m_appliedImpulse = cp.m_appliedImpulseLateral1 * infoGlobal.m_warmstartingFactor;
			if (bodies[bodyA->m_originalBodyIndex].m_invMass)
				bodyA->internalApplyImpulse(frictionConstraint1.m_contactNormal * bodies[bodyA->m_originalBodyIndex].m_invMass, frictionConstraint1.m_angularComponentA, frictionConstraint1.m_appliedImpulse);
			if (bodies[bodyB->m_originalBodyIndex].m_invMass)
				bodyB->internalApplyImpulse(frictionConstraint1.m_contactNormal * bodies[bodyB->m_originalBodyIndex].m_invMass, -frictionConstraint1.m_angularComponentB, -(b3Scalar)frictionConstraint1.m_appliedImpulse);
		}
		else
		{
			frictionConstraint1.m_appliedImpulse = 0.f;
		}
	}

	if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS))
	{
		b3SolverConstraint& frictionConstraint2 = m_tmpSolverContactFrictionConstraintPool[solverConstraint.m_frictionIndex + 1];
		if (infoGlobal.m_solverMode & B3_SOLVER_USE_WARMSTARTING)
		{
			frictionConstraint2.m_appliedImpulse = cp.m_appliedImpulseLateral2 * infoGlobal.m_warmstartingFactor;
			if (bodies[bodyA->m_originalBodyIndex].m_invMass)
				bodyA->internalApplyImpulse(frictionConstraint2.m_contactNormal * bodies[bodyA->m_originalBodyIndex].m_invMass, frictionConstraint2.m_angularComponentA, frictionConstraint2.m_appliedImpulse);
			if (bodies[bodyB->m_originalBodyIndex].m_invMass)
				bodyB->internalApplyImpulse(frictionConstraint2.m_contactNormal * bodies[bodyB->m_originalBodyIndex].m_invMass, -frictionConstraint2.m_angularComponentB, -(b3Scalar)frictionConstraint2.m_appliedImpulse);
		}
		else
		{
			frictionConstraint2.m_appliedImpulse = 0.f;
		}
	}
}

void b3PgsJacobiSolver::convertContact(b3RigidBodyData* bodies, b3InertiaData* inertias, b3Contact4* manifold, const b3ContactSolverInfo& infoGlobal)
{
	b3RigidBodyData *colObj0 = 0, *colObj1 = 0;

	int solverBodyIdA = getOrInitSolverBody(manifold->getBodyA(), bodies, inertias);
	int solverBodyIdB = getOrInitSolverBody(manifold->getBodyB(), bodies, inertias);

	//	b3RigidBody* bodyA = b3RigidBody::upcast(colObj0);
	//	b3RigidBody* bodyB = b3RigidBody::upcast(colObj1);

	b3SolverBody* solverBodyA = &m_tmpSolverBodyPool[solverBodyIdA];
	b3SolverBody* solverBodyB = &m_tmpSolverBodyPool[solverBodyIdB];

	///avoid collision response between two static objects
	if (solverBodyA->m_invMass.isZero() && solverBodyB->m_invMass.isZero())
		return;

	int rollingFriction = 1;
	int numContacts = getNumContacts(manifold);
	for (int j = 0; j < numContacts; j++)
	{
		b3ContactPoint cp;
		getContactPoint(manifold, j, cp);

		if (cp.getDistance() <= getContactProcessingThreshold(manifold))
		{
			b3Vector3 rel_pos1;
			b3Vector3 rel_pos2;
			b3Scalar relaxation;
			b3Scalar rel_vel;
			b3Vector3 vel;

			int frictionIndex = m_tmpSolverContactConstraintPool.size();
			b3SolverConstraint& solverConstraint = m_tmpSolverContactConstraintPool.expandNonInitializing();
			//			b3RigidBody* rb0 = b3RigidBody::upcast(colObj0);
			//			b3RigidBody* rb1 = b3RigidBody::upcast(colObj1);
			solverConstraint.m_solverBodyIdA = solverBodyIdA;
			solverConstraint.m_solverBodyIdB = solverBodyIdB;

			solverConstraint.m_originalContactPoint = &cp;

			setupContactConstraint(bodies, inertias, solverConstraint, solverBodyIdA, solverBodyIdB, cp, infoGlobal, vel, rel_vel, relaxation, rel_pos1, rel_pos2);

			//			const b3Vector3& pos1 = cp.getPositionWorldOnA();
			//			const b3Vector3& pos2 = cp.getPositionWorldOnB();

			/////setup the friction constraints

			solverConstraint.m_frictionIndex = m_tmpSolverContactFrictionConstraintPool.size();

			b3Vector3 angVelA, angVelB;
			solverBodyA->getAngularVelocity(angVelA);
			solverBodyB->getAngularVelocity(angVelB);
			b3Vector3 relAngVel = angVelB - angVelA;

			if ((cp.m_combinedRollingFriction > 0.f) && (rollingFriction > 0))
			{
				//only a single rollingFriction per manifold
				rollingFriction--;
				if (relAngVel.length() > infoGlobal.m_singleAxisRollingFrictionThreshold)
				{
					relAngVel.normalize();
					if (relAngVel.length() > 0.001)
						addRollingFrictionConstraint(bodies, inertias, relAngVel, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
				}
				else
				{
					addRollingFrictionConstraint(bodies, inertias, cp.m_normalWorldOnB, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
					b3Vector3 axis0, axis1;
					b3PlaneSpace1(cp.m_normalWorldOnB, axis0, axis1);
					if (axis0.length() > 0.001)
						addRollingFrictionConstraint(bodies, inertias, axis0, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
					if (axis1.length() > 0.001)
						addRollingFrictionConstraint(bodies, inertias, axis1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
				}
			}

			///Bullet has several options to set the friction directions
			///By default, each contact has only a single friction direction that is recomputed automatically very frame
			///based on the relative linear velocity.
			///If the relative velocity it zero, it will automatically compute a friction direction.

			///You can also enable two friction directions, using the B3_SOLVER_USE_2_FRICTION_DIRECTIONS.
			///In that case, the second friction direction will be orthogonal to both contact normal and first friction direction.
			///
			///If you choose B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION, then the friction will be independent from the relative projected velocity.
			///
			///The user can manually override the friction directions for certain contacts using a contact callback,
			///and set the cp.m_lateralFrictionInitialized to true
			///In that case, you can set the target relative motion in each friction direction (cp.m_contactMotion1 and cp.m_contactMotion2)
			///this will give a conveyor belt effect
			///
			if (!(infoGlobal.m_solverMode & B3_SOLVER_ENABLE_FRICTION_DIRECTION_CACHING) || !cp.m_lateralFrictionInitialized)
			{
				cp.m_lateralFrictionDir1 = vel - cp.m_normalWorldOnB * rel_vel;
				b3Scalar lat_rel_vel = cp.m_lateralFrictionDir1.length2();
				if (!(infoGlobal.m_solverMode & B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION) && lat_rel_vel > B3_EPSILON)
				{
					cp.m_lateralFrictionDir1 *= 1.f / b3Sqrt(lat_rel_vel);
					if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS))
					{
						cp.m_lateralFrictionDir2 = cp.m_lateralFrictionDir1.cross(cp.m_normalWorldOnB);
						cp.m_lateralFrictionDir2.normalize();  //??
						addFrictionConstraint(bodies, inertias, cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
					}

					addFrictionConstraint(bodies, inertias, cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
				}
				else
				{
					b3PlaneSpace1(cp.m_normalWorldOnB, cp.m_lateralFrictionDir1, cp.m_lateralFrictionDir2);

					if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS))
					{
						addFrictionConstraint(bodies, inertias, cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
					}

					addFrictionConstraint(bodies, inertias, cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);

					if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS) && (infoGlobal.m_solverMode & B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION))
					{
						cp.m_lateralFrictionInitialized = true;
					}
				}
			}
			else
			{
				addFrictionConstraint(bodies, inertias, cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, cp.m_contactMotion1, cp.m_contactCFM1);

				if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS))
					addFrictionConstraint(bodies, inertias, cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, cp.m_contactMotion2, cp.m_contactCFM2);

				setFrictionConstraintImpulse(bodies, inertias, solverConstraint, solverBodyIdA, solverBodyIdB, cp, infoGlobal);
			}
		}
	}
}

b3Scalar b3PgsJacobiSolver::solveGroupCacheFriendlySetup(b3RigidBodyData* bodies, b3InertiaData* inertias, int numBodies, b3Contact4* manifoldPtr, int numManifolds, b3TypedConstraint** constraints, int numConstraints, const b3ContactSolverInfo& infoGlobal)
{
	B3_PROFILE("solveGroupCacheFriendlySetup");

	m_maxOverrideNumSolverIterations = 0;

	m_tmpSolverBodyPool.resize(0);

	m_bodyCount.resize(0);
	m_bodyCount.resize(numBodies, 0);
	m_bodyCountCheck.resize(0);
	m_bodyCountCheck.resize(numBodies, 0);

	m_deltaLinearVelocities.resize(0);
	m_deltaLinearVelocities.resize(numBodies, b3MakeVector3(0, 0, 0));
	m_deltaAngularVelocities.resize(0);
	m_deltaAngularVelocities.resize(numBodies, b3MakeVector3(0, 0, 0));

	//int totalBodies = 0;

	for (int i = 0; i < numConstraints; i++)
	{
		int bodyIndexA = constraints[i]->getRigidBodyA();
		int bodyIndexB = constraints[i]->getRigidBodyB();
		if (m_usePgs)
		{
			m_bodyCount[bodyIndexA] = -1;
			m_bodyCount[bodyIndexB] = -1;
		}
		else
		{
			//didn't implement joints with Jacobi version yet
			b3Assert(0);
		}
	}
	for (int i = 0; i < numManifolds; i++)
	{
		int bodyIndexA = manifoldPtr[i].getBodyA();
		int bodyIndexB = manifoldPtr[i].getBodyB();
		if (m_usePgs)
		{
			m_bodyCount[bodyIndexA] = -1;
			m_bodyCount[bodyIndexB] = -1;
		}
		else
		{
			if (bodies[bodyIndexA].m_invMass)
			{
				//m_bodyCount[bodyIndexA]+=manifoldPtr[i].getNPoints();
				m_bodyCount[bodyIndexA]++;
			}
			else
				m_bodyCount[bodyIndexA] = -1;

			if (bodies[bodyIndexB].m_invMass)
				//	m_bodyCount[bodyIndexB]+=manifoldPtr[i].getNPoints();
				m_bodyCount[bodyIndexB]++;
			else
				m_bodyCount[bodyIndexB] = -1;
		}
	}

	if (1)
	{
		int j;
		for (j = 0; j < numConstraints; j++)
		{
			b3TypedConstraint* constraint = constraints[j];

			constraint->internalSetAppliedImpulse(0.0f);
		}
	}

	//b3RigidBody* rb0=0,*rb1=0;
	//if (1)
	{
		{
			int totalNumRows = 0;
			int i;

			m_tmpConstraintSizesPool.resizeNoInitialize(numConstraints);
			//calculate the total number of contraint rows
			for (i = 0; i < numConstraints; i++)
			{
				b3TypedConstraint::b3ConstraintInfo1& info1 = m_tmpConstraintSizesPool[i];
				b3JointFeedback* fb = constraints[i]->getJointFeedback();
				if (fb)
				{
					fb->m_appliedForceBodyA.setZero();
					fb->m_appliedTorqueBodyA.setZero();
					fb->m_appliedForceBodyB.setZero();
					fb->m_appliedTorqueBodyB.setZero();
				}

				if (constraints[i]->isEnabled())
				{
				}
				if (constraints[i]->isEnabled())
				{
					constraints[i]->getInfo1(&info1, bodies);
				}
				else
				{
					info1.m_numConstraintRows = 0;
					info1.nub = 0;
				}
				totalNumRows += info1.m_numConstraintRows;
			}
			m_tmpSolverNonContactConstraintPool.resizeNoInitialize(totalNumRows);

#ifndef DISABLE_JOINTS
			///setup the b3SolverConstraints
			int currentRow = 0;

			for (i = 0; i < numConstraints; i++)
			{
				const b3TypedConstraint::b3ConstraintInfo1& info1 = m_tmpConstraintSizesPool[i];

				if (info1.m_numConstraintRows)
				{
					b3Assert(currentRow < totalNumRows);

					b3SolverConstraint* currentConstraintRow = &m_tmpSolverNonContactConstraintPool[currentRow];
					b3TypedConstraint* constraint = constraints[i];

					b3RigidBodyData& rbA = bodies[constraint->getRigidBodyA()];
					//b3RigidBody& rbA = constraint->getRigidBodyA();
					//				b3RigidBody& rbB = constraint->getRigidBodyB();
					b3RigidBodyData& rbB = bodies[constraint->getRigidBodyB()];

					int solverBodyIdA = getOrInitSolverBody(constraint->getRigidBodyA(), bodies, inertias);
					int solverBodyIdB = getOrInitSolverBody(constraint->getRigidBodyB(), bodies, inertias);

					b3SolverBody* bodyAPtr = &m_tmpSolverBodyPool[solverBodyIdA];
					b3SolverBody* bodyBPtr = &m_tmpSolverBodyPool[solverBodyIdB];

					int overrideNumSolverIterations = constraint->getOverrideNumSolverIterations() > 0 ? constraint->getOverrideNumSolverIterations() : infoGlobal.m_numIterations;
					if (overrideNumSolverIterations > m_maxOverrideNumSolverIterations)
						m_maxOverrideNumSolverIterations = overrideNumSolverIterations;

					int j;
					for (j = 0; j < info1.m_numConstraintRows; j++)
					{
						memset(&currentConstraintRow[j], 0, sizeof(b3SolverConstraint));
						currentConstraintRow[j].m_lowerLimit = -B3_INFINITY;
						currentConstraintRow[j].m_upperLimit = B3_INFINITY;
						currentConstraintRow[j].m_appliedImpulse = 0.f;
						currentConstraintRow[j].m_appliedPushImpulse = 0.f;
						currentConstraintRow[j].m_solverBodyIdA = solverBodyIdA;
						currentConstraintRow[j].m_solverBodyIdB = solverBodyIdB;
						currentConstraintRow[j].m_overrideNumSolverIterations = overrideNumSolverIterations;
					}

					bodyAPtr->internalGetDeltaLinearVelocity().setValue(0.f, 0.f, 0.f);
					bodyAPtr->internalGetDeltaAngularVelocity().setValue(0.f, 0.f, 0.f);
					bodyAPtr->internalGetPushVelocity().setValue(0.f, 0.f, 0.f);
					bodyAPtr->internalGetTurnVelocity().setValue(0.f, 0.f, 0.f);
					bodyBPtr->internalGetDeltaLinearVelocity().setValue(0.f, 0.f, 0.f);
					bodyBPtr->internalGetDeltaAngularVelocity().setValue(0.f, 0.f, 0.f);
					bodyBPtr->internalGetPushVelocity().setValue(0.f, 0.f, 0.f);
					bodyBPtr->internalGetTurnVelocity().setValue(0.f, 0.f, 0.f);

					b3TypedConstraint::b3ConstraintInfo2 info2;
					info2.fps = 1.f / infoGlobal.m_timeStep;
					info2.erp = infoGlobal.m_erp;
					info2.m_J1linearAxis = currentConstraintRow->m_contactNormal;
					info2.m_J1angularAxis = currentConstraintRow->m_relpos1CrossNormal;
					info2.m_J2linearAxis = 0;
					info2.m_J2angularAxis = currentConstraintRow->m_relpos2CrossNormal;
					info2.rowskip = sizeof(b3SolverConstraint) / sizeof(b3Scalar);  //check this
																					///the size of b3SolverConstraint needs be a multiple of b3Scalar
					b3Assert(info2.rowskip * sizeof(b3Scalar) == sizeof(b3SolverConstraint));
					info2.m_constraintError = &currentConstraintRow->m_rhs;
					currentConstraintRow->m_cfm = infoGlobal.m_globalCfm;
					info2.m_damping = infoGlobal.m_damping;
					info2.cfm = &currentConstraintRow->m_cfm;
					info2.m_lowerLimit = &currentConstraintRow->m_lowerLimit;
					info2.m_upperLimit = &currentConstraintRow->m_upperLimit;
					info2.m_numIterations = infoGlobal.m_numIterations;
					constraints[i]->getInfo2(&info2, bodies);

					///finalize the constraint setup
					for (j = 0; j < info1.m_numConstraintRows; j++)
					{
						b3SolverConstraint& solverConstraint = currentConstraintRow[j];

						if (solverConstraint.m_upperLimit >= constraints[i]->getBreakingImpulseThreshold())
						{
							solverConstraint.m_upperLimit = constraints[i]->getBreakingImpulseThreshold();
						}

						if (solverConstraint.m_lowerLimit <= -constraints[i]->getBreakingImpulseThreshold())
						{
							solverConstraint.m_lowerLimit = -constraints[i]->getBreakingImpulseThreshold();
						}

						solverConstraint.m_originalContactPoint = constraint;

						b3Matrix3x3& invInertiaWorldA = inertias[constraint->getRigidBodyA()].m_invInertiaWorld;
						{
							//b3Vector3 angularFactorA(1,1,1);
							const b3Vector3& ftorqueAxis1 = solverConstraint.m_relpos1CrossNormal;
							solverConstraint.m_angularComponentA = invInertiaWorldA * ftorqueAxis1;  //*angularFactorA;
						}

						b3Matrix3x3& invInertiaWorldB = inertias[constraint->getRigidBodyB()].m_invInertiaWorld;
						{
							const b3Vector3& ftorqueAxis2 = solverConstraint.m_relpos2CrossNormal;
							solverConstraint.m_angularComponentB = invInertiaWorldB * ftorqueAxis2;  //*constraint->getRigidBodyB().getAngularFactor();
						}

						{
							//it is ok to use solverConstraint.m_contactNormal instead of -solverConstraint.m_contactNormal
							//because it gets multiplied iMJlB
							b3Vector3 iMJlA = solverConstraint.m_contactNormal * rbA.m_invMass;
							b3Vector3 iMJaA = invInertiaWorldA * solverConstraint.m_relpos1CrossNormal;
							b3Vector3 iMJlB = solverConstraint.m_contactNormal * rbB.m_invMass;  //sign of normal?
							b3Vector3 iMJaB = invInertiaWorldB * solverConstraint.m_relpos2CrossNormal;

							b3Scalar sum = iMJlA.dot(solverConstraint.m_contactNormal);
							sum += iMJaA.dot(solverConstraint.m_relpos1CrossNormal);
							sum += iMJlB.dot(solverConstraint.m_contactNormal);
							sum += iMJaB.dot(solverConstraint.m_relpos2CrossNormal);
							b3Scalar fsum = b3Fabs(sum);
							b3Assert(fsum > B3_EPSILON);
							solverConstraint.m_jacDiagABInv = fsum > B3_EPSILON ? b3Scalar(1.) / sum : 0.f;
						}

						///fix rhs
						///todo: add force/torque accelerators
						{
							b3Scalar rel_vel;
							b3Scalar vel1Dotn = solverConstraint.m_contactNormal.dot(rbA.m_linVel) + solverConstraint.m_relpos1CrossNormal.dot(rbA.m_angVel);
							b3Scalar vel2Dotn = -solverConstraint.m_contactNormal.dot(rbB.m_linVel) + solverConstraint.m_relpos2CrossNormal.dot(rbB.m_angVel);

							rel_vel = vel1Dotn + vel2Dotn;

							b3Scalar restitution = 0.f;
							b3Scalar positionalError = solverConstraint.m_rhs;  //already filled in by getConstraintInfo2
							b3Scalar velocityError = restitution - rel_vel * info2.m_damping;
							b3Scalar penetrationImpulse = positionalError * solverConstraint.m_jacDiagABInv;
							b3Scalar velocityImpulse = velocityError * solverConstraint.m_jacDiagABInv;
							solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;
							solverConstraint.m_appliedImpulse = 0.f;
						}
					}
				}
				currentRow += m_tmpConstraintSizesPool[i].m_numConstraintRows;
			}
#endif  //DISABLE_JOINTS
		}

		{
			int i;

			for (i = 0; i < numManifolds; i++)
			{
				b3Contact4& manifold = manifoldPtr[i];
				convertContact(bodies, inertias, &manifold, infoGlobal);
			}
		}
	}

	//	b3ContactSolverInfo info = infoGlobal;

	int numNonContactPool = m_tmpSolverNonContactConstraintPool.size();
	int numConstraintPool = m_tmpSolverContactConstraintPool.size();
	int numFrictionPool = m_tmpSolverContactFrictionConstraintPool.size();

	///@todo: use stack allocator for such temporarily memory, same for solver bodies/constraints
	m_orderNonContactConstraintPool.resizeNoInitialize(numNonContactPool);
	if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS))
		m_orderTmpConstraintPool.resizeNoInitialize(numConstraintPool * 2);
	else
		m_orderTmpConstraintPool.resizeNoInitialize(numConstraintPool);

	m_orderFrictionConstraintPool.resizeNoInitialize(numFrictionPool);
	{
		int i;
		for (i = 0; i < numNonContactPool; i++)
		{
			m_orderNonContactConstraintPool[i] = i;
		}
		for (i = 0; i < numConstraintPool; i++)
		{
			m_orderTmpConstraintPool[i] = i;
		}
		for (i = 0; i < numFrictionPool; i++)
		{
			m_orderFrictionConstraintPool[i] = i;
		}
	}

	return 0.f;
}

b3Scalar b3PgsJacobiSolver::solveSingleIteration(int iteration, b3TypedConstraint** constraints, int numConstraints, const b3ContactSolverInfo& infoGlobal)
{
	int numNonContactPool = m_tmpSolverNonContactConstraintPool.size();
	int numConstraintPool = m_tmpSolverContactConstraintPool.size();
	int numFrictionPool = m_tmpSolverContactFrictionConstraintPool.size();

	if (infoGlobal.m_solverMode & B3_SOLVER_RANDMIZE_ORDER)
	{
		if (1)  // uncomment this for a bit less random ((iteration & 7) == 0)
		{
			for (int j = 0; j < numNonContactPool; ++j)
			{
				int tmp = m_orderNonContactConstraintPool[j];
				int swapi = b3RandInt2(j + 1);
				m_orderNonContactConstraintPool[j] = m_orderNonContactConstraintPool[swapi];
				m_orderNonContactConstraintPool[swapi] = tmp;
			}

			//contact/friction constraints are not solved more than
			if (iteration < infoGlobal.m_numIterations)
			{
				for (int j = 0; j < numConstraintPool; ++j)
				{
					int tmp = m_orderTmpConstraintPool[j];
					int swapi = b3RandInt2(j + 1);
					m_orderTmpConstraintPool[j] = m_orderTmpConstraintPool[swapi];
					m_orderTmpConstraintPool[swapi] = tmp;
				}

				for (int j = 0; j < numFrictionPool; ++j)
				{
					int tmp = m_orderFrictionConstraintPool[j];
					int swapi = b3RandInt2(j + 1);
					m_orderFrictionConstraintPool[j] = m_orderFrictionConstraintPool[swapi];
					m_orderFrictionConstraintPool[swapi] = tmp;
				}
			}
		}
	}

	if (infoGlobal.m_solverMode & B3_SOLVER_SIMD)
	{
		///solve all joint constraints, using SIMD, if available
		for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++)
		{
			b3SolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[m_orderNonContactConstraintPool[j]];
			if (iteration < constraint.m_overrideNumSolverIterations)
				resolveSingleConstraintRowGenericSIMD(m_tmpSolverBodyPool[constraint.m_solverBodyIdA], m_tmpSolverBodyPool[constraint.m_solverBodyIdB], constraint);
		}

		if (iteration < infoGlobal.m_numIterations)
		{
			///solve all contact constraints using SIMD, if available
			if (infoGlobal.m_solverMode & B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS)
			{
				int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
				int multiplier = (infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS) ? 2 : 1;

				for (int c = 0; c < numPoolConstraints; c++)
				{
					b3Scalar totalImpulse = 0;

					{
						const b3SolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[c]];
						resolveSingleConstraintRowLowerLimitSIMD(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
						totalImpulse = solveManifold.m_appliedImpulse;
					}
					bool applyFriction = true;
					if (applyFriction)
					{
						{
							b3SolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[c * multiplier]];

							if (totalImpulse > b3Scalar(0))
							{
								solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
								solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

								resolveSingleConstraintRowGenericSIMD(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
							}
						}

						if (infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS)
						{
							b3SolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[c * multiplier + 1]];

							if (totalImpulse > b3Scalar(0))
							{
								solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
								solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

								resolveSingleConstraintRowGenericSIMD(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
							}
						}
					}
				}
			}
			else  //B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS
			{
				//solve the friction constraints after all contact constraints, don't interleave them
				int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
				int j;

				for (j = 0; j < numPoolConstraints; j++)
				{
					const b3SolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];
					resolveSingleConstraintRowLowerLimitSIMD(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
				}

				if (!m_usePgs)
					averageVelocities();

				///solve all friction constraints, using SIMD, if available

				int numFrictionPoolConstraints = m_tmpSolverContactFrictionConstraintPool.size();
				for (j = 0; j < numFrictionPoolConstraints; j++)
				{
					b3SolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[j]];
					b3Scalar totalImpulse = m_tmpSolverContactConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;

					if (totalImpulse > b3Scalar(0))
					{
						solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
						solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

						resolveSingleConstraintRowGenericSIMD(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
					}
				}

				int numRollingFrictionPoolConstraints = m_tmpSolverContactRollingFrictionConstraintPool.size();
				for (j = 0; j < numRollingFrictionPoolConstraints; j++)
				{
					b3SolverConstraint& rollingFrictionConstraint = m_tmpSolverContactRollingFrictionConstraintPool[j];
					b3Scalar totalImpulse = m_tmpSolverContactConstraintPool[rollingFrictionConstraint.m_frictionIndex].m_appliedImpulse;
					if (totalImpulse > b3Scalar(0))
					{
						b3Scalar rollingFrictionMagnitude = rollingFrictionConstraint.m_friction * totalImpulse;
						if (rollingFrictionMagnitude > rollingFrictionConstraint.m_friction)
							rollingFrictionMagnitude = rollingFrictionConstraint.m_friction;

						rollingFrictionConstraint.m_lowerLimit = -rollingFrictionMagnitude;
						rollingFrictionConstraint.m_upperLimit = rollingFrictionMagnitude;

						resolveSingleConstraintRowGenericSIMD(m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdA], m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdB], rollingFrictionConstraint);
					}
				}
			}
		}
	}
	else
	{
		//non-SIMD version
		///solve all joint constraints
		for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++)
		{
			b3SolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[m_orderNonContactConstraintPool[j]];
			if (iteration < constraint.m_overrideNumSolverIterations)
				resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[constraint.m_solverBodyIdA], m_tmpSolverBodyPool[constraint.m_solverBodyIdB], constraint);
		}

		if (iteration < infoGlobal.m_numIterations)
		{
			///solve all contact constraints
			int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
			for (int j = 0; j < numPoolConstraints; j++)
			{
				const b3SolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];
				resolveSingleConstraintRowLowerLimit(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
			}
			///solve all friction constraints
			int numFrictionPoolConstraints = m_tmpSolverContactFrictionConstraintPool.size();
			for (int j = 0; j < numFrictionPoolConstraints; j++)
			{
				b3SolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[j]];
				b3Scalar totalImpulse = m_tmpSolverContactConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;

				if (totalImpulse > b3Scalar(0))
				{
					solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
					solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

					resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
				}
			}

			int numRollingFrictionPoolConstraints = m_tmpSolverContactRollingFrictionConstraintPool.size();
			for (int j = 0; j < numRollingFrictionPoolConstraints; j++)
			{
				b3SolverConstraint& rollingFrictionConstraint = m_tmpSolverContactRollingFrictionConstraintPool[j];
				b3Scalar totalImpulse = m_tmpSolverContactConstraintPool[rollingFrictionConstraint.m_frictionIndex].m_appliedImpulse;
				if (totalImpulse > b3Scalar(0))
				{
					b3Scalar rollingFrictionMagnitude = rollingFrictionConstraint.m_friction * totalImpulse;
					if (rollingFrictionMagnitude > rollingFrictionConstraint.m_friction)
						rollingFrictionMagnitude = rollingFrictionConstraint.m_friction;

					rollingFrictionConstraint.m_lowerLimit = -rollingFrictionMagnitude;
					rollingFrictionConstraint.m_upperLimit = rollingFrictionMagnitude;

					resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdA], m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdB], rollingFrictionConstraint);
				}
			}
		}
	}
	return 0.f;
}

void b3PgsJacobiSolver::solveGroupCacheFriendlySplitImpulseIterations(b3TypedConstraint** constraints, int numConstraints, const b3ContactSolverInfo& infoGlobal)
{
	int iteration;
	if (infoGlobal.m_splitImpulse)
	{
		if (infoGlobal.m_solverMode & B3_SOLVER_SIMD)
		{
			for (iteration = 0; iteration < infoGlobal.m_numIterations; iteration++)
			{
				{
					int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
					int j;
					for (j = 0; j < numPoolConstraints; j++)
					{
						const b3SolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];

						resolveSplitPenetrationSIMD(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
					}
				}
			}
		}
		else
		{
			for (iteration = 0; iteration < infoGlobal.m_numIterations; iteration++)
			{
				{
					int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
					int j;
					for (j = 0; j < numPoolConstraints; j++)
					{
						const b3SolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];

						resolveSplitPenetrationImpulseCacheFriendly(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
					}
				}
			}
		}
	}
}

b3Scalar b3PgsJacobiSolver::solveGroupCacheFriendlyIterations(b3TypedConstraint** constraints, int numConstraints, const b3ContactSolverInfo& infoGlobal)
{
	B3_PROFILE("solveGroupCacheFriendlyIterations");

	{
		///this is a special step to resolve penetrations (just for contacts)
		solveGroupCacheFriendlySplitImpulseIterations(constraints, numConstraints, infoGlobal);

		int maxIterations = m_maxOverrideNumSolverIterations > infoGlobal.m_numIterations ? m_maxOverrideNumSolverIterations : infoGlobal.m_numIterations;

		for (int iteration = 0; iteration < maxIterations; iteration++)
		//for ( int iteration = maxIterations-1  ; iteration >= 0;iteration--)
		{
			solveSingleIteration(iteration, constraints, numConstraints, infoGlobal);

			if (!m_usePgs)
			{
				averageVelocities();
			}
		}
	}
	return 0.f;
}

void b3PgsJacobiSolver::averageVelocities()
{
	B3_PROFILE("averaging");
	//average the velocities
	int numBodies = m_bodyCount.size();

	m_deltaLinearVelocities.resize(0);
	m_deltaLinearVelocities.resize(numBodies, b3MakeVector3(0, 0, 0));
	m_deltaAngularVelocities.resize(0);
	m_deltaAngularVelocities.resize(numBodies, b3MakeVector3(0, 0, 0));

	for (int i = 0; i < m_tmpSolverBodyPool.size(); i++)
	{
		if (!m_tmpSolverBodyPool[i].m_invMass.isZero())
		{
			int orgBodyIndex = m_tmpSolverBodyPool[i].m_originalBodyIndex;
			m_deltaLinearVelocities[orgBodyIndex] += m_tmpSolverBodyPool[i].getDeltaLinearVelocity();
			m_deltaAngularVelocities[orgBodyIndex] += m_tmpSolverBodyPool[i].getDeltaAngularVelocity();
		}
	}

	for (int i = 0; i < m_tmpSolverBodyPool.size(); i++)
	{
		int orgBodyIndex = m_tmpSolverBodyPool[i].m_originalBodyIndex;

		if (!m_tmpSolverBodyPool[i].m_invMass.isZero())
		{
			b3Assert(m_bodyCount[orgBodyIndex] == m_bodyCountCheck[orgBodyIndex]);

			b3Scalar factor = 1.f / b3Scalar(m_bodyCount[orgBodyIndex]);

			m_tmpSolverBodyPool[i].m_deltaLinearVelocity = m_deltaLinearVelocities[orgBodyIndex] * factor;
			m_tmpSolverBodyPool[i].m_deltaAngularVelocity = m_deltaAngularVelocities[orgBodyIndex] * factor;
		}
	}
}

b3Scalar b3PgsJacobiSolver::solveGroupCacheFriendlyFinish(b3RigidBodyData* bodies, b3InertiaData* inertias, int numBodies, const b3ContactSolverInfo& infoGlobal)
{
	B3_PROFILE("solveGroupCacheFriendlyFinish");
	int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
	int i, j;

	if (infoGlobal.m_solverMode & B3_SOLVER_USE_WARMSTARTING)
	{
		for (j = 0; j < numPoolConstraints; j++)
		{
			const b3SolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[j];
			b3ContactPoint* pt = (b3ContactPoint*)solveManifold.m_originalContactPoint;
			b3Assert(pt);
			pt->m_appliedImpulse = solveManifold.m_appliedImpulse;
			//	float f = m_tmpSolverContactFrictionConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;
			//	printf("pt->m_appliedImpulseLateral1 = %f\n", f);
			pt->m_appliedImpulseLateral1 = m_tmpSolverContactFrictionConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;
			//printf("pt->m_appliedImpulseLateral1 = %f\n", pt->m_appliedImpulseLateral1);
			if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS))
			{
				pt->m_appliedImpulseLateral2 = m_tmpSolverContactFrictionConstraintPool[solveManifold.m_frictionIndex + 1].m_appliedImpulse;
			}
			//do a callback here?
		}
	}

	numPoolConstraints = m_tmpSolverNonContactConstraintPool.size();
	for (j = 0; j < numPoolConstraints; j++)
	{
		const b3SolverConstraint& solverConstr = m_tmpSolverNonContactConstraintPool[j];
		b3TypedConstraint* constr = (b3TypedConstraint*)solverConstr.m_originalContactPoint;
		b3JointFeedback* fb = constr->getJointFeedback();
		if (fb)
		{
			b3SolverBody* bodyA = &m_tmpSolverBodyPool[solverConstr.m_solverBodyIdA];
			b3SolverBody* bodyB = &m_tmpSolverBodyPool[solverConstr.m_solverBodyIdB];

			fb->m_appliedForceBodyA += solverConstr.m_contactNormal * solverConstr.m_appliedImpulse * bodyA->m_linearFactor / infoGlobal.m_timeStep;
			fb->m_appliedForceBodyB += -solverConstr.m_contactNormal * solverConstr.m_appliedImpulse * bodyB->m_linearFactor / infoGlobal.m_timeStep;
			fb->m_appliedTorqueBodyA += solverConstr.m_relpos1CrossNormal * bodyA->m_angularFactor * solverConstr.m_appliedImpulse / infoGlobal.m_timeStep;
			fb->m_appliedTorqueBodyB += -solverConstr.m_relpos1CrossNormal * bodyB->m_angularFactor * solverConstr.m_appliedImpulse / infoGlobal.m_timeStep;
		}

		constr->internalSetAppliedImpulse(solverConstr.m_appliedImpulse);
		if (b3Fabs(solverConstr.m_appliedImpulse) >= constr->getBreakingImpulseThreshold())
		{
			constr->setEnabled(false);
		}
	}

	{
		B3_PROFILE("write back velocities and transforms");
		for (i = 0; i < m_tmpSolverBodyPool.size(); i++)
		{
			int bodyIndex = m_tmpSolverBodyPool[i].m_originalBodyIndex;
			//b3Assert(i==bodyIndex);

			b3RigidBodyData* body = &bodies[bodyIndex];
			if (body->m_invMass)
			{
				if (infoGlobal.m_splitImpulse)
					m_tmpSolverBodyPool[i].writebackVelocityAndTransform(infoGlobal.m_timeStep, infoGlobal.m_splitImpulseTurnErp);
				else
					m_tmpSolverBodyPool[i].writebackVelocity();

				if (m_usePgs)
				{
					body->m_linVel = m_tmpSolverBodyPool[i].m_linearVelocity;
					body->m_angVel = m_tmpSolverBodyPool[i].m_angularVelocity;
				}
				else
				{
					b3Scalar factor = 1.f / b3Scalar(m_bodyCount[bodyIndex]);

					b3Vector3 deltaLinVel = m_deltaLinearVelocities[bodyIndex] * factor;
					b3Vector3 deltaAngVel = m_deltaAngularVelocities[bodyIndex] * factor;
					//printf("body %d\n",bodyIndex);
					//printf("deltaLinVel = %f,%f,%f\n",deltaLinVel.getX(),deltaLinVel.getY(),deltaLinVel.getZ());
					//printf("deltaAngVel = %f,%f,%f\n",deltaAngVel.getX(),deltaAngVel.getY(),deltaAngVel.getZ());

					body->m_linVel += deltaLinVel;
					body->m_angVel += deltaAngVel;
				}

				if (infoGlobal.m_splitImpulse)
				{
					body->m_pos = m_tmpSolverBodyPool[i].m_worldTransform.getOrigin();
					b3Quaternion orn;
					orn = m_tmpSolverBodyPool[i].m_worldTransform.getRotation();
					body->m_quat = orn;
				}
			}
		}
	}

	m_tmpSolverContactConstraintPool.resizeNoInitialize(0);
	m_tmpSolverNonContactConstraintPool.resizeNoInitialize(0);
	m_tmpSolverContactFrictionConstraintPool.resizeNoInitialize(0);
	m_tmpSolverContactRollingFrictionConstraintPool.resizeNoInitialize(0);

	m_tmpSolverBodyPool.resizeNoInitialize(0);
	return 0.f;
}

void b3PgsJacobiSolver::reset()
{
	m_btSeed2 = 0;
}