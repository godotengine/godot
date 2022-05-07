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

//#define COMPUTE_IMPULSE_DENOM 1
#ifdef BT_DEBUG
#	define BT_ADDITIONAL_DEBUG
#endif

//It is not necessary (redundant) to refresh contact manifolds, this refresh has been moved to the collision algorithms.

#include "btSequentialImpulseConstraintSolver.h"
#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"

#include "LinearMath/btIDebugDraw.h"
#include "LinearMath/btCpuFeatureUtility.h"

//#include "btJacobianEntry.h"
#include "LinearMath/btMinMax.h"
#include "BulletDynamics/ConstraintSolver/btTypedConstraint.h"
#include <new>
#include "LinearMath/btStackAlloc.h"
#include "LinearMath/btQuickprof.h"
//#include "btSolverBody.h"
//#include "btSolverConstraint.h"
#include "LinearMath/btAlignedObjectArray.h"
#include <string.h>  //for memset

int gNumSplitImpulseRecoveries = 0;

#include "BulletDynamics/Dynamics/btRigidBody.h"

//#define VERBOSE_RESIDUAL_PRINTF 1
///This is the scalar reference implementation of solving a single constraint row, the innerloop of the Projected Gauss Seidel/Sequential Impulse constraint solver
///Below are optional SSE2 and SSE4/FMA3 versions. We assume most hardware has SSE2. For SSE4/FMA3 we perform a CPU feature check.
static btScalar gResolveSingleConstraintRowGeneric_scalar_reference(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	btScalar deltaImpulse = c.m_rhs - btScalar(c.m_appliedImpulse) * c.m_cfm;
	const btScalar deltaVel1Dotn = c.m_contactNormal1.dot(bodyA.internalGetDeltaLinearVelocity()) + c.m_relpos1CrossNormal.dot(bodyA.internalGetDeltaAngularVelocity());
	const btScalar deltaVel2Dotn = c.m_contactNormal2.dot(bodyB.internalGetDeltaLinearVelocity()) + c.m_relpos2CrossNormal.dot(bodyB.internalGetDeltaAngularVelocity());

	//	const btScalar delta_rel_vel	=	deltaVel1Dotn-deltaVel2Dotn;
	deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
	deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;

	const btScalar sum = btScalar(c.m_appliedImpulse) + deltaImpulse;
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

	bodyA.internalApplyImpulse(c.m_contactNormal1 * bodyA.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
	bodyB.internalApplyImpulse(c.m_contactNormal2 * bodyB.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);

	return deltaImpulse * (1. / c.m_jacDiagABInv);
}

static btScalar gResolveSingleConstraintRowLowerLimit_scalar_reference(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	btScalar deltaImpulse = c.m_rhs - btScalar(c.m_appliedImpulse) * c.m_cfm;
	const btScalar deltaVel1Dotn = c.m_contactNormal1.dot(bodyA.internalGetDeltaLinearVelocity()) + c.m_relpos1CrossNormal.dot(bodyA.internalGetDeltaAngularVelocity());
	const btScalar deltaVel2Dotn = c.m_contactNormal2.dot(bodyB.internalGetDeltaLinearVelocity()) + c.m_relpos2CrossNormal.dot(bodyB.internalGetDeltaAngularVelocity());

	deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
	deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;
	const btScalar sum = btScalar(c.m_appliedImpulse) + deltaImpulse;
	if (sum < c.m_lowerLimit)
	{
		deltaImpulse = c.m_lowerLimit - c.m_appliedImpulse;
		c.m_appliedImpulse = c.m_lowerLimit;
	}
	else
	{
		c.m_appliedImpulse = sum;
	}
	bodyA.internalApplyImpulse(c.m_contactNormal1 * bodyA.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
	bodyB.internalApplyImpulse(c.m_contactNormal2 * bodyB.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);

	return deltaImpulse * (1. / c.m_jacDiagABInv);
}

#ifdef USE_SIMD
#include <emmintrin.h>

#define btVecSplat(x, e) _mm_shuffle_ps(x, x, _MM_SHUFFLE(e, e, e, e))
static inline __m128 btSimdDot3(__m128 vec0, __m128 vec1)
{
	__m128 result = _mm_mul_ps(vec0, vec1);
	return _mm_add_ps(btVecSplat(result, 0), _mm_add_ps(btVecSplat(result, 1), btVecSplat(result, 2)));
}

#if defined(BT_ALLOW_SSE4)
#include <intrin.h>

#define USE_FMA 1
#define USE_FMA3_INSTEAD_FMA4 1
#define USE_SSE4_DOT 1

#define SSE4_DP(a, b) _mm_dp_ps(a, b, 0x7f)
#define SSE4_DP_FP(a, b) _mm_cvtss_f32(_mm_dp_ps(a, b, 0x7f))

#if USE_SSE4_DOT
#define DOT_PRODUCT(a, b) SSE4_DP(a, b)
#else
#define DOT_PRODUCT(a, b) btSimdDot3(a, b)
#endif

#if USE_FMA
#if USE_FMA3_INSTEAD_FMA4
// a*b + c
#define FMADD(a, b, c) _mm_fmadd_ps(a, b, c)
// -(a*b) + c
#define FMNADD(a, b, c) _mm_fnmadd_ps(a, b, c)
#else  // USE_FMA3
// a*b + c
#define FMADD(a, b, c) _mm_macc_ps(a, b, c)
// -(a*b) + c
#define FMNADD(a, b, c) _mm_nmacc_ps(a, b, c)
#endif
#else  // USE_FMA
// c + a*b
#define FMADD(a, b, c) _mm_add_ps(c, _mm_mul_ps(a, b))
// c - a*b
#define FMNADD(a, b, c) _mm_sub_ps(c, _mm_mul_ps(a, b))
#endif
#endif

// Project Gauss Seidel or the equivalent Sequential Impulse
static btScalar gResolveSingleConstraintRowGeneric_sse2(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	__m128 cpAppliedImp = _mm_set1_ps(c.m_appliedImpulse);
	__m128 lowerLimit1 = _mm_set1_ps(c.m_lowerLimit);
	__m128 upperLimit1 = _mm_set1_ps(c.m_upperLimit);
	btSimdScalar deltaImpulse = _mm_sub_ps(_mm_set1_ps(c.m_rhs), _mm_mul_ps(_mm_set1_ps(c.m_appliedImpulse), _mm_set1_ps(c.m_cfm)));
	__m128 deltaVel1Dotn = _mm_add_ps(btSimdDot3(c.m_contactNormal1.mVec128, bodyA.internalGetDeltaLinearVelocity().mVec128), btSimdDot3(c.m_relpos1CrossNormal.mVec128, bodyA.internalGetDeltaAngularVelocity().mVec128));
	__m128 deltaVel2Dotn = _mm_add_ps(btSimdDot3(c.m_contactNormal2.mVec128, bodyB.internalGetDeltaLinearVelocity().mVec128), btSimdDot3(c.m_relpos2CrossNormal.mVec128, bodyB.internalGetDeltaAngularVelocity().mVec128));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel1Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel2Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	btSimdScalar sum = _mm_add_ps(cpAppliedImp, deltaImpulse);
	btSimdScalar resultLowerLess, resultUpperLess;
	resultLowerLess = _mm_cmplt_ps(sum, lowerLimit1);
	resultUpperLess = _mm_cmplt_ps(sum, upperLimit1);
	__m128 lowMinApplied = _mm_sub_ps(lowerLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowMinApplied), _mm_andnot_ps(resultLowerLess, deltaImpulse));
	c.m_appliedImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowerLimit1), _mm_andnot_ps(resultLowerLess, sum));
	__m128 upperMinApplied = _mm_sub_ps(upperLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultUpperLess, deltaImpulse), _mm_andnot_ps(resultUpperLess, upperMinApplied));
	c.m_appliedImpulse = _mm_or_ps(_mm_and_ps(resultUpperLess, c.m_appliedImpulse), _mm_andnot_ps(resultUpperLess, upperLimit1));
	__m128 linearComponentA = _mm_mul_ps(c.m_contactNormal1.mVec128, bodyA.internalGetInvMass().mVec128);
	__m128 linearComponentB = _mm_mul_ps((c.m_contactNormal2).mVec128, bodyB.internalGetInvMass().mVec128);
	__m128 impulseMagnitude = deltaImpulse;
	bodyA.internalGetDeltaLinearVelocity().mVec128 = _mm_add_ps(bodyA.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentA, impulseMagnitude));
	bodyA.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(bodyA.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentA.mVec128, impulseMagnitude));
	bodyB.internalGetDeltaLinearVelocity().mVec128 = _mm_add_ps(bodyB.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentB, impulseMagnitude));
	bodyB.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(bodyB.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentB.mVec128, impulseMagnitude));
	return deltaImpulse.m_floats[0] / c.m_jacDiagABInv;
}

// Enhanced version of gResolveSingleConstraintRowGeneric_sse2 with SSE4.1 and FMA3
static btScalar gResolveSingleConstraintRowGeneric_sse4_1_fma3(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
#if defined(BT_ALLOW_SSE4)
	__m128 tmp = _mm_set_ps1(c.m_jacDiagABInv);
	__m128 deltaImpulse = _mm_set_ps1(c.m_rhs - btScalar(c.m_appliedImpulse) * c.m_cfm);
	const __m128 lowerLimit = _mm_set_ps1(c.m_lowerLimit);
	const __m128 upperLimit = _mm_set_ps1(c.m_upperLimit);
	const __m128 deltaVel1Dotn = _mm_add_ps(DOT_PRODUCT(c.m_contactNormal1.mVec128, bodyA.internalGetDeltaLinearVelocity().mVec128), DOT_PRODUCT(c.m_relpos1CrossNormal.mVec128, bodyA.internalGetDeltaAngularVelocity().mVec128));
	const __m128 deltaVel2Dotn = _mm_add_ps(DOT_PRODUCT(c.m_contactNormal2.mVec128, bodyB.internalGetDeltaLinearVelocity().mVec128), DOT_PRODUCT(c.m_relpos2CrossNormal.mVec128, bodyB.internalGetDeltaAngularVelocity().mVec128));
	deltaImpulse = FMNADD(deltaVel1Dotn, tmp, deltaImpulse);
	deltaImpulse = FMNADD(deltaVel2Dotn, tmp, deltaImpulse);
	tmp = _mm_add_ps(c.m_appliedImpulse, deltaImpulse);  // sum
	const __m128 maskLower = _mm_cmpgt_ps(tmp, lowerLimit);
	const __m128 maskUpper = _mm_cmpgt_ps(upperLimit, tmp);
	deltaImpulse = _mm_blendv_ps(_mm_sub_ps(lowerLimit, c.m_appliedImpulse), _mm_blendv_ps(_mm_sub_ps(upperLimit, c.m_appliedImpulse), deltaImpulse, maskUpper), maskLower);
	c.m_appliedImpulse = _mm_blendv_ps(lowerLimit, _mm_blendv_ps(upperLimit, tmp, maskUpper), maskLower);
	bodyA.internalGetDeltaLinearVelocity().mVec128 = FMADD(_mm_mul_ps(c.m_contactNormal1.mVec128, bodyA.internalGetInvMass().mVec128), deltaImpulse, bodyA.internalGetDeltaLinearVelocity().mVec128);
	bodyA.internalGetDeltaAngularVelocity().mVec128 = FMADD(c.m_angularComponentA.mVec128, deltaImpulse, bodyA.internalGetDeltaAngularVelocity().mVec128);
	bodyB.internalGetDeltaLinearVelocity().mVec128 = FMADD(_mm_mul_ps(c.m_contactNormal2.mVec128, bodyB.internalGetInvMass().mVec128), deltaImpulse, bodyB.internalGetDeltaLinearVelocity().mVec128);
	bodyB.internalGetDeltaAngularVelocity().mVec128 = FMADD(c.m_angularComponentB.mVec128, deltaImpulse, bodyB.internalGetDeltaAngularVelocity().mVec128);
	btSimdScalar deltaImp = deltaImpulse;
	return deltaImp.m_floats[0] * (1. / c.m_jacDiagABInv);
#else
	return gResolveSingleConstraintRowGeneric_sse2(bodyA, bodyB, c);
#endif
}

static btScalar gResolveSingleConstraintRowLowerLimit_sse2(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	__m128 cpAppliedImp = _mm_set1_ps(c.m_appliedImpulse);
	__m128 lowerLimit1 = _mm_set1_ps(c.m_lowerLimit);
	__m128 upperLimit1 = _mm_set1_ps(c.m_upperLimit);
	btSimdScalar deltaImpulse = _mm_sub_ps(_mm_set1_ps(c.m_rhs), _mm_mul_ps(_mm_set1_ps(c.m_appliedImpulse), _mm_set1_ps(c.m_cfm)));
	__m128 deltaVel1Dotn = _mm_add_ps(btSimdDot3(c.m_contactNormal1.mVec128, bodyA.internalGetDeltaLinearVelocity().mVec128), btSimdDot3(c.m_relpos1CrossNormal.mVec128, bodyA.internalGetDeltaAngularVelocity().mVec128));
	__m128 deltaVel2Dotn = _mm_add_ps(btSimdDot3(c.m_contactNormal2.mVec128, bodyB.internalGetDeltaLinearVelocity().mVec128), btSimdDot3(c.m_relpos2CrossNormal.mVec128, bodyB.internalGetDeltaAngularVelocity().mVec128));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel1Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel2Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	btSimdScalar sum = _mm_add_ps(cpAppliedImp, deltaImpulse);
	btSimdScalar resultLowerLess, resultUpperLess;
	resultLowerLess = _mm_cmplt_ps(sum, lowerLimit1);
	resultUpperLess = _mm_cmplt_ps(sum, upperLimit1);
	__m128 lowMinApplied = _mm_sub_ps(lowerLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowMinApplied), _mm_andnot_ps(resultLowerLess, deltaImpulse));
	c.m_appliedImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowerLimit1), _mm_andnot_ps(resultLowerLess, sum));
	__m128 linearComponentA = _mm_mul_ps(c.m_contactNormal1.mVec128, bodyA.internalGetInvMass().mVec128);
	__m128 linearComponentB = _mm_mul_ps(c.m_contactNormal2.mVec128, bodyB.internalGetInvMass().mVec128);
	__m128 impulseMagnitude = deltaImpulse;
	bodyA.internalGetDeltaLinearVelocity().mVec128 = _mm_add_ps(bodyA.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentA, impulseMagnitude));
	bodyA.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(bodyA.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentA.mVec128, impulseMagnitude));
	bodyB.internalGetDeltaLinearVelocity().mVec128 = _mm_add_ps(bodyB.internalGetDeltaLinearVelocity().mVec128, _mm_mul_ps(linearComponentB, impulseMagnitude));
	bodyB.internalGetDeltaAngularVelocity().mVec128 = _mm_add_ps(bodyB.internalGetDeltaAngularVelocity().mVec128, _mm_mul_ps(c.m_angularComponentB.mVec128, impulseMagnitude));
	return deltaImpulse.m_floats[0] / c.m_jacDiagABInv;
}

// Enhanced version of gResolveSingleConstraintRowGeneric_sse2 with SSE4.1 and FMA3
static btScalar gResolveSingleConstraintRowLowerLimit_sse4_1_fma3(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
#ifdef BT_ALLOW_SSE4
	__m128 tmp = _mm_set_ps1(c.m_jacDiagABInv);
	__m128 deltaImpulse = _mm_set_ps1(c.m_rhs - btScalar(c.m_appliedImpulse) * c.m_cfm);
	const __m128 lowerLimit = _mm_set_ps1(c.m_lowerLimit);
	const __m128 deltaVel1Dotn = _mm_add_ps(DOT_PRODUCT(c.m_contactNormal1.mVec128, bodyA.internalGetDeltaLinearVelocity().mVec128), DOT_PRODUCT(c.m_relpos1CrossNormal.mVec128, bodyA.internalGetDeltaAngularVelocity().mVec128));
	const __m128 deltaVel2Dotn = _mm_add_ps(DOT_PRODUCT(c.m_contactNormal2.mVec128, bodyB.internalGetDeltaLinearVelocity().mVec128), DOT_PRODUCT(c.m_relpos2CrossNormal.mVec128, bodyB.internalGetDeltaAngularVelocity().mVec128));
	deltaImpulse = FMNADD(deltaVel1Dotn, tmp, deltaImpulse);
	deltaImpulse = FMNADD(deltaVel2Dotn, tmp, deltaImpulse);
	tmp = _mm_add_ps(c.m_appliedImpulse, deltaImpulse);
	const __m128 mask = _mm_cmpgt_ps(tmp, lowerLimit);
	deltaImpulse = _mm_blendv_ps(_mm_sub_ps(lowerLimit, c.m_appliedImpulse), deltaImpulse, mask);
	c.m_appliedImpulse = _mm_blendv_ps(lowerLimit, tmp, mask);
	bodyA.internalGetDeltaLinearVelocity().mVec128 = FMADD(_mm_mul_ps(c.m_contactNormal1.mVec128, bodyA.internalGetInvMass().mVec128), deltaImpulse, bodyA.internalGetDeltaLinearVelocity().mVec128);
	bodyA.internalGetDeltaAngularVelocity().mVec128 = FMADD(c.m_angularComponentA.mVec128, deltaImpulse, bodyA.internalGetDeltaAngularVelocity().mVec128);
	bodyB.internalGetDeltaLinearVelocity().mVec128 = FMADD(_mm_mul_ps(c.m_contactNormal2.mVec128, bodyB.internalGetInvMass().mVec128), deltaImpulse, bodyB.internalGetDeltaLinearVelocity().mVec128);
	bodyB.internalGetDeltaAngularVelocity().mVec128 = FMADD(c.m_angularComponentB.mVec128, deltaImpulse, bodyB.internalGetDeltaAngularVelocity().mVec128);
	btSimdScalar deltaImp = deltaImpulse;
	return deltaImp.m_floats[0] * (1. / c.m_jacDiagABInv);
#else
	return gResolveSingleConstraintRowLowerLimit_sse2(bodyA, bodyB, c);
#endif  //BT_ALLOW_SSE4
}

#endif  //USE_SIMD

btScalar btSequentialImpulseConstraintSolver::resolveSingleConstraintRowGenericSIMD(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	return m_resolveSingleConstraintRowGeneric(bodyA, bodyB, c);
}

// Project Gauss Seidel or the equivalent Sequential Impulse
btScalar btSequentialImpulseConstraintSolver::resolveSingleConstraintRowGeneric(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	return m_resolveSingleConstraintRowGeneric(bodyA, bodyB, c);
}

btScalar btSequentialImpulseConstraintSolver::resolveSingleConstraintRowLowerLimitSIMD(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	return m_resolveSingleConstraintRowLowerLimit(bodyA, bodyB, c);
}

btScalar btSequentialImpulseConstraintSolver::resolveSingleConstraintRowLowerLimit(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
	return m_resolveSingleConstraintRowLowerLimit(bodyA, bodyB, c);
}

static btScalar gResolveSplitPenetrationImpulse_scalar_reference(
	btSolverBody& bodyA,
	btSolverBody& bodyB,
	const btSolverConstraint& c)
{
	btScalar deltaImpulse = 0.f;

	if (c.m_rhsPenetration)
	{
		gNumSplitImpulseRecoveries++;
		deltaImpulse = c.m_rhsPenetration - btScalar(c.m_appliedPushImpulse) * c.m_cfm;
		const btScalar deltaVel1Dotn = c.m_contactNormal1.dot(bodyA.internalGetPushVelocity()) + c.m_relpos1CrossNormal.dot(bodyA.internalGetTurnVelocity());
		const btScalar deltaVel2Dotn = c.m_contactNormal2.dot(bodyB.internalGetPushVelocity()) + c.m_relpos2CrossNormal.dot(bodyB.internalGetTurnVelocity());

		deltaImpulse -= deltaVel1Dotn * c.m_jacDiagABInv;
		deltaImpulse -= deltaVel2Dotn * c.m_jacDiagABInv;
		const btScalar sum = btScalar(c.m_appliedPushImpulse) + deltaImpulse;
		if (sum < c.m_lowerLimit)
		{
			deltaImpulse = c.m_lowerLimit - c.m_appliedPushImpulse;
			c.m_appliedPushImpulse = c.m_lowerLimit;
		}
		else
		{
			c.m_appliedPushImpulse = sum;
		}
		bodyA.internalApplyPushImpulse(c.m_contactNormal1 * bodyA.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
		bodyB.internalApplyPushImpulse(c.m_contactNormal2 * bodyB.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
	}
	return deltaImpulse * (1. / c.m_jacDiagABInv);
}

static btScalar gResolveSplitPenetrationImpulse_sse2(btSolverBody& bodyA, btSolverBody& bodyB, const btSolverConstraint& c)
{
#ifdef USE_SIMD
	if (!c.m_rhsPenetration)
		return 0.f;

	gNumSplitImpulseRecoveries++;

	__m128 cpAppliedImp = _mm_set1_ps(c.m_appliedPushImpulse);
	__m128 lowerLimit1 = _mm_set1_ps(c.m_lowerLimit);
	__m128 upperLimit1 = _mm_set1_ps(c.m_upperLimit);
	__m128 deltaImpulse = _mm_sub_ps(_mm_set1_ps(c.m_rhsPenetration), _mm_mul_ps(_mm_set1_ps(c.m_appliedPushImpulse), _mm_set1_ps(c.m_cfm)));
	__m128 deltaVel1Dotn = _mm_add_ps(btSimdDot3(c.m_contactNormal1.mVec128, bodyA.internalGetPushVelocity().mVec128), btSimdDot3(c.m_relpos1CrossNormal.mVec128, bodyA.internalGetTurnVelocity().mVec128));
	__m128 deltaVel2Dotn = _mm_add_ps(btSimdDot3(c.m_contactNormal2.mVec128, bodyB.internalGetPushVelocity().mVec128), btSimdDot3(c.m_relpos2CrossNormal.mVec128, bodyB.internalGetTurnVelocity().mVec128));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel1Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	deltaImpulse = _mm_sub_ps(deltaImpulse, _mm_mul_ps(deltaVel2Dotn, _mm_set1_ps(c.m_jacDiagABInv)));
	btSimdScalar sum = _mm_add_ps(cpAppliedImp, deltaImpulse);
	btSimdScalar resultLowerLess, resultUpperLess;
	resultLowerLess = _mm_cmplt_ps(sum, lowerLimit1);
	resultUpperLess = _mm_cmplt_ps(sum, upperLimit1);
	__m128 lowMinApplied = _mm_sub_ps(lowerLimit1, cpAppliedImp);
	deltaImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowMinApplied), _mm_andnot_ps(resultLowerLess, deltaImpulse));
	c.m_appliedPushImpulse = _mm_or_ps(_mm_and_ps(resultLowerLess, lowerLimit1), _mm_andnot_ps(resultLowerLess, sum));
	__m128 linearComponentA = _mm_mul_ps(c.m_contactNormal1.mVec128, bodyA.internalGetInvMass().mVec128);
	__m128 linearComponentB = _mm_mul_ps(c.m_contactNormal2.mVec128, bodyB.internalGetInvMass().mVec128);
	__m128 impulseMagnitude = deltaImpulse;
	bodyA.internalGetPushVelocity().mVec128 = _mm_add_ps(bodyA.internalGetPushVelocity().mVec128, _mm_mul_ps(linearComponentA, impulseMagnitude));
	bodyA.internalGetTurnVelocity().mVec128 = _mm_add_ps(bodyA.internalGetTurnVelocity().mVec128, _mm_mul_ps(c.m_angularComponentA.mVec128, impulseMagnitude));
	bodyB.internalGetPushVelocity().mVec128 = _mm_add_ps(bodyB.internalGetPushVelocity().mVec128, _mm_mul_ps(linearComponentB, impulseMagnitude));
	bodyB.internalGetTurnVelocity().mVec128 = _mm_add_ps(bodyB.internalGetTurnVelocity().mVec128, _mm_mul_ps(c.m_angularComponentB.mVec128, impulseMagnitude));
	btSimdScalar deltaImp = deltaImpulse;
	return deltaImp.m_floats[0] * (1. / c.m_jacDiagABInv);
#else
	return gResolveSplitPenetrationImpulse_scalar_reference(bodyA, bodyB, c);
#endif
}

btSequentialImpulseConstraintSolver::btSequentialImpulseConstraintSolver()
{
	m_btSeed2 = 0;
	m_cachedSolverMode = 0;
	setupSolverFunctions(false);
}

void btSequentialImpulseConstraintSolver::setupSolverFunctions(bool useSimd)
{
	m_resolveSingleConstraintRowGeneric = gResolveSingleConstraintRowGeneric_scalar_reference;
	m_resolveSingleConstraintRowLowerLimit = gResolveSingleConstraintRowLowerLimit_scalar_reference;
	m_resolveSplitPenetrationImpulse = gResolveSplitPenetrationImpulse_scalar_reference;

	if (useSimd)
	{
#ifdef USE_SIMD
		m_resolveSingleConstraintRowGeneric = gResolveSingleConstraintRowGeneric_sse2;
		m_resolveSingleConstraintRowLowerLimit = gResolveSingleConstraintRowLowerLimit_sse2;
		m_resolveSplitPenetrationImpulse = gResolveSplitPenetrationImpulse_sse2;

#ifdef BT_ALLOW_SSE4
		int cpuFeatures = btCpuFeatureUtility::getCpuFeatures();
		if ((cpuFeatures & btCpuFeatureUtility::CPU_FEATURE_FMA3) && (cpuFeatures & btCpuFeatureUtility::CPU_FEATURE_SSE4_1))
		{
			m_resolveSingleConstraintRowGeneric = gResolveSingleConstraintRowGeneric_sse4_1_fma3;
			m_resolveSingleConstraintRowLowerLimit = gResolveSingleConstraintRowLowerLimit_sse4_1_fma3;
		}
#endif  //BT_ALLOW_SSE4
#endif  //USE_SIMD
	}
}

btSequentialImpulseConstraintSolver::~btSequentialImpulseConstraintSolver()
{
}

btSingleConstraintRowSolver btSequentialImpulseConstraintSolver::getScalarConstraintRowSolverGeneric()
{
	return gResolveSingleConstraintRowGeneric_scalar_reference;
}

btSingleConstraintRowSolver btSequentialImpulseConstraintSolver::getScalarConstraintRowSolverLowerLimit()
{
	return gResolveSingleConstraintRowLowerLimit_scalar_reference;
}

#ifdef USE_SIMD
btSingleConstraintRowSolver btSequentialImpulseConstraintSolver::getSSE2ConstraintRowSolverGeneric()
{
	return gResolveSingleConstraintRowGeneric_sse2;
}
btSingleConstraintRowSolver btSequentialImpulseConstraintSolver::getSSE2ConstraintRowSolverLowerLimit()
{
	return gResolveSingleConstraintRowLowerLimit_sse2;
}
#ifdef BT_ALLOW_SSE4
btSingleConstraintRowSolver btSequentialImpulseConstraintSolver::getSSE4_1ConstraintRowSolverGeneric()
{
	return gResolveSingleConstraintRowGeneric_sse4_1_fma3;
}
btSingleConstraintRowSolver btSequentialImpulseConstraintSolver::getSSE4_1ConstraintRowSolverLowerLimit()
{
	return gResolveSingleConstraintRowLowerLimit_sse4_1_fma3;
}
#endif  //BT_ALLOW_SSE4
#endif  //USE_SIMD

unsigned long btSequentialImpulseConstraintSolver::btRand2()
{
	m_btSeed2 = (1664525L * m_btSeed2 + 1013904223L) & 0xffffffff;
	return m_btSeed2;
}

//See ODE: adam's all-int straightforward(?) dRandInt (0..n-1)
int btSequentialImpulseConstraintSolver::btRandInt2(int n)
{
	// seems good; xor-fold and modulus
	const unsigned long un = static_cast<unsigned long>(n);
	unsigned long r = btRand2();

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

void btSequentialImpulseConstraintSolver::initSolverBody(btSolverBody* solverBody, btCollisionObject* collisionObject, btScalar timeStep)
{
	btRigidBody* rb = collisionObject ? btRigidBody::upcast(collisionObject) : 0;

	solverBody->internalGetDeltaLinearVelocity().setValue(0.f, 0.f, 0.f);
	solverBody->internalGetDeltaAngularVelocity().setValue(0.f, 0.f, 0.f);
	solverBody->internalGetPushVelocity().setValue(0.f, 0.f, 0.f);
	solverBody->internalGetTurnVelocity().setValue(0.f, 0.f, 0.f);

	if (rb)
	{
		solverBody->m_worldTransform = rb->getWorldTransform();
		solverBody->internalSetInvMass(btVector3(rb->getInvMass(), rb->getInvMass(), rb->getInvMass()) * rb->getLinearFactor());
		solverBody->m_originalBody = rb;
		solverBody->m_angularFactor = rb->getAngularFactor();
		solverBody->m_linearFactor = rb->getLinearFactor();
		solverBody->m_linearVelocity = rb->getLinearVelocity();
		solverBody->m_angularVelocity = rb->getAngularVelocity();
		solverBody->m_externalForceImpulse = rb->getTotalForce() * rb->getInvMass() * timeStep;
		solverBody->m_externalTorqueImpulse = rb->getTotalTorque() * rb->getInvInertiaTensorWorld() * timeStep;
	}
	else
		{
		solverBody->m_worldTransform.setIdentity();
		solverBody->internalSetInvMass(btVector3(0, 0, 0));
		solverBody->m_originalBody = 0;
		solverBody->m_angularFactor.setValue(1, 1, 1);
		solverBody->m_linearFactor.setValue(1, 1, 1);
		solverBody->m_linearVelocity.setValue(0, 0, 0);
		solverBody->m_angularVelocity.setValue(0, 0, 0);
		solverBody->m_externalForceImpulse.setValue(0, 0, 0);
		solverBody->m_externalTorqueImpulse.setValue(0, 0, 0);
					}
				}

btScalar btSequentialImpulseConstraintSolver::restitutionCurve(btScalar rel_vel, btScalar restitution, btScalar velocityThreshold)
{
	//printf("rel_vel =%f\n", rel_vel);
	if (btFabs(rel_vel) < velocityThreshold)
		return 0.;

	btScalar rest = restitution * -rel_vel;
	return rest;
}

void btSequentialImpulseConstraintSolver::applyAnisotropicFriction(btCollisionObject* colObj, btVector3& frictionDirection, int frictionMode)
{
	if (colObj && colObj->hasAnisotropicFriction(frictionMode))
	{
		// transform to local coordinates
		btVector3 loc_lateral = frictionDirection * colObj->getWorldTransform().getBasis();
		const btVector3& friction_scaling = colObj->getAnisotropicFriction();
		//apply anisotropic friction
		loc_lateral *= friction_scaling;
		// ... and transform it back to global coordinates
		frictionDirection = colObj->getWorldTransform().getBasis() * loc_lateral;
	}
}

void btSequentialImpulseConstraintSolver::setupFrictionConstraint(btSolverConstraint& solverConstraint, const btVector3& normalAxis, int solverBodyIdA, int solverBodyIdB, btManifoldPoint& cp, const btVector3& rel_pos1, const btVector3& rel_pos2, btCollisionObject* colObj0, btCollisionObject* colObj1, btScalar relaxation, const btContactSolverInfo& infoGlobal, btScalar desiredVelocity, btScalar cfmSlip)
{
	btSolverBody& solverBodyA = m_tmpSolverBodyPool[solverBodyIdA];
	btSolverBody& solverBodyB = m_tmpSolverBodyPool[solverBodyIdB];

	btRigidBody* body0 = m_tmpSolverBodyPool[solverBodyIdA].m_originalBody;
	btRigidBody* bodyA = m_tmpSolverBodyPool[solverBodyIdB].m_originalBody;

	solverConstraint.m_solverBodyIdA = solverBodyIdA;
	solverConstraint.m_solverBodyIdB = solverBodyIdB;

	solverConstraint.m_friction = cp.m_combinedFriction;
	solverConstraint.m_originalContactPoint = 0;

	solverConstraint.m_appliedImpulse = 0.f;
	solverConstraint.m_appliedPushImpulse = 0.f;

	if (body0)
	{
		solverConstraint.m_contactNormal1 = normalAxis;
		btVector3 ftorqueAxis1 = rel_pos1.cross(solverConstraint.m_contactNormal1);
		solverConstraint.m_relpos1CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentA = body0->getInvInertiaTensorWorld() * ftorqueAxis1 * body0->getAngularFactor();
	}
	else
	{
		solverConstraint.m_contactNormal1.setZero();
		solverConstraint.m_relpos1CrossNormal.setZero();
		solverConstraint.m_angularComponentA.setZero();
	}

	if (bodyA)
	{
		solverConstraint.m_contactNormal2 = -normalAxis;
		btVector3 ftorqueAxis1 = rel_pos2.cross(solverConstraint.m_contactNormal2);
		solverConstraint.m_relpos2CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentB = bodyA->getInvInertiaTensorWorld() * ftorqueAxis1 * bodyA->getAngularFactor();
	}
	else
	{
		solverConstraint.m_contactNormal2.setZero();
		solverConstraint.m_relpos2CrossNormal.setZero();
		solverConstraint.m_angularComponentB.setZero();
	}

	{
		btVector3 vec;
		btScalar denom0 = 0.f;
		btScalar denom1 = 0.f;
		if (body0)
		{
			vec = (solverConstraint.m_angularComponentA).cross(rel_pos1);
			denom0 = body0->getInvMass() + normalAxis.dot(vec);
		}
		if (bodyA)
		{
			vec = (-solverConstraint.m_angularComponentB).cross(rel_pos2);
			denom1 = bodyA->getInvMass() + normalAxis.dot(vec);
		}
		btScalar denom = relaxation / (denom0 + denom1);
		solverConstraint.m_jacDiagABInv = denom;
	}

	{
		btScalar rel_vel;
		btScalar vel1Dotn = solverConstraint.m_contactNormal1.dot(body0 ? solverBodyA.m_linearVelocity + solverBodyA.m_externalForceImpulse : btVector3(0, 0, 0)) + solverConstraint.m_relpos1CrossNormal.dot(body0 ? solverBodyA.m_angularVelocity : btVector3(0, 0, 0));
		btScalar vel2Dotn = solverConstraint.m_contactNormal2.dot(bodyA ? solverBodyB.m_linearVelocity + solverBodyB.m_externalForceImpulse : btVector3(0, 0, 0)) + solverConstraint.m_relpos2CrossNormal.dot(bodyA ? solverBodyB.m_angularVelocity : btVector3(0, 0, 0));

		rel_vel = vel1Dotn + vel2Dotn;

		//		btScalar positionalError = 0.f;

		btScalar velocityError = desiredVelocity - rel_vel;
		btScalar velocityImpulse = velocityError * solverConstraint.m_jacDiagABInv;

		btScalar penetrationImpulse = btScalar(0);

		if (cp.m_contactPointFlags & BT_CONTACT_FLAG_FRICTION_ANCHOR)
		{
			btScalar distance = (cp.getPositionWorldOnA() - cp.getPositionWorldOnB()).dot(normalAxis);
			btScalar positionalError = -distance * infoGlobal.m_frictionERP / infoGlobal.m_timeStep;
			penetrationImpulse = positionalError * solverConstraint.m_jacDiagABInv;
		}

		solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;
		solverConstraint.m_rhsPenetration = 0.f;
		solverConstraint.m_cfm = cfmSlip;
		solverConstraint.m_lowerLimit = -solverConstraint.m_friction;
		solverConstraint.m_upperLimit = solverConstraint.m_friction;
	}
}

btSolverConstraint& btSequentialImpulseConstraintSolver::addFrictionConstraint(const btVector3& normalAxis, int solverBodyIdA, int solverBodyIdB, int frictionIndex, btManifoldPoint& cp, const btVector3& rel_pos1, const btVector3& rel_pos2, btCollisionObject* colObj0, btCollisionObject* colObj1, btScalar relaxation, const btContactSolverInfo& infoGlobal, btScalar desiredVelocity, btScalar cfmSlip)
{
	btSolverConstraint& solverConstraint = m_tmpSolverContactFrictionConstraintPool.expandNonInitializing();
	solverConstraint.m_frictionIndex = frictionIndex;
	setupFrictionConstraint(solverConstraint, normalAxis, solverBodyIdA, solverBodyIdB, cp, rel_pos1, rel_pos2,
		colObj0, colObj1, relaxation, infoGlobal, desiredVelocity, cfmSlip);
	return solverConstraint;
}

void btSequentialImpulseConstraintSolver::setupTorsionalFrictionConstraint(btSolverConstraint& solverConstraint, const btVector3& normalAxis1, int solverBodyIdA, int solverBodyIdB,
	btManifoldPoint& cp, btScalar combinedTorsionalFriction, const btVector3& rel_pos1, const btVector3& rel_pos2,
	btCollisionObject* colObj0, btCollisionObject* colObj1, btScalar relaxation,
	btScalar desiredVelocity, btScalar cfmSlip)

{
	btVector3 normalAxis(0, 0, 0);

	solverConstraint.m_contactNormal1 = normalAxis;
	solverConstraint.m_contactNormal2 = -normalAxis;
	btSolverBody& solverBodyA = m_tmpSolverBodyPool[solverBodyIdA];
	btSolverBody& solverBodyB = m_tmpSolverBodyPool[solverBodyIdB];

	btRigidBody* body0 = m_tmpSolverBodyPool[solverBodyIdA].m_originalBody;
	btRigidBody* bodyA = m_tmpSolverBodyPool[solverBodyIdB].m_originalBody;

	solverConstraint.m_solverBodyIdA = solverBodyIdA;
	solverConstraint.m_solverBodyIdB = solverBodyIdB;

	solverConstraint.m_friction = combinedTorsionalFriction;
	solverConstraint.m_originalContactPoint = 0;

	solverConstraint.m_appliedImpulse = 0.f;
	solverConstraint.m_appliedPushImpulse = 0.f;

	{
		btVector3 ftorqueAxis1 = -normalAxis1;
		solverConstraint.m_relpos1CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentA = body0 ? body0->getInvInertiaTensorWorld() * ftorqueAxis1 * body0->getAngularFactor() : btVector3(0, 0, 0);
	}
	{
		btVector3 ftorqueAxis1 = normalAxis1;
		solverConstraint.m_relpos2CrossNormal = ftorqueAxis1;
		solverConstraint.m_angularComponentB = bodyA ? bodyA->getInvInertiaTensorWorld() * ftorqueAxis1 * bodyA->getAngularFactor() : btVector3(0, 0, 0);
	}

	{
		btVector3 iMJaA = body0 ? body0->getInvInertiaTensorWorld() * solverConstraint.m_relpos1CrossNormal : btVector3(0, 0, 0);
		btVector3 iMJaB = bodyA ? bodyA->getInvInertiaTensorWorld() * solverConstraint.m_relpos2CrossNormal : btVector3(0, 0, 0);
		btScalar sum = 0;
		sum += iMJaA.dot(solverConstraint.m_relpos1CrossNormal);
		sum += iMJaB.dot(solverConstraint.m_relpos2CrossNormal);
		solverConstraint.m_jacDiagABInv = btScalar(1.) / sum;
	}

	{
		btScalar rel_vel;
		btScalar vel1Dotn = solverConstraint.m_contactNormal1.dot(body0 ? solverBodyA.m_linearVelocity + solverBodyA.m_externalForceImpulse : btVector3(0, 0, 0)) + solverConstraint.m_relpos1CrossNormal.dot(body0 ? solverBodyA.m_angularVelocity : btVector3(0, 0, 0));
		btScalar vel2Dotn = solverConstraint.m_contactNormal2.dot(bodyA ? solverBodyB.m_linearVelocity + solverBodyB.m_externalForceImpulse : btVector3(0, 0, 0)) + solverConstraint.m_relpos2CrossNormal.dot(bodyA ? solverBodyB.m_angularVelocity : btVector3(0, 0, 0));

		rel_vel = vel1Dotn + vel2Dotn;

		//		btScalar positionalError = 0.f;

		btSimdScalar velocityError = desiredVelocity - rel_vel;
		btSimdScalar velocityImpulse = velocityError * btSimdScalar(solverConstraint.m_jacDiagABInv);
		solverConstraint.m_rhs = velocityImpulse;
		solverConstraint.m_cfm = cfmSlip;
		solverConstraint.m_lowerLimit = -solverConstraint.m_friction;
		solverConstraint.m_upperLimit = solverConstraint.m_friction;
	}
}

btSolverConstraint& btSequentialImpulseConstraintSolver::addTorsionalFrictionConstraint(const btVector3& normalAxis, int solverBodyIdA, int solverBodyIdB, int frictionIndex, btManifoldPoint& cp, btScalar combinedTorsionalFriction, const btVector3& rel_pos1, const btVector3& rel_pos2, btCollisionObject* colObj0, btCollisionObject* colObj1, btScalar relaxation, btScalar desiredVelocity, btScalar cfmSlip)
{
	btSolverConstraint& solverConstraint = m_tmpSolverContactRollingFrictionConstraintPool.expandNonInitializing();
	solverConstraint.m_frictionIndex = frictionIndex;
	setupTorsionalFrictionConstraint(solverConstraint, normalAxis, solverBodyIdA, solverBodyIdB, cp, combinedTorsionalFriction, rel_pos1, rel_pos2,
		colObj0, colObj1, relaxation, desiredVelocity, cfmSlip);
	return solverConstraint;
}

int btSequentialImpulseConstraintSolver::getOrInitSolverBody(btCollisionObject& body, btScalar timeStep)
{
#if BT_THREADSAFE
	int solverBodyId = -1;
	const bool isRigidBodyType = btRigidBody::upcast(&body) != NULL;
	const bool isStaticOrKinematic = body.isStaticOrKinematicObject();
	const bool isKinematic = body.isKinematicObject();
	if (isRigidBodyType && !isStaticOrKinematic)
	{
		// dynamic body
		// Dynamic bodies can only be in one island, so it's safe to write to the companionId
		solverBodyId = body.getCompanionId();
		if (solverBodyId < 0)
		{
			solverBodyId = m_tmpSolverBodyPool.size();
			btSolverBody& solverBody = m_tmpSolverBodyPool.expand();
			initSolverBody(&solverBody, &body, timeStep);
			body.setCompanionId(solverBodyId);
		}
	}
	else if (isRigidBodyType && isKinematic)
	{
		//
		// NOTE: must test for kinematic before static because some kinematic objects also
		//   identify as "static"
		//
		// Kinematic bodies can be in multiple islands at once, so it is a
		// race condition to write to them, so we use an alternate method
		// to record the solverBodyId
		int uniqueId = body.getWorldArrayIndex();
		const int INVALID_SOLVER_BODY_ID = -1;
		if (uniqueId >= m_kinematicBodyUniqueIdToSolverBodyTable.size())
		{
			m_kinematicBodyUniqueIdToSolverBodyTable.resize(uniqueId + 1, INVALID_SOLVER_BODY_ID);
		}
		solverBodyId = m_kinematicBodyUniqueIdToSolverBodyTable[uniqueId];
		// if no table entry yet,
		if (solverBodyId == INVALID_SOLVER_BODY_ID)
		{
			// create a table entry for this body
			solverBodyId = m_tmpSolverBodyPool.size();
			btSolverBody& solverBody = m_tmpSolverBodyPool.expand();
			initSolverBody(&solverBody, &body, timeStep);
			m_kinematicBodyUniqueIdToSolverBodyTable[uniqueId] = solverBodyId;
		}
	}
	else
	{
		bool isMultiBodyType = (body.getInternalType() & btCollisionObject::CO_FEATHERSTONE_LINK);
		// Incorrectly set collision object flags can degrade performance in various ways.
		if (!isMultiBodyType)
		{
			btAssert(body.isStaticOrKinematicObject());
		}
		//it could be a multibody link collider
		// all fixed bodies (inf mass) get mapped to a single solver id
		if (m_fixedBodyId < 0)
		{
			m_fixedBodyId = m_tmpSolverBodyPool.size();
			btSolverBody& fixedBody = m_tmpSolverBodyPool.expand();
			initSolverBody(&fixedBody, 0, timeStep);
		}
		solverBodyId = m_fixedBodyId;
	}
	btAssert(solverBodyId >= 0 && solverBodyId < m_tmpSolverBodyPool.size());
	return solverBodyId;
#else   // BT_THREADSAFE

	int solverBodyIdA = -1;

	if (body.getCompanionId() >= 0)
	{
		//body has already been converted
		solverBodyIdA = body.getCompanionId();
		btAssert(solverBodyIdA < m_tmpSolverBodyPool.size());
	}
	else
	{
		btRigidBody* rb = btRigidBody::upcast(&body);
		//convert both active and kinematic objects (for their velocity)
		if (rb && (rb->getInvMass() || rb->isKinematicObject()))
		{
			solverBodyIdA = m_tmpSolverBodyPool.size();
			btSolverBody& solverBody = m_tmpSolverBodyPool.expand();
			initSolverBody(&solverBody, &body, timeStep);
			body.setCompanionId(solverBodyIdA);
		}
		else
		{
			if (m_fixedBodyId < 0)
			{
				m_fixedBodyId = m_tmpSolverBodyPool.size();
				btSolverBody& fixedBody = m_tmpSolverBodyPool.expand();
				initSolverBody(&fixedBody, 0, timeStep);
			}
			return m_fixedBodyId;
			//			return 0;//assume first one is a fixed solver body
		}
	}

	return solverBodyIdA;
#endif  // BT_THREADSAFE
}
#include <stdio.h>

void btSequentialImpulseConstraintSolver::setupContactConstraint(btSolverConstraint& solverConstraint,
	int solverBodyIdA, int solverBodyIdB,
	btManifoldPoint& cp, const btContactSolverInfo& infoGlobal,
	btScalar& relaxation,
	const btVector3& rel_pos1, const btVector3& rel_pos2)
{
	//	const btVector3& pos1 = cp.getPositionWorldOnA();
	//	const btVector3& pos2 = cp.getPositionWorldOnB();

	btSolverBody* bodyA = &m_tmpSolverBodyPool[solverBodyIdA];
	btSolverBody* bodyB = &m_tmpSolverBodyPool[solverBodyIdB];

	btRigidBody* rb0 = bodyA->m_originalBody;
	btRigidBody* rb1 = bodyB->m_originalBody;

	//			btVector3 rel_pos1 = pos1 - colObj0->getWorldTransform().getOrigin();
	//			btVector3 rel_pos2 = pos2 - colObj1->getWorldTransform().getOrigin();
	//rel_pos1 = pos1 - bodyA->getWorldTransform().getOrigin();
	//rel_pos2 = pos2 - bodyB->getWorldTransform().getOrigin();

	relaxation = infoGlobal.m_sor;
	btScalar invTimeStep = btScalar(1) / infoGlobal.m_timeStep;

	//cfm = 1 /       ( dt * kp + kd )
	//erp = dt * kp / ( dt * kp + kd )

	btScalar cfm = infoGlobal.m_globalCfm;
	btScalar erp = infoGlobal.m_erp2;

	if ((cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_CFM) || (cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_ERP))
	{
		if (cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_CFM)
			cfm = cp.m_contactCFM;
		if (cp.m_contactPointFlags & BT_CONTACT_FLAG_HAS_CONTACT_ERP)
			erp = cp.m_contactERP;
	}
	else
	{
		if (cp.m_contactPointFlags & BT_CONTACT_FLAG_CONTACT_STIFFNESS_DAMPING)
		{
			btScalar denom = (infoGlobal.m_timeStep * cp.m_combinedContactStiffness1 + cp.m_combinedContactDamping1);
			if (denom < SIMD_EPSILON)
			{
				denom = SIMD_EPSILON;
			}
			cfm = btScalar(1) / denom;
			erp = (infoGlobal.m_timeStep * cp.m_combinedContactStiffness1) / denom;
		}
	}

	cfm *= invTimeStep;

	btVector3 torqueAxis0 = rel_pos1.cross(cp.m_normalWorldOnB);
	solverConstraint.m_angularComponentA = rb0 ? rb0->getInvInertiaTensorWorld() * torqueAxis0 * rb0->getAngularFactor() : btVector3(0, 0, 0);
	btVector3 torqueAxis1 = rel_pos2.cross(cp.m_normalWorldOnB);
	solverConstraint.m_angularComponentB = rb1 ? rb1->getInvInertiaTensorWorld() * -torqueAxis1 * rb1->getAngularFactor() : btVector3(0, 0, 0);

	{
#ifdef COMPUTE_IMPULSE_DENOM
		btScalar denom0 = rb0->computeImpulseDenominator(pos1, cp.m_normalWorldOnB);
		btScalar denom1 = rb1->computeImpulseDenominator(pos2, cp.m_normalWorldOnB);
#else
		btVector3 vec;
		btScalar denom0 = 0.f;
		btScalar denom1 = 0.f;
		if (rb0)
		{
			vec = (solverConstraint.m_angularComponentA).cross(rel_pos1);
			denom0 = rb0->getInvMass() + cp.m_normalWorldOnB.dot(vec);
		}
		if (rb1)
		{
			vec = (-solverConstraint.m_angularComponentB).cross(rel_pos2);
			denom1 = rb1->getInvMass() + cp.m_normalWorldOnB.dot(vec);
		}
#endif  //COMPUTE_IMPULSE_DENOM

		btScalar denom = relaxation / (denom0 + denom1 + cfm);
		solverConstraint.m_jacDiagABInv = denom;
	}

	if (rb0)
	{
		solverConstraint.m_contactNormal1 = cp.m_normalWorldOnB;
		solverConstraint.m_relpos1CrossNormal = torqueAxis0;
	}
	else
	{
		solverConstraint.m_contactNormal1.setZero();
		solverConstraint.m_relpos1CrossNormal.setZero();
	}
	if (rb1)
	{
		solverConstraint.m_contactNormal2 = -cp.m_normalWorldOnB;
		solverConstraint.m_relpos2CrossNormal = -torqueAxis1;
	}
	else
	{
		solverConstraint.m_contactNormal2.setZero();
		solverConstraint.m_relpos2CrossNormal.setZero();
	}

	btScalar restitution = 0.f;
	btScalar penetration = cp.getDistance() + infoGlobal.m_linearSlop;

	{
		btVector3 vel1, vel2;

		vel1 = rb0 ? rb0->getVelocityInLocalPoint(rel_pos1) : btVector3(0, 0, 0);
		vel2 = rb1 ? rb1->getVelocityInLocalPoint(rel_pos2) : btVector3(0, 0, 0);

		//			btVector3 vel2 = rb1 ? rb1->getVelocityInLocalPoint(rel_pos2) : btVector3(0,0,0);
		btVector3 vel = vel1 - vel2;
		btScalar rel_vel = cp.m_normalWorldOnB.dot(vel);

		solverConstraint.m_friction = cp.m_combinedFriction;

		restitution = restitutionCurve(rel_vel, cp.m_combinedRestitution, infoGlobal.m_restitutionVelocityThreshold);
		if (restitution <= btScalar(0.))
		{
			restitution = 0.f;
		};
	}

	///warm starting (or zero if disabled)
	if (infoGlobal.m_solverMode & SOLVER_USE_WARMSTARTING)
	{
		solverConstraint.m_appliedImpulse = cp.m_appliedImpulse * infoGlobal.m_warmstartingFactor;
		if (rb0)
			bodyA->internalApplyImpulse(solverConstraint.m_contactNormal1 * bodyA->internalGetInvMass(), solverConstraint.m_angularComponentA, solverConstraint.m_appliedImpulse);
		if (rb1)
			bodyB->internalApplyImpulse(-solverConstraint.m_contactNormal2 * bodyB->internalGetInvMass(), -solverConstraint.m_angularComponentB, -(btScalar)solverConstraint.m_appliedImpulse);
	}
	else
	{
		solverConstraint.m_appliedImpulse = 0.f;
	}

	solverConstraint.m_appliedPushImpulse = 0.f;

	{
		btVector3 externalForceImpulseA = bodyA->m_originalBody ? bodyA->m_externalForceImpulse : btVector3(0, 0, 0);
		btVector3 externalTorqueImpulseA = bodyA->m_originalBody ? bodyA->m_externalTorqueImpulse : btVector3(0, 0, 0);
		btVector3 externalForceImpulseB = bodyB->m_originalBody ? bodyB->m_externalForceImpulse : btVector3(0, 0, 0);
		btVector3 externalTorqueImpulseB = bodyB->m_originalBody ? bodyB->m_externalTorqueImpulse : btVector3(0, 0, 0);

		btScalar vel1Dotn = solverConstraint.m_contactNormal1.dot(bodyA->m_linearVelocity + externalForceImpulseA) + solverConstraint.m_relpos1CrossNormal.dot(bodyA->m_angularVelocity + externalTorqueImpulseA);
		btScalar vel2Dotn = solverConstraint.m_contactNormal2.dot(bodyB->m_linearVelocity + externalForceImpulseB) + solverConstraint.m_relpos2CrossNormal.dot(bodyB->m_angularVelocity + externalTorqueImpulseB);
		btScalar rel_vel = vel1Dotn + vel2Dotn;

		btScalar positionalError = 0.f;
		btScalar velocityError = restitution - rel_vel;  // * damping;

		if (penetration > 0)
		{
			positionalError = 0;

			velocityError -= penetration * invTimeStep;
		}
		else
		{
			positionalError = -penetration * erp * invTimeStep;
		}

		btScalar penetrationImpulse = positionalError * solverConstraint.m_jacDiagABInv;
		btScalar velocityImpulse = velocityError * solverConstraint.m_jacDiagABInv;

		if (!infoGlobal.m_splitImpulse || (penetration > infoGlobal.m_splitImpulsePenetrationThreshold))
		{
			//combine position and velocity into rhs
			solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;  //-solverConstraint.m_contactNormal1.dot(bodyA->m_externalForce*bodyA->m_invMass-bodyB->m_externalForce/bodyB->m_invMass)*solverConstraint.m_jacDiagABInv;
			solverConstraint.m_rhsPenetration = 0.f;
		}
		else
		{
			//split position and velocity into rhs and m_rhsPenetration
			solverConstraint.m_rhs = velocityImpulse;
			solverConstraint.m_rhsPenetration = penetrationImpulse;
		}
		solverConstraint.m_cfm = cfm * solverConstraint.m_jacDiagABInv;
		solverConstraint.m_lowerLimit = 0;
		solverConstraint.m_upperLimit = 1e10f;
	}
}

void btSequentialImpulseConstraintSolver::setFrictionConstraintImpulse(btSolverConstraint& solverConstraint,
	int solverBodyIdA, int solverBodyIdB,
	btManifoldPoint& cp, const btContactSolverInfo& infoGlobal)
{
	{
		btSolverConstraint& frictionConstraint1 = m_tmpSolverContactFrictionConstraintPool[solverConstraint.m_frictionIndex];
		
		frictionConstraint1.m_appliedImpulse = 0.f;
	}

	if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
	{
		btSolverConstraint& frictionConstraint2 = m_tmpSolverContactFrictionConstraintPool[solverConstraint.m_frictionIndex + 1];
		
		frictionConstraint2.m_appliedImpulse = 0.f;
	}
}

void btSequentialImpulseConstraintSolver::convertContact(btPersistentManifold* manifold, const btContactSolverInfo& infoGlobal)
{
	btCollisionObject *colObj0 = 0, *colObj1 = 0;

	colObj0 = (btCollisionObject*)manifold->getBody0();
	colObj1 = (btCollisionObject*)manifold->getBody1();

	int solverBodyIdA = getOrInitSolverBody(*colObj0, infoGlobal.m_timeStep);
	int solverBodyIdB = getOrInitSolverBody(*colObj1, infoGlobal.m_timeStep);

	//	btRigidBody* bodyA = btRigidBody::upcast(colObj0);
	//	btRigidBody* bodyB = btRigidBody::upcast(colObj1);

	btSolverBody* solverBodyA = &m_tmpSolverBodyPool[solverBodyIdA];
	btSolverBody* solverBodyB = &m_tmpSolverBodyPool[solverBodyIdB];

	///avoid collision response between two static objects
	if (!solverBodyA || (solverBodyA->m_invMass.fuzzyZero() && (!solverBodyB || solverBodyB->m_invMass.fuzzyZero())))
		return;

	int rollingFriction = 1;
	for (int j = 0; j < manifold->getNumContacts(); j++)
	{
		btManifoldPoint& cp = manifold->getContactPoint(j);

		if (cp.getDistance() <= manifold->getContactProcessingThreshold())
		{
			btVector3 rel_pos1;
			btVector3 rel_pos2;
			btScalar relaxation;

			int frictionIndex = m_tmpSolverContactConstraintPool.size();
			btSolverConstraint& solverConstraint = m_tmpSolverContactConstraintPool.expandNonInitializing();
			solverConstraint.m_solverBodyIdA = solverBodyIdA;
			solverConstraint.m_solverBodyIdB = solverBodyIdB;

			solverConstraint.m_originalContactPoint = &cp;

			const btVector3& pos1 = cp.getPositionWorldOnA();
			const btVector3& pos2 = cp.getPositionWorldOnB();

			rel_pos1 = pos1 - colObj0->getWorldTransform().getOrigin();
			rel_pos2 = pos2 - colObj1->getWorldTransform().getOrigin();

			btVector3 vel1;
			btVector3 vel2;

			solverBodyA->getVelocityInLocalPointNoDelta(rel_pos1, vel1);
			solverBodyB->getVelocityInLocalPointNoDelta(rel_pos2, vel2);

			btVector3 vel = vel1 - vel2;
			btScalar rel_vel = cp.m_normalWorldOnB.dot(vel);

			setupContactConstraint(solverConstraint, solverBodyIdA, solverBodyIdB, cp, infoGlobal, relaxation, rel_pos1, rel_pos2);

			/////setup the friction constraints

			solverConstraint.m_frictionIndex = m_tmpSolverContactFrictionConstraintPool.size();

			if ((cp.m_combinedRollingFriction > 0.f) && (rollingFriction > 0))
			{
				{
					addTorsionalFrictionConstraint(cp.m_normalWorldOnB, solverBodyIdA, solverBodyIdB, frictionIndex, cp, cp.m_combinedSpinningFriction, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
					btVector3 axis0, axis1;
					btPlaneSpace1(cp.m_normalWorldOnB, axis0, axis1);
					axis0.normalize();
					axis1.normalize();

					applyAnisotropicFriction(colObj0, axis0, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
					applyAnisotropicFriction(colObj1, axis0, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
					applyAnisotropicFriction(colObj0, axis1, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
					applyAnisotropicFriction(colObj1, axis1, btCollisionObject::CF_ANISOTROPIC_ROLLING_FRICTION);
					if (axis0.length() > 0.001)
						addTorsionalFrictionConstraint(axis0, solverBodyIdA, solverBodyIdB, frictionIndex, cp,
							cp.m_combinedRollingFriction, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
					if (axis1.length() > 0.001)
						addTorsionalFrictionConstraint(axis1, solverBodyIdA, solverBodyIdB, frictionIndex, cp,
							cp.m_combinedRollingFriction, rel_pos1, rel_pos2, colObj0, colObj1, relaxation);
					}
				}

			///Bullet has several options to set the friction directions
			///By default, each contact has only a single friction direction that is recomputed automatically very frame
			///based on the relative linear velocity.
			///If the relative velocity it zero, it will automatically compute a friction direction.

			///You can also enable two friction directions, using the SOLVER_USE_2_FRICTION_DIRECTIONS.
			///In that case, the second friction direction will be orthogonal to both contact normal and first friction direction.
			///
			///If you choose SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION, then the friction will be independent from the relative projected velocity.
			///
			///The user can manually override the friction directions for certain contacts using a contact callback,
			///and use contactPoint.m_contactPointFlags |= BT_CONTACT_FLAG_LATERAL_FRICTION_INITIALIZED
			///In that case, you can set the target relative motion in each friction direction (cp.m_contactMotion1 and cp.m_contactMotion2)
			///this will give a conveyor belt effect
			///

			if (!(infoGlobal.m_solverMode & SOLVER_ENABLE_FRICTION_DIRECTION_CACHING) || !(cp.m_contactPointFlags & BT_CONTACT_FLAG_LATERAL_FRICTION_INITIALIZED))
			{
				cp.m_lateralFrictionDir1 = vel - cp.m_normalWorldOnB * rel_vel;
				btScalar lat_rel_vel = cp.m_lateralFrictionDir1.length2();
				if (!(infoGlobal.m_solverMode & SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION) && lat_rel_vel > SIMD_EPSILON)
				{
					cp.m_lateralFrictionDir1 *= 1.f / btSqrt(lat_rel_vel);
					applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_FRICTION);
					applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_FRICTION);
					addFrictionConstraint(cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, infoGlobal);

					if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
					{
						cp.m_lateralFrictionDir2 = cp.m_lateralFrictionDir1.cross(cp.m_normalWorldOnB);
						cp.m_lateralFrictionDir2.normalize();  //??
						applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_FRICTION);
						applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_FRICTION);
						addFrictionConstraint(cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, infoGlobal);
					}
				}
				else
				{
					btPlaneSpace1(cp.m_normalWorldOnB, cp.m_lateralFrictionDir1, cp.m_lateralFrictionDir2);

					applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_FRICTION);
					applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir1, btCollisionObject::CF_ANISOTROPIC_FRICTION);
					addFrictionConstraint(cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, infoGlobal);

					if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
					{
						applyAnisotropicFriction(colObj0, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_FRICTION);
						applyAnisotropicFriction(colObj1, cp.m_lateralFrictionDir2, btCollisionObject::CF_ANISOTROPIC_FRICTION);
						addFrictionConstraint(cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, infoGlobal);
					}

					if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS) && (infoGlobal.m_solverMode & SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION))
					{
						cp.m_contactPointFlags |= BT_CONTACT_FLAG_LATERAL_FRICTION_INITIALIZED;
					}
				}
			}
			else
			{
				addFrictionConstraint(cp.m_lateralFrictionDir1, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, infoGlobal, cp.m_contactMotion1, cp.m_frictionCFM);

				if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
					addFrictionConstraint(cp.m_lateralFrictionDir2, solverBodyIdA, solverBodyIdB, frictionIndex, cp, rel_pos1, rel_pos2, colObj0, colObj1, relaxation, infoGlobal, cp.m_contactMotion2, cp.m_frictionCFM);
				}
			setFrictionConstraintImpulse(solverConstraint, solverBodyIdA, solverBodyIdB, cp, infoGlobal);
			}
		}
	}

void btSequentialImpulseConstraintSolver::convertContacts(btPersistentManifold** manifoldPtr, int numManifolds, const btContactSolverInfo& infoGlobal)
{
	int i;
	btPersistentManifold* manifold = 0;
	//			btCollisionObject* colObj0=0,*colObj1=0;

	for (i = 0; i < numManifolds; i++)
	{
		manifold = manifoldPtr[i];
		convertContact(manifold, infoGlobal);
	}
}

void btSequentialImpulseConstraintSolver::convertJoint(btSolverConstraint* currentConstraintRow,
	btTypedConstraint* constraint,
	const btTypedConstraint::btConstraintInfo1& info1,
	int solverBodyIdA,
	int solverBodyIdB,
	const btContactSolverInfo& infoGlobal)
{
	const btRigidBody& rbA = constraint->getRigidBodyA();
	const btRigidBody& rbB = constraint->getRigidBodyB();

	const btSolverBody* bodyAPtr = &m_tmpSolverBodyPool[solverBodyIdA];
	const btSolverBody* bodyBPtr = &m_tmpSolverBodyPool[solverBodyIdB];

	int overrideNumSolverIterations = constraint->getOverrideNumSolverIterations() > 0 ? constraint->getOverrideNumSolverIterations() : infoGlobal.m_numIterations;
	if (overrideNumSolverIterations > m_maxOverrideNumSolverIterations)
		m_maxOverrideNumSolverIterations = overrideNumSolverIterations;

	for (int j = 0; j < info1.m_numConstraintRows; j++)
	{
		memset(&currentConstraintRow[j], 0, sizeof(btSolverConstraint));
		currentConstraintRow[j].m_lowerLimit = -SIMD_INFINITY;
		currentConstraintRow[j].m_upperLimit = SIMD_INFINITY;
		currentConstraintRow[j].m_appliedImpulse = 0.f;
		currentConstraintRow[j].m_appliedPushImpulse = 0.f;
		currentConstraintRow[j].m_solverBodyIdA = solverBodyIdA;
		currentConstraintRow[j].m_solverBodyIdB = solverBodyIdB;
		currentConstraintRow[j].m_overrideNumSolverIterations = overrideNumSolverIterations;
	}

	// these vectors are already cleared in initSolverBody, no need to redundantly clear again
	btAssert(bodyAPtr->getDeltaLinearVelocity().isZero());
	btAssert(bodyAPtr->getDeltaAngularVelocity().isZero());
	btAssert(bodyAPtr->getPushVelocity().isZero());
	btAssert(bodyAPtr->getTurnVelocity().isZero());
	btAssert(bodyBPtr->getDeltaLinearVelocity().isZero());
	btAssert(bodyBPtr->getDeltaAngularVelocity().isZero());
	btAssert(bodyBPtr->getPushVelocity().isZero());
	btAssert(bodyBPtr->getTurnVelocity().isZero());
	//bodyAPtr->internalGetDeltaLinearVelocity().setValue(0.f,0.f,0.f);
	//bodyAPtr->internalGetDeltaAngularVelocity().setValue(0.f,0.f,0.f);
	//bodyAPtr->internalGetPushVelocity().setValue(0.f,0.f,0.f);
	//bodyAPtr->internalGetTurnVelocity().setValue(0.f,0.f,0.f);
	//bodyBPtr->internalGetDeltaLinearVelocity().setValue(0.f,0.f,0.f);
	//bodyBPtr->internalGetDeltaAngularVelocity().setValue(0.f,0.f,0.f);
	//bodyBPtr->internalGetPushVelocity().setValue(0.f,0.f,0.f);
	//bodyBPtr->internalGetTurnVelocity().setValue(0.f,0.f,0.f);

	btTypedConstraint::btConstraintInfo2 info2;
	info2.fps = 1.f / infoGlobal.m_timeStep;
	info2.erp = infoGlobal.m_erp;
	info2.m_J1linearAxis = currentConstraintRow->m_contactNormal1;
	info2.m_J1angularAxis = currentConstraintRow->m_relpos1CrossNormal;
	info2.m_J2linearAxis = currentConstraintRow->m_contactNormal2;
	info2.m_J2angularAxis = currentConstraintRow->m_relpos2CrossNormal;
	info2.rowskip = sizeof(btSolverConstraint) / sizeof(btScalar);  //check this
																	///the size of btSolverConstraint needs be a multiple of btScalar
	btAssert(info2.rowskip * sizeof(btScalar) == sizeof(btSolverConstraint));
	info2.m_constraintError = &currentConstraintRow->m_rhs;
	currentConstraintRow->m_cfm = infoGlobal.m_globalCfm;
	info2.m_damping = infoGlobal.m_damping;
	info2.cfm = &currentConstraintRow->m_cfm;
	info2.m_lowerLimit = &currentConstraintRow->m_lowerLimit;
	info2.m_upperLimit = &currentConstraintRow->m_upperLimit;
	info2.m_numIterations = infoGlobal.m_numIterations;
	constraint->getInfo2(&info2);

	///finalize the constraint setup
	for (int j = 0; j < info1.m_numConstraintRows; j++)
	{
		btSolverConstraint& solverConstraint = currentConstraintRow[j];

		if (solverConstraint.m_upperLimit >= constraint->getBreakingImpulseThreshold())
		{
			solverConstraint.m_upperLimit = constraint->getBreakingImpulseThreshold();
		}

		if (solverConstraint.m_lowerLimit <= -constraint->getBreakingImpulseThreshold())
		{
			solverConstraint.m_lowerLimit = -constraint->getBreakingImpulseThreshold();
		}

		solverConstraint.m_originalContactPoint = constraint;

		{
			const btVector3& ftorqueAxis1 = solverConstraint.m_relpos1CrossNormal;
			solverConstraint.m_angularComponentA = constraint->getRigidBodyA().getInvInertiaTensorWorld() * ftorqueAxis1 * constraint->getRigidBodyA().getAngularFactor();
		}
		{
			const btVector3& ftorqueAxis2 = solverConstraint.m_relpos2CrossNormal;
			solverConstraint.m_angularComponentB = constraint->getRigidBodyB().getInvInertiaTensorWorld() * ftorqueAxis2 * constraint->getRigidBodyB().getAngularFactor();
		}

		{
			btVector3 iMJlA = solverConstraint.m_contactNormal1 * rbA.getInvMass();
			btVector3 iMJaA = rbA.getInvInertiaTensorWorld() * solverConstraint.m_relpos1CrossNormal;
			btVector3 iMJlB = solverConstraint.m_contactNormal2 * rbB.getInvMass();  //sign of normal?
			btVector3 iMJaB = rbB.getInvInertiaTensorWorld() * solverConstraint.m_relpos2CrossNormal;

			btScalar sum = iMJlA.dot(solverConstraint.m_contactNormal1);
			sum += iMJaA.dot(solverConstraint.m_relpos1CrossNormal);
			sum += iMJlB.dot(solverConstraint.m_contactNormal2);
			sum += iMJaB.dot(solverConstraint.m_relpos2CrossNormal);
			btScalar fsum = btFabs(sum);
			btAssert(fsum > SIMD_EPSILON);
			btScalar sorRelaxation = 1.f;  //todo: get from globalInfo?
			solverConstraint.m_jacDiagABInv = fsum > SIMD_EPSILON ? sorRelaxation / sum : 0.f;
		}

		{
			btScalar rel_vel;
			btVector3 externalForceImpulseA = bodyAPtr->m_originalBody ? bodyAPtr->m_externalForceImpulse : btVector3(0, 0, 0);
			btVector3 externalTorqueImpulseA = bodyAPtr->m_originalBody ? bodyAPtr->m_externalTorqueImpulse : btVector3(0, 0, 0);

			btVector3 externalForceImpulseB = bodyBPtr->m_originalBody ? bodyBPtr->m_externalForceImpulse : btVector3(0, 0, 0);
			btVector3 externalTorqueImpulseB = bodyBPtr->m_originalBody ? bodyBPtr->m_externalTorqueImpulse : btVector3(0, 0, 0);

			btScalar vel1Dotn = solverConstraint.m_contactNormal1.dot(rbA.getLinearVelocity() + externalForceImpulseA) + solverConstraint.m_relpos1CrossNormal.dot(rbA.getAngularVelocity() + externalTorqueImpulseA);

			btScalar vel2Dotn = solverConstraint.m_contactNormal2.dot(rbB.getLinearVelocity() + externalForceImpulseB) + solverConstraint.m_relpos2CrossNormal.dot(rbB.getAngularVelocity() + externalTorqueImpulseB);

			rel_vel = vel1Dotn + vel2Dotn;
			btScalar restitution = 0.f;
			btScalar positionalError = solverConstraint.m_rhs;  //already filled in by getConstraintInfo2
			btScalar velocityError = restitution - rel_vel * info2.m_damping;
			btScalar penetrationImpulse = positionalError * solverConstraint.m_jacDiagABInv;
			btScalar velocityImpulse = velocityError * solverConstraint.m_jacDiagABInv;
			solverConstraint.m_rhs = penetrationImpulse + velocityImpulse;
			solverConstraint.m_appliedImpulse = 0.f;
		}
	}
}

void btSequentialImpulseConstraintSolver::convertJoints(btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal)
{
	BT_PROFILE("convertJoints");
	for (int j = 0; j < numConstraints; j++)
	{
		btTypedConstraint* constraint = constraints[j];
		constraint->buildJacobian();
		constraint->internalSetAppliedImpulse(0.0f);
	}

	int totalNumRows = 0;

	m_tmpConstraintSizesPool.resizeNoInitialize(numConstraints);
	//calculate the total number of contraint rows
	for (int i = 0; i < numConstraints; i++)
	{
		btTypedConstraint::btConstraintInfo1& info1 = m_tmpConstraintSizesPool[i];
		btJointFeedback* fb = constraints[i]->getJointFeedback();
		if (fb)
		{
			fb->m_appliedForceBodyA.setZero();
			fb->m_appliedTorqueBodyA.setZero();
			fb->m_appliedForceBodyB.setZero();
			fb->m_appliedTorqueBodyB.setZero();
		}

		if (constraints[i]->isEnabled())
		{
			constraints[i]->getInfo1(&info1);
		}
		else
		{
			info1.m_numConstraintRows = 0;
			info1.nub = 0;
		}
		totalNumRows += info1.m_numConstraintRows;
	}
	m_tmpSolverNonContactConstraintPool.resizeNoInitialize(totalNumRows);

	///setup the btSolverConstraints
	int currentRow = 0;

	for (int i = 0; i < numConstraints; i++)
	{
		const btTypedConstraint::btConstraintInfo1& info1 = m_tmpConstraintSizesPool[i];

		if (info1.m_numConstraintRows)
		{
			btAssert(currentRow < totalNumRows);

			btSolverConstraint* currentConstraintRow = &m_tmpSolverNonContactConstraintPool[currentRow];
			btTypedConstraint* constraint = constraints[i];
			btRigidBody& rbA = constraint->getRigidBodyA();
			btRigidBody& rbB = constraint->getRigidBodyB();

			int solverBodyIdA = getOrInitSolverBody(rbA, infoGlobal.m_timeStep);
			int solverBodyIdB = getOrInitSolverBody(rbB, infoGlobal.m_timeStep);

			convertJoint(currentConstraintRow, constraint, info1, solverBodyIdA, solverBodyIdB, infoGlobal);
		}
		currentRow += info1.m_numConstraintRows;
	}
}

void btSequentialImpulseConstraintSolver::convertBodies(btCollisionObject** bodies, int numBodies, const btContactSolverInfo& infoGlobal)
{
	BT_PROFILE("convertBodies");
	for (int i = 0; i < numBodies; i++)
	{
		bodies[i]->setCompanionId(-1);
	}
#if BT_THREADSAFE
	m_kinematicBodyUniqueIdToSolverBodyTable.resize(0);
#endif  // BT_THREADSAFE

	m_tmpSolverBodyPool.reserve(numBodies + 1);
	m_tmpSolverBodyPool.resize(0);

	//btSolverBody& fixedBody = m_tmpSolverBodyPool.expand();
	//initSolverBody(&fixedBody,0);

	for (int i = 0; i < numBodies; i++)
	{
		int bodyId = getOrInitSolverBody(*bodies[i], infoGlobal.m_timeStep);

		btRigidBody* body = btRigidBody::upcast(bodies[i]);
		if (body && body->getInvMass())
		{
			btSolverBody& solverBody = m_tmpSolverBodyPool[bodyId];
			btVector3 gyroForce(0, 0, 0);
			if (body->getFlags() & BT_ENABLE_GYROSCOPIC_FORCE_EXPLICIT)
			{
				gyroForce = body->computeGyroscopicForceExplicit(infoGlobal.m_maxGyroscopicForce);
				solverBody.m_externalTorqueImpulse -= gyroForce * body->getInvInertiaTensorWorld() * infoGlobal.m_timeStep;
			}
			if (body->getFlags() & BT_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_WORLD)
			{
				gyroForce = body->computeGyroscopicImpulseImplicit_World(infoGlobal.m_timeStep);
				solverBody.m_externalTorqueImpulse += gyroForce;
			}
			if (body->getFlags() & BT_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_BODY)
			{
				gyroForce = body->computeGyroscopicImpulseImplicit_Body(infoGlobal.m_timeStep);
				solverBody.m_externalTorqueImpulse += gyroForce;
			}
		}
	}
}

btScalar btSequentialImpulseConstraintSolver::solveGroupCacheFriendlySetup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer)
{
	m_fixedBodyId = -1;
	BT_PROFILE("solveGroupCacheFriendlySetup");
	(void)debugDrawer;

	// if solver mode has changed,
	if (infoGlobal.m_solverMode != m_cachedSolverMode)
	{
		// update solver functions to use SIMD or non-SIMD
		bool useSimd = !!(infoGlobal.m_solverMode & SOLVER_SIMD);
		setupSolverFunctions(useSimd);
		m_cachedSolverMode = infoGlobal.m_solverMode;
	}
	m_maxOverrideNumSolverIterations = 0;

#ifdef BT_ADDITIONAL_DEBUG
	//make sure that dynamic bodies exist for all (enabled) constraints
	for (int i = 0; i < numConstraints; i++)
	{
		btTypedConstraint* constraint = constraints[i];
		if (constraint->isEnabled())
		{
			if (!constraint->getRigidBodyA().isStaticOrKinematicObject())
			{
				bool found = false;
				for (int b = 0; b < numBodies; b++)
				{
					if (&constraint->getRigidBodyA() == bodies[b])
					{
						found = true;
						break;
					}
				}
				btAssert(found);
			}
			if (!constraint->getRigidBodyB().isStaticOrKinematicObject())
			{
				bool found = false;
				for (int b = 0; b < numBodies; b++)
				{
					if (&constraint->getRigidBodyB() == bodies[b])
					{
						found = true;
						break;
					}
				}
				btAssert(found);
			}
		}
	}
	//make sure that dynamic bodies exist for all contact manifolds
	for (int i = 0; i < numManifolds; i++)
	{
		if (!manifoldPtr[i]->getBody0()->isStaticOrKinematicObject())
		{
			bool found = false;
			for (int b = 0; b < numBodies; b++)
			{
				if (manifoldPtr[i]->getBody0() == bodies[b])
				{
					found = true;
					break;
				}
			}
			btAssert(found);
		}
		if (!manifoldPtr[i]->getBody1()->isStaticOrKinematicObject())
		{
			bool found = false;
			for (int b = 0; b < numBodies; b++)
			{
				if (manifoldPtr[i]->getBody1() == bodies[b])
				{
					found = true;
					break;
				}
			}
			btAssert(found);
		}
	}
#endif  //BT_ADDITIONAL_DEBUG

	//convert all bodies
	convertBodies(bodies, numBodies, infoGlobal);

	convertJoints(constraints, numConstraints, infoGlobal);

	convertContacts(manifoldPtr, numManifolds, infoGlobal);

	//	btContactSolverInfo info = infoGlobal;

	int numNonContactPool = m_tmpSolverNonContactConstraintPool.size();
	int numConstraintPool = m_tmpSolverContactConstraintPool.size();
	int numFrictionPool = m_tmpSolverContactFrictionConstraintPool.size();

	///@todo: use stack allocator for such temporarily memory, same for solver bodies/constraints
	m_orderNonContactConstraintPool.resizeNoInitialize(numNonContactPool);
	if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
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

btScalar btSequentialImpulseConstraintSolver::solveSingleIteration(int iteration, btCollisionObject** /*bodies */, int /*numBodies*/, btPersistentManifold** /*manifoldPtr*/, int /*numManifolds*/, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* /*debugDrawer*/)
{
	BT_PROFILE("solveSingleIteration");
	btScalar leastSquaresResidual = 0.f;

	int numNonContactPool = m_tmpSolverNonContactConstraintPool.size();
	int numConstraintPool = m_tmpSolverContactConstraintPool.size();
	int numFrictionPool = m_tmpSolverContactFrictionConstraintPool.size();

	if (infoGlobal.m_solverMode & SOLVER_RANDMIZE_ORDER)
	{
		if (1)  // uncomment this for a bit less random ((iteration & 7) == 0)
		{
			for (int j = 0; j < numNonContactPool; ++j)
			{
				int tmp = m_orderNonContactConstraintPool[j];
				int swapi = btRandInt2(j + 1);
				m_orderNonContactConstraintPool[j] = m_orderNonContactConstraintPool[swapi];
				m_orderNonContactConstraintPool[swapi] = tmp;
			}

			//contact/friction constraints are not solved more than
			if (iteration < infoGlobal.m_numIterations)
			{
				for (int j = 0; j < numConstraintPool; ++j)
				{
					int tmp = m_orderTmpConstraintPool[j];
					int swapi = btRandInt2(j + 1);
					m_orderTmpConstraintPool[j] = m_orderTmpConstraintPool[swapi];
					m_orderTmpConstraintPool[swapi] = tmp;
				}

				for (int j = 0; j < numFrictionPool; ++j)
				{
					int tmp = m_orderFrictionConstraintPool[j];
					int swapi = btRandInt2(j + 1);
					m_orderFrictionConstraintPool[j] = m_orderFrictionConstraintPool[swapi];
					m_orderFrictionConstraintPool[swapi] = tmp;
				}
			}
		}
	}

	///solve all joint constraints
	for (int j = 0; j < m_tmpSolverNonContactConstraintPool.size(); j++)
	{
		btSolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[m_orderNonContactConstraintPool[j]];
		if (iteration < constraint.m_overrideNumSolverIterations)
		{
			btScalar residual = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[constraint.m_solverBodyIdA], m_tmpSolverBodyPool[constraint.m_solverBodyIdB], constraint);
			leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);
		}
	}

	if (iteration < infoGlobal.m_numIterations)
	{
		for (int j = 0; j < numConstraints; j++)
		{
			if (constraints[j]->isEnabled())
			{
				int bodyAid = getOrInitSolverBody(constraints[j]->getRigidBodyA(), infoGlobal.m_timeStep);
				int bodyBid = getOrInitSolverBody(constraints[j]->getRigidBodyB(), infoGlobal.m_timeStep);
				btSolverBody& bodyA = m_tmpSolverBodyPool[bodyAid];
				btSolverBody& bodyB = m_tmpSolverBodyPool[bodyBid];
				constraints[j]->solveConstraintObsolete(bodyA, bodyB, infoGlobal.m_timeStep);
			}
		}

		///solve all contact constraints
		if (infoGlobal.m_solverMode & SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS)
		{
			int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
			int multiplier = (infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS) ? 2 : 1;

			for (int c = 0; c < numPoolConstraints; c++)
			{
				btScalar totalImpulse = 0;

				{
					const btSolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[c]];
					btScalar residual = resolveSingleConstraintRowLowerLimit(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
					leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);

					totalImpulse = solveManifold.m_appliedImpulse;
				}
				bool applyFriction = true;
				if (applyFriction)
				{
					{
						btSolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[c * multiplier]];

						if (totalImpulse > btScalar(0))
						{
							solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
							solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

							btScalar residual = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
							leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);
						}
					}

					if (infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS)
					{
						btSolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[c * multiplier + 1]];

						if (totalImpulse > btScalar(0))
						{
							solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
							solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

							btScalar residual = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
							leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);
						}
					}
				}
			}
		}
		else  //SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS
		{
			//solve the friction constraints after all contact constraints, don't interleave them
			int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
			int j;

			for (j = 0; j < numPoolConstraints; j++)
			{
				const btSolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];
				btScalar residual = resolveSingleConstraintRowLowerLimit(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
				leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);
			}

			///solve all friction constraints

			int numFrictionPoolConstraints = m_tmpSolverContactFrictionConstraintPool.size();
			for (j = 0; j < numFrictionPoolConstraints; j++)
			{
				btSolverConstraint& solveManifold = m_tmpSolverContactFrictionConstraintPool[m_orderFrictionConstraintPool[j]];
				btScalar totalImpulse = m_tmpSolverContactConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;

				if (totalImpulse > btScalar(0))
				{
					solveManifold.m_lowerLimit = -(solveManifold.m_friction * totalImpulse);
					solveManifold.m_upperLimit = solveManifold.m_friction * totalImpulse;

					btScalar residual = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
					leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);
				}
			}
		}

		int numRollingFrictionPoolConstraints = m_tmpSolverContactRollingFrictionConstraintPool.size();
		for (int j = 0; j < numRollingFrictionPoolConstraints; j++)
		{
			btSolverConstraint& rollingFrictionConstraint = m_tmpSolverContactRollingFrictionConstraintPool[j];
			btScalar totalImpulse = m_tmpSolverContactConstraintPool[rollingFrictionConstraint.m_frictionIndex].m_appliedImpulse;
			if (totalImpulse > btScalar(0))
			{
				btScalar rollingFrictionMagnitude = rollingFrictionConstraint.m_friction * totalImpulse;
				if (rollingFrictionMagnitude > rollingFrictionConstraint.m_friction)
					rollingFrictionMagnitude = rollingFrictionConstraint.m_friction;

				rollingFrictionConstraint.m_lowerLimit = -rollingFrictionMagnitude;
				rollingFrictionConstraint.m_upperLimit = rollingFrictionMagnitude;

				btScalar residual = resolveSingleConstraintRowGeneric(m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdA], m_tmpSolverBodyPool[rollingFrictionConstraint.m_solverBodyIdB], rollingFrictionConstraint);
				leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);
			}
		}
	}
	return leastSquaresResidual;
}

void btSequentialImpulseConstraintSolver::solveGroupCacheFriendlySplitImpulseIterations(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer)
{
	BT_PROFILE("solveGroupCacheFriendlySplitImpulseIterations");
	int iteration;
	if (infoGlobal.m_splitImpulse)
	{
		{
			for (iteration = 0; iteration < infoGlobal.m_numIterations; iteration++)
			{
				btScalar leastSquaresResidual = 0.f;
				{
					int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
					int j;
					for (j = 0; j < numPoolConstraints; j++)
					{
						const btSolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[m_orderTmpConstraintPool[j]];

						btScalar residual = resolveSplitPenetrationImpulse(m_tmpSolverBodyPool[solveManifold.m_solverBodyIdA], m_tmpSolverBodyPool[solveManifold.m_solverBodyIdB], solveManifold);
						leastSquaresResidual = btMax(leastSquaresResidual, residual * residual);
					}
				}
				if (leastSquaresResidual <= infoGlobal.m_leastSquaresResidualThreshold || iteration >= (infoGlobal.m_numIterations - 1))
				{
#ifdef VERBOSE_RESIDUAL_PRINTF
					printf("residual = %f at iteration #%d\n", leastSquaresResidual, iteration);
#endif
					break;
				}
			}
		}
	}
}

btScalar btSequentialImpulseConstraintSolver::solveGroupCacheFriendlyIterations(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer)
{
	BT_PROFILE("solveGroupCacheFriendlyIterations");

	{
		///this is a special step to resolve penetrations (just for contacts)
		solveGroupCacheFriendlySplitImpulseIterations(bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

		int maxIterations = m_maxOverrideNumSolverIterations > infoGlobal.m_numIterations ? m_maxOverrideNumSolverIterations : infoGlobal.m_numIterations;

		for (int iteration = 0; iteration < maxIterations; iteration++)
			//for ( int iteration = maxIterations-1  ; iteration >= 0;iteration--)
		{
			m_leastSquaresResidual = solveSingleIteration(iteration, bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

			if (m_leastSquaresResidual <= infoGlobal.m_leastSquaresResidualThreshold || (iteration >= (maxIterations - 1)))
			{
#ifdef VERBOSE_RESIDUAL_PRINTF
				printf("residual = %f at iteration #%d\n", m_leastSquaresResidual, iteration);
#endif
				m_analyticsData.m_numSolverCalls++;
				m_analyticsData.m_numIterationsUsed = iteration+1;
				m_analyticsData.m_islandId = -2;
				if (numBodies>0)
					m_analyticsData.m_islandId = bodies[0]->getCompanionId();
				m_analyticsData.m_numBodies = numBodies;
				m_analyticsData.m_numContactManifolds = numManifolds;
				m_analyticsData.m_remainingLeastSquaresResidual = m_leastSquaresResidual;
				break;
			}
		}
	}
	return 0.f;
}

void btSequentialImpulseConstraintSolver::writeBackContacts(int iBegin, int iEnd, const btContactSolverInfo& infoGlobal)
{
	for (int j = iBegin; j < iEnd; j++)
	{
		const btSolverConstraint& solveManifold = m_tmpSolverContactConstraintPool[j];
		btManifoldPoint* pt = (btManifoldPoint*)solveManifold.m_originalContactPoint;
		btAssert(pt);
		pt->m_appliedImpulse = solveManifold.m_appliedImpulse;
		//	float f = m_tmpSolverContactFrictionConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;
		//	printf("pt->m_appliedImpulseLateral1 = %f\n", f);
		pt->m_appliedImpulseLateral1 = m_tmpSolverContactFrictionConstraintPool[solveManifold.m_frictionIndex].m_appliedImpulse;
		//printf("pt->m_appliedImpulseLateral1 = %f\n", pt->m_appliedImpulseLateral1);
		if ((infoGlobal.m_solverMode & SOLVER_USE_2_FRICTION_DIRECTIONS))
		{
			pt->m_appliedImpulseLateral2 = m_tmpSolverContactFrictionConstraintPool[solveManifold.m_frictionIndex + 1].m_appliedImpulse;
		}
		//do a callback here?
	}
}

void btSequentialImpulseConstraintSolver::writeBackJoints(int iBegin, int iEnd, const btContactSolverInfo& infoGlobal)
{
	for (int j = iBegin; j < iEnd; j++)
	{
		const btSolverConstraint& solverConstr = m_tmpSolverNonContactConstraintPool[j];
		btTypedConstraint* constr = (btTypedConstraint*)solverConstr.m_originalContactPoint;
		btJointFeedback* fb = constr->getJointFeedback();
		if (fb)
		{
			fb->m_appliedForceBodyA += solverConstr.m_contactNormal1 * solverConstr.m_appliedImpulse * constr->getRigidBodyA().getLinearFactor() / infoGlobal.m_timeStep;
			fb->m_appliedForceBodyB += solverConstr.m_contactNormal2 * solverConstr.m_appliedImpulse * constr->getRigidBodyB().getLinearFactor() / infoGlobal.m_timeStep;
			fb->m_appliedTorqueBodyA += solverConstr.m_relpos1CrossNormal * constr->getRigidBodyA().getAngularFactor() * solverConstr.m_appliedImpulse / infoGlobal.m_timeStep;
			fb->m_appliedTorqueBodyB += solverConstr.m_relpos2CrossNormal * constr->getRigidBodyB().getAngularFactor() * solverConstr.m_appliedImpulse / infoGlobal.m_timeStep; /*RGM ???? */
		}

		constr->internalSetAppliedImpulse(solverConstr.m_appliedImpulse);
		if (btFabs(solverConstr.m_appliedImpulse) >= constr->getBreakingImpulseThreshold())
		{
			constr->setEnabled(false);
		}
	}
}

void btSequentialImpulseConstraintSolver::writeBackBodies(int iBegin, int iEnd, const btContactSolverInfo& infoGlobal)
{
	for (int i = iBegin; i < iEnd; i++)
	{
		btRigidBody* body = m_tmpSolverBodyPool[i].m_originalBody;
		if (body)
		{
			if (infoGlobal.m_splitImpulse)
				m_tmpSolverBodyPool[i].writebackVelocityAndTransform(infoGlobal.m_timeStep, infoGlobal.m_splitImpulseTurnErp);
			else
				m_tmpSolverBodyPool[i].writebackVelocity();

			m_tmpSolverBodyPool[i].m_originalBody->setLinearVelocity(
				m_tmpSolverBodyPool[i].m_linearVelocity +
				m_tmpSolverBodyPool[i].m_externalForceImpulse);

			m_tmpSolverBodyPool[i].m_originalBody->setAngularVelocity(
				m_tmpSolverBodyPool[i].m_angularVelocity +
				m_tmpSolverBodyPool[i].m_externalTorqueImpulse);

			if (infoGlobal.m_splitImpulse)
				m_tmpSolverBodyPool[i].m_originalBody->setWorldTransform(m_tmpSolverBodyPool[i].m_worldTransform);

			m_tmpSolverBodyPool[i].m_originalBody->setCompanionId(-1);
		}
	}
}

btScalar btSequentialImpulseConstraintSolver::solveGroupCacheFriendlyFinish(btCollisionObject** bodies, int numBodies, const btContactSolverInfo& infoGlobal)
{
	BT_PROFILE("solveGroupCacheFriendlyFinish");

	if (infoGlobal.m_solverMode & SOLVER_USE_WARMSTARTING)
	{
		writeBackContacts(0, m_tmpSolverContactConstraintPool.size(), infoGlobal);
	}

	writeBackJoints(0, m_tmpSolverNonContactConstraintPool.size(), infoGlobal);
	writeBackBodies(0, m_tmpSolverBodyPool.size(), infoGlobal);

	m_tmpSolverContactConstraintPool.resizeNoInitialize(0);
	m_tmpSolverNonContactConstraintPool.resizeNoInitialize(0);
	m_tmpSolverContactFrictionConstraintPool.resizeNoInitialize(0);
	m_tmpSolverContactRollingFrictionConstraintPool.resizeNoInitialize(0);

	m_tmpSolverBodyPool.resizeNoInitialize(0);
	return 0.f;
}

/// btSequentialImpulseConstraintSolver Sequentially applies impulses
btScalar btSequentialImpulseConstraintSolver::solveGroup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer, btDispatcher* /*dispatcher*/)
{
	BT_PROFILE("solveGroup");
	//you need to provide at least some bodies

	solveGroupCacheFriendlySetup(bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

	solveGroupCacheFriendlyIterations(bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

	solveGroupCacheFriendlyFinish(bodies, numBodies, infoGlobal);

	return 0.f;
}

void btSequentialImpulseConstraintSolver::reset()
{
	m_btSeed2 = 0;
}
