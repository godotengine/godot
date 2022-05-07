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

#ifndef B3_CONTACT_SOLVER_INFO
#define B3_CONTACT_SOLVER_INFO

#include "Bullet3Common/b3Scalar.h"

enum b3SolverMode
{
	B3_SOLVER_RANDMIZE_ORDER = 1,
	B3_SOLVER_FRICTION_SEPARATE = 2,
	B3_SOLVER_USE_WARMSTARTING = 4,
	B3_SOLVER_USE_2_FRICTION_DIRECTIONS = 16,
	B3_SOLVER_ENABLE_FRICTION_DIRECTION_CACHING = 32,
	B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION = 64,
	B3_SOLVER_CACHE_FRIENDLY = 128,
	B3_SOLVER_SIMD = 256,
	B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS = 512,
	B3_SOLVER_ALLOW_ZERO_LENGTH_FRICTION_DIRECTIONS = 1024
};

struct b3ContactSolverInfoData
{
	b3Scalar m_tau;
	b3Scalar m_damping;  //global non-contact constraint damping, can be locally overridden by constraints during 'getInfo2'.
	b3Scalar m_friction;
	b3Scalar m_timeStep;
	b3Scalar m_restitution;
	int m_numIterations;
	b3Scalar m_maxErrorReduction;
	b3Scalar m_sor;
	b3Scalar m_erp;        //used as Baumgarte factor
	b3Scalar m_erp2;       //used in Split Impulse
	b3Scalar m_globalCfm;  //constraint force mixing
	int m_splitImpulse;
	b3Scalar m_splitImpulsePenetrationThreshold;
	b3Scalar m_splitImpulseTurnErp;
	b3Scalar m_linearSlop;
	b3Scalar m_warmstartingFactor;

	int m_solverMode;
	int m_restingContactRestitutionThreshold;
	int m_minimumSolverBatchSize;
	b3Scalar m_maxGyroscopicForce;
	b3Scalar m_singleAxisRollingFrictionThreshold;
};

struct b3ContactSolverInfo : public b3ContactSolverInfoData
{
	inline b3ContactSolverInfo()
	{
		m_tau = b3Scalar(0.6);
		m_damping = b3Scalar(1.0);
		m_friction = b3Scalar(0.3);
		m_timeStep = b3Scalar(1.f / 60.f);
		m_restitution = b3Scalar(0.);
		m_maxErrorReduction = b3Scalar(20.);
		m_numIterations = 10;
		m_erp = b3Scalar(0.2);
		m_erp2 = b3Scalar(0.8);
		m_globalCfm = b3Scalar(0.);
		m_sor = b3Scalar(1.);
		m_splitImpulse = true;
		m_splitImpulsePenetrationThreshold = -.04f;
		m_splitImpulseTurnErp = 0.1f;
		m_linearSlop = b3Scalar(0.0);
		m_warmstartingFactor = b3Scalar(0.85);
		//m_solverMode =  B3_SOLVER_USE_WARMSTARTING |  B3_SOLVER_SIMD | B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION|B3_SOLVER_USE_2_FRICTION_DIRECTIONS|B3_SOLVER_ENABLE_FRICTION_DIRECTION_CACHING;// | B3_SOLVER_RANDMIZE_ORDER;
		m_solverMode = B3_SOLVER_USE_WARMSTARTING | B3_SOLVER_SIMD;  // | B3_SOLVER_RANDMIZE_ORDER;
		m_restingContactRestitutionThreshold = 2;                    //unused as of 2.81
		m_minimumSolverBatchSize = 128;                              //try to combine islands until the amount of constraints reaches this limit
		m_maxGyroscopicForce = 100.f;                                ///only used to clamp forces for bodies that have their B3_ENABLE_GYROPSCOPIC_FORCE flag set (using b3RigidBody::setFlag)
		m_singleAxisRollingFrictionThreshold = 1e30f;                ///if the velocity is above this threshold, it will use a single constraint row (axis), otherwise 3 rows.
	}
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct b3ContactSolverInfoDoubleData
{
	double m_tau;
	double m_damping;  //global non-contact constraint damping, can be locally overridden by constraints during 'getInfo2'.
	double m_friction;
	double m_timeStep;
	double m_restitution;
	double m_maxErrorReduction;
	double m_sor;
	double m_erp;        //used as Baumgarte factor
	double m_erp2;       //used in Split Impulse
	double m_globalCfm;  //constraint force mixing
	double m_splitImpulsePenetrationThreshold;
	double m_splitImpulseTurnErp;
	double m_linearSlop;
	double m_warmstartingFactor;
	double m_maxGyroscopicForce;
	double m_singleAxisRollingFrictionThreshold;

	int m_numIterations;
	int m_solverMode;
	int m_restingContactRestitutionThreshold;
	int m_minimumSolverBatchSize;
	int m_splitImpulse;
	char m_padding[4];
};
///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct b3ContactSolverInfoFloatData
{
	float m_tau;
	float m_damping;  //global non-contact constraint damping, can be locally overridden by constraints during 'getInfo2'.
	float m_friction;
	float m_timeStep;

	float m_restitution;
	float m_maxErrorReduction;
	float m_sor;
	float m_erp;  //used as Baumgarte factor

	float m_erp2;       //used in Split Impulse
	float m_globalCfm;  //constraint force mixing
	float m_splitImpulsePenetrationThreshold;
	float m_splitImpulseTurnErp;

	float m_linearSlop;
	float m_warmstartingFactor;
	float m_maxGyroscopicForce;
	float m_singleAxisRollingFrictionThreshold;

	int m_numIterations;
	int m_solverMode;
	int m_restingContactRestitutionThreshold;
	int m_minimumSolverBatchSize;

	int m_splitImpulse;
	char m_padding[4];
};

#endif  //B3_CONTACT_SOLVER_INFO
