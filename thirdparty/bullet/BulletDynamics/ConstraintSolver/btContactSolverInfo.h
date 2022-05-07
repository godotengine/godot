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

#ifndef BT_CONTACT_SOLVER_INFO
#define BT_CONTACT_SOLVER_INFO

#include "LinearMath/btScalar.h"

enum btSolverMode
{
	SOLVER_RANDMIZE_ORDER = 1,
	SOLVER_FRICTION_SEPARATE = 2,
	SOLVER_USE_WARMSTARTING = 4,
	SOLVER_USE_2_FRICTION_DIRECTIONS = 16,
	SOLVER_ENABLE_FRICTION_DIRECTION_CACHING = 32,
	SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION = 64,
	SOLVER_CACHE_FRIENDLY = 128,
	SOLVER_SIMD = 256,
	SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS = 512,
	SOLVER_ALLOW_ZERO_LENGTH_FRICTION_DIRECTIONS = 1024,
	SOLVER_DISABLE_IMPLICIT_CONE_FRICTION = 2048,
	SOLVER_USE_ARTICULATED_WARMSTARTING = 4096,
};

struct btContactSolverInfoData
{
	btScalar m_tau;
	btScalar m_damping;  //global non-contact constraint damping, can be locally overridden by constraints during 'getInfo2'.
	btScalar m_friction;
	btScalar m_timeStep;
	btScalar m_restitution;
	int m_numIterations;
	btScalar m_maxErrorReduction;
	btScalar m_sor;          //successive over-relaxation term
	btScalar m_erp;          //error reduction for non-contact constraints
	btScalar m_erp2;         //error reduction for contact constraints
	btScalar m_deformable_erp;          //error reduction for deformable constraints
	btScalar m_deformable_cfm;          //constraint force mixing for deformable constraints
	btScalar m_deformable_maxErrorReduction; // maxErrorReduction for deformable contact
	btScalar m_globalCfm;    //constraint force mixing for contacts and non-contacts
	btScalar m_frictionERP;  //error reduction for friction constraints
	btScalar m_frictionCFM;  //constraint force mixing for friction constraints

	int m_splitImpulse;
	btScalar m_splitImpulsePenetrationThreshold;
	btScalar m_splitImpulseTurnErp;
	btScalar m_linearSlop;
	btScalar m_warmstartingFactor;
	btScalar m_articulatedWarmstartingFactor;
	int m_solverMode;
	int m_restingContactRestitutionThreshold;
	int m_minimumSolverBatchSize;
	btScalar m_maxGyroscopicForce;
	btScalar m_singleAxisRollingFrictionThreshold;
	btScalar m_leastSquaresResidualThreshold;
	btScalar m_restitutionVelocityThreshold;
	bool m_jointFeedbackInWorldSpace;
	bool m_jointFeedbackInJointFrame;
	int m_reportSolverAnalytics;
	int m_numNonContactInnerIterations;
};

struct btContactSolverInfo : public btContactSolverInfoData
{
	inline btContactSolverInfo()
	{
		m_tau = btScalar(0.6);
		m_damping = btScalar(1.0);
		m_friction = btScalar(0.3);
		m_timeStep = btScalar(1.f / 60.f);
		m_restitution = btScalar(0.);
		m_maxErrorReduction = btScalar(20.);
		m_numIterations = 10;
		m_erp = btScalar(0.2);
		m_erp2 = btScalar(0.2);
		m_deformable_erp = btScalar(0.06);
		m_deformable_cfm = btScalar(0.01);
		m_deformable_maxErrorReduction = btScalar(0.1);
		m_globalCfm = btScalar(0.);
		m_frictionERP = btScalar(0.2);  //positional friction 'anchors' are disabled by default
		m_frictionCFM = btScalar(0.);
		m_sor = btScalar(1.);
		m_splitImpulse = true;
		m_splitImpulsePenetrationThreshold = -.04f;
		m_splitImpulseTurnErp = 0.1f;
		m_linearSlop = btScalar(0.0);
		m_warmstartingFactor = btScalar(0.85);
		m_articulatedWarmstartingFactor = btScalar(0.85);
		//m_solverMode =  SOLVER_USE_WARMSTARTING |  SOLVER_SIMD | SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION|SOLVER_USE_2_FRICTION_DIRECTIONS|SOLVER_ENABLE_FRICTION_DIRECTION_CACHING;// | SOLVER_RANDMIZE_ORDER;
		m_solverMode = SOLVER_USE_WARMSTARTING | SOLVER_SIMD;  // | SOLVER_RANDMIZE_ORDER;
		m_restingContactRestitutionThreshold = 2;              //unused as of 2.81
		m_minimumSolverBatchSize = 128;                        //try to combine islands until the amount of constraints reaches this limit
		m_maxGyroscopicForce = 100.f;                          ///it is only used for 'explicit' version of gyroscopic force
		m_singleAxisRollingFrictionThreshold = 1e30f;          ///if the velocity is above this threshold, it will use a single constraint row (axis), otherwise 3 rows.
		m_leastSquaresResidualThreshold = 0.f;
		m_restitutionVelocityThreshold = 0.2f;  //if the relative velocity is below this threshold, there is zero restitution
		m_jointFeedbackInWorldSpace = false;
		m_jointFeedbackInJointFrame = false;
		m_reportSolverAnalytics = 0;
		m_numNonContactInnerIterations = 1;   // the number of inner iterations for solving motor constraint in a single iteration of the constraint solve
	}
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btContactSolverInfoDoubleData
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
	double m_articulatedWarmstartingFactor;
	double m_maxGyroscopicForce;  ///it is only used for 'explicit' version of gyroscopic force
	double m_singleAxisRollingFrictionThreshold;

	int m_numIterations;
	int m_solverMode;
	int m_restingContactRestitutionThreshold;
	int m_minimumSolverBatchSize;
	int m_splitImpulse;
	char m_padding[4];
};
///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btContactSolverInfoFloatData
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
	float m_articulatedWarmstartingFactor;
	float m_maxGyroscopicForce;

	float m_singleAxisRollingFrictionThreshold;
	int m_numIterations;
	int m_solverMode;
	int m_restingContactRestitutionThreshold;

	int m_minimumSolverBatchSize;
	int m_splitImpulse;
	
};

#endif  //BT_CONTACT_SOLVER_INFO
