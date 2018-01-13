#ifndef B3_PGS_JACOBI_SOLVER
#define B3_PGS_JACOBI_SOLVER


struct b3Contact4;
struct b3ContactPoint;


class b3Dispatcher;

#include "b3TypedConstraint.h"
#include "b3ContactSolverInfo.h"
#include "b3SolverBody.h"
#include "b3SolverConstraint.h"

struct b3RigidBodyData;
struct b3InertiaData;

class b3PgsJacobiSolver
{

protected:
	b3AlignedObjectArray<b3SolverBody>      m_tmpSolverBodyPool;
	b3ConstraintArray			m_tmpSolverContactConstraintPool;
	b3ConstraintArray			m_tmpSolverNonContactConstraintPool;
	b3ConstraintArray			m_tmpSolverContactFrictionConstraintPool;
	b3ConstraintArray			m_tmpSolverContactRollingFrictionConstraintPool;

	b3AlignedObjectArray<int>	m_orderTmpConstraintPool;
	b3AlignedObjectArray<int>	m_orderNonContactConstraintPool;
	b3AlignedObjectArray<int>	m_orderFrictionConstraintPool;
	b3AlignedObjectArray<b3TypedConstraint::b3ConstraintInfo1> m_tmpConstraintSizesPool;
	
	b3AlignedObjectArray<int>		m_bodyCount;
	b3AlignedObjectArray<int>		m_bodyCountCheck;
	
	b3AlignedObjectArray<b3Vector3>	m_deltaLinearVelocities;
	b3AlignedObjectArray<b3Vector3>	m_deltaAngularVelocities;

	bool						m_usePgs;
	void						averageVelocities();

	int							m_maxOverrideNumSolverIterations;

	int							m_numSplitImpulseRecoveries;

	b3Scalar	getContactProcessingThreshold(b3Contact4* contact)
	{
		return 0.02f;
	}
	void setupFrictionConstraint(	b3RigidBodyData* bodies,b3InertiaData* inertias, b3SolverConstraint& solverConstraint, const b3Vector3& normalAxis,int solverBodyIdA,int  solverBodyIdB,
									b3ContactPoint& cp,const b3Vector3& rel_pos1,const b3Vector3& rel_pos2,
									b3RigidBodyData* colObj0,b3RigidBodyData* colObj1, b3Scalar relaxation, 
									b3Scalar desiredVelocity=0., b3Scalar cfmSlip=0.);

	void setupRollingFrictionConstraint(b3RigidBodyData* bodies,b3InertiaData* inertias,	b3SolverConstraint& solverConstraint, const b3Vector3& normalAxis,int solverBodyIdA,int  solverBodyIdB,
									b3ContactPoint& cp,const b3Vector3& rel_pos1,const b3Vector3& rel_pos2,
									b3RigidBodyData* colObj0,b3RigidBodyData* colObj1, b3Scalar relaxation, 
									b3Scalar desiredVelocity=0., b3Scalar cfmSlip=0.);

	b3SolverConstraint&	addFrictionConstraint(b3RigidBodyData* bodies,b3InertiaData* inertias,const b3Vector3& normalAxis,int solverBodyIdA,int solverBodyIdB,int frictionIndex,b3ContactPoint& cp,const b3Vector3& rel_pos1,const b3Vector3& rel_pos2,b3RigidBodyData* colObj0,b3RigidBodyData* colObj1, b3Scalar relaxation, b3Scalar desiredVelocity=0., b3Scalar cfmSlip=0.);
	b3SolverConstraint&	addRollingFrictionConstraint(b3RigidBodyData* bodies,b3InertiaData* inertias,const b3Vector3& normalAxis,int solverBodyIdA,int solverBodyIdB,int frictionIndex,b3ContactPoint& cp,const b3Vector3& rel_pos1,const b3Vector3& rel_pos2,b3RigidBodyData* colObj0,b3RigidBodyData* colObj1, b3Scalar relaxation, b3Scalar desiredVelocity=0, b3Scalar cfmSlip=0.f);


	void setupContactConstraint(b3RigidBodyData* bodies, b3InertiaData* inertias,
								b3SolverConstraint& solverConstraint, int solverBodyIdA, int solverBodyIdB, b3ContactPoint& cp, 
								const b3ContactSolverInfo& infoGlobal, b3Vector3& vel, b3Scalar& rel_vel, b3Scalar& relaxation, 
								b3Vector3& rel_pos1, b3Vector3& rel_pos2);

	void setFrictionConstraintImpulse( b3RigidBodyData* bodies, b3InertiaData* inertias,b3SolverConstraint& solverConstraint, int solverBodyIdA,int solverBodyIdB, 
										 b3ContactPoint& cp, const b3ContactSolverInfo& infoGlobal);

	///m_btSeed2 is used for re-arranging the constraint rows. improves convergence/quality of friction
	unsigned long	m_btSeed2;

	
	b3Scalar restitutionCurve(b3Scalar rel_vel, b3Scalar restitution);

	void	convertContact(b3RigidBodyData* bodies, b3InertiaData* inertias,b3Contact4* manifold,const b3ContactSolverInfo& infoGlobal);


	void	resolveSplitPenetrationSIMD(
     b3SolverBody& bodyA,b3SolverBody& bodyB,
        const b3SolverConstraint& contactConstraint);

	void	resolveSplitPenetrationImpulseCacheFriendly(
       b3SolverBody& bodyA,b3SolverBody& bodyB,
        const b3SolverConstraint& contactConstraint);

	//internal method
	int		getOrInitSolverBody(int bodyIndex, b3RigidBodyData* bodies,b3InertiaData* inertias);
	void	initSolverBody(int bodyIndex, b3SolverBody* solverBody, b3RigidBodyData* collisionObject);

	void	resolveSingleConstraintRowGeneric(b3SolverBody& bodyA,b3SolverBody& bodyB,const b3SolverConstraint& contactConstraint);

	void	resolveSingleConstraintRowGenericSIMD(b3SolverBody& bodyA,b3SolverBody& bodyB,const b3SolverConstraint& contactConstraint);
	
	void	resolveSingleConstraintRowLowerLimit(b3SolverBody& bodyA,b3SolverBody& bodyB,const b3SolverConstraint& contactConstraint);
	
	void	resolveSingleConstraintRowLowerLimitSIMD(b3SolverBody& bodyA,b3SolverBody& bodyB,const b3SolverConstraint& contactConstraint);
		
protected:

	virtual b3Scalar solveGroupCacheFriendlySetup(b3RigidBodyData* bodies, b3InertiaData* inertias,int numBodies,b3Contact4* manifoldPtr, int numManifolds,b3TypedConstraint** constraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);


	virtual b3Scalar solveGroupCacheFriendlyIterations(b3TypedConstraint** constraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);
	virtual void solveGroupCacheFriendlySplitImpulseIterations(b3TypedConstraint** constraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);
	b3Scalar solveSingleIteration(int iteration, b3TypedConstraint** constraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);


	virtual b3Scalar solveGroupCacheFriendlyFinish(b3RigidBodyData* bodies, b3InertiaData* inertias,int numBodies,const b3ContactSolverInfo& infoGlobal);


public:

	B3_DECLARE_ALIGNED_ALLOCATOR();
	
	b3PgsJacobiSolver(bool usePgs);
	virtual ~b3PgsJacobiSolver();

//	void	solveContacts(int numBodies, b3RigidBodyData* bodies, b3InertiaData* inertias, int numContacts, b3Contact4* contacts);
	void	solveContacts(int numBodies, b3RigidBodyData* bodies, b3InertiaData* inertias, int numContacts, b3Contact4* contacts, int numConstraints, b3TypedConstraint** constraints);

	b3Scalar solveGroup(b3RigidBodyData* bodies,b3InertiaData* inertias,int numBodies,b3Contact4* manifoldPtr, int numManifolds,b3TypedConstraint** constraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);

	///clear internal cached data and reset random seed
	virtual	void	reset();
	
	unsigned long b3Rand2();

	int b3RandInt2 (int n);

	void	setRandSeed(unsigned long seed)
	{
		m_btSeed2 = seed;
	}
	unsigned long	getRandSeed() const
	{
		return m_btSeed2;
	}




};

#endif //B3_PGS_JACOBI_SOLVER

