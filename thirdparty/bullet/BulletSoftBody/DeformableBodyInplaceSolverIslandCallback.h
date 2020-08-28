//
//  DeformableBodyInplaceSolverIslandCallback.h
//  BulletSoftBody
//
//  Created by Xuchen Han on 12/16/19.
//

#ifndef DeformableBodyInplaceSolverIslandCallback_h
#define DeformableBodyInplaceSolverIslandCallback_h

struct DeformableBodyInplaceSolverIslandCallback : public MultiBodyInplaceSolverIslandCallback
{
	btDeformableMultiBodyConstraintSolver* m_deformableSolver;

	DeformableBodyInplaceSolverIslandCallback(btDeformableMultiBodyConstraintSolver* solver,
		 btDispatcher* dispatcher)
	: MultiBodyInplaceSolverIslandCallback(solver, dispatcher), m_deformableSolver(solver)
	{
	}


	virtual void processConstraints(int islandId=-1)
	{
		btCollisionObject** bodies = m_bodies.size() ? &m_bodies[0] : 0;
		btCollisionObject** softBodies = m_softBodies.size() ? &m_softBodies[0] : 0;
		btPersistentManifold** manifold = m_manifolds.size() ? &m_manifolds[0] : 0;
		btTypedConstraint** constraints = m_constraints.size() ? &m_constraints[0] : 0;
		btMultiBodyConstraint** multiBodyConstraints = m_multiBodyConstraints.size() ? &m_multiBodyConstraints[0] : 0;

		//printf("mb contacts = %d, mb constraints = %d\n", mbContacts, m_multiBodyConstraints.size());

		m_deformableSolver->solveDeformableBodyGroup(bodies, m_bodies.size(), softBodies, m_softBodies.size(), manifold, m_manifolds.size(), constraints, m_constraints.size(), multiBodyConstraints, m_multiBodyConstraints.size(), *m_solverInfo, m_debugDrawer, m_dispatcher);
		if (m_bodies.size() && (m_solverInfo->m_reportSolverAnalytics&1))
		{
			m_deformableSolver->m_analyticsData.m_islandId = islandId;
			m_islandAnalyticsData.push_back(m_solver->m_analyticsData);
		}
		m_bodies.resize(0);
		m_softBodies.resize(0);
		m_manifolds.resize(0);
		m_constraints.resize(0);
		m_multiBodyConstraints.resize(0);
	}
};

#endif /* DeformableBodyInplaceSolverIslandCallback_h */
