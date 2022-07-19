/*
 Bullet Continuous Collision Detection and Physics Library
 Copyright (c) 2019 Google Inc. http://bulletphysics.org
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it freely,
 subject to the following restrictions:
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef BT_MULTIBODY_INPLACE_SOLVER_ISLAND_CALLBACK_H
#define BT_MULTIBODY_INPLACE_SOLVER_ISLAND_CALLBACK_H

#include "BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h"
#include "BulletCollision/CollisionDispatch/btSimulationIslandManager.h"
#include "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h"
#include "btMultiBodyConstraintSolver.h"

SIMD_FORCE_INLINE int btGetConstraintIslandId2(const btTypedConstraint* lhs)
{
    int islandId;
    
    const btCollisionObject& rcolObj0 = lhs->getRigidBodyA();
    const btCollisionObject& rcolObj1 = lhs->getRigidBodyB();
    islandId = rcolObj0.getIslandTag() >= 0 ? rcolObj0.getIslandTag() : rcolObj1.getIslandTag();
    return islandId;
}
class btSortConstraintOnIslandPredicate2
{
public:
    bool operator()(const btTypedConstraint* lhs, const btTypedConstraint* rhs) const
    {
        int rIslandId0, lIslandId0;
        rIslandId0 = btGetConstraintIslandId2(rhs);
        lIslandId0 = btGetConstraintIslandId2(lhs);
        return lIslandId0 < rIslandId0;
    }
};

SIMD_FORCE_INLINE int btGetMultiBodyConstraintIslandId(const btMultiBodyConstraint* lhs)
{
    int islandId;
    
    int islandTagA = lhs->getIslandIdA();
    int islandTagB = lhs->getIslandIdB();
    islandId = islandTagA >= 0 ? islandTagA : islandTagB;
    return islandId;
}

class btSortMultiBodyConstraintOnIslandPredicate
{
public:
    bool operator()(const btMultiBodyConstraint* lhs, const btMultiBodyConstraint* rhs) const
    {
        int rIslandId0, lIslandId0;
        rIslandId0 = btGetMultiBodyConstraintIslandId(rhs);
        lIslandId0 = btGetMultiBodyConstraintIslandId(lhs);
        return lIslandId0 < rIslandId0;
    }
};

struct MultiBodyInplaceSolverIslandCallback : public btSimulationIslandManager::IslandCallback
{

    btContactSolverInfo* m_solverInfo;
    btMultiBodyConstraintSolver* m_solver;
    btMultiBodyConstraint** m_multiBodySortedConstraints;
    int m_numMultiBodyConstraints;
    
    btTypedConstraint** m_sortedConstraints;
    int m_numConstraints;
    btIDebugDraw* m_debugDrawer;
    btDispatcher* m_dispatcher;
    
    btAlignedObjectArray<btCollisionObject*> m_bodies;
	btAlignedObjectArray<btCollisionObject*> m_softBodies;
    btAlignedObjectArray<btPersistentManifold*> m_manifolds;
    btAlignedObjectArray<btTypedConstraint*> m_constraints;
    btAlignedObjectArray<btMultiBodyConstraint*> m_multiBodyConstraints;
    
    btAlignedObjectArray<btSolverAnalyticsData> m_islandAnalyticsData;
    
    MultiBodyInplaceSolverIslandCallback(btMultiBodyConstraintSolver* solver,
                                         btDispatcher* dispatcher)
    : m_solverInfo(NULL),
    m_solver(solver),
    m_multiBodySortedConstraints(NULL),
    m_numConstraints(0),
    m_debugDrawer(NULL),
    m_dispatcher(dispatcher)
    {
    }
    
    MultiBodyInplaceSolverIslandCallback& operator=(const MultiBodyInplaceSolverIslandCallback& other)
    {
        btAssert(0);
        (void)other;
        return *this;
    }
    
    SIMD_FORCE_INLINE virtual void setup(btContactSolverInfo* solverInfo, btTypedConstraint** sortedConstraints, int numConstraints, btMultiBodyConstraint** sortedMultiBodyConstraints, int numMultiBodyConstraints, btIDebugDraw* debugDrawer)
    {
        m_islandAnalyticsData.clear();
        btAssert(solverInfo);
        m_solverInfo = solverInfo;
        
        m_multiBodySortedConstraints = sortedMultiBodyConstraints;
        m_numMultiBodyConstraints = numMultiBodyConstraints;
        m_sortedConstraints = sortedConstraints;
        m_numConstraints = numConstraints;
        
        m_debugDrawer = debugDrawer;
        m_bodies.resize(0);
        m_manifolds.resize(0);
        m_constraints.resize(0);
        m_multiBodyConstraints.resize(0);
    }
    
    void setMultiBodyConstraintSolver(btMultiBodyConstraintSolver* solver)
    {
        m_solver = solver;
    }
    
    virtual void processIsland(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifolds, int numManifolds, int islandId)
    {
        if (islandId < 0)
        {
            ///we don't split islands, so all constraints/contact manifolds/bodies are passed into the solver regardless the island id
            m_solver->solveMultiBodyGroup(bodies, numBodies, manifolds, numManifolds, m_sortedConstraints, m_numConstraints, &m_multiBodySortedConstraints[0], m_numConstraints, *m_solverInfo, m_debugDrawer, m_dispatcher);
            if (m_solverInfo->m_reportSolverAnalytics&1)
            {
                m_solver->m_analyticsData.m_islandId = islandId;
                m_islandAnalyticsData.push_back(m_solver->m_analyticsData);
            }
        }
        else
        {
            //also add all non-contact constraints/joints for this island
            btTypedConstraint** startConstraint = 0;
            btMultiBodyConstraint** startMultiBodyConstraint = 0;
            
            int numCurConstraints = 0;
            int numCurMultiBodyConstraints = 0;
            
            int i;
            
            //find the first constraint for this island
            
            for (i = 0; i < m_numConstraints; i++)
            {
                if (btGetConstraintIslandId2(m_sortedConstraints[i]) == islandId)
                {
                    startConstraint = &m_sortedConstraints[i];
                    break;
                }
            }
            //count the number of constraints in this island
            for (; i < m_numConstraints; i++)
            {
                if (btGetConstraintIslandId2(m_sortedConstraints[i]) == islandId)
                {
                    numCurConstraints++;
                }
            }
            
            for (i = 0; i < m_numMultiBodyConstraints; i++)
            {
                if (btGetMultiBodyConstraintIslandId(m_multiBodySortedConstraints[i]) == islandId)
                {
                    startMultiBodyConstraint = &m_multiBodySortedConstraints[i];
                    break;
                }
            }
            //count the number of multi body constraints in this island
            for (; i < m_numMultiBodyConstraints; i++)
            {
                if (btGetMultiBodyConstraintIslandId(m_multiBodySortedConstraints[i]) == islandId)
                {
                    numCurMultiBodyConstraints++;
                }
            }
            
            //if (m_solverInfo->m_minimumSolverBatchSize<=1)
            //{
            //    m_solver->solveGroup( bodies,numBodies,manifolds, numManifolds,startConstraint,numCurConstraints,*m_solverInfo,m_debugDrawer,m_dispatcher);
            //} else
            {
                for (i = 0; i < numBodies; i++)
				{
					bool isSoftBodyType = (bodies[i]->getInternalType() & btCollisionObject::CO_SOFT_BODY);
					if (!isSoftBodyType)
					{
						m_bodies.push_back(bodies[i]);
					}
					else
					{
						m_softBodies.push_back(bodies[i]);
					}
				}
                for (i = 0; i < numManifolds; i++)
                    m_manifolds.push_back(manifolds[i]);
                for (i = 0; i < numCurConstraints; i++)
                    m_constraints.push_back(startConstraint[i]);
                
                for (i = 0; i < numCurMultiBodyConstraints; i++)
                    m_multiBodyConstraints.push_back(startMultiBodyConstraint[i]);
                
                if ((m_multiBodyConstraints.size() + m_constraints.size() + m_manifolds.size()) > m_solverInfo->m_minimumSolverBatchSize)
                {
                    processConstraints(islandId);
                }
                else
                {
                    //printf("deferred\n");
                }
            }
        }
    }
    
    virtual void processConstraints(int islandId=-1)
    {
        btCollisionObject** bodies = m_bodies.size() ? &m_bodies[0] : 0;
        btPersistentManifold** manifold = m_manifolds.size() ? &m_manifolds[0] : 0;
        btTypedConstraint** constraints = m_constraints.size() ? &m_constraints[0] : 0;
        btMultiBodyConstraint** multiBodyConstraints = m_multiBodyConstraints.size() ? &m_multiBodyConstraints[0] : 0;
        
        //printf("mb contacts = %d, mb constraints = %d\n", mbContacts, m_multiBodyConstraints.size());
        
        m_solver->solveMultiBodyGroup(bodies, m_bodies.size(), manifold, m_manifolds.size(), constraints, m_constraints.size(), multiBodyConstraints, m_multiBodyConstraints.size(), *m_solverInfo, m_debugDrawer, m_dispatcher);
        if (m_bodies.size() && (m_solverInfo->m_reportSolverAnalytics&1))
        {
            m_solver->m_analyticsData.m_islandId = islandId;
            m_islandAnalyticsData.push_back(m_solver->m_analyticsData);
        }
        m_bodies.resize(0);
		m_softBodies.resize(0);
        m_manifolds.resize(0);
        m_constraints.resize(0);
        m_multiBodyConstraints.resize(0);
    }
};


#endif /*BT_MULTIBODY_INPLACE_SOLVER_ISLAND_CALLBACK_H */
