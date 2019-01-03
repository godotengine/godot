/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btMultiBodyDynamicsWorld.h"
#include "btMultiBodyConstraintSolver.h"
#include "btMultiBody.h"
#include "btMultiBodyLinkCollider.h"
#include "BulletCollision/CollisionDispatch/btSimulationIslandManager.h"
#include "LinearMath/btQuickprof.h"
#include "btMultiBodyConstraint.h"
#include "LinearMath/btIDebugDraw.h"
#include "LinearMath/btSerializer.h"

void btMultiBodyDynamicsWorld::addMultiBody(btMultiBody* body, int group, int mask)
{
	m_multiBodies.push_back(body);
}

void btMultiBodyDynamicsWorld::removeMultiBody(btMultiBody* body)
{
	m_multiBodies.remove(body);
}

void btMultiBodyDynamicsWorld::calculateSimulationIslands()
{
	BT_PROFILE("calculateSimulationIslands");

	getSimulationIslandManager()->updateActivationState(getCollisionWorld(), getCollisionWorld()->getDispatcher());

	{
		//merge islands based on speculative contact manifolds too
		for (int i = 0; i < this->m_predictiveManifolds.size(); i++)
		{
			btPersistentManifold* manifold = m_predictiveManifolds[i];

			const btCollisionObject* colObj0 = manifold->getBody0();
			const btCollisionObject* colObj1 = manifold->getBody1();

			if (((colObj0) && (!(colObj0)->isStaticOrKinematicObject())) &&
				((colObj1) && (!(colObj1)->isStaticOrKinematicObject())))
			{
				getSimulationIslandManager()->getUnionFind().unite((colObj0)->getIslandTag(), (colObj1)->getIslandTag());
			}
		}
	}

	{
		int i;
		int numConstraints = int(m_constraints.size());
		for (i = 0; i < numConstraints; i++)
		{
			btTypedConstraint* constraint = m_constraints[i];
			if (constraint->isEnabled())
			{
				const btRigidBody* colObj0 = &constraint->getRigidBodyA();
				const btRigidBody* colObj1 = &constraint->getRigidBodyB();

				if (((colObj0) && (!(colObj0)->isStaticOrKinematicObject())) &&
					((colObj1) && (!(colObj1)->isStaticOrKinematicObject())))
				{
					getSimulationIslandManager()->getUnionFind().unite((colObj0)->getIslandTag(), (colObj1)->getIslandTag());
				}
			}
		}
	}

	//merge islands linked by Featherstone link colliders
	for (int i = 0; i < m_multiBodies.size(); i++)
	{
		btMultiBody* body = m_multiBodies[i];
		{
			btMultiBodyLinkCollider* prev = body->getBaseCollider();

			for (int b = 0; b < body->getNumLinks(); b++)
			{
				btMultiBodyLinkCollider* cur = body->getLink(b).m_collider;

				if (((cur) && (!(cur)->isStaticOrKinematicObject())) &&
					((prev) && (!(prev)->isStaticOrKinematicObject())))
				{
					int tagPrev = prev->getIslandTag();
					int tagCur = cur->getIslandTag();
					getSimulationIslandManager()->getUnionFind().unite(tagPrev, tagCur);
				}
				if (cur && !cur->isStaticOrKinematicObject())
					prev = cur;
			}
		}
	}

	//merge islands linked by multibody constraints
	{
		for (int i = 0; i < this->m_multiBodyConstraints.size(); i++)
		{
			btMultiBodyConstraint* c = m_multiBodyConstraints[i];
			int tagA = c->getIslandIdA();
			int tagB = c->getIslandIdB();
			if (tagA >= 0 && tagB >= 0)
				getSimulationIslandManager()->getUnionFind().unite(tagA, tagB);
		}
	}

	//Store the island id in each body
	getSimulationIslandManager()->storeIslandActivationState(getCollisionWorld());
}

void btMultiBodyDynamicsWorld::updateActivationState(btScalar timeStep)
{
	BT_PROFILE("btMultiBodyDynamicsWorld::updateActivationState");

	for (int i = 0; i < m_multiBodies.size(); i++)
	{
		btMultiBody* body = m_multiBodies[i];
		if (body)
		{
			body->checkMotionAndSleepIfRequired(timeStep);
			if (!body->isAwake())
			{
				btMultiBodyLinkCollider* col = body->getBaseCollider();
				if (col && col->getActivationState() == ACTIVE_TAG)
				{
					col->setActivationState(WANTS_DEACTIVATION);
					col->setDeactivationTime(0.f);
				}
				for (int b = 0; b < body->getNumLinks(); b++)
				{
					btMultiBodyLinkCollider* col = body->getLink(b).m_collider;
					if (col && col->getActivationState() == ACTIVE_TAG)
					{
						col->setActivationState(WANTS_DEACTIVATION);
						col->setDeactivationTime(0.f);
					}
				}
			}
			else
			{
				btMultiBodyLinkCollider* col = body->getBaseCollider();
				if (col && col->getActivationState() != DISABLE_DEACTIVATION)
					col->setActivationState(ACTIVE_TAG);

				for (int b = 0; b < body->getNumLinks(); b++)
				{
					btMultiBodyLinkCollider* col = body->getLink(b).m_collider;
					if (col && col->getActivationState() != DISABLE_DEACTIVATION)
						col->setActivationState(ACTIVE_TAG);
				}
			}
		}
	}

	btDiscreteDynamicsWorld::updateActivationState(timeStep);
}

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
	btAlignedObjectArray<btPersistentManifold*> m_manifolds;
	btAlignedObjectArray<btTypedConstraint*> m_constraints;
	btAlignedObjectArray<btMultiBodyConstraint*> m_multiBodyConstraints;

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

	MultiBodyInplaceSolverIslandCallback& operator=(MultiBodyInplaceSolverIslandCallback& other)
	{
		btAssert(0);
		(void)other;
		return *this;
	}

	SIMD_FORCE_INLINE void setup(btContactSolverInfo* solverInfo, btTypedConstraint** sortedConstraints, int numConstraints, btMultiBodyConstraint** sortedMultiBodyConstraints, int numMultiBodyConstraints, btIDebugDraw* debugDrawer)
	{
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
			//	m_solver->solveGroup( bodies,numBodies,manifolds, numManifolds,startConstraint,numCurConstraints,*m_solverInfo,m_debugDrawer,m_dispatcher);
			//} else
			{
				for (i = 0; i < numBodies; i++)
					m_bodies.push_back(bodies[i]);
				for (i = 0; i < numManifolds; i++)
					m_manifolds.push_back(manifolds[i]);
				for (i = 0; i < numCurConstraints; i++)
					m_constraints.push_back(startConstraint[i]);

				for (i = 0; i < numCurMultiBodyConstraints; i++)
					m_multiBodyConstraints.push_back(startMultiBodyConstraint[i]);

				if ((m_multiBodyConstraints.size() + m_constraints.size() + m_manifolds.size()) > m_solverInfo->m_minimumSolverBatchSize)
				{
					processConstraints();
				}
				else
				{
					//printf("deferred\n");
				}
			}
		}
	}
	void processConstraints()
	{
		btCollisionObject** bodies = m_bodies.size() ? &m_bodies[0] : 0;
		btPersistentManifold** manifold = m_manifolds.size() ? &m_manifolds[0] : 0;
		btTypedConstraint** constraints = m_constraints.size() ? &m_constraints[0] : 0;
		btMultiBodyConstraint** multiBodyConstraints = m_multiBodyConstraints.size() ? &m_multiBodyConstraints[0] : 0;

		//printf("mb contacts = %d, mb constraints = %d\n", mbContacts, m_multiBodyConstraints.size());

		m_solver->solveMultiBodyGroup(bodies, m_bodies.size(), manifold, m_manifolds.size(), constraints, m_constraints.size(), multiBodyConstraints, m_multiBodyConstraints.size(), *m_solverInfo, m_debugDrawer, m_dispatcher);
		m_bodies.resize(0);
		m_manifolds.resize(0);
		m_constraints.resize(0);
		m_multiBodyConstraints.resize(0);
	}
};

btMultiBodyDynamicsWorld::btMultiBodyDynamicsWorld(btDispatcher* dispatcher, btBroadphaseInterface* pairCache, btMultiBodyConstraintSolver* constraintSolver, btCollisionConfiguration* collisionConfiguration)
	: btDiscreteDynamicsWorld(dispatcher, pairCache, constraintSolver, collisionConfiguration),
	  m_multiBodyConstraintSolver(constraintSolver)
{
	//split impulse is not yet supported for Featherstone hierarchies
	//	getSolverInfo().m_splitImpulse = false;
	getSolverInfo().m_solverMode |= SOLVER_USE_2_FRICTION_DIRECTIONS;
	m_solverMultiBodyIslandCallback = new MultiBodyInplaceSolverIslandCallback(constraintSolver, dispatcher);
}

btMultiBodyDynamicsWorld::~btMultiBodyDynamicsWorld()
{
	delete m_solverMultiBodyIslandCallback;
}

void btMultiBodyDynamicsWorld::setMultiBodyConstraintSolver(btMultiBodyConstraintSolver* solver)
{
	m_multiBodyConstraintSolver = solver;
	m_solverMultiBodyIslandCallback->setMultiBodyConstraintSolver(solver);
	btDiscreteDynamicsWorld::setConstraintSolver(solver);
}

void btMultiBodyDynamicsWorld::setConstraintSolver(btConstraintSolver* solver)
{
	if (solver->getSolverType() == BT_MULTIBODY_SOLVER)
	{
		m_multiBodyConstraintSolver = (btMultiBodyConstraintSolver*)solver;
	}
	btDiscreteDynamicsWorld::setConstraintSolver(solver);
}

void btMultiBodyDynamicsWorld::forwardKinematics()
{
	for (int b = 0; b < m_multiBodies.size(); b++)
	{
		btMultiBody* bod = m_multiBodies[b];
		bod->forwardKinematics(m_scratch_world_to_local, m_scratch_local_origin);
	}
}
void btMultiBodyDynamicsWorld::solveConstraints(btContactSolverInfo& solverInfo)
{
	forwardKinematics();

	BT_PROFILE("solveConstraints");

	clearMultiBodyConstraintForces();

	m_sortedConstraints.resize(m_constraints.size());
	int i;
	for (i = 0; i < getNumConstraints(); i++)
	{
		m_sortedConstraints[i] = m_constraints[i];
	}
	m_sortedConstraints.quickSort(btSortConstraintOnIslandPredicate2());
	btTypedConstraint** constraintsPtr = getNumConstraints() ? &m_sortedConstraints[0] : 0;

	m_sortedMultiBodyConstraints.resize(m_multiBodyConstraints.size());
	for (i = 0; i < m_multiBodyConstraints.size(); i++)
	{
		m_sortedMultiBodyConstraints[i] = m_multiBodyConstraints[i];
	}
	m_sortedMultiBodyConstraints.quickSort(btSortMultiBodyConstraintOnIslandPredicate());

	btMultiBodyConstraint** sortedMultiBodyConstraints = m_sortedMultiBodyConstraints.size() ? &m_sortedMultiBodyConstraints[0] : 0;

	m_solverMultiBodyIslandCallback->setup(&solverInfo, constraintsPtr, m_sortedConstraints.size(), sortedMultiBodyConstraints, m_sortedMultiBodyConstraints.size(), getDebugDrawer());
	m_constraintSolver->prepareSolve(getCollisionWorld()->getNumCollisionObjects(), getCollisionWorld()->getDispatcher()->getNumManifolds());

#ifndef BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY
	{
		BT_PROFILE("btMultiBody addForce");
		for (int i = 0; i < this->m_multiBodies.size(); i++)
		{
			btMultiBody* bod = m_multiBodies[i];

			bool isSleeping = false;

			if (bod->getBaseCollider() && bod->getBaseCollider()->getActivationState() == ISLAND_SLEEPING)
			{
				isSleeping = true;
			}
			for (int b = 0; b < bod->getNumLinks(); b++)
			{
				if (bod->getLink(b).m_collider && bod->getLink(b).m_collider->getActivationState() == ISLAND_SLEEPING)
					isSleeping = true;
			}

			if (!isSleeping)
			{
				//useless? they get resized in stepVelocities once again (AND DIFFERENTLY)
				m_scratch_r.resize(bod->getNumLinks() + 1);  //multidof? ("Y"s use it and it is used to store qdd)
				m_scratch_v.resize(bod->getNumLinks() + 1);
				m_scratch_m.resize(bod->getNumLinks() + 1);

				bod->addBaseForce(m_gravity * bod->getBaseMass());

				for (int j = 0; j < bod->getNumLinks(); ++j)
				{
					bod->addLinkForce(j, m_gravity * bod->getLinkMass(j));
				}
			}  //if (!isSleeping)
		}
	}
#endif  //BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY

	{
		BT_PROFILE("btMultiBody stepVelocities");
		for (int i = 0; i < this->m_multiBodies.size(); i++)
		{
			btMultiBody* bod = m_multiBodies[i];

			bool isSleeping = false;

			if (bod->getBaseCollider() && bod->getBaseCollider()->getActivationState() == ISLAND_SLEEPING)
			{
				isSleeping = true;
			}
			for (int b = 0; b < bod->getNumLinks(); b++)
			{
				if (bod->getLink(b).m_collider && bod->getLink(b).m_collider->getActivationState() == ISLAND_SLEEPING)
					isSleeping = true;
			}

			if (!isSleeping)
			{
				//useless? they get resized in stepVelocities once again (AND DIFFERENTLY)
				m_scratch_r.resize(bod->getNumLinks() + 1);  //multidof? ("Y"s use it and it is used to store qdd)
				m_scratch_v.resize(bod->getNumLinks() + 1);
				m_scratch_m.resize(bod->getNumLinks() + 1);
				bool doNotUpdatePos = false;
                bool isConstraintPass = false;
				{
					if (!bod->isUsingRK4Integration())
					{
						bod->computeAccelerationsArticulatedBodyAlgorithmMultiDof(solverInfo.m_timeStep,
						m_scratch_r, m_scratch_v, m_scratch_m,isConstraintPass,
						getSolverInfo().m_jointFeedbackInWorldSpace,
						getSolverInfo().m_jointFeedbackInJointFrame);
					}
					else
					{
						//
						int numDofs = bod->getNumDofs() + 6;
						int numPosVars = bod->getNumPosVars() + 7;
						btAlignedObjectArray<btScalar> scratch_r2;
						scratch_r2.resize(2 * numPosVars + 8 * numDofs);
						//convenience
						btScalar* pMem = &scratch_r2[0];
						btScalar* scratch_q0 = pMem;
						pMem += numPosVars;
						btScalar* scratch_qx = pMem;
						pMem += numPosVars;
						btScalar* scratch_qd0 = pMem;
						pMem += numDofs;
						btScalar* scratch_qd1 = pMem;
						pMem += numDofs;
						btScalar* scratch_qd2 = pMem;
						pMem += numDofs;
						btScalar* scratch_qd3 = pMem;
						pMem += numDofs;
						btScalar* scratch_qdd0 = pMem;
						pMem += numDofs;
						btScalar* scratch_qdd1 = pMem;
						pMem += numDofs;
						btScalar* scratch_qdd2 = pMem;
						pMem += numDofs;
						btScalar* scratch_qdd3 = pMem;
						pMem += numDofs;
						btAssert((pMem - (2 * numPosVars + 8 * numDofs)) == &scratch_r2[0]);

						/////
						//copy q0 to scratch_q0 and qd0 to scratch_qd0
						scratch_q0[0] = bod->getWorldToBaseRot().x();
						scratch_q0[1] = bod->getWorldToBaseRot().y();
						scratch_q0[2] = bod->getWorldToBaseRot().z();
						scratch_q0[3] = bod->getWorldToBaseRot().w();
						scratch_q0[4] = bod->getBasePos().x();
						scratch_q0[5] = bod->getBasePos().y();
						scratch_q0[6] = bod->getBasePos().z();
						//
						for (int link = 0; link < bod->getNumLinks(); ++link)
						{
							for (int dof = 0; dof < bod->getLink(link).m_posVarCount; ++dof)
								scratch_q0[7 + bod->getLink(link).m_cfgOffset + dof] = bod->getLink(link).m_jointPos[dof];
						}
						//
						for (int dof = 0; dof < numDofs; ++dof)
							scratch_qd0[dof] = bod->getVelocityVector()[dof];
						////
						struct
						{
							btMultiBody* bod;
							btScalar *scratch_qx, *scratch_q0;

							void operator()()
							{
								for (int dof = 0; dof < bod->getNumPosVars() + 7; ++dof)
									scratch_qx[dof] = scratch_q0[dof];
							}
						} pResetQx = {bod, scratch_qx, scratch_q0};
						//
						struct
						{
							void operator()(btScalar dt, const btScalar* pDer, const btScalar* pCurVal, btScalar* pVal, int size)
							{
								for (int i = 0; i < size; ++i)
									pVal[i] = pCurVal[i] + dt * pDer[i];
							}

						} pEulerIntegrate;
						//
						struct
						{
							void operator()(btMultiBody* pBody, const btScalar* pData)
							{
								btScalar* pVel = const_cast<btScalar*>(pBody->getVelocityVector());

								for (int i = 0; i < pBody->getNumDofs() + 6; ++i)
									pVel[i] = pData[i];
							}
						} pCopyToVelocityVector;
						//
						struct
						{
							void operator()(const btScalar* pSrc, btScalar* pDst, int start, int size)
							{
								for (int i = 0; i < size; ++i)
									pDst[i] = pSrc[start + i];
							}
						} pCopy;
						//

						btScalar h = solverInfo.m_timeStep;
#define output &m_scratch_r[bod->getNumDofs()]
						//calc qdd0 from: q0 & qd0
						bod->computeAccelerationsArticulatedBodyAlgorithmMultiDof(0., m_scratch_r, m_scratch_v, m_scratch_m,
						isConstraintPass,getSolverInfo().m_jointFeedbackInWorldSpace,
						getSolverInfo().m_jointFeedbackInJointFrame);
						pCopy(output, scratch_qdd0, 0, numDofs);
						//calc q1 = q0 + h/2 * qd0
						pResetQx();
						bod->stepPositionsMultiDof(btScalar(.5) * h, scratch_qx, scratch_qd0);
						//calc qd1 = qd0 + h/2 * qdd0
						pEulerIntegrate(btScalar(.5) * h, scratch_qdd0, scratch_qd0, scratch_qd1, numDofs);
						//
						//calc qdd1 from: q1 & qd1
						pCopyToVelocityVector(bod, scratch_qd1);
						bod->computeAccelerationsArticulatedBodyAlgorithmMultiDof(0., m_scratch_r, m_scratch_v, m_scratch_m,
						isConstraintPass,getSolverInfo().m_jointFeedbackInWorldSpace,
						getSolverInfo().m_jointFeedbackInJointFrame);
						pCopy(output, scratch_qdd1, 0, numDofs);
						//calc q2 = q0 + h/2 * qd1
						pResetQx();
						bod->stepPositionsMultiDof(btScalar(.5) * h, scratch_qx, scratch_qd1);
						//calc qd2 = qd0 + h/2 * qdd1
						pEulerIntegrate(btScalar(.5) * h, scratch_qdd1, scratch_qd0, scratch_qd2, numDofs);
						//
						//calc qdd2 from: q2 & qd2
						pCopyToVelocityVector(bod, scratch_qd2);
						bod->computeAccelerationsArticulatedBodyAlgorithmMultiDof(0., m_scratch_r, m_scratch_v, m_scratch_m,
						isConstraintPass,getSolverInfo().m_jointFeedbackInWorldSpace,
						getSolverInfo().m_jointFeedbackInJointFrame);
						pCopy(output, scratch_qdd2, 0, numDofs);
						//calc q3 = q0 + h * qd2
						pResetQx();
						bod->stepPositionsMultiDof(h, scratch_qx, scratch_qd2);
						//calc qd3 = qd0 + h * qdd2
						pEulerIntegrate(h, scratch_qdd2, scratch_qd0, scratch_qd3, numDofs);
						//
						//calc qdd3 from: q3 & qd3
						pCopyToVelocityVector(bod, scratch_qd3);
						bod->computeAccelerationsArticulatedBodyAlgorithmMultiDof(0., m_scratch_r, m_scratch_v, m_scratch_m,
						isConstraintPass,getSolverInfo().m_jointFeedbackInWorldSpace,
						getSolverInfo().m_jointFeedbackInJointFrame);
						pCopy(output, scratch_qdd3, 0, numDofs);

						//
						//calc q = q0 + h/6(qd0 + 2*(qd1 + qd2) + qd3)
						//calc qd = qd0 + h/6(qdd0 + 2*(qdd1 + qdd2) + qdd3)
						btAlignedObjectArray<btScalar> delta_q;
						delta_q.resize(numDofs);
						btAlignedObjectArray<btScalar> delta_qd;
						delta_qd.resize(numDofs);
						for (int i = 0; i < numDofs; ++i)
						{
							delta_q[i] = h / btScalar(6.) * (scratch_qd0[i] + 2 * scratch_qd1[i] + 2 * scratch_qd2[i] + scratch_qd3[i]);
							delta_qd[i] = h / btScalar(6.) * (scratch_qdd0[i] + 2 * scratch_qdd1[i] + 2 * scratch_qdd2[i] + scratch_qdd3[i]);
							//delta_q[i] = h*scratch_qd0[i];
							//delta_qd[i] = h*scratch_qdd0[i];
						}
						//
						pCopyToVelocityVector(bod, scratch_qd0);
						bod->applyDeltaVeeMultiDof(&delta_qd[0], 1);
						//
						if (!doNotUpdatePos)
						{
							btScalar* pRealBuf = const_cast<btScalar*>(bod->getVelocityVector());
							pRealBuf += 6 + bod->getNumDofs() + bod->getNumDofs() * bod->getNumDofs();

							for (int i = 0; i < numDofs; ++i)
								pRealBuf[i] = delta_q[i];

							//bod->stepPositionsMultiDof(1, 0, &delta_q[0]);
							bod->setPosUpdated(true);
						}

						//ugly hack which resets the cached data to t0 (needed for constraint solver)
						{
							for (int link = 0; link < bod->getNumLinks(); ++link)
								bod->getLink(link).updateCacheMultiDof();
							bod->computeAccelerationsArticulatedBodyAlgorithmMultiDof(0, m_scratch_r, m_scratch_v, m_scratch_m,
							isConstraintPass,getSolverInfo().m_jointFeedbackInWorldSpace,
							getSolverInfo().m_jointFeedbackInJointFrame);
						}
					}
				}

#ifndef BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY
				bod->clearForcesAndTorques();
#endif         //BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY
			}  //if (!isSleeping)
		}
	}

	/// solve all the constraints for this island
	m_islandManager->buildAndProcessIslands(getCollisionWorld()->getDispatcher(), getCollisionWorld(), m_solverMultiBodyIslandCallback);

	m_solverMultiBodyIslandCallback->processConstraints();

	m_constraintSolver->allSolved(solverInfo, m_debugDrawer);

	{
		BT_PROFILE("btMultiBody stepVelocities");
		for (int i = 0; i < this->m_multiBodies.size(); i++)
		{
			btMultiBody* bod = m_multiBodies[i];

			bool isSleeping = false;

			if (bod->getBaseCollider() && bod->getBaseCollider()->getActivationState() == ISLAND_SLEEPING)
			{
				isSleeping = true;
			}
			for (int b = 0; b < bod->getNumLinks(); b++)
			{
				if (bod->getLink(b).m_collider && bod->getLink(b).m_collider->getActivationState() == ISLAND_SLEEPING)
					isSleeping = true;
			}

			if (!isSleeping)
			{
				//useless? they get resized in stepVelocities once again (AND DIFFERENTLY)
				m_scratch_r.resize(bod->getNumLinks() + 1);  //multidof? ("Y"s use it and it is used to store qdd)
				m_scratch_v.resize(bod->getNumLinks() + 1);
				m_scratch_m.resize(bod->getNumLinks() + 1);

				{
					if (!bod->isUsingRK4Integration())
					{
						bool isConstraintPass = true;
						bod->computeAccelerationsArticulatedBodyAlgorithmMultiDof(solverInfo.m_timeStep, m_scratch_r, m_scratch_v, m_scratch_m, isConstraintPass,
						getSolverInfo().m_jointFeedbackInWorldSpace,
						getSolverInfo().m_jointFeedbackInJointFrame);
					}
				}
			}
		}
	}

	for (int i = 0; i < this->m_multiBodies.size(); i++)
	{
		btMultiBody* bod = m_multiBodies[i];
		bod->processDeltaVeeMultiDof2();
	}
}

void btMultiBodyDynamicsWorld::integrateTransforms(btScalar timeStep)
{
	btDiscreteDynamicsWorld::integrateTransforms(timeStep);

	{
		BT_PROFILE("btMultiBody stepPositions");
		//integrate and update the Featherstone hierarchies

		for (int b = 0; b < m_multiBodies.size(); b++)
		{
			btMultiBody* bod = m_multiBodies[b];
			bool isSleeping = false;
			if (bod->getBaseCollider() && bod->getBaseCollider()->getActivationState() == ISLAND_SLEEPING)
			{
				isSleeping = true;
			}
			for (int b = 0; b < bod->getNumLinks(); b++)
			{
				if (bod->getLink(b).m_collider && bod->getLink(b).m_collider->getActivationState() == ISLAND_SLEEPING)
					isSleeping = true;
			}

			if (!isSleeping)
			{
				int nLinks = bod->getNumLinks();

				///base + num m_links

				{
					if (!bod->isPosUpdated())
						bod->stepPositionsMultiDof(timeStep);
					else
					{
						btScalar* pRealBuf = const_cast<btScalar*>(bod->getVelocityVector());
						pRealBuf += 6 + bod->getNumDofs() + bod->getNumDofs() * bod->getNumDofs();

						bod->stepPositionsMultiDof(1, 0, pRealBuf);
						bod->setPosUpdated(false);
					}
				}

				m_scratch_world_to_local.resize(nLinks + 1);
				m_scratch_local_origin.resize(nLinks + 1);

				bod->updateCollisionObjectWorldTransforms(m_scratch_world_to_local, m_scratch_local_origin);
			}
			else
			{
				bod->clearVelocities();
			}
		}
	}
}

void btMultiBodyDynamicsWorld::addMultiBodyConstraint(btMultiBodyConstraint* constraint)
{
	m_multiBodyConstraints.push_back(constraint);
}

void btMultiBodyDynamicsWorld::removeMultiBodyConstraint(btMultiBodyConstraint* constraint)
{
	m_multiBodyConstraints.remove(constraint);
}

void btMultiBodyDynamicsWorld::debugDrawMultiBodyConstraint(btMultiBodyConstraint* constraint)
{
	constraint->debugDraw(getDebugDrawer());
}

void btMultiBodyDynamicsWorld::debugDrawWorld()
{
	BT_PROFILE("btMultiBodyDynamicsWorld debugDrawWorld");

	btDiscreteDynamicsWorld::debugDrawWorld();

	bool drawConstraints = false;
	if (getDebugDrawer())
	{
		int mode = getDebugDrawer()->getDebugMode();
		if (mode & (btIDebugDraw::DBG_DrawConstraints | btIDebugDraw::DBG_DrawConstraintLimits))
		{
			drawConstraints = true;
		}

		if (drawConstraints)
		{
			BT_PROFILE("btMultiBody debugDrawWorld");

			for (int c = 0; c < m_multiBodyConstraints.size(); c++)
			{
				btMultiBodyConstraint* constraint = m_multiBodyConstraints[c];
				debugDrawMultiBodyConstraint(constraint);
			}

			for (int b = 0; b < m_multiBodies.size(); b++)
			{
				btMultiBody* bod = m_multiBodies[b];
				bod->forwardKinematics(m_scratch_world_to_local1, m_scratch_local_origin1);

				if (mode & btIDebugDraw::DBG_DrawFrames)
				{
					getDebugDrawer()->drawTransform(bod->getBaseWorldTransform(), 0.1);
				}

				for (int m = 0; m < bod->getNumLinks(); m++)
				{
					const btTransform& tr = bod->getLink(m).m_cachedWorldTransform;
					if (mode & btIDebugDraw::DBG_DrawFrames)
					{
						getDebugDrawer()->drawTransform(tr, 0.1);
					}
					//draw the joint axis
					if (bod->getLink(m).m_jointType == btMultibodyLink::eRevolute)
					{
						btVector3 vec = quatRotate(tr.getRotation(), bod->getLink(m).m_axes[0].m_topVec) * 0.1;

						btVector4 color(0, 0, 0, 1);  //1,1,1);
						btVector3 from = vec + tr.getOrigin() - quatRotate(tr.getRotation(), bod->getLink(m).m_dVector);
						btVector3 to = tr.getOrigin() - quatRotate(tr.getRotation(), bod->getLink(m).m_dVector);
						getDebugDrawer()->drawLine(from, to, color);
					}
					if (bod->getLink(m).m_jointType == btMultibodyLink::eFixed)
					{
						btVector3 vec = quatRotate(tr.getRotation(), bod->getLink(m).m_axes[0].m_bottomVec) * 0.1;

						btVector4 color(0, 0, 0, 1);  //1,1,1);
						btVector3 from = vec + tr.getOrigin() - quatRotate(tr.getRotation(), bod->getLink(m).m_dVector);
						btVector3 to = tr.getOrigin() - quatRotate(tr.getRotation(), bod->getLink(m).m_dVector);
						getDebugDrawer()->drawLine(from, to, color);
					}
					if (bod->getLink(m).m_jointType == btMultibodyLink::ePrismatic)
					{
						btVector3 vec = quatRotate(tr.getRotation(), bod->getLink(m).m_axes[0].m_bottomVec) * 0.1;

						btVector4 color(0, 0, 0, 1);  //1,1,1);
						btVector3 from = vec + tr.getOrigin() - quatRotate(tr.getRotation(), bod->getLink(m).m_dVector);
						btVector3 to = tr.getOrigin() - quatRotate(tr.getRotation(), bod->getLink(m).m_dVector);
						getDebugDrawer()->drawLine(from, to, color);
					}
				}
			}
		}
	}
}

void btMultiBodyDynamicsWorld::applyGravity()
{
	btDiscreteDynamicsWorld::applyGravity();
#ifdef BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY
	BT_PROFILE("btMultiBody addGravity");
	for (int i = 0; i < this->m_multiBodies.size(); i++)
	{
		btMultiBody* bod = m_multiBodies[i];

		bool isSleeping = false;

		if (bod->getBaseCollider() && bod->getBaseCollider()->getActivationState() == ISLAND_SLEEPING)
		{
			isSleeping = true;
		}
		for (int b = 0; b < bod->getNumLinks(); b++)
		{
			if (bod->getLink(b).m_collider && bod->getLink(b).m_collider->getActivationState() == ISLAND_SLEEPING)
				isSleeping = true;
		}

		if (!isSleeping)
		{
			bod->addBaseForce(m_gravity * bod->getBaseMass());

			for (int j = 0; j < bod->getNumLinks(); ++j)
			{
				bod->addLinkForce(j, m_gravity * bod->getLinkMass(j));
			}
		}  //if (!isSleeping)
	}
#endif  //BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY
}

void btMultiBodyDynamicsWorld::clearMultiBodyConstraintForces()
{
	for (int i = 0; i < this->m_multiBodies.size(); i++)
	{
		btMultiBody* bod = m_multiBodies[i];
		bod->clearConstraintForces();
	}
}
void btMultiBodyDynamicsWorld::clearMultiBodyForces()
{
	{
		// BT_PROFILE("clearMultiBodyForces");
		for (int i = 0; i < this->m_multiBodies.size(); i++)
		{
			btMultiBody* bod = m_multiBodies[i];

			bool isSleeping = false;

			if (bod->getBaseCollider() && bod->getBaseCollider()->getActivationState() == ISLAND_SLEEPING)
			{
				isSleeping = true;
			}
			for (int b = 0; b < bod->getNumLinks(); b++)
			{
				if (bod->getLink(b).m_collider && bod->getLink(b).m_collider->getActivationState() == ISLAND_SLEEPING)
					isSleeping = true;
			}

			if (!isSleeping)
			{
				btMultiBody* bod = m_multiBodies[i];
				bod->clearForcesAndTorques();
			}
		}
	}
}
void btMultiBodyDynamicsWorld::clearForces()
{
	btDiscreteDynamicsWorld::clearForces();

#ifdef BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY
	clearMultiBodyForces();
#endif
}

void btMultiBodyDynamicsWorld::serialize(btSerializer* serializer)
{
	serializer->startSerialization();

	serializeDynamicsWorldInfo(serializer);

	serializeMultiBodies(serializer);

	serializeRigidBodies(serializer);

	serializeCollisionObjects(serializer);

	serializeContactManifolds(serializer);

	serializer->finishSerialization();
}

void btMultiBodyDynamicsWorld::serializeMultiBodies(btSerializer* serializer)
{
	int i;
	//serialize all collision objects
	for (i = 0; i < m_multiBodies.size(); i++)
	{
		btMultiBody* mb = m_multiBodies[i];
		{
			int len = mb->calculateSerializeBufferSize();
			btChunk* chunk = serializer->allocate(len, 1);
			const char* structType = mb->serialize(chunk->m_oldPtr, serializer);
			serializer->finalizeChunk(chunk, structType, BT_MULTIBODY_CODE, mb);
		}
	}

	//serialize all multibody links (collision objects)
	for (i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* colObj = m_collisionObjects[i];
		if (colObj->getInternalType() == btCollisionObject::CO_FEATHERSTONE_LINK)
		{
			int len = colObj->calculateSerializeBufferSize();
			btChunk* chunk = serializer->allocate(len, 1);
			const char* structType = colObj->serialize(chunk->m_oldPtr, serializer);
			serializer->finalizeChunk(chunk, structType, BT_MB_LINKCOLLIDER_CODE, colObj);
		}
	}
}
