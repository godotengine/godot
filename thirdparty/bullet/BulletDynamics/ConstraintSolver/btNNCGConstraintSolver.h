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

#ifndef BT_NNCG_CONSTRAINT_SOLVER_H
#define BT_NNCG_CONSTRAINT_SOLVER_H

#include "btSequentialImpulseConstraintSolver.h"

ATTRIBUTE_ALIGNED16(class)
btNNCGConstraintSolver : public btSequentialImpulseConstraintSolver
{
protected:
	btScalar m_deltafLengthSqrPrev;

	btAlignedObjectArray<btScalar> m_pNC;   // p for None Contact constraints
	btAlignedObjectArray<btScalar> m_pC;    // p for Contact constraints
	btAlignedObjectArray<btScalar> m_pCF;   // p for ContactFriction constraints
	btAlignedObjectArray<btScalar> m_pCRF;  // p for ContactRollingFriction constraints

	//These are recalculated in every iterations. We just keep these to prevent reallocation in each iteration.
	btAlignedObjectArray<btScalar> m_deltafNC;   // deltaf for NoneContact constraints
	btAlignedObjectArray<btScalar> m_deltafC;    // deltaf for Contact constraints
	btAlignedObjectArray<btScalar> m_deltafCF;   // deltaf for ContactFriction constraints
	btAlignedObjectArray<btScalar> m_deltafCRF;  // deltaf for ContactRollingFriction constraints

protected:
	virtual btScalar solveGroupCacheFriendlyFinish(btCollisionObject * *bodies, int numBodies, const btContactSolverInfo& infoGlobal);
	virtual btScalar solveSingleIteration(int iteration, btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer);

	virtual btScalar solveGroupCacheFriendlySetup(btCollisionObject * *bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer);

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btNNCGConstraintSolver() : btSequentialImpulseConstraintSolver(), m_onlyForNoneContact(false) {}

	virtual btConstraintSolverType getSolverType() const
	{
		return BT_NNCG_SOLVER;
	}

	bool m_onlyForNoneContact;
};

#endif  //BT_NNCG_CONSTRAINT_SOLVER_H
