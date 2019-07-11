/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
///original version written by Erwin Coumans, October 2013

#ifndef BT_MLCP_SOLVER_H
#define BT_MLCP_SOLVER_H

#include "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h"
#include "LinearMath/btMatrixX.h"
#include "BulletDynamics/MLCPSolvers/btMLCPSolverInterface.h"

class btMLCPSolver : public btSequentialImpulseConstraintSolver
{
protected:
	btMatrixXu m_A;
	btVectorXu m_b;
	btVectorXu m_x;
	btVectorXu m_lo;
	btVectorXu m_hi;

	///when using 'split impulse' we solve two separate (M)LCPs
	btVectorXu m_bSplit;
	btVectorXu m_xSplit;
	btVectorXu m_bSplit1;
	btVectorXu m_xSplit2;

	btAlignedObjectArray<int> m_limitDependencies;
	btAlignedObjectArray<btSolverConstraint*> m_allConstraintPtrArray;
	btMLCPSolverInterface* m_solver;
	int m_fallback;

	/// The following scratch variables are not stateful -- contents are cleared prior to each use.
	/// They are only cached here to avoid extra memory allocations and deallocations and to ensure
	/// that multiple instances of the solver can be run in parallel.
	btMatrixXu m_scratchJ3;
	btMatrixXu m_scratchJInvM3;
	btAlignedObjectArray<int> m_scratchOfs;
	btMatrixXu m_scratchMInv;
	btMatrixXu m_scratchJ;
	btMatrixXu m_scratchJTranspose;
	btMatrixXu m_scratchTmp;

	virtual btScalar solveGroupCacheFriendlySetup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer);
	virtual btScalar solveGroupCacheFriendlyIterations(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer);

	virtual void createMLCP(const btContactSolverInfo& infoGlobal);
	virtual void createMLCPFast(const btContactSolverInfo& infoGlobal);

	//return true is it solves the problem successfully
	virtual bool solveMLCP(const btContactSolverInfo& infoGlobal);

public:
	btMLCPSolver(btMLCPSolverInterface* solver);
	virtual ~btMLCPSolver();

	void setMLCPSolver(btMLCPSolverInterface* solver)
	{
		m_solver = solver;
	}

	int getNumFallbacks() const
	{
		return m_fallback;
	}
	void setNumFallbacks(int num)
	{
		m_fallback = num;
	}

	virtual btConstraintSolverType getSolverType() const
	{
		return BT_MLCP_SOLVER;
	}
};

#endif  //BT_MLCP_SOLVER_H
