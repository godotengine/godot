/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_CONSTRAINT_SOLVER_H
#define BT_CONSTRAINT_SOLVER_H

#include "LinearMath/btScalar.h"

class btPersistentManifold;
class btRigidBody;
class btCollisionObject;
class btTypedConstraint;
struct btContactSolverInfo;
struct btBroadphaseProxy;
class btIDebugDraw;
class btStackAlloc;
class btDispatcher;
/// btConstraintSolver provides solver interface

enum btConstraintSolverType
{
	BT_SEQUENTIAL_IMPULSE_SOLVER = 1,
	BT_MLCP_SOLVER = 2,
	BT_NNCG_SOLVER = 4,
	BT_MULTIBODY_SOLVER = 8,
	BT_BLOCK_SOLVER = 16,
};

class btConstraintSolver
{
public:
	virtual ~btConstraintSolver() {}

	virtual void prepareSolve(int /* numBodies */, int /* numManifolds */) { ; }

	///solve a group of constraints
	virtual btScalar solveGroup(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifold, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& info, class btIDebugDraw* debugDrawer, btDispatcher* dispatcher) = 0;

	virtual void allSolved(const btContactSolverInfo& /* info */, class btIDebugDraw* /* debugDrawer */) { ; }

	///clear internal cached data and reset random seed
	virtual void reset() = 0;

	virtual btConstraintSolverType getSolverType() const = 0;
};

#endif  //BT_CONSTRAINT_SOLVER_H
