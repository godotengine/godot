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

#ifndef BT_MULTIBODY_DYNAMICS_WORLD_H
#define BT_MULTIBODY_DYNAMICS_WORLD_H

#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"

#define BT_USE_VIRTUAL_CLEARFORCES_AND_GRAVITY

class btMultiBody;
class btMultiBodyConstraint;
class btMultiBodyConstraintSolver;
struct MultiBodyInplaceSolverIslandCallback;

///The btMultiBodyDynamicsWorld adds Featherstone multi body dynamics to Bullet
///This implementation is still preliminary/experimental.
class btMultiBodyDynamicsWorld : public btDiscreteDynamicsWorld
{
protected:
	btAlignedObjectArray<btMultiBody*> m_multiBodies;
	btAlignedObjectArray<btMultiBodyConstraint*> m_multiBodyConstraints;
	btAlignedObjectArray<btMultiBodyConstraint*> m_sortedMultiBodyConstraints;
	btMultiBodyConstraintSolver*	m_multiBodyConstraintSolver;
	MultiBodyInplaceSolverIslandCallback*	m_solverMultiBodyIslandCallback;

	//cached data to avoid memory allocations
	btAlignedObjectArray<btQuaternion> m_scratch_world_to_local;
	btAlignedObjectArray<btVector3> m_scratch_local_origin;
	btAlignedObjectArray<btQuaternion> m_scratch_world_to_local1;
	btAlignedObjectArray<btVector3> m_scratch_local_origin1;
	btAlignedObjectArray<btScalar> m_scratch_r;
	btAlignedObjectArray<btVector3> m_scratch_v;
	btAlignedObjectArray<btMatrix3x3> m_scratch_m;

	
	virtual void	calculateSimulationIslands();
	virtual void	updateActivationState(btScalar timeStep);
	virtual void	solveConstraints(btContactSolverInfo& solverInfo);
	
	virtual void	serializeMultiBodies(btSerializer* serializer);

public:

	btMultiBodyDynamicsWorld(btDispatcher* dispatcher,btBroadphaseInterface* pairCache,btMultiBodyConstraintSolver* constraintSolver,btCollisionConfiguration* collisionConfiguration);

	virtual ~btMultiBodyDynamicsWorld ();

	virtual void	addMultiBody(btMultiBody* body, int group= btBroadphaseProxy::DefaultFilter, int mask=btBroadphaseProxy::AllFilter);

	virtual void	removeMultiBody(btMultiBody* body);

	virtual int		getNumMultibodies() const
	{
		return m_multiBodies.size();
	}

	btMultiBody*	getMultiBody(int mbIndex)
	{
		return m_multiBodies[mbIndex];
	}

	virtual void	addMultiBodyConstraint( btMultiBodyConstraint* constraint);

	virtual int     getNumMultiBodyConstraints() const
	{
        return m_multiBodyConstraints.size();
	}

	virtual btMultiBodyConstraint*	getMultiBodyConstraint( int constraintIndex)
	{
        return m_multiBodyConstraints[constraintIndex];
	}

	virtual const btMultiBodyConstraint*	getMultiBodyConstraint( int constraintIndex) const
	{
        return m_multiBodyConstraints[constraintIndex];
	}

	virtual void	removeMultiBodyConstraint( btMultiBodyConstraint* constraint);

	virtual void	integrateTransforms(btScalar timeStep);

	virtual void	debugDrawWorld();

	virtual void	debugDrawMultiBodyConstraint(btMultiBodyConstraint* constraint);
	
	void	forwardKinematics();
	virtual void clearForces();
	virtual void clearMultiBodyConstraintForces();
	virtual void clearMultiBodyForces();
	virtual void applyGravity();
	
	virtual	void	serialize(btSerializer* serializer);

};
#endif //BT_MULTIBODY_DYNAMICS_WORLD_H
