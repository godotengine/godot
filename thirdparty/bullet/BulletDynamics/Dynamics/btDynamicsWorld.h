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

#ifndef BT_DYNAMICS_WORLD_H
#define BT_DYNAMICS_WORLD_H

#include "BulletCollision/CollisionDispatch/btCollisionWorld.h"
#include "BulletDynamics/ConstraintSolver/btContactSolverInfo.h"

class btTypedConstraint;
class btActionInterface;
class btConstraintSolver;
class btDynamicsWorld;

/// Type for the callback for each tick
typedef void (*btInternalTickCallback)(btDynamicsWorld* world, btScalar timeStep);

enum btDynamicsWorldType
{
	BT_SIMPLE_DYNAMICS_WORLD = 1,
	BT_DISCRETE_DYNAMICS_WORLD = 2,
	BT_CONTINUOUS_DYNAMICS_WORLD = 3,
	BT_SOFT_RIGID_DYNAMICS_WORLD = 4,
	BT_GPU_DYNAMICS_WORLD = 5,
	BT_SOFT_MULTIBODY_DYNAMICS_WORLD = 6,
    BT_DEFORMABLE_MULTIBODY_DYNAMICS_WORLD = 7
};

///The btDynamicsWorld is the interface class for several dynamics implementation, basic, discrete, parallel, and continuous etc.
class btDynamicsWorld : public btCollisionWorld
{
protected:
	btInternalTickCallback m_internalTickCallback;
	btInternalTickCallback m_internalPreTickCallback;
	void* m_worldUserInfo;

	btContactSolverInfo m_solverInfo;

public:
	btDynamicsWorld(btDispatcher* dispatcher, btBroadphaseInterface* broadphase, btCollisionConfiguration* collisionConfiguration)
		: btCollisionWorld(dispatcher, broadphase, collisionConfiguration), m_internalTickCallback(0), m_internalPreTickCallback(0), m_worldUserInfo(0)
	{
	}

	virtual ~btDynamicsWorld()
	{
	}

	///stepSimulation proceeds the simulation over 'timeStep', units in preferably in seconds.
	///By default, Bullet will subdivide the timestep in constant substeps of each 'fixedTimeStep'.
	///in order to keep the simulation real-time, the maximum number of substeps can be clamped to 'maxSubSteps'.
	///You can disable subdividing the timestep/substepping by passing maxSubSteps=0 as second argument to stepSimulation, but in that case you have to keep the timeStep constant.
	virtual int stepSimulation(btScalar timeStep, int maxSubSteps = 1, btScalar fixedTimeStep = btScalar(1.) / btScalar(60.)) = 0;

	virtual void debugDrawWorld() = 0;

	virtual void addConstraint(btTypedConstraint* constraint, bool disableCollisionsBetweenLinkedBodies = false)
	{
		(void)constraint;
		(void)disableCollisionsBetweenLinkedBodies;
	}

	virtual void removeConstraint(btTypedConstraint* constraint) { (void)constraint; }

	virtual void addAction(btActionInterface* action) = 0;

	virtual void removeAction(btActionInterface* action) = 0;

	//once a rigidbody is added to the dynamics world, it will get this gravity assigned
	//existing rigidbodies in the world get gravity assigned too, during this method
	virtual void setGravity(const btVector3& gravity) = 0;
	virtual btVector3 getGravity() const = 0;

	virtual void synchronizeMotionStates() = 0;

	virtual void addRigidBody(btRigidBody* body) = 0;

	virtual void addRigidBody(btRigidBody* body, int group, int mask) = 0;

	virtual void removeRigidBody(btRigidBody* body) = 0;

	virtual void setConstraintSolver(btConstraintSolver* solver) = 0;

	virtual btConstraintSolver* getConstraintSolver() = 0;

	virtual int getNumConstraints() const { return 0; }

	virtual btTypedConstraint* getConstraint(int index)
	{
		(void)index;
		return 0;
	}

	virtual const btTypedConstraint* getConstraint(int index) const
	{
		(void)index;
		return 0;
	}

	virtual btDynamicsWorldType getWorldType() const = 0;

	virtual void clearForces() = 0;

	/// Set the callback for when an internal tick (simulation substep) happens, optional user info
	void setInternalTickCallback(btInternalTickCallback cb, void* worldUserInfo = 0, bool isPreTick = false)
	{
		if (isPreTick)
		{
			m_internalPreTickCallback = cb;
		}
		else
		{
			m_internalTickCallback = cb;
		}
		m_worldUserInfo = worldUserInfo;
	}

	void setWorldUserInfo(void* worldUserInfo)
	{
		m_worldUserInfo = worldUserInfo;
	}

	void* getWorldUserInfo() const
	{
		return m_worldUserInfo;
	}

	btContactSolverInfo& getSolverInfo()
	{
		return m_solverInfo;
	}

	const btContactSolverInfo& getSolverInfo() const
	{
		return m_solverInfo;
	}

	///obsolete, use addAction instead.
	virtual void addVehicle(btActionInterface* vehicle) { (void)vehicle; }
	///obsolete, use removeAction instead
	virtual void removeVehicle(btActionInterface* vehicle) { (void)vehicle; }
	///obsolete, use addAction instead.
	virtual void addCharacter(btActionInterface* character) { (void)character; }
	///obsolete, use removeAction instead
	virtual void removeCharacter(btActionInterface* character) { (void)character; }
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btDynamicsWorldDoubleData
{
	btContactSolverInfoDoubleData m_solverInfo;
	btVector3DoubleData m_gravity;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btDynamicsWorldFloatData
{
	btContactSolverInfoFloatData m_solverInfo;
	btVector3FloatData m_gravity;
};

#endif  //BT_DYNAMICS_WORLD_H
