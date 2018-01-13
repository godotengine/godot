/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef BT_DISCRETE_DYNAMICS_WORLD_MT_H
#define BT_DISCRETE_DYNAMICS_WORLD_MT_H

#include "btDiscreteDynamicsWorld.h"
#include "btSimulationIslandManagerMt.h"
#include "BulletDynamics/ConstraintSolver/btConstraintSolver.h"

struct InplaceSolverIslandCallbackMt;

///
/// btConstraintSolverPoolMt - masquerades as a constraint solver, but really it is a threadsafe pool of them.
///
///  Each solver in the pool is protected by a mutex.  When solveGroup is called from a thread,
///  the pool looks for a solver that isn't being used by another thread, locks it, and dispatches the
///  call to the solver.
///  So long as there are at least as many solvers as there are hardware threads, it should never need to
///  spin wait.
///
class btConstraintSolverPoolMt : public btConstraintSolver
{
public:
    // create the solvers for me
    explicit btConstraintSolverPoolMt( int numSolvers );

    // pass in fully constructed solvers (destructor will delete them)
    btConstraintSolverPoolMt( btConstraintSolver** solvers, int numSolvers );

    virtual ~btConstraintSolverPoolMt();

    ///solve a group of constraints
    virtual btScalar solveGroup( btCollisionObject** bodies,
        int numBodies,
        btPersistentManifold** manifolds,
        int numManifolds,
        btTypedConstraint** constraints,
        int numConstraints,
        const btContactSolverInfo& info,
        btIDebugDraw* debugDrawer,
        btDispatcher* dispatcher
    ) BT_OVERRIDE;

    virtual	void reset() BT_OVERRIDE;
    virtual btConstraintSolverType getSolverType() const BT_OVERRIDE { return m_solverType; }

private:
    const static size_t kCacheLineSize = 128;
    struct ThreadSolver
    {
        btConstraintSolver* solver;
        btSpinMutex mutex;
        char _cachelinePadding[ kCacheLineSize - sizeof( btSpinMutex ) - sizeof( void* ) ];  // keep mutexes from sharing a cache line
    };
    btAlignedObjectArray<ThreadSolver> m_solvers;
    btConstraintSolverType m_solverType;

    ThreadSolver* getAndLockThreadSolver();
    void init( btConstraintSolver** solvers, int numSolvers );
};



///
/// btDiscreteDynamicsWorldMt -- a version of DiscreteDynamicsWorld with some minor changes to support
///                              solving simulation islands on multiple threads.
///
///  Should function exactly like btDiscreteDynamicsWorld.
///  Also 3 methods that iterate over all of the rigidbodies can run in parallel:
///     - predictUnconstraintMotion
///     - integrateTransforms
///     - createPredictiveContacts
///
ATTRIBUTE_ALIGNED16(class) btDiscreteDynamicsWorldMt : public btDiscreteDynamicsWorld
{
protected:
    InplaceSolverIslandCallbackMt* m_solverIslandCallbackMt;

    virtual void solveConstraints(btContactSolverInfo& solverInfo) BT_OVERRIDE;

    virtual void predictUnconstraintMotion( btScalar timeStep ) BT_OVERRIDE;

    struct UpdaterCreatePredictiveContacts : public btIParallelForBody
    {
        btScalar timeStep;
        btRigidBody** rigidBodies;
        btDiscreteDynamicsWorldMt* world;

        void forLoop( int iBegin, int iEnd ) const BT_OVERRIDE
        {
            world->createPredictiveContactsInternal( &rigidBodies[ iBegin ], iEnd - iBegin, timeStep );
        }
    };
    virtual void createPredictiveContacts( btScalar timeStep ) BT_OVERRIDE;

    struct UpdaterIntegrateTransforms : public btIParallelForBody
    {
        btScalar timeStep;
        btRigidBody** rigidBodies;
        btDiscreteDynamicsWorldMt* world;

        void forLoop( int iBegin, int iEnd ) const BT_OVERRIDE
        {
            world->integrateTransformsInternal( &rigidBodies[ iBegin ], iEnd - iBegin, timeStep );
        }
    };
    virtual void integrateTransforms( btScalar timeStep ) BT_OVERRIDE;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	btDiscreteDynamicsWorldMt(btDispatcher* dispatcher,
        btBroadphaseInterface* pairCache,
        btConstraintSolverPoolMt* constraintSolver,   // Note this should be a solver-pool for multi-threading
        btCollisionConfiguration* collisionConfiguration
    );
	virtual ~btDiscreteDynamicsWorldMt();
};

#endif //BT_DISCRETE_DYNAMICS_WORLD_H
