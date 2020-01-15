/*
 Written by Xuchen Han <xuchenhan2015@u.northwestern.edu>
 
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

#ifndef BT_DEFORMABLE_MULTIBODY_DYNAMICS_WORLD_H
#define BT_DEFORMABLE_MULTIBODY_DYNAMICS_WORLD_H

#include "btSoftMultiBodyDynamicsWorld.h"
#include "btDeformableLagrangianForce.h"
#include "btDeformableMassSpringForce.h"
#include "btDeformableBodySolver.h"
#include "btDeformableMultiBodyConstraintSolver.h"
#include "btSoftBodyHelpers.h"
#include "BulletCollision/CollisionDispatch/btSimulationIslandManager.h"
#include <functional>
typedef btAlignedObjectArray<btSoftBody*> btSoftBodyArray;

class btDeformableBodySolver;
class btDeformableLagrangianForce;
struct MultiBodyInplaceSolverIslandCallback;
struct DeformableBodyInplaceSolverIslandCallback;
class btDeformableMultiBodyConstraintSolver;

typedef btAlignedObjectArray<btSoftBody*> btSoftBodyArray;

class btDeformableMultiBodyDynamicsWorld : public btMultiBodyDynamicsWorld
{
    typedef btAlignedObjectArray<btVector3> TVStack;
    ///Solver classes that encapsulate multiple deformable bodies for solving
    btDeformableBodySolver* m_deformableBodySolver;
    btSoftBodyArray m_softBodies;
    int m_drawFlags;
    bool m_drawNodeTree;
    bool m_drawFaceTree;
    bool m_drawClusterTree;
    btSoftBodyWorldInfo m_sbi;
    btScalar m_internalTime;
    int m_contact_iterations;
    bool m_implicit;
    bool m_lineSearch;
    bool m_selfCollision;
    DeformableBodyInplaceSolverIslandCallback* m_solverDeformableBodyIslandCallback;
    
    typedef void (*btSolverCallback)(btScalar time, btDeformableMultiBodyDynamicsWorld* world);
    btSolverCallback m_solverCallback;
    
protected:
    virtual void internalSingleStepSimulation(btScalar timeStep);
    
    virtual void integrateTransforms(btScalar timeStep);
    
    void positionCorrection(btScalar timeStep);
    
    void solveConstraints(btScalar timeStep);
    
    void updateActivationState(btScalar timeStep);
    
    void clearGravity();
    
public:
	btDeformableMultiBodyDynamicsWorld(btDispatcher* dispatcher, btBroadphaseInterface* pairCache, btDeformableMultiBodyConstraintSolver* constraintSolver, btCollisionConfiguration* collisionConfiguration, btDeformableBodySolver* deformableBodySolver = 0);

    virtual int stepSimulation(btScalar timeStep, int maxSubSteps = 1, btScalar fixedTimeStep = btScalar(1.) / btScalar(60.));

	virtual void debugDrawWorld();

    void setSolverCallback(btSolverCallback cb)
    {
        m_solverCallback = cb;
    }
    
    virtual ~btDeformableMultiBodyDynamicsWorld()
    {
    }
    
    virtual btMultiBodyDynamicsWorld* getMultiBodyDynamicsWorld()
    {
        return (btMultiBodyDynamicsWorld*)(this);
    }
    
    virtual const btMultiBodyDynamicsWorld* getMultiBodyDynamicsWorld() const
    {
        return (const btMultiBodyDynamicsWorld*)(this);
    }
    
    virtual btDynamicsWorldType getWorldType() const
    {
        return BT_DEFORMABLE_MULTIBODY_DYNAMICS_WORLD;
    }
    
    virtual void predictUnconstraintMotion(btScalar timeStep);
    
    virtual void addSoftBody(btSoftBody* body, int collisionFilterGroup = btBroadphaseProxy::DefaultFilter, int collisionFilterMask = btBroadphaseProxy::AllFilter);
    
    btSoftBodyArray& getSoftBodyArray()
    {
        return m_softBodies;
    }
    
    const btSoftBodyArray& getSoftBodyArray() const
    {
        return m_softBodies;
    }
    
    btSoftBodyWorldInfo& getWorldInfo()
    {
        return m_sbi;
    }
    
    const btSoftBodyWorldInfo& getWorldInfo() const
    {
        return m_sbi;
    }
    
    void reinitialize(btScalar timeStep);
    
    void applyRigidBodyGravity(btScalar timeStep);
    
    void beforeSolverCallbacks(btScalar timeStep);
    
    void afterSolverCallbacks(btScalar timeStep);
    
    void addForce(btSoftBody* psb, btDeformableLagrangianForce* force);
    
    void removeSoftBody(btSoftBody* body);
    
    void removeCollisionObject(btCollisionObject* collisionObject);
    
    int getDrawFlags() const { return (m_drawFlags); }
    void setDrawFlags(int f) { m_drawFlags = f; }
    
    void setupConstraints();
    
    void solveMultiBodyConstraints();
    
    void solveContactConstraints();
    
    void sortConstraints();
    
    void softBodySelfCollision();
    
    void setImplicit(bool implicit)
    {
        m_implicit = implicit;
    }
    
    void setLineSearch(bool lineSearch)
    {
        m_lineSearch = lineSearch;
    }

};

#endif  //BT_DEFORMABLE_MULTIBODY_DYNAMICS_WORLD_H
