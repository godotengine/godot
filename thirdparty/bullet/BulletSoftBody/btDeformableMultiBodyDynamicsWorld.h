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
	int m_ccdIterations;
	bool m_implicit;
	bool m_lineSearch;
	bool m_useProjection;
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

	virtual ~btDeformableMultiBodyDynamicsWorld();

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

	void removeForce(btSoftBody* psb, btDeformableLagrangianForce* force);

	void removeSoftBodyForce(btSoftBody* psb);

	void removeSoftBody(btSoftBody* body);

	void removeCollisionObject(btCollisionObject* collisionObject);

	int getDrawFlags() const { return (m_drawFlags); }
	void setDrawFlags(int f) { m_drawFlags = f; }

	void setupConstraints();

	void performDeformableCollisionDetection();

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

	void setUseProjection(bool useProjection)
	{
		m_useProjection = useProjection;
	}

	void applyRepulsionForce(btScalar timeStep);

	void performGeometricCollisions(btScalar timeStep);

	struct btDeformableSingleRayCallback : public btBroadphaseRayCallback
	{
		btVector3 m_rayFromWorld;
		btVector3 m_rayToWorld;
		btTransform m_rayFromTrans;
		btTransform m_rayToTrans;
		btVector3 m_hitNormal;

		const btDeformableMultiBodyDynamicsWorld* m_world;
		btCollisionWorld::RayResultCallback& m_resultCallback;

		btDeformableSingleRayCallback(const btVector3& rayFromWorld, const btVector3& rayToWorld, const btDeformableMultiBodyDynamicsWorld* world, btCollisionWorld::RayResultCallback& resultCallback)
			: m_rayFromWorld(rayFromWorld),
			  m_rayToWorld(rayToWorld),
			  m_world(world),
			  m_resultCallback(resultCallback)
		{
			m_rayFromTrans.setIdentity();
			m_rayFromTrans.setOrigin(m_rayFromWorld);
			m_rayToTrans.setIdentity();
			m_rayToTrans.setOrigin(m_rayToWorld);

			btVector3 rayDir = (rayToWorld - rayFromWorld);

			rayDir.normalize();
			///what about division by zero? --> just set rayDirection[i] to INF/1e30
			m_rayDirectionInverse[0] = rayDir[0] == btScalar(0.0) ? btScalar(1e30) : btScalar(1.0) / rayDir[0];
			m_rayDirectionInverse[1] = rayDir[1] == btScalar(0.0) ? btScalar(1e30) : btScalar(1.0) / rayDir[1];
			m_rayDirectionInverse[2] = rayDir[2] == btScalar(0.0) ? btScalar(1e30) : btScalar(1.0) / rayDir[2];
			m_signs[0] = m_rayDirectionInverse[0] < 0.0;
			m_signs[1] = m_rayDirectionInverse[1] < 0.0;
			m_signs[2] = m_rayDirectionInverse[2] < 0.0;

			m_lambda_max = rayDir.dot(m_rayToWorld - m_rayFromWorld);
		}

		virtual bool process(const btBroadphaseProxy* proxy)
		{
			///terminate further ray tests, once the closestHitFraction reached zero
			if (m_resultCallback.m_closestHitFraction == btScalar(0.f))
				return false;

			btCollisionObject* collisionObject = (btCollisionObject*)proxy->m_clientObject;

			//only perform raycast if filterMask matches
			if (m_resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
			{
				//RigidcollisionObject* collisionObject = ctrl->GetRigidcollisionObject();
				//btVector3 collisionObjectAabbMin,collisionObjectAabbMax;
#if 0
#ifdef RECALCULATE_AABB
                btVector3 collisionObjectAabbMin,collisionObjectAabbMax;
                collisionObject->getCollisionShape()->getAabb(collisionObject->getWorldTransform(),collisionObjectAabbMin,collisionObjectAabbMax);
#else
                //getBroadphase()->getAabb(collisionObject->getBroadphaseHandle(),collisionObjectAabbMin,collisionObjectAabbMax);
                const btVector3& collisionObjectAabbMin = collisionObject->getBroadphaseHandle()->m_aabbMin;
                const btVector3& collisionObjectAabbMax = collisionObject->getBroadphaseHandle()->m_aabbMax;
#endif
#endif
				//btScalar hitLambda = m_resultCallback.m_closestHitFraction;
				//culling already done by broadphase
				//if (btRayAabb(m_rayFromWorld,m_rayToWorld,collisionObjectAabbMin,collisionObjectAabbMax,hitLambda,m_hitNormal))
				{
					m_world->rayTestSingle(m_rayFromTrans, m_rayToTrans,
										   collisionObject,
										   collisionObject->getCollisionShape(),
										   collisionObject->getWorldTransform(),
										   m_resultCallback);
				}
			}
			return true;
		}
	};

	void rayTest(const btVector3& rayFromWorld, const btVector3& rayToWorld, RayResultCallback& resultCallback) const
	{
		BT_PROFILE("rayTest");
		/// use the broadphase to accelerate the search for objects, based on their aabb
		/// and for each object with ray-aabb overlap, perform an exact ray test
		btDeformableSingleRayCallback rayCB(rayFromWorld, rayToWorld, this, resultCallback);

#ifndef USE_BRUTEFORCE_RAYBROADPHASE
		m_broadphasePairCache->rayTest(rayFromWorld, rayToWorld, rayCB);
#else
		for (int i = 0; i < this->getNumCollisionObjects(); i++)
		{
			rayCB.process(m_collisionObjects[i]->getBroadphaseHandle());
		}
#endif  //USE_BRUTEFORCE_RAYBROADPHASE
	}

	void rayTestSingle(const btTransform& rayFromTrans, const btTransform& rayToTrans,
					   btCollisionObject* collisionObject,
					   const btCollisionShape* collisionShape,
					   const btTransform& colObjWorldTransform,
					   RayResultCallback& resultCallback) const
	{
		if (collisionShape->isSoftBody())
		{
			btSoftBody* softBody = btSoftBody::upcast(collisionObject);
			if (softBody)
			{
				btSoftBody::sRayCast softResult;
				if (softBody->rayFaceTest(rayFromTrans.getOrigin(), rayToTrans.getOrigin(), softResult))
				{
					if (softResult.fraction <= resultCallback.m_closestHitFraction)
					{
						btCollisionWorld::LocalShapeInfo shapeInfo;
						shapeInfo.m_shapePart = 0;
						shapeInfo.m_triangleIndex = softResult.index;
						// get the normal
						btVector3 rayDir = rayToTrans.getOrigin() - rayFromTrans.getOrigin();
						btVector3 normal = -rayDir;
						normal.normalize();
						{
							normal = softBody->m_faces[softResult.index].m_normal;
							if (normal.dot(rayDir) > 0)
							{
								// normal always point toward origin of the ray
								normal = -normal;
							}
						}

						btCollisionWorld::LocalRayResult rayResult(collisionObject,
																   &shapeInfo,
																   normal,
																   softResult.fraction);
						bool normalInWorldSpace = true;
						resultCallback.addSingleResult(rayResult, normalInWorldSpace);
					}
				}
			}
		}
		else
		{
			btCollisionWorld::rayTestSingle(rayFromTrans, rayToTrans, collisionObject, collisionShape, colObjWorldTransform, resultCallback);
		}
	}
};

#endif  //BT_DEFORMABLE_MULTIBODY_DYNAMICS_WORLD_H
