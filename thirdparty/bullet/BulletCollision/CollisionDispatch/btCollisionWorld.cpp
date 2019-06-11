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

#include "btCollisionWorld.h"
#include "btCollisionDispatcher.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletCollision/CollisionShapes/btConvexShape.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h"
#include "BulletCollision/CollisionShapes/btSphereShape.h"                 //for raycasting
#include "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h"        //for raycasting
#include "BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h"  //for raycasting
#include "BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h"     //for raycasting
#include "BulletCollision/NarrowPhaseCollision/btRaycastCallback.h"
#include "BulletCollision/CollisionShapes/btCompoundShape.h"
#include "BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h"
#include "BulletCollision/NarrowPhaseCollision/btGjkConvexCast.h"
#include "BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.h"
#include "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h"
#include "BulletCollision/BroadphaseCollision/btDbvt.h"
#include "LinearMath/btAabbUtil2.h"
#include "LinearMath/btQuickprof.h"
#include "LinearMath/btSerializer.h"
#include "BulletCollision/CollisionShapes/btConvexPolyhedron.h"
#include "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h"

//#define DISABLE_DBVT_COMPOUNDSHAPE_RAYCAST_ACCELERATION

//#define USE_BRUTEFORCE_RAYBROADPHASE 1
//RECALCULATE_AABB is slower, but benefit is that you don't need to call 'stepSimulation'  or 'updateAabbs' before using a rayTest
//#define RECALCULATE_AABB_RAYCAST 1

//When the user doesn't provide dispatcher or broadphase, create basic versions (and delete them in destructor)
#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btSimpleBroadphase.h"
#include "BulletCollision/CollisionDispatch/btCollisionConfiguration.h"

///for debug drawing

//for debug rendering
#include "BulletCollision/CollisionShapes/btBoxShape.h"
#include "BulletCollision/CollisionShapes/btCapsuleShape.h"
#include "BulletCollision/CollisionShapes/btCompoundShape.h"
#include "BulletCollision/CollisionShapes/btConeShape.h"
#include "BulletCollision/CollisionShapes/btConvexTriangleMeshShape.h"
#include "BulletCollision/CollisionShapes/btCylinderShape.h"
#include "BulletCollision/CollisionShapes/btMultiSphereShape.h"
#include "BulletCollision/CollisionShapes/btPolyhedralConvexShape.h"
#include "BulletCollision/CollisionShapes/btSphereShape.h"
#include "BulletCollision/CollisionShapes/btTriangleCallback.h"
#include "BulletCollision/CollisionShapes/btTriangleMeshShape.h"
#include "BulletCollision/CollisionShapes/btStaticPlaneShape.h"

btCollisionWorld::btCollisionWorld(btDispatcher* dispatcher, btBroadphaseInterface* pairCache, btCollisionConfiguration* collisionConfiguration)
	: m_dispatcher1(dispatcher),
	  m_broadphasePairCache(pairCache),
	  m_debugDrawer(0),
	  m_forceUpdateAllAabbs(true)
{
}

btCollisionWorld::~btCollisionWorld()
{
	//clean up remaining objects
	int i;
	for (i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* collisionObject = m_collisionObjects[i];

		btBroadphaseProxy* bp = collisionObject->getBroadphaseHandle();
		if (bp)
		{
			//
			// only clear the cached algorithms
			//
			getBroadphase()->getOverlappingPairCache()->cleanProxyFromPairs(bp, m_dispatcher1);
			getBroadphase()->destroyProxy(bp, m_dispatcher1);
			collisionObject->setBroadphaseHandle(0);
		}
	}
}

void btCollisionWorld::refreshBroadphaseProxy(btCollisionObject* collisionObject)
{
	if (collisionObject->getBroadphaseHandle())
	{
		int collisionFilterGroup = collisionObject->getBroadphaseHandle()->m_collisionFilterGroup;
		int collisionFilterMask = collisionObject->getBroadphaseHandle()->m_collisionFilterMask;

		getBroadphase()->destroyProxy(collisionObject->getBroadphaseHandle(), getDispatcher());

		//calculate new AABB
		btTransform trans = collisionObject->getWorldTransform();

		btVector3 minAabb;
		btVector3 maxAabb;
		collisionObject->getCollisionShape()->getAabb(trans, minAabb, maxAabb);

		int type = collisionObject->getCollisionShape()->getShapeType();
		collisionObject->setBroadphaseHandle(getBroadphase()->createProxy(
			minAabb,
			maxAabb,
			type,
			collisionObject,
			collisionFilterGroup,
			collisionFilterMask,
			m_dispatcher1));
	}
}

void btCollisionWorld::addCollisionObject(btCollisionObject* collisionObject, int collisionFilterGroup, int collisionFilterMask)
{
	btAssert(collisionObject);

	//check that the object isn't already added
	btAssert(m_collisionObjects.findLinearSearch(collisionObject) == m_collisionObjects.size());
	btAssert(collisionObject->getWorldArrayIndex() == -1);  // do not add the same object to more than one collision world

	collisionObject->setWorldArrayIndex(m_collisionObjects.size());
	m_collisionObjects.push_back(collisionObject);

	//calculate new AABB
	btTransform trans = collisionObject->getWorldTransform();

	btVector3 minAabb;
	btVector3 maxAabb;
	collisionObject->getCollisionShape()->getAabb(trans, minAabb, maxAabb);

	int type = collisionObject->getCollisionShape()->getShapeType();
	collisionObject->setBroadphaseHandle(getBroadphase()->createProxy(
		minAabb,
		maxAabb,
		type,
		collisionObject,
		collisionFilterGroup,
		collisionFilterMask,
		m_dispatcher1));
}

void btCollisionWorld::updateSingleAabb(btCollisionObject* colObj)
{
	btVector3 minAabb, maxAabb;
	colObj->getCollisionShape()->getAabb(colObj->getWorldTransform(), minAabb, maxAabb);
	//need to increase the aabb for contact thresholds
	btVector3 contactThreshold(gContactBreakingThreshold, gContactBreakingThreshold, gContactBreakingThreshold);
	minAabb -= contactThreshold;
	maxAabb += contactThreshold;

	if (getDispatchInfo().m_useContinuous && colObj->getInternalType() == btCollisionObject::CO_RIGID_BODY && !colObj->isStaticOrKinematicObject())
	{
		btVector3 minAabb2, maxAabb2;
		colObj->getCollisionShape()->getAabb(colObj->getInterpolationWorldTransform(), minAabb2, maxAabb2);
		minAabb2 -= contactThreshold;
		maxAabb2 += contactThreshold;
		minAabb.setMin(minAabb2);
		maxAabb.setMax(maxAabb2);
	}

	btBroadphaseInterface* bp = (btBroadphaseInterface*)m_broadphasePairCache;

	//moving objects should be moderately sized, probably something wrong if not
	if (colObj->isStaticObject() || ((maxAabb - minAabb).length2() < btScalar(1e12)))
	{
		bp->setAabb(colObj->getBroadphaseHandle(), minAabb, maxAabb, m_dispatcher1);
	}
	else
	{
		//something went wrong, investigate
		//this assert is unwanted in 3D modelers (danger of loosing work)
		colObj->setActivationState(DISABLE_SIMULATION);

		static bool reportMe = true;
		if (reportMe && m_debugDrawer)
		{
			reportMe = false;
			m_debugDrawer->reportErrorWarning("Overflow in AABB, object removed from simulation");
			m_debugDrawer->reportErrorWarning("If you can reproduce this, please email bugs@continuousphysics.com\n");
			m_debugDrawer->reportErrorWarning("Please include above information, your Platform, version of OS.\n");
			m_debugDrawer->reportErrorWarning("Thanks.\n");
		}
	}
}

void btCollisionWorld::updateAabbs()
{
	BT_PROFILE("updateAabbs");

	btTransform predictedTrans;
	for (int i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* colObj = m_collisionObjects[i];
		btAssert(colObj->getWorldArrayIndex() == i);

		//only update aabb of active objects
		if (m_forceUpdateAllAabbs || colObj->isActive())
		{
			updateSingleAabb(colObj);
		}
	}
}

void btCollisionWorld::computeOverlappingPairs()
{
	BT_PROFILE("calculateOverlappingPairs");
	m_broadphasePairCache->calculateOverlappingPairs(m_dispatcher1);
}

void btCollisionWorld::performDiscreteCollisionDetection()
{
	BT_PROFILE("performDiscreteCollisionDetection");

	btDispatcherInfo& dispatchInfo = getDispatchInfo();

	updateAabbs();

	computeOverlappingPairs();

	btDispatcher* dispatcher = getDispatcher();
	{
		BT_PROFILE("dispatchAllCollisionPairs");
		if (dispatcher)
			dispatcher->dispatchAllCollisionPairs(m_broadphasePairCache->getOverlappingPairCache(), dispatchInfo, m_dispatcher1);
	}
}

void btCollisionWorld::removeCollisionObject(btCollisionObject* collisionObject)
{
	//bool removeFromBroadphase = false;

	{
		btBroadphaseProxy* bp = collisionObject->getBroadphaseHandle();
		if (bp)
		{
			//
			// only clear the cached algorithms
			//
			getBroadphase()->getOverlappingPairCache()->cleanProxyFromPairs(bp, m_dispatcher1);
			getBroadphase()->destroyProxy(bp, m_dispatcher1);
			collisionObject->setBroadphaseHandle(0);
		}
	}

	int iObj = collisionObject->getWorldArrayIndex();
	//    btAssert(iObj >= 0 && iObj < m_collisionObjects.size()); // trying to remove an object that was never added or already removed previously?
	if (iObj >= 0 && iObj < m_collisionObjects.size())
	{
		btAssert(collisionObject == m_collisionObjects[iObj]);
		m_collisionObjects.swap(iObj, m_collisionObjects.size() - 1);
		m_collisionObjects.pop_back();
		if (iObj < m_collisionObjects.size())
		{
			m_collisionObjects[iObj]->setWorldArrayIndex(iObj);
		}
	}
	else
	{
		// slow linear search
		//swapremove
		m_collisionObjects.remove(collisionObject);
	}
	collisionObject->setWorldArrayIndex(-1);
}

void btCollisionWorld::rayTestSingle(const btTransform& rayFromTrans, const btTransform& rayToTrans,
									 btCollisionObject* collisionObject,
									 const btCollisionShape* collisionShape,
									 const btTransform& colObjWorldTransform,
									 RayResultCallback& resultCallback)
{
	btCollisionObjectWrapper colObWrap(0, collisionShape, collisionObject, colObjWorldTransform, -1, -1);
	btCollisionWorld::rayTestSingleInternal(rayFromTrans, rayToTrans, &colObWrap, resultCallback);
}

void btCollisionWorld::rayTestSingleInternal(const btTransform& rayFromTrans, const btTransform& rayToTrans,
											 const btCollisionObjectWrapper* collisionObjectWrap,
											 RayResultCallback& resultCallback)
{
	btSphereShape pointShape(btScalar(0.0));
	pointShape.setMargin(0.f);
	const btConvexShape* castShape = &pointShape;
	const btCollisionShape* collisionShape = collisionObjectWrap->getCollisionShape();
	const btTransform& colObjWorldTransform = collisionObjectWrap->getWorldTransform();

	if (collisionShape->isConvex())
	{
		//		BT_PROFILE("rayTestConvex");
		btConvexCast::CastResult castResult;
		castResult.m_fraction = resultCallback.m_closestHitFraction;

		btConvexShape* convexShape = (btConvexShape*)collisionShape;
		btVoronoiSimplexSolver simplexSolver;
		btSubsimplexConvexCast subSimplexConvexCaster(castShape, convexShape, &simplexSolver);

		btGjkConvexCast gjkConvexCaster(castShape, convexShape, &simplexSolver);

		//btContinuousConvexCollision convexCaster(castShape,convexShape,&simplexSolver,0);

		btConvexCast* convexCasterPtr = 0;
		//use kF_UseSubSimplexConvexCastRaytest by default
		if (resultCallback.m_flags & btTriangleRaycastCallback::kF_UseGjkConvexCastRaytest)
			convexCasterPtr = &gjkConvexCaster;
		else
			convexCasterPtr = &subSimplexConvexCaster;

		btConvexCast& convexCaster = *convexCasterPtr;

		if (convexCaster.calcTimeOfImpact(rayFromTrans, rayToTrans, colObjWorldTransform, colObjWorldTransform, castResult))
		{
			//add hit
			if (castResult.m_normal.length2() > btScalar(0.0001))
			{
				if (castResult.m_fraction < resultCallback.m_closestHitFraction)
				{
					//todo: figure out what this is about. When is rayFromTest.getBasis() not identity?
#ifdef USE_SUBSIMPLEX_CONVEX_CAST
					//rotate normal into worldspace
					castResult.m_normal = rayFromTrans.getBasis() * castResult.m_normal;
#endif  //USE_SUBSIMPLEX_CONVEX_CAST

					castResult.m_normal.normalize();
					btCollisionWorld::LocalRayResult localRayResult(
						collisionObjectWrap->getCollisionObject(),
						0,
						castResult.m_normal,
						castResult.m_fraction);

					bool normalInWorldSpace = true;
					resultCallback.addSingleResult(localRayResult, normalInWorldSpace);
				}
			}
		}
	}
	else
	{
		if (collisionShape->isConcave())
		{
			//ConvexCast::CastResult
			struct BridgeTriangleRaycastCallback : public btTriangleRaycastCallback
			{
				btCollisionWorld::RayResultCallback* m_resultCallback;
				const btCollisionObject* m_collisionObject;
				const btConcaveShape* m_triangleMesh;

				btTransform m_colObjWorldTransform;

				BridgeTriangleRaycastCallback(const btVector3& from, const btVector3& to,
											  btCollisionWorld::RayResultCallback* resultCallback, const btCollisionObject* collisionObject, const btConcaveShape* triangleMesh, const btTransform& colObjWorldTransform) :  //@BP Mod
																																																							btTriangleRaycastCallback(from, to, resultCallback->m_flags),
																																																							m_resultCallback(resultCallback),
																																																							m_collisionObject(collisionObject),
																																																							m_triangleMesh(triangleMesh),
																																																							m_colObjWorldTransform(colObjWorldTransform)
				{
				}

				virtual btScalar reportHit(const btVector3& hitNormalLocal, btScalar hitFraction, int partId, int triangleIndex)
				{
					btCollisionWorld::LocalShapeInfo shapeInfo;
					shapeInfo.m_shapePart = partId;
					shapeInfo.m_triangleIndex = triangleIndex;

					btVector3 hitNormalWorld = m_colObjWorldTransform.getBasis() * hitNormalLocal;

					btCollisionWorld::LocalRayResult rayResult(m_collisionObject,
															   &shapeInfo,
															   hitNormalWorld,
															   hitFraction);

					bool normalInWorldSpace = true;
					return m_resultCallback->addSingleResult(rayResult, normalInWorldSpace);
				}
			};

			btTransform worldTocollisionObject = colObjWorldTransform.inverse();
			btVector3 rayFromLocal = worldTocollisionObject * rayFromTrans.getOrigin();
			btVector3 rayToLocal = worldTocollisionObject * rayToTrans.getOrigin();

			//			BT_PROFILE("rayTestConcave");
			if (collisionShape->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				///optimized version for btBvhTriangleMeshShape
				btBvhTriangleMeshShape* triangleMesh = (btBvhTriangleMeshShape*)collisionShape;

				BridgeTriangleRaycastCallback rcb(rayFromLocal, rayToLocal, &resultCallback, collisionObjectWrap->getCollisionObject(), triangleMesh, colObjWorldTransform);
				rcb.m_hitFraction = resultCallback.m_closestHitFraction;
				triangleMesh->performRaycast(&rcb, rayFromLocal, rayToLocal);
			}
			else if (collisionShape->getShapeType() == SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				///optimized version for btScaledBvhTriangleMeshShape
				btScaledBvhTriangleMeshShape* scaledTriangleMesh = (btScaledBvhTriangleMeshShape*)collisionShape;
				btBvhTriangleMeshShape* triangleMesh = (btBvhTriangleMeshShape*)scaledTriangleMesh->getChildShape();

				//scale the ray positions
				btVector3 scale = scaledTriangleMesh->getLocalScaling();
				btVector3 rayFromLocalScaled = rayFromLocal / scale;
				btVector3 rayToLocalScaled = rayToLocal / scale;

				//perform raycast in the underlying btBvhTriangleMeshShape
				BridgeTriangleRaycastCallback rcb(rayFromLocalScaled, rayToLocalScaled, &resultCallback, collisionObjectWrap->getCollisionObject(), triangleMesh, colObjWorldTransform);
				rcb.m_hitFraction = resultCallback.m_closestHitFraction;
				triangleMesh->performRaycast(&rcb, rayFromLocalScaled, rayToLocalScaled);
			}
			else if (((resultCallback.m_flags&btTriangleRaycastCallback::kF_DisableHeightfieldAccelerator)==0) 
				&& collisionShape->getShapeType() == TERRAIN_SHAPE_PROXYTYPE 
				)
			{
				///optimized version for btHeightfieldTerrainShape
				btHeightfieldTerrainShape* heightField = (btHeightfieldTerrainShape*)collisionShape;
				btTransform worldTocollisionObject = colObjWorldTransform.inverse();
				btVector3 rayFromLocal = worldTocollisionObject * rayFromTrans.getOrigin();
				btVector3 rayToLocal = worldTocollisionObject * rayToTrans.getOrigin();

				BridgeTriangleRaycastCallback rcb(rayFromLocal, rayToLocal, &resultCallback, collisionObjectWrap->getCollisionObject(), heightField, colObjWorldTransform);
				rcb.m_hitFraction = resultCallback.m_closestHitFraction;
				heightField->performRaycast(&rcb, rayFromLocal, rayToLocal);
			}
			else
			{
				//generic (slower) case
				btConcaveShape* concaveShape = (btConcaveShape*)collisionShape;

				btTransform worldTocollisionObject = colObjWorldTransform.inverse();

				btVector3 rayFromLocal = worldTocollisionObject * rayFromTrans.getOrigin();
				btVector3 rayToLocal = worldTocollisionObject * rayToTrans.getOrigin();

				//ConvexCast::CastResult

				struct BridgeTriangleRaycastCallback : public btTriangleRaycastCallback
				{
					btCollisionWorld::RayResultCallback* m_resultCallback;
					const btCollisionObject* m_collisionObject;
					btConcaveShape* m_triangleMesh;

					btTransform m_colObjWorldTransform;

					BridgeTriangleRaycastCallback(const btVector3& from, const btVector3& to,
												  btCollisionWorld::RayResultCallback* resultCallback, const btCollisionObject* collisionObject, btConcaveShape* triangleMesh, const btTransform& colObjWorldTransform) :  //@BP Mod
																																																						  btTriangleRaycastCallback(from, to, resultCallback->m_flags),
																																																						  m_resultCallback(resultCallback),
																																																						  m_collisionObject(collisionObject),
																																																						  m_triangleMesh(triangleMesh),
																																																						  m_colObjWorldTransform(colObjWorldTransform)
					{
					}

					virtual btScalar reportHit(const btVector3& hitNormalLocal, btScalar hitFraction, int partId, int triangleIndex)
					{
						btCollisionWorld::LocalShapeInfo shapeInfo;
						shapeInfo.m_shapePart = partId;
						shapeInfo.m_triangleIndex = triangleIndex;

						btVector3 hitNormalWorld = m_colObjWorldTransform.getBasis() * hitNormalLocal;

						btCollisionWorld::LocalRayResult rayResult(m_collisionObject,
																   &shapeInfo,
																   hitNormalWorld,
																   hitFraction);

						bool normalInWorldSpace = true;
						return m_resultCallback->addSingleResult(rayResult, normalInWorldSpace);
					}
				};

				BridgeTriangleRaycastCallback rcb(rayFromLocal, rayToLocal, &resultCallback, collisionObjectWrap->getCollisionObject(), concaveShape, colObjWorldTransform);
				rcb.m_hitFraction = resultCallback.m_closestHitFraction;

				btVector3 rayAabbMinLocal = rayFromLocal;
				rayAabbMinLocal.setMin(rayToLocal);
				btVector3 rayAabbMaxLocal = rayFromLocal;
				rayAabbMaxLocal.setMax(rayToLocal);

				concaveShape->processAllTriangles(&rcb, rayAabbMinLocal, rayAabbMaxLocal);
			}
		}
		else
		{
			//			BT_PROFILE("rayTestCompound");
			if (collisionShape->isCompound())
			{
				struct LocalInfoAdder2 : public RayResultCallback
				{
					RayResultCallback* m_userCallback;
					int m_i;

					LocalInfoAdder2(int i, RayResultCallback* user)
						: m_userCallback(user), m_i(i)
					{
						m_closestHitFraction = m_userCallback->m_closestHitFraction;
						m_flags = m_userCallback->m_flags;
					}
					virtual bool needsCollision(btBroadphaseProxy* p) const
					{
						return m_userCallback->needsCollision(p);
					}

					virtual btScalar addSingleResult(btCollisionWorld::LocalRayResult& r, bool b)
					{
						btCollisionWorld::LocalShapeInfo shapeInfo;
						shapeInfo.m_shapePart = -1;
						shapeInfo.m_triangleIndex = m_i;
						if (r.m_localShapeInfo == NULL)
							r.m_localShapeInfo = &shapeInfo;

						const btScalar result = m_userCallback->addSingleResult(r, b);
						m_closestHitFraction = m_userCallback->m_closestHitFraction;
						return result;
					}
				};

				struct RayTester : btDbvt::ICollide
				{
					const btCollisionObject* m_collisionObject;
					const btCompoundShape* m_compoundShape;
					const btTransform& m_colObjWorldTransform;
					const btTransform& m_rayFromTrans;
					const btTransform& m_rayToTrans;
					RayResultCallback& m_resultCallback;

					RayTester(const btCollisionObject* collisionObject,
							  const btCompoundShape* compoundShape,
							  const btTransform& colObjWorldTransform,
							  const btTransform& rayFromTrans,
							  const btTransform& rayToTrans,
							  RayResultCallback& resultCallback) : m_collisionObject(collisionObject),
																   m_compoundShape(compoundShape),
																   m_colObjWorldTransform(colObjWorldTransform),
																   m_rayFromTrans(rayFromTrans),
																   m_rayToTrans(rayToTrans),
																   m_resultCallback(resultCallback)
					{
					}

					void ProcessLeaf(int i)
					{
						const btCollisionShape* childCollisionShape = m_compoundShape->getChildShape(i);
						const btTransform& childTrans = m_compoundShape->getChildTransform(i);
						btTransform childWorldTrans = m_colObjWorldTransform * childTrans;

						btCollisionObjectWrapper tmpOb(0, childCollisionShape, m_collisionObject, childWorldTrans, -1, i);
						// replace collision shape so that callback can determine the triangle

						LocalInfoAdder2 my_cb(i, &m_resultCallback);

						rayTestSingleInternal(
							m_rayFromTrans,
							m_rayToTrans,
							&tmpOb,
							my_cb);
					}

					void Process(const btDbvtNode* leaf)
					{
						ProcessLeaf(leaf->dataAsInt);
					}
				};

				const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(collisionShape);
				const btDbvt* dbvt = compoundShape->getDynamicAabbTree();

				RayTester rayCB(
					collisionObjectWrap->getCollisionObject(),
					compoundShape,
					colObjWorldTransform,
					rayFromTrans,
					rayToTrans,
					resultCallback);
#ifndef DISABLE_DBVT_COMPOUNDSHAPE_RAYCAST_ACCELERATION
				if (dbvt)
				{
					btVector3 localRayFrom = colObjWorldTransform.inverseTimes(rayFromTrans).getOrigin();
					btVector3 localRayTo = colObjWorldTransform.inverseTimes(rayToTrans).getOrigin();
					btDbvt::rayTest(dbvt->m_root, localRayFrom, localRayTo, rayCB);
				}
				else
#endif  //DISABLE_DBVT_COMPOUNDSHAPE_RAYCAST_ACCELERATION
				{
					for (int i = 0, n = compoundShape->getNumChildShapes(); i < n; ++i)
					{
						rayCB.ProcessLeaf(i);
					}
				}
			}
		}
	}
}

void btCollisionWorld::objectQuerySingle(const btConvexShape* castShape, const btTransform& convexFromTrans, const btTransform& convexToTrans,
										 btCollisionObject* collisionObject,
										 const btCollisionShape* collisionShape,
										 const btTransform& colObjWorldTransform,
										 ConvexResultCallback& resultCallback, btScalar allowedPenetration)
{
	btCollisionObjectWrapper tmpOb(0, collisionShape, collisionObject, colObjWorldTransform, -1, -1);
	btCollisionWorld::objectQuerySingleInternal(castShape, convexFromTrans, convexToTrans, &tmpOb, resultCallback, allowedPenetration);
}

void btCollisionWorld::objectQuerySingleInternal(const btConvexShape* castShape, const btTransform& convexFromTrans, const btTransform& convexToTrans,
												 const btCollisionObjectWrapper* colObjWrap,
												 ConvexResultCallback& resultCallback, btScalar allowedPenetration)
{
	const btCollisionShape* collisionShape = colObjWrap->getCollisionShape();
	const btTransform& colObjWorldTransform = colObjWrap->getWorldTransform();

	if (collisionShape->isConvex())
	{
		//BT_PROFILE("convexSweepConvex");
		btConvexCast::CastResult castResult;
		castResult.m_allowedPenetration = allowedPenetration;
		castResult.m_fraction = resultCallback.m_closestHitFraction;  //btScalar(1.);//??

		btConvexShape* convexShape = (btConvexShape*)collisionShape;
		btVoronoiSimplexSolver simplexSolver;
		btGjkEpaPenetrationDepthSolver gjkEpaPenetrationSolver;

		btContinuousConvexCollision convexCaster1(castShape, convexShape, &simplexSolver, &gjkEpaPenetrationSolver);
		//btGjkConvexCast convexCaster2(castShape,convexShape,&simplexSolver);
		//btSubsimplexConvexCast convexCaster3(castShape,convexShape,&simplexSolver);

		btConvexCast* castPtr = &convexCaster1;

		if (castPtr->calcTimeOfImpact(convexFromTrans, convexToTrans, colObjWorldTransform, colObjWorldTransform, castResult))
		{
			//add hit
			if (castResult.m_normal.length2() > btScalar(0.0001))
			{
				if (castResult.m_fraction < resultCallback.m_closestHitFraction)
				{
					castResult.m_normal.normalize();
					btCollisionWorld::LocalConvexResult localConvexResult(
						colObjWrap->getCollisionObject(),
						0,
						castResult.m_normal,
						castResult.m_hitPoint,
						castResult.m_fraction);

					bool normalInWorldSpace = true;
					resultCallback.addSingleResult(localConvexResult, normalInWorldSpace);
				}
			}
		}
	}
	else
	{
		if (collisionShape->isConcave())
		{
			if (collisionShape->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				//BT_PROFILE("convexSweepbtBvhTriangleMesh");
				btBvhTriangleMeshShape* triangleMesh = (btBvhTriangleMeshShape*)collisionShape;
				btTransform worldTocollisionObject = colObjWorldTransform.inverse();
				btVector3 convexFromLocal = worldTocollisionObject * convexFromTrans.getOrigin();
				btVector3 convexToLocal = worldTocollisionObject * convexToTrans.getOrigin();
				// rotation of box in local mesh space = MeshRotation^-1 * ConvexToRotation
				btTransform rotationXform = btTransform(worldTocollisionObject.getBasis() * convexToTrans.getBasis());

				//ConvexCast::CastResult
				struct BridgeTriangleConvexcastCallback : public btTriangleConvexcastCallback
				{
					btCollisionWorld::ConvexResultCallback* m_resultCallback;
					const btCollisionObject* m_collisionObject;
					btTriangleMeshShape* m_triangleMesh;

					BridgeTriangleConvexcastCallback(const btConvexShape* castShape, const btTransform& from, const btTransform& to,
													 btCollisionWorld::ConvexResultCallback* resultCallback, const btCollisionObject* collisionObject, btTriangleMeshShape* triangleMesh, const btTransform& triangleToWorld) : btTriangleConvexcastCallback(castShape, from, to, triangleToWorld, triangleMesh->getMargin()),
																																																								m_resultCallback(resultCallback),
																																																								m_collisionObject(collisionObject),
																																																								m_triangleMesh(triangleMesh)
					{
					}

					virtual btScalar reportHit(const btVector3& hitNormalLocal, const btVector3& hitPointLocal, btScalar hitFraction, int partId, int triangleIndex)
					{
						btCollisionWorld::LocalShapeInfo shapeInfo;
						shapeInfo.m_shapePart = partId;
						shapeInfo.m_triangleIndex = triangleIndex;
						if (hitFraction <= m_resultCallback->m_closestHitFraction)
						{
							btCollisionWorld::LocalConvexResult convexResult(m_collisionObject,
																			 &shapeInfo,
																			 hitNormalLocal,
																			 hitPointLocal,
																			 hitFraction);

							bool normalInWorldSpace = true;

							return m_resultCallback->addSingleResult(convexResult, normalInWorldSpace);
						}
						return hitFraction;
					}
				};

				BridgeTriangleConvexcastCallback tccb(castShape, convexFromTrans, convexToTrans, &resultCallback, colObjWrap->getCollisionObject(), triangleMesh, colObjWorldTransform);
				tccb.m_hitFraction = resultCallback.m_closestHitFraction;
				tccb.m_allowedPenetration = allowedPenetration;
				btVector3 boxMinLocal, boxMaxLocal;
				castShape->getAabb(rotationXform, boxMinLocal, boxMaxLocal);
				triangleMesh->performConvexcast(&tccb, convexFromLocal, convexToLocal, boxMinLocal, boxMaxLocal);
			}
			else
			{
				if (collisionShape->getShapeType() == STATIC_PLANE_PROXYTYPE)
				{
					btConvexCast::CastResult castResult;
					castResult.m_allowedPenetration = allowedPenetration;
					castResult.m_fraction = resultCallback.m_closestHitFraction;
					btStaticPlaneShape* planeShape = (btStaticPlaneShape*)collisionShape;
					btContinuousConvexCollision convexCaster1(castShape, planeShape);
					btConvexCast* castPtr = &convexCaster1;

					if (castPtr->calcTimeOfImpact(convexFromTrans, convexToTrans, colObjWorldTransform, colObjWorldTransform, castResult))
					{
						//add hit
						if (castResult.m_normal.length2() > btScalar(0.0001))
						{
							if (castResult.m_fraction < resultCallback.m_closestHitFraction)
							{
								castResult.m_normal.normalize();
								btCollisionWorld::LocalConvexResult localConvexResult(
									colObjWrap->getCollisionObject(),
									0,
									castResult.m_normal,
									castResult.m_hitPoint,
									castResult.m_fraction);

								bool normalInWorldSpace = true;
								resultCallback.addSingleResult(localConvexResult, normalInWorldSpace);
							}
						}
					}
				}
				else
				{
					//BT_PROFILE("convexSweepConcave");
					btConcaveShape* concaveShape = (btConcaveShape*)collisionShape;
					btTransform worldTocollisionObject = colObjWorldTransform.inverse();
					btVector3 convexFromLocal = worldTocollisionObject * convexFromTrans.getOrigin();
					btVector3 convexToLocal = worldTocollisionObject * convexToTrans.getOrigin();
					// rotation of box in local mesh space = MeshRotation^-1 * ConvexToRotation
					btTransform rotationXform = btTransform(worldTocollisionObject.getBasis() * convexToTrans.getBasis());

					//ConvexCast::CastResult
					struct BridgeTriangleConvexcastCallback : public btTriangleConvexcastCallback
					{
						btCollisionWorld::ConvexResultCallback* m_resultCallback;
						const btCollisionObject* m_collisionObject;
						btConcaveShape* m_triangleMesh;

						BridgeTriangleConvexcastCallback(const btConvexShape* castShape, const btTransform& from, const btTransform& to,
														 btCollisionWorld::ConvexResultCallback* resultCallback, const btCollisionObject* collisionObject, btConcaveShape* triangleMesh, const btTransform& triangleToWorld) : btTriangleConvexcastCallback(castShape, from, to, triangleToWorld, triangleMesh->getMargin()),
																																																							   m_resultCallback(resultCallback),
																																																							   m_collisionObject(collisionObject),
																																																							   m_triangleMesh(triangleMesh)
						{
						}

						virtual btScalar reportHit(const btVector3& hitNormalLocal, const btVector3& hitPointLocal, btScalar hitFraction, int partId, int triangleIndex)
						{
							btCollisionWorld::LocalShapeInfo shapeInfo;
							shapeInfo.m_shapePart = partId;
							shapeInfo.m_triangleIndex = triangleIndex;
							if (hitFraction <= m_resultCallback->m_closestHitFraction)
							{
								btCollisionWorld::LocalConvexResult convexResult(m_collisionObject,
																				 &shapeInfo,
																				 hitNormalLocal,
																				 hitPointLocal,
																				 hitFraction);

								bool normalInWorldSpace = true;

								return m_resultCallback->addSingleResult(convexResult, normalInWorldSpace);
							}
							return hitFraction;
						}
					};

					BridgeTriangleConvexcastCallback tccb(castShape, convexFromTrans, convexToTrans, &resultCallback, colObjWrap->getCollisionObject(), concaveShape, colObjWorldTransform);
					tccb.m_hitFraction = resultCallback.m_closestHitFraction;
					tccb.m_allowedPenetration = allowedPenetration;
					btVector3 boxMinLocal, boxMaxLocal;
					castShape->getAabb(rotationXform, boxMinLocal, boxMaxLocal);

					btVector3 rayAabbMinLocal = convexFromLocal;
					rayAabbMinLocal.setMin(convexToLocal);
					btVector3 rayAabbMaxLocal = convexFromLocal;
					rayAabbMaxLocal.setMax(convexToLocal);
					rayAabbMinLocal += boxMinLocal;
					rayAabbMaxLocal += boxMaxLocal;
					concaveShape->processAllTriangles(&tccb, rayAabbMinLocal, rayAabbMaxLocal);
				}
			}
		}
		else
		{
			if (collisionShape->isCompound())
			{
				struct btCompoundLeafCallback : btDbvt::ICollide
				{
					btCompoundLeafCallback(
						const btCollisionObjectWrapper* colObjWrap,
						const btConvexShape* castShape,
						const btTransform& convexFromTrans,
						const btTransform& convexToTrans,
						btScalar allowedPenetration,
						const btCompoundShape* compoundShape,
						const btTransform& colObjWorldTransform,
						ConvexResultCallback& resultCallback)
						: m_colObjWrap(colObjWrap),
						  m_castShape(castShape),
						  m_convexFromTrans(convexFromTrans),
						  m_convexToTrans(convexToTrans),
						  m_allowedPenetration(allowedPenetration),
						  m_compoundShape(compoundShape),
						  m_colObjWorldTransform(colObjWorldTransform),
						  m_resultCallback(resultCallback)
					{
					}

					const btCollisionObjectWrapper* m_colObjWrap;
					const btConvexShape* m_castShape;
					const btTransform& m_convexFromTrans;
					const btTransform& m_convexToTrans;
					btScalar m_allowedPenetration;
					const btCompoundShape* m_compoundShape;
					const btTransform& m_colObjWorldTransform;
					ConvexResultCallback& m_resultCallback;

				public:
					void ProcessChild(int index, const btTransform& childTrans, const btCollisionShape* childCollisionShape)
					{
						btTransform childWorldTrans = m_colObjWorldTransform * childTrans;

						struct LocalInfoAdder : public ConvexResultCallback
						{
							ConvexResultCallback* m_userCallback;
							int m_i;

							LocalInfoAdder(int i, ConvexResultCallback* user)
								: m_userCallback(user), m_i(i)
							{
								m_closestHitFraction = m_userCallback->m_closestHitFraction;
							}
							virtual bool needsCollision(btBroadphaseProxy* p) const
							{
								return m_userCallback->needsCollision(p);
							}
							virtual btScalar addSingleResult(btCollisionWorld::LocalConvexResult& r, bool b)
							{
								btCollisionWorld::LocalShapeInfo shapeInfo;
								shapeInfo.m_shapePart = -1;
								shapeInfo.m_triangleIndex = m_i;
								if (r.m_localShapeInfo == NULL)
									r.m_localShapeInfo = &shapeInfo;
								const btScalar result = m_userCallback->addSingleResult(r, b);
								m_closestHitFraction = m_userCallback->m_closestHitFraction;
								return result;
							}
						};

						LocalInfoAdder my_cb(index, &m_resultCallback);

						btCollisionObjectWrapper tmpObj(m_colObjWrap, childCollisionShape, m_colObjWrap->getCollisionObject(), childWorldTrans, -1, index);

						objectQuerySingleInternal(m_castShape, m_convexFromTrans, m_convexToTrans, &tmpObj, my_cb, m_allowedPenetration);
					}

					void Process(const btDbvtNode* leaf)
					{
						// Processing leaf node
						int index = leaf->dataAsInt;

						btTransform childTrans = m_compoundShape->getChildTransform(index);
						const btCollisionShape* childCollisionShape = m_compoundShape->getChildShape(index);

						ProcessChild(index, childTrans, childCollisionShape);
					}
				};

				BT_PROFILE("convexSweepCompound");
				const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(collisionShape);

				btVector3 fromLocalAabbMin, fromLocalAabbMax;
				btVector3 toLocalAabbMin, toLocalAabbMax;

				castShape->getAabb(colObjWorldTransform.inverse() * convexFromTrans, fromLocalAabbMin, fromLocalAabbMax);
				castShape->getAabb(colObjWorldTransform.inverse() * convexToTrans, toLocalAabbMin, toLocalAabbMax);

				fromLocalAabbMin.setMin(toLocalAabbMin);
				fromLocalAabbMax.setMax(toLocalAabbMax);

				btCompoundLeafCallback callback(colObjWrap, castShape, convexFromTrans, convexToTrans,
												allowedPenetration, compoundShape, colObjWorldTransform, resultCallback);

				const btDbvt* tree = compoundShape->getDynamicAabbTree();
				if (tree)
				{
					const ATTRIBUTE_ALIGNED16(btDbvtVolume) bounds = btDbvtVolume::FromMM(fromLocalAabbMin, fromLocalAabbMax);
					tree->collideTV(tree->m_root, bounds, callback);
				}
				else
				{
					int i;
					for (i = 0; i < compoundShape->getNumChildShapes(); i++)
					{
						const btCollisionShape* childCollisionShape = compoundShape->getChildShape(i);
						btTransform childTrans = compoundShape->getChildTransform(i);
						callback.ProcessChild(i, childTrans, childCollisionShape);
					}
				}
			}
		}
	}
}

struct btSingleRayCallback : public btBroadphaseRayCallback
{
	btVector3 m_rayFromWorld;
	btVector3 m_rayToWorld;
	btTransform m_rayFromTrans;
	btTransform m_rayToTrans;
	btVector3 m_hitNormal;

	const btCollisionWorld* m_world;
	btCollisionWorld::RayResultCallback& m_resultCallback;

	btSingleRayCallback(const btVector3& rayFromWorld, const btVector3& rayToWorld, const btCollisionWorld* world, btCollisionWorld::RayResultCallback& resultCallback)
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
		///what about division by zero? --> just set rayDirection[i] to INF/BT_LARGE_FLOAT
		m_rayDirectionInverse[0] = rayDir[0] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[0];
		m_rayDirectionInverse[1] = rayDir[1] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[1];
		m_rayDirectionInverse[2] = rayDir[2] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[2];
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

void btCollisionWorld::rayTest(const btVector3& rayFromWorld, const btVector3& rayToWorld, RayResultCallback& resultCallback) const
{
	//BT_PROFILE("rayTest");
	/// use the broadphase to accelerate the search for objects, based on their aabb
	/// and for each object with ray-aabb overlap, perform an exact ray test
	btSingleRayCallback rayCB(rayFromWorld, rayToWorld, this, resultCallback);

#ifndef USE_BRUTEFORCE_RAYBROADPHASE
	m_broadphasePairCache->rayTest(rayFromWorld, rayToWorld, rayCB);
#else
	for (int i = 0; i < this->getNumCollisionObjects(); i++)
	{
		rayCB.process(m_collisionObjects[i]->getBroadphaseHandle());
	}
#endif  //USE_BRUTEFORCE_RAYBROADPHASE
}

struct btSingleSweepCallback : public btBroadphaseRayCallback
{
	btTransform m_convexFromTrans;
	btTransform m_convexToTrans;
	btVector3 m_hitNormal;
	const btCollisionWorld* m_world;
	btCollisionWorld::ConvexResultCallback& m_resultCallback;
	btScalar m_allowedCcdPenetration;
	const btConvexShape* m_castShape;

	btSingleSweepCallback(const btConvexShape* castShape, const btTransform& convexFromTrans, const btTransform& convexToTrans, const btCollisionWorld* world, btCollisionWorld::ConvexResultCallback& resultCallback, btScalar allowedPenetration)
		: m_convexFromTrans(convexFromTrans),
		  m_convexToTrans(convexToTrans),
		  m_world(world),
		  m_resultCallback(resultCallback),
		  m_allowedCcdPenetration(allowedPenetration),
		  m_castShape(castShape)
	{
		btVector3 unnormalizedRayDir = (m_convexToTrans.getOrigin() - m_convexFromTrans.getOrigin());
		btVector3 rayDir = unnormalizedRayDir.normalized();
		///what about division by zero? --> just set rayDirection[i] to INF/BT_LARGE_FLOAT
		m_rayDirectionInverse[0] = rayDir[0] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[0];
		m_rayDirectionInverse[1] = rayDir[1] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[1];
		m_rayDirectionInverse[2] = rayDir[2] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[2];
		m_signs[0] = m_rayDirectionInverse[0] < 0.0;
		m_signs[1] = m_rayDirectionInverse[1] < 0.0;
		m_signs[2] = m_rayDirectionInverse[2] < 0.0;

		m_lambda_max = rayDir.dot(unnormalizedRayDir);
	}

	virtual bool process(const btBroadphaseProxy* proxy)
	{
		///terminate further convex sweep tests, once the closestHitFraction reached zero
		if (m_resultCallback.m_closestHitFraction == btScalar(0.f))
			return false;

		btCollisionObject* collisionObject = (btCollisionObject*)proxy->m_clientObject;

		//only perform raycast if filterMask matches
		if (m_resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			//RigidcollisionObject* collisionObject = ctrl->GetRigidcollisionObject();
			m_world->objectQuerySingle(m_castShape, m_convexFromTrans, m_convexToTrans,
									   collisionObject,
									   collisionObject->getCollisionShape(),
									   collisionObject->getWorldTransform(),
									   m_resultCallback,
									   m_allowedCcdPenetration);
		}

		return true;
	}
};

void btCollisionWorld::convexSweepTest(const btConvexShape* castShape, const btTransform& convexFromWorld, const btTransform& convexToWorld, ConvexResultCallback& resultCallback, btScalar allowedCcdPenetration) const
{
	BT_PROFILE("convexSweepTest");
	/// use the broadphase to accelerate the search for objects, based on their aabb
	/// and for each object with ray-aabb overlap, perform an exact ray test
	/// unfortunately the implementation for rayTest and convexSweepTest duplicated, albeit practically identical

	btTransform convexFromTrans, convexToTrans;
	convexFromTrans = convexFromWorld;
	convexToTrans = convexToWorld;
	btVector3 castShapeAabbMin, castShapeAabbMax;
	/* Compute AABB that encompasses angular movement */
	{
		btVector3 linVel, angVel;
		btTransformUtil::calculateVelocity(convexFromTrans, convexToTrans, 1.0f, linVel, angVel);
		btVector3 zeroLinVel;
		zeroLinVel.setValue(0, 0, 0);
		btTransform R;
		R.setIdentity();
		R.setRotation(convexFromTrans.getRotation());
		castShape->calculateTemporalAabb(R, zeroLinVel, angVel, 1.0f, castShapeAabbMin, castShapeAabbMax);
	}

#ifndef USE_BRUTEFORCE_RAYBROADPHASE

	btSingleSweepCallback convexCB(castShape, convexFromWorld, convexToWorld, this, resultCallback, allowedCcdPenetration);

	m_broadphasePairCache->rayTest(convexFromTrans.getOrigin(), convexToTrans.getOrigin(), convexCB, castShapeAabbMin, castShapeAabbMax);

#else
	/// go over all objects, and if the ray intersects their aabb + cast shape aabb,
	// do a ray-shape query using convexCaster (CCD)
	int i;
	for (i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* collisionObject = m_collisionObjects[i];
		//only perform raycast if filterMask matches
		if (resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			//RigidcollisionObject* collisionObject = ctrl->GetRigidcollisionObject();
			btVector3 collisionObjectAabbMin, collisionObjectAabbMax;
			collisionObject->getCollisionShape()->getAabb(collisionObject->getWorldTransform(), collisionObjectAabbMin, collisionObjectAabbMax);
			AabbExpand(collisionObjectAabbMin, collisionObjectAabbMax, castShapeAabbMin, castShapeAabbMax);
			btScalar hitLambda = btScalar(1.);  //could use resultCallback.m_closestHitFraction, but needs testing
			btVector3 hitNormal;
			if (btRayAabb(convexFromWorld.getOrigin(), convexToWorld.getOrigin(), collisionObjectAabbMin, collisionObjectAabbMax, hitLambda, hitNormal))
			{
				objectQuerySingle(castShape, convexFromTrans, convexToTrans,
								  collisionObject,
								  collisionObject->getCollisionShape(),
								  collisionObject->getWorldTransform(),
								  resultCallback,
								  allowedCcdPenetration);
			}
		}
	}
#endif  //USE_BRUTEFORCE_RAYBROADPHASE
}

struct btBridgedManifoldResult : public btManifoldResult
{
	btCollisionWorld::ContactResultCallback& m_resultCallback;

	btBridgedManifoldResult(const btCollisionObjectWrapper* obj0Wrap, const btCollisionObjectWrapper* obj1Wrap, btCollisionWorld::ContactResultCallback& resultCallback)
		: btManifoldResult(obj0Wrap, obj1Wrap),
		  m_resultCallback(resultCallback)
	{
	}

	virtual void addContactPoint(const btVector3& normalOnBInWorld, const btVector3& pointInWorld, btScalar depth)
	{
		bool isSwapped = m_manifoldPtr->getBody0() != m_body0Wrap->getCollisionObject();
		btVector3 pointA = pointInWorld + normalOnBInWorld * depth;
		btVector3 localA;
		btVector3 localB;
		if (isSwapped)
		{
			localA = m_body1Wrap->getCollisionObject()->getWorldTransform().invXform(pointA);
			localB = m_body0Wrap->getCollisionObject()->getWorldTransform().invXform(pointInWorld);
		}
		else
		{
			localA = m_body0Wrap->getCollisionObject()->getWorldTransform().invXform(pointA);
			localB = m_body1Wrap->getCollisionObject()->getWorldTransform().invXform(pointInWorld);
		}

		btManifoldPoint newPt(localA, localB, normalOnBInWorld, depth);
		newPt.m_positionWorldOnA = pointA;
		newPt.m_positionWorldOnB = pointInWorld;

		//BP mod, store contact triangles.
		if (isSwapped)
		{
			newPt.m_partId0 = m_partId1;
			newPt.m_partId1 = m_partId0;
			newPt.m_index0 = m_index1;
			newPt.m_index1 = m_index0;
		}
		else
		{
			newPt.m_partId0 = m_partId0;
			newPt.m_partId1 = m_partId1;
			newPt.m_index0 = m_index0;
			newPt.m_index1 = m_index1;
		}

		//experimental feature info, for per-triangle material etc.
		const btCollisionObjectWrapper* obj0Wrap = isSwapped ? m_body1Wrap : m_body0Wrap;
		const btCollisionObjectWrapper* obj1Wrap = isSwapped ? m_body0Wrap : m_body1Wrap;
		m_resultCallback.addSingleResult(newPt, obj0Wrap, newPt.m_partId0, newPt.m_index0, obj1Wrap, newPt.m_partId1, newPt.m_index1);
	}
};

struct btSingleContactCallback : public btBroadphaseAabbCallback
{
	btCollisionObject* m_collisionObject;
	btCollisionWorld* m_world;
	btCollisionWorld::ContactResultCallback& m_resultCallback;

	btSingleContactCallback(btCollisionObject* collisionObject, btCollisionWorld* world, btCollisionWorld::ContactResultCallback& resultCallback)
		: m_collisionObject(collisionObject),
		  m_world(world),
		  m_resultCallback(resultCallback)
	{
	}

	virtual bool process(const btBroadphaseProxy* proxy)
	{
		btCollisionObject* collisionObject = (btCollisionObject*)proxy->m_clientObject;
		if (collisionObject == m_collisionObject)
			return true;

		//only perform raycast if filterMask matches
		if (m_resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			btCollisionObjectWrapper ob0(0, m_collisionObject->getCollisionShape(), m_collisionObject, m_collisionObject->getWorldTransform(), -1, -1);
			btCollisionObjectWrapper ob1(0, collisionObject->getCollisionShape(), collisionObject, collisionObject->getWorldTransform(), -1, -1);

			btCollisionAlgorithm* algorithm = m_world->getDispatcher()->findAlgorithm(&ob0, &ob1, 0, BT_CLOSEST_POINT_ALGORITHMS);
			if (algorithm)
			{
				btBridgedManifoldResult contactPointResult(&ob0, &ob1, m_resultCallback);
				//discrete collision detection query

				algorithm->processCollision(&ob0, &ob1, m_world->getDispatchInfo(), &contactPointResult);

				algorithm->~btCollisionAlgorithm();
				m_world->getDispatcher()->freeCollisionAlgorithm(algorithm);
			}
		}
		return true;
	}
};

///contactTest performs a discrete collision test against all objects in the btCollisionWorld, and calls the resultCallback.
///it reports one or more contact points for every overlapping object (including the one with deepest penetration)
void btCollisionWorld::contactTest(btCollisionObject* colObj, ContactResultCallback& resultCallback)
{
	btVector3 aabbMin, aabbMax;
	colObj->getCollisionShape()->getAabb(colObj->getWorldTransform(), aabbMin, aabbMax);
	btSingleContactCallback contactCB(colObj, this, resultCallback);

	m_broadphasePairCache->aabbTest(aabbMin, aabbMax, contactCB);
}

///contactTest performs a discrete collision test between two collision objects and calls the resultCallback if overlap if detected.
///it reports one or more contact points (including the one with deepest penetration)
void btCollisionWorld::contactPairTest(btCollisionObject* colObjA, btCollisionObject* colObjB, ContactResultCallback& resultCallback)
{
	btCollisionObjectWrapper obA(0, colObjA->getCollisionShape(), colObjA, colObjA->getWorldTransform(), -1, -1);
	btCollisionObjectWrapper obB(0, colObjB->getCollisionShape(), colObjB, colObjB->getWorldTransform(), -1, -1);

	btCollisionAlgorithm* algorithm = getDispatcher()->findAlgorithm(&obA, &obB, 0, BT_CLOSEST_POINT_ALGORITHMS);
	if (algorithm)
	{
		btBridgedManifoldResult contactPointResult(&obA, &obB, resultCallback);
		contactPointResult.m_closestPointDistanceThreshold = resultCallback.m_closestDistanceThreshold;
		//discrete collision detection query
		algorithm->processCollision(&obA, &obB, getDispatchInfo(), &contactPointResult);

		algorithm->~btCollisionAlgorithm();
		getDispatcher()->freeCollisionAlgorithm(algorithm);
	}
}

class DebugDrawcallback : public btTriangleCallback, public btInternalTriangleIndexCallback
{
	btIDebugDraw* m_debugDrawer;
	btVector3 m_color;
	btTransform m_worldTrans;

public:
	DebugDrawcallback(btIDebugDraw* debugDrawer, const btTransform& worldTrans, const btVector3& color) : m_debugDrawer(debugDrawer),
																										  m_color(color),
																										  m_worldTrans(worldTrans)
	{
	}

	virtual void internalProcessTriangleIndex(btVector3* triangle, int partId, int triangleIndex)
	{
		processTriangle(triangle, partId, triangleIndex);
	}

	virtual void processTriangle(btVector3* triangle, int partId, int triangleIndex)
	{
		(void)partId;
		(void)triangleIndex;

		btVector3 wv0, wv1, wv2;
		wv0 = m_worldTrans * triangle[0];
		wv1 = m_worldTrans * triangle[1];
		wv2 = m_worldTrans * triangle[2];
		btVector3 center = (wv0 + wv1 + wv2) * btScalar(1. / 3.);

		if (m_debugDrawer->getDebugMode() & btIDebugDraw::DBG_DrawNormals)
		{
			btVector3 normal = (wv1 - wv0).cross(wv2 - wv0);
			normal.normalize();
			btVector3 normalColor(1, 1, 0);
			m_debugDrawer->drawLine(center, center + normal, normalColor);
		}
		m_debugDrawer->drawLine(wv0, wv1, m_color);
		m_debugDrawer->drawLine(wv1, wv2, m_color);
		m_debugDrawer->drawLine(wv2, wv0, m_color);
	}
};

void btCollisionWorld::debugDrawObject(const btTransform& worldTransform, const btCollisionShape* shape, const btVector3& color)
{
	// Draw a small simplex at the center of the object
	if (getDebugDrawer() && getDebugDrawer()->getDebugMode() & btIDebugDraw::DBG_DrawFrames)
	{
		getDebugDrawer()->drawTransform(worldTransform, .1);
	}

	if (shape->getShapeType() == COMPOUND_SHAPE_PROXYTYPE)
	{
		const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(shape);
		for (int i = compoundShape->getNumChildShapes() - 1; i >= 0; i--)
		{
			btTransform childTrans = compoundShape->getChildTransform(i);
			const btCollisionShape* colShape = compoundShape->getChildShape(i);
			debugDrawObject(worldTransform * childTrans, colShape, color);
		}
	}
	else
	{
		switch (shape->getShapeType())
		{
			case BOX_SHAPE_PROXYTYPE:
			{
				const btBoxShape* boxShape = static_cast<const btBoxShape*>(shape);
				btVector3 halfExtents = boxShape->getHalfExtentsWithMargin();
				getDebugDrawer()->drawBox(-halfExtents, halfExtents, worldTransform, color);
				break;
			}

			case SPHERE_SHAPE_PROXYTYPE:
			{
				const btSphereShape* sphereShape = static_cast<const btSphereShape*>(shape);
				btScalar radius = sphereShape->getMargin();  //radius doesn't include the margin, so draw with margin

				getDebugDrawer()->drawSphere(radius, worldTransform, color);
				break;
			}
			case MULTI_SPHERE_SHAPE_PROXYTYPE:
			{
				const btMultiSphereShape* multiSphereShape = static_cast<const btMultiSphereShape*>(shape);

				btTransform childTransform;
				childTransform.setIdentity();

				for (int i = multiSphereShape->getSphereCount() - 1; i >= 0; i--)
				{
					childTransform.setOrigin(multiSphereShape->getSpherePosition(i));
					getDebugDrawer()->drawSphere(multiSphereShape->getSphereRadius(i), worldTransform * childTransform, color);
				}

				break;
			}
			case CAPSULE_SHAPE_PROXYTYPE:
			{
				const btCapsuleShape* capsuleShape = static_cast<const btCapsuleShape*>(shape);

				btScalar radius = capsuleShape->getRadius();
				btScalar halfHeight = capsuleShape->getHalfHeight();

				int upAxis = capsuleShape->getUpAxis();
				getDebugDrawer()->drawCapsule(radius, halfHeight, upAxis, worldTransform, color);
				break;
			}
			case CONE_SHAPE_PROXYTYPE:
			{
				const btConeShape* coneShape = static_cast<const btConeShape*>(shape);
				btScalar radius = coneShape->getRadius();  //+coneShape->getMargin();
				btScalar height = coneShape->getHeight();  //+coneShape->getMargin();

				int upAxis = coneShape->getConeUpIndex();
				getDebugDrawer()->drawCone(radius, height, upAxis, worldTransform, color);
				break;
			}
			case CYLINDER_SHAPE_PROXYTYPE:
			{
				const btCylinderShape* cylinder = static_cast<const btCylinderShape*>(shape);
				int upAxis = cylinder->getUpAxis();
				btScalar radius = cylinder->getRadius();
				btScalar halfHeight = cylinder->getHalfExtentsWithMargin()[upAxis];
				getDebugDrawer()->drawCylinder(radius, halfHeight, upAxis, worldTransform, color);
				break;
			}

			case STATIC_PLANE_PROXYTYPE:
			{
				const btStaticPlaneShape* staticPlaneShape = static_cast<const btStaticPlaneShape*>(shape);
				btScalar planeConst = staticPlaneShape->getPlaneConstant();
				const btVector3& planeNormal = staticPlaneShape->getPlaneNormal();
				getDebugDrawer()->drawPlane(planeNormal, planeConst, worldTransform, color);
				break;
			}
			default:
			{
				/// for polyhedral shapes
				if (shape->isPolyhedral())
				{
					btPolyhedralConvexShape* polyshape = (btPolyhedralConvexShape*)shape;

					int i;
					if (polyshape->getConvexPolyhedron())
					{
						const btConvexPolyhedron* poly = polyshape->getConvexPolyhedron();
						for (i = 0; i < poly->m_faces.size(); i++)
						{
							btVector3 centroid(0, 0, 0);
							int numVerts = poly->m_faces[i].m_indices.size();
							if (numVerts)
							{
								int lastV = poly->m_faces[i].m_indices[numVerts - 1];
								for (int v = 0; v < poly->m_faces[i].m_indices.size(); v++)
								{
									int curVert = poly->m_faces[i].m_indices[v];
									centroid += poly->m_vertices[curVert];
									getDebugDrawer()->drawLine(worldTransform * poly->m_vertices[lastV], worldTransform * poly->m_vertices[curVert], color);
									lastV = curVert;
								}
							}
							centroid *= btScalar(1.f) / btScalar(numVerts);
							if (getDebugDrawer()->getDebugMode() & btIDebugDraw::DBG_DrawNormals)
							{
								btVector3 normalColor(1, 1, 0);
								btVector3 faceNormal(poly->m_faces[i].m_plane[0], poly->m_faces[i].m_plane[1], poly->m_faces[i].m_plane[2]);
								getDebugDrawer()->drawLine(worldTransform * centroid, worldTransform * (centroid + faceNormal), normalColor);
							}
						}
					}
					else
					{
						for (i = 0; i < polyshape->getNumEdges(); i++)
						{
							btVector3 a, b;
							polyshape->getEdge(i, a, b);
							btVector3 wa = worldTransform * a;
							btVector3 wb = worldTransform * b;
							getDebugDrawer()->drawLine(wa, wb, color);
						}
					}
				}

				if (shape->isConcave())
				{
					btConcaveShape* concaveMesh = (btConcaveShape*)shape;

					///@todo pass camera, for some culling? no -> we are not a graphics lib
					btVector3 aabbMax(btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT));
					btVector3 aabbMin(btScalar(-BT_LARGE_FLOAT), btScalar(-BT_LARGE_FLOAT), btScalar(-BT_LARGE_FLOAT));

					DebugDrawcallback drawCallback(getDebugDrawer(), worldTransform, color);
					concaveMesh->processAllTriangles(&drawCallback, aabbMin, aabbMax);
				}

				if (shape->getShapeType() == CONVEX_TRIANGLEMESH_SHAPE_PROXYTYPE)
				{
					btConvexTriangleMeshShape* convexMesh = (btConvexTriangleMeshShape*)shape;
					//todo: pass camera for some culling
					btVector3 aabbMax(btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT));
					btVector3 aabbMin(btScalar(-BT_LARGE_FLOAT), btScalar(-BT_LARGE_FLOAT), btScalar(-BT_LARGE_FLOAT));
					//DebugDrawcallback drawCallback;
					DebugDrawcallback drawCallback(getDebugDrawer(), worldTransform, color);
					convexMesh->getMeshInterface()->InternalProcessAllTriangles(&drawCallback, aabbMin, aabbMax);
				}
			}
		}
	}
}

void btCollisionWorld::debugDrawWorld()
{
	if (getDebugDrawer())
	{
		getDebugDrawer()->clearLines();

		btIDebugDraw::DefaultColors defaultColors = getDebugDrawer()->getDefaultColors();

		if (getDebugDrawer()->getDebugMode() & btIDebugDraw::DBG_DrawContactPoints)
		{
			if (getDispatcher())
			{
				int numManifolds = getDispatcher()->getNumManifolds();

				for (int i = 0; i < numManifolds; i++)
				{
					btPersistentManifold* contactManifold = getDispatcher()->getManifoldByIndexInternal(i);
					//btCollisionObject* obA = static_cast<btCollisionObject*>(contactManifold->getBody0());
					//btCollisionObject* obB = static_cast<btCollisionObject*>(contactManifold->getBody1());

					int numContacts = contactManifold->getNumContacts();
					for (int j = 0; j < numContacts; j++)
					{
						btManifoldPoint& cp = contactManifold->getContactPoint(j);
						getDebugDrawer()->drawContactPoint(cp.m_positionWorldOnB, cp.m_normalWorldOnB, cp.getDistance(), cp.getLifeTime(), defaultColors.m_contactPoint);
					}
				}
			}
		}

		if ((getDebugDrawer()->getDebugMode() & (btIDebugDraw::DBG_DrawWireframe | btIDebugDraw::DBG_DrawAabb)))
		{
			int i;

			for (i = 0; i < m_collisionObjects.size(); i++)
			{
				btCollisionObject* colObj = m_collisionObjects[i];
				if ((colObj->getCollisionFlags() & btCollisionObject::CF_DISABLE_VISUALIZE_OBJECT) == 0)
				{
					if (getDebugDrawer() && (getDebugDrawer()->getDebugMode() & btIDebugDraw::DBG_DrawWireframe))
					{
						btVector3 color(btScalar(0.4), btScalar(0.4), btScalar(0.4));

						switch (colObj->getActivationState())
						{
							case ACTIVE_TAG:
								color = defaultColors.m_activeObject;
								break;
							case ISLAND_SLEEPING:
								color = defaultColors.m_deactivatedObject;
								break;
							case WANTS_DEACTIVATION:
								color = defaultColors.m_wantsDeactivationObject;
								break;
							case DISABLE_DEACTIVATION:
								color = defaultColors.m_disabledDeactivationObject;
								break;
							case DISABLE_SIMULATION:
								color = defaultColors.m_disabledSimulationObject;
								break;
							default:
							{
								color = btVector3(btScalar(.3), btScalar(0.3), btScalar(0.3));
							}
						};

						colObj->getCustomDebugColor(color);

						debugDrawObject(colObj->getWorldTransform(), colObj->getCollisionShape(), color);
					}
					if (m_debugDrawer && (m_debugDrawer->getDebugMode() & btIDebugDraw::DBG_DrawAabb))
					{
						btVector3 minAabb, maxAabb;
						btVector3 colorvec = defaultColors.m_aabb;
						colObj->getCollisionShape()->getAabb(colObj->getWorldTransform(), minAabb, maxAabb);
						btVector3 contactThreshold(gContactBreakingThreshold, gContactBreakingThreshold, gContactBreakingThreshold);
						minAabb -= contactThreshold;
						maxAabb += contactThreshold;

						btVector3 minAabb2, maxAabb2;

						if (getDispatchInfo().m_useContinuous && colObj->getInternalType() == btCollisionObject::CO_RIGID_BODY && !colObj->isStaticOrKinematicObject())
						{
							colObj->getCollisionShape()->getAabb(colObj->getInterpolationWorldTransform(), minAabb2, maxAabb2);
							minAabb2 -= contactThreshold;
							maxAabb2 += contactThreshold;
							minAabb.setMin(minAabb2);
							maxAabb.setMax(maxAabb2);
						}

						m_debugDrawer->drawAabb(minAabb, maxAabb, colorvec);
					}
				}
			}
		}
	}
}

void btCollisionWorld::serializeCollisionObjects(btSerializer* serializer)
{
	int i;

	///keep track of shapes already serialized
	btHashMap<btHashPtr, btCollisionShape*> serializedShapes;

	for (i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* colObj = m_collisionObjects[i];
		btCollisionShape* shape = colObj->getCollisionShape();

		if (!serializedShapes.find(shape))
		{
			serializedShapes.insert(shape, shape);
			shape->serializeSingleShape(serializer);
		}
	}

	//serialize all collision objects
	for (i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* colObj = m_collisionObjects[i];
		if (colObj->getInternalType() == btCollisionObject::CO_COLLISION_OBJECT)
		{
			colObj->serializeSingleObject(serializer);
		}
	}
}

void btCollisionWorld::serializeContactManifolds(btSerializer* serializer)
{
	if (serializer->getSerializationFlags() & BT_SERIALIZE_CONTACT_MANIFOLDS)
	{
		int numManifolds = getDispatcher()->getNumManifolds();
		for (int i = 0; i < numManifolds; i++)
		{
			const btPersistentManifold* manifold = getDispatcher()->getInternalManifoldPointer()[i];
			//don't serialize empty manifolds, they just take space
			//(may have to do it anyway if it destroys determinism)
			if (manifold->getNumContacts() == 0)
				continue;

			btChunk* chunk = serializer->allocate(manifold->calculateSerializeBufferSize(), 1);
			const char* structType = manifold->serialize(manifold, chunk->m_oldPtr, serializer);
			serializer->finalizeChunk(chunk, structType, BT_CONTACTMANIFOLD_CODE, (void*)manifold);
		}
	}
}

void btCollisionWorld::serialize(btSerializer* serializer)
{
	serializer->startSerialization();

	serializeCollisionObjects(serializer);

	serializeContactManifolds(serializer);

	serializer->finishSerialization();
}
