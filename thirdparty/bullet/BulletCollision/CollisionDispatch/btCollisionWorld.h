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


/**
 * @mainpage Bullet Documentation
 *
 * @section intro_sec Introduction
 * Bullet is a Collision Detection and Rigid Body Dynamics Library. The Library is Open Source and free for commercial use, under the ZLib license ( http://opensource.org/licenses/zlib-license.php ).
 *
 * The main documentation is Bullet_User_Manual.pdf, included in the source code distribution.
 * There is the Physics Forum for feedback and general Collision Detection and Physics discussions.
 * Please visit http://www.bulletphysics.org
 *
 * @section install_sec Installation
 *
 * @subsection step1 Step 1: Download
 * You can download the Bullet Physics Library from the github repository: https://github.com/bulletphysics/bullet3/releases 
 *
 * @subsection step2 Step 2: Building
 * Bullet has multiple build systems, including premake, cmake and autotools. Premake and cmake support all platforms.
 * Premake is included in the Bullet/build folder for Windows, Mac OSX and Linux. 
 * Under Windows you can click on Bullet/build/vs2010.bat to create Microsoft Visual Studio projects. 
 * On Mac OSX and Linux you can open a terminal and generate Makefile, codeblocks or Xcode4 projects:
 * cd Bullet/build
 * ./premake4_osx gmake or ./premake4_linux gmake or ./premake4_linux64 gmake or (for Mac) ./premake4_osx xcode4
 * cd Bullet/build/gmake
 * make
 * 
 * An alternative to premake is cmake. You can download cmake from http://www.cmake.org
 * cmake can autogenerate projectfiles for Microsoft Visual Studio, Apple Xcode, KDevelop and Unix Makefiles.
 * The easiest is to run the CMake cmake-gui graphical user interface and choose the options and generate projectfiles.
 * You can also use cmake in the command-line. Here are some examples for various platforms:
 * cmake . -G "Visual Studio 9 2008"
 * cmake . -G Xcode
 * cmake . -G "Unix Makefiles"
 * Although cmake is recommended, you can also use autotools for UNIX: ./autogen.sh ./configure to create a Makefile and then run make.
 * 
 * @subsection step3 Step 3: Testing demos
 * Try to run and experiment with BasicDemo executable as a starting point.
 * Bullet can be used in several ways, as Full Rigid Body simulation, as Collision Detector Library or Low Level / Snippets like the GJK Closest Point calculation.
 * The Dependencies can be seen in this documentation under Directories
 * 
 * @subsection step4 Step 4: Integrating in your application, full Rigid Body and Soft Body simulation
 * Check out BasicDemo how to create a btDynamicsWorld, btRigidBody and btCollisionShape, Stepping the simulation and synchronizing your graphics object transform.
 * Check out SoftDemo how to use soft body dynamics, using btSoftRigidDynamicsWorld.
 * @subsection step5 Step 5 : Integrate the Collision Detection Library (without Dynamics and other Extras)
 * Bullet Collision Detection can also be used without the Dynamics/Extras.
 * Check out btCollisionWorld and btCollisionObject, and the CollisionInterfaceDemo.
 * @subsection step6 Step 6 : Use Snippets like the GJK Closest Point calculation.
 * Bullet has been designed in a modular way keeping dependencies to a minimum. The ConvexHullDistance demo demonstrates direct use of btGjkPairDetector.
 *
 * @section copyright Copyright
 * For up-to-data information and copyright and contributors list check out the Bullet_User_Manual.pdf
 * 
 */
 
 

#ifndef BT_COLLISION_WORLD_H
#define BT_COLLISION_WORLD_H

class btCollisionShape;
class btConvexShape;
class btBroadphaseInterface;
class btSerializer;

#include "LinearMath/btVector3.h"
#include "LinearMath/btTransform.h"
#include "btCollisionObject.h"
#include "btCollisionDispatcher.h"
#include "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h"
#include "LinearMath/btAlignedObjectArray.h"

///CollisionWorld is interface and container for the collision detection
class btCollisionWorld
{

	
protected:

	btAlignedObjectArray<btCollisionObject*>	m_collisionObjects;
	
	btDispatcher*	m_dispatcher1;

	btDispatcherInfo	m_dispatchInfo;

	btBroadphaseInterface*	m_broadphasePairCache;

	btIDebugDraw*	m_debugDrawer;

	///m_forceUpdateAllAabbs can be set to false as an optimization to only update active object AABBs
	///it is true by default, because it is error-prone (setting the position of static objects wouldn't update their AABB)
	bool m_forceUpdateAllAabbs;

	void	serializeCollisionObjects(btSerializer* serializer);

	void serializeContactManifolds(btSerializer* serializer);


public:

	//this constructor doesn't own the dispatcher and paircache/broadphase
	btCollisionWorld(btDispatcher* dispatcher,btBroadphaseInterface* broadphasePairCache, btCollisionConfiguration* collisionConfiguration);

	virtual ~btCollisionWorld();

	void	setBroadphase(btBroadphaseInterface*	pairCache)
	{
		m_broadphasePairCache = pairCache;
	}

	const btBroadphaseInterface*	getBroadphase() const
	{
		return m_broadphasePairCache;
	}

	btBroadphaseInterface*	getBroadphase()
	{
		return m_broadphasePairCache;
	}

	btOverlappingPairCache*	getPairCache()
	{
		return m_broadphasePairCache->getOverlappingPairCache();
	}


	btDispatcher*	getDispatcher()
	{
		return m_dispatcher1;
	}

	const btDispatcher*	getDispatcher() const
	{
		return m_dispatcher1;
	}

	void	updateSingleAabb(btCollisionObject* colObj);

	virtual void	updateAabbs();

	///the computeOverlappingPairs is usually already called by performDiscreteCollisionDetection (or stepSimulation)
	///it can be useful to use if you perform ray tests without collision detection/simulation
	virtual void	computeOverlappingPairs();

	
	virtual void	setDebugDrawer(btIDebugDraw*	debugDrawer)
	{
			m_debugDrawer = debugDrawer;
	}

	virtual btIDebugDraw*	getDebugDrawer()
	{
		return m_debugDrawer;
	}

	virtual void	debugDrawWorld();

	virtual void debugDrawObject(const btTransform& worldTransform, const btCollisionShape* shape, const btVector3& color);


	///LocalShapeInfo gives extra information for complex shapes
	///Currently, only btTriangleMeshShape is available, so it just contains triangleIndex and subpart
	struct	LocalShapeInfo
	{
		int	m_shapePart;
		int	m_triangleIndex;
		
		//const btCollisionShape*	m_shapeTemp;
		//const btTransform*	m_shapeLocalTransform;
	};

	struct	LocalRayResult
	{
		LocalRayResult(const btCollisionObject*	collisionObject, 
			LocalShapeInfo*	localShapeInfo,
			const btVector3&		hitNormalLocal,
			btScalar hitFraction)
		:m_collisionObject(collisionObject),
		m_localShapeInfo(localShapeInfo),
		m_hitNormalLocal(hitNormalLocal),
		m_hitFraction(hitFraction)
		{
		}

		const btCollisionObject*		m_collisionObject;
		LocalShapeInfo*			m_localShapeInfo;
		btVector3				m_hitNormalLocal;
		btScalar				m_hitFraction;

	};

	///RayResultCallback is used to report new raycast results
	struct	RayResultCallback
	{
		btScalar	m_closestHitFraction;
		const btCollisionObject*		m_collisionObject;
		int	m_collisionFilterGroup;
		int	m_collisionFilterMask;
		//@BP Mod - Custom flags, currently used to enable backface culling on tri-meshes, see btRaycastCallback.h. Apply any of the EFlags defined there on m_flags here to invoke.
		unsigned int m_flags;

		virtual ~RayResultCallback()
		{
		}
		bool	hasHit() const
		{
			return (m_collisionObject != 0);
		}

		RayResultCallback()
			:m_closestHitFraction(btScalar(1.)),
			m_collisionObject(0),
			m_collisionFilterGroup(btBroadphaseProxy::DefaultFilter),
			m_collisionFilterMask(btBroadphaseProxy::AllFilter),
			//@BP Mod
			m_flags(0)
		{
		}

		virtual bool needsCollision(btBroadphaseProxy* proxy0) const
		{
			bool collides = (proxy0->m_collisionFilterGroup & m_collisionFilterMask) != 0;
			collides = collides && (m_collisionFilterGroup & proxy0->m_collisionFilterMask);
			return collides;
		}


		virtual	btScalar	addSingleResult(LocalRayResult& rayResult,bool normalInWorldSpace) = 0;
	};

	struct	ClosestRayResultCallback : public RayResultCallback
	{
		ClosestRayResultCallback(const btVector3&	rayFromWorld,const btVector3&	rayToWorld)
		:m_rayFromWorld(rayFromWorld),
		m_rayToWorld(rayToWorld)
		{
		}

		btVector3	m_rayFromWorld;//used to calculate hitPointWorld from hitFraction
		btVector3	m_rayToWorld;

		btVector3	m_hitNormalWorld;
		btVector3	m_hitPointWorld;
			
		virtual	btScalar	addSingleResult(LocalRayResult& rayResult,bool normalInWorldSpace)
		{
			//caller already does the filter on the m_closestHitFraction
			btAssert(rayResult.m_hitFraction <= m_closestHitFraction);
			
			m_closestHitFraction = rayResult.m_hitFraction;
			m_collisionObject = rayResult.m_collisionObject;
			if (normalInWorldSpace)
			{
				m_hitNormalWorld = rayResult.m_hitNormalLocal;
			} else
			{
				///need to transform normal into worldspace
				m_hitNormalWorld = m_collisionObject->getWorldTransform().getBasis()*rayResult.m_hitNormalLocal;
			}
			m_hitPointWorld.setInterpolate3(m_rayFromWorld,m_rayToWorld,rayResult.m_hitFraction);
			return rayResult.m_hitFraction;
		}
	};

	struct	AllHitsRayResultCallback : public RayResultCallback
	{
		AllHitsRayResultCallback(const btVector3&	rayFromWorld,const btVector3&	rayToWorld)
		:m_rayFromWorld(rayFromWorld),
		m_rayToWorld(rayToWorld)
		{
		}

		btAlignedObjectArray<const btCollisionObject*>		m_collisionObjects;

		btVector3	m_rayFromWorld;//used to calculate hitPointWorld from hitFraction
		btVector3	m_rayToWorld;

		btAlignedObjectArray<btVector3>	m_hitNormalWorld;
		btAlignedObjectArray<btVector3>	m_hitPointWorld;
		btAlignedObjectArray<btScalar> m_hitFractions;
			
		virtual	btScalar	addSingleResult(LocalRayResult& rayResult,bool normalInWorldSpace)
		{
			m_collisionObject = rayResult.m_collisionObject;
			m_collisionObjects.push_back(rayResult.m_collisionObject);
			btVector3 hitNormalWorld;
			if (normalInWorldSpace)
			{
				hitNormalWorld = rayResult.m_hitNormalLocal;
			} else
			{
				///need to transform normal into worldspace
				hitNormalWorld = m_collisionObject->getWorldTransform().getBasis()*rayResult.m_hitNormalLocal;
			}
			m_hitNormalWorld.push_back(hitNormalWorld);
			btVector3 hitPointWorld;
			hitPointWorld.setInterpolate3(m_rayFromWorld,m_rayToWorld,rayResult.m_hitFraction);
			m_hitPointWorld.push_back(hitPointWorld);
			m_hitFractions.push_back(rayResult.m_hitFraction);
			return m_closestHitFraction;
		}
	};


	struct LocalConvexResult
	{
		LocalConvexResult(const btCollisionObject*	hitCollisionObject, 
			LocalShapeInfo*	localShapeInfo,
			const btVector3&		hitNormalLocal,
			const btVector3&		hitPointLocal,
			btScalar hitFraction
			)
		:m_hitCollisionObject(hitCollisionObject),
		m_localShapeInfo(localShapeInfo),
		m_hitNormalLocal(hitNormalLocal),
		m_hitPointLocal(hitPointLocal),
		m_hitFraction(hitFraction)
		{
		}

		const btCollisionObject*		m_hitCollisionObject;
		LocalShapeInfo*			m_localShapeInfo;
		btVector3				m_hitNormalLocal;
		btVector3				m_hitPointLocal;
		btScalar				m_hitFraction;
	};

	///RayResultCallback is used to report new raycast results
	struct	ConvexResultCallback
	{
		btScalar	m_closestHitFraction;
		int	m_collisionFilterGroup;
		int	m_collisionFilterMask;
		
		ConvexResultCallback()
			:m_closestHitFraction(btScalar(1.)),
			m_collisionFilterGroup(btBroadphaseProxy::DefaultFilter),
			m_collisionFilterMask(btBroadphaseProxy::AllFilter)
		{
		}

		virtual ~ConvexResultCallback()
		{
		}
		
		bool	hasHit() const
		{
			return (m_closestHitFraction < btScalar(1.));
		}

		

		virtual bool needsCollision(btBroadphaseProxy* proxy0) const
		{
			bool collides = (proxy0->m_collisionFilterGroup & m_collisionFilterMask) != 0;
			collides = collides && (m_collisionFilterGroup & proxy0->m_collisionFilterMask);
			return collides;
		}

		virtual	btScalar	addSingleResult(LocalConvexResult& convexResult,bool normalInWorldSpace) = 0;
	};

	struct	ClosestConvexResultCallback : public ConvexResultCallback
	{
		ClosestConvexResultCallback(const btVector3&	convexFromWorld,const btVector3&	convexToWorld)
		:m_convexFromWorld(convexFromWorld),
		m_convexToWorld(convexToWorld),
		m_hitCollisionObject(0)
		{
		}

		btVector3	m_convexFromWorld;//used to calculate hitPointWorld from hitFraction
		btVector3	m_convexToWorld;

		btVector3	m_hitNormalWorld;
		btVector3	m_hitPointWorld;
		const btCollisionObject*	m_hitCollisionObject;
		
		virtual	btScalar	addSingleResult(LocalConvexResult& convexResult,bool normalInWorldSpace)
		{
//caller already does the filter on the m_closestHitFraction
			btAssert(convexResult.m_hitFraction <= m_closestHitFraction);
						
			m_closestHitFraction = convexResult.m_hitFraction;
			m_hitCollisionObject = convexResult.m_hitCollisionObject;
			if (normalInWorldSpace)
			{
				m_hitNormalWorld = convexResult.m_hitNormalLocal;
			} else
			{
				///need to transform normal into worldspace
				m_hitNormalWorld = m_hitCollisionObject->getWorldTransform().getBasis()*convexResult.m_hitNormalLocal;
			}
			m_hitPointWorld = convexResult.m_hitPointLocal;
			return convexResult.m_hitFraction;
		}
	};

	///ContactResultCallback is used to report contact points
	struct	ContactResultCallback
	{
		int	m_collisionFilterGroup;
		int	m_collisionFilterMask;
		btScalar	m_closestDistanceThreshold;

		ContactResultCallback()
			:m_collisionFilterGroup(btBroadphaseProxy::DefaultFilter),
			m_collisionFilterMask(btBroadphaseProxy::AllFilter),
			m_closestDistanceThreshold(0)
		{
		}

		virtual ~ContactResultCallback()
		{
		}
		
		virtual bool needsCollision(btBroadphaseProxy* proxy0) const
		{
			bool collides = (proxy0->m_collisionFilterGroup & m_collisionFilterMask) != 0;
			collides = collides && (m_collisionFilterGroup & proxy0->m_collisionFilterMask);
			return collides;
		}

		virtual	btScalar	addSingleResult(btManifoldPoint& cp,	const btCollisionObjectWrapper* colObj0Wrap,int partId0,int index0,const btCollisionObjectWrapper* colObj1Wrap,int partId1,int index1) = 0;
	};



	int	getNumCollisionObjects() const
	{
		return int(m_collisionObjects.size());
	}

	/// rayTest performs a raycast on all objects in the btCollisionWorld, and calls the resultCallback
	/// This allows for several queries: first hit, all hits, any hit, dependent on the value returned by the callback.
	virtual void rayTest(const btVector3& rayFromWorld, const btVector3& rayToWorld, RayResultCallback& resultCallback) const; 

	/// convexTest performs a swept convex cast on all objects in the btCollisionWorld, and calls the resultCallback
	/// This allows for several queries: first hit, all hits, any hit, dependent on the value return by the callback.
	void    convexSweepTest (const btConvexShape* castShape, const btTransform& from, const btTransform& to, ConvexResultCallback& resultCallback,  btScalar allowedCcdPenetration = btScalar(0.)) const;

	///contactTest performs a discrete collision test between colObj against all objects in the btCollisionWorld, and calls the resultCallback.
	///it reports one or more contact points for every overlapping object (including the one with deepest penetration)
	void	contactTest(btCollisionObject* colObj, ContactResultCallback& resultCallback);

	///contactTest performs a discrete collision test between two collision objects and calls the resultCallback if overlap if detected.
	///it reports one or more contact points (including the one with deepest penetration)
	void	contactPairTest(btCollisionObject* colObjA, btCollisionObject* colObjB, ContactResultCallback& resultCallback);


	/// rayTestSingle performs a raycast call and calls the resultCallback. It is used internally by rayTest.
	/// In a future implementation, we consider moving the ray test as a virtual method in btCollisionShape.
	/// This allows more customization.
	static void	rayTestSingle(const btTransform& rayFromTrans,const btTransform& rayToTrans,
					  btCollisionObject* collisionObject,
					  const btCollisionShape* collisionShape,
					  const btTransform& colObjWorldTransform,
					  RayResultCallback& resultCallback);

	static void	rayTestSingleInternal(const btTransform& rayFromTrans,const btTransform& rayToTrans,
					  const btCollisionObjectWrapper* collisionObjectWrap,
					  RayResultCallback& resultCallback);

	/// objectQuerySingle performs a collision detection query and calls the resultCallback. It is used internally by rayTest.
	static void	objectQuerySingle(const btConvexShape* castShape, const btTransform& rayFromTrans,const btTransform& rayToTrans,
					  btCollisionObject* collisionObject,
					  const btCollisionShape* collisionShape,
					  const btTransform& colObjWorldTransform,
					  ConvexResultCallback& resultCallback, btScalar	allowedPenetration);

	static void	objectQuerySingleInternal(const btConvexShape* castShape,const btTransform& convexFromTrans,const btTransform& convexToTrans,
											const btCollisionObjectWrapper* colObjWrap,
											ConvexResultCallback& resultCallback, btScalar allowedPenetration);

	virtual void	addCollisionObject(btCollisionObject* collisionObject, int collisionFilterGroup=btBroadphaseProxy::DefaultFilter, int collisionFilterMask=btBroadphaseProxy::AllFilter);

	btCollisionObjectArray& getCollisionObjectArray()
	{
		return m_collisionObjects;
	}

	const btCollisionObjectArray& getCollisionObjectArray() const
	{
		return m_collisionObjects;
	}


	virtual void	removeCollisionObject(btCollisionObject* collisionObject);

	virtual void	performDiscreteCollisionDetection();

	btDispatcherInfo& getDispatchInfo()
	{
		return m_dispatchInfo;
	}

	const btDispatcherInfo& getDispatchInfo() const
	{
		return m_dispatchInfo;
	}
	
	bool	getForceUpdateAllAabbs() const
	{
		return m_forceUpdateAllAabbs;
	}
	void setForceUpdateAllAabbs( bool forceUpdateAllAabbs)
	{
		m_forceUpdateAllAabbs = forceUpdateAllAabbs;
	}

	///Preliminary serialization test for Bullet 2.76. Loading those files requires a separate parser (Bullet/Demos/SerializeDemo)
	virtual	void	serialize(btSerializer* serializer);

};


#endif //BT_COLLISION_WORLD_H
