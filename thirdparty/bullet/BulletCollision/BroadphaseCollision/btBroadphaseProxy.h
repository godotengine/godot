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

#ifndef BT_BROADPHASE_PROXY_H
#define BT_BROADPHASE_PROXY_H

#include "LinearMath/btScalar.h"  //for SIMD_FORCE_INLINE
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedAllocator.h"

/// btDispatcher uses these types
/// IMPORTANT NOTE:The types are ordered polyhedral, implicit convex and concave
/// to facilitate type checking
/// CUSTOM_POLYHEDRAL_SHAPE_TYPE,CUSTOM_CONVEX_SHAPE_TYPE and CUSTOM_CONCAVE_SHAPE_TYPE can be used to extend Bullet without modifying source code
enum BroadphaseNativeTypes
{
	// polyhedral convex shapes
	BOX_SHAPE_PROXYTYPE,
	TRIANGLE_SHAPE_PROXYTYPE,
	TETRAHEDRAL_SHAPE_PROXYTYPE,
	CONVEX_TRIANGLEMESH_SHAPE_PROXYTYPE,
	CONVEX_HULL_SHAPE_PROXYTYPE,
	CONVEX_POINT_CLOUD_SHAPE_PROXYTYPE,
	CUSTOM_POLYHEDRAL_SHAPE_TYPE,
	//implicit convex shapes
	IMPLICIT_CONVEX_SHAPES_START_HERE,
	SPHERE_SHAPE_PROXYTYPE,
	MULTI_SPHERE_SHAPE_PROXYTYPE,
	CAPSULE_SHAPE_PROXYTYPE,
	CONE_SHAPE_PROXYTYPE,
	CONVEX_SHAPE_PROXYTYPE,
	CYLINDER_SHAPE_PROXYTYPE,
	UNIFORM_SCALING_SHAPE_PROXYTYPE,
	MINKOWSKI_SUM_SHAPE_PROXYTYPE,
	MINKOWSKI_DIFFERENCE_SHAPE_PROXYTYPE,
	BOX_2D_SHAPE_PROXYTYPE,
	CONVEX_2D_SHAPE_PROXYTYPE,
	CUSTOM_CONVEX_SHAPE_TYPE,
	//concave shapes
	CONCAVE_SHAPES_START_HERE,
	//keep all the convex shapetype below here, for the check IsConvexShape in broadphase proxy!
	TRIANGLE_MESH_SHAPE_PROXYTYPE,
	SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE,
	///used for demo integration FAST/Swift collision library and Bullet
	FAST_CONCAVE_MESH_PROXYTYPE,
	//terrain
	TERRAIN_SHAPE_PROXYTYPE,
	///Used for GIMPACT Trimesh integration
	GIMPACT_SHAPE_PROXYTYPE,
	///Multimaterial mesh
	MULTIMATERIAL_TRIANGLE_MESH_PROXYTYPE,

	EMPTY_SHAPE_PROXYTYPE,
	STATIC_PLANE_PROXYTYPE,
	CUSTOM_CONCAVE_SHAPE_TYPE,
	SDF_SHAPE_PROXYTYPE = CUSTOM_CONCAVE_SHAPE_TYPE,
	CONCAVE_SHAPES_END_HERE,

	COMPOUND_SHAPE_PROXYTYPE,

	SOFTBODY_SHAPE_PROXYTYPE,
	HFFLUID_SHAPE_PROXYTYPE,
	HFFLUID_BUOYANT_CONVEX_SHAPE_PROXYTYPE,
	INVALID_SHAPE_PROXYTYPE,

	MAX_BROADPHASE_COLLISION_TYPES

};

///The btBroadphaseProxy is the main class that can be used with the Bullet broadphases.
///It stores collision shape type information, collision filter information and a client object, typically a btCollisionObject or btRigidBody.
ATTRIBUTE_ALIGNED16(struct)
btBroadphaseProxy
{
	BT_DECLARE_ALIGNED_ALLOCATOR();

	///optional filtering to cull potential collisions
	enum CollisionFilterGroups
	{
		DefaultFilter = 1,
		StaticFilter = 2,
		KinematicFilter = 4,
		DebrisFilter = 8,
		SensorTrigger = 16,
		CharacterFilter = 32,
		AllFilter = -1  //all bits sets: DefaultFilter | StaticFilter | KinematicFilter | DebrisFilter | SensorTrigger
	};

	//Usually the client btCollisionObject or Rigidbody class
	void* m_clientObject;
	int m_collisionFilterGroup;
	int m_collisionFilterMask;

	int m_uniqueId;  //m_uniqueId is introduced for paircache. could get rid of this, by calculating the address offset etc.

	btVector3 m_aabbMin;
	btVector3 m_aabbMax;

	SIMD_FORCE_INLINE int getUid() const
	{
		return m_uniqueId;
	}

	//used for memory pools
	btBroadphaseProxy() : m_clientObject(0)
	{
	}

	btBroadphaseProxy(const btVector3& aabbMin, const btVector3& aabbMax, void* userPtr, int collisionFilterGroup, int collisionFilterMask)
		: m_clientObject(userPtr),
		  m_collisionFilterGroup(collisionFilterGroup),
		  m_collisionFilterMask(collisionFilterMask),
		  m_aabbMin(aabbMin),
		  m_aabbMax(aabbMax)
	{
	}

	static SIMD_FORCE_INLINE bool isPolyhedral(int proxyType)
	{
		return (proxyType < IMPLICIT_CONVEX_SHAPES_START_HERE);
	}

	static SIMD_FORCE_INLINE bool isConvex(int proxyType)
	{
		return (proxyType < CONCAVE_SHAPES_START_HERE);
	}

	static SIMD_FORCE_INLINE bool isNonMoving(int proxyType)
	{
		return (isConcave(proxyType) && !(proxyType == GIMPACT_SHAPE_PROXYTYPE));
	}

	static SIMD_FORCE_INLINE bool isConcave(int proxyType)
	{
		return ((proxyType > CONCAVE_SHAPES_START_HERE) &&
				(proxyType < CONCAVE_SHAPES_END_HERE));
	}
	static SIMD_FORCE_INLINE bool isCompound(int proxyType)
	{
		return (proxyType == COMPOUND_SHAPE_PROXYTYPE);
	}

	static SIMD_FORCE_INLINE bool isSoftBody(int proxyType)
	{
		return (proxyType == SOFTBODY_SHAPE_PROXYTYPE);
	}

	static SIMD_FORCE_INLINE bool isInfinite(int proxyType)
	{
		return (proxyType == STATIC_PLANE_PROXYTYPE);
	}

	static SIMD_FORCE_INLINE bool isConvex2d(int proxyType)
	{
		return (proxyType == BOX_2D_SHAPE_PROXYTYPE) || (proxyType == CONVEX_2D_SHAPE_PROXYTYPE);
	}
};

class btCollisionAlgorithm;

struct btBroadphaseProxy;

///The btBroadphasePair class contains a pair of aabb-overlapping objects.
///A btDispatcher can search a btCollisionAlgorithm that performs exact/narrowphase collision detection on the actual collision shapes.
ATTRIBUTE_ALIGNED16(struct)
btBroadphasePair
{
	btBroadphasePair()
		: m_pProxy0(0),
		  m_pProxy1(0),
		  m_algorithm(0),
		  m_internalInfo1(0)
	{
	}

	BT_DECLARE_ALIGNED_ALLOCATOR();

	btBroadphasePair(btBroadphaseProxy & proxy0, btBroadphaseProxy & proxy1)
	{
		//keep them sorted, so the std::set operations work
		if (proxy0.m_uniqueId < proxy1.m_uniqueId)
		{
			m_pProxy0 = &proxy0;
			m_pProxy1 = &proxy1;
		}
		else
		{
			m_pProxy0 = &proxy1;
			m_pProxy1 = &proxy0;
		}

		m_algorithm = 0;
		m_internalInfo1 = 0;
	}

	btBroadphaseProxy* m_pProxy0;
	btBroadphaseProxy* m_pProxy1;

	mutable btCollisionAlgorithm* m_algorithm;
	union {
		void* m_internalInfo1;
		int m_internalTmpValue;
	};  //don't use this data, it will be removed in future version.
};

/*
//comparison for set operation, see Solid DT_Encounter
SIMD_FORCE_INLINE bool operator<(const btBroadphasePair& a, const btBroadphasePair& b) 
{ 
    return a.m_pProxy0 < b.m_pProxy0 || 
        (a.m_pProxy0 == b.m_pProxy0 && a.m_pProxy1 < b.m_pProxy1); 
}
*/

class btBroadphasePairSortPredicate
{
public:
	bool operator()(const btBroadphasePair& a, const btBroadphasePair& b) const
	{
		const int uidA0 = a.m_pProxy0 ? a.m_pProxy0->m_uniqueId : -1;
		const int uidB0 = b.m_pProxy0 ? b.m_pProxy0->m_uniqueId : -1;
		const int uidA1 = a.m_pProxy1 ? a.m_pProxy1->m_uniqueId : -1;
		const int uidB1 = b.m_pProxy1 ? b.m_pProxy1->m_uniqueId : -1;

		return uidA0 > uidB0 ||
			   (a.m_pProxy0 == b.m_pProxy0 && uidA1 > uidB1) ||
			   (a.m_pProxy0 == b.m_pProxy0 && a.m_pProxy1 == b.m_pProxy1 && a.m_algorithm > b.m_algorithm);
	}
};

SIMD_FORCE_INLINE bool operator==(const btBroadphasePair& a, const btBroadphasePair& b)
{
	return (a.m_pProxy0 == b.m_pProxy0) && (a.m_pProxy1 == b.m_pProxy1);
}

#endif  //BT_BROADPHASE_PROXY_H
