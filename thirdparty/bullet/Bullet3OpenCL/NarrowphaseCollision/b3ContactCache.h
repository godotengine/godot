
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

#ifndef B3_CONTACT_CACHE_H
#define B3_CONTACT_CACHE_H

#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3Transform.h"
#include "Bullet3Common/b3AlignedAllocator.h"

///maximum contact breaking and merging threshold
extern b3Scalar gContactBreakingThreshold;

#define MANIFOLD_CACHE_SIZE 4

///b3ContactCache is a contact point cache, it stays persistent as long as objects are overlapping in the broadphase.
///Those contact points are created by the collision narrow phase.
///The cache can be empty, or hold 1,2,3 or 4 points. Some collision algorithms (GJK) might only add one point at a time.
///updates/refreshes old contact points, and throw them away if necessary (distance becomes too large)
///reduces the cache to 4 points, when more then 4 points are added, using following rules:
///the contact point with deepest penetration is always kept, and it tries to maximuze the area covered by the points
///note that some pairs of objects might have more then one contact manifold.
B3_ATTRIBUTE_ALIGNED16(class)
b3ContactCache
{
	/// sort cached points so most isolated points come first
	int sortCachedPoints(const b3Vector3& pt);

public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	int addManifoldPoint(const b3Vector3& newPoint);

	/*void replaceContactPoint(const b3Vector3& newPoint,int insertIndex)
	{
		b3Assert(validContactDistance(newPoint));
		m_pointCache[insertIndex] = newPoint;
	}
	*/

	static bool validContactDistance(const b3Vector3& pt);

	/// calculated new worldspace coordinates and depth, and reject points that exceed the collision margin
	static void refreshContactPoints(const b3Transform& trA, const b3Transform& trB, struct b3Contact4Data& newContactCache);

	static void removeContactPoint(struct b3Contact4Data & newContactCache, int i);
};

#endif  //B3_CONTACT_CACHE_H
