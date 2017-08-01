
#if 0
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


#include "b3ContactCache.h"
#include "Bullet3Common/b3Transform.h"

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"

b3Scalar					gContactBreakingThreshold = b3Scalar(0.02);

///gContactCalcArea3Points will approximate the convex hull area using 3 points
///when setting it to false, it will use 4 points to compute the area: it is more accurate but slower
bool						gContactCalcArea3Points = true;




static inline b3Scalar calcArea4Points(const b3Vector3 &p0,const b3Vector3 &p1,const b3Vector3 &p2,const b3Vector3 &p3)
{
	// It calculates possible 3 area constructed from random 4 points and returns the biggest one.

	b3Vector3 a[3],b[3];
	a[0] = p0 - p1;
	a[1] = p0 - p2;
	a[2] = p0 - p3;
	b[0] = p2 - p3;
	b[1] = p1 - p3;
	b[2] = p1 - p2;

	//todo: Following 3 cross production can be easily optimized by SIMD.
	b3Vector3 tmp0 = a[0].cross(b[0]);
	b3Vector3 tmp1 = a[1].cross(b[1]);
	b3Vector3 tmp2 = a[2].cross(b[2]);

	return b3Max(b3Max(tmp0.length2(),tmp1.length2()),tmp2.length2());
}
#if 0

//using localPointA for all points
int b3ContactCache::sortCachedPoints(const b3Vector3& pt) 
{
		//calculate 4 possible cases areas, and take biggest area
		//also need to keep 'deepest'
		
		int maxPenetrationIndex = -1;
#define KEEP_DEEPEST_POINT 1
#ifdef KEEP_DEEPEST_POINT
		b3Scalar maxPenetration = pt.getDistance();
		for (int i=0;i<4;i++)
		{
			if (m_pointCache[i].getDistance() < maxPenetration)
			{
				maxPenetrationIndex = i;
				maxPenetration = m_pointCache[i].getDistance();
			}
		}
#endif //KEEP_DEEPEST_POINT
		
		b3Scalar res0(b3Scalar(0.)),res1(b3Scalar(0.)),res2(b3Scalar(0.)),res3(b3Scalar(0.));

	if (gContactCalcArea3Points)
	{
		if (maxPenetrationIndex != 0)
		{
			b3Vector3 a0 = pt.m_localPointA-m_pointCache[1].m_localPointA;
			b3Vector3 b0 = m_pointCache[3].m_localPointA-m_pointCache[2].m_localPointA;
			b3Vector3 cross = a0.cross(b0);
			res0 = cross.length2();
		}
		if (maxPenetrationIndex != 1)
		{
			b3Vector3 a1 = pt.m_localPointA-m_pointCache[0].m_localPointA;
			b3Vector3 b1 = m_pointCache[3].m_localPointA-m_pointCache[2].m_localPointA;
			b3Vector3 cross = a1.cross(b1);
			res1 = cross.length2();
		}

		if (maxPenetrationIndex != 2)
		{
			b3Vector3 a2 = pt.m_localPointA-m_pointCache[0].m_localPointA;
			b3Vector3 b2 = m_pointCache[3].m_localPointA-m_pointCache[1].m_localPointA;
			b3Vector3 cross = a2.cross(b2);
			res2 = cross.length2();
		}

		if (maxPenetrationIndex != 3)
		{
			b3Vector3 a3 = pt.m_localPointA-m_pointCache[0].m_localPointA;
			b3Vector3 b3 = m_pointCache[2].m_localPointA-m_pointCache[1].m_localPointA;
			b3Vector3 cross = a3.cross(b3);
			res3 = cross.length2();
		}
	} 
	else
	{
		if(maxPenetrationIndex != 0) {
			res0 = calcArea4Points(pt.m_localPointA,m_pointCache[1].m_localPointA,m_pointCache[2].m_localPointA,m_pointCache[3].m_localPointA);
		}

		if(maxPenetrationIndex != 1) {
			res1 = calcArea4Points(pt.m_localPointA,m_pointCache[0].m_localPointA,m_pointCache[2].m_localPointA,m_pointCache[3].m_localPointA);
		}

		if(maxPenetrationIndex != 2) {
			res2 = calcArea4Points(pt.m_localPointA,m_pointCache[0].m_localPointA,m_pointCache[1].m_localPointA,m_pointCache[3].m_localPointA);
		}

		if(maxPenetrationIndex != 3) {
			res3 = calcArea4Points(pt.m_localPointA,m_pointCache[0].m_localPointA,m_pointCache[1].m_localPointA,m_pointCache[2].m_localPointA);
		}
	}
	b3Vector4 maxvec(res0,res1,res2,res3);
	int biggestarea = maxvec.closestAxis4();
	return biggestarea;
	
}


int b3ContactCache::getCacheEntry(const b3Vector3& newPoint) const
{
	b3Scalar shortestDist =  getContactBreakingThreshold() * getContactBreakingThreshold();
	int size = getNumContacts();
	int nearestPoint = -1;
	for( int i = 0; i < size; i++ )
	{
		const b3Vector3 &mp = m_pointCache[i];

		b3Vector3 diffA =  mp.m_localPointA- newPoint.m_localPointA;
		const b3Scalar distToManiPoint = diffA.dot(diffA);
		if( distToManiPoint < shortestDist )
		{
			shortestDist = distToManiPoint;
			nearestPoint = i;
		}
	}
	return nearestPoint;
}

int b3ContactCache::addManifoldPoint(const b3Vector3& newPoint)
{
	b3Assert(validContactDistance(newPoint));
	
	int insertIndex = getNumContacts();
	if (insertIndex == MANIFOLD_CACHE_SIZE)
	{
#if MANIFOLD_CACHE_SIZE >= 4
		//sort cache so best points come first, based on area
		insertIndex = sortCachedPoints(newPoint);
#else
		insertIndex = 0;
#endif
		clearUserCache(m_pointCache[insertIndex]);
		
	} else
	{
		m_cachedPoints++;

		
	}
	if (insertIndex<0)
		insertIndex=0;

	//b3Assert(m_pointCache[insertIndex].m_userPersistentData==0);
	m_pointCache[insertIndex] = newPoint;
	return insertIndex;
}

#endif

bool b3ContactCache::validContactDistance(const b3Vector3& pt)
{
	return pt.w <= gContactBreakingThreshold;
}

void b3ContactCache::removeContactPoint(struct b3Contact4Data& newContactCache,int i)
{
	int numContacts = b3Contact4Data_getNumPoints(&newContactCache);
	if (i!=(numContacts-1))
	{
		b3Swap(newContactCache.m_localPosA[i],newContactCache.m_localPosA[numContacts-1]);
		b3Swap(newContactCache.m_localPosB[i],newContactCache.m_localPosB[numContacts-1]);
		b3Swap(newContactCache.m_worldPosB[i],newContactCache.m_worldPosB[numContacts-1]);
	}
	b3Contact4Data_setNumPoints(&newContactCache,numContacts-1);

}


void b3ContactCache::refreshContactPoints(const b3Transform& trA,const b3Transform& trB, struct b3Contact4Data& contacts)
{

	int numContacts = b3Contact4Data_getNumPoints(&contacts);
	

	int i;
	/// first refresh worldspace positions and distance
	for (i=numContacts-1;i>=0;i--)
	{
		b3Vector3 worldPosA = trA( contacts.m_localPosA[i]);
		b3Vector3 worldPosB = trB( contacts.m_localPosB[i]);
		contacts.m_worldPosB[i] = worldPosB;
		float distance = (worldPosA -  worldPosB).dot(contacts.m_worldNormalOnB);
		contacts.m_worldPosB[i].w = distance;
	}

	/// then 
	b3Scalar distance2d;
	b3Vector3 projectedDifference,projectedPoint;
	for (i=numContacts-1;i>=0;i--)
	{
		b3Vector3 worldPosA = trA( contacts.m_localPosA[i]);
		b3Vector3 worldPosB = trB( contacts.m_localPosB[i]);
		b3Vector3&pt = contacts.m_worldPosB[i];
		//contact becomes invalid when signed distance exceeds margin (projected on contactnormal direction)
		if (!validContactDistance(pt))
		{
			removeContactPoint(contacts,i);
		} else
		{
			//contact also becomes invalid when relative movement orthogonal to normal exceeds margin
			projectedPoint = worldPosA - contacts.m_worldNormalOnB * contacts.m_worldPosB[i].w;
			projectedDifference = contacts.m_worldPosB[i] - projectedPoint;
			distance2d = projectedDifference.dot(projectedDifference);
			if (distance2d  > gContactBreakingThreshold*gContactBreakingThreshold )
			{
				removeContactPoint(contacts,i);
			} else
			{
				////contact point processed callback
				//if (gContactProcessedCallback)
				//	(*gContactProcessedCallback)(manifoldPoint,(void*)m_body0,(void*)m_body1);
			}
		}
	}
	

}





#endif
