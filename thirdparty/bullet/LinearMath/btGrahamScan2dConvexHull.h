/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef GRAHAM_SCAN_2D_CONVEX_HULL_H
#define GRAHAM_SCAN_2D_CONVEX_HULL_H


#include "btVector3.h"
#include "btAlignedObjectArray.h"

struct GrahamVector3 : public btVector3
{
	GrahamVector3(const btVector3& org, int orgIndex)
		:btVector3(org),
			m_orgIndex(orgIndex)
	{
	}
	btScalar	m_angle;
	int m_orgIndex;
};


struct btAngleCompareFunc {
	btVector3 m_anchor;
	btAngleCompareFunc(const btVector3& anchor)
	: m_anchor(anchor) 
	{
	}
	bool operator()(const GrahamVector3& a, const GrahamVector3& b) const {
		if (a.m_angle != b.m_angle)
			return a.m_angle < b.m_angle;
		else
		{
			btScalar al = (a-m_anchor).length2();
			btScalar bl = (b-m_anchor).length2();
			if (al != bl)
				return  al < bl;
			else
			{
				return a.m_orgIndex < b.m_orgIndex;
			}
		}
	}
};

inline void GrahamScanConvexHull2D(btAlignedObjectArray<GrahamVector3>& originalPoints, btAlignedObjectArray<GrahamVector3>& hull, const btVector3& normalAxis)
{
	btVector3 axis0,axis1;
	btPlaneSpace1(normalAxis,axis0,axis1);
	

	if (originalPoints.size()<=1)
	{
		for (int i=0;i<originalPoints.size();i++)
			hull.push_back(originalPoints[0]);
		return;
	}
	//step1 : find anchor point with smallest projection on axis0 and move it to first location
	for (int i=0;i<originalPoints.size();i++)
	{
//		const btVector3& left = originalPoints[i];
//		const btVector3& right = originalPoints[0];
		btScalar projL = originalPoints[i].dot(axis0);
		btScalar projR = originalPoints[0].dot(axis0);
		if (projL < projR)
		{
			originalPoints.swap(0,i);
		}
	}

	//also precompute angles
	originalPoints[0].m_angle = -1e30f;
	for (int i=1;i<originalPoints.size();i++)
	{
	    btVector3 ar = originalPoints[i]-originalPoints[0];
	    btScalar ar1 = axis1.dot(ar);
	    btScalar ar0 = axis0.dot(ar);
	    if( ar1*ar1+ar0*ar0 < FLT_EPSILON ) 
	    {
	      originalPoints[i].m_angle = 0.0f;
	    }
	    else
	    {
	      originalPoints[i].m_angle = btAtan2Fast(ar1, ar0);
	    }
	}

	//step 2: sort all points, based on 'angle' with this anchor
	btAngleCompareFunc comp(originalPoints[0]);
	originalPoints.quickSortInternal(comp,1,originalPoints.size()-1);

	int i;
	for (i = 0; i<2; i++) 
		hull.push_back(originalPoints[i]);

	//step 3: keep all 'convex' points and discard concave points (using back tracking)
	for (; i != originalPoints.size(); i++) 
	{
		bool isConvex = false;
		while (!isConvex&& hull.size()>1) {
			btVector3& a = hull[hull.size()-2];
			btVector3& b = hull[hull.size()-1];
			isConvex = btCross(a-b,a-originalPoints[i]).dot(normalAxis)> 0;
			if (!isConvex)
				hull.pop_back();
			else 
				hull.push_back(originalPoints[i]);
		}

	    if( hull.size() == 1 )
	    {
	      hull.push_back( originalPoints[i] );
	    }
	}
}

#endif //GRAHAM_SCAN_2D_CONVEX_HULL_H
