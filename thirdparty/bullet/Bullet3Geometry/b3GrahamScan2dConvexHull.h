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

#ifndef B3_GRAHAM_SCAN_2D_CONVEX_HULL_H
#define B3_GRAHAM_SCAN_2D_CONVEX_HULL_H

#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

struct b3GrahamVector3 : public b3Vector3
{
	b3GrahamVector3(const b3Vector3& org, int orgIndex)
		: b3Vector3(org),
		  m_orgIndex(orgIndex)
	{
	}
	b3Scalar m_angle;
	int m_orgIndex;
};

struct b3AngleCompareFunc
{
	b3Vector3 m_anchor;
	b3AngleCompareFunc(const b3Vector3& anchor)
		: m_anchor(anchor)
	{
	}
	bool operator()(const b3GrahamVector3& a, const b3GrahamVector3& b) const
	{
		if (a.m_angle != b.m_angle)
			return a.m_angle < b.m_angle;
		else
		{
			b3Scalar al = (a - m_anchor).length2();
			b3Scalar bl = (b - m_anchor).length2();
			if (al != bl)
				return al < bl;
			else
			{
				return a.m_orgIndex < b.m_orgIndex;
			}
		}
	}
};

inline void b3GrahamScanConvexHull2D(b3AlignedObjectArray<b3GrahamVector3>& originalPoints, b3AlignedObjectArray<b3GrahamVector3>& hull, const b3Vector3& normalAxis)
{
	b3Vector3 axis0, axis1;
	b3PlaneSpace1(normalAxis, axis0, axis1);

	if (originalPoints.size() <= 1)
	{
		for (int i = 0; i < originalPoints.size(); i++)
			hull.push_back(originalPoints[0]);
		return;
	}
	//step1 : find anchor point with smallest projection on axis0 and move it to first location
	for (int i = 0; i < originalPoints.size(); i++)
	{
		//		const b3Vector3& left = originalPoints[i];
		//		const b3Vector3& right = originalPoints[0];
		b3Scalar projL = originalPoints[i].dot(axis0);
		b3Scalar projR = originalPoints[0].dot(axis0);
		if (projL < projR)
		{
			originalPoints.swap(0, i);
		}
	}

	//also precompute angles
	originalPoints[0].m_angle = -1e30f;
	for (int i = 1; i < originalPoints.size(); i++)
	{
		b3Vector3 xvec = axis0;
		b3Vector3 ar = originalPoints[i] - originalPoints[0];
		originalPoints[i].m_angle = b3Cross(xvec, ar).dot(normalAxis) / ar.length();
	}

	//step 2: sort all points, based on 'angle' with this anchor
	b3AngleCompareFunc comp(originalPoints[0]);
	originalPoints.quickSortInternal(comp, 1, originalPoints.size() - 1);

	int i;
	for (i = 0; i < 2; i++)
		hull.push_back(originalPoints[i]);

	//step 3: keep all 'convex' points and discard concave points (using back tracking)
	for (; i != originalPoints.size(); i++)
	{
		bool isConvex = false;
		while (!isConvex && hull.size() > 1)
		{
			b3Vector3& a = hull[hull.size() - 2];
			b3Vector3& b = hull[hull.size() - 1];
			isConvex = b3Cross(a - b, a - originalPoints[i]).dot(normalAxis) > 0;
			if (!isConvex)
				hull.pop_back();
			else
				hull.push_back(originalPoints[i]);
		}
	}
}

#endif  //B3_GRAHAM_SCAN_2D_CONVEX_HULL_H
