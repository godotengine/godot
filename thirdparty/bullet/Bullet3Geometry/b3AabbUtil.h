/*
Copyright (c) 2003-2006 Gino van den Bergen / Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_AABB_UTIL2
#define B3_AABB_UTIL2

#include "Bullet3Common/b3Transform.h"
#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3MinMax.h"

B3_FORCE_INLINE void b3AabbExpand(b3Vector3& aabbMin,
								  b3Vector3& aabbMax,
								  const b3Vector3& expansionMin,
								  const b3Vector3& expansionMax)
{
	aabbMin = aabbMin + expansionMin;
	aabbMax = aabbMax + expansionMax;
}

/// conservative test for overlap between two aabbs
B3_FORCE_INLINE bool b3TestPointAgainstAabb2(const b3Vector3& aabbMin1, const b3Vector3& aabbMax1,
											 const b3Vector3& point)
{
	bool overlap = true;
	overlap = (aabbMin1.getX() > point.getX() || aabbMax1.getX() < point.getX()) ? false : overlap;
	overlap = (aabbMin1.getZ() > point.getZ() || aabbMax1.getZ() < point.getZ()) ? false : overlap;
	overlap = (aabbMin1.getY() > point.getY() || aabbMax1.getY() < point.getY()) ? false : overlap;
	return overlap;
}

/// conservative test for overlap between two aabbs
B3_FORCE_INLINE bool b3TestAabbAgainstAabb2(const b3Vector3& aabbMin1, const b3Vector3& aabbMax1,
											const b3Vector3& aabbMin2, const b3Vector3& aabbMax2)
{
	bool overlap = true;
	overlap = (aabbMin1.getX() > aabbMax2.getX() || aabbMax1.getX() < aabbMin2.getX()) ? false : overlap;
	overlap = (aabbMin1.getZ() > aabbMax2.getZ() || aabbMax1.getZ() < aabbMin2.getZ()) ? false : overlap;
	overlap = (aabbMin1.getY() > aabbMax2.getY() || aabbMax1.getY() < aabbMin2.getY()) ? false : overlap;
	return overlap;
}

/// conservative test for overlap between triangle and aabb
B3_FORCE_INLINE bool b3TestTriangleAgainstAabb2(const b3Vector3* vertices,
												const b3Vector3& aabbMin, const b3Vector3& aabbMax)
{
	const b3Vector3& p1 = vertices[0];
	const b3Vector3& p2 = vertices[1];
	const b3Vector3& p3 = vertices[2];

	if (b3Min(b3Min(p1[0], p2[0]), p3[0]) > aabbMax[0]) return false;
	if (b3Max(b3Max(p1[0], p2[0]), p3[0]) < aabbMin[0]) return false;

	if (b3Min(b3Min(p1[2], p2[2]), p3[2]) > aabbMax[2]) return false;
	if (b3Max(b3Max(p1[2], p2[2]), p3[2]) < aabbMin[2]) return false;

	if (b3Min(b3Min(p1[1], p2[1]), p3[1]) > aabbMax[1]) return false;
	if (b3Max(b3Max(p1[1], p2[1]), p3[1]) < aabbMin[1]) return false;
	return true;
}

B3_FORCE_INLINE int b3Outcode(const b3Vector3& p, const b3Vector3& halfExtent)
{
	return (p.getX() < -halfExtent.getX() ? 0x01 : 0x0) |
		   (p.getX() > halfExtent.getX() ? 0x08 : 0x0) |
		   (p.getY() < -halfExtent.getY() ? 0x02 : 0x0) |
		   (p.getY() > halfExtent.getY() ? 0x10 : 0x0) |
		   (p.getZ() < -halfExtent.getZ() ? 0x4 : 0x0) |
		   (p.getZ() > halfExtent.getZ() ? 0x20 : 0x0);
}

B3_FORCE_INLINE bool b3RayAabb2(const b3Vector3& rayFrom,
								const b3Vector3& rayInvDirection,
								const unsigned int raySign[3],
								const b3Vector3 bounds[2],
								b3Scalar& tmin,
								b3Scalar lambda_min,
								b3Scalar lambda_max)
{
	b3Scalar tmax, tymin, tymax, tzmin, tzmax;
	tmin = (bounds[raySign[0]].getX() - rayFrom.getX()) * rayInvDirection.getX();
	tmax = (bounds[1 - raySign[0]].getX() - rayFrom.getX()) * rayInvDirection.getX();
	tymin = (bounds[raySign[1]].getY() - rayFrom.getY()) * rayInvDirection.getY();
	tymax = (bounds[1 - raySign[1]].getY() - rayFrom.getY()) * rayInvDirection.getY();

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bounds[raySign[2]].getZ() - rayFrom.getZ()) * rayInvDirection.getZ();
	tzmax = (bounds[1 - raySign[2]].getZ() - rayFrom.getZ()) * rayInvDirection.getZ();

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	return ((tmin < lambda_max) && (tmax > lambda_min));
}

B3_FORCE_INLINE bool b3RayAabb(const b3Vector3& rayFrom,
							   const b3Vector3& rayTo,
							   const b3Vector3& aabbMin,
							   const b3Vector3& aabbMax,
							   b3Scalar& param, b3Vector3& normal)
{
	b3Vector3 aabbHalfExtent = (aabbMax - aabbMin) * b3Scalar(0.5);
	b3Vector3 aabbCenter = (aabbMax + aabbMin) * b3Scalar(0.5);
	b3Vector3 source = rayFrom - aabbCenter;
	b3Vector3 target = rayTo - aabbCenter;
	int sourceOutcode = b3Outcode(source, aabbHalfExtent);
	int targetOutcode = b3Outcode(target, aabbHalfExtent);
	if ((sourceOutcode & targetOutcode) == 0x0)
	{
		b3Scalar lambda_enter = b3Scalar(0.0);
		b3Scalar lambda_exit = param;
		b3Vector3 r = target - source;
		int i;
		b3Scalar normSign = 1;
		b3Vector3 hitNormal = b3MakeVector3(0, 0, 0);
		int bit = 1;

		for (int j = 0; j < 2; j++)
		{
			for (i = 0; i != 3; ++i)
			{
				if (sourceOutcode & bit)
				{
					b3Scalar lambda = (-source[i] - aabbHalfExtent[i] * normSign) / r[i];
					if (lambda_enter <= lambda)
					{
						lambda_enter = lambda;
						hitNormal.setValue(0, 0, 0);
						hitNormal[i] = normSign;
					}
				}
				else if (targetOutcode & bit)
				{
					b3Scalar lambda = (-source[i] - aabbHalfExtent[i] * normSign) / r[i];
					b3SetMin(lambda_exit, lambda);
				}
				bit <<= 1;
			}
			normSign = b3Scalar(-1.);
		}
		if (lambda_enter <= lambda_exit)
		{
			param = lambda_enter;
			normal = hitNormal;
			return true;
		}
	}
	return false;
}

B3_FORCE_INLINE void b3TransformAabb(const b3Vector3& halfExtents, b3Scalar margin, const b3Transform& t, b3Vector3& aabbMinOut, b3Vector3& aabbMaxOut)
{
	b3Vector3 halfExtentsWithMargin = halfExtents + b3MakeVector3(margin, margin, margin);
	b3Matrix3x3 abs_b = t.getBasis().absolute();
	b3Vector3 center = t.getOrigin();
	b3Vector3 extent = halfExtentsWithMargin.dot3(abs_b[0], abs_b[1], abs_b[2]);
	aabbMinOut = center - extent;
	aabbMaxOut = center + extent;
}

B3_FORCE_INLINE void b3TransformAabb(const b3Vector3& localAabbMin, const b3Vector3& localAabbMax, b3Scalar margin, const b3Transform& trans, b3Vector3& aabbMinOut, b3Vector3& aabbMaxOut)
{
	//b3Assert(localAabbMin.getX() <= localAabbMax.getX());
	//b3Assert(localAabbMin.getY() <= localAabbMax.getY());
	//b3Assert(localAabbMin.getZ() <= localAabbMax.getZ());
	b3Vector3 localHalfExtents = b3Scalar(0.5) * (localAabbMax - localAabbMin);
	localHalfExtents += b3MakeVector3(margin, margin, margin);

	b3Vector3 localCenter = b3Scalar(0.5) * (localAabbMax + localAabbMin);
	b3Matrix3x3 abs_b = trans.getBasis().absolute();
	b3Vector3 center = trans(localCenter);
	b3Vector3 extent = localHalfExtents.dot3(abs_b[0], abs_b[1], abs_b[2]);
	aabbMinOut = center - extent;
	aabbMaxOut = center + extent;
}

#define B3_USE_BANCHLESS 1
#ifdef B3_USE_BANCHLESS
//This block replaces the block below and uses no branches, and replaces the 8 bit return with a 32 bit return for improved performance (~3x on XBox 360)
B3_FORCE_INLINE unsigned b3TestQuantizedAabbAgainstQuantizedAabb(const unsigned short int* aabbMin1, const unsigned short int* aabbMax1, const unsigned short int* aabbMin2, const unsigned short int* aabbMax2)
{
	return static_cast<unsigned int>(b3Select((unsigned)((aabbMin1[0] <= aabbMax2[0]) & (aabbMax1[0] >= aabbMin2[0]) & (aabbMin1[2] <= aabbMax2[2]) & (aabbMax1[2] >= aabbMin2[2]) & (aabbMin1[1] <= aabbMax2[1]) & (aabbMax1[1] >= aabbMin2[1])),
											  1, 0));
}
#else
B3_FORCE_INLINE bool b3TestQuantizedAabbAgainstQuantizedAabb(const unsigned short int* aabbMin1, const unsigned short int* aabbMax1, const unsigned short int* aabbMin2, const unsigned short int* aabbMax2)
{
	bool overlap = true;
	overlap = (aabbMin1[0] > aabbMax2[0] || aabbMax1[0] < aabbMin2[0]) ? false : overlap;
	overlap = (aabbMin1[2] > aabbMax2[2] || aabbMax1[2] < aabbMin2[2]) ? false : overlap;
	overlap = (aabbMin1[1] > aabbMax2[1] || aabbMax1[1] < aabbMin2[1]) ? false : overlap;
	return overlap;
}
#endif  //B3_USE_BANCHLESS

#endif  //B3_AABB_UTIL2
