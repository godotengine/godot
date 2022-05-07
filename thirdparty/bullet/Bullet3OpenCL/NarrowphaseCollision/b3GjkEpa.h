/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2008 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software in a
product, an acknowledgment in the product documentation would be appreciated
but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

/*
GJK-EPA collision solver by Nathanael Presson, 2008
*/
#ifndef B3_GJK_EPA2_H
#define B3_GJK_EPA2_H

#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Transform.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"

///btGjkEpaSolver contributed under zlib by Nathanael Presson
struct b3GjkEpaSolver2
{
	struct sResults
	{
		enum eStatus
		{
			Separated,   /* Shapes doesnt penetrate												*/
			Penetrating, /* Shapes are penetrating												*/
			GJK_Failed,  /* GJK phase fail, no big issue, shapes are probably just 'touching'	*/
			EPA_Failed   /* EPA phase fail, bigger problem, need to save parameters, and debug	*/
		} status;
		b3Vector3 witnesses[2];
		b3Vector3 normal;
		b3Scalar distance;
	};

	static int StackSizeRequirement();

	static bool Distance(const b3Transform& transA, const b3Transform& transB,
						 const b3ConvexPolyhedronData* hullA, const b3ConvexPolyhedronData* hullB,
						 const b3AlignedObjectArray<b3Vector3>& verticesA,
						 const b3AlignedObjectArray<b3Vector3>& verticesB,
						 const b3Vector3& guess,
						 sResults& results);

	static bool Penetration(const b3Transform& transA, const b3Transform& transB,
							const b3ConvexPolyhedronData* hullA, const b3ConvexPolyhedronData* hullB,
							const b3AlignedObjectArray<b3Vector3>& verticesA,
							const b3AlignedObjectArray<b3Vector3>& verticesB,
							const b3Vector3& guess,
							sResults& results,
							bool usemargins = true);
#if 0
static b3Scalar	SignedDistance(	const b3Vector3& position,
								b3Scalar margin,
								const btConvexShape* shape,
								const btTransform& wtrs,
								sResults& results);
							
static bool		SignedDistance(	const btConvexShape* shape0,const btTransform& wtrs0,
								const btConvexShape* shape1,const btTransform& wtrs1,
								const b3Vector3& guess,
								sResults& results);
#endif
};

#endif  //B3_GJK_EPA2_H
