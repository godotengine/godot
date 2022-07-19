
/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

#ifndef _BT_CONVEX_UTILITY_H
#define _BT_CONVEX_UTILITY_H

#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Transform.h"

struct b3MyFace
{
	b3AlignedObjectArray<int> m_indices;
	b3Scalar m_plane[4];
};

B3_ATTRIBUTE_ALIGNED16(class)
b3ConvexUtility
{
public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	b3Vector3 m_localCenter;
	b3Vector3 m_extents;
	b3Vector3 mC;
	b3Vector3 mE;
	b3Scalar m_radius;

	b3AlignedObjectArray<b3Vector3> m_vertices;
	b3AlignedObjectArray<b3MyFace> m_faces;
	b3AlignedObjectArray<b3Vector3> m_uniqueEdges;

	b3ConvexUtility()
	{
	}
	virtual ~b3ConvexUtility();

	bool initializePolyhedralFeatures(const b3Vector3* orgVertices, int numVertices, bool mergeCoplanarTriangles = true);

	void initialize();
	bool testContainment() const;
};
#endif
