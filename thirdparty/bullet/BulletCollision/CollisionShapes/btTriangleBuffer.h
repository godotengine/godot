/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_TRIANGLE_BUFFER_H
#define BT_TRIANGLE_BUFFER_H

#include "btTriangleCallback.h"
#include "LinearMath/btAlignedObjectArray.h"

struct	btTriangle
{
	btVector3	m_vertex0;
	btVector3	m_vertex1;
	btVector3	m_vertex2;
	int	m_partId;
	int	m_triangleIndex;
};

///The btTriangleBuffer callback can be useful to collect and store overlapping triangles between AABB and concave objects that support 'processAllTriangles'
///Example usage of this class:
///			btTriangleBuffer	triBuf;
///			concaveShape->processAllTriangles(&triBuf,aabbMin, aabbMax);
///			for (int i=0;i<triBuf.getNumTriangles();i++)
///			{
///				const btTriangle& tri = triBuf.getTriangle(i);
///				//do something useful here with the triangle
///			}
class btTriangleBuffer : public btTriangleCallback
{

	btAlignedObjectArray<btTriangle>	m_triangleBuffer;
	
public:


	virtual void processTriangle(btVector3* triangle, int partId, int triangleIndex);
	
	int	getNumTriangles() const
	{
		return int(m_triangleBuffer.size());
	}
	
	const btTriangle&	getTriangle(int index) const
	{
		return m_triangleBuffer[index];
	}

	void	clearBuffer()
	{
		m_triangleBuffer.clear();
	}
	
};


#endif //BT_TRIANGLE_BUFFER_H

