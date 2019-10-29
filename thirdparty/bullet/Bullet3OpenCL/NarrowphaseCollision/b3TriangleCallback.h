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

#ifndef B3_TRIANGLE_CALLBACK_H
#define B3_TRIANGLE_CALLBACK_H

#include "Bullet3Common/b3Vector3.h"

///The b3TriangleCallback provides a callback for each overlapping triangle when calling processAllTriangles.
///This callback is called by processAllTriangles for all b3ConcaveShape derived class, such as  b3BvhTriangleMeshShape, b3StaticPlaneShape and b3HeightfieldTerrainShape.
class b3TriangleCallback
{
public:
	virtual ~b3TriangleCallback();
	virtual void processTriangle(b3Vector3* triangle, int partId, int triangleIndex) = 0;
};

class b3InternalTriangleIndexCallback
{
public:
	virtual ~b3InternalTriangleIndexCallback();
	virtual void internalProcessTriangleIndex(b3Vector3* triangle, int partId, int triangleIndex) = 0;
};

#endif  //B3_TRIANGLE_CALLBACK_H
