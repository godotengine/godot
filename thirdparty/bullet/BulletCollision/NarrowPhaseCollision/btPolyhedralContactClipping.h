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

///This file was written by Erwin Coumans

#ifndef BT_POLYHEDRAL_CONTACT_CLIPPING_H
#define BT_POLYHEDRAL_CONTACT_CLIPPING_H

#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btTransform.h"
#include "btDiscreteCollisionDetectorInterface.h"

class btConvexPolyhedron;

typedef btAlignedObjectArray<btVector3> btVertexArray;

// Clips a face to the back of a plane
struct btPolyhedralContactClipping
{
	static void clipHullAgainstHull(const btVector3& separatingNormal1, const btConvexPolyhedron& hullA, const btConvexPolyhedron& hullB, const btTransform& transA, const btTransform& transB, const btScalar minDist, btScalar maxDist, btVertexArray& worldVertsB1, btVertexArray& worldVertsB2, btDiscreteCollisionDetectorInterface::Result& resultOut);

	static void clipFaceAgainstHull(const btVector3& separatingNormal, const btConvexPolyhedron& hullA, const btTransform& transA, btVertexArray& worldVertsB1, btVertexArray& worldVertsB2, const btScalar minDist, btScalar maxDist, btDiscreteCollisionDetectorInterface::Result& resultOut);

	static bool findSeparatingAxis(const btConvexPolyhedron& hullA, const btConvexPolyhedron& hullB, const btTransform& transA, const btTransform& transB, btVector3& sep, btDiscreteCollisionDetectorInterface::Result& resultOut);

	///the clipFace method is used internally
	static void clipFace(const btVertexArray& pVtxIn, btVertexArray& ppVtxOut, const btVector3& planeNormalWS, btScalar planeEqWS);
};

#endif  // BT_POLYHEDRAL_CONTACT_CLIPPING_H
