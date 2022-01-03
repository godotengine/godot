/*************************************************************************/
/*  btRayShape.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "btRayShape.h"

#include "core/math/math_funcs.h"

#include <LinearMath/btAabbUtil2.h>

/**
	@author AndreaCatania
*/

btRayShape::btRayShape(btScalar length) :
		btConvexInternalShape() {
	m_shapeType = CUSTOM_CONVEX_SHAPE_TYPE;
	setLength(length);
}

btRayShape::~btRayShape() {
}

void btRayShape::setLength(btScalar p_length) {
	m_length = p_length;
	reload_cache();
}

void btRayShape::setMargin(btScalar margin) {
	btConvexInternalShape::setMargin(margin);
	reload_cache();
}

void btRayShape::setSlipsOnSlope(bool p_slipsOnSlope) {
	slipsOnSlope = p_slipsOnSlope;
}

btVector3 btRayShape::localGetSupportingVertex(const btVector3 &vec) const {
	return localGetSupportingVertexWithoutMargin(vec) + (m_shapeAxis * m_collisionMargin);
}

btVector3 btRayShape::localGetSupportingVertexWithoutMargin(const btVector3 &vec) const {
	if (vec.z() > 0) {
		return m_shapeAxis * m_cacheScaledLength;
	} else {
		return btVector3(0, 0, 0);
	}
}

void btRayShape::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3 *vectors, btVector3 *supportVerticesOut, int numVectors) const {
	for (int i = 0; i < numVectors; ++i) {
		supportVerticesOut[i] = localGetSupportingVertexWithoutMargin(vectors[i]);
	}
}

void btRayShape::getAabb(const btTransform &t, btVector3 &aabbMin, btVector3 &aabbMax) const {
	btVector3 localAabbMin(0, 0, 0);
	btVector3 localAabbMax(m_shapeAxis * m_cacheScaledLength);
	btTransformAabb(localAabbMin, localAabbMax, m_collisionMargin, t, aabbMin, aabbMax);
}

void btRayShape::calculateLocalInertia(btScalar mass, btVector3 &inertia) const {
	inertia.setZero();
}

int btRayShape::getNumPreferredPenetrationDirections() const {
	return 0;
}

void btRayShape::getPreferredPenetrationDirection(int index, btVector3 &penetrationVector) const {
	penetrationVector.setZero();
}

void btRayShape::reload_cache() {
	m_cacheScaledLength = m_length * m_localScaling[2];

	m_cacheSupportPoint.setIdentity();
	m_cacheSupportPoint.setOrigin(m_shapeAxis * m_cacheScaledLength);
}
