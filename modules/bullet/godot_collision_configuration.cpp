/*************************************************************************/
/*  godot_collision_configuration.cpp                                    */
/*  Author: AndreaCatania                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "godot_collision_configuration.h"
#include "thirdparty/Bullet/src/BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"

GodotRayConvexAlgorithm::GodotRayConvexAlgorithm(btPersistentManifold *mf, const btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, btConvexPenetrationDepthSolver *pdSolver, int numPerturbationIterations, int minimumPointsPerturbationThreshold, bool isSwapped)
	: btConvexConvexAlgorithm(mf, ci, body0Wrap, body1Wrap, pdSolver, numPerturbationIterations, minimumPointsPerturbationThreshold), m_isSwapped(isSwapped) {
}

void GodotRayConvexAlgorithm::processCollision(const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut) {
	btConvexConvexAlgorithm::processCollision(body0Wrap, body1Wrap, dispatchInfo, resultOut);

	btVector3 rayNormal(0, 1, 0);
	if (m_isSwapped) {
		rayNormal *= -1;
	}

	// Clamping all lateral contacts
	for (int x = resultOut->getPersistentManifold()->getNumContacts() - 1; 0 <= x; --x) {
		resultOut->getPersistentManifold()->getContactPoint(x).m_normalWorldOnB = rayNormal;
	}
}

GodotRayConvexAlgorithm::CreateFunc::CreateFunc(btConvexPenetrationDepthSolver *pdSolver)
	: btConvexConvexAlgorithm::CreateFunc(pdSolver) {}

GodotRayConvexAlgorithm::SwappedCreateFunc::SwappedCreateFunc(btConvexPenetrationDepthSolver *pdSolver)
	: btConvexConvexAlgorithm::CreateFunc(pdSolver) {}

GodotRayConcaveAlgorithm::GodotRayConcaveAlgorithm(const btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, bool isSwapped1)
	: btConvexConcaveCollisionAlgorithm(ci, body0Wrap, body1Wrap, isSwapped1), m_isSwapped(isSwapped1) {
}

void GodotRayConcaveAlgorithm::processCollision(const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut) {
	btConvexConcaveCollisionAlgorithm::processCollision(body0Wrap, body1Wrap, dispatchInfo, resultOut);

	btVector3 rayNormal(0, 1, 0);
	if (m_isSwapped) {
		rayNormal *= -1;
	}

	// Clamping all lateral contacts
	for (int x = resultOut->getPersistentManifold()->getNumContacts() - 1; 0 <= x; --x) {
		resultOut->getPersistentManifold()->getContactPoint(x).m_normalWorldOnB = rayNormal;
	}
}

GodotCollisionConfiguration::GodotCollisionConfiguration(const btDefaultCollisionConstructionInfo &constructionInfo)
	: btDefaultCollisionConfiguration(constructionInfo) {

	void *mem = NULL;

	mem = btAlignedAlloc(sizeof(GodotRayConvexAlgorithm::CreateFunc), 16);
	m_rayConvexCF = new (mem) GodotRayConvexAlgorithm::CreateFunc(m_pdSolver);

	mem = btAlignedAlloc(sizeof(GodotRayConvexAlgorithm::SwappedCreateFunc), 16);
	m_swappedRayConvexCF = new (mem) GodotRayConvexAlgorithm::SwappedCreateFunc(m_pdSolver);

	mem = btAlignedAlloc(sizeof(GodotRayConcaveAlgorithm::CreateFunc), 16);
	m_rayConcaveCF = new (mem) GodotRayConcaveAlgorithm::CreateFunc;

	mem = btAlignedAlloc(sizeof(GodotRayConcaveAlgorithm::SwappedCreateFunc), 16);
	m_swappedRayConcaveCF = new (mem) GodotRayConcaveAlgorithm::SwappedCreateFunc;
}

GodotCollisionConfiguration::~GodotCollisionConfiguration() {
	m_rayConvexCF->~btCollisionAlgorithmCreateFunc();
	btAlignedFree(m_rayConvexCF);

	m_swappedRayConvexCF->~btCollisionAlgorithmCreateFunc();
	btAlignedFree(m_swappedRayConvexCF);

	m_rayConcaveCF->~btCollisionAlgorithmCreateFunc();
	btAlignedFree(m_rayConcaveCF);

	m_swappedRayConcaveCF->~btCollisionAlgorithmCreateFunc();
	btAlignedFree(m_swappedRayConcaveCF);
}

btCollisionAlgorithmCreateFunc *GodotCollisionConfiguration::getCollisionAlgorithmCreateFunc(int proxyType0, int proxyType1) {

	if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		// This collision is not supported
		return m_emptyCreateFunc;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && btBroadphaseProxy::isConvex(proxyType1)) {

		return m_rayConvexCF;
	} else if (btBroadphaseProxy::isConvex(proxyType0) && CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		return m_swappedRayConvexCF;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && btBroadphaseProxy::isConcave(proxyType1)) {

		return m_rayConcaveCF;
	} else if (btBroadphaseProxy::isConcave(proxyType0) && CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		return m_swappedRayConcaveCF;
	} else {

		return btDefaultCollisionConfiguration::getCollisionAlgorithmCreateFunc(proxyType0, proxyType1);
	}
}

btCollisionAlgorithmCreateFunc *GodotCollisionConfiguration::getClosestPointsAlgorithmCreateFunc(int proxyType0, int proxyType1) {

	if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		// This collision is not supported
		return m_emptyCreateFunc;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && btBroadphaseProxy::isConvex(proxyType1)) {

		return m_rayConvexCF;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && btBroadphaseProxy::isConcave(proxyType1)) {

		return m_rayConcaveCF;
	} else if (btBroadphaseProxy::isConcave(proxyType0) && CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		return m_swappedRayConcaveCF;
	} else {

		return btDefaultCollisionConfiguration::getClosestPointsAlgorithmCreateFunc(proxyType0, proxyType1);
	}
}
