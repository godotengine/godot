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
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"

// TODO Move to new file
#include "Collision_object_bullet.h"
#include "btRayShape.h"
#include "space_bullet.h"

GodotRayWorldAlgorithm::CreateFunc::CreateFunc(const btDiscreteDynamicsWorld *world)
	: m_world(world) {}

GodotRayWorldAlgorithm::SwappedCreateFunc::SwappedCreateFunc(const btDiscreteDynamicsWorld *world)
	: m_world(world) {}

GodotRayWorldAlgorithm::GodotRayWorldAlgorithm(const btDiscreteDynamicsWorld *world, btPersistentManifold *mf, const btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, bool isSwapped)
	: btActivatingCollisionAlgorithm(ci, body0Wrap, body1Wrap),
	  m_manifoldPtr(mf),
	  m_world(world),
	  m_isSwapped(isSwapped) {}

void GodotRayWorldAlgorithm::processCollision(const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut) {

	if (!m_manifoldPtr) {
		if (m_isSwapped) {
			m_manifoldPtr = m_dispatcher->getNewManifold(body1Wrap->getCollisionObject(), body0Wrap->getCollisionObject());
		} else {
			m_manifoldPtr = m_dispatcher->getNewManifold(body0Wrap->getCollisionObject(), body1Wrap->getCollisionObject());
		}
	}
	resultOut->setPersistentManifold(m_manifoldPtr);

	const btCollisionObject *ray_object;
	const btRayShape *ray_shape;
	btTransform ray_transform;

	btCollisionObject *other_object;
	const btCollisionShape *other_shape;
	btTransform other_transform;

	const btCollisionObjectWrapper *other_co_wrapper;

	if (m_isSwapped) {

		ray_object = body1Wrap->getCollisionObject();
		ray_shape = static_cast<const btRayShape *>(body1Wrap->getCollisionShape());
		ray_transform = body1Wrap->getWorldTransform();

		other_object = const_cast<btCollisionObject *>(body0Wrap->getCollisionObject());
		other_shape = static_cast<const btRayShape *>(body0Wrap->getCollisionShape());
		other_transform = body0Wrap->getWorldTransform();
		other_co_wrapper = body0Wrap;
	} else {

		ray_object = body0Wrap->getCollisionObject();
		ray_shape = static_cast<const btRayShape *>(body0Wrap->getCollisionShape());
		ray_transform = body0Wrap->getWorldTransform();

		other_object = const_cast<btCollisionObject *>(body1Wrap->getCollisionObject());
		other_shape = static_cast<const btRayShape *>(body1Wrap->getCollisionShape());
		other_transform = body1Wrap->getWorldTransform();
		other_co_wrapper = body1Wrap;
	}

	btTransform to(ray_transform * ray_shape->getSupportPoint());

	btCollisionWorld::ClosestRayResultCallback btResult(ray_transform.getOrigin(), to.getOrigin());
	btResult.m_collisionFilterGroup = INT_MAX;
	btResult.m_collisionFilterMask = INT_MAX;

	//m_world->rayTestSingle(ray_transform, to, other_object, other_shape, other_transform, btResult);
	m_world->rayTestSingleInternal(ray_transform, to, other_co_wrapper, btResult);

	if (btResult.hasHit()) {
		resultOut->addContactPoint(btResult.m_hitNormalWorld, btResult.m_hitPointWorld, 0);
	}
}

btScalar GodotRayWorldAlgorithm::calculateTimeOfImpact(btCollisionObject *body0, btCollisionObject *body1, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut) {
	return 0;
}

void GodotRayWorldAlgorithm::getAllContactManifolds(btManifoldArray &manifoldArray) {
}

GodotCollisionConfiguration::GodotCollisionConfiguration(const btDiscreteDynamicsWorld *world, const btDefaultCollisionConstructionInfo &constructionInfo)
	: btDefaultCollisionConfiguration(constructionInfo) {

	void *mem = NULL;

	mem = btAlignedAlloc(sizeof(GodotRayWorldAlgorithm::CreateFunc), 16);
	m_rayWorldCF = new (mem) GodotRayWorldAlgorithm::CreateFunc(world);

	mem = btAlignedAlloc(sizeof(GodotRayWorldAlgorithm::SwappedCreateFunc), 16);
	m_swappedRayWorldCF = new (mem) GodotRayWorldAlgorithm::SwappedCreateFunc(world);
}

GodotCollisionConfiguration::~GodotCollisionConfiguration() {
	m_rayWorldCF->~btCollisionAlgorithmCreateFunc();
	btAlignedFree(m_rayWorldCF);

	m_swappedRayWorldCF->~btCollisionAlgorithmCreateFunc();
	btAlignedFree(m_swappedRayWorldCF);
}

btCollisionAlgorithmCreateFunc *GodotCollisionConfiguration::getCollisionAlgorithmCreateFunc(int proxyType0, int proxyType1) {

	if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		// This collision is not supported
		return m_emptyCreateFunc;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0) {

		return m_rayWorldCF;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		return m_swappedRayWorldCF;
	} else {

		return btDefaultCollisionConfiguration::getCollisionAlgorithmCreateFunc(proxyType0, proxyType1);
	}
}

btCollisionAlgorithmCreateFunc *GodotCollisionConfiguration::getClosestPointsAlgorithmCreateFunc(int proxyType0, int proxyType1) {

	if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0 && CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		// This collision is not supported
		return m_emptyCreateFunc;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType0) {

		return m_rayWorldCF;
	} else if (CUSTOM_CONVEX_SHAPE_TYPE == proxyType1) {

		return m_swappedRayWorldCF;
	} else {

		return btDefaultCollisionConfiguration::getClosestPointsAlgorithmCreateFunc(proxyType0, proxyType1);
	}
}
