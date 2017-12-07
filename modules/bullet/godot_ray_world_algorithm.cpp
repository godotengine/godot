/*************************************************************************/
/*  godot_ray_world_algorithm.cpp                                        */
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

#include "godot_ray_world_algorithm.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "btRayShape.h"
#include "collision_object_bullet.h"

GodotRayWorldAlgorithm::CreateFunc::CreateFunc(const btDiscreteDynamicsWorld *world) :
		m_world(world) {}

GodotRayWorldAlgorithm::SwappedCreateFunc::SwappedCreateFunc(const btDiscreteDynamicsWorld *world) :
		m_world(world) {}

GodotRayWorldAlgorithm::GodotRayWorldAlgorithm(const btDiscreteDynamicsWorld *world, btPersistentManifold *mf, const btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, bool isSwapped) :
		btActivatingCollisionAlgorithm(ci, body0Wrap, body1Wrap),
		m_manifoldPtr(mf),
		m_ownManifold(false),
		m_world(world),
		m_isSwapped(isSwapped) {}

GodotRayWorldAlgorithm::~GodotRayWorldAlgorithm() {
	if (m_ownManifold && m_manifoldPtr) {
		m_dispatcher->releaseManifold(m_manifoldPtr);
	}
}

void GodotRayWorldAlgorithm::processCollision(const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut) {

	if (!m_manifoldPtr) {
		if (m_isSwapped) {
			m_manifoldPtr = m_dispatcher->getNewManifold(body1Wrap->getCollisionObject(), body0Wrap->getCollisionObject());
		} else {
			m_manifoldPtr = m_dispatcher->getNewManifold(body0Wrap->getCollisionObject(), body1Wrap->getCollisionObject());
		}
		m_ownManifold = true;
	}
	m_manifoldPtr->clearManifold();
	resultOut->setPersistentManifold(m_manifoldPtr);

	const btRayShape *ray_shape;
	btTransform ray_transform;

	const btCollisionObjectWrapper *other_co_wrapper;

	if (m_isSwapped) {

		ray_shape = static_cast<const btRayShape *>(body1Wrap->getCollisionShape());
		ray_transform = body1Wrap->getWorldTransform();

		other_co_wrapper = body0Wrap;
	} else {

		ray_shape = static_cast<const btRayShape *>(body0Wrap->getCollisionShape());
		ray_transform = body0Wrap->getWorldTransform();

		other_co_wrapper = body1Wrap;
	}

	btTransform to(ray_transform * ray_shape->getSupportPoint());

	btCollisionWorld::ClosestRayResultCallback btResult(ray_transform.getOrigin(), to.getOrigin());

	m_world->rayTestSingleInternal(ray_transform, to, other_co_wrapper, btResult);

	if (btResult.hasHit()) {
		btVector3 ray_normal(to.getOrigin() - ray_transform.getOrigin());
		ray_normal.normalize();
		ray_normal *= -1;
		resultOut->addContactPoint(ray_normal, btResult.m_hitPointWorld, ray_shape->getScaledLength() * (btResult.m_closestHitFraction - 1));
	}
}

btScalar GodotRayWorldAlgorithm::calculateTimeOfImpact(btCollisionObject *body0, btCollisionObject *body1, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut) {
	return 1;
}
