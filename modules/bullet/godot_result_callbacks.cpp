/*************************************************************************/
/*  godot_result_callbacks.cpp                                           */
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

#include "godot_result_callbacks.h"
#include "bullet_types_converter.h"
#include "collision_object_bullet.h"
#include "rigid_body_bullet.h"

bool GodotFilterCallback::test_collision_filters(uint32_t body0_collision_layer, uint32_t body0_collision_mask, uint32_t body1_collision_layer, uint32_t body1_collision_mask) {
	return body0_collision_layer & body1_collision_mask || body1_collision_layer & body0_collision_mask;
}

bool GodotFilterCallback::needBroadphaseCollision(btBroadphaseProxy *proxy0, btBroadphaseProxy *proxy1) const {
	return GodotFilterCallback::test_collision_filters(proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask, proxy1->m_collisionFilterGroup, proxy1->m_collisionFilterMask);
}

bool GodotClosestRayResultCallback::needsCollision(btBroadphaseProxy *proxy0) const {
	const bool needs = GodotFilterCallback::test_collision_filters(m_collisionFilterGroup, m_collisionFilterMask, proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask);
	if (needs) {
		btCollisionObject *btObj = static_cast<btCollisionObject *>(proxy0->m_clientObject);
		CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(btObj->getUserPointer());
		if (m_pickRay && gObj->is_ray_pickable()) {
			return true;
		} else if (m_exclude->has(gObj->get_self())) {
			return false;
		}
		return true;
	} else {
		return false;
	}
}

bool GodotAllConvexResultCallback::needsCollision(btBroadphaseProxy *proxy0) const {
	const bool needs = GodotFilterCallback::test_collision_filters(m_collisionFilterGroup, m_collisionFilterMask, proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask);
	if (needs) {
		btCollisionObject *btObj = static_cast<btCollisionObject *>(proxy0->m_clientObject);
		CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(btObj->getUserPointer());
		if (m_exclude->has(gObj->get_self())) {
			return false;
		}
		return true;
	} else {
		return false;
	}
}

btScalar GodotAllConvexResultCallback::addSingleResult(btCollisionWorld::LocalConvexResult &convexResult, bool normalInWorldSpace) {
	CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(convexResult.m_hitCollisionObject->getUserPointer());

	PhysicsDirectSpaceState::ShapeResult &result = m_results[count];

	result.shape = convexResult.m_localShapeInfo->m_triangleIndex; // "m_triangleIndex" Is a odd name but contains the compound shape ID
	result.rid = gObj->get_self();
	result.collider_id = gObj->get_instance_id();
	result.collider = 0 == result.collider_id ? NULL : ObjectDB::get_instance(result.collider_id);

	++count;
	return count < m_resultMax;
}

bool GodotKinClosestConvexResultCallback::needsCollision(btBroadphaseProxy *proxy0) const {
	const bool needs = GodotFilterCallback::test_collision_filters(m_collisionFilterGroup, m_collisionFilterMask, proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask);
	if (needs) {
		btCollisionObject *btObj = static_cast<btCollisionObject *>(proxy0->m_clientObject);
		CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(btObj->getUserPointer());
		if (gObj == m_self_object) {
			return false;
		} else {
			if (m_ignore_areas && gObj->getType() == CollisionObjectBullet::TYPE_AREA) {
				return false;
			} else if (m_self_object->has_collision_exception(gObj)) {
				return false;
			}
		}
		return true;
	} else {
		return false;
	}
}

bool GodotClosestConvexResultCallback::needsCollision(btBroadphaseProxy *proxy0) const {
	const bool needs = GodotFilterCallback::test_collision_filters(m_collisionFilterGroup, m_collisionFilterMask, proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask);
	if (needs) {
		btCollisionObject *btObj = static_cast<btCollisionObject *>(proxy0->m_clientObject);
		CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(btObj->getUserPointer());
		if (m_exclude->has(gObj->get_self())) {
			return false;
		}
		return true;
	} else {
		return false;
	}
}

btScalar GodotClosestConvexResultCallback::addSingleResult(btCollisionWorld::LocalConvexResult &convexResult, bool normalInWorldSpace) {
	btScalar res = btCollisionWorld::ClosestConvexResultCallback::addSingleResult(convexResult, normalInWorldSpace);
	m_shapeId = convexResult.m_localShapeInfo->m_triangleIndex; // "m_triangleIndex" Is a odd name but contains the compound shape ID
	return res;
}

bool GodotAllContactResultCallback::needsCollision(btBroadphaseProxy *proxy0) const {
	const bool needs = GodotFilterCallback::test_collision_filters(m_collisionFilterGroup, m_collisionFilterMask, proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask);
	if (needs) {
		btCollisionObject *btObj = static_cast<btCollisionObject *>(proxy0->m_clientObject);
		CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(btObj->getUserPointer());
		if (m_exclude->has(gObj->get_self())) {
			return false;
		}
		return true;
	} else {
		return false;
	}
}

btScalar GodotAllContactResultCallback::addSingleResult(btManifoldPoint &cp, const btCollisionObjectWrapper *colObj0Wrap, int partId0, int index0, const btCollisionObjectWrapper *colObj1Wrap, int partId1, int index1) {

	if (m_count >= m_resultMax)
		return cp.getDistance();

	if (cp.getDistance() <= 0) {

		PhysicsDirectSpaceState::ShapeResult &result = m_results[m_count];
		// Penetrated

		CollisionObjectBullet *colObj;
		if (m_self_object == colObj0Wrap->getCollisionObject()) {
			colObj = static_cast<CollisionObjectBullet *>(colObj1Wrap->getCollisionObject()->getUserPointer());
			result.shape = cp.m_index1;
		} else {
			colObj = static_cast<CollisionObjectBullet *>(colObj0Wrap->getCollisionObject()->getUserPointer());
			result.shape = cp.m_index0;
		}

		if (colObj)
			result.collider_id = colObj->get_instance_id();
		else
			result.collider_id = 0;
		result.collider = 0 == result.collider_id ? NULL : ObjectDB::get_instance(result.collider_id);
		result.rid = colObj->get_self();
		++m_count;
	}

	return cp.getDistance();
}

bool GodotContactPairContactResultCallback::needsCollision(btBroadphaseProxy *proxy0) const {
	const bool needs = GodotFilterCallback::test_collision_filters(m_collisionFilterGroup, m_collisionFilterMask, proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask);
	if (needs) {
		btCollisionObject *btObj = static_cast<btCollisionObject *>(proxy0->m_clientObject);
		CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(btObj->getUserPointer());
		if (m_exclude->has(gObj->get_self())) {
			return false;
		}
		return true;
	} else {
		return false;
	}
}

btScalar GodotContactPairContactResultCallback::addSingleResult(btManifoldPoint &cp, const btCollisionObjectWrapper *colObj0Wrap, int partId0, int index0, const btCollisionObjectWrapper *colObj1Wrap, int partId1, int index1) {

	if (m_self_object == colObj0Wrap->getCollisionObject()) {
		B_TO_G(cp.m_localPointA, m_results[m_count * 2 + 0]); // Local contact
		B_TO_G(cp.m_localPointB, m_results[m_count * 2 + 1]);
	} else {
		B_TO_G(cp.m_localPointB, m_results[m_count * 2 + 0]); // Local contact
		B_TO_G(cp.m_localPointA, m_results[m_count * 2 + 1]);
	}

	++m_count;

	return m_count < m_resultMax;
}

bool GodotRestInfoContactResultCallback::needsCollision(btBroadphaseProxy *proxy0) const {
	const bool needs = GodotFilterCallback::test_collision_filters(m_collisionFilterGroup, m_collisionFilterMask, proxy0->m_collisionFilterGroup, proxy0->m_collisionFilterMask);
	if (needs) {
		btCollisionObject *btObj = static_cast<btCollisionObject *>(proxy0->m_clientObject);
		CollisionObjectBullet *gObj = static_cast<CollisionObjectBullet *>(btObj->getUserPointer());
		if (m_exclude->has(gObj->get_self())) {
			return false;
		}
		return true;
	} else {
		return false;
	}
}

btScalar GodotRestInfoContactResultCallback::addSingleResult(btManifoldPoint &cp, const btCollisionObjectWrapper *colObj0Wrap, int partId0, int index0, const btCollisionObjectWrapper *colObj1Wrap, int partId1, int index1) {

	if (cp.getDistance() <= m_min_distance) {
		m_min_distance = cp.getDistance();

		CollisionObjectBullet *colObj;
		if (m_self_object == colObj0Wrap->getCollisionObject()) {
			colObj = static_cast<CollisionObjectBullet *>(colObj1Wrap->getCollisionObject()->getUserPointer());
			m_result->shape = cp.m_index1;
			B_TO_G(cp.getPositionWorldOnB(), m_result->point);
			m_rest_info_bt_point = cp.getPositionWorldOnB();
			m_rest_info_collision_object = colObj1Wrap->getCollisionObject();
		} else {
			colObj = static_cast<CollisionObjectBullet *>(colObj0Wrap->getCollisionObject()->getUserPointer());
			m_result->shape = cp.m_index0;
			B_TO_G(cp.m_normalWorldOnB * -1, m_result->normal);
			m_rest_info_bt_point = cp.getPositionWorldOnA();
			m_rest_info_collision_object = colObj0Wrap->getCollisionObject();
		}

		if (colObj)
			m_result->collider_id = colObj->get_instance_id();
		else
			m_result->collider_id = 0;
		m_result->rid = colObj->get_self();

		m_collided = true;
	}

	return cp.getDistance();
}

void GodotDeepPenetrationContactResultCallback::addContactPoint(const btVector3 &normalOnBInWorld, const btVector3 &pointInWorldOnB, btScalar depth) {

	if (depth < 0) {
		// Has penetration
		if (m_most_penetrated_distance > depth) {

			bool isSwapped = m_manifoldPtr->getBody0() != m_body0Wrap->getCollisionObject();

			m_most_penetrated_distance = depth;
			m_pointCollisionObject = (isSwapped ? m_body0Wrap : m_body1Wrap)->getCollisionObject();
			m_other_compound_shape_index = isSwapped ? m_index1 : m_index0;
			m_pointNormalWorld = isSwapped ? normalOnBInWorld * -1 : normalOnBInWorld;
			m_pointWorld = isSwapped ? (pointInWorldOnB + normalOnBInWorld * depth) : pointInWorldOnB;
			m_penetration_distance = depth;
		}
	}
}
