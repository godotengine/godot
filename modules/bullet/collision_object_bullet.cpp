/*************************************************************************/
/*  collision_object_bullet.cpp                                          */
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

#include "collision_object_bullet.h"

#include "area_bullet.h"
#include "bullet_physics_server.h"
#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "shape_bullet.h"
#include "space_bullet.h"

#include <btBulletCollisionCommon.h>

/**
	@author AndreaCatania
*/

// We enable dynamic AABB tree so that we can actually perform a broadphase on bodies with compound collision shapes.
// This is crucial for the performance of kinematic bodies and for bodies with transforming shapes.
#define enableDynamicAabbTree true

CollisionObjectBullet::ShapeWrapper::~ShapeWrapper() {}

void CollisionObjectBullet::ShapeWrapper::set_transform(const Transform &p_transform) {
	G_TO_B(p_transform.get_basis().get_scale_abs(), scale);
	G_TO_B(p_transform, transform);
	UNSCALE_BT_BASIS(transform);
}

void CollisionObjectBullet::ShapeWrapper::set_transform(const btTransform &p_transform) {
	transform = p_transform;
}

btTransform CollisionObjectBullet::ShapeWrapper::get_adjusted_transform() const {
	if (shape->get_type() == PhysicsServer::SHAPE_HEIGHTMAP) {
		const HeightMapShapeBullet *hm_shape = (const HeightMapShapeBullet *)shape; // should be safe to cast now
		btTransform adjusted_transform;

		// Bullet centers our heightmap:
		// https://github.com/bulletphysics/bullet3/blob/master/src/BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h#L33
		// This is really counter intuitive so we're adjusting for it

		adjusted_transform.setIdentity();
		adjusted_transform.setOrigin(btVector3(0.0, hm_shape->min_height + ((hm_shape->max_height - hm_shape->min_height) * 0.5), 0.0));
		adjusted_transform *= transform;

		return adjusted_transform;
	} else {
		return transform;
	}
}

void CollisionObjectBullet::ShapeWrapper::claim_bt_shape(const btVector3 &body_scale) {
	if (!bt_shape) {
		if (active) {
			bt_shape = shape->create_bt_shape(scale * body_scale);
		} else {
			bt_shape = ShapeBullet::create_shape_empty();
		}
	}
}

CollisionObjectBullet::CollisionObjectBullet(Type p_type) :
		RIDBullet(),
		type(p_type),
		instance_id(0),
		collisionLayer(0),
		collisionMask(0),
		collisionsEnabled(true),
		m_isStatic(false),
		ray_pickable(false),
		bt_collision_object(nullptr),
		body_scale(1., 1., 1.),
		force_shape_reset(false),
		space(nullptr) {}

CollisionObjectBullet::~CollisionObjectBullet() {
	for (int i = 0; i < areasOverlapped.size(); i++) {
		areasOverlapped[i]->remove_object_overlaps(this);
	}
	destroyBulletCollisionObject();
}

bool equal(real_t first, real_t second) {
	return Math::abs(first - second) <= 0.001f;
}

void CollisionObjectBullet::set_body_scale(const Vector3 &p_new_scale) {
	if (!equal(p_new_scale[0], body_scale[0]) || !equal(p_new_scale[1], body_scale[1]) || !equal(p_new_scale[2], body_scale[2])) {
		body_scale = p_new_scale;
		body_scale_changed();
	}
}

btVector3 CollisionObjectBullet::get_bt_body_scale() const {
	btVector3 s;
	G_TO_B(body_scale, s);
	return s;
}

void CollisionObjectBullet::body_scale_changed() {
	force_shape_reset = true;
}

void CollisionObjectBullet::destroyBulletCollisionObject() {
	bulletdelete(bt_collision_object);
}

void CollisionObjectBullet::setupBulletCollisionObject(btCollisionObject *p_collisionObject) {
	bt_collision_object = p_collisionObject;
	bt_collision_object->setUserPointer(this);
	bt_collision_object->setUserIndex(type);
	// Force the enabling of collision and avoid problems
	set_collision_enabled(collisionsEnabled);
	p_collisionObject->setCollisionFlags(p_collisionObject->getCollisionFlags() | btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK);
}

void CollisionObjectBullet::add_collision_exception(const CollisionObjectBullet *p_ignoreCollisionObject) {
	exceptions.insert(p_ignoreCollisionObject->get_self());
	if (!bt_collision_object) {
		return;
	}
	bt_collision_object->setIgnoreCollisionCheck(p_ignoreCollisionObject->bt_collision_object, true);
	if (space) {
		space->get_broadphase()->getOverlappingPairCache()->cleanProxyFromPairs(bt_collision_object->getBroadphaseHandle(), space->get_dispatcher());
	}
}

void CollisionObjectBullet::remove_collision_exception(const CollisionObjectBullet *p_ignoreCollisionObject) {
	exceptions.erase(p_ignoreCollisionObject->get_self());
	if (!bt_collision_object) {
		return;
	}
	bt_collision_object->setIgnoreCollisionCheck(p_ignoreCollisionObject->bt_collision_object, false);
	if (space) {
		space->get_broadphase()->getOverlappingPairCache()->cleanProxyFromPairs(bt_collision_object->getBroadphaseHandle(), space->get_dispatcher());
	}
}

bool CollisionObjectBullet::has_collision_exception(const CollisionObjectBullet *p_otherCollisionObject) const {
	return exceptions.has(p_otherCollisionObject->get_self());
}

void CollisionObjectBullet::set_collision_enabled(bool p_enabled) {
	collisionsEnabled = p_enabled;
	if (!bt_collision_object) {
		return;
	}
	if (collisionsEnabled) {
		bt_collision_object->setCollisionFlags(bt_collision_object->getCollisionFlags() & (~btCollisionObject::CF_NO_CONTACT_RESPONSE));
	} else {
		bt_collision_object->setCollisionFlags(bt_collision_object->getCollisionFlags() | btCollisionObject::CF_NO_CONTACT_RESPONSE);
	}
}

bool CollisionObjectBullet::is_collisions_response_enabled() {
	return collisionsEnabled;
}

void CollisionObjectBullet::notify_new_overlap(AreaBullet *p_area) {
	if (areasOverlapped.find(p_area) == -1) {
		areasOverlapped.push_back(p_area);
	}
}

void CollisionObjectBullet::on_exit_area(AreaBullet *p_area) {
	areasOverlapped.erase(p_area);
}

void CollisionObjectBullet::set_godot_object_flags(int flags) {
	bt_collision_object->setUserIndex2(flags);
	updated = true;
}

int CollisionObjectBullet::get_godot_object_flags() const {
	return bt_collision_object->getUserIndex2();
}

void CollisionObjectBullet::set_transform(const Transform &p_global_transform) {
	set_body_scale(p_global_transform.basis.get_scale_abs());

	btTransform bt_transform;
	G_TO_B(p_global_transform, bt_transform);
	UNSCALE_BT_BASIS(bt_transform);

	set_transform__bullet(bt_transform);
}

Transform CollisionObjectBullet::get_transform() const {
	Transform t;
	B_TO_G(get_transform__bullet(), t);
	t.basis.scale(body_scale);
	return t;
}

void CollisionObjectBullet::set_transform__bullet(const btTransform &p_global_transform) {
	bt_collision_object->setWorldTransform(p_global_transform);
	notify_transform_changed();
}

const btTransform &CollisionObjectBullet::get_transform__bullet() const {
	return bt_collision_object->getWorldTransform();
}

void CollisionObjectBullet::notify_transform_changed() {
	updated = true;
}

RigidCollisionObjectBullet::RigidCollisionObjectBullet(Type p_type) :
		CollisionObjectBullet(p_type),
		mainShape(nullptr) {
}

RigidCollisionObjectBullet::~RigidCollisionObjectBullet() {
	remove_all_shapes(true, true);
	if (mainShape && mainShape->isCompound()) {
		bulletdelete(mainShape);
	}
}

void RigidCollisionObjectBullet::add_shape(ShapeBullet *p_shape, const Transform &p_transform, bool p_disabled) {
	shapes.push_back(ShapeWrapper(p_shape, p_transform, !p_disabled));
	p_shape->add_owner(this);
	reload_shapes();
}

void RigidCollisionObjectBullet::set_shape(int p_index, ShapeBullet *p_shape) {
	ShapeWrapper &shp = shapes.write[p_index];
	shp.shape->remove_owner(this);
	p_shape->add_owner(this);
	shp.shape = p_shape;
	reload_shapes();
}

int RigidCollisionObjectBullet::get_shape_count() const {
	return shapes.size();
}

ShapeBullet *RigidCollisionObjectBullet::get_shape(int p_index) const {
	return shapes[p_index].shape;
}

btCollisionShape *RigidCollisionObjectBullet::get_bt_shape(int p_index) const {
	return shapes[p_index].bt_shape;
}

int RigidCollisionObjectBullet::find_shape(ShapeBullet *p_shape) const {
	const int size = shapes.size();
	for (int i = 0; i < size; ++i) {
		if (shapes[i].shape == p_shape) {
			return i;
		}
	}
	return -1;
}

void RigidCollisionObjectBullet::remove_shape_full(ShapeBullet *p_shape) {
	// Remove the shape, all the times it appears
	// Reverse order required for delete.
	for (int i = shapes.size() - 1; 0 <= i; --i) {
		if (p_shape == shapes[i].shape) {
			internal_shape_destroy(i);
			shapes.remove(i);
		}
	}
	reload_shapes();
}

void RigidCollisionObjectBullet::remove_shape_full(int p_index) {
	ERR_FAIL_INDEX(p_index, get_shape_count());
	internal_shape_destroy(p_index);
	shapes.remove(p_index);
	reload_shapes();
}

void RigidCollisionObjectBullet::remove_all_shapes(bool p_permanentlyFromThisBody, bool p_force_not_reload) {
	// Reverse order required for delete.
	for (int i = shapes.size() - 1; 0 <= i; --i) {
		internal_shape_destroy(i, p_permanentlyFromThisBody);
	}
	shapes.clear();
	if (!p_force_not_reload) {
		reload_shapes();
	}
}

void RigidCollisionObjectBullet::set_shape_transform(int p_index, const Transform &p_transform) {
	ERR_FAIL_INDEX(p_index, get_shape_count());

	shapes.write[p_index].set_transform(p_transform);
	shape_changed(p_index);
}

const btTransform &RigidCollisionObjectBullet::get_bt_shape_transform(int p_index) const {
	return shapes[p_index].transform;
}

Transform RigidCollisionObjectBullet::get_shape_transform(int p_index) const {
	Transform trs;
	B_TO_G(shapes[p_index].transform, trs);
	return trs;
}

void RigidCollisionObjectBullet::set_shape_disabled(int p_index, bool p_disabled) {
	if (shapes[p_index].active != p_disabled) {
		return;
	}
	shapes.write[p_index].active = !p_disabled;
	shape_changed(p_index);
}

bool RigidCollisionObjectBullet::is_shape_disabled(int p_index) {
	return !shapes[p_index].active;
}

void RigidCollisionObjectBullet::shape_changed(int p_shape_index) {
	ShapeWrapper &shp = shapes.write[p_shape_index];
	if (shp.bt_shape == mainShape) {
		mainShape = nullptr;
	}
	bulletdelete(shp.bt_shape);
	reload_shapes();
}

void RigidCollisionObjectBullet::reload_shapes() {
	if (mainShape && mainShape->isCompound()) {
		// Destroy compound
		bulletdelete(mainShape);
	}

	mainShape = nullptr;

	ShapeWrapper *shpWrapper;
	const int shape_count = shapes.size();

	// Reset shape if required
	if (force_shape_reset) {
		for (int i(0); i < shape_count; ++i) {
			shpWrapper = &shapes.write[i];
			bulletdelete(shpWrapper->bt_shape);
		}
		force_shape_reset = false;
	}

	const btVector3 body_scale(get_bt_body_scale());

	// Try to optimize by not using compound
	if (1 == shape_count) {
		shpWrapper = &shapes.write[0];
		btTransform transform = shpWrapper->get_adjusted_transform();
		if (transform.getOrigin().isZero() && transform.getBasis() == transform.getBasis().getIdentity()) {
			shpWrapper->claim_bt_shape(body_scale);
			mainShape = shpWrapper->bt_shape;
			main_shape_changed();
			return;
		}
	}

	// Optimization not possible use a compound shape
	btCompoundShape *compoundShape = bulletnew(btCompoundShape(enableDynamicAabbTree, shape_count));

	for (int i(0); i < shape_count; ++i) {
		shpWrapper = &shapes.write[i];
		shpWrapper->claim_bt_shape(body_scale);
		btTransform scaled_shape_transform(shpWrapper->get_adjusted_transform());
		scaled_shape_transform.getOrigin() *= body_scale;
		compoundShape->addChildShape(scaled_shape_transform, shpWrapper->bt_shape);
	}

	compoundShape->recalculateLocalAabb();
	mainShape = compoundShape;
	main_shape_changed();
}

void RigidCollisionObjectBullet::body_scale_changed() {
	CollisionObjectBullet::body_scale_changed();
	reload_shapes();
}

void RigidCollisionObjectBullet::internal_shape_destroy(int p_index, bool p_permanentlyFromThisBody) {
	ShapeWrapper &shp = shapes.write[p_index];
	shp.shape->remove_owner(this, p_permanentlyFromThisBody);
	if (shp.bt_shape == mainShape) {
		mainShape = nullptr;
	}
	bulletdelete(shp.bt_shape);
}
