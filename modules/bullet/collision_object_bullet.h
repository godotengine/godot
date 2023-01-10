/**************************************************************************/
/*  collision_object_bullet.h                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef COLLISION_OBJECT_BULLET_H
#define COLLISION_OBJECT_BULLET_H

#include "core/math/transform.h"
#include "core/math/vector3.h"
#include "core/object.h"
#include "core/vset.h"
#include "shape_owner_bullet.h"

#if defined(__clang__) && (__clang_major__ >= 13)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-copy-with-user-provided-copy"
#endif

#include <LinearMath/btTransform.h>

#if defined(__clang__) && (__clang_major__ >= 13)
#pragma clang diagnostic pop
#endif

/**
	@author AndreaCatania
*/

class AreaBullet;
class ShapeBullet;
class btCollisionObject;
class btCompoundShape;
class btCollisionShape;
class SpaceBullet;

class CollisionObjectBullet : public RIDBullet {
public:
	enum GodotObjectFlags {
		GOF_IS_MONITORING_AREA = 1 << 0
		// FLAG2 = 1 << 1,
		// FLAG3 = 1 << 2,
		// FLAG4 = 1 << 3,
		// FLAG5 = 1 << 4,
		// FLAG6 = 1 << 5
		// etc..
	};
	enum Type {
		TYPE_AREA = 0,
		TYPE_RIGID_BODY,
		TYPE_SOFT_BODY,
		TYPE_KINEMATIC_GHOST_BODY
	};

	struct ShapeWrapper {
		ShapeBullet *shape;
		btCollisionShape *bt_shape;
		btTransform transform;
		btVector3 scale;
		bool active;

		ShapeWrapper() :
				shape(nullptr),
				bt_shape(nullptr),
				active(true) {}

		ShapeWrapper(ShapeBullet *p_shape, const btTransform &p_transform, bool p_active) :
				shape(p_shape),
				bt_shape(nullptr),
				active(p_active) {
			set_transform(p_transform);
		}

		ShapeWrapper(ShapeBullet *p_shape, const Transform &p_transform, bool p_active) :
				shape(p_shape),
				bt_shape(nullptr),
				active(p_active) {
			set_transform(p_transform);
		}
		~ShapeWrapper();

		ShapeWrapper(const ShapeWrapper &otherShape) {
			operator=(otherShape);
		}

		void operator=(const ShapeWrapper &otherShape) {
			shape = otherShape.shape;
			bt_shape = otherShape.bt_shape;
			transform = otherShape.transform;
			scale = otherShape.scale;
			active = otherShape.active;
		}

		void set_transform(const Transform &p_transform);
		void set_transform(const btTransform &p_transform);
		btTransform get_adjusted_transform() const;

		void claim_bt_shape(const btVector3 &body_scale);
	};

protected:
	Type type;
	ObjectID instance_id;
	uint32_t collisionLayer;
	uint32_t collisionMask;
	bool collisionsEnabled;
	bool m_isStatic;
	bool ray_pickable;
	btCollisionObject *bt_collision_object;
	Vector3 body_scale;
	bool force_shape_reset;
	SpaceBullet *space;

	VSet<RID> exceptions;

	/// This array is used to know all areas where this Object is overlapped in
	/// New area is added when overlap with new area (AreaBullet::addOverlap), then is removed when it exit (CollisionObjectBullet::onExitArea)
	/// This array is used mainly to know which area hold the pointer of this object
	Vector<AreaBullet *> areasOverlapped;
	bool updated = false;

public:
	CollisionObjectBullet(Type p_type);
	virtual ~CollisionObjectBullet();

	Type getType() { return type; }

protected:
	void destroyBulletCollisionObject();
	void setupBulletCollisionObject(btCollisionObject *p_collisionObject);

public:
	_FORCE_INLINE_ btCollisionObject *get_bt_collision_object() { return bt_collision_object; }

	_FORCE_INLINE_ void set_instance_id(const ObjectID &p_instance_id) { instance_id = p_instance_id; }
	_FORCE_INLINE_ ObjectID get_instance_id() const { return instance_id; }

	_FORCE_INLINE_ bool is_static() const { return m_isStatic; }

	_FORCE_INLINE_ void set_ray_pickable(bool p_enable) { ray_pickable = p_enable; }
	_FORCE_INLINE_ bool is_ray_pickable() const { return ray_pickable; }

	void set_body_scale(const Vector3 &p_new_scale);
	const Vector3 &get_body_scale() const { return body_scale; }
	btVector3 get_bt_body_scale() const;
	virtual void body_scale_changed();

	void add_collision_exception(const CollisionObjectBullet *p_ignoreCollisionObject);
	void remove_collision_exception(const CollisionObjectBullet *p_ignoreCollisionObject);
	bool has_collision_exception(const CollisionObjectBullet *p_otherCollisionObject) const;
	_FORCE_INLINE_ const VSet<RID> &get_exceptions() const { return exceptions; }

	_FORCE_INLINE_ void set_collision_layer(uint32_t p_layer) {
		if (collisionLayer != p_layer) {
			collisionLayer = p_layer;
			on_collision_filters_change();
		}
	}
	_FORCE_INLINE_ uint32_t get_collision_layer() const { return collisionLayer; }

	_FORCE_INLINE_ void set_collision_mask(uint32_t p_mask) {
		if (collisionMask != p_mask) {
			collisionMask = p_mask;
			on_collision_filters_change();
		}
	}
	_FORCE_INLINE_ uint32_t get_collision_mask() const { return collisionMask; }

	virtual void on_collision_filters_change() = 0;

	_FORCE_INLINE_ bool test_collision_mask(CollisionObjectBullet *p_other) const {
		return collisionLayer & p_other->collisionMask || p_other->collisionLayer & collisionMask;
	}

	virtual void reload_body() = 0;
	virtual void set_space(SpaceBullet *p_space) = 0;
	_FORCE_INLINE_ SpaceBullet *get_space() const { return space; }

	virtual void on_collision_checker_start() = 0;
	virtual void on_collision_checker_end() = 0;

	virtual void dispatch_callbacks() = 0;

	void set_collision_enabled(bool p_enabled);
	bool is_collisions_response_enabled();

	void notify_new_overlap(AreaBullet *p_area);
	virtual void on_enter_area(AreaBullet *p_area) = 0;
	virtual void on_exit_area(AreaBullet *p_area);

	void set_godot_object_flags(int flags);
	int get_godot_object_flags() const;

	void set_transform(const Transform &p_global_transform);
	Transform get_transform() const;
	virtual void set_transform__bullet(const btTransform &p_global_transform);
	virtual const btTransform &get_transform__bullet() const;
	virtual void notify_transform_changed();

	bool is_updated() const { return updated; }
};

class RigidCollisionObjectBullet : public CollisionObjectBullet, public ShapeOwnerBullet {
protected:
	btCollisionShape *mainShape;
	Vector<ShapeWrapper> shapes;

public:
	RigidCollisionObjectBullet(Type p_type);
	~RigidCollisionObjectBullet();

	_FORCE_INLINE_ const Vector<ShapeWrapper> &get_shapes_wrappers() const { return shapes; }

	_FORCE_INLINE_ btCollisionShape *get_main_shape() const { return mainShape; }

	void add_shape(ShapeBullet *p_shape, const Transform &p_transform = Transform(), bool p_disabled = false);
	void set_shape(int p_index, ShapeBullet *p_shape);

	int get_shape_count() const;
	ShapeBullet *get_shape(int p_index) const;
	btCollisionShape *get_bt_shape(int p_index) const;

	int find_shape(ShapeBullet *p_shape) const;

	virtual void remove_shape_full(ShapeBullet *p_shape);
	void remove_shape_full(int p_index);
	void remove_all_shapes(bool p_permanentlyFromThisBody = false, bool p_force_not_reload = false);

	void set_shape_transform(int p_index, const Transform &p_transform);

	const btTransform &get_bt_shape_transform(int p_index) const;
	Transform get_shape_transform(int p_index) const;

	void set_shape_disabled(int p_index, bool p_disabled);
	bool is_shape_disabled(int p_index);

	virtual void shape_changed(int p_shape_index);
	virtual void reload_shapes();

	virtual void main_shape_changed() = 0;
	virtual void body_scale_changed();

private:
	void internal_shape_destroy(int p_index, bool p_permanentlyFromThisBody = false);
};

#endif // COLLISION_OBJECT_BULLET_H
