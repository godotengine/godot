/*************************************************************************/
/*  collision_object_bullet.h                                            */
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

#ifndef COLLISION_OBJECT_BULLET_H
#define COLLISION_OBJECT_BULLET_H

#include "core/vset.h"
#include "shape_owner_bullet.h"
#include "space_bullet.h"

class AreaBullet;
class ShapeBullet;
class btCollisionObject;
class btCompoundShape;
class btCollisionShape;

class CollisionObjectBullet : public ShapeOwnerBullet {
protected:
	class HackedBasis : public Basis {
	public:
		void get_ABS_scale(Vector3 &out_vec) const {
			out_vec[0] = Vector3(elements[0][0], elements[1][0], elements[2][0]).length();
			out_vec[1] = Vector3(elements[0][1], elements[1][1], elements[2][1]).length();
			out_vec[2] = Vector3(elements[0][2], elements[1][2], elements[2][2]).length();
		}
	};

public:
	enum Type {
		TYPE_AREA = 0,
		TYPE_BODY
	};

	struct ShapeWrapper {
		ShapeBullet *shape;
		btCollisionShape *bt_shape;
		Transform transform;
		bool active;

		btTransform cache_transform;
		btVector3 cache_shape_scale;

		ShapeWrapper()
			: shape(NULL), bt_shape(NULL), active(true) {}

		ShapeWrapper(ShapeBullet *p_shape, const btTransform &p_transform, bool p_active)
			: shape(p_shape), bt_shape(NULL), active(p_active) {
			set_transform(p_transform);
		}

		ShapeWrapper(ShapeBullet *p_shape, const Transform &p_transform, bool p_active)
			: shape(p_shape), bt_shape(NULL), active(p_active) {
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
			active = otherShape.active;
		}

		void set_transform(const Transform &p_transform);
		void set_transform(const btTransform &p_transform);
	};

protected:
	Type type;
	ObjectID instance_id;
	uint32_t collisionLayer;
	uint32_t collisionMask;
	bool collisionsEnabled;
	bool m_isStatic;
	bool ray_pickable;
	btCollisionObject *collisionObject;
	Vector3 global_scale;

	/// This is required to combine some shapes together.
	/// Since Godot allow to have multiple shapes for each body with custom relative location,
	/// each body will attach the shapes using this class even if there is only one shape.
	btCompoundShape *compoundShape;
	Vector<ShapeWrapper> shapes;
	VSet<RID> exceptions;

	/// This array is used to know all areas where this Object is overlapped in
	/// New area is added when overlap with new area (AreaBullet::addOverlap), then is removed when it exit (CollisionObjectBullet::onExitArea)
	/// This array is used mainly to know which area hold the pointer of this object
	Vector<AreaBullet *> areasOverlapped;

public:
	CollisionObjectBullet(Type p_type);
	virtual ~CollisionObjectBullet();

	Type getType() { return type; }

protected:
	void setupCollisionObject(btCollisionObject *p_collisionObject);

public:
	_FORCE_INLINE_ const Vector<ShapeWrapper> &get_shapes_wrappers() const { return shapes; }
	_FORCE_INLINE_ btCollisionObject *get_bt_collision_object() { return collisionObject; }

	_FORCE_INLINE_ void set_instance_id(const ObjectID &p_instance_id) { instance_id = p_instance_id; }
	_FORCE_INLINE_ ObjectID get_instance_id() const { return instance_id; }

	_FORCE_INLINE_ bool is_static() const { return m_isStatic; }

	_FORCE_INLINE_ void set_ray_pickable(bool p_enable) { ray_pickable = p_enable; }
	_FORCE_INLINE_ bool is_ray_pickable() const { return ray_pickable; }

	void set_global_ABS_scale(const Vector3 &p_new_scale);

	void add_collision_exception(const CollisionObjectBullet *p_ignoreCollisionObject);
	void remove_collision_exception(const CollisionObjectBullet *p_ignoreCollisionObject);
	bool has_collision_exception(const CollisionObjectBullet *p_otherCollisionObject) const;
	_FORCE_INLINE_ const VSet<RID> &get_exceptions() const { return exceptions; }

	/// This is used to set new shape or replace existing
	//virtual void _internal_replaceShape(btCollisionShape *p_old_shape, btCollisionShape *p_new_shape) = 0;
	void add_shape(ShapeBullet *p_shape, const Transform &p_transform = Transform());
	void set_shape(int p_index, ShapeBullet *p_shape);
	void set_shape_transform(int p_index, const Transform &p_transform);
	virtual void remove_shape(ShapeBullet *p_shape);
	void remove_shape(int p_index);
	void remove_all_shapes(bool p_permanentlyFromThisBody = false);

	virtual void on_shape_changed(const ShapeBullet *const p_shape);
	virtual void on_shapes_changed();

	_FORCE_INLINE_ btCompoundShape *get_compound_shape() const { return compoundShape; }
	int get_shape_count() const;
	ShapeBullet *get_shape(int p_index) const;
	btCollisionShape *get_bt_shape(int p_index) const;
	Transform get_shape_transform(int p_index) const;

	_FORCE_INLINE_ void set_collision_layer(uint32_t p_layer) {
		collisionLayer = p_layer;
		on_collision_filters_change();
	}
	_FORCE_INLINE_ uint32_t get_collision_layer() const { return collisionLayer; }

	_FORCE_INLINE_ void set_collision_mask(uint32_t p_mask) {
		collisionMask = p_mask;
		on_collision_filters_change();
	}
	_FORCE_INLINE_ uint32_t get_collision_mask() const { return collisionMask; }

	virtual void on_collision_filters_change() = 0;

	_FORCE_INLINE_ bool test_collision_mask(CollisionObjectBullet *p_other) const {
		return collisionLayer & p_other->collisionMask || p_other->collisionLayer & collisionMask;
	}

	virtual void reload_body() = 0;
	virtual void set_space(SpaceBullet *p_space) = 0;
	virtual SpaceBullet *get_space() const = 0;
	/// This is an event that is called when a collision checker starts
	virtual void on_collision_checker_start() = 0;

	virtual void dispatch_callbacks() = 0;

	void set_shape_disabled(int p_index, bool p_disabled);
	bool is_shape_disabled(int p_index);
	void set_collision_enabled(bool p_enabled);
	bool is_collisions_response_enabled();

	void notify_new_overlap(AreaBullet *p_area);
	virtual void on_enter_area(AreaBullet *p_area) = 0;
	virtual void on_exit_area(AreaBullet *p_area);

private:
	void internal_destroy(int p_index, bool p_permanentlyFromThisBody = false);
};

#endif
