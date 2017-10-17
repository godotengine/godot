/*************************************************************************/
/*  collision_object_2d_sw.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef COLLISION_OBJECT_2D_SW_H
#define COLLISION_OBJECT_2D_SW_H

#include "broad_phase_2d_sw.h"
#include "self_list.h"
#include "servers/physics_2d_server.h"
#include "shape_2d_sw.h"

class Space2DSW;

class CollisionObject2DSW : public ShapeOwner2DSW {
public:
	enum Type {
		TYPE_AREA,
		TYPE_BODY
	};

private:
	Type type;
	RID self;
	ObjectID instance_id;
	bool pickable;

	struct Shape {

		Transform2D xform;
		Transform2D xform_inv;
		BroadPhase2DSW::ID bpid;
		Rect2 aabb_cache; //for rayqueries
		Shape2DSW *shape;
		Variant metadata;
		bool disabled;
		bool one_way_collision;
		Shape() {
			disabled = false;
			one_way_collision = false;
		}
	};

	Vector<Shape> shapes;
	Space2DSW *space;
	Transform2D transform;
	Transform2D inv_transform;
	uint32_t collision_mask;
	uint32_t collision_layer;
	bool _static;

	void _update_shapes();

protected:
	void _update_shapes_with_motion(const Vector2 &p_motion);
	void _unregister_shapes();

	_FORCE_INLINE_ void _set_transform(const Transform2D &p_transform, bool p_update_shapes = true) {
		transform = p_transform;
		if (p_update_shapes) {
			_update_shapes();
		}
	}
	_FORCE_INLINE_ void _set_inv_transform(const Transform2D &p_transform) { inv_transform = p_transform; }
	void _set_static(bool p_static);

	virtual void _shapes_changed() = 0;
	void _set_space(Space2DSW *p_space);

	CollisionObject2DSW(Type p_type);

public:
	_FORCE_INLINE_ void set_self(const RID &p_self) { self = p_self; }
	_FORCE_INLINE_ RID get_self() const { return self; }

	_FORCE_INLINE_ void set_instance_id(const ObjectID &p_instance_id) { instance_id = p_instance_id; }
	_FORCE_INLINE_ ObjectID get_instance_id() const { return instance_id; }

	void _shape_changed();

	_FORCE_INLINE_ Type get_type() const { return type; }
	void add_shape(Shape2DSW *p_shape, const Transform2D &p_transform = Transform2D());
	void set_shape(int p_index, Shape2DSW *p_shape);
	void set_shape_transform(int p_index, const Transform2D &p_transform);
	void set_shape_metadata(int p_index, const Variant &p_metadata);

	_FORCE_INLINE_ int get_shape_count() const { return shapes.size(); }
	_FORCE_INLINE_ Shape2DSW *get_shape(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, shapes.size(), NULL);
		return shapes[p_index].shape;
	}
	_FORCE_INLINE_ const Transform2D &get_shape_transform(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, shapes.size(), Transform2D());
		return shapes[p_index].xform;
	}
	_FORCE_INLINE_ const Transform2D &get_shape_inv_transform(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, shapes.size(), Transform2D());
		return shapes[p_index].xform_inv;
	}
	_FORCE_INLINE_ const Rect2 &get_shape_aabb(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, shapes.size(), Rect2());
		return shapes[p_index].aabb_cache;
	}
	_FORCE_INLINE_ const Variant &get_shape_metadata(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, shapes.size(), Variant());
		return shapes[p_index].metadata;
	}

	_FORCE_INLINE_ Transform2D get_transform() const { return transform; }
	_FORCE_INLINE_ Transform2D get_inv_transform() const { return inv_transform; }
	_FORCE_INLINE_ Space2DSW *get_space() const { return space; }

	_FORCE_INLINE_ void set_shape_as_disabled(int p_idx, bool p_disabled) {
		ERR_FAIL_INDEX(p_idx, shapes.size());
		shapes[p_idx].disabled = p_disabled;
	}
	_FORCE_INLINE_ bool is_shape_set_as_disabled(int p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, shapes.size(), false);
		return shapes[p_idx].disabled;
	}

	_FORCE_INLINE_ void set_shape_as_one_way_collision(int p_idx, bool p_one_way_collision) {
		ERR_FAIL_INDEX(p_idx, shapes.size());
		shapes[p_idx].one_way_collision = p_one_way_collision;
	}
	_FORCE_INLINE_ bool is_shape_set_as_one_way_collision(int p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, shapes.size(), false);
		return shapes[p_idx].one_way_collision;
	}

	void set_collision_mask(uint32_t p_mask) { collision_mask = p_mask; }
	_FORCE_INLINE_ uint32_t get_collision_mask() const { return collision_mask; }

	void set_collision_layer(uint32_t p_layer) { collision_layer = p_layer; }
	_FORCE_INLINE_ uint32_t get_collision_layer() const { return collision_layer; }

	void remove_shape(Shape2DSW *p_shape);
	void remove_shape(int p_index);

	virtual void set_space(Space2DSW *p_space) = 0;

	_FORCE_INLINE_ bool is_static() const { return _static; }

	void set_pickable(bool p_pickable) { pickable = p_pickable; }
	_FORCE_INLINE_ bool is_pickable() const { return pickable; }

	_FORCE_INLINE_ bool test_collision_mask(CollisionObject2DSW *p_other) const {

		return collision_layer & p_other->collision_mask || p_other->collision_layer & collision_mask;
	}

	virtual ~CollisionObject2DSW() {}
};

#endif // COLLISION_OBJECT_2D_SW_H
