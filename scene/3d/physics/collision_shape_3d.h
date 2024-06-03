/**************************************************************************/
/*  collision_shape_3d.h                                                  */
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

#ifndef COLLISION_SHAPE_3D_H
#define COLLISION_SHAPE_3D_H

#include "scene/3d/node_3d.h"
#include "scene/resources/3d/shape_3d.h"

class CollisionObject3D;
class CollisionShape3D : public Node3D {
	GDCLASS(CollisionShape3D, Node3D);

	Ref<Shape3D> shape;

	uint32_t owner_id = 0;
	CollisionObject3D *collision_object = nullptr;

#ifndef DISABLE_DEPRECATED
	void resource_changed(Ref<Resource> res);
#endif
	bool disabled = false;

protected:
	void _update_in_shape_owner(bool p_xform_only = false);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void make_convex_from_siblings();

	void set_shape(const Ref<Shape3D> &p_shape);
	Ref<Shape3D> get_shape() const;

	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	PackedStringArray get_configuration_warnings() const override;

	CollisionShape3D();
	~CollisionShape3D();
};
class CollisionObject3DConnectionShape : public RefCounted {
	GDCLASS(CollisionObject3DConnectionShape, RefCounted);
	static void _bind_methods();
public:
	void set_shape(const Ref<Shape3D> &p_shape) {
		shape = p_shape;
		update_transform();
	}

	Ref<Shape3D> get_shape() const {
		return shape;
	}
	void set_position(const Vector3 &p_position) {
		local_origin = p_position;
		update_transform();
	}

	Vector3 get_position() const {
		return local_origin;
	}

	void set_rotation(const Vector3 &p_rotation) {
		local_rotation = p_rotation;
		update_transform();
	}

	Vector3 get_rotation() const {
		return local_rotation;
	}

	void set_scale(const Vector3 &p_scale) {
		local_scale = p_scale;
		update_transform();
	}

	Vector3 get_scale() const {
		return local_scale;
	}
	CollisionObject3DConnectionShape() {}
	~CollisionObject3DConnectionShape() {set_link_target(nullptr);}
protected:
friend class CollisionObject3DConnection;
	void set_link_target(Node3D *p_target);
	void update_transform();
protected:
	Node3D* link_target = nullptr;
	CollisionShape3D *shape_node = nullptr;
	Ref<Shape3D> shape;
	Vector3 local_origin = Vector3(0, 0, 0);
	Vector3 local_rotation = Vector3(0, 0, 0);
	Vector3 local_scale = Vector3(1, 1, 1);

};


class CollisionObject3DConnection : public RefCounted {
	GDCLASS(CollisionObject3DConnection, RefCounted);


public:
	void set_shapes(const TypedArray<CollisionObject3DConnectionShape> &p_shapes) {
		shapes = p_shapes;
		update_link_target();
	}
	TypedArray<CollisionObject3DConnectionShape> get_shapes() const {
		return shapes;
	}

	void set_link_target(Node3D *p_target) ;
	void update_link_target();
	~CollisionObject3DConnection()
	{
		link_target = nullptr;
		update_link_target();
	}
	void on_taeget_free() {
		link_target = nullptr;
	}
protected:


	
	TypedArray<CollisionObject3DConnectionShape> shapes;
	Node3D* link_target = nullptr;
};

#endif // COLLISION_SHAPE_3D_H
