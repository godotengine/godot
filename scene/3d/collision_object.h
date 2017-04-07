/*************************************************************************/
/*  collision_object.h                                                   */
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
#ifndef COLLISION_OBJECT_H
#define COLLISION_OBJECT_H

#include "scene/3d/spatial.h"
#include "scene/resources/shape.h"

class CollisionObject : public Spatial {

	GDCLASS(CollisionObject, Spatial);

	bool area;
	RID rid;

	struct ShapeData {
		Transform xform;
		Ref<Shape> shape;
		bool trigger;

		ShapeData() {
			trigger = false;
		}
	};

	bool capture_input_on_drag;
	bool ray_pickable;
	Vector<ShapeData> shapes;

	void _update_pickable();
	void _update_shapes();

	friend class CollisionShape;
	friend class CollisionPolygon;
	void _update_shapes_from_children();

protected:
	CollisionObject(RID p_rid, bool p_area);

	void _notification(int p_what);
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();
	friend class Viewport;
	virtual void _input_event(Node *p_camera, const InputEvent &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape);
	virtual void _mouse_enter();
	virtual void _mouse_exit();

public:
	void add_shape(const Ref<Shape> &p_shape, const Transform &p_transform = Transform());
	int get_shape_count() const;
	void set_shape(int p_shape_idx, const Ref<Shape> &p_shape);
	void set_shape_transform(int p_shape_idx, const Transform &p_transform);
	Ref<Shape> get_shape(int p_shape_idx) const;
	Transform get_shape_transform(int p_shape_idx) const;
	void remove_shape(int p_shape_idx);
	void clear_shapes();
	void set_shape_as_trigger(int p_shape_idx, bool p_trigger);
	bool is_shape_set_as_trigger(int p_shape_idx) const;

	void set_ray_pickable(bool p_ray_pickable);
	bool is_ray_pickable() const;

	void set_capture_input_on_drag(bool p_capture);
	bool get_capture_input_on_drag() const;

	_FORCE_INLINE_ RID get_rid() const { return rid; }

	CollisionObject();
	~CollisionObject();
};

#endif // COLLISION_OBJECT__H
