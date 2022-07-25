/*************************************************************************/
/*  collision_object.h                                                   */
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

#ifndef COLLISION_OBJECT_H
#define COLLISION_OBJECT_H

#include "scene/3d/spatial.h"
#include "scene/resources/shape.h"

class CollisionObject : public Spatial {
	GDCLASS(CollisionObject, Spatial);

	uint32_t collision_layer = 1;
	uint32_t collision_mask = 1;

	bool area;

	RID rid;

	struct ShapeData {
		ObjectID owner_id;
		Transform xform;
		struct ShapeBase {
			RID debug_shape;
			Ref<Shape> shape;
			int index;
		};

		Vector<ShapeBase> shapes;
		bool disabled;

		ShapeData() {
			disabled = false;
			owner_id = 0;
		}
	};

	int total_subshapes;

	Map<uint32_t, ShapeData> shapes;
	bool only_update_transform_changes = false; //this is used for sync physics in KinematicBody

	bool capture_input_on_drag;
	bool ray_pickable;

	Set<uint32_t> debug_shapes_to_update;
	int debug_shapes_count = 0;
	Transform debug_shape_old_transform;

	void _update_pickable();

	bool _are_collision_shapes_visible();
	void _update_shape_data(uint32_t p_owner);
	void _shape_changed(const Ref<Shape> &p_shape);
	void _update_debug_shapes();
	void _clear_debug_shapes();

protected:
	CollisionObject(RID p_rid, bool p_area);

	void _notification(int p_what);
	static void _bind_methods();
	friend class Viewport;
	virtual void _input_event(Node *p_camera, const Ref<InputEvent> &p_input_event, const Vector3 &p_pos, const Vector3 &p_normal, int p_shape);
	virtual void _mouse_enter();
	virtual void _mouse_exit();

	void set_only_update_transform_changes(bool p_enable);

	void _on_transform_changed();

public:
	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_layer_bit(int p_bit, bool p_value);
	bool get_collision_layer_bit(int p_bit) const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	uint32_t create_shape_owner(Object *p_owner);
	void remove_shape_owner(uint32_t owner);
	void get_shape_owners(List<uint32_t> *r_owners);
	Array _get_shape_owners();

	void shape_owner_set_transform(uint32_t p_owner, const Transform &p_transform);
	Transform shape_owner_get_transform(uint32_t p_owner) const;
	Object *shape_owner_get_owner(uint32_t p_owner) const;

	void shape_owner_set_disabled(uint32_t p_owner, bool p_disabled);
	bool is_shape_owner_disabled(uint32_t p_owner) const;

	void shape_owner_add_shape(uint32_t p_owner, const Ref<Shape> &p_shape);
	int shape_owner_get_shape_count(uint32_t p_owner) const;
	Ref<Shape> shape_owner_get_shape(uint32_t p_owner, int p_shape) const;
	int shape_owner_get_shape_index(uint32_t p_owner, int p_shape) const;

	void shape_owner_remove_shape(uint32_t p_owner, int p_shape);
	void shape_owner_clear_shapes(uint32_t p_owner);

	uint32_t shape_find_owner(int p_shape_index) const;

	void set_ray_pickable(bool p_ray_pickable);
	bool is_ray_pickable() const;

	void set_capture_input_on_drag(bool p_capture);
	bool get_capture_input_on_drag() const;

	_FORCE_INLINE_ RID get_rid() const { return rid; }

	virtual String get_configuration_warning() const;

	CollisionObject();
	~CollisionObject();
};

#endif // COLLISION_OBJECT_H
