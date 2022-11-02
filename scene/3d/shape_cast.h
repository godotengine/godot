/*************************************************************************/
/*  shape_cast.h                                                         */
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

#ifndef SHAPE_CAST_H
#define SHAPE_CAST_H

#include "scene/3d/spatial.h"
#include "scene/resources/shape.h"

class ShapeCast : public Spatial {
	GDCLASS(ShapeCast, Spatial);

	bool enabled = true;
	void resource_changed(Ref<Resource> p_res);

	Ref<Shape> shape;
	RID shape_rid;
	Vector3 target_position = Vector3(0, -1, 0);

	Set<RID> exclude;
	real_t margin = 0.0;
	uint32_t collision_mask = 1;
	bool exclude_parent_body = true;
	bool collide_with_areas = false;
	bool collide_with_bodies = true;

	Node *debug_shape = nullptr;
	Ref<Material> debug_material;
	Color debug_shape_custom_color = Color(0.0, 0.0, 0.0);
	Vector<Vector3> debug_shape_vertices;
	Vector<Vector3> debug_line_vertices;

	void _create_debug_shape();
	void _update_debug_shape();
	void _update_debug_shape_material(bool p_check_collision = false);
	void _update_debug_shape_vertices();
	void _clear_debug_shape();

	// Result
	int max_results = 32;
	Vector<PhysicsDirectSpaceState::ShapeRestInfo> result;
	bool collided = false;
	real_t collision_safe_fraction = 1.0;
	real_t collision_unsafe_fraction = 1.0;

	Array _get_collision_result() const;

	~ShapeCast();

protected:
	void _notification(int p_what);
	void _update_shapecast_state();
	static void _bind_methods();

public:
	void set_collide_with_areas(bool p_clip);
	bool is_collide_with_areas_enabled() const;

	void set_collide_with_bodies(bool p_clip);
	bool is_collide_with_bodies_enabled() const;

	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_shape(const Ref<Shape> &p_shape);
	Ref<Shape> get_shape() const;

	void set_target_position(const Vector3 &p_point);
	Vector3 get_target_position() const;

	void set_margin(real_t p_margin);
	real_t get_margin() const;

	void set_max_results(int p_max_results);
	int get_max_results() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_mask_value(int p_layer_number, bool p_value);
	bool get_collision_mask_value(int p_layer_number) const;

	void set_exclude_parent_body(bool p_exclude_parent_body);
	bool get_exclude_parent_body() const;

	const Color &get_debug_shape_custom_color() const;
	void set_debug_shape_custom_color(const Color &p_color);

	const Vector<Vector3> &get_debug_shape_vertices() const;
	const Vector<Vector3> &get_debug_line_vertices() const;

	Ref<SpatialMaterial> get_debug_material();

	int get_collision_count() const;
	Object *get_collider(int p_idx) const;
	RID get_collider_rid(int p_idx) const;
	int get_collider_shape(int p_idx) const;
	Vector3 get_collision_point(int p_idx) const;
	Vector3 get_collision_normal(int p_idx) const;

	real_t get_closest_collision_safe_fraction() const;
	real_t get_closest_collision_unsafe_fraction() const;

	void force_shapecast_update();
	bool is_colliding() const;

	void add_exception_rid(const RID &p_rid);
	void add_exception(const Object *p_object);
	void remove_exception_rid(const RID &p_rid);
	void remove_exception(const Object *p_object);
	void clear_exceptions();

	virtual String get_configuration_warning() const;
};

#endif // SHAPE_CAST_H
