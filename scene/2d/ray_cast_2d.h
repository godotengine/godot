/*************************************************************************/
/*  ray_cast_2d.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RAY_CAST_2D_H
#define RAY_CAST_2D_H

#include "scene/2d/node_2d.h"

class RayCast2D : public Node2D {
	GDCLASS(RayCast2D, Node2D);

	bool enabled;
	bool collided;
	ObjectID against;
	int against_shape;
	Vector2 collision_point;
	Vector2 collision_normal;
	Set<RID> exclude;
	uint32_t collision_mask;
	bool exclude_parent_body;

	Vector2 target_position;

	bool collide_with_areas;
	bool collide_with_bodies;

protected:
	void _notification(int p_what);
	void _update_raycast_state();
	static void _bind_methods();

public:
	void set_collide_with_areas(bool p_clip);
	bool is_collide_with_areas_enabled() const;

	void set_collide_with_bodies(bool p_clip);
	bool is_collide_with_bodies_enabled() const;

	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_target_position(const Vector2 &p_point);
	Vector2 get_target_position() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	void set_exclude_parent_body(bool p_exclude_parent_body);
	bool get_exclude_parent_body() const;

	void force_raycast_update();

	bool is_colliding() const;
	Object *get_collider() const;
	int get_collider_shape() const;
	Vector2 get_collision_point() const;
	Vector2 get_collision_normal() const;

	void add_exception_rid(const RID &p_rid);
	void add_exception(const Object *p_object);
	void remove_exception_rid(const RID &p_rid);
	void remove_exception(const Object *p_object);
	void clear_exceptions();

	RayCast2D();
};

#endif // RAY_CAST_2D_H
