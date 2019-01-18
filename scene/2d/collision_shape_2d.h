/*************************************************************************/
/*  collision_shape_2d.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef COLLISION_SHAPE_2D_H
#define COLLISION_SHAPE_2D_H

#include "scene/2d/node_2d.h"
#include "scene/resources/shape_2d.h"

class CollisionObject2D;

class CollisionShape2D : public Node2D {

	GDCLASS(CollisionShape2D, Node2D)
	Ref<Shape2D> shape;
	Rect2 rect;
	uint32_t owner_id;
	CollisionObject2D *parent;
	void _shape_changed();
	bool disabled;
	bool one_way_collision;
	float one_way_collision_margin;

	void _update_in_shape_owner(bool p_xform_only = false);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const;

	void set_shape(const Ref<Shape2D> &p_shape);
	Ref<Shape2D> get_shape() const;

	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	void set_one_way_collision(bool p_enable);
	bool is_one_way_collision_enabled() const;

	void set_one_way_collision_margin(float p_margin);
	float get_one_way_collision_margin() const;

	virtual String get_configuration_warning() const;

	CollisionShape2D();
};

#endif // COLLISION_SHAPE_2D_H
