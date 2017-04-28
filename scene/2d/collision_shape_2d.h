/*************************************************************************/
/*  collision_shape_2d.h                                                 */
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
#ifndef COLLISION_SHAPE_2D_H
#define COLLISION_SHAPE_2D_H

#include "scene/2d/node_2d.h"
#include "scene/resources/shape_2d.h"

class CollisionShape2D : public Node2D {

	GDCLASS(CollisionShape2D, Node2D);
	Ref<Shape2D> shape;
	Rect2 rect;
	bool trigger;
	bool unparenting;
	bool can_update_body;
	void _shape_changed();
	int update_shape_index;

	void _set_update_shape_index(int p_index);
	int _get_update_shape_index() const;

protected:
	void _update_parent();
	void _notification(int p_what);
	static void _bind_methods();

	void _add_to_collision_object(Object *p_obj);

public:
	void set_shape(const Ref<Shape2D> &p_shape);
	Ref<Shape2D> get_shape() const;
	virtual Rect2 get_item_rect() const;
	void set_trigger(bool p_trigger);
	bool is_trigger() const;

	int get_collision_object_shape_index() const { return _get_update_shape_index(); }

	virtual String get_configuration_warning() const;

	CollisionShape2D();
};

#endif // COLLISION_SHAPE_2D_H
