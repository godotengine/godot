/*************************************************************************/
/*  collision_polygon_2d.h                                               */
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
#ifndef COLLISION_POLYGON_2D_H
#define COLLISION_POLYGON_2D_H

#include "scene/2d/node_2d.h"
#include "scene/resources/shape_2d.h"

class CollisionPolygon2D : public Node2D {

	GDCLASS(CollisionPolygon2D, Node2D);

public:
	enum BuildMode {
		BUILD_SOLIDS,
		BUILD_SEGMENTS,
	};

protected:
	Rect2 aabb;
	BuildMode build_mode;
	Vector<Point2> polygon;
	bool trigger;
	bool unparenting;

	void _add_to_collision_object(Object *p_obj);
	void _update_parent();

	bool can_update_body;
	int shape_from;
	int shape_to;

	void _set_shape_range(const Vector2 &p_range);
	Vector2 _get_shape_range() const;

	Vector<Vector<Vector2> > _decompose_in_convex();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_trigger(bool p_trigger);
	bool is_trigger() const;

	void set_build_mode(BuildMode p_mode);
	BuildMode get_build_mode() const;

	void set_polygon(const Vector<Point2> &p_polygon);
	Vector<Point2> get_polygon() const;

	virtual Rect2 get_item_rect() const;

	int get_collision_object_first_shape() const { return shape_from; }
	int get_collision_object_last_shape() const { return shape_to; }

	virtual String get_configuration_warning() const;

	CollisionPolygon2D();
};

VARIANT_ENUM_CAST(CollisionPolygon2D::BuildMode);

#endif // COLLISION_POLYGON_2D_H
