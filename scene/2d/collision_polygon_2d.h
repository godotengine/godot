/*************************************************************************/
/*  collision_polygon_2d.h                                               */
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
#ifndef COLLISION_POLYGON_2D_H
#define COLLISION_POLYGON_2D_H

#include "scene/2d/node_2d.h"
#include "scene/resources/shape_2d.h"
#include "scene/2d/editable_polygon_2d.h"

class CollisionObject2D;

class CollisionPolygon2D : public Node2D, public EditablePolygon2D {

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
	uint32_t owner_id;
	CollisionObject2D *parent;
	bool disabled;
	bool one_way_collision;

	Vector<Vector<Vector2> > _decompose_in_convex();

	void _build_polygon();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_build_mode(BuildMode p_mode);
	BuildMode get_build_mode() const;

	void set_polygon(const Vector<Point2> &p_polygon);
	Vector<Point2> get_polygon() const;

	virtual Rect2 get_item_rect() const;

	virtual String get_configuration_warning() const;

	void set_disabled(bool p_disabled);
	bool is_disabled() const;

	void set_one_way_collision(bool p_enable);
	bool is_one_way_collision_enabled() const;

	virtual int edit_get_polygon_count() const;
	virtual Vector<Vector2> edit_get_polygon(int p_polygon) const;
	virtual void edit_set_polygon(int p_polygon, const Vector<Vector2> &p_points);
	virtual bool edit_is_wip_destructive() const;
	virtual void edit_create_wip_close_action(UndoRedo *undo_redo, const Vector<Vector2> &p_wip);
	virtual void edit_create_edit_poly_action(UndoRedo *undo_redo, int p_polygon, const Vector<Vector2> &p_before, const Vector<Vector2> &p_after);
	virtual void edit_create_remove_point_action(UndoRedo *undo_redo, int p_polygon, int p_point);
	virtual Color edit_get_previous_outline_color() const;

	CollisionPolygon2D();
};

VARIANT_ENUM_CAST(CollisionPolygon2D::BuildMode);

#endif // COLLISION_POLYGON_2D_H
