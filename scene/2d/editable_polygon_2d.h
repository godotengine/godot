/*************************************************************************/
/*  editable_polygon_2d.h                                                */
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
#ifndef EDITABLEPOLYGON2D_H
#define EDITABLEPOLYGON2D_H

#include "core/undo_redo.h"
#include "core/resource.h"
#include "scene/resources/texture.h"
#include "scene/2d/node_2d.h"

class AbstractPolygon2D : public Resource {

	GDCLASS(AbstractPolygon2D, Resource);

protected:
	PoolVector<Point2> vertices;

	mutable bool rect_cache_dirty;
	mutable Rect2 item_rect;

protected:
	static void _bind_methods();

public:
	Vector<Point2> get_vertices() const;
	void set_vertices(const Vector<Point2> &p_vertices);

	bool is_empty() const;
	Rect2 get_item_rect() const;

	virtual Vector2 get_offset() const;
	virtual Color get_outline_color() const;

	AbstractPolygon2D();
};

class Outline2D : public AbstractPolygon2D {

	GDCLASS(Outline2D, AbstractPolygon2D);
};

class EditablePolygonNode2D : public Node2D {

	GDCLASS(EditablePolygonNode2D, Node2D);

protected:
	static void _bind_methods();

public:
	virtual bool _has_resource() const = 0;
	virtual void _create_resource(UndoRedo *undo_redo) = 0;

	virtual int get_polygon_count() const = 0;
	virtual Ref<AbstractPolygon2D> get_nth_polygon(int p_idx) const = 0;

	virtual void append_polygon(const Vector<Point2> &p_vertices) = 0;
	virtual void add_polygon_at_index(int p_idx, Ref<AbstractPolygon2D> p_polygon) = 0;
	virtual void set_vertices(int p_idx, const Vector<Point2> &p_vertices) = 0;
	virtual void remove_polygon(int p_idx) = 0;
};

#endif // EDITABLEPOLYGON2D_H
