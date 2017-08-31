/*************************************************************************/
/*  navigation_polygon.h                                                 */
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
#ifndef NAVIGATION_POLYGON_H
#define NAVIGATION_POLYGON_H

#include "scene/2d/node_2d.h"
#include "scene/2d/editable_polygon_2d.h"

class NavigationPolygon : public Resource {

	GDCLASS(NavigationPolygon, Resource);

	PoolVector<Vector2> vertices;
	struct Polygon {
		Vector<int> indices;
	};
	Vector<Polygon> polygons;
	Vector<Ref<Outline2D> > outlines;

protected:
	static void _bind_methods();

	void _set_polygons(const Array &p_array);
	Array _get_polygons() const;

	void _set_outlines(const Array &p_array);
	Array _get_outlines() const;

public:
	void set_vertices(const PoolVector<Vector2> &p_vertices);
	PoolVector<Vector2> get_vertices() const;

	void add_polygon(const Vector<int> &p_polygon);
	int get_polygon_count() const;

	void add_outline(const Ref<Outline2D> &p_outline);
	void add_outline_at_index(const Ref<Outline2D> &p_outline, int p_index);
	void set_outline(int p_idx, const Ref<Outline2D> &p_outline);
	Ref<Outline2D> get_outline(int p_idx) const;
	void remove_outline(int p_idx);
	int get_outline_count() const;

	void clear_outlines();
	void make_polygons_from_outlines();

	Vector<int> get_polygon(int p_idx);
	void clear_polygons();

	NavigationPolygon();
};

class Navigation2D;

class NavigationPolygonInstance : public EditablePolygonNode2D {

	GDCLASS(NavigationPolygonInstance, EditablePolygonNode2D);

	bool enabled;
	int nav_id;
	Navigation2D *navigation;
	Ref<NavigationPolygon> navpoly;

	void _navpoly_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_navigation_polygon(const Ref<NavigationPolygon> &p_navpoly);
	Ref<NavigationPolygon> get_navigation_polygon() const;

	String get_configuration_warning() const;

	virtual bool _has_resource() const;
	virtual void _create_resource(UndoRedo *undo_redo);

	virtual int get_polygon_count() const;
	virtual Ref<AbstractPolygon2D> get_nth_polygon(int p_idx) const;

	virtual void append_polygon(const Vector<Point2> &p_vertices);
	virtual void add_polygon_at_index(int p_idx, Ref<AbstractPolygon2D> p_polygon);
	virtual void set_vertices(int p_idx, const Vector<Point2> &p_vertices);
	virtual void remove_polygon(int p_idx);

	NavigationPolygonInstance();
};

#endif // NAVIGATIONPOLYGON_H
