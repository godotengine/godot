/*************************************************************************/
/*  polygon_2d.h                                                         */
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
#ifndef POLYGON_2D_H
#define POLYGON_2D_H

#include "scene/2d/node_2d.h"
#include "scene/2d/editable_polygon_2d.h"

class Polygon2D : public AbstractPolygon2D {

	GDCLASS(Polygon2D, AbstractPolygon2D);

	PoolVector<Vector2> uv;
	PoolVector<Color> vertex_colors;
	Color color;
	Ref<Texture> texture;
	Size2 tex_scale;
	Vector2 tex_ofs;
	bool tex_tile;
	float tex_rot;
	bool invert;
	float invert_border;
	bool antialiased;

	Vector2 offset;

	void _set_texture_rotationd(float p_rot);
	float _get_texture_rotationd() const;

protected:
	static void _bind_methods();

public:
	void set_uv(const PoolVector<Vector2> &p_uv);
	PoolVector<Vector2> get_uv() const;

	void set_color(const Color &p_color);
	Color get_color() const;

	void set_vertex_colors(const PoolVector<Color> &p_colors);
	PoolVector<Color> get_vertex_colors() const;

	void set_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_texture() const;

	void set_texture_offset(const Vector2 &p_offset);
	Vector2 get_texture_offset() const;

	void set_texture_rotation(float p_rot);
	float get_texture_rotation() const;

	void set_texture_scale(const Size2 &p_scale);
	Size2 get_texture_scale() const;

	void set_invert(bool p_invert);
	bool get_invert() const;

	void set_antialiased(bool p_antialiased);
	bool get_antialiased() const;

	void set_invert_border(float p_invert_border);
	float get_invert_border() const;

	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;

	void draw(RID p_canvas_item);

	//editor stuff

	virtual void edit_set_pivot(const Point2 &p_pivot);
	virtual Point2 edit_get_pivot() const;
	virtual bool edit_has_pivot() const;

	virtual PoolVector<Vector2> edit_get_uv() const;
	virtual void edit_set_uv(const PoolVector<Vector2> &p_uv);
	virtual Ref<Texture> edit_get_texture() const;

	Polygon2D();
};

class Polygon2DInstance : public EditablePolygonNode2D {

	GDCLASS(Polygon2DInstance, EditablePolygonNode2D);

	Ref<Polygon2D> polygon;

	void _polygon_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_polygon(const Ref<Polygon2D> &p_polygon);
	Ref<Polygon2D> get_polygon() const;

	virtual bool _has_resource() const;
	virtual void _create_resource(UndoRedo *undo_redo);

	virtual int get_polygon_count() const;
	virtual Ref<AbstractPolygon2D> get_nth_polygon(int p_idx) const;

	virtual void append_polygon(const Vector<Point2> &p_vertices);
	virtual void add_polygon_at_index(int p_idx, Ref<AbstractPolygon2D> p_polygon);
	virtual void set_vertices(int p_idx, const Vector<Point2> &p_vertices);
	virtual void remove_polygon(int p_idx);

	virtual String get_configuration_warning() const;
	virtual Rect2 get_item_rect() const;

	Polygon2DInstance();
};

#endif // POLYGON_2D_H
