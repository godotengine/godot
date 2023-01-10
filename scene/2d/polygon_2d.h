/**************************************************************************/
/*  polygon_2d.h                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef POLYGON_2D_H
#define POLYGON_2D_H

#include "scene/2d/node_2d.h"

class Polygon2D : public Node2D {
	GDCLASS(Polygon2D, Node2D);

	PoolVector<Vector2> polygon;
	PoolVector<Vector2> uv;
	PoolVector<Color> vertex_colors;
	Array polygons;
	int internal_vertices;

	struct Bone {
		NodePath path;
		PoolVector<float> weights;
	};

	Vector<Bone> bone_weights;

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
	mutable bool rect_cache_dirty;
	mutable Rect2 item_rect;

	NodePath skeleton;
	ObjectID current_skeleton_id;

	Array _get_bones() const;
	void _set_bones(const Array &p_bones);

	void _skeleton_bone_setup_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
#ifdef TOOLS_ENABLED
	virtual Dictionary _edit_get_state() const;
	virtual void _edit_set_state(const Dictionary &p_state);

	virtual void _edit_set_pivot(const Point2 &p_pivot);
	virtual Point2 _edit_get_pivot() const;
	virtual bool _edit_use_pivot() const;
	virtual Rect2 _edit_get_rect() const;
	virtual bool _edit_use_rect() const;

	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const;
#endif

	void set_polygon(const PoolVector<Vector2> &p_polygon);
	PoolVector<Vector2> get_polygon() const;

	void set_internal_vertex_count(int p_count);
	int get_internal_vertex_count() const;

	void set_uv(const PoolVector<Vector2> &p_uv);
	PoolVector<Vector2> get_uv() const;

	void set_polygons(const Array &p_polygons);
	Array get_polygons() const;

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

	void set_texture_rotation_degrees(float p_rot);
	float get_texture_rotation_degrees() const;

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

	void add_bone(const NodePath &p_path = NodePath(), const PoolVector<float> &p_weights = PoolVector<float>());
	int get_bone_count() const;
	NodePath get_bone_path(int p_index) const;
	PoolVector<float> get_bone_weights(int p_index) const;
	void erase_bone(int p_idx);
	void clear_bones();
	void set_bone_weights(int p_index, const PoolVector<float> &p_weights);
	void set_bone_path(int p_index, const NodePath &p_path);

	void set_skeleton(const NodePath &p_skeleton);
	NodePath get_skeleton() const;

	Polygon2D();
};

#endif // POLYGON_2D_H
