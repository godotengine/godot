/**************************************************************************/
/*  canvas_item_editor_gizmos.h                                           */
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

#pragma once

#include "scene/main/canvas_item.h"

class EditorCanvasItemGizmoPlugin;

class EditorCanvasItemGizmo : public CanvasItemGizmo {
	GDCLASS(EditorCanvasItemGizmo, CanvasItemGizmo);

	struct Instance {
		RID instance;
		Transform2D xform;

		void create_instance(CanvasItem *p_base, bool p_hidden = false);
	};

	bool selected;

	Vector<Vector2> collision_segments;
	Vector<Rect2> collision_rects;
	Vector<Vector<Vector2>> collision_polygons;
	// TODO: GIZMOS: i think we're going to need exclusion shapes, so plugins can properly model
	//  selection of 2D shapes with holes that cannot be modeled with simple polygons (e.g. a torus shape)

	Vector<Vector2> handles;
	Vector<int> handle_ids;
	Vector<Vector2> secondary_handles;
	Vector<int> secondary_handle_ids;

	bool valid;
	bool hidden;
	Vector<Instance> instances;
	CanvasItem *canvas_item = nullptr;

	void _set_canvas_item(CanvasItem *p_canvas_item) { set_canvas_item(Object::cast_to<CanvasItem>(p_canvas_item)); }

protected:
	static void _bind_methods();

	EditorCanvasItemGizmoPlugin *gizmo_plugin = nullptr;

	GDVIRTUAL0(_redraw)
	GDVIRTUAL2RC(String, _get_handle_name, int, bool)
	GDVIRTUAL2RC(bool, _is_handle_highlighted, int, bool)
	GDVIRTUAL2RC(Variant, _get_handle_value, int, bool)
	GDVIRTUAL2(_begin_handle_action, int, bool)
	GDVIRTUAL3(_set_handle, int, bool, Vector2)
	GDVIRTUAL4(_commit_handle, int, bool, Variant, bool)

	GDVIRTUAL2RC(int, _subgizmos_intersect_point, Vector2, real_t)
	GDVIRTUAL1RC(Vector<int>, _subgizmos_intersect_rect, Rect2)
	GDVIRTUAL1RC(Transform2D, _get_subgizmo_transform, int)
	GDVIRTUAL2(_set_subgizmo_transform, int, Transform2D)
	GDVIRTUAL3(_commit_subgizmos, Vector<int>, TypedArray<Transform2D>, bool)

public:
	void add_circle(const Vector2 &p_pos, float p_radius, const Color &p_color = Color(1, 1, 1));
	void add_polygon(const Vector<Vector2> &p_polygon, const Color &p_color = Color(1, 1, 1));
	void add_polyline(const Vector<Vector2> &p_points, const Color &p_color = Color(1, 1, 1));
	void add_rect(const Rect2 &p_rect, const Color &p_color = Color(1, 1, 1));

	void add_collision_segments(const Vector<Vector2> &p_lines);
	void add_collision_rect(const Rect2 &p_rect);
	void add_collision_polygon(const Vector<Vector2> &p_polygon);

	void add_handles(const Vector<Vector2> &p_handles, Ref<Texture2D> p_texture, const Vector<int> &p_ids = Vector<int>(), bool p_secondary = false);

	virtual bool is_handle_highlighted(int p_id, bool p_secondary) const;
	virtual String get_handle_name(int p_id, bool p_secondary) const;
	virtual Variant get_handle_value(int p_id, bool p_secondary) const;
	virtual void begin_handle_action(int p_id, bool p_secondary);
	virtual void set_handle(int p_id, bool p_secondary, const Point2 &p_point);
	virtual void commit_handle(int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false);

	virtual int subgizmos_intersect_point(const Point2 &p_point, real_t p_max_distance) const; // TODO: GIZMOS: docs -> this position is in local space.
	virtual Vector<int> subgizmos_intersect_rect(const Rect2 &p_rect) const; // TODO: GIZMOS: docs -> this rect is in canvas space
	virtual Transform2D get_subgizmo_transform(int p_id) const;
	virtual void set_subgizmo_transform(int p_id, const Transform2D &p_xform);
	virtual void commit_subgizmos(const Vector<int> &p_ids, const Vector<Transform2D> &p_transforms, bool p_cancel = false);

	void set_selected(bool p_selected) { selected = p_selected; }
	bool is_selected() const { return selected; }

	void set_canvas_item(CanvasItem *p_canvas_item);
	CanvasItem *get_canvas_item() const { return canvas_item; }

	Ref<EditorCanvasItemGizmoPlugin> get_plugin() const { return gizmo_plugin; }

	bool intersect_rect(const Rect2 &p_rect) const; // TODO: GIZMOS: docs -> this rect is in canvas space
	void handles_intersect_point(const Point2 &p_point, real_t p_max_distance, bool p_shift_pressed, int &r_id, bool &r_secondary); // TODO: GIZMOS: docs -> this position is in local space, so is distance
	bool intersect_point(const Point2 &p_point, real_t p_max_distance) const; // TODO: GIZMOS: docs -> this position is in local space, so is max_distance
	bool is_subgizmo_selected(int p_id) const;
	Vector<int> get_subgizmo_selection() const;

	virtual void clear() override;
	virtual void create() override;
	virtual void transform() override;
	virtual void redraw() override;
	virtual void free() override;

	virtual bool is_editable() const;

	void set_hidden(bool p_hidden);
	void set_plugin(EditorCanvasItemGizmoPlugin *p_plugin);

	EditorCanvasItemGizmo();
	~EditorCanvasItemGizmo();
};

class EditorCanvasItemGizmoPlugin : public Resource {
	GDCLASS(EditorCanvasItemGizmoPlugin, Resource);

public:
	static const int VISIBLE = 0;
	static const int HIDDEN = 1;

protected:
	int current_state;
	HashSet<EditorCanvasItemGizmo *> current_gizmos;

	static void _bind_methods();
	virtual bool has_gizmo(CanvasItem *p_canvas_item);
	virtual Ref<EditorCanvasItemGizmo> create_gizmo(CanvasItem *p_canvas_item);

	GDVIRTUAL1RC(bool, _has_gizmo, CanvasItem *)
	GDVIRTUAL1RC(Ref<EditorCanvasItemGizmo>, _create_gizmo, CanvasItem *)

	GDVIRTUAL0RC(String, _get_gizmo_name)
	GDVIRTUAL0RC(int, _get_priority)
	GDVIRTUAL0RC(bool, _can_be_hidden)
	GDVIRTUAL0RC(bool, _is_selectable_when_hidden)

	GDVIRTUAL1(_redraw, Ref<EditorCanvasItemGizmo>)
	GDVIRTUAL3RC(String, _get_handle_name, Ref<EditorCanvasItemGizmo>, int, bool)
	GDVIRTUAL3RC(bool, _is_handle_highlighted, Ref<EditorCanvasItemGizmo>, int, bool)
	GDVIRTUAL3RC(Variant, _get_handle_value, Ref<EditorCanvasItemGizmo>, int, bool)

	GDVIRTUAL3(_begin_handle_action, Ref<EditorCanvasItemGizmo>, int, bool)
	GDVIRTUAL4(_set_handle, Ref<EditorCanvasItemGizmo>, int, bool, Vector2)
	GDVIRTUAL5(_commit_handle, Ref<EditorCanvasItemGizmo>, int, bool, Variant, bool)

	GDVIRTUAL3RC(int, _subgizmos_intersect_point, Ref<EditorCanvasItemGizmo>, Vector2, real_t)
	GDVIRTUAL2RC(Vector<int>, _subgizmos_intersect_rect, Ref<EditorCanvasItemGizmo>, Rect2)
	GDVIRTUAL2RC(Transform2D, _get_subgizmo_transform, Ref<EditorCanvasItemGizmo>, int)
	GDVIRTUAL3(_set_subgizmo_transform, Ref<EditorCanvasItemGizmo>, int, Transform2D)
	GDVIRTUAL4(_commit_subgizmos, Ref<EditorCanvasItemGizmo>, Vector<int>, TypedArray<Transform2D>, bool)

public:
	virtual String get_gizmo_name() const;
	virtual int get_priority() const;
	virtual bool can_be_hidden() const;
	virtual bool is_selectable_when_hidden() const;
	virtual bool can_commit_handle_on_click() const;

	virtual void redraw(EditorCanvasItemGizmo *p_gizmo);
	virtual bool is_handle_highlighted(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const;
	virtual String get_handle_name(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const;
	virtual Variant get_handle_value(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const;
	virtual void begin_handle_action(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary);
	virtual void set_handle(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary, const Point2 &p_point);
	virtual void commit_handle(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel = false);

	virtual int subgizmos_intersect_point(const EditorCanvasItemGizmo *p_gizmo, const Vector2 &p_point, real_t p_max_distance) const;
	virtual Vector<int> subgizmos_intersect_rect(const EditorCanvasItemGizmo *p_gizmo, const Rect2 &p_rect) const;
	virtual Transform2D get_subgizmo_transform(const EditorCanvasItemGizmo *p_gizmo, int p_id) const;
	virtual void set_subgizmo_transform(const EditorCanvasItemGizmo *p_gizmo, int p_id, const Transform2D &p_xform);
	virtual void commit_subgizmos(const EditorCanvasItemGizmo *p_gizmo, const Vector<int> &p_ids, const Vector<Transform2D> &p_transforms, bool p_cancel = false);

	Ref<EditorCanvasItemGizmo> get_gizmo(CanvasItem *p_canvas_item);
	void set_state(int p_state);
	int get_state() const;
	void unregister_gizmo(EditorCanvasItemGizmo *p_gizmo);

	EditorCanvasItemGizmoPlugin();
	virtual ~EditorCanvasItemGizmoPlugin();
};
