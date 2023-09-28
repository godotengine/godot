/**************************************************************************/
/*  navigation_polygon.h                                                  */
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

#ifndef NAVIGATION_POLYGON_H
#define NAVIGATION_POLYGON_H

#include "scene/2d/node_2d.h"
#include "scene/resources/navigation_mesh.h"

class NavigationPolygon : public Resource {
	GDCLASS(NavigationPolygon, Resource);

	Vector<Vector2> vertices;
	struct Polygon {
		Vector<int> indices;
	};
	Vector<Polygon> polygons;
	Vector<Vector<Vector2>> outlines;
	Vector<Vector<Vector2>> baked_outlines;

	mutable Rect2 item_rect;
	mutable bool rect_cache_dirty = true;

	Mutex navigation_mesh_generation;
	// Navigation mesh
	Ref<NavigationMesh> navigation_mesh;

	real_t cell_size = 1.0f; // Must match ProjectSettings default 2D cell_size.

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

	void _set_polygons(const TypedArray<Vector<int32_t>> &p_array);
	TypedArray<Vector<int32_t>> _get_polygons() const;

	void _set_outlines(const TypedArray<Vector<Vector2>> &p_array);
	TypedArray<Vector<Vector2>> _get_outlines() const;

public:
#ifdef TOOLS_ENABLED
	Rect2 _edit_get_rect() const;
	bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const;
#endif

	enum ParsedGeometryType {
		PARSED_GEOMETRY_MESH_INSTANCES = 0,
		PARSED_GEOMETRY_STATIC_COLLIDERS,
		PARSED_GEOMETRY_BOTH,
		PARSED_GEOMETRY_MAX
	};

	enum SourceGeometryMode {
		SOURCE_GEOMETRY_ROOT_NODE_CHILDREN = 0,
		SOURCE_GEOMETRY_GROUPS_WITH_CHILDREN,
		SOURCE_GEOMETRY_GROUPS_EXPLICIT,
		SOURCE_GEOMETRY_MAX
	};

	real_t agent_radius = 10.0f;

	ParsedGeometryType parsed_geometry_type = PARSED_GEOMETRY_BOTH;
	uint32_t parsed_collision_mask = 0xFFFFFFFF;

	SourceGeometryMode source_geometry_mode = SOURCE_GEOMETRY_ROOT_NODE_CHILDREN;
	StringName source_geometry_group_name = "navigation_polygon_source_group";

	void set_vertices(const Vector<Vector2> &p_vertices);
	Vector<Vector2> get_vertices() const;

	void add_polygon(const Vector<int> &p_polygon);
	int get_polygon_count() const;

	void add_outline(const Vector<Vector2> &p_outline);
	void add_outline_at_index(const Vector<Vector2> &p_outline, int p_index);
	void set_outline(int p_idx, const Vector<Vector2> &p_outline);
	Vector<Vector2> get_outline(int p_idx) const;
	void remove_outline(int p_idx);
	int get_outline_count() const;

	void clear_outlines();
#ifndef DISABLE_DEPRECATED
	void make_polygons_from_outlines();
#endif // DISABLE_DEPRECATED

	void set_polygons(const Vector<Vector<int>> &p_polygons);
	const Vector<Vector<int>> &get_polygons() const;
	Vector<int> get_polygon(int p_idx);
	void clear_polygons();

	void set_parsed_geometry_type(ParsedGeometryType p_geometry_type);
	ParsedGeometryType get_parsed_geometry_type() const;

	void set_parsed_collision_mask(uint32_t p_mask);
	uint32_t get_parsed_collision_mask() const;

	void set_parsed_collision_mask_value(int p_layer_number, bool p_value);
	bool get_parsed_collision_mask_value(int p_layer_number) const;

	void set_source_geometry_mode(SourceGeometryMode p_geometry_mode);
	SourceGeometryMode get_source_geometry_mode() const;

	void set_source_geometry_group_name(StringName p_group_name);
	StringName get_source_geometry_group_name() const;

	void set_agent_radius(real_t p_value);
	real_t get_agent_radius() const;

	Ref<NavigationMesh> get_navigation_mesh();

	void set_cell_size(real_t p_cell_size);
	real_t get_cell_size() const;

	void clear();

	NavigationPolygon();
	~NavigationPolygon() {}
};

VARIANT_ENUM_CAST(NavigationPolygon::ParsedGeometryType);
VARIANT_ENUM_CAST(NavigationPolygon::SourceGeometryMode);

#endif // NAVIGATION_POLYGON_H
