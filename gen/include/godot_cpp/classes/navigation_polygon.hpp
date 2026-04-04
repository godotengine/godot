/**************************************************************************/
/*  navigation_polygon.hpp                                                */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class NavigationMesh;

class NavigationPolygon : public Resource {
	GDEXTENSION_CLASS(NavigationPolygon, Resource)

public:
	enum SamplePartitionType {
		SAMPLE_PARTITION_CONVEX_PARTITION = 0,
		SAMPLE_PARTITION_TRIANGULATE = 1,
		SAMPLE_PARTITION_MAX = 2,
	};

	enum ParsedGeometryType {
		PARSED_GEOMETRY_MESH_INSTANCES = 0,
		PARSED_GEOMETRY_STATIC_COLLIDERS = 1,
		PARSED_GEOMETRY_BOTH = 2,
		PARSED_GEOMETRY_MAX = 3,
	};

	enum SourceGeometryMode {
		SOURCE_GEOMETRY_ROOT_NODE_CHILDREN = 0,
		SOURCE_GEOMETRY_GROUPS_WITH_CHILDREN = 1,
		SOURCE_GEOMETRY_GROUPS_EXPLICIT = 2,
		SOURCE_GEOMETRY_MAX = 3,
	};

	void set_vertices(const PackedVector2Array &p_vertices);
	PackedVector2Array get_vertices() const;
	void add_polygon(const PackedInt32Array &p_polygon);
	int32_t get_polygon_count() const;
	PackedInt32Array get_polygon(int32_t p_idx);
	void clear_polygons();
	Ref<NavigationMesh> get_navigation_mesh();
	void add_outline(const PackedVector2Array &p_outline);
	void add_outline_at_index(const PackedVector2Array &p_outline, int32_t p_index);
	int32_t get_outline_count() const;
	void set_outline(int32_t p_idx, const PackedVector2Array &p_outline);
	PackedVector2Array get_outline(int32_t p_idx) const;
	void remove_outline(int32_t p_idx);
	void clear_outlines();
	void make_polygons_from_outlines();
	void set_cell_size(float p_cell_size);
	float get_cell_size() const;
	void set_border_size(float p_border_size);
	float get_border_size() const;
	void set_sample_partition_type(NavigationPolygon::SamplePartitionType p_sample_partition_type);
	NavigationPolygon::SamplePartitionType get_sample_partition_type() const;
	void set_parsed_geometry_type(NavigationPolygon::ParsedGeometryType p_geometry_type);
	NavigationPolygon::ParsedGeometryType get_parsed_geometry_type() const;
	void set_parsed_collision_mask(uint32_t p_mask);
	uint32_t get_parsed_collision_mask() const;
	void set_parsed_collision_mask_value(int32_t p_layer_number, bool p_value);
	bool get_parsed_collision_mask_value(int32_t p_layer_number) const;
	void set_source_geometry_mode(NavigationPolygon::SourceGeometryMode p_geometry_mode);
	NavigationPolygon::SourceGeometryMode get_source_geometry_mode() const;
	void set_source_geometry_group_name(const StringName &p_group_name);
	StringName get_source_geometry_group_name() const;
	void set_agent_radius(float p_agent_radius);
	float get_agent_radius() const;
	void set_baking_rect(const Rect2 &p_rect);
	Rect2 get_baking_rect() const;
	void set_baking_rect_offset(const Vector2 &p_rect_offset);
	Vector2 get_baking_rect_offset() const;
	void clear();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(NavigationPolygon::SamplePartitionType);
VARIANT_ENUM_CAST(NavigationPolygon::ParsedGeometryType);
VARIANT_ENUM_CAST(NavigationPolygon::SourceGeometryMode);

