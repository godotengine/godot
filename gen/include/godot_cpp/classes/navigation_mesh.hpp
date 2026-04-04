/**************************************************************************/
/*  navigation_mesh.hpp                                                   */
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
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Mesh;

class NavigationMesh : public Resource {
	GDEXTENSION_CLASS(NavigationMesh, Resource)

public:
	enum SamplePartitionType {
		SAMPLE_PARTITION_WATERSHED = 0,
		SAMPLE_PARTITION_MONOTONE = 1,
		SAMPLE_PARTITION_LAYERS = 2,
		SAMPLE_PARTITION_MAX = 3,
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

	void set_sample_partition_type(NavigationMesh::SamplePartitionType p_sample_partition_type);
	NavigationMesh::SamplePartitionType get_sample_partition_type() const;
	void set_parsed_geometry_type(NavigationMesh::ParsedGeometryType p_geometry_type);
	NavigationMesh::ParsedGeometryType get_parsed_geometry_type() const;
	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;
	void set_collision_mask_value(int32_t p_layer_number, bool p_value);
	bool get_collision_mask_value(int32_t p_layer_number) const;
	void set_source_geometry_mode(NavigationMesh::SourceGeometryMode p_mask);
	NavigationMesh::SourceGeometryMode get_source_geometry_mode() const;
	void set_source_group_name(const StringName &p_mask);
	StringName get_source_group_name() const;
	void set_cell_size(float p_cell_size);
	float get_cell_size() const;
	void set_cell_height(float p_cell_height);
	float get_cell_height() const;
	void set_border_size(float p_border_size);
	float get_border_size() const;
	void set_agent_height(float p_agent_height);
	float get_agent_height() const;
	void set_agent_radius(float p_agent_radius);
	float get_agent_radius();
	void set_agent_max_climb(float p_agent_max_climb);
	float get_agent_max_climb() const;
	void set_agent_max_slope(float p_agent_max_slope);
	float get_agent_max_slope() const;
	void set_region_min_size(float p_region_min_size);
	float get_region_min_size() const;
	void set_region_merge_size(float p_region_merge_size);
	float get_region_merge_size() const;
	void set_edge_max_length(float p_edge_max_length);
	float get_edge_max_length() const;
	void set_edge_max_error(float p_edge_max_error);
	float get_edge_max_error() const;
	void set_vertices_per_polygon(float p_vertices_per_polygon);
	float get_vertices_per_polygon() const;
	void set_detail_sample_distance(float p_detail_sample_dist);
	float get_detail_sample_distance() const;
	void set_detail_sample_max_error(float p_detail_sample_max_error);
	float get_detail_sample_max_error() const;
	void set_filter_low_hanging_obstacles(bool p_filter_low_hanging_obstacles);
	bool get_filter_low_hanging_obstacles() const;
	void set_filter_ledge_spans(bool p_filter_ledge_spans);
	bool get_filter_ledge_spans() const;
	void set_filter_walkable_low_height_spans(bool p_filter_walkable_low_height_spans);
	bool get_filter_walkable_low_height_spans() const;
	void set_filter_baking_aabb(const AABB &p_baking_aabb);
	AABB get_filter_baking_aabb() const;
	void set_filter_baking_aabb_offset(const Vector3 &p_baking_aabb_offset);
	Vector3 get_filter_baking_aabb_offset() const;
	void set_vertices(const PackedVector3Array &p_vertices);
	PackedVector3Array get_vertices() const;
	void add_polygon(const PackedInt32Array &p_polygon);
	int32_t get_polygon_count() const;
	PackedInt32Array get_polygon(int32_t p_idx);
	void clear_polygons();
	void create_from_mesh(const Ref<Mesh> &p_mesh);
	void clear();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(NavigationMesh::SamplePartitionType);
VARIANT_ENUM_CAST(NavigationMesh::ParsedGeometryType);
VARIANT_ENUM_CAST(NavigationMesh::SourceGeometryMode);

