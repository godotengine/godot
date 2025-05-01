/**************************************************************************/
/*  navigation_mesh.cpp                                                   */
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

#include "navigation_mesh.h"

#ifdef DEBUG_ENABLED
#include "servers/navigation_server_3d.h"
#endif // DEBUG_ENABLED

void NavigationMesh::create_from_mesh(const Ref<Mesh> &p_mesh) {
	RWLockWrite write_lock(rwlock);
	ERR_FAIL_COND(p_mesh.is_null());

	vertices = Vector<Vector3>();
	polygons.clear();

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			WARN_PRINT("A mesh surface was skipped when creating a NavigationMesh due to wrong primitive type in the source mesh. Mesh surface must be made out of triangles.");
			continue;
		}
		Array arr = p_mesh->surface_get_arrays(i);
		ERR_CONTINUE(arr.size() != Mesh::ARRAY_MAX);

		Vector<Vector3> varr = arr[Mesh::ARRAY_VERTEX];
		Vector<int> iarr = arr[Mesh::ARRAY_INDEX];
		if (varr.is_empty() || iarr.is_empty()) {
			WARN_PRINT("A mesh surface was skipped when creating a NavigationMesh due to an empty vertex or index array.");
			continue;
		}

		int from = vertices.size();
		vertices.append_array(varr);
		int rlen = iarr.size();
		const int *r = iarr.ptr();

		Vector<int> polygon;
		for (int j = 0; j < rlen; j += 3) {
			polygon.resize(3);
			polygon.write[0] = r[j + 0] + from;
			polygon.write[1] = r[j + 1] + from;
			polygon.write[2] = r[j + 2] + from;
			polygons.push_back(polygon);
		}
	}
}

void NavigationMesh::set_sample_partition_type(SamplePartitionType p_value) {
	ERR_FAIL_INDEX(p_value, SAMPLE_PARTITION_MAX);
	partition_type = p_value;
}

NavigationMesh::SamplePartitionType NavigationMesh::get_sample_partition_type() const {
	return partition_type;
}

void NavigationMesh::set_parsed_geometry_type(ParsedGeometryType p_value) {
	ERR_FAIL_INDEX(p_value, PARSED_GEOMETRY_MAX);
	parsed_geometry_type = p_value;
	notify_property_list_changed();
}

NavigationMesh::ParsedGeometryType NavigationMesh::get_parsed_geometry_type() const {
	return parsed_geometry_type;
}

void NavigationMesh::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t NavigationMesh::get_collision_mask() const {
	return collision_mask;
}

void NavigationMesh::set_collision_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_collision_mask(mask);
}

bool NavigationMesh::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

void NavigationMesh::set_source_geometry_mode(SourceGeometryMode p_geometry_mode) {
	ERR_FAIL_INDEX(p_geometry_mode, SOURCE_GEOMETRY_MAX);
	source_geometry_mode = p_geometry_mode;
	notify_property_list_changed();
}

NavigationMesh::SourceGeometryMode NavigationMesh::get_source_geometry_mode() const {
	return source_geometry_mode;
}

void NavigationMesh::set_source_group_name(const StringName &p_group_name) {
	source_group_name = p_group_name;
}

StringName NavigationMesh::get_source_group_name() const {
	return source_group_name;
}

void NavigationMesh::set_cell_size(float p_value) {
	ERR_FAIL_COND(p_value <= 0);
	cell_size = p_value;
}

float NavigationMesh::get_cell_size() const {
	return cell_size;
}

void NavigationMesh::set_cell_height(float p_value) {
	ERR_FAIL_COND(p_value <= 0);
	cell_height = p_value;
}

float NavigationMesh::get_cell_height() const {
	return cell_height;
}

void NavigationMesh::set_border_size(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	border_size = p_value;
}

float NavigationMesh::get_border_size() const {
	return border_size;
}

void NavigationMesh::set_agent_height(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	agent_height = p_value;
}

float NavigationMesh::get_agent_height() const {
	return agent_height;
}

void NavigationMesh::set_agent_radius(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	agent_radius = p_value;
}

float NavigationMesh::get_agent_radius() {
	return agent_radius;
}

void NavigationMesh::set_agent_max_climb(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	agent_max_climb = p_value;
}

float NavigationMesh::get_agent_max_climb() const {
	return agent_max_climb;
}

void NavigationMesh::set_agent_max_slope(float p_value) {
	ERR_FAIL_COND(p_value < 0 || p_value > 90);
	agent_max_slope = p_value;
}

float NavigationMesh::get_agent_max_slope() const {
	return agent_max_slope;
}

void NavigationMesh::set_region_min_size(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	region_min_size = p_value;
}

float NavigationMesh::get_region_min_size() const {
	return region_min_size;
}

void NavigationMesh::set_region_merge_size(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	region_merge_size = p_value;
}

float NavigationMesh::get_region_merge_size() const {
	return region_merge_size;
}

void NavigationMesh::set_edge_max_length(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	edge_max_length = p_value;
}

float NavigationMesh::get_edge_max_length() const {
	return edge_max_length;
}

void NavigationMesh::set_edge_max_error(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	edge_max_error = p_value;
}

float NavigationMesh::get_edge_max_error() const {
	return edge_max_error;
}

void NavigationMesh::set_vertices_per_polygon(float p_value) {
	ERR_FAIL_COND(p_value < 3);
	vertices_per_polygon = p_value;
}

float NavigationMesh::get_vertices_per_polygon() const {
	return vertices_per_polygon;
}

void NavigationMesh::set_detail_sample_distance(float p_value) {
	ERR_FAIL_COND(p_value < 0.1);
	detail_sample_distance = p_value;
}

float NavigationMesh::get_detail_sample_distance() const {
	return detail_sample_distance;
}

void NavigationMesh::set_detail_sample_max_error(float p_value) {
	ERR_FAIL_COND(p_value < 0);
	detail_sample_max_error = p_value;
}

float NavigationMesh::get_detail_sample_max_error() const {
	return detail_sample_max_error;
}

void NavigationMesh::set_filter_low_hanging_obstacles(bool p_value) {
	filter_low_hanging_obstacles = p_value;
}

bool NavigationMesh::get_filter_low_hanging_obstacles() const {
	return filter_low_hanging_obstacles;
}

void NavigationMesh::set_filter_ledge_spans(bool p_value) {
	filter_ledge_spans = p_value;
}

bool NavigationMesh::get_filter_ledge_spans() const {
	return filter_ledge_spans;
}

void NavigationMesh::set_filter_walkable_low_height_spans(bool p_value) {
	filter_walkable_low_height_spans = p_value;
}

bool NavigationMesh::get_filter_walkable_low_height_spans() const {
	return filter_walkable_low_height_spans;
}

void NavigationMesh::set_filter_baking_aabb(const AABB &p_aabb) {
	filter_baking_aabb = p_aabb;
	emit_changed();
}

AABB NavigationMesh::get_filter_baking_aabb() const {
	return filter_baking_aabb;
}

void NavigationMesh::set_filter_baking_aabb_offset(const Vector3 &p_aabb_offset) {
	filter_baking_aabb_offset = p_aabb_offset;
	emit_changed();
}

Vector3 NavigationMesh::get_filter_baking_aabb_offset() const {
	return filter_baking_aabb_offset;
}

void NavigationMesh::set_vertices(const Vector<Vector3> &p_vertices) {
	RWLockWrite write_lock(rwlock);
	vertices = p_vertices;
	notify_property_list_changed();
}

Vector<Vector3> NavigationMesh::get_vertices() const {
	RWLockRead read_lock(rwlock);
	return vertices;
}

void NavigationMesh::_set_polygons(const Array &p_array) {
	RWLockWrite write_lock(rwlock);
	polygons.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		polygons.write[i] = p_array[i];
	}
	notify_property_list_changed();
}

Array NavigationMesh::_get_polygons() const {
	RWLockRead read_lock(rwlock);
	Array ret;
	ret.resize(polygons.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = polygons[i];
	}

	return ret;
}

void NavigationMesh::set_polygons(const Vector<Vector<int>> &p_polygons) {
	RWLockWrite write_lock(rwlock);
	polygons = p_polygons;
	notify_property_list_changed();
}

Vector<Vector<int>> NavigationMesh::get_polygons() const {
	RWLockRead read_lock(rwlock);
	return polygons;
}

void NavigationMesh::add_polygon(const Vector<int> &p_polygon) {
	RWLockWrite write_lock(rwlock);
	polygons.push_back(p_polygon);
	notify_property_list_changed();
}

int NavigationMesh::get_polygon_count() const {
	RWLockRead read_lock(rwlock);
	return polygons.size();
}

Vector<int> NavigationMesh::get_polygon(int p_idx) {
	RWLockRead read_lock(rwlock);
	ERR_FAIL_INDEX_V(p_idx, polygons.size(), Vector<int>());
	return polygons[p_idx];
}

void NavigationMesh::clear_polygons() {
	RWLockWrite write_lock(rwlock);
	polygons.clear();
}

void NavigationMesh::clear() {
	RWLockWrite write_lock(rwlock);
	polygons.clear();
	vertices.clear();
}

void NavigationMesh::set_data(const Vector<Vector3> &p_vertices, const Vector<Vector<int>> &p_polygons) {
	RWLockWrite write_lock(rwlock);
	vertices = p_vertices;
	polygons = p_polygons;
}

void NavigationMesh::get_data(Vector<Vector3> &r_vertices, Vector<Vector<int>> &r_polygons) {
	RWLockRead read_lock(rwlock);
	r_vertices = vertices;
	r_polygons = polygons;
}

#ifdef DEBUG_ENABLED
Ref<ArrayMesh> NavigationMesh::get_debug_mesh() {
	if (debug_mesh.is_valid()) {
		// Blocks further updates for now, code below is intended for dynamic updates e.g. when settings change.
		return debug_mesh;
	}

	if (debug_mesh.is_null()) {
		debug_mesh.instantiate();
	} else {
		debug_mesh->clear_surfaces();
	}

	if (vertices.is_empty()) {
		return debug_mesh;
	}

	RWLockRead read_lock(rwlock);

	int polygon_count = get_polygon_count();

	if (polygon_count < 1) {
		// no face, no play
		return debug_mesh;
	}

	// build geometry face surface
	Vector<Vector3> face_vertex_array;
	face_vertex_array.resize(polygon_count * 3);

	for (int i = 0; i < polygon_count; i++) {
		Vector<int> polygon = get_polygon(i);

		face_vertex_array.push_back(vertices[polygon[0]]);
		face_vertex_array.push_back(vertices[polygon[1]]);
		face_vertex_array.push_back(vertices[polygon[2]]);
	}

	Array face_mesh_array;
	face_mesh_array.resize(Mesh::ARRAY_MAX);
	face_mesh_array[Mesh::ARRAY_VERTEX] = face_vertex_array;

	// if enabled add vertex colors to colorize each face individually
	bool enabled_geometry_face_random_color = NavigationServer3D::get_singleton()->get_debug_navigation_enable_geometry_face_random_color();
	if (enabled_geometry_face_random_color) {
		Color debug_navigation_geometry_face_color = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_color();
		Color polygon_color = debug_navigation_geometry_face_color;

		Vector<Color> face_color_array;
		face_color_array.resize(polygon_count * 3);

		for (int i = 0; i < polygon_count; i++) {
			polygon_color = debug_navigation_geometry_face_color * (Color(Math::randf(), Math::randf(), Math::randf()));

			face_color_array.push_back(polygon_color);
			face_color_array.push_back(polygon_color);
			face_color_array.push_back(polygon_color);
		}
		face_mesh_array[Mesh::ARRAY_COLOR] = face_color_array;
	}

	debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, face_mesh_array);
	Ref<StandardMaterial3D> debug_geometry_face_material = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_face_material();
	debug_mesh->surface_set_material(0, debug_geometry_face_material);

	// if enabled build geometry edge line surface
	bool enabled_edge_lines = NavigationServer3D::get_singleton()->get_debug_navigation_enable_edge_lines();

	if (enabled_edge_lines) {
		Vector<Vector3> line_vertex_array;
		line_vertex_array.resize(polygon_count * 6);

		for (int i = 0; i < polygon_count; i++) {
			Vector<int> polygon = get_polygon(i);

			line_vertex_array.push_back(vertices[polygon[0]]);
			line_vertex_array.push_back(vertices[polygon[1]]);
			line_vertex_array.push_back(vertices[polygon[1]]);
			line_vertex_array.push_back(vertices[polygon[2]]);
			line_vertex_array.push_back(vertices[polygon[2]]);
			line_vertex_array.push_back(vertices[polygon[0]]);
		}

		Array line_mesh_array;
		line_mesh_array.resize(Mesh::ARRAY_MAX);
		line_mesh_array[Mesh::ARRAY_VERTEX] = line_vertex_array;
		debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, line_mesh_array);
		Ref<StandardMaterial3D> debug_geometry_edge_material = NavigationServer3D::get_singleton()->get_debug_navigation_geometry_edge_material();
		debug_mesh->surface_set_material(1, debug_geometry_edge_material);
	}

	return debug_mesh;
}
#endif // DEBUG_ENABLED

void NavigationMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sample_partition_type", "sample_partition_type"), &NavigationMesh::set_sample_partition_type);
	ClassDB::bind_method(D_METHOD("get_sample_partition_type"), &NavigationMesh::get_sample_partition_type);

	ClassDB::bind_method(D_METHOD("set_parsed_geometry_type", "geometry_type"), &NavigationMesh::set_parsed_geometry_type);
	ClassDB::bind_method(D_METHOD("get_parsed_geometry_type"), &NavigationMesh::get_parsed_geometry_type);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &NavigationMesh::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &NavigationMesh::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &NavigationMesh::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &NavigationMesh::get_collision_mask_value);

	ClassDB::bind_method(D_METHOD("set_source_geometry_mode", "mask"), &NavigationMesh::set_source_geometry_mode);
	ClassDB::bind_method(D_METHOD("get_source_geometry_mode"), &NavigationMesh::get_source_geometry_mode);

	ClassDB::bind_method(D_METHOD("set_source_group_name", "mask"), &NavigationMesh::set_source_group_name);
	ClassDB::bind_method(D_METHOD("get_source_group_name"), &NavigationMesh::get_source_group_name);

	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &NavigationMesh::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &NavigationMesh::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_cell_height", "cell_height"), &NavigationMesh::set_cell_height);
	ClassDB::bind_method(D_METHOD("get_cell_height"), &NavigationMesh::get_cell_height);

	ClassDB::bind_method(D_METHOD("set_border_size", "border_size"), &NavigationMesh::set_border_size);
	ClassDB::bind_method(D_METHOD("get_border_size"), &NavigationMesh::get_border_size);

	ClassDB::bind_method(D_METHOD("set_agent_height", "agent_height"), &NavigationMesh::set_agent_height);
	ClassDB::bind_method(D_METHOD("get_agent_height"), &NavigationMesh::get_agent_height);

	ClassDB::bind_method(D_METHOD("set_agent_radius", "agent_radius"), &NavigationMesh::set_agent_radius);
	ClassDB::bind_method(D_METHOD("get_agent_radius"), &NavigationMesh::get_agent_radius);

	ClassDB::bind_method(D_METHOD("set_agent_max_climb", "agent_max_climb"), &NavigationMesh::set_agent_max_climb);
	ClassDB::bind_method(D_METHOD("get_agent_max_climb"), &NavigationMesh::get_agent_max_climb);

	ClassDB::bind_method(D_METHOD("set_agent_max_slope", "agent_max_slope"), &NavigationMesh::set_agent_max_slope);
	ClassDB::bind_method(D_METHOD("get_agent_max_slope"), &NavigationMesh::get_agent_max_slope);

	ClassDB::bind_method(D_METHOD("set_region_min_size", "region_min_size"), &NavigationMesh::set_region_min_size);
	ClassDB::bind_method(D_METHOD("get_region_min_size"), &NavigationMesh::get_region_min_size);

	ClassDB::bind_method(D_METHOD("set_region_merge_size", "region_merge_size"), &NavigationMesh::set_region_merge_size);
	ClassDB::bind_method(D_METHOD("get_region_merge_size"), &NavigationMesh::get_region_merge_size);

	ClassDB::bind_method(D_METHOD("set_edge_max_length", "edge_max_length"), &NavigationMesh::set_edge_max_length);
	ClassDB::bind_method(D_METHOD("get_edge_max_length"), &NavigationMesh::get_edge_max_length);

	ClassDB::bind_method(D_METHOD("set_edge_max_error", "edge_max_error"), &NavigationMesh::set_edge_max_error);
	ClassDB::bind_method(D_METHOD("get_edge_max_error"), &NavigationMesh::get_edge_max_error);

	ClassDB::bind_method(D_METHOD("set_vertices_per_polygon", "vertices_per_polygon"), &NavigationMesh::set_vertices_per_polygon);
	ClassDB::bind_method(D_METHOD("get_vertices_per_polygon"), &NavigationMesh::get_vertices_per_polygon);

	ClassDB::bind_method(D_METHOD("set_detail_sample_distance", "detail_sample_dist"), &NavigationMesh::set_detail_sample_distance);
	ClassDB::bind_method(D_METHOD("get_detail_sample_distance"), &NavigationMesh::get_detail_sample_distance);

	ClassDB::bind_method(D_METHOD("set_detail_sample_max_error", "detail_sample_max_error"), &NavigationMesh::set_detail_sample_max_error);
	ClassDB::bind_method(D_METHOD("get_detail_sample_max_error"), &NavigationMesh::get_detail_sample_max_error);

	ClassDB::bind_method(D_METHOD("set_filter_low_hanging_obstacles", "filter_low_hanging_obstacles"), &NavigationMesh::set_filter_low_hanging_obstacles);
	ClassDB::bind_method(D_METHOD("get_filter_low_hanging_obstacles"), &NavigationMesh::get_filter_low_hanging_obstacles);

	ClassDB::bind_method(D_METHOD("set_filter_ledge_spans", "filter_ledge_spans"), &NavigationMesh::set_filter_ledge_spans);
	ClassDB::bind_method(D_METHOD("get_filter_ledge_spans"), &NavigationMesh::get_filter_ledge_spans);

	ClassDB::bind_method(D_METHOD("set_filter_walkable_low_height_spans", "filter_walkable_low_height_spans"), &NavigationMesh::set_filter_walkable_low_height_spans);
	ClassDB::bind_method(D_METHOD("get_filter_walkable_low_height_spans"), &NavigationMesh::get_filter_walkable_low_height_spans);
	ClassDB::bind_method(D_METHOD("set_filter_baking_aabb", "baking_aabb"), &NavigationMesh::set_filter_baking_aabb);
	ClassDB::bind_method(D_METHOD("get_filter_baking_aabb"), &NavigationMesh::get_filter_baking_aabb);
	ClassDB::bind_method(D_METHOD("set_filter_baking_aabb_offset", "baking_aabb_offset"), &NavigationMesh::set_filter_baking_aabb_offset);
	ClassDB::bind_method(D_METHOD("get_filter_baking_aabb_offset"), &NavigationMesh::get_filter_baking_aabb_offset);

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationMesh::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationMesh::get_vertices);

	ClassDB::bind_method(D_METHOD("add_polygon", "polygon"), &NavigationMesh::add_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon_count"), &NavigationMesh::get_polygon_count);
	ClassDB::bind_method(D_METHOD("get_polygon", "idx"), &NavigationMesh::get_polygon);
	ClassDB::bind_method(D_METHOD("clear_polygons"), &NavigationMesh::clear_polygons);

	ClassDB::bind_method(D_METHOD("create_from_mesh", "mesh"), &NavigationMesh::create_from_mesh);

	ClassDB::bind_method(D_METHOD("_set_polygons", "polygons"), &NavigationMesh::_set_polygons);
	ClassDB::bind_method(D_METHOD("_get_polygons"), &NavigationMesh::_get_polygons);

	ClassDB::bind_method(D_METHOD("clear"), &NavigationMesh::clear);

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR3_ARRAY, "vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "polygons", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_polygons", "_get_polygons");

	ADD_GROUP("Sampling", "sample_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_partition_type", PROPERTY_HINT_ENUM, "Watershed,Monotone,Layers"), "set_sample_partition_type", "get_sample_partition_type");
	ADD_GROUP("Geometry", "geometry_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "geometry_parsed_geometry_type", PROPERTY_HINT_ENUM, "Mesh Instances,Static Colliders,Both"), "set_parsed_geometry_type", "get_parsed_geometry_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "geometry_collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY_DEFAULT("geometry_collision_mask", 0xFFFFFFFF);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "geometry_source_geometry_mode", PROPERTY_HINT_ENUM, "Root Node Children,Group With Children,Group Explicit"), "set_source_geometry_mode", "get_source_geometry_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "geometry_source_group_name"), "set_source_group_name", "get_source_group_name");
	ADD_PROPERTY_DEFAULT("geometry_source_group_name", StringName("navigation_mesh_source_group"));
	ADD_GROUP("Cells", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cell_size", PROPERTY_HINT_RANGE, "0.01,500.0,0.01,or_greater,suffix:m"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cell_height", PROPERTY_HINT_RANGE, "0.01,500.0,0.01,or_greater,suffix:m"), "set_cell_height", "get_cell_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "border_size", PROPERTY_HINT_RANGE, "0.0,500.0,0.01,or_greater,suffix:m"), "set_border_size", "get_border_size");
	ADD_GROUP("Agents", "agent_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_height", PROPERTY_HINT_RANGE, "0.0,500.0,0.01,or_greater,suffix:m"), "set_agent_height", "get_agent_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_radius", PROPERTY_HINT_RANGE, "0.0,500.0,0.01,or_greater,suffix:m"), "set_agent_radius", "get_agent_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_max_climb", PROPERTY_HINT_RANGE, "0.0,500.0,0.01,or_greater,suffix:m"), "set_agent_max_climb", "get_agent_max_climb");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "agent_max_slope", PROPERTY_HINT_RANGE, "0.02,90.0,0.01,degrees"), "set_agent_max_slope", "get_agent_max_slope");
	ADD_GROUP("Regions", "region_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "region_min_size", PROPERTY_HINT_RANGE, "0.0,150.0,0.01,or_greater"), "set_region_min_size", "get_region_min_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "region_merge_size", PROPERTY_HINT_RANGE, "0.0,150.0,0.01,or_greater"), "set_region_merge_size", "get_region_merge_size");
	ADD_GROUP("Edges", "edge_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "edge_max_length", PROPERTY_HINT_RANGE, "0.0,50.0,0.01,or_greater,suffix:m"), "set_edge_max_length", "get_edge_max_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "edge_max_error", PROPERTY_HINT_RANGE, "0.1,3.0,0.01,or_greater,suffix:m"), "set_edge_max_error", "get_edge_max_error");
	ADD_GROUP("Polygons", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vertices_per_polygon", PROPERTY_HINT_RANGE, "3.0,12.0,1.0,or_greater"), "set_vertices_per_polygon", "get_vertices_per_polygon");
	ADD_GROUP("Details", "detail_");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "detail_sample_distance", PROPERTY_HINT_RANGE, "0.1,16.0,0.01,or_greater,suffix:m"), "set_detail_sample_distance", "get_detail_sample_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "detail_sample_max_error", PROPERTY_HINT_RANGE, "0.0,16.0,0.01,or_greater,suffix:m"), "set_detail_sample_max_error", "get_detail_sample_max_error");
	ADD_GROUP("Filters", "filter_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_low_hanging_obstacles"), "set_filter_low_hanging_obstacles", "get_filter_low_hanging_obstacles");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_ledge_spans"), "set_filter_ledge_spans", "get_filter_ledge_spans");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_walkable_low_height_spans"), "set_filter_walkable_low_height_spans", "get_filter_walkable_low_height_spans");
	ADD_PROPERTY(PropertyInfo(Variant::AABB, "filter_baking_aabb"), "set_filter_baking_aabb", "get_filter_baking_aabb");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "filter_baking_aabb_offset"), "set_filter_baking_aabb_offset", "get_filter_baking_aabb_offset");

	BIND_ENUM_CONSTANT(SAMPLE_PARTITION_WATERSHED);
	BIND_ENUM_CONSTANT(SAMPLE_PARTITION_MONOTONE);
	BIND_ENUM_CONSTANT(SAMPLE_PARTITION_LAYERS);
	BIND_ENUM_CONSTANT(SAMPLE_PARTITION_MAX);

	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_MESH_INSTANCES);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_STATIC_COLLIDERS);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_BOTH);
	BIND_ENUM_CONSTANT(PARSED_GEOMETRY_MAX);

	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_ROOT_NODE_CHILDREN);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_GROUPS_WITH_CHILDREN);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_GROUPS_EXPLICIT);
	BIND_ENUM_CONSTANT(SOURCE_GEOMETRY_MAX);
}

void NavigationMesh::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "geometry_collision_mask") {
		if (parsed_geometry_type == PARSED_GEOMETRY_MESH_INSTANCES) {
			p_property.usage = PROPERTY_USAGE_NONE;
			return;
		}
	}

	if (p_property.name == "geometry_source_group_name") {
		if (source_geometry_mode == SOURCE_GEOMETRY_ROOT_NODE_CHILDREN) {
			p_property.usage = PROPERTY_USAGE_NONE;
			return;
		}
	}
}

#ifndef DISABLE_DEPRECATED
bool NavigationMesh::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "polygon_verts_per_poly") { // Renamed in 4.0 beta 9.
		set_vertices_per_polygon(p_value);
		return true;
	}
	return false;
}

bool NavigationMesh::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "polygon_verts_per_poly") { // Renamed in 4.0 beta 9.
		r_ret = get_vertices_per_polygon();
		return true;
	}
	return false;
}
#endif // DISABLE_DEPRECATED
