/*************************************************************************/
/*  navigation_mesh.cpp                                                  */
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
#include "navigation_mesh.h"
#include "mesh_instance.h"
#include "navigation.h"

void NavigationMesh::create_from_mesh(const Ref<Mesh> &p_mesh) {

	vertices = PoolVector<Vector3>();
	clear_polygons();

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {

		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
			continue;
		Array arr = p_mesh->surface_get_arrays(i);
		PoolVector<Vector3> varr = arr[Mesh::ARRAY_VERTEX];
		PoolVector<int> iarr = arr[Mesh::ARRAY_INDEX];
		if (varr.size() == 0 || iarr.size() == 0)
			continue;

		int from = vertices.size();
		vertices.append_array(varr);
		int rlen = iarr.size();
		PoolVector<int>::Read r = iarr.read();

		for (int j = 0; j < rlen; j += 3) {
			Vector<int> vi;
			vi.resize(3);
			vi[0] = r[j + 0] + from;
			vi[1] = r[j + 1] + from;
			vi[2] = r[j + 2] + from;

			add_polygon(vi);
		}
	}
}

void NavigationMesh::set_sample_partition_type(int p_value) {
	ERR_FAIL_COND(p_value >= SAMPLE_PARTITION_MAX);
	partition_type = static_cast<SamplePartitionType>(p_value);
}

int NavigationMesh::get_sample_partition_type() const {
	return static_cast<int>(partition_type);
}

void NavigationMesh::set_cell_size(float p_value) {
	cell_size = p_value;
}

float NavigationMesh::get_cell_size() const {
	return cell_size;
}

void NavigationMesh::set_cell_height(float p_value) {
	cell_height = p_value;
}

float NavigationMesh::get_cell_height() const {
	return cell_height;
}

void NavigationMesh::set_agent_height(float p_value) {
	agent_height = p_value;
}

float NavigationMesh::get_agent_height() const {
	return agent_height;
}

void NavigationMesh::set_agent_radius(float p_value) {
	agent_radius = p_value;
}

float NavigationMesh::get_agent_radius() {
	return agent_radius;
}

void NavigationMesh::set_agent_max_climb(float p_value) {
	agent_max_climb = p_value;
}

float NavigationMesh::get_agent_max_climb() const {
	return agent_max_climb;
}

void NavigationMesh::set_agent_max_slope(float p_value) {
	agent_max_slope = p_value;
}

float NavigationMesh::get_agent_max_slope() const {
	return agent_max_slope;
}

void NavigationMesh::set_region_min_size(float p_value) {
	region_min_size = p_value;
}

float NavigationMesh::get_region_min_size() const {
	return region_min_size;
}

void NavigationMesh::set_region_merge_size(float p_value) {
	region_merge_size = p_value;
}

float NavigationMesh::get_region_merge_size() const {
	return region_merge_size;
}

void NavigationMesh::set_edge_max_length(float p_value) {
	edge_max_length = p_value;
}

float NavigationMesh::get_edge_max_length() const {
	return edge_max_length;
}

void NavigationMesh::set_edge_max_error(float p_value) {
	edge_max_error = p_value;
}

float NavigationMesh::get_edge_max_error() const {
	return edge_max_error;
}

void NavigationMesh::set_verts_per_poly(float p_value) {
	verts_per_poly = p_value;
}

float NavigationMesh::get_verts_per_poly() const {
	return verts_per_poly;
}

void NavigationMesh::set_detail_sample_distance(float p_value) {
	detail_sample_distance = p_value;
}

float NavigationMesh::get_detail_sample_distance() const {
	return detail_sample_distance;
}

void NavigationMesh::set_detail_sample_max_error(float p_value) {
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

void NavigationMesh::set_vertices(const PoolVector<Vector3> &p_vertices) {

	vertices = p_vertices;
}

PoolVector<Vector3> NavigationMesh::get_vertices() const {

	return vertices;
}

void NavigationMesh::_set_polygons(const Array &p_array) {

	polygons.resize(p_array.size());
	for (int i = 0; i < p_array.size(); i++) {
		polygons[i].indices = p_array[i];
	}
}

Array NavigationMesh::_get_polygons() const {

	Array ret;
	ret.resize(polygons.size());
	for (int i = 0; i < ret.size(); i++) {
		ret[i] = polygons[i].indices;
	}

	return ret;
}

void NavigationMesh::add_polygon(const Vector<int> &p_polygon) {

	Polygon polygon;
	polygon.indices = p_polygon;
	polygons.push_back(polygon);
}
int NavigationMesh::get_polygon_count() const {

	return polygons.size();
}
Vector<int> NavigationMesh::get_polygon(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx, polygons.size(), Vector<int>());
	return polygons[p_idx].indices;
}
void NavigationMesh::clear_polygons() {

	polygons.clear();
}

Ref<Mesh> NavigationMesh::get_debug_mesh() {

	if (debug_mesh.is_valid())
		return debug_mesh;

	PoolVector<Vector3> vertices = get_vertices();
	PoolVector<Vector3>::Read vr = vertices.read();
	List<Face3> faces;
	for (int i = 0; i < get_polygon_count(); i++) {
		Vector<int> p = get_polygon(i);

		for (int j = 2; j < p.size(); j++) {
			Face3 f;
			f.vertex[0] = vr[p[0]];
			f.vertex[1] = vr[p[j - 1]];
			f.vertex[2] = vr[p[j]];

			faces.push_back(f);
		}
	}

	Map<_EdgeKey, bool> edge_map;
	PoolVector<Vector3> tmeshfaces;
	tmeshfaces.resize(faces.size() * 3);

	{
		PoolVector<Vector3>::Write tw = tmeshfaces.write();
		int tidx = 0;

		for (List<Face3>::Element *E = faces.front(); E; E = E->next()) {

			const Face3 &f = E->get();

			for (int j = 0; j < 3; j++) {

				tw[tidx++] = f.vertex[j];
				_EdgeKey ek;
				ek.from = f.vertex[j].snapped(Vector3(CMP_EPSILON, CMP_EPSILON, CMP_EPSILON));
				ek.to = f.vertex[(j + 1) % 3].snapped(Vector3(CMP_EPSILON, CMP_EPSILON, CMP_EPSILON));
				if (ek.from < ek.to)
					SWAP(ek.from, ek.to);

				Map<_EdgeKey, bool>::Element *E = edge_map.find(ek);

				if (E) {

					E->get() = false;

				} else {

					edge_map[ek] = true;
				}
			}
		}
	}
	List<Vector3> lines;

	for (Map<_EdgeKey, bool>::Element *E = edge_map.front(); E; E = E->next()) {

		if (E->get()) {
			lines.push_back(E->key().from);
			lines.push_back(E->key().to);
		}
	}

	PoolVector<Vector3> varr;
	varr.resize(lines.size());
	{
		PoolVector<Vector3>::Write w = varr.write();
		int idx = 0;
		for (List<Vector3>::Element *E = lines.front(); E; E = E->next()) {
			w[idx++] = E->get();
		}
	}

	debug_mesh = Ref<ArrayMesh>(memnew(ArrayMesh));

	Array arr;
	arr.resize(Mesh::ARRAY_MAX);
	arr[Mesh::ARRAY_VERTEX] = varr;

	debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, arr);

	return debug_mesh;
}

void NavigationMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sample_partition_type", "sample_partition_type"), &NavigationMesh::set_sample_partition_type);
	ClassDB::bind_method(D_METHOD("get_sample_partition_type"), &NavigationMesh::get_sample_partition_type);

	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &NavigationMesh::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &NavigationMesh::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_cell_height", "cell_height"), &NavigationMesh::set_cell_height);
	ClassDB::bind_method(D_METHOD("get_cell_height"), &NavigationMesh::get_cell_height);

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

	ClassDB::bind_method(D_METHOD("set_verts_per_poly", "verts_per_poly"), &NavigationMesh::set_verts_per_poly);
	ClassDB::bind_method(D_METHOD("get_verts_per_poly"), &NavigationMesh::get_verts_per_poly);

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

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationMesh::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationMesh::get_vertices);

	ClassDB::bind_method(D_METHOD("add_polygon", "polygon"), &NavigationMesh::add_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon_count"), &NavigationMesh::get_polygon_count);
	ClassDB::bind_method(D_METHOD("get_polygon", "idx"), &NavigationMesh::get_polygon);
	ClassDB::bind_method(D_METHOD("clear_polygons"), &NavigationMesh::clear_polygons);

	ClassDB::bind_method(D_METHOD("create_from_mesh", "mesh"), &NavigationMesh::create_from_mesh);

	ClassDB::bind_method(D_METHOD("_set_polygons", "polygons"), &NavigationMesh::_set_polygons);
	ClassDB::bind_method(D_METHOD("_get_polygons"), &NavigationMesh::_get_polygons);

	BIND_CONSTANT(SAMPLE_PARTITION_WATERSHED);
	BIND_CONSTANT(SAMPLE_PARTITION_MONOTONE);
	BIND_CONSTANT(SAMPLE_PARTITION_LAYERS);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR3_ARRAY, "vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "polygons", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_polygons", "_get_polygons");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "sample_partition_type/sample_partition_type", PROPERTY_HINT_ENUM, "Watershed,Monotone,Layers"), "set_sample_partition_type", "get_sample_partition_type");

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "cell/size", PROPERTY_HINT_RANGE, "0.1,1.0,0.01"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "cell/height", PROPERTY_HINT_RANGE, "0.1,1.0,0.01"), "set_cell_height", "get_cell_height");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "agent/height", PROPERTY_HINT_RANGE, "0.1,5.0,0.01"), "set_agent_height", "get_agent_height");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "agent/radius", PROPERTY_HINT_RANGE, "0.1,5.0,0.01"), "set_agent_radius", "get_agent_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "agent/max_climb", PROPERTY_HINT_RANGE, "0.1,5.0,0.01"), "set_agent_max_climb", "get_agent_max_climb");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "agent/max_slope", PROPERTY_HINT_RANGE, "0.0,90.0,0.1"), "set_agent_max_slope", "get_agent_max_slope");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "region/min_size", PROPERTY_HINT_RANGE, "0.0,150.0,0.01"), "set_region_min_size", "get_region_min_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "region/merge_size", PROPERTY_HINT_RANGE, "0.0,150.0,0.01"), "set_region_merge_size", "get_region_merge_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge/max_length", PROPERTY_HINT_RANGE, "0.0,50.0,0.01"), "set_edge_max_length", "get_edge_max_length");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge/max_error", PROPERTY_HINT_RANGE, "0.1,3.0,0.01"), "set_edge_max_error", "get_edge_max_error");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "polygon/verts_per_poly", PROPERTY_HINT_RANGE, "3.0,12.0,1.0"), "set_verts_per_poly", "get_verts_per_poly");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detail/sample_distance", PROPERTY_HINT_RANGE, "0.0,16.0,0.01"), "set_detail_sample_distance", "get_detail_sample_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "detail/sample_max_error", PROPERTY_HINT_RANGE, "0.0,16.0,0.01"), "set_detail_sample_max_error", "get_detail_sample_max_error");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter/low_hanging_obstacles"), "set_filter_low_hanging_obstacles", "get_filter_low_hanging_obstacles");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter/ledge_spans"), "set_filter_ledge_spans", "get_filter_ledge_spans");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter/filter_walkable_low_height_spans"), "set_filter_walkable_low_height_spans", "get_filter_walkable_low_height_spans");
}

NavigationMesh::NavigationMesh() {
	cell_size = 0.3f;
	cell_height = 0.2f;
	agent_height = 2.0f;
	agent_radius = 0.6f;
	agent_max_climb = 0.9f;
	agent_max_slope = 45.0f;
	region_min_size = 8.0f;
	region_merge_size = 20.0f;
	edge_max_length = 12.0f;
	edge_max_error = 1.3f;
	verts_per_poly = 6.0f;
	detail_sample_distance = 6.0f;
	detail_sample_max_error = 1.0f;

	partition_type = SAMPLE_PARTITION_WATERSHED;

	filter_low_hanging_obstacles = false;
	filter_ledge_spans = false;
	filter_walkable_low_height_spans = false;
}

void NavigationMeshInstance::set_enabled(bool p_enabled) {

	if (enabled == p_enabled)
		return;
	enabled = p_enabled;

	if (!is_inside_tree())
		return;

	if (!enabled) {

		if (nav_id != -1) {
			navigation->navmesh_remove(nav_id);
			nav_id = -1;
		}
	} else {

		if (navigation) {

			if (navmesh.is_valid()) {

				nav_id = navigation->navmesh_create(navmesh, get_relative_transform(navigation), this);
			}
		}
	}

	if (debug_view) {
		MeshInstance *dm = Object::cast_to<MeshInstance>(debug_view);
		if (is_enabled()) {
			dm->set_material_override(get_tree()->get_debug_navigation_material());
		} else {
			dm->set_material_override(get_tree()->get_debug_navigation_disabled_material());
		}
	}

	update_gizmo();
}

bool NavigationMeshInstance::is_enabled() const {

	return enabled;
}

/////////////////////////////

void NavigationMeshInstance::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {

			Spatial *c = this;
			while (c) {

				navigation = Object::cast_to<Navigation>(c);
				if (navigation) {

					if (enabled && navmesh.is_valid()) {

						nav_id = navigation->navmesh_create(navmesh, get_relative_transform(navigation), this);
					}
					break;
				}

				c = c->get_parent_spatial();
			}

			if (navmesh.is_valid() && get_tree()->is_debugging_navigation_hint()) {

				MeshInstance *dm = memnew(MeshInstance);
				dm->set_mesh(navmesh->get_debug_mesh());
				if (is_enabled()) {
					dm->set_material_override(get_tree()->get_debug_navigation_material());
				} else {
					dm->set_material_override(get_tree()->get_debug_navigation_disabled_material());
				}
				add_child(dm);
				debug_view = dm;
			}

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {

			if (navigation && nav_id != -1) {
				navigation->navmesh_set_transform(nav_id, get_relative_transform(navigation));
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (navigation) {

				if (nav_id != -1) {
					navigation->navmesh_remove(nav_id);
					nav_id = -1;
				}
			}

			if (debug_view) {
				debug_view->queue_delete();
				debug_view = NULL;
			}
			navigation = NULL;
		} break;
	}
}

void NavigationMeshInstance::set_navigation_mesh(const Ref<NavigationMesh> &p_navmesh) {

	if (p_navmesh == navmesh)
		return;

	if (navigation && nav_id != -1) {
		navigation->navmesh_remove(nav_id);
		nav_id = -1;
	}
	navmesh = p_navmesh;

	if (navigation && navmesh.is_valid() && enabled) {
		nav_id = navigation->navmesh_create(navmesh, get_relative_transform(navigation), this);
	}

	if (debug_view && navmesh.is_valid()) {
		Object::cast_to<MeshInstance>(debug_view)->set_mesh(navmesh->get_debug_mesh());
	}

	update_gizmo();
	update_configuration_warning();
}

Ref<NavigationMesh> NavigationMeshInstance::get_navigation_mesh() const {

	return navmesh;
}

String NavigationMeshInstance::get_configuration_warning() const {

	if (!is_visible_in_tree() || !is_inside_tree())
		return String();

	if (!navmesh.is_valid()) {
		return TTR("A NavigationMesh resource must be set or created for this node to work.");
	}
	const Spatial *c = this;
	while (c) {

		if (Object::cast_to<Navigation>(c))
			return String();

		c = Object::cast_to<Spatial>(c->get_parent());
	}

	return TTR("NavigationMeshInstance must be a child or grandchild to a Navigation node. It only provides navigation data.");
}

void NavigationMeshInstance::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_navigation_mesh", "navmesh"), &NavigationMeshInstance::set_navigation_mesh);
	ClassDB::bind_method(D_METHOD("get_navigation_mesh"), &NavigationMeshInstance::get_navigation_mesh);

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &NavigationMeshInstance::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &NavigationMeshInstance::is_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navmesh", PROPERTY_HINT_RESOURCE_TYPE, "NavigationMesh"), "set_navigation_mesh", "get_navigation_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
}

NavigationMeshInstance::NavigationMeshInstance() {

	debug_view = NULL;
	navigation = NULL;
	nav_id = -1;
	enabled = true;
	set_notify_transform(true);
}
