/*************************************************************************/
/*  navigation_mesh.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
				ek.from = f.vertex[j].snapped(CMP_EPSILON);
				ek.to = f.vertex[(j + 1) % 3].snapped(CMP_EPSILON);
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

	debug_mesh = Ref<Mesh>(memnew(Mesh));

	Array arr;
	arr.resize(Mesh::ARRAY_MAX);
	arr[Mesh::ARRAY_VERTEX] = varr;

	debug_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, arr);

	return debug_mesh;
}

void NavigationMesh::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_vertices", "vertices"), &NavigationMesh::set_vertices);
	ClassDB::bind_method(D_METHOD("get_vertices"), &NavigationMesh::get_vertices);

	ClassDB::bind_method(D_METHOD("add_polygon", "polygon"), &NavigationMesh::add_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon_count"), &NavigationMesh::get_polygon_count);
	ClassDB::bind_method(D_METHOD("get_polygon", "idx"), &NavigationMesh::get_polygon);
	ClassDB::bind_method(D_METHOD("clear_polygons"), &NavigationMesh::clear_polygons);

	ClassDB::bind_method(D_METHOD("_set_polygons", "polygons"), &NavigationMesh::_set_polygons);
	ClassDB::bind_method(D_METHOD("_get_polygons"), &NavigationMesh::_get_polygons);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR3_ARRAY, "vertices", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "set_vertices", "get_vertices");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "polygons", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_polygons", "_get_polygons");
}

NavigationMesh::NavigationMesh() {
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
		MeshInstance *dm = debug_view->cast_to<MeshInstance>();
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

				navigation = c->cast_to<Navigation>();
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
		debug_view->cast_to<MeshInstance>()->set_mesh(navmesh->get_debug_mesh());
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

		if (c->cast_to<Navigation>())
			return String();

		c = c->get_parent()->cast_to<Spatial>();
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
