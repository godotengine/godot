/**************************************************************************/
/*  nav_region.cpp                                                        */
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

#include "nav_region.h"

#include "nav_map.h"

void NavRegion::set_map(NavMap *p_map) {
	if (map == p_map) {
		return;
	}

	if (map) {
		map->remove_region(this);
	}

	map = p_map;
	polygons_dirty = true;

	connections.clear();

	if (map) {
		map->add_region(this);
	}
}

void NavRegion::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	enabled = p_enabled;

	// TODO: This should not require a full rebuild as the region has not really changed.
	polygons_dirty = true;
};

void NavRegion::set_use_edge_connections(bool p_enabled) {
	if (use_edge_connections != p_enabled) {
		use_edge_connections = p_enabled;
		polygons_dirty = true;
	}
}

void NavRegion::set_transform(Transform3D p_transform) {
	if (transform == p_transform) {
		return;
	}
	transform = p_transform;
	polygons_dirty = true;
}

void NavRegion::set_mesh(Ref<NavigationMesh> p_mesh) {
	mesh = p_mesh;
	polygons_dirty = true;
}

int NavRegion::get_connections_count() const {
	if (!map) {
		return 0;
	}
	return connections.size();
}

Vector3 NavRegion::get_connection_pathway_start(int p_connection_id) const {
	ERR_FAIL_NULL_V(map, Vector3());
	ERR_FAIL_INDEX_V(p_connection_id, connections.size(), Vector3());
	return connections[p_connection_id].pathway_start;
}

Vector3 NavRegion::get_connection_pathway_end(int p_connection_id) const {
	ERR_FAIL_NULL_V(map, Vector3());
	ERR_FAIL_INDEX_V(p_connection_id, connections.size(), Vector3());
	return connections[p_connection_id].pathway_end;
}

bool NavRegion::sync() {
	bool something_changed = polygons_dirty /* || something_dirty? */;

	update_polygons();

	return something_changed;
}

void NavRegion::update_polygons() {
	if (!polygons_dirty) {
		return;
	}
	polygons.clear();
	polygons_dirty = false;

	if (map == nullptr) {
		return;
	}

	if (mesh.is_null()) {
		return;
	}

#ifdef DEBUG_ENABLED
	if (!Math::is_equal_approx(double(map->get_cell_size()), double(mesh->get_cell_size()))) {
		ERR_PRINT_ONCE(vformat("Navigation map synchronization error. Attempted to update a navigation region with a navigation mesh that uses a `cell_size` of %s while assigned to a navigation map set to a `cell_size` of %s. The cell size for navigation maps can be changed by using the NavigationServer map_set_cell_size() function. The cell size for default navigation maps can also be changed in the ProjectSettings.", double(mesh->get_cell_size()), double(map->get_cell_size())));
	}

	if (!Math::is_equal_approx(double(map->get_cell_height()), double(mesh->get_cell_height()))) {
		ERR_PRINT_ONCE(vformat("Navigation map synchronization error. Attempted to update a navigation region with a navigation mesh that uses a `cell_height` of %s while assigned to a navigation map set to a `cell_height` of %s. The cell height for navigation maps can be changed by using the NavigationServer map_set_cell_height() function. The cell height for default navigation maps can also be changed in the ProjectSettings.", double(mesh->get_cell_height()), double(map->get_cell_height())));
	}

	if (map && Math::rad_to_deg(map->get_up().angle_to(transform.basis.get_column(1))) >= 90.0f) {
		ERR_PRINT_ONCE("Navigation map synchronization error. Attempted to update a navigation region transform rotated 90 degrees or more away from the current navigation map UP orientation.");
	}
#endif // DEBUG_ENABLED

	Vector<Vector3> vertices = mesh->get_vertices();
	int len = vertices.size();
	if (len == 0) {
		return;
	}

	const Vector3 *vertices_r = vertices.ptr();

	polygons.resize(mesh->get_polygon_count());

	// Build
	for (size_t i(0); i < polygons.size(); i++) {
		gd::Polygon &p = polygons[i];
		p.owner = this;

		Vector<int> mesh_poly = mesh->get_polygon(i);
		const int *indices = mesh_poly.ptr();
		bool valid(true);
		p.points.resize(mesh_poly.size());
		p.edges.resize(mesh_poly.size());

		Vector3 center;
		real_t sum(0);

		for (int j(0); j < mesh_poly.size(); j++) {
			int idx = indices[j];
			if (idx < 0 || idx >= len) {
				valid = false;
				break;
			}

			Vector3 point_position = transform.xform(vertices_r[idx]);
			p.points[j].pos = point_position;
			p.points[j].key = map->get_point_key(point_position);

			center += point_position; // Composing the center of the polygon

			if (j >= 2) {
				Vector3 epa = transform.xform(vertices_r[indices[j - 2]]);
				Vector3 epb = transform.xform(vertices_r[indices[j - 1]]);

				sum += map->get_up().dot((epb - epa).cross(point_position - epa));
			}
		}

		if (!valid) {
			ERR_BREAK_MSG(!valid, "The navigation mesh set in this region is not valid!");
		}

		p.clockwise = sum > 0;
		if (mesh_poly.size() != 0) {
			p.center = center / real_t(mesh_poly.size());
		}
	}
}
