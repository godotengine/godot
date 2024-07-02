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

#ifdef DEBUG_ENABLED
	if (map && Math::rad_to_deg(map->get_up().angle_to(transform.basis.get_column(1))) >= 90.0f) {
		ERR_PRINT_ONCE("Attempted to update a navigation region transform rotated 90 degrees or more away from the current navigation map UP orientation.");
	}
#endif // DEBUG_ENABLED
}

void NavRegion::set_navigation_mesh(Ref<NavigationMesh> p_navigation_mesh) {
#ifdef DEBUG_ENABLED
	if (map && p_navigation_mesh.is_valid() && !Math::is_equal_approx(double(map->get_cell_size()), double(p_navigation_mesh->get_cell_size()))) {
		ERR_PRINT_ONCE(vformat("Attempted to update a navigation region with a navigation mesh that uses a `cell_size` of %s while assigned to a navigation map set to a `cell_size` of %s. The cell size for navigation maps can be changed by using the NavigationServer map_set_cell_size() function. The cell size for default navigation maps can also be changed in the ProjectSettings.", double(p_navigation_mesh->get_cell_size()), double(map->get_cell_size())));
	}

	if (map && p_navigation_mesh.is_valid() && !Math::is_equal_approx(double(map->get_cell_height()), double(p_navigation_mesh->get_cell_height()))) {
		ERR_PRINT_ONCE(vformat("Attempted to update a navigation region with a navigation mesh that uses a `cell_height` of %s while assigned to a navigation map set to a `cell_height` of %s. The cell height for navigation maps can be changed by using the NavigationServer map_set_cell_height() function. The cell height for default navigation maps can also be changed in the ProjectSettings.", double(p_navigation_mesh->get_cell_height()), double(map->get_cell_height())));
	}
#endif // DEBUG_ENABLED

	RWLockWrite write_lock(navmesh_rwlock);

	pending_navmesh_vertices.clear();
	pending_navmesh_polygons.clear();

	if (p_navigation_mesh.is_valid()) {
		p_navigation_mesh->get_data(pending_navmesh_vertices, pending_navmesh_polygons);
	}

	polygons_dirty = true;
}

Vector3 NavRegion::get_random_point(uint32_t p_navigation_layers, bool p_uniformly) const {
	if (!get_enabled()) {
		return Vector3();
	}

	const LocalVector<gd::Polygon> &region_polygons = get_polygons();

	if (region_polygons.is_empty()) {
		return Vector3();
	}

	if (p_uniformly) {
		real_t accumulated_area = 0;
		RBMap<real_t, uint32_t> region_area_map;

		for (uint32_t rp_index = 0; rp_index < region_polygons.size(); rp_index++) {
			const gd::Polygon &region_polygon = region_polygons[rp_index];
			real_t polyon_area = region_polygon.surface_area;

			if (polyon_area == 0.0) {
				continue;
			}
			region_area_map[accumulated_area] = rp_index;
			accumulated_area += polyon_area;
		}
		if (region_area_map.is_empty() || accumulated_area == 0) {
			// All polygons have no real surface / no area.
			return Vector3();
		}

		real_t region_area_map_pos = Math::random(real_t(0), accumulated_area);

		RBMap<real_t, uint32_t>::Iterator region_E = region_area_map.find_closest(region_area_map_pos);
		ERR_FAIL_COND_V(!region_E, Vector3());
		uint32_t rrp_polygon_index = region_E->value;
		ERR_FAIL_UNSIGNED_INDEX_V(rrp_polygon_index, region_polygons.size(), Vector3());

		const gd::Polygon &rr_polygon = region_polygons[rrp_polygon_index];

		real_t accumulated_polygon_area = 0;
		RBMap<real_t, uint32_t> polygon_area_map;

		for (uint32_t rpp_index = 2; rpp_index < rr_polygon.points.size(); rpp_index++) {
			real_t face_area = Face3(rr_polygon.points[0].pos, rr_polygon.points[rpp_index - 1].pos, rr_polygon.points[rpp_index].pos).get_area();

			if (face_area == 0.0) {
				continue;
			}
			polygon_area_map[accumulated_polygon_area] = rpp_index;
			accumulated_polygon_area += face_area;
		}
		if (polygon_area_map.is_empty() || accumulated_polygon_area == 0) {
			// All faces have no real surface / no area.
			return Vector3();
		}

		real_t polygon_area_map_pos = Math::random(real_t(0), accumulated_polygon_area);

		RBMap<real_t, uint32_t>::Iterator polygon_E = polygon_area_map.find_closest(polygon_area_map_pos);
		ERR_FAIL_COND_V(!polygon_E, Vector3());
		uint32_t rrp_face_index = polygon_E->value;
		ERR_FAIL_UNSIGNED_INDEX_V(rrp_face_index, rr_polygon.points.size(), Vector3());

		const Face3 face(rr_polygon.points[0].pos, rr_polygon.points[rrp_face_index - 1].pos, rr_polygon.points[rrp_face_index].pos);

		Vector3 face_random_position = face.get_random_point_inside();
		return face_random_position;

	} else {
		uint32_t rrp_polygon_index = Math::random(int(0), region_polygons.size() - 1);

		const gd::Polygon &rr_polygon = region_polygons[rrp_polygon_index];

		uint32_t rrp_face_index = Math::random(int(2), rr_polygon.points.size() - 1);

		const Face3 face(rr_polygon.points[0].pos, rr_polygon.points[rrp_face_index - 1].pos, rr_polygon.points[rrp_face_index].pos);

		Vector3 face_random_position = face.get_random_point_inside();
		return face_random_position;
	}
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
	surface_area = 0.0;
	polygons_dirty = false;

	if (map == nullptr) {
		return;
	}

	RWLockRead read_lock(navmesh_rwlock);

	if (pending_navmesh_vertices.is_empty() || pending_navmesh_polygons.is_empty()) {
		return;
	}

	int len = pending_navmesh_vertices.size();
	if (len == 0) {
		return;
	}

	const Vector3 *vertices_r = pending_navmesh_vertices.ptr();

	polygons.resize(pending_navmesh_polygons.size());

	real_t _new_region_surface_area = 0.0;

	// Build
	int navigation_mesh_polygon_index = 0;
	for (gd::Polygon &polygon : polygons) {
		polygon.owner = this;
		polygon.surface_area = 0.0;

		Vector<int> navigation_mesh_polygon = pending_navmesh_polygons[navigation_mesh_polygon_index];
		navigation_mesh_polygon_index += 1;

		int navigation_mesh_polygon_size = navigation_mesh_polygon.size();
		if (navigation_mesh_polygon_size < 3) {
			continue;
		}

		const int *indices = navigation_mesh_polygon.ptr();
		bool valid(true);

		polygon.points.resize(navigation_mesh_polygon_size);
		polygon.edges.resize(navigation_mesh_polygon_size);

		real_t _new_polygon_surface_area = 0.0;

		for (int j(2); j < navigation_mesh_polygon_size; j++) {
			const Face3 face = Face3(
					transform.xform(vertices_r[indices[0]]),
					transform.xform(vertices_r[indices[j - 1]]),
					transform.xform(vertices_r[indices[j]]));

			_new_polygon_surface_area += face.get_area();
		}

		polygon.surface_area = _new_polygon_surface_area;
		_new_region_surface_area += _new_polygon_surface_area;

		for (int j(0); j < navigation_mesh_polygon_size; j++) {
			int idx = indices[j];
			if (idx < 0 || idx >= len) {
				valid = false;
				break;
			}

			Vector3 point_position = transform.xform(vertices_r[idx]);
			polygon.points[j].pos = point_position;
			polygon.points[j].key = map->get_point_key(point_position);
		}

		if (!valid) {
			ERR_BREAK_MSG(!valid, "The navigation mesh set in this region is not valid!");
		}
	}

	surface_area = _new_region_surface_area;
}
