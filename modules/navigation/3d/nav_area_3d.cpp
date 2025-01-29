/**************************************************************************/
/*  nav_area_3d.cpp                                                       */
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

#include "nav_area_3d.h"

#include "../nav_map.h"

#include "core/math/geometry_2d.h"

void NavArea3D::set_map(NavMap *p_map) {
	if (map == p_map) {
		return;
	}

	cancel_sync_request();

	if (map) {
		map->remove_area(this);
	}

	map = p_map;
	area_dirty = true;

	if (map) {
		map->add_area(this);
		request_sync();
	}
}

void NavArea3D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}

	enabled = p_enabled;

	area_dirty = true;
	request_sync();
}

void NavArea3D::set_height(float p_height) {
	if (height == p_height) {
		return;
	}

	height = p_height;
	size.y = height;

	area_dirty = true;
	request_sync();
}

void NavArea3D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}

	navigation_layers = p_navigation_layers;

	area_dirty = true;
	request_sync();
}
void NavArea3D::set_priority(int p_priority) {
	if (priority == p_priority) {
		return;
	}

	priority = p_priority;

	area_dirty = true;
	request_sync();
}

void NavArea3D::set_shape_type(NavigationServer3D::AreaShapeType3D p_shape_type) {
	if (shape_type == p_shape_type) {
		return;
	}

	shape_type = p_shape_type;

	area_dirty = true;
	request_sync();
}

void NavArea3D::set_position(Vector3 p_position) {
	if (position == p_position) {
		return;
	}

	position = p_position;

	area_dirty = true;
	request_sync();
}

void NavArea3D::set_size(Vector3 p_size) {
	if (size == p_size) {
		return;
	}

	size = p_size;
	set_height(p_size.y);

	area_dirty = true;
	request_sync();
}

void NavArea3D::set_radius(float p_radius) {
	if (radius == p_radius) {
		return;
	}

	radius = p_radius;

	area_dirty = true;
	request_sync();
}

void NavArea3D::set_vertices(const Vector<Vector3> &p_vertices) {
	vertices.resize(p_vertices.size());

	const Vector3 *r = p_vertices.ptr();
	Vector3 *w = vertices.ptr();

	for (uint32_t i = 0; i < vertices.size(); i++) {
		w[i] = r[i];
	}

	area_dirty = true;
	request_sync();
}

bool NavArea3D::has_point(const Vector3 &p_point) {
	switch (shape_type) {
		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_BOX: {
			return get_bounds().has_point(p_point);
		} break;

		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_CYLINDER: {
			if (p_point.y < position.y || p_point.y > position.y + height) {
				return false;
			}
			return Vector3(position.x, p_point.y, position.z).distance_to(p_point) <= radius;
		} break;

		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_POLYGON: {
			if (p_point.y < position.y || p_point.y > (position.y + height)) {
				return false;
			}
			if (vertices.size() < 3) {
				return false;
			}

			const Vector3 *vertices_ptr = vertices.ptr();
			Vector<Vector2> polygon;
			polygon.resize(vertices.size());
			Vector2 *polygon_ptrw = polygon.ptrw();
			for (uint32_t i = 0; i < vertices.size(); i++) {
				polygon_ptrw[i] = Vector2(position.x + vertices_ptr[i].x, position.z + vertices_ptr[i].z);
			}
			return Geometry2D::is_point_in_polygon(Vector2(p_point.x, p_point.z), polygon);
		} break;

		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_NONE: {
			return false;
		} break;

		default:
			ERR_PRINT_ONCE("Invalid NavigationServer3D::AreaShapeType3D.");
	}

	return false;
}

void NavArea3D::_update_bounds() {
	AABB new_bounds;

	switch (shape_type) {
		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_BOX: {
			if (size.x > 0.0 && size.z > 0.0) {
				LocalVector<Vector3> b_vertices;
				b_vertices.resize(4);
				b_vertices[0] = Vector3(-size.x * 0.5, 0.0, -size.z * 0.5);
				b_vertices[1] = Vector3(size.x * 0.5, 0.0, -size.z * 0.5);
				b_vertices[2] = Vector3(size.x * 0.5, 0.0, size.z * 0.5);
				b_vertices[3] = Vector3(-size.x * 0.5, 0.0, size.z * 0.5);
				new_bounds.position = position + b_vertices[0];
				for (const Vector3 &vertex : b_vertices) {
					new_bounds.expand_to(position + vertex);
				}
				const Vector3 height_offset = Vector3(0.0, height, 0.0);
				for (const Vector3 &vertex : b_vertices) {
					new_bounds.expand_to(position + vertex + height_offset);
				}
			}
		} break;

		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_CYLINDER: {
			if (radius > 0.0) {
				LocalVector<Vector3> c_vertices;
				c_vertices.resize(4);
				c_vertices[0] = Vector3(-radius, 0.0, -radius);
				c_vertices[1] = Vector3(radius, 0.0, -radius);
				c_vertices[2] = Vector3(radius, 0.0, radius);
				c_vertices[3] = Vector3(-radius, 0.0, radius);

				new_bounds.position = position + c_vertices[0];
				for (const Vector3 &vertex : c_vertices) {
					new_bounds.expand_to(position + vertex);
				}
				const Vector3 height_offset = Vector3(0.0, height, 0.0);
				for (const Vector3 &vertex : c_vertices) {
					new_bounds.expand_to(position + vertex + height_offset);
				}
			}
		} break;

		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_POLYGON: {
			if (!vertices.is_empty()) {
				new_bounds.position = position + vertices[0];

				for (const Vector3 &vertex : vertices) {
					new_bounds.expand_to(position + vertex);
				}
				const Vector3 height_offset = Vector3(0.0, height, 0.0);
				for (const Vector3 &vertex : vertices) {
					new_bounds.expand_to(position + vertex + height_offset);
				}
			}
		} break;

		case NavigationServer3D::AreaShapeType3D::AREA_SHAPE_NONE: {
		} break;

		default:
			ERR_PRINT_ONCE("Invalid NavigationServer3D::AreaShapeType3D.");
	}

	bounds = new_bounds;
}

bool NavArea3D::sync() {
	if (!area_dirty) {
		return false;
	}
	RWLockWrite write_lock(area_rwlock);
	bounds = AABB();
	area_dirty = false;

	_update_bounds();

	return true;
}

void NavArea3D::request_sync() {
	if (map && !sync_dirty_request_list_element.in_list()) {
		map->add_area_sync_dirty_request(&sync_dirty_request_list_element);
	}
}

void NavArea3D::cancel_sync_request() {
	if (map && sync_dirty_request_list_element.in_list()) {
		map->remove_area_sync_dirty_request(&sync_dirty_request_list_element);
	}
}

NavArea3D::NavArea3D() :
		sync_dirty_request_list_element(this) {
}

NavArea3D::~NavArea3D() {
	cancel_sync_request();
}
