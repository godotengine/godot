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

#include "nav_map_3d.h"

NavArea3D::NavArea3D() :
		sync_dirty_request_list_element(this) {
	iteration = 0;
}

NavArea3D::~NavArea3D() {
	cancel_sync_request();

	iteration = 0;
}


void NavArea3D::set_id(uint16_t p_id) {
	if (id == p_id) {
		return;
	}

	id = p_id;
	print_line("NavArea3D::set_id");
	// iteration_dirty = true;

	// request_sync();
}

void NavArea3D::set_map(NavMap3D *p_map) {
	if (map == p_map) {
		return;
	}

	cancel_sync_request();

	if (map) {
		map->remove_area(this);
	}

	map = p_map;
	iteration_dirty = true;

	if (map) {
		map->add_area(this);
		request_sync();
	}
}

void NavArea3D::set_position(const Vector3 p_position) {
	if (position == p_position) {
		return;
	}

	position = p_position;
}

void NavArea3D::set_shape_type(NavigationMeshSourceGeometryData3D::ProjectedArea::ShapeType p_shape_type) {
	if (shape_type == p_shape_type) {
		return;
	}
	shape_type = p_shape_type;
}

void NavArea3D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	enabled = p_enabled;
}

void NavArea3D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}
	navigation_layers = p_navigation_layers;
	iteration_dirty = true;
	print_line("NavArea3D::set_navigation_layers");

	request_sync();
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

bool NavArea3D::sync() {
	bool requires_map_update = false;
	if (!map) {
		return requires_map_update;
	}

	if (iteration_dirty) {
		print_line("NavArea3D::sync");
		// build iteration:
		iteration_dirty = false;
		RWLockWrite write_lock(iteration_rwlock);
		iteration = get_navigation_layers();
		requires_map_update = true;
	}

	return requires_map_update;
}


uint32_t NavArea3D::get_iteration() const {
	RWLockRead read_lock(iteration_rwlock);
	return iteration;
}

void NavArea3D::set_bake_priority(int p_priority) {
	if (priority == p_priority) {
		return;
	}
	priority = p_priority;
}

void NavArea3D::set_size(const Vector3 p_size) {
	if (xyz == p_size) {
		return;
	}
	xyz = p_size;
}

void NavArea3D::set_elevation(const real_t p_elevation) {
	if (xyz.z == p_elevation) {
		return;
	}

	xyz.z = p_elevation;
}

void NavArea3D::set_height(const real_t p_height) {
	if (xyz.y == p_height) {
		return;
	}

	xyz.y = p_height;
}

void NavArea3D::set_radius(real_t p_radius) {
	if (xyz.x == p_radius) {
		return;
	}
	xyz.x = p_radius;
}

void NavArea3D::set_vertices(const Vector<Vector3> &p_vertices) {
	if (vertices == p_vertices) {
		return;
	}
	vertices = p_vertices;
}
