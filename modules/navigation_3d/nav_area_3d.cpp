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

NavArea3D::NavArea3D() {
}

NavArea3D::~NavArea3D() {
}

void NavArea3D::set_map(NavMap3D *p_map) {
	if (map == p_map) {
		return;
	}

	if (map) {
		map->remove_area(this);
	}

	map = p_map;

	if (map) {
		map->add_area(this);
	}
}

void NavArea3D::set_position(const Vector3 p_position) {
	if (position == p_position) {
		return;
	}

	position = p_position;
}

void NavArea3D::set_enabled(bool p_enabled) {
	if (enabled == p_enabled) {
		return;
	}
	enabled = p_enabled;
}

void NavArea3D::set_height(const real_t p_height) {
	if (height == p_height) {
		return;
	}

	height = p_height;
}

void NavArea3D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}
	navigation_layers = p_navigation_layers;
}

void NavArea3D::set_priority(int p_priority) {
	if (priority == p_priority) {
		return;
	}
	priority = p_priority;
}

void NavAreaBox3D::set_size(const Vector3 p_size) {
	if (size == p_size) {
		return;
	}
	size = p_size;
}

void NavAreaCylinder3D::set_radius(real_t p_radius) {
	if (radius == p_radius) {
		return;
	}
	radius = p_radius;
}

void NavAreaPolygon3D::set_vertices(const Vector<Vector3> &p_vertices) {
	if (vertices == p_vertices) {
		return;
	}
	vertices = p_vertices;
}
