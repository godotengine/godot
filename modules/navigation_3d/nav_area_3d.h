/**************************************************************************/
/*  nav_area_3d.h                                                         */
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

#pragma once

#include "nav_rid_3d.h"

#include "core/math/vector3.h"
#include "core/templates/self_list.h"

class NavMap3D;

class NavArea3D : public NavRid3D {
	NavMap3D *map = nullptr;
	Vector3 position;

	bool enabled = false;
	real_t height = 0.0;
	uint32_t navigation_layers = 1;
	int priority = 0;

public:
	NavArea3D();
	~NavArea3D();

	void set_map(NavMap3D *p_map);
	NavMap3D *get_map() const { return map; }

	void set_position(const Vector3 p_position);
	const Vector3 &get_position() const { return position; }

	void set_enabled(bool p_enabled);
	bool get_enabled() const { return enabled; }

	void set_height(const real_t p_height);
	real_t get_height() const { return height; }

	void set_navigation_layers(uint32_t p_layers);
	uint32_t get_navigation_layers() const { return navigation_layers; }

	void set_priority(int p_priority);
	int get_prioset_priority() const { return priority; }
};

class NavAreaBox3D : public NavArea3D {
	Vector3 size = Vector3(1.0, 1.0, 1.0);

public:
	void set_size(const Vector3 p_size);
	const Vector3 &get_size() const { return size; }
};

class NavAreaCylinder3D : public NavArea3D {
	real_t radius = 1.0;

public:
	void set_radius(real_t p_radius);
	real_t get_radius() const { return radius; }
};

class NavAreaPolygon3D : public NavArea3D {
	Vector<Vector3> vertices;
	// FIXME:
	// bool vertices_are_clockwise = true;
	// bool vertices_are_valid = true;

public:
	void set_vertices(const Vector<Vector3> &p_vertices);
	const Vector<Vector3> &get_vertices() const { return vertices; }
};
