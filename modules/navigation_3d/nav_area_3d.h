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
#include "scene/resources/3d/navigation_mesh_source_geometry_data_3d.h"

class NavMap3D;

// For parsing navigation mesh areas created via navigation server.
// Lives inside a NavMap.
// Variant of NavigationMeshSourceGeometryData3D::ProjectedArea.
class NavArea3D : public NavRid3D {
	NavMap3D *map = nullptr;
	Vector3 position;
	// Channel-packing Legend:
	// For AreaBox only: xyz means `size`.
	// For AreaCylinder: x means `radius`, y means `height`.
	// For AreaPolygon: z means `elevation`, y means `height`.
	Vector3 xyz = Vector3(1.0, 1.0, 1.0);

	Vector<Vector3> vertices; // For AreaPolygon only.

	NavigationMeshSourceGeometryData3D::ProjectedArea::ShapeType shape_type = NavigationMeshSourceGeometryData3D::ProjectedArea::ShapeType::NONE;

	bool enabled = false;
	uint32_t navigation_layers = 1;
	int priority = 0;

public:
	NavArea3D();
	~NavArea3D();

	void set_map(NavMap3D *p_map);
	NavMap3D *get_map() const { return map; }

	void set_position(const Vector3 p_position);
	const Vector3 &get_position() const { return position; }

	void set_shape_type(NavigationMeshSourceGeometryData3D::ProjectedArea::ShapeType p_shape_type);
	NavigationMeshSourceGeometryData3D::ProjectedArea::ShapeType get_shape_type() const { return shape_type; }

	void set_enabled(bool p_enabled);
	bool get_enabled() const { return enabled; }

	void set_navigation_layers(uint32_t p_layers);
	uint32_t get_navigation_layers() const { return navigation_layers; }

	void set_bake_priority(int p_priority);
	int get_bake_priority() const { return priority; }

	void set_size(const Vector3 p_size);
	const Vector3 &get_size() const { return xyz; }

	void set_elevation(const real_t p_elevation);
	real_t get_elevation() const { return xyz.z; }

	void set_height(const real_t p_height);
	real_t get_height() const { return xyz.y; }

	void set_radius(real_t p_radius);
	real_t get_radius() const { return xyz.x; }

	void set_vertices(const Vector<Vector3> &p_vertices);
	const Vector<Vector3> &get_vertices() const { return vertices; }
};
