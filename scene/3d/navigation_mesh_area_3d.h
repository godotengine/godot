/**************************************************************************/
/*  navigation_mesh_area_3d.h                                             */
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

#ifndef NAVIGATION_MESH_AREA_3D_H
#define NAVIGATION_MESH_AREA_3D_H

#include "scene/3d/node_3d.h"

class NavigationMesh;
class NavigationMeshSourceGeometryData3D;

class NavigationMeshArea3D : public Node3D {
	GDCLASS(NavigationMeshArea3D, Node3D);

protected:
	RID area;
	RID map_override;

	bool enabled = true;
	float height = 1.0;
	uint32_t navigation_layers = 1;
	int priority = 0;
	AABB bounds;
	bool bounds_dirty = true;

	virtual void _update_bounds();
	static AABB _xform_bounds(const Vector<Vector3> &p_vertices, const Transform3D &p_gt, float p_height);

	static void _bind_methods();
	void _notification(int p_what);

#ifdef DEBUG_ENABLED
	RID debug_instance_rid;
	RID debug_mesh_rid;

	virtual void _update_debug();
	void _clear_debug();
#endif // DEBUG_ENABLED

public:
	RID get_rid() const;

	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_navigation_map(RID p_navigation_map);
	RID get_navigation_map() const;

	void set_height(float p_height);
	float get_height() const;

	void set_navigation_layers(uint32_t p_navigation_layers);
	uint32_t get_navigation_layers() const;

	void set_navigation_layer_value(int p_layer_number, bool p_value);
	bool get_navigation_layer_value(int p_layer_number) const;

	void set_priority(int p_priority);
	int get_priority() const;

	AABB get_bounds();

	NavigationMeshArea3D();
	~NavigationMeshArea3D();

private:
	static Callable _navmesh_source_geometry_parsing_callback;
	static RID _navmesh_source_geometry_parser;

public:
	static void navmesh_parse_init();
	static void navmesh_parse_source_geometry(const Ref<NavigationMesh> &p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Node *p_node);
};

class NavigationMeshAreaBox3D : public NavigationMeshArea3D {
	GDCLASS(NavigationMeshAreaBox3D, NavigationMeshArea3D);

	Vector3 size = Vector3(1.0, 1.0, 1.0);

	virtual void _update_bounds() override;

protected:
	static void _bind_methods();

#ifdef DEBUG_ENABLED
	virtual void _update_debug() override;
#endif // DEBUG_ENABLED

public:
	void set_size(const Vector3 &p_size);
	const Vector3 &get_size() const;

	PackedStringArray get_configuration_warnings() const override;

	NavigationMeshAreaBox3D();
	~NavigationMeshAreaBox3D();
};

class NavigationMeshAreaCylinder3D : public NavigationMeshArea3D {
	GDCLASS(NavigationMeshAreaCylinder3D, NavigationMeshArea3D);

	float radius = 1.0;

	virtual void _update_bounds() override;

protected:
	static void _bind_methods();

#ifdef DEBUG_ENABLED
	virtual void _update_debug() override;
#endif // DEBUG_ENABLED

public:
	void set_radius(float p_radius);
	float get_radius() const;

	PackedStringArray get_configuration_warnings() const override;

	NavigationMeshAreaCylinder3D();
	~NavigationMeshAreaCylinder3D();
};

class NavigationMeshAreaPolygon3D : public NavigationMeshArea3D {
	GDCLASS(NavigationMeshAreaPolygon3D, NavigationMeshArea3D);

	Vector<Vector3> vertices;
	bool vertices_are_clockwise = true;
	bool vertices_are_valid = true;

	virtual void _update_bounds() override;

protected:
	static void _bind_methods();

#ifdef DEBUG_ENABLED
	virtual void _update_debug() override;
#endif // DEBUG_ENABLED

public:
	void set_vertices(const Vector<Vector3> &p_vertices);
	const Vector<Vector3> &get_vertices() const;

	bool are_vertices_clockwise() const;
	bool are_vertices_valid() const;

	PackedStringArray get_configuration_warnings() const override;

	NavigationMeshAreaPolygon3D();
	~NavigationMeshAreaPolygon3D();
};

#endif // NAVIGATION_MESH_AREA_3D_H
