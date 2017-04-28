/*************************************************************************/
/*  navigation_mesh.h                                                    */
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
#ifndef NAVIGATION_MESH_H
#define NAVIGATION_MESH_H

#include "scene/3d/spatial.h"
#include "scene/resources/mesh.h"

class Mesh;

class NavigationMesh : public Resource {

	GDCLASS(NavigationMesh, Resource);

	PoolVector<Vector3> vertices;
	struct Polygon {
		Vector<int> indices;
	};
	Vector<Polygon> polygons;
	Ref<Mesh> debug_mesh;

	struct _EdgeKey {

		Vector3 from;
		Vector3 to;

		bool operator<(const _EdgeKey &p_with) const { return from == p_with.from ? to < p_with.to : from < p_with.from; }
	};

protected:
	static void _bind_methods();

	void _set_polygons(const Array &p_array);
	Array _get_polygons() const;

public:
	void create_from_mesh(const Ref<Mesh> &p_mesh);

	void set_vertices(const PoolVector<Vector3> &p_vertices);
	PoolVector<Vector3> get_vertices() const;

	void add_polygon(const Vector<int> &p_polygon);
	int get_polygon_count() const;
	Vector<int> get_polygon(int p_idx);
	void clear_polygons();

	Ref<Mesh> get_debug_mesh();

	NavigationMesh();
};

class Navigation;

class NavigationMeshInstance : public Spatial {

	GDCLASS(NavigationMeshInstance, Spatial);

	bool enabled;
	int nav_id;
	Navigation *navigation;
	Ref<NavigationMesh> navmesh;

	Node *debug_view;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_navigation_mesh(const Ref<NavigationMesh> &p_navmesh);
	Ref<NavigationMesh> get_navigation_mesh() const;

	String get_configuration_warning() const;

	NavigationMeshInstance();
};

#endif // NAVIGATION_MESH_H
