/*************************************************************************/
/*  mesh_library.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GRID_THEME_H
#define GRID_THEME_H

#include "core/map.h"
#include "core/resource.h"
#include "mesh.h"
#include "scene/3d/navigation_mesh.h"
#include "shape.h"

class MeshLibrary : public Resource {
	GDCLASS(MeshLibrary, Resource);
	RES_BASE_EXTENSION("meshlib");

public:
	struct ShapeData {
		Ref<Shape> shape;
		Transform local_transform;
	};
	struct Item {
		String name;
		Ref<Mesh> mesh;
		Vector<ShapeData> shapes;
		Ref<Texture> preview;
		Transform navmesh_transform;
		Transform mesh_transform;
		Ref<NavigationMesh> navmesh;
	};

	Map<int, Item> item_map;

	void _set_item_shapes(int p_item, const Array &p_shapes);
	Array _get_item_shapes(int p_item) const;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	void create_item(int p_item);
	void set_item_name(int p_item, const String &p_name);
	void set_item_mesh(int p_item, const Ref<Mesh> &p_mesh);
	void set_item_mesh_transform(int p_item, const Transform &p_transform);
	void set_item_navmesh(int p_item, const Ref<NavigationMesh> &p_navmesh);
	void set_item_navmesh_transform(int p_item, const Transform &p_transform);
	void set_item_shapes(int p_item, const Vector<ShapeData> &p_shapes);
	void set_item_preview(int p_item, const Ref<Texture> &p_preview);
	String get_item_name(int p_item) const;
	Ref<Mesh> get_item_mesh(int p_item) const;
	Transform get_item_mesh_transform(int p_item) const;
	Ref<NavigationMesh> get_item_navmesh(int p_item) const;
	Transform get_item_navmesh_transform(int p_item) const;
	Vector<ShapeData> get_item_shapes(int p_item) const;
	Ref<Texture> get_item_preview(int p_item) const;

	void remove_item(int p_item);
	bool has_item(int p_item) const;

	void clear();

	int find_item_by_name(const String &p_name) const;

	Vector<int> get_item_list() const;
	int get_last_unused_item_id() const;

	MeshLibrary();
	~MeshLibrary();
};

#endif // CUBE_GRID_THEME_H
