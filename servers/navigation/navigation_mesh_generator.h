/**************************************************************************/
/*  navigation_mesh_generator.h                                           */
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

#ifndef NAVIGATION_MESH_GENERATOR_H
#define NAVIGATION_MESH_GENERATOR_H

#include "core/object/class_db.h"
#include "core/templates/rid.h"

#include "scene/2d/navigation_geometry_parser_2d.h"
#ifndef _3D_DISABLED
#include "scene/3d/navigation_geometry_parser_3d.h"
#endif // _3D_DISABLED
#include "scene/main/node.h"
#include "scene/resources/navigation_mesh.h"
#include "scene/resources/navigation_mesh_source_geometry_data_2d.h"
#ifndef _3D_DISABLED
#include "scene/resources/navigation_mesh_source_geometry_data_3d.h"
#endif // _3D_DISABLED

class NavigationMeshGenerator : public Object {
	GDCLASS(NavigationMeshGenerator, Object);

	static NavigationMeshGenerator *singleton;

protected:
	static void _bind_methods();

public:
	static NavigationMeshGenerator *get_singleton();

	virtual void process() = 0;
	virtual void cleanup() = 0;

	// 2D //////////////////////////////
	virtual void register_geometry_parser_2d(Ref<NavigationGeometryParser2D> p_geometry_parser) = 0;
	virtual void unregister_geometry_parser_2d(Ref<NavigationGeometryParser2D> p_geometry_parser) = 0;

	virtual Ref<NavigationMeshSourceGeometryData2D> parse_2d_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Node *p_root_node, Callable p_callback = Callable()) = 0;
	virtual void bake_2d_from_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry_data, Callable p_callback = Callable()) = 0;

	virtual void parse_and_bake_2d(Ref<NavigationPolygon> p_navigation_polygon, Node *p_root_node, Callable p_callback = Callable()) = 0;

	virtual bool is_navigation_polygon_baking(Ref<NavigationPolygon> p_navigation_polygon) const = 0;

	// 3D //////////////////////////////
#ifndef _3D_DISABLED
	virtual void register_geometry_parser_3d(Ref<NavigationGeometryParser3D> p_geometry_parser) = 0;
	virtual void unregister_geometry_parser_3d(Ref<NavigationGeometryParser3D> p_geometry_parser) = 0;

	virtual Ref<NavigationMeshSourceGeometryData3D> parse_3d_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node, Callable p_callback = Callable()) = 0;
	virtual void bake_3d_from_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Callable p_callback = Callable()) = 0;

	virtual void parse_and_bake_3d(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node, Callable p_callback = Callable()) = 0;

	virtual bool is_navigation_mesh_baking(Ref<NavigationMesh> p_navigation_mesh) const = 0;
#endif // _3D_DISABLED

	NavigationMeshGenerator();
	~NavigationMeshGenerator() override;
};

/// NavigationMeshGeneratorManager ////////////////////////////////////////////////////

class NavigationMeshGeneratorManager : public Object {
	GDCLASS(NavigationMeshGeneratorManager, Object);

	static NavigationMeshGeneratorManager *singleton;

	struct ClassInfo {
		String name;
		Callable create_callback;
	};

	Vector<ClassInfo> navigation_mesh_generators;
	int default_server_id = -1;
	int default_server_priority = -1;

	void on_servers_changed();

protected:
	static void _bind_methods();

public:
	static const String setting_property_name;

	static NavigationMeshGeneratorManager *get_singleton();

	void register_server(const String &p_name, const Callable &p_create_callback);
	void set_default_server(const String &p_name, int p_priority = 0);
	int find_server_id(const String &p_name) const;

	NavigationMeshGenerator *new_default_server() const;
	NavigationMeshGenerator *new_server(const String &p_name) const;

	NavigationMeshGeneratorManager();
	~NavigationMeshGeneratorManager() override;
};

#endif // NAVIGATION_MESH_GENERATOR_H
