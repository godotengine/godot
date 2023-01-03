/**************************************************************************/
/*  navigation_mesh_generator_dummy.h                                     */
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

#ifndef NAVIGATION_MESH_GENERATOR_DUMMY_H
#define NAVIGATION_MESH_GENERATOR_DUMMY_H

#include "servers/navigation/navigation_mesh_generator.h"

class NavigationMeshGeneratorDummy : public NavigationMeshGenerator {
	GDCLASS(NavigationMeshGeneratorDummy, NavigationMeshGenerator);

public:
	virtual void cleanup() override {}
	virtual void process() override {}

	// 2D //////////////////////////////
	virtual void register_geometry_parser_2d(Ref<NavigationGeometryParser2D> p_geometry_parser) override {}
	virtual void unregister_geometry_parser_2d(Ref<NavigationGeometryParser2D> p_geometry_parser) override {}

	virtual Ref<NavigationMeshSourceGeometryData2D> parse_2d_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Node *p_root_node, Callable p_callback = Callable()) override { return Ref<NavigationMeshSourceGeometryData2D>(); }
	virtual void bake_2d_from_source_geometry_data(Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry_data, Callable p_callback = Callable()) override {}

	virtual void parse_and_bake_2d(Ref<NavigationPolygon> p_navigation_polygon, Node *p_root_node, Callable p_callback = Callable()) override {}

	virtual bool is_navigation_polygon_baking(Ref<NavigationPolygon> p_navigation_polygon) const override { return false; }

	// 3D //////////////////////////////
#ifndef _3D_DISABLED
	virtual void register_geometry_parser_3d(Ref<NavigationGeometryParser3D> p_geometry_parser) override {}
	virtual void unregister_geometry_parser_3d(Ref<NavigationGeometryParser3D> p_geometry_parser) override {}

	virtual Ref<NavigationMeshSourceGeometryData3D> parse_3d_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node, Callable p_callback = Callable()) override { return Ref<NavigationMeshSourceGeometryData3D>(); }
	virtual void bake_3d_from_source_geometry_data(Ref<NavigationMesh> p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Callable p_callback = Callable()) override {}

	virtual void parse_and_bake_3d(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node, Callable p_callback = Callable()) override {}

	virtual bool is_navigation_mesh_baking(Ref<NavigationMesh> p_navigation_mesh) const override { return false; }
#endif // _3D_DISABLED
};

#endif // NAVIGATION_MESH_GENERATOR_DUMMY_H
