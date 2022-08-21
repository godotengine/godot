/*************************************************************************/
/*  csgshape3d_navigation_geometry_parser_3d.h                           */
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

#ifndef CSGSHAPE3D_NAVIGATION_GEOMETRY_PARSER_3D_H
#define CSGSHAPE3D_NAVIGATION_GEOMETRY_PARSER_3D_H

#include "modules/csg/csg_shape.h"
#include "modules/navigation/navigation_geometry_parser_3d.h"

class CSGShape3DNavigationGeometryParser3D : public NavigationGeometryParser3D {
public:
	virtual bool parses_node(Node *p_node) override {
		return (Object::cast_to<CSGShape3D>(p_node) != nullptr);
	}

	virtual void parse_geometry(Node *p_node, Ref<NavigationMesh> p_navigationmesh) override {
		NavigationMesh::ParsedGeometryType parsed_geometry_type = p_navigationmesh->get_parsed_geometry_type();

		if (Object::cast_to<CSGShape3D>(p_node) && parsed_geometry_type != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
			CSGShape3D *csg_shape = Object::cast_to<CSGShape3D>(p_node);
			Array meshes = csg_shape->get_meshes();
			if (!meshes.is_empty()) {
				Ref<Mesh> mesh = meshes[1];
				if (mesh.is_valid()) {
					add_mesh(mesh, csg_shape->get_global_transform());
				}
			}
		}
	}
};

#endif // CSGSHAPE3D_NAVIGATION_GEOMETRY_PARSER_3D_H
