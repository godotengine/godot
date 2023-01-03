/**************************************************************************/
/*  multimeshinstance3d_navigation_geometry_parser_3d.cpp                 */
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

#include "multimeshinstance3d_navigation_geometry_parser_3d.h"

#include "scene/3d/multimesh_instance_3d.h"

bool MultiMeshInstance3DNavigationGeometryParser3D::parses_node(Node *p_node) {
	return (Object::cast_to<MultiMeshInstance3D>(p_node) != nullptr);
}

void MultiMeshInstance3DNavigationGeometryParser3D::parse_geometry(Node *p_node, Ref<NavigationMesh> p_navigationmesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry) {
	NavigationMesh::ParsedGeometryType parsed_geometry_type = p_navigationmesh->get_parsed_geometry_type();

	if (Object::cast_to<MultiMeshInstance3D>(p_node) && parsed_geometry_type != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		MultiMeshInstance3D *multimesh_instance = Object::cast_to<MultiMeshInstance3D>(p_node);
		Ref<MultiMesh> multimesh = multimesh_instance->get_multimesh();
		if (multimesh.is_valid()) {
			Ref<Mesh> mesh = multimesh->get_mesh();
			if (mesh.is_valid()) {
				int n = multimesh->get_visible_instance_count();
				if (n == -1) {
					n = multimesh->get_instance_count();
				}
				for (int i = 0; i < n; i++) {
					p_source_geometry->add_mesh(mesh, multimesh_instance->get_global_transform() * multimesh->get_instance_transform(i));
				}
			}
		}
	}
}
