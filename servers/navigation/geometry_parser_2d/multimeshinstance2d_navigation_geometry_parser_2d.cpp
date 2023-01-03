/**************************************************************************/
/*  multimeshinstance2d_navigation_geometry_parser_2d.cpp                 */
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

#include "multimeshinstance2d_navigation_geometry_parser_2d.h"

#include "scene/2d/multimesh_instance_2d.h"

#ifdef CLIPPER_ENABLED
#include "thirdparty/clipper2/include/clipper2/clipper.h"
#endif // CLIPPER_ENABLED

bool MultiMeshInstance2DNavigationGeometryParser2D::parses_node(Node *p_node) {
	return (Object::cast_to<MultiMeshInstance2D>(p_node) != nullptr);
}

void MultiMeshInstance2DNavigationGeometryParser2D::parse_geometry(Node *p_node, Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry) {
#ifdef CLIPPER_ENABLED
	NavigationPolygon::ParsedGeometryType parsed_geometry_type = p_navigation_polygon->get_parsed_geometry_type();

	if (Object::cast_to<MultiMeshInstance2D>(p_node) && parsed_geometry_type != NavigationPolygon::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		MultiMeshInstance2D *multimesh_instance = Object::cast_to<MultiMeshInstance2D>(p_node);
		Ref<MultiMesh> multimesh = multimesh_instance->get_multimesh();
		if (multimesh.is_valid() && multimesh->get_transform_format() == MultiMesh::TRANSFORM_2D) {
			Ref<Mesh> mesh = multimesh->get_mesh();
			if (mesh.is_valid()) {
				using namespace Clipper2Lib;

				Paths64 mesh_subject_paths, dummy_clip_paths;

				for (int i = 0; i < mesh->get_surface_count(); i++) {
					if (mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
						continue;
					}

					if (!(mesh->surface_get_format(i) & Mesh::ARRAY_FLAG_USE_2D_VERTICES)) {
						continue;
					}

					Path64 subject_path;

					int index_count = 0;
					if (mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
						index_count = mesh->surface_get_array_index_len(i);
					} else {
						index_count = mesh->surface_get_array_len(i);
					}

					ERR_CONTINUE((index_count == 0 || (index_count % 3) != 0));

					Array a = mesh->surface_get_arrays(i);

					Vector<Vector2> mesh_vertices = a[Mesh::ARRAY_VERTEX];

					if (mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
						Vector<int> mesh_indices = a[Mesh::ARRAY_INDEX];
						for (int vertex_index : mesh_indices) {
							const Vector2 &vertex = mesh_vertices[vertex_index];
							const Point64 &point = Point64(vertex.x, vertex.y);
							subject_path.push_back(point);
						}
					} else {
						for (const Vector2 &vertex : mesh_vertices) {
							const Point64 &point = Point64(vertex.x, vertex.y);
							subject_path.push_back(point);
						}
					}
					mesh_subject_paths.push_back(subject_path);
				}

				Paths64 mesh_path_solution = Union(mesh_subject_paths, dummy_clip_paths, FillRule::NonZero);

				//path_solution = RamerDouglasPeucker(path_solution, 0.025);

				int multimesh_instance_count = multimesh->get_visible_instance_count();
				if (multimesh_instance_count == -1) {
					multimesh_instance_count = multimesh->get_instance_count();
				}
				for (int i = 0; i < multimesh_instance_count; i++) {
					const Transform2D multimesh_instance_transform = multimesh_instance->get_transform() * multimesh->get_instance_transform_2d(i);

					for (const Path64 &mesh_path : mesh_path_solution) {
						Vector<Vector2> shape_outline;

						for (const Point64 &mesh_path_point : mesh_path) {
							shape_outline.push_back(Point2(static_cast<real_t>(mesh_path_point.x), static_cast<real_t>(mesh_path_point.y)));
						}

						for (int j = 0; j < shape_outline.size(); j++) {
							shape_outline.write[j] = multimesh_instance_transform.xform(shape_outline[j]);
						}
						p_source_geometry->add_obstruction_outline(shape_outline);
					}
				}
			}
		}
	}
#endif // CLIPPER_ENABLED
}
