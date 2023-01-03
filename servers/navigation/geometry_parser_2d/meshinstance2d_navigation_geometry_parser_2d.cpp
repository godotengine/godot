/**************************************************************************/
/*  meshinstance2d_navigation_geometry_parser_2d.cpp                      */
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

#include "meshinstance2d_navigation_geometry_parser_2d.h"

#include "scene/2d/mesh_instance_2d.h"

#ifdef CLIPPER_ENABLED
#include "thirdparty/clipper2/include/clipper2/clipper.h"
#endif // CLIPPER_ENABLED

bool MeshInstance2DNavigationGeometryParser2D::parses_node(Node *p_node) {
	return (Object::cast_to<MeshInstance2D>(p_node) != nullptr);
}

void MeshInstance2DNavigationGeometryParser2D::parse_geometry(Node *p_node, Ref<NavigationPolygon> p_navigation_polygon, Ref<NavigationMeshSourceGeometryData2D> p_source_geometry) {
#ifdef CLIPPER_ENABLED
	NavigationPolygon::ParsedGeometryType parsed_geometry_type = p_navigation_polygon->get_parsed_geometry_type();

	if (Object::cast_to<MeshInstance2D>(p_node) && parsed_geometry_type != NavigationPolygon::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		MeshInstance2D *mesh_instance = Object::cast_to<MeshInstance2D>(p_node);
		Ref<Mesh> mesh = mesh_instance->get_mesh();
		if (!mesh.is_valid()) {
			return;
		}

		const Transform2D transform = mesh_instance->get_transform();

		using namespace Clipper2Lib;

		Paths64 subject_paths, dummy_clip_paths;

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
			subject_paths.push_back(subject_path);
		}

		Paths64 path_solution;

		path_solution = Union(subject_paths, dummy_clip_paths, FillRule::NonZero);

		//path_solution = RamerDouglasPeucker(path_solution, 0.025);

		Vector<Vector<Vector2>> polypaths;

		for (const Path64 &scaled_path : path_solution) {
			Vector<Vector2> shape_outline;
			for (const Point64 &scaled_point : scaled_path) {
				shape_outline.push_back(Point2(static_cast<real_t>(scaled_point.x), static_cast<real_t>(scaled_point.y)));
			}

			for (int i = 0; i < shape_outline.size(); i++) {
				shape_outline.write[i] = transform.xform(shape_outline[i]);
			}

			p_source_geometry->add_obstruction_outline(shape_outline);
		}
	}
#endif // CLIPPER_ENABLED
}
