/**************************************************************************/
/*  staticbody3d_navigation_geometry_parser_3d.cpp                        */
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

#include "staticbody3d_navigation_geometry_parser_3d.h"

#include "core/math/convex_hull.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/resources/box_shape_3d.h"
#include "scene/resources/capsule_shape_3d.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/convex_polygon_shape_3d.h"
#include "scene/resources/cylinder_shape_3d.h"
#include "scene/resources/height_map_shape_3d.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/shape_3d.h"
#include "scene/resources/sphere_shape_3d.h"
#include "scene/resources/world_boundary_shape_3d.h"

bool StaticBody3DNavigationGeometryParser3D::parses_node(Node *p_node) {
	return (Object::cast_to<StaticBody3D>(p_node) != nullptr);
}

void StaticBody3DNavigationGeometryParser3D::parse_geometry(Node *p_node, Ref<NavigationMesh> p_navigationmesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry) {
	NavigationMesh::ParsedGeometryType parsed_geometry_type = p_navigationmesh->get_parsed_geometry_type();
	uint32_t navigationmesh_collision_mask = p_navigationmesh->get_collision_mask();

	if (Object::cast_to<StaticBody3D>(p_node) && parsed_geometry_type != NavigationMesh::PARSED_GEOMETRY_MESH_INSTANCES) {
		StaticBody3D *static_body = Object::cast_to<StaticBody3D>(p_node);
		if (static_body->get_collision_layer() & navigationmesh_collision_mask) {
			List<uint32_t> shape_owners;
			static_body->get_shape_owners(&shape_owners);
			for (uint32_t shape_owner : shape_owners) {
				if (static_body->is_shape_owner_disabled(shape_owner)) {
					continue;
				}
				const int shape_count = static_body->shape_owner_get_shape_count(shape_owner);
				for (int shape_index = 0; shape_index < shape_count; shape_index++) {
					Ref<Shape3D> s = static_body->shape_owner_get_shape(shape_owner, shape_index);
					if (s.is_null()) {
						continue;
					}

					const Transform3D transform = static_body->get_global_transform() * static_body->shape_owner_get_transform(shape_owner);

					BoxShape3D *box = Object::cast_to<BoxShape3D>(*s);
					if (box) {
						Array arr;
						arr.resize(RS::ARRAY_MAX);
						BoxMesh::create_mesh_array(arr, box->get_size());
						p_source_geometry->add_mesh_array(arr, transform);
					}

					CapsuleShape3D *capsule = Object::cast_to<CapsuleShape3D>(*s);
					if (capsule) {
						Array arr;
						arr.resize(RS::ARRAY_MAX);
						CapsuleMesh::create_mesh_array(arr, capsule->get_radius(), capsule->get_height());
						p_source_geometry->add_mesh_array(arr, transform);
					}

					CylinderShape3D *cylinder = Object::cast_to<CylinderShape3D>(*s);
					if (cylinder) {
						Array arr;
						arr.resize(RS::ARRAY_MAX);
						CylinderMesh::create_mesh_array(arr, cylinder->get_radius(), cylinder->get_radius(), cylinder->get_height());
						p_source_geometry->add_mesh_array(arr, transform);
					}

					SphereShape3D *sphere = Object::cast_to<SphereShape3D>(*s);
					if (sphere) {
						Array arr;
						arr.resize(RS::ARRAY_MAX);
						SphereMesh::create_mesh_array(arr, sphere->get_radius(), sphere->get_radius() * 2.0);
						p_source_geometry->add_mesh_array(arr, transform);
					}

					ConcavePolygonShape3D *concave_polygon = Object::cast_to<ConcavePolygonShape3D>(*s);
					if (concave_polygon) {
						p_source_geometry->add_faces(concave_polygon->get_faces(), transform);
					}

					ConvexPolygonShape3D *convex_polygon = Object::cast_to<ConvexPolygonShape3D>(*s);
					if (convex_polygon) {
						Vector<Vector3> varr = Variant(convex_polygon->get_points());
						Geometry3D::MeshData md;

						Error err = ConvexHullComputer::convex_hull(varr, md);

						if (err == OK) {
							PackedVector3Array faces;

							for (const Geometry3D::MeshData::Face &face : md.faces) {
								for (uint32_t k = 2; k < face.indices.size(); ++k) {
									faces.push_back(md.vertices[face.indices[0]]);
									faces.push_back(md.vertices[face.indices[k - 1]]);
									faces.push_back(md.vertices[face.indices[k]]);
								}
							}

							p_source_geometry->add_faces(faces, transform);
						}
					}

					HeightMapShape3D *heightmap_shape = Object::cast_to<HeightMapShape3D>(*s);
					if (heightmap_shape) {
						int heightmap_depth = heightmap_shape->get_map_depth();
						int heightmap_width = heightmap_shape->get_map_width();

						if (heightmap_depth >= 2 && heightmap_width >= 2) {
							const Vector<real_t> &map_data = heightmap_shape->get_map_data();

							Vector2 heightmap_gridsize(heightmap_width - 1, heightmap_depth - 1);
							Vector2 start = heightmap_gridsize * -0.5;

							Vector<Vector3> vertex_array;
							vertex_array.resize((heightmap_depth - 1) * (heightmap_width - 1) * 6);
							int map_data_current_index = 0;

							for (int d = 0; d < heightmap_depth; d++) {
								for (int w = 0; w < heightmap_width; w++) {
									if (map_data_current_index + 1 + heightmap_depth < map_data.size()) {
										float top_left_height = map_data[map_data_current_index];
										float top_right_height = map_data[map_data_current_index + 1];
										float bottom_left_height = map_data[map_data_current_index + heightmap_depth];
										float bottom_right_height = map_data[map_data_current_index + 1 + heightmap_depth];

										Vector3 top_left = Vector3(start.x + w, top_left_height, start.y + d);
										Vector3 top_right = Vector3(start.x + w + 1.0, top_right_height, start.y + d);
										Vector3 bottom_left = Vector3(start.x + w, bottom_left_height, start.y + d + 1.0);
										Vector3 bottom_right = Vector3(start.x + w + 1.0, bottom_right_height, start.y + d + 1.0);

										vertex_array.push_back(top_right);
										vertex_array.push_back(bottom_left);
										vertex_array.push_back(top_left);
										vertex_array.push_back(top_right);
										vertex_array.push_back(bottom_right);
										vertex_array.push_back(bottom_left);
									}
									map_data_current_index += 1;
								}
							}
							if (vertex_array.size() > 0) {
								p_source_geometry->add_faces(vertex_array, transform);
							}
						}
					}
				}
			}
		}
	}
}
