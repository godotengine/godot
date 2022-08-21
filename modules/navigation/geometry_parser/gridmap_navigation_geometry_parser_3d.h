/*************************************************************************/
/*  gridmap_navigation_geometry_parser_3d.h                              */
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

#ifndef GRIDMAP_NAVIGATION_GEOMETRY_PARSER_3D_H
#define GRIDMAP_NAVIGATION_GEOMETRY_PARSER_3D_H

#include "modules/gridmap/grid_map.h"
#include "modules/navigation/navigation_geometry_parser_3d.h"

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

class GridMap3DNavigationGeometryParser3D : public NavigationGeometryParser3D {
public:
	virtual bool parses_node(Node *p_node) override {
		return (Object::cast_to<GridMap>(p_node) != nullptr);
	}

	virtual void parse_geometry(Node *p_node, Ref<NavigationMesh> p_navigationmesh) override {
		GridMap *gridmap = Object::cast_to<GridMap>(p_node);
		NavigationMesh::ParsedGeometryType parsed_geometry_type = p_navigationmesh->get_parsed_geometry_type();
		uint32_t navigationmesh_collision_mask = p_navigationmesh->get_collision_mask();

		if (gridmap) {
			if (parsed_geometry_type != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
				Array meshes = gridmap->get_meshes();
				Transform3D xform = gridmap->get_global_transform();
				for (int i = 0; i < meshes.size(); i += 2) {
					Ref<Mesh> mesh = meshes[i + 1];
					if (mesh.is_valid()) {
						add_mesh(mesh, xform * (Transform3D)meshes[i]);
					}
				}
			}

			if (parsed_geometry_type != NavigationMesh::PARSED_GEOMETRY_MESH_INSTANCES && (gridmap->get_collision_layer() & navigationmesh_collision_mask)) {
				Array shapes = gridmap->get_collision_shapes();
				for (int i = 0; i < shapes.size(); i += 2) {
					RID shape = shapes[i + 1];
					PhysicsServer3D::ShapeType type = PhysicsServer3D::get_singleton()->shape_get_type(shape);
					Variant data = PhysicsServer3D::get_singleton()->shape_get_data(shape);

					switch (type) {
						case PhysicsServer3D::SHAPE_SPHERE: {
							real_t radius = data;
							Array arr;
							arr.resize(RS::ARRAY_MAX);
							SphereMesh::create_mesh_array(arr, radius, radius * 2.0);
							add_mesh_array(arr, shapes[i]);
						} break;
						case PhysicsServer3D::SHAPE_BOX: {
							Vector3 extents = data;
							Array arr;
							arr.resize(RS::ARRAY_MAX);
							BoxMesh::create_mesh_array(arr, extents * 2.0);
							add_mesh_array(arr, shapes[i]);
						} break;
						case PhysicsServer3D::SHAPE_CAPSULE: {
							Dictionary dict = data;
							real_t radius = dict["radius"];
							real_t height = dict["height"];
							Array arr;
							arr.resize(RS::ARRAY_MAX);
							CapsuleMesh::create_mesh_array(arr, radius, height);
							add_mesh_array(arr, shapes[i]);
						} break;
						case PhysicsServer3D::SHAPE_CYLINDER: {
							Dictionary dict = data;
							real_t radius = dict["radius"];
							real_t height = dict["height"];
							Array arr;
							arr.resize(RS::ARRAY_MAX);
							CylinderMesh::create_mesh_array(arr, radius, radius, height);
							add_mesh_array(arr, shapes[i]);
						} break;
						case PhysicsServer3D::SHAPE_CONVEX_POLYGON: {
							PackedVector3Array vertices = data;
							Geometry3D::MeshData md;

							Error err = ConvexHullComputer::convex_hull(vertices, md);

							if (err == OK) {
								PackedVector3Array faces;

								for (uint32_t j = 0; j < md.faces.size(); ++j) {
									const Geometry3D::MeshData::Face &face = md.faces[j];

									for (uint32_t k = 2; k < face.indices.size(); ++k) {
										faces.push_back(md.vertices[face.indices[0]]);
										faces.push_back(md.vertices[face.indices[k - 1]]);
										faces.push_back(md.vertices[face.indices[k]]);
									}
								}

								add_faces(faces, shapes[i]);
							}
						} break;
						case PhysicsServer3D::SHAPE_CONCAVE_POLYGON: {
							Dictionary dict = data;
							PackedVector3Array faces = Variant(dict["faces"]);
							add_faces(faces, shapes[i]);
						} break;
						case PhysicsServer3D::SHAPE_HEIGHTMAP: {
							Dictionary dict = data;
							///< dict( int:"width", int:"depth",float:"cell_size", float_array:"heights"
							int heightmap_depth = dict["depth"];
							int heightmap_width = dict["width"];

							if (heightmap_depth >= 2 && heightmap_width >= 2) {
								const Vector<real_t> &map_data = dict["heights"];

								Vector2 heightmap_gridsize(heightmap_width - 1, heightmap_depth - 1);
								Vector2 start = heightmap_gridsize * -0.5;

								Vector<Vector3> vertex_array;
								vertex_array.resize((heightmap_depth - 1) * (heightmap_width - 1) * 6);
								int map_data_current_index = 0;

								for (int d = 0; d < heightmap_depth - 1; d++) {
									for (int w = 0; w < heightmap_width - 1; w++) {
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
									add_faces(vertex_array, shapes[i]);
								}
							}
						} break;
						default: {
							WARN_PRINT("Unsupported collision shape type.");
						} break;
					}
				}
			}
		}
	}
};

#endif // GRIDMAP_NAVIGATION_GEOMETRY_PARSER_3D_H
