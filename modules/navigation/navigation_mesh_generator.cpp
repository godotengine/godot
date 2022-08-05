/*************************************************************************/
/*  navigation_mesh_generator.cpp                                        */
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

#include "core/math/convex_hull.h"
#ifndef _3D_DISABLED

#include "navigation_mesh_generator.h"
//#include "core/math/quick_hull.h"
//#include "core/math/convex_hull.h"
#include "core/os/thread.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/3d/physics_body.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/cylinder_shape.h"
#include "scene/resources/height_map_shape.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/shape.h"
#include "scene/resources/sphere_shape.h"

#include "modules/modules_enabled.gen.h" // For csg, gridmap.

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#endif

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#ifdef MODULE_GRIDMAP_ENABLED
#include "modules/gridmap/grid_map.h"
#endif

NavigationMeshGenerator *NavigationMeshGenerator::singleton = NULL;

void NavigationMeshGenerator::_add_vertex(const Vector3 &p_vec3, Vector<float> &p_vertices) {
	p_vertices.push_back(p_vec3.x);
	p_vertices.push_back(p_vec3.y);
	p_vertices.push_back(p_vec3.z);
}

void NavigationMeshGenerator::_add_mesh(const Ref<Mesh> &p_mesh, const Transform &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices) {
	int current_vertex_count;

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		current_vertex_count = p_vertices.size() / 3;

		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
			continue;
		}

		int index_count = 0;
		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
			index_count = p_mesh->surface_get_array_index_len(i);
		} else {
			index_count = p_mesh->surface_get_array_len(i);
		}

		ERR_CONTINUE((index_count == 0 || (index_count % 3) != 0));

		int face_count = index_count / 3;

		Array a = p_mesh->surface_get_arrays(i);

		PoolVector<Vector3> mesh_vertices = a[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr = mesh_vertices.read();

		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
			PoolVector<int> mesh_indices = a[Mesh::ARRAY_INDEX];
			PoolVector<int>::Read ir = mesh_indices.read();

			for (int j = 0; j < mesh_vertices.size(); j++) {
				_add_vertex(p_xform.xform(vr[j]), p_vertices);
			}

			for (int j = 0; j < face_count; j++) {
				// CCW
				p_indices.push_back(current_vertex_count + (ir[j * 3 + 0]));
				p_indices.push_back(current_vertex_count + (ir[j * 3 + 2]));
				p_indices.push_back(current_vertex_count + (ir[j * 3 + 1]));
			}
		} else {
			face_count = mesh_vertices.size() / 3;
			for (int j = 0; j < face_count; j++) {
				_add_vertex(p_xform.xform(vr[j * 3 + 0]), p_vertices);
				_add_vertex(p_xform.xform(vr[j * 3 + 2]), p_vertices);
				_add_vertex(p_xform.xform(vr[j * 3 + 1]), p_vertices);

				p_indices.push_back(current_vertex_count + (j * 3 + 0));
				p_indices.push_back(current_vertex_count + (j * 3 + 1));
				p_indices.push_back(current_vertex_count + (j * 3 + 2));
			}
		}
	}
}

void NavigationMeshGenerator::_add_mesh_array(const Array &p_array, const Transform &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices) {
	PoolVector<Vector3> mesh_vertices = p_array[Mesh::ARRAY_VERTEX];
	PoolVector<Vector3>::Read vr = mesh_vertices.read();

	PoolVector<int> mesh_indices = p_array[Mesh::ARRAY_INDEX];
	PoolVector<int>::Read ir = mesh_indices.read();

	const int face_count = mesh_indices.size() / 3;
	const int current_vertex_count = p_vertices.size() / 3;

	for (int j = 0; j < mesh_vertices.size(); j++) {
		_add_vertex(p_xform.xform(vr[j]), p_vertices);
	}

	for (int j = 0; j < face_count; j++) {
		// CCW
		p_indices.push_back(current_vertex_count + (ir[j * 3 + 0]));
		p_indices.push_back(current_vertex_count + (ir[j * 3 + 2]));
		p_indices.push_back(current_vertex_count + (ir[j * 3 + 1]));
	}
}

void NavigationMeshGenerator::_add_faces(const PoolVector3Array &p_faces, const Transform &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices) {
	int face_count = p_faces.size() / 3;
	int current_vertex_count = p_vertices.size() / 3;

	for (int j = 0; j < face_count; j++) {
		_add_vertex(p_xform.xform(p_faces[j * 3 + 0]), p_vertices);
		_add_vertex(p_xform.xform(p_faces[j * 3 + 1]), p_vertices);
		_add_vertex(p_xform.xform(p_faces[j * 3 + 2]), p_vertices);

		p_indices.push_back(current_vertex_count + (j * 3 + 0));
		p_indices.push_back(current_vertex_count + (j * 3 + 2));
		p_indices.push_back(current_vertex_count + (j * 3 + 1));
	}
}

void NavigationMeshGenerator::_parse_geometry(const Transform &p_navmesh_xform, Node *p_node, Vector<float> &p_vertices, Vector<int> &p_indices, int p_generate_from, uint32_t p_collision_mask, bool p_recurse_children) {
	if (Object::cast_to<MeshInstance>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		MeshInstance *mesh_instance = Object::cast_to<MeshInstance>(p_node);
		Ref<Mesh> mesh = mesh_instance->get_mesh();
		if (mesh.is_valid()) {
			_add_mesh(mesh, p_navmesh_xform * mesh_instance->get_global_transform(), p_vertices, p_indices);
		}
	}

	if (Object::cast_to<MultiMeshInstance>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		MultiMeshInstance *multimesh_instance = Object::cast_to<MultiMeshInstance>(p_node);
		Ref<MultiMesh> multimesh = multimesh_instance->get_multimesh();
		if (multimesh.is_valid()) {
			Ref<Mesh> mesh = multimesh->get_mesh();
			if (mesh.is_valid()) {
				int n = multimesh->get_visible_instance_count();
				if (n == -1) {
					n = multimesh->get_instance_count();
				}
				for (int i = 0; i < n; i++) {
					_add_mesh(mesh, p_navmesh_xform * multimesh_instance->get_global_transform() * multimesh->get_instance_transform(i), p_vertices, p_indices);
				}
			}
		}
	}

#ifdef MODULE_CSG_ENABLED
	if (Object::cast_to<CSGShape>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		CSGShape *csg_shape = Object::cast_to<CSGShape>(p_node);
		Array meshes = csg_shape->get_meshes();
		if (!meshes.empty()) {
			Ref<Mesh> mesh = meshes[1];
			if (mesh.is_valid()) {
				_add_mesh(mesh, p_navmesh_xform * csg_shape->get_global_transform(), p_vertices, p_indices);
			}
		}
	}
#endif

	if (Object::cast_to<StaticBody>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_MESH_INSTANCES) {
		StaticBody *static_body = Object::cast_to<StaticBody>(p_node);

		if (static_body->get_collision_layer() & p_collision_mask) {
			List<uint32_t> shape_owners;
			static_body->get_shape_owners(&shape_owners);
			for (List<uint32_t>::Element *E = shape_owners.front(); E; E = E->next()) {
				uint32_t shape_owner = E->get();
				const int shape_count = static_body->shape_owner_get_shape_count(shape_owner);
				for (int i = 0; i < shape_count; i++) {
					Ref<Shape> s = static_body->shape_owner_get_shape(shape_owner, i);
					if (s.is_null()) {
						continue;
					}

					const Transform transform = p_navmesh_xform * static_body->get_global_transform() * static_body->shape_owner_get_transform(shape_owner);

					BoxShape *box = Object::cast_to<BoxShape>(*s);
					if (box) {
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						CubeMesh::create_mesh_array(arr, box->get_extents() * 2.0);
						_add_mesh_array(arr, transform, p_vertices, p_indices);
					}

					CapsuleShape *capsule = Object::cast_to<CapsuleShape>(*s);
					if (capsule) {
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						CapsuleMesh::create_mesh_array(arr, capsule->get_radius(), capsule->get_height() / 2.0);
						_add_mesh_array(arr, transform, p_vertices, p_indices);
					}

					CylinderShape *cylinder = Object::cast_to<CylinderShape>(*s);
					if (cylinder) {
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						CylinderMesh::create_mesh_array(arr, cylinder->get_radius(), cylinder->get_radius(), cylinder->get_height());
						_add_mesh_array(arr, transform, p_vertices, p_indices);
					}

					SphereShape *sphere = Object::cast_to<SphereShape>(*s);
					if (sphere) {
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						SphereMesh::create_mesh_array(arr, sphere->get_radius(), sphere->get_radius() * 2.0);
						_add_mesh_array(arr, transform, p_vertices, p_indices);
					}

					ConcavePolygonShape *concave_polygon = Object::cast_to<ConcavePolygonShape>(*s);
					if (concave_polygon) {
						_add_faces(concave_polygon->get_faces(), transform, p_vertices, p_indices);
					}

					ConvexPolygonShape *convex_polygon = Object::cast_to<ConvexPolygonShape>(*s);
					if (convex_polygon) {
						Vector<Vector3> varr = Variant(convex_polygon->get_points());
						Geometry::MeshData md;

						Error err = ConvexHullComputer::convex_hull(varr, md);

						if (err == OK) {
							PoolVector3Array faces;

							for (int j = 0; j < md.faces.size(); ++j) {
								Geometry::MeshData::Face face = md.faces[j];

								for (int k = 2; k < face.indices.size(); ++k) {
									faces.push_back(md.vertices[face.indices[0]]);
									faces.push_back(md.vertices[face.indices[k - 1]]);
									faces.push_back(md.vertices[face.indices[k]]);
								}
							}

							_add_faces(faces, transform, p_vertices, p_indices);
						}
					}

					HeightMapShape *heightmap_shape = Object::cast_to<HeightMapShape>(*s);
					if (heightmap_shape) {
						int heightmap_depth = heightmap_shape->get_map_depth();
						int heightmap_width = heightmap_shape->get_map_width();

						if (heightmap_depth >= 2 && heightmap_width >= 2) {
							const PoolRealArray &map_data = heightmap_shape->get_map_data();

							Vector2 heightmap_gridsize(heightmap_width - 1, heightmap_depth - 1);
							Vector2 start = heightmap_gridsize * -0.5;

							PoolVector3Array vertex_array;
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
								_add_faces(vertex_array, transform, p_vertices, p_indices);
							}
						}
					}
				}
			}
		}
	}

#ifdef MODULE_GRIDMAP_ENABLED
	GridMap *gridmap = Object::cast_to<GridMap>(p_node);

	if (gridmap) {
		if (p_generate_from != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
			Array meshes = gridmap->get_meshes();
			Transform xform = gridmap->get_global_transform();
			for (int i = 0; i < meshes.size(); i += 2) {
				Ref<Mesh> mesh = meshes[i + 1];
				if (mesh.is_valid()) {
					Transform mesh_xform = meshes[i];
					_add_mesh(mesh, p_navmesh_xform * xform * mesh_xform, p_vertices, p_indices);
				}
			}
		}

		if (p_generate_from != NavigationMesh::PARSED_GEOMETRY_MESH_INSTANCES && (gridmap->get_collision_layer() & p_collision_mask)) {
			Array shapes = gridmap->get_collision_shapes();
			for (int i = 0; i < shapes.size(); i += 2) {
				RID shape = shapes[i + 1];
				PhysicsServer::ShapeType type = PhysicsServer::get_singleton()->shape_get_type(shape);
				Variant data = PhysicsServer::get_singleton()->shape_get_data(shape);

				switch (type) {
					case PhysicsServer::SHAPE_SPHERE: {
						real_t radius = data;
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						SphereMesh::create_mesh_array(arr, radius, radius * 2.0);
						_add_mesh_array(arr, shapes[i], p_vertices, p_indices);
					} break;
					case PhysicsServer::SHAPE_BOX: {
						Vector3 extents = data;
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						CubeMesh::create_mesh_array(arr, extents * 2.0);
						_add_mesh_array(arr, shapes[i], p_vertices, p_indices);
					} break;
					case PhysicsServer::SHAPE_CAPSULE: {
						Dictionary dict = data;
						real_t radius = dict["radius"];
						real_t height = dict["height"];
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						CapsuleMesh::create_mesh_array(arr, radius, height * 0.5);
						_add_mesh_array(arr, shapes[i], p_vertices, p_indices);
					} break;
					case PhysicsServer::SHAPE_CYLINDER: {
						Dictionary dict = data;
						real_t radius = dict["radius"];
						real_t height = dict["height"];
						Array arr;
						arr.resize(VS::ARRAY_MAX);
						CylinderMesh::create_mesh_array(arr, radius, radius, height);
						_add_mesh_array(arr, shapes[i], p_vertices, p_indices);
					} break;
					case PhysicsServer::SHAPE_CONVEX_POLYGON: {
						PoolVector3Array vertices = data;
						Geometry::MeshData md;

						Error err = ConvexHullComputer::convex_hull(vertices, md);

						if (err == OK) {
							PoolVector3Array faces;

							for (int j = 0; j < md.faces.size(); ++j) {
								Geometry::MeshData::Face face = md.faces[j];

								for (int k = 2; k < face.indices.size(); ++k) {
									faces.push_back(md.vertices[face.indices[0]]);
									faces.push_back(md.vertices[face.indices[k - 1]]);
									faces.push_back(md.vertices[face.indices[k]]);
								}
							}

							_add_faces(faces, shapes[i], p_vertices, p_indices);
						}
					} break;
					case PhysicsServer::SHAPE_CONCAVE_POLYGON: {
						PoolVector3Array faces = data;
						_add_faces(faces, shapes[i], p_vertices, p_indices);
					} break;
					case PhysicsServer::SHAPE_HEIGHTMAP: {
						Dictionary dict = data;
						///< dict( int:"width", int:"depth",float:"cell_size", float_array:"heights"
						int heightmap_depth = dict["depth"];
						int heightmap_width = dict["width"];

						if (heightmap_depth >= 2 && heightmap_width >= 2) {
							const PoolRealArray &map_data = dict["heights"];

							Vector2 heightmap_gridsize(heightmap_width - 1, heightmap_depth - 1);
							Vector2 start = heightmap_gridsize * -0.5;

							PoolVector3Array vertex_array;
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
								_add_faces(vertex_array, shapes[i], p_vertices, p_indices);
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
#endif

	if (p_recurse_children) {
		for (int i = 0; i < p_node->get_child_count(); i++) {
			_parse_geometry(p_navmesh_xform, p_node->get_child(i), p_vertices, p_indices, p_generate_from, p_collision_mask, p_recurse_children);
		}
	}
}

void NavigationMeshGenerator::_convert_detail_mesh_to_native_navigation_mesh(const rcPolyMeshDetail *p_detail_mesh, Ref<NavigationMesh> p_nav_mesh) {
	PoolVector<Vector3> nav_vertices;

	for (int i = 0; i < p_detail_mesh->nverts; i++) {
		const float *v = &p_detail_mesh->verts[i * 3];
		nav_vertices.append(Vector3(v[0], v[1], v[2]));
	}
	p_nav_mesh->set_vertices(nav_vertices);

	for (int i = 0; i < p_detail_mesh->nmeshes; i++) {
		const unsigned int *m = &p_detail_mesh->meshes[i * 4];
		const unsigned int bverts = m[0];
		const unsigned int btris = m[2];
		const unsigned int ntris = m[3];
		const unsigned char *tris = &p_detail_mesh->tris[btris * 4];
		for (unsigned int j = 0; j < ntris; j++) {
			Vector<int> nav_indices;
			nav_indices.resize(3);
			// Polygon order in recast is opposite than godot's
			nav_indices.write[0] = ((int)(bverts + tris[j * 4 + 0]));
			nav_indices.write[1] = ((int)(bverts + tris[j * 4 + 2]));
			nav_indices.write[2] = ((int)(bverts + tris[j * 4 + 1]));
			p_nav_mesh->add_polygon(nav_indices);
		}
	}
}

void NavigationMeshGenerator::_build_recast_navigation_mesh(
		Ref<NavigationMesh> p_nav_mesh,
#ifdef TOOLS_ENABLED
		EditorProgress *ep,
#endif
		rcHeightfield *hf,
		rcCompactHeightfield *chf,
		rcContourSet *cset,
		rcPolyMesh *poly_mesh,
		rcPolyMeshDetail *detail_mesh,
		Vector<float> &vertices,
		Vector<int> &indices) {
	rcContext ctx;

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Setting up Configuration..."), 1);
#endif

	const float *verts = vertices.ptr();
	const int nverts = vertices.size() / 3;
	const int *tris = indices.ptr();
	const int ntris = indices.size() / 3;

	float bmin[3], bmax[3];
	rcCalcBounds(verts, nverts, bmin, bmax);

	rcConfig cfg;
	memset(&cfg, 0, sizeof(cfg));

	cfg.cs = p_nav_mesh->get_cell_size();
	cfg.ch = p_nav_mesh->get_cell_height();
	cfg.walkableSlopeAngle = p_nav_mesh->get_agent_max_slope();
	cfg.walkableHeight = (int)Math::ceil(p_nav_mesh->get_agent_height() / cfg.ch);
	cfg.walkableClimb = (int)Math::floor(p_nav_mesh->get_agent_max_climb() / cfg.ch);
	cfg.walkableRadius = (int)Math::ceil(p_nav_mesh->get_agent_radius() / cfg.cs);
	cfg.maxEdgeLen = (int)(p_nav_mesh->get_edge_max_length() / p_nav_mesh->get_cell_size());
	cfg.maxSimplificationError = p_nav_mesh->get_edge_max_error();
	cfg.minRegionArea = (int)(p_nav_mesh->get_region_min_size() * p_nav_mesh->get_region_min_size());
	cfg.mergeRegionArea = (int)(p_nav_mesh->get_region_merge_size() * p_nav_mesh->get_region_merge_size());
	cfg.maxVertsPerPoly = (int)p_nav_mesh->get_verts_per_poly();
	cfg.detailSampleDist = MAX(p_nav_mesh->get_cell_size() * p_nav_mesh->get_detail_sample_distance(), 0.1f);
	cfg.detailSampleMaxError = p_nav_mesh->get_cell_height() * p_nav_mesh->get_detail_sample_max_error();

	cfg.bmin[0] = bmin[0];
	cfg.bmin[1] = bmin[1];
	cfg.bmin[2] = bmin[2];
	cfg.bmax[0] = bmax[0];
	cfg.bmax[1] = bmax[1];
	cfg.bmax[2] = bmax[2];

	AABB baking_aabb = p_nav_mesh->get_filter_baking_aabb();

	bool aabb_has_no_volume = baking_aabb.has_no_area();

	if (!aabb_has_no_volume) {
		Vector3 baking_aabb_offset = p_nav_mesh->get_filter_baking_aabb_offset();

		cfg.bmin[0] = baking_aabb.position[0] + baking_aabb_offset.x;
		cfg.bmin[1] = baking_aabb.position[1] + baking_aabb_offset.y;
		cfg.bmin[2] = baking_aabb.position[2] + baking_aabb_offset.z;
		cfg.bmax[0] = cfg.bmin[0] + baking_aabb.size[0];
		cfg.bmax[1] = cfg.bmin[1] + baking_aabb.size[1];
		cfg.bmax[2] = cfg.bmin[2] + baking_aabb.size[2];
	}

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Calculating grid size..."), 2);
#endif
	rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &cfg.width, &cfg.height);

	// ~30000000 seems to be around sweetspot where Editor baking breaks
	if ((cfg.width * cfg.height) > 30000000) {
		WARN_PRINT("NavigationMesh baking process will likely fail."
				   "\nSource geometry is suspiciously big for the current Cell Size and Cell Height in the NavMesh Resource bake settings."
				   "\nIf baking does not fail, the resulting NavigationMesh will create serious pathfinding performance issues."
				   "\nIt is advised to increase Cell Size and/or Cell Height in the NavMesh Resource bake settings or reduce the size / scale of the source geometry.");
	}

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Creating heightfield..."), 3);
#endif
	hf = rcAllocHeightfield();

	ERR_FAIL_COND(!hf);
	ERR_FAIL_COND(!rcCreateHeightfield(&ctx, *hf, cfg.width, cfg.height, cfg.bmin, cfg.bmax, cfg.cs, cfg.ch));

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Marking walkable triangles..."), 4);
#endif
	{
		Vector<unsigned char> tri_areas;
		tri_areas.resize(ntris);

		ERR_FAIL_COND(tri_areas.size() == 0);

		memset(tri_areas.ptrw(), 0, ntris * sizeof(unsigned char));
		rcMarkWalkableTriangles(&ctx, cfg.walkableSlopeAngle, verts, nverts, tris, ntris, tri_areas.ptrw());

		ERR_FAIL_COND(!rcRasterizeTriangles(&ctx, verts, nverts, tris, tri_areas.ptr(), ntris, *hf, cfg.walkableClimb));
	}

	if (p_nav_mesh->get_filter_low_hanging_obstacles()) {
		rcFilterLowHangingWalkableObstacles(&ctx, cfg.walkableClimb, *hf);
	}
	if (p_nav_mesh->get_filter_ledge_spans()) {
		rcFilterLedgeSpans(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf);
	}
	if (p_nav_mesh->get_filter_walkable_low_height_spans()) {
		rcFilterWalkableLowHeightSpans(&ctx, cfg.walkableHeight, *hf);
	}

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Constructing compact heightfield..."), 5);
#endif

	chf = rcAllocCompactHeightfield();

	ERR_FAIL_COND(!chf);
	ERR_FAIL_COND(!rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf, *chf));

	rcFreeHeightField(hf);
	hf = 0;

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Eroding walkable area..."), 6);
#endif

	ERR_FAIL_COND(!rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf));

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Partitioning..."), 7);
#endif

	if (p_nav_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_WATERSHED) {
		ERR_FAIL_COND(!rcBuildDistanceField(&ctx, *chf));
		ERR_FAIL_COND(!rcBuildRegions(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea));
	} else if (p_nav_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_MONOTONE) {
		ERR_FAIL_COND(!rcBuildRegionsMonotone(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea));
	} else {
		ERR_FAIL_COND(!rcBuildLayerRegions(&ctx, *chf, 0, cfg.minRegionArea));
	}

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Creating contours..."), 8);
#endif

	cset = rcAllocContourSet();

	ERR_FAIL_COND(!cset);
	ERR_FAIL_COND(!rcBuildContours(&ctx, *chf, cfg.maxSimplificationError, cfg.maxEdgeLen, *cset));

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Creating polymesh..."), 9);
#endif

	poly_mesh = rcAllocPolyMesh();
	ERR_FAIL_COND(!poly_mesh);
	ERR_FAIL_COND(!rcBuildPolyMesh(&ctx, *cset, cfg.maxVertsPerPoly, *poly_mesh));

	detail_mesh = rcAllocPolyMeshDetail();
	ERR_FAIL_COND(!detail_mesh);
	ERR_FAIL_COND(!rcBuildPolyMeshDetail(&ctx, *poly_mesh, *chf, cfg.detailSampleDist, cfg.detailSampleMaxError, *detail_mesh));

	rcFreeCompactHeightfield(chf);
	chf = 0;
	rcFreeContourSet(cset);
	cset = 0;

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Converting to native navigation mesh..."), 10);
#endif

	_convert_detail_mesh_to_native_navigation_mesh(detail_mesh, p_nav_mesh);

	rcFreePolyMesh(poly_mesh);
	poly_mesh = 0;
	rcFreePolyMeshDetail(detail_mesh);
	detail_mesh = 0;
}

NavigationMeshGenerator *NavigationMeshGenerator::get_singleton() {
	return singleton;
}

NavigationMeshGenerator::NavigationMeshGenerator() {
	singleton = this;
}

NavigationMeshGenerator::~NavigationMeshGenerator() {
}

void NavigationMeshGenerator::bake(Ref<NavigationMesh> p_nav_mesh, Node *p_node) {
	ERR_FAIL_COND_MSG(!p_nav_mesh.is_valid(), "Invalid Navigation Mesh");

#ifdef TOOLS_ENABLED
	EditorProgress *ep(nullptr);
	// FIXME
#endif
#if 0
	// After discussion on devchat disabled EditorProgress for now as it is not thread-safe and uses hacks and Main::iteration() for steps.
	// EditorProgress randomly crashes the Engine when the bake function is used with a thread e.g. inside Editor with a tool script and procedural navigation
	// This was not a problem in older versions as previously Godot was unable to (re)bake NavigationMesh at runtime.
	// If EditorProgress is fixed and made thread-safe this should be enabled again.
	if (Engine::get_singleton()->is_editor_hint()) {
		ep = memnew(EditorProgress("bake", TTR("Navigation Mesh Generator Setup:"), 11));
	}

	if (ep)
		ep->step(TTR("Parsing Geometry..."), 0);
#endif

	Vector<float> vertices;
	Vector<int> indices;

	List<Node *> parse_nodes;

	if (p_nav_mesh->get_source_geometry_mode() == NavigationMesh::SOURCE_GEOMETRY_NAVMESH_CHILDREN) {
		parse_nodes.push_back(p_node);
	} else {
		p_node->get_tree()->get_nodes_in_group(p_nav_mesh->get_source_group_name(), &parse_nodes);
	}

	Transform navmesh_xform = Object::cast_to<Spatial>(p_node)->get_global_transform().affine_inverse();
	for (const List<Node *>::Element *E = parse_nodes.front(); E; E = E->next()) {
		NavigationMesh::ParsedGeometryType geometry_type = p_nav_mesh->get_parsed_geometry_type();
		uint32_t collision_mask = p_nav_mesh->get_collision_mask();
		bool recurse_children = p_nav_mesh->get_source_geometry_mode() != NavigationMesh::SOURCE_GEOMETRY_GROUPS_EXPLICIT;
		_parse_geometry(navmesh_xform, E->get(), vertices, indices, geometry_type, collision_mask, recurse_children);
	}

	if (vertices.size() > 0 && indices.size() > 0) {
		rcHeightfield *hf = nullptr;
		rcCompactHeightfield *chf = nullptr;
		rcContourSet *cset = nullptr;
		rcPolyMesh *poly_mesh = nullptr;
		rcPolyMeshDetail *detail_mesh = nullptr;

		_build_recast_navigation_mesh(
				p_nav_mesh,
#ifdef TOOLS_ENABLED
				ep,
#endif
				hf,
				chf,
				cset,
				poly_mesh,
				detail_mesh,
				vertices,
				indices);

		rcFreeHeightField(hf);
		hf = 0;

		rcFreeCompactHeightfield(chf);
		chf = 0;

		rcFreeContourSet(cset);
		cset = 0;

		rcFreePolyMesh(poly_mesh);
		poly_mesh = 0;

		rcFreePolyMeshDetail(detail_mesh);
		detail_mesh = 0;
	}

#ifdef TOOLS_ENABLED
	if (ep)
		ep->step(TTR("Done!"), 11);

	if (ep)
		memdelete(ep);
#endif

	p_nav_mesh->property_list_changed_notify();
}

void NavigationMeshGenerator::clear(Ref<NavigationMesh> p_nav_mesh) {
	if (p_nav_mesh.is_valid()) {
		p_nav_mesh->clear_polygons();
		p_nav_mesh->set_vertices(PoolVector<Vector3>());
	}
}

void NavigationMeshGenerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bake", "nav_mesh", "root_node"), &NavigationMeshGenerator::bake);
	ClassDB::bind_method(D_METHOD("clear", "nav_mesh"), &NavigationMeshGenerator::clear);
}

#endif
