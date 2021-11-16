/*************************************************************************/
/*  navigation_mesh_generator.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "navigation_mesh_generator.h"

#include "core/math/convex_hull.h"
#include "core/os/thread.h"
#include "editor/editor_settings.h"
#include "scene/3d/collision_shape.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/physics_body.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/cylinder_shape.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/shape.h"
#include "scene/resources/sphere_shape.h"

#include "modules/modules_enabled.gen.h" // For csg, gridmap.
#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#ifdef MODULE_GRIDMAP_ENABLED
#include "modules/gridmap/grid_map.h"
#endif

EditorNavigationMeshGenerator *EditorNavigationMeshGenerator::singleton = nullptr;

void EditorNavigationMeshGenerator::_add_vertex(const Vector3 &p_vec3, Vector<float> &p_verticies) {
	p_verticies.push_back(p_vec3.x);
	p_verticies.push_back(p_vec3.y);
	p_verticies.push_back(p_vec3.z);
}

void EditorNavigationMeshGenerator::_add_mesh(const Ref<Mesh> &p_mesh, const Transform &p_xform, Vector<float> &p_verticies, Vector<int> &p_indices) {
	int current_vertex_count;

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		current_vertex_count = p_verticies.size() / 3;

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
				_add_vertex(p_xform.xform(vr[j]), p_verticies);
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
				_add_vertex(p_xform.xform(vr[j * 3 + 0]), p_verticies);
				_add_vertex(p_xform.xform(vr[j * 3 + 2]), p_verticies);
				_add_vertex(p_xform.xform(vr[j * 3 + 1]), p_verticies);

				p_indices.push_back(current_vertex_count + (j * 3 + 0));
				p_indices.push_back(current_vertex_count + (j * 3 + 1));
				p_indices.push_back(current_vertex_count + (j * 3 + 2));
			}
		}
	}
}

void EditorNavigationMeshGenerator::_add_faces(const PoolVector3Array &p_faces, const Transform &p_xform, Vector<float> &p_verticies, Vector<int> &p_indices) {
	int face_count = p_faces.size() / 3;
	int current_vertex_count = p_verticies.size() / 3;

	for (int j = 0; j < face_count; j++) {
		_add_vertex(p_xform.xform(p_faces[j * 3 + 0]), p_verticies);
		_add_vertex(p_xform.xform(p_faces[j * 3 + 1]), p_verticies);
		_add_vertex(p_xform.xform(p_faces[j * 3 + 2]), p_verticies);

		p_indices.push_back(current_vertex_count + (j * 3 + 0));
		p_indices.push_back(current_vertex_count + (j * 3 + 2));
		p_indices.push_back(current_vertex_count + (j * 3 + 1));
	}
}

void EditorNavigationMeshGenerator::_parse_geometry(Transform p_accumulated_transform, Node *p_node, Vector<float> &p_verticies, Vector<int> &p_indices, NavigationMesh::ParsedGeometryType p_generate_from, uint32_t p_collision_mask, bool p_recurse_children) {
	if (Object::cast_to<MeshInstance>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		MeshInstance *mesh_instance = Object::cast_to<MeshInstance>(p_node);
		Ref<Mesh> mesh = mesh_instance->get_mesh();
		if (mesh.is_valid()) {
			_add_mesh(mesh, p_accumulated_transform * mesh_instance->get_transform(), p_verticies, p_indices);
		}
	}

#ifdef MODULE_CSG_ENABLED
	if (Object::cast_to<CSGShape>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		CSGShape *csg_shape = Object::cast_to<CSGShape>(p_node);
		Array meshes = csg_shape->get_meshes();
		if (!meshes.empty()) {
			Ref<Mesh> mesh = meshes[1];
			if (mesh.is_valid()) {
				_add_mesh(mesh, p_accumulated_transform * csg_shape->get_transform(), p_verticies, p_indices);
			}
		}
	}
#endif

	if (Object::cast_to<StaticBody>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_MESH_INSTANCES) {
		StaticBody *static_body = Object::cast_to<StaticBody>(p_node);

		if (static_body->get_collision_layer() & p_collision_mask) {
			for (int i = 0; i < p_node->get_child_count(); ++i) {
				Node *child = p_node->get_child(i);
				if (Object::cast_to<CollisionShape>(child)) {
					CollisionShape *col_shape = Object::cast_to<CollisionShape>(child);

					Transform transform = p_accumulated_transform * static_body->get_transform() * col_shape->get_transform();

					Ref<Mesh> mesh;
					Ref<Shape> s = col_shape->get_shape();

					BoxShape *box = Object::cast_to<BoxShape>(*s);
					if (box) {
						Ref<CubeMesh> cube_mesh;
						cube_mesh.instance();
						cube_mesh->set_size(box->get_extents() * 2.0);
						mesh = cube_mesh;
					}

					CapsuleShape *capsule = Object::cast_to<CapsuleShape>(*s);
					if (capsule) {
						Ref<CapsuleMesh> capsule_mesh;
						capsule_mesh.instance();
						capsule_mesh->set_radius(capsule->get_radius());
						capsule_mesh->set_mid_height(capsule->get_height() / 2.0);
						mesh = capsule_mesh;
					}

					CylinderShape *cylinder = Object::cast_to<CylinderShape>(*s);
					if (cylinder) {
						Ref<CylinderMesh> cylinder_mesh;
						cylinder_mesh.instance();
						cylinder_mesh->set_height(cylinder->get_height());
						cylinder_mesh->set_bottom_radius(cylinder->get_radius());
						cylinder_mesh->set_top_radius(cylinder->get_radius());
						mesh = cylinder_mesh;
					}

					SphereShape *sphere = Object::cast_to<SphereShape>(*s);
					if (sphere) {
						Ref<SphereMesh> sphere_mesh;
						sphere_mesh.instance();
						sphere_mesh->set_radius(sphere->get_radius());
						sphere_mesh->set_height(sphere->get_radius() * 2.0);
						mesh = sphere_mesh;
					}

					ConcavePolygonShape *concave_polygon = Object::cast_to<ConcavePolygonShape>(*s);
					if (concave_polygon) {
						_add_faces(concave_polygon->get_faces(), transform, p_verticies, p_indices);
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

							_add_faces(faces, transform, p_verticies, p_indices);
						}
					}

					if (mesh.is_valid()) {
						_add_mesh(mesh, transform, p_verticies, p_indices);
					}
				}
			}
		}
	}

#ifdef MODULE_GRIDMAP_ENABLED
	if (Object::cast_to<GridMap>(p_node) && p_generate_from != NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS) {
		GridMap *gridmap_instance = Object::cast_to<GridMap>(p_node);
		Array meshes = gridmap_instance->get_meshes();
		Transform xform = gridmap_instance->get_transform();
		for (int i = 0; i < meshes.size(); i += 2) {
			Ref<Mesh> mesh = meshes[i + 1];
			if (mesh.is_valid()) {
				_add_mesh(mesh, p_accumulated_transform * xform * meshes[i], p_verticies, p_indices);
			}
		}
	}
#endif

	if (Object::cast_to<Spatial>(p_node)) {
		Spatial *spatial = Object::cast_to<Spatial>(p_node);
		p_accumulated_transform = p_accumulated_transform * spatial->get_transform();
	}

	if (p_recurse_children) {
		for (int i = 0; i < p_node->get_child_count(); i++) {
			_parse_geometry(p_accumulated_transform, p_node->get_child(i), p_verticies, p_indices, p_generate_from, p_collision_mask, p_recurse_children);
		}
	}
}

void EditorNavigationMeshGenerator::_convert_detail_mesh_to_native_navigation_mesh(const rcPolyMeshDetail *p_detail_mesh, Ref<NavigationMesh> p_nav_mesh) {
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

void EditorNavigationMeshGenerator::_build_recast_navigation_mesh(Ref<NavigationMesh> p_nav_mesh, EditorProgress *ep,
		rcHeightfield *hf, rcCompactHeightfield *chf, rcContourSet *cset, rcPolyMesh *poly_mesh, rcPolyMeshDetail *detail_mesh,
		Vector<float> &vertices, Vector<int> &indices) {
	rcContext ctx;
	ep->step(TTR("Setting up Configuration..."), 1);

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
	cfg.detailSampleDist = p_nav_mesh->get_detail_sample_distance() < 0.9f ? 0 : p_nav_mesh->get_cell_size() * p_nav_mesh->get_detail_sample_distance();
	cfg.detailSampleMaxError = p_nav_mesh->get_cell_height() * p_nav_mesh->get_detail_sample_max_error();

	cfg.bmin[0] = bmin[0];
	cfg.bmin[1] = bmin[1];
	cfg.bmin[2] = bmin[2];
	cfg.bmax[0] = bmax[0];
	cfg.bmax[1] = bmax[1];
	cfg.bmax[2] = bmax[2];

	ep->step(TTR("Calculating grid size..."), 2);
	rcCalcGridSize(cfg.bmin, cfg.bmax, cfg.cs, &cfg.width, &cfg.height);

	ep->step(TTR("Creating heightfield..."), 3);
	hf = rcAllocHeightfield();

	ERR_FAIL_COND(!hf);
	ERR_FAIL_COND(!rcCreateHeightfield(&ctx, *hf, cfg.width, cfg.height, cfg.bmin, cfg.bmax, cfg.cs, cfg.ch));

	ep->step(TTR("Marking walkable triangles..."), 4);
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

	ep->step(TTR("Constructing compact heightfield..."), 5);

	chf = rcAllocCompactHeightfield();

	ERR_FAIL_COND(!chf);
	ERR_FAIL_COND(!rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf, *chf));

	rcFreeHeightField(hf);
	hf = nullptr;

	ep->step(TTR("Eroding walkable area..."), 6);
	ERR_FAIL_COND(!rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf));

	ep->step(TTR("Partitioning..."), 7);
	if (p_nav_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_WATERSHED) {
		ERR_FAIL_COND(!rcBuildDistanceField(&ctx, *chf));
		ERR_FAIL_COND(!rcBuildRegions(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea));
	} else if (p_nav_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_MONOTONE) {
		ERR_FAIL_COND(!rcBuildRegionsMonotone(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea));
	} else {
		ERR_FAIL_COND(!rcBuildLayerRegions(&ctx, *chf, 0, cfg.minRegionArea));
	}

	ep->step(TTR("Creating contours..."), 8);

	cset = rcAllocContourSet();

	ERR_FAIL_COND(!cset);
	ERR_FAIL_COND(!rcBuildContours(&ctx, *chf, cfg.maxSimplificationError, cfg.maxEdgeLen, *cset));

	ep->step(TTR("Creating polymesh..."), 9);

	poly_mesh = rcAllocPolyMesh();
	ERR_FAIL_COND(!poly_mesh);
	ERR_FAIL_COND(!rcBuildPolyMesh(&ctx, *cset, cfg.maxVertsPerPoly, *poly_mesh));

	detail_mesh = rcAllocPolyMeshDetail();
	ERR_FAIL_COND(!detail_mesh);
	ERR_FAIL_COND(!rcBuildPolyMeshDetail(&ctx, *poly_mesh, *chf, cfg.detailSampleDist, cfg.detailSampleMaxError, *detail_mesh));

	rcFreeCompactHeightfield(chf);
	chf = nullptr;
	rcFreeContourSet(cset);
	cset = nullptr;

	ep->step(TTR("Converting to native navigation mesh..."), 10);

	_convert_detail_mesh_to_native_navigation_mesh(detail_mesh, p_nav_mesh);

	rcFreePolyMesh(poly_mesh);
	poly_mesh = nullptr;
	rcFreePolyMeshDetail(detail_mesh);
	detail_mesh = nullptr;
}

EditorNavigationMeshGenerator *EditorNavigationMeshGenerator::get_singleton() {
	return singleton;
}

EditorNavigationMeshGenerator::EditorNavigationMeshGenerator() {
	singleton = this;
}

EditorNavigationMeshGenerator::~EditorNavigationMeshGenerator() {
}

void EditorNavigationMeshGenerator::bake(Ref<NavigationMesh> p_nav_mesh, Node *p_node) {
	if (!Engine::get_singleton()->is_editor_hint()) {
		ERR_PRINT("Invoking EditorNavigationMeshGenerator::bake(...) in-game is not supported in Godot 3.2 or below. Aborting bake...");
		return;
	}

	ERR_FAIL_COND(!p_nav_mesh.is_valid());

	EditorProgress ep("bake", TTR("Navigation Mesh Generator Setup:"), 11);
	ep.step(TTR("Parsing Geometry..."), 0);

	Vector<float> vertices;
	Vector<int> indices;

	List<Node *> parse_nodes;

	if (p_nav_mesh->get_source_geometry_mode() == NavigationMesh::SOURCE_GEOMETRY_NAVMESH_CHILDREN) {
		parse_nodes.push_back(p_node);
	} else {
		p_node->get_tree()->get_nodes_in_group(p_nav_mesh->get_source_group_name(), &parse_nodes);
	}

	Transform navmesh_xform = Object::cast_to<Spatial>(p_node)->get_transform().affine_inverse();
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

		_build_recast_navigation_mesh(p_nav_mesh, &ep, hf, chf, cset, poly_mesh, detail_mesh, vertices, indices);

		rcFreeHeightField(hf);
		hf = nullptr;

		rcFreeCompactHeightfield(chf);
		chf = nullptr;

		rcFreeContourSet(cset);
		cset = nullptr;

		rcFreePolyMesh(poly_mesh);
		poly_mesh = nullptr;

		rcFreePolyMeshDetail(detail_mesh);
		detail_mesh = nullptr;
	}
	ep.step(TTR("Done!"), 11);
}

void EditorNavigationMeshGenerator::clear(Ref<NavigationMesh> p_nav_mesh) {
	if (p_nav_mesh.is_valid()) {
		p_nav_mesh->clear_polygons();
		p_nav_mesh->set_vertices(PoolVector<Vector3>());
	}
}

void EditorNavigationMeshGenerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bake", "nav_mesh", "root_node"), &EditorNavigationMeshGenerator::bake);
	ClassDB::bind_method(D_METHOD("clear", "nav_mesh"), &EditorNavigationMeshGenerator::clear);
}
