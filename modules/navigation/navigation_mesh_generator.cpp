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

#ifndef _3D_DISABLED

#include "navigation_mesh_generator.h"

#include "core/math/convex_hull.h"
#include "core/os/thread.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
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

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#endif

#include "modules/modules_enabled.gen.h" // For csg, gridmap.

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#ifdef MODULE_GRIDMAP_ENABLED
#include "modules/gridmap/grid_map.h"
#endif

NavigationMeshGenerator *NavigationMeshGenerator::singleton = nullptr;

void NavigationMeshGenerator::_add_vertex(const Vector3 &p_vec3, Vector<float> &p_vertices) {
	p_vertices.push_back(p_vec3.x);
	p_vertices.push_back(p_vec3.y);
	p_vertices.push_back(p_vec3.z);
}

void NavigationMeshGenerator::_add_mesh(const Ref<Mesh> &p_mesh, const Transform3D &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices) {
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

		Vector<Vector3> mesh_vertices = a[Mesh::ARRAY_VERTEX];
		const Vector3 *vr = mesh_vertices.ptr();

		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
			Vector<int> mesh_indices = a[Mesh::ARRAY_INDEX];
			const int *ir = mesh_indices.ptr();

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

void NavigationMeshGenerator::_add_mesh_array(const Array &p_array, const Transform3D &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices) {
	Vector<Vector3> mesh_vertices = p_array[Mesh::ARRAY_VERTEX];
	const Vector3 *vr = mesh_vertices.ptr();

	Vector<int> mesh_indices = p_array[Mesh::ARRAY_INDEX];
	const int *ir = mesh_indices.ptr();

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

void NavigationMeshGenerator::_add_faces(const PackedVector3Array &p_faces, const Transform3D &p_xform, Vector<float> &p_vertices, Vector<int> &p_indices) {
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

void NavigationMeshGenerator::_parse_geometry_node(const Transform3D &p_navmesh_transform, Ref<NavigationMesh> p_navigationmesh, Node *p_node, Vector<float> &p_vertices, Vector<int> &p_indices, bool p_recurse_children) {
	// find a geometry parser that handles the node
	for (uint32_t i(0); i < NavigationMeshGenerator::get_singleton()->geometry_3d_parsers.size(); i++) {
		if (NavigationMeshGenerator::get_singleton()->geometry_3d_parsers[i]->parses_node(p_node)) {
			NavigationMeshGenerator::get_singleton()->geometry_3d_parsers[i]->parse_node_geometry(p_navmesh_transform, p_navigationmesh, p_node, p_vertices, p_indices);
			break;
		};
	};

	if (p_recurse_children) {
		for (int i = 0; i < p_node->get_child_count(); i++) {
			_parse_geometry_node(p_navmesh_transform, p_navigationmesh, p_node->get_child(i), p_vertices, p_indices, p_recurse_children);
		}
	}
}

void NavigationMeshGenerator::_convert_detail_mesh_to_native_navigation_mesh(const rcPolyMeshDetail *p_detail_mesh, Ref<NavigationMesh> p_navigation_mesh) {
	Vector<Vector3> nav_vertices;

	for (int i = 0; i < p_detail_mesh->nverts; i++) {
		const float *v = &p_detail_mesh->verts[i * 3];
		nav_vertices.push_back(Vector3(v[0], v[1], v[2]));
	}
	p_navigation_mesh->set_vertices(nav_vertices);

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
			p_navigation_mesh->add_polygon(nav_indices);
		}
	}
}

void NavigationMeshGenerator::_build_recast_navigation_mesh(
		Ref<NavigationMesh> p_navigation_mesh,
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
	if (ep) {
		ep->step(TTR("Setting up Configuration..."), 1);
	}
#endif

	const float *verts = vertices.ptr();
	const int nverts = vertices.size() / 3;
	const int *tris = indices.ptr();
	const int ntris = indices.size() / 3;

	float bmin[3], bmax[3];
	rcCalcBounds(verts, nverts, bmin, bmax);

	rcConfig cfg;
	memset(&cfg, 0, sizeof(cfg));

	cfg.cs = p_navigation_mesh->get_cell_size();
	cfg.ch = p_navigation_mesh->get_cell_height();
	cfg.walkableSlopeAngle = p_navigation_mesh->get_agent_max_slope();
	cfg.walkableHeight = (int)Math::ceil(p_navigation_mesh->get_agent_height() / cfg.ch);
	cfg.walkableClimb = (int)Math::floor(p_navigation_mesh->get_agent_max_climb() / cfg.ch);
	cfg.walkableRadius = (int)Math::ceil(p_navigation_mesh->get_agent_radius() / cfg.cs);
	cfg.maxEdgeLen = (int)(p_navigation_mesh->get_edge_max_length() / p_navigation_mesh->get_cell_size());
	cfg.maxSimplificationError = p_navigation_mesh->get_edge_max_error();
	cfg.minRegionArea = (int)(p_navigation_mesh->get_region_min_size() * p_navigation_mesh->get_region_min_size());
	cfg.mergeRegionArea = (int)(p_navigation_mesh->get_region_merge_size() * p_navigation_mesh->get_region_merge_size());
	cfg.maxVertsPerPoly = (int)p_navigation_mesh->get_vertices_per_polyon();
	cfg.detailSampleDist = MAX(p_navigation_mesh->get_cell_size() * p_navigation_mesh->get_detail_sample_distance(), 0.1f);
	cfg.detailSampleMaxError = p_navigation_mesh->get_cell_height() * p_navigation_mesh->get_detail_sample_max_error();

	if (!Math::is_equal_approx((float)cfg.walkableHeight * cfg.ch, p_navigation_mesh->get_agent_height())) {
		WARN_PRINT("Property agent_height is ceiled to cell_height voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.walkableClimb * cfg.ch, p_navigation_mesh->get_agent_max_climb())) {
		WARN_PRINT("Property agent_max_climb is floored to cell_height voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.walkableRadius * cfg.cs, p_navigation_mesh->get_agent_radius())) {
		WARN_PRINT("Property agent_radius is ceiled to cell_size voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.maxEdgeLen * cfg.cs, p_navigation_mesh->get_edge_max_length())) {
		WARN_PRINT("Property edge_max_length is rounded to cell_size voxel units and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.minRegionArea, p_navigation_mesh->get_region_min_size() * p_navigation_mesh->get_region_min_size())) {
		WARN_PRINT("Property region_min_size is converted to int and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.mergeRegionArea, p_navigation_mesh->get_region_merge_size() * p_navigation_mesh->get_region_merge_size())) {
		WARN_PRINT("Property region_merge_size is converted to int and loses precision.");
	}
	if (!Math::is_equal_approx((float)cfg.maxVertsPerPoly, p_navigation_mesh->get_vertices_per_polyon())) {
		WARN_PRINT("Property vertices_per_polyon is converted to int and loses precision.");
	}
	if (p_navigation_mesh->get_cell_size() * p_navigation_mesh->get_detail_sample_distance() < 0.1f) {
		WARN_PRINT("Property detail_sample_distance is clamped to 0.1 world units as the resulting value from multiplying with cell_size is too low.");
	}

	cfg.bmin[0] = bmin[0];
	cfg.bmin[1] = bmin[1];
	cfg.bmin[2] = bmin[2];
	cfg.bmax[0] = bmax[0];
	cfg.bmax[1] = bmax[1];
	cfg.bmax[2] = bmax[2];

	AABB baking_aabb = p_navigation_mesh->get_filter_baking_aabb();
	if (baking_aabb.has_volume()) {
		Vector3 baking_aabb_offset = p_navigation_mesh->get_filter_baking_aabb_offset();
		cfg.bmin[0] = baking_aabb.position[0] + baking_aabb_offset.x;
		cfg.bmin[1] = baking_aabb.position[1] + baking_aabb_offset.y;
		cfg.bmin[2] = baking_aabb.position[2] + baking_aabb_offset.z;
		cfg.bmax[0] = cfg.bmin[0] + baking_aabb.size[0];
		cfg.bmax[1] = cfg.bmin[1] + baking_aabb.size[1];
		cfg.bmax[2] = cfg.bmin[2] + baking_aabb.size[2];
	}

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Calculating grid size..."), 2);
	}
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
	if (ep) {
		ep->step(TTR("Creating heightfield..."), 3);
	}
#endif
	hf = rcAllocHeightfield();

	ERR_FAIL_COND(!hf);
	ERR_FAIL_COND(!rcCreateHeightfield(&ctx, *hf, cfg.width, cfg.height, cfg.bmin, cfg.bmax, cfg.cs, cfg.ch));

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Marking walkable triangles..."), 4);
	}
#endif
	{
		Vector<unsigned char> tri_areas;
		tri_areas.resize(ntris);

		ERR_FAIL_COND(tri_areas.size() == 0);

		memset(tri_areas.ptrw(), 0, ntris * sizeof(unsigned char));
		rcMarkWalkableTriangles(&ctx, cfg.walkableSlopeAngle, verts, nverts, tris, ntris, tri_areas.ptrw());

		ERR_FAIL_COND(!rcRasterizeTriangles(&ctx, verts, nverts, tris, tri_areas.ptr(), ntris, *hf, cfg.walkableClimb));
	}

	if (p_navigation_mesh->get_filter_low_hanging_obstacles()) {
		rcFilterLowHangingWalkableObstacles(&ctx, cfg.walkableClimb, *hf);
	}
	if (p_navigation_mesh->get_filter_ledge_spans()) {
		rcFilterLedgeSpans(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf);
	}
	if (p_navigation_mesh->get_filter_walkable_low_height_spans()) {
		rcFilterWalkableLowHeightSpans(&ctx, cfg.walkableHeight, *hf);
	}

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Constructing compact heightfield..."), 5);
	}
#endif

	chf = rcAllocCompactHeightfield();

	ERR_FAIL_COND(!chf);
	ERR_FAIL_COND(!rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf, *chf));

	rcFreeHeightField(hf);
	hf = nullptr;

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Eroding walkable area..."), 6);
	}
#endif

	ERR_FAIL_COND(!rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf));

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Partitioning..."), 7);
	}
#endif

	if (p_navigation_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_WATERSHED) {
		ERR_FAIL_COND(!rcBuildDistanceField(&ctx, *chf));
		ERR_FAIL_COND(!rcBuildRegions(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea));
	} else if (p_navigation_mesh->get_sample_partition_type() == NavigationMesh::SAMPLE_PARTITION_MONOTONE) {
		ERR_FAIL_COND(!rcBuildRegionsMonotone(&ctx, *chf, 0, cfg.minRegionArea, cfg.mergeRegionArea));
	} else {
		ERR_FAIL_COND(!rcBuildLayerRegions(&ctx, *chf, 0, cfg.minRegionArea));
	}

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Creating contours..."), 8);
	}
#endif

	cset = rcAllocContourSet();

	ERR_FAIL_COND(!cset);
	ERR_FAIL_COND(!rcBuildContours(&ctx, *chf, cfg.maxSimplificationError, cfg.maxEdgeLen, *cset));

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Creating polymesh..."), 9);
	}
#endif

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

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Converting to native navigation mesh..."), 10);
	}
#endif

	_convert_detail_mesh_to_native_navigation_mesh(detail_mesh, p_navigation_mesh);

	rcFreePolyMesh(poly_mesh);
	poly_mesh = nullptr;
	rcFreePolyMeshDetail(detail_mesh);
	detail_mesh = nullptr;
}

NavigationMeshGenerator *NavigationMeshGenerator::get_singleton() {
	return singleton;
}

NavigationMeshGenerator::NavigationMeshGenerator() {
	singleton = this;
}

NavigationMeshGenerator::~NavigationMeshGenerator() {
	geometry_3d_parsers.clear();
}

void NavigationMeshGenerator::bake(Ref<NavigationMesh> p_navigation_mesh, Node *p_root_node) {
	ERR_FAIL_COND_MSG(!p_navigation_mesh.is_valid(), "Invalid navigation mesh.");

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

	if (ep) {
		ep->step(TTR("Parsing Geometry..."), 0);
	}
#endif

	Vector<float> vertices;
	Vector<int> indices;

	List<Node *> parse_nodes;

	if (p_navigation_mesh->get_source_geometry_mode() == NavigationMesh::SOURCE_GEOMETRY_ROOT_NODE_CHILDREN) {
		parse_nodes.push_back(p_root_node);
	} else {
		p_root_node->get_tree()->get_nodes_in_group(p_navigation_mesh->get_source_group_name(), &parse_nodes);
	}

	Transform3D navmesh_xform = Object::cast_to<Node3D>(p_root_node)->get_global_transform().affine_inverse();
	bool recurse_children = p_navigation_mesh->get_source_geometry_mode() != NavigationMesh::SOURCE_GEOMETRY_GROUPS_EXPLICIT;
	for (Node *E : parse_nodes) {
		_parse_geometry_node(navmesh_xform, p_navigation_mesh, E, vertices, indices, recurse_children);
	}

	if (vertices.size() > 0 && indices.size() > 0) {
		rcHeightfield *hf = nullptr;
		rcCompactHeightfield *chf = nullptr;
		rcContourSet *cset = nullptr;
		rcPolyMesh *poly_mesh = nullptr;
		rcPolyMeshDetail *detail_mesh = nullptr;

		_build_recast_navigation_mesh(
				p_navigation_mesh,
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

#ifdef TOOLS_ENABLED
	if (ep) {
		ep->step(TTR("Done!"), 11);
	}

	if (ep) {
		memdelete(ep);
	}
#endif
}

void NavigationMeshGenerator::clear(Ref<NavigationMesh> p_navigation_mesh) {
	if (p_navigation_mesh.is_valid()) {
		p_navigation_mesh->clear_polygons();
		p_navigation_mesh->set_vertices(Vector<Vector3>());
	}
}

void NavigationMeshGenerator::register_geometry_parser_3d(const Ref<NavigationGeometryParser3D> &p_geometry_parser) {
	if (NavigationMeshGenerator::get_singleton()->geometry_3d_parsers.find(p_geometry_parser) == -1) {
		// add front so user custom parsers that extend existing node types are called before build-in parsers
		NavigationMeshGenerator::get_singleton()->geometry_3d_parsers.insert(0, p_geometry_parser);
	}
}

void NavigationMeshGenerator::unregister_geometry_parser_3d(const Ref<NavigationGeometryParser3D> &p_geometry_parser) {
	NavigationMeshGenerator::get_singleton()->geometry_3d_parsers.erase(p_geometry_parser);
}

void NavigationMeshGenerator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bake", "navigation_mesh", "root_node"), &NavigationMeshGenerator::bake);
	ClassDB::bind_method(D_METHOD("clear", "navigation_mesh"), &NavigationMeshGenerator::clear);

	ClassDB::bind_method(D_METHOD("register_geometry_parser_3d", "geometry_parser"), &NavigationMeshGenerator::register_geometry_parser_3d);
	ClassDB::bind_method(D_METHOD("unregister_geometry_parser_3d", "geometry_parser"), &NavigationMeshGenerator::unregister_geometry_parser_3d);
}

#endif
