/*************************************************************************/
/*  navigation_mesh_generator.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifdef RECAST_ENABLED

void NavigationMeshGenerator::_add_vertex(const Vector3 &p_vec3, Vector<float> &p_verticies) {
	p_verticies.push_back(p_vec3.x);
	p_verticies.push_back(p_vec3.y);
	p_verticies.push_back(p_vec3.z);
}

void NavigationMeshGenerator::_add_mesh(const Ref<Mesh> &p_mesh, const Transform &p_xform, Vector<float> &p_verticies, Vector<int> &p_indices) {
	int current_vertex_count = p_verticies.size() / 3;

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {
		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
			continue;

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

			for (int i = 0; i < mesh_vertices.size(); i++) {
				_add_vertex(p_xform.xform(vr[i]), p_verticies);
			}

			for (int i = 0; i < face_count; i++) {
				// CCW
				p_indices.push_back(current_vertex_count + (ir[i * 3 + 0]));
				p_indices.push_back(current_vertex_count + (ir[i * 3 + 2]));
				p_indices.push_back(current_vertex_count + (ir[i * 3 + 1]));
			}
		} else {
			face_count = mesh_vertices.size() / 3;
			for (int i = 0; i < face_count; i++) {
				_add_vertex(p_xform.xform(vr[i * 3 + 0]), p_verticies);
				_add_vertex(p_xform.xform(vr[i * 3 + 2]), p_verticies);
				_add_vertex(p_xform.xform(vr[i * 3 + 1]), p_verticies);

				p_indices.push_back(current_vertex_count + (i * 3 + 0));
				p_indices.push_back(current_vertex_count + (i * 3 + 1));
				p_indices.push_back(current_vertex_count + (i * 3 + 2));
			}
		}
	}
}

void NavigationMeshGenerator::_parse_geometry(const Transform &p_base_inverse, Node *p_node, Vector<float> &p_verticies, Vector<int> &p_indices) {

	if (Object::cast_to<MeshInstance>(p_node)) {

		MeshInstance *mesh_instance = Object::cast_to<MeshInstance>(p_node);
		Ref<Mesh> mesh = mesh_instance->get_mesh();
		if (mesh.is_valid()) {
			_add_mesh(mesh, p_base_inverse * mesh_instance->get_global_transform(), p_verticies, p_indices);
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_parse_geometry(p_base_inverse, p_node->get_child(i), p_verticies, p_indices);
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
			nav_indices[0] = ((int)(bverts + tris[j * 4 + 0]));
			nav_indices[1] = ((int)(bverts + tris[j * 4 + 1]));
			nav_indices[2] = ((int)(bverts + tris[j * 4 + 2]));
			p_nav_mesh->add_polygon(nav_indices);
		}
	}
}

void NavigationMeshGenerator::_build_recast_navigation_mesh(Ref<NavigationMesh> p_nav_mesh, EditorProgress *ep,
		rcHeightfield *hf, rcCompactHeightfield *chf, rcContourSet *cset, rcPolyMesh *poly_mesh, rcPolyMeshDetail *detail_mesh,
		Vector<float> &verticies, Vector<int> &indices) {
	rcContext ctx;
	ep->step(TTR("Setting up Configuration..."), 1);

	const float *verts = verticies.ptr();
	const int nverts = verticies.size() / 3;
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

		memset(tri_areas.ptr(), 0, ntris * sizeof(unsigned char));
		rcMarkWalkableTriangles(&ctx, cfg.walkableSlopeAngle, verts, nverts, tris, ntris, tri_areas.ptr());

		ERR_FAIL_COND(!rcRasterizeTriangles(&ctx, verts, nverts, tris, tri_areas.ptr(), ntris, *hf, cfg.walkableClimb));
	}

	if (p_nav_mesh->get_filter_low_hanging_obstacles())
		rcFilterLowHangingWalkableObstacles(&ctx, cfg.walkableClimb, *hf);
	if (p_nav_mesh->get_filter_ledge_spans())
		rcFilterLedgeSpans(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf);
	if (p_nav_mesh->get_filter_walkable_low_height_spans())
		rcFilterWalkableLowHeightSpans(&ctx, cfg.walkableHeight, *hf);

	ep->step(TTR("Constructing compact heightfield..."), 5);

	chf = rcAllocCompactHeightfield();

	ERR_FAIL_COND(!chf);
	ERR_FAIL_COND(!rcBuildCompactHeightfield(&ctx, cfg.walkableHeight, cfg.walkableClimb, *hf, *chf));

	rcFreeHeightField(hf);
	hf = 0;

	ep->step(TTR("Eroding walkable area..."), 6);
	ERR_FAIL_COND(!rcErodeWalkableArea(&ctx, cfg.walkableRadius, *chf));

	ep->step(TTR("Partioning..."), 7);
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
	chf = 0;
	rcFreeContourSet(cset);
	cset = 0;

	ep->step(TTR("Converting to native navigation mesh..."), 10);

	_convert_detail_mesh_to_native_navigation_mesh(detail_mesh, p_nav_mesh);

	rcFreePolyMesh(poly_mesh);
	poly_mesh = 0;
	rcFreePolyMeshDetail(detail_mesh);
	detail_mesh = 0;
}

void NavigationMeshGenerator::bake(Ref<NavigationMesh> p_nav_mesh, Node *p_node) {

	ERR_FAIL_COND(!p_nav_mesh.is_valid());

	EditorProgress ep("bake", TTR("Navigation Mesh Generator Setup:"), 11);
	ep.step(TTR("Parsing Geometry..."), 0);

	Vector<float> verticies;
	Vector<int> indices;

	_parse_geometry(Object::cast_to<Spatial>(p_node)->get_global_transform().affine_inverse(), p_node, verticies, indices);

	if (verticies.size() > 0 && indices.size() > 0) {

		rcHeightfield *hf = NULL;
		rcCompactHeightfield *chf = NULL;
		rcContourSet *cset = NULL;
		rcPolyMesh *poly_mesh = NULL;
		rcPolyMeshDetail *detail_mesh = NULL;

		_build_recast_navigation_mesh(p_nav_mesh, &ep, hf, chf, cset, poly_mesh, detail_mesh, verticies, indices);

		if (hf) {
			rcFreeHeightField(hf);
			hf = 0;
		}
		if (chf) {
			rcFreeCompactHeightfield(chf);
			chf = 0;
		}
		if (cset) {
			rcFreeContourSet(cset);
			cset = 0;
		}
		if (poly_mesh) {
			rcFreePolyMesh(poly_mesh);
			poly_mesh = 0;
		}
		if (detail_mesh) {
			rcFreePolyMeshDetail(detail_mesh);
			detail_mesh = 0;
		}
	}
	ep.step(TTR("Done!"), 11);
}

void NavigationMeshGenerator::clear(Ref<NavigationMesh> p_nav_mesh) {
	if (p_nav_mesh.is_valid()) {
		p_nav_mesh->clear_polygons();
		p_nav_mesh->set_vertices(PoolVector<Vector3>());
	}
}

#endif //RECAST_ENABLED
