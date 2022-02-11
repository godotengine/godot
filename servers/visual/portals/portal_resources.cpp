/*************************************************************************/
/*  portal_resources.cpp                                                 */
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

#include "portal_resources.h"

OccluderResourceHandle PortalResources::occluder_resource_create() {
	uint32_t pool_id = 0;
	VSOccluder_Resource *occ = _occluder_resource_pool.request(pool_id);
	occ->create();

	OccluderResourceHandle handle = pool_id + 1;
	return handle;
}

void PortalResources::occluder_resource_destroy(OccluderResourceHandle p_handle) {
	p_handle--;

	// Depending on the occluder resource type, remove the spheres, polys, holes etc
	// We can reuse the update methods for this.
	VSOccluder_Resource &occ = _occluder_resource_pool[p_handle];
	switch (occ.type) {
		case VSOccluder_Instance::OT_SPHERE: {
			occluder_resource_update_spheres(p_handle + 1, Vector<Plane>());
		} break;
		case VSOccluder_Instance::OT_MESH: {
			occluder_resource_update_mesh(p_handle + 1, Geometry::OccluderMeshData());
		} break;
		default: {
		} break;
	}

	// This also clears the occluder
	occ.create();

	_occluder_resource_pool.free(p_handle);
}

void PortalResources::occluder_resource_prepare(OccluderResourceHandle p_handle, VSOccluder_Instance::Type p_type) {
	p_handle--;

	// depending on the occluder type, remove the spheres etc
	VSOccluder_Resource &occ = _occluder_resource_pool[p_handle];

	if (occ.type != VSOccluder_Instance::OT_UNDEFINED) {
		ERR_PRINT_ONCE("occluder_resource_prepare should be called only once.");
	}

	occ.type = p_type;
	ERR_FAIL_COND(p_type == VSOccluder_Instance::OT_UNDEFINED);
}

void PortalResources::occluder_resource_update_spheres(OccluderResourceHandle p_handle, const Vector<Plane> &p_spheres) {
	p_handle--;
	VSOccluder_Resource &occ = _occluder_resource_pool[p_handle];
	ERR_FAIL_COND(occ.type != VSOccluder_Resource::OT_SPHERE);

	// first deal with the situation where the number of spheres has changed (rare)
	if (occ.list_ids.size() != p_spheres.size()) {
		// not the most efficient, but works...
		// remove existing
		for (int n = 0; n < occ.list_ids.size(); n++) {
			uint32_t id = occ.list_ids[n];
			_occluder_local_sphere_pool.free(id);
		}

		occ.list_ids.clear();
		// create new
		for (int n = 0; n < p_spheres.size(); n++) {
			uint32_t id;
			VSOccluder_Sphere *sphere = _occluder_local_sphere_pool.request(id);
			sphere->create();
			occ.list_ids.push_back(id);
		}
	}

	// new positions
	for (int n = 0; n < occ.list_ids.size(); n++) {
		uint32_t id = occ.list_ids[n];
		VSOccluder_Sphere &sphere = _occluder_local_sphere_pool[id];
		sphere.from_plane(p_spheres[n]);
	}

	// mark as dirty as the world space spheres will be out of date next time this resource is used
	occ.revision += 1;
}

void PortalResources::occluder_resource_update_mesh(OccluderResourceHandle p_handle, const Geometry::OccluderMeshData &p_mesh_data) {
	p_handle--;
	VSOccluder_Resource &occ = _occluder_resource_pool[p_handle];
	ERR_FAIL_COND(occ.type != VSOccluder_Resource::OT_MESH);

	// mark as dirty, needs world points updating next time this resource is used
	occ.revision += 1;

	const LocalVectori<Geometry::OccluderMeshData::Face> &faces = p_mesh_data.faces;
	const LocalVectori<Vector3> &vertices = p_mesh_data.vertices;

	// first deal with the situation where the number of polys has changed (rare)
	if (occ.list_ids.size() != faces.size()) {
		// not the most efficient, but works...
		// remove existing
		for (int n = 0; n < occ.list_ids.size(); n++) {
			uint32_t id = occ.list_ids[n];

			// must also free the holes
			VSOccluder_Poly &opoly = _occluder_local_poly_pool[id];
			for (int h = 0; h < opoly.num_holes; h++) {
				_occluder_local_hole_pool.free(opoly.hole_pool_ids[h]);

				// perhaps debug only
				opoly.hole_pool_ids[h] = UINT32_MAX;
			}

			_occluder_local_poly_pool.free(id);
		}

		occ.list_ids.clear();
		// create new
		for (int n = 0; n < faces.size(); n++) {
			uint32_t id;
			VSOccluder_Poly *poly = _occluder_local_poly_pool.request(id);
			poly->create();
			occ.list_ids.push_back(id);
		}
	}

	// new data
	for (int n = 0; n < occ.list_ids.size(); n++) {
		uint32_t id = occ.list_ids[n];

		VSOccluder_Poly &opoly = _occluder_local_poly_pool[id];
		Occlusion::PolyPlane &poly = opoly.poly;

		// source face
		const Geometry::OccluderMeshData::Face &face = faces[n];
		opoly.two_way = face.two_way;

		// make sure the number of holes is correct
		if (face.holes.size() != opoly.num_holes) {
			// slow but hey ho
			// delete existing holes
			for (int i = 0; i < opoly.num_holes; i++) {
				_occluder_local_hole_pool.free(opoly.hole_pool_ids[i]);
				opoly.hole_pool_ids[i] = UINT32_MAX;
			}
			// create any new holes
			opoly.num_holes = face.holes.size();
			for (int i = 0; i < opoly.num_holes; i++) {
				uint32_t hole_id;
				VSOccluder_Hole *hole = _occluder_local_hole_pool.request(hole_id);
				opoly.hole_pool_ids[i] = hole_id;
				hole->create();
			}
		}

		// set up the poly basics, plane and verts
		poly.plane = face.plane;
		poly.num_verts = MIN(face.indices.size(), Occlusion::PolyPlane::MAX_POLY_VERTS);

		for (int c = 0; c < poly.num_verts; c++) {
			int vert_index = face.indices[c];

			if (vert_index < vertices.size()) {
				poly.verts[c] = vertices[vert_index];
			} else {
				WARN_PRINT_ONCE("occluder_update_mesh : poly index out of range");
			}
		}

		// set up any holes that are present
		for (int h = 0; h < opoly.num_holes; h++) {
			VSOccluder_Hole &dhole = get_pool_occluder_local_hole(opoly.hole_pool_ids[h]);
			const Geometry::OccluderMeshData::Hole &shole = face.holes[h];

			dhole.num_verts = shole.indices.size();
			dhole.num_verts = MIN(dhole.num_verts, Occlusion::Poly::MAX_POLY_VERTS);

			for (int c = 0; c < dhole.num_verts; c++) {
				int vert_index = shole.indices[c];
				if (vert_index < vertices.size()) {
					dhole.verts[c] = vertices[vert_index];
				} else {
					WARN_PRINT_ONCE("occluder_update_mesh : hole index out of range");
				}
			} // for c through hole verts
		} // for h through holes

	} // for n through occluders
}
