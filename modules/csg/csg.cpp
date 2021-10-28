/*************************************************************************/
/*  csg.cpp                                                              */
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

#include "csg.h"

#include "core/math/geometry.h"
#include "core/math/math_funcs.h"
#include "core/sort_array.h"

// Static helper functions.

inline static bool is_snapable(const Vector3 &p_point1, const Vector3 &p_point2, real_t p_distance) {
	return (p_point1 - p_point2).length_squared() < p_distance * p_distance;
}

inline static Vector2 interpolate_segment_uv(const Vector2 p_segement_points[2], const Vector2 p_uvs[2], const Vector2 &p_interpolation_point) {
	float segment_length = (p_segement_points[1] - p_segement_points[0]).length();
	if (segment_length < CMP_EPSILON) {
		return p_uvs[0];
	}

	float distance = (p_interpolation_point - p_segement_points[0]).length();
	float fraction = distance / segment_length;

	return p_uvs[0].linear_interpolate(p_uvs[1], fraction);
}

inline static Vector2 interpolate_triangle_uv(const Vector2 p_vertices[3], const Vector2 p_uvs[3], const Vector2 &p_interpolation_point) {
	if (p_interpolation_point.distance_squared_to(p_vertices[0]) < CMP_EPSILON2) {
		return p_uvs[0];
	}
	if (p_interpolation_point.distance_squared_to(p_vertices[1]) < CMP_EPSILON2) {
		return p_uvs[1];
	}
	if (p_interpolation_point.distance_squared_to(p_vertices[2]) < CMP_EPSILON2) {
		return p_uvs[2];
	}

	Vector2 edge1 = p_vertices[1] - p_vertices[0];
	Vector2 edge2 = p_vertices[2] - p_vertices[0];
	Vector2 interpolation = p_interpolation_point - p_vertices[0];

	float edge1_on_edge1 = edge1.dot(edge1);
	float edge1_on_edge2 = edge1.dot(edge2);
	float edge2_on_edge2 = edge2.dot(edge2);
	float inter_on_edge1 = interpolation.dot(edge1);
	float inter_on_edge2 = interpolation.dot(edge2);
	float scale = (edge1_on_edge1 * edge2_on_edge2 - edge1_on_edge2 * edge1_on_edge2);
	if (scale == 0) {
		return p_uvs[0];
	}

	float v = (edge2_on_edge2 * inter_on_edge1 - edge1_on_edge2 * inter_on_edge2) / scale;
	float w = (edge1_on_edge1 * inter_on_edge2 - edge1_on_edge2 * inter_on_edge1) / scale;
	float u = 1.0f - v - w;

	return p_uvs[0] * u + p_uvs[1] * v + p_uvs[2] * w;
}

static inline bool ray_intersects_triangle(const Vector3 &p_from, const Vector3 &p_dir, const Vector3 p_vertices[3], float p_tolerance, Vector3 &r_intersection_point) {
	Vector3 edge1 = p_vertices[1] - p_vertices[0];
	Vector3 edge2 = p_vertices[2] - p_vertices[0];
	Vector3 h = p_dir.cross(edge2);
	real_t a = edge1.dot(h);
	// Check if ray is parallel to triangle.
	if (Math::is_zero_approx(a)) {
		return false;
	}
	real_t f = 1.0 / a;

	Vector3 s = p_from - p_vertices[0];
	real_t u = f * s.dot(h);
	if (u < 0.0 - p_tolerance || u > 1.0 + p_tolerance) {
		return false;
	}

	Vector3 q = s.cross(edge1);
	real_t v = f * p_dir.dot(q);
	if (v < 0.0 - p_tolerance || u + v > 1.0 + p_tolerance) {
		return false;
	}

	// Ray intersects triangle.
	// Calculate distance.
	real_t t = f * edge2.dot(q);
	// Confirm triangle is in front of ray.
	if (t >= p_tolerance) {
		r_intersection_point = p_from + p_dir * t;
		return true;
	} else {
		return false;
	}
}

inline bool is_point_in_triangle(const Vector3 &p_point, const Vector3 p_vertices[3], int p_shifted = 0) {
	real_t det = p_vertices[0].dot(p_vertices[1].cross(p_vertices[2]));

	// If determinant is, zero try shift the triangle and the point.
	if (Math::is_zero_approx(det)) {
		if (p_shifted > 2) {
			// Triangle appears degenerate, so ignore it.
			return false;
		}
		Vector3 shift_by;
		shift_by[p_shifted] = 1;
		Vector3 shifted_point = p_point + shift_by;
		Vector3 shifted_vertices[3] = { p_vertices[0] + shift_by, p_vertices[1] + shift_by, p_vertices[2] + shift_by };
		return is_point_in_triangle(shifted_point, shifted_vertices, p_shifted + 1);
	}

	// Find the barycentric coordinates of the point with respect to the vertices.
	real_t lambda[3];
	lambda[0] = p_vertices[1].cross(p_vertices[2]).dot(p_point) / det;
	lambda[1] = p_vertices[2].cross(p_vertices[0]).dot(p_point) / det;
	lambda[2] = p_vertices[0].cross(p_vertices[1]).dot(p_point) / det;

	// Point is in the plane if all lambdas sum to 1.
	if (!Math::is_equal_approx(lambda[0] + lambda[1] + lambda[2], 1)) {
		return false;
	}

	// Point is inside the triangle if all lambdas are positive.
	if (lambda[0] < 0 || lambda[1] < 0 || lambda[2] < 0) {
		return false;
	}

	return true;
}

inline static bool is_triangle_degenerate(const Vector2 p_vertices[3], real_t p_vertex_snap2) {
	real_t det = p_vertices[0].x * p_vertices[1].y - p_vertices[0].x * p_vertices[2].y +
			p_vertices[0].y * p_vertices[2].x - p_vertices[0].y * p_vertices[1].x +
			p_vertices[1].x * p_vertices[2].y - p_vertices[1].y * p_vertices[2].x;

	return det < p_vertex_snap2;
}

inline static bool are_segements_parallel(const Vector2 p_segment1_points[2], const Vector2 p_segment2_points[2], float p_vertex_snap2) {
	Vector2 segment1 = p_segment1_points[1] - p_segment1_points[0];
	Vector2 segment2 = p_segment2_points[1] - p_segment2_points[0];
	real_t segment1_length2 = segment1.dot(segment1);
	real_t segment2_length2 = segment2.dot(segment2);
	real_t segment_onto_segment = segment2.dot(segment1);

	if (segment1_length2 < p_vertex_snap2 || segment2_length2 < p_vertex_snap2) {
		return true;
	}

	real_t max_separation2;
	if (segment1_length2 > segment2_length2) {
		max_separation2 = segment2_length2 - segment_onto_segment * segment_onto_segment / segment1_length2;
	} else {
		max_separation2 = segment1_length2 - segment_onto_segment * segment_onto_segment / segment2_length2;
	}

	return max_separation2 < p_vertex_snap2;
}

// CSGBrush

void CSGBrush::_regen_face_aabbs() {
	for (int i = 0; i < faces.size(); i++) {
		faces.write[i].aabb = AABB();
		faces.write[i].aabb.position = faces[i].vertices[0];
		faces.write[i].aabb.expand_to(faces[i].vertices[1]);
		faces.write[i].aabb.expand_to(faces[i].vertices[2]);
	}
}

void CSGBrush::build_from_faces(const PoolVector<Vector3> &p_vertices, const PoolVector<Vector2> &p_uvs, const PoolVector<bool> &p_smooth, const PoolVector<Ref<Material>> &p_materials, const PoolVector<bool> &p_invert_faces) {
	faces.clear();

	int vc = p_vertices.size();

	ERR_FAIL_COND((vc % 3) != 0);

	PoolVector<Vector3>::Read rv = p_vertices.read();
	int uvc = p_uvs.size();
	PoolVector<Vector2>::Read ruv = p_uvs.read();
	int sc = p_smooth.size();
	PoolVector<bool>::Read rs = p_smooth.read();
	int mc = p_materials.size();
	PoolVector<Ref<Material>>::Read rm = p_materials.read();
	int ic = p_invert_faces.size();
	PoolVector<bool>::Read ri = p_invert_faces.read();

	Map<Ref<Material>, int> material_map;

	faces.resize(p_vertices.size() / 3);

	for (int i = 0; i < faces.size(); i++) {
		Face &f = faces.write[i];
		f.vertices[0] = rv[i * 3 + 0];
		f.vertices[1] = rv[i * 3 + 1];
		f.vertices[2] = rv[i * 3 + 2];

		if (uvc == vc) {
			f.uvs[0] = ruv[i * 3 + 0];
			f.uvs[1] = ruv[i * 3 + 1];
			f.uvs[2] = ruv[i * 3 + 2];
		}

		if (sc == vc / 3) {
			f.smooth = rs[i];
		} else {
			f.smooth = false;
		}

		if (ic == vc / 3) {
			f.invert = ri[i];
		} else {
			f.invert = false;
		}

		if (mc == vc / 3) {
			Ref<Material> mat = rm[i];
			if (mat.is_valid()) {
				const Map<Ref<Material>, int>::Element *E = material_map.find(mat);

				if (E) {
					f.material = E->get();
				} else {
					f.material = material_map.size();
					material_map[mat] = f.material;
				}

			} else {
				f.material = -1;
			}
		}
	}

	materials.resize(material_map.size());
	for (Map<Ref<Material>, int>::Element *E = material_map.front(); E; E = E->next()) {
		materials.write[E->get()] = E->key();
	}

	_regen_face_aabbs();
}

void CSGBrush::copy_from(const CSGBrush &p_brush, const Transform &p_xform) {
	faces = p_brush.faces;
	materials = p_brush.materials;

	for (int i = 0; i < faces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			faces.write[i].vertices[j] = p_xform.xform(p_brush.faces[i].vertices[j]);
		}
	}

	_regen_face_aabbs();
}

// CSGBrushOperation

void CSGBrushOperation::merge_brushes(Operation p_operation, const CSGBrush &p_brush_a, const CSGBrush &p_brush_b, CSGBrush &r_merged_brush, float p_vertex_snap) {
	// Check for face collisions and add necessary faces.
	Build2DFaceCollection build2DFaceCollection;
	for (int i = 0; i < p_brush_a.faces.size(); i++) {
		for (int j = 0; j < p_brush_b.faces.size(); j++) {
			if (p_brush_a.faces[i].aabb.intersects_inclusive(p_brush_b.faces[j].aabb)) {
				update_faces(p_brush_a, i, p_brush_b, j, build2DFaceCollection, p_vertex_snap);
			}
		}
	}

	// Add faces to MeshMerge.
	MeshMerge mesh_merge;
	mesh_merge.vertex_snap = p_vertex_snap;

	for (int i = 0; i < p_brush_a.faces.size(); i++) {
		Ref<Material> material;
		if (p_brush_a.faces[i].material != -1) {
			material = p_brush_a.materials[p_brush_a.faces[i].material];
		}

		if (build2DFaceCollection.build2DFacesA.has(i)) {
			build2DFaceCollection.build2DFacesA[i].addFacesToMesh(mesh_merge, p_brush_a.faces[i].smooth, p_brush_a.faces[i].invert, material, false);
		} else {
			Vector3 points[3];
			Vector2 uvs[3];
			for (int j = 0; j < 3; j++) {
				points[j] = p_brush_a.faces[i].vertices[j];
				uvs[j] = p_brush_a.faces[i].uvs[j];
			}
			mesh_merge.add_face(points, uvs, p_brush_a.faces[i].smooth, p_brush_a.faces[i].invert, material, false);
		}
	}

	for (int i = 0; i < p_brush_b.faces.size(); i++) {
		Ref<Material> material;
		if (p_brush_b.faces[i].material != -1) {
			material = p_brush_b.materials[p_brush_b.faces[i].material];
		}

		if (build2DFaceCollection.build2DFacesB.has(i)) {
			build2DFaceCollection.build2DFacesB[i].addFacesToMesh(mesh_merge, p_brush_b.faces[i].smooth, p_brush_b.faces[i].invert, material, true);
		} else {
			Vector3 points[3];
			Vector2 uvs[3];
			for (int j = 0; j < 3; j++) {
				points[j] = p_brush_b.faces[i].vertices[j];
				uvs[j] = p_brush_b.faces[i].uvs[j];
			}
			mesh_merge.add_face(points, uvs, p_brush_b.faces[i].smooth, p_brush_b.faces[i].invert, material, true);
		}
	}

	// Mark faces that ended up inside the intersection.
	mesh_merge.mark_inside_faces();

	// Create new brush and fill with new faces.
	r_merged_brush.faces.clear();

	switch (p_operation) {
		case OPERATION_UNION: {
			int outside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (mesh_merge.faces[i].inside) {
					continue;
				}
				outside_count++;
			}

			r_merged_brush.faces.resize(outside_count);

			outside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (mesh_merge.faces[i].inside) {
					continue;
				}

				for (int j = 0; j < 3; j++) {
					r_merged_brush.faces.write[outside_count].vertices[j] = mesh_merge.points[mesh_merge.faces[i].points[j]];
					r_merged_brush.faces.write[outside_count].uvs[j] = mesh_merge.faces[i].uvs[j];
				}

				r_merged_brush.faces.write[outside_count].smooth = mesh_merge.faces[i].smooth;
				r_merged_brush.faces.write[outside_count].invert = mesh_merge.faces[i].invert;
				r_merged_brush.faces.write[outside_count].material = mesh_merge.faces[i].material_idx;
				outside_count++;
			}

			r_merged_brush._regen_face_aabbs();

		} break;

		case OPERATION_INTERSECTION: {
			int inside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (!mesh_merge.faces[i].inside) {
					continue;
				}
				inside_count++;
			}

			r_merged_brush.faces.resize(inside_count);

			inside_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (!mesh_merge.faces[i].inside) {
					continue;
				}

				for (int j = 0; j < 3; j++) {
					r_merged_brush.faces.write[inside_count].vertices[j] = mesh_merge.points[mesh_merge.faces[i].points[j]];
					r_merged_brush.faces.write[inside_count].uvs[j] = mesh_merge.faces[i].uvs[j];
				}

				r_merged_brush.faces.write[inside_count].smooth = mesh_merge.faces[i].smooth;
				r_merged_brush.faces.write[inside_count].invert = mesh_merge.faces[i].invert;
				r_merged_brush.faces.write[inside_count].material = mesh_merge.faces[i].material_idx;
				inside_count++;
			}

			r_merged_brush._regen_face_aabbs();

		} break;

		case OPERATION_SUBTRACTION: {
			int face_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (mesh_merge.faces[i].from_b && !mesh_merge.faces[i].inside) {
					continue;
				}
				if (!mesh_merge.faces[i].from_b && mesh_merge.faces[i].inside) {
					continue;
				}
				face_count++;
			}

			r_merged_brush.faces.resize(face_count);

			face_count = 0;

			for (int i = 0; i < mesh_merge.faces.size(); i++) {
				if (mesh_merge.faces[i].from_b && !mesh_merge.faces[i].inside) {
					continue;
				}
				if (!mesh_merge.faces[i].from_b && mesh_merge.faces[i].inside) {
					continue;
				}

				for (int j = 0; j < 3; j++) {
					r_merged_brush.faces.write[face_count].vertices[j] = mesh_merge.points[mesh_merge.faces[i].points[j]];
					r_merged_brush.faces.write[face_count].uvs[j] = mesh_merge.faces[i].uvs[j];
				}

				if (mesh_merge.faces[i].from_b) {
					//invert facing of insides of B
					SWAP(r_merged_brush.faces.write[face_count].vertices[1], r_merged_brush.faces.write[face_count].vertices[2]);
					SWAP(r_merged_brush.faces.write[face_count].uvs[1], r_merged_brush.faces.write[face_count].uvs[2]);
				}

				r_merged_brush.faces.write[face_count].smooth = mesh_merge.faces[i].smooth;
				r_merged_brush.faces.write[face_count].invert = mesh_merge.faces[i].invert;
				r_merged_brush.faces.write[face_count].material = mesh_merge.faces[i].material_idx;
				face_count++;
			}

			r_merged_brush._regen_face_aabbs();

		} break;
	}

	// Update the list of materials.
	r_merged_brush.materials.resize(mesh_merge.materials.size());
	for (const Map<Ref<Material>, int>::Element *E = mesh_merge.materials.front(); E; E = E->next()) {
		r_merged_brush.materials.write[E->get()] = E->key();
	}
}

// CSGBrushOperation::MeshMerge

// Use a limit to speed up bvh and limit the depth.
#define BVH_LIMIT 8

int CSGBrushOperation::MeshMerge::_create_bvh(FaceBVH *facebvhptr, FaceBVH **facebvhptrptr, int p_from, int p_size, int p_depth, int &r_max_depth, int &r_max_alloc) {
	if (p_depth > r_max_depth) {
		r_max_depth = p_depth;
	}

	if (p_size == 0) {
		return -1;
	}

	if (p_size <= BVH_LIMIT) {
		for (int i = 0; i < p_size - 1; i++) {
			facebvhptrptr[p_from + i]->next = facebvhptrptr[p_from + i + 1] - facebvhptr;
		}
		return facebvhptrptr[p_from] - facebvhptr;
	}

	AABB aabb;
	aabb = facebvhptrptr[p_from]->aabb;
	for (int i = 1; i < p_size; i++) {
		aabb.merge_with(facebvhptrptr[p_from + i]->aabb);
	}

	int li = aabb.get_longest_axis_index();

	switch (li) {
		case Vector3::AXIS_X: {
			SortArray<FaceBVH *, FaceBVHCmpX> sort_x;
			sort_x.nth_element(0, p_size, p_size / 2, &facebvhptrptr[p_from]);
			//sort_x.sort(&p_bb[p_from],p_size);
		} break;

		case Vector3::AXIS_Y: {
			SortArray<FaceBVH *, FaceBVHCmpY> sort_y;
			sort_y.nth_element(0, p_size, p_size / 2, &facebvhptrptr[p_from]);
			//sort_y.sort(&p_bb[p_from],p_size);
		} break;

		case Vector3::AXIS_Z: {
			SortArray<FaceBVH *, FaceBVHCmpZ> sort_z;
			sort_z.nth_element(0, p_size, p_size / 2, &facebvhptrptr[p_from]);
			//sort_z.sort(&p_bb[p_from],p_size);
		} break;
	}

	int left = _create_bvh(facebvhptr, facebvhptrptr, p_from, p_size / 2, p_depth + 1, r_max_depth, r_max_alloc);
	int right = _create_bvh(facebvhptr, facebvhptrptr, p_from + p_size / 2, p_size - p_size / 2, p_depth + 1, r_max_depth, r_max_alloc);

	int index = r_max_alloc++;
	FaceBVH *_new = &facebvhptr[index];
	_new->aabb = aabb;
	_new->center = aabb.position + aabb.size * 0.5;
	_new->face = -1;
	_new->left = left;
	_new->right = right;
	_new->next = -1;

	return index;
}

void CSGBrushOperation::MeshMerge::_add_distance(List<real_t> &r_intersectionsA, List<real_t> &r_intersectionsB, bool p_from_B, real_t p_distance) const {
	List<real_t> &intersections = p_from_B ? r_intersectionsB : r_intersectionsA;

	// Check if distance exists.
	for (const List<real_t>::Element *E = intersections.front(); E; E = E->next()) {
		if (Math::is_equal_approx(**E, p_distance)) {
			return;
		}
	}

	intersections.push_back(p_distance);
}

bool CSGBrushOperation::MeshMerge::_bvh_inside(FaceBVH *facebvhptr, int p_max_depth, int p_bvh_first, int p_face_idx) const {
	Face face = faces[p_face_idx];
	Vector3 face_points[3] = {
		points[face.points[0]],
		points[face.points[1]],
		points[face.points[2]]
	};
	Vector3 face_center = (face_points[0] + face_points[1] + face_points[2]) / 3.0;
	Vector3 face_normal = Plane(face_points[0], face_points[1], face_points[2]).normal;

	uint32_t *stack = (uint32_t *)alloca(sizeof(int) * p_max_depth);

	enum {
		TEST_AABB_BIT = 0,
		VISIT_LEFT_BIT = 1,
		VISIT_RIGHT_BIT = 2,
		VISIT_DONE_BIT = 3,
		VISITED_BIT_SHIFT = 29,
		NODE_IDX_MASK = (1 << VISITED_BIT_SHIFT) - 1,
		VISITED_BIT_MASK = ~NODE_IDX_MASK
	};

	List<real_t> intersectionsA;
	List<real_t> intersectionsB;

	int level = 0;
	int pos = p_bvh_first;
	stack[0] = pos;

	while (true) {
		uint32_t node = stack[level] & NODE_IDX_MASK;
		const FaceBVH *current_facebvhptr = &(facebvhptr[node]);
		bool done = false;

		switch (stack[level] >> VISITED_BIT_SHIFT) {
			case TEST_AABB_BIT: {
				if (current_facebvhptr->face >= 0) {
					while (current_facebvhptr) {
						if (p_face_idx != current_facebvhptr->face &&
								current_facebvhptr->aabb.intersects_ray(face_center, face_normal)) {
							const Face &current_face = faces[current_facebvhptr->face];
							Vector3 current_points[3] = {
								points[current_face.points[0]],
								points[current_face.points[1]],
								points[current_face.points[2]]
							};
							Vector3 current_normal = Plane(current_points[0], current_points[1], current_points[2]).normal;
							Vector3 intersection_point;

							// Check if faces are co-planar.
							if ((current_normal - face_normal).length_squared() < CMP_EPSILON2 &&
									is_point_in_triangle(face_center, current_points)) {
								// Only add an intersection if not a B face.
								if (!face.from_b) {
									_add_distance(intersectionsA, intersectionsB, current_face.from_b, 0);
								}
							} else if (ray_intersects_triangle(face_center, face_normal, current_points, CMP_EPSILON, intersection_point)) {
								real_t distance = (intersection_point - face_center).length();
								_add_distance(intersectionsA, intersectionsB, current_face.from_b, distance);
							}
						}

						if (current_facebvhptr->next != -1) {
							current_facebvhptr = &facebvhptr[current_facebvhptr->next];
						} else {
							current_facebvhptr = nullptr;
						}
					}

					stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;

				} else {
					bool valid = current_facebvhptr->aabb.intersects_ray(face_center, face_normal);

					if (!valid) {
						stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;
					} else {
						stack[level] = (VISIT_LEFT_BIT << VISITED_BIT_SHIFT) | node;
					}
				}
				continue;
			}

			case VISIT_LEFT_BIT: {
				stack[level] = (VISIT_RIGHT_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = current_facebvhptr->left | TEST_AABB_BIT;
				level++;
				continue;
			}

			case VISIT_RIGHT_BIT: {
				stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = current_facebvhptr->right | TEST_AABB_BIT;
				level++;
				continue;
			}

			case VISIT_DONE_BIT: {
				if (level == 0) {
					done = true;
					break;
				} else {
					level--;
				}
				continue;
			}
		}

		if (done) {
			break;
		}
	}

	// Inside if face normal intersects other faces an odd number of times.
	return (intersectionsA.size() + intersectionsB.size()) & 1;
}

void CSGBrushOperation::MeshMerge::mark_inside_faces() {
	// Mark faces that are inside. This helps later do the boolean ops when merging.
	// This approach is very brute force with a bunch of optimizations,
	// such as BVH and pre AABB intersection test.

	Vector<FaceBVH> bvhvec;
	bvhvec.resize(faces.size() * 3); // Will never be larger than this (TODO: Make better)
	FaceBVH *facebvh = bvhvec.ptrw();

	AABB aabb_a;
	AABB aabb_b;

	bool first_a = true;
	bool first_b = true;

	for (int i = 0; i < faces.size(); i++) {
		facebvh[i].left = -1;
		facebvh[i].right = -1;
		facebvh[i].face = i;
		facebvh[i].aabb.position = points[faces[i].points[0]];
		facebvh[i].aabb.expand_to(points[faces[i].points[1]]);
		facebvh[i].aabb.expand_to(points[faces[i].points[2]]);
		facebvh[i].center = facebvh[i].aabb.position + facebvh[i].aabb.size * 0.5;
		facebvh[i].aabb.grow_by(vertex_snap);
		facebvh[i].next = -1;

		if (faces[i].from_b) {
			if (first_b) {
				aabb_b = facebvh[i].aabb;
				first_b = false;
			} else {
				aabb_b.merge_with(facebvh[i].aabb);
			}
		} else {
			if (first_a) {
				aabb_a = facebvh[i].aabb;
				first_a = false;
			} else {
				aabb_a.merge_with(facebvh[i].aabb);
			}
		}
	}

	AABB intersection_aabb = aabb_a.intersection(aabb_b);

	// Check if shape AABBs intersect.
	if (intersection_aabb.size == Vector3()) {
		return;
	}

	Vector<FaceBVH *> bvhtrvec;
	bvhtrvec.resize(faces.size());
	FaceBVH **bvhptr = bvhtrvec.ptrw();
	for (int i = 0; i < faces.size(); i++) {
		bvhptr[i] = &facebvh[i];
	}

	int max_depth = 0;
	int max_alloc = faces.size();
	_create_bvh(facebvh, bvhptr, 0, faces.size(), 1, max_depth, max_alloc);

	for (int i = 0; i < faces.size(); i++) {
		// Check if face AABB intersects the intersection AABB.
		if (!intersection_aabb.intersects_inclusive(facebvh[i].aabb)) {
			continue;
		}

		if (_bvh_inside(facebvh, max_depth, max_alloc - 1, i)) {
			faces.write[i].inside = true;
		}
	}
}

void CSGBrushOperation::MeshMerge::add_face(const Vector3 p_points[], const Vector2 p_uvs[], bool p_smooth, bool p_invert, const Ref<Material> &p_material, bool p_from_b) {
	int indices[3];
	for (int i = 0; i < 3; i++) {
		VertexKey vk;
		vk.x = int((double(p_points[i].x) + double(vertex_snap) * 0.31234) / double(vertex_snap));
		vk.y = int((double(p_points[i].y) + double(vertex_snap) * 0.31234) / double(vertex_snap));
		vk.z = int((double(p_points[i].z) + double(vertex_snap) * 0.31234) / double(vertex_snap));

		int res;
		if (snap_cache.lookup(vk, res)) {
			indices[i] = res;
		} else {
			indices[i] = points.size();
			points.push_back(p_points[i]);
			snap_cache.set(vk, indices[i]);
		}
	}

	// Don't add degenerate faces.
	if (indices[0] == indices[2] || indices[0] == indices[1] || indices[1] == indices[2]) {
		return;
	}

	MeshMerge::Face face;
	face.from_b = p_from_b;
	face.inside = false;
	face.smooth = p_smooth;
	face.invert = p_invert;

	if (p_material.is_valid()) {
		if (!materials.has(p_material)) {
			face.material_idx = materials.size();
			materials[p_material] = face.material_idx;
		} else {
			face.material_idx = materials[p_material];
		}
	} else {
		face.material_idx = -1;
	}

	for (int k = 0; k < 3; k++) {
		face.points[k] = indices[k];
		face.uvs[k] = p_uvs[k];
	}

	faces.push_back(face);
}

// CSGBrushOperation::Build2DFaces

int CSGBrushOperation::Build2DFaces::_get_point_idx(const Vector2 &p_point) {
	for (int vertex_idx = 0; vertex_idx < vertices.size(); ++vertex_idx) {
		if ((p_point - vertices[vertex_idx].point).length_squared() < vertex_snap2) {
			return vertex_idx;
		}
	}
	return -1;
}

int CSGBrushOperation::Build2DFaces::_add_vertex(const Vertex2D &p_vertex) {
	// Check if vertex exists.
	int vertex_id = _get_point_idx(p_vertex.point);
	if (vertex_id != -1) {
		return vertex_id;
	}

	vertices.push_back(p_vertex);
	return vertices.size() - 1;
}

void CSGBrushOperation::Build2DFaces::_add_vertex_idx_sorted(Vector<int> &r_vertex_indices, int p_new_vertex_index) {
	if (p_new_vertex_index >= 0 && r_vertex_indices.find(p_new_vertex_index) == -1) {
		ERR_FAIL_COND_MSG(p_new_vertex_index >= vertices.size(), "Invalid vertex index.");

		// The first vertex.
		if (r_vertex_indices.size() == 0) {
			// Simply add it.
			r_vertex_indices.push_back(p_new_vertex_index);
			return;
		}

		// The second vertex.
		if (r_vertex_indices.size() == 1) {
			Vector2 first_point = vertices[r_vertex_indices[0]].point;
			Vector2 new_point = vertices[p_new_vertex_index].point;

			// Sort along the axis with the greatest difference.
			int axis = 0;
			if (Math::abs(new_point.x - first_point.x) < Math::abs(new_point.y - first_point.y)) {
				axis = 1;
			}

			// Add it to the beginning or the end appropriately.
			if (new_point[axis] < first_point[axis]) {
				r_vertex_indices.insert(0, p_new_vertex_index);
			} else {
				r_vertex_indices.push_back(p_new_vertex_index);
			}

			return;
		}

		// Third or later vertices.
		Vector2 first_point = vertices[r_vertex_indices[0]].point;
		Vector2 last_point = vertices[r_vertex_indices[r_vertex_indices.size() - 1]].point;
		Vector2 new_point = vertices[p_new_vertex_index].point;

		// Determine axis being sorted against i.e. the axis with the greatest difference.
		int axis = 0;
		if (Math::abs(last_point.x - first_point.x) < Math::abs(last_point.y - first_point.y)) {
			axis = 1;
		}

		// Insert the point at the appropriate index.
		for (int insert_idx = 0; insert_idx < r_vertex_indices.size(); ++insert_idx) {
			Vector2 insert_point = vertices[r_vertex_indices[insert_idx]].point;
			if (new_point[axis] < insert_point[axis]) {
				r_vertex_indices.insert(insert_idx, p_new_vertex_index);
				return;
			}
		}

		// New largest, add it to the end.
		r_vertex_indices.push_back(p_new_vertex_index);
	}
}

void CSGBrushOperation::Build2DFaces::_merge_faces(const Vector<int> &p_segment_indices) {
	int segments = p_segment_indices.size() - 1;
	if (segments < 2) {
		return;
	}

	// Faces around an inner vertex are merged by moving the inner vertex to the first vertex.
	for (int sorted_idx = 1; sorted_idx < segments; ++sorted_idx) {
		int closest_idx = 0;
		int inner_idx = p_segment_indices[sorted_idx];

		if (sorted_idx > segments / 2) {
			// Merge to other segment end.
			closest_idx = segments;
			// Reverse the merge order.
			inner_idx = p_segment_indices[segments + segments / 2 - sorted_idx];
		}

		// Find the mergeable faces.
		Vector<int> merge_faces_idx;
		Vector<Face2D> merge_faces;
		Vector<int> merge_faces_inner_vertex_idx;
		for (int face_idx = 0; face_idx < faces.size(); ++face_idx) {
			for (int face_vertex_idx = 0; face_vertex_idx < 3; ++face_vertex_idx) {
				if (faces[face_idx].vertex_idx[face_vertex_idx] == inner_idx) {
					merge_faces_idx.push_back(face_idx);
					merge_faces.push_back(faces[face_idx]);
					merge_faces_inner_vertex_idx.push_back(face_vertex_idx);
				}
			}
		}

		Vector<int> degenerate_points;

		// Create the new faces.
		for (int merge_idx = 0; merge_idx < merge_faces.size(); ++merge_idx) {
			int outer_edge_idx[2];
			outer_edge_idx[0] = merge_faces[merge_idx].vertex_idx[(merge_faces_inner_vertex_idx[merge_idx] + 1) % 3];
			outer_edge_idx[1] = merge_faces[merge_idx].vertex_idx[(merge_faces_inner_vertex_idx[merge_idx] + 2) % 3];

			// Skip flattened faces.
			if (outer_edge_idx[0] == p_segment_indices[closest_idx] ||
					outer_edge_idx[1] == p_segment_indices[closest_idx]) {
				continue;
			}

			//Don't create degenerate triangles.
			Vector2 edge1[2] = {
				vertices[outer_edge_idx[0]].point,
				vertices[p_segment_indices[closest_idx]].point
			};
			Vector2 edge2[2] = {
				vertices[outer_edge_idx[1]].point,
				vertices[p_segment_indices[closest_idx]].point
			};
			if (are_segements_parallel(edge1, edge2, vertex_snap2)) {
				if (!degenerate_points.find(outer_edge_idx[0])) {
					degenerate_points.push_back(outer_edge_idx[0]);
				}
				if (!degenerate_points.find(outer_edge_idx[1])) {
					degenerate_points.push_back(outer_edge_idx[1]);
				}
				continue;
			}

			// Create new faces.
			Face2D new_face;
			new_face.vertex_idx[0] = p_segment_indices[closest_idx];
			new_face.vertex_idx[1] = outer_edge_idx[0];
			new_face.vertex_idx[2] = outer_edge_idx[1];
			faces.push_back(new_face);
		}

		// Delete the old faces in reverse index order.
		merge_faces_idx.sort();
		merge_faces_idx.invert();
		for (int i = 0; i < merge_faces_idx.size(); ++i) {
			faces.remove(merge_faces_idx[i]);
		}

		if (degenerate_points.size() == 0) {
			continue;
		}

		// Split faces using degenerate points.
		for (int face_idx = 0; face_idx < faces.size(); ++face_idx) {
			Face2D face = faces[face_idx];
			Vertex2D face_vertices[3] = {
				vertices[face.vertex_idx[0]],
				vertices[face.vertex_idx[1]],
				vertices[face.vertex_idx[2]]
			};
			Vector2 face_points[3] = {
				face_vertices[0].point,
				face_vertices[1].point,
				face_vertices[2].point
			};

			for (int point_idx = 0; point_idx < degenerate_points.size(); ++point_idx) {
				int degenerate_idx = degenerate_points[point_idx];
				Vector2 point_2D = vertices[degenerate_idx].point;

				// Check if point is existing face vertex.
				bool existing = false;
				for (int i = 0; i < 3; ++i) {
					if ((point_2D - face_vertices[i].point).length_squared() < vertex_snap2) {
						existing = true;
						break;
					}
				}
				if (existing) {
					continue;
				}

				// Check if point is on an each edge.
				for (int face_edge_idx = 0; face_edge_idx < 3; ++face_edge_idx) {
					Vector2 edge_points[2] = {
						face_points[face_edge_idx],
						face_points[(face_edge_idx + 1) % 3]
					};
					Vector2 closest_point = Geometry::get_closest_point_to_segment_2d(point_2D, edge_points);

					if ((closest_point - point_2D).length_squared() < vertex_snap2) {
						int opposite_vertex_idx = face.vertex_idx[(face_edge_idx + 2) % 3];

						// If new vertex snaps to degenerate vertex, just delete this face.
						if (degenerate_idx == opposite_vertex_idx) {
							faces.remove(face_idx);
							// Update index.
							--face_idx;
							break;
						}

						// Create two new faces around the new edge and remove this face.
						// The new edge is the last edge.
						Face2D left_face;
						left_face.vertex_idx[0] = degenerate_idx;
						left_face.vertex_idx[1] = face.vertex_idx[(face_edge_idx + 1) % 3];
						left_face.vertex_idx[2] = opposite_vertex_idx;
						Face2D right_face;
						right_face.vertex_idx[0] = opposite_vertex_idx;
						right_face.vertex_idx[1] = face.vertex_idx[face_edge_idx];
						right_face.vertex_idx[2] = degenerate_idx;
						faces.remove(face_idx);
						faces.insert(face_idx, right_face);
						faces.insert(face_idx, left_face);

						// Don't check against the new faces.
						++face_idx;

						// No need to check other edges.
						break;
					}
				}
			}
		}
	}
}

void CSGBrushOperation::Build2DFaces::_find_edge_intersections(const Vector2 p_segment_points[2], Vector<int> &r_segment_indices) {
	// For each face.
	for (int face_idx = 0; face_idx < faces.size(); ++face_idx) {
		Face2D face = faces[face_idx];
		Vertex2D face_vertices[3] = {
			vertices[face.vertex_idx[0]],
			vertices[face.vertex_idx[1]],
			vertices[face.vertex_idx[2]]
		};

		// Check each edge.
		for (int face_edge_idx = 0; face_edge_idx < 3; ++face_edge_idx) {
			Vector2 edge_points[2] = {
				face_vertices[face_edge_idx].point,
				face_vertices[(face_edge_idx + 1) % 3].point
			};
			Vector2 edge_uvs[2] = {
				face_vertices[face_edge_idx].uv,
				face_vertices[(face_edge_idx + 1) % 3].uv
			};
			Vector2 intersection_point;

			// First check if the ends of the segment are on the edge.
			bool on_edge = false;
			for (int edge_point_idx = 0; edge_point_idx < 2; ++edge_point_idx) {
				intersection_point = Geometry::get_closest_point_to_segment_2d(p_segment_points[edge_point_idx], edge_points);
				if ((intersection_point - p_segment_points[edge_point_idx]).length_squared() < vertex_snap2) {
					on_edge = true;
					break;
				}
			}

			// Else check if the segment intersects the edge.
			if (on_edge || Geometry::segment_intersects_segment_2d(p_segment_points[0], p_segment_points[1], edge_points[0], edge_points[1], &intersection_point)) {
				// Check if intersection point is an edge point.
				if ((intersection_point - edge_points[0]).length_squared() < vertex_snap2 ||
						(intersection_point - edge_points[1]).length_squared() < vertex_snap2) {
					continue;
				}

				// Check if edge exists, by checking if the intersecting segment is parallel to the edge.
				if (are_segements_parallel(p_segment_points, edge_points, vertex_snap2)) {
					continue;
				}

				// Add the intersection point as a new vertex.
				Vertex2D new_vertex;
				new_vertex.point = intersection_point;
				new_vertex.uv = interpolate_segment_uv(edge_points, edge_uvs, intersection_point);
				int new_vertex_idx = _add_vertex(new_vertex);
				int opposite_vertex_idx = face.vertex_idx[(face_edge_idx + 2) % 3];
				_add_vertex_idx_sorted(r_segment_indices, new_vertex_idx);

				// If new vertex snaps to opposite vertex, just delete this face.
				if (new_vertex_idx == opposite_vertex_idx) {
					faces.remove(face_idx);
					// Update index.
					--face_idx;
					break;
				}

				// If opposite point is on the segemnt, add its index to segment indices too.
				Vector2 closest_point = Geometry::get_closest_point_to_segment_2d(vertices[opposite_vertex_idx].point, p_segment_points);
				if ((closest_point - vertices[opposite_vertex_idx].point).length_squared() < vertex_snap2) {
					_add_vertex_idx_sorted(r_segment_indices, opposite_vertex_idx);
				}

				// Create two new faces around the new edge and remove this face.
				// The new edge is the last edge.
				Face2D left_face;
				left_face.vertex_idx[0] = new_vertex_idx;
				left_face.vertex_idx[1] = face.vertex_idx[(face_edge_idx + 1) % 3];
				left_face.vertex_idx[2] = opposite_vertex_idx;
				Face2D right_face;
				right_face.vertex_idx[0] = opposite_vertex_idx;
				right_face.vertex_idx[1] = face.vertex_idx[face_edge_idx];
				right_face.vertex_idx[2] = new_vertex_idx;
				faces.remove(face_idx);
				faces.insert(face_idx, right_face);
				faces.insert(face_idx, left_face);

				// Check against the new faces.
				--face_idx;
				break;
			}
		}
	}
}

int CSGBrushOperation::Build2DFaces::_insert_point(const Vector2 &p_point) {
	int new_vertex_idx = -1;

	for (int face_idx = 0; face_idx < faces.size(); ++face_idx) {
		Face2D face = faces[face_idx];
		Vertex2D face_vertices[3] = {
			vertices[face.vertex_idx[0]],
			vertices[face.vertex_idx[1]],
			vertices[face.vertex_idx[2]]
		};
		Vector2 points[3] = {
			face_vertices[0].point,
			face_vertices[1].point,
			face_vertices[2].point
		};
		Vector2 uvs[3] = {
			face_vertices[0].uv,
			face_vertices[1].uv,
			face_vertices[2].uv
		};

		// Skip degenerate triangles.
		if (is_triangle_degenerate(points, vertex_snap2)) {
			continue;
		}

		// Check if point is existing face vertex.
		for (int i = 0; i < 3; ++i) {
			if ((p_point - face_vertices[i].point).length_squared() < vertex_snap2) {
				return face.vertex_idx[i];
			}
		}

		// Check if point is on an each edge.
		bool on_edge = false;
		for (int face_edge_idx = 0; face_edge_idx < 3; ++face_edge_idx) {
			Vector2 edge_points[2] = {
				points[face_edge_idx],
				points[(face_edge_idx + 1) % 3]
			};
			Vector2 edge_uvs[2] = {
				uvs[face_edge_idx],
				uvs[(face_edge_idx + 1) % 3]
			};

			Vector2 closest_point = Geometry::get_closest_point_to_segment_2d(p_point, edge_points);
			if ((closest_point - p_point).length_squared() < vertex_snap2) {
				on_edge = true;

				// Add the point as a new vertex.
				Vertex2D new_vertex;
				new_vertex.point = p_point;
				new_vertex.uv = interpolate_segment_uv(edge_points, edge_uvs, p_point);
				new_vertex_idx = _add_vertex(new_vertex);
				int opposite_vertex_idx = face.vertex_idx[(face_edge_idx + 2) % 3];

				// If new vertex snaps to opposite vertex, just delete this face.
				if (new_vertex_idx == opposite_vertex_idx) {
					faces.remove(face_idx);
					// Update index.
					--face_idx;
					break;
				}

				// Don't create degenerate triangles.
				Vector2 split_edge1[2] = { vertices[new_vertex_idx].point, edge_points[0] };
				Vector2 split_edge2[2] = { vertices[new_vertex_idx].point, edge_points[1] };
				Vector2 new_edge[2] = { vertices[new_vertex_idx].point, vertices[opposite_vertex_idx].point };
				if (are_segements_parallel(split_edge1, new_edge, vertex_snap2) &&
						are_segements_parallel(split_edge2, new_edge, vertex_snap2)) {
					break;
				}

				// Create two new faces around the new edge and remove this face.
				// The new edge is the last edge.
				Face2D left_face;
				left_face.vertex_idx[0] = new_vertex_idx;
				left_face.vertex_idx[1] = face.vertex_idx[(face_edge_idx + 1) % 3];
				left_face.vertex_idx[2] = opposite_vertex_idx;
				Face2D right_face;
				right_face.vertex_idx[0] = opposite_vertex_idx;
				right_face.vertex_idx[1] = face.vertex_idx[face_edge_idx];
				right_face.vertex_idx[2] = new_vertex_idx;
				faces.remove(face_idx);
				faces.insert(face_idx, right_face);
				faces.insert(face_idx, left_face);

				// Don't check against the new faces.
				++face_idx;

				// No need to check other edges.
				break;
			}
		}

		// If not on an edge, check if the point is inside the face.
		if (!on_edge && Geometry::is_point_in_triangle(p_point, face_vertices[0].point, face_vertices[1].point, face_vertices[2].point)) {
			// Add the point as a new vertex.
			Vertex2D new_vertex;
			new_vertex.point = p_point;
			new_vertex.uv = interpolate_triangle_uv(points, uvs, p_point);
			new_vertex_idx = _add_vertex(new_vertex);

			// Create three new faces around this point and remove this face.
			// The new vertex is the last vertex.
			for (int i = 0; i < 3; ++i) {
				// Don't create degenerate triangles.
				Vector2 new_points[3] = { points[i], points[(i + 1) % 3], vertices[new_vertex_idx].point };
				if (is_triangle_degenerate(new_points, vertex_snap2)) {
					continue;
				}

				Face2D new_face;
				new_face.vertex_idx[0] = face.vertex_idx[i];
				new_face.vertex_idx[1] = face.vertex_idx[(i + 1) % 3];
				new_face.vertex_idx[2] = new_vertex_idx;
				faces.push_back(new_face);
			}
			faces.remove(face_idx);

			// No need to check other faces.
			break;
		}
	}

	return new_vertex_idx;
}

void CSGBrushOperation::Build2DFaces::insert(const CSGBrush &p_brush, int p_face_idx) {
	// Find edge points that cross the plane and face points that are in the plane.
	// Map those points to 2D.
	// Create new faces from those points.

	Vector2 points_2D[3];
	int points_count = 0;

	for (int i = 0; i < 3; i++) {
		Vector3 point_3D = p_brush.faces[p_face_idx].vertices[i];

		if (plane.has_point(point_3D)) {
			// Point is in the plane, add it.
			Vector3 point_2D = plane.project(point_3D);
			point_2D = to_2D.xform(point_2D);
			points_2D[points_count++] = Vector2(point_2D.x, point_2D.y);

		} else {
			Vector3 next_point_3D = p_brush.faces[p_face_idx].vertices[(i + 1) % 3];

			if (plane.has_point(next_point_3D)) {
				continue; // Next point is in plane, it will be added separately.
			}
			if (plane.is_point_over(point_3D) == plane.is_point_over(next_point_3D)) {
				continue; // Both points on the same side of the plane, ignore.
			}

			// Edge crosses the plane, find and add the intersection point.
			Vector3 point_2D;
			if (plane.intersects_segment(point_3D, next_point_3D, &point_2D)) {
				point_2D = to_2D.xform(point_2D);
				points_2D[points_count++] = Vector2(point_2D.x, point_2D.y);
			}
		}
	}

	Vector<int> segment_indices;
	Vector2 segment[2];
	int inserted_index[3] = { -1, -1, -1 };

	// Insert points.
	for (int i = 0; i < points_count; ++i) {
		inserted_index[i] = _insert_point(points_2D[i]);
	}

	if (points_count == 2) {
		// Insert a single segment.
		segment[0] = points_2D[0];
		segment[1] = points_2D[1];
		_find_edge_intersections(segment, segment_indices);
		for (int i = 0; i < 2; ++i) {
			_add_vertex_idx_sorted(segment_indices, inserted_index[i]);
		}
		_merge_faces(segment_indices);
	}

	if (points_count == 3) {
		// Insert three segments.
		for (int edge_idx = 0; edge_idx < 3; ++edge_idx) {
			segment[0] = points_2D[edge_idx];
			segment[1] = points_2D[(edge_idx + 1) % 3];
			_find_edge_intersections(segment, segment_indices);
			for (int i = 0; i < 2; ++i) {
				_add_vertex_idx_sorted(segment_indices, inserted_index[(edge_idx + i) % 3]);
			}
			_merge_faces(segment_indices);
			segment_indices.clear();
		}
	}
}

void CSGBrushOperation::Build2DFaces::addFacesToMesh(MeshMerge &r_mesh_merge, bool p_smooth, bool p_invert, const Ref<Material> &p_material, bool p_from_b) {
	for (int face_idx = 0; face_idx < faces.size(); ++face_idx) {
		Face2D face = faces[face_idx];
		Vertex2D fv[3] = {
			vertices[face.vertex_idx[0]],
			vertices[face.vertex_idx[1]],
			vertices[face.vertex_idx[2]]
		};

		// Convert 2D vertex points to 3D.
		Vector3 points_3D[3];
		Vector2 uvs[3];
		for (int i = 0; i < 3; ++i) {
			Vector3 point_2D(fv[i].point.x, fv[i].point.y, 0);
			points_3D[i] = to_3D.xform(point_2D);
			uvs[i] = fv[i].uv;
		}

		r_mesh_merge.add_face(points_3D, uvs, p_smooth, p_invert, p_material, p_from_b);
	}
}

CSGBrushOperation::Build2DFaces::Build2DFaces(const CSGBrush &p_brush, int p_face_idx, float p_vertex_snap2) :
		vertex_snap2(p_vertex_snap2 * p_vertex_snap2) {
	// Convert 3D vertex points to 2D.
	Vector3 points_3D[3] = {
		p_brush.faces[p_face_idx].vertices[0],
		p_brush.faces[p_face_idx].vertices[1],
		p_brush.faces[p_face_idx].vertices[2],
	};

	plane = Plane(points_3D[0], points_3D[1], points_3D[2]);
	to_3D.origin = points_3D[0];
	to_3D.basis.set_axis(2, plane.normal);
	to_3D.basis.set_axis(0, (points_3D[1] - points_3D[2]).normalized());
	to_3D.basis.set_axis(1, to_3D.basis.get_axis(0).cross(to_3D.basis.get_axis(2)).normalized());
	to_2D = to_3D.affine_inverse();

	Face2D face;
	for (int i = 0; i < 3; i++) {
		Vertex2D vertex;
		Vector3 point_2D = to_2D.xform(points_3D[i]);
		vertex.point.x = point_2D.x;
		vertex.point.y = point_2D.y;
		vertex.uv = p_brush.faces[p_face_idx].uvs[i];
		vertices.push_back(vertex);
		face.vertex_idx[i] = i;
	}
	faces.push_back(face);
}

void CSGBrushOperation::update_faces(const CSGBrush &p_brush_a, const int p_face_idx_a, const CSGBrush &p_brush_b, const int p_face_idx_b, Build2DFaceCollection &p_collection, float p_vertex_snap) {
	Vector3 vertices_a[3] = {
		p_brush_a.faces[p_face_idx_a].vertices[0],
		p_brush_a.faces[p_face_idx_a].vertices[1],
		p_brush_a.faces[p_face_idx_a].vertices[2],
	};

	Vector3 vertices_b[3] = {
		p_brush_b.faces[p_face_idx_b].vertices[0],
		p_brush_b.faces[p_face_idx_b].vertices[1],
		p_brush_b.faces[p_face_idx_b].vertices[2],
	};

	// Don't use degenerate faces.
	bool has_degenerate = false;
	if (is_snapable(vertices_a[0], vertices_a[1], p_vertex_snap) ||
			is_snapable(vertices_a[0], vertices_a[2], p_vertex_snap) ||
			is_snapable(vertices_a[1], vertices_a[2], p_vertex_snap)) {
		p_collection.build2DFacesA[p_face_idx_a] = Build2DFaces();
		has_degenerate = true;
	}

	if (is_snapable(vertices_b[0], vertices_b[1], p_vertex_snap) ||
			is_snapable(vertices_b[0], vertices_b[2], p_vertex_snap) ||
			is_snapable(vertices_b[1], vertices_b[2], p_vertex_snap)) {
		p_collection.build2DFacesB[p_face_idx_b] = Build2DFaces();
		has_degenerate = true;
	}
	if (has_degenerate) {
		return;
	}

	// Ensure B has points either side of or in the plane of A.
	int in_plane_count = 0, over_count = 0, under_count = 0;
	Plane plane_a(vertices_a[0], vertices_a[1], vertices_a[2]);
	ERR_FAIL_COND_MSG(plane_a.normal == Vector3(), "Couldn't form plane from Brush A face.");

	for (int i = 0; i < 3; i++) {
		if (plane_a.has_point(vertices_b[i])) {
			in_plane_count++;
		} else if (plane_a.is_point_over(vertices_b[i])) {
			over_count++;
		} else {
			under_count++;
		}
	}
	// If all points under or over the plane, there is no intesection.
	if (over_count == 3 || under_count == 3) {
		return;
	}

	// Ensure A has points either side of or in the plane of B.
	in_plane_count = 0;
	over_count = 0;
	under_count = 0;
	Plane plane_b(vertices_b[0], vertices_b[1], vertices_b[2]);
	ERR_FAIL_COND_MSG(plane_b.normal == Vector3(), "Couldn't form plane from Brush B face.");

	for (int i = 0; i < 3; i++) {
		if (plane_b.has_point(vertices_a[i])) {
			in_plane_count++;
		} else if (plane_b.is_point_over(vertices_a[i])) {
			over_count++;
		} else {
			under_count++;
		}
	}
	// If all points under or over the plane, there is no intesection.
	if (over_count == 3 || under_count == 3) {
		return;
	}

	// Check for intersection using the SAT theorem.
	{
		// Edge pair cross product combinations.
		for (int i = 0; i < 3; i++) {
			Vector3 axis_a = (vertices_a[i] - vertices_a[(i + 1) % 3]).normalized();

			for (int j = 0; j < 3; j++) {
				Vector3 axis_b = (vertices_b[j] - vertices_b[(j + 1) % 3]).normalized();

				Vector3 sep_axis = axis_a.cross(axis_b);
				if (sep_axis == Vector3()) {
					continue; //colineal
				}
				sep_axis.normalize();

				real_t min_a = 1e20, max_a = -1e20;
				real_t min_b = 1e20, max_b = -1e20;

				for (int k = 0; k < 3; k++) {
					real_t d = sep_axis.dot(vertices_a[k]);
					min_a = MIN(min_a, d);
					max_a = MAX(max_a, d);
					d = sep_axis.dot(vertices_b[k]);
					min_b = MIN(min_b, d);
					max_b = MAX(max_b, d);
				}

				min_b -= (max_a - min_a) * 0.5;
				max_b += (max_a - min_a) * 0.5;

				real_t dmin = min_b - (min_a + max_a) * 0.5;
				real_t dmax = max_b - (min_a + max_a) * 0.5;

				if (dmin > CMP_EPSILON || dmax < -CMP_EPSILON) {
					return; // Does not contain zero, so they don't overlap.
				}
			}
		}
	}

	// If we're still here, the faces probably intersect, so add new faces.
	if (!p_collection.build2DFacesA.has(p_face_idx_a)) {
		p_collection.build2DFacesA[p_face_idx_a] = Build2DFaces(p_brush_a, p_face_idx_a, p_vertex_snap);
	}
	p_collection.build2DFacesA[p_face_idx_a].insert(p_brush_b, p_face_idx_b);

	if (!p_collection.build2DFacesB.has(p_face_idx_b)) {
		p_collection.build2DFacesB[p_face_idx_b] = Build2DFaces(p_brush_b, p_face_idx_b, p_vertex_snap);
	}
	p_collection.build2DFacesB[p_face_idx_b].insert(p_brush_a, p_face_idx_a);
}
