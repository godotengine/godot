/*************************************************************************/
/*  csg.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef CSG_H
#define CSG_H

#include "core/dvector.h"
#include "core/map.h"
#include "core/math/aabb.h"
#include "core/math/plane.h"
#include "core/math/rect2.h"
#include "core/math/transform.h"
#include "core/math/vector3.h"
#include "core/oa_hash_map.h"
#include "scene/resources/material.h"

struct CSGBrush {

	struct Face {

		Vector3 vertices[3];
		Vector2 uvs[3];
		AABB aabb;
		bool smooth;
		bool invert;
		int material;
	};

	Vector<Face> faces;
	Vector<Ref<Material> > materials;

	void _regen_face_aabbs();
	//create a brush from faces
	void build_from_faces(const PoolVector<Vector3> &p_vertices, const PoolVector<Vector2> &p_uvs, const PoolVector<bool> &p_smooth, const PoolVector<Ref<Material> > &p_materials, const PoolVector<bool> &p_invert_faces);
	void copy_from(const CSGBrush &p_brush, const Transform &p_xform);

	void clear();
};

struct CSGBrushOperation {

	enum Operation {
		OPERATION_UNION,
		OPERATION_INTERSECTION,
		OPERATION_SUBSTRACTION,

	};

	struct MeshMerge {

		struct BVH {
			int face;
			int left;
			int right;
			int next;
			Vector3 center;
			AABB aabb;
		};

		struct BVHCmpX {

			bool operator()(const BVH *p_left, const BVH *p_right) const {

				return p_left->center.x < p_right->center.x;
			}
		};

		struct BVHCmpY {

			bool operator()(const BVH *p_left, const BVH *p_right) const {

				return p_left->center.y < p_right->center.y;
			}
		};
		struct BVHCmpZ {

			bool operator()(const BVH *p_left, const BVH *p_right) const {

				return p_left->center.z < p_right->center.z;
			}
		};

		int _bvh_count_intersections(BVH *bvhptr, int p_max_depth, int p_bvh_first, const Vector3 &p_begin, const Vector3 &p_end, int p_exclude) const;
		int _create_bvh(BVH *p_bvh, BVH **p_bb, int p_from, int p_size, int p_depth, int &max_depth, int &max_alloc);

		struct VertexKey {
			int32_t x, y, z;
			_FORCE_INLINE_ bool operator<(const VertexKey &p_key) const {
				if (x == p_key.x) {
					if (y == p_key.y) {
						return z < p_key.z;
					} else {
						return y < p_key.y;
					}
				} else {
					return x < p_key.x;
				}
			}

			_FORCE_INLINE_ bool operator==(const VertexKey &p_key) const {
				return (x == p_key.x && y == p_key.y && z == p_key.z);
			}
		};

		struct VertexKeyHash {
			static _FORCE_INLINE_ uint32_t hash(const VertexKey &p_vk) {
				uint32_t h = hash_djb2_one_32(p_vk.x);
				h = hash_djb2_one_32(p_vk.y, h);
				h = hash_djb2_one_32(p_vk.z, h);
				return h;
			}
		};

		OAHashMap<VertexKey, int, VertexKeyHash> snap_cache;

		Vector<Vector3> points;

		struct Face {
			bool from_b;
			bool inside;
			int points[3];
			Vector2 uvs[3];
			bool smooth;
			bool invert;
			int material_idx;
		};

		Vector<Face> faces;

		Map<Ref<Material>, int> materials;

		Map<Vector3, int> vertex_map;
		void add_face(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, const Vector2 &p_uv_a, const Vector2 &p_uv_b, const Vector2 &p_uv_c, bool p_smooth, bool p_invert, const Ref<Material> &p_material, bool p_from_b);
		//		void add_face(const Vector3 &p_a, const Vector3 &p_b, const Vector3 &p_c, bool p_from_b);

		float vertex_snap;
		void mark_inside_faces();
	};

	struct BuildPoly {

		Plane plane;
		Transform to_poly;
		Transform to_world;
		int face_index;

		struct Point {
			Vector2 point;
			Vector2 uv;
		};

		Vector<Point> points;

		struct Edge {
			bool outer;
			int points[2];
			Edge() {
				outer = false;
			}
		};

		Vector<Edge> edges;
		Ref<Material> material;
		bool smooth;
		bool invert;

		int base_edges; //edges from original triangle, even if split

		void _clip_segment(const CSGBrush *p_brush, int p_face, const Vector2 *segment, MeshMerge &mesh_merge, bool p_for_B);

		void create(const CSGBrush *p_brush, int p_face, MeshMerge &mesh_merge, bool p_for_B);
		void clip(const CSGBrush *p_brush, int p_face, MeshMerge &mesh_merge, bool p_for_B);
	};

	struct PolyPoints {

		Vector<int> points;

		Vector<Vector<int> > holes;
	};

	struct EdgeSort {
		int edge;
		int prev_point;
		int edge_point;
		float angle;
		bool operator<(const EdgeSort &p_edge) const { return angle < p_edge.angle; }
	};

	struct CallbackData {
		const CSGBrush *A;
		const CSGBrush *B;
		int face_a;
		CSGBrushOperation *self;
		Map<int, BuildPoly> build_polys_A;
		Map<int, BuildPoly> build_polys_B;
	};

	void _add_poly_points(const BuildPoly &p_poly, int p_edge, int p_from_point, int p_to_point, const Vector<Vector<int> > &vertex_process, Vector<bool> &edge_process, Vector<PolyPoints> &r_poly);
	void _add_poly_outline(const BuildPoly &p_poly, int p_from_point, int p_to_point, const Vector<Vector<int> > &vertex_process, Vector<int> &r_outline);
	void _merge_poly(MeshMerge &mesh, int p_face_idx, const BuildPoly &p_poly, bool p_from_b);

	void _collision_callback(const CSGBrush *A, int p_face_a, Map<int, BuildPoly> &build_polys_a, const CSGBrush *B, int p_face_b, Map<int, BuildPoly> &build_polys_b, MeshMerge &mesh_merge);

	static void _collision_callbacks(void *ud, int p_face_b);
	void merge_brushes(Operation p_operation, const CSGBrush &p_A, const CSGBrush &p_B, CSGBrush &result, float p_snap = 0.001);
};

#endif // CSG_H
