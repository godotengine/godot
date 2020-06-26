/*************************************************************************/
/*  csg.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/list.h"
#include "core/map.h"
#include "core/math/aabb.h"
#include "core/math/plane.h"
#include "core/math/transform.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/oa_hash_map.h"
#include "core/reference.h"
#include "core/vector.h"
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
	Vector<Ref<Material>> materials;

	inline void _regen_face_aabbs();

	// Create a brush from faces.
	void build_from_faces(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uvs, const Vector<bool> &p_smooth, const Vector<Ref<Material>> &p_materials, const Vector<bool> &p_invert_faces);
	void copy_from(const CSGBrush &p_brush, const Transform &p_xform);
};

struct CSGBrushOperation {
	enum Operation {
		OPERATION_UNION,
		OPERATION_INTERSECTION,
		OPERATION_SUBSTRACTION,
	};

	void merge_brushes(Operation p_operation, const CSGBrush &p_brush_a, const CSGBrush &p_brush_b, CSGBrush &r_merged_brush, float p_vertex_snap);

	struct MeshMerge {
		struct Face {
			bool from_b;
			bool inside;
			int points[3];
			Vector2 uvs[3];
			bool smooth;
			bool invert;
			int material_idx;
		};

		struct FaceBVH {
			int face;
			int left;
			int right;
			int next;
			Vector3 center;
			AABB aabb;
		};

		struct FaceBVHCmpX {
			_FORCE_INLINE_ bool operator()(const FaceBVH *p_left, const FaceBVH *p_right) const {
				return p_left->center.x < p_right->center.x;
			}
		};

		struct FaceBVHCmpY {
			_FORCE_INLINE_ bool operator()(const FaceBVH *p_left, const FaceBVH *p_right) const {
				return p_left->center.y < p_right->center.y;
			}
		};
		struct FaceBVHCmpZ {
			_FORCE_INLINE_ bool operator()(const FaceBVH *p_left, const FaceBVH *p_right) const {
				return p_left->center.z < p_right->center.z;
			}
		};

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

		Vector<Vector3> points;
		Vector<Face> faces;
		Map<Ref<Material>, int> materials;
		Map<Vector3, int> vertex_map;
		OAHashMap<VertexKey, int, VertexKeyHash> snap_cache;
		float vertex_snap;

		inline void _add_distance(List<real_t> &r_intersectionsA, List<real_t> &r_intersectionsB, bool p_from_B, real_t p_distance) const;
		inline bool _bvh_inside(FaceBVH *facebvhptr, int p_max_depth, int p_bvh_first, int p_face_idx) const;
		inline int _create_bvh(FaceBVH *facebvhptr, FaceBVH **facebvhptrptr, int p_from, int p_size, int p_depth, int &r_max_depth, int &r_max_alloc);

		void add_face(const Vector3 p_points[3], const Vector2 p_uvs[3], bool p_smooth, bool p_invert, const Ref<Material> &p_material, bool p_from_b);
		void mark_inside_faces();
	};

	struct Build2DFaces {
		struct Vertex2D {
			Vector2 point;
			Vector2 uv;
		};

		struct Face2D {
			int vertex_idx[3];
		};

		Vector<Vertex2D> vertices;
		Vector<Face2D> faces;
		Plane plane;
		Transform to_2D;
		Transform to_3D;
		float vertex_snap2;

		inline int _get_point_idx(const Vector2 &p_point);
		inline int _add_vertex(const Vertex2D &p_vertex);
		inline void _add_vertex_idx_sorted(Vector<int> &r_vertex_indices, int p_new_vertex_index);
		inline void _merge_faces(const Vector<int> &p_segment_indices);
		inline void _find_edge_intersections(const Vector2 p_segment_points[2], Vector<int> &r_segment_indices);
		inline int _insert_point(const Vector2 &p_point);

		void insert(const CSGBrush &p_brush, int p_brush_face);
		void addFacesToMesh(MeshMerge &r_mesh_merge, bool p_smooth, bool p_invert, const Ref<Material> &p_material, bool p_from_b);

		Build2DFaces() {}
		Build2DFaces(const CSGBrush &p_brush, int p_brush_face, float p_vertex_snap2);
	};

	struct Build2DFaceCollection {
		Map<int, Build2DFaces> build2DFacesA;
		Map<int, Build2DFaces> build2DFacesB;
	};

	void update_faces(const CSGBrush &p_brush_a, const int p_face_idx_a, const CSGBrush &p_brush_b, const int p_face_idx_b, Build2DFaceCollection &p_collection, float p_vertex_snap);
};

#endif // CSG_H
