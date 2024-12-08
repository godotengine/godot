/**************************************************************************/
/*  csg.h                                                                 */
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

#ifndef CSG_H
#define CSG_H

#include "core/math/aabb.h"
#include "core/math/plane.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/object/ref_counted.h"
#include "core/templates/list.h"
#include "core/templates/oa_hash_map.h"
#include "core/templates/vector.h"
#include "scene/resources/material.h"

struct CSGBrush {
	struct Face {
		Vector3 vertices[3];
		Vector2 uvs[3];
		AABB aabb;
		bool smooth = false;
		bool invert = false;
		int material = 0;
	};

	Vector<Face> faces;
	Vector<Ref<Material>> materials;

	inline void _regen_face_aabbs();

	// Create a brush from faces.
	void build_from_faces(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uvs, const Vector<bool> &p_smooth, const Vector<Ref<Material>> &p_materials, const Vector<bool> &p_invert_faces);
	void copy_from(const CSGBrush &p_brush, const Transform3D &p_xform);
};

struct CSGBrushOperation {
	enum Operation {
		OPERATION_UNION,
		OPERATION_INTERSECTION,
		OPERATION_SUBTRACTION,
	};

	void merge_brushes(Operation p_operation, const CSGBrush &p_brush_a, const CSGBrush &p_brush_b, CSGBrush &r_merged_brush, float p_vertex_snap);

	struct MeshMerge {
		struct Face {
			bool from_b = false;
			bool inside = false;
			int points[3] = {};
			Vector2 uvs[3];
			bool smooth = false;
			bool invert = false;
			int material_idx = 0;
		};

		struct FaceBVH {
			int face = 0;
			int left = 0;
			int right = 0;
			int next = 0;
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
				uint32_t h = hash_murmur3_one_32(p_vk.x);
				h = hash_murmur3_one_32(p_vk.y, h);
				h = hash_murmur3_one_32(p_vk.z, h);
				return h;
			}
		};
		struct Intersection {
			bool found = false;
			real_t conormal = FLT_MAX;
			real_t distance_squared = FLT_MAX;
			real_t origin_angle = FLT_MAX;
		};

		struct IntersectionDistance {
			bool is_conormal;
			real_t distance_squared;
		};

		Vector<Vector3> points;
		Vector<Face> faces;
		HashMap<Ref<Material>, int> materials;
		HashMap<Vector3, int> vertex_map;
		OAHashMap<VertexKey, int, VertexKeyHash> snap_cache;
		float vertex_snap = 0.0;

		inline void _add_distance(List<IntersectionDistance> &r_intersectionsA, List<IntersectionDistance> &r_intersectionsB, bool p_from_B, real_t p_distance, bool p_is_conormal) const;
		inline bool _bvh_inside(FaceBVH *r_facebvhptr, int p_max_depth, int p_bvh_first, int p_face_idx) const;
		inline int _create_bvh(FaceBVH *r_facebvhptr, FaceBVH **r_facebvhptrptr, int p_from, int p_size, int p_depth, int &r_max_depth, int &r_max_alloc);

		void add_face(const Vector3 p_points[3], const Vector2 p_uvs[3], bool p_smooth, bool p_invert, const Ref<Material> &p_material, bool p_from_b);
		void mark_inside_faces();
	};

	struct Build2DFaces {
		struct Vertex2D {
			Vector2 point;
			Vector2 uv;
		};

		struct Face2D {
			int vertex_idx[3] = {};
		};

		Vector<Vertex2D> vertices;
		Vector<Face2D> faces;
		Plane plane;
		Transform3D to_2D;
		Transform3D to_3D;
		float vertex_snap2 = 0.0;

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
		HashMap<int, Build2DFaces> build2DFacesA;
		HashMap<int, Build2DFaces> build2DFacesB;
	};

	void update_faces(const CSGBrush &p_brush_a, const int p_face_idx_a, const CSGBrush &p_brush_b, const int p_face_idx_b, Build2DFaceCollection &p_collection, float p_vertex_snap);
};

#endif // CSG_H
