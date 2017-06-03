/*************************************************************************/
/*  triangle_mesh.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "face3.h"
#include "reference.h"
class TriangleMesh : public Reference {

	GDCLASS(TriangleMesh, Reference);

	struct Triangle {

		Vector3 normal;
		int indices[3];
	};

	PoolVector<Triangle> triangles;
	PoolVector<Vector3> vertices;

	struct BVH {

		Rect3 aabb;
		Vector3 center; //used for sorting
		int left;
		int right;

		int face_index;
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

	int _create_bvh(BVH *p_bvh, BVH **p_bb, int p_from, int p_size, int p_depth, int &max_depth, int &max_alloc);

	PoolVector<BVH> bvh;
	int max_depth;
	bool valid;

public:
	bool is_valid() const;
	bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_point, Vector3 &r_normal) const;
	bool intersect_ray(const Vector3 &p_begin, const Vector3 &p_dir, Vector3 &r_point, Vector3 &r_normal) const;
	Vector3 get_area_normal(const Rect3 &p_aabb) const;
	PoolVector<Face3> get_faces() const;

	void create(const PoolVector<Vector3> &p_faces);
	TriangleMesh();
};

#endif // TRIANGLE_MESH_H
