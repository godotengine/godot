/*************************************************************************/
/*  bvh_simple.h                                                         */
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

#ifndef BVH_SIMPLE_H
#define BVH_SIMPLE_H

#include "bvh_tree.h"
#include "dynamic_bvh.h"

#define GODOT_BVH_SIMPLE_TREE_CLASS BVH_Tree<T, 2, 256, false, true>

template <class T>
class BVH_SimpleT {
public:
	struct ID {
		ID() { _handle.set_invalid(); }
		BVHHandle _handle;

	public:
		bool is_valid() const { return !_handle.is_invalid(); }
	};

	ID insert(const AABB &p_box, T *p_userdata) {
		ID id;
		id._handle = _tree.item_add(p_userdata, true, p_box, 0, false, 0, 0);
		return id;
	}

	bool update(const ID &p_id, const AABB &p_box) {
		return _tree.item_move(p_id._handle, p_box);
	}

	void remove(const ID &p_id) {
		_tree.item_remove(p_id._handle);
	}

	void set_index(uint32_t p_index) { _index = p_index; }
	uint32_t get_index() const { return _index; }

	void clear() { _tree.clear(); }
	bool is_empty() const { return _tree.is_empty(); }
	void optimize_incremental(int passes) {
		if (passes) {
			_tree.incremental_optimize();
		}
	}

	/* Discouraged, but works as a reference on how it must be used */
	struct DefaultQueryResult {
		virtual bool operator()(T *p_data) = 0; //return true whether you want to continue the query
		virtual ~DefaultQueryResult() {}
	};

	template <class QueryResult>
	_FORCE_INLINE_ void aabb_query(const AABB &p_aabb, QueryResult &r_result) {
		typename GODOT_BVH_SIMPLE_TREE_CLASS::CullParams params;

		params.result_count_overall = 0;
		params.result_max = 0;
		params.result_array = nullptr;
		params.subindex_array = nullptr;
		params.mask = 0;
		params.pairable_type = 0;
		params.test_pairable_only = false;
		params.abb.from(p_aabb);

		_tree.cull_aabb(params, false, &r_result);
	}

	template <class QueryResult>
	_FORCE_INLINE_ void convex_query(const Plane *p_planes, int p_plane_count, const Vector3 *p_points, int p_point_count, QueryResult &r_result) {
		if (!p_plane_count) {
			return;
		}

		Vector<Vector3> convex_points = Geometry3D::compute_convex_mesh_points(p_planes, p_plane_count);
		if (convex_points.size() == 0) {
			return;
		}

		typename GODOT_BVH_SIMPLE_TREE_CLASS::CullParams params;
		params.result_count_overall = 0;
		params.result_max = 0;
		params.result_array = nullptr;
		params.subindex_array = nullptr;
		params.mask = 0;
		params.pairable_type = 0;

		params.hull.planes = p_planes;
		params.hull.num_planes = p_plane_count;
		params.hull.points = &convex_points[0];
		params.hull.num_points = convex_points.size();

		_tree.cull_convex(params, false, &r_result);
	}

	template <class QueryResult>
	_FORCE_INLINE_ void ray_query(const Vector3 &p_from, const Vector3 &p_to, QueryResult &r_result) {
		typename GODOT_BVH_SIMPLE_TREE_CLASS::CullParams params;

		params.result_count_overall = 0;
		params.result_max = 0;
		params.result_array = nullptr;
		params.subindex_array = nullptr;
		params.mask = 0;
		params.pairable_type = 0;

		params.segment.from = p_from;
		params.segment.to = p_to;

		_tree.cull_segment(params, false, &r_result);
	}

private:
	GODOT_BVH_SIMPLE_TREE_CLASS _tree;
	uint32_t _index = 0;
};

// Define this if you want to compare the performance of the old dynamic BVH.
// There is no runtime switching, only this compile time switching...
//#define GODOT_USE_OLD_DYNAMIC_BVH
#ifdef GODOT_USE_OLD_DYNAMIC_BVH
class BVH_Simple : public DynamicBVH {
};
#else
class BVH_Simple : public BVH_SimpleT<void> {
};
#endif

#endif
