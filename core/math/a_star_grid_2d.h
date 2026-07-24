/**************************************************************************/
/*  a_star_grid_2d.h                                                      */
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

#pragma once

#include "core/object/gdvirtual.gen.h"
#include "core/object/ref_counted.h"
#include "core/templates/local_vector.h"

class AStarGrid2D : public RefCounted {
	GDCLASS(AStarGrid2D, RefCounted);

public:
	enum DiagonalMode {
		DIAGONAL_MODE_ALWAYS,
		DIAGONAL_MODE_NEVER,
		DIAGONAL_MODE_AT_LEAST_ONE_WALKABLE,
		DIAGONAL_MODE_ONLY_IF_NO_OBSTACLES,
		DIAGONAL_MODE_MAX,
	};

	enum Heuristic {
		HEURISTIC_EUCLIDEAN,
		HEURISTIC_MANHATTAN,
		HEURISTIC_OCTILE,
		HEURISTIC_CHEBYSHEV,
		HEURISTIC_MAX,
	};

	enum CellShape {
		CELL_SHAPE_SQUARE,
		CELL_SHAPE_ISOMETRIC_RIGHT,
		CELL_SHAPE_ISOMETRIC_DOWN,
		CELL_SHAPE_MAX,
	};

private:
	enum ClusterAdjacency {
		CLUSTER_ADJACENCY_NONE,
		CLUSTER_ADJACENCY_LATERAL,
		CLUSTER_ADJACENCY_VERTICAL,
		CLUSTER_ADJACENCY_DIAGONAL,
	};

	enum HPAPathResult {
		HPA_PATH_RESULT_FOUND,
		HPA_PATH_RESULT_UNREACHABLE,
		HPA_PATH_RESULT_FAILED,
	};

	Rect2i region;
	Vector2 offset;
	Size2 cell_size = Size2(1, 1);
	bool dirty = false;
	CellShape cell_shape = CELL_SHAPE_SQUARE;

	bool jumping_enabled = false;
	DiagonalMode diagonal_mode = DIAGONAL_MODE_ALWAYS;
	Heuristic default_compute_heuristic = HEURISTIC_EUCLIDEAN;
	Heuristic default_estimate_heuristic = HEURISTIC_EUCLIDEAN;

	bool hpa_enabled = false;
	bool hpa_dirty = false;
	int32_t hpa_cluster_size = 5;

	// Number of clusters along each axis, used to map a cell to its cluster index.
	int32_t hpa_cluster_cols = 0;
	int32_t hpa_cluster_rows = 0;
	// Separate pass counter for the abstract graph search (independent of the cell-level `pass`).
	uint64_t hpa_pass = 1;

	struct Point {
		Vector2i id;

		Vector2 pos;
		real_t weight_scale = 1.0;

		// Used for pathfinding.
		Point *prev_point = nullptr;
		real_t g_score = 0;
		real_t f_score = 0;
		uint64_t open_pass = 0;
		uint64_t closed_pass = 0;

		// Used for getting last_closest_point.
		real_t abs_g_score = 0;
		real_t abs_f_score = 0;

		Point() {}

		Point(const Vector2i &p_id, const Vector2 &p_pos) :
				id(p_id), pos(p_pos) {}
	};

	struct SortPoints {
		_FORCE_INLINE_ bool operator()(const Point *p_left, const Point *p_right) const { // Returns true when the Point A is worse than Point B.
			if (p_left->f_score > p_right->f_score) {
				return true;
			} else if (p_left->f_score < p_right->f_score) {
				return false;
			} else {
				return p_left->g_score < p_right->g_score; // If the f_costs are the same then prioritize the points that are further away from the start.
			}
		}
	};

	struct HierEdge {
		int32_t to_node_idx = -1;
		real_t distance = 0.0;
		bool is_inter_cluster = false;
	};

	struct HierNode {
		int32_t id = -1;
		Vector2i pos;
		int32_t cluster_idx = -1;
		LocalVector<HierEdge> edges;

		// Used for the abstract A* search.
		real_t g_score = 0;
		real_t f_score = 0;
		int32_t prev_node_idx = -1;
		uint64_t open_pass = 0;
		uint64_t closed_pass = 0;
	};

	struct Cluster {
		int32_t cluster_width = 0;
		int32_t cluster_height = 0;
		Vector2i cluster_pos;
		LocalVector<int32_t> node_ids;
	};

	struct Entrance {
		int32_t cluster_1_idx = -1;
		int32_t cluster_2_idx = -1;
		Vector2i c1_pos; // First boundary cell of the entrance on cluster 1's side.
		Vector2i c2_pos; // First boundary cell of the entrance on cluster 2's side.
		int32_t width = 0;
		int32_t height = 0;
		ClusterAdjacency dir = CLUSTER_ADJACENCY_NONE;
	};

	LocalVector<Cluster> hpa_clusters;
	LocalVector<Entrance> hpa_entrances;
	LocalVector<HierNode> hpa_nodes;

	LocalVector<bool> solid_mask;
	LocalVector<LocalVector<Point>> points;
	Point *end = nullptr;
	Point *last_closest_point = nullptr;

	uint64_t pass = 1;

private: // Internal routines.
	_FORCE_INLINE_ size_t _to_mask_index(int32_t p_x, int32_t p_y) const {
		return ((p_y - region.position.y + 1) * (region.size.x + 2)) + p_x - region.position.x + 1;
	}

	_FORCE_INLINE_ bool _is_walkable(int32_t p_x, int32_t p_y) const {
		return !solid_mask[_to_mask_index(p_x, p_y)];
	}

	_FORCE_INLINE_ Point *_get_point(int32_t p_x, int32_t p_y) {
		if (region.has_point(Vector2i(p_x, p_y))) {
			return &points[p_y - region.position.y][p_x - region.position.x];
		}
		return nullptr;
	}

	_FORCE_INLINE_ void _set_solid_unchecked(int32_t p_x, int32_t p_y, bool p_solid) {
		solid_mask[_to_mask_index(p_x, p_y)] = p_solid;
	}

	_FORCE_INLINE_ void _set_solid_unchecked(const Vector2i &p_id, bool p_solid) {
		solid_mask[_to_mask_index(p_id.x, p_id.y)] = p_solid;
	}

	_FORCE_INLINE_ bool _get_solid_unchecked(const Vector2i &p_id) const {
		return solid_mask[_to_mask_index(p_id.x, p_id.y)];
	}

	_FORCE_INLINE_ Point *_get_point_unchecked(int32_t p_x, int32_t p_y) {
		return &points[p_y - region.position.y][p_x - region.position.x];
	}

	_FORCE_INLINE_ Point *_get_point_unchecked(const Vector2i &p_id) {
		return &points[p_id.y - region.position.y][p_id.x - region.position.x];
	}

	_FORCE_INLINE_ const Point *_get_point_unchecked(const Vector2i &p_id) const {
		return &points[p_id.y - region.position.y][p_id.x - region.position.x];
	}

	void _get_nbors(Point *p_point, LocalVector<Point *> &r_nbors);
	ClusterAdjacency _clusters_are_adjacent(const Cluster *p_c1, const Cluster *p_c2) const;
	void _create_entrances(int32_t p_c1_idx, int32_t p_c2_idx, ClusterAdjacency p_adjacent_dir);
	void _add_entrance(int32_t p_c1_idx, int32_t p_c2_idx, const Vector2i &p_c1_pos, const Vector2i &p_c2_pos, int32_t p_width, int32_t p_height, ClusterAdjacency p_dir);
	void _build_clusters();
	void _build_entrances();
	void _create_hpa_nodes();
	void _build_node_and_inter_edges(const Vector2i &p_pos1, int32_t p_c1_idx, const Vector2i &p_pos2, int32_t p_c2_idx);
	void _build_intra_edges();
	_FORCE_INLINE_ Rect2i _cluster_rect(const Cluster *p_c) const {
		return Rect2i(p_c->cluster_pos, Size2i(p_c->cluster_width, p_c->cluster_height));
	}
	int32_t _get_cluster_index(const Vector2i &p_pos) const;
	real_t _solve_bounded(const Vector2i &p_from, const Vector2i &p_to, const Rect2i &p_bounds, LocalVector<Vector2i> *r_path = nullptr);
	bool _abstract_astar(int32_t p_start_idx, int32_t p_goal_idx, LocalVector<Vector2i> &r_node_path);
	HPAPathResult _hpa_solve_abstract(const Vector2i &p_from, const Vector2i &p_to, LocalVector<Vector2i> &r_node_path);
	HPAPathResult _hpa_get_cell_path(const Vector2i &p_from, const Vector2i &p_to, LocalVector<Vector2i> &r_cells);
	Point *_jump(Point *p_from, Point *p_to);
	bool _solve(Point *p_begin_point, Point *p_end_point, bool p_allow_partial_path);
	Point *_forced_successor(int32_t p_x, int32_t p_y, int32_t p_dx, int32_t p_dy, bool p_inclusive = false);

protected:
	static void _bind_methods();

	virtual real_t _estimate_cost(const Vector2i &p_from_id, const Vector2i &p_end_id);
	virtual real_t _compute_cost(const Vector2i &p_from_id, const Vector2i &p_to_id);

	GDVIRTUAL2RC(real_t, _estimate_cost, Vector2i, Vector2i)
	GDVIRTUAL2RC(real_t, _compute_cost, Vector2i, Vector2i)

#ifndef DISABLE_DEPRECATED
	TypedArray<Vector2i> _get_id_path_bind_compat_88047(const Vector2i &p_from, const Vector2i &p_to);
	Vector<Vector2> _get_point_path_bind_compat_88047(const Vector2i &p_from, const Vector2i &p_to);
	static void _bind_compatibility_methods();
#endif

public:
	void set_region(const Rect2i &p_region);
	Rect2i get_region() const;

	void set_size(const Size2i &p_size);
	Size2i get_size() const;

	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;

	void set_cell_size(const Size2 &p_cell_size);
	Size2 get_cell_size() const;

	void set_cell_shape(CellShape p_cell_shape);
	CellShape get_cell_shape() const;

	void update();

	bool is_in_bounds(int32_t p_x, int32_t p_y) const;
	bool is_in_boundsv(const Vector2i &p_id) const;
	bool is_dirty() const;

	bool is_hpa_enabled() const;
	bool is_hpa_dirty() const;
	void set_hpa_enabled(bool p_enabled);
	int32_t get_hpa_cluster_size() const;
	void set_hpa_cluster_size(int32_t p_cluster_size);
	void update_hpa();

	void set_jumping_enabled(bool p_enabled);
	bool is_jumping_enabled() const;

	void set_diagonal_mode(DiagonalMode p_diagonal_mode);
	DiagonalMode get_diagonal_mode() const;

	void set_default_compute_heuristic(Heuristic p_heuristic);
	Heuristic get_default_compute_heuristic() const;

	void set_default_estimate_heuristic(Heuristic p_heuristic);
	Heuristic get_default_estimate_heuristic() const;

	void set_point_solid(const Vector2i &p_id, bool p_solid = true);
	bool is_point_solid(const Vector2i &p_id) const;

	void set_point_weight_scale(const Vector2i &p_id, real_t p_weight_scale);
	real_t get_point_weight_scale(const Vector2i &p_id) const;

	void fill_solid_region(const Rect2i &p_region, bool p_solid = true);
	void fill_weight_scale_region(const Rect2i &p_region, real_t p_weight_scale);

	void clear();

	Vector2 get_point_position(const Vector2i &p_id) const;
	TypedArray<Dictionary> get_point_data_in_region(const Rect2i &p_region) const;
	Vector<Vector2> get_point_path(const Vector2i &p_from, const Vector2i &p_to, bool p_allow_partial_path = false);
	TypedArray<Vector2i> get_id_path(const Vector2i &p_from, const Vector2i &p_to, bool p_allow_partial_path = false);
};

VARIANT_ENUM_CAST(AStarGrid2D::DiagonalMode);
VARIANT_ENUM_CAST(AStarGrid2D::Heuristic);
VARIANT_ENUM_CAST(AStarGrid2D::CellShape)
