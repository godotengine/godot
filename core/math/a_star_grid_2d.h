/*************************************************************************/
/*  a_star_grid_2d.h                                                     */
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

#ifndef A_STAR_GRID_2D_H
#define A_STAR_GRID_2D_H

#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"
#include "core/object/script_language.h"
#include "core/templates/list.h"
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

private:
	Size2i size;
	Vector2 offset;
	Size2 cell_size = Size2(1, 1);
	bool dirty = false;

	bool jumping_enabled = false;
	DiagonalMode diagonal_mode = DIAGONAL_MODE_ALWAYS;
	Heuristic default_heuristic = HEURISTIC_EUCLIDEAN;

	struct Point {
		Vector2i id;

		bool solid = false;
		Vector2 pos;

		// Used for pathfinding.
		Point *prev_point = nullptr;
		real_t g_score = 0;
		real_t f_score = 0;
		uint64_t open_pass = 0;
		uint64_t closed_pass = 0;

		Point() {}

		Point(const Vector2i &p_id, const Vector2 &p_pos) :
				id(p_id), pos(p_pos) {}
	};

	struct SortPoints {
		_FORCE_INLINE_ bool operator()(const Point *A, const Point *B) const { // Returns true when the Point A is worse than Point B.
			if (A->f_score > B->f_score) {
				return true;
			} else if (A->f_score < B->f_score) {
				return false;
			} else {
				return A->g_score < B->g_score; // If the f_costs are the same then prioritize the points that are further away from the start.
			}
		}
	};

	LocalVector<LocalVector<Point>> points;
	Point *end = nullptr;

	uint64_t pass = 1;

private: // Internal routines.
	_FORCE_INLINE_ bool _is_walkable(int64_t p_x, int64_t p_y) const {
		if (p_x >= 0 && p_y >= 0 && p_x < size.width && p_y < size.height) {
			return !points[p_y][p_x].solid;
		}
		return false;
	}

	_FORCE_INLINE_ Point *_get_point(int64_t p_x, int64_t p_y) {
		if (p_x >= 0 && p_y >= 0 && p_x < size.width && p_y < size.height) {
			return &points[p_y][p_x];
		}
		return nullptr;
	}

	_FORCE_INLINE_ Point *_get_point_unchecked(int64_t p_x, int64_t p_y) {
		return &points[p_y][p_x];
	}

	void _get_nbors(Point *p_point, List<Point *> &r_nbors);
	Point *_jump(Point *p_from, Point *p_to);
	bool _solve(Point *p_begin_point, Point *p_end_point);

protected:
	static void _bind_methods();

	virtual real_t _estimate_cost(const Vector2i &p_from_id, const Vector2i &p_to_id);
	virtual real_t _compute_cost(const Vector2i &p_from_id, const Vector2i &p_to_id);

	GDVIRTUAL2RC(real_t, _estimate_cost, Vector2i, Vector2i)
	GDVIRTUAL2RC(real_t, _compute_cost, Vector2i, Vector2i)

public:
	void set_size(const Size2i &p_size);
	Size2i get_size() const;

	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;

	void set_cell_size(const Size2 &p_cell_size);
	Size2 get_cell_size() const;

	void update();

	int get_width() const;
	int get_height() const;

	bool is_in_bounds(int p_x, int p_y) const;
	bool is_in_boundsv(const Vector2i &p_id) const;
	bool is_dirty() const;

	void set_jumping_enabled(bool p_enabled);
	bool is_jumping_enabled() const;

	void set_diagonal_mode(DiagonalMode p_diagonal_mode);
	DiagonalMode get_diagonal_mode() const;

	void set_default_heuristic(Heuristic p_heuristic);
	Heuristic get_default_heuristic() const;

	void set_point_solid(const Vector2i &p_id, bool p_solid = true);
	bool is_point_solid(const Vector2i &p_id) const;

	void clear();

	Vector<Vector2> get_point_path(const Vector2i &p_from, const Vector2i &p_to);
	Vector<Vector2> get_id_path(const Vector2i &p_from, const Vector2i &p_to);
};

VARIANT_ENUM_CAST(AStarGrid2D::DiagonalMode);
VARIANT_ENUM_CAST(AStarGrid2D::Heuristic);

#endif // A_STAR_GRID_2D_H
