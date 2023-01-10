/**************************************************************************/
/*  a_star.h                                                              */
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

#ifndef A_STAR_H
#define A_STAR_H

#include "core/oa_hash_map.h"
#include "core/reference.h"

/**
	A* pathfinding algorithm

	@author Juan Linietsky <reduzio@gmail.com>
*/

class AStar : public Reference {
	GDCLASS(AStar, Reference);
	friend class AStar2D;

	struct Point {
		Point() :
				neighbours(4u),
				unlinked_neighbours(4u) {}

		int id;
		Vector3 pos;
		real_t weight_scale;
		bool enabled;

		OAHashMap<int, Point *> neighbours;
		OAHashMap<int, Point *> unlinked_neighbours;

		// Used for pathfinding.
		Point *prev_point;
		real_t g_score;
		real_t f_score;
		uint64_t open_pass;
		uint64_t closed_pass;
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

	struct Segment {
		union {
			struct {
				int32_t u;
				int32_t v;
			};
			uint64_t key;
		};

		enum {
			NONE = 0,
			FORWARD = 1,
			BACKWARD = 2,
			BIDIRECTIONAL = FORWARD | BACKWARD
		};
		unsigned char direction;

		bool operator<(const Segment &p_s) const { return key < p_s.key; }
		Segment() {
			key = 0;
			direction = NONE;
		}
		Segment(int p_from, int p_to) {
			if (p_from < p_to) {
				u = p_from;
				v = p_to;
				direction = FORWARD;
			} else {
				u = p_to;
				v = p_from;
				direction = BACKWARD;
			}
		}
	};

	int last_free_id;
	uint64_t pass;

	OAHashMap<int, Point *> points;
	Set<Segment> segments;

	bool _solve(Point *begin_point, Point *end_point);

protected:
	static void _bind_methods();

	virtual real_t _estimate_cost(int p_from_id, int p_to_id);
	virtual real_t _compute_cost(int p_from_id, int p_to_id);

public:
	int get_available_point_id() const;

	void add_point(int p_id, const Vector3 &p_pos, real_t p_weight_scale = 1);
	Vector3 get_point_position(int p_id) const;
	void set_point_position(int p_id, const Vector3 &p_pos);
	real_t get_point_weight_scale(int p_id) const;
	void set_point_weight_scale(int p_id, real_t p_weight_scale);
	void remove_point(int p_id);
	bool has_point(int p_id) const;
	PoolVector<int> get_point_connections(int p_id);
	Array get_points();

	void set_point_disabled(int p_id, bool p_disabled = true);
	bool is_point_disabled(int p_id) const;

	void connect_points(int p_id, int p_with_id, bool bidirectional = true);
	void disconnect_points(int p_id, int p_with_id, bool bidirectional = true);
	bool are_points_connected(int p_id, int p_with_id, bool bidirectional = true) const;

	int get_point_count() const;
	int get_point_capacity() const;
	void reserve_space(int p_num_nodes);
	void clear();

	int get_closest_point(const Vector3 &p_point, bool p_include_disabled = false) const;
	Vector3 get_closest_position_in_segment(const Vector3 &p_point) const;

	PoolVector<Vector3> get_point_path(int p_from_id, int p_to_id);
	PoolVector<int> get_id_path(int p_from_id, int p_to_id);

	AStar();
	~AStar();
};

class AStar2D : public Reference {
	GDCLASS(AStar2D, Reference);
	AStar astar;

	bool _solve(AStar::Point *begin_point, AStar::Point *end_point);

protected:
	static void _bind_methods();

	virtual real_t _estimate_cost(int p_from_id, int p_to_id);
	virtual real_t _compute_cost(int p_from_id, int p_to_id);

public:
	int get_available_point_id() const;

	void add_point(int p_id, const Vector2 &p_pos, real_t p_weight_scale = 1);
	Vector2 get_point_position(int p_id) const;
	void set_point_position(int p_id, const Vector2 &p_pos);
	real_t get_point_weight_scale(int p_id) const;
	void set_point_weight_scale(int p_id, real_t p_weight_scale);
	void remove_point(int p_id);
	bool has_point(int p_id) const;
	PoolVector<int> get_point_connections(int p_id);
	Array get_points();

	void set_point_disabled(int p_id, bool p_disabled = true);
	bool is_point_disabled(int p_id) const;

	void connect_points(int p_id, int p_with_id, bool p_bidirectional = true);
	void disconnect_points(int p_id, int p_with_id, bool p_bidirectional = true);
	bool are_points_connected(int p_id, int p_with_id, bool p_bidirectional = true) const;

	int get_point_count() const;
	int get_point_capacity() const;
	void reserve_space(int p_num_nodes);
	void clear();

	int get_closest_point(const Vector2 &p_point, bool p_include_disabled = false) const;
	Vector2 get_closest_position_in_segment(const Vector2 &p_point) const;

	PoolVector<Vector2> get_point_path(int p_from_id, int p_to_id);
	PoolVector<int> get_id_path(int p_from_id, int p_to_id);

	AStar2D();
	~AStar2D();
};

#endif // A_STAR_H
