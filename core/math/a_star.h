/*************************************************************************/
/*  a_star.h                                                        */
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
#ifndef ASTAR_H
#define ASTAR_H

#include "reference.h"
#include "self_list.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class AStar : public Reference {

	GDCLASS(AStar, Reference)

	uint64_t pass;

	struct Point {

		SelfList<Point> list;

		int id;
		Vector3 pos;
		real_t weight_scale;
		uint64_t last_pass;

		Vector<Point *> neighbours;

		//used for pathfinding
		Point *prev_point;
		real_t distance;

		Point()
			: list(this) {}
	};

	Map<int, Point *> points;

	struct Segment {
		union {
			struct {
				int32_t from;
				int32_t to;
			};
			uint64_t key;
		};

		Point *from_point;
		Point *to_point;

		bool operator<(const Segment &p_s) const { return key < p_s.key; }
		Segment() { key = 0; }
		Segment(int p_from, int p_to) {
			if (p_from > p_to) {
				SWAP(p_from, p_to);
			}

			from = p_from;
			to = p_to;
		}
	};

	Set<Segment> segments;

	bool _solve(Point *begin_point, Point *end_point);

protected:
	static void _bind_methods();

	virtual float _estimate_cost(int p_from_id, int p_to_id);
	virtual float _compute_cost(int p_from_id, int p_to_id);

public:
	int get_available_point_id() const;

	void add_point(int p_id, const Vector3 &p_pos, real_t p_weight_scale = 1);
	Vector3 get_point_pos(int p_id) const;
	real_t get_point_weight_scale(int p_id) const;
	void remove_point(int p_id);

	void connect_points(int p_id, int p_with_id);
	void disconnect_points(int p_id, int p_with_id);
	bool are_points_connected(int p_id, int p_with_id) const;

	void clear();

	int get_closest_point(const Vector3 &p_point) const;
	Vector3 get_closest_pos_in_segment(const Vector3 &p_point) const;

	PoolVector<Vector3> get_point_path(int p_from_id, int p_to_id);
	PoolVector<int> get_id_path(int p_from_id, int p_to_id);

	AStar();
	~AStar();
};

#endif // ASTAR_H
