/**************************************************************************/
/*  bentley_ottmann.h                                                     */
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

#ifndef BENTLEY_OTTMANN_H
#define BENTLEY_OTTMANN_H

#include "core/math/vector2.h"
#include "core/templates/local_vector.h"

class BentleyOttmann {
public:
	BentleyOttmann(Vector<Vector2> p_edges, Vector<int> p_winding, bool p_winding_even_odd);
	Vector<Vector2> out_points;
	Vector<int32_t> out_triangles;

private:
	int winding_mask;

	struct TreeNode {
		uint32_t parent = 0;
		uint32_t left = 0;
		uint32_t right = 0;
		uint32_t prev = 0;
		uint32_t next = 0;
		bool is_heavy = false;
		uint32_t element = 0;
	};
	thread_local static LocalVector<TreeNode> tree_nodes;

	struct ListNode {
		uint32_t anchor = 0;
		uint32_t prev = 0;
		uint32_t next = 0;
		uint32_t element = 0;
	};
	thread_local static LocalVector<ListNode> list_nodes;

	struct Slice {
		int64_t x;
		uint32_t vertical_tree;
	};
	thread_local static LocalVector<Slice> slices;
	uint32_t slices_tree;

	struct Point {
		int64_t x;
		int64_t y;
		int64_t factor;
		uint32_t outgoing_tree;
		uint32_t listnode_edge_prev;
		uint32_t listnode_edge_next;
		uint32_t used = 0;
	};
	thread_local static LocalVector<Point> points;
	uint32_t points_tree;

	struct Edge {
		uint32_t point_start;
		uint32_t point_end;
		uint32_t point_outgoing;
		uint32_t treenode_edges;
		uint32_t treenode_outgoing;
		uint32_t points_prev_list;
		uint32_t points_next_list;
		int64_t dir_x;
		int64_t dir_y;
		int64_t cross;
		int64_t min_y;
		int64_t max_y;
		int winding;
		int winding_total = 0;
	};
	thread_local static LocalVector<Edge> edges;
	uint32_t edges_tree;

	struct Vertical {
		int64_t y;
		bool is_start;
	};
	thread_local static LocalVector<Vertical> verticals;

	thread_local static LocalVector<uint32_t> triangles;

	uint32_t add_slice(int64_t p_x);
	uint32_t add_point(int64_t p_x, int64_t p_y, int64_t p_factor);
	void point_add_outgoing(uint32_t p_point, uint32_t p_edge);
	void add_edge(uint32_t p_point_start, uint32_t p_point_end, int p_winding);
	void add_vertical_edge(uint32_t p_slice, int64_t p_y_start, int64_t p_y_end);
	void edge_intersect_x(uint32_t p_edge, int64_t p_x);
	void edge_intersect_edge(uint32_t p_edge1, uint32_t p_edge2);
	void edge_set_outgoing(uint32_t p_edge, uint32_t p_point);
	void edge_add_point_before(uint32_t p_edge, uint32_t p_point);
	void edge_add_point_after(uint32_t p_edge, uint32_t p_point);
	uint32_t get_edge_before(int64_t p_x, int64_t p_y);
	uint32_t get_edge_before_point(uint32_t p_point);
	void check_intersection(uint32_t p_treenode_edge);

	int cmp_point_point(int64_t p_x, int64_t p_y, int64_t factor, uint32_t p_point);
	int cmp_point_edge(uint32_t p_point, uint32_t p_edge);
	int cmp_cross(uint32_t p_point1, uint32_t p_point2, uint32_t p_point_rel);

	uint32_t tree_create(uint32_t p_element = 0);
	void tree_clear(uint32_t p_tree);
	void tree_insert(uint32_t p_insert_item, uint32_t p_insert_after);
	void tree_remove(uint32_t p_remove_item);
	void tree_rotate(uint32_t p_item);
	void tree_swap(uint32_t p_item1, uint32_t p_item2);
	void tree_replace(uint32_t p_item1, uint32_t p_item2);

	uint32_t list_create(uint32_t p_element = 0);
	void list_insert(uint32_t p_insert_item, uint32_t p_list);
	void list_remove(uint32_t p_remove_item);
};

#endif // BENTLEY_OTTMANN_H
