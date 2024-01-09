/**************************************************************************/
/*  bentley_ottmann.cpp                                                   */
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

#include "bentley_ottmann.h"

#define EXP_MIN -65536

thread_local LocalVector<BentleyOttmann::TreeNode> BentleyOttmann::tree_nodes;
thread_local LocalVector<BentleyOttmann::ListNode> BentleyOttmann::list_nodes;
thread_local LocalVector<BentleyOttmann::Slice> BentleyOttmann::slices;
thread_local LocalVector<BentleyOttmann::Point> BentleyOttmann::points;
thread_local LocalVector<BentleyOttmann::Edge> BentleyOttmann::edges;
thread_local LocalVector<BentleyOttmann::Vertical> BentleyOttmann::verticals;
thread_local LocalVector<uint32_t> BentleyOttmann::triangles;

BentleyOttmann::BentleyOttmann(Vector<Vector2> p_edges, Vector<int> p_winding, bool p_winding_even_odd) {
	tree_nodes.clear();
	list_nodes.clear();
	slices.clear();
	points.clear();
	edges.clear();
	verticals.clear();
	triangles.clear();
	// The cost of an explicit nil node is lower than having a special nil value.
	// This also ensures that tree_nodes[0].element is 0 instead of a null pointer exception.
	TreeNode nil_node;
	tree_nodes.push_back(nil_node);
	edges_tree = tree_create();
	slices_tree = tree_create();
	points_tree = tree_create();
	winding_mask = p_winding_even_odd ? 1 : -1;

	ERR_FAIL_COND(p_edges.size() & 1);
	ERR_FAIL_COND((p_edges.size() >> 1) != p_winding.size());
	if (p_edges.size() < 1) {
		return;
	}
	int x_exp = EXP_MIN;
	int y_exp = EXP_MIN;
	for (int i = 0; i < p_edges.size(); i++) {
		if (isnormal(p_edges[i].x)) {
			int exp;
			frexp(p_edges[i].x, &exp);
			if (x_exp < exp) {
				x_exp = exp;
			}
		}
		if (isnormal(p_edges[i].y)) {
			int exp;
			frexp(p_edges[i].y, &exp);
			if (y_exp < exp) {
				y_exp = exp;
			}
		}
	}
	if (x_exp == EXP_MIN) {
		x_exp = 0;
	} else {
		x_exp -= 20;
	}
	if (y_exp == EXP_MIN) {
		y_exp = 0;
	} else {
		y_exp -= 20;
	}
	for (int i = 0, j = 0; i < p_winding.size(); i++, j += 2) {
		int64_t start_x = static_cast<int64_t>(ldexp(p_edges[j].x, -x_exp));
		int64_t start_y = static_cast<int64_t>(ldexp(p_edges[j].y, -y_exp));
		int64_t end_x = static_cast<int64_t>(ldexp(p_edges[j + 1].x, -x_exp));
		int64_t end_y = static_cast<int64_t>(ldexp(p_edges[j + 1].y, -y_exp));
		if (start_x < end_x) {
			add_edge(add_point(start_x, start_y, 1), add_point(end_x, end_y, 1), p_winding[i]);
		} else if (start_x > end_x) {
			add_edge(add_point(end_x, end_y, 1), add_point(start_x, start_y, 1), -p_winding[i]);
		} else if (start_y < end_y) {
			add_vertical_edge(add_slice(start_x), start_y, end_y);
		} else if (start_y > end_y) {
			add_vertical_edge(add_slice(start_x), end_y, start_y);
		}
	}

	uint32_t slice_iter = tree_nodes[slices_tree].next;
	uint32_t point_iter = points_tree;
	while (true) {
		uint32_t point_iter_next = tree_nodes[point_iter].next;
		if (unlikely(point_iter_next == points_tree || (slice_iter != slices_tree && points[tree_nodes[point_iter_next].element].x >= slices[tree_nodes[slice_iter].element].x * points[tree_nodes[point_iter_next].element].factor))) {
			// There's at least two points per slice, so slices are less likely than points.
			if (unlikely(slice_iter == slices_tree)) {
				// And the end of the shape only happens once in the loop.
				break;
			}
			uint32_t slice = tree_nodes[slice_iter].element;
			slice_iter = tree_nodes[slice_iter].next;
			int64_t slice_x = slices[slice].x;

			// Mark intersection of passthrough edges with vertical edges.
			uint32_t vertical_iter = tree_nodes[slices[slice].vertical_tree].next;
			while (vertical_iter != slices[slice].vertical_tree) {
				DEV_ASSERT(verticals[tree_nodes[vertical_iter].element].is_start);
				uint32_t treenode_edge = get_edge_before(slice_x, verticals[tree_nodes[vertical_iter].element].y);
				vertical_iter = tree_nodes[vertical_iter].next;
				DEV_ASSERT(vertical_iter != slices[slice].vertical_tree);
				DEV_ASSERT(!verticals[tree_nodes[vertical_iter].element].is_start);
				int64_t y_end = verticals[tree_nodes[vertical_iter].element].y;
				while (tree_nodes[treenode_edge].next != edges_tree) {
					treenode_edge = tree_nodes[treenode_edge].next;
					const Edge &edge = edges[tree_nodes[treenode_edge].element];
					if (y_end * edge.dir_x - slice_x * edge.dir_y <= edge.cross) {
						break;
					}
					edge_intersect_x(tree_nodes[treenode_edge].element, slice_x);
				}
				vertical_iter = tree_nodes[vertical_iter].next;
			}
		} else {
			uint32_t point = tree_nodes[point_iter = point_iter_next].element;
			uint32_t treenode_edge_before = get_edge_before_point(point);

			// Find and remove edges going through this point.
			while (tree_nodes[treenode_edge_before].next != edges_tree) {
				uint32_t edge = tree_nodes[tree_nodes[treenode_edge_before].next].element;
				if (edges[edge].point_end == point) {
					edge_set_outgoing(edge, point);
				} else {
					if (cmp_point_edge(point, edge) < 0) {
						break;
					}
					edge_set_outgoing(edge, point);
					// An edge merely passing through this point will be re-added below.
					point_add_outgoing(point, edge);
				}
				tree_remove(tree_nodes[treenode_edge_before].next);
			}

			// Add point to surrounding edges.
			if (treenode_edge_before != edges_tree) {
				edge_add_point_after(tree_nodes[treenode_edge_before].element, point);
			}
			if (tree_nodes[treenode_edge_before].next != edges_tree) {
				edge_add_point_before(tree_nodes[tree_nodes[treenode_edge_before].next].element, point);
			}

			// Add outgoing edges.
			int winding = 0;
			if (treenode_edge_before != edges_tree) {
				winding = edges[tree_nodes[treenode_edge_before].element].winding_total;
			}
			uint32_t treenode_outgoing_iter = tree_nodes[points[point].outgoing_tree].next;
			while (treenode_outgoing_iter != points[point].outgoing_tree) {
				winding += edges[tree_nodes[treenode_outgoing_iter].element].winding;
				edges[tree_nodes[treenode_outgoing_iter].element].winding_total = winding;
				treenode_outgoing_iter = tree_nodes[treenode_outgoing_iter].next;
			}
			treenode_outgoing_iter = tree_nodes[points[point].outgoing_tree].prev;
			while (treenode_outgoing_iter != points[point].outgoing_tree) {
				tree_insert(edges[tree_nodes[treenode_outgoing_iter].element].treenode_edges, treenode_edge_before);
				treenode_outgoing_iter = tree_nodes[treenode_outgoing_iter].prev;
			}

			// Check intersections.
			if (treenode_edge_before != edges_tree && tree_nodes[treenode_edge_before].next != edges_tree) {
				check_intersection(treenode_edge_before);
			}
			if (tree_nodes[points[point].outgoing_tree].prev != points[point].outgoing_tree) {
				uint32_t check = edges[tree_nodes[tree_nodes[points[point].outgoing_tree].prev].element].treenode_edges;
				if (tree_nodes[check].next != edges_tree) {
					check_intersection(check);
				}
			}

			// Cleanup.
			tree_clear(points[point].outgoing_tree);
		}
	}

	DEV_ASSERT(tree_nodes[edges_tree].right == 0);

	// Optimize points and flush to final buffers.
	DEV_ASSERT((triangles.size() % 3) == 0);
	for (uint32_t i = 0; i < triangles.size();) {
		if (triangles[i] == triangles[i + 1] || triangles[i] == triangles[i + 2] || triangles[i + 1] == triangles[i + 2]) {
			i += 3;
			continue;
		}
		for (uint32_t j = 0; j < 3; i++, j++) {
			if (!points[triangles[i]].used) {
				out_points.push_back(Vector2(ldexp(static_cast<real_t>(points[triangles[i]].x / points[triangles[i]].factor), x_exp), ldexp(static_cast<real_t>(points[triangles[i]].y / points[triangles[i]].factor), y_exp)));
				points[triangles[i]].used = out_points.size();
			}
			out_triangles.push_back(points[triangles[i]].used - 1);
		}
	}
}

uint32_t BentleyOttmann::add_slice(int64_t p_x) {
	uint32_t insert_after = slices_tree;
	uint32_t current = tree_nodes[slices_tree].right;
	if (current) {
		while (true) {
			int64_t x = p_x - slices[tree_nodes[current].element].x;
			if (x < 0) {
				if (tree_nodes[current].left) {
					current = tree_nodes[current].left;
					continue;
				}
				insert_after = tree_nodes[current].prev;
				break;
			}
			if (x > 0) {
				if (tree_nodes[current].right) {
					current = tree_nodes[current].right;
					continue;
				}
				insert_after = current;
				break;
			}
			return tree_nodes[current].element;
		}
	}
	Slice slice;
	slice.x = p_x;
	slice.vertical_tree = tree_create();
	tree_insert(tree_create(slices.size()), insert_after);
	slices.push_back(slice);
	return slices.size() - 1;
}

uint32_t BentleyOttmann::add_point(int64_t p_x, int64_t p_y, int64_t p_factor) {
	DEV_ASSERT(p_factor > 0 && p_factor < 0x100000000000LL);
	if (p_factor > 1 && (p_x % p_factor) == 0 && (p_y % p_factor) == 0) {
		// Factor 1 offers a faster path for some operations, since no-overflow is guaranteed.
		// Try to optimize only to 1. Optimizing to 2 or more is pointless.
		p_x /= p_factor;
		p_y /= p_factor;
		p_factor = 1;
	}
	uint32_t insert_after = points_tree;
	uint32_t current = tree_nodes[points_tree].right;
	if (current) {
		while (true) {
			int cmp = cmp_point_point(p_x, p_y, p_factor, tree_nodes[current].element);
			if (cmp < 0) {
				if (tree_nodes[current].left) {
					current = tree_nodes[current].left;
					continue;
				}
				insert_after = tree_nodes[current].prev;
				break;
			}
			if (cmp > 0) {
				if (tree_nodes[current].right) {
					current = tree_nodes[current].right;
					continue;
				}
				insert_after = current;
				break;
			}
			return tree_nodes[current].element;
		}
	}
	Point point;
	point.x = p_x;
	point.y = p_y;
	point.factor = p_factor;
	point.outgoing_tree = tree_create();
	point.listnode_edge_prev = list_create(points.size());
	point.listnode_edge_next = list_create(points.size());
	tree_insert(tree_create(points.size()), insert_after);
	points.push_back(point);
	return points.size() - 1;
}

void BentleyOttmann::point_add_outgoing(uint32_t p_point, uint32_t p_edge) {
retry:
	DEV_ASSERT(tree_nodes[edges[p_edge].treenode_outgoing].parent == 0);
	uint32_t current = tree_nodes[points[p_point].outgoing_tree].right;
	if (!current) {
		tree_insert(edges[p_edge].treenode_outgoing, points[p_point].outgoing_tree);
		return;
	}
	while (true) {
		uint32_t point = edges[tree_nodes[current].element].point_end;
		int64_t cmp = points[point].x * edges[p_edge].dir_y - points[point].y * edges[p_edge].dir_x + edges[p_edge].cross;
		if (cmp < 0) {
			if (tree_nodes[current].left) {
				current = tree_nodes[current].left;
				continue;
			}
			tree_insert(edges[p_edge].treenode_outgoing, tree_nodes[current].prev);
			return;
		}
		if (cmp > 0) {
			if (tree_nodes[current].right) {
				current = tree_nodes[current].right;
				continue;
			}
			tree_insert(edges[p_edge].treenode_outgoing, current);
			return;
		}
		uint32_t point_end = edges[p_edge].point_end;
		if (points[point_end].x >= points[point].x) {
			edges[tree_nodes[current].element].winding += edges[p_edge].winding;
			if (points[point_end].x == points[point].x) {
				DEV_ASSERT(points[point_end].y == points[point].y);
				return;
			}
			p_point = point;
			edges[p_edge].point_outgoing = p_point;
			goto retry;
		}
		tree_replace(edges[p_edge].treenode_outgoing, current);
		edges[p_edge].winding += edges[tree_nodes[current].element].winding;
		p_edge = tree_nodes[current].element;
		p_point = point_end;
		edges[p_edge].point_outgoing = p_point;
		goto retry;
	}
}

void BentleyOttmann::add_edge(uint32_t p_point_start, uint32_t p_point_end, int p_winding) {
	DEV_ASSERT(points[p_point_start].factor == 1 && points[p_point_end].factor == 1);
	Edge edge;
	edge.point_start = edge.point_outgoing = p_point_start;
	edge.point_end = p_point_end;
	edge.winding = p_winding;
	edge.treenode_edges = tree_create(edges.size());
	edge.treenode_outgoing = tree_create(edges.size());
	edge.points_prev_list = list_create();
	edge.points_next_list = list_create();
	edge.dir_x = points[p_point_end].x - points[p_point_start].x;
	edge.dir_y = points[p_point_end].y - points[p_point_start].y;
	if (edge.dir_y >= 0) {
		edge.min_y = points[p_point_start].y;
		edge.max_y = points[p_point_end].y;
	} else {
		edge.min_y = points[p_point_end].y;
		edge.max_y = points[p_point_start].y;
	}
	DEV_ASSERT(edge.dir_x > 0);
	edge.cross = points[p_point_start].y * edge.dir_x - points[p_point_start].x * edge.dir_y;
	edges.push_back(edge);
	point_add_outgoing(p_point_start, edges.size() - 1);
}

void BentleyOttmann::add_vertical_edge(uint32_t p_slice, int64_t p_y_start, int64_t p_y_end) {
	uint32_t start;
	uint32_t current = tree_nodes[slices[p_slice].vertical_tree].right;
	if (!current) {
		Vertical vertical;
		vertical.y = p_y_start;
		vertical.is_start = true;
		start = tree_create(verticals.size());
		verticals.push_back(vertical);
		tree_insert(start, slices[p_slice].vertical_tree);
	} else {
		while (true) {
			int64_t y = p_y_start - verticals[tree_nodes[current].element].y;
			if (y < 0) {
				if (tree_nodes[current].left) {
					current = tree_nodes[current].left;
					continue;
				}
				if (verticals[tree_nodes[current].element].is_start) {
					Vertical vertical;
					vertical.y = p_y_start;
					vertical.is_start = true;
					start = tree_create(verticals.size());
					verticals.push_back(vertical);
					tree_insert(start, tree_nodes[current].prev);
				} else {
					start = tree_nodes[current].prev;
				}
				break;
			}
			if (y > 0) {
				if (tree_nodes[current].right) {
					current = tree_nodes[current].right;
					continue;
				}
				if (!verticals[tree_nodes[current].element].is_start) {
					Vertical vertical;
					vertical.y = p_y_start;
					vertical.is_start = true;
					start = tree_create(verticals.size());
					verticals.push_back(vertical);
					tree_insert(start, current);
				} else {
					start = current;
				}
				break;
			}
			if (verticals[tree_nodes[current].element].is_start) {
				start = current;
			} else {
				start = tree_nodes[current].prev;
			}
			break;
		}
	}
	while (tree_nodes[start].next != slices[p_slice].vertical_tree) {
		int64_t y = p_y_end - verticals[tree_nodes[tree_nodes[start].next].element].y;
		if (y < 0 || (y == 0 && !verticals[tree_nodes[tree_nodes[start].next].element].is_start)) {
			break;
		}
		tree_remove(tree_nodes[start].next);
	}
	if (tree_nodes[start].next == slices[p_slice].vertical_tree || verticals[tree_nodes[tree_nodes[start].next].element].is_start) {
		Vertical vertical;
		vertical.y = p_y_end;
		vertical.is_start = false;
		tree_insert(tree_create(verticals.size()), start);
		verticals.push_back(vertical);
	}
}

void BentleyOttmann::edge_intersect_x(uint32_t p_edge, int64_t p_x) {
	const Edge &edge = edges[p_edge];
	if (points[edge.point_end].x <= p_x) {
		return;
	}
	int64_t total = p_x * edge.dir_y + edge.cross;
	add_point(p_x * edge.dir_x, total, edge.dir_x);
}

void BentleyOttmann::edge_intersect_edge(uint32_t p_edge1, uint32_t p_edge2) {
	const Edge &edge1 = edges[p_edge1];
	const Edge &edge2 = edges[p_edge2];
	int64_t total_x = edge2.cross * edge1.dir_x - edge1.cross * edge2.dir_x;
	int64_t total_y = edge2.cross * edge1.dir_y - edge1.cross * edge2.dir_y;
	int64_t factor = edge1.dir_y * edge2.dir_x - edge2.dir_y * edge1.dir_x;
	add_point(total_x, total_y, factor);
	DEV_ASSERT(cmp_point_edge(add_point(total_x, total_y, factor), p_edge1) == 0 && cmp_point_edge(add_point(total_x, total_y, factor), p_edge2) == 0);
}

void BentleyOttmann::edge_set_outgoing(uint32_t p_edge, uint32_t p_point) {
	if ((edges[p_edge].winding_total - edges[p_edge].winding) & winding_mask) {
		uint32_t next = list_nodes[edges[p_edge].points_prev_list].next;
		if (next != edges[p_edge].points_prev_list) {
			uint32_t next_next = list_nodes[next].next;
			while (next_next != edges[p_edge].points_prev_list) {
				triangles.push_back(p_point);
				triangles.push_back(list_nodes[next_next].element);
				triangles.push_back(list_nodes[next].element);
				list_remove(next);
				next = next_next;
				next_next = list_nodes[next].next;
			}
			triangles.push_back(p_point);
			triangles.push_back(edges[p_edge].point_outgoing);
			triangles.push_back(list_nodes[next].element);
			list_remove(next);
		}
	}
	if (edges[p_edge].winding_total & winding_mask) {
		uint32_t next = list_nodes[edges[p_edge].points_next_list].next;
		if (next != edges[p_edge].points_next_list) {
			uint32_t next_next = list_nodes[next].next;
			while (next_next != edges[p_edge].points_next_list) {
				triangles.push_back(p_point);
				triangles.push_back(list_nodes[next].element);
				triangles.push_back(list_nodes[next_next].element);
				list_remove(next);
				next = next_next;
				next_next = list_nodes[next].next;
			}
			triangles.push_back(p_point);
			triangles.push_back(list_nodes[next].element);
			triangles.push_back(edges[p_edge].point_outgoing);
			list_remove(next);
		}
	}
	edges[p_edge].point_outgoing = p_point;
}

void BentleyOttmann::edge_add_point_before(uint32_t p_edge, uint32_t p_point) {
	if (!((edges[p_edge].winding_total - edges[p_edge].winding) & winding_mask)) {
		return;
	}
	uint32_t next = list_nodes[edges[p_edge].points_prev_list].next;
	if (next != edges[p_edge].points_prev_list) {
		uint32_t next_next = list_nodes[next].next;
		while (next_next != edges[p_edge].points_prev_list) {
			if (cmp_cross(list_nodes[next].element, list_nodes[next_next].element, p_point) <= 0) {
				goto done;
			}
			triangles.push_back(p_point);
			triangles.push_back(list_nodes[next_next].element);
			triangles.push_back(list_nodes[next].element);
			list_remove(next);
			next = next_next;
			next_next = list_nodes[next].next;
		}
		if (cmp_cross(list_nodes[next].element, edges[p_edge].point_outgoing, p_point) > 0) {
			triangles.push_back(p_point);
			triangles.push_back(edges[p_edge].point_outgoing);
			triangles.push_back(list_nodes[next].element);
			list_remove(next);
		}
	}
done:
	list_insert(points[p_point].listnode_edge_prev, edges[p_edge].points_prev_list);
}

void BentleyOttmann::edge_add_point_after(uint32_t p_edge, uint32_t p_point) {
	if (!(edges[p_edge].winding_total & winding_mask)) {
		return;
	}
	uint32_t next = list_nodes[edges[p_edge].points_next_list].next;
	if (next != edges[p_edge].points_next_list) {
		uint32_t next_next = list_nodes[next].next;
		while (next_next != edges[p_edge].points_next_list) {
			if (cmp_cross(list_nodes[next].element, list_nodes[next_next].element, p_point) >= 0) {
				goto done;
			}
			triangles.push_back(p_point);
			triangles.push_back(list_nodes[next].element);
			triangles.push_back(list_nodes[next_next].element);
			list_remove(next);
			next = next_next;
			next_next = list_nodes[next].next;
		}
		if (cmp_cross(list_nodes[next].element, edges[p_edge].point_outgoing, p_point) < 0) {
			triangles.push_back(p_point);
			triangles.push_back(list_nodes[next].element);
			triangles.push_back(edges[p_edge].point_outgoing);
			list_remove(next);
		}
	}
done:
	list_insert(points[p_point].listnode_edge_next, edges[p_edge].points_next_list);
}

uint32_t BentleyOttmann::get_edge_before(int64_t p_x, int64_t p_y) {
	uint32_t current = tree_nodes[edges_tree].right;
	if (!current) {
		return edges_tree;
	}
	while (true) {
		const Edge &edge = edges[tree_nodes[current].element];
		int64_t cross = p_y * edge.dir_x - p_x * edge.dir_y - edge.cross;
		if (cross > 0) {
			if (tree_nodes[current].right) {
				current = tree_nodes[current].right;
				continue;
			}
			return current;
		}
		if (tree_nodes[current].left) {
			current = tree_nodes[current].left;
			continue;
		}
		return tree_nodes[current].prev;
	}
}

uint32_t BentleyOttmann::get_edge_before_point(uint32_t p_point) {
	uint32_t current = tree_nodes[edges_tree].right;
	if (!current) {
		return edges_tree;
	}
	while (true) {
		int cmp = cmp_point_edge(p_point, tree_nodes[current].element);
		if (cmp > 0) {
			if (tree_nodes[current].right) {
				current = tree_nodes[current].right;
				continue;
			}
			return current;
		}
		if (tree_nodes[current].left) {
			current = tree_nodes[current].left;
			continue;
		}
		return tree_nodes[current].prev;
	}
}

void BentleyOttmann::check_intersection(uint32_t p_treenode_edge) {
	DEV_ASSERT(p_treenode_edge != edges_tree && tree_nodes[p_treenode_edge].next != edges_tree);
	Edge &edge1 = edges[tree_nodes[p_treenode_edge].element];
	Edge &edge2 = edges[tree_nodes[tree_nodes[p_treenode_edge].next].element];
	if (edge1.max_y < edge2.min_y || edge1.point_start == edge2.point_start || edge1.point_end == edge2.point_end) {
		return;
	}
	int64_t max;
	if (points[edge1.point_end].x < points[edge2.point_end].x) {
		max = points[edge1.point_end].x;
	} else {
		max = points[edge2.point_end].x;
	}
	if ((max * edge2.dir_y + edge2.cross) * edge1.dir_x >= (max * edge1.dir_y + edge1.cross) * edge2.dir_x) {
		return;
	}
	edge_intersect_edge(tree_nodes[p_treenode_edge].element, tree_nodes[tree_nodes[p_treenode_edge].next].element);
}

int BentleyOttmann::cmp_point_point(int64_t p_x, int64_t p_y, int64_t p_factor, uint32_t p_point) {
	if (p_factor == 1) {
		int64_t cmp = p_x * points[p_point].factor - points[p_point].x;
		if (cmp > 0) {
			return 1;
		}
		if (cmp < 0) {
			return -1;
		}
		cmp = p_y * points[p_point].factor - points[p_point].y;
		if (cmp > 0) {
			return 1;
		}
		if (cmp < 0) {
			return -1;
		}
		return 0;
	}
	if (points[p_point].factor == 1) {
		int64_t cmp = p_x - points[p_point].x * p_factor;
		if (cmp > 0) {
			return 1;
		}
		if (cmp < 0) {
			return -1;
		}
		cmp = p_y - points[p_point].y * p_factor;
		if (cmp > 0) {
			return 1;
		}
		if (cmp < 0) {
			return -1;
		}
		return 0;
	}
	int64_t factor1_lo = p_factor & 0xFFFFFF;
	int64_t factor1_hi = p_factor >> 24;
	int64_t factor2_lo = points[p_point].factor & 0xFFFFFF;
	int64_t factor2_hi = points[p_point].factor >> 24;
	int64_t v1_lo, v1_hi, v2_lo, v2_hi, m1_lo, m1_hi, m2_lo, m2_hi;
	int64_t m1_t1, m1_t2, m1_t3, m2_t1, m2_t2, m2_t3;
	v1_lo = p_x & 0xFFFFFFFF;
	v1_hi = p_x >> 32;
	v2_lo = points[p_point].x & 0xFFFFFFFF;
	v2_hi = points[p_point].x >> 32;
	// m1 = x1 * factor2
	// m2 = x2 * factor1
	m1_t1 = factor2_lo * v1_lo;
	m2_t1 = factor1_lo * v2_lo;
	m1_t2 = factor2_hi * v1_lo + (m1_t1 >> 24);
	m2_t2 = factor1_hi * v2_lo + (m2_t1 >> 24);
	m1_t3 = factor2_lo * v1_hi + (m1_t2 >> 8);
	m2_t3 = factor1_lo * v2_hi + (m2_t2 >> 8);
	m1_hi = factor2_hi * v1_hi + (m1_t3 >> 24);
	m2_hi = factor1_hi * v2_hi + (m2_t3 >> 24);
	m1_lo = (m1_t1 & 0xFFFFFF) | ((m1_t2 & 0xFF) << 24) | ((m1_t3 & 0xFFFFFF) << 32);
	m2_lo = (m2_t1 & 0xFFFFFF) | ((m2_t2 & 0xFF) << 24) | ((m2_t3 & 0xFFFFFF) << 32);
	// sign(m1 - m2)
	if (m1_hi > m2_hi) {
		return 1;
	}
	if (m1_hi < m2_hi) {
		return -1;
	}
	if (m1_lo > m2_lo) {
		return 1;
	}
	if (m1_lo < m2_lo) {
		return -1;
	}
	v1_lo = p_y & 0xFFFFFFFF;
	v1_hi = p_y >> 32;
	v2_lo = points[p_point].y & 0xFFFFFFFF;
	v2_hi = points[p_point].y >> 32;
	// m1 = y1 * factor2
	// m2 = y2 * factor1
	m1_t1 = factor2_lo * v1_lo;
	m2_t1 = factor1_lo * v2_lo;
	m1_t2 = factor2_hi * v1_lo + (m1_t1 >> 24);
	m2_t2 = factor1_hi * v2_lo + (m2_t1 >> 24);
	m1_t3 = factor2_lo * v1_hi + (m1_t2 >> 8);
	m2_t3 = factor1_lo * v2_hi + (m2_t2 >> 8);
	m1_hi = factor2_hi * v1_hi + (m1_t3 >> 24);
	m2_hi = factor1_hi * v2_hi + (m2_t3 >> 24);
	m1_lo = (m1_t1 & 0xFFFFFF) | ((m1_t2 & 0xFF) << 24) | ((m1_t3 & 0xFFFFFF) << 32);
	m2_lo = (m2_t1 & 0xFFFFFF) | ((m2_t2 & 0xFF) << 24) | ((m2_t3 & 0xFFFFFF) << 32);
	// sign(m1 - m2)
	if (m1_hi > m2_hi) {
		return 1;
	}
	if (m1_hi < m2_hi) {
		return -1;
	}
	if (m1_lo > m2_lo) {
		return 1;
	}
	if (m1_lo < m2_lo) {
		return -1;
	}
	return 0;
}

int BentleyOttmann::cmp_point_edge(uint32_t p_point, uint32_t p_edge) {
	if (points[p_point].factor == 1) {
		int64_t cmp = points[p_point].y * edges[p_edge].dir_x - points[p_point].x * edges[p_edge].dir_y - edges[p_edge].cross;
		if (cmp > 0) {
			return 1;
		}
		if (cmp < 0) {
			return -1;
		}
		return 0;
	}
	int64_t m1_lo, m1_hi, m2_lo, m2_hi, m3_lo, m3_hi;
	int64_t v_lo, v_hi;
	int64_t v1_lo, v1_hi, v2_lo, v2_hi, t1, t2, t3;
	v_lo = points[p_point].y & 0xFFFFFFFF;
	v_hi = points[p_point].y >> 32;
	v_lo *= edges[p_edge].dir_x;
	v_hi = v_hi * edges[p_edge].dir_x + (v_lo >> 32);
	m1_lo = (v_lo & 0xFFFFFFFF) | ((v_hi & 0xFFFFFFF) << 32);
	m1_hi = v_hi >> 28;
	v_lo = points[p_point].x & 0xFFFFFFFF;
	v_hi = points[p_point].x >> 32;
	v_lo *= edges[p_edge].dir_y;
	v_hi = v_hi * edges[p_edge].dir_y + (v_lo >> 32);
	m2_lo = (v_lo & 0xFFFFFFFF) | ((v_hi & 0xFFFFFFF) << 32);
	m2_hi = v_hi >> 28;
	v1_lo = points[p_point].factor & 0xFFFFFF;
	v1_hi = points[p_point].factor >> 24;
	v2_lo = edges[p_edge].cross & 0xFFFFFF;
	v2_hi = edges[p_edge].cross >> 24;
	t1 = v1_lo * v2_lo;
	t2 = v1_lo * v2_hi + v1_hi * v2_lo + (t1 >> 24);
	t3 = v1_hi * v2_hi + (t2 >> 24);
	m3_lo = (t1 & 0xFFFFFF) | ((t2 & 0xFFFFFF) << 24) | ((t3 & 0xFFF) << 48);
	m3_hi = t3 >> 12;
	m2_lo += m3_lo;
	m2_hi += m3_hi + (m2_lo >> 60);
	m2_lo &= 0xFFFFFFFFFFFFFFF;
	if (m1_hi > m2_hi) {
		return 1;
	}
	if (m1_hi < m2_hi) {
		return -1;
	}
	if (m1_lo > m2_lo) {
		return 1;
	}
	if (m1_lo < m2_lo) {
		return -1;
	}
	return 0;
}

int BentleyOttmann::cmp_cross(uint32_t p_point1, uint32_t p_point2, uint32_t p_point_rel) {
	if (points[p_point1].factor == 1 && points[p_point2].factor == 1 && points[p_point_rel].factor == 1) {
		int64_t cmp = (points[p_point1].y - points[p_point_rel].y) * (points[p_point2].x - points[p_point_rel].x) - (points[p_point1].x - points[p_point_rel].x) * (points[p_point2].y - points[p_point_rel].y);
		if (cmp > 0) {
			return 1;
		}
		if (cmp < 0) {
			return -1;
		}
		return 0;
	}

	int64_t factor1_lo = points[p_point1].factor & 0xFFFFFF;
	int64_t factor1_hi = points[p_point1].factor >> 24;
	int64_t factor2_lo = points[p_point2].factor & 0xFFFFFF;
	int64_t factor2_hi = points[p_point2].factor >> 24;
	int64_t factor_rel_lo = points[p_point_rel].factor & 0xFFFFFF;
	int64_t factor_rel_hi = points[p_point_rel].factor >> 24;
	int64_t x1_lo = points[p_point1].x & 0xFFFFFFFF;
	int64_t x1_hi = points[p_point1].x >> 32;
	int64_t x2_lo = points[p_point2].x & 0xFFFFFFFF;
	int64_t x2_hi = points[p_point2].x >> 32;
	int64_t x_rel_lo = (-points[p_point_rel].x) & 0xFFFFFFFF;
	int64_t x_rel_hi = (-points[p_point_rel].x) >> 32;
	int64_t y1_lo = points[p_point1].y & 0xFFFFFFFF;
	int64_t y1_hi = points[p_point1].y >> 32;
	int64_t y2_lo = points[p_point2].y & 0xFFFFFFFF;
	int64_t y2_hi = points[p_point2].y >> 32;
	int64_t y_rel_lo = (-points[p_point_rel].y) & 0xFFFFFFFF;
	int64_t y_rel_hi = (-points[p_point_rel].y) >> 32;

	// m1 = point1 * factor_rel - point_rel * factor1
	// m2 = point2 * factor_rel - point_rel * factor2
	int64_t mx1_t0 = x1_lo * factor_rel_lo + x_rel_lo * factor1_lo;
	int64_t my1_t0 = y1_lo * factor_rel_lo + y_rel_lo * factor1_lo;
	int64_t mx2_t0 = x2_lo * factor_rel_lo + x_rel_lo * factor2_lo;
	int64_t my2_t0 = y2_lo * factor_rel_lo + y_rel_lo * factor2_lo;
	int64_t mx1_t1 = x1_lo * factor_rel_hi + x_rel_lo * factor1_hi + (mx1_t0 >> 24);
	int64_t my1_t1 = y1_lo * factor_rel_hi + y_rel_lo * factor1_hi + (my1_t0 >> 24);
	int64_t mx2_t1 = x2_lo * factor_rel_hi + x_rel_lo * factor2_hi + (mx2_t0 >> 24);
	int64_t my2_t1 = y2_lo * factor_rel_hi + y_rel_lo * factor2_hi + (my2_t0 >> 24);
	int64_t mx1_t2 = x1_hi * factor_rel_lo + x_rel_hi * factor1_lo + (mx1_t1 >> 8);
	int64_t my1_t2 = y1_hi * factor_rel_lo + y_rel_hi * factor1_lo + (my1_t1 >> 8);
	int64_t mx2_t2 = x2_hi * factor_rel_lo + x_rel_hi * factor2_lo + (mx2_t1 >> 8);
	int64_t my2_t2 = y2_hi * factor_rel_lo + y_rel_hi * factor2_lo + (my2_t1 >> 8);
	int64_t mx1_t3 = x1_hi * factor_rel_hi + x_rel_hi * factor1_hi + (mx1_t2 >> 24);
	int64_t my1_t3 = y1_hi * factor_rel_hi + y_rel_hi * factor1_hi + (my1_t2 >> 24);
	int64_t mx2_t3 = x2_hi * factor_rel_hi + x_rel_hi * factor2_hi + (mx2_t2 >> 24);
	int64_t my2_t3 = y2_hi * factor_rel_hi + y_rel_hi * factor2_hi + (my2_t2 >> 24);
	int64_t mx1_0 = (mx1_t0 & 0xFFFFFF) | ((mx1_t1 & 0x1F) << 24);
	int64_t my1_0 = (my1_t0 & 0xFFFFFF) | ((my1_t1 & 0x1F) << 24);
	int64_t mx2_0 = (mx2_t0 & 0xFFFFFF) | ((mx2_t1 & 0x1F) << 24);
	int64_t my2_0 = (my2_t0 & 0xFFFFFF) | ((my2_t1 & 0x1F) << 24);
	int64_t mx1_1 = ((mx1_t1 >> 5) & 0x7) | ((mx1_t2 & 0xFFFFFF) << 3) | ((mx1_t3 & 0x3) << 27);
	int64_t my1_1 = ((my1_t1 >> 5) & 0x7) | ((my1_t2 & 0xFFFFFF) << 3) | ((my1_t3 & 0x3) << 27);
	int64_t mx2_1 = ((mx2_t1 >> 5) & 0x7) | ((mx2_t2 & 0xFFFFFF) << 3) | ((mx2_t3 & 0x3) << 27);
	int64_t my2_1 = ((my2_t1 >> 5) & 0x7) | ((my2_t2 & 0xFFFFFF) << 3) | ((my2_t3 & 0x3) << 27);
	int64_t mx1_2 = (mx1_t3 >> 2) & 0x1FFFFFFF;
	int64_t my1_2 = (my1_t3 >> 2) & 0x1FFFFFFF;
	int64_t mx2_2 = (mx2_t3 >> 2) & 0x1FFFFFFF;
	int64_t my2_2 = (my2_t3 >> 2) & 0x1FFFFFFF;
	int64_t mx1_3 = mx1_t3 >> 31;
	int64_t my1_3 = my1_t3 >> 31;
	int64_t mx2_3 = mx2_t3 >> 31;
	int64_t my2_3 = my2_t3 >> 31;

	// a = m1.y * m2.x
	// b = m1.x * m2.y
	int64_t a_0 = my1_0 * mx2_0;
	int64_t b_0 = mx1_0 * my2_0;
	int64_t a_1 = my1_1 * mx2_0 + my1_0 * mx2_1 + (a_0 >> 29);
	int64_t b_1 = mx1_1 * my2_0 + mx1_0 * my2_1 + (b_0 >> 29);
	int64_t a_2 = my1_2 * mx2_0 + my1_1 * mx2_1 + my1_0 * mx2_2 + (a_1 >> 29);
	int64_t b_2 = mx1_2 * my2_0 + mx1_1 * my2_1 + mx1_0 * my2_2 + (b_1 >> 29);
	int64_t a_3 = my1_3 * mx2_0 + my1_2 * mx2_1 + my1_1 * mx2_2 + my1_0 * mx2_3 + (a_2 >> 29);
	int64_t b_3 = mx1_3 * my2_0 + mx1_2 * my2_1 + mx1_1 * my2_2 + mx1_0 * my2_3 + (b_2 >> 29);
	int64_t a_4 = my1_3 * mx2_1 + my1_2 * mx2_2 + my1_1 * mx2_3 + (a_3 >> 29);
	int64_t b_4 = mx1_3 * my2_1 + mx1_2 * my2_2 + mx1_1 * my2_3 + (b_3 >> 29);
	int64_t a_5 = my1_3 * mx2_2 + my1_2 * mx2_3 + (a_4 >> 29);
	int64_t b_5 = mx1_3 * my2_2 + mx1_2 * my2_3 + (b_4 >> 29);
	int64_t a_6 = my1_3 * mx2_3 + (a_5 >> 29);
	int64_t b_6 = mx1_3 * my2_3 + (b_5 >> 29);

	a_0 &= 0x1FFFFFFF;
	b_0 &= 0x1FFFFFFF;
	a_1 &= 0x1FFFFFFF;
	b_1 &= 0x1FFFFFFF;
	a_2 &= 0x1FFFFFFF;
	b_2 &= 0x1FFFFFFF;
	a_3 &= 0x1FFFFFFF;
	b_3 &= 0x1FFFFFFF;
	a_4 &= 0x1FFFFFFF;
	b_4 &= 0x1FFFFFFF;
	a_5 &= 0x1FFFFFFF;
	b_5 &= 0x1FFFFFFF;

	// sign(a - b)
	if (a_6 > b_6) {
		return 1;
	}
	if (a_6 < b_6) {
		return -1;
	}
	if (a_5 > b_5) {
		return 1;
	}
	if (a_5 < b_5) {
		return -1;
	}
	if (a_4 > b_4) {
		return 1;
	}
	if (a_4 < b_4) {
		return -1;
	}
	if (a_3 > b_3) {
		return 1;
	}
	if (a_3 < b_3) {
		return -1;
	}
	if (a_2 > b_2) {
		return 1;
	}
	if (a_2 < b_2) {
		return -1;
	}
	if (a_1 > b_1) {
		return 1;
	}
	if (a_1 < b_1) {
		return -1;
	}
	if (a_0 > b_0) {
		return 1;
	}
	if (a_0 < b_0) {
		return -1;
	}
	return 0;
}

uint32_t BentleyOttmann::tree_create(uint32_t p_element) {
	TreeNode node;
	node.prev = node.next = tree_nodes.size();
	node.element = p_element;
	tree_nodes.push_back(node);
	return node.next;
}

void BentleyOttmann::tree_clear(uint32_t p_tree) {
	uint32_t iter = tree_nodes[p_tree].next;
	while (iter != p_tree) {
		uint32_t next = tree_nodes[iter].next;
		tree_nodes[iter].left = tree_nodes[iter].right = tree_nodes[iter].parent = 0;
		tree_nodes[iter].prev = tree_nodes[iter].next = iter;
		tree_nodes[iter].is_heavy = false;
		iter = next;
	}
	tree_nodes[p_tree].left = tree_nodes[p_tree].right = tree_nodes[p_tree].parent = 0;
	tree_nodes[p_tree].prev = tree_nodes[p_tree].next = iter;
	tree_nodes[p_tree].is_heavy = false;
}

void BentleyOttmann::tree_insert(uint32_t p_insert_item, uint32_t p_insert_after) {
	DEV_ASSERT(p_insert_item != 0 && p_insert_after != 0);
	if (tree_nodes[p_insert_after].right == 0) {
		tree_nodes[p_insert_after].right = p_insert_item;
		tree_nodes[p_insert_item].parent = p_insert_after;
	} else {
		DEV_ASSERT(tree_nodes[tree_nodes[p_insert_after].next].left == 0);
		tree_nodes[tree_nodes[p_insert_after].next].left = p_insert_item;
		tree_nodes[p_insert_item].parent = tree_nodes[p_insert_after].next;
	}
	tree_nodes[p_insert_item].prev = p_insert_after;
	tree_nodes[p_insert_item].next = tree_nodes[p_insert_after].next;
	tree_nodes[tree_nodes[p_insert_after].next].prev = p_insert_item;
	tree_nodes[p_insert_after].next = p_insert_item;
	uint32_t item = p_insert_item;
	uint32_t parent = tree_nodes[item].parent;
	while (tree_nodes[parent].parent) {
		uint32_t sibling = tree_nodes[parent].left;
		if (sibling == item) {
			sibling = tree_nodes[parent].right;
		}
		if (tree_nodes[sibling].is_heavy) {
			tree_nodes[sibling].is_heavy = false;
			return;
		}
		if (!tree_nodes[item].is_heavy) {
			tree_nodes[item].is_heavy = true;
			item = parent;
			parent = tree_nodes[item].parent;
			continue;
		}
		uint32_t move;
		uint32_t unmove;
		uint32_t move_move;
		uint32_t move_unmove;
		if (item == tree_nodes[parent].left) {
			move = tree_nodes[item].right;
			unmove = tree_nodes[item].left;
			move_move = tree_nodes[move].left;
			move_unmove = tree_nodes[move].right;
		} else {
			move = tree_nodes[item].left;
			unmove = tree_nodes[item].right;
			move_move = tree_nodes[move].right;
			move_unmove = tree_nodes[move].left;
		}
		if (!tree_nodes[move].is_heavy) {
			tree_rotate(item);
			tree_nodes[item].is_heavy = tree_nodes[parent].is_heavy;
			tree_nodes[parent].is_heavy = !tree_nodes[unmove].is_heavy;
			if (tree_nodes[unmove].is_heavy) {
				tree_nodes[unmove].is_heavy = false;
				return;
			}
			DEV_ASSERT(move != 0);
			tree_nodes[move].is_heavy = true;
			parent = tree_nodes[item].parent;
			continue;
		}
		tree_rotate(move);
		tree_rotate(move);
		tree_nodes[move].is_heavy = tree_nodes[parent].is_heavy;
		if (unmove != 0) {
			tree_nodes[unmove].is_heavy = tree_nodes[move_unmove].is_heavy;
		}
		if (sibling != 0) {
			tree_nodes[sibling].is_heavy = tree_nodes[move_move].is_heavy;
		}
		tree_nodes[item].is_heavy = false;
		tree_nodes[parent].is_heavy = false;
		tree_nodes[move_move].is_heavy = false;
		if (move_unmove != 0) {
			tree_nodes[move_unmove].is_heavy = false;
		}
		return;
	}
}

void BentleyOttmann::tree_remove(uint32_t p_remove_item) {
	DEV_ASSERT(tree_nodes[p_remove_item].parent != 0);
	if (tree_nodes[p_remove_item].left != 0 && tree_nodes[p_remove_item].right != 0) {
		uint32_t prev = tree_nodes[p_remove_item].prev;
		DEV_ASSERT(tree_nodes[prev].parent != 0 && tree_nodes[prev].right == 0);
		tree_swap(p_remove_item, prev);
	}
	DEV_ASSERT(tree_nodes[p_remove_item].left == 0 || tree_nodes[p_remove_item].right == 0);
	uint32_t prev = tree_nodes[p_remove_item].prev;
	uint32_t next = tree_nodes[p_remove_item].next;
	tree_nodes[prev].next = next;
	tree_nodes[next].prev = prev;
	uint32_t parent = tree_nodes[p_remove_item].parent;
	uint32_t replacement = tree_nodes[p_remove_item].left;
	if (replacement == 0) {
		replacement = tree_nodes[p_remove_item].right;
	}
	if (replacement != 0) {
		tree_nodes[replacement].parent = parent;
		tree_nodes[replacement].is_heavy = tree_nodes[p_remove_item].is_heavy;
	}
	if (tree_nodes[parent].left == p_remove_item) {
		tree_nodes[parent].left = replacement;
	} else {
		tree_nodes[parent].right = replacement;
	}
	tree_nodes[p_remove_item].left = tree_nodes[p_remove_item].right = tree_nodes[p_remove_item].parent = 0;
	tree_nodes[p_remove_item].prev = tree_nodes[p_remove_item].next = p_remove_item;
	tree_nodes[p_remove_item].is_heavy = false;
	uint32_t item = replacement;
	if (tree_nodes[parent].left == 0 && tree_nodes[parent].right == 0) {
		item = parent;
		parent = tree_nodes[item].parent;
	}
	while (tree_nodes[parent].parent != 0) {
		uint32_t sibling = tree_nodes[parent].left;
		if (sibling == item) {
			sibling = tree_nodes[parent].right;
		}
		DEV_ASSERT(sibling != 0);
		if (tree_nodes[item].is_heavy) {
			tree_nodes[item].is_heavy = false;
			item = parent;
			parent = tree_nodes[item].parent;
			continue;
		}
		if (!tree_nodes[sibling].is_heavy) {
			tree_nodes[sibling].is_heavy = true;
			return;
		}
		uint32_t move;
		uint32_t unmove;
		uint32_t move_move;
		uint32_t move_unmove;
		if (sibling == tree_nodes[parent].left) {
			move = tree_nodes[sibling].right;
			unmove = tree_nodes[sibling].left;
			move_move = tree_nodes[move].left;
			move_unmove = tree_nodes[move].right;
		} else {
			move = tree_nodes[sibling].left;
			unmove = tree_nodes[sibling].right;
			move_move = tree_nodes[move].right;
			move_unmove = tree_nodes[move].left;
		}
		if (!tree_nodes[move].is_heavy) {
			tree_rotate(sibling);
			tree_nodes[sibling].is_heavy = tree_nodes[parent].is_heavy;
			tree_nodes[parent].is_heavy = !tree_nodes[unmove].is_heavy;
			if (tree_nodes[unmove].is_heavy) {
				tree_nodes[unmove].is_heavy = false;
				item = sibling;
				parent = tree_nodes[item].parent;
				continue;
			}
			DEV_ASSERT(move != 0);
			tree_nodes[move].is_heavy = true;
			return;
		}
		tree_rotate(move);
		tree_rotate(move);
		tree_nodes[move].is_heavy = tree_nodes[parent].is_heavy;
		if (unmove != 0) {
			tree_nodes[unmove].is_heavy = tree_nodes[move_unmove].is_heavy;
		}
		if (item != 0) {
			tree_nodes[item].is_heavy = tree_nodes[move_move].is_heavy;
		}
		tree_nodes[sibling].is_heavy = false;
		tree_nodes[parent].is_heavy = false;
		tree_nodes[move_move].is_heavy = false;
		if (move_unmove != 0) {
			tree_nodes[move_unmove].is_heavy = false;
		}
		item = move;
		parent = tree_nodes[item].parent;
		continue;
	}
}

void BentleyOttmann::tree_rotate(uint32_t p_item) {
	DEV_ASSERT(tree_nodes[tree_nodes[p_item].parent].parent != 0);
	uint32_t parent = tree_nodes[p_item].parent;
	if (tree_nodes[parent].left == p_item) {
		uint32_t move = tree_nodes[p_item].right;
		tree_nodes[parent].left = move;
		tree_nodes[p_item].right = parent;
		if (move) {
			tree_nodes[move].parent = parent;
		}
	} else {
		uint32_t move = tree_nodes[p_item].left;
		tree_nodes[parent].right = move;
		tree_nodes[p_item].left = parent;
		if (move) {
			tree_nodes[move].parent = parent;
		}
	}
	uint32_t grandparent = tree_nodes[parent].parent;
	tree_nodes[p_item].parent = grandparent;
	if (tree_nodes[grandparent].left == parent) {
		tree_nodes[grandparent].left = p_item;
	} else {
		tree_nodes[grandparent].right = p_item;
	}
	tree_nodes[parent].parent = p_item;
}

void BentleyOttmann::tree_swap(uint32_t p_item1, uint32_t p_item2) {
	DEV_ASSERT(tree_nodes[p_item1].parent != 0 && tree_nodes[p_item2].parent != 0);
	uint32_t parent1 = tree_nodes[p_item1].parent;
	uint32_t left1 = tree_nodes[p_item1].left;
	uint32_t right1 = tree_nodes[p_item1].right;
	uint32_t prev1 = tree_nodes[p_item1].prev;
	uint32_t next1 = tree_nodes[p_item1].next;
	uint32_t parent2 = tree_nodes[p_item2].parent;
	uint32_t left2 = tree_nodes[p_item2].left;
	uint32_t right2 = tree_nodes[p_item2].right;
	uint32_t prev2 = tree_nodes[p_item2].prev;
	uint32_t next2 = tree_nodes[p_item2].next;
	if (tree_nodes[parent1].left == p_item1) {
		tree_nodes[parent1].left = p_item2;
	} else {
		tree_nodes[parent1].right = p_item2;
	}
	if (tree_nodes[parent2].left == p_item2) {
		tree_nodes[parent2].left = p_item1;
	} else {
		tree_nodes[parent2].right = p_item1;
	}
	if (left1) {
		tree_nodes[left1].parent = p_item2;
	}
	if (right1) {
		tree_nodes[right1].parent = p_item2;
	}
	if (left2) {
		tree_nodes[left2].parent = p_item1;
	}
	if (right2) {
		tree_nodes[right2].parent = p_item1;
	}
	tree_nodes[prev1].next = p_item2;
	tree_nodes[next1].prev = p_item2;
	tree_nodes[prev2].next = p_item1;
	tree_nodes[next2].prev = p_item1;
	parent1 = tree_nodes[p_item1].parent;
	left1 = tree_nodes[p_item1].left;
	right1 = tree_nodes[p_item1].right;
	prev1 = tree_nodes[p_item1].prev;
	next1 = tree_nodes[p_item1].next;
	parent2 = tree_nodes[p_item2].parent;
	left2 = tree_nodes[p_item2].left;
	right2 = tree_nodes[p_item2].right;
	prev2 = tree_nodes[p_item2].prev;
	next2 = tree_nodes[p_item2].next;
	tree_nodes[p_item2].parent = parent1;
	tree_nodes[p_item2].left = left1;
	tree_nodes[p_item2].right = right1;
	tree_nodes[p_item2].prev = prev1;
	tree_nodes[p_item2].next = next1;
	tree_nodes[p_item1].parent = parent2;
	tree_nodes[p_item1].left = left2;
	tree_nodes[p_item1].right = right2;
	tree_nodes[p_item1].prev = prev2;
	tree_nodes[p_item1].next = next2;
	bool is_heavy = tree_nodes[p_item1].is_heavy;
	tree_nodes[p_item1].is_heavy = tree_nodes[p_item2].is_heavy;
	tree_nodes[p_item2].is_heavy = is_heavy;
}

void BentleyOttmann::tree_replace(uint32_t p_item1, uint32_t p_item2) {
	DEV_ASSERT(tree_nodes[p_item1].parent == 0 && tree_nodes[p_item2].parent != 0);
	uint32_t parent = tree_nodes[p_item2].parent;
	uint32_t left = tree_nodes[p_item2].left;
	uint32_t right = tree_nodes[p_item2].right;
	uint32_t prev = tree_nodes[p_item2].prev;
	uint32_t next = tree_nodes[p_item2].next;
	if (tree_nodes[parent].left == p_item2) {
		tree_nodes[parent].left = p_item1;
	} else {
		tree_nodes[parent].right = p_item1;
	}
	if (left) {
		tree_nodes[left].parent = p_item1;
	}
	if (right) {
		tree_nodes[right].parent = p_item1;
	}
	tree_nodes[prev].next = p_item1;
	tree_nodes[next].prev = p_item1;
	tree_nodes[p_item1].parent = parent;
	tree_nodes[p_item1].left = left;
	tree_nodes[p_item1].right = right;
	tree_nodes[p_item1].prev = prev;
	tree_nodes[p_item1].next = next;
	tree_nodes[p_item1].is_heavy = tree_nodes[p_item2].is_heavy;
	tree_nodes[p_item2].left = tree_nodes[p_item2].right = tree_nodes[p_item2].parent = 0;
	tree_nodes[p_item2].prev = tree_nodes[p_item2].next = p_item2;
	tree_nodes[p_item2].is_heavy = false;
}

uint32_t BentleyOttmann::list_create(uint32_t p_element) {
	ListNode node;
	node.anchor = node.prev = node.next = list_nodes.size();
	node.element = p_element;
	list_nodes.push_back(node);
	return node.next;
}

void BentleyOttmann::list_insert(uint32_t p_insert_item, uint32_t p_list) {
	DEV_ASSERT(p_insert_item != p_list);
	DEV_ASSERT(list_nodes[p_list].anchor == p_list);
	if (list_nodes[p_insert_item].anchor == p_list) {
		return;
	}
	if (list_nodes[p_insert_item].anchor != p_insert_item) {
		list_remove(p_insert_item);
	}
	list_nodes[p_insert_item].anchor = p_list;
	list_nodes[p_insert_item].prev = p_list;
	list_nodes[p_insert_item].next = list_nodes[p_list].next;
	list_nodes[list_nodes[p_list].next].prev = p_insert_item;
	list_nodes[p_list].next = p_insert_item;
}

void BentleyOttmann::list_remove(uint32_t p_remove_item) {
	list_nodes[list_nodes[p_remove_item].next].prev = list_nodes[p_remove_item].prev;
	list_nodes[list_nodes[p_remove_item].prev].next = list_nodes[p_remove_item].next;
	list_nodes[p_remove_item].anchor = list_nodes[p_remove_item].prev = list_nodes[p_remove_item].next = p_remove_item;
}
