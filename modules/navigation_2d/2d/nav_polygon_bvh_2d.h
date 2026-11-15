/**************************************************************************/
/*  nav_polygon_bvh_2d.h                                                  */
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

#include "../nav_utils_2d.h"
#include "../triangle2.h"

#include "core/math/rect2.h"
#include "core/templates/local_vector.h"
#include "core/templates/sort_array.h"

class NavPolygonBVH2D {
	// Use a leaf size of 8 to balance tree construction and polygon scanning costs.
	static const uint32_t LEAF_SIZE = 8;

	struct Node {
		Rect2 bounds;
		// Store the right child index if `count` is zero, otherwise an offset into `polygon_indices`.
		uint32_t right_or_first = 0;
		uint32_t count = 0;
	};

	struct CenterLess {
		const LocalVector<Rect2> *bounds = nullptr;
		int compare_axis = 0;
		bool operator()(const uint32_t &p_a, const uint32_t &p_b) const {
			const real_t a_center = (*bounds)[p_a].get_center()[compare_axis];
			const real_t b_center = (*bounds)[p_b].get_center()[compare_axis];
			return a_center == b_center ? p_a < p_b : a_center < b_center;
		}
	};

	LocalVector<Node> nodes;
	LocalVector<uint32_t> polygon_indices;

	uint32_t _build(uint32_t p_first, uint32_t p_count, const LocalVector<Rect2> &p_bounds) {
		const uint32_t index = nodes.size();
		nodes.push_back(Node());
		Rect2 combined_bounds = p_bounds[polygon_indices[p_first]];
		for (uint32_t i = 1; i < p_count; i++) {
			combined_bounds = combined_bounds.merge(p_bounds[polygon_indices[p_first + i]]);
		}
		nodes[index].bounds = combined_bounds;

		if (p_count <= LEAF_SIZE) {
			nodes[index].right_or_first = p_first;
			nodes[index].count = p_count;
			return index;
		}

		SortArray<uint32_t, CenterLess> sort_array;
		CenterLess center_less;
		center_less.bounds = &p_bounds;
		center_less.compare_axis = combined_bounds.size.max_axis_index();
		sort_array.compare = center_less;
		const uint32_t middle = p_first + p_count / 2;
		// Partition around the median without fully sorting to balance the polygon counts.
		sort_array.nth_element(p_first, p_first + p_count, middle, polygon_indices.ptr());
		// Build the left subtree at `index + 1` and store the right child index explicitly.
		_build(p_first, p_count / 2, p_bounds);
		nodes[index].right_or_first = _build(middle, p_first + p_count - middle, p_bounds);
		return index;
	}

	static real_t _distance_squared(const Vector2 &p_point, const Rect2 &p_bounds) {
		const Vector2 end = p_bounds.get_end();
		const Vector2 closest = p_point.clamp(p_bounds.position, end);
		return p_point.distance_squared_to(closest);
	}

	static void _find_closest_polygon(const Nav2D::Polygon &p_polygon, uint32_t p_polygon_id, const Vector2 &p_point, uint32_t &r_polygon_id, real_t &r_distance, const Nav2D::Polygon *&r_polygon, Vector2 &r_position) {
		for (const Vector2 &vertex : p_polygon.vertices) {
			if (!vertex.is_finite()) {
				return;
			}
		}
		if (p_polygon.vertices.size() >= 3) {
			const Vector2 v0 = p_polygon.vertices[0];
			for (uint32_t i = 1; i + 1 < p_polygon.vertices.size(); i++) {
				const Vector2 v1 = p_polygon.vertices[i];
				const Vector2 v2 = p_polygon.vertices[i + 1];
				const Triangle2 face(v0, v1, v2);
				const Vector2 closest = face.get_closest_point_to(p_point);
				// Compare distances as in the original scan to preserve ties after rounding.
				const real_t distance = closest.distance_to(p_point);
				const bool is_closer = distance < r_distance;
				// Within a region, break ties by the original polygon index to match the linear scan.
				const bool wins_tie = distance == r_distance && r_polygon_id != UINT32_MAX && p_polygon_id < r_polygon_id;
				if (is_closer || wins_tie) {
					r_polygon_id = p_polygon_id;
					r_distance = distance;
					r_position = closest;
					r_polygon = &p_polygon;
				}
			}
		}
	}

public:
	bool is_empty() const { return nodes.is_empty(); }

	uint64_t get_size_bytes() const {
		const uint64_t node_bytes = uint64_t(nodes.get_capacity()) * sizeof(Node);
		const uint64_t index_bytes = uint64_t(polygon_indices.get_capacity()) * sizeof(uint32_t);
		return node_bytes + index_bytes;
	}

	void find_closest_point(const LocalVector<Nav2D::Polygon> &p_polygons, const Vector2 &p_point, real_t &r_distance, const Nav2D::Polygon *&r_polygon, Vector2 &r_position) const {
		find_closest_polygon(p_point, p_polygons, r_distance, r_polygon, r_position);
	}

	void find_closest_polygon(const Vector2 &p_point, const LocalVector<Nav2D::Polygon> &p_polygons, real_t &r_distance, const Nav2D::Polygon *&r_polygon, Vector2 &r_position) const {
		uint32_t closest_polygon_index = UINT32_MAX;
		if (nodes.is_empty()) {
			for (uint32_t i = 0; i < p_polygons.size(); i++) {
				_find_closest_polygon(p_polygons[i], i, p_point, closest_polygon_index, r_distance, r_polygon, r_position);
			}
			return;
		}
		// Median splits limit the depth to 32 levels; 64 entries cover pending branches.
		const uint32_t STACK_SIZE = 64;
		uint32_t search_stack[STACK_SIZE];
		uint32_t stack_top = 0;
		search_stack[stack_top++] = 0;
		while (stack_top > 0) {
			const uint32_t search_index = search_stack[--stack_top];

			const Node &node = nodes[search_index];
			// Allow rounding slack when pruning to retain potentially equal-distance candidates.
			const real_t prune_distance = r_distance + CMP_EPSILON;
			if (Math::sqrt(_distance_squared(p_point, node.bounds)) > prune_distance) {
				continue;
			}
			if (node.count != 0) {
				for (uint32_t i = 0; i < node.count; i++) {
					const uint32_t indices_index = node.right_or_first + i;
					const uint32_t polygon_index = polygon_indices[indices_index];
					_find_closest_polygon(p_polygons[polygon_index], polygon_index, p_point, closest_polygon_index, r_distance, r_polygon, r_position);
				}
			} else {
				const uint32_t left_index = search_index + 1;
				const uint32_t right_index = node.right_or_first;
				const Node &left_node = nodes[left_index];
				const Node &right_node = nodes[right_index];
				const real_t left_distance = Math::sqrt(_distance_squared(p_point, left_node.bounds));
				const real_t right_distance = Math::sqrt(_distance_squared(p_point, right_node.bounds));
				DEV_ASSERT(stack_top + 2 <= STACK_SIZE);
				// Push the farther child first to visit the nearer child and tighten the bound first.
				if (left_distance < right_distance) {
					if (right_distance <= prune_distance) {
						search_stack[stack_top++] = right_index;
					}
					if (left_distance <= prune_distance) {
						search_stack[stack_top++] = left_index;
					}

				} else {
					if (left_distance <= prune_distance) {
						search_stack[stack_top++] = left_index;
					}
					if (right_distance <= prune_distance) {
						search_stack[stack_top++] = right_index;
					}
				}
			}
		}
	}

	void build(const LocalVector<Nav2D::Polygon> &p_polygons) {
		nodes.reset();
		polygon_indices.reset();

		const uint32_t count = p_polygons.size();
		// Scan small regions directly to avoid tree construction overhead.
		if (count <= LEAF_SIZE) {
			return;
		}
		LocalVector<Rect2> bounds;
		const uint32_t polygons_count = p_polygons.size();
		bounds.resize(polygons_count);
		polygon_indices.reserve(polygons_count);
		for (uint32_t i = 0; i < polygons_count; i++) {
			const Nav2D::Polygon &polygon = p_polygons[i];
			if (polygon.vertices.size() < 3) {
				continue;
			}
			Rect2 box(polygon.vertices[0], Vector2());

			bool valid_vertices = polygon.vertices[0].is_finite();
			for (uint32_t j = 1; j < polygon.vertices.size(); j++) {
				if (!polygon.vertices[j].is_finite()) {
					valid_vertices = false;
					break;
				}
				box.expand_to(polygon.vertices[j]);
			}
			if (!valid_vertices) {
				continue;
			}
			// Pad bounds by coordinate scale to allow for rounding in `Triangle2` closest-point results.
			const Vector2 abs_position = box.position.abs();
			const Vector2 abs_end = box.get_end().abs();
			const real_t max_position = abs_position[abs_position.max_axis_index()];
			const real_t max_end = abs_end[abs_end.max_axis_index()];
			const real_t max_coordinate = MAX(max_position, max_end);
			const real_t scale = MAX(real_t(1.0), max_coordinate);
			box.grow_by(CMP_EPSILON * scale);
			bounds[i] = box;
			// Keep original polygon indices for lookup and index comparisons on equal distances.
			polygon_indices.push_back(i);
		}
		if (!polygon_indices.is_empty()) {
			_build(0, polygon_indices.size(), bounds);
		}
	}
};
