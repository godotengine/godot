/**************************************************************************/
/*  bit_map.cpp                                                           */
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

#include "bit_map.h"
#include "bit_map.compat.inc"

#include "core/variant/typed_array.h"

void BitMap::create(const Size2i &p_size) {
	ERR_FAIL_COND(p_size.width < 1);
	ERR_FAIL_COND(p_size.height < 1);

	ERR_FAIL_COND(static_cast<int64_t>(p_size.width) * static_cast<int64_t>(p_size.height) > INT32_MAX);

	Error err = bitmask.resize(Math::division_round_up(p_size.width * p_size.height, 8));
	ERR_FAIL_COND(err != OK);

	width = p_size.width;
	height = p_size.height;

	memset(bitmask.ptrw(), 0, bitmask.size());
}

void BitMap::create_from_image_alpha(const Ref<Image> &p_image, float p_threshold) {
	ERR_FAIL_COND(p_image.is_null() || p_image->is_empty());
	Ref<Image> img = p_image->duplicate();
	img->convert(Image::FORMAT_LA8);
	ERR_FAIL_COND(img->get_format() != Image::FORMAT_LA8);

	create(Size2i(img->get_width(), img->get_height()));

	const uint8_t *r = img->get_data().ptr();
	uint8_t *w = bitmask.ptrw();

	for (int i = 0; i < width * height; i++) {
		int bbyte = i / 8;
		int bbit = i % 8;
		if (r[i * 2 + 1] / 255.0 > p_threshold) {
			w[bbyte] |= (1 << bbit);
		}
	}
}

void BitMap::set_bit_rect(const Rect2i &p_rect, bool p_value) {
	Rect2i current = Rect2i(0, 0, width, height).intersection(p_rect);
	uint8_t *data = bitmask.ptrw();

	for (int i = current.position.x; i < current.position.x + current.size.x; i++) {
		for (int j = current.position.y; j < current.position.y + current.size.y; j++) {
			int ofs = width * j + i;
			int bbyte = ofs / 8;
			int bbit = ofs % 8;

			uint8_t b = data[bbyte];

			if (p_value) {
				b |= (1 << bbit);
			} else {
				b &= ~(1 << bbit);
			}

			data[bbyte] = b;
		}
	}
}

int BitMap::get_true_bit_count() const {
	int ds = bitmask.size();
	const uint8_t *d = bitmask.ptr();
	int c = 0;

	// Fast, almost branchless version.

	for (int i = 0; i < ds; i++) {
		c += (d[i] & (1 << 7)) >> 7;
		c += (d[i] & (1 << 6)) >> 6;
		c += (d[i] & (1 << 5)) >> 5;
		c += (d[i] & (1 << 4)) >> 4;
		c += (d[i] & (1 << 3)) >> 3;
		c += (d[i] & (1 << 2)) >> 2;
		c += (d[i] & (1 << 1)) >> 1;
		c += d[i] & 1;
	}

	return c;
}

void BitMap::set_bitv(const Point2i &p_pos, bool p_value) {
	set_bit(p_pos.x, p_pos.y, p_value);
}

void BitMap::set_bit(int p_x, int p_y, bool p_value) {
	ERR_FAIL_INDEX(p_x, width);
	ERR_FAIL_INDEX(p_y, height);

	int ofs = width * p_y + p_x;
	int bbyte = ofs / 8;
	int bbit = ofs % 8;

	uint8_t b = bitmask[bbyte];

	if (p_value) {
		b |= (1 << bbit);
	} else {
		b &= ~(1 << bbit);
	}

	bitmask.write[bbyte] = b;
}

bool BitMap::get_bitv(const Point2i &p_pos) const {
	return get_bit(p_pos.x, p_pos.y);
}

bool BitMap::get_bit(int p_x, int p_y) const {
	ERR_FAIL_INDEX_V(p_x, width, false);
	ERR_FAIL_INDEX_V(p_y, height, false);

	int ofs = width * p_y + p_x;
	int bbyte = ofs / 8;
	int bbit = ofs % 8;

	return (bitmask[bbyte] & (1 << bbit)) != 0;
}

Size2i BitMap::get_size() const {
	return Size2i(width, height);
}

void BitMap::_set_data(const Dictionary &p_d) {
	ERR_FAIL_COND(!p_d.has("size"));
	ERR_FAIL_COND(!p_d.has("data"));

	create(p_d["size"]);
	bitmask = p_d["data"];
}

Dictionary BitMap::_get_data() const {
	Dictionary d;
	d["size"] = get_size();
	d["data"] = bitmask;
	return d;
}

Vector<Vector<Vector2>> BitMap::_march_square(const Rect2i &p_rect, const Point2i &p_start) const {
	int stepx = 0;
	int stepy = 0;
	int prevx = 0;
	int prevy = 0;
	int startx = p_start.x;
	int starty = p_start.y;
	int curx = startx;
	int cury = starty;
	unsigned int count = 0;

	HashMap<Point2i, int> cross_map;

	Vector<Vector2> _points;
	int points_size = 0;

	Vector<Vector<Vector2>> ret;

	// Add starting entry at start of return.
	ret.resize(1);

	do {
		int sv = 0;
		{ // Square value

			/*
			checking the 2x2 pixel grid, assigning these values to each pixel, if not transparent
			+---+---+
			| 1 | 2 |
			+---+---+
			| 4 | 8 | <- current pixel (curx,cury)
			+---+---+
			*/
			Point2i tl = Point2i(curx - 1, cury - 1);
			sv += (p_rect.has_point(tl) && get_bitv(tl)) ? 1 : 0;
			Point2i tr = Point2i(curx, cury - 1);
			sv += (p_rect.has_point(tr) && get_bitv(tr)) ? 2 : 0;
			Point2i bl = Point2i(curx - 1, cury);
			sv += (p_rect.has_point(bl) && get_bitv(bl)) ? 4 : 0;
			Point2i br = Point2i(curx, cury);
			sv += (p_rect.has_point(br) && get_bitv(br)) ? 8 : 0;
			ERR_FAIL_COND_V(sv == 0 || sv == 15, Vector<Vector<Vector2>>());
		}

		switch (sv) {
			case 1:
			case 5:
			case 13:
				/* going UP with these cases:
				1          5           13
				+---+---+  +---+---+  +---+---+
				| 1 |   |  | 1 |   |  | 1 |   |
				+---+---+  +---+---+  +---+---+
				|   |   |  | 4 |   |  | 4 | 8 |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = 0;
				stepy = -1;
				break;

			case 8:
			case 10:
			case 11:
				/* going DOWN with these cases:
				8          10         11
				+---+---+  +---+---+  +---+---+
				|   |   |  |   | 2 |  | 1 | 2 |
				+---+---+  +---+---+  +---+---+
				|   | 8 |  |   | 8 |  |   | 8 |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = 0;
				stepy = 1;
				break;

			case 4:
			case 12:
			case 14:
				/* going LEFT with these cases:
				4          12         14
				+---+---+  +---+---+  +---+---+
				|   |   |  |   |   |  |   | 2 |
				+---+---+  +---+---+  +---+---+
				| 4 |   |  | 4 | 8 |  | 4 | 8 |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = -1;
				stepy = 0;
				break;

			case 2:
			case 3:
			case 7:
				/* going RIGHT with these cases:
				2          3          7
				+---+---+  +---+---+  +---+---+
				|   | 2 |  | 1 | 2 |  | 1 | 2 |
				+---+---+  +---+---+  +---+---+
				|   |   |  |   |   |  | 4 |   |
				+---+---+  +---+---+  +---+---+
				*/
				stepx = 1;
				stepy = 0;
				break;
			case 9:
				/* Going DOWN if coming from the LEFT, otherwise go UP.
				9
				+---+---+
				| 1 |   |
				+---+---+
				|   | 8 |
				+---+---+
				*/

				if (prevx == 1) {
					stepx = 0;
					stepy = 1;
				} else {
					stepx = 0;
					stepy = -1;
				}
				break;
			case 6:
				/* Going RIGHT if coming from BELOW, otherwise go LEFT.
				6
				+---+---+
				|   | 2 |
				+---+---+
				| 4 |   |
				+---+---+
				*/

				if (prevy == -1) {
					stepx = 1;
					stepy = 0;
				} else {
					stepx = -1;
					stepy = 0;
				}
				break;
			default:
				ERR_PRINT("this shouldn't happen.");
		}

		// Handle crossing points.
		if (sv == 6 || sv == 9) {
			const Point2i cur_pos(curx, cury);

			// Find if this point has occurred before.
			if (HashMap<Point2i, int>::Iterator found = cross_map.find(cur_pos)) {
				// Add points after the previous crossing to the result.
				ret.push_back(_points.slice(found->value + 1, points_size));

				// Remove points after crossing point.
				points_size = found->value + 1;

				// Erase trailing map elements.
				while (cross_map.last() != found) {
					cross_map.remove(cross_map.last());
				}

				cross_map.erase(cur_pos);
			} else {
				// Add crossing point to map.
				cross_map.insert(cur_pos, points_size - 1);
			}
		}

		// Small optimization:
		// If the previous direction is same as the current direction,
		// then we should modify the last vector to current.
		curx += stepx;
		cury += stepy;
		if (stepx == prevx && stepy == prevy) {
			_points.set(points_size - 1, Vector2(curx, cury) - p_rect.position);
		} else {
			_points.resize(MAX(points_size + 1, _points.size()));
			_points.set(points_size, Vector2(curx, cury) - p_rect.position);
			points_size++;
		}

		count++;
		prevx = stepx;
		prevy = stepy;

		ERR_FAIL_COND_V((int)count > 2 * (width * height + 1), Vector<Vector<Vector2>>());
	} while (curx != startx || cury != starty);

	// Add remaining points to result.
	_points.resize(points_size);

	ret.set(0, _points);

	return ret;
}

/**
 * Check if a point(b) is between two line segment (a and c) perpendicular range. Does not include endpoints.
 * Uses dot product to get the directions for both endpoints and if their signs are different then the point is out of range
 */
static bool is_in_line_range(const Vector2 &b, const Vector2 &a, const Vector2 &c) {
	Vector2 ba = a - b;
	Vector2 bc = c - b;
	Vector2 ac = c - a;

	float dot1 = ba.dot(ac);
	float dot2 = bc.dot(ac);
	return (dot1 * dot2 < 0);
}

static float perpendicular_distance(const Vector2 &i, const Vector2 &start, const Vector2 &end) {
	float res;
	float slope;
	float intercept;

	if (is_in_line_range(i, start, end)) {
		if (start.x == end.x) {
			res = Math::absf(i.x - end.x);
		} else if (start.y == end.y) {
			res = Math::absf(i.y - end.y);
		} else {
			slope = (end.y - start.y) / (end.x - start.x);
			intercept = start.y - (slope * start.x);
			res = Math::absf(slope * i.x - i.y + intercept) / Math::sqrt(Math::pow(slope, 2.0f) + 1.0);
		}
	} else {
		res = MIN(i.distance_to(start), i.distance_to(end));

	}
	return res;
}

static Vector<Vector2> rdp(const Vector<Vector2> &v, float optimization) {
	if (v.size() < 3) {
		return v;
	}

	int index = -1;
	float dist = 0.0;
	// Not looping first and last point.
	for (size_t i = 1, size = v.size(); i < size - 1; ++i) {
		float cdist = perpendicular_distance(v[i], v[0], v[size - 1]);
		if (cdist > dist) {
			dist = cdist;
			index = static_cast<int>(i);
		}
	}
	if (dist > optimization) {
		Vector<Vector2> left, right;
		left.resize(index + 1);
		for (int i = 0; i < index + 1; i++) {
			left.write[i] = v[i];
		}
		right.resize(v.size() - index);
		for (int i = 0; i < right.size(); i++) {
			right.write[i] = v[index + i];
		}
		Vector<Vector2> r1 = rdp(left, optimization);
		Vector<Vector2> r2 = rdp(right, optimization);

		int middle = r1.size() - 1;
		r1.resize(r1.size() + r2.size() - 1);
		for (int i = 0; i < r2.size(); i++) {
			r1.write[middle + i] = r2[i];
		}
		return r1;
	} else {
		Vector<Vector2> ret;
		ret.push_back(v[0]);
		ret.push_back(v[v.size() - 1]);
		return ret;
	}
}

// Check if ABC is counterclockwise
static bool cntrcw(const Vector2 &A, const Vector2 &B, const Vector2 &C) {
	return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

// Line segments AB and CD intersect check
static bool intersect(const Vector2 &A, const Vector2 &B, const Vector2 &C, const Vector2 &D) {
	return cntrcw(A, C, D) != cntrcw(B, C, D) && cntrcw(A, B, C) != cntrcw(A, B, D);
}

static float angle_between_two_pnts(const Vector2 center_pnt, const Vector2 prev_pnt, const Vector2 new_pnt) {
	Vector2 v1 = Vector2(prev_pnt.y - center_pnt.y, prev_pnt.x - center_pnt.x);
	Vector2 v2 = Vector2(new_pnt.y - center_pnt.y, new_pnt.x - center_pnt.x);

	float dot = v1.x * v2.x + v1.y * v2.y;
	float det = v1.x * v2.y - v1.y * v2.x;

	return Math::atan2(det, dot);
}

int get_triangle_pnt(const Vector<PackedInt32Array> &graph, const int &ep1_i, const int &ep2_i, Vector<bool> &seen) {
	Vector<int> points = graph[ep1_i];
	Vector<bool> pTouched;
	pTouched.resize_zeroed(graph.size());

	for (int p : points) {
		pTouched.write[p] = true;
	}

	points = graph[ep2_i];
	for (int p : points) {
		if (pTouched[p] && !seen[p]) { // Only return the point if it hasn't already been processed
			return p;
		}
	}

	return -1; // None of the triangles available have not been processed
}

void tri_search(const Vector<PackedInt32Array> graph, const Vector<Vector2> &pl, const Vector2 &star_center, Vector<bool> &seen, Vector<int> &pKeep, int ep1_i, int ep2_i, const float special_contraint = -99) {
	Vector2 ep1 = pl[ep1_i];
	Vector2 ep2 = pl[ep2_i];

	PackedFloat32Array constraints = { angle_between_two_pnts(star_center, pl[0], ep1), angle_between_two_pnts(star_center, pl[0], ep2) };
	constraints.sort();

	int point_idx = get_triangle_pnt(graph, ep1_i, ep2_i, seen);
	if (point_idx == -1) {
		return;
	}

	Vector2 point = pl[point_idx];

	seen.write[point_idx] = true;
	float angle = angle_between_two_pnts(star_center, pl[0], point);
	if (constraints[0] < angle && angle < constraints[1] && (0 < angle && angle <= special_contraint)) {
		pKeep.append(point_idx);
		if (Math::abs(ep1_i - ep2_i) != 1) {
			tri_search(graph, pl, star_center, seen, pKeep, ep1_i, point_idx, special_contraint);
			tri_search(graph, pl, star_center, seen, pKeep, point_idx, ep2_i, special_contraint);
		}
	}
}

static bool is_between(const Vector2 &b, const Vector2 &a, const Vector2 &c) {
	return Math::absf((a.distance_to(b) + b.distance_to(c)) - a.distance_to(c)) < .001;
}

// Gets the star region for the star_center of the given polyline( pl ) in a clockwise pattern
static PackedInt32Array get_star_region(Vector<Vector2> &pl, const Vector2 &star_center) {
	pl.append(star_center);
	PackedInt32Array triangulation = Geometry2D::triangulate_polygon(pl);
	if (triangulation.size() == 0) { // Edge case where the polygon would self-intersect when closed
		pl.remove_at(pl.size() - 1);
		return PackedInt32Array({ 0 });
	}

	PackedInt32Array star_region;
	// Generate a neighbor graph for the triangulation
	Vector<PackedInt32Array> graph;
	graph.resize(pl.size());
	for (int i = 0; i < triangulation.size(); i += 3) {
		int p1_i = triangulation[i];
		int p2_i = triangulation[i + 1];
		int p3_i = triangulation[i + 2];

		graph.write[p1_i].append(p2_i);
		graph.write[p1_i].append(p3_i);

		graph.write[p2_i].append(p1_i);
		graph.write[p2_i].append(p3_i);

		graph.write[p3_i].append(p1_i);
		graph.write[p3_i].append(p2_i);
	}

	PackedInt32Array star_connections = graph[pl.size() - 1];
	Vector<bool> seen;
	seen.resize_zeroed(pl.size());
	seen.write[pl.size() - 1] = true;

	for (int i = 0; i < star_connections.size(); i += 2) {
		if (!seen[star_connections[i]]) {
			seen.write[star_connections[i]] = true;
			star_region.append(star_connections[i]);
		}

		if (!seen[star_connections[i + 1]]) {
			seen.write[star_connections[i + 1]] = true;
			star_region.append(star_connections[i + 1]);
		}

		// The angle needs to be positive facing from edge points 1 - 2
		if (angle_between_two_pnts(star_center, pl[star_connections[i]], pl[star_connections[i + 1]])) {
			tri_search(graph, pl, star_center, seen, star_region, star_connections[i], star_connections[i + 1], Math_PI);
		} else {
			tri_search(graph, pl, star_center, seen, star_region, star_connections[i + 1], star_connections[i], Math_PI);
		}
	}

	star_region.sort();

	pl.remove_at(pl.size() - 1); // Remove the add star_center from the list

	if (star_region[star_region.size() - 1] != pl.size() - 1) {
		star_region.push_back(pl.size() - 1);
	}

	return star_region;
}

// Calculates signed distance sign
bool is_outside_the_line_segment(const Vector2 &pnt, const Vector2 &start, const Vector2 &end) {
	return !cntrcw(start, pnt, end) || is_between(pnt, start, end);
}

int fix_polyline_intersection(Vector<Vector2> &simplified_pl, Vector<Vector2> &pl) {
	// Split into sections between sub-polyline
	PackedInt32Array convex_segment;
	Vector2 last_val = simplified_pl[simplified_pl.size() - 1];
	int result_cntr = 0;
	for (int i = 0; i < pl.size(); ++i) {
		if (pl[i] == simplified_pl[result_cntr]) {
			convex_segment.push_back(i);
			result_cntr += 1;
			if (pl[i] == last_val) {
				break;
			}
		}
	}
	int error_idx = -1; // Location of the error
	int segment_i = 0;
	while (segment_i < convex_segment.size() - 1) {
		Vector2i segment = { segment_i, segment_i + 1 };
		if (convex_segment[segment[1]] - convex_segment[segment[0]] == 1) { // Skip if no in-between points to make the convex hull from
			segment_i++;
			continue;
		}
		error_idx = segment[0];
		// Create the convex hull to test against and find the point that will be used to fix the convex hull
		Vector<Vector2> convex_hull;
		Vector2 start = simplified_pl[segment[0]];
		Vector2 end = simplified_pl[segment[1]];
		for (int pl_idx = convex_segment[segment[0]]; pl_idx < convex_segment[segment[1]] + 1; ++pl_idx) {
			convex_hull.push_back(pl[pl_idx]);
		}

		// Iterate through every sub-polyline section
		int furthest_intersecting_pnt_idx = -1;
		float max_dist = -1.f;
		for (int i = 0; i < simplified_pl.size(); ++i) {
			// Check if there is a point that intersects the hull
			Vector2 simp_pnt = simplified_pl[i];
			if (simp_pnt != start && simp_pnt != end && simp_pnt != convex_hull[0] && simp_pnt != convex_hull[convex_hull.size() - 1] && (Geometry2D::is_point_in_polygon(simp_pnt, convex_hull) || is_between(simp_pnt, start, end))) {
				float dist = perpendicular_distance(simp_pnt, start, end);
				if (dist > max_dist) {
					furthest_intersecting_pnt_idx = i;
					max_dist = dist;
				}
			}
		}
		if (furthest_intersecting_pnt_idx != -1) {
			// Find a point in the convex hull to fix the error
			Vector2 prob_end;
			Vector2 furthest_intersecting_pnt = simplified_pl[furthest_intersecting_pnt_idx];
			if (furthest_intersecting_pnt_idx == simplified_pl.size() - 1) {
				prob_end = simplified_pl[furthest_intersecting_pnt_idx - 1];
			} else if (furthest_intersecting_pnt_idx == 0) {
				prob_end = simplified_pl[furthest_intersecting_pnt_idx + 1];
			} else {
				if (simplified_pl[furthest_intersecting_pnt_idx - 1] == end || intersect(start, end, furthest_intersecting_pnt, simplified_pl[furthest_intersecting_pnt_idx + 1])) {
					prob_end = simplified_pl[furthest_intersecting_pnt_idx + 1];
				} else {
					prob_end = simplified_pl[furthest_intersecting_pnt_idx - 1];
				}
			}
			bool furthest_pnt_orientation = cntrcw(start, furthest_intersecting_pnt, end);
			bool special_orientation = is_between(furthest_intersecting_pnt, start, end);
			bool fixed = false;
			for (int pl_i = convex_segment[segment[0]] + 1; pl_i < convex_segment[segment[1]]; ++pl_i) {
				Vector2 pnt = pl[pl_i];
				if ((furthest_pnt_orientation == cntrcw(start, pnt, end)) || special_orientation) { // Only look through points that have the same orientation
					if (!intersect(start, pnt, furthest_intersecting_pnt, prob_end) && !intersect(pnt, end, furthest_intersecting_pnt, prob_end) && !is_between(furthest_intersecting_pnt, start, pnt) && !is_between(furthest_intersecting_pnt, pnt, end)) {
						simplified_pl.insert(segment[0] + 1, pnt);
						convex_segment.insert(segment_i + 1, pl_i);
						segment_i = -1; // Reset the intersection check
						fixed = true;
						break;
					}
				}
			}
			if (!fixed) {
				Vector2 pnt = pl[convex_segment[segment[1]] - 1];
				simplified_pl.insert(segment[0] + 1, pnt);
				convex_segment.insert(segment_i + 1, convex_segment[segment[1]] - 1);
				segment_i = -1; // Reset the intersection check
			}
		}
		segment_i++;
	}
	return error_idx; // Return the index of the bug
}

// Copy vector into another vector. In python: destination[start_idx:end_idx] = source
static void copy_section(Vector<Vector2> &destination, const Vector<Vector2> &source, int start_idx, int end_idx) {
	if ((end_idx - start_idx + 1) > source.size()) {
		for (int i = 0; i < source.size(); ++i)
			destination.write[start_idx + i] = source[i];

		// Remove any excess points in the replacement range that aren't used anymore
		int iter_end = start_idx + source.size();
		for (int i = start_idx + source.size(); i < end_idx; ++i)
			destination.remove_at(iter_end);

	} else {
		int end_iter = MIN(destination.size() - start_idx, end_idx - start_idx);

		for (int i = 0; i < end_iter; ++i)
			destination.write[start_idx + i] = source[i];

		end_iter = source.size() - end_iter + 1;
		for (int i = 1; i < end_iter; ++i)
			destination.insert(end_idx, source[source.size() - i]);
	}
}

static Vector<Vector2> star_shaped_rdp(Vector<Vector2> &v, float optimization) {
	if (v.size() < 3) {
		return v;
	}
	// 1 - (a) Determine the intersection of v1 vn and the original polyline. If there is no intersection go to (2).
	Vector2 v1_vn_intersection = (v[0] + v[v.size() - 1]) / 2.0;

	// 2 - (a) Determine the sequence of vertices Si lying in a star-shaped region.
	PackedInt32Array star_region = get_star_region(v, v1_vn_intersection);

	// Nothing was found in the search due to a self-intersection. Fix the self-intersection and rescan
	if (star_region.size() <= 2 && v.size() != 3) {
		int error_idx = -1;
		Vector2 end = v[v.size() - 1];
		Vector2 start = v[0];

		bool between_flag = false;
		if (star_region.size() == 1) { // The first point cannot see the last point
			for (int i = 1; i < v.size() - 1; ++i) {
				if (cntrcw(start, v[i], end)) {
					error_idx = i;
					break;
				}
			}
		} else { // The first and last points are being covered by another edge
			for (int i = 1; i < v.size() - 1; ++i) {
				if (is_outside_the_line_segment(v[i], start, end)) {
					error_idx = i;
					between_flag = true;
					break;
				}
			}
		}

		if (error_idx != -1) {
			// Split at the error
			Vector<Vector2> left, right;
			left.resize(error_idx + 1);
			for (int i = 0; i < error_idx + 1; i++) {
				left.write[i] = v[i];
			}
			right.resize(v.size() - error_idx);
			for (int i = 0; i < right.size(); i++) {
				right.write[i] = v[error_idx + i];
			}

			Vector<Vector2> r1 = star_shaped_rdp(left, optimization);
			Vector<Vector2> r2 = star_shaped_rdp(right, optimization);

			if (!between_flag) {
				r1.remove_at(r1.size() - 1);
			}
			r2.remove_at(0);

			r1.append_array(r2);

			return r1;
		}
	}

	// 2 - (b) Apply the Douglas-Peucker algorithm for these vertices
	Vector<Vector2> star_region_polyline;
	star_region_polyline.resize(star_region.size());
	for (int i = 0; i < star_region.size(); ++i) {
		star_region_polyline.write[i] = v[star_region[i]];
	}
	star_region_polyline = rdp(star_region_polyline, optimization);

	if (star_region.size() == star_region_polyline.size()) {
		return rdp(v, optimization);
	}

	if (star_region_polyline.size() == 2) {
		for (int i = star_region.size() - 2; i > 0; --i) {
			v.remove_at(star_region[i]);
		}
	}
	// 2 - (c) Check the distance of vertices out of star_region_polyline.
	bool no_point_larger_than_eps = true; // Skip checking the perpendicular distance when a point is found that is larger than the optimization
	Vector<Vector2> edge = { star_region_polyline[0], star_region_polyline[1] };
	int edge_idx = 2; // The index of the next edge
	int start_idx = 0; // The index of the start of the polyline section
	int polyline_size = v.size();

	for (int i = 1; i < polyline_size; ++i) {
		Vector2 pnt = v[i];
		if (!star_region_polyline.has(pnt)) {
			if (no_point_larger_than_eps) {
				if (perpendicular_distance(pnt, edge[0], edge[1]) > optimization) {
					no_point_larger_than_eps = false;
				}
			}
		} else {
			// 2 - (c) If some have a distance greater than the specified tolerance, go to (2a).
			if (!no_point_larger_than_eps) {
				Vector<Vector2> new_polyline_section;
				for (int j = start_idx; j < i + 1; j++)
					new_polyline_section.push_back(v[j]);

				Vector<Vector2> res = star_shaped_rdp(new_polyline_section, optimization);
				copy_section(v, res, start_idx, i + 1);

				// Update the size and index pointer
				int old_size = polyline_size;
				polyline_size = v.size();
				i -= old_size - polyline_size;

			} else { // Remove the points between the two new points as they are in the optimization
				copy_section(v, edge, start_idx, i + 1);

				// Update the size and index pointer
				int old_size = polyline_size;
				polyline_size = v.size();
				i -= old_size - polyline_size;
			}

			// Move on to the next section if not at the polyline end
			if (i != (polyline_size - 1)) {
				edge.write[0] = edge[1];
				edge.write[1] = star_region_polyline[edge_idx];
				edge_idx += 1;
				start_idx = i;
			}
		}
	}

	return v;
}

static Vector<Vector2> reduce(Vector<Vector2> &points, const Rect2i &rect, float epsilon, bool p_star_rdp) {
	int size = points.size();
	// If there are less than 3 points, then we have nothing.
	ERR_FAIL_COND_V(size < 3, Vector<Vector2>());
	// If there are less than 9 points (but more than 3), then we don't need to reduce it.
	if (size < 9) {
		return points;
	}

	float maxEp = MIN(rect.size.width, rect.size.height);
	float ep = CLAMP(epsilon, 0.0, maxEp / 2);

	Vector<Vector2> result;
	if (p_star_rdp) {
		Vector<Vector2> orig_points(points); // For the self-intersection fix
		result = star_shaped_rdp(points, ep);
		fix_polyline_intersection(result, orig_points);
	} else {
		result = rdp(points, epsilon);
	}

	Vector2 last = result[result.size() - 1];

	if (last.y > result[0].y && last.distance_to(result[0]) < ep * 0.5f) {
		result.write[0].y = last.y;
		result.resize(result.size() - 1);
	}
	return result;
}

struct FillBitsStackEntry {
	Point2i pos;
	int i = 0;
	int j = 0;
};

static void fill_bits(const BitMap *p_src, Ref<BitMap> &p_map, const Point2i &p_pos, const Rect2i &rect) {
	// Using a custom stack to work iteratively to avoid stack overflow on big bitmaps.
	Vector<FillBitsStackEntry> stack;
	// Tracking size since we won't be shrinking the stack vector.
	int stack_size = 0;

	Point2i pos = p_pos;
	int next_i = 0;
	int next_j = 0;

	bool reenter = true;
	bool popped = false;
	do {
		if (reenter) {
			next_i = pos.x - 1;
			next_j = pos.y - 1;
			reenter = false;
		}

		for (int i = next_i; i <= pos.x + 1; i++) {
			for (int j = next_j; j <= pos.y + 1; j++) {
				if (popped) {
					// The next loop over j must start normally.
					next_j = pos.y - 1;
					popped = false;
					// Skip because an iteration was already executed with current counter values.
					continue;
				}

				if (i < rect.position.x || i >= rect.position.x + rect.size.x) {
					continue;
				}
				if (j < rect.position.y || j >= rect.position.y + rect.size.y) {
					continue;
				}

				if (p_map->get_bit(i, j)) {
					continue;

				} else if (p_src->get_bit(i, j)) {
					p_map->set_bit(i, j, true);

					FillBitsStackEntry se = { pos, i, j };
					stack.resize(MAX(stack_size + 1, stack.size()));
					stack.set(stack_size, se);
					stack_size++;

					pos = Point2i(i, j);
					reenter = true;
					break;
				}
			}
			if (reenter) {
				break;
			}
		}
		if (!reenter) {
			if (stack_size) {
				FillBitsStackEntry se = stack.get(stack_size - 1);
				stack_size--;
				pos = se.pos;
				next_i = se.i;
				next_j = se.j;
				popped = true;
			}
		}
	} while (reenter || popped);
}

Vector<Vector<Vector2>> BitMap::clip_opaque_to_polygons(const Rect2i &p_rect, float p_epsilon, bool p_star_rdp) const {
	Rect2i r = Rect2i(0, 0, width, height).intersection(p_rect);

	Ref<BitMap> fill;
	fill.instantiate();
	fill->create(get_size());

	Vector<Vector<Vector2>> polygons;
	for (int i = r.position.y; i < r.position.y + r.size.height; i++) {
		for (int j = r.position.x; j < r.position.x + r.size.width; j++) {
			if (!fill->get_bit(j, i) && get_bit(j, i)) {
				fill_bits(this, fill, Point2i(j, i), r);

				for (Vector<Vector2> polygon : _march_square(r, Point2i(j, i))) {
					polygon = reduce(polygon, r, p_epsilon, p_star_rdp);

					if (polygon.size() < 3) {
						print_verbose("Invalid polygon, skipped");
						continue;
					}

					polygons.push_back(polygon);
				}
			}
		}
	}

	return polygons;
}

void BitMap::grow_mask(int p_pixels, const Rect2i &p_rect) {
	if (p_pixels == 0) {
		return;
	}

	bool bit_value = p_pixels > 0;
	p_pixels = Math::abs(p_pixels);
	const int pixels2 = p_pixels * p_pixels;

	Rect2i r = Rect2i(0, 0, width, height).intersection(p_rect);

	Ref<BitMap> copy;
	copy.instantiate();
	copy->create(get_size());
	copy->bitmask = bitmask;

	for (int i = r.position.y; i < r.position.y + r.size.height; i++) {
		for (int j = r.position.x; j < r.position.x + r.size.width; j++) {
			if (bit_value == get_bit(j, i)) {
				continue;
			}

			bool found = false;

			for (int y = i - p_pixels; y <= i + p_pixels; y++) {
				for (int x = j - p_pixels; x <= j + p_pixels; x++) {
					bool outside = false;

					if ((x < p_rect.position.x) || (x >= p_rect.position.x + p_rect.size.x) || (y < p_rect.position.y) || (y >= p_rect.position.y + p_rect.size.y)) {
						// Outside of rectangle counts as bit not set.
						if (!bit_value) {
							outside = true;
						} else {
							continue;
						}
					}

					float d = Point2(j, i).distance_squared_to(Point2(x, y)) - CMP_EPSILON2;
					if (d > pixels2) {
						continue;
					}

					if (outside || (bit_value == copy->get_bit(x, y))) {
						found = true;
						break;
					}
				}
				if (found) {
					break;
				}
			}

			if (found) {
				set_bit(j, i, bit_value);
			}
		}
	}
}

void BitMap::shrink_mask(int p_pixels, const Rect2i &p_rect) {
	grow_mask(-p_pixels, p_rect);
}

TypedArray<PackedVector2Array> BitMap::_opaque_to_polygons_bind(const Rect2i &p_rect, float p_epsilon, bool p_star_rdp) const {
	Vector<Vector<Vector2>> result = clip_opaque_to_polygons(p_rect, p_epsilon, p_star_rdp);

	// Convert result to bindable types.

	TypedArray<PackedVector2Array> result_array;
	result_array.resize(result.size());
	for (int i = 0; i < result.size(); i++) {
		const Vector<Vector2> &polygon = result[i];

		PackedVector2Array polygon_array;
		polygon_array.resize(polygon.size());

		{
			Vector2 *w = polygon_array.ptrw();
			for (int j = 0; j < polygon.size(); j++) {
				w[j] = polygon[j];
			}
		}

		result_array[i] = polygon_array;
	}

	return result_array;
}

void BitMap::resize(const Size2i &p_new_size) {
	ERR_FAIL_COND(p_new_size.width < 0 || p_new_size.height < 0);
	if (p_new_size == get_size()) {
		return;
	}

	Ref<BitMap> new_bitmap;
	new_bitmap.instantiate();
	new_bitmap->create(p_new_size);
	// also allow for upscaling
	int lw = (width == 0) ? 0 : p_new_size.width;
	int lh = (height == 0) ? 0 : p_new_size.height;

	float scale_x = ((float)width / p_new_size.width);
	float scale_y = ((float)height / p_new_size.height);
	for (int x = 0; x < lw; x++) {
		for (int y = 0; y < lh; y++) {
			bool new_bit = get_bit(x * scale_x, y * scale_y);
			new_bitmap->set_bit(x, y, new_bit);
		}
	}

	width = new_bitmap->width;
	height = new_bitmap->height;
	bitmask = new_bitmap->bitmask;
}

Ref<Image> BitMap::convert_to_image() const {
	Ref<Image> image = Image::create_empty(width, height, false, Image::FORMAT_L8);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			image->set_pixel(i, j, get_bit(i, j) ? Color(1, 1, 1) : Color(0, 0, 0));
		}
	}

	return image;
}

void BitMap::blit(const Vector2i &p_pos, const Ref<BitMap> &p_bitmap) {
	ERR_FAIL_COND_MSG(p_bitmap.is_null(), "It's not a reference to a valid BitMap object.");

	int x = p_pos.x;
	int y = p_pos.y;
	int w = p_bitmap->get_size().width;
	int h = p_bitmap->get_size().height;

	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			int px = x + i;
			int py = y + j;
			if (px < 0 || px >= width) {
				continue;
			}
			if (py < 0 || py >= height) {
				continue;
			}
			if (p_bitmap->get_bit(i, j)) {
				set_bit(px, py, true);
			}
		}
	}
}

void BitMap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create", "size"), &BitMap::create);
	ClassDB::bind_method(D_METHOD("create_from_image_alpha", "image", "threshold"), &BitMap::create_from_image_alpha, DEFVAL(0.1));

	ClassDB::bind_method(D_METHOD("set_bitv", "position", "bit"), &BitMap::set_bitv);
	ClassDB::bind_method(D_METHOD("set_bit", "x", "y", "bit"), &BitMap::set_bit);
	ClassDB::bind_method(D_METHOD("get_bitv", "position"), &BitMap::get_bitv);
	ClassDB::bind_method(D_METHOD("get_bit", "x", "y"), &BitMap::get_bit);

	ClassDB::bind_method(D_METHOD("set_bit_rect", "rect", "bit"), &BitMap::set_bit_rect);
	ClassDB::bind_method(D_METHOD("get_true_bit_count"), &BitMap::get_true_bit_count);

	ClassDB::bind_method(D_METHOD("get_size"), &BitMap::get_size);
	ClassDB::bind_method(D_METHOD("resize", "new_size"), &BitMap::resize);

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &BitMap::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &BitMap::_get_data);

	ClassDB::bind_method(D_METHOD("grow_mask", "pixels", "rect"), &BitMap::grow_mask);
	ClassDB::bind_method(D_METHOD("convert_to_image"), &BitMap::convert_to_image);
	ClassDB::bind_method(D_METHOD("opaque_to_polygons", "rect", "epsilon", "star_rdp"), &BitMap::_opaque_to_polygons_bind, DEFVAL(2.0), DEFVAL(false));

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}
