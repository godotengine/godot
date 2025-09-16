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

#include "core/math/geometry_2d.h"
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
			res = Math::abs(i.x - end.x);
		} else if (start.y == end.y) {
			res = Math::abs(i.y - end.y);
		} else {
			slope = (end.y - start.y) / (end.x - start.x);
			intercept = start.y - (slope * start.x);
			res = Math::abs(slope * i.x - i.y + intercept) / Math::sqrt(Math::pow(slope, 2.0f) + 1.0);
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

// X-Axis dependent. Stores the pointer (address) of the point
static List<List<Vector2>::Element *> generate_mono_chains(List<Vector2> &pl) {
	List<List<Vector2>::Element *> mono_chain_lst;

	List<Vector2>::Element *iter_node = pl.front();
	mono_chain_lst.push_back(iter_node);

	float dx = iter_node->next()->get()[0] - iter_node->get()[0];
	iter_node = iter_node->next();
	while (iter_node->next()) {
		iter_node = iter_node->next();
		float ndx = iter_node->get()[0] - iter_node->prev()->get()[0];
		if (dx * ndx < 0) { // If they are not the same direction
			mono_chain_lst.push_back(iter_node->prev());
			dx = ndx;
		}
	}
	mono_chain_lst.push_back(pl.back()); // Get the end

	return mono_chain_lst;
}

static Vector2 project_rectangle(const PackedVector2Array &rect, const Vector2 &axis) {
	float p0 = rect[0].dot(axis);
	float p1 = rect[1].dot(axis);
	float p2 = rect[2].dot(axis);
	float p3 = rect[3].dot(axis);

	float min = MIN(MIN(p0, p1), MIN(p2, p3));
	float max = MAX(MAX(p0, p1), MAX(p2, p3));

	return Vector2(min, max);
}

static bool seperated_axis_theorum(const PackedVector2Array &rect1, const PackedVector2Array &rect2) {
	PackedVector2Array rect;
	Vector2 edge1, edge2, min_max_1, min_max_2;
	for (int i = 0; i < 2; ++i) {
		if (i == 0) {
			rect = rect1;
		} else {
			rect = rect2;
		}
		// Get two edge vectors (assumes rectangle points are ordered)
		edge1 = rect[1] - rect[0];
		edge2 = rect[3] - rect[0];
		PackedVector2Array axes = {
			Vector2(-edge1[1], edge1[0]).normalized(), // perpendicular to edge1
			Vector2(-edge2[1], edge2[0]).normalized() // perpendicular to edge2
		};

		for (const Vector2 &axis : axes) {
			min_max_1 = project_rectangle(rect1, axis);
			min_max_2 = project_rectangle(rect2, axis);
			if (min_max_1[1] < min_max_2[0] || min_max_2[1] < min_max_1[0]) {
				return false; // Separating axis found -> no intersection
			}
		}
	}
	return true; // No separating axis -> they intersect
}

// Rotating Calipers Algorithm to determine the Minimum Area Enclosed Rectangle
static PackedVector2Array generate_bbox_from_polyline(const Vector<Vector2> &pl) {
	if (pl.size() <= 2) {
		return pl;
	}

	PackedVector2Array polygon = Geometry2D::convex_hull(pl);
	polygon.remove_at(polygon.size() - 1);

	int n = polygon.size();
	float min_area = INFINITY;

	PackedVector2Array best_rect;
	best_rect.resize(4);

	Vector2 p1, p2, p_edge;
	for (int i = 0; i < n; ++i) {
		// Edge vector
		p1 = polygon[i];
		p2 = polygon[(i + 1) % n];
		float dx = p2.x - p1.x;
		float dy = p2.y - p1.y;
		float length = 1.0 / sqrt(dx * dx + dy * dy);

		float ux = dx * length; // Unit edge vector
		float uy = dy * length;

		// Perp. vector
		float vx = -uy;
		float vy = ux;

		// Min/max projections along edge and perpendicular
		float min_u = polygon[0].x * ux + polygon[0].y * uy;
		float max_u = min_u;
		float min_v = polygon[0].x * vx + polygon[0].y * vy;
		float max_v = min_v;

		for (int j = 1; j < n; ++j) {
			p_edge = polygon[j];
			float proj_u = p_edge.x * ux + p_edge.y * uy;
			float proj_v = p_edge.x * vx + p_edge.y * vy;

			min_u = MIN(min_u, proj_u);
			max_u = MAX(max_u, proj_u);
			min_v = MIN(min_v, proj_v);
			max_v = MAX(max_v, proj_v);
		}

		float width = max_u - min_u;
		float height = max_v - min_v;
		float area = width * height;

		if (area < min_area) {
			min_area = area;

			// Origin (bottom-left corner)
			float ox = ux * min_u + vx * min_v;
			float oy = uy * min_u + vy * min_v;

			// Construct the new MER(minimum-area enclosed Rectangle)
			best_rect = PackedVector2Array(
					{ { ox, oy },
							{ ox + ux * width, oy + uy * width },
							{ 0, 0 },
							{ ox + vx * height, oy + vy * height } });
			best_rect.write[2] = { best_rect[1].x + vx * height, best_rect[1].y + vy * height };
		}
	}

	return best_rect;
}

/**
 * This does not include endpoints because if 2 lines intersect at their end point, it is impossible
 * for any point that is added to remove such an intersection, causing the algorithm to try and put
 * non-existent points between the segments, causing a crash.
 */
static bool non_endpoint_segment_intersection(const Vector<Vector2> &line1, const Vector<Vector2> &line2) {
	// Returns true if they intersect, false otherwise.
	bool intersect = Geometry2D::segment_intersects_segment(line1[0], line1[1], line2[0], line2[1], nullptr); // No Result. Only care if they collide

	// Exclude the endpoints
	if (intersect &&
			(line1[0].is_equal_approx(line2[0]) ||
					line1[0].is_equal_approx(line2[1]) ||
					line1[1].is_equal_approx(line2[0]) ||
					line1[1].is_equal_approx(line2[1]))) {
		intersect = false;
	}

	return intersect;
}

static bool does_bbox_collide_with_line(const Vector<Vector2> &bbox, const Vector<Vector2> &line) {
	Vector2 edges[] = {
		bbox[0],
		bbox[1],
		bbox[1],
		bbox[2],
		bbox[2],
		bbox[3],
		bbox[3],
		bbox[0],
	};

	for (int i = 0; i < 8; i += 2) {
		if (non_endpoint_segment_intersection(line, { edges[i], edges[i + 1] })) {
			return true;
		}
	}

	// Edge case where 1 point matches a bbox point, while the other point is inside of the bbox
	if (Geometry2D::is_point_in_polygon(line[0], bbox) || Geometry2D::is_point_in_polygon(line[1], bbox)) {
		return true;
	}

	return false;
}

static bool does_polyline_bboxes_collide(const PackedVector2Array &p1_line, const PackedVector2Array &p2_line) {
	int p1_len = p1_line.size();
	int p2_len = p2_line.size();

	// If the polyline is not 4 points(OBB) and is not a line its needs it's bbox calculated
	bool p1_is_poly = (p1_len != 4 && p1_len > 2);
	bool p2_is_poly = (p2_len != 4 && p2_len > 2);

	PackedVector2Array p1_bbox, p2_bbox;
	if (p1_is_poly) {
		p1_bbox = generate_bbox_from_polyline(p1_line);
	} else {
		p1_bbox = p1_line; // p1_line is a obb
	}
	if (p2_is_poly) {
		p2_bbox = generate_bbox_from_polyline(p2_line);
	} else {
		p2_bbox = p2_line; // p2_line is a obb
	}

	// OBB vs Line
	if (p1_len == 2 && !p1_is_poly) { // p1 is a line segment
		return does_bbox_collide_with_line(p2_bbox, p1_line);
	}
	if (p2_len == 2 && !p2_is_poly) { // p2 is a line segment
		return does_bbox_collide_with_line(p1_bbox, p2_line);
	}

	// OBB vs OBB
	return seperated_axis_theorum(p1_bbox, p2_bbox) and seperated_axis_theorum(p2_bbox, p1_bbox);
}

// Squared distance
static float high_speed_perp_dist(const Vector2 &p, const Vector2 &e1, const Vector2 &e2) {
	float res = -1.0;
	if (e2.x - e1.x == 0) {
		res = Math::abs(p.x - e2.x);

	} else if (e2.y - e1.y == 0) {
		res = Math::abs(p.y - e2.y);

	} else {
		float slope = (e2.y - e1.y) / (e2.x - e1.x);
		float intercept = e1.y - (slope * e1.x);
		float numerator = slope * p.x - p.y + intercept;
		res = (numerator * numerator) / (slope * slope + 1);
	}
	return res;
}

// Returns the index of the point with the furthest perpendicular distance from the edge in the given range
static int find_furthest_perp_point_from_edge(const PackedVector2Array &pl, const int start, const int end) {
	int max_dist = -1;
	int idx = -1;

	Vector2 st_pnt = pl[start];
	Vector2 end_pnt = pl[end];
	for (int pnt_i = start + 1; pnt_i < end; ++pnt_i) {
		float dist = Math::abs(high_speed_perp_dist(pl[pnt_i], st_pnt, end_pnt));
		if (dist > max_dist) {
			max_dist = dist;
			idx = pnt_i;
		}
	}
	return idx;
}

static Vector<Vector2> retrieve_list_from_id_range(List<Vector2>::Element *start_ptr, const List<Vector2>::Element *end_ptr) {
	Vector<Vector2> list;

	int count = 0;
	List<Vector2>::Element *it = start_ptr;
	while (it) {
		count++;
		if (it == end_ptr) {
			break;
		}
		it = it->next();
	}

	list.resize(count);

	it = start_ptr;
	for (int i = 0; i < count; i++) {
		list.write[i] = it->get();
		it = it->next();
	}

	return list;
}

static Vector4 generate_aabb(const PackedVector2Array &bbox) {
	float min_x = INFINITY;
	float min_y = INFINITY;
	float max_x = -INFINITY;
	float max_y = -INFINITY;
	for (int i = 0; i < bbox.size(); ++i) {
		Vector2 p = bbox[i];
		min_x = MIN(min_x, p.x);
		min_y = MIN(min_y, p.y);
		max_x = MAX(max_x, p.x);
		max_y = MAX(max_y, p.y);
	}

	return Vector4({ min_x, min_y, max_x, max_y });
}

struct ChainBoxData {
	Vector4 aabb;
	List<List<Vector2>::Element *>::Element *chain_end_node;
	PackedVector2Array obb;
};

static Vector<ChainBoxData> generate_obbs_aabbs(List<List<Vector2>::Element *> &mono_chain_lst) {
	Vector<ChainBoxData> obb_aabb_data;

	PackedVector2Array obb;
	List<List<Vector2>::Element *>::Element *iter_node = mono_chain_lst.front();
	List<List<Vector2>::Element *>::Element *ch_end_node;
	while (iter_node->next()) {
		ch_end_node = iter_node->next();

		obb = generate_bbox_from_polyline(retrieve_list_from_id_range(iter_node->get(), ch_end_node->get()));

		obb_aabb_data.push_back({ generate_aabb(obb),
				ch_end_node,
				obb });
		iter_node = iter_node->next();
	}

	return obb_aabb_data;
}

// Returns an array of intersecting aabbs: (aabb1_idx, aabb9_idx)
static Vector<Vector2i> find_intersections_sweep(const Vector<ChainBoxData> &obb_aabb_data) {
	class SweepEvent {
	public:
		float x;
		int type; // type: 0 - start | 1 - end
		float y1, y2;
		int index;

		SweepEvent() {}

		SweepEvent(float p_x, int p_type, float p_y1, float p_y2, int p_index) :
				x(p_x), type(p_type), y1(p_y1), y2(p_y2), index(p_index) {}

		bool operator<(const SweepEvent &p_ev) const { // for sort()
			return (x < p_ev.x || (x == p_ev.x && type < p_ev.type));
		}
	};

	// Each event: (x, type, y_min, y_max, index).
	Vector<SweepEvent> events;

	for (int i = 0; i < obb_aabb_data.size(); ++i) {
		const Vector4 rect = obb_aabb_data[i].aabb;
		events.push_back({ rect[0], 0, rect[1], rect[3], i });
		events.push_back({ rect[2], 1, rect[1], rect[3], i });
	}
	// Sort by x; 'start' comes before 'end' if equal
	events.sort();

	struct ActiveEntry {
		float ay1, ay2;
		int aidx;
	};

	Vector<ActiveEntry> active;
	Vector<Vector2i> result;

	for (SweepEvent event : events) {
		if (event.type == 0) {
			for (ActiveEntry &a : active) {
				if (!(event.y2 <= a.ay1 || a.ay2 <= event.y1)) {
					result.push_back({ a.aidx, event.index });
				}
			}
			active.push_back({ event.y1, event.y2, event.index });
		} else {
			// Remove all entries from active where aidx == event.index
			for (int i = active.size() - 1; i >= 0; --i) {
				if (active[i].aidx == event.index) {
					active.remove_at(i);
				}
			}
		}
	}

	return result;
}

/*
 * Stores the intersecting location for each chain.
 * Each intersection should have a reference to their mono atomic chain they are in as well as the
 * actual node that is colliding in the result.
 */
struct ChainIntersection {
	List<List<Vector2>::Element *>::Element *ch1_mono_chain;
	List<Vector2>::Element *ch1_end_node;
	List<List<Vector2>::Element *>::Element *ch2_mono_chain;
	List<Vector2>::Element *ch2_end_node;
};

void recursive_obb_collision_check(List<List<Vector2>::Element *>::Element *ch1_list, List<Vector2>::Element *ch1_start, List<Vector2>::Element *ch1_end,
		List<List<Vector2>::Element *>::Element *ch2_list, List<Vector2>::Element *ch2_start, List<Vector2>::Element *ch2_end,
		Vector<ChainIntersection> &possible_intersections) {
	// Step 5: Recursively finds the smallest segment pair causing a collision.
	Vector<Vector2> pl1 = retrieve_list_from_id_range(ch1_start, ch1_end);
	Vector<Vector2> pl2 = retrieve_list_from_id_range(ch2_start, ch2_end);

	int len1 = pl1.size();
	int len2 = pl2.size();

	// Collision possible
	if (len1 <= 2 && len2 <= 2) {
		if (len1 == 2 && len2 == 2 && non_endpoint_segment_intersection(pl1, pl2)) {
			possible_intersections.append({ ch1_list, ch1_end, ch2_list, ch2_end }); // Collision found
		}
		return;
	}

	// No collision
	if (!does_polyline_bboxes_collide(pl1, pl2)) {
		return;
	}

	if (len1 >= len2 && len1 > 2) {
		// Split Chain 1
		int mid_idx = int((len1 + 1) / 2);
		List<Vector2>::Element *mid_node = ch1_start;
		for (int i = 0; i < mid_idx - 1; ++i) {
			mid_node = mid_node->next();
		}

		// Recursively check both halves of Chain 1 against Chain 2
		recursive_obb_collision_check(ch1_list, ch1_start, mid_node,
				ch2_list, ch2_start, ch2_end,
				possible_intersections);
		recursive_obb_collision_check(ch1_list, mid_node, ch1_end,
				ch2_list, ch2_start, ch2_end,
				possible_intersections);

	} else if (len2 > 2) {
		// Split Chain 2
		int mid_idx = int((len2 + 1) / 2);
		List<Vector2>::Element *mid_node = ch2_start;
		for (int i = 0; i < mid_idx - 1; ++i) {
			mid_node = mid_node->next();
		}

		// Recursively check Chain 1 against both halves of Chain 2
		recursive_obb_collision_check(ch1_list, ch1_start, ch1_end,
				ch2_list, ch2_start, mid_node,
				possible_intersections);
		recursive_obb_collision_check(ch1_list, ch1_start, ch1_end,
				ch2_list, mid_node, ch2_end,
				possible_intersections);
	}
}

Vector<ChainIntersection> find_intersections(List<List<Vector2>::Element *> &mono_chain_lst) {
	Vector<ChainBoxData> obbs_data = generate_obbs_aabbs(mono_chain_lst);
	Vector<Vector2i> aabb_collisions = find_intersections_sweep(obbs_data); // Use AABBs to avoid costly OBB generation + recursion

	// Stores tuples of (ch1_mono_chain, ch1_end_node, ch2_mono_chain, ch2_end_node) for colliding segments
	Vector<ChainIntersection> intersections;

	for (Vector2i &collision : aabb_collisions) {
		List<List<Vector2>::Element *>::Element *ch1_end_node = obbs_data[collision[0]].chain_end_node;
		List<List<Vector2>::Element *>::Element *ch2_end_node = obbs_data[collision[1]].chain_end_node;

		List<List<Vector2>::Element *>::Element *ch1_start_node = ch1_end_node->prev();
		List<List<Vector2>::Element *>::Element *ch2_start_node = ch2_end_node->prev();

		PackedVector2Array obb1 = obbs_data[collision[0]].obb;
		PackedVector2Array obb2 = obbs_data[collision[1]].obb;

		// In case the 2 obbs are just 2 lines
		if (obb1.size() == 2 and obb2.size() == 2) {
			if (non_endpoint_segment_intersection(obb1, obb2)) {
				intersections.append({ ch1_start_node, ch1_end_node->get(), ch2_start_node, ch2_end_node->get() }); // Collision found
			}
			continue;
		}

		if (does_polyline_bboxes_collide(obb1, obb2)) {
			recursive_obb_collision_check(ch1_start_node, ch1_start_node->get(), ch1_end_node->get(),
					ch2_start_node, ch2_start_node->get(), ch2_end_node->get(),
					intersections);
		}
	}

	return intersections;
}

// Find the index of an item based on it's value in a linked list
static int list_index(List<Vector2> &list, const List<Vector2>::Element *item) {
	int idx = 0;

	List<Vector2>::Element *iter_node = list.front();
	while (iter_node) {
		if (item == iter_node) {
			return idx;
		}
		iter_node = iter_node->next();
		++idx;
	}

	return -1;
}

static Vector<Vector2> monotonic_chain_rdp(Vector<Vector2> &pl, const float optimization) {
	Vector<Vector2> orig_res = rdp(pl, optimization);

	List<Vector2> result;
	for (int i = 0; i < orig_res.size(); ++i) {
		result.push_back(orig_res[i]);
	}
	List<List<Vector2>::Element *> mono_chain_lst = generate_mono_chains(result);

	Vector<ChainIntersection> intersections = find_intersections(mono_chain_lst);

	// Result mapped to the original points
	PackedInt64Array mapped_result;
	if (intersections.size() > 0) {
		mapped_result.resize(result.size());
		int res_idx = 0;
		List<Vector2>::Element *res_idx_node = result.front();
		for (int i = 0; i < pl.size(); ++i) {
			if (pl[i] == res_idx_node->get()) {
				mapped_result.write[res_idx] = i;
				++res_idx;
				res_idx_node = res_idx_node->next();
			}
		}
	}

	// Step 8: Iterative refinement loop
	while (!intersections.is_empty()) {
		Vector<ChainIntersection> next_intersections; // List of new possible intersections after the current pass
		while (!intersections.is_empty()) {
			ChainIntersection intersection = intersections[0];
			intersections.remove_at(0);

			List<List<Vector2>::Element *>::Element *ch1_list = intersection.ch1_mono_chain;
			List<Vector2>::Element *ch1_end_node = intersection.ch1_end_node;

			List<List<Vector2>::Element *>::Element *ch2_list = intersection.ch2_mono_chain;
			List<Vector2>::Element *ch2_end_node = intersection.ch2_end_node;

			// Get the index of the ends of both chains in the original list
			int ch1_idx = list_index(result, ch1_end_node);
			int ch2_idx = list_index(result, ch2_end_node);
			PackedInt64Array edges = {
				mapped_result[ch1_idx - 1], // First chain edges
				mapped_result[ch1_idx],

				mapped_result[ch2_idx - 1], // Second chain edges
				mapped_result[ch2_idx]
			};

			Vector2i pnt_additions = { -1, -1 };
			int res = find_furthest_perp_point_from_edge(pl, edges[0], edges[1]); // Point addition for the first chain
			if (res != -1) {
				pnt_additions[0] = res;
			}

			res = find_furthest_perp_point_from_edge(pl, edges[2], edges[3]); // Point addition for the first chain
			if (res != -1) {
				pnt_additions[1] = res;
			}

			// Add the points
			Vector<List<Vector2>::Element *> new_chain_end_nodes; // List that stores if the chain intersects. [first_new_chain, second_new_chain].
			for (int pnt_i = 0; pnt_i < 2; ++pnt_i) { // 0 = first chain, 1 = second chain
				int pnt = pnt_additions[pnt_i];

				List<Vector2>::Element *chain_node_to_split;
				List<List<Vector2>::Element *>::Element *ch_list;

				if (pnt_i == 0) {
					chain_node_to_split = ch1_end_node;
					ch_list = ch1_list;
				} else {
					chain_node_to_split = ch2_end_node;
					ch_list = ch2_list;
				}

				if (pnt != -1) {
					// Bin-search to find the idx to insert the point
					int low = 0;
					int high = mapped_result.size();
					while (low < high) {
						int mid = floor((low + high) / 2);
						if (mapped_result[mid] < pnt) {
							low = mid + 1;
						} else {
							high = mid;
						}
					}
					// Insert the point
					mapped_result.insert(low, pnt);

					List<Vector2, DefaultAllocator>::Element *insert_node = result.front();
					for (int i = 0; i < low; ++i) {
						insert_node = insert_node->next();
					}
					result.insert_before(insert_node, pl[pnt]);

					// Add the point as an new split in the monochain
					mono_chain_lst.insert_before(ch_list->next(), insert_node->prev());

					// Add the two new chains to the list
					new_chain_end_nodes.push_back(chain_node_to_split->prev()); // low - 1 -> low
					new_chain_end_nodes.push_back(chain_node_to_split); // low -> low + 1
				} else {
					new_chain_end_nodes.push_back(chain_node_to_split);
					new_chain_end_nodes.push_back(NULL);
				}
			}
			// Check if the added point still intersects the other edge
			for (int chain_i = 0; chain_i < 2; ++chain_i) { // Check the first chain splits
				if (new_chain_end_nodes[chain_i] != NULL) {
					List<Vector2>::Element *ch1 = new_chain_end_nodes[chain_i];

					for (int chain_j = 2; chain_j < 4; ++chain_j) { // Check the second chain splits
						if (new_chain_end_nodes[chain_j] != NULL) {
							List<Vector2>::Element *ch2 = new_chain_end_nodes[chain_j];

							recursive_obb_collision_check(
									ch1_list, ch1->prev(), ch1,
									ch2_list, ch2->prev(), ch2,
									next_intersections);
						}
					}
				}
			}
		}
		intersections = next_intersections;
	}
	// Turn it back into a Vector and return the fixed polygon
	return retrieve_list_from_id_range(result.front(), result.back());
}

static Vector<Vector2> reduce(Vector<Vector2> &points, const Rect2i &rect, float epsilon, bool p_advanced_rdp) {
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
	if (p_advanced_rdp) {
		result = monotonic_chain_rdp(points, ep);
	} else {
		result = rdp(points, ep);
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

TypedArray<PackedVector2Array> BitMap::_opaque_to_polygons_bind(const Rect2i &p_rect, float p_epsilon, bool p_advanced_rdp) const {
	Vector<Vector<Vector2>> result = clip_opaque_to_polygons(p_rect, p_epsilon, p_advanced_rdp);

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
	ClassDB::bind_method(D_METHOD("opaque_to_polygons", "rect", "epsilon", "advanced_rdp"), &BitMap::_opaque_to_polygons_bind, DEFVAL(2.0), DEFVAL(false));

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}
