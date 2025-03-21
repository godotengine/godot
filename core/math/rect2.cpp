/**************************************************************************/
/*  rect2.cpp                                                             */
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

#include "rect2.h"

#include "core/math/rect2i.h"
#include "core/math/transform_2d.h"
#include "core/string/ustring.h"

bool Rect2::is_equal_approx(const Rect2 &p_rect) const {
	return position.is_equal_approx(p_rect.position) && size.is_equal_approx(p_rect.size);
}

bool Rect2::is_same(const Rect2 &p_rect) const {
	return position.is_same(p_rect.position) && size.is_same(p_rect.size);
}

bool Rect2::is_finite() const {
	return position.is_finite() && size.is_finite();
}

bool Rect2::intersects_segment(const Point2 &p_from, const Point2 &p_to, Point2 *r_pos, Point2 *r_normal) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0)) {
		ERR_PRINT("Rect2 size is negative, this is not supported. Use Rect2.abs() to get a Rect2 with a positive size.");
	}
#endif
	real_t min = 0, max = 1;
	int axis = 0;
	real_t sign = 0;

	for (int i = 0; i < 2; i++) {
		real_t seg_from = p_from[i];
		real_t seg_to = p_to[i];
		real_t box_begin = position[i];
		real_t box_end = box_begin + size[i];
		real_t cmin, cmax;
		real_t csign;

		if (seg_from < seg_to) {
			if (seg_from > box_end || seg_to < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from < box_begin) ? ((box_begin - seg_from) / length) : 0;
			cmax = (seg_to > box_end) ? ((box_end - seg_from) / length) : 1;
			csign = -1.0;

		} else {
			if (seg_to > box_end || seg_from < box_begin) {
				return false;
			}
			real_t length = seg_to - seg_from;
			cmin = (seg_from > box_end) ? (box_end - seg_from) / length : 0;
			cmax = (seg_to < box_begin) ? (box_begin - seg_from) / length : 1;
			csign = 1.0;
		}

		if (cmin > min) {
			min = cmin;
			axis = i;
			sign = csign;
		}
		if (cmax < max) {
			max = cmax;
		}
		if (max < min) {
			return false;
		}
	}

	Vector2 rel = p_to - p_from;

	if (r_normal) {
		Vector2 normal;
		normal[axis] = sign;
		*r_normal = normal;
	}

	if (r_pos) {
		*r_pos = p_from + rel * min;
	}

	return true;
}

bool Rect2::intersects_transformed(const Transform2D &p_xform, const Rect2 &p_rect) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || p_rect.size.x < 0 || p_rect.size.y < 0)) {
		ERR_PRINT("Rect2 size is negative, this is not supported. Use Rect2.abs() to get a Rect2 with a positive size.");
	}
#endif
	//SAT intersection between local and transformed rect2

	Vector2 xf_points[4] = {
		p_xform.xform(p_rect.position),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y)),
		p_xform.xform(Vector2(p_rect.position.x, p_rect.position.y + p_rect.size.y)),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y + p_rect.size.y)),
	};

	real_t low_limit;

	//base rect2 first (faster)

	if (xf_points[0].y > position.y) {
		goto next1;
	}
	if (xf_points[1].y > position.y) {
		goto next1;
	}
	if (xf_points[2].y > position.y) {
		goto next1;
	}
	if (xf_points[3].y > position.y) {
		goto next1;
	}

	return false;

next1:

	low_limit = position.y + size.y;

	if (xf_points[0].y < low_limit) {
		goto next2;
	}
	if (xf_points[1].y < low_limit) {
		goto next2;
	}
	if (xf_points[2].y < low_limit) {
		goto next2;
	}
	if (xf_points[3].y < low_limit) {
		goto next2;
	}

	return false;

next2:

	if (xf_points[0].x > position.x) {
		goto next3;
	}
	if (xf_points[1].x > position.x) {
		goto next3;
	}
	if (xf_points[2].x > position.x) {
		goto next3;
	}
	if (xf_points[3].x > position.x) {
		goto next3;
	}

	return false;

next3:

	low_limit = position.x + size.x;

	if (xf_points[0].x < low_limit) {
		goto next4;
	}
	if (xf_points[1].x < low_limit) {
		goto next4;
	}
	if (xf_points[2].x < low_limit) {
		goto next4;
	}
	if (xf_points[3].x < low_limit) {
		goto next4;
	}

	return false;

next4:

	Vector2 xf_points2[4] = {
		position,
		Vector2(position.x + size.x, position.y),
		Vector2(position.x, position.y + size.y),
		Vector2(position.x + size.x, position.y + size.y),
	};

	real_t maxa = p_xform.columns[0].dot(xf_points2[0]);
	real_t mina = maxa;

	real_t dp = p_xform.columns[0].dot(xf_points2[1]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[0].dot(xf_points2[2]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[0].dot(xf_points2[3]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	real_t maxb = p_xform.columns[0].dot(xf_points[0]);
	real_t minb = maxb;

	dp = p_xform.columns[0].dot(xf_points[1]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[0].dot(xf_points[2]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[0].dot(xf_points[3]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	if (mina > maxb) {
		return false;
	}
	if (minb > maxa) {
		return false;
	}

	maxa = p_xform.columns[1].dot(xf_points2[0]);
	mina = maxa;

	dp = p_xform.columns[1].dot(xf_points2[1]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[1].dot(xf_points2[2]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.columns[1].dot(xf_points2[3]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	maxb = p_xform.columns[1].dot(xf_points[0]);
	minb = maxb;

	dp = p_xform.columns[1].dot(xf_points[1]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[1].dot(xf_points[2]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.columns[1].dot(xf_points[3]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	if (mina > maxb) {
		return false;
	}
	if (minb > maxa) {
		return false;
	}

	return true;
}

Rect2 Rect2::intersection_transformed(const Transform2D &p_xform, const Rect2 &p_rect) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || p_rect.size.x < 0 || p_rect.size.y < 0)) {
		ERR_PRINT("Rect2 size is negative, this is not supported. Use Rect2.abs() to get a Rect2 with a positive size.");
	}
#endif

	if ((p_xform.columns[0].y == 0.0f && p_xform.columns[1].x == 0.0f) || (p_xform.columns[0].x == 0.0f && p_xform.columns[1].y == 0.0f)) {
		return intersection(p_xform.xform(p_rect));
	}

	Vector2 xf_points[4] = {
		p_xform.xform(p_rect.position),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y)),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y + p_rect.size.y)),
		p_xform.xform(Vector2(p_rect.position.x, p_rect.position.y + p_rect.size.y)),
	};

	const Point2 end = position + size;

	Vector2i flags[4]; // The points are distributed in a nine-square grid.
	Vector2i flag_changed_index; // Record the point index at which the flag changes for the first time.
	Vector<Point2> points;

	for (int idx = 0; idx < 5; idx++) {
		if (idx < 4) {
			const Point2 point = xf_points[idx];
			for (int axis = 0; axis < 2; axis++) {
				flags[idx][axis] = point[axis] < position[axis] ? -1 : (point[axis] > end[axis] ? 1 : 0);

				if (idx != 0 && flag_changed_index[axis] == 0 && flags[idx][axis] != flags[0][axis]) {
					flag_changed_index[axis] = idx;
				}
			}

			if (flags[idx] == Vector2i()) {
				points.push_back(point);
			}
			if (idx == 0) {
				continue;
			}
		}

		// Contains some points of p_rect.

		const int index = idx % 4;
		const int prev_idx = (index + 3) % 4;
		const bool inside = flags[index] == Vector2i();
		if ((flags[prev_idx] == Vector2i()) == inside) {
			continue;
		}

		const Vector2i outer_flag = inside ? flags[prev_idx] : flags[index];
		const Point2 outer = inside ? xf_points[prev_idx] : xf_points[index];
		const Point2 inner = inside ? xf_points[index] : xf_points[prev_idx];
		Point2 intersected = outer;

		for (int axis = 0; axis < 2; axis++) {
			if (outer_flag[axis] == 0) {
				axis++;
			}
			intersected[axis] = CLAMP(intersected[axis], position[axis], end[axis]);
			if (inner[axis] == intersected[axis]) {
				break; // No extra intersection points, as the inner point is on the closest border.
			}

			const int axis_1 = (axis + 1) % 2;
			intersected[axis_1] = outer[axis_1] + (intersected[axis] - outer[axis]) * (inner[axis_1] - outer[axis_1]) / (inner[axis] - outer[axis]);
			if (axis == 1 || (intersected[axis_1] >= position[axis_1] && intersected[axis_1] <= end[axis_1])) {
				points.push_back(intersected);
				break;
			}
		}
	}

	if (points.size()) {
		return Rect2(points.ptr(), points.size());
	}

	int axis = flag_changed_index.x == 0 ? 0 : (flag_changed_index.y == 0 ? 1 : -1);

	if (axis != -1) {
		if (flags[0][axis] != 0) {
			return Rect2(); // No intersection, since all points are on one side.
		}

		// The points are located on both sides of the middle.

		const int axis_1 = (axis + 1) % 2;
		int idx = flag_changed_index[axis_1];

		for (int line_count = 0; line_count < 2; line_count++) {
			const Point2 prev = xf_points[(idx + 3) % 4];
			const Point2 point = xf_points[idx];

			Point2 intersected;
			intersected[axis_1] = position[axis_1];
			intersected[axis] = point[axis] + (position[axis_1] - point[axis_1]) * (prev[axis] - point[axis]) / (prev[axis_1] - point[axis_1]);
			points.push_back(intersected);

			intersected[axis_1] = end[axis_1];
			intersected[axis] = point[axis] + (end[axis_1] - point[axis_1]) * (prev[axis] - point[axis]) / (prev[axis_1] - point[axis_1]);
			points.push_back(intersected);

			idx = flags[idx][axis_1] != flags[(idx + 1) % 4][axis_1] ? (idx + 1) % 4 : (idx + 2) % 4; // Adjacent or opposite.
		}

		return Rect2(points.ptr(), points.size());
	}

	for (int idx = MIN(flag_changed_index.x, flag_changed_index.y) - 1; idx < 4; idx++) {
		const int next_idx = (idx + 1) % 4;

		if ((flags[idx].x == flags[next_idx].x && flags[idx].x != 0) || (flags[idx].y == flags[next_idx].y && flags[idx].y != 0)) {
			continue;
		}

		const Point2 point = xf_points[idx];
		const Point2 next = xf_points[next_idx];

		for (axis = 0; axis < 2; axis++) {
			const int axis_1 = (axis + 1) % 2;
			const int flag_sum = flags[idx][axis] + flags[next_idx][axis];

			Point2 intersected;

			switch (flag_sum) {
				case -1:
				case 0: {
					intersected[axis] = position[axis];
					intersected[axis_1] = point[axis_1] + (intersected[axis] - point[axis]) * (next[axis_1] - point[axis_1]) / (next[axis] - point[axis]);
					if (intersected[axis_1] >= position[axis_1] && intersected[axis_1] <= end[axis_1]) {
						points.push_back(intersected);
					}
					if (flag_sum == -1) {
						break;
					}
					[[fallthrough]];
				}
				case 1: {
					intersected[axis] = end[axis];
					intersected[axis_1] = point[axis_1] + (intersected[axis] - point[axis]) * (next[axis_1] - point[axis_1]) / (next[axis] - point[axis]);
					if (intersected[axis_1] >= position[axis_1] && intersected[axis_1] <= end[axis_1]) {
						points.push_back(intersected);
					}
				} break;
			}
		}
	}
	if (points.size() > 2) {
		return Rect2(points.ptr(), points.size());
	}

	// At most one edge intersects.

	const Vector2 _points[4] = { position, Point2(end.x, position.y), end, Point2(position.x, end.y) };

	for (int point_idx = 0; point_idx < 4; point_idx++) {
		bool was_left = false;
		bool inside = true;
		for (int idx = 0; idx < 4; idx++) {
			const int next_idx = (idx + 1) % 4;
			Vector2 v0 = xf_points[next_idx] - xf_points[idx];
			Vector2 v1 = _points[point_idx] - xf_points[idx];

			bool is_left = v0.cross(v1) > 0;
			if (idx > 0 && is_left != was_left) {
				inside = false;
				break;
			}
			was_left = is_left;
		}
		if (inside) {
			points.push_back(_points[point_idx]);
		}
	}

	if (points.size()) {
		return Rect2(points.ptr(), points.size());
	}

	return Rect2();
}

Rect2::operator String() const {
	return "[P: " + position.operator String() + ", S: " + size.operator String() + "]";
}

Rect2::operator Rect2i() const {
	return Rect2i(position, size);
}
