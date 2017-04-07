/*************************************************************************/
/*  rect3.cpp                                                            */
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
#include "rect3.h"

#include "print_string.h"

real_t Rect3::get_area() const {

	return size.x * size.y * size.z;
}

bool Rect3::operator==(const Rect3 &p_rval) const {

	return ((pos == p_rval.pos) && (size == p_rval.size));
}
bool Rect3::operator!=(const Rect3 &p_rval) const {

	return ((pos != p_rval.pos) || (size != p_rval.size));
}

void Rect3::merge_with(const Rect3 &p_aabb) {

	Vector3 beg_1, beg_2;
	Vector3 end_1, end_2;
	Vector3 min, max;

	beg_1 = pos;
	beg_2 = p_aabb.pos;
	end_1 = Vector3(size.x, size.y, size.z) + beg_1;
	end_2 = Vector3(p_aabb.size.x, p_aabb.size.y, p_aabb.size.z) + beg_2;

	min.x = (beg_1.x < beg_2.x) ? beg_1.x : beg_2.x;
	min.y = (beg_1.y < beg_2.y) ? beg_1.y : beg_2.y;
	min.z = (beg_1.z < beg_2.z) ? beg_1.z : beg_2.z;

	max.x = (end_1.x > end_2.x) ? end_1.x : end_2.x;
	max.y = (end_1.y > end_2.y) ? end_1.y : end_2.y;
	max.z = (end_1.z > end_2.z) ? end_1.z : end_2.z;

	pos = min;
	size = max - min;
}

Rect3 Rect3::intersection(const Rect3 &p_aabb) const {

	Vector3 src_min = pos;
	Vector3 src_max = pos + size;
	Vector3 dst_min = p_aabb.pos;
	Vector3 dst_max = p_aabb.pos + p_aabb.size;

	Vector3 min, max;

	if (src_min.x > dst_max.x || src_max.x < dst_min.x)
		return Rect3();
	else {

		min.x = (src_min.x > dst_min.x) ? src_min.x : dst_min.x;
		max.x = (src_max.x < dst_max.x) ? src_max.x : dst_max.x;
	}

	if (src_min.y > dst_max.y || src_max.y < dst_min.y)
		return Rect3();
	else {

		min.y = (src_min.y > dst_min.y) ? src_min.y : dst_min.y;
		max.y = (src_max.y < dst_max.y) ? src_max.y : dst_max.y;
	}

	if (src_min.z > dst_max.z || src_max.z < dst_min.z)
		return Rect3();
	else {

		min.z = (src_min.z > dst_min.z) ? src_min.z : dst_min.z;
		max.z = (src_max.z < dst_max.z) ? src_max.z : dst_max.z;
	}

	return Rect3(min, max - min);
}

bool Rect3::intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *r_clip, Vector3 *r_normal) const {

	Vector3 c1, c2;
	Vector3 end = pos + size;
	real_t near = -1e20;
	real_t far = 1e20;
	int axis = 0;

	for (int i = 0; i < 3; i++) {
		if (p_dir[i] == 0) {
			if ((p_from[i] < pos[i]) || (p_from[i] > end[i])) {
				return false;
			}
		} else { // ray not parallel to planes in this direction
			c1[i] = (pos[i] - p_from[i]) / p_dir[i];
			c2[i] = (end[i] - p_from[i]) / p_dir[i];

			if (c1[i] > c2[i]) {
				SWAP(c1, c2);
			}
			if (c1[i] > near) {
				near = c1[i];
				axis = i;
			}
			if (c2[i] < far) {
				far = c2[i];
			}
			if ((near > far) || (far < 0)) {
				return false;
			}
		}
	}

	if (r_clip)
		*r_clip = c1;
	if (r_normal) {
		*r_normal = Vector3();
		(*r_normal)[axis] = p_dir[axis] ? -1 : 1;
	}

	return true;
}

bool Rect3::intersects_segment(const Vector3 &p_from, const Vector3 &p_to, Vector3 *r_clip, Vector3 *r_normal) const {

	real_t min = 0, max = 1;
	int axis = 0;
	real_t sign = 0;

	for (int i = 0; i < 3; i++) {
		real_t seg_from = p_from[i];
		real_t seg_to = p_to[i];
		real_t box_begin = pos[i];
		real_t box_end = box_begin + size[i];
		real_t cmin, cmax;
		real_t csign;

		if (seg_from < seg_to) {

			if (seg_from > box_end || seg_to < box_begin)
				return false;
			real_t length = seg_to - seg_from;
			cmin = (seg_from < box_begin) ? ((box_begin - seg_from) / length) : 0;
			cmax = (seg_to > box_end) ? ((box_end - seg_from) / length) : 1;
			csign = -1.0;

		} else {

			if (seg_to > box_end || seg_from < box_begin)
				return false;
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
		if (cmax < max)
			max = cmax;
		if (max < min)
			return false;
	}

	Vector3 rel = p_to - p_from;

	if (r_normal) {
		Vector3 normal;
		normal[axis] = sign;
		*r_normal = normal;
	}

	if (r_clip)
		*r_clip = p_from + rel * min;

	return true;
}

bool Rect3::intersects_plane(const Plane &p_plane) const {

	Vector3 points[8] = {
		Vector3(pos.x, pos.y, pos.z),
		Vector3(pos.x, pos.y, pos.z + size.z),
		Vector3(pos.x, pos.y + size.y, pos.z),
		Vector3(pos.x, pos.y + size.y, pos.z + size.z),
		Vector3(pos.x + size.x, pos.y, pos.z),
		Vector3(pos.x + size.x, pos.y, pos.z + size.z),
		Vector3(pos.x + size.x, pos.y + size.y, pos.z),
		Vector3(pos.x + size.x, pos.y + size.y, pos.z + size.z),
	};

	bool over = false;
	bool under = false;

	for (int i = 0; i < 8; i++) {

		if (p_plane.distance_to(points[i]) > 0)
			over = true;
		else
			under = true;
	}

	return under && over;
}

Vector3 Rect3::get_longest_axis() const {

	Vector3 axis(1, 0, 0);
	real_t max_size = size.x;

	if (size.y > max_size) {
		axis = Vector3(0, 1, 0);
		max_size = size.y;
	}

	if (size.z > max_size) {
		axis = Vector3(0, 0, 1);
		max_size = size.z;
	}

	return axis;
}
int Rect3::get_longest_axis_index() const {

	int axis = 0;
	real_t max_size = size.x;

	if (size.y > max_size) {
		axis = 1;
		max_size = size.y;
	}

	if (size.z > max_size) {
		axis = 2;
		max_size = size.z;
	}

	return axis;
}

Vector3 Rect3::get_shortest_axis() const {

	Vector3 axis(1, 0, 0);
	real_t max_size = size.x;

	if (size.y < max_size) {
		axis = Vector3(0, 1, 0);
		max_size = size.y;
	}

	if (size.z < max_size) {
		axis = Vector3(0, 0, 1);
		max_size = size.z;
	}

	return axis;
}
int Rect3::get_shortest_axis_index() const {

	int axis = 0;
	real_t max_size = size.x;

	if (size.y < max_size) {
		axis = 1;
		max_size = size.y;
	}

	if (size.z < max_size) {
		axis = 2;
		max_size = size.z;
	}

	return axis;
}

Rect3 Rect3::merge(const Rect3 &p_with) const {

	Rect3 aabb = *this;
	aabb.merge_with(p_with);
	return aabb;
}
Rect3 Rect3::expand(const Vector3 &p_vector) const {
	Rect3 aabb = *this;
	aabb.expand_to(p_vector);
	return aabb;
}
Rect3 Rect3::grow(real_t p_by) const {

	Rect3 aabb = *this;
	aabb.grow_by(p_by);
	return aabb;
}

void Rect3::get_edge(int p_edge, Vector3 &r_from, Vector3 &r_to) const {

	ERR_FAIL_INDEX(p_edge, 12);
	switch (p_edge) {

		case 0: {

			r_from = Vector3(pos.x + size.x, pos.y, pos.z);
			r_to = Vector3(pos.x, pos.y, pos.z);
		} break;
		case 1: {

			r_from = Vector3(pos.x + size.x, pos.y, pos.z + size.z);
			r_to = Vector3(pos.x + size.x, pos.y, pos.z);
		} break;
		case 2: {
			r_from = Vector3(pos.x, pos.y, pos.z + size.z);
			r_to = Vector3(pos.x + size.x, pos.y, pos.z + size.z);

		} break;
		case 3: {

			r_from = Vector3(pos.x, pos.y, pos.z);
			r_to = Vector3(pos.x, pos.y, pos.z + size.z);

		} break;
		case 4: {

			r_from = Vector3(pos.x, pos.y + size.y, pos.z);
			r_to = Vector3(pos.x + size.x, pos.y + size.y, pos.z);
		} break;
		case 5: {

			r_from = Vector3(pos.x + size.x, pos.y + size.y, pos.z);
			r_to = Vector3(pos.x + size.x, pos.y + size.y, pos.z + size.z);
		} break;
		case 6: {
			r_from = Vector3(pos.x + size.x, pos.y + size.y, pos.z + size.z);
			r_to = Vector3(pos.x, pos.y + size.y, pos.z + size.z);

		} break;
		case 7: {

			r_from = Vector3(pos.x, pos.y + size.y, pos.z + size.z);
			r_to = Vector3(pos.x, pos.y + size.y, pos.z);

		} break;
		case 8: {

			r_from = Vector3(pos.x, pos.y, pos.z + size.z);
			r_to = Vector3(pos.x, pos.y + size.y, pos.z + size.z);

		} break;
		case 9: {

			r_from = Vector3(pos.x, pos.y, pos.z);
			r_to = Vector3(pos.x, pos.y + size.y, pos.z);

		} break;
		case 10: {

			r_from = Vector3(pos.x + size.x, pos.y, pos.z);
			r_to = Vector3(pos.x + size.x, pos.y + size.y, pos.z);

		} break;
		case 11: {

			r_from = Vector3(pos.x + size.x, pos.y, pos.z + size.z);
			r_to = Vector3(pos.x + size.x, pos.y + size.y, pos.z + size.z);

		} break;
	}
}

Rect3::operator String() const {

	return String() + pos + " - " + size;
}
