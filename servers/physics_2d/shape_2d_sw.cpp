/*************************************************************************/
/*  shape_2d_sw.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "shape_2d_sw.h"

#include "core/math/geometry_2d.h"
#include "core/templates/sort_array.h"

void Shape2DSW::configure(const Rect2 &p_aabb) {
	aabb = p_aabb;
	configured = true;
	for (Map<ShapeOwner2DSW *, int>::Element *E = owners.front(); E; E = E->next()) {
		ShapeOwner2DSW *co = (ShapeOwner2DSW *)E->key();
		co->_shape_changed();
	}
}

Vector2 Shape2DSW::get_support(const Vector2 &p_normal) const {
	Vector2 res[2];
	int amnt;
	get_supports(p_normal, res, amnt);
	return res[0];
}

void Shape2DSW::add_owner(ShapeOwner2DSW *p_owner) {
	Map<ShapeOwner2DSW *, int>::Element *E = owners.find(p_owner);
	if (E) {
		E->get()++;
	} else {
		owners[p_owner] = 1;
	}
}

void Shape2DSW::remove_owner(ShapeOwner2DSW *p_owner) {
	Map<ShapeOwner2DSW *, int>::Element *E = owners.find(p_owner);
	ERR_FAIL_COND(!E);
	E->get()--;
	if (E->get() == 0) {
		owners.erase(E);
	}
}

bool Shape2DSW::is_owner(ShapeOwner2DSW *p_owner) const {
	return owners.has(p_owner);
}

const Map<ShapeOwner2DSW *, int> &Shape2DSW::get_owners() const {
	return owners;
}

Shape2DSW::Shape2DSW() {
	custom_bias = 0;
	configured = false;
}

Shape2DSW::~Shape2DSW() {
	ERR_FAIL_COND(owners.size());
}

/*********************************************************/
/*********************************************************/
/*********************************************************/

void LineShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	r_amount = 0;
}

bool LineShape2DSW::contains_point(const Vector2 &p_point) const {
	return normal.dot(p_point) < d;
}

bool LineShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	Vector2 segment = p_begin - p_end;
	real_t den = normal.dot(segment);

	//printf("den is %i\n",den);
	if (Math::abs(den) <= CMP_EPSILON) {
		return false;
	}

	real_t dist = (normal.dot(p_begin) - d) / den;
	//printf("dist is %i\n",dist);

	if (dist < -CMP_EPSILON || dist > (1.0 + CMP_EPSILON)) {
		return false;
	}

	r_point = p_begin + segment * -dist;
	r_normal = normal;

	return true;
}

real_t LineShape2DSW::get_moment_of_inertia(real_t p_mass, const Size2 &p_scale) const {
	return 0;
}

void LineShape2DSW::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::ARRAY);
	Array arr = p_data;
	ERR_FAIL_COND(arr.size() != 2);
	normal = arr[0];
	d = arr[1];
	configure(Rect2(Vector2(-1e4, -1e4), Vector2(1e4 * 2, 1e4 * 2)));
}

Variant LineShape2DSW::get_data() const {
	Array arr;
	arr.resize(2);
	arr[0] = normal;
	arr[1] = d;
	return arr;
}

/*********************************************************/
/*********************************************************/
/*********************************************************/

void RayShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	r_amount = 1;

	if (p_normal.y > 0) {
		*r_supports = Vector2(0, length);
	} else {
		*r_supports = Vector2();
	}
}

bool RayShape2DSW::contains_point(const Vector2 &p_point) const {
	return false;
}

bool RayShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	return false; //rays can't be intersected
}

real_t RayShape2DSW::get_moment_of_inertia(real_t p_mass, const Size2 &p_scale) const {
	return 0; //rays are mass-less
}

void RayShape2DSW::set_data(const Variant &p_data) {
	Dictionary d = p_data;
	length = d["length"];
	slips_on_slope = d["slips_on_slope"];
	configure(Rect2(0, 0, 0.001, length));
}

Variant RayShape2DSW::get_data() const {
	Dictionary d;
	d["length"] = length;
	d["slips_on_slope"] = slips_on_slope;
	return d;
}

/*********************************************************/
/*********************************************************/
/*********************************************************/

void SegmentShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	if (Math::abs(p_normal.dot(n)) > _SEGMENT_IS_VALID_SUPPORT_THRESHOLD) {
		r_supports[0] = a;
		r_supports[1] = b;
		r_amount = 2;
		return;
	}

	real_t dp = p_normal.dot(b - a);
	if (dp > 0) {
		*r_supports = b;
	} else {
		*r_supports = a;
	}
	r_amount = 1;
}

bool SegmentShape2DSW::contains_point(const Vector2 &p_point) const {
	return false;
}

bool SegmentShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	if (!Geometry2D::segment_intersects_segment(p_begin, p_end, a, b, &r_point)) {
		return false;
	}

	if (n.dot(p_begin) > n.dot(a)) {
		r_normal = n;
	} else {
		r_normal = -n;
	}

	return true;
}

real_t SegmentShape2DSW::get_moment_of_inertia(real_t p_mass, const Size2 &p_scale) const {
	return p_mass * ((a * p_scale).distance_squared_to(b * p_scale)) / 12;
}

void SegmentShape2DSW::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::RECT2);

	Rect2 r = p_data;
	a = r.position;
	b = r.size;
	n = (b - a).orthogonal();

	Rect2 aabb;
	aabb.position = a;
	aabb.expand_to(b);
	if (aabb.size.x == 0) {
		aabb.size.x = 0.001;
	}
	if (aabb.size.y == 0) {
		aabb.size.y = 0.001;
	}
	configure(aabb);
}

Variant SegmentShape2DSW::get_data() const {
	Rect2 r;
	r.position = a;
	r.size = b;
	return r;
}

/*********************************************************/
/*********************************************************/
/*********************************************************/

void CircleShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	r_amount = 1;
	*r_supports = p_normal * radius;
}

bool CircleShape2DSW::contains_point(const Vector2 &p_point) const {
	return p_point.length_squared() < radius * radius;
}

bool CircleShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	Vector2 line_vec = p_end - p_begin;

	real_t a, b, c;

	a = line_vec.dot(line_vec);
	b = 2 * p_begin.dot(line_vec);
	c = p_begin.dot(p_begin) - radius * radius;

	real_t sqrtterm = b * b - 4 * a * c;

	if (sqrtterm < 0) {
		return false;
	}
	sqrtterm = Math::sqrt(sqrtterm);
	real_t res = (-b - sqrtterm) / (2 * a);

	if (res < 0 || res > 1 + CMP_EPSILON) {
		return false;
	}

	r_point = p_begin + line_vec * res;
	r_normal = r_point.normalized();
	return true;
}

real_t CircleShape2DSW::get_moment_of_inertia(real_t p_mass, const Size2 &p_scale) const {
	real_t a = radius * p_scale.x;
	real_t b = radius * p_scale.y;
	return p_mass * (a * a + b * b) / 4;
}

void CircleShape2DSW::set_data(const Variant &p_data) {
	ERR_FAIL_COND(!p_data.is_num());
	radius = p_data;
	configure(Rect2(-radius, -radius, radius * 2, radius * 2));
}

Variant CircleShape2DSW::get_data() const {
	return radius;
}

/*********************************************************/
/*********************************************************/
/*********************************************************/

void RectangleShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	for (int i = 0; i < 2; i++) {
		Vector2 ag;
		ag[i] = 1.0;
		real_t dp = ag.dot(p_normal);
		if (Math::abs(dp) < _SEGMENT_IS_VALID_SUPPORT_THRESHOLD) {
			continue;
		}

		real_t sgn = dp > 0 ? 1.0 : -1.0;

		r_amount = 2;

		r_supports[0][i] = half_extents[i] * sgn;
		r_supports[0][i ^ 1] = half_extents[i ^ 1];

		r_supports[1][i] = half_extents[i] * sgn;
		r_supports[1][i ^ 1] = -half_extents[i ^ 1];

		return;
	}

	/* USE POINT */

	r_amount = 1;
	r_supports[0] = Vector2(
			(p_normal.x < 0) ? -half_extents.x : half_extents.x,
			(p_normal.y < 0) ? -half_extents.y : half_extents.y);
}

bool RectangleShape2DSW::contains_point(const Vector2 &p_point) const {
	real_t x = p_point.x;
	real_t y = p_point.y;
	real_t edge_x = half_extents.x;
	real_t edge_y = half_extents.y;
	return (x >= -edge_x) && (x < edge_x) && (y >= -edge_y) && (y < edge_y);
}

bool RectangleShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	return get_aabb().intersects_segment(p_begin, p_end, &r_point, &r_normal);
}

real_t RectangleShape2DSW::get_moment_of_inertia(real_t p_mass, const Size2 &p_scale) const {
	Vector2 he2 = half_extents * 2 * p_scale;
	return p_mass * he2.dot(he2) / 12.0;
}

void RectangleShape2DSW::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::VECTOR2);

	half_extents = p_data;
	configure(Rect2(-half_extents, half_extents * 2.0));
}

Variant RectangleShape2DSW::get_data() const {
	return half_extents;
}

/*********************************************************/
/*********************************************************/
/*********************************************************/

void CapsuleShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	Vector2 n = p_normal;

	real_t d = n.y;

	if (Math::abs(d) < (1.0 - _SEGMENT_IS_VALID_SUPPORT_THRESHOLD)) {
		// make it flat
		n.y = 0.0;
		n.normalize();
		n *= radius;

		r_amount = 2;
		r_supports[0] = n;
		r_supports[0].y += height * 0.5;
		r_supports[1] = n;
		r_supports[1].y -= height * 0.5;

	} else {
		real_t h = (d > 0) ? height : -height;

		n *= radius;
		n.y += h * 0.5;
		r_amount = 1;
		*r_supports = n;
	}
}

bool CapsuleShape2DSW::contains_point(const Vector2 &p_point) const {
	Vector2 p = p_point;
	p.y = Math::abs(p.y);
	p.y -= height * 0.5;
	if (p.y < 0) {
		p.y = 0;
	}

	return p.length_squared() < radius * radius;
}

bool CapsuleShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	real_t d = 1e10;
	Vector2 n = (p_end - p_begin).normalized();
	bool collided = false;

	//try spheres
	for (int i = 0; i < 2; i++) {
		Vector2 begin = p_begin;
		Vector2 end = p_end;
		real_t ofs = (i == 0) ? -height * 0.5 : height * 0.5;
		begin.y += ofs;
		end.y += ofs;

		Vector2 line_vec = end - begin;

		real_t a, b, c;

		a = line_vec.dot(line_vec);
		b = 2 * begin.dot(line_vec);
		c = begin.dot(begin) - radius * radius;

		real_t sqrtterm = b * b - 4 * a * c;

		if (sqrtterm < 0) {
			continue;
		}

		sqrtterm = Math::sqrt(sqrtterm);
		real_t res = (-b - sqrtterm) / (2 * a);

		if (res < 0 || res > 1 + CMP_EPSILON) {
			continue;
		}

		Vector2 point = begin + line_vec * res;
		Vector2 pointf(point.x, point.y - ofs);
		real_t pd = n.dot(pointf);
		if (pd < d) {
			r_point = pointf;
			r_normal = point.normalized();
			d = pd;
			collided = true;
		}
	}

	Vector2 rpos, rnorm;
	if (Rect2(Point2(-radius, -height * 0.5), Size2(radius * 2.0, height)).intersects_segment(p_begin, p_end, &rpos, &rnorm)) {
		real_t pd = n.dot(rpos);
		if (pd < d) {
			r_point = rpos;
			r_normal = rnorm;
			d = pd;
			collided = true;
		}
	}

	//return get_aabb().intersects_segment(p_begin,p_end,&r_point,&r_normal);
	return collided; //todo
}

real_t CapsuleShape2DSW::get_moment_of_inertia(real_t p_mass, const Size2 &p_scale) const {
	Vector2 he2 = Vector2(radius * 2, height + radius * 2) * p_scale;
	return p_mass * he2.dot(he2) / 12.0;
}

void CapsuleShape2DSW::set_data(const Variant &p_data) {
	ERR_FAIL_COND(p_data.get_type() != Variant::ARRAY && p_data.get_type() != Variant::VECTOR2);

	if (p_data.get_type() == Variant::ARRAY) {
		Array arr = p_data;
		ERR_FAIL_COND(arr.size() != 2);
		height = arr[0];
		radius = arr[1];
	} else {
		Point2 p = p_data;
		radius = p.x;
		height = p.y;
	}

	Point2 he(radius, height * 0.5 + radius);
	configure(Rect2(-he, he * 2));
}

Variant CapsuleShape2DSW::get_data() const {
	return Point2(height, radius);
}

/*********************************************************/
/*********************************************************/
/*********************************************************/

void ConvexPolygonShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	int support_idx = -1;
	real_t d = -1e10;
	r_amount = 0;

	for (int i = 0; i < point_count; i++) {
		//test point
		real_t ld = p_normal.dot(points[i].pos);
		if (ld > d) {
			support_idx = i;
			d = ld;
		}

		//test segment
		if (points[i].normal.dot(p_normal) > _SEGMENT_IS_VALID_SUPPORT_THRESHOLD) {
			r_amount = 2;
			r_supports[0] = points[i].pos;
			r_supports[1] = points[(i + 1) % point_count].pos;
			return;
		}
	}

	ERR_FAIL_COND_MSG(support_idx == -1, "Convex polygon shape support not found.");

	r_amount = 1;
	r_supports[0] = points[support_idx].pos;
}

bool ConvexPolygonShape2DSW::contains_point(const Vector2 &p_point) const {
	bool out = false;
	bool in = false;

	for (int i = 0; i < point_count; i++) {
		real_t d = points[i].normal.dot(p_point) - points[i].normal.dot(points[i].pos);
		if (d > 0) {
			out = true;
		} else {
			in = true;
		}
	}

	return in != out;
}

bool ConvexPolygonShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	Vector2 n = (p_end - p_begin).normalized();
	real_t d = 1e10;
	bool inters = false;

	for (int i = 0; i < point_count; i++) {
		//hmm.. no can do..
		/*
		if (d.dot(points[i].normal)>=0)
			continue;
		*/

		Vector2 res;

		if (!Geometry2D::segment_intersects_segment(p_begin, p_end, points[i].pos, points[(i + 1) % point_count].pos, &res)) {
			continue;
		}

		real_t nd = n.dot(res);
		if (nd < d) {
			d = nd;
			r_point = res;
			r_normal = points[i].normal;
			inters = true;
		}
	}

	if (inters) {
		if (n.dot(r_normal) > 0) {
			r_normal = -r_normal;
		}
	}

	//return get_aabb().intersects_segment(p_begin,p_end,&r_point,&r_normal);
	return inters; //todo
}

real_t ConvexPolygonShape2DSW::get_moment_of_inertia(real_t p_mass, const Size2 &p_scale) const {
	ERR_FAIL_COND_V_MSG(point_count == 0, 0, "Convex polygon shape has no points.");
	Rect2 aabb;
	aabb.position = points[0].pos * p_scale;
	for (int i = 0; i < point_count; i++) {
		aabb.expand_to(points[i].pos * p_scale);
	}

	return p_mass * aabb.size.dot(aabb.size) / 12.0;
}

void ConvexPolygonShape2DSW::set_data(const Variant &p_data) {
#ifdef REAL_T_IS_DOUBLE
	ERR_FAIL_COND(p_data.get_type() != Variant::PACKED_VECTOR2_ARRAY && p_data.get_type() != Variant::PACKED_FLOAT64_ARRAY);
#else
	ERR_FAIL_COND(p_data.get_type() != Variant::PACKED_VECTOR2_ARRAY && p_data.get_type() != Variant::PACKED_FLOAT32_ARRAY);
#endif

	if (points) {
		memdelete_arr(points);
	}
	points = nullptr;
	point_count = 0;

	if (p_data.get_type() == Variant::PACKED_VECTOR2_ARRAY) {
		Vector<Vector2> arr = p_data;
		ERR_FAIL_COND(arr.size() == 0);
		point_count = arr.size();
		points = memnew_arr(Point, point_count);
		const Vector2 *r = arr.ptr();

		for (int i = 0; i < point_count; i++) {
			points[i].pos = r[i];
		}

		for (int i = 0; i < point_count; i++) {
			Vector2 p = points[i].pos;
			Vector2 pn = points[(i + 1) % point_count].pos;
			points[i].normal = (pn - p).orthogonal().normalized();
		}
	} else {
		Vector<real_t> dvr = p_data;
		point_count = dvr.size() / 4;
		ERR_FAIL_COND(point_count == 0);

		points = memnew_arr(Point, point_count);
		const real_t *r = dvr.ptr();

		for (int i = 0; i < point_count; i++) {
			int idx = i << 2;
			points[i].pos.x = r[idx + 0];
			points[i].pos.y = r[idx + 1];
			points[i].normal.x = r[idx + 2];
			points[i].normal.y = r[idx + 3];
		}
	}

	ERR_FAIL_COND(point_count == 0);
	Rect2 aabb;
	aabb.position = points[0].pos;
	for (int i = 1; i < point_count; i++) {
		aabb.expand_to(points[i].pos);
	}

	configure(aabb);
}

Variant ConvexPolygonShape2DSW::get_data() const {
	Vector<Vector2> dvr;

	dvr.resize(point_count);

	for (int i = 0; i < point_count; i++) {
		dvr.set(i, points[i].pos);
	}

	return dvr;
}

ConvexPolygonShape2DSW::ConvexPolygonShape2DSW() {
	points = nullptr;
	point_count = 0;
}

ConvexPolygonShape2DSW::~ConvexPolygonShape2DSW() {
	if (points) {
		memdelete_arr(points);
	}
}

//////////////////////////////////////////////////

void ConcavePolygonShape2DSW::get_supports(const Vector2 &p_normal, Vector2 *r_supports, int &r_amount) const {
	real_t d = -1e10;
	int idx = -1;
	for (int i = 0; i < points.size(); i++) {
		real_t ld = p_normal.dot(points[i]);
		if (ld > d) {
			d = ld;
			idx = i;
		}
	}

	r_amount = 1;
	ERR_FAIL_COND(idx == -1);
	*r_supports = points[idx];
}

bool ConcavePolygonShape2DSW::contains_point(const Vector2 &p_point) const {
	return false; //sorry
}

bool ConcavePolygonShape2DSW::intersect_segment(const Vector2 &p_begin, const Vector2 &p_end, Vector2 &r_point, Vector2 &r_normal) const {
	if (segments.size() == 0 || points.size() == 0) {
		return false;
	}

	uint32_t *stack = (uint32_t *)alloca(sizeof(int) * bvh_depth);

	enum {
		TEST_AABB_BIT = 0,
		VISIT_LEFT_BIT = 1,
		VISIT_RIGHT_BIT = 2,
		VISIT_DONE_BIT = 3,
		VISITED_BIT_SHIFT = 29,
		NODE_IDX_MASK = (1 << VISITED_BIT_SHIFT) - 1,
		VISITED_BIT_MASK = ~NODE_IDX_MASK,

	};

	Vector2 n = (p_end - p_begin).normalized();
	real_t d = 1e10;
	bool inters = false;

	/*
	for(int i=0;i<bvh_depth;i++)
		stack[i]=0;
	*/

	int level = 0;

	const Segment *segmentptr = &segments[0];
	const Vector2 *pointptr = &points[0];
	const BVH *bvhptr = &bvh[0];

	stack[0] = 0;
	while (true) {
		uint32_t node = stack[level] & NODE_IDX_MASK;
		const BVH &bvh = bvhptr[node];
		bool done = false;

		switch (stack[level] >> VISITED_BIT_SHIFT) {
			case TEST_AABB_BIT: {
				bool valid = bvh.aabb.intersects_segment(p_begin, p_end);
				if (!valid) {
					stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;

				} else {
					if (bvh.left < 0) {
						const Segment &s = segmentptr[bvh.right];
						Vector2 a = pointptr[s.points[0]];
						Vector2 b = pointptr[s.points[1]];

						Vector2 res;

						if (Geometry2D::segment_intersects_segment(p_begin, p_end, a, b, &res)) {
							real_t nd = n.dot(res);
							if (nd < d) {
								d = nd;
								r_point = res;
								r_normal = (b - a).orthogonal().normalized();
								inters = true;
							}
						}

						stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;

					} else {
						stack[level] = (VISIT_LEFT_BIT << VISITED_BIT_SHIFT) | node;
					}
				}
			}
				continue;
			case VISIT_LEFT_BIT: {
				stack[level] = (VISIT_RIGHT_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = bvh.left | TEST_AABB_BIT;
				level++;
			}
				continue;
			case VISIT_RIGHT_BIT: {
				stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = bvh.right | TEST_AABB_BIT;
				level++;
			}
				continue;
			case VISIT_DONE_BIT: {
				if (level == 0) {
					done = true;
					break;
				} else {
					level--;
				}
			}
				continue;
		}

		if (done) {
			break;
		}
	}

	if (inters) {
		if (n.dot(r_normal) > 0) {
			r_normal = -r_normal;
		}
	}

	return inters;
}

int ConcavePolygonShape2DSW::_generate_bvh(BVH *p_bvh, int p_len, int p_depth) {
	if (p_len == 1) {
		bvh_depth = MAX(p_depth, bvh_depth);
		bvh.push_back(*p_bvh);
		return bvh.size() - 1;
	}

	//else sort best

	Rect2 global_aabb = p_bvh[0].aabb;
	for (int i = 1; i < p_len; i++) {
		global_aabb = global_aabb.merge(p_bvh[i].aabb);
	}

	if (global_aabb.size.x > global_aabb.size.y) {
		SortArray<BVH, BVH_CompareX> sort;
		sort.sort(p_bvh, p_len);

	} else {
		SortArray<BVH, BVH_CompareY> sort;
		sort.sort(p_bvh, p_len);
	}

	int median = p_len / 2;

	BVH node;
	node.aabb = global_aabb;
	int node_idx = bvh.size();
	bvh.push_back(node);

	int l = _generate_bvh(p_bvh, median, p_depth + 1);
	int r = _generate_bvh(&p_bvh[median], p_len - median, p_depth + 1);
	bvh.write[node_idx].left = l;
	bvh.write[node_idx].right = r;

	return node_idx;
}

void ConcavePolygonShape2DSW::set_data(const Variant &p_data) {
#ifdef REAL_T_IS_DOUBLE
	ERR_FAIL_COND(p_data.get_type() != Variant::PACKED_VECTOR2_ARRAY && p_data.get_type() != Variant::PACKED_FLOAT64_ARRAY);
#else
	ERR_FAIL_COND(p_data.get_type() != Variant::PACKED_VECTOR2_ARRAY && p_data.get_type() != Variant::PACKED_FLOAT32_ARRAY);
#endif

	Rect2 aabb;

	if (p_data.get_type() == Variant::PACKED_VECTOR2_ARRAY) {
		Vector<Vector2> p2arr = p_data;
		int len = p2arr.size();
		ERR_FAIL_COND(len % 2);

		segments.clear();
		points.clear();
		bvh.clear();
		bvh_depth = 1;

		if (len == 0) {
			configure(aabb);
			return;
		}

		const Vector2 *arr = p2arr.ptr();

		Map<Point2, int> pointmap;
		for (int i = 0; i < len; i += 2) {
			Point2 p1 = arr[i];
			Point2 p2 = arr[i + 1];
			int idx_p1, idx_p2;

			if (pointmap.has(p1)) {
				idx_p1 = pointmap[p1];
			} else {
				idx_p1 = pointmap.size();
				pointmap[p1] = idx_p1;
			}

			if (pointmap.has(p2)) {
				idx_p2 = pointmap[p2];
			} else {
				idx_p2 = pointmap.size();
				pointmap[p2] = idx_p2;
			}

			Segment s;
			s.points[0] = idx_p1;
			s.points[1] = idx_p2;
			segments.push_back(s);
		}

		points.resize(pointmap.size());
		aabb.position = pointmap.front()->key();
		for (Map<Point2, int>::Element *E = pointmap.front(); E; E = E->next()) {
			aabb.expand_to(E->key());
			points.write[E->get()] = E->key();
		}

		Vector<BVH> main_vbh;
		main_vbh.resize(segments.size());
		for (int i = 0; i < main_vbh.size(); i++) {
			main_vbh.write[i].aabb.position = points[segments[i].points[0]];
			main_vbh.write[i].aabb.expand_to(points[segments[i].points[1]]);
			main_vbh.write[i].left = -1;
			main_vbh.write[i].right = i;
		}

		_generate_bvh(main_vbh.ptrw(), main_vbh.size(), 1);

	} else {
		//dictionary with arrays
	}

	configure(aabb);
}

Variant ConcavePolygonShape2DSW::get_data() const {
	Vector<Vector2> rsegments;
	int len = segments.size();
	rsegments.resize(len * 2);
	Vector2 *w = rsegments.ptrw();
	for (int i = 0; i < len; i++) {
		w[(i << 1) + 0] = points[segments[i].points[0]];
		w[(i << 1) + 1] = points[segments[i].points[1]];
	}

	return rsegments;
}

void ConcavePolygonShape2DSW::cull(const Rect2 &p_local_aabb, Callback p_callback, void *p_userdata) const {
	uint32_t *stack = (uint32_t *)alloca(sizeof(int) * bvh_depth);

	enum {
		TEST_AABB_BIT = 0,
		VISIT_LEFT_BIT = 1,
		VISIT_RIGHT_BIT = 2,
		VISIT_DONE_BIT = 3,
		VISITED_BIT_SHIFT = 29,
		NODE_IDX_MASK = (1 << VISITED_BIT_SHIFT) - 1,
		VISITED_BIT_MASK = ~NODE_IDX_MASK,

	};

	/*
	for(int i=0;i<bvh_depth;i++)
		stack[i]=0;
	*/

	if (segments.size() == 0 || points.size() == 0 || bvh.size() == 0) {
		return;
	}

	int level = 0;

	const Segment *segmentptr = &segments[0];
	const Vector2 *pointptr = &points[0];
	const BVH *bvhptr = &bvh[0];

	stack[0] = 0;
	while (true) {
		uint32_t node = stack[level] & NODE_IDX_MASK;
		const BVH &bvh = bvhptr[node];

		switch (stack[level] >> VISITED_BIT_SHIFT) {
			case TEST_AABB_BIT: {
				bool valid = p_local_aabb.intersects(bvh.aabb);
				if (!valid) {
					stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;

				} else {
					if (bvh.left < 0) {
						const Segment &s = segmentptr[bvh.right];
						Vector2 a = pointptr[s.points[0]];
						Vector2 b = pointptr[s.points[1]];

						SegmentShape2DSW ss(a, b, (b - a).orthogonal().normalized());

						p_callback(p_userdata, &ss);
						stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;

					} else {
						stack[level] = (VISIT_LEFT_BIT << VISITED_BIT_SHIFT) | node;
					}
				}
			}
				continue;
			case VISIT_LEFT_BIT: {
				stack[level] = (VISIT_RIGHT_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = bvh.left | TEST_AABB_BIT;
				level++;
			}
				continue;
			case VISIT_RIGHT_BIT: {
				stack[level] = (VISIT_DONE_BIT << VISITED_BIT_SHIFT) | node;
				stack[level + 1] = bvh.right | TEST_AABB_BIT;
				level++;
			}
				continue;
			case VISIT_DONE_BIT: {
				if (level == 0) {
					return;
				} else {
					level--;
				}
			}
				continue;
		}
	}
}
