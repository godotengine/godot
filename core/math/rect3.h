/*************************************************************************/
/*  rect3.h                                                              */
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
#ifndef AABB_H
#define AABB_H

#include "math_defs.h"
#include "plane.h"
#include "vector3.h"

/**
 * AABB / AABB (Axis Aligned Bounding Box)
 * This is implemented by a point (pos) and the box size
 */

class Rect3 {
public:
	Vector3 pos;
	Vector3 size;

	real_t get_area() const; /// get area
	_FORCE_INLINE_ bool has_no_area() const {

		return (size.x <= CMP_EPSILON || size.y <= CMP_EPSILON || size.z <= CMP_EPSILON);
	}

	_FORCE_INLINE_ bool has_no_surface() const {

		return (size.x <= CMP_EPSILON && size.y <= CMP_EPSILON && size.z <= CMP_EPSILON);
	}

	const Vector3 &get_pos() const { return pos; }
	void set_pos(const Vector3 &p_pos) { pos = p_pos; }
	const Vector3 &get_size() const { return size; }
	void set_size(const Vector3 &p_size) { size = p_size; }

	bool operator==(const Rect3 &p_rval) const;
	bool operator!=(const Rect3 &p_rval) const;

	_FORCE_INLINE_ bool intersects(const Rect3 &p_aabb) const; /// Both AABBs overlap
	_FORCE_INLINE_ bool intersects_inclusive(const Rect3 &p_aabb) const; /// Both AABBs (or their faces) overlap
	_FORCE_INLINE_ bool encloses(const Rect3 &p_aabb) const; /// p_aabb is completely inside this

	Rect3 merge(const Rect3 &p_with) const;
	void merge_with(const Rect3 &p_aabb); ///merge with another AABB
	Rect3 intersection(const Rect3 &p_aabb) const; ///get box where two intersect, empty if no intersection occurs
	bool intersects_segment(const Vector3 &p_from, const Vector3 &p_to, Vector3 *r_clip = NULL, Vector3 *r_normal = NULL) const;
	bool intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *r_clip = NULL, Vector3 *r_normal = NULL) const;
	_FORCE_INLINE_ bool smits_intersect_ray(const Vector3 &from, const Vector3 &p_dir, real_t t0, real_t t1) const;

	_FORCE_INLINE_ bool intersects_convex_shape(const Plane *p_plane, int p_plane_count) const;
	bool intersects_plane(const Plane &p_plane) const;

	_FORCE_INLINE_ bool has_point(const Vector3 &p_point) const;
	_FORCE_INLINE_ Vector3 get_support(const Vector3 &p_normal) const;

	Vector3 get_longest_axis() const;
	int get_longest_axis_index() const;
	_FORCE_INLINE_ real_t get_longest_axis_size() const;

	Vector3 get_shortest_axis() const;
	int get_shortest_axis_index() const;
	_FORCE_INLINE_ real_t get_shortest_axis_size() const;

	Rect3 grow(real_t p_by) const;
	_FORCE_INLINE_ void grow_by(real_t p_amount);

	void get_edge(int p_edge, Vector3 &r_from, Vector3 &r_to) const;
	_FORCE_INLINE_ Vector3 get_endpoint(int p_point) const;

	Rect3 expand(const Vector3 &p_vector) const;
	_FORCE_INLINE_ void project_range_in_plane(const Plane &p_plane, real_t &r_min, real_t &r_max) const;
	_FORCE_INLINE_ void expand_to(const Vector3 &p_vector); /** expand to contain a point if necessary */

	operator String() const;

	_FORCE_INLINE_ Rect3() {}
	inline Rect3(const Vector3 &p_pos, const Vector3 &p_size) {
		pos = p_pos;
		size = p_size;
	}
};

inline bool Rect3::intersects(const Rect3 &p_aabb) const {

	if (pos.x >= (p_aabb.pos.x + p_aabb.size.x))
		return false;
	if ((pos.x + size.x) <= p_aabb.pos.x)
		return false;
	if (pos.y >= (p_aabb.pos.y + p_aabb.size.y))
		return false;
	if ((pos.y + size.y) <= p_aabb.pos.y)
		return false;
	if (pos.z >= (p_aabb.pos.z + p_aabb.size.z))
		return false;
	if ((pos.z + size.z) <= p_aabb.pos.z)
		return false;

	return true;
}

inline bool Rect3::intersects_inclusive(const Rect3 &p_aabb) const {

	if (pos.x > (p_aabb.pos.x + p_aabb.size.x))
		return false;
	if ((pos.x + size.x) < p_aabb.pos.x)
		return false;
	if (pos.y > (p_aabb.pos.y + p_aabb.size.y))
		return false;
	if ((pos.y + size.y) < p_aabb.pos.y)
		return false;
	if (pos.z > (p_aabb.pos.z + p_aabb.size.z))
		return false;
	if ((pos.z + size.z) < p_aabb.pos.z)
		return false;

	return true;
}

inline bool Rect3::encloses(const Rect3 &p_aabb) const {

	Vector3 src_min = pos;
	Vector3 src_max = pos + size;
	Vector3 dst_min = p_aabb.pos;
	Vector3 dst_max = p_aabb.pos + p_aabb.size;

	return (
			(src_min.x <= dst_min.x) &&
			(src_max.x > dst_max.x) &&
			(src_min.y <= dst_min.y) &&
			(src_max.y > dst_max.y) &&
			(src_min.z <= dst_min.z) &&
			(src_max.z > dst_max.z));
}

Vector3 Rect3::get_support(const Vector3 &p_normal) const {

	Vector3 half_extents = size * 0.5;
	Vector3 ofs = pos + half_extents;

	return Vector3(
				   (p_normal.x > 0) ? -half_extents.x : half_extents.x,
				   (p_normal.y > 0) ? -half_extents.y : half_extents.y,
				   (p_normal.z > 0) ? -half_extents.z : half_extents.z) +
		   ofs;
}

Vector3 Rect3::get_endpoint(int p_point) const {

	switch (p_point) {
		case 0: return Vector3(pos.x, pos.y, pos.z);
		case 1: return Vector3(pos.x, pos.y, pos.z + size.z);
		case 2: return Vector3(pos.x, pos.y + size.y, pos.z);
		case 3: return Vector3(pos.x, pos.y + size.y, pos.z + size.z);
		case 4: return Vector3(pos.x + size.x, pos.y, pos.z);
		case 5: return Vector3(pos.x + size.x, pos.y, pos.z + size.z);
		case 6: return Vector3(pos.x + size.x, pos.y + size.y, pos.z);
		case 7: return Vector3(pos.x + size.x, pos.y + size.y, pos.z + size.z);
	};

	ERR_FAIL_V(Vector3());
}

bool Rect3::intersects_convex_shape(const Plane *p_planes, int p_plane_count) const {

#if 1

	Vector3 half_extents = size * 0.5;
	Vector3 ofs = pos + half_extents;

	for (int i = 0; i < p_plane_count; i++) {
		const Plane &p = p_planes[i];
		Vector3 point(
				(p.normal.x > 0) ? -half_extents.x : half_extents.x,
				(p.normal.y > 0) ? -half_extents.y : half_extents.y,
				(p.normal.z > 0) ? -half_extents.z : half_extents.z);
		point += ofs;
		if (p.is_point_over(point))
			return false;
	}

	return true;
#else
	//cache all points to check against!
	// #warning should be easy to optimize, just use the same as when taking the support and use only that point
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

	for (int i = 0; i < p_plane_count; i++) { //for each plane

		const Plane &plane = p_planes[i];
		bool all_points_over = true;
		//test if it has all points over!

		for (int j = 0; j < 8; j++) {

			if (!plane.is_point_over(points[j])) {

				all_points_over = false;
				break;
			}
		}

		if (all_points_over) {

			return false;
		}
	}
	return true;
#endif
}

bool Rect3::has_point(const Vector3 &p_point) const {

	if (p_point.x < pos.x)
		return false;
	if (p_point.y < pos.y)
		return false;
	if (p_point.z < pos.z)
		return false;
	if (p_point.x > pos.x + size.x)
		return false;
	if (p_point.y > pos.y + size.y)
		return false;
	if (p_point.z > pos.z + size.z)
		return false;

	return true;
}

inline void Rect3::expand_to(const Vector3 &p_vector) {

	Vector3 begin = pos;
	Vector3 end = pos + size;

	if (p_vector.x < begin.x)
		begin.x = p_vector.x;
	if (p_vector.y < begin.y)
		begin.y = p_vector.y;
	if (p_vector.z < begin.z)
		begin.z = p_vector.z;

	if (p_vector.x > end.x)
		end.x = p_vector.x;
	if (p_vector.y > end.y)
		end.y = p_vector.y;
	if (p_vector.z > end.z)
		end.z = p_vector.z;

	pos = begin;
	size = end - begin;
}

void Rect3::project_range_in_plane(const Plane &p_plane, real_t &r_min, real_t &r_max) const {

	Vector3 half_extents(size.x * 0.5, size.y * 0.5, size.z * 0.5);
	Vector3 center(pos.x + half_extents.x, pos.y + half_extents.y, pos.z + half_extents.z);

	real_t length = p_plane.normal.abs().dot(half_extents);
	real_t distance = p_plane.distance_to(center);
	r_min = distance - length;
	r_max = distance + length;
}

inline real_t Rect3::get_longest_axis_size() const {

	real_t max_size = size.x;

	if (size.y > max_size) {
		max_size = size.y;
	}

	if (size.z > max_size) {
		max_size = size.z;
	}

	return max_size;
}

inline real_t Rect3::get_shortest_axis_size() const {

	real_t max_size = size.x;

	if (size.y < max_size) {
		max_size = size.y;
	}

	if (size.z < max_size) {
		max_size = size.z;
	}

	return max_size;
}

bool Rect3::smits_intersect_ray(const Vector3 &from, const Vector3 &dir, real_t t0, real_t t1) const {

	real_t divx = 1.0 / dir.x;
	real_t divy = 1.0 / dir.y;
	real_t divz = 1.0 / dir.z;

	Vector3 upbound = pos + size;
	real_t tmin, tmax, tymin, tymax, tzmin, tzmax;
	if (dir.x >= 0) {
		tmin = (pos.x - from.x) * divx;
		tmax = (upbound.x - from.x) * divx;
	} else {
		tmin = (upbound.x - from.x) * divx;
		tmax = (pos.x - from.x) * divx;
	}
	if (dir.y >= 0) {
		tymin = (pos.y - from.y) * divy;
		tymax = (upbound.y - from.y) * divy;
	} else {
		tymin = (upbound.y - from.y) * divy;
		tymax = (pos.y - from.y) * divy;
	}
	if ((tmin > tymax) || (tymin > tmax))
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	if (dir.z >= 0) {
		tzmin = (pos.z - from.z) * divz;
		tzmax = (upbound.z - from.z) * divz;
	} else {
		tzmin = (upbound.z - from.z) * divz;
		tzmax = (pos.z - from.z) * divz;
	}
	if ((tmin > tzmax) || (tzmin > tmax))
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	return ((tmin < t1) && (tmax > t0));
}

void Rect3::grow_by(real_t p_amount) {

	pos.x -= p_amount;
	pos.y -= p_amount;
	pos.z -= p_amount;
	size.x += 2.0 * p_amount;
	size.y += 2.0 * p_amount;
	size.z += 2.0 * p_amount;
}

#endif // AABB_H
