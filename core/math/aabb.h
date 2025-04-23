/**************************************************************************/
/*  aabb.h                                                                */
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

#include "core/math/plane.h"
#include "core/math/vector3.h"

/**
 * AABB (Axis Aligned Bounding Box)
 * This is implemented by a point (position) and the box size.
 */

struct [[nodiscard]] AABB {
	Vector3 position;
	Vector3 size;

	_FORCE_INLINE_ constexpr real_t get_volume() const;
	_FORCE_INLINE_ constexpr bool has_volume() const {
		return size.x > 0.0f && size.y > 0.0f && size.z > 0.0f;
	}

	_FORCE_INLINE_ constexpr bool has_surface() const {
		return size.x > 0.0f || size.y > 0.0f || size.z > 0.0f;
	}

	_FORCE_INLINE_ constexpr const Vector3 &get_position() const { return position; }
	_FORCE_INLINE_ constexpr void set_position(const Vector3 &p_pos) { position = p_pos; }
	_FORCE_INLINE_ constexpr const Vector3 &get_size() const { return size; }
	_FORCE_INLINE_ constexpr void set_size(const Vector3 &p_size) { size = p_size; }

	_FORCE_INLINE_ constexpr bool operator==(const AABB &p_rval) const {
		return position == p_rval.position && size == p_rval.size;
	}
	_FORCE_INLINE_ constexpr bool operator!=(const AABB &p_rval) const {
		return position != p_rval.position || size != p_rval.size;
	}

	_FORCE_INLINE_ bool is_equal_approx(const AABB &p_aabb) const;
	_FORCE_INLINE_ bool is_same(const AABB &p_aabb) const;
	_FORCE_INLINE_ bool is_finite() const;
	_FORCE_INLINE_ bool intersects(const AABB &p_aabb) const; /// Both AABBs overlap
	_FORCE_INLINE_ bool intersects_inclusive(const AABB &p_aabb) const; /// Both AABBs (or their faces) overlap
	_FORCE_INLINE_ bool encloses(const AABB &p_aabb) const; /// p_aabb is completely inside this

	_FORCE_INLINE_ AABB merge(const AABB &p_with) const;
	_FORCE_INLINE_ void merge_with(const AABB &p_aabb); ///merge with another AABB
	_FORCE_INLINE_ AABB intersection(const AABB &p_aabb) const; ///get box where two intersect, empty if no intersection occurs
	_FORCE_INLINE_ bool smits_intersect_ray(const Vector3 &p_from, const Vector3 &p_dir, real_t p_t0, real_t p_t1) const;

	_FORCE_INLINE_ bool intersects_segment(const Vector3 &p_from, const Vector3 &p_to, Vector3 *r_intersection_point = nullptr, Vector3 *r_normal = nullptr) const;
	_FORCE_INLINE_ bool intersects_ray(const Vector3 &p_from, const Vector3 &p_dir) const {
		bool inside;
		return find_intersects_ray(p_from, p_dir, inside);
	}
	_FORCE_INLINE_ bool find_intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, bool &r_inside, Vector3 *r_intersection_point = nullptr, Vector3 *r_normal = nullptr) const;

	_FORCE_INLINE_ bool intersects_convex_shape(const Plane *p_planes, int p_plane_count, const Vector3 *p_points, int p_point_count) const;
	_FORCE_INLINE_ bool inside_convex_shape(const Plane *p_planes, int p_plane_count) const;
	_FORCE_INLINE_ bool intersects_plane(const Plane &p_plane) const;

	_FORCE_INLINE_ bool has_point(const Vector3 &p_point) const;
	_FORCE_INLINE_ Vector3 get_support(const Vector3 &p_direction) const;

	_FORCE_INLINE_ Vector3 get_longest_axis() const;
	_FORCE_INLINE_ int get_longest_axis_index() const;
	_FORCE_INLINE_ real_t get_longest_axis_size() const;

	_FORCE_INLINE_ Vector3 get_shortest_axis() const;
	_FORCE_INLINE_ int get_shortest_axis_index() const;
	_FORCE_INLINE_ real_t get_shortest_axis_size() const;

	_FORCE_INLINE_ AABB grow(real_t p_by) const;
	_FORCE_INLINE_ void grow_by(real_t p_amount);

	_FORCE_INLINE_ void get_edge(int p_edge, Vector3 &r_from, Vector3 &r_to) const;
	_FORCE_INLINE_ Vector3 get_endpoint(int p_point) const;

	_FORCE_INLINE_ AABB expand(const Vector3 &p_vector) const;
	_FORCE_INLINE_ void project_range_in_plane(const Plane &p_plane, real_t &r_min, real_t &r_max) const;
	_FORCE_INLINE_ void expand_to(const Vector3 &p_vector); /** expand to contain a point if necessary */

	_FORCE_INLINE_ AABB abs() const {
		return AABB(position + size.minf(0), size.abs());
	}

	Variant intersects_segment_bind(const Vector3 &p_from, const Vector3 &p_to) const;
	Variant intersects_ray_bind(const Vector3 &p_from, const Vector3 &p_dir) const;

	_FORCE_INLINE_ void quantize(real_t p_unit);
	_FORCE_INLINE_ AABB quantized(real_t p_unit) const;

	_FORCE_INLINE_ constexpr void set_end(const Vector3 &p_end) {
		size = p_end - position;
	}

	_FORCE_INLINE_ constexpr Vector3 get_end() const {
		return position + size;
	}

	_FORCE_INLINE_ constexpr Vector3 get_center() const {
		return position + (size * 0.5f);
	}

	operator String() const;

	_FORCE_INLINE_ constexpr AABB() = default;
	_FORCE_INLINE_ constexpr AABB(const Vector3 &p_pos, const Vector3 &p_size) :
			position(p_pos),
			size(p_size) {
	}
};

bool AABB::intersects(const AABB &p_aabb) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0 || p_aabb.size.x < 0 || p_aabb.size.y < 0 || p_aabb.size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	if (position.x >= (p_aabb.position.x + p_aabb.size.x)) {
		return false;
	}
	if ((position.x + size.x) <= p_aabb.position.x) {
		return false;
	}
	if (position.y >= (p_aabb.position.y + p_aabb.size.y)) {
		return false;
	}
	if ((position.y + size.y) <= p_aabb.position.y) {
		return false;
	}
	if (position.z >= (p_aabb.position.z + p_aabb.size.z)) {
		return false;
	}
	if ((position.z + size.z) <= p_aabb.position.z) {
		return false;
	}

	return true;
}

bool AABB::intersects_inclusive(const AABB &p_aabb) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0 || p_aabb.size.x < 0 || p_aabb.size.y < 0 || p_aabb.size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	if (position.x > (p_aabb.position.x + p_aabb.size.x)) {
		return false;
	}
	if ((position.x + size.x) < p_aabb.position.x) {
		return false;
	}
	if (position.y > (p_aabb.position.y + p_aabb.size.y)) {
		return false;
	}
	if ((position.y + size.y) < p_aabb.position.y) {
		return false;
	}
	if (position.z > (p_aabb.position.z + p_aabb.size.z)) {
		return false;
	}
	if ((position.z + size.z) < p_aabb.position.z) {
		return false;
	}

	return true;
}

bool AABB::encloses(const AABB &p_aabb) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0 || p_aabb.size.x < 0 || p_aabb.size.y < 0 || p_aabb.size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	Vector3 src_min = position;
	Vector3 src_max = position + size;
	Vector3 dst_min = p_aabb.position;
	Vector3 dst_max = p_aabb.position + p_aabb.size;

	return (
			(src_min.x <= dst_min.x) &&
			(src_max.x >= dst_max.x) &&
			(src_min.y <= dst_min.y) &&
			(src_max.y >= dst_max.y) &&
			(src_min.z <= dst_min.z) &&
			(src_max.z >= dst_max.z));
}

Vector3 AABB::get_support(const Vector3 &p_direction) const {
	Vector3 support = position;
	if (p_direction.x > 0.0f) {
		support.x += size.x;
	}
	if (p_direction.y > 0.0f) {
		support.y += size.y;
	}
	if (p_direction.z > 0.0f) {
		support.z += size.z;
	}
	return support;
}

Vector3 AABB::get_endpoint(int p_point) const {
	switch (p_point) {
		case 0:
			return Vector3(position.x, position.y, position.z);
		case 1:
			return Vector3(position.x, position.y, position.z + size.z);
		case 2:
			return Vector3(position.x, position.y + size.y, position.z);
		case 3:
			return Vector3(position.x, position.y + size.y, position.z + size.z);
		case 4:
			return Vector3(position.x + size.x, position.y, position.z);
		case 5:
			return Vector3(position.x + size.x, position.y, position.z + size.z);
		case 6:
			return Vector3(position.x + size.x, position.y + size.y, position.z);
		case 7:
			return Vector3(position.x + size.x, position.y + size.y, position.z + size.z);
	}

	ERR_FAIL_V(Vector3());
}

bool AABB::intersects_convex_shape(const Plane *p_planes, int p_plane_count, const Vector3 *p_points, int p_point_count) const {
	Vector3 half_extents = size * 0.5f;
	Vector3 ofs = position + half_extents;

	for (int i = 0; i < p_plane_count; i++) {
		const Plane &p = p_planes[i];
		Vector3 point(
				(p.normal.x > 0) ? -half_extents.x : half_extents.x,
				(p.normal.y > 0) ? -half_extents.y : half_extents.y,
				(p.normal.z > 0) ? -half_extents.z : half_extents.z);
		point += ofs;
		if (p.is_point_over(point)) {
			return false;
		}
	}

	// Make sure all points in the shape aren't fully separated from the AABB on
	// each axis.
	int bad_point_counts_positive[3] = { 0 };
	int bad_point_counts_negative[3] = { 0 };

	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < p_point_count; i++) {
			if (p_points[i].coord[k] > ofs.coord[k] + half_extents.coord[k]) {
				bad_point_counts_positive[k]++;
			}
			if (p_points[i].coord[k] < ofs.coord[k] - half_extents.coord[k]) {
				bad_point_counts_negative[k]++;
			}
		}

		if (bad_point_counts_negative[k] == p_point_count) {
			return false;
		}
		if (bad_point_counts_positive[k] == p_point_count) {
			return false;
		}
	}

	return true;
}

bool AABB::inside_convex_shape(const Plane *p_planes, int p_plane_count) const {
	Vector3 half_extents = size * 0.5f;
	Vector3 ofs = position + half_extents;

	for (int i = 0; i < p_plane_count; i++) {
		const Plane &p = p_planes[i];
		Vector3 point(
				(p.normal.x < 0) ? -half_extents.x : half_extents.x,
				(p.normal.y < 0) ? -half_extents.y : half_extents.y,
				(p.normal.z < 0) ? -half_extents.z : half_extents.z);
		point += ofs;
		if (p.is_point_over(point)) {
			return false;
		}
	}

	return true;
}

bool AABB::has_point(const Vector3 &p_point) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	if (p_point.x < position.x) {
		return false;
	}
	if (p_point.y < position.y) {
		return false;
	}
	if (p_point.z < position.z) {
		return false;
	}
	if (p_point.x > position.x + size.x) {
		return false;
	}
	if (p_point.y > position.y + size.y) {
		return false;
	}
	if (p_point.z > position.z + size.z) {
		return false;
	}

	return true;
}

void AABB::expand_to(const Vector3 &p_vector) {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	Vector3 begin = position;
	Vector3 end = position + size;

	if (p_vector.x < begin.x) {
		begin.x = p_vector.x;
	}
	if (p_vector.y < begin.y) {
		begin.y = p_vector.y;
	}
	if (p_vector.z < begin.z) {
		begin.z = p_vector.z;
	}

	if (p_vector.x > end.x) {
		end.x = p_vector.x;
	}
	if (p_vector.y > end.y) {
		end.y = p_vector.y;
	}
	if (p_vector.z > end.z) {
		end.z = p_vector.z;
	}

	position = begin;
	size = end - begin;
}

void AABB::project_range_in_plane(const Plane &p_plane, real_t &r_min, real_t &r_max) const {
	Vector3 half_extents(size.x * 0.5f, size.y * 0.5f, size.z * 0.5f);
	Vector3 center(position.x + half_extents.x, position.y + half_extents.y, position.z + half_extents.z);

	real_t length = p_plane.normal.abs().dot(half_extents);
	real_t distance = p_plane.distance_to(center);
	r_min = distance - length;
	r_max = distance + length;
}

real_t AABB::get_longest_axis_size() const {
	real_t max_size = size.x;

	if (size.y > max_size) {
		max_size = size.y;
	}

	if (size.z > max_size) {
		max_size = size.z;
	}

	return max_size;
}

real_t AABB::get_shortest_axis_size() const {
	real_t max_size = size.x;

	if (size.y < max_size) {
		max_size = size.y;
	}

	if (size.z < max_size) {
		max_size = size.z;
	}

	return max_size;
}

bool AABB::smits_intersect_ray(const Vector3 &p_from, const Vector3 &p_dir, real_t p_t0, real_t p_t1) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	real_t divx = 1.0f / p_dir.x;
	real_t divy = 1.0f / p_dir.y;
	real_t divz = 1.0f / p_dir.z;

	Vector3 upbound = position + size;
	real_t tmin, tmax, tymin, tymax, tzmin, tzmax;
	if (p_dir.x >= 0) {
		tmin = (position.x - p_from.x) * divx;
		tmax = (upbound.x - p_from.x) * divx;
	} else {
		tmin = (upbound.x - p_from.x) * divx;
		tmax = (position.x - p_from.x) * divx;
	}
	if (p_dir.y >= 0) {
		tymin = (position.y - p_from.y) * divy;
		tymax = (upbound.y - p_from.y) * divy;
	} else {
		tymin = (upbound.y - p_from.y) * divy;
		tymax = (position.y - p_from.y) * divy;
	}
	if ((tmin > tymax) || (tymin > tmax)) {
		return false;
	}
	if (tymin > tmin) {
		tmin = tymin;
	}
	if (tymax < tmax) {
		tmax = tymax;
	}
	if (p_dir.z >= 0) {
		tzmin = (position.z - p_from.z) * divz;
		tzmax = (upbound.z - p_from.z) * divz;
	} else {
		tzmin = (upbound.z - p_from.z) * divz;
		tzmax = (position.z - p_from.z) * divz;
	}
	if ((tmin > tzmax) || (tzmin > tmax)) {
		return false;
	}
	if (tzmin > tmin) {
		tmin = tzmin;
	}
	if (tzmax < tmax) {
		tmax = tzmax;
	}
	return ((tmin < p_t1) && (tmax > p_t0));
}

void AABB::grow_by(real_t p_amount) {
	position.x -= p_amount;
	position.y -= p_amount;
	position.z -= p_amount;
	size.x += 2.0f * p_amount;
	size.y += 2.0f * p_amount;
	size.z += 2.0f * p_amount;
}

void AABB::quantize(real_t p_unit) {
	size += position;

	position.x -= Math::fposmodp(position.x, p_unit);
	position.y -= Math::fposmodp(position.y, p_unit);
	position.z -= Math::fposmodp(position.z, p_unit);

	size.x -= Math::fposmodp(size.x, p_unit);
	size.y -= Math::fposmodp(size.y, p_unit);
	size.z -= Math::fposmodp(size.z, p_unit);

	size.x += p_unit;
	size.y += p_unit;
	size.z += p_unit;

	size -= position;
}

AABB AABB::quantized(real_t p_unit) const {
	AABB ret = *this;
	ret.quantize(p_unit);
	return ret;
}

real_t constexpr AABB::get_volume() const {
	return size.x * size.y * size.z;
}

void AABB::merge_with(const AABB &p_aabb) {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0 || p_aabb.size.x < 0 || p_aabb.size.y < 0 || p_aabb.size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	Vector3 beg_1, beg_2;
	Vector3 end_1, end_2;
	Vector3 min, max;

	beg_1 = position;
	beg_2 = p_aabb.position;
	end_1 = size + beg_1;
	end_2 = p_aabb.size + beg_2;

	min.x = (beg_1.x < beg_2.x) ? beg_1.x : beg_2.x;
	min.y = (beg_1.y < beg_2.y) ? beg_1.y : beg_2.y;
	min.z = (beg_1.z < beg_2.z) ? beg_1.z : beg_2.z;

	max.x = (end_1.x > end_2.x) ? end_1.x : end_2.x;
	max.y = (end_1.y > end_2.y) ? end_1.y : end_2.y;
	max.z = (end_1.z > end_2.z) ? end_1.z : end_2.z;

	position = min;
	size = max - min;
}

bool AABB::is_equal_approx(const AABB &p_aabb) const {
	return position.is_equal_approx(p_aabb.position) && size.is_equal_approx(p_aabb.size);
}

bool AABB::is_same(const AABB &p_aabb) const {
	return position.is_same(p_aabb.position) && size.is_same(p_aabb.size);
}

bool AABB::is_finite() const {
	return position.is_finite() && size.is_finite();
}

AABB AABB::intersection(const AABB &p_aabb) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0 || p_aabb.size.x < 0 || p_aabb.size.y < 0 || p_aabb.size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	Vector3 src_min = position;
	Vector3 src_max = position + size;
	Vector3 dst_min = p_aabb.position;
	Vector3 dst_max = p_aabb.position + p_aabb.size;

	Vector3 min, max;

	if (src_min.x > dst_max.x || src_max.x < dst_min.x) {
		return AABB();
	} else {
		min.x = (src_min.x > dst_min.x) ? src_min.x : dst_min.x;
		max.x = (src_max.x < dst_max.x) ? src_max.x : dst_max.x;
	}

	if (src_min.y > dst_max.y || src_max.y < dst_min.y) {
		return AABB();
	} else {
		min.y = (src_min.y > dst_min.y) ? src_min.y : dst_min.y;
		max.y = (src_max.y < dst_max.y) ? src_max.y : dst_max.y;
	}

	if (src_min.z > dst_max.z || src_max.z < dst_min.z) {
		return AABB();
	} else {
		min.z = (src_min.z > dst_min.z) ? src_min.z : dst_min.z;
		max.z = (src_max.z < dst_max.z) ? src_max.z : dst_max.z;
	}

	return AABB(min, max - min);
}

// Note that this routine returns the BACKTRACKED (i.e. behind the ray origin)
// intersection point + normal if INSIDE the AABB.
// The caller can therefore decide when INSIDE whether to use the
// backtracked intersection, or use p_from as the intersection, and
// carry on progressing without e.g. reflecting against the normal.
bool AABB::find_intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, bool &r_inside, Vector3 *r_intersection_point, Vector3 *r_normal) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	Vector3 end = position + size;
	real_t tmin = -1e20;
	real_t tmax = 1e20;
	int axis = 0;

	// Make sure r_inside is always initialized,
	// to prevent reading uninitialized data in the client code.
	r_inside = false;

	for (int i = 0; i < 3; i++) {
		if (p_dir[i] == 0) {
			if ((p_from[i] < position[i]) || (p_from[i] > end[i])) {
				return false;
			}
		} else { // ray not parallel to planes in this direction
			real_t t1 = (position[i] - p_from[i]) / p_dir[i];
			real_t t2 = (end[i] - p_from[i]) / p_dir[i];

			if (t1 > t2) {
				SWAP(t1, t2);
			}
			if (t1 >= tmin) {
				tmin = t1;
				axis = i;
			}
			if (t2 < tmax) {
				if (t2 < 0) {
					return false;
				}
				tmax = t2;
			}
			if (tmin > tmax) {
				return false;
			}
		}
	}

	// Did the ray start from inside the box?
	// In which case the intersection returned is the point of entry
	// (behind the ray start) or the calling routine can use the ray origin as intersection point.
	r_inside = tmin < 0;

	if (r_intersection_point) {
		*r_intersection_point = p_from + p_dir * tmin;

		// Prevent float error by making sure the point is exactly
		// on the AABB border on the relevant axis.
		r_intersection_point->coord[axis] = (p_dir[axis] >= 0) ? position.coord[axis] : end.coord[axis];
	}
	if (r_normal) {
		*r_normal = Vector3();
		(*r_normal)[axis] = (p_dir[axis] >= 0) ? -1 : 1;
	}

	return true;
}

bool AABB::intersects_segment(const Vector3 &p_from, const Vector3 &p_to, Vector3 *r_intersection_point, Vector3 *r_normal) const {
#ifdef MATH_CHECKS
	if (unlikely(size.x < 0 || size.y < 0 || size.z < 0)) {
		ERR_PRINT("AABB size is negative, this is not supported. Use AABB.abs() to get an AABB with a positive size.");
	}
#endif
	real_t min = 0, max = 1;
	int axis = 0;
	real_t sign = 0;

	for (int i = 0; i < 3; i++) {
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

	Vector3 rel = p_to - p_from;

	if (r_normal) {
		Vector3 normal;
		normal[axis] = sign;
		*r_normal = normal;
	}

	if (r_intersection_point) {
		*r_intersection_point = p_from + rel * min;
	}

	return true;
}

bool AABB::intersects_plane(const Plane &p_plane) const {
	Vector3 points[8] = {
		Vector3(position.x, position.y, position.z),
		Vector3(position.x, position.y, position.z + size.z),
		Vector3(position.x, position.y + size.y, position.z),
		Vector3(position.x, position.y + size.y, position.z + size.z),
		Vector3(position.x + size.x, position.y, position.z),
		Vector3(position.x + size.x, position.y, position.z + size.z),
		Vector3(position.x + size.x, position.y + size.y, position.z),
		Vector3(position.x + size.x, position.y + size.y, position.z + size.z),
	};

	bool over = false;
	bool under = false;

	for (int i = 0; i < 8; i++) {
		if (p_plane.distance_to(points[i]) > 0) {
			over = true;
		} else {
			under = true;
		}
	}

	return under && over;
}

Vector3 AABB::get_longest_axis() const {
	Vector3 axis(1, 0, 0);
	real_t max_size = size.x;

	if (size.y > max_size) {
		axis = Vector3(0, 1, 0);
		max_size = size.y;
	}

	if (size.z > max_size) {
		axis = Vector3(0, 0, 1);
	}

	return axis;
}

int AABB::get_longest_axis_index() const {
	int axis = 0;
	real_t max_size = size.x;

	if (size.y > max_size) {
		axis = 1;
		max_size = size.y;
	}

	if (size.z > max_size) {
		axis = 2;
	}

	return axis;
}

Vector3 AABB::get_shortest_axis() const {
	Vector3 axis(1, 0, 0);
	real_t min_size = size.x;

	if (size.y < min_size) {
		axis = Vector3(0, 1, 0);
		min_size = size.y;
	}

	if (size.z < min_size) {
		axis = Vector3(0, 0, 1);
	}

	return axis;
}

int AABB::get_shortest_axis_index() const {
	int axis = 0;
	real_t min_size = size.x;

	if (size.y < min_size) {
		axis = 1;
		min_size = size.y;
	}

	if (size.z < min_size) {
		axis = 2;
	}

	return axis;
}

AABB AABB::merge(const AABB &p_with) const {
	AABB aabb = *this;
	aabb.merge_with(p_with);
	return aabb;
}

AABB AABB::expand(const Vector3 &p_vector) const {
	AABB aabb = *this;
	aabb.expand_to(p_vector);
	return aabb;
}

AABB AABB::grow(real_t p_by) const {
	AABB aabb = *this;
	aabb.grow_by(p_by);
	return aabb;
}

void AABB::get_edge(int p_edge, Vector3 &r_from, Vector3 &r_to) const {
	ERR_FAIL_INDEX(p_edge, 12);
	switch (p_edge) {
		case 0: {
			r_from = Vector3(position.x + size.x, position.y, position.z);
			r_to = Vector3(position.x, position.y, position.z);
		} break;
		case 1: {
			r_from = Vector3(position.x + size.x, position.y, position.z + size.z);
			r_to = Vector3(position.x + size.x, position.y, position.z);
		} break;
		case 2: {
			r_from = Vector3(position.x, position.y, position.z + size.z);
			r_to = Vector3(position.x + size.x, position.y, position.z + size.z);

		} break;
		case 3: {
			r_from = Vector3(position.x, position.y, position.z);
			r_to = Vector3(position.x, position.y, position.z + size.z);

		} break;
		case 4: {
			r_from = Vector3(position.x, position.y + size.y, position.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z);
		} break;
		case 5: {
			r_from = Vector3(position.x + size.x, position.y + size.y, position.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z + size.z);
		} break;
		case 6: {
			r_from = Vector3(position.x + size.x, position.y + size.y, position.z + size.z);
			r_to = Vector3(position.x, position.y + size.y, position.z + size.z);

		} break;
		case 7: {
			r_from = Vector3(position.x, position.y + size.y, position.z + size.z);
			r_to = Vector3(position.x, position.y + size.y, position.z);

		} break;
		case 8: {
			r_from = Vector3(position.x, position.y, position.z + size.z);
			r_to = Vector3(position.x, position.y + size.y, position.z + size.z);

		} break;
		case 9: {
			r_from = Vector3(position.x, position.y, position.z);
			r_to = Vector3(position.x, position.y + size.y, position.z);

		} break;
		case 10: {
			r_from = Vector3(position.x + size.x, position.y, position.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z);

		} break;
		case 11: {
			r_from = Vector3(position.x + size.x, position.y, position.z + size.z);
			r_to = Vector3(position.x + size.x, position.y + size.y, position.z + size.z);

		} break;
	}
}

template <>
struct is_zero_constructible<AABB> : std::true_type {};
