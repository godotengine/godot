#include "AABB.hpp"
#include "Plane.hpp"
#include "Vector3.hpp"

#include <algorithm>

namespace godot {

bool AABB::intersects(const AABB &p_aabb) const {

	if (position.x >= (p_aabb.position.x + p_aabb.size.x))
		return false;
	if ((position.x + size.x) <= p_aabb.position.x)
		return false;
	if (position.y >= (p_aabb.position.y + p_aabb.size.y))
		return false;
	if ((position.y + size.y) <= p_aabb.position.y)
		return false;
	if (position.z >= (p_aabb.position.z + p_aabb.size.z))
		return false;
	if ((position.z + size.z) <= p_aabb.position.z)
		return false;

	return true;
}

bool AABB::intersects_inclusive(const AABB &p_aabb) const {

	if (position.x > (p_aabb.position.x + p_aabb.size.x))
		return false;
	if ((position.x + size.x) < p_aabb.position.x)
		return false;
	if (position.y > (p_aabb.position.y + p_aabb.size.y))
		return false;
	if ((position.y + size.y) < p_aabb.position.y)
		return false;
	if (position.z > (p_aabb.position.z + p_aabb.size.z))
		return false;
	if ((position.z + size.z) < p_aabb.position.z)
		return false;

	return true;
}

bool AABB::encloses(const AABB &p_aabb) const {

	Vector3 src_min = position;
	Vector3 src_max = position + size;
	Vector3 dst_min = p_aabb.position;
	Vector3 dst_max = p_aabb.position + p_aabb.size;

	return (
			(src_min.x <= dst_min.x) &&
			(src_max.x > dst_max.x) &&
			(src_min.y <= dst_min.y) &&
			(src_max.y > dst_max.y) &&
			(src_min.z <= dst_min.z) &&
			(src_max.z > dst_max.z));
}

Vector3 AABB::get_support(const Vector3 &p_normal) const {

	Vector3 half_extents = size * 0.5;
	Vector3 ofs = position + half_extents;

	return Vector3(
				   (p_normal.x > 0) ? -half_extents.x : half_extents.x,
				   (p_normal.y > 0) ? -half_extents.y : half_extents.y,
				   (p_normal.z > 0) ? -half_extents.z : half_extents.z) +
		   ofs;
}

Vector3 AABB::get_endpoint(int p_point) const {

	switch (p_point) {
		case 0: return Vector3(position.x, position.y, position.z);
		case 1: return Vector3(position.x, position.y, position.z + size.z);
		case 2: return Vector3(position.x, position.y + size.y, position.z);
		case 3: return Vector3(position.x, position.y + size.y, position.z + size.z);
		case 4: return Vector3(position.x + size.x, position.y, position.z);
		case 5: return Vector3(position.x + size.x, position.y, position.z + size.z);
		case 6: return Vector3(position.x + size.x, position.y + size.y, position.z);
		case 7: return Vector3(position.x + size.x, position.y + size.y, position.z + size.z);
	};

	ERR_FAIL_V(Vector3());
}

bool AABB::intersects_convex_shape(const Plane *p_planes, int p_plane_count) const {

	Vector3 half_extents = size * 0.5;
	Vector3 ofs = position + half_extents;

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
}

bool AABB::has_point(const Vector3 &p_point) const {

	if (p_point.x < position.x)
		return false;
	if (p_point.y < position.y)
		return false;
	if (p_point.z < position.z)
		return false;
	if (p_point.x > position.x + size.x)
		return false;
	if (p_point.y > position.y + size.y)
		return false;
	if (p_point.z > position.z + size.z)
		return false;

	return true;
}

void AABB::expand_to(const Vector3 &p_vector) {

	Vector3 begin = position;
	Vector3 end = position + size;

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

	position = begin;
	size = end - begin;
}

void AABB::project_range_in_plane(const Plane &p_plane, real_t &r_min, real_t &r_max) const {

	Vector3 half_extents(size.x * 0.5, size.y * 0.5, size.z * 0.5);
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

bool AABB::smits_intersect_ray(const Vector3 &from, const Vector3 &dir, real_t t0, real_t t1) const {

	real_t divx = 1.0 / dir.x;
	real_t divy = 1.0 / dir.y;
	real_t divz = 1.0 / dir.z;

	Vector3 upbound = position + size;
	real_t tmin, tmax, tymin, tymax, tzmin, tzmax;
	if (dir.x >= 0) {
		tmin = (position.x - from.x) * divx;
		tmax = (upbound.x - from.x) * divx;
	} else {
		tmin = (upbound.x - from.x) * divx;
		tmax = (position.x - from.x) * divx;
	}
	if (dir.y >= 0) {
		tymin = (position.y - from.y) * divy;
		tymax = (upbound.y - from.y) * divy;
	} else {
		tymin = (upbound.y - from.y) * divy;
		tymax = (position.y - from.y) * divy;
	}
	if ((tmin > tymax) || (tymin > tmax))
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	if (dir.z >= 0) {
		tzmin = (position.z - from.z) * divz;
		tzmax = (upbound.z - from.z) * divz;
	} else {
		tzmin = (upbound.z - from.z) * divz;
		tzmax = (position.z - from.z) * divz;
	}
	if ((tmin > tzmax) || (tzmin > tmax))
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	return ((tmin < t1) && (tmax > t0));
}

void AABB::grow_by(real_t p_amount) {

	position.x -= p_amount;
	position.y -= p_amount;
	position.z -= p_amount;
	size.x += 2.0 * p_amount;
	size.y += 2.0 * p_amount;
	size.z += 2.0 * p_amount;
}

real_t AABB::get_area() const {

	return size.x * size.y * size.z;
}

bool AABB::operator==(const AABB &p_rval) const {

	return ((position == p_rval.position) && (size == p_rval.size));
}
bool AABB::operator!=(const AABB &p_rval) const {

	return ((position != p_rval.position) || (size != p_rval.size));
}

void AABB::merge_with(const AABB &p_aabb) {

	Vector3 beg_1, beg_2;
	Vector3 end_1, end_2;
	Vector3 min, max;

	beg_1 = position;
	beg_2 = p_aabb.position;
	end_1 = Vector3(size.x, size.y, size.z) + beg_1;
	end_2 = Vector3(p_aabb.size.x, p_aabb.size.y, p_aabb.size.z) + beg_2;

	min.x = (beg_1.x < beg_2.x) ? beg_1.x : beg_2.x;
	min.y = (beg_1.y < beg_2.y) ? beg_1.y : beg_2.y;
	min.z = (beg_1.z < beg_2.z) ? beg_1.z : beg_2.z;

	max.x = (end_1.x > end_2.x) ? end_1.x : end_2.x;
	max.y = (end_1.y > end_2.y) ? end_1.y : end_2.y;
	max.z = (end_1.z > end_2.z) ? end_1.z : end_2.z;

	position = min;
	size = max - min;
}

AABB AABB::intersection(const AABB &p_aabb) const {

	Vector3 src_min = position;
	Vector3 src_max = position + size;
	Vector3 dst_min = p_aabb.position;
	Vector3 dst_max = p_aabb.position + p_aabb.size;

	Vector3 min, max;

	if (src_min.x > dst_max.x || src_max.x < dst_min.x)
		return AABB();
	else {

		min.x = (src_min.x > dst_min.x) ? src_min.x : dst_min.x;
		max.x = (src_max.x < dst_max.x) ? src_max.x : dst_max.x;
	}

	if (src_min.y > dst_max.y || src_max.y < dst_min.y)
		return AABB();
	else {

		min.y = (src_min.y > dst_min.y) ? src_min.y : dst_min.y;
		max.y = (src_max.y < dst_max.y) ? src_max.y : dst_max.y;
	}

	if (src_min.z > dst_max.z || src_max.z < dst_min.z)
		return AABB();
	else {

		min.z = (src_min.z > dst_min.z) ? src_min.z : dst_min.z;
		max.z = (src_max.z < dst_max.z) ? src_max.z : dst_max.z;
	}

	return AABB(min, max - min);
}

bool AABB::intersects_ray(const Vector3 &p_from, const Vector3 &p_dir, Vector3 *r_clip, Vector3 *r_normal) const {

	Vector3 c1, c2;
	Vector3 end = position + size;
	real_t near = -1e20;
	real_t far = 1e20;
	int axis = 0;

	for (int i = 0; i < 3; i++) {
		if (p_dir[i] == 0) {
			if ((p_from[i] < position[i]) || (p_from[i] > end[i])) {
				return false;
			}
		} else { // ray not parallel to planes in this direction
			c1[i] = (position[i] - p_from[i]) / p_dir[i];
			c2[i] = (end[i] - p_from[i]) / p_dir[i];

			if (c1[i] > c2[i]) {
				std::swap(c1, c2);
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

bool AABB::intersects_segment(const Vector3 &p_from, const Vector3 &p_to, Vector3 *r_clip, Vector3 *r_normal) const {

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

		if (p_plane.distance_to(points[i]) > 0)
			over = true;
		else
			under = true;
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
		max_size = size.z;
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
		max_size = size.z;
	}

	return axis;
}

Vector3 AABB::get_shortest_axis() const {

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
int AABB::get_shortest_axis_index() const {

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

AABB::operator String() const {

	return String() + position + " - " + size;
}

} // namespace godot
