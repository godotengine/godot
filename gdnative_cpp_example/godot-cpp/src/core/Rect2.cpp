#include "Rect2.hpp"
#include "String.hpp"
#include "Transform2D.hpp"
#include "Vector2.hpp"

#include <cmath>

namespace godot {

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a, b) (a < b ? a : b)
#endif

real_t Rect2::distance_to(const Vector2 &p_point) const {

	real_t dist = 1e20;

	if (p_point.x < position.x) {
		dist = MIN(dist, position.x - p_point.x);
	}
	if (p_point.y < position.y) {
		dist = MIN(dist, position.y - p_point.y);
	}
	if (p_point.x >= (position.x + size.x)) {
		dist = MIN(p_point.x - (position.x + size.x), dist);
	}
	if (p_point.y >= (position.y + size.y)) {
		dist = MIN(p_point.y - (position.y + size.y), dist);
	}

	if (dist == 1e20)
		return 0;
	else
		return dist;
}

Rect2 Rect2::clip(const Rect2 &p_rect) const { /// return a clipped rect

	Rect2 new_rect = p_rect;

	if (!intersects(new_rect))
		return Rect2();

	new_rect.position.x = MAX(p_rect.position.x, position.x);
	new_rect.position.y = MAX(p_rect.position.y, position.y);

	Point2 p_rect_end = p_rect.position + p_rect.size;
	Point2 end = position + size;

	new_rect.size.x = MIN(p_rect_end.x, end.x) - new_rect.position.x;
	new_rect.size.y = MIN(p_rect_end.y, end.y) - new_rect.position.y;

	return new_rect;
}

Rect2 Rect2::merge(const Rect2 &p_rect) const { ///< return a merged rect

	Rect2 new_rect;

	new_rect.position.x = MIN(p_rect.position.x, position.x);
	new_rect.position.y = MIN(p_rect.position.y, position.y);

	new_rect.size.x = MAX(p_rect.position.x + p_rect.size.x, position.x + size.x);
	new_rect.size.y = MAX(p_rect.position.y + p_rect.size.y, position.y + size.y);

	new_rect.size = new_rect.size - new_rect.position; //make relative again

	return new_rect;
}

Rect2::operator String() const {
	return String(position) + ", " + String(size);
}

bool Rect2::intersects_segment(const Point2 &p_from, const Point2 &p_to, Point2 *r_position, Point2 *r_normal) const {

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

	Vector2 rel = p_to - p_from;

	if (r_normal) {
		Vector2 normal;
		normal[axis] = sign;
		*r_normal = normal;
	}

	if (r_position)
		*r_position = p_from + rel * min;

	return true;
}

bool Rect2::intersects_transformed(const Transform2D &p_xform, const Rect2 &p_rect) const {

	//SAT intersection between local and transformed rect2

	Vector2 xf_points[4] = {
		p_xform.xform(p_rect.position),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y)),
		p_xform.xform(Vector2(p_rect.position.x, p_rect.position.y + p_rect.size.y)),
		p_xform.xform(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y + p_rect.size.y)),
	};

	real_t low_limit;

	//base rect2 first (faster)

	if (xf_points[0].y > position.y)
		goto next1;
	if (xf_points[1].y > position.y)
		goto next1;
	if (xf_points[2].y > position.y)
		goto next1;
	if (xf_points[3].y > position.y)
		goto next1;

	return false;

next1:

	low_limit = position.y + size.y;

	if (xf_points[0].y < low_limit)
		goto next2;
	if (xf_points[1].y < low_limit)
		goto next2;
	if (xf_points[2].y < low_limit)
		goto next2;
	if (xf_points[3].y < low_limit)
		goto next2;

	return false;

next2:

	if (xf_points[0].x > position.x)
		goto next3;
	if (xf_points[1].x > position.x)
		goto next3;
	if (xf_points[2].x > position.x)
		goto next3;
	if (xf_points[3].x > position.x)
		goto next3;

	return false;

next3:

	low_limit = position.x + size.x;

	if (xf_points[0].x < low_limit)
		goto next4;
	if (xf_points[1].x < low_limit)
		goto next4;
	if (xf_points[2].x < low_limit)
		goto next4;
	if (xf_points[3].x < low_limit)
		goto next4;

	return false;

next4:

	Vector2 xf_points2[4] = {
		position,
		Vector2(position.x + size.x, position.y),
		Vector2(position.x, position.y + size.y),
		Vector2(position.x + size.x, position.y + size.y),
	};

	real_t maxa = p_xform.elements[0].dot(xf_points2[0]);
	real_t mina = maxa;

	real_t dp = p_xform.elements[0].dot(xf_points2[1]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.elements[0].dot(xf_points2[2]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.elements[0].dot(xf_points2[3]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	real_t maxb = p_xform.elements[0].dot(xf_points[0]);
	real_t minb = maxb;

	dp = p_xform.elements[0].dot(xf_points[1]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.elements[0].dot(xf_points[2]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.elements[0].dot(xf_points[3]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	if (mina > maxb)
		return false;
	if (minb > maxa)
		return false;

	maxa = p_xform.elements[1].dot(xf_points2[0]);
	mina = maxa;

	dp = p_xform.elements[1].dot(xf_points2[1]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.elements[1].dot(xf_points2[2]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	dp = p_xform.elements[1].dot(xf_points2[3]);
	maxa = MAX(dp, maxa);
	mina = MIN(dp, mina);

	maxb = p_xform.elements[1].dot(xf_points[0]);
	minb = maxb;

	dp = p_xform.elements[1].dot(xf_points[1]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.elements[1].dot(xf_points[2]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	dp = p_xform.elements[1].dot(xf_points[3]);
	maxb = MAX(dp, maxb);
	minb = MIN(dp, minb);

	if (mina > maxb)
		return false;
	if (minb > maxa)
		return false;

	return true;
}

} // namespace godot
