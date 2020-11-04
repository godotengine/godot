/*************************************************************************/
/*  rect2.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RECT2_H
#define RECT2_H

#include "core/math/vector2.h" // also includes math_funcs and ustring

struct Transform2D;

struct Rect2 {
	Point2 position;
	Size2 size;

	const Vector2 &get_position() const { return position; }
	void set_position(const Vector2 &p_pos) { position = p_pos; }
	const Vector2 &get_size() const { return size; }
	void set_size(const Vector2 &p_size) { size = p_size; }

	real_t get_area() const { return size.width * size.height; }

	inline bool intersects(const Rect2 &p_rect, const bool p_include_borders = false) const {
		if (p_include_borders) {
			if (position.x > (p_rect.position.x + p_rect.size.width)) {
				return false;
			}
			if ((position.x + size.width) < p_rect.position.x) {
				return false;
			}
			if (position.y > (p_rect.position.y + p_rect.size.height)) {
				return false;
			}
			if ((position.y + size.height) < p_rect.position.y) {
				return false;
			}
		} else {
			if (position.x >= (p_rect.position.x + p_rect.size.width)) {
				return false;
			}
			if ((position.x + size.width) <= p_rect.position.x) {
				return false;
			}
			if (position.y >= (p_rect.position.y + p_rect.size.height)) {
				return false;
			}
			if ((position.y + size.height) <= p_rect.position.y) {
				return false;
			}
		}

		return true;
	}

	inline real_t distance_to(const Vector2 &p_point) const {
		real_t dist = 0.0;
		bool inside = true;

		if (p_point.x < position.x) {
			real_t d = position.x - p_point.x;
			dist = d;
			inside = false;
		}
		if (p_point.y < position.y) {
			real_t d = position.y - p_point.y;
			dist = inside ? d : MIN(dist, d);
			inside = false;
		}
		if (p_point.x >= (position.x + size.x)) {
			real_t d = p_point.x - (position.x + size.x);
			dist = inside ? d : MIN(dist, d);
			inside = false;
		}
		if (p_point.y >= (position.y + size.y)) {
			real_t d = p_point.y - (position.y + size.y);
			dist = inside ? d : MIN(dist, d);
			inside = false;
		}

		if (inside) {
			return 0;
		} else {
			return dist;
		}
	}

	bool intersects_transformed(const Transform2D &p_xform, const Rect2 &p_rect) const;

	bool intersects_segment(const Point2 &p_from, const Point2 &p_to, Point2 *r_pos = nullptr, Point2 *r_normal = nullptr) const;

	inline bool encloses(const Rect2 &p_rect) const {
		return (p_rect.position.x >= position.x) && (p_rect.position.y >= position.y) &&
			   ((p_rect.position.x + p_rect.size.x) <= (position.x + size.x)) &&
			   ((p_rect.position.y + p_rect.size.y) <= (position.y + size.y));
	}

	_FORCE_INLINE_ bool has_no_area() const {
		return (size.x <= 0 || size.y <= 0);
	}
	inline Rect2 clip(const Rect2 &p_rect) const { /// return a clipped rect

		Rect2 new_rect = p_rect;

		if (!intersects(new_rect)) {
			return Rect2();
		}

		new_rect.position.x = MAX(p_rect.position.x, position.x);
		new_rect.position.y = MAX(p_rect.position.y, position.y);

		Point2 p_rect_end = p_rect.position + p_rect.size;
		Point2 end = position + size;

		new_rect.size.x = MIN(p_rect_end.x, end.x) - new_rect.position.x;
		new_rect.size.y = MIN(p_rect_end.y, end.y) - new_rect.position.y;

		return new_rect;
	}

	inline Rect2 merge(const Rect2 &p_rect) const { ///< return a merged rect

		Rect2 new_rect;

		new_rect.position.x = MIN(p_rect.position.x, position.x);
		new_rect.position.y = MIN(p_rect.position.y, position.y);

		new_rect.size.x = MAX(p_rect.position.x + p_rect.size.x, position.x + size.x);
		new_rect.size.y = MAX(p_rect.position.y + p_rect.size.y, position.y + size.y);

		new_rect.size = new_rect.size - new_rect.position; //make relative again

		return new_rect;
	}
	inline bool has_point(const Point2 &p_point) const {
		if (p_point.x < position.x) {
			return false;
		}
		if (p_point.y < position.y) {
			return false;
		}

		if (p_point.x >= (position.x + size.x)) {
			return false;
		}
		if (p_point.y >= (position.y + size.y)) {
			return false;
		}

		return true;
	}
	bool is_equal_approx(const Rect2 &p_rect) const;

	bool operator==(const Rect2 &p_rect) const { return position == p_rect.position && size == p_rect.size; }
	bool operator!=(const Rect2 &p_rect) const { return position != p_rect.position || size != p_rect.size; }

	inline Rect2 grow(real_t p_by) const {
		Rect2 g = *this;
		g.position.x -= p_by;
		g.position.y -= p_by;
		g.size.width += p_by * 2;
		g.size.height += p_by * 2;
		return g;
	}

	inline Rect2 grow_margin(Margin p_margin, real_t p_amount) const {
		Rect2 g = *this;
		g = g.grow_individual((MARGIN_LEFT == p_margin) ? p_amount : 0,
				(MARGIN_TOP == p_margin) ? p_amount : 0,
				(MARGIN_RIGHT == p_margin) ? p_amount : 0,
				(MARGIN_BOTTOM == p_margin) ? p_amount : 0);
		return g;
	}

	inline Rect2 grow_margin_bind(uint32_t p_margin, real_t p_amount) const {
		return grow_margin(Margin(p_margin), p_amount);
	}

	inline Rect2 grow_individual(real_t p_left, real_t p_top, real_t p_right, real_t p_bottom) const {
		Rect2 g = *this;
		g.position.x -= p_left;
		g.position.y -= p_top;
		g.size.width += p_left + p_right;
		g.size.height += p_top + p_bottom;

		return g;
	}

	_FORCE_INLINE_ Rect2 expand(const Vector2 &p_vector) const {
		Rect2 r = *this;
		r.expand_to(p_vector);
		return r;
	}

	inline void expand_to(const Vector2 &p_vector) { //in place function for speed

		Vector2 begin = position;
		Vector2 end = position + size;

		if (p_vector.x < begin.x) {
			begin.x = p_vector.x;
		}
		if (p_vector.y < begin.y) {
			begin.y = p_vector.y;
		}

		if (p_vector.x > end.x) {
			end.x = p_vector.x;
		}
		if (p_vector.y > end.y) {
			end.y = p_vector.y;
		}

		position = begin;
		size = end - begin;
	}

	_FORCE_INLINE_ Rect2 abs() const {
		return Rect2(Point2(position.x + MIN(size.x, 0), position.y + MIN(size.y, 0)), size.abs());
	}

	Vector2 get_support(const Vector2 &p_normal) const {
		Vector2 half_extents = size * 0.5;
		Vector2 ofs = position + half_extents;
		return Vector2(
					   (p_normal.x > 0) ? -half_extents.x : half_extents.x,
					   (p_normal.y > 0) ? -half_extents.y : half_extents.y) +
			   ofs;
	}

	_FORCE_INLINE_ bool intersects_filled_polygon(const Vector2 *p_points, int p_point_count) const {
		Vector2 center = position + size * 0.5;
		int side_plus = 0;
		int side_minus = 0;
		Vector2 end = position + size;

		int i_f = p_point_count - 1;
		for (int i = 0; i < p_point_count; i++) {
			const Vector2 &a = p_points[i_f];
			const Vector2 &b = p_points[i];
			i_f = i;

			Vector2 r = (b - a);
			float l = r.length();
			if (l == 0.0) {
				continue;
			}

			//check inside
			Vector2 tg = r.tangent();
			float s = tg.dot(center) - tg.dot(a);
			if (s < 0.0) {
				side_plus++;
			} else {
				side_minus++;
			}

			//check ray box
			r /= l;
			Vector2 ir(1.0 / r.x, 1.0 / r.y);

			// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
			// r.org is origin of ray
			Vector2 t13 = (position - a) * ir;
			Vector2 t24 = (end - a) * ir;

			float tmin = MAX(MIN(t13.x, t24.x), MIN(t13.y, t24.y));
			float tmax = MIN(MAX(t13.x, t24.x), MAX(t13.y, t24.y));

			// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
			if (tmax < 0 || tmin > tmax || tmin >= l) {
				continue;
			}

			return true;
		}

		if (side_plus * side_minus == 0) {
			return true; //all inside
		} else {
			return false;
		}
	}
	operator String() const { return String(position) + ", " + String(size); }

	Rect2() {}
	Rect2(real_t p_x, real_t p_y, real_t p_width, real_t p_height) :
			position(Point2(p_x, p_y)),
			size(Size2(p_width, p_height)) {
	}
	Rect2(const Point2 &p_pos, const Size2 &p_size) :
			position(p_pos),
			size(p_size) {
	}
};

struct Rect2i {
	Point2i position;
	Size2i size;

	const Point2i &get_position() const { return position; }
	void set_position(const Point2i &p_position) { position = p_position; }
	const Size2i &get_size() const { return size; }
	void set_size(const Size2i &p_size) { size = p_size; }

	int get_area() const { return size.width * size.height; }

	inline bool intersects(const Rect2i &p_rect) const {
		if (position.x > (p_rect.position.x + p_rect.size.width)) {
			return false;
		}
		if ((position.x + size.width) < p_rect.position.x) {
			return false;
		}
		if (position.y > (p_rect.position.y + p_rect.size.height)) {
			return false;
		}
		if ((position.y + size.height) < p_rect.position.y) {
			return false;
		}

		return true;
	}

	inline bool encloses(const Rect2i &p_rect) const {
		return (p_rect.position.x >= position.x) && (p_rect.position.y >= position.y) &&
			   ((p_rect.position.x + p_rect.size.x) < (position.x + size.x)) &&
			   ((p_rect.position.y + p_rect.size.y) < (position.y + size.y));
	}

	_FORCE_INLINE_ bool has_no_area() const {
		return (size.x <= 0 || size.y <= 0);
	}
	inline Rect2i clip(const Rect2i &p_rect) const { /// return a clipped rect

		Rect2i new_rect = p_rect;

		if (!intersects(new_rect)) {
			return Rect2i();
		}

		new_rect.position.x = MAX(p_rect.position.x, position.x);
		new_rect.position.y = MAX(p_rect.position.y, position.y);

		Point2i p_rect_end = p_rect.position + p_rect.size;
		Point2i end = position + size;

		new_rect.size.x = (int)(MIN(p_rect_end.x, end.x) - new_rect.position.x);
		new_rect.size.y = (int)(MIN(p_rect_end.y, end.y) - new_rect.position.y);

		return new_rect;
	}

	inline Rect2i merge(const Rect2i &p_rect) const { ///< return a merged rect

		Rect2i new_rect;

		new_rect.position.x = MIN(p_rect.position.x, position.x);
		new_rect.position.y = MIN(p_rect.position.y, position.y);

		new_rect.size.x = MAX(p_rect.position.x + p_rect.size.x, position.x + size.x);
		new_rect.size.y = MAX(p_rect.position.y + p_rect.size.y, position.y + size.y);

		new_rect.size = new_rect.size - new_rect.position; //make relative again

		return new_rect;
	}
	bool has_point(const Point2i &p_point) const {
		if (p_point.x < position.x) {
			return false;
		}
		if (p_point.y < position.y) {
			return false;
		}

		if (p_point.x >= (position.x + size.x)) {
			return false;
		}
		if (p_point.y >= (position.y + size.y)) {
			return false;
		}

		return true;
	}

	bool operator==(const Rect2i &p_rect) const { return position == p_rect.position && size == p_rect.size; }
	bool operator!=(const Rect2i &p_rect) const { return position != p_rect.position || size != p_rect.size; }

	Rect2i grow(int p_by) const {
		Rect2i g = *this;
		g.position.x -= p_by;
		g.position.y -= p_by;
		g.size.width += p_by * 2;
		g.size.height += p_by * 2;
		return g;
	}

	inline Rect2i grow_margin(Margin p_margin, int p_amount) const {
		Rect2i g = *this;
		g = g.grow_individual((MARGIN_LEFT == p_margin) ? p_amount : 0,
				(MARGIN_TOP == p_margin) ? p_amount : 0,
				(MARGIN_RIGHT == p_margin) ? p_amount : 0,
				(MARGIN_BOTTOM == p_margin) ? p_amount : 0);
		return g;
	}

	inline Rect2i grow_margin_bind(uint32_t p_margin, int p_amount) const {
		return grow_margin(Margin(p_margin), p_amount);
	}

	inline Rect2i grow_individual(int p_left, int p_top, int p_right, int p_bottom) const {
		Rect2i g = *this;
		g.position.x -= p_left;
		g.position.y -= p_top;
		g.size.width += p_left + p_right;
		g.size.height += p_top + p_bottom;

		return g;
	}

	_FORCE_INLINE_ Rect2i expand(const Vector2i &p_vector) const {
		Rect2i r = *this;
		r.expand_to(p_vector);
		return r;
	}

	inline void expand_to(const Point2i &p_vector) {
		Point2i begin = position;
		Point2i end = position + size;

		if (p_vector.x < begin.x) {
			begin.x = p_vector.x;
		}
		if (p_vector.y < begin.y) {
			begin.y = p_vector.y;
		}

		if (p_vector.x > end.x) {
			end.x = p_vector.x;
		}
		if (p_vector.y > end.y) {
			end.y = p_vector.y;
		}

		position = begin;
		size = end - begin;
	}

	_FORCE_INLINE_ Rect2i abs() const {
		return Rect2i(Point2i(position.x + MIN(size.x, 0), position.y + MIN(size.y, 0)), size.abs());
	}

	operator String() const { return String(position) + ", " + String(size); }

	operator Rect2() const { return Rect2(position, size); }

	Rect2i() {}
	Rect2i(const Rect2 &p_r2) :
			position(p_r2.position),
			size(p_r2.size) {
	}
	Rect2i(int p_x, int p_y, int p_width, int p_height) :
			position(Point2i(p_x, p_y)),
			size(Size2i(p_width, p_height)) {
	}
	Rect2i(const Point2i &p_pos, const Size2i &p_size) :
			position(p_pos),
			size(p_size) {
	}
};

#endif // RECT2_H
