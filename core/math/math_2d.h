/*************************************************************************/
/*  math_2d.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef MATH_2D_H
#define MATH_2D_H

#include "math_funcs.h"
#include "ustring.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
enum Margin {

	MARGIN_LEFT,
	MARGIN_TOP,
	MARGIN_RIGHT,
	MARGIN_BOTTOM
};

enum Orientation {

	HORIZONTAL,
	VERTICAL
};

enum HAlign {

	HALIGN_LEFT,
	HALIGN_CENTER,
	HALIGN_RIGHT
};

enum VAlign {

	VALIGN_TOP,
	VALIGN_CENTER,
	VALIGN_BOTTOM
};

struct Vector2 {

	union {
		float x;
		float width;
	};
	union {
		float y;
		float height;
	};

	_FORCE_INLINE_ float &operator[](int p_idx) {
		return p_idx ? y : x;
	}
	_FORCE_INLINE_ const float &operator[](int p_idx) const {
		return p_idx ? y : x;
	}

	void normalize();
	Vector2 normalized() const;

	float length() const;
	float length_squared() const;

	float distance_to(const Vector2 &p_vector2) const;
	float distance_squared_to(const Vector2 &p_vector2) const;
	float angle_to(const Vector2 &p_vector2) const;
	float angle_to_point(const Vector2 &p_vector2) const;

	float dot(const Vector2 &p_other) const;
	float cross(const Vector2 &p_other) const;
	Vector2 cross(real_t p_other) const;
	Vector2 project(const Vector2 &p_vec) const;

	Vector2 plane_project(real_t p_d, const Vector2 &p_vec) const;

	Vector2 clamped(real_t p_len) const;

	_FORCE_INLINE_ static Vector2 linear_interpolate(const Vector2 &p_a, const Vector2 &p_b, float p_t);
	_FORCE_INLINE_ Vector2 linear_interpolate(const Vector2 &p_b, float p_t) const;
	Vector2 cubic_interpolate(const Vector2 &p_b, const Vector2 &p_pre_a, const Vector2 &p_post_b, float p_t) const;
	Vector2 cubic_interpolate_soft(const Vector2 &p_b, const Vector2 &p_pre_a, const Vector2 &p_post_b, float p_t) const;

	Vector2 slide(const Vector2 &p_vec) const;
	Vector2 reflect(const Vector2 &p_vec) const;

	Vector2 operator+(const Vector2 &p_v) const;
	void operator+=(const Vector2 &p_v);
	Vector2 operator-(const Vector2 &p_v) const;
	void operator-=(const Vector2 &p_v);
	Vector2 operator*(const Vector2 &p_v1) const;

	Vector2 operator*(const float &rvalue) const;
	void operator*=(const float &rvalue);
	void operator*=(const Vector2 &rvalue) { *this = *this * rvalue; }

	Vector2 operator/(const Vector2 &p_v1) const;

	Vector2 operator/(const float &rvalue) const;

	void operator/=(const float &rvalue);

	Vector2 operator-() const;

	bool operator==(const Vector2 &p_vec2) const;
	bool operator!=(const Vector2 &p_vec2) const;

	bool operator<(const Vector2 &p_vec2) const { return (x == p_vec2.x) ? (y < p_vec2.y) : (x < p_vec2.x); }
	bool operator<=(const Vector2 &p_vec2) const { return (x == p_vec2.x) ? (y <= p_vec2.y) : (x <= p_vec2.x); }

	real_t angle() const;

	void set_rotation(float p_radians) {

		x = Math::sin(p_radians);
		y = Math::cos(p_radians);
	}

	_FORCE_INLINE_ Vector2 abs() const {

		return Vector2(Math::abs(x), Math::abs(y));
	}

	Vector2 rotated(float p_by) const;
	Vector2 tangent() const {

		return Vector2(y, -x);
	}

	Vector2 floor() const;
	Vector2 snapped(const Vector2 &p_by) const;
	float get_aspect() const { return width / height; }

	operator String() const { return String::num(x) + ", " + String::num(y); }

	_FORCE_INLINE_ Vector2(float p_x, float p_y) {
		x = p_x;
		y = p_y;
	}
	_FORCE_INLINE_ Vector2() {
		x = 0;
		y = 0;
	}
};

_FORCE_INLINE_ Vector2 Vector2::plane_project(real_t p_d, const Vector2 &p_vec) const {

	return p_vec - *this * (dot(p_vec) - p_d);
}

_FORCE_INLINE_ Vector2 operator*(float p_scalar, const Vector2 &p_vec) {

	return p_vec * p_scalar;
}

Vector2 Vector2::linear_interpolate(const Vector2 &p_b, float p_t) const {

	Vector2 res = *this;

	res.x += (p_t * (p_b.x - x));
	res.y += (p_t * (p_b.y - y));

	return res;
}

Vector2 Vector2::linear_interpolate(const Vector2 &p_a, const Vector2 &p_b, float p_t) {

	Vector2 res = p_a;

	res.x += (p_t * (p_b.x - p_a.x));
	res.y += (p_t * (p_b.y - p_a.y));

	return res;
}

typedef Vector2 Size2;
typedef Vector2 Point2;

struct Matrix32;

struct Rect2 {

	Point2 pos;
	Size2 size;

	const Vector2 &get_pos() const { return pos; }
	void set_pos(const Vector2 &p_pos) { pos = p_pos; }
	const Vector2 &get_size() const { return size; }
	void set_size(const Vector2 &p_size) { size = p_size; }

	float get_area() const { return size.width * size.height; }

	inline bool intersects(const Rect2 &p_rect) const {
		if (pos.x >= (p_rect.pos.x + p_rect.size.width))
			return false;
		if ((pos.x + size.width) <= p_rect.pos.x)
			return false;
		if (pos.y >= (p_rect.pos.y + p_rect.size.height))
			return false;
		if ((pos.y + size.height) <= p_rect.pos.y)
			return false;

		return true;
	}

	inline float distance_to(const Vector2 &p_point) const {

		float dist = 1e20;

		if (p_point.x < pos.x) {
			dist = MIN(dist, pos.x - p_point.x);
		}
		if (p_point.y < pos.y) {
			dist = MIN(dist, pos.y - p_point.y);
		}
		if (p_point.x >= (pos.x + size.x)) {
			dist = MIN(p_point.x - (pos.x + size.x), dist);
		}
		if (p_point.y >= (pos.y + size.y)) {
			dist = MIN(p_point.y - (pos.y + size.y), dist);
		}

		if (dist == 1e20)
			return 0;
		else
			return dist;
	}

	_FORCE_INLINE_ bool intersects_transformed(const Matrix32 &p_xform, const Rect2 &p_rect) const;

	bool intersects_segment(const Point2 &p_from, const Point2 &p_to, Point2 *r_pos = NULL, Point2 *r_normal = NULL) const;

	inline bool encloses(const Rect2 &p_rect) const {

		return (p_rect.pos.x >= pos.x) && (p_rect.pos.y >= pos.y) &&
			   ((p_rect.pos.x + p_rect.size.x) < (pos.x + size.x)) &&
			   ((p_rect.pos.y + p_rect.size.y) < (pos.y + size.y));
	}

	inline bool has_no_area() const {

		return (size.x <= 0 || size.y <= 0);
	}
	inline Rect2 clip(const Rect2 &p_rect) const { /// return a clipped rect

		Rect2 new_rect = p_rect;

		if (!intersects(new_rect))
			return Rect2();

		new_rect.pos.x = MAX(p_rect.pos.x, pos.x);
		new_rect.pos.y = MAX(p_rect.pos.y, pos.y);

		Point2 p_rect_end = p_rect.pos + p_rect.size;
		Point2 end = pos + size;

		new_rect.size.x = MIN(p_rect_end.x, end.x) - new_rect.pos.x;
		new_rect.size.y = MIN(p_rect_end.y, end.y) - new_rect.pos.y;

		return new_rect;
	}

	inline Rect2 merge(const Rect2 &p_rect) const { ///< return a merged rect

		Rect2 new_rect;

		new_rect.pos.x = MIN(p_rect.pos.x, pos.x);
		new_rect.pos.y = MIN(p_rect.pos.y, pos.y);

		new_rect.size.x = MAX(p_rect.pos.x + p_rect.size.x, pos.x + size.x);
		new_rect.size.y = MAX(p_rect.pos.y + p_rect.size.y, pos.y + size.y);

		new_rect.size = new_rect.size - new_rect.pos; //make relative again

		return new_rect;
	};
	inline bool has_point(const Point2 &p_point) const {
		if (p_point.x < pos.x)
			return false;
		if (p_point.y < pos.y)
			return false;

		if (p_point.x >= (pos.x + size.x))
			return false;
		if (p_point.y >= (pos.y + size.y))
			return false;

		return true;
	}

	inline bool no_area() const { return (size.width <= 0 || size.height <= 0); }

	bool operator==(const Rect2 &p_rect) const { return pos == p_rect.pos && size == p_rect.size; }
	bool operator!=(const Rect2 &p_rect) const { return pos != p_rect.pos || size != p_rect.size; }

	inline Rect2 grow(real_t p_by) const {

		Rect2 g = *this;
		g.pos.x -= p_by;
		g.pos.y -= p_by;
		g.size.width += p_by * 2;
		g.size.height += p_by * 2;
		return g;
	}
	inline Rect2 grow_margin(Margin p_margin, real_t p_amount) const {
		Rect2 g = *this;
		g.grow_individual((MARGIN_LEFT == p_margin) ? p_amount : 0,
				(MARGIN_TOP == p_margin) ? p_amount : 0,
				(MARGIN_RIGHT == p_margin) ? p_amount : 0,
				(MARGIN_BOTTOM == p_margin) ? p_amount : 0);
		return g;
	}

	inline Rect2 grow_individual(real_t p_left, real_t p_top, real_t p_right, real_t p_bottom) const {

		Rect2 g = *this;
		g.pos.x -= p_left;
		g.pos.y -= p_top;
		g.size.width += p_left + p_right;
		g.size.height += p_top + p_bottom;

		return g;
	}

	inline Rect2 expand(const Vector2 &p_vector) const {

		Rect2 r = *this;
		r.expand_to(p_vector);
		return r;
	}

	inline void expand_to(const Vector2 &p_vector) { //in place function for speed

		Vector2 begin = pos;
		Vector2 end = pos + size;

		if (p_vector.x < begin.x)
			begin.x = p_vector.x;
		if (p_vector.y < begin.y)
			begin.y = p_vector.y;

		if (p_vector.x > end.x)
			end.x = p_vector.x;
		if (p_vector.y > end.y)
			end.y = p_vector.y;

		pos = begin;
		size = end - begin;
	}

	operator String() const { return String(pos) + ", " + String(size); }

	Rect2() {}
	Rect2(float p_x, float p_y, float p_width, float p_height) {
		pos = Point2(p_x, p_y);
		size = Size2(p_width, p_height);
	}
	Rect2(const Point2 &p_pos, const Size2 &p_size) {
		pos = p_pos;
		size = p_size;
	}
};

/* INTEGER STUFF */

struct Point2i {

	union {
		int x;
		int width;
	};
	union {
		int y;
		int height;
	};

	_FORCE_INLINE_ int &operator[](int p_idx) {
		return p_idx ? y : x;
	}
	_FORCE_INLINE_ const int &operator[](int p_idx) const {
		return p_idx ? y : x;
	}

	Point2i operator+(const Point2i &p_v) const;
	void operator+=(const Point2i &p_v);
	Point2i operator-(const Point2i &p_v) const;
	void operator-=(const Point2i &p_v);
	Point2i operator*(const Point2i &p_v1) const;

	Point2i operator*(const int &rvalue) const;
	void operator*=(const int &rvalue);

	Point2i operator/(const Point2i &p_v1) const;

	Point2i operator/(const int &rvalue) const;

	void operator/=(const int &rvalue);

	Point2i operator-() const;
	bool operator<(const Point2i &p_vec2) const { return (x == p_vec2.x) ? (y < p_vec2.y) : (x < p_vec2.x); }
	bool operator>(const Point2i &p_vec2) const { return (x == p_vec2.x) ? (y > p_vec2.y) : (x > p_vec2.x); }

	bool operator==(const Point2i &p_vec2) const;
	bool operator!=(const Point2i &p_vec2) const;

	float get_aspect() const { return width / (float)height; }

	operator String() const { return String::num(x) + ", " + String::num(y); }

	operator Vector2() const { return Vector2(x, y); }
	inline Point2i(const Vector2 &p_vec2) {
		x = (int)p_vec2.x;
		y = (int)p_vec2.y;
	}
	inline Point2i(int p_x, int p_y) {
		x = p_x;
		y = p_y;
	}
	inline Point2i() {
		x = 0;
		y = 0;
	}
};

typedef Point2i Size2i;

struct Rect2i {

	Point2i pos;
	Size2i size;

	const Point2i &get_pos() const { return pos; }
	void set_pos(const Point2i &p_pos) { pos = p_pos; }
	const Point2i &get_size() const { return size; }
	void set_size(const Point2i &p_size) { size = p_size; }

	int get_area() const { return size.width * size.height; }

	inline bool intersects(const Rect2i &p_rect) const {
		if (pos.x > (p_rect.pos.x + p_rect.size.width))
			return false;
		if ((pos.x + size.width) < p_rect.pos.x)
			return false;
		if (pos.y > (p_rect.pos.y + p_rect.size.height))
			return false;
		if ((pos.y + size.height) < p_rect.pos.y)
			return false;

		return true;
	}

	inline bool encloses(const Rect2i &p_rect) const {

		return (p_rect.pos.x >= pos.x) && (p_rect.pos.y >= pos.y) &&
			   ((p_rect.pos.x + p_rect.size.x) < (pos.x + size.x)) &&
			   ((p_rect.pos.y + p_rect.size.y) < (pos.y + size.y));
	}

	inline bool has_no_area() const {

		return (size.x <= 0 || size.y <= 0);
	}
	inline Rect2i clip(const Rect2i &p_rect) const { /// return a clipped rect

		Rect2i new_rect = p_rect;

		if (!intersects(new_rect))
			return Rect2i();

		new_rect.pos.x = MAX(p_rect.pos.x, pos.x);
		new_rect.pos.y = MAX(p_rect.pos.y, pos.y);

		Point2 p_rect_end = p_rect.pos + p_rect.size;
		Point2 end = pos + size;

		new_rect.size.x = (int)(MIN(p_rect_end.x, end.x) - new_rect.pos.x);
		new_rect.size.y = (int)(MIN(p_rect_end.y, end.y) - new_rect.pos.y);

		return new_rect;
	}

	inline Rect2i merge(const Rect2i &p_rect) const { ///< return a merged rect

		Rect2i new_rect;

		new_rect.pos.x = MIN(p_rect.pos.x, pos.x);
		new_rect.pos.y = MIN(p_rect.pos.y, pos.y);

		new_rect.size.x = MAX(p_rect.pos.x + p_rect.size.x, pos.x + size.x);
		new_rect.size.y = MAX(p_rect.pos.y + p_rect.size.y, pos.y + size.y);

		new_rect.size = new_rect.size - new_rect.pos; //make relative again

		return new_rect;
	};
	bool has_point(const Point2 &p_point) const {
		if (p_point.x < pos.x)
			return false;
		if (p_point.y < pos.y)
			return false;

		if (p_point.x >= (pos.x + size.x))
			return false;
		if (p_point.y >= (pos.y + size.y))
			return false;

		return true;
	}

	bool no_area() { return (size.width <= 0 || size.height <= 0); }

	bool operator==(const Rect2i &p_rect) const { return pos == p_rect.pos && size == p_rect.size; }
	bool operator!=(const Rect2i &p_rect) const { return pos != p_rect.pos || size != p_rect.size; }

	Rect2i grow(int p_by) const {

		Rect2i g = *this;
		g.pos.x -= p_by;
		g.pos.y -= p_by;
		g.size.width += p_by * 2;
		g.size.height += p_by * 2;
		return g;
	}

	inline void expand_to(const Point2i &p_vector) {

		Point2i begin = pos;
		Point2i end = pos + size;

		if (p_vector.x < begin.x)
			begin.x = p_vector.x;
		if (p_vector.y < begin.y)
			begin.y = p_vector.y;

		if (p_vector.x > end.x)
			end.x = p_vector.x;
		if (p_vector.y > end.y)
			end.y = p_vector.y;

		pos = begin;
		size = end - begin;
	}

	operator String() const { return String(pos) + ", " + String(size); }

	operator Rect2() const { return Rect2(pos, size); }
	Rect2i(const Rect2 &p_r2) {
		pos = p_r2.pos;
		size = p_r2.size;
	}
	Rect2i() {}
	Rect2i(int p_x, int p_y, int p_width, int p_height) {
		pos = Point2(p_x, p_y);
		size = Size2(p_width, p_height);
	}
	Rect2i(const Point2 &p_pos, const Size2 &p_size) {
		pos = p_pos;
		size = p_size;
	}
};

struct Matrix32 {

	Vector2 elements[3];

	_FORCE_INLINE_ float tdotx(const Vector2 &v) const { return elements[0][0] * v.x + elements[1][0] * v.y; }
	_FORCE_INLINE_ float tdoty(const Vector2 &v) const { return elements[0][1] * v.x + elements[1][1] * v.y; }

	const Vector2 &operator[](int p_idx) const { return elements[p_idx]; }
	Vector2 &operator[](int p_idx) { return elements[p_idx]; }

	_FORCE_INLINE_ Vector2 get_axis(int p_axis) const {
		ERR_FAIL_INDEX_V(p_axis, 3, Vector2());
		return elements[p_axis];
	}
	_FORCE_INLINE_ void set_axis(int p_axis, const Vector2 &p_vec) {
		ERR_FAIL_INDEX(p_axis, 3);
		elements[p_axis] = p_vec;
	}

	void invert();
	Matrix32 inverse() const;

	void affine_invert();
	Matrix32 affine_inverse() const;

	void set_rotation(real_t p_phi);
	real_t get_rotation() const;
	_FORCE_INLINE_ void set_rotation_and_scale(real_t p_phi, const Size2 &p_scale);
	void rotate(real_t p_phi);

	void scale(const Size2 &p_scale);
	void scale_basis(const Size2 &p_scale);
	void translate(real_t p_tx, real_t p_ty);
	void translate(const Vector2 &p_translation);

	float basis_determinant() const;

	Size2 get_scale() const;

	_FORCE_INLINE_ const Vector2 &get_origin() const { return elements[2]; }
	_FORCE_INLINE_ void set_origin(const Vector2 &p_origin) { elements[2] = p_origin; }

	Matrix32 scaled(const Size2 &p_scale) const;
	Matrix32 basis_scaled(const Size2 &p_scale) const;
	Matrix32 translated(const Vector2 &p_offset) const;
	Matrix32 rotated(float p_phi) const;

	Matrix32 untranslated() const;

	void orthonormalize();
	Matrix32 orthonormalized() const;

	bool operator==(const Matrix32 &p_transform) const;
	bool operator!=(const Matrix32 &p_transform) const;

	void operator*=(const Matrix32 &p_transform);
	Matrix32 operator*(const Matrix32 &p_transform) const;

	Matrix32 interpolate_with(const Matrix32 &p_transform, float p_c) const;

	_FORCE_INLINE_ Vector2 basis_xform(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Vector2 basis_xform_inv(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Vector2 xform(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Vector2 xform_inv(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Rect2 xform(const Rect2 &p_vec) const;
	_FORCE_INLINE_ Rect2 xform_inv(const Rect2 &p_vec) const;

	operator String() const;

	Matrix32(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy) {

		elements[0][0] = xx;
		elements[0][1] = xy;
		elements[1][0] = yx;
		elements[1][1] = yy;
		elements[2][0] = ox;
		elements[2][1] = oy;
	}

	Matrix32(real_t p_rot, const Vector2 &p_pos);
	Matrix32() {
		elements[0][0] = 1.0;
		elements[1][1] = 1.0;
	}
};

bool Rect2::intersects_transformed(const Matrix32 &p_xform, const Rect2 &p_rect) const {

	//SAT intersection between local and transformed rect2

	Vector2 xf_points[4] = {
		p_xform.xform(p_rect.pos),
		p_xform.xform(Vector2(p_rect.pos.x + p_rect.size.x, p_rect.pos.y)),
		p_xform.xform(Vector2(p_rect.pos.x, p_rect.pos.y + p_rect.size.y)),
		p_xform.xform(Vector2(p_rect.pos.x + p_rect.size.x, p_rect.pos.y + p_rect.size.y)),
	};

	real_t low_limit;

	//base rect2 first (faster)

	if (xf_points[0].y > pos.y)
		goto next1;
	if (xf_points[1].y > pos.y)
		goto next1;
	if (xf_points[2].y > pos.y)
		goto next1;
	if (xf_points[3].y > pos.y)
		goto next1;

	return false;

next1:

	low_limit = pos.y + size.y;

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

	if (xf_points[0].x > pos.x)
		goto next3;
	if (xf_points[1].x > pos.x)
		goto next3;
	if (xf_points[2].x > pos.x)
		goto next3;
	if (xf_points[3].x > pos.x)
		goto next3;

	return false;

next3:

	low_limit = pos.x + size.x;

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
		pos,
		Vector2(pos.x + size.x, pos.y),
		Vector2(pos.x, pos.y + size.y),
		Vector2(pos.x + size.x, pos.y + size.y),
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

Vector2 Matrix32::basis_xform(const Vector2 &v) const {

	return Vector2(
			tdotx(v),
			tdoty(v));
}

Vector2 Matrix32::basis_xform_inv(const Vector2 &v) const {

	return Vector2(
			elements[0].dot(v),
			elements[1].dot(v));
}

Vector2 Matrix32::xform(const Vector2 &v) const {

	return Vector2(
				   tdotx(v),
				   tdoty(v)) +
		   elements[2];
}
Vector2 Matrix32::xform_inv(const Vector2 &p_vec) const {

	Vector2 v = p_vec - elements[2];

	return Vector2(
			elements[0].dot(v),
			elements[1].dot(v));
}
Rect2 Matrix32::xform(const Rect2 &p_rect) const {

	Vector2 x = elements[0] * p_rect.size.x;
	Vector2 y = elements[1] * p_rect.size.y;
	Vector2 pos = xform(p_rect.pos);

	Rect2 new_rect;
	new_rect.pos = pos;
	new_rect.expand_to(pos + x);
	new_rect.expand_to(pos + y);
	new_rect.expand_to(pos + x + y);
	return new_rect;
}

void Matrix32::set_rotation_and_scale(real_t p_rot, const Size2 &p_scale) {

	elements[0][0] = Math::cos(p_rot) * p_scale.x;
	elements[1][1] = Math::cos(p_rot) * p_scale.y;
	elements[0][1] = -Math::sin(p_rot) * p_scale.x;
	elements[1][0] = Math::sin(p_rot) * p_scale.y;
}

Rect2 Matrix32::xform_inv(const Rect2 &p_rect) const {

	Vector2 ends[4] = {
		xform_inv(p_rect.pos),
		xform_inv(Vector2(p_rect.pos.x, p_rect.pos.y + p_rect.size.y)),
		xform_inv(Vector2(p_rect.pos.x + p_rect.size.x, p_rect.pos.y + p_rect.size.y)),
		xform_inv(Vector2(p_rect.pos.x + p_rect.size.x, p_rect.pos.y))
	};

	Rect2 new_rect;
	new_rect.pos = ends[0];
	new_rect.expand_to(ends[1]);
	new_rect.expand_to(ends[2]);
	new_rect.expand_to(ends[3]);

	return new_rect;
}

#endif
