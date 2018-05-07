/*************************************************************************/
/*  math_2d.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "math_2d.h"

bool Rect2::intersects_segment(const Point2 &p_from, const Point2 &p_to, Point2 *r_pos, Point2 *r_normal) const {

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

	if (r_pos)
		*r_pos = p_from + rel * min;

	return true;
}

/* Point2i */

Point2i Point2i::operator+(const Point2i &p_v) const {

	return Point2i(x + p_v.x, y + p_v.y);
}
void Point2i::operator+=(const Point2i &p_v) {

	x += p_v.x;
	y += p_v.y;
}
Point2i Point2i::operator-(const Point2i &p_v) const {

	return Point2i(x - p_v.x, y - p_v.y);
}
void Point2i::operator-=(const Point2i &p_v) {

	x -= p_v.x;
	y -= p_v.y;
}

Point2i Point2i::operator*(const Point2i &p_v1) const {

	return Point2i(x * p_v1.x, y * p_v1.y);
};

Point2i Point2i::operator*(const int &rvalue) const {

	return Point2i(x * rvalue, y * rvalue);
};
void Point2i::operator*=(const int &rvalue) {

	x *= rvalue;
	y *= rvalue;
};

Point2i Point2i::operator/(const Point2i &p_v1) const {

	return Point2i(x / p_v1.x, y / p_v1.y);
};

Point2i Point2i::operator/(const int &rvalue) const {

	return Point2i(x / rvalue, y / rvalue);
};

void Point2i::operator/=(const int &rvalue) {

	x /= rvalue;
	y /= rvalue;
};

Point2i Point2i::operator-() const {

	return Point2i(-x, -y);
}

bool Point2i::operator==(const Point2i &p_vec2) const {

	return x == p_vec2.x && y == p_vec2.y;
}
bool Point2i::operator!=(const Point2i &p_vec2) const {

	return x != p_vec2.x || y != p_vec2.y;
}

void Transform2D::invert() {
	// FIXME: this function assumes the basis is a rotation matrix, with no scaling.
	// Transform2D::affine_inverse can handle matrices with scaling, so GDScript should eventually use that.
	SWAP(elements[0][1], elements[1][0]);
	elements[2] = basis_xform(-elements[2]);
}

Transform2D Transform2D::inverse() const {

	Transform2D inv = *this;
	inv.invert();
	return inv;
}

void Transform2D::affine_invert() {

	real_t det = basis_determinant();
#ifdef MATH_CHECKS
	ERR_FAIL_COND(det == 0);
#endif
	real_t idet = 1.0 / det;

	SWAP(elements[0][0], elements[1][1]);
	elements[0] *= Vector2(idet, -idet);
	elements[1] *= Vector2(-idet, idet);

	elements[2] = basis_xform(-elements[2]);
}

Transform2D Transform2D::affine_inverse() const {

	Transform2D inv = *this;
	inv.affine_invert();
	return inv;
}

void Transform2D::rotate(real_t p_phi) {
	*this = Transform2D(p_phi, Vector2()) * (*this);
}

real_t Transform2D::get_rotation() const {
	real_t det = basis_determinant();
	Transform2D m = orthonormalized();
	if (det < 0) {
		m.scale_basis(Size2(1, -1)); // convention to separate rotation and reflection for 2D is to absorb a flip along y into scaling.
	}
	return Math::atan2(m[0].y, m[0].x);
}

void Transform2D::set_rotation(real_t p_rot) {

	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	elements[0][0] = cr;
	elements[0][1] = sr;
	elements[1][0] = -sr;
	elements[1][1] = cr;
}

Transform2D::Transform2D(real_t p_rot, const Vector2 &p_pos) {

	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	elements[0][0] = cr;
	elements[0][1] = sr;
	elements[1][0] = -sr;
	elements[1][1] = cr;
	elements[2] = p_pos;
}

Size2 Transform2D::get_scale() const {
	real_t det_sign = basis_determinant() > 0 ? 1 : -1;
	return Size2(elements[0].length(), det_sign * elements[1].length());
}

void Transform2D::scale(const Size2 &p_scale) {
	scale_basis(p_scale);
	elements[2] *= p_scale;
}
void Transform2D::scale_basis(const Size2 &p_scale) {

	elements[0][0] *= p_scale.x;
	elements[0][1] *= p_scale.y;
	elements[1][0] *= p_scale.x;
	elements[1][1] *= p_scale.y;
}
void Transform2D::translate(real_t p_tx, real_t p_ty) {

	translate(Vector2(p_tx, p_ty));
}
void Transform2D::translate(const Vector2 &p_translation) {

	elements[2] += basis_xform(p_translation);
}

void Transform2D::orthonormalize() {

	// Gram-Schmidt Process

	Vector2 x = elements[0];
	Vector2 y = elements[1];

	x.normalize();
	y = (y - x * (x.dot(y)));
	y.normalize();

	elements[0] = x;
	elements[1] = y;
}
Transform2D Transform2D::orthonormalized() const {

	Transform2D on = *this;
	on.orthonormalize();
	return on;
}

bool Transform2D::operator==(const Transform2D &p_transform) const {

	for (int i = 0; i < 3; i++) {
		if (elements[i] != p_transform.elements[i])
			return false;
	}

	return true;
}

bool Transform2D::operator!=(const Transform2D &p_transform) const {

	for (int i = 0; i < 3; i++) {
		if (elements[i] != p_transform.elements[i])
			return true;
	}

	return false;
}

void Transform2D::operator*=(const Transform2D &p_transform) {

	elements[2] = xform(p_transform.elements[2]);

	real_t x0, x1, y0, y1;

	x0 = tdotx(p_transform.elements[0]);
	x1 = tdoty(p_transform.elements[0]);
	y0 = tdotx(p_transform.elements[1]);
	y1 = tdoty(p_transform.elements[1]);

	elements[0][0] = x0;
	elements[0][1] = x1;
	elements[1][0] = y0;
	elements[1][1] = y1;
}

Transform2D Transform2D::operator*(const Transform2D &p_transform) const {

	Transform2D t = *this;
	t *= p_transform;
	return t;
}

Transform2D Transform2D::scaled(const Size2 &p_scale) const {

	Transform2D copy = *this;
	copy.scale(p_scale);
	return copy;
}

Transform2D Transform2D::basis_scaled(const Size2 &p_scale) const {

	Transform2D copy = *this;
	copy.scale_basis(p_scale);
	return copy;
}

Transform2D Transform2D::untranslated() const {

	Transform2D copy = *this;
	copy.elements[2] = Vector2();
	return copy;
}

Transform2D Transform2D::translated(const Vector2 &p_offset) const {

	Transform2D copy = *this;
	copy.translate(p_offset);
	return copy;
}

Transform2D Transform2D::rotated(real_t p_phi) const {

	Transform2D copy = *this;
	copy.rotate(p_phi);
	return copy;
}

real_t Transform2D::basis_determinant() const {

	return elements[0].x * elements[1].y - elements[0].y * elements[1].x;
}

Transform2D Transform2D::interpolate_with(const Transform2D &p_transform, real_t p_c) const {

	//extract parameters
	Vector2 p1 = get_origin();
	Vector2 p2 = p_transform.get_origin();

	real_t r1 = get_rotation();
	real_t r2 = p_transform.get_rotation();

	Size2 s1 = get_scale();
	Size2 s2 = p_transform.get_scale();

	//slerp rotation
	Vector2 v1(Math::cos(r1), Math::sin(r1));
	Vector2 v2(Math::cos(r2), Math::sin(r2));

	real_t dot = v1.dot(v2);

	dot = (dot < -1.0) ? -1.0 : ((dot > 1.0) ? 1.0 : dot); //clamp dot to [-1,1]

	Vector2 v;

	if (dot > 0.9995) {
		v = Vector2::linear_interpolate(v1, v2, p_c).normalized(); //linearly interpolate to avoid numerical precision issues
	} else {
		real_t angle = p_c * Math::acos(dot);
		Vector2 v3 = (v2 - v1 * dot).normalized();
		v = v1 * Math::cos(angle) + v3 * Math::sin(angle);
	}

	//construct matrix
	Transform2D res(Math::atan2(v.y, v.x), Vector2::linear_interpolate(p1, p2, p_c));
	res.scale_basis(Vector2::linear_interpolate(s1, s2, p_c));
	return res;
}

Transform2D::operator String() const {

	return String(String() + elements[0] + ", " + elements[1] + ", " + elements[2]);
}
