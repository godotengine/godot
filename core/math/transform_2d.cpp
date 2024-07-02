/**************************************************************************/
/*  transform_2d.cpp                                                      */
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

#include "transform_2d.h"

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
	real_t det = determinant();
#ifdef MATH_CHECKS
	ERR_FAIL_COND(det == 0);
#endif
	real_t idet = 1 / det;

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

void Transform2D::rotate(real_t p_angle) {
	*this = Transform2D(p_angle, Vector2()) * (*this);
}

real_t Transform2D::get_rotation() const {
	return Math::atan2(elements[0].y, elements[0].x);
}

void Transform2D::set_rotation(real_t p_rot) {
	Size2 scale = get_scale();
	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	elements[0][0] = cr;
	elements[0][1] = sr;
	elements[1][0] = -sr;
	elements[1][1] = cr;
	set_scale(scale);
}

void Transform2D::set_skew(real_t p_angle) {
	real_t det = determinant();
	elements[1] = SGN(det) * elements[0].rotated(((real_t)Math_PI * 0.5f + p_angle)).normalized() * elements[1].length();
}

real_t Transform2D::get_skew() const {
	real_t det = determinant();
	return Math::acos(elements[0].normalized().dot(SGN(det) * elements[1].normalized())) - (real_t)Math_PI * 0.5f;
}

Transform2D::Transform2D(real_t p_rot, const Size2 &p_scale, real_t p_skew, const Vector2 &p_pos) {
	elements[0][0] = Math::cos(p_rot) * p_scale.x;
	elements[1][1] = Math::cos(p_rot + p_skew) * p_scale.y;
	elements[1][0] = -Math::sin(p_rot + p_skew) * p_scale.y;
	elements[0][1] = Math::sin(p_rot) * p_scale.x;
	elements[2] = p_pos;
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
	real_t det_sign = SGN(determinant());
	return Size2(elements[0].length(), det_sign * elements[1].length());
}

void Transform2D::set_scale(const Size2 &p_scale) {
	elements[0].normalize();
	elements[1].normalize();
	elements[0] *= p_scale.x;
	elements[1] *= p_scale.y;
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

bool Transform2D::is_equal_approx(const Transform2D &p_transform) const {
	return elements[0].is_equal_approx(p_transform.elements[0]) && elements[1].is_equal_approx(p_transform.elements[1]) && elements[2].is_equal_approx(p_transform.elements[2]);
}

bool Transform2D::operator==(const Transform2D &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (elements[i] != p_transform.elements[i]) {
			return false;
		}
	}

	return true;
}

bool Transform2D::operator!=(const Transform2D &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (elements[i] != p_transform.elements[i]) {
			return true;
		}
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

Transform2D Transform2D::rotated(real_t p_angle) const {
	Transform2D copy = *this;
	copy.rotate(p_angle);
	return copy;
}

real_t Transform2D::determinant() const {
	return elements[0].x * elements[1].y - elements[0].y * elements[1].x;
}

Transform2D::operator String() const {
	return String(String() + elements[0] + ", " + elements[1] + ", " + elements[2]);
}
