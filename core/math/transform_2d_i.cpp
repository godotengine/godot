/**************************************************************************/
/*  transform_2d_i.cpp                                                      */
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

#include "transform_2d_i.h"

#include "core/string/ustring.h"

void Transform2Di::invert() {
	// FIXME: this function assumes the basis is a rotation matrix, with no scaling.
	// Transform2Di::affine_inverse can handle matrices with scaling, so GDScript should eventually use that.
	SWAP(columns[0][1], columns[1][0]);
	columns[2] = basis_xform(-columns[2]);
}

Transform2Di Transform2Di::inverse() const {
	Transform2Di inv = *this;
	inv.invert();
	return inv;
}

void Transform2Di::affine_invert() {
	int32_t det = determinant();
#ifdef MATH_CHECKS
	ERR_FAIL_COND(det == 0);
#endif
	int32_t idet = 1 / det;

	SWAP(columns[0][0], columns[1][1]);
	columns[0] *= Vector2i(idet, -idet);
	columns[1] *= Vector2i(-idet, idet);

	columns[2] = basis_xform(-columns[2]);
}

Transform2Di Transform2Di::affine_inverse() const {
	Transform2Di inv = *this;
	inv.affine_invert();
	return inv;
}

Transform2Di::Transform2Di(const Vector2i &p_pos) {
	columns[0][0] = 1;
	columns[0][1] = 0;
	columns[1][0] = 0;
	columns[1][1] = 1;
	columns[2] = p_pos;
}

Transform2Di::Transform2Di(const Size2i &p_scale, const Vector2i &p_pos) {
	columns[0][0] = p_scale.x;
	columns[0][1] = 0;
	columns[1][0] = 0;
	columns[1][1] = p_scale.y;
	columns[2] = p_pos;
}

Size2i Transform2Di::get_scale() const {
	int32_t det_sign = SIGN(determinant());
	return Size2i(columns[0].length(), det_sign * columns[1].length());
}

void Transform2Di::set_scale(const Size2i &p_scale) {
	columns[0].normalize();
	columns[1].normalize();
	columns[0] *= p_scale.x;
	columns[1] *= p_scale.y;
}

void Transform2Di::scale(const Size2i &p_scale) {
	scale_basis(p_scale);
	columns[2] *= p_scale;
}

void Transform2Di::scale_basis(const Size2i &p_scale) {
	columns[0][0] *= p_scale.x;
	columns[0][1] *= p_scale.y;
	columns[1][0] *= p_scale.x;
	columns[1][1] *= p_scale.y;
}

void Transform2Di::translate_local(int32_t p_tx, int32_t p_ty) {
	translate_local(Vector2i(p_tx, p_ty));
}

void Transform2Di::translate_local(const Vector2i &p_translation) {
	columns[2] += basis_xform(p_translation);
}

void Transform2Di::orthonormalize() {
	// Gram-Schmidt Process

	Vector2i x = columns[0];
	Vector2i y = columns[1];

	x.normalize();
	y = y - x * x.dot(y);
	y.normalize();

	columns[0] = x;
	columns[1] = y;
}

Transform2Di Transform2Di::orthonormalized() const {
	Transform2Di ortho = *this;
	ortho.orthonormalize();
	return ortho;
}

bool Transform2Di::is_conformal() const {
	// Non-flipped case.
	if ((columns[0][0] == columns[1][1]) && (columns[0][1] == -columns[1][0])) {
		return true;
	}
	// Flipped case.
	if ((columns[0][0] == -columns[1][1]) && (columns[0][1] == columns[1][0])) {
		return true;
	}
	return false;
}

bool Transform2Di::is_equal(const Transform2Di &p_transform) const {
	return columns[0].is_equal(p_transform.columns[0]) && columns[1].is_equal(p_transform.columns[1]) && columns[2].is_equal(p_transform.columns[2]);
}

bool Transform2Di::operator==(const Transform2Di &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (columns[i] != p_transform.columns[i]) {
			return false;
		}
	}

	return true;
}

bool Transform2Di::operator!=(const Transform2Di &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (columns[i] != p_transform.columns[i]) {
			return true;
		}
	}

	return false;
}

void Transform2Di::operator*=(const Transform2Di &p_transform) {
	columns[2] = xform(p_transform.columns[2]);

	int32_t x0, x1, y0, y1;

	x0 = tdotx(p_transform.columns[0]);
	x1 = tdoty(p_transform.columns[0]);
	y0 = tdotx(p_transform.columns[1]);
	y1 = tdoty(p_transform.columns[1]);

	columns[0][0] = x0;
	columns[0][1] = x1;
	columns[1][0] = y0;
	columns[1][1] = y1;
}

Transform2Di Transform2Di::operator*(const Transform2Di &p_transform) const {
	Transform2Di t = *this;
	t *= p_transform;
	return t;
}

Transform2Di Transform2Di::scaled(const Size2i &p_scale) const {
	// Equivalent to left multiplication
	Transform2Di copy = *this;
	copy.scale(p_scale);
	return copy;
}

Transform2Di Transform2Di::scaled_local(const Size2i &p_scale) const {
	// Equivalent to right multiplication
	return Transform2Di(columns[0] * p_scale.x, columns[1] * p_scale.y, columns[2]);
}

Transform2Di Transform2Di::untranslated() const {
	Transform2Di copy = *this;
	copy.columns[2] = Vector2i();
	return copy;
}

Transform2Di Transform2Di::translated(const Vector2i &p_offset) const {
	// Equivalent to left multiplication
	return Transform2Di(columns[0], columns[1], columns[2] + p_offset);
}

Transform2Di Transform2Di::translated_local(const Vector2i &p_offset) const {
	// Equivalent to right multiplication
	return Transform2Di(columns[0], columns[1], columns[2] + basis_xform(p_offset));
}

int32_t Transform2Di::determinant() const {
	return columns[0].x * columns[1].y - columns[0].y * columns[1].x;
}

Transform2Di Transform2Di::interpolate_with(const Transform2Di &p_transform, real_t p_weight) const {
	return Transform2Di(
			get_scale().lerp(p_transform.get_scale(), p_weight),
			get_origin().lerp(p_transform.get_origin(), p_weight));
}

void Transform2Di::operator*=(int32_t p_val) {
	columns[0] *= p_val;
	columns[1] *= p_val;
	columns[2] *= p_val;
}

Transform2Di Transform2Di::operator*(int32_t p_val) const {
	Transform2Di ret(*this);
	ret *= p_val;
	return ret;
}

void Transform2Di::operator/=(int32_t p_val) {
	columns[0] /= p_val;
	columns[1] /= p_val;
	columns[2] /= p_val;
}

Transform2Di Transform2Di::operator/(int32_t p_val) const {
	Transform2Di ret(*this);
	ret /= p_val;
	return ret;
}

Transform2Di::operator String() const {
	return "[X: " + columns[0].operator String() +
			", Y: " + columns[1].operator String() +
			", O: " + columns[2].operator String() + "]";
}
