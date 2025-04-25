/**************************************************************************/
/*  transform_2d.h                                                        */
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

#include "core/math/math_funcs.h"
#include "core/math/rect2.h"
#include "core/math/vector2.h"
#include "core/templates/vector.h"

struct [[nodiscard]] Transform2D {
	// WARNING: The basis of Transform2D is stored differently from Basis.
	// In terms of columns array, the basis matrix looks like "on paper":
	// M = (columns[0][0] columns[1][0])
	//     (columns[0][1] columns[1][1])
	// This is such that the columns, which can be interpreted as basis vectors
	// of the coordinate system "painted" on the object, can be accessed as columns[i].
	// NOTE: This is the opposite of the indices in mathematical texts,
	// meaning: $M_{12}$ in a math book corresponds to columns[1][0] here.
	// This requires additional care when working with explicit indices.
	// See https://en.wikipedia.org/wiki/Row-_and_column-major_order for further reading.

	// WARNING: Be aware that unlike 3D code, 2D code uses a left-handed coordinate system:
	// Y-axis points down, and angle is measure from +X to +Y in a clockwise-fashion.

	Vector2 columns[3] = {
		{ 1, 0 },
		{ 0, 1 },
		{ 0, 0 },
	};

	_FORCE_INLINE_ real_t tdotx(const Vector2 &p_v) const { return columns[0][0] * p_v.x + columns[1][0] * p_v.y; }
	_FORCE_INLINE_ real_t tdoty(const Vector2 &p_v) const { return columns[0][1] * p_v.x + columns[1][1] * p_v.y; }

	_FORCE_INLINE_ constexpr const Vector2 &operator[](int p_idx) const { return columns[p_idx]; }
	_FORCE_INLINE_ constexpr Vector2 &operator[](int p_idx) { return columns[p_idx]; }

	_FORCE_INLINE_ void invert();
	_FORCE_INLINE_ Transform2D inverse() const;

	_FORCE_INLINE_ void affine_invert();
	_FORCE_INLINE_ Transform2D affine_inverse() const;

	_FORCE_INLINE_ void set_rotation(real_t p_rot);
	_FORCE_INLINE_ real_t get_rotation() const;
	_FORCE_INLINE_ real_t get_skew() const;
	_FORCE_INLINE_ void set_skew(real_t p_angle);
	_FORCE_INLINE_ void set_rotation_and_scale(real_t p_rot, const Size2 &p_scale);
	_FORCE_INLINE_ void set_rotation_scale_and_skew(real_t p_rot, const Size2 &p_scale, real_t p_skew);
	_FORCE_INLINE_ void rotate(real_t p_angle);

	_FORCE_INLINE_ void scale(const Size2 &p_scale);
	_FORCE_INLINE_ void scale_basis(const Size2 &p_scale);
	_FORCE_INLINE_ void translate_local(real_t p_tx, real_t p_ty);
	_FORCE_INLINE_ void translate_local(const Vector2 &p_translation);

	_FORCE_INLINE_ real_t determinant() const;

	_FORCE_INLINE_ Size2 get_scale() const;
	_FORCE_INLINE_ void set_scale(const Size2 &p_scale);

	_FORCE_INLINE_ const Vector2 &get_origin() const { return columns[2]; }
	_FORCE_INLINE_ void set_origin(const Vector2 &p_origin) { columns[2] = p_origin; }

	_FORCE_INLINE_ Transform2D scaled(const Size2 &p_scale) const;
	_FORCE_INLINE_ Transform2D scaled_local(const Size2 &p_scale) const;
	_FORCE_INLINE_ Transform2D translated(const Vector2 &p_offset) const;
	_FORCE_INLINE_ Transform2D translated_local(const Vector2 &p_offset) const;
	_FORCE_INLINE_ Transform2D rotated(real_t p_angle) const;
	_FORCE_INLINE_ Transform2D rotated_local(real_t p_angle) const;

	_FORCE_INLINE_ Transform2D untranslated() const;

	_FORCE_INLINE_ void orthonormalize();
	_FORCE_INLINE_ Transform2D orthonormalized() const;
	_FORCE_INLINE_ bool is_conformal() const;
	_FORCE_INLINE_ bool is_equal_approx(const Transform2D &p_transform) const;
	_FORCE_INLINE_ bool is_same(const Transform2D &p_transform) const;
	_FORCE_INLINE_ bool is_finite() const;

	_FORCE_INLINE_ Transform2D looking_at(const Vector2 &p_target) const;

	_FORCE_INLINE_ constexpr bool operator==(const Transform2D &p_transform) const;
	_FORCE_INLINE_ constexpr bool operator!=(const Transform2D &p_transform) const;

	_FORCE_INLINE_ void operator*=(const Transform2D &p_transform);
	_FORCE_INLINE_ Transform2D operator*(const Transform2D &p_transform) const;
	_FORCE_INLINE_ constexpr void operator*=(real_t p_val);
	_FORCE_INLINE_ constexpr Transform2D operator*(real_t p_val) const;
	_FORCE_INLINE_ constexpr void operator/=(real_t p_val);
	_FORCE_INLINE_ constexpr Transform2D operator/(real_t p_val) const;

	_FORCE_INLINE_ Transform2D interpolate_with(const Transform2D &p_transform, real_t p_c) const;

	_FORCE_INLINE_ Vector2 basis_xform(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Vector2 basis_xform_inv(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Vector2 xform(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Vector2 xform_inv(const Vector2 &p_vec) const;
	_FORCE_INLINE_ Rect2 xform(const Rect2 &p_rect) const;
	_FORCE_INLINE_ Rect2 xform_inv(const Rect2 &p_rect) const;
	_FORCE_INLINE_ Vector<Vector2> xform(const Vector<Vector2> &p_array) const;
	_FORCE_INLINE_ Vector<Vector2> xform_inv(const Vector<Vector2> &p_array) const;

	operator String() const;

	_FORCE_INLINE_ constexpr Transform2D(real_t p_xx, real_t p_xy, real_t p_yx, real_t p_yy, real_t p_ox, real_t p_oy) :
			columns{
				{ p_xx, p_xy },
				{ p_yx, p_yy },
				{ p_ox, p_oy },
			} {}

	_FORCE_INLINE_ constexpr Transform2D(const Vector2 &p_x, const Vector2 &p_y, const Vector2 &p_origin) :
			columns{ p_x, p_y, p_origin } {}

	_FORCE_INLINE_ Transform2D(real_t p_rot, const Vector2 &p_pos);

	_FORCE_INLINE_ Transform2D(real_t p_rot, const Size2 &p_scale, real_t p_skew, const Vector2 &p_pos);

	_FORCE_INLINE_ constexpr Transform2D() = default;
};

constexpr bool Transform2D::operator==(const Transform2D &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (columns[i] != p_transform.columns[i]) {
			return false;
		}
	}

	return true;
}

constexpr bool Transform2D::operator!=(const Transform2D &p_transform) const {
	for (int i = 0; i < 3; i++) {
		if (columns[i] != p_transform.columns[i]) {
			return true;
		}
	}

	return false;
}

constexpr void Transform2D::operator*=(real_t p_val) {
	columns[0] *= p_val;
	columns[1] *= p_val;
	columns[2] *= p_val;
}

constexpr Transform2D Transform2D::operator*(real_t p_val) const {
	Transform2D ret(*this);
	ret *= p_val;
	return ret;
}

constexpr void Transform2D::operator/=(real_t p_val) {
	columns[0] /= p_val;
	columns[1] /= p_val;
	columns[2] /= p_val;
}

constexpr Transform2D Transform2D::operator/(real_t p_val) const {
	Transform2D ret(*this);
	ret /= p_val;
	return ret;
}

Vector2 Transform2D::basis_xform(const Vector2 &p_vec) const {
	return Vector2(
			tdotx(p_vec),
			tdoty(p_vec));
}

Vector2 Transform2D::basis_xform_inv(const Vector2 &p_vec) const {
	return Vector2(
			columns[0].dot(p_vec),
			columns[1].dot(p_vec));
}

Vector2 Transform2D::xform(const Vector2 &p_vec) const {
	return Vector2(
				   tdotx(p_vec),
				   tdoty(p_vec)) +
			columns[2];
}

Vector2 Transform2D::xform_inv(const Vector2 &p_vec) const {
	Vector2 v = p_vec - columns[2];

	return Vector2(
			columns[0].dot(v),
			columns[1].dot(v));
}

Rect2 Transform2D::xform(const Rect2 &p_rect) const {
	Vector2 x = columns[0] * p_rect.size.x;
	Vector2 y = columns[1] * p_rect.size.y;
	Vector2 pos = xform(p_rect.position);

	Rect2 new_rect;
	new_rect.position = pos;
	new_rect.expand_to(pos + x);
	new_rect.expand_to(pos + y);
	new_rect.expand_to(pos + x + y);
	return new_rect;
}

void Transform2D::set_rotation_and_scale(real_t p_rot, const Size2 &p_scale) {
	columns[0][0] = Math::cos(p_rot) * p_scale.x;
	columns[1][1] = Math::cos(p_rot) * p_scale.y;
	columns[1][0] = -Math::sin(p_rot) * p_scale.y;
	columns[0][1] = Math::sin(p_rot) * p_scale.x;
}

void Transform2D::set_rotation_scale_and_skew(real_t p_rot, const Size2 &p_scale, real_t p_skew) {
	columns[0][0] = Math::cos(p_rot) * p_scale.x;
	columns[1][1] = Math::cos(p_rot + p_skew) * p_scale.y;
	columns[1][0] = -Math::sin(p_rot + p_skew) * p_scale.y;
	columns[0][1] = Math::sin(p_rot) * p_scale.x;
}

Rect2 Transform2D::xform_inv(const Rect2 &p_rect) const {
	Vector2 ends[4] = {
		xform_inv(p_rect.position),
		xform_inv(Vector2(p_rect.position.x, p_rect.position.y + p_rect.size.y)),
		xform_inv(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y + p_rect.size.y)),
		xform_inv(Vector2(p_rect.position.x + p_rect.size.x, p_rect.position.y))
	};

	Rect2 new_rect;
	new_rect.position = ends[0];
	new_rect.expand_to(ends[1]);
	new_rect.expand_to(ends[2]);
	new_rect.expand_to(ends[3]);

	return new_rect;
}

Vector<Vector2> Transform2D::xform(const Vector<Vector2> &p_array) const {
	Vector<Vector2> array;
	array.resize(p_array.size());

	const Vector2 *r = p_array.ptr();
	Vector2 *w = array.ptrw();

	for (int i = 0; i < p_array.size(); ++i) {
		w[i] = xform(r[i]);
	}
	return array;
}

Vector<Vector2> Transform2D::xform_inv(const Vector<Vector2> &p_array) const {
	Vector<Vector2> array;
	array.resize(p_array.size());

	const Vector2 *r = p_array.ptr();
	Vector2 *w = array.ptrw();

	for (int i = 0; i < p_array.size(); ++i) {
		w[i] = xform_inv(r[i]);
	}
	return array;
}

void Transform2D::invert() {
	// FIXME: this function assumes the basis is a rotation matrix, with no scaling.
	// Transform2D::affine_inverse can handle matrices with scaling, so GDScript should eventually use that.
	SWAP(columns[0][1], columns[1][0]);
	columns[2] = basis_xform(-columns[2]);
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
	real_t idet = 1.0f / det;

	SWAP(columns[0][0], columns[1][1]);
	columns[0] *= Vector2(idet, -idet);
	columns[1] *= Vector2(-idet, idet);

	columns[2] = basis_xform(-columns[2]);
}

Transform2D Transform2D::affine_inverse() const {
	Transform2D inv = *this;
	inv.affine_invert();
	return inv;
}

void Transform2D::rotate(real_t p_angle) {
	*this = Transform2D(p_angle, Vector2()) * (*this);
}

real_t Transform2D::get_skew() const {
	real_t det = determinant();
	return Math::acos(columns[0].normalized().dot(SIGN(det) * columns[1].normalized())) - (real_t)Math::PI * 0.5f;
}

void Transform2D::set_skew(real_t p_angle) {
	real_t det = determinant();
	columns[1] = SIGN(det) * columns[0].rotated(((real_t)Math::PI * 0.5f + p_angle)).normalized() * columns[1].length();
}

real_t Transform2D::get_rotation() const {
	return Math::atan2(columns[0].y, columns[0].x);
}

void Transform2D::set_rotation(real_t p_rot) {
	Size2 scale = get_scale();
	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	columns[0][0] = cr;
	columns[0][1] = sr;
	columns[1][0] = -sr;
	columns[1][1] = cr;
	set_scale(scale);
}

Transform2D::Transform2D(real_t p_rot, const Vector2 &p_pos) {
	real_t cr = Math::cos(p_rot);
	real_t sr = Math::sin(p_rot);
	columns[0][0] = cr;
	columns[0][1] = sr;
	columns[1][0] = -sr;
	columns[1][1] = cr;
	columns[2] = p_pos;
}

Transform2D::Transform2D(real_t p_rot, const Size2 &p_scale, real_t p_skew, const Vector2 &p_pos) {
	columns[0][0] = Math::cos(p_rot) * p_scale.x;
	columns[1][1] = Math::cos(p_rot + p_skew) * p_scale.y;
	columns[1][0] = -Math::sin(p_rot + p_skew) * p_scale.y;
	columns[0][1] = Math::sin(p_rot) * p_scale.x;
	columns[2] = p_pos;
}

Size2 Transform2D::get_scale() const {
	real_t det_sign = SIGN(determinant());
	return Size2(columns[0].length(), det_sign * columns[1].length());
}

void Transform2D::set_scale(const Size2 &p_scale) {
	columns[0].normalize();
	columns[1].normalize();
	columns[0] *= p_scale.x;
	columns[1] *= p_scale.y;
}

void Transform2D::scale(const Size2 &p_scale) {
	scale_basis(p_scale);
	columns[2] *= p_scale;
}

void Transform2D::scale_basis(const Size2 &p_scale) {
	columns[0][0] *= p_scale.x;
	columns[0][1] *= p_scale.y;
	columns[1][0] *= p_scale.x;
	columns[1][1] *= p_scale.y;
}

void Transform2D::translate_local(real_t p_tx, real_t p_ty) {
	translate_local(Vector2(p_tx, p_ty));
}

void Transform2D::translate_local(const Vector2 &p_translation) {
	columns[2] += basis_xform(p_translation);
}

void Transform2D::orthonormalize() {
	// Gram-Schmidt Process

	Vector2 x = columns[0];
	Vector2 y = columns[1];

	x.normalize();
	y = y - x * x.dot(y);
	y.normalize();

	columns[0] = x;
	columns[1] = y;
}

Transform2D Transform2D::orthonormalized() const {
	Transform2D ortho = *this;
	ortho.orthonormalize();
	return ortho;
}

bool Transform2D::is_conformal() const {
	// Non-flipped case.
	if (Math::is_equal_approx(columns[0][0], columns[1][1]) && Math::is_equal_approx(columns[0][1], -columns[1][0])) {
		return true;
	}
	// Flipped case.
	if (Math::is_equal_approx(columns[0][0], -columns[1][1]) && Math::is_equal_approx(columns[0][1], columns[1][0])) {
		return true;
	}
	return false;
}

bool Transform2D::is_equal_approx(const Transform2D &p_transform) const {
	return columns[0].is_equal_approx(p_transform.columns[0]) && columns[1].is_equal_approx(p_transform.columns[1]) && columns[2].is_equal_approx(p_transform.columns[2]);
}

bool Transform2D::is_same(const Transform2D &p_transform) const {
	return columns[0].is_same(p_transform.columns[0]) && columns[1].is_same(p_transform.columns[1]) && columns[2].is_same(p_transform.columns[2]);
}

bool Transform2D::is_finite() const {
	return columns[0].is_finite() && columns[1].is_finite() && columns[2].is_finite();
}

Transform2D Transform2D::looking_at(const Vector2 &p_target) const {
	Transform2D return_trans = Transform2D(get_rotation(), get_origin());
	Vector2 target_position = affine_inverse().xform(p_target);
	return_trans.set_rotation(return_trans.get_rotation() + (target_position * get_scale()).angle());
	return return_trans;
}

void Transform2D::operator*=(const Transform2D &p_transform) {
	columns[2] = xform(p_transform.columns[2]);

	real_t x0, x1, y0, y1;

	x0 = tdotx(p_transform.columns[0]);
	x1 = tdoty(p_transform.columns[0]);
	y0 = tdotx(p_transform.columns[1]);
	y1 = tdoty(p_transform.columns[1]);

	columns[0][0] = x0;
	columns[0][1] = x1;
	columns[1][0] = y0;
	columns[1][1] = y1;
}

Transform2D Transform2D::operator*(const Transform2D &p_transform) const {
	Transform2D t = *this;
	t *= p_transform;
	return t;
}

Transform2D Transform2D::scaled(const Size2 &p_scale) const {
	// Equivalent to left multiplication
	Transform2D copy = *this;
	copy.scale(p_scale);
	return copy;
}

Transform2D Transform2D::scaled_local(const Size2 &p_scale) const {
	// Equivalent to right multiplication
	return Transform2D(columns[0] * p_scale.x, columns[1] * p_scale.y, columns[2]);
}

Transform2D Transform2D::untranslated() const {
	Transform2D copy = *this;
	copy.columns[2] = Vector2();
	return copy;
}

Transform2D Transform2D::translated(const Vector2 &p_offset) const {
	// Equivalent to left multiplication
	return Transform2D(columns[0], columns[1], columns[2] + p_offset);
}

Transform2D Transform2D::translated_local(const Vector2 &p_offset) const {
	// Equivalent to right multiplication
	return Transform2D(columns[0], columns[1], columns[2] + basis_xform(p_offset));
}

Transform2D Transform2D::rotated(real_t p_angle) const {
	// Equivalent to left multiplication
	return Transform2D(p_angle, Vector2()) * (*this);
}

Transform2D Transform2D::rotated_local(real_t p_angle) const {
	// Equivalent to right multiplication
	return (*this) * Transform2D(p_angle, Vector2()); // Could be optimized, because origin transform can be skipped.
}

real_t Transform2D::determinant() const {
	return columns[0].x * columns[1].y - columns[0].y * columns[1].x;
}

Transform2D Transform2D::interpolate_with(const Transform2D &p_transform, real_t p_weight) const {
	return Transform2D(
			Math::lerp_angle(get_rotation(), p_transform.get_rotation(), p_weight),
			get_scale().lerp(p_transform.get_scale(), p_weight),
			Math::lerp_angle(get_skew(), p_transform.get_skew(), p_weight),
			get_origin().lerp(p_transform.get_origin(), p_weight));
}
