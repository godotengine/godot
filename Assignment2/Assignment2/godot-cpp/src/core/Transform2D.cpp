#include "Transform2D.hpp"
#include "Rect2.hpp"
#include "String.hpp"
#include "Vector2.hpp"

#include <algorithm>

namespace godot {

Transform2D::Transform2D(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy) {

	elements[0][0] = xx;
	elements[0][1] = xy;
	elements[1][0] = yx;
	elements[1][1] = yy;
	elements[2][0] = ox;
	elements[2][1] = oy;
}

Vector2 Transform2D::basis_xform(const Vector2 &v) const {

	return Vector2(
			tdotx(v),
			tdoty(v));
}

Vector2 Transform2D::basis_xform_inv(const Vector2 &v) const {

	return Vector2(
			elements[0].dot(v),
			elements[1].dot(v));
}

Vector2 Transform2D::xform(const Vector2 &v) const {

	return Vector2(
				   tdotx(v),
				   tdoty(v)) +
		   elements[2];
}
Vector2 Transform2D::xform_inv(const Vector2 &p_vec) const {

	Vector2 v = p_vec - elements[2];

	return Vector2(
			elements[0].dot(v),
			elements[1].dot(v));
}
Rect2 Transform2D::xform(const Rect2 &p_rect) const {

	Vector2 x = elements[0] * p_rect.size.x;
	Vector2 y = elements[1] * p_rect.size.y;
	Vector2 position = xform(p_rect.position);

	Rect2 new_rect;
	new_rect.position = position;
	new_rect.expand_to(position + x);
	new_rect.expand_to(position + y);
	new_rect.expand_to(position + x + y);
	return new_rect;
}

void Transform2D::set_rotation_and_scale(real_t p_rot, const Size2 &p_scale) {

	elements[0][0] = ::cos(p_rot) * p_scale.x;
	elements[1][1] = ::cos(p_rot) * p_scale.y;
	elements[1][0] = -::sin(p_rot) * p_scale.y;
	elements[0][1] = ::sin(p_rot) * p_scale.x;
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

void Transform2D::invert() {
	// FIXME: this function assumes the basis is a rotation matrix, with no scaling.
	// Transform2D::affine_inverse can handle matrices with scaling, so GDScript should eventually use that.
	std::swap(elements[0][1], elements[1][0]);
	elements[2] = basis_xform(-elements[2]);
}

Transform2D Transform2D::inverse() const {

	Transform2D inv = *this;
	inv.invert();
	return inv;
}

void Transform2D::affine_invert() {

	real_t det = basis_determinant();
	ERR_FAIL_COND(det == 0);
	real_t idet = 1.0 / det;

	std::swap(elements[0][0], elements[1][1]);
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
		m.scale_basis(Size2(-1, -1));
	}
	return ::atan2(m[0].y, m[0].x);
}

void Transform2D::set_rotation(real_t p_rot) {

	real_t cr = ::cos(p_rot);
	real_t sr = ::sin(p_rot);
	elements[0][0] = cr;
	elements[0][1] = sr;
	elements[1][0] = -sr;
	elements[1][1] = cr;
}

Transform2D::Transform2D(real_t p_rot, const Vector2 &p_position) {

	real_t cr = ::cos(p_rot);
	real_t sr = ::sin(p_rot);
	elements[0][0] = cr;
	elements[0][1] = sr;
	elements[1][0] = -sr;
	elements[1][1] = cr;
	elements[2] = p_position;
}

Size2 Transform2D::get_scale() const {
	real_t det_sign = basis_determinant() > 0 ? 1 : -1;
	return det_sign * Size2(elements[0].length(), elements[1].length());
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
	Vector2 v1(::cos(r1), ::sin(r1));
	Vector2 v2(::cos(r2), ::sin(r2));

	real_t dot = v1.dot(v2);

	dot = (dot < -1.0) ? -1.0 : ((dot > 1.0) ? 1.0 : dot); //clamp dot to [-1,1]

	Vector2 v;

	if (dot > 0.9995) {
		v = Vector2::linear_interpolate(v1, v2, p_c).normalized(); //linearly interpolate to avoid numerical precision issues
	} else {
		real_t angle = p_c * ::acos(dot);
		Vector2 v3 = (v2 - v1 * dot).normalized();
		v = v1 * ::cos(angle) + v3 * ::sin(angle);
	}

	//construct matrix
	Transform2D res(::atan2(v.y, v.x), Vector2::linear_interpolate(p1, p2, p_c));
	res.scale_basis(Vector2::linear_interpolate(s1, s2, p_c));
	return res;
}

Transform2D::operator String() const {

	return String(String() + elements[0] + ", " + elements[1] + ", " + elements[2]);
}

} // namespace godot
