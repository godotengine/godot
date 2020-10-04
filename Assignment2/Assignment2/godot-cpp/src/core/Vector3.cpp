#include "Vector3.hpp"

#include "String.hpp"

#include <stdlib.h>

#include "Basis.hpp"

namespace godot {

bool Vector3::operator<(const Vector3 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y)
			return z < p_v.z;
		else
			return y < p_v.y;
	} else {
		return x < p_v.x;
	}
}

bool Vector3::operator<=(const Vector3 &p_v) const {
	if (x == p_v.x) {
		if (y == p_v.y)
			return z <= p_v.z;
		else
			return y < p_v.y;
	} else {
		return x < p_v.x;
	}
}

Vector3 Vector3::cubic_interpolate(const Vector3 &b, const Vector3 &pre_a, const Vector3 &post_b, const real_t t) const {
	Vector3 p0 = pre_a;
	Vector3 p1 = *this;
	Vector3 p2 = b;
	Vector3 p3 = post_b;

	real_t t2 = t * t;
	real_t t3 = t2 * t;

	Vector3 out;
	out = ((p1 * 2.0) +
				  (-p0 + p2) * t +
				  (p0 * 2.0 - p1 * 5.0 + p2 * 4 - p3) * t2 +
				  (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * t3) *
		  0.5;
	return out;
}

Basis Vector3::outer(const Vector3 &b) const {
	Vector3 row0(x * b.x, x * b.y, x * b.z);
	Vector3 row1(y * b.x, y * b.y, y * b.z);
	Vector3 row2(z * b.x, z * b.y, z * b.z);
	return Basis(row0, row1, row2);
}

int Vector3::max_axis() const {
	return x < y ? (y < z ? 2 : 1) : (x < z ? 2 : 0);
}

int Vector3::min_axis() const {
	return x < y ? (x < z ? 0 : 2) : (y < z ? 1 : 2);
}

void Vector3::rotate(const Vector3 &p_axis, real_t p_phi) {
	*this = Basis(p_axis, p_phi).xform(*this);
}

void Vector3::snap(real_t p_val) {
	x = Math::stepify(x, p_val);
	y = Math::stepify(y, p_val);
	z = Math::stepify(z, p_val);
}

Vector3::operator String() const {
	return String::num(x) + ", " + String::num(y) + ", " + String::num(z);
}

} // namespace godot
