#include "Quat.hpp"
#include "Basis.hpp"
#include "Defs.hpp"
#include "Vector3.hpp"

#include <cmath>

namespace godot {

// set_euler_xyz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses XYZ convention (Z is the first rotation).
void Quat::set_euler_xyz(const Vector3 &p_euler) {
	real_t half_a1 = p_euler.x * 0.5;
	real_t half_a2 = p_euler.y * 0.5;
	real_t half_a3 = p_euler.z * 0.5;

	// R = X(a1).Y(a2).Z(a3) convention for Euler angles.
	// Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-2)
	// a3 is the angle of the first rotation, following the notation in this reference.

	real_t cos_a1 = ::cos(half_a1);
	real_t sin_a1 = ::sin(half_a1);
	real_t cos_a2 = ::cos(half_a2);
	real_t sin_a2 = ::sin(half_a2);
	real_t cos_a3 = ::cos(half_a3);
	real_t sin_a3 = ::sin(half_a3);

	set(sin_a1 * cos_a2 * cos_a3 + sin_a2 * sin_a3 * cos_a1,
			-sin_a1 * sin_a3 * cos_a2 + sin_a2 * cos_a1 * cos_a3,
			sin_a1 * sin_a2 * cos_a3 + sin_a3 * cos_a1 * cos_a2,
			-sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3);
}

// get_euler_xyz returns a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses XYZ convention (Z is the first rotation).
Vector3 Quat::get_euler_xyz() const {
	Basis m(*this);
	return m.get_euler_xyz();
}

// set_euler_yxz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses YXZ convention (Z is the first rotation).
void Quat::set_euler_yxz(const Vector3 &p_euler) {
	real_t half_a1 = p_euler.y * 0.5;
	real_t half_a2 = p_euler.x * 0.5;
	real_t half_a3 = p_euler.z * 0.5;

	// R = Y(a1).X(a2).Z(a3) convention for Euler angles.
	// Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-6)
	// a3 is the angle of the first rotation, following the notation in this reference.

	real_t cos_a1 = ::cos(half_a1);
	real_t sin_a1 = ::sin(half_a1);
	real_t cos_a2 = ::cos(half_a2);
	real_t sin_a2 = ::sin(half_a2);
	real_t cos_a3 = ::cos(half_a3);
	real_t sin_a3 = ::sin(half_a3);

	set(sin_a1 * cos_a2 * sin_a3 + cos_a1 * sin_a2 * cos_a3,
			sin_a1 * cos_a2 * cos_a3 - cos_a1 * sin_a2 * sin_a3,
			-sin_a1 * sin_a2 * cos_a3 + cos_a1 * sin_a2 * sin_a3,
			sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3);
}

// get_euler_yxz returns a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// This implementation uses YXZ convention (Z is the first rotation).
Vector3 Quat::get_euler_yxz() const {
	Basis m(*this);
	return m.get_euler_yxz();
}

real_t Quat::length() const {
	return ::sqrt(length_squared());
}

void Quat::normalize() {
	*this /= length();
}

Quat Quat::normalized() const {
	return *this / length();
}

bool Quat::is_normalized() const {
	return std::abs(length_squared() - 1.0) < 0.00001;
}

Quat Quat::inverse() const {
	return Quat(-x, -y, -z, w);
}

Quat Quat::slerp(const Quat &q, const real_t &t) const {

	Quat to1;
	real_t omega, cosom, sinom, scale0, scale1;

	// calc cosine
	cosom = dot(q);

	// adjust signs (if necessary)
	if (cosom < 0.0) {
		cosom = -cosom;
		to1.x = -q.x;
		to1.y = -q.y;
		to1.z = -q.z;
		to1.w = -q.w;
	} else {
		to1.x = q.x;
		to1.y = q.y;
		to1.z = q.z;
		to1.w = q.w;
	}

	// calculate coefficients

	if ((1.0 - cosom) > CMP_EPSILON) {
		// standard case (slerp)
		omega = ::acos(cosom);
		sinom = ::sin(omega);
		scale0 = ::sin((1.0 - t) * omega) / sinom;
		scale1 = ::sin(t * omega) / sinom;
	} else {
		// "from" and "to" quaternions are very close
		//  ... so we can do a linear interpolation
		scale0 = 1.0 - t;
		scale1 = t;
	}
	// calculate final values
	return Quat(
			scale0 * x + scale1 * to1.x,
			scale0 * y + scale1 * to1.y,
			scale0 * z + scale1 * to1.z,
			scale0 * w + scale1 * to1.w);
}

Quat Quat::slerpni(const Quat &q, const real_t &t) const {

	const Quat &from = *this;

	real_t dot = from.dot(q);

	if (::fabs(dot) > 0.9999) return from;

	real_t theta = ::acos(dot),
		   sinT = 1.0 / ::sin(theta),
		   newFactor = ::sin(t * theta) * sinT,
		   invFactor = ::sin((1.0 - t) * theta) * sinT;

	return Quat(invFactor * from.x + newFactor * q.x,
			invFactor * from.y + newFactor * q.y,
			invFactor * from.z + newFactor * q.z,
			invFactor * from.w + newFactor * q.w);
}

Quat Quat::cubic_slerp(const Quat &q, const Quat &prep, const Quat &postq, const real_t &t) const {
	//the only way to do slerp :|
	real_t t2 = (1.0 - t) * t * 2;
	Quat sp = this->slerp(q, t);
	Quat sq = prep.slerpni(postq, t);
	return sp.slerpni(sq, t2);
}

void Quat::get_axis_and_angle(Vector3 &r_axis, real_t &r_angle) const {
	r_angle = 2 * ::acos(w);
	r_axis.x = x / ::sqrt(1 - w * w);
	r_axis.y = y / ::sqrt(1 - w * w);
	r_axis.z = z / ::sqrt(1 - w * w);
}

void Quat::set_axis_angle(const Vector3 &axis, const float angle) {
	ERR_FAIL_COND(!axis.is_normalized());

	real_t d = axis.length();
	if (d == 0)
		set(0, 0, 0, 0);
	else {
		real_t sin_angle = ::sin(angle * 0.5);
		real_t cos_angle = ::cos(angle * 0.5);
		real_t s = sin_angle / d;
		set(axis.x * s, axis.y * s, axis.z * s,
				cos_angle);
	}
}

Quat Quat::operator*(const Vector3 &v) const {
	return Quat(w * v.x + y * v.z - z * v.y,
			w * v.y + z * v.x - x * v.z,
			w * v.z + x * v.y - y * v.x,
			-x * v.x - y * v.y - z * v.z);
}

Vector3 Quat::xform(const Vector3 &v) const {

	Quat q = *this * v;
	q *= this->inverse();
	return Vector3(q.x, q.y, q.z);
}

Quat::operator String() const {
	return String(); // @Todo
}

Quat::Quat(const Vector3 &axis, const real_t &angle) {
	real_t d = axis.length();
	if (d == 0)
		set(0, 0, 0, 0);
	else {
		real_t sin_angle = ::sin(angle * 0.5);
		real_t cos_angle = ::cos(angle * 0.5);
		real_t s = sin_angle / d;
		set(axis.x * s, axis.y * s, axis.z * s,
				cos_angle);
	}
}

Quat::Quat(const Vector3 &v0, const Vector3 &v1) // shortest arc
{
	Vector3 c = v0.cross(v1);
	real_t d = v0.dot(v1);

	if (d < -1.0 + CMP_EPSILON) {
		x = 0;
		y = 1;
		z = 0;
		w = 0;
	} else {

		real_t s = ::sqrt((1.0 + d) * 2.0);
		real_t rs = 1.0 / s;

		x = c.x * rs;
		y = c.y * rs;
		z = c.z * rs;
		w = s * 0.5;
	}
}

real_t Quat::dot(const Quat &q) const {
	return x * q.x + y * q.y + z * q.z + w * q.w;
}

real_t Quat::length_squared() const {
	return dot(*this);
}

void Quat::operator+=(const Quat &q) {
	x += q.x;
	y += q.y;
	z += q.z;
	w += q.w;
}

void Quat::operator-=(const Quat &q) {
	x -= q.x;
	y -= q.y;
	z -= q.z;
	w -= q.w;
}

void Quat::operator*=(const Quat &q) {
	set(w * q.x + x * q.w + y * q.z - z * q.y,
			w * q.y + y * q.w + z * q.x - x * q.z,
			w * q.z + z * q.w + x * q.y - y * q.x,
			w * q.w - x * q.x - y * q.y - z * q.z);
}

void Quat::operator*=(const real_t &s) {
	x *= s;
	y *= s;
	z *= s;
	w *= s;
}

void Quat::operator/=(const real_t &s) {

	*this *= 1.0 / s;
}

Quat Quat::operator+(const Quat &q2) const {
	const Quat &q1 = *this;
	return Quat(q1.x + q2.x, q1.y + q2.y, q1.z + q2.z, q1.w + q2.w);
}

Quat Quat::operator-(const Quat &q2) const {
	const Quat &q1 = *this;
	return Quat(q1.x - q2.x, q1.y - q2.y, q1.z - q2.z, q1.w - q2.w);
}

Quat Quat::operator*(const Quat &q2) const {
	Quat q1 = *this;
	q1 *= q2;
	return q1;
}

Quat Quat::operator-() const {
	const Quat &q2 = *this;
	return Quat(-q2.x, -q2.y, -q2.z, -q2.w);
}

Quat Quat::operator*(const real_t &s) const {
	return Quat(x * s, y * s, z * s, w * s);
}

Quat Quat::operator/(const real_t &s) const {
	return *this * (1.0 / s);
}

bool Quat::operator==(const Quat &p_quat) const {
	return x == p_quat.x && y == p_quat.y && z == p_quat.z && w == p_quat.w;
}

bool Quat::operator!=(const Quat &p_quat) const {
	return x != p_quat.x || y != p_quat.y || z != p_quat.z || w != p_quat.w;
}

} // namespace godot
