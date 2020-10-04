#include "Basis.hpp"
#include "Defs.hpp"
#include "Quat.hpp"
#include "Vector3.hpp"

#include <algorithm>

namespace godot {

Basis::Basis(const Vector3 &row0, const Vector3 &row1, const Vector3 &row2) {
	elements[0] = row0;
	elements[1] = row1;
	elements[2] = row2;
}

Basis::Basis(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz) {

	set(xx, xy, xz, yx, yy, yz, zx, zy, zz);
}

Basis::Basis() {

	elements[0][0] = 1;
	elements[0][1] = 0;
	elements[0][2] = 0;
	elements[1][0] = 0;
	elements[1][1] = 1;
	elements[1][2] = 0;
	elements[2][0] = 0;
	elements[2][1] = 0;
	elements[2][2] = 1;
}

#define cofac(row1, col1, row2, col2) \
	(elements[row1][col1] * elements[row2][col2] - elements[row1][col2] * elements[row2][col1])

void Basis::invert() {
	real_t co[3] = {
		cofac(1, 1, 2, 2), cofac(1, 2, 2, 0), cofac(1, 0, 2, 1)
	};
	real_t det = elements[0][0] * co[0] +
				 elements[0][1] * co[1] +
				 elements[0][2] * co[2];

	ERR_FAIL_COND(det == 0);

	real_t s = 1.0 / det;

	set(co[0] * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s,
			co[1] * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s,
			co[2] * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s);
}
#undef cofac

bool Basis::isequal_approx(const Basis &a, const Basis &b) const {

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if ((::fabs(a.elements[i][j] - b.elements[i][j]) < CMP_EPSILON) == false)
				return false;
		}
	}

	return true;
}

bool Basis::is_orthogonal() const {
	Basis id;
	Basis m = (*this) * transposed();

	return isequal_approx(id, m);
}

bool Basis::is_rotation() const {
	return ::fabs(determinant() - 1) < CMP_EPSILON && is_orthogonal();
}

void Basis::transpose() {
	std::swap(elements[0][1], elements[1][0]);
	std::swap(elements[0][2], elements[2][0]);
	std::swap(elements[1][2], elements[2][1]);
}

Basis Basis::inverse() const {
	Basis b = *this;
	b.invert();
	return b;
}

Basis Basis::transposed() const {
	Basis b = *this;
	b.transpose();
	return b;
}

real_t Basis::determinant() const {
	return elements[0][0] * (elements[1][1] * elements[2][2] - elements[2][1] * elements[1][2]) -
		   elements[1][0] * (elements[0][1] * elements[2][2] - elements[2][1] * elements[0][2]) +
		   elements[2][0] * (elements[0][1] * elements[1][2] - elements[1][1] * elements[0][2]);
}

Vector3 Basis::get_axis(int p_axis) const {
	// get actual basis axis (elements is transposed for performance)
	return Vector3(elements[0][p_axis], elements[1][p_axis], elements[2][p_axis]);
}
void Basis::set_axis(int p_axis, const Vector3 &p_value) {
	// get actual basis axis (elements is transposed for performance)
	elements[0][p_axis] = p_value.x;
	elements[1][p_axis] = p_value.y;
	elements[2][p_axis] = p_value.z;
}

void Basis::rotate(const Vector3 &p_axis, real_t p_phi) {
	*this = rotated(p_axis, p_phi);
}

Basis Basis::rotated(const Vector3 &p_axis, real_t p_phi) const {
	return Basis(p_axis, p_phi) * (*this);
}

void Basis::scale(const Vector3 &p_scale) {
	elements[0][0] *= p_scale.x;
	elements[0][1] *= p_scale.x;
	elements[0][2] *= p_scale.x;
	elements[1][0] *= p_scale.y;
	elements[1][1] *= p_scale.y;
	elements[1][2] *= p_scale.y;
	elements[2][0] *= p_scale.z;
	elements[2][1] *= p_scale.z;
	elements[2][2] *= p_scale.z;
}

Basis Basis::scaled(const Vector3 &p_scale) const {
	Basis b = *this;
	b.scale(p_scale);
	return b;
}

Vector3 Basis::get_scale() const {
	// We are assuming M = R.S, and performing a polar decomposition to extract R and S.
	// FIXME: We eventually need a proper polar decomposition.
	// As a cheap workaround until then, to ensure that R is a proper rotation matrix with determinant +1
	// (such that it can be represented by a Quat or Euler angles), we absorb the sign flip into the scaling matrix.
	// As such, it works in conjuction with get_rotation().
	real_t det_sign = determinant() > 0 ? 1 : -1;
	return det_sign * Vector3(
							  Vector3(elements[0][0], elements[1][0], elements[2][0]).length(),
							  Vector3(elements[0][1], elements[1][1], elements[2][1]).length(),
							  Vector3(elements[0][2], elements[1][2], elements[2][2]).length());
}

// TODO: implement this directly without using quaternions to make it more efficient
Basis Basis::slerp(Basis b, float t) const {
	ERR_FAIL_COND_V(!is_rotation(), Basis());
	ERR_FAIL_COND_V(!b.is_rotation(), Basis());
	Quat from(*this);
	Quat to(b);
	return Basis(from.slerp(to, t));
}

// get_euler_xyz returns a vector containing the Euler angles in the format
// (a1,a2,a3), where a3 is the angle of the first rotation, and a1 is the last
// (following the convention they are commonly defined in the literature).
//
// The current implementation uses XYZ convention (Z is the first rotation),
// so euler.z is the angle of the (first) rotation around Z axis and so on,
//
// And thus, assuming the matrix is a rotation matrix, this function returns
// the angles in the decomposition R = X(a1).Y(a2).Z(a3) where Z(a) rotates
// around the z-axis by a and so on.
Vector3 Basis::get_euler_xyz() const {

	// Euler angles in XYZ convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cy*cz          -cy*sz           sy
	//        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
	//       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy

	Vector3 euler;

	ERR_FAIL_COND_V(is_rotation() == false, euler);

	real_t sy = elements[0][2];
	if (sy < 1.0) {
		if (sy > -1.0) {
			// is this a pure Y rotation?
			if (elements[1][0] == 0.0 && elements[0][1] == 0.0 && elements[1][2] == 0 && elements[2][1] == 0 && elements[1][1] == 1) {
				// return the simplest form (human friendlier in editor and scripts)
				euler.x = 0;
				euler.y = atan2(elements[0][2], elements[0][0]);
				euler.z = 0;
			} else {
				euler.x = ::atan2(-elements[1][2], elements[2][2]);
				euler.y = ::asin(sy);
				euler.z = ::atan2(-elements[0][1], elements[0][0]);
			}
		} else {
			euler.x = -::atan2(elements[0][1], elements[1][1]);
			euler.y = -Math_PI / 2.0;
			euler.z = 0.0;
		}
	} else {
		euler.x = ::atan2(elements[0][1], elements[1][1]);
		euler.y = Math_PI / 2.0;
		euler.z = 0.0;
	}
	return euler;
}

// set_euler_xyz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// The current implementation uses XYZ convention (Z is the first rotation).
void Basis::set_euler_xyz(const Vector3 &p_euler) {

	real_t c, s;

	c = ::cos(p_euler.x);
	s = ::sin(p_euler.x);
	Basis xmat(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);

	c = ::cos(p_euler.y);
	s = ::sin(p_euler.y);
	Basis ymat(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);

	c = ::cos(p_euler.z);
	s = ::sin(p_euler.z);
	Basis zmat(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);

	//optimizer will optimize away all this anyway
	*this = xmat * (ymat * zmat);
}

// get_euler_yxz returns a vector containing the Euler angles in the YXZ convention,
// as in first-Z, then-X, last-Y. The angles for X, Y, and Z rotations are returned
// as the x, y, and z components of a Vector3 respectively.
Vector3 Basis::get_euler_yxz() const {

	// Euler angles in YXZ convention.
	// See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
	//
	// rot =  cy*cz+sy*sx*sz    cz*sy*sx-cy*sz        cx*sy
	//        cx*sz             cx*cz                 -sx
	//        cy*sx*sz-cz*sy    cy*cz*sx+sy*sz        cy*cx

	Vector3 euler;

	ERR_FAIL_COND_V(is_rotation() == false, euler);

	real_t m12 = elements[1][2];

	if (m12 < 1) {
		if (m12 > -1) {
			// is this a pure X rotation?
			if (elements[1][0] == 0 && elements[0][1] == 0 && elements[0][2] == 0 && elements[2][0] == 0 && elements[0][0] == 1) {
				// return the simplest form (human friendlier in editor and scripts)
				euler.x = atan2(-m12, elements[1][1]);
				euler.y = 0;
				euler.z = 0;
			} else {
				euler.x = asin(-m12);
				euler.y = atan2(elements[0][2], elements[2][2]);
				euler.z = atan2(elements[1][0], elements[1][1]);
			}
		} else { // m12 == -1
			euler.x = Math_PI * 0.5;
			euler.y = -atan2(-elements[0][1], elements[0][0]);
			euler.z = 0;
		}
	} else { // m12 == 1
		euler.x = -Math_PI * 0.5;
		euler.y = -atan2(-elements[0][1], elements[0][0]);
		euler.z = 0;
	}

	return euler;
}

// set_euler_yxz expects a vector containing the Euler angles in the format
// (ax,ay,az), where ax is the angle of rotation around x axis,
// and similar for other axes.
// The current implementation uses YXZ convention (Z is the first rotation).
void Basis::set_euler_yxz(const Vector3 &p_euler) {

	real_t c, s;

	c = ::cos(p_euler.x);
	s = ::sin(p_euler.x);
	Basis xmat(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);

	c = ::cos(p_euler.y);
	s = ::sin(p_euler.y);
	Basis ymat(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);

	c = ::cos(p_euler.z);
	s = ::sin(p_euler.z);
	Basis zmat(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);

	//optimizer will optimize away all this anyway
	*this = ymat * xmat * zmat;
}

// transposed dot products
real_t Basis::tdotx(const Vector3 &v) const {
	return elements[0][0] * v[0] + elements[1][0] * v[1] + elements[2][0] * v[2];
}
real_t Basis::tdoty(const Vector3 &v) const {
	return elements[0][1] * v[0] + elements[1][1] * v[1] + elements[2][1] * v[2];
}
real_t Basis::tdotz(const Vector3 &v) const {
	return elements[0][2] * v[0] + elements[1][2] * v[1] + elements[2][2] * v[2];
}

bool Basis::operator==(const Basis &p_matrix) const {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (elements[i][j] != p_matrix.elements[i][j])
				return false;
		}
	}

	return true;
}

bool Basis::operator!=(const Basis &p_matrix) const {
	return (!(*this == p_matrix));
}

Vector3 Basis::xform(const Vector3 &p_vector) const {

	return Vector3(
			elements[0].dot(p_vector),
			elements[1].dot(p_vector),
			elements[2].dot(p_vector));
}

Vector3 Basis::xform_inv(const Vector3 &p_vector) const {

	return Vector3(
			(elements[0][0] * p_vector.x) + (elements[1][0] * p_vector.y) + (elements[2][0] * p_vector.z),
			(elements[0][1] * p_vector.x) + (elements[1][1] * p_vector.y) + (elements[2][1] * p_vector.z),
			(elements[0][2] * p_vector.x) + (elements[1][2] * p_vector.y) + (elements[2][2] * p_vector.z));
}
void Basis::operator*=(const Basis &p_matrix) {
	set(
			p_matrix.tdotx(elements[0]), p_matrix.tdoty(elements[0]), p_matrix.tdotz(elements[0]),
			p_matrix.tdotx(elements[1]), p_matrix.tdoty(elements[1]), p_matrix.tdotz(elements[1]),
			p_matrix.tdotx(elements[2]), p_matrix.tdoty(elements[2]), p_matrix.tdotz(elements[2]));
}

Basis Basis::operator*(const Basis &p_matrix) const {
	return Basis(
			p_matrix.tdotx(elements[0]), p_matrix.tdoty(elements[0]), p_matrix.tdotz(elements[0]),
			p_matrix.tdotx(elements[1]), p_matrix.tdoty(elements[1]), p_matrix.tdotz(elements[1]),
			p_matrix.tdotx(elements[2]), p_matrix.tdoty(elements[2]), p_matrix.tdotz(elements[2]));
}

void Basis::operator+=(const Basis &p_matrix) {

	elements[0] += p_matrix.elements[0];
	elements[1] += p_matrix.elements[1];
	elements[2] += p_matrix.elements[2];
}

Basis Basis::operator+(const Basis &p_matrix) const {

	Basis ret(*this);
	ret += p_matrix;
	return ret;
}

void Basis::operator-=(const Basis &p_matrix) {

	elements[0] -= p_matrix.elements[0];
	elements[1] -= p_matrix.elements[1];
	elements[2] -= p_matrix.elements[2];
}

Basis Basis::operator-(const Basis &p_matrix) const {

	Basis ret(*this);
	ret -= p_matrix;
	return ret;
}

void Basis::operator*=(real_t p_val) {

	elements[0] *= p_val;
	elements[1] *= p_val;
	elements[2] *= p_val;
}

Basis Basis::operator*(real_t p_val) const {

	Basis ret(*this);
	ret *= p_val;
	return ret;
}

Basis::operator String() const {
	String s;
	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 3; j++) {

			if (i != 0 || j != 0)
				s += ", ";

			s += String::num(elements[i][j]);
		}
	}
	return s;
}

/* create / set */

void Basis::set(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz) {

	elements[0][0] = xx;
	elements[0][1] = xy;
	elements[0][2] = xz;
	elements[1][0] = yx;
	elements[1][1] = yy;
	elements[1][2] = yz;
	elements[2][0] = zx;
	elements[2][1] = zy;
	elements[2][2] = zz;
}
Vector3 Basis::get_column(int i) const {

	return Vector3(elements[0][i], elements[1][i], elements[2][i]);
}

Vector3 Basis::get_row(int i) const {

	return Vector3(elements[i][0], elements[i][1], elements[i][2]);
}
Vector3 Basis::get_main_diagonal() const {
	return Vector3(elements[0][0], elements[1][1], elements[2][2]);
}

void Basis::set_row(int i, const Vector3 &p_row) {
	elements[i][0] = p_row.x;
	elements[i][1] = p_row.y;
	elements[i][2] = p_row.z;
}

Basis Basis::transpose_xform(const Basis &m) const {
	return Basis(
			elements[0].x * m[0].x + elements[1].x * m[1].x + elements[2].x * m[2].x,
			elements[0].x * m[0].y + elements[1].x * m[1].y + elements[2].x * m[2].y,
			elements[0].x * m[0].z + elements[1].x * m[1].z + elements[2].x * m[2].z,
			elements[0].y * m[0].x + elements[1].y * m[1].x + elements[2].y * m[2].x,
			elements[0].y * m[0].y + elements[1].y * m[1].y + elements[2].y * m[2].y,
			elements[0].y * m[0].z + elements[1].y * m[1].z + elements[2].y * m[2].z,
			elements[0].z * m[0].x + elements[1].z * m[1].x + elements[2].z * m[2].x,
			elements[0].z * m[0].y + elements[1].z * m[1].y + elements[2].z * m[2].y,
			elements[0].z * m[0].z + elements[1].z * m[1].z + elements[2].z * m[2].z);
}

void Basis::orthonormalize() {
	ERR_FAIL_COND(determinant() == 0);

	// Gram-Schmidt Process

	Vector3 x = get_axis(0);
	Vector3 y = get_axis(1);
	Vector3 z = get_axis(2);

	x.normalize();
	y = (y - x * (x.dot(y)));
	y.normalize();
	z = (z - x * (x.dot(z)) - y * (y.dot(z)));
	z.normalize();

	set_axis(0, x);
	set_axis(1, y);
	set_axis(2, z);
}

Basis Basis::orthonormalized() const {
	Basis b = *this;
	b.orthonormalize();
	return b;
}

bool Basis::is_symmetric() const {
	if (::fabs(elements[0][1] - elements[1][0]) > CMP_EPSILON)
		return false;
	if (::fabs(elements[0][2] - elements[2][0]) > CMP_EPSILON)
		return false;
	if (::fabs(elements[1][2] - elements[2][1]) > CMP_EPSILON)
		return false;

	return true;
}

Basis Basis::diagonalize() {
	// I love copy paste

	if (!is_symmetric())
		return Basis();

	const int ite_max = 1024;

	real_t off_matrix_norm_2 = elements[0][1] * elements[0][1] + elements[0][2] * elements[0][2] + elements[1][2] * elements[1][2];

	int ite = 0;
	Basis acc_rot;
	while (off_matrix_norm_2 > CMP_EPSILON2 && ite++ < ite_max) {
		real_t el01_2 = elements[0][1] * elements[0][1];
		real_t el02_2 = elements[0][2] * elements[0][2];
		real_t el12_2 = elements[1][2] * elements[1][2];
		// Find the pivot element
		int i, j;
		if (el01_2 > el02_2) {
			if (el12_2 > el01_2) {
				i = 1;
				j = 2;
			} else {
				i = 0;
				j = 1;
			}
		} else {
			if (el12_2 > el02_2) {
				i = 1;
				j = 2;
			} else {
				i = 0;
				j = 2;
			}
		}

		// Compute the rotation angle
		real_t angle;
		if (::fabs(elements[j][j] - elements[i][i]) < CMP_EPSILON) {
			angle = Math_PI / 4;
		} else {
			angle = 0.5 * ::atan(2 * elements[i][j] / (elements[j][j] - elements[i][i]));
		}

		// Compute the rotation matrix
		Basis rot;
		rot.elements[i][i] = rot.elements[j][j] = ::cos(angle);
		rot.elements[i][j] = -(rot.elements[j][i] = ::sin(angle));

		// Update the off matrix norm
		off_matrix_norm_2 -= elements[i][j] * elements[i][j];

		// Apply the rotation
		*this = rot * *this * rot.transposed();
		acc_rot = rot * acc_rot;
	}

	return acc_rot;
}

static const Basis _ortho_bases[24] = {
	Basis(1, 0, 0, 0, 1, 0, 0, 0, 1),
	Basis(0, -1, 0, 1, 0, 0, 0, 0, 1),
	Basis(-1, 0, 0, 0, -1, 0, 0, 0, 1),
	Basis(0, 1, 0, -1, 0, 0, 0, 0, 1),
	Basis(1, 0, 0, 0, 0, -1, 0, 1, 0),
	Basis(0, 0, 1, 1, 0, 0, 0, 1, 0),
	Basis(-1, 0, 0, 0, 0, 1, 0, 1, 0),
	Basis(0, 0, -1, -1, 0, 0, 0, 1, 0),
	Basis(1, 0, 0, 0, -1, 0, 0, 0, -1),
	Basis(0, 1, 0, 1, 0, 0, 0, 0, -1),
	Basis(-1, 0, 0, 0, 1, 0, 0, 0, -1),
	Basis(0, -1, 0, -1, 0, 0, 0, 0, -1),
	Basis(1, 0, 0, 0, 0, 1, 0, -1, 0),
	Basis(0, 0, -1, 1, 0, 0, 0, -1, 0),
	Basis(-1, 0, 0, 0, 0, -1, 0, -1, 0),
	Basis(0, 0, 1, -1, 0, 0, 0, -1, 0),
	Basis(0, 0, 1, 0, 1, 0, -1, 0, 0),
	Basis(0, -1, 0, 0, 0, 1, -1, 0, 0),
	Basis(0, 0, -1, 0, -1, 0, -1, 0, 0),
	Basis(0, 1, 0, 0, 0, -1, -1, 0, 0),
	Basis(0, 0, 1, 0, -1, 0, 1, 0, 0),
	Basis(0, 1, 0, 0, 0, 1, 1, 0, 0),
	Basis(0, 0, -1, 0, 1, 0, 1, 0, 0),
	Basis(0, -1, 0, 0, 0, -1, 1, 0, 0)
};

int Basis::get_orthogonal_index() const {
	//could be sped up if i come up with a way
	Basis orth = *this;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {

			real_t v = orth[i][j];
			if (v > 0.5)
				v = 1.0;
			else if (v < -0.5)
				v = -1.0;
			else
				v = 0;

			orth[i][j] = v;
		}
	}

	for (int i = 0; i < 24; i++) {

		if (_ortho_bases[i] == orth)
			return i;
	}

	return 0;
}

void Basis::set_orthogonal_index(int p_index) {

	//there only exist 24 orthogonal bases in r3
	ERR_FAIL_COND(p_index >= 24);

	*this = _ortho_bases[p_index];
}

Basis::Basis(const Vector3 &p_euler) {

	set_euler(p_euler);
}

} // namespace godot

#include "Quat.hpp"

namespace godot {

Basis::Basis(const Quat &p_quat) {

	real_t d = p_quat.length_squared();
	real_t s = 2.0 / d;
	real_t xs = p_quat.x * s, ys = p_quat.y * s, zs = p_quat.z * s;
	real_t wx = p_quat.w * xs, wy = p_quat.w * ys, wz = p_quat.w * zs;
	real_t xx = p_quat.x * xs, xy = p_quat.x * ys, xz = p_quat.x * zs;
	real_t yy = p_quat.y * ys, yz = p_quat.y * zs, zz = p_quat.z * zs;
	set(1.0 - (yy + zz), xy - wz, xz + wy,
			xy + wz, 1.0 - (xx + zz), yz - wx,
			xz - wy, yz + wx, 1.0 - (xx + yy));
}

Basis::Basis(const Vector3 &p_axis, real_t p_phi) {
	// Rotation matrix from axis and angle, see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

	Vector3 axis_sq(p_axis.x * p_axis.x, p_axis.y * p_axis.y, p_axis.z * p_axis.z);

	real_t cosine = ::cos(p_phi);
	real_t sine = ::sin(p_phi);

	elements[0][0] = axis_sq.x + cosine * (1.0 - axis_sq.x);
	elements[0][1] = p_axis.x * p_axis.y * (1.0 - cosine) - p_axis.z * sine;
	elements[0][2] = p_axis.z * p_axis.x * (1.0 - cosine) + p_axis.y * sine;

	elements[1][0] = p_axis.x * p_axis.y * (1.0 - cosine) + p_axis.z * sine;
	elements[1][1] = axis_sq.y + cosine * (1.0 - axis_sq.y);
	elements[1][2] = p_axis.y * p_axis.z * (1.0 - cosine) - p_axis.x * sine;

	elements[2][0] = p_axis.z * p_axis.x * (1.0 - cosine) - p_axis.y * sine;
	elements[2][1] = p_axis.y * p_axis.z * (1.0 - cosine) + p_axis.x * sine;
	elements[2][2] = axis_sq.z + cosine * (1.0 - axis_sq.z);
}

Basis::operator Quat() const {
	//commenting this check because precision issues cause it to fail when it shouldn't
	//ERR_FAIL_COND_V(is_rotation() == false, Quat());

	real_t trace = elements[0][0] + elements[1][1] + elements[2][2];
	real_t temp[4];

	if (trace > 0.0) {
		real_t s = ::sqrt(trace + 1.0);
		temp[3] = (s * 0.5);
		s = 0.5 / s;

		temp[0] = ((elements[2][1] - elements[1][2]) * s);
		temp[1] = ((elements[0][2] - elements[2][0]) * s);
		temp[2] = ((elements[1][0] - elements[0][1]) * s);
	} else {
		int i = elements[0][0] < elements[1][1] ?
						(elements[1][1] < elements[2][2] ? 2 : 1) :
						(elements[0][0] < elements[2][2] ? 2 : 0);
		int j = (i + 1) % 3;
		int k = (i + 2) % 3;

		real_t s = ::sqrt(elements[i][i] - elements[j][j] - elements[k][k] + 1.0);
		temp[i] = s * 0.5;
		s = 0.5 / s;

		temp[3] = (elements[k][j] - elements[j][k]) * s;
		temp[j] = (elements[j][i] + elements[i][j]) * s;
		temp[k] = (elements[k][i] + elements[i][k]) * s;
	}

	return Quat(temp[0], temp[1], temp[2], temp[3]);
}

} // namespace godot
