/*************************************************************************/
/*  basis.h                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef BASIS_H
#define BASIS_H

#include "core/math/quaternion.h"
#include "core/math/vector3.h"

class Basis {
private:
	void _set_diagonal(const Vector3 &p_diag);

public:
	Vector3 elements[3] = {
		Vector3(1, 0, 0),
		Vector3(0, 1, 0),
		Vector3(0, 0, 1)
	};

	_FORCE_INLINE_ const Vector3 &operator[](int axis) const {
		return elements[axis];
	}
	_FORCE_INLINE_ Vector3 &operator[](int axis) {
		return elements[axis];
	}

	void invert();
	void transpose();

	Basis inverse() const;
	Basis transposed() const;

	_FORCE_INLINE_ real_t determinant() const;

	void from_z(const Vector3 &p_z);

	_FORCE_INLINE_ Vector3 get_axis(int p_axis) const {
		// get actual basis axis (elements is transposed for performance)
		return Vector3(elements[0][p_axis], elements[1][p_axis], elements[2][p_axis]);
	}
	_FORCE_INLINE_ void set_axis(int p_axis, const Vector3 &p_value) {
		// get actual basis axis (elements is transposed for performance)
		elements[0][p_axis] = p_value.x;
		elements[1][p_axis] = p_value.y;
		elements[2][p_axis] = p_value.z;
	}

	void rotate(const Vector3 &p_axis, real_t p_phi);
	Basis rotated(const Vector3 &p_axis, real_t p_phi) const;

	void rotate_local(const Vector3 &p_axis, real_t p_phi);
	Basis rotated_local(const Vector3 &p_axis, real_t p_phi) const;

	void rotate(const Vector3 &p_euler);
	Basis rotated(const Vector3 &p_euler) const;

	void rotate(const Quaternion &p_quaternion);
	Basis rotated(const Quaternion &p_quaternion) const;

	enum EulerOrder {
		EULER_ORDER_XYZ,
		EULER_ORDER_XZY,
		EULER_ORDER_YXZ,
		EULER_ORDER_YZX,
		EULER_ORDER_ZXY,
		EULER_ORDER_ZYX
	};

	Vector3 get_euler_normalized(EulerOrder p_order = EULER_ORDER_YXZ) const;
	void get_rotation_axis_angle(Vector3 &p_axis, real_t &p_angle) const;
	void get_rotation_axis_angle_local(Vector3 &p_axis, real_t &p_angle) const;
	Quaternion get_rotation_quaternion() const;

	void rotate_to_align(Vector3 p_start_direction, Vector3 p_end_direction);

	Vector3 rotref_posscale_decomposition(Basis &rotref) const;

	Vector3 get_euler(EulerOrder p_order = EULER_ORDER_YXZ) const;
	void set_euler(const Vector3 &p_euler, EulerOrder p_order = EULER_ORDER_YXZ);
	static Basis from_euler(const Vector3 &p_euler, EulerOrder p_order = EULER_ORDER_YXZ) {
		Basis b;
		b.set_euler(p_euler, p_order);
		return b;
	}

	Quaternion get_quaternion() const;
	void set_quaternion(const Quaternion &p_quaternion);

	void get_axis_angle(Vector3 &r_axis, real_t &r_angle) const;
	void set_axis_angle(const Vector3 &p_axis, real_t p_phi);

	void scale(const Vector3 &p_scale);
	Basis scaled(const Vector3 &p_scale) const;

	void scale_local(const Vector3 &p_scale);
	Basis scaled_local(const Vector3 &p_scale) const;

	void scale_orthogonal(const Vector3 &p_scale);
	Basis scaled_orthogonal(const Vector3 &p_scale) const;

	void make_scale_uniform();
	float get_uniform_scale() const;

	Vector3 get_scale() const;
	Vector3 get_scale_abs() const;
	Vector3 get_scale_local() const;

	void set_axis_angle_scale(const Vector3 &p_axis, real_t p_phi, const Vector3 &p_scale);
	void set_euler_scale(const Vector3 &p_euler, const Vector3 &p_scale);
	void set_quaternion_scale(const Quaternion &p_quaternion, const Vector3 &p_scale);

	// transposed dot products
	_FORCE_INLINE_ real_t tdotx(const Vector3 &v) const {
		return elements[0][0] * v[0] + elements[1][0] * v[1] + elements[2][0] * v[2];
	}
	_FORCE_INLINE_ real_t tdoty(const Vector3 &v) const {
		return elements[0][1] * v[0] + elements[1][1] * v[1] + elements[2][1] * v[2];
	}
	_FORCE_INLINE_ real_t tdotz(const Vector3 &v) const {
		return elements[0][2] * v[0] + elements[1][2] * v[1] + elements[2][2] * v[2];
	}

	bool is_equal_approx(const Basis &p_basis) const;

	bool operator==(const Basis &p_matrix) const;
	bool operator!=(const Basis &p_matrix) const;

	_FORCE_INLINE_ Vector3 xform(const Vector3 &p_vector) const;
	_FORCE_INLINE_ Vector3 xform_inv(const Vector3 &p_vector) const;
	_FORCE_INLINE_ void operator*=(const Basis &p_matrix);
	_FORCE_INLINE_ Basis operator*(const Basis &p_matrix) const;
	_FORCE_INLINE_ void operator+=(const Basis &p_matrix);
	_FORCE_INLINE_ Basis operator+(const Basis &p_matrix) const;
	_FORCE_INLINE_ void operator-=(const Basis &p_matrix);
	_FORCE_INLINE_ Basis operator-(const Basis &p_matrix) const;
	_FORCE_INLINE_ void operator*=(const real_t p_val);
	_FORCE_INLINE_ Basis operator*(const real_t p_val) const;

	int get_orthogonal_index() const;
	void set_orthogonal_index(int p_index);

	bool is_orthogonal() const;
	bool is_diagonal() const;
	bool is_rotation() const;

	Basis lerp(const Basis &p_to, const real_t &p_weight) const;
	Basis slerp(const Basis &p_to, const real_t &p_weight) const;
	void rotate_sh(real_t *p_values);

	operator String() const;

	/* create / set */

	_FORCE_INLINE_ void set(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz) {
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
	_FORCE_INLINE_ void set(const Vector3 &p_x, const Vector3 &p_y, const Vector3 &p_z) {
		set_axis(0, p_x);
		set_axis(1, p_y);
		set_axis(2, p_z);
	}
	_FORCE_INLINE_ Vector3 get_column(int i) const {
		return Vector3(elements[0][i], elements[1][i], elements[2][i]);
	}

	_FORCE_INLINE_ Vector3 get_row(int i) const {
		return Vector3(elements[i][0], elements[i][1], elements[i][2]);
	}
	_FORCE_INLINE_ Vector3 get_main_diagonal() const {
		return Vector3(elements[0][0], elements[1][1], elements[2][2]);
	}

	_FORCE_INLINE_ void set_row(int i, const Vector3 &p_row) {
		elements[i][0] = p_row.x;
		elements[i][1] = p_row.y;
		elements[i][2] = p_row.z;
	}

	_FORCE_INLINE_ void set_zero() {
		elements[0].zero();
		elements[1].zero();
		elements[2].zero();
	}

	_FORCE_INLINE_ Basis transpose_xform(const Basis &m) const {
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
	Basis(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz) {
		set(xx, xy, xz, yx, yy, yz, zx, zy, zz);
	}

	void orthonormalize();
	Basis orthonormalized() const;

	void orthogonalize();
	Basis orthogonalized() const;

#ifdef MATH_CHECKS
	bool is_symmetric() const;
#endif
	Basis diagonalize();

	operator Quaternion() const { return get_quaternion(); }

	static Basis looking_at(const Vector3 &p_target, const Vector3 &p_up = Vector3(0, 1, 0));

	Basis(const Quaternion &p_quaternion) { set_quaternion(p_quaternion); };
	Basis(const Quaternion &p_quaternion, const Vector3 &p_scale) { set_quaternion_scale(p_quaternion, p_scale); }

	Basis(const Vector3 &p_axis, real_t p_phi) { set_axis_angle(p_axis, p_phi); }
	Basis(const Vector3 &p_axis, real_t p_phi, const Vector3 &p_scale) { set_axis_angle_scale(p_axis, p_phi, p_scale); }
	static Basis from_scale(const Vector3 &p_scale);

	_FORCE_INLINE_ Basis(const Vector3 &row0, const Vector3 &row1, const Vector3 &row2) {
		elements[0] = row0;
		elements[1] = row1;
		elements[2] = row2;
	}

	_FORCE_INLINE_ Basis() {}
};

_FORCE_INLINE_ void Basis::operator*=(const Basis &p_matrix) {
	set(
			p_matrix.tdotx(elements[0]), p_matrix.tdoty(elements[0]), p_matrix.tdotz(elements[0]),
			p_matrix.tdotx(elements[1]), p_matrix.tdoty(elements[1]), p_matrix.tdotz(elements[1]),
			p_matrix.tdotx(elements[2]), p_matrix.tdoty(elements[2]), p_matrix.tdotz(elements[2]));
}

_FORCE_INLINE_ Basis Basis::operator*(const Basis &p_matrix) const {
	return Basis(
			p_matrix.tdotx(elements[0]), p_matrix.tdoty(elements[0]), p_matrix.tdotz(elements[0]),
			p_matrix.tdotx(elements[1]), p_matrix.tdoty(elements[1]), p_matrix.tdotz(elements[1]),
			p_matrix.tdotx(elements[2]), p_matrix.tdoty(elements[2]), p_matrix.tdotz(elements[2]));
}

_FORCE_INLINE_ void Basis::operator+=(const Basis &p_matrix) {
	elements[0] += p_matrix.elements[0];
	elements[1] += p_matrix.elements[1];
	elements[2] += p_matrix.elements[2];
}

_FORCE_INLINE_ Basis Basis::operator+(const Basis &p_matrix) const {
	Basis ret(*this);
	ret += p_matrix;
	return ret;
}

_FORCE_INLINE_ void Basis::operator-=(const Basis &p_matrix) {
	elements[0] -= p_matrix.elements[0];
	elements[1] -= p_matrix.elements[1];
	elements[2] -= p_matrix.elements[2];
}

_FORCE_INLINE_ Basis Basis::operator-(const Basis &p_matrix) const {
	Basis ret(*this);
	ret -= p_matrix;
	return ret;
}

_FORCE_INLINE_ void Basis::operator*=(const real_t p_val) {
	elements[0] *= p_val;
	elements[1] *= p_val;
	elements[2] *= p_val;
}

_FORCE_INLINE_ Basis Basis::operator*(const real_t p_val) const {
	Basis ret(*this);
	ret *= p_val;
	return ret;
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

real_t Basis::determinant() const {
	return elements[0][0] * (elements[1][1] * elements[2][2] - elements[2][1] * elements[1][2]) -
			elements[1][0] * (elements[0][1] * elements[2][2] - elements[2][1] * elements[0][2]) +
			elements[2][0] * (elements[0][1] * elements[1][2] - elements[1][1] * elements[0][2]);
}
#endif // BASIS_H
